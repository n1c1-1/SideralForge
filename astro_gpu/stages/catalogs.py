"""Gaia DR3 catalogues of the field: bright/deep stars (photometry), faint stars (model), XP spectra (colour)."""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from ..core.imgutil import safe_world2pix
from ..core.params import ADVANCED, P, StageParams
from ..core.stage import Context, Stage, StageError

CACHE = Path.home() / ".astro_gpu" / "cache"


class CatalogsParams(StageParams):
    g_deep: float = P(18.5, label="Measured stars limit", level=ADVANCED, ge=12, le=21, step=0.5, unit="G mag",
                      help="Stars brighter than this Gaia magnitude are measured individually (forced photometry) "
                           "and removed from the nebula layer.")
    g_faint: float = P(20.5, label="Modelled stars limit", level=ADVANCED, ge=0, le=21, step=0.5, unit="G mag",
                       help="Fainter stars down to this magnitude are not measured but modelled from their Gaia "
                            "magnitude; they form the grain of the Milky Way and are moved to the star layer.",
                       tip="Set it equal to the measured-stars limit to disable faint-star modelling.")
    max_faint: int = P(4_000_000, label="Maximum modelled stars", level=ADVANCED, ge=0,
                       help="Upper bound on the number of faint stars (very wide Milky Way fields).")
    xp_gmin: float = P(7.5, label="Colour stars: brightest", level=ADVANCED, ge=3, le=15, step=0.5, unit="G mag",
                       help="Brightest Gaia stars whose XP spectrum is used for the colour calibration.")
    xp_gmax: float = P(14.0, label="Colour stars: faintest", level=ADVANCED, ge=8, le=17.5, step=0.5, unit="G mag",
                       help="Faintest Gaia stars whose XP spectrum is used for the colour calibration.")
    xp_max: int = P(4000, label="Colour stars: number", level=ADVANCED, ge=100, le=20000,
                    help="Maximum number of XP spectra downloaded (they are cached and reused).")
    cell_deg: float = P(1.0, label="Query cell size", level=ADVANCED, ge=0.1, le=5, unit="deg",
                        help="Size of the cones in which the Gaia archive is queried.")


class Catalogs(Stage):
    name = "catalogs"
    title = "Gaia catalogues"
    description = ("Downloads from the Gaia DR3 archive every star of the field: positions (propagated to the date of "
                   "the images), magnitudes and colours, plus the low-resolution XP spectra of a few thousand "
                   "isolated stars used to calibrate the colour of the sensor. Results are cached and shared "
                   "between projects.")
    deps = ("inventory", "astrometry", "stack")
    Params = CatalogsParams
    version = "2"          # 2: the bright stars missing from Gaia are added from Hipparcos
    network = True

    def run(self, ctx: Context, p: CatalogsParams) -> None:
        from astropy.io import fits
        from astropy.wcs import WCS
        w = WCS(fits.getheader(ctx.input("astrometry", "wcs.fits")))
        meta = ctx.load_json("stack", "drz_meta.json")
        W, H = meta["width"], meta["height"]
        scale = float(ctx.metric_of("astrometry", "scale_arcsec"))
        epoch = _epoch(ctx.metric_of("inventory", "first_time"))
        centres, radius = _cells(w, W, H, scale, p.cell_deg)
        ctx.log(f"{len(centres)} sky cells of {p.cell_deg} deg, epoch {epoch:.2f}")

        # ---------------- deep catalogue (measured stars)
        cols = ["Source", "RA_ICRS", "DE_ICRS", "Gmag", "BPmag", "RPmag", "pmRA", "pmDE", "XPsamp"]
        c = _query_cells(ctx, centres, radius, cols, f"<{p.g_deep}", "deep", 0.0, 0.5)
        dt = epoch - 2016.0
        dec_e = c["DE_ICRS"] + np.nan_to_num(c["pmDE"]) / 3.6e6 * dt
        ra_e = c["RA_ICRS"] + np.nan_to_num(c["pmRA"]) / 3.6e6 * dt / np.cos(np.radians(c["DE_ICRS"]))
        x, y = safe_world2pix(w, ra_e, dec_e, W, H)
        inside = (x > -30) & (x < W + 30) & (y > -30) & (y < H + 30) & np.isfinite(c["Gmag"])
        deep = dict(source=c["Source"][inside], ra=ra_e[inside], dec=dec_e[inside], x=x[inside], y=y[inside],
                    g=c["Gmag"][inside], bp=c["BPmag"][inside], rp=c["RPmag"][inside], xp=c["XPsamp"][inside])
        add = _bright_missing(ctx, w, W, H, scale, epoch, deep)
        if add is not None:
            deep = {k: np.concatenate([deep[k], add[k].astype(deep[k].dtype)]) for k in deep}
        np.savez(ctx.out("gaia_deep.npz"), **deep)
        ctx.metric("stars_deep", int(inside.sum()))
        ctx.log(f"{inside.sum()} Gaia stars G < {p.g_deep} in the field")
        if inside.sum() < 100:
            raise StageError("almost no Gaia star in the field: the astrometric solution is probably wrong")

        # ---------------- faint catalogue (modelled stars)
        if p.g_faint > p.g_deep:
            f = _query_cells(ctx, centres, radius, ["RA_ICRS", "DE_ICRS", "Gmag"], f"{p.g_deep}..{p.g_faint}",
                             "faint", 0.5, 0.85)
            fx, fy = safe_world2pix(w, f["RA_ICRS"], f["DE_ICRS"], W, H)
            fin = (fx > -30) & (fx < W + 30) & (fy > -30) & (fy < H + 30) & np.isfinite(f["Gmag"])
            fx, fy, fg = fx[fin], fy[fin], f["Gmag"][fin]
            if len(fg) > p.max_faint:
                o = np.argsort(fg)[: p.max_faint]
                ctx.log(f"{len(fg)} faint stars: keeping the {p.max_faint} brightest (G < {fg[o[-1]]:.2f})", "warning")
                fx, fy, fg = fx[o], fy[o], fg[o]
        else:
            fx = fy = fg = np.zeros(0)
        np.savez(ctx.out("gaia_faint.npz"), x=fx.astype(np.float32), y=fy.astype(np.float32), g=fg.astype(np.float32))
        ctx.metric("stars_faint", int(len(fg)))
        ctx.log(f"{len(fg)} faint Gaia stars {p.g_deep} < G < {p.g_faint} modelled")

        # ---------------- XP spectra of isolated stars (colour calibration)
        d = np.load(ctx.out("gaia_deep.npz"))
        sel = _isolated_xp(d, p, W, H, scale)
        ctx.log(f"{len(sel)} isolated stars with an XP spectrum selected for the colour calibration")
        rows = _xp_spectra(ctx, d["source"][sel].astype(np.int64), 0.85, 1.0)
        if len(rows) < 50:
            raise StageError(f"only {len(rows)} XP spectra available: the colour calibration needs at least 50")
        keep = np.array([i for i, r in zip(sel, rows) if r is not None])
        lam = next(r for r in rows if r is not None)[0]
        flux = np.stack([r[1] for r in rows if r is not None])
        xyz = _xyz(lam, flux)
        np.savez(ctx.out("gaia_xp.npz"), source=d["source"][keep], x=d["x"][keep], y=d["y"][keep], g=d["g"][keep],
                 lam=lam, flux=flux, xyz=xyz)
        ctx.metric("xp_spectra", int(len(keep)))


def _epoch(t: str | None) -> float:
    if not t:
        return 2016.0
    dt = datetime.fromisoformat(t)
    y0 = datetime(dt.year, 1, 1)
    return dt.year + (dt - y0).total_seconds() / (365.25 * 86400)


def _cells(w, W, H, scale, cell_deg):
    """Cone centres covering the frame (+ margin), laid out on the image grid (no RA wrap / pole issues)."""
    cell_px = cell_deg * 3600 / scale
    m = 0.05 * max(W, H)
    nx = max(1, int(np.ceil((W + 2 * m) / cell_px)))
    ny = max(1, int(np.ceil((H + 2 * m) / cell_px)))
    xs = -m + (np.arange(nx) + 0.5) * (W + 2 * m) / nx
    ys = -m + (np.arange(ny) + 0.5) * (H + 2 * m) / ny
    gx, gy = np.meshgrid(xs, ys)
    sky = w.pixel_to_world(gx.ravel(), gy.ravel())
    step = max((W + 2 * m) / nx, (H + 2 * m) / ny) * scale / 3600
    return np.c_[sky.ra.deg, sky.dec.deg], step * np.sqrt(2) / 2 * 1.05 + 0.01


def _query_cells(ctx, centres, radius, cols, gfilter, label, p0, p1):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    def one(c):
        key = hashlib.sha1(f"gaiadr3|{','.join(cols)}|{gfilter}|{c[0]:.5f}|{c[1]:.5f}|{radius:.4f}".encode()).hexdigest()
        cp = CACHE / "gaia" / f"{key}.npz"
        if cp.exists():
            return dict(np.load(cp))
        for attempt in range(5):
            try:
                v = Vizier(columns=cols, column_filters={"Gmag": gfilter}, row_limit=-1, timeout=900)
                r = v.query_region(SkyCoord(c[0], c[1], unit="deg"), radius=radius * u.deg, catalog="I/355/gaiadr3")
                out = {k: (np.ma.filled(np.ma.asarray(r[0][k]).astype(np.float64), np.nan) if len(r) else np.zeros(0))
                       for k in cols}
                if "Source" in out and len(r):
                    out["Source"] = np.asarray(r[0]["Source"]).astype(np.int64)
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cp, **out)
                return out
            except Exception as e:  # noqa: BLE001 - network errors: retry
                err = e
                time.sleep(3 * (attempt + 1))
        raise StageError(f"Gaia archive query failed ({err}); check the internet connection and retry")

    parts = []
    with ThreadPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(one, centres)):
            parts.append(r)
            ctx.progress(p0 + (p1 - p0) * (i + 1) / len(centres), f"Gaia {label}: cell {i + 1}/{len(centres)}")
    c = {k: np.concatenate([q[k] for q in parts if len(q[k])]) if any(len(q[k]) for q in parts) else np.zeros(0)
         for k in cols}
    # duplicates from overlapping cones: exact position (Gaia ids must stay int64: as floats they collide)
    key = np.round(c["RA_ICRS"] * 3.6e5).astype(np.int64) * 10 ** 9 + np.round(c["DE_ICRS"] * 3.6e5).astype(np.int64)
    _, idx = np.unique(key, return_index=True)
    return {k: v[idx] for k, v in c.items()}


def _bright_missing(ctx, w, W, H, scale, epoch, deep, vmax=9.0):
    """The brightest stars of the sky are partly missing from Gaia DR3 (its detector saturates; Alnitak, by the
    Horsehead, is absent). Missing, such a star is not modelled: its core and its glare stay in the nebula layer.
    Hipparcos stars of the field with no Gaia counterpart are added. Their Gaia magnitude and colour come from V and
    B-V through relations fitted on the Hipparcos stars of the same field that Gaia does have (no assumed
    photometric transformation). Returns the entries to append to the deep catalogue, or None."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier
    from scipy.spatial import cKDTree
    centre = w.pixel_to_world(W / 2, H / 2)
    radius = 0.5 * np.hypot(W, H) * scale / 3600 * 1.05
    cols = ["HIP", "RAICRS", "DEICRS", "Vmag", "B-V", "pmRA", "pmDE"]
    key = hashlib.sha1(f"hip|{centre.ra.deg:.4f}|{centre.dec.deg:.4f}|{radius:.3f}|{vmax}".encode()).hexdigest()
    cp = CACHE / "hipparcos" / f"{key}.npz"
    if cp.exists():
        t = dict(np.load(cp))
    else:
        t = None
        for attempt in range(4):
            try:
                v = Vizier(columns=cols, column_filters={"Vmag": f"<{vmax}"}, row_limit=-1, timeout=300)
                r = v.query_region(SkyCoord(centre.ra.deg, centre.dec.deg, unit="deg"), radius=radius * u.deg,
                                   catalog="I/239/hip_main")
                t = {k: (np.ma.filled(np.ma.asarray(r[0][k]).astype(np.float64), np.nan) if len(r) else np.zeros(0))
                     for k in cols}
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cp, **t)
                break
            except Exception:  # noqa: BLE001 - network errors: retry, then go on without the supplement
                time.sleep(3 * (attempt + 1))
        if t is None:
            ctx.log("Hipparcos query failed: bright stars missing from Gaia are not added", "warning")
            return None
    if not len(t["HIP"]):
        return None
    dt = epoch - 1991.25
    dec = t["DEICRS"] + np.nan_to_num(t["pmDE"]) / 3.6e6 * dt
    ra = t["RAICRS"] + np.nan_to_num(t["pmRA"]) / 3.6e6 * dt / np.cos(np.radians(t["DEICRS"]))
    x, y = safe_world2pix(w, ra, dec, W, H)
    V, BV = t["Vmag"], np.nan_to_num(t["B-V"], nan=0.6)
    ok = (x > -30) & (x < W + 30) & (y > -30) & (y < H + 30) & np.isfinite(V)
    if not ok.any():
        return None
    tree = cKDTree(np.c_[deep["x"], deep["y"]])
    rpx = max(5.0 / scale, 2.0)
    gd, col = deep["g"], deep["bp"] - deep["rp"]
    pairs, missing = [], []
    for i in np.where(ok)[0]:
        nb = [j for j in tree.query_ball_point([x[i], y[i]], rpx) if gd[j] < V[i] + 1.0]
        if nb:
            j = min(nb, key=lambda j: gd[j])
            if np.isfinite(col[j]) and np.isfinite(t["B-V"][i]):
                pairs.append((BV[i], V[i], gd[j], col[j]))
        else:
            missing.append(i)
    if not missing:
        return None
    if len(pairs) >= 8:
        P_ = np.array(pairs)
        cg = np.polyfit(P_[:, 0], P_[:, 2] - P_[:, 1], 2)
        cc = np.polyfit(P_[:, 0], P_[:, 3], 2)
        how = f"relations fitted on {len(pairs)} stars of the field"
    else:                                   # too few: G - V of Riello et al. 2021, a mean colour relation
        cg = np.array([-0.2156, 0.01424, -0.02704])
        cc = np.array([0.0, 1.2, 0.08])
        how = "mean relations (too few Hipparcos stars matched in the field)"
    m = np.array(missing)
    g_new = V[m] + np.polyval(cg, BV[m])
    c_new = np.polyval(cc, BV[m])
    for i, gg in zip(m, g_new):
        ctx.log(f"HIP {int(t['HIP'][i])} (V = {V[i]:.2f}) is missing from Gaia: added with G = {gg:.2f} ({how})")
    ctx.metric("bright_stars_added", [int(t["HIP"][i]) for i in m])
    return dict(source=-t["HIP"][m].astype(np.int64), ra=ra[m], dec=dec[m], x=x[m], y=y[m], g=g_new,
                bp=g_new + c_new / 2, rp=g_new - c_new / 2, xp=np.zeros(len(m)))


def _isolated_xp(d, p, W, H, scale):
    from scipy.spatial import cKDTree
    g = d["g"]
    xp = np.nan_to_num(d["xp"]) == 1
    cand = np.where(xp & (g > p.xp_gmin) & (g < p.xp_gmax) & (d["x"] > 60) & (d["x"] < W - 60)
                    & (d["y"] > 60) & (d["y"] < H - 60))[0]
    tree = cKDTree(np.c_[d["x"], d["y"]])
    keep = []
    rpx = 40.0 / scale                     # isolation: no star brighter than G + 3 within 40"
    for i in cand:
        nb = tree.query_ball_point([d["x"][i], d["y"][i]], rpx)
        if all(j == i or g[j] >= g[i] + 3 for j in nb):
            keep.append(i)
    keep = np.array(keep, int)
    return keep[np.argsort(g[keep])][: p.xp_max]


def _xp_spectra(ctx, sources, p0, p1):
    from astroquery.vizier import Vizier

    def fetch(sid):
        cp = CACHE / "xp" / f"{sid}.npy"
        if cp.exists():
            a = np.load(cp)
            return (a[0], a[1]) if a.size else None
        for _ in range(3):
            try:
                v = Vizier(columns=["Source", "lambda", "Flux"], row_limit=-1, timeout=120)
                rr = v.query_constraints(catalog="I/355/xpsample", Source=f"={sid}")
                cp.parent.mkdir(parents=True, exist_ok=True)
                if len(rr) and len(rr[0]):
                    lam = np.asarray(rr[0]["lambda"], float)
                    o = np.argsort(lam)
                    out = (lam[o], np.asarray(rr[0]["Flux"], float)[o])
                    np.save(cp, np.stack(out))
                    return out
                np.save(cp, np.zeros(0))
                return None
            except Exception:  # noqa: BLE001
                time.sleep(2)
        return None

    rows = []
    with ThreadPoolExecutor(10) as ex:
        for i, r in enumerate(ex.map(fetch, [int(s) for s in sources])):
            rows.append(r)
            ctx.progress(p0 + (p1 - p0) * (i + 1) / len(sources), f"XP spectra {i + 1}/{len(sources)}")
    return rows


def _xyz(lam, flux):
    import colour
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    wl, xb = cmfs.wavelengths, cmfs.values
    s = (lam >= wl.min()) & (lam <= wl.max())
    xbar = np.stack([np.interp(lam[s], wl, xb[:, i]) for i in range(3)], 1)
    return flux[:, s] @ xbar * np.gradient(lam[s])[:, None].mean()
