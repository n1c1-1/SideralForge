"""Astrometry: plate solving of the stack directly on Gaia DR3 (TAN-SIP world coordinate system)."""
from __future__ import annotations

import json
import re
import threading

import numpy as np

from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage, StageError

GOOD_PAIRS = 15     # matched star pairs that make a triangle match certain (false matches raise instead)
TRIAL_G = 11.0      # the stars of the scale search trials when they are enough (see the search)
#: under this many stars in the stack, a field is solved by ``_match_sparse``
SPARSE_FIELD = 50
#: the fewest stars that can prove a solution: the 3 of the triangle that proposes it, and 3 it did not use
SPARSE_MIN_STARS = 6
#: the detected stars, and twice as many catalogue stars, whose triangles are compared (every triple of them)
SPARSE_TRIANGLE_STARS = 30
#: the published decision on a solution (D. Lang et al., "Astrometry.net", AJ 139, 1782, 2010, and the defaults of its
#: solver): the Bayes factor it must reach, the stars' position error (px), the share of stars with no counterpart
ODDS_TO_SOLVE, VERIFY_PX, DISTRACTORS = 1e9, 1.0, 0.25


class AstrometryParams(StageParams):
    target: str = P("", label="Target name",
                    help="Name of the main object (e.g. 'NGC 7822', 'IC 1805', 'Heart Nebula'), resolved with SIMBAD "
                         "to get the approximate centre of the field.",
                    tip="Not needed when RA/Dec are given or when the FITS files contain the pointing.")
    ra: float | None = P(None, label="Centre RA", unit="deg", ge=0, le=360,
                         help="Approximate right ascension of the field centre (degrees). Within ~20 % of the field "
                              "width is enough.")
    dec: float | None = P(None, label="Centre Dec", unit="deg", ge=-90, le=90,
                          help="Approximate declination of the field centre (degrees).")
    focal_mm: float | None = P(None, label="Focal length", unit="mm", gt=0,
                               help="Effective focal length of the instrument (with reducer / flattener). With the pixel "
                                    "size of the camera it gives the image scale, which makes the astrometry immediate.",
                               tip="Read from the files when the camera records it; telescopes usually do not report "
                                   "it. Leave empty to let the solver search the scale (a few minutes).")
    pixel_um: float | None = P(None, label="Pixel size", unit="um", gt=0, level=ADVANCED,
                               help="Physical size of a sensor pixel, from the camera database or the FITS header.")
    scale: float | None = P(None, label="Image scale", unit="arcsec/px", gt=0, level=ADVANCED,
                            help="Approximate size of a pixel of the stacked image on the sky. Computed from the focal "
                                 "length and the pixel size when they are known; otherwise searched automatically.",
                            tip="206.265 x pixel size (um) / focal length (mm), divided by the drizzle scale.")
    blind: bool = P(True, label="Blind solving", level=ADVANCED,
                    help="When the field centre is unknown (or the guided search fails), identify the field on the "
                         "whole sky from its brightest stars (the shapes of groups of four stars, looked up in an "
                         "index of the sky).",
                    tip="Works for fields wider than about 1 deg; the first use downloads two bright-star catalogues.")
    search_offsets: float = P(0.2, label="Search range", level=ADVANCED, ge=0.0, le=1.0, step=0.05,
                              unit="x field", help="How far from the given centre the solver looks for the field.")
    sip_order: int = P(3, label="Distortion order", level=ADVANCED, ge=0, le=5,
                       help="Order of the SIP polynomial describing the optical distortion.",
                       tip="3 for refractors with reducers / wide fields; 2 for small fields; 0 = pure tangent plane.")


class Astrometry(Stage):
    name = "astrometry"
    title = "Astrometry"
    description = ("Finds exactly where the image points on the sky by matching its stars to Gaia DR3 (triangle "
                   "matching, then a tangent-plane + distortion fit on thousands of stars). Every later step that "
                   "uses the sky (Gaia photometry, colour calibration, sky surveys) relies on this solution; the "
                   "typical residual is a few hundredths of a pixel.")
    deps = ("inventory", "stack")
    Params = AstrometryParams
    version = "1"
    network = True

    def auto(self, ctx: Context) -> dict[str, Auto]:
        a: dict[str, Auto] = {}
        tgt = ctx.project.config.target
        if tgt.ra is not None and tgt.dec is not None:
            a["ra"], a["dec"] = Auto(tgt.ra, "project target"), Auto(tgt.dec, "project target")
        elif ctx.metric_of("inventory", "pointing_ra") is not None:
            a["ra"] = Auto(ctx.metric_of("inventory", "pointing_ra"), "telescope pointing in the FITS headers")
            a["dec"] = Auto(ctx.metric_of("inventory", "pointing_dec"), "telescope pointing in the FITS headers")
        # the target entered, else the object the FITS headers name; never the project's name (issue #38: a
        # project named "test" was looked up in the catalogues and on the network)
        name = tgt.name or ctx.metric_of("inventory", "object")
        if name and (tgt.ra is None or tgt.dec is None):    # with its coordinates, its name changes nothing
            a["target"] = Auto(name, "project / FITS object name")
        pix, foc = ctx.metric_of("inventory", "pixel_um"), ctx.metric_of("inventory", "focal_mm")
        if getattr(tgt, "pixel_um", None) and not pix:
            a["pixel_um"] = Auto(tgt.pixel_um, "entered when the project was created")
        if pix:
            a["pixel_um"] = Auto(pix, "camera {camera}", {"camera": ctx.metric_of('inventory', 'camera')})
        if foc:
            a["focal_mm"] = Auto(foc, "recorded in the files")
        if ctx.project.config.target.focal_mm:
            a["focal_mm"] = Auto(ctx.project.config.target.focal_mm, "entered when the project was created")
        sc = ctx.metric_of("inventory", "pixel_scale")
        if sc and not (foc and (pix or getattr(tgt, "pixel_um", None))):
            try:
                S = float(ctx.load_json("stack", "drz_meta.json")["scale"])
            except Exception:  # noqa: BLE001 - no stack to read yet
                S = 1.0
            a["scale"] = Auto(round(float(sc) / S, 4), "pixel scale in the FITS headers")
        return a

    def run(self, ctx: Context, p: AstrometryParams) -> None:
        import astroalign as aa
        from astropy.io import fits
        from astropy.wcs.utils import fit_wcs_from_points
        from scipy.spatial import cKDTree

        ra0, dec0 = p.ra, p.dec
        S_drz = float(ctx.load_json("stack", "drz_meta.json")["scale"])
        if p.scale is None and p.focal_mm and p.pixel_um:
            p = p.model_copy(update={"scale": 206.265 * p.pixel_um / p.focal_mm / S_drz})
            ctx.log(f"image scale {p.scale:.3f}\"/px from focal {p.focal_mm} mm and pixel {p.pixel_um} um")
        if (ra0 is None or dec0 is None) and p.target:
            try:
                ra0, dec0 = resolve_name(p.target, ctx.log)
                ctx.log(f"'{p.target}' resolved: RA {ra0:.4f} Dec {dec0:.4f}")
            except StageError as e:
                ctx.log(str(e), "warning")
        d = ctx.load("stack", "drz_full.npy", mmap=True)
        g = np.array(d[1], dtype=np.float32, copy=True)
        H, W = g.shape
        o = detect(g)
        o_all = np.sort(o, order="flux")[::-1]
        brightdet = np.c_[o_all["x"], o_all["y"]][:150]
        o = np.sort(o[o["flag"] == 0], order="flux")[::-1]
        det = np.c_[o["x"], o["y"]]
        ctx.log(f"{len(det)} stars detected in the stack")
        if len(det) < SPARSE_MIN_STARS:
            raise StageError(f"only {len(det)} stars found in the stack, at least {SPARSE_MIN_STARS} are needed to "
                             f"identify the field: add light frames, or take Astrometry out of this workflow")
        sparse = len(det) < SPARSE_FIELD
        unproved = StageError(f"the {len(det)} stars of this sparse field do not identify it on the sky beyond doubt, so "
                              "no solution is given rather than a wrong one: take Astrometry out of this workflow")
        odds = 0.0
        if ra0 is None or dec0 is None:
            if not p.blind:
                raise StageError("the approximate field centre is unknown: enter the target name or its RA/Dec")
            ra0, dec0, sc_b = _blind(ctx, brightdet, W, H)
            p = p.model_copy(update={"scale": sc_b})

        # triangle matching is scale-invariant: a coarse grid of scales is enough when the scale is unknown
        scales = [p.scale] if p.scale else list(np.geomspace(0.4, 40, 22))
        tol = 0.15 if p.scale else 0.13
        pix = p.pixel_um
        camera = ctx.metric_of("inventory", "camera")
        if not p.scale:
            # most plausible first: the focal lengths this camera was solved at before, then a ~500 mm focal length
            # when the pixel size is known, else ~3"/px
            typical = 206.265 * pix / 500.0 / S_drz if pix else 3.0
            scales.sort(key=lambda v: abs(np.log(v / typical)))
            hints = [206.265 * pix / f / S_drz for f in focal_hints(camera, pix)] if pix else []
            if hints:
                ctx.log(f"focal length unknown: trying first the ones this camera was solved at before "
                        f"({', '.join(f'{f:g} mm' for f in focal_hints(camera, pix))})")
            scales = hints + scales
        best = None
        flips = mirror_hint(camera, pix)
        # pass 1: every scale with the given centre only (fast); pass 2: also shifted centres
        passes = [0.0, p.search_offsets] if p.search_offsets > 0 else [0.0]
        for ip, off in enumerate(passes):
            queue = list(scales)
            tried: list[float] = []
            k = -1
            while queue:
                sc = queue.pop(0)
                tried.append(sc)
                k += 1
                ctx.progress(0.6 * (ip + min(k, len(scales) - 1) / len(scales)) / len(passes),
                             f"searching the field at {sc:.2f}\"/px" + (" (shifted centres)" if off else ""))
                fw, fh = W * sc / 3600, H * sc / 3600
                if not p.scale and (fw > 40 or fw < 0.1):
                    continue
                area = fw * fh
                glim = float(np.clip(12.5 - 1.25 * np.log10(max(area, 1e-3) / 37.0), 10.5, 16.0))
                radius = 0.5 * np.hypot(fw, fh) * (1 + 2 * p.search_offsets) + 0.2
                try:
                    gs, gm = None, None
                    if not p.scale and glim > TRIAL_G:
                        # a trial only matches the 150 brightest stars in its frame: when the stars brighter than
                        # TRIAL_G give 150 at every trial centre, they are the same 150 (one small download shared
                        # by every trial scale; the full depth is fetched once, for the scale that matches)
                        gs, gm = gaia_bright(ra0, dec0, radius, TRIAL_G, ctx.log)
                        if _fewest_in_frame(gs, ra0, dec0, sc, W, H, off, fw, fh) < 150:
                            gs, gm = None, None
                    if gs is None:
                        gs, gm = gaia_bright(ra0, dec0, radius, glim, ctx.log)
                except Exception as e:  # noqa: BLE001 - no catalogue for this trial scale (offline, services down)
                    if p.scale:
                        raise StageError("the star catalogue for this field cannot be read: check the network "
                                         "connection, or download the star positions in Settings") from e
                    ctx.log(f"no Gaia stars at {sc:.2f}\"/px ({str(e)[:150]})", "warning")
                    continue
                if len(gm) < 20:
                    continue
                if sparse:
                    res = _match_sparse(gs, gm, ra0, dec0, sc, W, H, det, off, fw, fh, tol, flips)
                    if res is not None:
                        best, odds = (*res[:4], sc, glim, radius), res[4]
                        break
                    continue
                seen: list[float] = []
                # scale unknown: 60 control points per trial (a vain trial costs ~2 s instead of ~70 s), and the
                # scale that matches is matched again with the usual 100, as when the scale is known
                res = _match(gs, gm, ra0, dec0, sc, W, H, brightdet, off, fw, fh, aa, tol, flips,
                             seen if not p.scale else None, mcp=100 if p.scale else 60)
                if res is not None and not p.scale and res[2] >= GOOD_PAIRS:
                    full = _match(gs, gm, ra0, dec0, sc, W, H, brightdet, off, fw, fh, aa, tol, (res[0],))
                    res = full if full is not None and full[2] >= res[2] else res
                if res is not None and (best is None or res[2] > best[2]):
                    best = (*res, sc, glim, radius)
                    if res[2] >= GOOD_PAIRS:
                        break
                # a certain match found at another scale than the trial one (triangles do not see the scale):
                # try that scale next instead of walking the grid
                for s2 in seen:
                    if len(tried) < len(scales) + 3 and all(abs(np.log(s2 / q)) > 0.02 for q in tried):
                        ctx.log(f"stars matched at {sc:.2f}\"/px trial: the image scale is about {s2:.2f}\"/px")
                        queue.insert(0, s2)
                        break
            if best is not None and (sparse or best[2] >= GOOD_PAIRS):
                break
        if best is None and p.blind:
            ctx.log("guided search failed: trying the all-sky blind solver", "warning")
            ra0, dec0, sc_b = _blind(ctx, brightdet, W, H)
            fw, fh = W * sc_b / 3600, H * sc_b / 3600
            glim = float(np.clip(12.5 - 1.25 * np.log10(max(fw * fh, 1e-3) / 37.0), 10.5, 16.0))
            radius = 0.5 * np.hypot(fw, fh) * (1 + 2 * p.search_offsets) + 0.2
            gs, gm = gaia_bright(ra0, dec0, radius, glim, ctx.log)
            if sparse:
                res = _match_sparse(gs, gm, ra0, dec0, sc_b, W, H, det, 0.0, fw, fh, 0.15, flips)
                if res is not None:
                    best, odds = (*res[:4], sc_b, glim, radius), res[4]
            else:
                res = _match(gs, gm, ra0, dec0, sc_b, W, H, brightdet, 0.0, fw, fh, aa, 0.15, flips)
                if res is not None:
                    best = (*res, sc_b, glim, radius)
        if best is None:
            if sparse:
                raise unproved
            raise StageError("the stars of the image match no star of the catalogue: check the target or its "
                             "coordinates, and the focal length and pixel size")
        flip, T, nm, crval, sc, glim, radius = best
        w0 = _tan(crval[0], crval[1], sc, W, H)
        ctx.log(f"triangle match: {nm} stars, scale {sc * T.scale:.3f}\"/px, mirror {flip}")
        if sparse:
            ctx.log(f"sparse field: the solution holds {odds / np.log(10):.0f} powers of ten over chance, 9 needed "
                    f"({nm} of {len(det)} stars)")
        # with a few dozen stars, a distortion polynomial would follow their noise
        sip = None if sparse else (p.sip_order or None)
        q = det[:4000].copy()
        if flip:
            q[:, 1] = H - 1 - q[:, 1]
        pred = T(q)
        sky = w0.pixel_to_world(pred[:, 0], pred[:, 1])
        w1 = fit_wcs_from_points((det[:4000, 0], det[:4000, 1]), sky, projection="TAN", sip_degree=sip)
        gs, gm = gaia_bright(ra0, dec0, radius, glim, ctx.log)
        res = None
        for it, tol in enumerate((3.0, 1.5, 1.0)):
            ctx.progress(0.7 + 0.1 * it, "refining the solution")
            gx, gy = w1.world_to_pixel(gs)
            ins = (gx > 5) & (gx < W - 5) & (gy > 5) & (gy < H - 5)
            tr = cKDTree(det[:8000])
            dd, j = tr.query(np.c_[gx[ins], gy[ins]], distance_upper_bound=tol)
            m = np.isfinite(dd)
            if sparse and m.sum() < SPARSE_MIN_STARS:
                raise unproved
            w1 = fit_wcs_from_points((det[:8000][j[m], 0], det[:8000][j[m], 1]), gs[ins][m], projection="TAN",
                                     sip_degree=sip)
            gx2, gy2 = w1.world_to_pixel(gs[ins][m])
            res = np.hypot(gx2 - det[:8000][j[m], 0], gy2 - det[:8000][j[m], 1])
        if sparse and float(np.median(res)) > VERIFY_PX:
            raise unproved
        fits.PrimaryHDU(header=w1.to_header(relax=True)).writeto(ctx.out("wcs.fits"), overwrite=True)
        c = w1.pixel_to_world(W / 2, H / 2)
        scale = abs(w1.proj_plane_pixel_scales()[0].to_value("arcsec"))
        north = w1.pixel_to_world(W / 2, H / 2 - 100)
        ang = float(c.position_angle(north).deg)
        S = float(ctx.load_json("stack", "drz_meta.json")["scale"])
        ctx.metric("ra", round(float(c.ra.deg), 5))
        ctx.metric("dec", round(float(c.dec.deg), 5))
        ctx.metric("scale_arcsec", round(scale, 4))
        ctx.metric("sensor_scale_arcsec", round(scale * S, 4))
        ctx.metric("field_deg", [round(W * scale / 3600, 3), round(H * scale / 3600, 3)])
        ctx.metric("up_position_angle", round(ang, 2))
        ctx.metric("parity_flipped", bool(np.linalg.det(w1.pixel_scale_matrix) > 0))
        ctx.metric("residual_px", round(float(np.median(res)), 3))
        ctx.metric("stars_used", int(len(res)))
        pix = ctx.metric_of("inventory", "pixel_um")
        if pix:
            ctx.metric("focal_estimate_mm", round(206.265 * pix / (scale * S), 1))
        remember_instrument(camera, pix, 206.265 * pix / (scale * S) if pix else None, bool(flip))
        ctx.log(f"solved: centre RA {c.ra.deg:.4f} Dec {c.dec.deg:.4f}, {scale:.3f}\"/px, field "
                f"{W * scale / 3600:.2f} x {H * scale / 3600:.2f} deg, residual {np.median(res):.3f} px on {len(res)} stars")


def detect(g: np.ndarray, tile_above: int = 20_000_000) -> np.ndarray:
    """Sources of the stack (sep: background mesh 64 px, 15 sigma of the global noise, 6 pixels at least). A large
    stack is cut in 4 x 4 tiles with 128 px of overlap, extracted in parallel threads (sep releases the GIL), each
    source kept by the tile whose core holds its centre: the same stars at the same positions (NGC 7822, 6384 x 4272,
    nebulous: the 33 653 sources of the star list identical, in 26 s instead of 172 s)."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    import sep

    from ..core.imgutil import sep_extract
    sep.set_extract_pixstack(3000000)
    bk = sep.Background(g, bw=64, bh=64)
    sub = g - bk.back()
    rms = bk.globalrms
    H, W = sub.shape
    if H * W < tile_above:
        return sep_extract(sub, 15, err=rms, minarea=6)
    n, ov = 4, 128
    ys = np.linspace(0, H, n + 1).astype(int)
    xs = np.linspace(0, W, n + 1).astype(int)

    def one(ij):
        i, j = ij
        y0, y1, x0, x1 = max(0, ys[i] - ov), min(H, ys[i + 1] + ov), max(0, xs[j] - ov), min(W, xs[j + 1] + ov)
        o = sep_extract(np.ascontiguousarray(sub[y0:y1, x0:x1]), 15, err=rms, minarea=6)
        o = o.copy()
        for k, off in (("x", x0), ("xmin", x0), ("xmax", x0), ("xpeak", x0), ("xcpeak", x0),
                       ("y", y0), ("ymin", y0), ("ymax", y0), ("ypeak", y0), ("ycpeak", y0)):
            o[k] += off
        keep = ((o["x"] >= xs[j] - 0.5) & (o["x"] < xs[j + 1] - 0.5) & (o["y"] >= ys[i] - 0.5)
                & (o["y"] < ys[i + 1] - 0.5))
        return o[keep]
    with ThreadPoolExecutor(min(n * n, os.cpu_count() or 4)) as ex:
        return np.concatenate(list(ex.map(one, [(i, j) for i in range(n) for j in range(n)])))


def _instruments() -> dict:
    from ..core import skycache
    p = skycache.find("instruments.json")
    if p is None:
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _instrument_key(camera, pix) -> str:
    return f"{camera or 'camera'}|{float(pix or 0):.2f}"


def focal_hints(camera, pix) -> list[float]:
    """Focal lengths (mm) this camera (and pixel size) was solved at before, most recent first (``cache/
    instruments.json``): tried first when the focal length is unknown, so the scale search usually stops at once."""
    if not pix:
        return []
    return [float(f) for f in _instruments().get("focal", {}).get(_instrument_key(camera, pix), [])][:4]


def mirror_hint(camera, pix) -> tuple[bool, bool]:
    """The order in which to try the two mirror images: the one this camera was solved with last, else mirrored
    first (the stacks are stored top row first, the sky's parity flipped: every case so far)."""
    m = _instruments().get("mirror", {}).get(_instrument_key(camera, pix))
    return (False, True) if m is False else (True, False)


def remember_instrument(camera, pix, focal: float | None, mirror: bool) -> None:
    from ..core import skycache
    cur = skycache.cache_dir() / "instruments.json"
    try:
        d = json.loads(cur.read_text(encoding="utf-8")) if cur.exists() else {}
        k = _instrument_key(camera, pix)
        d.setdefault("mirror", {})[k] = mirror
        if pix and focal and np.isfinite(focal) and focal > 0:
            f = round(float(focal), 0)
            old = [x for x in d.setdefault("focal", {}).get(k, []) if abs(x - f) / f > 0.03]   # one per focal (3 %)
            d["focal"][k] = ([f] + old)[:4]
        skycache.save_json(cur, d)
    except (OSError, ValueError, AttributeError):
        pass


def _blind(ctx, xy, W, H):
    """All-sky blind solve of the brightest stars: (ra, dec, scale) of the field centre."""
    from ..core.blindsolve import solve_any
    ctx.progress(0.05, "all-sky blind solving (first use builds the star index, a few minutes)")
    r = solve_any(xy[:80], W, H)
    if r is None:
        raise StageError("the field could not be identified on the sky: enter the target name or its coordinates")
    ctx.log(f"blind solve: centre RA {r['ra']:.3f} Dec {r['dec']:+.3f}, {r['scale']:.2f}\"/px, "
            f"{r['matches']} catalogue stars matched (chance level {r['chance']})")
    ctx.metric("blind_solved", True)
    return r["ra"], r["dec"], r["scale"]


def _tan(ra, dec, scale, W, H):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [W / 2, H / 2]
    w.wcs.cdelt = [-scale / 3600, scale / 3600]
    return w


def _centres(ra0, dec0, off, fw, fh):
    dra_l = [0.0] + ([-off * fw, off * fw] if off > 0 else [])
    ddec_l = [0.0] + ([-off * fh, off * fh] if off > 0 else [])
    for dra in dra_l:
        for ddec in ddec_l:
            dec_c = float(np.clip(dec0 + ddec, -89.9, 89.9))
            yield ra0 + dra / np.cos(np.radians(dec_c)), dec_c


def _fewest_in_frame(gs, ra0, dec0, sc, W, H, off, fw, fh) -> int:
    """The fewest of the stars ``_match`` keeps in a trial frame (the frame + 15 %), over its trial centres."""
    n = []
    for ra, dec in _centres(ra0, dec0, off, fw, fh):
        px, py = _tan(ra, dec, sc, W, H).world_to_pixel(gs)
        n.append(int(((px > -0.15 * W) & (px < 1.15 * W) & (py > -0.15 * H) & (py < 1.15 * H)).sum()))
    return min(n) if n else 0


def _match(gs, gm, ra0, dec0, sc, W, H, brightdet, off, fw, fh, aa, tol, flips=(True, False), seen=None,
           mcp=100):
    """Triangle match of the brightest detections with the Gaia stars projected at the trial scale and centres:
    (mirror, transform, matched pairs, centre) or None. The likely mirror is tried first and a certain match (>=
    GOOD_PAIRS pairs: the other mirror cannot give one) ends the search: a vain match costs ~35 s of exhausted
    triangle lists, a good one ~1 s. ``seen``: where to put the scales of certain matches rejected for their scale."""
    best = None
    dra_l = [0.0] + ([-off * fw, off * fw] if off > 0 else [])
    ddec_l = [0.0] + ([-off * fh, off * fh] if off > 0 else [])
    for dra in dra_l:
        for ddec in ddec_l:
            dec_c = float(np.clip(dec0 + ddec, -89.9, 89.9))
            w0 = _tan(ra0 + dra / np.cos(np.radians(dec_c)), dec_c, sc, W, H)
            px, py = w0.world_to_pixel(gs)
            inb = (px > -0.15 * W) & (px < 1.15 * W) & (py > -0.15 * H) & (py < 1.15 * H)
            if inb.sum() < 20:
                continue
            order = np.argsort(gm[inb], kind="stable")
            gpos = np.c_[px[inb], py[inb]][order][:150]
            for flip in flips:
                q = brightdet.copy()
                if flip:
                    q[:, 1] = H - 1 - q[:, 1]
                try:
                    T, (a, _) = aa.find_transform(q, gpos, max_control_points=mcp)
                except Exception:  # noqa: BLE001 - no match for this hypothesis
                    continue
                if abs(np.log(T.scale)) < tol and (best is None or len(a) > best[2]):
                    best = (flip, T, len(a), list(w0.wcs.crval))
                elif seen is not None and len(a) >= GOOD_PAIRS and 0.2 < T.scale < 5:
                    seen.append(float(sc * T.scale))
                if best is not None and best[2] >= GOOD_PAIRS:
                    break
        if best is not None and best[2] >= GOOD_PAIRS:
            break
    return best


def log_odds(pred, cat, s, t, tree, W, H, used) -> tuple[float, int]:
    """(natural log of the Bayes factor of a solution, stars matched) by the decision of Lang et al. 2010 (section 2.4):
    against the hypothesis that the stars fall anywhere in the image (1 / area), each star either has no counterpart
    (DISTRACTORS of them) or lies in a Gaussian of VERIFY_PX around one of the catalogue stars the solution puts in the
    frame. ``pred``: the stars' positions through the solution, complex; ``cat``: the catalogue's; ``used``: the three
    stars that proposed the solution, which prove nothing and are left out of the product (they still count as
    matched); a catalogue star is taken once, by its nearest star. Every other star counts, so that stars that match
    nothing weigh against the solution."""
    z = (cat - t) / s
    n_ref = int(((z.real >= 0) & (z.real < W) & (z.imag >= 0) & (z.imag < H)).sum())
    if n_ref == 0:
        return -np.inf, 0
    rest = np.delete(pred, used)
    d, j = tree.query(np.c_[rest.real, rest.imag], distance_upper_bound=3 * VERIFY_PX)
    hit = np.where(np.isfinite(d))[0]
    hit = hit[np.argsort(d[hit], kind="stable")]
    hit = hit[np.unique(j[hit], return_index=True)[1]]
    fg = (1 - DISTRACTORS) / n_ref * np.exp(-d[hit] ** 2 / (2 * VERIFY_PX ** 2)) / (2 * np.pi * VERIFY_PX ** 2)
    lo = float(np.log(DISTRACTORS + W * H * fg).sum() + (len(rest) - len(hit)) * np.log(DISTRACTORS))
    return lo, len(hit) + len(used)


def _match_sparse(gs, gm, ra0, dec0, sc, W, H, det, off, fw, fh, tol, flips):
    """The field of a stack of few stars on the catalogue: (mirror, transform, matched stars, centre, log odds) or None.
    Every triple of its brightest stars and of the catalogue's (the triangle matching of Groth 1986 and Valdes et al.
    1995: the nearest-neighbour triangles of the frames' registration need two lists of the same stars, which a list
    of detections and a catalogue are not) proposes a similarity; the one the stars prove best stands if its Bayes
    factor reaches ODDS_TO_SOLVE (``log_odds``). No random draw: the same stars give the same solution."""
    from scipy.spatial import cKDTree
    from skimage.transform import SimilarityTransform

    from .register import triangle_similarities
    for ra, dec in _centres(ra0, dec0, off, fw, fh):
        w0 = _tan(ra, dec, sc, W, H)
        px, py = w0.world_to_pixel(gs)
        inb = (px > -0.15 * W) & (px < 1.15 * W) & (py > -0.15 * H) & (py < 1.15 * H)
        if inb.sum() < SPARSE_MIN_STARS:
            continue
        gpos = np.c_[px[inb], py[inb]][np.argsort(gm[inb], kind="stable")]
        gz = gpos[:, 0] + 1j * gpos[:, 1]
        tree = cKDTree(gpos)
        for flip in flips:
            q = det.copy()
            if flip:
                q[:, 1] = H - 1 - q[:, 1]
            sim, tra, tri = triangle_similarities(q[:SPARSE_TRIANGLE_STARS], gpos[:2 * SPARSE_TRIANGLE_STARS],
                                                  nearest=False, max_scale=tol)
            zq = q[:, 0] + 1j * q[:, 1]
            best = None
            for s, t, used in zip(sim, tra, tri):
                lo, nm = log_odds(s * zq + t, gz, s, t, tree, W, H, used)
                if best is None or lo > best[0]:
                    best = (lo, nm, s, t)
            if best is not None and best[0] >= np.log(ODDS_TO_SOLVE):
                lo, nm, s, t = best
                T = SimilarityTransform(matrix=np.array([[s.real, -s.imag, t.real], [s.imag, s.real, t.imag], [0, 0, 1]]))
                return flip, T, nm, list(w0.wcs.crval), lo
    return None


_GAIA_CACHE: dict = {}
_GAIA_LOCK = threading.Lock()


def bright_stars(ra, dec, radius, glim, log=None) -> tuple[dict, str]:
    """Gaia DR3 stars brighter than glim within radius (deg) of (ra, dec), epoch 2016.0: ({"RA_ICRS", "DE_ICRS",
    "Gmag"}, where they came from). The offline astrometric extract when it is installed (local, instant; it keeps the 127
    brightest stars of every 14' cell, which in the Milky Way can stop short of glim: plenty for a solution), else
    the sky tiles of the cache, else the network (tiles downloaded and cached)."""
    from ..core import sirilcat, skycache
    s = sirilcat.astro_cone(ra, dec, radius, glim)
    if s is not None:
        src = f"the offline Gaia extract ({sirilcat.locate()['astro'].path.name})"
        if s["complete_to"] < glim:
            src += f", complete to G {s['complete_to']:.1f} here"
        return s, src
    t = skycache.gaia_stars([(ra, dec, radius)], ["RA_ICRS", "DE_ICRS", "Gmag"], f"<{glim:.2f}", log=log)
    return t, "the Gaia sky tiles (cache, else network)"


def gaia_bright(ra, dec, radius, glim, log=None):
    """Gaia DR3 stars brighter than glim within radius (deg) of (ra, dec): (SkyCoord, G), brightest first."""
    from astropy.coordinates import SkyCoord
    key = (round(ra, 2), round(dec, 2), round(radius, 1), round(glim, 1))
    with _GAIA_LOCK:
        hit = _GAIA_CACHE.get(key)
    if hit is None:
        t, src = bright_stars(ra, dec, radius, float(f"{glim:.2f}"), log)
        ok = np.isfinite(t["RA_ICRS"]) & np.isfinite(t["DE_ICRS"])
        ra_, dec_, g = (np.asarray(t[k], np.float64)[ok] for k in ("RA_ICRS", "DE_ICRS", "Gmag"))
        o = np.lexsort((dec_, ra_, g))              # one order whatever the source
        hit = (SkyCoord(ra_[o], dec_[o], unit="deg"), g[o])
        with _GAIA_LOCK:
            _GAIA_CACHE[key] = hit
        if log is not None:
            from ..core import online
            online.sayer(log)(f"Gaia stars for the astrometry from {src}", "info")
    return hit


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+]", "", s.lower().replace("nebula", "").replace("galaxy", "")) or s.lower()


#: the names made comparable, kept with the index they come from (issue #115: made again at each call, 15 000 names,
#: 25 ms each time the graph drew its Project block, six times as a project opened). Read only by its callers
_LOCAL: dict = {"index": None, "names": None}
_LOCAL_LOCK = threading.Lock()


def _local_names() -> dict[str, tuple[float, float]]:
    """Object names -> (ra, dec) from the catalogues shipped with the package (core/skyobjects.names_index: Messier,
    NGC, IC, Sharpless with their common names such as 'Whirlpool Galaxy', and the Lynds dark nebulae). Made once per
    index: a new catalogue (``skyobjects.reset``, other files) makes it again."""
    from ..core import skyobjects
    out: dict[str, tuple[float, float]] = {}
    try:
        index = skyobjects.names_index()
    except skyobjects.CatalogueUnavailable:
        return out
    with _LOCAL_LOCK:
        if _LOCAL["index"] is index:
            return _LOCAL["names"]
        for n, (ra, dec, _d) in index.items():
            out.setdefault(_norm(n), (ra, dec))
        _LOCAL.update(index=index, names=out)
    return out


def find_offline(name: str) -> tuple[float, float] | None:
    """(ra, dec) of a named object without the network: the names resolved before (the cache) and the catalogues
    shipped with the package; None when neither knows it (New project checks a target with it, issue #38)."""
    from ..core import skycache
    if not name or not name.strip():
        return None
    p = skycache.find("names.json")
    if p is not None:
        try:
            known = json.loads(p.read_text(encoding="utf-8"))
            if _norm(name) in known:
                return tuple(float(v) for v in known[_norm(name)][:2])
        except (OSError, ValueError):
            pass
    local = _local_names()
    for c in dict.fromkeys([name, name.replace("_", " ")]):
        if _norm(c) in local:
            ra, dec = local[_norm(c)]
            return float(ra), float(dec)
    return None


def resolve_name(name: str, log=None) -> tuple[float, float]:
    """(ra, dec) of a named object: names resolved before (cache), the shipped catalogues, SIMBAD, NED (each service
    given a few seconds; one line per fallback)."""
    from ..core import online, skycache
    say = online.sayer(log)
    candidates = list(dict.fromkeys([name, name.replace("_", " "),
                                     name.replace("_nebula", " Nebula").replace("_", " ")]))
    key = _norm(name)
    p = skycache.find("names.json")
    known = {}
    if p is not None:
        try:
            known = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            known = {}
    if key in known:
        return tuple(known[key][:2])

    def remember(ra, dec, src):
        try:
            cur = skycache.cache_dir() / "names.json"
            d = json.loads(cur.read_text(encoding="utf-8")) if cur.exists() else {}
            d[key] = [ra, dec, src]
            skycache.save_json(cur, d)
        except OSError:
            pass
        return ra, dec

    local = _local_names()
    for c in candidates:
        if _norm(c) in local:
            ra, dec = local[_norm(c)]
            say(f"'{name}' found in the deep-sky catalogues: RA {ra:.4f} Dec {dec:.4f}", "info")
            return float(ra), float(dec)

    def simbad():
        from astroquery.simbad import Simbad
        s = Simbad()
        s.TIMEOUT = online.READ_S
        for c in candidates:
            t = s.query_object(c)
            if t is not None and len(t):
                import astropy.units as u
                from astropy.coordinates import SkyCoord
                cols = {k.lower(): k for k in t.colnames}
                if "ra" in cols and np.issubdtype(np.asarray(t[cols["ra"]]).dtype, np.floating):
                    return float(t[cols["ra"]][0]), float(t[cols["dec"]][0])
                sc = SkyCoord(t[cols.get("ra", "RA")][0], t[cols.get("dec", "DEC")][0], unit=(u.hourangle, u.deg))
                return float(sc.ra.deg), float(sc.dec.deg)
        return None

    def ned():                      # NED (NASA/IPAC) when SIMBAD (CDS) does not answer
        from astroquery.ipac.ned import Ned
        for c in candidates:
            try:
                t = Ned.query_object(c)
            except Exception as e:  # noqa: BLE001 - an unknown name raises too: the next spelling
                if online.network_error(e):
                    raise
                continue
            if t is not None and len(t):
                return float(t["RA"][0]), float(t["DEC"][0])
        return None

    for svc, fn in (("simbad", simbad), ("ned", ned)):
        try:
            r = online.call(svc, fn, 20.0)
        except online.ServiceDown as e:
            say(f"{online.label(svc)} {e.why}: the name '{name}' is looked up elsewhere")
            continue
        except Exception:  # noqa: BLE001 - this service failed on the name: the next one
            continue
        if r is not None:
            return remember(r[0], r[1], svc)
    raise StageError(f"'{name}' is not in the deep-sky catalogues and neither SIMBAD nor NED could give it: enter the "
                     "RA/Dec of the field centre")
