"""Catalogue pre-fetch for the Base runs: the Gaia cones of a Base project built from the ones its Express run
downloaded, written into the bench cache (bench/_home/.astro_gpu/cache), so that the Base catalogs stage reads them
instead of the network.

    python tools/bench/prefetch.py CASE [--from express] [--to base]

Why: the catalogs stage caches each cone under the exact cell centre (5 decimals) and radius, and the cells follow the
stack's astrometric solution. The Base stack is solved a little differently from the Express one (another reference
frame, other stars), so none of its cones would be found in the cache, and a wide or dense field (Heart: 54 cells,
M8) takes the better part of an hour from the Gaia services when VizieR is down.

How: the Express cones of the case (recomputed from the Express solution, read from the cache) are merged; for every
Base cone entirely inside them, the stars within its radius are written under the key the stage will ask for (the
same columns and magnitude filter). A Base cone that sticks out of the Express cover is left alone: the stage
downloads it. The Hipparcos cone (bright stars missing from Gaia) is handled the same way. XP spectra are cached by
Gaia id: they are shared already. Run it after the Base astrometry, before its catalogs stage (ours.py --stop-before
catalogs, then this, then ours.py again).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

C.engine_home()
sys.path.insert(0, str(C.REPO))
C.utf8_console()

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from astro_gpu.stages.catalogs import BRIGHT_G  # noqa: E402

#: the columns of the field's stars (stages/catalogs.py: the bright stars, G < BRIGHT_G, when the offline Gaia extract
#: is not there)
COLS = ["Source", "RA_ICRS", "DE_ICRS", "Gmag", "BPmag", "RPmag", "pmRA", "pmDE", "XPsamp"]
HIP_COLS = ["HIP", "RAICRS", "DEICRS", "Vmag", "B-V", "pmRA", "pmDE"]


def _sep(ra1, dec1, ra2, dec2):
    c = np.radians
    v = np.sin(c(dec1)) * np.sin(c(dec2)) + np.cos(c(dec1)) * np.cos(c(dec2)) * np.cos(c(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(v, -1, 1)))


def field(case: str, level: str) -> dict:
    """The cells the catalogs stage of this project computes (same code, same inputs)."""
    from astro_gpu import api
    from astro_gpu.stages.catalogs import CatalogsParams, _cells
    p = api.open_project(C.BENCH / case / f"ours_{level}")
    w = next(w for w in p.workflows if w.name == "Classic stack")
    ast = api.stage_outputs(p, w.id, "astrometry")
    stk = api.stage_outputs(p, w.id, "stack")
    if ast is None or stk is None:
        raise SystemExit(f"{case} {level}: astrometry not done")
    wcs = WCS(fits.getheader(Path(ast.dir) / "wcs.fits"))
    meta = json.loads((Path(stk.dir) / "drz_meta.json").read_text(encoding="utf-8"))
    W, H = meta["width"], meta["height"]
    scale = float(ast.metrics["scale_arcsec"])
    pr = CatalogsParams()
    cat = api.stage_outputs(p, w.id, "catalogs")
    if cat is not None and cat.params:
        pr = CatalogsParams(**{k: v for k, v in cat.params.items() if k in CatalogsParams.model_fields})
    centres, radius = _cells(wcs, W, H, scale, pr.cell_deg)
    centre = wcs.pixel_to_world(W / 2, H / 2)
    hip_r = 0.5 * np.hypot(W, H) * scale / 3600 * 1.05
    return {"centres": centres, "radius": radius, "p": pr, "hip": (centre.ra.deg, centre.dec.deg, hip_r),
            "wcs": wcs, "W": W, "H": H}


def gaia_key(cols, gfilter, c, radius) -> str:
    return hashlib.sha1(f"gaiadr3|{','.join(cols)}|{gfilter}|{c[0]:.5f}|{c[1]:.5f}|{radius:.4f}".encode()).hexdigest()


def hip_key(ra, dec, radius, vmax=9.0) -> str:
    return hashlib.sha1(f"hip|{ra:.4f}|{dec:.4f}|{radius:.3f}|{vmax}".encode()).hexdigest()


def covered(c, r, cones: np.ndarray, rads: np.ndarray, fd: dict | None = None) -> bool:
    """Whether the part of the cone (c, r) the stage keeps (the image of the target project and 2 % around it; the
    whole cone without fd) lies inside the union of the cones: sampled on rings and at the centre."""
    pts = [np.asarray(c, float)]
    for rr in (0.25, 0.5, 0.75, 1.0):
        a = np.radians(np.arange(0, 360, 5))
        dec = c[1] + rr * r * np.sin(a)
        ra = c[0] + rr * r * np.cos(a) / np.maximum(np.cos(np.radians(dec)), 1e-3)
        pts += list(np.c_[ra, dec])
    pts = np.array(pts)
    if fd is not None:
        x, y = fd["wcs"].all_world2pix(pts[:, 0], pts[:, 1], 0)
        m = 0.02 * max(fd["W"], fd["H"])
        keep = (x > -m) & (y > -m) & (x < fd["W"] + m) & (y < fd["H"] + m)
        pts = pts[keep]
    for x, y in pts:
        if not np.any(_sep(x, y, cones[:, 0], cones[:, 1]) <= rads - 1e-6):
            return False
    return True


def _cone_of(ra: np.ndarray, dec: np.ndarray) -> tuple[float, float, float]:
    """The centre and (a lower bound of the) radius of a cached cone, from its stars."""
    r, d = np.radians(ra), np.radians(dec)
    v = np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)].mean(0)
    c_ra = float(np.degrees(np.arctan2(v[1], v[0])) % 360)
    c_dec = float(np.degrees(np.arcsin(v[2] / np.linalg.norm(v))))
    return c_ra, c_dec, float(np.max(_sep(ra, dec, c_ra, c_dec)))


def pool_cones(case: str, dst: str) -> int:
    """Every cone file of the cache that touches the field (whatever solution it was cut for: an earlier run of the
    bench) pooled, and the target project's cones written from it where the pool covers them."""
    cache = C.AG_HOME / "cache"
    fd = field(case, dst)
    centre = fd["wcs"].pixel_to_world(fd["W"] / 2, fd["H"] / 2)
    reach = 0.5 * np.hypot(fd["W"], fd["H"]) * float(np.abs(np.linalg.det(fd["wcs"].pixel_scale_matrix)) ** 0.5) + 1.5
    report = {}
    for label, cols, gf in (("bright", COLS, f"<{BRIGHT_G:g}"),):
        parts, cones = [], []
        for f in (cache / "gaia").glob("*.npz"):
            try:
                z = dict(np.load(f))
            except (OSError, ValueError):
                continue
            if set(z) != set(cols) or not len(z["Gmag"]):
                continue
            c = _cone_of(z["RA_ICRS"], z["DE_ICRS"])
            if _sep(c[0], c[1], centre.ra.deg, centre.dec.deg) > reach + c[2]:
                continue
            parts.append(z)
            cones.append(c)
        if not parts:
            report[label] = "no cached cone near the field"
            continue
        pl = {k: np.concatenate([q[k] for q in parts]) for k in cols}
        key = np.round(pl["RA_ICRS"] * 3.6e5).astype(np.int64) * 10 ** 9 + np.round(pl["DE_ICRS"] * 3.6e5).astype(np.int64)
        _, idx = np.unique(key, return_index=True)
        pl = {k: v[idx] for k, v in pl.items()}
        cc = np.array([[c[0], c[1]] for c in cones])
        rr = np.array([c[2] for c in cones])
        made = kept = out = 0
        for c in fd["centres"]:
            f = cache / "gaia" / f"{gaia_key(cols, gf, c, fd['radius'])}.npz"
            if f.exists():
                kept += 1
                continue
            if not covered(c, fd["radius"], cc, rr, fd):
                out += 1
                continue
            sel = _sep(pl["RA_ICRS"], pl["DE_ICRS"], c[0], c[1]) <= fd["radius"]
            np.savez(f, **{k: v[sel] for k, v in pl.items()})
            made += 1
        report[label] = {"cached_cones_used": len(parts), "pool_stars": int(len(idx)), "made": made, "already": kept,
                         "left_to_network": out}
    print(json.dumps({"case": case, "to": dst, "pool": True, **report}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=list(C.CASES))
    ap.add_argument("--from", dest="src", default="express")
    ap.add_argument("--to", dest="dst", default="base")
    ap.add_argument("--pool", action="store_true", help="cut the target's cones from every cached cone of the field "
                                                       "(earlier runs of the bench), not from one project's")
    a = ap.parse_args()
    if a.pool:
        return pool_cones(a.case, a.dst)
    cache = C.AG_HOME / "cache"
    fs, fd = field(a.case, a.src), field(a.case, a.dst)
    report = {}
    for label, cols, gf in (("bright", COLS, f"<{BRIGHT_G:g}"),):
        parts = []
        for c in fs["centres"]:
            f = cache / "gaia" / f"{gaia_key(cols, gf, c, fs['radius'])}.npz"
            if not f.exists():
                raise SystemExit(f"{a.case} {a.src}: the {label} cone {c} is not in the cache: run {a.src} first")
            parts.append(dict(np.load(f)))
        pool = {k: np.concatenate([q[k] for q in parts]) for k in cols}
        key = np.round(pool["RA_ICRS"] * 3.6e5).astype(np.int64) * 10 ** 9 + \
            np.round(pool["DE_ICRS"] * 3.6e5).astype(np.int64)
        _, idx = np.unique(key, return_index=True)
        pool = {k: v[idx] for k, v in pool.items()}
        made = kept = out = 0
        rads = np.full(len(fs["centres"]), fs["radius"])
        for c in fd["centres"]:
            f = cache / "gaia" / f"{gaia_key(cols, gf, c, fd['radius'])}.npz"
            if f.exists():
                kept += 1
                continue
            if not covered(c, fd["radius"], fs["centres"], rads, fd):
                out += 1
                continue
            sel = _sep(pool["RA_ICRS"], pool["DE_ICRS"], c[0], c[1]) <= fd["radius"]
            np.savez(f, **{k: v[sel] for k, v in pool.items()})
            made += 1
        report[label] = {"pool_stars": int(len(idx)), "made": made, "already": kept, "left_to_network": out}
    # Hipparcos
    ra, dec, r = fs["hip"]
    f = cache / "hipparcos" / f"{hip_key(ra, dec, r)}.npz"
    rd, dd, r2 = fd["hip"]
    g = cache / "hipparcos" / f"{hip_key(rd, dd, r2)}.npz"
    if g.exists():
        report["hipparcos"] = "already"
    elif f.exists() and covered((rd, dd), r2, np.array([[ra, dec]]), np.array([r]), fd):
        t = dict(np.load(f))
        sel = _sep(t["RAICRS"], t["DEICRS"], rd, dd) <= r2
        np.savez(g, **{k: v[sel] for k, v in t.items()})
        report["hipparcos"] = "made"
    else:
        report["hipparcos"] = "left to the network"
    print(json.dumps({"case": a.case, "from": a.src, "to": a.dst, **report}), flush=True)
    log = C.BENCH / a.case / f"prefetch_{a.dst}.json"
    log.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
