"""StarNet against our star removal on the bench cases: five measures and side-by-side sheets
(docs/chantiers/etoiles_etat_de_lart.md §3; the samples: tools/bench/stars_samples.py).

    python tools/bench/stars_compare.py [--cases heart,iris,...]

Every method starts from the same linear image O (``stars_samples.py export``: the denoised stack minus its gradient,
times the white balance) and gives a starless image S in the same units; the star layer is O - S (linear, not
clipped). The methods: ``starnet`` (StarNet 2, ``--linear``), ``ours_model`` (our Gaia star model subtracted, then
denoised: the starless stage's ``n2n_final``), ``ours_neural`` (our per-image star removal network's layer subtracted:
``n2n_final_learned``), each times the same white balance and scale as O.

The measures (the published tests of the state of the art, transposed to images whose truth is unknown):
1. **Star light left** (the injection test of Madarász et al. 2025, done on the real stars: our method only removes
   catalogued stars, so injected ones would say nothing about it): for each Gaia star of the field, the flux left in
   S over its flux in O, in the same aperture with each image's own local background; the median by magnitude, on the
   sky and on nebula (the stars whose local background is in the top fifth of the field), and the share of "holes"
   (more than 10 % of the star's flux taken below the background).
2. **Off the stars, nothing changes** (Casassus et al. 2026, appendix C): far from every star, the straight line
   S = a O + b (both slightly smoothed): a should be 1; the median of S - O, in noise units of O.
3. **Texture where the stars were** (power spectrum / histogram under the masks, Madarász et al. 2025): for stars
   brighter than G 14, the fine-scale spread of S inside the star's disc over the one in a ring around it (1: the
   filled area looks like its surroundings; below 1: a smooth blotch; above: residual structure).
4. **The two layers give back the original** (exactness): for StarNet, its own star layer (unscreen U: O = S +
   U (1 - S)) recombined against O, the 99.9th percentile of the error in noise units; our engine's layers are O - S
   by construction (0).
5. **Compact objects stay** (maskfill / photutils practice): the named galaxies and planetary nebulae smaller than 3'
   in the field (offline catalogue), their flux in S over their flux in O (1: kept; 0: taken as a star).

Sheets: F:/astro_essais/stars_compare/<case>_sheet.jpg (three areas: the brightest star, the brightest nebula, a
plain star field; columns: original, then each method's starless image; one stretch for all) and
<case>_stars.jpg (the star layers O - S, same stretch). Results: F:/astro_essais/stars_compare/results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BENCH = Path("F:/astro_essais/bench")
OUT = Path("F:/astro_essais/stars_compare")
CASES = ("heart", "iris", "m51", "m8", "rosette", "seestar")
SOURCE = "Classic stack · denoise"
OURS = {"ours_model": ("Full · denoise", "n2n_final.npy"),
        "ours_neural": ("Full + neural stars · denoise", "n2n_final_learned.npy")}
BINS = [(-5, 10), (10, 12), (12, 14), (14, 16), (16, 18.5)]
LABELS = {"original": "original", "starnet": "StarNet", "ours_model": "ours: Gaia model", "ours_neural": "ours: neural",
          "a_telea": "A: mask + fill"}


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def _api():
    import os
    os.environ.setdefault("ASTRO_GPU_HOME", str(BENCH / "_state_probe"))
    from astro_gpu import api
    return api


def read_fits(path: Path) -> np.ndarray:
    from astropy.io import fits
    with fits.open(path) as h:
        d = np.asarray(h[0].data, np.float32)
    return d if d.shape[0] == 3 else np.moveaxis(d, -1, 0)


def load_case(api, case: str):
    """O, the starless images by method, the Gaia stars (x, y, G), the WCS, the star FWHM (px)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    d = OUT / case
    meta = json.loads((d / "meta.json").read_text())
    O = read_fits(d / "input.fits")
    S = {}
    if (d / "starnet_starless.fits").exists():
        S["starnet"] = read_fits(d / "starnet_starless.fits")
    if (d / "a_telea_starless.fits").exists():                     # path A without its network (stars_mask_fill.py)
        S["a_telea"] = read_fits(d / "a_telea_starless.fits")
    p = api.open_project(BENCH / case / "ours_express")
    src = next(w for w in p.workflows if w.name == SOURCE)
    col = api.stage_outputs(p, src.id, "color")
    M = np.load(Path(col.dir) / "M.npy") if col is not None else np.eye(3)
    for key, (wname, fname) in OURS.items():
        w = next((x for x in p.workflows if x.name == wname), None)
        o = api.stage_outputs(p, w.id, "starless") if w is not None else None
        if o is not None and (Path(o.dir) / fname).exists():
            S[key] = np.einsum("ij,jhw->ihw", M, np.load(Path(o.dir) / fname).astype(np.float32)) / meta["scale"]
    cat = api.stage_outputs(p, src.id, "catalogs")
    stars = None
    if cat is not None:
        z = np.load(Path(cat.dir) / "gaia_deep.npz")
        stars = (np.asarray(z["x"], float), np.asarray(z["y"], float), np.asarray(z["g"], float))
    ast = api.stage_outputs(p, src.id, "astrometry")
    wcs = WCS(fits.getheader(Path(ast.dir) / "wcs.fits")) if ast is not None else None
    fwhm = float(api.stage_outputs(p, src.id, "register").metrics.get("fwhm_px") or 4.0)
    return O, S, stars, wcs, fwhm, meta


def lum(x):
    return x.mean(0)


def noise_of(O) -> float:
    L = lum(O)
    hp = L - cv2.GaussianBlur(L, (0, 0), 1.5)
    return float(1.4826 * np.median(np.abs(hp[::3, ::3] - np.median(hp[::3, ::3]))))


def aperture_flux(img, x, y, r):
    """Flux in a disc of radius r minus the median of a ring 2r..3r, on the luminance, for each (x, y)."""
    L = lum(img)
    H, W = L.shape
    R = int(np.ceil(3 * r)) + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy)
    disc, ring = rr <= r, (rr >= 2 * r) & (rr <= 3 * r)
    out = np.full(len(x), np.nan)
    for i, (cx, cy) in enumerate(zip(np.round(x).astype(int), np.round(y).astype(int))):
        if cx - R < 0 or cy - R < 0 or cx + R >= W or cy + R >= H:
            continue
        patch = L[cy - R:cy + R + 1, cx - R:cx + R + 1]
        out[i] = float(patch[disc].sum() - np.median(patch[ring]) * disc.sum())
    return out


def star_light_left(O, Sm, stars, fwhm, sigma) -> dict:
    x, y, g = stars
    r = max(3.0, 1.5 * fwhm)
    fo = aperture_flux(O, x, y, r)
    fs = aperture_flux(Sm, x, y, r)
    big = cv2.blur(lum(O), (65, 65))
    H, W = big.shape
    xi, yi = np.clip(np.round(x).astype(int), 0, W - 1), np.clip(np.round(y).astype(int), 0, H - 1)
    bgl = big[yi, xi]
    neb = bgl >= np.nanpercentile(bgl, 80)
    ok = np.isfinite(fo) & np.isfinite(fs) & (fo > 10 * sigma * np.sqrt(np.pi) * r)    # stars well above the noise
    out = {}
    for lo, hi in BINS:
        for where, sel in (("sky", ~neb), ("nebula", neb)):
            m = ok & sel & (g >= lo) & (g < hi)
            if m.sum() < 5:
                continue
            frac = fs[m] / fo[m]
            out[f"G{lo if lo > 0 else ''}-{hi} {where}"] = {"stars": int(m.sum()),
                                                            "left_pct": round(100 * float(np.median(frac)), 1),
                                                            "holes_pct": round(100 * float((frac < -0.1).mean()), 1)}
    return out


def star_mask(O, stars, fwhm):
    """The stars and their surroundings: Gaia stars (radius growing with brightness) and what stands out of the
    luminance's local median, widened."""
    from scipy.ndimage import median_filter
    L = lum(O)
    H, W = L.shape
    m = np.zeros((H, W), np.uint8)
    if stars is not None:
        x, y, g = stars
        for cx, cy, gg in zip(x, y, g):
            rad = int(2 * fwhm + max(0.0, 15 - gg) * 4)
            if 0 <= cx < W and 0 <= cy < H:
                cv2.circle(m, (int(cx), int(cy)), rad, 1, -1)
    sm = cv2.GaussianBlur(L, (0, 0), 1.0)
    small = median_filter(sm[::2, ::2], 9)
    bump = sm - cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    s = noise_of(O)
    m |= (bump > 5 * s).astype(np.uint8)
    return cv2.dilate(m, np.ones((7, 7), np.uint8)) > 0


def off_stars(O, Sm, mask, sigma) -> dict:
    a, b = cv2.GaussianBlur(lum(O), (0, 0), 2), cv2.GaussianBlur(lum(Sm), (0, 0), 2)
    keep = ~mask
    keep[:40], keep[-40:], keep[:, :40], keep[:, -40:] = False, False, False, False
    xa, xb = a[keep].astype(np.float64), b[keep].astype(np.float64)       # centred, in double: the values are close
    ca, cb = xa - xa.mean(), xb - xb.mean()
    slope = float((ca * cb).sum() / (ca * ca).sum())
    d = float(np.median(xb - xa))
    return {"slope": round(slope, 4), "median_diff_sigma": round(d / sigma, 2),
            "median_diff_pct_of_sky": round(100 * d / float(np.median(xa)), 3),
            "pixels_pct": round(100 * float(keep.mean()), 1)}


def texture(Sm, stars, fwhm) -> dict:
    if stars is None:
        return {}
    x, y, g = stars
    L = lum(Sm)
    hp = L - cv2.GaussianBlur(L, (0, 0), 1.5)
    H, W = L.shape
    sel = g < 14
    ratios = []
    for cx, cy, gg in zip(x[sel], y[sel], g[sel]):
        r = int(max(4, 1.5 * fwhm + max(0.0, 12 - gg) * 2))
        cx, cy = int(round(cx)), int(round(cy))
        if cx - 2 * r < 0 or cy - 2 * r < 0 or cx + 2 * r >= W or cy + 2 * r >= H:
            continue
        p = hp[cy - 2 * r:cy + 2 * r + 1, cx - 2 * r:cx + 2 * r + 1]
        yy, xx = np.mgrid[-2 * r:2 * r + 1, -2 * r:2 * r + 1]
        rr = np.hypot(xx, yy)
        inner, ring = p[rr <= r], p[(rr > 1.3 * r) & (rr <= 2 * r)]
        si, so = np.std(inner), 1.4826 * np.median(np.abs(ring - np.median(ring)))
        if so > 0:
            ratios.append(si / so)
    if not ratios:
        return {}
    q = np.percentile(ratios, [10, 50, 90])
    return {"stars": len(ratios), "ratio_p10": round(float(q[0]), 2), "ratio_median": round(float(q[1]), 2),
            "ratio_p90": round(float(q[2]), 2)}


def compact_objects(O, S, wcs) -> list:
    if wcs is None:
        return []
    from astro_gpu.core import skyobjects
    H, W = O.shape[1:]
    try:
        objs = skyobjects.objects_in_field(wcs, W, H)
    except Exception:  # noqa: BLE001
        return []
    scale = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600   # arcsec per px
    out = []
    for o in objs:
        if o.kind not in ("galaxy", "planetary nebula") or not 0 < o.major <= 3:
            continue
        x, y = wcs.all_world2pix([[o.ra, o.dec]], 0)[0]
        if not (20 < x < W - 20 and 20 < y < H - 20):
            continue
        r = max(4.0, 0.5 * o.major * 60 / scale)
        fo = aperture_flux(O, np.array([x]), np.array([y]), r)[0]
        if not np.isfinite(fo) or fo <= 0:
            continue
        out.append({"name": o.name, "kind": o.kind, "size_arcmin": o.major,
                    **{k: round(float(aperture_flux(v, np.array([x]), np.array([y]), r)[0] / fo), 2)
                       for k, v in S.items()}})
    return out


def exactness(case: str, O, S_sn, sigma) -> dict:
    f = OUT / case / "starnet_unscreen.fits"
    if S_sn is None or not f.exists():
        return {}
    U = read_fits(f)
    err = np.abs(S_sn + U * (1 - S_sn) - O)
    return {"starnet_p999_sigma": round(float(np.percentile(err, 99.9)) / sigma, 2),
            "starnet_max_sigma": round(float(err.max()) / sigma, 1),
            "ours": "O - S by construction: exact"}


def sheets(case: str, O, S, stars) -> None:
    L = lum(O)
    H, W = L.shape
    bk = np.median(O.reshape(3, -1)[:, ::7], axis=1)[:, None, None]             # the sky of each colour: neutral
    s = noise_of(O) * 3
    top = np.arcsinh(float(np.percentile(O - bk, 99.95)) / s)

    def show(img):
        v = np.arcsinh(np.clip((img - bk) / s, -1, None)) / top
        return (np.clip(v, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)

    big = cv2.blur(L, (65, 65))
    C = 360
    pts = []
    if stars is not None:                                                          # the brightest Gaia star
        x, y, g = stars
        inside = (x > C) & (x < W - C) & (y > C) & (y < H - C)
        i = int(np.argmin(np.where(inside, g, np.inf)))
        pts.append((int(y[i]), int(x[i])))
    else:
        sm = cv2.GaussianBlur(L, (0, 0), 3)
        yb, xb = np.unravel_index(np.argmax(sm[C:-C, C:-C]), (H - 2 * C, W - 2 * C))
        pts.append((yb + C, xb + C))
    yn, xn = np.unravel_index(np.argmax(big[C:-C, C:-C]), (H - 2 * C, W - 2 * C))
    pts.append((yn + C, xn + C))                                                   # the brightest nebula
    pts.append((H // 2 + H // 5, W // 2 - W // 5))                                 # a plain star field
    cols = [("original", O)] + list(S.items())
    for name, kind in (("sheet", "image"), ("stars", "layer")):
        rows = []
        for (cy, cx) in pts:
            y0, x0 = int(np.clip(cy - C // 2, 0, H - C)), int(np.clip(cx - C // 2, 0, W - C))
            tiles = []
            for k, img in cols:
                crop = img[:, y0:y0 + C, x0:x0 + C]
                if kind == "layer":
                    crop = (O[:, y0:y0 + C, x0:x0 + C] - crop) + bk if k != "original" else crop
                t = np.ascontiguousarray(show(crop))
                cv2.putText(t, LABELS.get(k, k), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                tiles.append(t)
            rows.append(np.hstack(tiles))
        cv2.imwrite(str(OUT / f"{case}_{name}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=",".join(CASES))
    a = ap.parse_args(argv)
    api = _api()
    res_f = OUT / "results.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    for case in a.cases.split(","):
        t = time.time()
        try:
            O, S, stars, wcs, fwhm, meta = load_case(api, case)
        except Exception as e:  # noqa: BLE001
            log(f"{case}: not ready ({type(e).__name__}: {e})")
            continue
        sigma = noise_of(O)
        mask = star_mask(O, stars, fwhm)
        r = {"methods": sorted(S), "fwhm_px": fwhm, "colour": meta.get("colour"),
             "exact": exactness(case, O, S.get("starnet"), sigma),
             "compact_objects": compact_objects(O, S, wcs), "by_method": {}}
        for k, Sm in S.items():
            r["by_method"][k] = {"off_stars": off_stars(O, Sm, mask, sigma), "texture": texture(Sm, stars, fwhm),
                                 "star_light_left": star_light_left(O, Sm, stars, fwhm, sigma) if stars else {}}
        sheets(case, O, S, stars)
        res[case] = r
        res_f.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        log(f"{case}: {sorted(S)} measured in {time.time() - t:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
