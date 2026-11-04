"""Path A of the star removal plan, without its network yet: mask the stars, fill the mask (docs/chantiers/etoiles.md).

    python tools/bench/stars_mask_fill.py [--cases rosette,heart]

The mask: the Gaia stars of the field (the catalogue stage's ``gaia_deep.npz``) whose peak measured on the image stands
PEAK_K noise sigmas above the local median, each with the radius where its own profile falls to EDGE_K noise sigmas (a
Moffat profile, beta 2.5, of the registration's FWHM, scaled to the star's peak; at most 80 px), plus the stars Gaia
does not have: the local maxima standing 5 noise sigmas above the local median (a disc of 2 FWHM). The fill: Telea's
fast marching inpainting (Telea 2004, "An image inpainting technique based on the fast marching method", J. Graphics
Tools 9(1); OpenCV ``INPAINT_TELEA``), channel by channel, on the linear image. Outside the mask the image is not
touched.

Input: the common original of ``stars_samples.py`` (F:/astro_essais/stars_compare/<case>/input.fits). Output, beside
it: ``a_telea_starless.fits`` (same units) and ``a_mask.png``; ``stars_compare.py`` measures it as ``a_telea``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "bench"))
import stars_compare as SC  # noqa: E402

BETA = 2.5
MAX_R = 80
#: a star is masked when its peak stands this many noise sigmas above the local median, out to where its profile falls
#: to EDGE_K sigmas (with every Gaia star and the edge at half a sigma, the mask covered 98 % of Heart, 65 % of Rosette)
PEAK_K, EDGE_K = 20.0, 3.0


def star_mask(O, stars, fwhm, sigma):
    L = SC.lum(O)
    H, W = L.shape
    from scipy.ndimage import maximum_filter, median_filter
    small = median_filter(L[::2, ::2], 15)
    bg = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    m = np.zeros((H, W), np.uint8)
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / BETA) - 1))
    x, y, g = stars
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    inside = (xi >= 1) & (yi >= 1) & (xi < W - 1) & (yi < H - 1)
    pk = np.zeros(len(x))
    pk[inside] = np.array([L[b - 1:b + 2, a - 1:a + 2].max() - bg[b, a] for a, b in zip(xi[inside], yi[inside])])
    radii = []
    for a, b, p in zip(xi[inside], yi[inside], pk[inside]):
        if p < PEAK_K * sigma:
            continue
        r = alpha * np.sqrt(max((p / (EDGE_K * sigma)) ** (1 / BETA) - 1, 0))
        r = int(np.clip(np.ceil(max(r, 1.0 * fwhm)), 2, MAX_R))
        radii.append(r)
        cv2.circle(m, (int(a), int(b)), r, 1, -1)
    # the stars Gaia does not have
    sm = cv2.GaussianBlur(L, (0, 0), 1.0)
    bump = sm - bg
    peaks = (bump > 5 * sigma) & (sm == maximum_filter(sm, 5)) & (m == 0)
    py, px = np.nonzero(peaks)
    for a, b in zip(px, py):
        cv2.circle(m, (int(a), int(b)), int(np.ceil(2 * fwhm)), 1, -1)
    return m.astype(bool), {"gaia_stars": int(inside.sum()), "gaia_masked": len(radii), "extra_stars": int(len(px)),
                            "median_radius_px": float(np.median(radii)) if radii else 0.0}


def fill(O, mask, sigma):
    """Telea on a 16-bit coding of each channel (OpenCV's float path filled the mask with a checkered pattern on these
    small linear values): an arcsinh stretch, undone after the fill; outside the mask the original values are kept."""
    out = O.copy()
    m8 = mask.astype(np.uint8)
    for c in range(3):
        bk = float(np.median(O[c][::5, ::5]))
        s = 3 * sigma
        top = float(np.arcsinh((O[c].max() - bk) / s)) + 1e-6
        lo = float(np.arcsinh((O[c].min() - bk) / s))
        y = (np.arcsinh((O[c] - bk) / s) - lo) / (top - lo)
        q = np.round(np.clip(y, 0, 1) * 65535).astype(np.uint16)
        f = cv2.inpaint(q, m8, 3, cv2.INPAINT_TELEA).astype(np.float32) / 65535
        back = bk + s * np.sinh(f * (top - lo) + lo)
        out[c][mask] = back[mask]
    return out


def main(argv=None) -> int:
    from astropy.io import fits
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default="rosette,heart")
    a = ap.parse_args(argv)
    api = SC._api()
    for case in a.cases.split(","):
        t = time.time()
        O, _S, stars, _wcs, fwhm, _meta = SC.load_case(api, case)
        sigma = SC.noise_of(O)
        mask, info = star_mask(O, stars, fwhm, sigma)
        t1 = time.time()
        S = fill(O, mask, sigma)
        d = SC.OUT / case
        fits.PrimaryHDU(S.astype(np.float32)).writeto(d / "a_telea_starless.fits", overwrite=True)
        cv2.imwrite(str(d / "a_mask.png"), (mask * 255).astype(np.uint8))
        info.update({"masked_pct": round(100 * float(mask.mean()), 1), "mask_s": round(t1 - t, 1),
                     "fill_s": round(time.time() - t1, 1)})
        (d / "a_info.json").write_text(json.dumps(info, indent=1))
        SC.log(f"{case}: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
