"""Aperture photometry of the bench: exact circular apertures, aperture correction, empty-aperture depth.

No side effect on import (measure.py sets the bench's environment; this file only needs numpy and photutils), so the
tests can import it alone (tests/test_bench_aperture.py).

Sources of the method:
- exact apertures: each pixel counts for the fraction of its area inside the circle. SExtractor (Bertin & Arnouts 1996,
  A&AS 117, 393), DAOPHOT (Stetson 1987, PASP 99, 191) and photutils (``aperture_photometry(method="exact")``) do
  the same. A circle of whole pixels jumps from 13 to 21 pixels at r = sqrt(5) = 2.236 px: the noise it collects
  jumps by 25 % with no change in the image;
- aperture correction: the curve of growth of the stars of the image itself (Howell 1989, PASP 101, 616; Stetson 1990,
  PASP 102, 932): the share of the flux of a star that the detection aperture holds;
- depth: the noise of empty apertures (Labbe et al. 2003, AJ 125, 1107; Skelton et al. 2014, ApJS 214, 24, 3.4).
"""
from __future__ import annotations

import numpy as np

MAD = 1.4826


def rstd(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(MAD * np.median(np.abs(v - np.median(v)))) if v.size else float("nan")


def ring_offsets(r1: float, r2: float) -> tuple[np.ndarray, np.ndarray]:
    R = int(np.ceil(r2))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d2 = xx ** 2 + yy ** 2
    m = (d2 >= r1 * r1) & (d2 <= r2 * r2)
    return yy[m], xx[m]


def gather(img: np.ndarray, x: np.ndarray, y: np.ndarray, off: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    H, W = img.shape
    xi = np.clip(np.round(x).astype(int)[:, None] + off[1][None], 0, W - 1)
    yi = np.clip(np.round(y).astype(int)[:, None] + off[0][None], 0, H - 1)
    return img[yi, xi]


def aperture_sums(img: np.ndarray, x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    """Sum of the pixels inside circles of radius ``r`` (pixels) centred on (x, y), each pixel weighted by the exact
    fraction of its area inside the circle (a pixel centre is an integer coordinate): photutils' "exact" method."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size == 0:
        return np.zeros(0)
    from photutils.aperture import CircularAperture, aperture_photometry
    ap = CircularAperture(np.c_[x, y], r=float(r))
    return np.asarray(aperture_photometry(np.asarray(img, np.float32), ap, method="exact")["aperture_sum"].data, float)


def sky_level(img: np.ndarray, x: np.ndarray, y: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
    """Local sky of each star: the median of the pixels of an annulus around it (the usual sky annulus of aperture
    photometry; whole pixels are fine for a median)."""
    return np.median(gather(img, x, y, ring_offsets(r_in, r_out)), 1)


def star_flux(img: np.ndarray, x: np.ndarray, y: np.ndarray, r: float, r_in: float, r_out: float) -> np.ndarray:
    """Background-subtracted flux of stars in an exact circular aperture of radius ``r``: the exact sum minus the
    local sky times the exact area pi r^2 (not a pixel count)."""
    return aperture_sums(img, x, y, r) - sky_level(img, x, y, r_in, r_out) * np.pi * r * r


#: radii (in FWHM) at which the curve of growth is reported
GROWTH_RADII = (0.5, 1.0, 1.5, 2.0)


def aperture_depth(R: np.ndarray, xs: np.ndarray, ys: np.ndarray, g: np.ndarray, ex: np.ndarray, ey: np.ndarray,
                   fwhm_px: float, sigma_ap: float | None = None) -> dict:
    """Zero point, aperture correction and 5-sigma depth by the standard aperture photometry.

    - zero point: G + 2.5 log10(flux in the exact aperture of 2.5 FWHM), median over the stars (xs, ys, g) (sky from
      an annulus of 3.5-5 FWHM);
    - aperture correction (curve of growth): on the same stars of the same image, the share of the 2.5 FWHM flux that
      falls inside the detection aperture (radius max(1.5, 1 FWHM)), and inside smaller ones (``growth``); no assumed
      profile;
    - noise: the robust spread of the flux in empty apertures (ex, ey; exact, radius of the detection aperture);
    - depth: the magnitude whose flux inside the detection aperture is 5 sigma, i.e. whose flux in the 2.5 FWHM
      aperture (the zero point's) is 5 sigma / fraction: zp - 2.5 log10(5 sigma / fraction).
    """
    r_det, r_big = max(1.5, 1.0 * fwhm_px), 2.5 * fwhm_px
    r1, r2 = 3.5 * fwhm_px, 5.0 * fwhm_px
    loc = sky_level(R, xs, ys, r1, r2)
    F = aperture_sums(R, xs, ys, r_big) - loc * np.pi * r_big ** 2
    Fd = aperture_sums(R, xs, ys, r_det) - loc * np.pi * r_det ** 2
    okf = F > 0
    zp = float(np.median(g[okf] + 2.5 * np.log10(F[okf]))) if okf.sum() >= 10 else float("nan")
    ok = okf & (Fd > 0)
    frac = float(np.median(Fd[ok] / F[ok])) if ok.sum() >= 10 else float("nan")
    growth = {}
    for k in GROWTH_RADII:
        rk = k * fwhm_px
        Fk = aperture_sums(R, xs, ys, rk) - loc * np.pi * rk * rk
        okk = okf & (Fk > 0)
        growth[f"{k:g}"] = float(np.median(Fk[okk] / F[okk])) if okk.sum() >= 10 else float("nan")
    sig = rstd(aperture_sums(R, ex, ey, r_det)) if sigma_ap is None else sigma_ap
    good = np.isfinite(zp) and np.isfinite(frac) and frac > 0
    return {"zp": zp, "zp_stars": int(okf.sum()), "aperture_fraction": frac, "growth": growth, "r_det_px": r_det,
            "sigma_ap": sig,
            "depth5_mag": zp - 2.5 * np.log10(5 * sig / frac) if good else float("nan"),
            "depth5_uncorrected_mag": zp - 2.5 * np.log10(5 * sig) if np.isfinite(zp) else float("nan")}
