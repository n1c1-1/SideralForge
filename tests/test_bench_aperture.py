"""Aperture photometry of the bench (tools/bench/aperture.py): exact circular apertures, aperture correction, depth.

The defect these tests guard against: an aperture made of whole pixels jumps from 13 to 21 pixels when the radius
crosses sqrt(5) = 2.236 px, so a star width moving from 2.17 to 2.24 px moved the 5-sigma depth by 0.45 mag with no
real change in the image. The standard method (SExtractor / DAOPHOT / photutils "exact" apertures plus an aperture
correction measured on the stars of the image) varies smoothly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("photutils")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import aperture as AP  # noqa: E402

BETA = 4.0
SIGMA = 1.0
PEDESTAL = 40.0          # a sky level the ring must remove


def alpha_of(fwhm: float, beta: float = BETA) -> float:
    return fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))


def moffat_total(flux: float, fwhm: float, beta: float = BETA):
    """Amplitude of a Moffat star of the given total flux."""
    a = alpha_of(fwhm, beta)
    return flux * (beta - 1) / (np.pi * a * a), a


def add_star(img: np.ndarray, x0: float, y0: float, flux: float, fwhm: float, beta: float = BETA, sub: int = 5) -> None:
    """A Moffat star integrated over each pixel (sub x sub samples per pixel), like a real sensor."""
    A, a = moffat_total(flux, fwhm, beta)
    r = int(np.ceil(8 * fwhm))
    xi, yi = int(round(x0)), int(round(y0))
    off = (np.arange(sub) + 0.5) / sub - 0.5
    yy, xx = np.mgrid[yi - r:yi + r + 1, xi - r:xi + r + 1].astype(float)
    acc = np.zeros_like(xx)
    for dy in off:
        for dx in off:
            acc += (1 + ((xx + dx - x0) ** 2 + (yy + dy - y0) ** 2) / a ** 2) ** (-beta)
    img[yi - r:yi + r + 1, xi - r:xi + r + 1] += A * acc / sub ** 2


def field(fwhm: float, seed: int = 3, n_side: int = 8, size: int = 480):
    """Isolated stars on a jittered grid (fixed positions and fluxes, whatever the FWHM), white noise (the same
    realisation for every FWHM), a pedestal; empty-aperture positions away from every star."""
    rng = np.random.default_rng(seed)
    step = size // n_side
    gy, gx = np.mgrid[step // 2:size:step, step // 2:size:step]
    xy = np.c_[gx.ravel(), gy.ravel()] + rng.uniform(-6, 6, (gx.size, 2))
    flux = 10 ** rng.uniform(4.3, 4.8, len(xy))
    noise = np.random.default_rng(seed + 100).normal(0, SIGMA, (size, size))
    img = PEDESTAL + noise
    for (x0, y0), f in zip(xy, flux):
        add_star(img, x0, y0, f, fwhm)
    g = 20.0 - 2.5 * np.log10(flux)
    erng = np.random.default_rng(seed + 200)
    e = erng.uniform(25, size - 25, (4000, 2))
    d = np.hypot(e[:, None, 0] - xy[None, :, 0], e[:, None, 1] - xy[None, :, 1]).min(1)
    e = e[d > 8 * 2.5 + 6]
    return img, xy, flux, g, e


def depth_of(fwhm: float) -> dict:
    img, xy, flux, g, e = field(fwhm)
    return AP.aperture_depth(img, xy[:, 0], xy[:, 1], g, e[:, 0], e[:, 1], fwhm)


def old_whole_pixel_depth(fwhm: float) -> float:
    """The previous measure (disk of whole pixels, centres rounded, no aperture correction), for contrast."""
    img, xy, flux, g, e = field(fwhm)
    r_det = max(1.5, fwhm)

    def disk(r):
        R = int(np.ceil(r))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        m = xx ** 2 + yy ** 2 <= r * r
        return yy[m], xx[m]
    big, ap, ring = disk(2.5 * fwhm), disk(r_det), AP.ring_offsets(3.5 * fwhm, 5.0 * fwhm)
    loc = np.median(AP.gather(img, xy[:, 0], xy[:, 1], ring), 1)
    F = AP.gather(img, xy[:, 0], xy[:, 1], big).sum(1) - loc * len(big[0])
    zp = float(np.median(g + 2.5 * np.log10(F)))
    sig = AP.rstd(AP.gather(img - np.median(img), np.round(e[:, 0]), np.round(e[:, 1]), ap).sum(1))
    return zp - 2.5 * np.log10(5 * sig)


def test_exact_aperture_area_is_continuous():
    """A flat image of ones: the sum is pi r^2 for any radius and any centre (a pixel count is not)."""
    img = np.ones((60, 60), np.float32)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(20, 40, 50), rng.uniform(20, 40, 50)
    for r in (1.5, 2.0, 2.236, 2.24, 3.7, 5.6):
        s = AP.aperture_sums(img, x, y, r)
        assert np.allclose(s, np.pi * r * r, rtol=2e-4), (r, s.min(), s.max(), np.pi * r * r)
    # the whole-pixel disk is a step function of r: 13 pixels up to sqrt(5), 21 after
    n = lambda r: int((np.mgrid[-4:5, -4:5][0] ** 2 + np.mgrid[-4:5, -4:5][1] ** 2 <= r * r).sum())  # noqa: E731
    assert (n(2.23), n(2.24)) == (13, 21)


@pytest.mark.parametrize("fwhm", [2.0, 2.3, 2.5])
def test_moffat_flux_matches_true_flux(fwhm):
    """Synthetic Moffat stars (beta 4, pixel-integrated, noise, a sky pedestal) of known total flux: the flux measured
    as the bench does matches the true flux within 1 %. The share of the profile beyond 2.5 FWHM (0.5 % for beta = 4)
    is all the zero point's aperture leaves out. The detection aperture's flux, corrected by the curve of growth of
    the other stars, is the same within 1 % on the median star (a single star by up to ~1.5 %: the exact area of a
    small circle depends slightly on where the centre falls in its pixel)."""
    img, xy, flux, g, e = field(fwhm)
    x, y = xy[:, 0], xy[:, 1]
    f_big = AP.star_flux(img, x, y, 2.5 * fwhm, 3.5 * fwhm, 5.0 * fwhm)
    assert np.abs(f_big / flux - 1).max() < 0.01, np.abs(f_big / flux - 1).max()
    ad = AP.aperture_depth(img, x[1:], y[1:], g[1:], e[:, 0], e[:, 1], fwhm)            # the correction without star 0
    f_det = AP.star_flux(img, x, y, max(1.5, fwhm), 3.5 * fwhm, 5.0 * fwhm)
    err = f_det / ad["aperture_fraction"] / flux - 1
    assert abs(np.median(err)) < 0.01 and np.abs(err).max() < 0.02, (np.median(err), np.abs(err).max())
    # the curve of growth is the Moffat profile's, relative to the 2.5 FWHM aperture
    a = alpha_of(fwhm)
    enc = lambda r: 1 - (1 + (r / a) ** 2) ** (1 - BETA)  # noqa: E731
    for k, fr in ad["growth"].items():
        if float(k) < 1.5:     # small circles: the pixel box widens the sampled star (3-4 % less flux inside 2 px)
            continue
        assert abs(fr - enc(float(k) * fwhm) / enc(2.5 * fwhm)) < 0.02, (k, fr)


def test_depth_varies_smoothly_with_the_fwhm():
    """FWHM swept across 2.0-2.5 px (it crosses sqrt(5) = 2.236, where whole-pixel apertures jump): the depth falls as
    -2.5 log10(FWHM) (empty-aperture noise grows with the aperture, the corrected flux does not), without a step."""
    fw = np.round(np.arange(2.0, 2.5001, 0.025), 3)
    d = np.array([depth_of(f)["depth5_mag"] for f in fw])
    assert np.all(np.isfinite(d))
    assert np.abs(np.diff(d)).max() < 0.03, np.abs(np.diff(d)).max()                  # no step (a jump is 0.2-0.5 mag)
    line = np.polyval(np.polyfit(-2.5 * np.log10(fw), d, 1), -2.5 * np.log10(fw))
    assert np.abs(d - line).max() < 0.03, np.abs(d - line).max()
    assert np.polyfit(-2.5 * np.log10(fw), d, 1)[0] == pytest.approx(1.0, abs=0.35)
    # the same sweep with whole pixels jumps at sqrt(5): the defect this replaces
    old = np.array([old_whole_pixel_depth(f) for f in (2.20, 2.27)])
    assert abs(old[1] - old[0]) > 0.15, old
