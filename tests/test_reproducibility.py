"""Run-to-run reproducibility: the PSF fit of the optics stage, the tone curve of the rendering, deterministic mode."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import erf

from astro_gpu.core.determinism import deterministic
from astro_gpu.render.engine import fit_tone_curve
from astro_gpu.stages.optics import fit_channel

# a small field and few iterations: the fit is the same code on 60 stars as on 260 (the full-size field took 30 s)
ITERS = 50


def _field(seed=0, H=300, W=450, n=60, sky=(50.0, 0.01, -0.006)):
    """Isolated stars (pixel-integrated Gaussians, wider towards the right), on a tilted sky, with noise."""
    rng = np.random.default_rng(seed)
    gy, gx = np.mgrid[40:H - 40:36, 40:W - 40:36]
    xs = (gx.ravel() + rng.uniform(-6, 6, gx.size))[:n].astype(np.float64)
    ys = (gy.ravel() + rng.uniform(-6, 6, gy.size))[:n].astype(np.float64)
    flux = 10 ** rng.uniform(3.5, 4.8, len(xs))
    cols = rng.uniform(0.5, 2.0, len(xs))
    yy, xx = np.mgrid[0:H, 0:W]
    img = sky[0] + sky[1] * xx + sky[2] * yy + np.zeros((H, W))
    s2 = math.sqrt(2.0)
    for x0, y0, f in zip(xs, ys, flux):
        sg = 1.1 + 0.5 * x0 / W
        sl = (slice(int(y0) - 12, int(y0) + 13), slice(int(x0) - 12, int(x0) + 13))
        gxv = 0.5 * (erf((xx[sl] - x0 + 0.5) / (sg * s2)) - erf((xx[sl] - x0 - 0.5) / (sg * s2)))
        gyv = 0.5 * (erf((yy[sl] - y0 + 0.5) / (sg * s2)) - erf((yy[sl] - y0 - 0.5) / (sg * s2)))
        img[sl] += f * gxv * gyv
    img = img + rng.normal(0, 2.0, img.shape)
    return img.astype(np.float32), xs, ys, cols, flux


def test_psf_fit_gives_the_same_answer_every_time():
    img, xs, ys, cols, flux = _field()
    H, W = img.shape
    dev = torch.device("cpu")
    E1, f1, r1 = fit_channel(img, xs, ys, cols, W, H, ITERS, dev)
    E2, f2, r2 = fit_channel(img, xs, ys, cols, W, H, ITERS, dev)
    assert torch.equal(E1, E2) and np.array_equal(f1, f2) and r1 == r2
    ratio = f1 / flux
    assert np.median(np.abs(ratio / np.median(ratio) - 1)) < 0.01        # consistent fluxes (up to one common scale)
    # a flat offset of the image (a different sky level) does not move the fluxes: the background is measured
    # around each star, not traded with a flat part of the PSF model
    _, f3, _ = fit_channel(img + 25.0, xs, ys, cols, W, H, ITERS, dev)
    assert np.max(np.abs(f3 / f1 - 1)) < 1e-3
    # a small change of the image gives a small change of the fluxes (no jump to another solution)
    _, f4, _ = fit_channel(img + np.random.default_rng(9).normal(0, 0.02, img.shape).astype(np.float32),
                           xs, ys, cols, W, H, ITERS, dev)
    assert np.median(np.abs(f4 / f1 - 1)) < 1e-3


def test_tone_curve_moves_smoothly_with_its_inputs():
    t_dark, t_sky, toe = 10 / 255, 24 / 255, 0.015
    lo0, b0, _ = fit_tone_curve(1.5, 2.5, 30.0, t_dark, t_sky, toe)
    for d in (1e-4, 1e-3):
        lo1, b1, _ = fit_tone_curve(1.5 + d, 2.5, 30.0, t_dark, t_sky, toe)
        assert abs(lo1 - lo0) < 5 * d and abs(b1 / b0 - 1) < 50 * d       # no jump of a grid step


def test_deterministic_mode_is_restored():
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark)
    with deterministic():
        assert torch.are_deterministic_algorithms_enabled()
    assert (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark) == before
