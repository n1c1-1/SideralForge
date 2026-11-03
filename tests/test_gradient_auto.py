"""The automatic sky model (stages/gradient_auto.py, Siril's ``subsky -auto`` written again), on synthetic skies."""
from __future__ import annotations

import numpy as np

from astro_gpu.stages import gradient_auto as G


def test_lowpass_follows_a_ramp_to_the_edges():
    """The image is continued along its slope beyond the edges: a plane comes out within 1 % of its range, borders
    included (replicated edges, the published source's way, sag by 6 % here)."""
    yy, xx = np.mgrid[0:120, 0:200].astype(np.float32)
    plane = 0.2 + 0.001 * xx - 0.0005 * yy
    assert np.abs(G.lowpass(plane, 12.0) - plane).max() < 0.01 * float(plane.max() - plane.min())


def test_histogram_median_is_the_median():
    x = np.random.default_rng(0).normal(0, 1, 100_001).astype(np.float32)
    assert abs(G._histogram_median(x) - float(np.median(x))) < 2e-3


def test_sky_model_recovers_a_gradient_under_stars_and_a_nebula():
    rng = np.random.default_rng(1)
    H, W = 800, 1200
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W], np.float32)[:, None, None]
    sky = 0.2 + 0.05 * xx + 0.03 * yy ** 2
    img = sky + 0.03 * np.exp(-((xx - 0.4) ** 2 + (yy - 0.5) ** 2) / 0.005)               # a nebula
    stars = np.zeros((H, W), np.float32)
    stars[rng.integers(0, H, 300), rng.integers(0, W, 300)] = rng.uniform(0.2, 0.8, 300)
    img = (img + stars + rng.normal(0, 0.002, (H, W))).astype(np.float32)
    bg = G.enlarge(G.sky_model(img, G.AutoSettings()), H, W)
    err = (bg - np.median(bg)) - (sky - np.median(sky))
    assert np.sqrt((err ** 2).mean()) < 0.03 * (sky.max() - sky.min())       # within 3 % of the gradient, RMS


def test_valid_map_keeps_the_borders_out():
    """Uncovered borders (zeros) never enter the fit: the model under the covered field is that of the field."""
    rng = np.random.default_rng(2)
    img = (0.3 + rng.normal(0, 0.002, (400, 600))).astype(np.float32)
    framed = img.copy()
    framed[:, :40] = 0
    valid = np.ones(img.shape, bool)
    valid[:, :40] = False
    m = G.enlarge(G.sky_model(framed, G.AutoSettings(), valid), 400, 600)
    assert abs(float(np.median(m[:, 100:])) - 0.3) < 0.002
