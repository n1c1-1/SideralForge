"""Frame registration: the usual drift, and a field that turns (alt-azimuth mounts such as smart telescopes)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from astro_gpu.stages.register import apply_poly, fit_frame


def _field(n=400, W=1080, H=1920, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform([0, 0], [W, H], (n, 2))
    flux = np.sort(rng.pareto(1.5, n) + 1)[::-1] * 1000
    return np.c_[xy, flux, np.full(n, 2.0)]


@pytest.mark.parametrize("ang,shift", [(0.0, (35.0, -12.0)), (12.0, (20.0, 40.0)), (-27.0, (-60.0, 5.0))])
def test_a_turned_field_is_aligned(ang, shift):
    W, H = 1080, 1920
    ref = _field(W=W, H=H)
    t = np.deg2rad(ang)
    c = np.array([W / 2, H / 2])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    xy = (ref[:, :2] - c) @ R.T + c + np.array(shift)          # the frame: the sky turned and shifted
    inside = (xy[:, 0] > 0) & (xy[:, 0] < W) & (xy[:, 1] > 0) & (xy[:, 1] < H)
    cat = np.c_[xy, ref[:, 2:]][inside]
    cat[:, :2] += np.random.default_rng(1).normal(0, 0.2, (len(cat), 2))
    r = fit_frame(cat, ref, cKDTree(ref[:, :2]), W, H)
    assert r is not None
    coef, nmatch, rms, ratio = r
    assert nmatch > 0.8 * len(cat) and rms < 0.6 and abs(ratio - 1) < 0.01
    px, py = apply_poly(coef, cat[:, 0], cat[:, 1], W, H)
    assert np.median(np.hypot(px - ref[inside, 0], py - ref[inside, 1])) < 0.3
