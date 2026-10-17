"""Continuum smoothing from the gaps between stars (render_prep)."""
import cv2
import numpy as np

from astro_gpu.stages.render_prep import smooth_from_gaps


def test_smoothing_ignores_star_residuals():
    rng = np.random.default_rng(0)
    c = np.full((200, 200), 10.0, np.float32)
    w = np.ones_like(c)
    ys, xs = rng.integers(0, 200, 400), rng.integers(0, 200, 400)
    c[ys, xs] += 50.0                                       # residual stars
    w[ys, xs] = 0.0                                         # where star light dominates
    plain = cv2.GaussianBlur(c, (0, 0), 24)
    gaps = smooth_from_gaps(c, w, 24)
    assert abs(float(np.median(gaps)) - 10.0) < 0.05        # the level of the gaps
    assert float(np.median(plain)) > 10.3                   # a plain average is raised by the residuals


def test_no_gap_within_reach_falls_back_to_plain():
    c = np.random.default_rng(1).normal(5, 1, (120, 120)).astype(np.float32)
    w = np.zeros_like(c)
    assert np.allclose(smooth_from_gaps(c, w, 10), cv2.GaussianBlur(c, (0, 0), 10), atol=1e-5)
