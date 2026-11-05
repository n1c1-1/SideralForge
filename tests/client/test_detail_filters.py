"""Highlight compression and faint area smoothing (client/postfilters/detail.py): identity at 0, what they touch and
what they leave alone."""
from __future__ import annotations

import cv2
import numpy as np

from astro_gpu.client.postfilters import detail as pd


def scene(seed=0):
    """A display image (h, w, 3): a faint noisy sky at 0.12, a bright reddish core at 0.85 with fine structure."""
    rng = np.random.default_rng(seed)
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    core = np.exp(-(((xx - 220) / 50.0) ** 2 + ((yy - 120) / 50.0) ** 2))
    L = 0.12 + 0.73 * core + 0.04 * core * np.sin(xx / 2.0) + 0.02 * rng.standard_normal((h, w))
    rgb = np.stack([L * 1.1, L * 0.9, L * 0.95], -1)
    return np.clip(rgb, 0, 1).astype(np.float32)


def lum(x):
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def test_highlights_bring_the_core_down_keep_its_detail_and_its_colour():
    x = scene()
    assert pd.highlights(x, 0.0, 0.5, 0.3, 40) is x
    y = pd.highlights(x, 1.0, 0.5, 0.3, 40)
    core, sky = (slice(105, 135), slice(205, 235)), (slice(10, 40), slice(10, 60))
    assert lum(y)[core].mean() < lum(x)[core].mean() - 0.05                  # the core comes down
    assert abs(lum(y)[sky].mean() - lum(x)[sky].mean()) < 0.01               # under the knee: untouched
    hp = lambda a: lum(a) - cv2.GaussianBlur(lum(a), (0, 0), 3)               # noqa: E731
    assert hp(y)[core].std() > 0.5 * hp(x)[core].std()                       # its fine structure kept
    rx, ry = x[core][..., 0] / x[core][..., 1], y[core][..., 0] / y[core][..., 1]
    assert np.allclose(rx, ry, atol=1e-3)                                    # the colour kept
    assert y.max() <= 1 and y.min() >= 0


def test_faint_smoothing_smooths_the_sky_and_leaves_the_core():
    x = scene()
    assert pd.faint_smoothing(x, 0.0, 0.3, 12) is x
    y = pd.faint_smoothing(x, 1.0, 0.3, 12)
    sky, core = (slice(20, 80), slice(20, 100)), (slice(110, 130), slice(210, 230))
    assert lum(y)[sky].std() < 0.3 * lum(x)[sky].std()                      # the grain of the sky gone
    assert np.abs(y[core] - x[core]).max() < 1e-3                           # the bright core untouched
