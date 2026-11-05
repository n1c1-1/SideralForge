"""Issue #37: a frame whose stars give no clear vote (a dense field: the brightest detections change from frame to
frame) is aligned from the shift its image's phase correlation gives. Synthetic frames: a dense field of stars, the
second shifted; the phase correlation finds the shift to a fraction of a pixel."""
from __future__ import annotations

import numpy as np

from astro_gpu.stages.register import phase_shift


def _field(H, W, xs, ys, fl, dx=0.0, dy=0.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.full((H, W), 100.0, np.float32)
    for x, y, f in zip(xs, ys, fl):
        cx, cy = x + dx, y + dy
        if 5 < cx < W - 5 and 5 < cy < H - 5:
            sl = (slice(int(cy) - 6, int(cy) + 7), slice(int(cx) - 6, int(cx) + 7))
            img[sl] += f * np.exp(-((xx[sl] - cx) ** 2 + (yy[sl] - cy) ** 2) / (2 * 1.8 ** 2))
    return img


def test_the_phase_correlation_finds_the_shift_of_a_dense_field():
    rng = np.random.default_rng(4)
    H, W, n = 768, 1024, 3000
    xs, ys = rng.uniform(0, W, n), rng.uniform(0, H, n)
    fl = rng.lognormal(5, 1, n)
    mm = np.stack([_field(H, W, xs, ys, fl), _field(H, W, xs, ys, fl, 37.3, -55.1)])
    mm += rng.normal(0, 3, mm.shape).astype(np.float32)
    s = phase_shift(mm, 1, 0, H, W)
    assert s is not None and np.allclose(s, [37.3, -55.1], atol=2.0)        # to the binned grid, then the fit refines
