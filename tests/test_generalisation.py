"""Data-driven safeguards that make the processing independent of the instrument: fixed pattern of the sensor, sky
survey tiles."""
from __future__ import annotations

import numpy as np
import torch

from astro_gpu.stages.stack import fixed_pattern


def test_fixed_pattern_recovered_from_drifting_frames():
    rng = np.random.default_rng(2)
    H, W, n = 128, 192, 40
    pattern = rng.normal(0, 3, (H, W)).astype(np.float32)
    yy, xx = np.mgrid[0:H + 80, 0:W + 80]
    sky = (100 + 0.05 * xx + 0.03 * yy).astype(np.float32)                 # smooth sky with drifting stars
    for x0, y0 in rng.uniform(0, [W + 80, H + 80], (120, 2)):
        sky += 200 * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 1.5 ** 2)
    frames = np.zeros((n, H, W), np.float16)
    coef = np.zeros((n, 2, 10))
    for i in range(n):
        dx, dy = i * 1.5, i * 1.0
        frames[i] = sky[int(dy):int(dy) + H, int(dx):int(dx) + W] + pattern + rng.normal(0, 4, (H, W))
        coef[i, 0, 0], coef[i, 1, 0] = W / 2 + dx, H / 2 + dy
    est, info = fixed_pattern(frames, np.arange(n), coef, torch.device("cpu"), H, W, 24)
    assert info["applied"]
    e = est.numpy()
    corr = np.corrcoef(e[8:-8, 8:-8].ravel(), pattern[8:-8, 8:-8].ravel())[0, 1]
    assert corr > 0.8
    # no drift: not applied
    coef[:, 0, 0], coef[:, 1, 0] = W / 2, H / 2
    assert fixed_pattern(frames, np.arange(n), coef, torch.device("cpu"), H, W, 24)[0] is None


def test_nested_healpix_index_splits_into_tile_coordinates():
    from astro_gpu.stages.surveys import _compact
    x, y = np.arange(512, dtype=np.int64), np.arange(511, -1, -1, dtype=np.int64)
    idx = np.zeros_like(x)
    for b in range(9):                                   # interleave: x on even bits, y on odd bits
        idx |= ((x >> b) & 1) << (2 * b) | ((y >> b) & 1) << (2 * b + 1)
    assert np.array_equal(_compact(idx), x) and np.array_equal(_compact(idx >> 1), y)
