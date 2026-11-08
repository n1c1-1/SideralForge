"""The compiled loops of the drizzle (CPU) against the PyTorch code that runs on a GPU: the same results."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from astro_gpu.stages import stack
from astro_gpu.stages.stack import Prediction, drizzle_deposit, touching


def frame(H=480, W=640, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    X, Y = xx + 0.37 + 1e-3 * yy, yy - 0.21 + 1e-3 * xx
    val = torch.from_numpy(rng.uniform(100, 200, (H, W)).astype(np.float32))
    ch = torch.tensor([0, 1, 1, 2])[2 * (yy.long() % 2) + (xx.long() % 2)]
    return X, Y, val, ch, torch.ones((H, W), dtype=torch.bool), H, W


@pytest.mark.parametrize("h, S", [(0.35, 1.0), (0.5, 1.0), (0.75, 1.5), (1.0, 2.0)])
def test_the_compiled_deposit_is_the_generic_one(monkeypatch, h, S):
    X, Y, val, ch, keep, H, W = frame()
    Ho, Wo = int(H * S), int(W * S)
    Xo, Yo = (X + 0.5) * S - 0.5, (Y + 0.5) * S - 0.5

    def run(limit, den):
        monkeypatch.setattr(stack, "PARALLEL_SAMPLES", limit)
        num, d = torch.zeros(3 * Ho * Wo), torch.zeros(3 * Ho * Wo) if den else None
        drizzle_deposit(num, d, val, Xo, Yo, 0.8, keep, ch, h, S, Ho, Wo)
        return num, d

    before = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        a, b = run(10 ** 12, True), run(1, True)
        assert torch.allclose(a[0], b[0], rtol=2e-6) and torch.allclose(a[1], b[1], rtol=2e-6)
        assert torch.allclose(run(10 ** 12, False)[0], run(1, False)[0], rtol=2e-6)     # no weights kept (near, clip)
    finally:
        torch.set_num_threads(before)


def test_the_compiled_sampling_is_grid_sample(monkeypatch):
    rng = np.random.default_rng(2)
    H, W = 200, 300
    X, Y, _val, ch, _k, _h, _w = frame(H, W)
    num = torch.from_numpy(rng.uniform(1, 9, 3 * H * W).astype(np.float32)) * 5
    den = torch.from_numpy(rng.uniform(0, 5, 3 * H * W).astype(np.float32)) * (torch.rand(3 * H * W) > 0.05)
    model = Prediction(num, den, 10, H, W, torch.device("cpu"))

    def run(limit):
        monkeypatch.setattr(stack, "PARALLEL_SAMPLES", limit)
        return model.sample(X - 3.0, Y + 2.0, ch)

    a, b = run(10 ** 12), run(1)
    for k in ("pred", "cover", "bkg", "floor"):
        assert torch.allclose(a[k], b[k], rtol=2e-5, atol=2e-5), k


def test_touching_is_a_three_by_three_dilation():
    m = torch.from_numpy(np.random.default_rng(3).random((60, 80)) < 0.02)
    want = F.max_pool2d(m.float()[None, None], 3, 1, 1)[0, 0] > 0
    assert torch.equal(touching(m), want)
