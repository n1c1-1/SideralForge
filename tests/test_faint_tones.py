"""The layer's Faint tones (render/simple.faint_tones): a generalised hyperbolic stretch of the range above the sky."""
from __future__ import annotations

import torch

from astro_gpu.render.simple import faint_tones

SKY = 0.2


def ramp() -> torch.Tensor:
    v = torch.linspace(0, 1, 101)
    return torch.stack([v * 1.0, v * 0.6, v * 0.7])[:, None, :]      # a pink ramp: the colour ratios to keep


def lum(y):
    return 0.2126 * y[0] + 0.7152 * y[1] + 0.0722 * y[2]


def test_the_sky_stays_and_the_faint_tones_rise_most():
    x = ramp()
    L0 = lum(x)[0]
    for k in (0.8, 1.6):
        L = lum(faint_tones(x.clone(), SKY, k))[0]
        below = L0 <= SKY
        assert torch.allclose(L[below], L0[below], atol=1e-6)                     # at and below the sky: unchanged
        assert bool((L[1:] >= L[:-1] - 1e-6).all())                               # still increasing
        above = (L0 > SKY + 0.02) & (L0 < 0.6)                                   # red not yet at 1
        gain = (L[above] - SKY) / (L0[above] - SKY)
        assert bool((gain > 1).all()) and float(gain[0]) > float(gain[-1])      # just above the sky rises most
    assert torch.equal(faint_tones(x.clone(), SKY, 0.0), x)


def test_below_zero_is_the_inverse_and_the_colours_are_kept():
    x = ramp()
    up, down = faint_tones(x.clone(), SKY, 0.8), faint_tones(x.clone(), SKY, -0.8)
    back = faint_tones(up.clone(), SKY, -0.8)
    i = 60
    assert float(lum(down)[0, i]) < float(lum(x)[0, i]) < float(lum(up)[0, i])
    assert abs(float(lum(back)[0, i]) - float(lum(x)[0, i])) < 0.01                # the inverse undoes the raise
    r = up[:, 0, i] / up[0, 0, i]
    assert torch.allclose(r, torch.tensor([1.0, 0.6, 0.7]), atol=1e-4)             # the colour ratios kept
