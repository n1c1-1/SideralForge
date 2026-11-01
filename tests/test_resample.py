"""Resampling onto the reference grid for the Classic stacking method (astro_gpu/core/resample.py: Lanczos-4 with
Siril's or PixInsight's clamping, bilinear; the registration polynomial inverted).

Synthetic images whose values at any position are known: smooth band-limited fields, linear ramps, undersampled stars
on a flat sky (the ringing case), shifts given by registration polynomials of register.py's form.
"""
import math

import numpy as np
import pytest
import torch

from astro_gpu.core import normalize_local as nl
from astro_gpu.core import resample as rs


def grid(H, W):
    return torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")


def coef_shift(W, H, dx, dy, th=0.0):
    """register.py's polynomial: frame sensor -> reference, a turn th about the centre and a shift (dx, dy)."""
    c = np.zeros((2, 10))
    c[0, :3] = [W / 2 + dx, math.cos(th) * W / 2, -math.sin(th) * H / 2]
    c[1, :3] = [H / 2 + dy, math.sin(th) * W / 2, math.cos(th) * H / 2]
    return c


def smooth(x, y):
    return 1000 + 200 * torch.sin(x / 6.0 + 0.3) * torch.cos(y / 7.0) + 80 * torch.sin((x + y) / 11.0)


def test_the_kernel_weights():
    f = torch.rand(500, dtype=torch.float64)
    off = torch.arange(-3, 5, dtype=torch.float64)[:, None]
    for kind in ("opencv", "pcl"):
        w = rs.lanczos_weights(f, kind)
        assert torch.allclose(w.sum(0), torch.ones(500, dtype=torch.float64), atol=1e-12)
    w = rs.lanczos_weights(f, "pcl")
    assert torch.allclose((w * (off - f)).sum(0), torch.zeros(500, dtype=torch.float64), atol=1e-12)
    w0 = rs.lanczos_weights(torch.zeros(1, dtype=torch.float64), "opencv")[:, 0]
    assert torch.allclose(w0, torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float64), atol=1e-12)
    t = torch.tensor([0.0, 1.0, 2.0, 3.5, 4.0, 5.0], dtype=torch.float64)
    assert rs.lanczos(t).tolist()[:3] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert float(rs.lanczos(t)[4]) == 0.0 and float(rs.lanczos(t)[5]) == 0.0


@pytest.mark.parametrize("clamping", rs.CLAMPINGS)
def test_whole_pixel_shifts_copy_the_image_and_the_border_has_no_value(clamping):
    H, W = 40, 50
    yy, xx = grid(H, W)
    img = smooth(xx, yy).float()[None].repeat(3, 1, 1)
    out = rs.resample(img, xx + 3, yy - 2, clamping=clamping)
    inner = out[:, 5:-5, 5:-8]
    assert torch.allclose(inner, img[:, 3:-7, 8:-5], atol=1e-3)
    assert bool(torch.isnan(out[:, :, -4:]).all()) and bool(torch.isnan(out[:, :4]).all())


def test_lanczos_follows_a_smooth_field_better_than_bilinear_and_pcl_keeps_a_ramp_exactly():
    H, W = 48, 64
    yy, xx = grid(H, W)
    img = smooth(xx, yy).float()[None]
    x, y = xx + 0.37, yy + 0.61
    truth = smooth(x, y)
    err = {}
    for name, kw in (("lanczos", dict(clamping="none")), ("bilinear", dict(method="bilinear"))):
        o = rs.resample(img, x, y, **kw)[0].double()
        err[name] = float((o - truth)[6:-6, 6:-6].abs().max())
    assert err["lanczos"] < 0.5 and err["lanczos"] < 0.3 * err["bilinear"]
    ramp = (3.0 * xx - 2.0 * yy + 500).float()[None]
    o = rs.resample(ramp, x, y, clamping="pixinsight", threshold=-1)[0].double()
    assert float((o - (3.0 * x - 2.0 * y + 500))[6:-6, 6:-6].abs().max()) < 1e-3


def test_clamping_removes_the_dark_ring_around_an_undersampled_star():
    H, W = 64, 64
    yy, xx = grid(H, W)
    sky = 100.0
    img = (sky + 5000 * torch.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 0.7 ** 2))).float()[None]
    x, y = xx + 0.5, yy + 0.37
    low = {}
    for c in rs.CLAMPINGS:
        o = rs.resample(img, x, y, clamping=c)[0]
        low[c] = float(o[16:48, 16:48].min())
    assert low["none"] < sky - 10                       # the ring
    assert low["siril"] > sky - 0.5                     # replaced by the bilinear guide around it
    assert low["none"] < low["pixinsight"] < sky + 1    # PCL shrinks the negative lobes


def test_sirils_clamping_takes_the_guide_on_a_cross_around_each_undershoot():
    H, W = 30, 30
    yy, xx = grid(H, W)
    img = torch.full((1, H, W), 100.0)
    img[0, 15, 15] = 10000.0
    x, y = xx + 0.5, yy.clone()
    plain = rs.resample(img, x, y, clamping="none")[0]
    guide = rs.resample(img, x, y, method="bilinear")[0]
    sir = rs.resample(img, x, y, clamping="siril")[0]
    under = plain < rs.CLAMPING_FACTOR * guide
    cross = under.clone()
    cross[1:] |= under[:-1]
    cross[:-1] |= under[1:]
    cross[:, 1:] |= under[:, :-1]
    cross[:, :-1] |= under[:, 1:]
    fin = torch.isfinite(plain)
    assert bool(under.any())
    assert torch.equal(sir[cross & fin], guide[cross & fin])
    assert torch.equal(sir[~cross & fin], plain[~cross & fin])


def test_a_frame_goes_to_the_reference_grid_through_its_registration_polynomial():
    H, W = 60, 80
    dx, dy, th = 2.3, -1.7, 0.004
    c = coef_shift(W, H, dx, dy, th)
    # the frame sees the sky shifted: the reference position of its pixel (x, y) is P(x, y)
    yy, xx = grid(H, W)
    X, Y = nl.poly_apply(c, xx, yy, W, H)
    frame = smooth(X, Y).float()[None]
    x, y = rs.source_positions(c, H, W, W, H)
    Xb, Yb = nl.poly_apply(c, x, y, W, H)
    assert float((Xb - xx).abs().max()) < 1e-6 and float((Yb - yy).abs().max()) < 1e-6
    out = rs.to_reference(frame, c, W, H, clamping="siril")[0].double()
    inner = torch.isfinite(out)
    assert float(inner.float().mean()) > 0.7
    assert float((out - smooth(xx, yy))[inner].abs().max()) < 1.0          # 0.1 % of the field
