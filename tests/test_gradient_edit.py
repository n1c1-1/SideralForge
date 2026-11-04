"""The Gradient block's RBF, placed samples and excluded areas (stages/gradient.py), on a synthetic sky."""
from __future__ import annotations

import numpy as np

from astro_gpu.stages.gradient import GradientParams, exclusion_mask, read_grid, sky_model_of

H, W = 300, 450


def scene(nebula: float = 0.0):
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W], np.float32)[:, None, None]
    sky = 100 + 20 * xx + 10 * yy ** 2 + 15 * np.exp(-np.hypot(xx - 1, yy) / 0.3)       # a corner glow
    neb = nebula * np.exp(-((xx - 0.45) ** 2 + (yy - 0.5) ** 2) / 0.03)
    img = np.stack([sky + neb] * 3) + rng.normal(0, 0.5, (3, H, W))
    return img.astype(np.float32), sky


def model(img, **kw):
    m = sky_model_of(img, np.ones((H, W), bool), GradientParams(**kw))
    return read_grid(m["grid"], m["info"], H, W)[0], m


def test_rbf_follows_a_corner_glow_better_than_the_polynomial():
    img, sky = scene()
    err = {}
    for meth in ("polynomial", "rbf"):
        g, _m = model(img, method=meth, sample_points=[], rbf_smooth=0.0) if meth == "rbf" else model(img, method=meth)
        d = (g - np.median(g)) - (sky - np.median(sky))
        err[meth] = float(np.sqrt((d ** 2).mean()))
    assert err["rbf"] < 0.85 * err["polynomial"], err


def test_placed_samples_are_the_samples():
    img, _sky = scene()
    pts = [[40.0, 40.0], [400.0, 40.0], [40.0, 260.0], [400.0, 260.0], [220.0, 150.0], [300.0, 100.0], [120.0, 200.0]]
    _g, m = model(img, method="polynomial", degree=1, sample_points=pts)
    assert m["samples"]["manual"] and len(m["samples"]["x"]) == len(pts) and m["samples"]["used"].all()


def test_an_excluded_ellipse_keeps_a_nebula_out_of_the_automatic_model():
    img, sky = scene(nebula=2.0)                          # a faint, wide nebula
    ell = [[0.45 * W, 0.5 * H, 0.25 * W, 0.3 * H]]
    c = (slice(140, 160), slice(190, 215))
    lift = []
    for kw in ({}, {"exclude_ellipses": ell}):
        g, _ = model(img, method="automatic", **kw)
        d = g - sky
        lift.append(float(d[c].mean() - np.median(d)))  # how far the model rises into the nebula
    assert lift[0] > 1.5 and lift[1] < 0.5 * lift[0], lift


def test_exclusion_mask_paints_and_erases_in_order():
    p = GradientParams(exclude_ellipses=[[100, 100, 50, 30]], exclude_brush=[[100, 100, 10, 0], [300, 50, 20, 1]])
    m = exclusion_mask(p, 200, 400)
    assert m[100, 70] and not m[100, 100] and m[50, 300] and not m[180, 380]
    half = exclusion_mask(p, 100, 200, 0.5)
    assert half.shape == (100, 200) and half[25, 150]
    assert exclusion_mask(GradientParams(), 10, 10) is None


def test_a_turned_ellipse_is_excluded_along_its_axes():
    from astro_gpu.client.gradientedit import _inside, _local, _world
    e = [200.0, 100.0, 80.0, 20.0, 90.0]                    # turned a quarter: its long axis vertical
    m = exclusion_mask(GradientParams(exclude_ellipses=[e]), 200, 400)
    assert m[160, 200] and not m[100, 270]
    assert _inside(e, 200, 160) and not _inside(e, 270, 100)
    u, v = _local(e, *_world(e, 30.0, -7.0))
    assert abs(u - 30) < 1e-9 and abs(v + 7) < 1e-9
