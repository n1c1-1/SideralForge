"""The drizzle coverage map and its rule (astro_gpu/core/coverage.py, stacking v2 points 1 and 1b): the geometry is
register's; the fast estimate equals the exact map where it looks; no dither gives holes, random dithers a coverage
that varies as 1 / sqrt(frames); the suggestion of scale, drop size and method with its reasons; and, on the real
heart_a7s transforms (slow), the estimate against the weight map of the stack actually made."""
from __future__ import annotations

import ast
import glob
import json
import os
import string
from pathlib import Path

import numpy as np
import pytest

from astro_gpu.core import coverage as cv
from astro_gpu.core.params import Auto
from astro_gpu.stages.register import apply_poly

W, H = 4256, 2848


def transforms(n, W=W, H=H, shift=8.0, rot_deg=0.0, jitter=0.0, same=False, cubic=0.0, seed=0):
    """register-like polynomials (n, 2, 10): a turn about the centre (sd ``rot_deg``), a shift (sd ``shift`` px, or the
    same for all with ``same``) plus ``jitter``, and a small cubic distortion (``cubic`` px at the corners)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 2, 10))
    for i in range(n):
        t = np.radians(rng.normal(0, rot_deg)) if rot_deg else 0.0
        sx, sy = (0.0, 0.0) if same else rng.normal(0, shift, 2)
        sx, sy = sx + rng.normal(0, jitter), sy + rng.normal(0, jitter)
        c, s = np.cos(t), np.sin(t)
        out[i, 0, :3] = W / 2 + sx, c * W / 2, -s * H / 2
        out[i, 1, :3] = H / 2 + sy, s * W / 2, c * H / 2
        out[i, :, 6:] = rng.normal(0, cubic, (2, 4)) if cubic else 0.0
    return out


def test_the_geometry_is_registers():
    c = transforms(4, rot_deg=0.5, cubic=2.0, seed=1)
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, W, 50), rng.uniform(0, H, 50)
    for i in range(4):
        np.testing.assert_allclose(cv.poly_eval(c[i], x, y, W, H), apply_poly(c[i], x, y, W, H), atol=1e-9)
    X, Y = cv.poly_eval(c, np.broadcast_to(x, (4, 50)), np.broadcast_to(y, (4, 50)), W, H)
    xi, yi = cv.invert(c, X[0], Y[0], W, H)                    # frame 0's targets, found in every frame
    X2, Y2 = cv.poly_eval(c, xi, yi, W, H)
    assert np.abs(X2 - X[0]).max() < 1e-6 and np.abs(Y2 - Y[0]).max() < 1e-6
    # the estimate's tangent of a frame's polynomial over a block of +-24 px (scale 1, patches of 32): 0.002 px off at
    # most near a corner, for cubic terms of a few px (the bench's registrations: up to 2.3 px, quadratic up to 2.1 px)
    c = transforms(1, rot_deg=0.5, cubic=1.0, seed=3)
    c[0, :, 3:6] = 2.0
    x0, y0 = np.array([[100.0]]), np.array([[2700.0]])
    X0, Y0 = cv.poly_eval(c, x0, y0, W, H)
    Xu, Xv, Yu, Yv = cv._poly_jacobian(c, x0, y0, W, H)
    d = np.arange(-24.0, 25.0, 4.0)
    dx, dy = np.meshgrid(d, d)
    Xe, Ye = cv.poly_eval(c[0], x0[0, 0] + dx, y0[0, 0] + dy, W, H)
    assert np.abs(X0[0, 0] + Xu[0, 0] * dx + Xv[0, 0] * dy - Xe).max() < 2e-3
    assert np.abs(Y0[0, 0] + Yu[0, 0] * dx + Yv[0, 0] * dy - Ye).max() < 2e-3


@pytest.mark.parametrize("scale,pixfrac", [(1.0, 1.0), (1.0, 0.7), (1.5, 0.7), (2.0, 0.5)])
def test_the_estimate_is_the_exact_map_where_it_looks(scale, pixfrac):
    """The patches of the estimate against the full weight map of every sample (small sensor, turned, distorted: a
    cubic of 0.02 px on 320 px bends as much per pixel as one of 3.5 px on 4256 px)."""
    w, h = 320, 240
    c = transforms(12, W=w, H=h, shift=5.0, rot_deg=0.4, cubic=0.02, seed=3)
    wts = np.random.default_rng(4).uniform(0.5, 1.5, 12)
    full = cv.weight_map(c, w, h, "RGGB", scale, pixfrac, weights=wts)
    est = cv.coverage(c, w, h, "RGGB", scale, pixfrac, weights=wts, grid=(2, 2), patch=24)
    assert (est.frames == 12).all()
    for py in range(2):
        for px in range(2):
            ox, oy = est.origins[py, px]
            for ch, name in enumerate("RGB"):
                exact = cv.variation(full[ch, oy:oy + 24, ox:ox + 24])
                assert abs(est.cv_map[name][py, px] - exact) < 2e-3 * max(1.0, exact), (name, py, px)
    for name in "RGB":
        assert abs(est.cv[name] - np.sqrt(np.mean(est.cv_map[name] ** 2))) < 1e-12


def test_pure_shifts_match_the_analytic_weights():
    """Frames shifted only: the red weight map has period 2 and four values, the sums of each frame's overlaps."""
    for seed in range(3):
        c = transforms(40, seed=seed)
        vals = np.zeros((2, 2))
        for sx, sy in zip(c[:, 0, 0] - W / 2, c[:, 1, 0] - H / 2):
            ov = [np.array([max(0, 1 - abs(f)) + max(0, 1 - abs(f - 2)), max(0, 1 - abs(f - 1))])
                  for f in (sx % 2.0, sy % 2.0)]
            vals += np.outer(ov[1], ov[0])
        est = cv.coverage(c, W, H, "RGGB", 1.0, 1.0, channels="R")
        assert abs(est.cv["R"] - vals.std() / np.median(vals)) < 1e-6


def test_no_dither_leaves_holes():
    c = transforms(30, same=True, jitter=0.05, seed=5)          # the same place every time: tracking noise only
    est = cv.coverage(c, W, H, "RGGB", 1.0, 1.0)
    assert est.cv["R"] > 1.0 and est.cv["B"] > 1.0 and est.cv["G"] > 0.5
    s = cv.suggest_drizzle(c, W, H, "RGGB", 1.8)
    assert s.method.value == "classic" and s.scale.value == 1.0
    assert "Classic advised" in s.method.reason
    holes = transforms(30, same=True, seed=5)                    # exactly the same place: half the red pixels empty
    s = cv.suggest_drizzle(holes, W, H, "RGGB", 1.8)
    assert s.method.reason == "30 frames: the red and blue samples leave holes at 1×: Classic advised"
    assert "would leave holes" in s.scale.reason


def test_random_dithers_cover_well_and_better_with_more_frames():
    """Random sub-pixel phases: the red weight map varies as 0.88 / sqrt(frames) at scale 1, drop 1 (a drop covers
    (1 - |t|) of a pixel in each direction: 7/144 of variance per frame around a mean of 1/4)."""
    got = {}
    for n in (16, 64):
        got[n] = np.mean([cv.coverage(transforms(n, rot_deg=0.2, seed=s), W, H, "RGGB", 1.0, 1.0,
                                      channels="RB").red_blue for s in range(3)])
        assert abs(got[n] - np.sqrt(7 / 144) / 0.25 / np.sqrt(n)) < 0.3 * (0.882 / np.sqrt(n)), (n, got[n])
    assert got[64] < cv.LIMIT < got[16] * 1.5
    s = cv.suggest_drizzle(transforms(64, rot_deg=0.2, seed=9), W, H, "RGGB", 3.0)
    assert s.method.value == "drizzle" and s.method.reason.endswith("at 1×: Bayer drizzle")


def test_scale_and_drop_from_the_rule():
    c = transforms(80, rot_deg=0.2, seed=11)
    # well-sampled stars: 1x and a drop of 1, whatever the coverage
    s = cv.suggest_drizzle(c, W, H, "RGGB", 3.0)
    assert (s.scale.value, s.pixfrac.value) == (1.0, 1.0)
    assert s.scale.reason == "stars of 3.0 px: sampled well enough, drizzle would add no detail: 1×"
    assert s.pixfrac.reason == "scale 1×: drops of a whole pixel"
    assert list(s.coverage) == [(1.0, 1.0)]                     # nothing else computed
    s = cv.suggest_drizzle(c, W, H, "RGGB", float("nan"))
    assert s.scale.value == 1.0 and s.scale.reason == "star size not measured: 1×"
    # undersampled, many frames: 1.5x (it samples 1.8 px stars at 2.7 px), the finest drop that keeps the rule
    s = cv.suggest_drizzle(c, W, H, "RGGB", 1.8)
    assert s.scale.value == 1.5
    pf = s.pixfrac.value
    steps = list(cv.pixfrac_steps(1.5))
    assert steps[0] == 0.7 and steps[-1] == 1.0 and pf in steps
    assert s.coverage[(1.5, pf)].holds()
    if pf > steps[0]:
        below = cv.coverage(c, W, H, "RGGB", 1.5, steps[steps.index(pf) - 1], channels="RB")
        assert not below.holds()
    assert s.scale.reason.startswith("stars of 1.8 px: undersampled; coverage stays at ")
    assert s.scale.reason.endswith(" %: 1.5×")
    assert s.scale.args["cov"] < 100 * cv.LIMIT
    assert s.pixfrac.reason.startswith("the finest drop that keeps the red and blue coverage under 20 % (")
    # few frames: the rule fails at 1.5x and 2x, so 1x with the reason
    s = cv.suggest_drizzle(transforms(10, rot_deg=0.2, seed=12), W, H, "RGGB", 1.8)
    assert s.scale.value == 1.0 and s.pixfrac.value == 1.0
    assert s.scale.reason.startswith("stars of 1.8 px: undersampled, but at 1.5× the coverage would vary by ")


def test_very_small_stars_prefer_2x_when_it_holds():
    c = transforms(160, rot_deg=0.2, seed=13)
    s = cv.suggest_drizzle(c, W, H, "RGGB", 1.2)              # 1.5x samples them at 1.8 px only: under 2 px
    assert s.coverage[(1.5, 1.0)].holds() and s.coverage[(2.0, s.pixfrac.value)].holds()
    assert s.scale.value == 2.0 and s.scale.reason.endswith(" %: 2×")
    s = cv.suggest_drizzle(c, W, H, "RGGB", 1.6)              # 1.5x samples them at 2.4 px: enough
    assert s.scale.value == 1.5


def test_frame_weights_count():
    c = transforms(40, rot_deg=0.2, seed=14)
    w = np.ones(40)
    w[20:] = 0.0                                                # half the frames out: as 20 frames
    a = cv.coverage(c, W, H, "RGGB", 1.0, 1.0, weights=w)
    b = cv.coverage(c[:20], W, H, "RGGB", 1.0, 1.0)
    assert a.n_frames == 20 and abs(a.red_blue - b.red_blue) < 1e-12
    s = cv.suggest_drizzle(c, W, H, "RGGB", 3.0, weights=w)
    assert s.method.args["n"] == 20


def test_map_variation_of_a_full_map():
    m = np.ones((3, 100, 100))
    m[0, ::2, ::2] = 3.0
    m[:, :5] = 0.0                                              # an edge outside the margin does not count
    v = cv.map_variation(m)
    assert v["G"] == 0.0 and v["B"] == 0.0 and abs(v["R"] - np.std([3, 1, 1, 1])) < 1e-12
    assert cv.variation(np.zeros(10)) == float("inf") and np.isnan(cv.variation(np.array([])))


def test_reasons_are_templates_with_a_french():
    """Every Auto written in the module: a plain template (texts.reason_sites' rules), its arguments named in a dict,
    and its French in FRENCH with the same fields."""
    tree = ast.parse(Path(cv.__file__).read_text(encoding="utf-8"))
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Auto":
            t = node.args[1]
            assert isinstance(t, ast.Constant) and isinstance(t.value, str)
            fields = {f for _, f, _, _ in string.Formatter().parse(t.value) if f}
            names = {k.value for k in node.args[2].keys} if len(node.args) > 2 else set()
            assert fields == names, t.value
            assert t.value in cv.FRENCH, t.value
            spec_en = {(f, sp) for _, f, sp, _ in string.Formatter().parse(t.value) if f}
            spec_fr = {(f, sp) for _, f, sp, _ in string.Formatter().parse(cv.FRENCH[t.value]) if f}
            assert spec_en == spec_fr and cv.FRENCH[t.value] != t.value
            seen.add(t.value)
    assert seen == set(cv.FRENCH)
    a = Auto(1.5, "stars of {fwhm:.1f} px: undersampled; coverage stays at {cov:.0f} %: {scale:g}×",
             {"fwhm": 1.8, "cov": 14.2, "scale": 1.5})
    assert a.reason == "stars of 1.8 px: undersampled; coverage stays at 14 %: 1.5×"


def test_fast_enough():
    import time
    c = transforms(100, rot_deg=0.2, cubic=1.0, seed=15)
    t = time.perf_counter()
    cv.suggest_drizzle(c, W, H, "RGGB", 1.8)
    assert time.perf_counter() - t < 8.0                        # about 0.6 s alone on this machine


# ------------------------------------------------------------------------------------------------ real data
HEART = "F:/astro_essais/bench/heart/ours_express/store"


@pytest.mark.slow
def test_heart_a7s_estimate_against_the_real_weight_map():
    """The real transforms of heart_a7s (read-only): the estimate at the stack's own scale and drop size, with its
    frame weights, against the drz_weight.npy of that stack."""
    stacks = [d for d in glob.glob(f"{HEART}/stack/*") if os.path.exists(f"{d}/drz_weight.npy")
              and os.path.exists(f"{d}/frames.json") and os.path.exists(f"{d}/manifest.json")]
    if not stacks:
        pytest.skip("the heart_a7s bench stack is not here")
    d = sorted(stacks, key=os.path.getmtime)[-1]
    man = json.load(open(f"{d}/manifest.json"))
    reg = np.load(f"{HEART}/register/{man['inputs']['register']}/reg.npz")
    cal = json.load(open(f"{HEART}/calibrate/{man['inputs']['calibrate']}/cache.json"))
    w = np.array([f["weight"] for f in json.load(open(f"{d}/frames.json"))])
    used = w > 0
    S, pf = man["params"]["scale"], man["params"]["pixfrac"]
    est = cv.coverage(reg["coef"][used], cal["width"], cal["height"], cal["cfa"], S, pf, weights=w[used])
    real = cv.map_variation(np.load(f"{d}/drz_weight.npy", mmap_mode="r"))
    for c in "RGB":
        assert abs(est.cv[c] - real[c]) < 0.06, (c, est.cv[c], real[c])
