"""Rejection of the Classic stacking method (astro_gpu/core/rejection_classic.py: WBPP's rule by the number of frames,
Siril's percentile / winsorized / GESDT, PixInsight's linear fit), and the whole Classic path
(astro_gpu/core/stack_classic.py) on synthetic frames.

The vectorised methods are checked against straight ports of the tools' loops for one pixel (Siril's
apply_rejection_float, PixInsight's LinearFitRejectionThread with an exact L1 line) and against the NIST worked
example of the generalized ESD test (Rosner 1983). The whole path stacks synthetic CFA frames of a known sky (stars,
gradients), registered by shifts and turns, with a satellite trail, a cosmic ray and bad pixels.
"""
import itertools
import json
import math

import numpy as np
import pytest
import torch

from astro_gpu.core import debayer as db
from astro_gpu.core import normalize_local as nl
from astro_gpu.core import rejection_classic as rc
from astro_gpu.core import stack_classic as sc


# ------------------------------------------------------------------------------------------------ the rule
@pytest.mark.parametrize("n, method, low, high", [(2, "none", 0, 0), (3, "percentile", 0.2, 0.1),
                                                  (5, "percentile", 0.2, 0.1), (6, "winsorized", 3, 3),
                                                  (15, "winsorized", 3, 3), (16, "linear_fit", 5, 3.5),
                                                  (51, "linear_fit", 5, 3.5)])
def test_wbpp_rule_by_the_number_of_frames(n, method, low, high):
    ch = rc.choose(n)
    assert (ch.method, ch.low, ch.high) == (method, low, high)
    assert str(n) in ch.reason


def test_gesdt_option_above_50_frames_and_chosen_methods():
    assert rc.choose(50, gesd_above_50=True).method == "linear_fit"
    ch = rc.choose(51, gesd_above_50=True)
    assert (ch.method, ch.low, ch.high) == ("gesdt", 0.3, 0.05)
    ch = rc.choose(30, "winsorized", 2.5, None)
    assert (ch.method, ch.low, ch.high) == ("winsorized", 2.5, 3.0)
    with pytest.raises(ValueError):
        rc.choose(30, "kappa")


# ------------------------------------------------------------------------------------------------ the methods
def test_percentile_catches_a_trail_in_four_frames():
    g = torch.Generator().manual_seed(0)
    v = 1000 + 20 * torch.randn(4, 5000, generator=g)
    v[2, :100] += 800
    v[1, 200:300] = 300
    lo, hi = rc.percentile_clip(v, 0.2, 0.1)
    assert bool(hi[2, :100].all()) and bool(lo[1, 200:300].all())
    assert int((lo | hi).sum()) == 200


def _sd(x):
    m = sum(x) / len(x)
    return math.sqrt(sum((a - m) ** 2 for a in x) / (len(x) - 1))


def _med(x):
    s = sorted(x)
    n = len(s)
    return 0.5 * (s[(n - 1) // 2] + s[n // 2])


def siril_winsorized(vals, low, high):
    """Siril's WINSORIZED branch of apply_rejection_float for one pixel (rejected values, low and high)."""
    stack, N, r = list(vals), len(vals), 0
    lows, highs = [], []
    while True:
        sigma, med = _sd(stack), _med(stack)
        w = list(stack)
        while True:
            m0, m1 = med - 1.5 * sigma, med + 1.5 * sigma
            w = [min(m1, max(m0, x)) for x in w]
            s0, sigma = sigma, 1.134 * _sd(w)
            if not abs(sigma - s0) > s0 * 0.0005:
                break
        keep = []
        for x in stack:
            f = 0
            if N - r > 4:
                f = -1 if med - x > sigma * low else (1 if x - med > sigma * high else 0)
                r += f != 0
            (lows if f < 0 else highs if f > 0 else keep).append(x)
        changed = len(keep) != N
        N, stack = len(keep), keep
        if not (changed and N > 3):
            break
    return sorted(lows), sorted(highs)


def test_winsorized_is_sirils_loop():
    rng = np.random.default_rng(5)
    N, P = 12, 300
    v = rng.normal(1000, 20, (N, P))
    for j in range(P):                                  # one or two outliers per pixel, of either sign
        for i in rng.choice(N, rng.integers(0, 3), replace=False):
            v[i, j] += rng.choice([-1, 1]) * rng.uniform(40, 400)
    lo, hi = rc.winsorized_clip(torch.from_numpy(v), 3.0, 3.0)
    for j in range(P):
        el, eh = siril_winsorized(v[:, j], 3.0, 3.0)
        assert sorted(v[lo[:, j].numpy(), j]) == el and sorted(v[hi[:, j].numpy(), j]) == eh, j


def test_winsorized_keeps_sirils_guard_of_four_values():
    v = torch.tensor([[1000.0], [1001.0], [999.0], [1000.5], [5000.0], [6000.0], [7000.0]])
    v = torch.cat([v[:3], v[3:]])                       # 7 values, 3 far above
    lo, hi = rc.winsorized_clip(v, 3.0, 3.0)
    assert int(hi.sum()) <= 3 and int((~(lo | hi)).sum()) >= 4
    lo, hi = rc.winsorized_clip(v[:4], 3.0, 3.0)        # 4 values: never
    assert not bool((lo | hi).any())


def l1_exact(y):
    """The least mean absolute deviation line of y against 0..n-1: the best line through two of the points."""
    x = np.arange(len(y))
    best = (np.inf, 0.0, 0.0)
    for i, j in itertools.combinations(range(len(y)), 2):
        b = (y[j] - y[i]) / (j - i)
        a = y[i] - b * i
        best = min(best, (float(np.abs(y - a - b * x).mean()), a, b))
    return best


def test_the_median_line_fit_minimises_the_mean_absolute_deviation():
    rng = np.random.default_rng(11)
    for K in (5, 12, 24):
        y = np.sort(rng.normal(100, 10, (60, K)), axis=1)
        y[::7] = np.round(y[::7])                           # ties too
        a, b, adev = rc.l1_line(torch.from_numpy(y), torch.full((60,), K))
        best = np.array([l1_exact(r)[0] for r in y])
        assert np.all(adev.numpy() <= best * (1 + 2e-3) + 1e-12)


def pixinsight_linear_fit(vals, low, high):
    """PixInsight's LinearFitRejectionThread for one pixel (2016 code), with the exact L1 line."""
    r = sorted(vals)
    lows, highs = [], []
    if len(r) < 5:
        return lows, highs
    while True:
        y = np.array(r)
        adev, a, b = l1_exact(y)
        sigma = 2 * adev
        if sigma == 0:
            break
        keep, nc = [], 0
        for j, v in enumerate(r):
            line = a + b * j
            if v < line and (line - v) / sigma >= low:
                lows.append(v)
                nc += 1
            elif v >= line and (v - line) / sigma >= high:
                highs.append(v)
                nc += 1
            else:
                keep.append(v)
        if nc == 0:
            break
        r = sorted(keep)
        if len(r) < 3:
            break
    return sorted(lows), sorted(highs)


def test_linear_fit_is_pixinsights_loop():
    rng = np.random.default_rng(9)
    N, P = 16, 150
    v = rng.normal(1000, 20, (N, P))
    v += np.linspace(-30, 30, N)[:, None] * rng.uniform(0, 1, P)[None]      # a gradient across the frames
    for j in range(P):
        for i in rng.choice(N, rng.integers(0, 3), replace=False):
            v[i, j] += rng.choice([-1, 1]) * rng.uniform(100, 900)
    lo, hi = rc.linear_fit_clip(torch.from_numpy(v), 5.0, 3.5)
    same = 0
    for j in range(P):
        el, eh = pixinsight_linear_fit(v[:, j], 5.0, 3.5)
        same += sorted(v[lo[:, j].numpy(), j]) == el and sorted(v[hi[:, j].numpy(), j]) == eh
    assert same >= 0.97 * P                     # the L1 line is not always unique: rare ties at a threshold


# NIST/SEMATECH e-Handbook, 1.3.5.17.3 Generalized ESD test: Rosner's 54 values, alpha 0.05, up to 10 outliers
ROSNER = [-0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26, 1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56, 1.58, 1.65, 1.69, 1.70,
          1.76, 1.77, 1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10, 2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37, 2.40,
          2.47, 2.54, 2.62, 2.64, 2.90, 2.92, 2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68, 4.30, 4.64, 5.34, 5.42, 6.01]
NIST_LAMBDA = [3.158, 3.151, 3.143, 3.136, 3.128, 3.120, 3.111, 3.103, 3.094, 3.085]


def test_gesdt_on_the_nist_example():
    lam = rc.gesd_critical(54, 0.19, 0.05)
    assert len(lam) == 10
    assert np.allclose(lam, NIST_LAMBDA, atol=1.5e-3)
    rng = np.random.default_rng(0)
    v = torch.tensor(np.array([rng.permutation(ROSNER) for _ in range(5)]).T)            # 5 pixels, any order
    lo, hi = rc.gesd_clip(v, 0.19, 0.05)
    for j in range(5):
        assert sorted(v[hi[:, j], j].tolist()) == [5.34, 5.42, 6.01] and not bool(lo[:, j].any())


def test_false_rejections_on_pure_noise_stay_small():
    g = torch.Generator().manual_seed(4)
    for n, top in ((4, 0.001), (10, 1.5), (30, 1.0), (60, 0.2)):
        v = 1000 + 10 * torch.randn(n, 40000, generator=g)
        ch = rc.choose(n, gesd_above_50=True)
        lo, hi = rc.reject(v, ch)
        assert 100 * float((lo | hi).float().mean()) <= top, (n, ch.method)


def test_combine_weights_the_kept_values_and_falls_back_to_the_median():
    v = torch.tensor([[10.0, 1.0, float("nan")], [20.0, 2.0, float("nan")], [40.0, 3.0, float("nan")]])
    keep = torch.tensor([[True, False, False], [True, False, False], [False, False, False]])
    w = torch.tensor([1.0, 3.0, 5.0])
    mean, den, k = rc.combine(v, keep, w)
    assert float(mean[0]) == pytest.approx((10 + 3 * 20) / 4) and float(den[0]) == 4
    assert float(mean[1]) == 2.0 and int(k[1]) == 0               # nothing kept: the median
    assert math.isnan(float(mean[2]))


# ------------------------------------------------------------------------------------------------ the whole path
H, W = 72, 88
NF = 16
SIGMA = 4.0
TRAIL_FRAME, CR_FRAME = 3, 6


def sky(X, Y):
    """The sky on the reference grid (3, ...) at positions X, Y: gradients and stars of several colours."""
    base = torch.stack([300 + 0.8 * X, 400 + 0.5 * Y, 250 + 0.3 * (X + Y)])
    stars = [(20.3, 18.6, 3000, (0.8, 1.0, 0.6)), (60.7, 40.2, 2000, (1.0, 0.9, 1.1)), (35.1, 55.4, 2500, (0.6, 1.0, 1.3)),
             (70.2, 15.9, 1500, (1.2, 1.0, 0.7)), (45.5, 30.5, 1200, (1.0, 1.0, 1.0))]
    for x0, y0, a, col in stars:
        p = a * torch.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * 1.6 ** 2))
        base = base + torch.stack([p * col[0], p * col[1], p * col[2]])
    return base


def make_frames():
    rng = np.random.default_rng(21)
    coefs, frames = [], []
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    col = db.colour_of("RGGB", H, W)
    scale = rng.uniform(0.8, 1.1, (NF, 3))
    off = rng.uniform(-40, 40, (NF, 3))
    Q = np.zeros((NF, 3, 6))
    Q[:, :, 0] = off
    for i in range(NF):
        dx, dy, th = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-0.01, 0.01)
        c = np.zeros((2, 10))
        c[0, :3] = [W / 2 + dx, math.cos(th) * W / 2, -math.sin(th) * H / 2]
        c[1, :3] = [H / 2 + dy, math.sin(th) * W / 2, math.cos(th) * H / 2]
        X, Y = nl.poly_apply(c, xx, yy, W, H)
        truth = sky(X, Y)
        cfa = torch.gather(truth, 0, col[None])[0].numpy()
        cfa = cfa * scale[i][col.numpy()] + off[i][col.numpy()] + rng.normal(0, SIGMA, (H, W))
        if i == TRAIL_FRAME:                           # a satellite: a bright line across the frame
            d = np.abs(yy.numpy() - (0.4 * xx.numpy() + 10))
            cfa += 1500 * (d < 1.2)
        if i == CR_FRAME:
            cfa[30, 41] += 6000                        # a cosmic ray
        cfa[10, 12] = cfa[50, 70] = np.nan             # bad pixels of the sensor
        coefs.append(c)
        frames.append(cfa.astype(np.float32))
    return np.array(coefs), frames, sc.Norm(scale, Q)


def test_the_classic_path_stacks_a_known_sky(tmp_path):
    coef, frames, norm = make_frames()
    res = sc.stack_classic(lambda i: frames[i], coef, None, norm, np.ones(NF), sc.ClassicParams(half_block=4),
                           "cpu", used=range(NF), ch_of_k=[0, 1, 1, 2], H=H, W=W)
    assert res.choice.method == "linear_fit"
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    truth = sky(xx, yy).numpy()
    m = 12
    err = (res.full - truth)[:, m:-m, m:-m]
    starless = (truth[1] - (400 + 0.5 * yy.numpy()))[m:-m, m:-m] < 20       # away from the stars
    assert np.isfinite(res.full).all()
    assert float(np.median(np.abs(err))) < 2.0                       # the noise of 16 frames, ~1 ADU
    assert float(np.abs(err).max()) < 0.12 * 3000                    # star peaks, through RCD and Lanczos
    # the trail and the cosmic ray are gone from the stack, and counted
    d = np.abs(yy.numpy() - (0.4 * xx.numpy() + 10))
    on = (d < 1.0)[m:-m, m:-m] & starless
    assert float(np.abs(err[:, on]).max()) < 25
    fr = {f["index"]: f for f in res.frames}
    assert fr[TRAIL_FRAME]["rejected_high"] > 3 * 30
    assert fr[CR_FRAME]["rejected_high"] >= 3
    clean = [f["rejected_high"] for i, f in fr.items() if i not in (TRAIL_FRAME, CR_FRAME)]
    assert fr[TRAIL_FRAME]["rejected_high"] > 10 * max(np.median(clean), 1)
    assert 0 < res.rejected_pct < 3 and int(res.rejection_high.sum()) > 0
    # bad pixels filled, weights, halves, noise
    assert all(f["bad_filled"] == 2 for f in res.frames)
    assert res.weight.shape == (3, H, W) and float(np.median(res.weight[:, m:-m, m:-m])) == NF
    assert res.A.shape == res.B.shape == (3, H, W) and np.abs(res.A - res.B)[:, m:-m, m:-m].mean() > 0
    for c in "RGB":
        assert 0.2 * SIGMA / math.sqrt(NF) < res.noise[c] < 2.0 * SIGMA / math.sqrt(NF)
    res.save(tmp_path)
    meta = json.loads((tmp_path / "classic.json").read_text(encoding="utf-8"))
    assert meta["meta"]["method"] == "classic" and meta["diagnostics"]["rejection"]["method"] == "linear_fit"
    z = np.load(tmp_path / "rejection.npz")
    assert set(z.files) == {"low", "high", "factor", "grid"} and list(z["grid"]) == [W, H]
    for name in ("drz_full.npy", "drz_A.npy", "drz_B.npy", "drz_weight.npy", "saturation.npy"):
        assert (tmp_path / name).exists()


def test_few_frames_take_the_percentile_and_two_frames_no_rejection():
    coef, frames, norm = make_frames()
    res = sc.stack_classic(lambda i: frames[i], coef, None, norm, None, sc.ClassicParams(), "cpu",
                           used=[0, 1, 2, TRAIL_FRAME], ch_of_k="RGGB", H=H, W=W)
    assert res.choice.method == "percentile"
    assert {f["index"]: f for f in res.frames}[TRAIL_FRAME]["rejected_high"] > 3 * 20
    res = sc.stack_classic(lambda i: frames[i], coef, None, norm, None, sc.ClassicParams(), "cpu",
                           used=[0, 1], ch_of_k="RGGB", H=H, W=W)
    assert res.choice.method == "none" and res.rejected_pct == 0
