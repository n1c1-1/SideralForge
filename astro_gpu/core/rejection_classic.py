"""Pixel rejection of the Classic stacking method, chosen by the number of frames as PixInsight's WBPP does (stacking
v2: docs/spec/STACKING.md point 2, docs/pilot/STATE_OF_THE_ART_STACKING.md sections 4 and 9.2).

The rule (WBPP 3.x, BPP-Global.js: the automatic rejection by the size of the group [P15]; the same ranges in the
documentation of PixInsight [P1] and of Siril [S1]):

    frames      method                        low / high        formulas from
    < 3         none (too few to tell an outlier from the others; said in the reason)
    3 to 5      percentile clipping           0.2 / 0.1         Siril = PixInsight [S6][P20]
    6 to 15     winsorized sigma clipping     3 / 3             Siril's code (its default thresholds) [S2][S6]
    16 and up   linear fit clipping           5 / 3.5           PixInsight's code (WBPP's thresholds) [P15][P20]
    option > 50 generalized ESD test (GESDT)  0.3 / 0.05        Siril's code (its defaults) [S1][S6]

Each method is the published formula of the tool whose thresholds it uses (the thresholds mean nothing with another
tool's formula: Siril's linear fit divides by the mean absolute deviation, PixInsight's by twice it):

* **Percentile** (Siril ``percentile_clipping`` = PixInsight PercentileClipRejectionThread): a value is rejected low when
  median - v > low x median, high when v - median > high x median; one pass; the median of an even count is the mean
  of the two middle values. Nothing with fewer than 2 values or a zero median.
* **Winsorized sigma clipping** (Siril ``apply_rejection_float``, WINSORIZED; PixInsight's RejectionWinsorization is
  the same algorithm): sigma = the standard deviation (n - 1) and m = the median of the values; the values are then
  winsorized again and again at m +- 1.5 sigma, sigma = 1.134 x their standard deviation, until sigma changes by at
  most 0.05 %; a value is rejected when m - v > low x sigma or v - m > high x sigma; repeated on what is left while
  something was rejected and more than 3 values are left. Siril's guard: no rejection once n - r <= 4 (r: the
  rejections so far). Siril applies it in the order of its array (scrambled by its quickselect median); here, when it
  binds, the most deviant values go first.
* **Linear fit clipping** (PixInsight LinearFitRejectionThread, ImageIntegration code of 2016 [P20]): the values are
  sorted and a straight line y = a + b k is fitted to them against their rank k = 0..n-1 by minimising the mean
  absolute deviation (PCL's LinearFit; the median fit of Numerical Recipes, ``medfit`` / ``rofunc``, done here);
  sigma = 2 x that mean absolute deviation ("mult. by 2 to compatibilize with sigma clipping"); a value below the
  line is rejected when (line - v) / sigma >= low, above it when (v - line) / sigma >= high; repeated on what is left
  while something was rejected and at least 3 values are left. Needs 5 values to start.
* **GESDT** (Siril ``apply_rejection_float``, GESDT, with the critical values of ``stack_mean_or_median``; Rosner 1983
  [L8]): at most floor(0.3 x N) outliers (N the frames of the stack, missing values counted as already removed); at
  each step the value farthest from the mean (in standard deviations, n - 1) of what is left is set aside and its
  Grubbs statistic compared with lambda_j = (n - j - 1) t / sqrt((n - j) (n - j - 2 + t^2)), t the Student quantile
  of 1 - alpha / (2 (n - j)) with n - j - 2 degrees of freedom, alpha = 0.05; the outliers are all those set aside up
  to the last step whose statistic exceeds its lambda; above the median: high, below: low.

Missing values (NaN: the frame does not cover the pixel) are left out of every statistic, as Siril leaves out its
zeros. The rejections are per frame, colour and pixel; ``combine`` makes the weighted mean of what is kept, as Siril's
``mean_and_reject`` does (with the frame weights; the median of all values where nothing is kept).

All functions work on a stack ``v`` (N frames, P pixels) on any device, by chunks the caller chooses.

Sources: [S1] https://siril.readthedocs.io/en/latest/preprocessing/stacking.html ; [S2] Siril's ``stack`` command;
[S6] Siril 1.4 src/stacking/rejection_float.c, median_and_mean.c, siril_fit_linear.c; [P1] ImageIntegration
documentation (2013); [P15] WBPP (BatchPreprocessing scripts); [P20] PixInsight ImageIntegration code (2016:
ImageIntegrationInstance.cpp), PCL include/pcl/LinearFit.h; Numerical Recipes in C, 2nd ed., section 15.7 (medfit,
rofunc); [L8] NIST/SEMATECH e-Handbook, generalized ESD test (Rosner 1983).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

METHODS = ("none", "percentile", "winsorized", "linear_fit", "gesdt")
#: the published thresholds (low, high) of each method (see the module's docstring)
DEFAULTS = {"percentile": (0.2, 0.1), "winsorized": (3.0, 3.0), "linear_fit": (5.0, 3.5), "gesdt": (0.3, 0.05)}
#: WBPP's ranges: percentile below this many frames, winsorized below LINEAR_FROM, linear fit from it on
WINSORIZED_FROM = 6
LINEAR_FROM = 16
#: the GESDT option takes over above this many frames (Siril: "excellent above 50")
GESD_ABOVE = 50
#: fewest frames for any rejection (the drizzle's rule too: under 3 nothing can be told apart)
MIN_FRAMES = 3
#: Numerical Recipes' rofunc tolerance
NR_EPS = 1.0e-7

REASONS = {
    "few": "{n} frames: too few to tell an outlier from the others, no rejection",
    "percentile": "{n} frames: percentile clipping ({low:g} / {high:g}), as PixInsight's WBPP below 6 frames",
    "winsorized": "{n} frames: winsorized sigma clipping ({low:g} / {high:g} σ), as PixInsight's WBPP from 6 to 15 "
                  "frames, with Siril's thresholds",
    "linear_fit": "{n} frames: linear fit clipping ({low:g} / {high:g} σ), as PixInsight's WBPP from 16 frames",
    "gesdt": "{n} frames: generalized ESD test (up to {low:.0%} outliers, significance {high:g}), as Siril advises "
             "above 50 frames",
    "chosen": "{method} chosen ({low:g} / {high:g})",
}


@dataclass(frozen=True)
class Choice:
    """The rejection to run: method, thresholds and the reason (an English template and its numbers)."""
    method: str
    low: float
    high: float
    template: str
    args: dict = field(default_factory=dict)

    @property
    def reason(self) -> str:
        return self.template.format(**self.args)

    def as_dict(self) -> dict:
        return {"method": self.method, "low": self.low, "high": self.high, "reason": self.reason}


def choose(n_frames: int, method: str = "auto", low: float | None = None, high: float | None = None,
           gesd_above_50: bool = False) -> Choice:
    """The method for a stack of ``n_frames`` frames: WBPP's rule with ``method="auto"`` (GESDT above 50 frames with
    ``gesd_above_50``), else the method named; ``low`` / ``high`` replace its published thresholds."""
    n = int(n_frames)
    if method == "auto":
        if n < MIN_FRAMES:
            return Choice("none", 0.0, 0.0, REASONS["few"], {"n": n})
        if n < WINSORIZED_FROM:
            m = "percentile"
        elif n < LINEAR_FROM:
            m = "winsorized"
        elif gesd_above_50 and n > GESD_ABOVE:
            m = "gesdt"
        else:
            m = "linear_fit"
        lo, hi = DEFAULTS[m]
        lo, hi = (lo if low is None else low), (hi if high is None else high)
        return Choice(m, lo, hi, REASONS[m], {"n": n, "low": lo, "high": hi})
    if method not in METHODS:
        raise ValueError(f"unknown rejection {method!r}: one of {('auto',) + METHODS}")
    if method == "none":
        return Choice("none", 0.0, 0.0, "no rejection (chosen)", {})
    lo, hi = DEFAULTS[method]
    lo, hi = (lo if low is None else low), (hi if high is None else high)
    return Choice(method, lo, hi, REASONS["chosen"], {"method": method, "low": lo, "high": hi})


# ------------------------------------------------------------------------------------------------ statistics
def _count(valid: torch.Tensor) -> torch.Tensor:
    return valid.sum(0)


def median(v: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """(P,) the median of the valid values of each column of v (N, P) (NaN where none), the mean of the two middle
    values for an even count (gsl's and Siril's quickmedian, PixInsight's RejectionMedian)."""
    valid = torch.isfinite(v) if valid is None else valid
    x = torch.where(valid, v, torch.full_like(v, float("inf")))
    s = torch.sort(x, dim=0).values
    n = _count(valid)
    i1 = ((n - 1).clamp(min=0) // 2)
    i2 = (n // 2).clamp(max=v.shape[0] - 1)
    m = 0.5 * (s.gather(0, i1[None])[0] + s.gather(0, i2[None])[0])
    return torch.where(n > 0, m, torch.full_like(m, float("nan")))


def std(v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(P,) the standard deviation (n - 1) of the valid values (0 under 2 values), in float64."""
    x = torch.where(valid, v.double(), torch.zeros((), dtype=torch.float64, device=v.device))
    n = _count(valid).double()
    mean = x.sum(0) / n.clamp(min=1)
    d = torch.where(valid, v.double() - mean, torch.zeros((), dtype=torch.float64, device=v.device))
    var = (d * d).sum(0) / (n - 1).clamp(min=1)
    return torch.where(n >= 2, var.sqrt(), torch.zeros_like(var))


# ------------------------------------------------------------------------------------------------ the methods
def percentile_clip(v: torch.Tensor, low: float, high: float, valid: torch.Tensor | None = None):
    """(low, high) rejected (N, P): Siril's / PixInsight's percentile clipping."""
    valid = torch.isfinite(v) if valid is None else valid
    med = median(v, valid)
    ok = valid & ((_count(valid) >= 2) & (med != 0) & torch.isfinite(med))[None]
    lo = ok & (med[None] - v > low * med[None])
    hi = ok & (v - med[None] > high * med[None])
    return lo, hi


def winsorized_clip(v: torch.Tensor, low: float, high: float, valid: torch.Tensor | None = None,
                    max_rounds: int = 100, max_winsor: int = 200):
    """(low, high) rejected (N, P): Siril's winsorized sigma clipping (see the module's docstring).

    Each round runs on the whole band with the finished columns masked out, not on the columns ``torch.nonzero``
    picks: the bench of 1 November 2026 stopped twice here on PyTorch's bound check of such an index (an index_put
    in deterministic mode, ``TensorCompare.cu:109``) while the GPU driver logged errors on both cards in the same
    seconds and the machine was short of memory; that index cannot leave its range when the GPU computes right
    (40 bands of 1 Mpx with gaps, plateaus and outliers ran clean alone). Same results, about 1.4 x the time."""
    valid = torch.isfinite(v) if valid is None else valid
    N, P = v.shape
    x = v.double()
    nan = torch.full_like(x, float("nan"))
    lo = torch.zeros_like(valid)
    hi = torch.zeros_like(valid)
    active = valid.clone()
    r = torch.zeros(P, dtype=torch.long, device=v.device)
    go = _count(active) >= 2
    # every step works on the whole band, the finished columns masked out: no column index is ever computed on the
    # device, so no out-of-range index can reach an indexing kernel (see _guard_most_deviant)
    for _ in range(max_rounds):
        if not bool(go.any()):
            break
        act = active & go[None]
        n = _count(act)
        sigma = std(x, act)
        m = median(x, act)
        w = torch.where(act, x, nan)
        todo = go.clone()
        for _ in range(max_winsor):
            a, b = (m - 1.5 * sigma)[None], (m + 1.5 * sigma)[None]
            w2 = torch.where(todo[None], torch.minimum(torch.maximum(w, a), b), w)
            w = torch.where(torch.isfinite(w), w2, w)
            s0 = sigma
            s1 = 1.134 * std(w, act)
            sigma = torch.where(todo, s1, sigma)
            todo = todo & ((sigma - s0).abs() > s0 * 0.0005)
            if not bool(todo.any()):
                break
        cl = act & (m[None] - x > sigma[None] * low)
        ch = act & (x - m[None] > sigma[None] * high)
        cand = cl | ch
        allowed = (n - r - 4).clamp(min=0)
        if bool((cand.sum(0) > allowed).any()):     # Siril's guard: the most deviant first
            cand = _guard_most_deviant(cand, (x - m[None]).abs(), allowed)
            cl, ch = cl & cand, ch & cand
        nrej = cand.sum(0)
        lo |= cl
        hi |= ch
        active &= ~cand
        r += nrej
        go &= (nrej > 0) & (n - nrej > 3)
    return lo, hi


def _guard_most_deviant(cand: torch.Tensor, dev: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """The candidates ``cand`` (N, P) cut to the ``allowed`` (P,) most deviant of each column (``dev`` (N, P), ties in
    frame order): by a threshold on the sorted deviations and a running count, element-wise, with no index written
    or gathered from a device-computed permutation."""
    N = cand.shape[0]
    d = torch.where(cand, dev, torch.full_like(dev, -1.0))
    s = torch.sort(d, dim=0, descending=True).values
    k = (allowed - 1).clamp(min=0, max=N - 1)                  # arithmetic on counts: always in [0, N - 1]
    kth = s.gather(0, k[None])[0]
    greater = cand & (d > kth[None])
    need = allowed - greater.sum(0)
    eq = cand & (d == kth[None])
    keep = greater | (eq & (eq.long().cumsum(0) <= need[None]))
    keep = keep & (allowed > 0)[None]
    return torch.where((cand.sum(0) > allowed)[None], keep, cand)


def _gather_sorted(v: torch.Tensor, valid: torch.Tensor):
    """The valid values of each column sorted ascending and moved to the top (NaN below), their count and the
    permutation (sorted row -> frame)."""
    x = torch.where(valid, v, torch.full_like(v, float("inf")))
    s, perm = torch.sort(x, dim=0)
    n = _count(valid)
    s = torch.where(torch.arange(v.shape[0], device=v.device)[:, None] < n[None], s, torch.full_like(s, float("nan")))
    return s, n, perm


def _rofunc(y: torch.Tensor, x: torch.Tensor, valid: torch.Tensor, n: torch.Tensor, b: torch.Tensor):
    """Numerical Recipes' rofunc for every row of y (M, K) (the first n valid): sum_j x_j sgn(y_j - b x_j - a), with
    a the median of y - b x; returns (that sum, a, sum |y - a - b x|)."""
    arr = torch.where(valid, y - b[:, None] * x, torch.full_like(y, float("inf")))
    s = torch.sort(arr, dim=1).values
    i1 = ((n - 1).clamp(min=0) // 2)[:, None]
    i2 = (n // 2).clamp(max=y.shape[1] - 1)[:, None]
    aa = 0.5 * (s.gather(1, i1)[:, 0] + s.gather(1, i2)[:, 0])
    d = y - (b[:, None] * x + aa[:, None])
    abdev = torch.where(valid, d.abs(), torch.zeros_like(d)).sum(1)
    dd = torch.where(y != 0, d / y.abs(), d)
    f = torch.where(valid & (dd.abs() > NR_EPS), torch.where(d >= 0, x, -x), torch.zeros_like(x)).sum(1)
    return f, aa, abdev


def l1_line(y: torch.Tensor, n: torch.Tensor, max_iter: int = 100):
    """(a, b, mean absolute deviation) (M,) of the straight line y = a + b k minimising the mean absolute deviation
    of the first n values of each row of y (M, K) against their index k (Numerical Recipes' medfit, PCL's LinearFit);
    two points: the line through them."""
    M, K = y.shape
    y = y.double()
    dev = y.device
    x = torch.arange(K, device=dev, dtype=torch.float64)[None].expand(M, K)
    valid = x < n[:, None]
    nf = n.double()
    xv = torch.where(valid, x, torch.zeros_like(x))
    yv = torch.where(valid, y, torch.zeros_like(y))
    sx, sy = xv.sum(1), yv.sum(1)
    sxy, sxx = (xv * yv).sum(1), (xv * xv).sum(1)
    det = nf * sxx - sx * sx
    det_ok = det != 0
    dets = torch.where(det_ok, det, torch.ones_like(det))
    aa = (sxx * sy - sx * sxy) / dets
    bb = torch.where(det_ok, (nf * sxy - sx * sy) / dets, torch.zeros_like(det))
    chisq = torch.where(valid, (y - (aa[:, None] + bb[:, None] * x)) ** 2, torch.zeros_like(y)).sum(1)
    sigb = torch.sqrt(torch.where(det_ok, chisq / dets, torch.zeros_like(det)))
    b1 = bb.clone()
    f1 = _rofunc(y, x, valid, n, b1)[0]
    b2 = bb + torch.where(f1 >= 0, 3.0 * sigb, -3.0 * sigb)
    f2 = _rofunc(y, x, valid, n, b2)[0]
    # a slope where rofunc is 0 is a minimum (the minimum of an L1 fit can be a whole interval): stop there
    # (Numerical Recipes' loop would take a zero for a sign and leave the interval)
    last = torch.where((sigb > 0) & (f1 != 0), b2, b1)
    done = (sigb <= 0) | (b2 == b1) | (n < 3) | (f1 == 0)
    hit = ~done & (f2 == 0)
    done = done | hit
    for _ in range(max_iter):                       # bracket the root of rofunc
        act = ~done & (f1 * f2 > 0)
        if not bool(act.any()):
            break
        i = torch.nonzero(act, as_tuple=True)[0]
        bn = b2[i] + 1.6 * (b2[i] - b1[i])
        b1[i], f1[i], b2[i] = b2[i], f2[i], bn
        f2[i] = _rofunc(y[i], x[i], valid[i], n[i], bn)[0]
        last[i] = bn
        done[i] = done[i] | (f2[i] == 0)
    tol = 0.01 * sigb
    for _ in range(max_iter):                       # then bisect it
        act = ~done & ((b2 - b1).abs() > tol)
        if not bool(act.any()):
            break
        i = torch.nonzero(act, as_tuple=True)[0]
        bm = b1[i] + 0.5 * (b2[i] - b1[i])
        stuck = (bm == b1[i]) | (bm == b2[i])
        done[i[stuck]] = True
        i, bm = i[~stuck], bm[~stuck]
        if i.numel() == 0:
            continue
        f = _rofunc(y[i], x[i], valid[i], n[i], bm)[0]
        done[i] = done[i] | (f == 0)
        same = f * f1[i] > 0
        b1[i] = torch.where(same, bm, b1[i])
        f1[i] = torch.where(same, f, f1[i])
        b2[i] = torch.where(same, b2[i], bm)
        f2[i] = torch.where(same, f2[i], f)
        last[i] = bm
    b = torch.where(sigb > 0, last, bb)
    _, a, abdev = _rofunc(y, x, valid, n, b)
    adev = abdev / nf.clamp(min=1)
    two = n == 2                                     # PCL: the line through the two points, no deviation
    if bool(two.any()):
        y0, y1 = y[:, 0], y[:, 1]
        b = torch.where(two, y1 - y0, b)
        a = torch.where(two, y0, a)
        adev = torch.where(two, torch.zeros_like(adev), adev)
    return a, b, adev


def linear_fit_clip(v: torch.Tensor, low: float, high: float, valid: torch.Tensor | None = None,
                    max_rounds: int = 100):
    """(low, high) rejected (N, P): PixInsight's linear fit clipping (see the module's docstring)."""
    valid = torch.isfinite(v) if valid is None else valid
    N, P = v.shape
    lo = torch.zeros_like(valid)
    hi = torch.zeros_like(valid)
    active = valid.clone()
    go = _count(active) >= 5
    ar = torch.arange(N, device=v.device, dtype=torch.float64)
    for _ in range(max_rounds):
        idx = torch.nonzero(go, as_tuple=True)[0]
        if idx.numel() == 0:
            break
        s, n, perm = _gather_sorted(v[:, idx].double(), active[:, idx])
        a, b, adev = l1_line(s.T, n)
        sigma = 2.0 * adev
        line = a[None] + b[None] * ar[:, None]
        ok = (ar[:, None] < n[None]) & (sigma > 0)[None]
        sg = torch.where(sigma > 0, sigma, torch.ones_like(sigma))[None]
        below = s < line
        rl = ok & below & ((line - s) / sg >= low)
        rh = ok & ~below & ((s - line) / sg >= high)
        nc = (rl | rh).sum(0)
        fl = torch.zeros_like(rl).scatter_(0, perm, rl)
        fh = torch.zeros_like(rh).scatter_(0, perm, rh)
        lo[:, idx] |= fl
        hi[:, idx] |= fh
        active[:, idx] &= ~(fl | fh)
        go[idx] = (nc > 0) & (n - nc >= 3)
    return lo, hi


_CRIT: dict = {}


def gesd_critical(nb_frames: int, frac: float = 0.3, alpha: float = 0.05) -> np.ndarray:
    """Siril's GESDT critical values lambda_j, j = 0 .. floor(frac x nb_frames) - 1 (median_and_mean.c)."""
    key = (int(nb_frames), float(frac), float(alpha))
    if key not in _CRIT:
        from scipy.stats import t as student
        m = int(math.floor(nb_frames * frac))
        out = np.zeros(m)
        for j in range(m):
            size = nb_frames - j
            if size - 2 <= 0:
                out[j] = np.inf
                continue
            tq = float(student.ppf(1 - alpha / (2 * size), size - 2))
            out[j] = (size - 1) * tq / (math.sqrt(size) * math.sqrt(size - 2 + tq * tq))
        _CRIT[key] = out
    return _CRIT[key]


def gesd_clip(v: torch.Tensor, frac: float = 0.3, alpha: float = 0.05, valid: torch.Tensor | None = None):
    """(low, high) rejected (N, P): Siril's generalized ESD test (see the module's docstring)."""
    valid = torch.isfinite(v) if valid is None else valid
    N, P = v.shape
    dev = v.device
    lo = torch.zeros_like(valid)
    hi = torch.zeros_like(valid)
    crit = gesd_critical(N, frac, alpha)
    max_out = len(crit)
    if max_out == 0:
        return lo, hi
    s, n, perm = _gather_sorted(v.double(), valid)
    removed = N - n
    m_out = (max_out - removed).clamp(min=0)
    med = median(s, torch.isfinite(s))
    z = torch.where(torch.isfinite(s), s, torch.zeros_like(s))
    S1 = torch.cat([torch.zeros(1, P, dtype=torch.float64, device=dev), z.cumsum(0)])
    S2 = torch.cat([torch.zeros(1, P, dtype=torch.float64, device=dev), (z * z).cumsum(0)])
    crit_t = torch.as_tensor(np.append(crit, np.inf), dtype=torch.float64, device=dev)
    a = torch.zeros(P, dtype=torch.long, device=dev)            # what is left: sorted rows [a, b)
    b = n.clone()
    outs, pos, val = [], [], []
    for it in range(max_out):
        act = (it < m_out) & (b - a >= 3)
        cnt = (b - a).double().clamp(min=1)
        mean = (S1.gather(0, b[None])[0] - S1.gather(0, a[None])[0]) / cnt
        var = (S2.gather(0, b[None])[0] - S2.gather(0, a[None])[0] - cnt * mean * mean) / (cnt - 1).clamp(min=1)
        sd = var.clamp(min=0).sqrt()
        vlo = s.gather(0, a.clamp(max=N - 1)[None])[0]
        vhi = s.gather(0, (b - 1).clamp(min=0)[None])[0]
        dlo, dhi = mean - vlo, vhi - mean
        top = dhi > dlo
        g = torch.where(top, dhi, dlo) / torch.where(sd > 0, sd, torch.full_like(sd, float("inf")))
        lam = crit_t.gather(0, (removed + it).clamp(max=max_out))
        outs.append(act & (g > lam))
        pos.append(torch.where(top, b - 1, a))
        val.append(torch.where(top, vhi, vlo))
        a = torch.where(act & ~top, a + 1, a)
        b = torch.where(act & top, b - 1, b)
    O = torch.stack(outs)                                        # (max_out, P)
    steps = torch.arange(max_out, device=dev)[:, None]
    last = torch.where(O, steps, torch.full_like(steps, -1)).max(0).values
    rej = steps <= last[None]
    Pos, Val = torch.stack(pos), torch.stack(val)
    high_side = Val >= med[None]
    rs_lo = torch.zeros((N, P), dtype=torch.bool, device=dev)
    rs_hi = torch.zeros_like(rs_lo)
    for k in range(max_out):                                     # few steps: scatter one by one
        p = Pos[k][None]
        rs_lo.scatter_(0, p, rs_lo.gather(0, p) | (rej[k] & ~high_side[k])[None])
        rs_hi.scatter_(0, p, rs_hi.gather(0, p) | (rej[k] & high_side[k])[None])
    lo.scatter_(0, perm, rs_lo)
    hi.scatter_(0, perm, rs_hi)
    return lo & valid, hi & valid


def reject(v: torch.Tensor, choice: Choice, valid: torch.Tensor | None = None):
    """(low, high) rejected (N, P) by the method of ``choice`` (``choose``)."""
    valid = torch.isfinite(v) if valid is None else valid
    if choice.method == "none":
        z = torch.zeros_like(valid)
        return z, z.clone()
    if choice.method == "percentile":
        return percentile_clip(v, choice.low, choice.high, valid)
    if choice.method == "winsorized":
        return winsorized_clip(v, choice.low, choice.high, valid)
    if choice.method == "linear_fit":
        return linear_fit_clip(v, choice.low, choice.high, valid)
    if choice.method == "gesdt":
        return gesd_clip(v, choice.low, choice.high, valid)
    raise ValueError(f"unknown rejection {choice.method!r}")


def combine(v: torch.Tensor, keep: torch.Tensor, w: torch.Tensor):
    """(mean, sum of weights, kept count) (P,): the weighted mean of the kept values of v (N, P) with the frame
    weights w (N,); where the kept values weigh nothing, their plain mean; where nothing is kept, the median of all
    the values (Siril's mean_and_reject); NaN where there is no value."""
    wk = torch.where(keep, w.to(v.dtype)[:, None].expand_as(v), torch.zeros_like(v))
    x = torch.where(keep, v, torch.zeros_like(v))
    num = (wk * x).sum(0)
    den = wk.sum(0)
    k = keep.sum(0)
    plain = x.sum(0) / k.clamp(min=1)
    mean = torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)), plain)
    valid = torch.isfinite(v)
    med = median(v, valid)
    mean = torch.where(k > 0, mean, med)
    return mean, den, k
