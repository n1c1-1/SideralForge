"""The Bayer drizzle's per-sample loops for the CPU, compiled by Numba: one loop that does a sample's whole computation,
instead of generic array operations that each walk the 12 million samples and allocate a temporary array (about 15
times faster for the deposit). They compute what the PyTorch code of stages/stack.py computes on a GPU, in the same
precision (float32); tests/test_stack_kernels.py keeps the two the same.

``cache=True``: compiled once, kept next to the module.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange, set_num_threads


@njit(nogil=True, parallel=True, cache=True)
def _drops(num, den, v, xo, yo, c, starts, ends, w, h, Ho, Wo, span):
    half = np.float32(0.5)
    for b in prange(len(starts)):
        for i in range(starts[b], ends[b]):
            x, y = xo[i], yo[i]
            jx0, iy0 = np.floor(x - h + half), np.floor(y - h + half)
            for di in range(span):
                iy = iy0 + np.float32(di)
                wy = min(y + h, iy + half) - max(y - h, iy - half)
                if wy <= 0 or iy < 0 or iy >= Ho:
                    continue
                for dj in range(span):
                    jx = jx0 + np.float32(dj)
                    wx = min(x + h, jx + half) - max(x - h, jx - half)
                    if wx <= 0 or jx < 0 or jx >= Wo:
                        continue
                    ww = wx * wy * w
                    k = int(c[i]) * Ho * Wo + int(iy) * Wo + int(jx)
                    num[k] += v[i] * ww
                    if den.size:
                        den[k] += ww


def deposit(num, den, v, xo, yo, c, bands: list[tuple[int, int]], w: float, h: float, Ho: int, Wo: int, span: int,
            workers: int) -> None:
    """The drops of the samples (numpy arrays) in ``bands`` (stack.deposit_bands: the even ones, then the odd ones,
    never write to the same rows). ``den``: an empty array when the caller has none."""
    set_num_threads(max(1, workers))
    starts, ends = np.array([b[0] for b in bands], np.int64), np.array([b[1] for b in bands], np.int64)
    for g in (0, 1):
        _drops(num, den, v, xo, yo, c, starts[g::2], ends[g::2], np.float32(w), np.float32(h), Ho, Wo, span)


@njit(nogil=True, parallel=True, cache=True)
def _sample(mv, valid, cover, bkg, x, y, c, pred, w, cov, bk, Ho, Wo):
    for i in prange(len(x)):
        x0, y0 = np.floor(x[i]), np.floor(y[i])
        ax, ay = x[i] - x0, y[i] - y0
        ch = int(c[i])
        ix, iy = int(x0), int(y0)
        p = wg = cv = bg = np.float32(0)
        for dy in range(2):
            ky = ay if dy else np.float32(1) - ay
            for dx in range(2):
                kx = ax if dx else np.float32(1) - ax
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < Wo and 0 <= jy < Ho:
                    k = kx * ky
                    p += k * mv[ch, jy, jx]
                    wg += k * valid[ch, jy, jx]
                    cv += k * cover[ch, jy, jx]
                    bg += k * bkg[ch, jy, jx]
        pred[i], w[i], cov[i], bk[i] = p, wg, cv, bg


def sample(mv, valid, cover, bkg, x, y, c, workers: int):
    """Bilinear values of (3, Ho, Wo) fields at the samples (x, y), each in its own channel ``c``; a place outside
    counts as 0, as grid_sample's zeros padding: (mean x valid, valid, cover, background)."""
    set_num_threads(max(1, workers))
    n = x.shape[0]
    out = [np.empty(n, np.float32) for _ in range(4)]
    _sample(mv, valid, cover, bkg, x, y, c, *out, mv.shape[1], mv.shape[2])
    return out
