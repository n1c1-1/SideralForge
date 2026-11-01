"""Debayering (demosaicing) of a calibrated CFA frame, for the Classic stacking method (stacking v2:
docs/spec/STACKING.md point 1, docs/pilot/STATE_OF_THE_ART_STACKING.md sections 1, 8 and 9.1).

Two methods, torch, on the device of the frame (GPU or CPU):

* ``rcd`` (default): **Ratio Corrected Demosaicing**, Luis Sanz Rodríguez, release 2.3 (2017-11-25) [L10], exactly as
  Siril runs it (Siril's default since 1.2 [S6][S11]: ``src/algos/demosaicing_rtp.cpp`` calls librtprocess's
  ``rcd_demosaic``). The reference code is librtprocess ``src/demosaic/rcd.cc`` (RawTherapee's tiled version by Ingo
  Weyrich, with the author's exact shorter form of the directional statistics) and ``src/demosaic/border.cc``
  (``bayerborder_demosaic``); this module follows it step by step:

  - Siril's float path first maps the frame from [min, max] to [0, 65535] (``debayer_buffer_new_float``) and back
    afterwards; rcd.cc divides by 65536 and clips to [0, 1] (LIM01), and its output is max(0, value) x 65536;
  - step 1: vertical / horizontal discrimination VH = V / (V + H), V and H the sums over three rows (columns) of the
    squared colour-difference high-pass ``(c[-3] - c[-1] - c[+1] + c[+3]) - 3 (c[-2] + c[+2]) + 6 c``; this short form
    equals the long quadratic form of the original v2.3 (tests/test_debayer.py checks it to float precision);
  - step 2: the low-pass ``c + 0.5 (N + S + W + E) + 0.25 (NW + NE + SW + SE)`` at the red and blue samples;
  - step 3: green at red and blue: ratio-corrected cardinal estimates ``c_N 2 lpf / (eps + lpf + lpf_N)`` weighted by
    the inverse cardinal gradients, blended by VH (or by the mean VH of the four diagonal neighbours when that one is
    more decided);
  - step 4: red at blue and blue at red along the two diagonals (colour differences, P/Q discrimination), then red and
    blue at green along the cardinals (colour differences, VH discrimination);
  - the image is processed in tiles of 194 pixels overlapping by 9 (as rcd.cc: each tile keeps its own zeroed
    discrimination buffer, which the result depends on slightly near the tile seams), and the outer 9 pixels of the
    frame are filled by ``bayerborder_demosaic``: the mean of the samples of each colour in the 3 x 3 neighbourhood;
  - rcd.cc stores the diagonal high-pass terms of step 4 at half resolution (``indx / 2``) for the odd columns only,
    so that a sample reads the term of the odd column of its pair (x | 1), not always its own diagonal as RCD v2.3
    does. Kept as Siril runs it.

  Validation (1 November 2026, F:/astro_essais/classic_check/rcd): a calibrated frame of heart (Sony A7S, RGGB,
  4256 x 2848) debayered by Siril 1.4.0-beta3 (``calibrate_single -debayer``, RCD) and here: relative difference
  median 4e-7, 99.99th percentile 9e-6, rms 1.6e-7 of the [0, 1] range (signal rms 4.9e-3); 126 pixels of 12 million
  differ by more than 1e-6, where a rounding flips a direction choice. With the diagonal terms at their own place
  (RCD v2.3's intent) the difference grows 2700-fold (rms 4.3e-4): Siril runs librtprocess's indexing.
  tests/test_debayer.py repeats it (marked slow) when the files are there.

  RCD was chosen by Siril because it renders round, sharp-edged objects (stars) without the zipper and the
  overshoot of the older methods [S11].
* ``bilinear``: the textbook bilinear demosaicing, each missing colour the mean of the samples of that colour in the
  3 x 3 neighbourhood (DSS's default [D1]; librtprocess's ``bayerborder_demosaic`` formula everywhere). Not Siril's
  "bilinear", which is librtprocess's ``bayerfast`` (gradient-weighted green): a simple option, not the reference.

Before demosaicing, the samples the calibration left out (NaN in the calibrated cache: hot, cold and dusty pixels) are
filled as Siril's cosmetic correction fills a hot pixel of a CFA image (``getAverage3x3`` in
``src/filters/cosmetic_correction.c``: the mean of its 8 nearest samples of the same colour, 2 pixels away), here of
those that are valid; a sample with none valid gets the mean of the next ring, then the colour's median
(``fill_bad``). Saturated samples are the caller's to replace (stack_classic puts the saturation level there).

Sources: [L10] https://github.com/LuisSR/RCD-Demosaicing (rcd_demosaicing.c, release 2.3);
librtprocess https://github.com/CarVac/librtprocess (src/demosaic/rcd.cc, border.cc); Siril 1.4
https://gitlab.com/free-astro/siril/-/blob/1.4/src/algos/demosaicing_rtp.cpp and src/filters/cosmetic_correction.c;
[S11] https://siril.org/2020/09/whats-new-in-siril-0.99.4/ ; [D1] DeepSkyStacker technical help.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

METHODS = ("rcd", "bilinear")
#: rcd.cc: tile side, the overlap kept on each side of a tile, the frame border filled by bayerborder_demosaic
TILE = 194
TILE_BORDER = 9
RCD_BORDER = 9
STEP = TILE - 2 * TILE_BORDER
#: rcd.cc: tolerances against a division by zero
EPS = 1e-5
EPSSQ = 1e-10
#: rcd.cc divides by this (and multiplies back); Siril's float path maps the frame onto [0, NORM] first
SCALE = 65536.0
NORM = 65535.0
#: tiles processed together (memory: about 40 arrays of TILE x TILE floats per tile)
TILES_PER_BATCH = 96


def channel_map(cfa) -> list[int]:
    """The colour (0 R, 1 G, 2 B) of each 2 x 2 position k = 2 * row + column, from a pattern ("RGGB") or the list
    itself (io.frames.cfa_channel_map's)."""
    if isinstance(cfa, str):
        return ["RGB".index(c) for c in cfa.upper()]
    ch = [int(c) for c in cfa]
    if len(ch) != 4 or sorted(ch) != [0, 1, 1, 2]:
        raise ValueError(f"not a Bayer pattern: {cfa!r} (two greens, one red, one blue)")
    return ch


def colour_of(ch_of_k, H: int, W: int, device="cpu") -> torch.Tensor:
    """(H, W) long: the colour of every sensor pixel."""
    k = torch.tensor(channel_map(ch_of_k), device=device)
    yy = torch.arange(H, device=device)[:, None] % 2
    xx = torch.arange(W, device=device)[None, :] % 2
    return k[2 * yy + xx]


def _shifter(a: torch.Tensor, pad: int = 4):
    """at(dy, dx): the array a (..., h, w) read at (row + dy, column + dx), zero outside."""
    h, w = a.shape[-2:]
    ap = F.pad(a, (pad, pad, pad, pad))

    def at(dy: int, dx: int) -> torch.Tensor:
        return ap[..., pad + dy:pad + dy + h, pad + dx:pad + dx + w]
    return at


# ------------------------------------------------------------------------------------------------ bad samples
def fill_bad(cfa: torch.Tensor, bad: torch.Tensor | None = None, rounds: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """The CFA frame with its bad samples (NaN, or ``bad``) filled as Siril's cosmetic correction fills a hot pixel
    (getAverage3x3, CFA step 2): the mean of the valid samples of the same colour among the 8 nearest (2 pixels
    away); those with none take the mean of the next ring (4 pixels away), up to ``rounds`` times, then the median of
    their colour. Returns (filled frame, the mask of the samples filled)."""
    x = cfa.float()
    miss = torch.isnan(x)
    if bad is not None:
        miss = miss | bad.to(x.device, torch.bool)
    if not bool(miss.any()):
        return x, miss
    out = torch.where(miss, torch.zeros_like(x), x)
    todo = miss.clone()
    k = torch.ones(1, 1, 3, 3, device=x.device, dtype=x.dtype)
    k[0, 0, 1, 1] = 0
    for r in range(rounds):
        ok = (~todo).float()
        d = 2 * (r + 1)
        num = F.conv2d((out * ok)[None, None], k, padding=d, dilation=d)[0, 0]
        den = F.conv2d(ok[None, None], k, padding=d, dilation=d)[0, 0]
        got = todo & (den > 0)
        out = torch.where(got, num / den.clamp(min=1e-12), out)
        todo = todo & ~got
        if not bool(todo.any()):
            break
    if bool(todo.any()):                     # nothing valid nearby: the median of the samples of its 2 x 2 position
        par = 2 * (torch.arange(x.shape[0], device=x.device)[:, None] % 2) + \
            torch.arange(x.shape[1], device=x.device)[None, :] % 2
        for p in range(4):
            sel = (par == p)
            good = sel & ~miss
            med = x[good].median() if bool(good.any()) else torch.zeros((), device=x.device)
            out = torch.where(todo & sel, med, out)
    return out, miss


# ------------------------------------------------------------------------------------------------ bilinear
def bilinear(cfa: torch.Tensor, ch_of_k) -> torch.Tensor:
    """(3, H, W): each missing colour the mean of the samples of that colour in the 3 x 3 neighbourhood (inside the
    frame), each sample kept as it is."""
    x = cfa.float()
    H, W = x.shape
    col = colour_of(ch_of_k, H, W, x.device)
    k = torch.ones(1, 1, 3, 3, device=x.device, dtype=x.dtype)
    out = torch.empty((3, H, W), device=x.device, dtype=x.dtype)
    for c in range(3):
        m = (col == c).to(x.dtype)
        num = F.conv2d((x * m)[None, None], k, padding=1)[0, 0]
        den = F.conv2d(m[None, None], k, padding=1)[0, 0]
        out[c] = torch.where(col == c, x, num / den.clamp(min=1e-12))
    return out


# ------------------------------------------------------------------------------------------------ RCD
def _rcd_tiles(c: torch.Tensor, col: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """rcd.cc on a batch of tiles: ``c`` (b, T, T) the normalised CFA samples (0..1), ``col`` (T, T) the colour of
    each tile pixel (the tiles start on even rows and columns), ``rows`` / ``cols`` (b,) the real size of each tile
    (smaller at the right and bottom edges). Returns (3, b, T, T), the scale of rcd.cc's output (x 65536)."""
    b, T, _ = c.shape
    dev = c.device
    r = torch.arange(T, device=dev)
    R, C = r.view(1, T, 1), r.view(1, 1, T)
    tr, tc = rows.view(b, 1, 1), cols.view(b, 1, 1)

    def inside(k: int) -> torch.Tensor:
        return (R >= k) & (R < tr - k) & (C >= k) & (C < tc - k)

    green = (col == 1)[None]
    rb = ~green
    m4 = inside(4)
    cc = _shifter(c)
    zero = torch.zeros((), device=dev, dtype=c.dtype)

    # ---- step 1: the squared vertical / horizontal colour-difference high-pass, summed over three rows / columns
    v = (cc(-3, 0) - cc(-1, 0) - cc(1, 0) + cc(3, 0)) - 3.0 * (cc(-2, 0) + cc(2, 0)) + 6.0 * c
    v = v * v
    h = (cc(0, -3) - cc(0, -1) - cc(0, 1) + cc(0, 3)) - 3.0 * (cc(0, -2) + cc(0, 2)) + 6.0 * c
    h = h * h
    vs, hs = _shifter(v), _shifter(h)
    V = torch.clamp((vs(-1, 0) + v) + vs(1, 0), min=EPSSQ)
    Hs = torch.clamp((hs(0, -1) + h) + hs(0, 1), min=EPSSQ)
    VH = torch.where(m4, V / (V + Hs), zero)
    del v, h, vs, hs, V, Hs
    vhs = _shifter(VH)
    vh_n = 0.25 * ((vhs(-1, -1) + vhs(-1, 1)) + (vhs(1, -1) + vhs(1, 1)))
    vh_disc = torch.where((0.5 - VH).abs() < (0.5 - vh_n).abs(), vh_n, VH)
    del vhs, vh_n, VH

    # ---- step 2: the low-pass at the red and blue samples
    lpf = c + 0.5 * (((cc(-1, 0) + cc(1, 0)) + cc(0, -1)) + cc(0, 1)) + \
        0.25 * (((cc(-1, -1) + cc(-1, 1)) + cc(1, -1)) + cc(1, 1))
    lpf = torch.where(inside(2) & rb, lpf, zero)
    ls = _shifter(lpf)

    # ---- step 3: green at the red and blue samples
    n1, s1, w1, e1 = cc(-1, 0), cc(1, 0), cc(0, -1), cc(0, 1)
    n2, s2, w2, e2 = cc(-2, 0), cc(2, 0), cc(0, -2), cc(0, 2)
    ns = (n1 - s1).abs()
    we = (w1 - e1).abs()
    n_grad = EPS + (ns + (c - n2).abs()) + ((n1 - cc(-3, 0)).abs() + (n2 - cc(-4, 0)).abs())
    s_grad = EPS + (ns + (c - s2).abs()) + ((s1 - cc(3, 0)).abs() + (s2 - cc(4, 0)).abs())
    w_grad = EPS + (we + (c - w2).abs()) + ((w1 - cc(0, -3)).abs() + (w2 - cc(0, -4)).abs())
    e_grad = EPS + (we + (c - e2).abs()) + ((e1 - cc(0, 3)).abs() + (e2 - cc(0, 4)).abs())
    l2 = lpf + lpf
    n_est = n1 * l2 / (EPS + lpf + ls(-2, 0))
    s_est = s1 * l2 / (EPS + lpf + ls(2, 0))
    w_est = w1 * l2 / (EPS + lpf + ls(0, -2))
    e_est = e1 * l2 / (EPS + lpf + ls(0, 2))
    v_est = (s_grad * n_est + n_grad * s_est) / (n_grad + s_grad)
    h_est = (w_grad * e_est + e_grad * w_est) / (e_grad + w_grad)
    g = torch.where(rb & m4, vh_disc * (h_est - v_est) + v_est, c)
    del n1, s1, w1, e1, n2, s2, w2, e2, ns, we, n_grad, s_grad, w_grad, e_grad, l2, n_est, s_est, w_est, e_est
    del v_est, h_est, lpf, ls

    # ---- step 4.0 / 4.1: the diagonal colour-difference high-pass (odd columns, read at x | 1 as rcd.cc's
    # half-resolution buffer does) and the P / Q discrimination at the red and blue samples
    odd = (C % 2 == 1)
    m3 = (R >= 3) & (R < tr - 3) & (C >= 3) & (C < tc - 3) & odd
    P = (cc(-3, -3) - cc(-1, -1) - cc(1, 1) + cc(3, 3)) - 3.0 * (cc(-2, -2) + cc(2, 2)) + 6.0 * c
    Q = (cc(-3, 3) - cc(-1, 1) - cc(1, -1) + cc(3, -3)) - 3.0 * (cc(-2, 2) + cc(2, -2)) + 6.0 * c
    P = torch.where(m3, P * P, zero)
    Q = torch.where(m3, Q * Q, zero)
    P = P[..., 1::2].repeat_interleave(2, dim=-1)
    Q = Q[..., 1::2].repeat_interleave(2, dim=-1)
    ps, qs = _shifter(P), _shifter(Q)
    p_stat = torch.clamp((ps(-1, -1) + P) + ps(1, 1), min=EPSSQ)
    q_stat = torch.clamp((qs(-1, 1) + Q) + qs(1, -1), min=EPSSQ)
    PQ = torch.where(rb & m4, p_stat / (p_stat + q_stat), zero)
    del P, Q, ps, qs, p_stat, q_stat
    pqs = _shifter(PQ)
    pq_n = 0.25 * (((pqs(-1, -1) + pqs(-1, 1)) + pqs(1, -1)) + pqs(1, 1))
    pq_disc = torch.where((0.5 - PQ).abs() < (0.5 - pq_n).abs(), pq_n, PQ)
    del pqs, pq_n, PQ

    # ---- step 4.2: red at blue and blue at red (the diagonal neighbours are samples of the other colour)
    gs = _shifter(g)
    nw, ne, sw, se = cc(-1, -1), cc(-1, 1), cc(1, -1), cc(1, 1)
    d1 = (nw - se).abs()
    d2 = (ne - sw).abs()
    nw_grad = EPS + d1 + (nw - cc(-3, -3)).abs() + (g - gs(-2, -2)).abs()
    ne_grad = EPS + d2 + (ne - cc(-3, 3)).abs() + (g - gs(-2, 2)).abs()
    sw_grad = EPS + d2 + (sw - cc(3, -3)).abs() + (g - gs(2, -2)).abs()
    se_grad = EPS + d1 + (se - cc(3, 3)).abs() + (g - gs(2, 2)).abs()
    nw_est, ne_est = nw - gs(-1, -1), ne - gs(-1, 1)
    sw_est, se_est = sw - gs(1, -1), se - gs(1, 1)
    p_est = (nw_grad * se_est + se_grad * nw_est) / (nw_grad + se_grad)
    q_est = (ne_grad * sw_est + sw_grad * ne_est) / (ne_grad + sw_grad)
    opp = torch.where(rb & m4, g + (pq_disc * (q_est - p_est) + p_est), zero)
    del nw, ne, sw, se, d1, d2, nw_grad, ne_grad, sw_grad, se_grad, nw_est, ne_est, sw_est, se_est, p_est, q_est
    del pq_disc

    # ---- step 4.3: red and blue at the green samples
    is_r, is_b = (col == 0)[None], (col == 2)[None]
    n1g = EPS + (g - gs(-2, 0)).abs()
    s1g = EPS + (g - gs(2, 0)).abs()
    w1g = EPS + (g - gs(0, -2)).abs()
    e1g = EPS + (g - gs(0, 2)).abs()
    gm = green & m4
    out = []
    for X in (torch.where(is_r, c, torch.where(is_b, opp, zero)), torch.where(is_b, c, torch.where(is_r, opp, zero))):
        xs = _shifter(X)
        sn = (xs(-1, 0) - xs(1, 0)).abs()
        ew = (xs(0, -1) - xs(0, 1)).abs()
        n_grad = n1g + sn + (xs(-1, 0) - xs(-3, 0)).abs()
        s_grad = s1g + sn + (xs(1, 0) - xs(3, 0)).abs()
        w_grad = w1g + ew + (xs(0, -1) - xs(0, -3)).abs()
        e_grad = e1g + ew + (xs(0, 1) - xs(0, 3)).abs()
        n_est, s_est = xs(-1, 0) - gs(-1, 0), xs(1, 0) - gs(1, 0)
        w_est, e_est = xs(0, -1) - gs(0, -1), xs(0, 1) - gs(0, 1)
        v_est = (n_grad * s_est + s_grad * n_est) / (n_grad + s_grad)
        h_est = (e_grad * w_est + w_grad * e_est) / (e_grad + w_grad)
        out.append(torch.where(gm, g + (vh_disc * (h_est - v_est) + v_est), X))
        del xs, sn, ew, n_grad, s_grad, w_grad, e_grad, n_est, s_est, w_est, e_est, v_est, h_est
    return torch.stack([out[0], g, out[1]]).mul_(SCALE).clamp_(min=0.0)


def _border(raw: torch.Tensor, col: torch.Tensor, rgb: torch.Tensor, border: int) -> None:
    """librtprocess bayerborder_demosaic: the outer ``border`` pixels of ``rgb`` (3, H, W) from the 3 x 3 means of
    each colour of ``raw`` (in place)."""
    H, W = raw.shape
    edge = torch.ones((H, W), dtype=torch.bool, device=raw.device)
    edge[border:H - border, border:W - border] = False
    k = torch.ones(1, 1, 3, 3, device=raw.device, dtype=raw.dtype)
    for c in range(3):
        m = (col == c).to(raw.dtype)
        num = F.conv2d((raw * m)[None, None], k, padding=1)[0, 0]
        den = F.conv2d(m[None, None], k, padding=1)[0, 0]
        val = torch.where(col == c, raw, num / den.clamp(min=1e-12))
        rgb[c] = torch.where(edge, val, rgb[c])


def rcd(cfa: torch.Tensor, ch_of_k, *, siril_range: bool = True, tiles_per_batch: int = TILES_PER_BATCH) -> torch.Tensor:
    """(3, H, W): the CFA frame (H, W) demosaiced by RCD as Siril runs it (see the module's docstring). ``ch_of_k``:
    the pattern ("RGGB") or the colour of each 2 x 2 position. ``siril_range``: map the frame's [min, max] onto
    [0, 65535] first and back after, as Siril's float path does (False: the values are taken as 16-bit data, clipped
    to [0, 65536] by rcd.cc). No NaN allowed (``fill_bad`` first). Same device and float32."""
    x = cfa.float()
    H, W = x.shape
    if H < 2 * RCD_BORDER + 2 or W < 2 * RCD_BORDER + 2:
        raise ValueError(f"frame too small for RCD: {W} x {H}")
    if bool(torch.isnan(x).any()):
        raise ValueError("NaN samples: fill them first (debayer.fill_bad)")
    dev = x.device
    lo = hi = 0.0
    if siril_range:
        lo, hi = float(x.min()), float(x.max())
        if hi == lo:
            return x.expand(3, H, W).clone()
        raw = (x - lo) * (NORM / (hi - lo))
    else:
        raw = x
    col_full = colour_of(ch_of_k, H, W, dev)
    nth, ntw = -(-H // STEP), -(-W // STEP)
    Hp, Wp = nth * STEP + 2 * TILE_BORDER, ntw * STEP + 2 * TILE_BORDER
    cn = torch.zeros((Hp, Wp), device=dev, dtype=torch.float32)
    cn[:H, :W] = (raw / SCALE).clamp(0.0, 1.0)
    tiles = cn.unfold(0, TILE, STEP).unfold(1, TILE, STEP).reshape(nth * ntw, TILE, TILE)
    starts_r = torch.arange(nth, device=dev) * STEP
    starts_c = torch.arange(ntw, device=dev) * STEP
    rows = (H - starts_r).clamp(max=TILE)[:, None].expand(nth, ntw).reshape(-1)
    cols = (W - starts_c).clamp(max=TILE)[None, :].expand(nth, ntw).reshape(-1)
    col_tile = colour_of(ch_of_k, TILE, TILE, dev)          # the tiles start on even rows and columns
    inner = torch.zeros((3, nth * ntw, STEP, STEP), device=dev, dtype=torch.float32)
    for s in range(0, nth * ntw, tiles_per_batch):
        e = min(s + tiles_per_batch, nth * ntw)
        res = _rcd_tiles(tiles[s:e].contiguous(), col_tile, rows[s:e], cols[s:e])
        inner[:, s:e] = res[:, :, TILE_BORDER:TILE_BORDER + STEP, TILE_BORDER:TILE_BORDER + STEP]
        del res
    rgb = inner.view(3, nth, ntw, STEP, STEP).permute(0, 1, 3, 2, 4).reshape(3, nth * STEP, ntw * STEP)
    out = torch.zeros((3, H, W), device=dev, dtype=torch.float32)
    hh, ww = min(H - TILE_BORDER, nth * STEP), min(W - TILE_BORDER, ntw * STEP)
    out[:, TILE_BORDER:TILE_BORDER + hh, TILE_BORDER:TILE_BORDER + ww] = rgb[:, :hh, :ww]
    del rgb, inner, tiles, cn
    _border(raw, col_full, out, RCD_BORDER)
    if siril_range:
        out = out * ((hi - lo) / NORM) + lo
    return out


def debayer(cfa: torch.Tensor, ch_of_k, method: str = "rcd") -> torch.Tensor:
    """(3, H, W) float32 from a CFA frame (H, W) without NaN: ``method`` one of METHODS."""
    if method == "rcd":
        return rcd(cfa, ch_of_k)
    if method == "bilinear":
        return bilinear(cfa, ch_of_k)
    raise ValueError(f"unknown debayering {method!r}: one of {METHODS}")


def rcd_v23_vh_stat(c: Sequence[float]) -> float:
    """The vertical (or horizontal) statistic of RCD v2.3 as originally written (rcd_demosaicing.c, step 1.1: the
    long quadratic form of the 9 samples c[-4..4] along the direction), for the test that rcd.cc's short form is the
    same. ``c``: 9 values, c[4] the centre."""
    m4, m3, m2, m1, x, p1, p2, p3, p4 = [float(v) for v in c]
    return max(
        -18 * x * m1 - 18 * x * p1 - 36 * x * m2 - 36 * x * p2 + 18 * x * m3 + 18 * x * p3 - 2 * x * m4 - 2 * x * p4
        + 38 * x * x - 70 * m1 * p1 - 12 * m1 * m2 + 24 * m1 * p2 - 38 * m1 * m3 + 16 * m1 * p3 + 12 * m1 * m4
        - 6 * m1 * p4 + 46 * m1 * m1 + 24 * p1 * m2 - 12 * p1 * p2 + 16 * p1 * m3 - 38 * p1 * p3 - 6 * p1 * m4
        + 12 * p1 * p4 + 46 * p1 * p1 + 14 * m2 * p2 - 12 * m2 * p3 - 2 * m2 * m4 + 2 * m2 * p4 + 11 * m2 * m2
        - 12 * p2 * m3 + 2 * p2 * m4 - 2 * p2 * p4 + 11 * p2 * p2 + 2 * m3 * p3 - 6 * m3 * m4 + 10 * m3 * m3
        - 6 * p3 * p4 + 10 * p3 * p3 + m4 * m4 + p4 * p4, EPSSQ)


def rcd_short_vh_stat(c: Sequence[float]) -> float:
    """rcd.cc's short form of the same statistic: the sum of the squared high-pass at the centre and its two
    neighbours along the direction."""
    a = np.asarray(c, float)

    def hp(i):
        return (a[i - 3] - a[i - 1] - a[i + 1] + a[i + 3]) - 3 * (a[i - 2] + a[i + 2]) + 6 * a[i]
    return max(sum(hp(i) ** 2 for i in (3, 4, 5)), EPSSQ) if len(a) >= 9 else EPSSQ
