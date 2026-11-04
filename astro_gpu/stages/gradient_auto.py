"""The automatic (sample-free) sky model of Siril 1.5 (``subsky -auto``), written again for our engine.

Source: Siril's documentation (Background extraction, "Automatic (sample-free) method") and, for what it leaves out
(the rejection bounds, the stopping rule, the scale in pixels, the protection mask), the values of Siril's
``src/algos/background_extraction.c`` (development version of 2026-11-02, the ``ag_*`` functions). The method:

* each channel apart, on a copy reduced ``downsample`` times (block means; the last incomplete blocks are dropped);
* the background B is first fitted on every pixel, then, until the kept pixels settle (20 rounds at most, stop when
  fewer than 1 in 10 000 change): the residual r = image - B, its median and sigma (1.4826 x MAD); a pixel is kept
  when med - 4 sigma <= r <= med + 2 sigma (bright structures go first); with the structure protection, the pixels
  of a smooth mask grown around r - med > ``protect_threshold`` are left out too; never fewer than 2 % of the pixels;
  B fitted again on the kept pixels;
* the surface B: the kept pixels filled across the holes by repeated low-passes (10 rounds, the kept pixels restored
  each time; the holes start at the kept pixels' mean), then low-passed once more. The low-pass is 3 running box
  blurs matching a Gaussian; its radius is ``scale`` % of the reduced image's shorter side;
* a last low-pass of ``smoothness`` x that radius; the model brought back to full size bilinearly (corners aligned);
* the "simplified model": the same rounds with a polynomial of degree ``degree`` instead of the surface (pixel
  coordinates mapped to -1..1), removed first; the surface is then fitted on what it leaves; the background is the sum.

The image beyond its edges: the published source replicates the edge pixels, which makes the surface sag at the
borders; the rejection then drops the border pixels round after round and the model drifts (on a smooth synthetic sky,
the source's own functions compiled apart leave 2.5 % of the sky's level at the edges). The program Siril distributes
(1.5.0-489c8be, 2026-11-02) does not: it follows a linear ramp to the edge. Extending each box blur point-symmetrically
(the image continued along its slope) reproduces that program: on a synthetic sky, the backgrounds differ by 0.3 % of
their spread (RMS), against 27 % with replicated edges.

The rejection's median and sigma: the source takes them over the pixels kept so far; the distributed program behaves
as if they were taken over the whole image (on Heart with an injected corner glow, 2.1 % of the background's spread
RMS from it this way, 5.4 % the source's way; the source's way drops a strong gradient round after round). The whole
image is taken here.

Siril's thresholds are in units of an image scaled to 0..1; ``sky_model`` takes the image so scaled. Siril fits the
whole frame and asks for the uncovered borders to be cropped first; here a ``valid`` map can be given instead: the
pixels outside it are never kept, the surface is filled across them like across any rejected area. With every pixel
valid, the result is Siril's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

HIGH_K = 2.0                 # upper rejection bound, in sigmas above the median residual
LOW_K = 4.0                  # lower rejection bound, in sigmas below
ROUNDS = 20                  # rejection rounds at most
SETTLED = 1e-4               # the rounds stop when fewer than this share of the pixels change
MIN_KEEP = 0.02              # never fewer kept pixels than this share (16 at least)
BOX_PASSES = 3               # box blurs per low-pass
FILL_ROUNDS = 10              # low-pass rounds that fill the rejected areas


@dataclass
class AutoSettings:
    """Siril's settings and their defaults (``subsky -auto``)."""
    scale: float = 5.0               # 1..10: the low-pass radius, in % of the reduced image's shorter side
    smoothness: float = 1.0          # the last low-pass, in units of that radius (0: none)
    protect: bool = True             # mask the extended bright structures
    protect_threshold: float = 0.05  # brightness above the model at which a pixel is a structure (image in 0..1)
    protect_amount: float = 0.5      # how far the mask grows around the structures
    simplified: bool = False         # a stiff polynomial first
    degree: int = 1                  # its degree (1: a plane)
    downsample: int = 4              # 1, 2, 4 or 8
    protect_sigmas: float = 0.0      # not 0: the protection threshold in sigmas of the residual (not Siril's)


def _round(x: float) -> int:
    """C's lround: halves away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def box_sigma(r: int, passes: int) -> float:
    """The Gaussian sigma of ``passes`` box blurs of radius ``r``."""
    return math.sqrt(passes * r * (r + 1) / 3.0) if r >= 1 else 0.0


def lowpass(a: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian of ``sigma`` approximated by 3 box blurs of the matching radius, the image continued beyond each
    edge along its slope (point-symmetric extension, see the module); below a quarter pixel, or a radius under 1, the
    image itself."""
    r = _round((-1.0 + math.sqrt(1.0 + 12.0 * sigma * sigma / BOX_PASSES)) / 2.0)
    if sigma < 0.25 or r < 1:
        return a.copy()
    out = a
    for _ in range(BOX_PASSES):
        for axis in (1, 0):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (r, r)
            f = uniform_filter1d(np.pad(out, pad, mode="symmetric", reflect_type="odd"), 2 * r + 1, axis=axis,
                                 mode="nearest")
            out = f[r:-r] if axis == 0 else f[:, r:-r]
    return out


def inpaint(img: np.ndarray, keep: np.ndarray, radius: int) -> np.ndarray:
    """The smooth surface of ``img`` over the ``keep`` pixels, bridging the holes (harmonic-style filling)."""
    sigma = box_sigma(radius, 2)
    if not keep.any():
        return np.zeros_like(img)
    if keep.all():
        return lowpass(img, sigma)
    filled = np.where(keep, img, np.float32(img[keep].mean(dtype=np.float64)))
    for _ in range(FILL_ROUNDS):
        filled = np.where(keep, img, lowpass(filled, sigma))
    return lowpass(filled, sigma)


def structure_mask(res: np.ndarray, radius: int, threshold: float, amount: float) -> np.ndarray:
    """The extended bright structures: the pixels above ``threshold``, spread by a low-pass and cut at a level set
    by ``amount`` (a larger amount grows the mask further)."""
    det = (res > threshold).astype(np.float32)
    if not det.any():
        return np.zeros(res.shape, bool)
    grow = max(1, _round(radius * (0.5 + amount)))
    return lowpass(det, box_sigma(grow, 2)) > (1.0 - amount) * 0.5 + 1e-3


def _median(x: np.ndarray) -> float:
    """The median as Siril's quickselect gives it: the element of rank n // 2."""
    k = x.size // 2
    return float(np.partition(x, k)[k])


def _histogram_median(x: np.ndarray) -> float:
    """The median as Siril's MAD takes it (RawTherapee's percentile search, rt_algo.cc): a histogram of up to 65536
    bins between the minimum and the maximum, the bin where half of the values are reached, interpolated."""
    x = x.astype(np.float32, copy=False)
    lo, hi = float(x.min()), float(x.max())
    if hi == lo:
        return lo
    size = min(65536, x.size)
    scale = np.float32((size - 1) / (hi - lo))
    histo = np.bincount((scale * (x - np.float32(lo))).astype(np.uint16), minlength=size)
    thresh = np.float32(0.5) * np.float32(x.size)
    if thresh <= 0:
        return lo
    cs = np.cumsum(histo)
    i = int(np.searchsorted(cs, thresh, side="left"))
    count = float(cs[i])
    c0, c1 = count - float(thresh), float(thresh) - (count - float(histo[i]))
    k = i + 1
    v = (c1 * k + c0 * (k - 1)) / (c0 + c1)
    return min(max(v / float(scale) + lo, lo), hi)


def poly_fit(img: np.ndarray, keep: np.ndarray, degree: int) -> np.ndarray:
    """The polynomial of ``degree`` (x^i y^j, i + j <= degree, x and y mapped to -1..1) fitted on the ``keep``
    pixels, over the whole image."""
    h, w = img.shape
    xn = np.arange(w) / max(w - 1, 1) * 2.0 - 1.0
    yn = np.arange(h) / max(h - 1, 1) * 2.0 - 1.0
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    yy, xx = np.nonzero(keep)
    A = np.stack([xn[xx] ** i * yn[yy] ** j for i, j in terms], 1)
    AtA = A.T @ A + 1e-9 * np.eye(len(terms))
    coef = np.linalg.solve(AtA, A.T @ img[yy, xx].astype(np.float64))
    out = np.zeros((h, w))
    for c, (i, j) in zip(coef, terms):
        out += c * np.multiply.outer(yn ** j, xn ** i)
    return out.astype(np.float32)


def estimate(img: np.ndarray, valid: np.ndarray, radius: int, s: AutoSettings, use_poly: bool) -> np.ndarray:
    """The background of one reduced channel: the rounds of fit and rejection (see the module)."""
    n = img.size
    radius = max(radius, 1)

    def fit(keep):
        return poly_fit(img, keep, s.degree) if use_poly else inpaint(img, keep, radius)
    keep = valid.copy()
    model = fit(keep)
    prev = int(keep.sum())
    min_keep = max(16, int(MIN_KEEP * n))
    for it in range(ROUNDS):
        res = img - model
        ref = res[valid]                                    # the whole image: see the module
        med = _median(ref)
        sigma = 1.4826 * _histogram_median(np.abs(ref - np.float32(med))) + 1e-12
        new = (res <= med + HIGH_K * sigma) & (res >= med - LOW_K * sigma) & valid
        if s.protect:
            thr = s.protect_sigmas * sigma if s.protect_sigmas > 0 else s.protect_threshold
            new &= ~structure_mask(res - med, radius, thr, s.protect_amount)
        kept = int(new.sum())
        if kept < min_keep:                                 # never empty the fit set
            thr = np.sort(res[valid], axis=None)[min(min_keep, int(valid.sum()) - 1)]
            new = (res <= thr) & valid
            kept = int(new.sum())
        model = fit(new)
        change = abs(kept - prev) / n
        keep, prev = new, kept
        if it > 0 and change < SETTLED:
            break
    if not use_poly and s.smoothness > 0:
        model = lowpass(model, box_sigma(max(1, _round(radius * s.smoothness)), BOX_PASSES))
    return model


def reduce(a: np.ndarray, f: int) -> np.ndarray:
    """Block means of ``f`` x ``f`` (the last incomplete blocks dropped)."""
    h, w = a.shape[0] // f, a.shape[1] // f
    return a[:h * f, :w * f].reshape(h, f, w, f).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def enlarge(a: np.ndarray, H: int, W: int) -> np.ndarray:
    """Bilinear resize to (H, W), the corners aligned (the first and last rows and columns land on the image's)."""
    sh, sw = a.shape
    if (sh, sw) == (H, W):
        return a.copy()
    fy = np.arange(H) * (sh - 1) / max(H - 1, 1)
    fx = np.arange(W) * (sw - 1) / max(W - 1, 1)
    y0 = fy.astype(int)
    x0 = fx.astype(int)
    y1 = np.minimum(y0 + 1, sh - 1)
    x1 = np.minimum(x0 + 1, sw - 1)
    wy = (fy - y0)[:, None]
    wx = (fx - x0)[None, :]
    a = a.astype(np.float64)
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def sky_model(img: np.ndarray, s: AutoSettings, valid: np.ndarray | None = None) -> np.ndarray:
    """The background of one channel (H, W) of an image scaled to 0..1, on the reduced grid
    (H // downsample, W // downsample); ``enlarge`` brings it to full size."""
    f = s.downsample if 1 <= s.downsample <= min(img.shape) else 1
    small = reduce(img.astype(np.float32), f)
    if small.shape[0] < 2 or small.shape[1] < 2:
        raise ValueError(f"the image ({img.shape[1]} x {img.shape[0]}) is too small for the automatic model")
    ok = np.ones(small.shape, bool) if valid is None else reduce(valid.astype(np.float32), f) >= 0.99
    radius = max(1, _round(s.scale / 100.0 * min(small.shape)))
    if s.simplified:
        poly = estimate(small, ok, radius, s, True)
        return estimate(small - poly, ok, radius, s, False) + poly
    return estimate(small, ok, radius, s, False)
