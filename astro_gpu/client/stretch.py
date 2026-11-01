"""The display stretch of the image viewer (``imageview.py``): pure numpy, no Qt, no GPU.

The viewer never changes the data: it shows them through a look-up table. Each tile of the display pyramid is
quantized once to 16 bits, on a square-root scale (fine steps in the shadows, where the sky lives: 0.05 ADU at a sky of
175 ADU on a 16 000 ADU range), and every stretch is one table of 65 536 entries per channel:

* the pre-scaling of "Neutral sky" (``neutral_transform``): for an image still in the camera's colours, each channel is
  scaled so that its sky (the median) and its stars (the 99.5th percentile) line up with the green channel;
* black, mid and white points per channel (linked: the same three numbers for all channels); the mid point is the
  value shown as mid grey;
* the curve between black and white: ``linear``, ``mtf`` (the midtones transfer function of Siril and PixInsight),
  ``asinh``, or ``histogram`` (equalisation).

A slider move rebuilds three tables (about a millisecond) and re-indexes the tiles in view. The histogram of the data
is counted once, on a sample, when the image is opened (``Stats``); the median, the noise, the percentiles and the
clipped fractions come from it.

This replaces, for the viewer, ``services/inspect.stretch_params / neutral_params / render8 / histogram8 /
stretch_image`` (one fixed 8-bit rendering made by the worker); the automatic numbers are the same (black 2.8 sigma
under the sky, white at the 99.99th percentile, sky at a quarter of the range).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Q = 65535                                  # the top of the 16-bit quantization
MODES = ("linear", "mtf", "asinh", "histogram")
TARGET = 0.25                              # where the automatic stretch puts the sky (0 black, 1 white)
SHADOWS = 2.8                              # the automatic black point, in noise sigmas under the sky


# ---------------------------------------------------------------------------------------------- quantization
def quantize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Float data -> uint16 on a square-root scale between ``lo`` and ``hi`` (NaN and values under ``lo`` -> 0)."""
    scalar = np.ndim(x) == 0
    y = np.subtract(np.atleast_1d(x), np.float32(lo), dtype=np.float32)
    y *= np.float32(1.0 / (hi - lo))
    np.fmax(y, np.float32(0.0), out=y)                      # fmax drops NaN
    np.fmin(y, np.float32(1.0), out=y)
    np.sqrt(y, out=y)
    y *= np.float32(Q)
    y += np.float32(0.5)
    q = y.astype(np.uint16)
    return q[0] if scalar else q


def level_values(lo: float, hi: float) -> np.ndarray:
    """The data value each of the 65 536 quantized levels stands for (float64)."""
    u = np.arange(Q + 1, dtype=np.float64) / Q
    return lo + (hi - lo) * u * u


# ---------------------------------------------------------------------------------------------- statistics
def _weighted_pct(vals: np.ndarray, counts: np.ndarray, p: float) -> float:
    cs = np.cumsum(counts)
    if cs[-1] <= 0:
        return float(vals[0])
    i = int(np.searchsorted(cs, p * cs[-1], side="left"))
    return float(vals[min(i, len(vals) - 1)])


@dataclass
class Stats:
    """What the viewer knows of an image's values, per channel: counted once on a sample when it is opened."""

    lo: float                       # the quantization range (data units)
    hi: float
    hist: np.ndarray                # (C, 65536) counts of the quantized sample
    median: np.ndarray              # (C,)
    sigma: np.ndarray               # (C,) noise: 1.4826 x the median absolute deviation
    low: np.ndarray                 # (C,) 0.01th percentile
    high: np.ndarray                # (C,) 99.99th percentile
    ref: np.ndarray                 # (C,) 99.5th percentile (the stars, for the neutral sky)
    vmin: np.ndarray                # (C,) the data's own minimum and maximum
    vmax: np.ndarray
    _vals: np.ndarray | None = field(default=None, repr=False)

    @property
    def channels(self) -> int:
        return int(self.hist.shape[0])

    @property
    def values(self) -> np.ndarray:
        if self._vals is None:
            self._vals = level_values(self.lo, self.hi)
        return self._vals


_POOLS: dict = {}


def pool(name: str = "prepare"):
    """The threads the viewer computes on (numpy and OpenCV release the GIL), one per core, at most 8: "prepare" for
    the images being opened (worker threads), "render" for the tiles of the view (the GUI thread waits on them), so
    that an image opening in the background never queues before a repaint."""
    p = _POOLS.get(name)
    if p is None:
        import os
        from concurrent.futures import ThreadPoolExecutor
        p = _POOLS[name] = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4))),
                                              thread_name_prefix=f"imageview-{name}")
    return p


def _minmax(a: np.ndarray) -> tuple[float, float]:
    """The finite minimum and maximum of a 2-D float array (OpenCV skips NaN)."""
    import cv2
    a = np.ascontiguousarray(a, np.float32)
    lo, hi = cv2.minMaxLoc(a)[:2]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        f = a[np.isfinite(a)]
        lo, hi = (float(f.min()), float(f.max())) if f.size else (0.0, 1.0)
    return float(lo), float(hi)


def _hist(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    import cv2
    q = np.ascontiguousarray(quantize(x, lo, hi))
    return cv2.calcHist([q], [0], None, [Q + 1], [0, Q + 1]).ravel().astype(np.int64)


def _channel_numbers(vals: np.ndarray, hk: np.ndarray) -> tuple[float, ...]:
    med = _weighted_pct(vals, hk, 0.5)
    dev = np.abs(vals - med)
    order = np.argsort(dev, kind="stable")
    sig = 1.4826 * _weighted_pct(dev[order], hk[order], 0.5)
    return (med, sig, _weighted_pct(vals, hk, 1e-4), _weighted_pct(vals, hk, 0.9999),
            _weighted_pct(vals, hk, 0.995))


def compute_stats(level0: np.ndarray, target_samples: float = 1.5e6) -> Stats:
    """The statistics of a (C, H, W) float image: its true minimum and maximum, then the histogram of a regular sample
    (about ``target_samples`` values per channel), from which the median, the noise and the percentiles are read. The
    channels are counted in parallel."""
    c, h, w = level0.shape
    step = max(1, int(np.sqrt(h * w / target_samples)))
    ex = pool()
    mm = list(ex.map(lambda k: _minmax(level0[k]), range(c)))
    vmin = np.array([m[0] for m in mm])
    vmax = np.array([m[1] for m in mm])
    lo, hi = float(vmin.min()), float(vmax.max())
    if not hi > lo:
        hi = lo + max(abs(lo) * 1e-6, 1e-6)
    sample = level0[:, ::step, ::step]

    def count(lo_, hi_):
        hist_ = np.stack(list(ex.map(lambda k: _hist(sample[k], lo_, hi_), range(c))))
        vals_ = level_values(lo_, hi_)
        return hist_, vals_, np.array(list(ex.map(lambda k: _channel_numbers(vals_, hist_[k]), range(c))))

    hist, vals, num = count(lo, hi)
    # outliers far under the sky would waste the fine steps of the quantization: count again from 64 sigmas under it
    robust = float((num[:, 0] - 64 * np.maximum(num[:, 1], (hi - lo) * 1e-7)).min())
    if robust > lo + (hi - lo) * 1e-3:
        lo = robust
        hist, vals, num = count(lo, hi)
    med, sig, low, high, ref = (num[:, i].copy() for i in range(5))
    sig = np.maximum(sig, (hi - lo) * 1e-7)
    return Stats(lo, hi, hist, med, sig, low, high, ref, vmin, vmax, vals)


# ---------------------------------------------------------------------------------------------- the stretch
def mtf(m, x):
    """The midtones transfer function: 0 -> 0, m -> 1/2, 1 -> 1 (and mtf(mtf(t, x), x) = t)."""
    return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)


def asinh_beta(t: float, y: float) -> float:
    """The strength b of the curve asinh(b x) / asinh(b) that sends ``t`` to ``y`` (0 < t < y < 1); ~0 (linear) when
    t >= y."""
    t = float(np.clip(t, 1e-9, 1 - 1e-9))
    if t >= y:
        return 1e-6
    lo, hi = np.log(1e-6), np.log(1e12)
    for _i in range(80):                               # f(b) = asinh(b t) / asinh(b) grows from t to 1 with b
        mid = 0.5 * (lo + hi)
        b = np.exp(mid)
        if np.arcsinh(b * t) / np.arcsinh(b) < y:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def asinh_curve(t, beta: float):
    if beta < 1e-5:
        return t
    return np.arcsinh(beta * t) / np.arcsinh(beta)


@dataclass
class Stretch:
    """How an image is shown. ``black``, ``mid`` and ``white`` are per channel, in the units of the data after the
    pre-scaling ``y = x * gain + shift`` (the identity unless ``neutral``). ``mid`` is the value shown as mid grey
    (``linear`` and ``histogram`` do not use it). ``auto``: the numbers follow the automatic stretch."""

    mode: str
    black: np.ndarray
    mid: np.ndarray
    white: np.ndarray
    gain: np.ndarray
    shift: np.ndarray
    linked: bool = True
    neutral: bool = False
    auto: bool = True

    @property
    def channels(self) -> int:
        return int(len(self.black))

    def copy(self) -> Stretch:
        return Stretch(self.mode, self.black.copy(), self.mid.copy(), self.white.copy(), self.gain.copy(),
                       self.shift.copy(), self.linked, self.neutral, self.auto)

    def same(self, o: Stretch | None) -> bool:
        return (o is not None and o.mode == self.mode and o.linked == self.linked and o.neutral == self.neutral
                and o.auto == self.auto and o.channels == self.channels
                and all(np.array_equal(a, b) for a, b in ((self.black, o.black), (self.mid, o.mid),
                                                            (self.white, o.white), (self.gain, o.gain),
                                                            (self.shift, o.shift))))

    def adapted(self, channels: int) -> Stretch:
        """The same stretch for an image with another number of channels (a mono frame after a colour one)."""
        if channels == self.channels:
            return self.copy()

        def fit(a, how):
            v = float(how(a))
            return np.full(channels, v)
        return Stretch(self.mode, fit(self.black, np.mean), fit(self.mid, np.mean), fit(self.white, np.max),
                       np.ones(channels), np.zeros(channels), True, False, self.auto)


def neutral_transform(stats: Stats) -> tuple[np.ndarray, np.ndarray]:
    """(gain, shift) per channel making the sky (median) and the stars (99.5th percentile) of each channel those of
    the green one: the sky of a camera-colour image comes out grey and its stars white. Identity for one channel."""
    c = stats.channels
    if c != 3:
        return np.ones(c), np.zeros(c)
    med = stats.median
    span = np.maximum(stats.ref - med, stats.sigma)
    gain = np.clip(span[1] / span, 0.1, 10.0)
    shift = med[1] - med * gain
    return gain, shift


def looks_camera(stats: Stats, ratio: float = 1.15, sigmas: float = 5.0) -> bool:
    """Whether a colour image still looks in the camera's colours: the skies of its channels apart by more than
    ``ratio`` and by more than ``sigmas`` noise sigmas (a calibrated image has a grey sky)."""
    if stats.channels != 3:
        return False
    med = stats.median - min(stats.lo, 0.0)
    lo, hi = float(med.min()), float(med.max())
    return lo > 0 and hi / lo > ratio and (hi - lo) > sigmas * float(stats.sigma.max())


def transformed(stats: Stats, gain: np.ndarray, shift: np.ndarray) -> dict[str, np.ndarray]:
    """The channel statistics after the pre-scaling."""
    return {"median": stats.median * gain + shift, "sigma": stats.sigma * gain, "high": stats.high * gain + shift,
            "lo": stats.lo * gain + shift, "hi": stats.hi * gain + shift, "ref": stats.ref * gain + shift}


def auto_stretch(stats: Stats, mode: str = "mtf", linked: bool = True, neutral: bool = False,
                 target: float = TARGET, shadows: float = SHADOWS) -> Stretch:
    """The automatic stretch: black ``shadows`` sigmas under the sky, white at the 99.99th percentile, and the sky shown
    at ``target`` (``mtf`` and ``asinh``; ``linear`` and ``histogram`` have no mid point)."""
    c = stats.channels
    gain, shift = neutral_transform(stats) if neutral and c == 3 else (np.ones(c), np.zeros(c))
    t = transformed(stats, gain, shift)
    med, sig = t["median"], t["sigma"]
    eps = max((stats.hi - stats.lo) * 1e-6, 1e-12)
    b = np.maximum(med - shadows * sig, t["lo"])
    w = np.maximum(t["high"], b + eps)
    if linked and c > 1:
        b = np.full(c, b.mean())
        w = np.full(c, w.max())
        med = np.full(c, med.mean())
    n = np.clip((med - b) / (w - b), 1e-6, 0.99)
    if mode == "mtf":
        mid = b + mtf(target, n) * (w - b)
    elif mode == "asinh":
        mid = np.empty(c)
        for k in range(c):
            beta = asinh_beta(float(n[k]), target)
            th = np.sinh(0.5 * np.arcsinh(beta)) / beta if beta > 1e-5 else 0.5
            mid[k] = b[k] + th * (w[k] - b[k])
    else:
        mid = 0.5 * (b + w)
    return Stretch(mode if mode in MODES else "mtf", b, mid, w, gain, shift, linked and c > 1, neutral and c == 3,
                   True)


def manual(st: Stretch, which: str, value: float, channel: int = -1, stats: Stats | None = None) -> Stretch:
    """A copy of ``st`` with one point moved by hand (``which``: black, mid or white) on one channel, or on all of them
    (``channel`` -1: the same value when linked, the same shift of each channel's point when not). The points keep
    their order; the stretch is no longer automatic."""
    out = st.copy()
    out.auto = False
    c = out.channels
    chans = range(c) if channel < 0 else [channel]
    arr = getattr(out, which)
    delta = value - (float(arr[channel]) if channel >= 0 else float(np.mean(arr)))
    for k in chans:
        arr[k] = value if (out.linked or channel >= 0 or c == 1) else arr[k] + delta
    span = (stats.hi - stats.lo) if stats is not None else max(float(np.max(out.white - out.black)), 1e-9)
    gap = max(span * 1e-6, 1e-12)
    for k in range(c):
        b, m, w = out.black[k], out.mid[k], out.white[k]
        if which == "black":
            b = min(b, w - 2 * gap)
            m = min(max(m, b + gap), w - gap)
        elif which == "white":
            w = max(w, b + 2 * gap)
            m = min(max(m, b + gap), w - gap)
        else:
            m = min(max(m, b + gap), w - gap)
        out.black[k], out.mid[k], out.white[k] = b, m, w
    return out


def _curve(t: np.ndarray, st: Stretch, k: int) -> np.ndarray:
    b, m, w = float(st.black[k]), float(st.mid[k]), float(st.white[k])
    if st.mode == "mtf":
        mm = float(np.clip((m - b) / (w - b), 1e-6, 1 - 1e-6))
        return mtf(mm, t)
    if st.mode == "asinh":
        mm = float(np.clip((m - b) / (w - b), 1e-9, 1 - 1e-9))
        return asinh_curve(t, asinh_beta(mm, 0.5))
    return t


def curves(st: Stretch, stats: Stats) -> np.ndarray:
    """(C, 65536) float32: the shown value (0..1) of each quantized level, per channel."""
    vals = stats.values
    c = st.channels
    out = np.empty((c, Q + 1), np.float32)
    ys = [vals * st.gain[k] + st.shift[k] for k in range(c)]
    if st.mode == "histogram":
        hist = stats.hist.astype(np.float64)
        masks = [(ys[k] >= st.black[k]) & (ys[k] <= st.white[k]) for k in range(c)]
        if st.linked and c > 1:                            # one equalisation of the pooled values
            ally = np.concatenate(ys)
            allw = np.concatenate([hist[k] * masks[k] for k in range(c)])
            order = np.argsort(ally, kind="stable")
            cdf = np.cumsum(allw[order])
            cdf /= max(cdf[-1], 1.0)
            for k in range(c):
                out[k] = np.interp(ys[k], ally[order], cdf)
        else:
            for k in range(c):
                cdf = np.cumsum(hist[k] * masks[k])
                out[k] = cdf / max(cdf[-1], 1.0)
        for k in range(c):
            out[k][ys[k] < st.black[k]] = 0.0
            out[k][ys[k] > st.white[k]] = 1.0
        return out
    for k in range(c):
        t = np.clip((ys[k] - st.black[k]) / max(st.white[k] - st.black[k], 1e-30), 0.0, 1.0)
        out[k] = np.clip(_curve(t, st, k), 0.0, 1.0)
    return out


def luts(st: Stretch, stats: Stats) -> np.ndarray:
    """(C, 65536) uint8: the tables the tiles are shown through."""
    return (curves(st, stats) * 255.0 + 0.5).astype(np.uint8)


def render(q: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """A quantized tile (C, h, w) uint16 through the tables -> (h, w, 3) uint8, or (h, w) uint8 for one channel."""
    if q.shape[0] == 1:
        return lut[0][q[0]]
    out = np.empty(q.shape[1:] + (3,), np.uint8)
    for k in range(3):
        out[..., k] = lut[k][q[k]]
    return out


def clipped(st: Stretch, stats: Stats) -> tuple[float, float]:
    """The fractions of the pixels shown pure black and pure white (the mean of the channels)."""
    vals = stats.values
    lo_f, hi_f = [], []
    for k in range(st.channels):
        y = vals * st.gain[k] + st.shift[k]
        h = stats.hist[k]
        tot = max(int(h.sum()), 1)
        lo_f.append(float(h[y < st.black[k]].sum()) / tot)
        hi_f.append(float(h[y > st.white[k]].sum()) / tot)
    return float(np.mean(lo_f)), float(np.mean(hi_f))


def apply(data: np.ndarray, st: Stretch, stats: Stats, rows: int = 256) -> np.ndarray:
    """The stretched image at full precision, (C, H, W) float32 in 0..1 (the export of "the stretched view"): the
    curves are evaluated on the float data (``histogram`` goes through its 16-bit table)."""
    c, h, w = data.shape
    out = np.empty((c, h, w), np.float32)
    table = curves(st, stats) if st.mode == "histogram" else None
    for r0 in range(0, h, rows):
        blk = data[:, r0:r0 + rows]
        for k in range(c):
            if table is not None:
                out[k, r0:r0 + rows] = table[k][quantize(blk[k], stats.lo, stats.hi)]
                continue
            y = blk[k].astype(np.float64) * st.gain[k] + st.shift[k]
            t = np.clip((y - st.black[k]) / max(st.white[k] - st.black[k], 1e-30), 0.0, 1.0)
            t = np.nan_to_num(t, nan=0.0)
            out[k, r0:r0 + rows] = np.clip(_curve(t, st, k), 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------------------------- the pyramid
def reduce2(a: np.ndarray) -> np.ndarray:
    """(C, h, w) float32 -> (C, h // 2, w // 2), each pixel the mean of 2 x 2 (an odd last row or column is dropped)."""
    import cv2
    c, h, w = a.shape
    size = (max(1, w // 2), max(1, h // 2))
    out = np.empty((c, size[1], size[0]), np.float32)

    def one(k):
        src = a[k, :size[1] * 2, :size[0] * 2]
        if not src.flags["C_CONTIGUOUS"] or src.dtype != np.float32:
            src = np.ascontiguousarray(src, np.float32)
        cv2.resize(src, size, dst=out[k], interpolation=cv2.INTER_AREA)
    list(pool().map(one, range(c)))
    return out


def pyramid(level0: np.ndarray, smallest: int = 320) -> list[np.ndarray]:
    """Level 0 (the given array, not copied), then halves until the long side is at most ``smallest``."""
    levels = [level0]
    while max(levels[-1].shape[1:]) > smallest and min(levels[-1].shape[1:]) >= 4:
        levels.append(reduce2(levels[-1]))
    return levels


CHANNEL = {"R": 0, "G": 1, "B": 2}


def superpixel(mosaic: np.ndarray, cfa: str) -> np.ndarray:
    """A Bayer mosaic (H, W) -> (3, H // 2, W // 2) float32: each 2 x 2 cell one pixel, its two greens averaged."""
    a = np.asarray(mosaic)
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    out = np.zeros((3, h // 2, w // 2), np.float32)
    n = np.zeros(3, np.float32)
    for i, ch in enumerate(cfa.upper()[:4]):
        k = CHANNEL.get(ch, 1)
        out[k] += a[i // 2:h:2, i % 2:w:2]
        n[k] += 1
    out /= np.maximum(n, 1)[:, None, None]
    return out


def cfa_letter(cfa: str, x: int, y: int) -> str:
    """The colour of the photosite (x, y) of a mosaic: "R", "G" or "B"."""
    if len(cfa) < 4:
        return ""
    return cfa.upper()[(y % 2) * 2 + (x % 2)]


__all__ = ["Q", "MODES", "Stats", "Stretch", "quantize", "level_values", "compute_stats", "mtf", "asinh_beta",
           "neutral_transform", "looks_camera", "auto_stretch", "manual", "curves", "luts", "render", "clipped", "apply",
           "reduce2", "pyramid", "superpixel", "cfa_letter"]
