"""Multiscale detail and local contrast of Post.

* **The starlet transform** (the isotropic undecimated wavelet transform, B3 spline a trous: J.-L. Starck, F. Murtagh
  and J. Fadili, "Sparse Image and Signal Processing", 2010, ch. 3; J.-L. Starck and F. Murtagh, "Astronomical Image and
  Data Analysis", 2006): c(j+1) = h(j) * c(j) with h = [1, 4, 6, 4, 1] / 16 dilated by 2^j, w(j+1) = c(j) - c(j+1); the
  image is the sum of its layers and the residual. Layer j holds the structures of about 2^(j-1) pixels. Siril's
  Wavelets tool and PixInsight's MultiscaleLinearTransform use it.
* **Multiscale sharpening** (PixInsight MultiscaleLinearTransform "bias", Siril Wavelets): each layer multiplied by
  (1 + bias). On the CIE L* lightness (PixInsight's default target: the colours kept) or on each RGB channel.
* **Multiscale noise reduction** (PixInsight MultiscaleLinearTransform noise reduction; Starck and Murtagh 2006, 2.3:
  soft thresholding): each layer's coefficients shrunk towards zero by k sigma(j), sigma(j) = sigma(1) e(j) / e(1) with
  sigma(1) the MAD of the first layer / 0.6745 and e(j) the noise of each starlet layer for a unit Gaussian noise
  (0.889, 0.200, 0.086, 0.041, 0.020), then mixed with the layer by the amount. On the lightness, the chrominance
  (CIE a* b*) or each RGB channel.
* **CLAHE** (K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", Graphics Gems IV, 1994; Siril's CLAHE,
  PixInsight's LocalHistogramEqualization): on the CIE L* lightness, histograms of 256 levels per tile (PixInsight's
  default resolution), clipped at the contrast limit times the mean and redistributed as OpenCV's ``createCLAHE`` does,
  the tables interpolated bilinearly between tile centres and linearly between levels (no posterisation of smooth
  gradients), then mixed with the original lightness by the amount (PixInsight's Amount).

* **Highlight compression** (a base and its detail by the guided filter: K. He, J. Sun, X. Tang, "Guided Image
  Filtering", IEEE TPAMI 35(6), 2013, section 4.1 "detail enhancement and HDR compression"): the lightness split into a
  smooth base (the mean of two guided filters, at a third of the radius and at the radius) and the detail above it; the
  base compressed above a knee by a logarithm that brings 1 down to the ceiling, the detail kept (scaled by the local
  slope, never below a half, then raised by Detail); the three channels scaled by the lightness's gain (the colours
  kept). The bright cores of a nebula come down while their structure stays.
* **Faint area smoothing**: the faint areas (lightness under a threshold, with a soft transition) replaced by a
  Gaussian blur of themselves, the bright ones untouched, so that the faint extensions can be stretched up without
  their grain.

Sizes are full-resolution pixels: at a reduced resolution (a draft, the screen) the layers finer than a pixel are
left out and the tiles are smaller, so that the reduced image looks like the full one reduced.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import colour

LAYERS = 5                                             # 1, 2, 4, 8 and 16 px
NOISE_E = (0.889, 0.200, 0.086, 0.041, 0.020, 0.010)   # starlet layer noise for a unit Gaussian noise (B3 spline)
TARGETS = ["Lightness (CIE L*)", "RGB"]
NR_TARGETS = ["Lightness (CIE L*)", "Chrominance (CIE a* b*)", "RGB"]
_H = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32) / 16


def starlet(img: np.ndarray, n: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    """The n first starlet layers and the residual of an image (h, w) or (h, w, c), as tensors (c, h, w)."""
    a = torch.from_numpy(colour.f32(img))
    if a.ndim == 2:
        a = a[..., None]
    c = a.permute(2, 0, 1).contiguous()[None]           # (1, ch, h, w)
    ch = c.shape[1]
    layers = []
    for j in range(n):
        s = 2 ** j
        ky = _H.view(1, 1, 5, 1).repeat(ch, 1, 1, 1)
        kx = _H.view(1, 1, 1, 5).repeat(ch, 1, 1, 1)
        h, w = c.shape[-2:]
        py, px = min(2 * s, h - 1), min(2 * s, w - 1)
        if py < 2 * s or px < 2 * s:                   # the image is smaller than the kernel: no coarser layer
            layers.append(torch.zeros_like(c[0]))
            continue
        t = F.pad(c, (0, 0, py, py), mode="reflect")
        t = F.conv2d(t, ky, groups=ch, dilation=(s, 1))
        t = F.pad(t, (px, px, 0, 0), mode="reflect")
        t = F.conv2d(t, kx, groups=ch, dilation=(1, s))
        layers.append((c - t)[0])
        c = t
    return layers, c[0]


def _first_layer(level: int) -> int:
    """The first full-resolution layer (0 = 1 px) that a resolution 2^level still shows."""
    return min(level, LAYERS)


def _on(x: np.ndarray, target: str, fn) -> np.ndarray:
    """``fn`` (an image (h, w, c) -> the same) on the lightness, the chrominance or the RGB channels."""
    if target == "RGB":
        return fn(colour.f32(x))
    lab = colour.to_lab(x)
    if target.startswith("Lightness"):
        lab[..., 0] = fn(np.ascontiguousarray(lab[..., :1] / 100))[..., 0] * 100
    else:
        lab[..., 1:] = fn(np.ascontiguousarray(lab[..., 1:] / 128)) * 128
    return colour.from_lab(lab)


def _back(layers, residual) -> np.ndarray:
    out = residual.clone()
    for w in layers:
        out += w
    return out.permute(1, 2, 0).contiguous().numpy()


def sharpen(x: np.ndarray, biases: list[float], target: str, level: int = 0) -> np.ndarray:
    """Multiscale sharpening: layer j (2^j px at full resolution) times (1 + bias j)."""
    if not any(abs(b) > 1e-9 for b in biases):
        return x
    first = _first_layer(level)
    gains = [1 + b for b in biases[first:]]
    if not any(abs(g - 1) > 1e-9 for g in gains):
        return x

    def fn(img):
        layers, res = starlet(img, len(gains))
        return _back([w * g for w, g in zip(layers, gains)], res)
    return _on(x, target, fn)


def soft(w: torch.Tensor, t) -> torch.Tensor:
    """Soft thresholding: sign(w) max(|w| - t, 0)."""
    return torch.sign(w) * (w.abs() - t).clamp_(min=0)


def noise_sigma(w1: torch.Tensor) -> torch.Tensor:
    """sigma of the first starlet layer of each channel (MAD / 0.6745), divided by its unit-noise e(1)."""
    flat = w1.reshape(w1.shape[0], -1)
    if flat.shape[1] > 400_000:                         # a regular subsample is enough for a median
        flat = flat[:, :: flat.shape[1] // 400_000]
    mad = (flat - flat.median(1, keepdim=True).values).abs().median(1).values
    return mad / 0.6745 / NOISE_E[0]


def denoise(x: np.ndarray, thresholds: list[float], amount: float, target: str, level: int = 0) -> np.ndarray:
    """Multiscale noise reduction: layer j's coefficients soft-thresholded at k_j sigma_j, mixed by the amount."""
    if amount <= 0 or not any(t > 0 for t in thresholds):
        return x
    first = _first_layer(level)
    ks = list(thresholds[first:])
    if not any(t > 0 for t in ks):
        return x

    def fn(img):
        layers, res = starlet(img, len(ks))
        sig = noise_sigma(layers[0])                     # the noise of this resolution's finest layer
        out = []
        for j, (w, k) in enumerate(zip(layers, ks)):
            if k <= 0:
                out.append(w)
                continue
            t = (k * sig * NOISE_E[j]).view(-1, 1, 1)
            out.append(w + amount * (soft(w, t) - w))
        return _back(out, res)
    return _on(x, target, fn)


# ---------------------------------------------------------------------------------------------- CLAHE
def _grid(W: int, H: int, tile: float) -> tuple[int, int, int, int]:
    """OpenCV's tile grid of a W x H image for tiles of about ``tile`` px: (gx, gy, tile width, tile height); an
    image that is not a multiple of the grid is padded on both sides by g - n % g, as createCLAHE does."""
    gx, gy = max(1, int(round(W / tile))), max(1, int(round(H / tile)))
    if W % gx or H % gy:
        W, H = W + gx - W % gx, H + gy - H % gy
    return gx, gy, W // gx, H // gy


def _clip_lut(hist: np.ndarray, clip: float, area: int) -> np.ndarray:
    """The tables of the tiles' histograms (tiles, 256): clipped at clip x the mean and the excess spread over every
    level as OpenCV does (the whole part on every level, the rest one by one, evenly spaced), then the cumulated sum."""
    limit = max(1, int(clip * area / 256)) if clip > 0 else 0
    if limit:
        excess = np.clip(hist - limit, 0, None).sum(1, keepdims=True)
        hist = np.minimum(hist, limit)
        hist += np.floor(excess / 256)
        rest = (excess[:, 0] - np.floor(excess[:, 0] / 256) * 256).astype(int)
        for k, r in enumerate(rest):
            if r > 0:
                step = max(256 // r, 1)
                hist[k, ::step][:r] += 1
    return np.clip(np.cumsum(hist, 1) * (255.0 / area), 0, 255).astype(np.float32)


def clahe_lightness(L: np.ndarray, clip: float, tile: float, origin=(0, 0), canvas=None) -> np.ndarray:
    """CLAHE of a lightness in [0, 1] (h, w), tiles of about ``tile`` pixels: Zuiderveld's algorithm as OpenCV's
    createCLAHE computes it (256-level histograms, clipped), evaluated at the continuous value (the tables
    interpolated between levels). ``origin`` and ``canvas`` (W, H): L is the box of a larger image starting there (a
    region at full resolution); the grid is then the whole image's, and the tiles the box cuts are counted from the
    pixels it has, scaled to a whole tile (so that the region looks like the same place of the whole image)."""
    h, w = L.shape
    q = np.clip(L, 0, 1).astype(np.float32) * 255
    ox, oy = origin
    W, H = canvas or (w, h)
    gx, gy, tw, th = _grid(W, H, tile)
    whole = (ox, oy, W, H) == (0, 0, w, h)
    if whole:
        pad = q if (pw := tw * gx - w) == 0 and th * gy == h else \
            np.pad(q, ((0, th * gy - h), (0, pw)), mode="reflect")
        ib = np.clip((pad + 0.5).astype(np.int32), 0, 255)
        tiles = ib.reshape(gy, th, gx, tw).transpose(0, 2, 1, 3).reshape(gy * gx, th * tw)
        hist = np.stack([np.bincount(t, minlength=256) for t in tiles]).astype(np.float64)
    else:
        ib = np.clip((q + 0.5).astype(np.int32), 0, 255)
        ty = np.clip((np.arange(h) + oy) // th, 0, gy - 1)
        tx = np.clip((np.arange(w) + ox) // tw, 0, gx - 1)
        idx = (ty[:, None] * gx + tx[None, :]) * 256 + ib
        hist = np.bincount(idx.ravel(), minlength=gx * gy * 256).reshape(gx * gy, 256).astype(np.float64)
        n = hist.sum(1, keepdims=True)
        hist = np.where(n > 0, hist * (th * tw) / np.maximum(n, 1), 0)
        empty = n[:, 0] == 0
        hist[empty] = th * tw / 256                                # a tile the box does not reach: never read
    lut = _clip_lut(hist, clip, th * tw)
    t = torch.from_numpy(q)
    lo = t.floor().clamp_(0, 254)
    fr = t - lo
    lo = lo.long()
    yy = (torch.arange(h, dtype=torch.float32) + oy) / th - 0.5
    xx = (torch.arange(w, dtype=torch.float32) + ox) / tw - 0.5
    y1 = yy.floor().clamp(0, gy - 1).long()
    x1 = xx.floor().clamp(0, gx - 1).long()
    y2 = (yy.floor() + 1).clamp(0, gy - 1).long()
    x2 = (xx.floor() + 1).clamp(0, gx - 1).long()
    wy = (yy - yy.floor()).clamp(0, 1)[:, None]
    wx = (xx - xx.floor()).clamp(0, 1)[None, :]
    lt = torch.from_numpy(lut).reshape(-1)

    def at(ty, tx):
        base = (ty[:, None] * gx + tx[None, :]) * 256 + lo
        return lt[base] + fr * (lt[base + 1] - lt[base])

    top = at(y1, x1) * (1 - wx) + at(y1, x2) * wx
    bot = at(y2, x1) * (1 - wx) + at(y2, x2) * wx
    return ((top * (1 - wy) + bot * wy) / 255).numpy()


def clahe(x: np.ndarray, amount: float, clip: float, radius: float, level: int = 0, origin=(0, 0),
          canvas=None) -> np.ndarray:
    """CLAHE on the CIE L* lightness, mixed with it by the amount. ``origin``, ``canvas``: x is a region of a larger
    image (clahe_lightness)."""
    if amount <= 0:
        return x
    lab = colour.to_lab(x)
    L = lab[..., 0] / 100
    tile = max(4.0, 2 * radius / 2 ** level)
    L2 = clahe_lightness(L, clip, tile, origin, canvas)
    lab[..., 0] = (L + amount * (L2 - L)) * 100
    return colour.from_lab(lab)


def guided(I: np.ndarray, p: np.ndarray, r: float, eps: float) -> np.ndarray:
    """The guided filter (He et al. 2013, algorithm 1) of p guided by I, box radius r (2-D float32 arrays)."""
    import cv2
    k = (2 * max(1, int(round(r))) + 1,) * 2

    def box(a):
        return cv2.boxFilter(a, -1, k, borderType=cv2.BORDER_REFLECT)
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def highlights(x: np.ndarray, strength: float, knee: float, detail: float, radius: float,
               level: int = 0) -> np.ndarray:
    """The bright areas brought down above the knee, their detail kept (He et al. 2013's base and detail): the
    lightness's base is compressed by c log(1 + (B - knee) / c), c chosen so that 1 goes to the ceiling
    knee + (1 - knee)(1 - strength / 2); the detail is scaled by the curve's slope (never below a half) and raised by
    ``detail`` where the base is compressed; the channels follow the lightness's gain."""
    if strength <= 0:
        return x
    x = np.ascontiguousarray(x, np.float32)
    L = np.clip(0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2], 0, None).astype(np.float32)
    r = radius / 2 ** level
    B = 0.5 * guided(L, L, r / 3, 0.01) + 0.5 * guided(L, L, r, 0.01)
    D = L - B
    span = max(1.0 - knee, 1e-3)
    top = (1.0 - knee) * (1.0 - 0.5 * min(strength, 1.0))
    lo, hi = 1e-4, 1e3                                # c log(1 + span / c) = top: c by bisection (monotonic in c)
    for _ in range(60):
        c = float(np.sqrt(lo * hi))
        if c * np.log1p(span / c) > top:
            hi = c
        else:
            lo = c
    over = np.clip(B - knee, 0, None)
    above = B > knee
    Bc = np.where(above, knee + c * np.log1p(over / c), B)
    slope = np.where(above, 1.0 / (1.0 + over / c), 1.0)
    gD = np.clip(slope, 0.5, 1.0) * (1.0 + detail * np.clip(over / span, 0, 1))
    Lt = np.clip(Bc + gD * D, 0, 1)
    gain = Lt / np.maximum(L, 1e-6)
    return np.clip(x * gain[..., None], 0, 1).astype(np.float32)


def faint_smoothing(x: np.ndarray, amount: float, threshold: float, radius: float, level: int = 0) -> np.ndarray:
    """The areas fainter than ``threshold`` (lightness; a soft transition over +- 0.1) mixed with their Gaussian blur
    of ``radius`` px by ``amount``; the brighter ones untouched."""
    if amount <= 0:
        return x
    import cv2
    x = np.ascontiguousarray(x, np.float32)
    L = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    s = radius / 2 ** level
    Ls = cv2.GaussianBlur(L, (0, 0), max(s / 4, 0.5), borderType=cv2.BORDER_REFLECT)   # decided on the smooth level
    t = np.clip((Ls - (threshold - 0.1)) / 0.2, 0, 1)
    w = (amount * (1 - t * t * (3 - 2 * t)))[..., None]
    blur = cv2.GaussianBlur(x, (0, 0), max(s, 0.5), borderType=cv2.BORDER_REFLECT)
    return (x + w * (blur - x)).astype(np.float32)
