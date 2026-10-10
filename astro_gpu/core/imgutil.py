"""Small numerical helpers shared by several stages (CPU, numpy / OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np


def robust_sigma(x: np.ndarray) -> tuple[float, float]:
    """Median and MAD-based sigma of the finite values of x."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    m = float(np.median(x))
    return m, float(1.4826 * np.median(np.abs(x - m)))


def hp_noise(img: np.ndarray, sub: int = 7, sigma: float = 2.0) -> float:
    """Pixel noise estimated on a high-pass version of a 2-D image (robust to structures and stars)."""
    a = np.asarray(img, np.float32)
    hp = a - cv2.GaussianBlur(a, (0, 0), sigma)
    v = hp[::sub, ::sub]
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def poly_terms(x: np.ndarray, y: np.ndarray, width: float, height: float, deg: int) -> np.ndarray:
    """2-D monomials u^i v^j (i + j <= deg) on coordinates normalised to [-1, 1]."""
    u = np.asarray(x, np.float64) / (width / 2) - 1
    v = np.asarray(y, np.float64) / (height / 2) - 1
    return np.stack([u ** i * v ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


def robust_lstsq(T: np.ndarray, r: np.ndarray, iters: int = 5, k: float = 3.0):
    """Least squares with iterative k-sigma rejection. Returns (coefficients, sigma, kept mask)."""
    ok = np.isfinite(r)
    q, s = None, 0.0
    for _ in range(iters):
        q = np.linalg.lstsq(T[ok], r[ok], rcond=None)[0]
        e = r - T @ q
        s = 1.4826 * np.median(np.abs(e[ok]))
        ok = np.isfinite(r) & (np.abs(e) < k * s + 1e-12)
    return q, float(s), ok


def fill_holes(img: np.ndarray, mask: np.ndarray, scales=(2, 4, 8, 16, 32, 64, 128)) -> np.ndarray:
    """Fill masked pixels by multi-scale normalised convolution of the surrounding valid pixels.

    Each hole takes the value of the nearest valid neighbourhood; large holes (saturated stars at the frame edge)
    use the coarse scales, and anything still unfilled gets the image median - a raw value is never left in a hole.
    """
    out = img.astype(np.float32).copy()
    keep = (~mask).astype(np.float32)
    todo = mask.copy()
    src = img.astype(np.float32) * keep
    for s in scales:
        num = cv2.GaussianBlur(src, (0, 0), s)
        den = cv2.GaussianBlur(keep, (0, 0), s)
        ok = todo & (den > (0.05 if s <= 32 else 1e-4))
        out[ok] = num[ok] / den[ok]
        todo &= ~ok
        if not todo.any():
            break
    if todo.any():
        out[todo] = np.median(img[keep > 0][::50])
    return out


def starlet(img: np.ndarray, nj: int):
    """Isotropic undecimated wavelet transform (B3 spline, a trous). Returns (details[0..nj-1], coarse)."""
    k1 = np.array([1, 4, 6, 4, 1], np.float32) / 16
    c = img.astype(np.float32)
    out = []
    for j in range(nj):
        step = 2 ** j
        k = np.zeros(4 * step + 1, np.float32)
        k[::step] = k1
        c2 = cv2.sepFilter2D(c, -1, k, k, borderType=cv2.BORDER_REFLECT)
        out.append(c - c2)
        c = c2
    return out, c


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter (He et al. 2013): edge-preserving smoothing of p guided by I."""
    def box(a):
        return cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def bin_median(a: np.ndarray, b: int) -> np.ndarray:
    a = np.asarray(a, np.float32)
    hb, wb = a.shape[0] // b, a.shape[1] // b
    return np.nanmedian(a[: hb * b, : wb * b].reshape(hb, b, wb, b), axis=(1, 3))


def normalized_blur(a: np.ndarray, m: np.ndarray, sigma: float, scale: float = 0.125) -> np.ndarray:
    """Gaussian blur ignoring pixels where m == 0, computed on a reduced grid (for very large sigmas)."""
    small = cv2.resize(a * m, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sm = cv2.resize(m.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    num = cv2.GaussianBlur(small, (0, 0), sigma * scale)
    den = cv2.GaussianBlur(sm, (0, 0), sigma * scale)
    out = num / np.maximum(den, 1e-3)
    return cv2.resize(out, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_CUBIC)


def safe_world2pix(w, ra, dec, width: int, height: int, margin: int = 400):
    """World -> pixel with a SIP WCS, robust far from the frame (the SIP polynomial diverges there).

    Stars are pre-selected with the linear WCS; only those near the frame get the full projection. Others get
    -1e6 so that any "inside the frame" test rejects them.
    """
    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    x0, y0 = w.wcs_world2pix(ra, dec, 0)
    near = (np.isfinite(x0) & np.isfinite(y0) & (x0 > -margin) & (x0 < width + margin)
            & (y0 > -margin) & (y0 < height + margin))
    x = np.full(ra.shape, -1e6)
    y = np.full(ra.shape, -1e6)
    if near.any():
        x[near], y[near] = w.all_world2pix(ra[near], dec[near], 0, quiet=True)
    return x, y


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def asinh_preview(img: np.ndarray, M: np.ndarray | None = None, k: float = 60.0) -> np.ndarray:
    """Quick colour preview of a linear (3, H, W) image: optional colour matrix, background at 5 %, asinh."""
    a = np.asarray(img, np.float32)
    if M is not None:
        a = np.tensordot(M, a, 1)
    sub = a[:, ::4, ::4]
    bk = np.percentile(sub, 5, axis=(1, 2))[:, None, None]
    hi = np.percentile(sub - bk, 99.7)
    v = np.clip((a - bk) / max(hi, 1e-6), 0, None)
    return np.clip(np.arcsinh(v * k) / np.arcsinh(k), 0, 1)
