"""Resampling of a debayered frame onto the reference grid, for the Classic stacking method (``to_reference``).

Geometry: the frame's registration polynomial ``coef`` (reg.npz, the one stages/stack.py drizzles with: frame sensor
pixel (x, y) -> reference sensor position (X, Y), a cubic in x / (W/2) - 1, y / (H/2) - 1) is inverted at every
output pixel (core/poly.poly_inverse: an inverse cubic, then Newton steps on the polynomial itself; 1e-6 px).
Output pixel (j, i) is the reference sensor position X = j, Y = i: the grid of the drizzle at scale 1, the sensor
resolution. A pixel whose interpolation window leaves the frame has no value (NaN): the frame does not cover it (as
OpenCV's BORDER_TRANSPARENT leaves it untouched in Siril, which then counts it as missing).

Interpolation, ``lanczos4`` (default) or ``bilinear``:

* **Lanczos-4**, Siril's default interpolation for registration (``register -interp=lanczos4``) and PixInsight's
  StarAlignment "Auto" choice without a change of scale [S3][S4][P13]: L(t) = sinc(t) sinc(t / 4) for |t| < 4, the 8 x 8
  samples around the position (offsets -3..+4 from its integer part), separable weights.
  - ``clamping="siril"`` (default): Siril 1.4's clamping (src/opencv/opencv.cpp, ``cvTransformImage``): the frame is
    warped twice, with OpenCV's INTER_LANCZOS4 (weights normalised to a sum of 1, OpenCV's ``interpolateLanczos4``)
    and with the "area" interpolation as a guide (OpenCV's warps do not have it and run INTER_LINEAR, bilinear); where
    the Lanczos value is below CLAMPING_FACTOR = 0.98 x the guide, that pixel and its 4 nearest neighbours (the 3 x 3
    "ellipse" of getStructuringElement, a cross) take the guide's value, colour by colour. It removes the dark rings
    Lanczos leaves around stars; Siril's documentation credits it with "often 8 % smaller FWHM" than the pixel-area
    method [S3]. OpenCV quantises the sub-pixel position to 1/32 px (INTER_BITS); here it is exact.
  - ``clamping="pixinsight"``: PCL's LanczosInterpolation (include/pcl/LanczosInterpolation.h and
    LanczosRowInterpolation.h): the weights corrected for their first two moments (PCL_CorrectLanczosMoments, sum 1
    and centroid at the position: no sub-pixel position bias, the PixInsight 1.9.5 fix [P8]); the positive and
    negative contributions s+ and s- summed apart with their weights w+ and w-; r = s- / s+ above the clamping
    threshold c (0.3, the default of PCL and of StarAlignment [P13]) shrinks the negative lobes by 1 - ((r - c) /
    (1 - c))^2, r >= 1 keeps the positive lobes only; value = (s+ - s-) / (w+ - w-).
  - ``clamping="none"``: OpenCV's Lanczos-4 alone.
* **Bilinear**: the 2 x 2 samples around the position (an option, and the guide of Siril's clamping).

Sources: [S3] https://gitlab.com/free-astro/siril-doc/-/raw/1.4/doc/preprocessing/registration.rst ; [S4] .../commands/
register.rst ; Siril 1.4 https://gitlab.com/free-astro/siril/-/blob/1.4/src/opencv/opencv.cpp ; OpenCV
modules/imgproc/src/imgwarp.cpp (interpolateLanczos4, remapLanczos4, INTER_AREA -> INTER_LINEAR in the warps);
PCL https://gitlab.com/pixinsight/PCL (include/pcl/LanczosInterpolation.h, LanczosRowInterpolation.h); [P8] PixInsight
1.9.5 release notes; [P13] StarAlignment documentation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .poly import poly_inverse

METHODS = ("lanczos4", "bilinear")
CLAMPINGS = ("siril", "pixinsight", "none")
#: Siril (opencv.cpp): a Lanczos value below this fraction of the guide is replaced by the guide
CLAMPING_FACTOR = 0.98
#: PCL / StarAlignment default clamping threshold (__PCL_LANCZOS_CLAMPING_THRESHOLD)
PCL_CLAMPING_THRESHOLD = 0.3
#: Lanczos order and window: the taps are at offsets -N + 1 .. N from the integer part
N = 4
#: output pixels processed together (memory: about 20 x 4 bytes x 3 colours each)
CHUNK = 1 << 20


def lanczos(t: torch.Tensor, n: int = N) -> torch.Tensor:
    """L(t; n) = sinc(t) sinc(t / n) for |t| < n, else 0 (sinc(x) = sin(pi x) / (pi x))."""
    return torch.where(t.abs() < n, torch.sinc(t) * torch.sinc(t / n), torch.zeros_like(t))


def lanczos_weights(frac: torch.Tensor, kind: str = "opencv", n: int = N) -> torch.Tensor:
    """(2n, ...) the weights of the samples at offsets -n + 1 .. n from the integer part of a position whose
    fractional part is ``frac``: ``opencv`` normalised to a sum of 1 (interpolateLanczos4), ``pcl`` corrected for
    their first two moments (PCL_CorrectLanczosMoments: sum 1, centroid at the position)."""
    off = torch.arange(-n + 1, n + 1, device=frac.device, dtype=frac.dtype).view(-1, *([1] * frac.dim()))
    t = off - frac[None]
    w = lanczos(t, n)
    if kind == "opencv":
        return w / w.sum(0, keepdim=True)
    if kind == "pcl":
        S0 = 2.0 * n
        S1 = t.sum(0)
        S2 = (t * t).sum(0)
        M0 = w.sum(0)
        M1 = (w * t).sum(0)
        det = S0 * S2 - S1 * S1
        a = ((1 - M0) * S2 + M1 * S1) / det
        b = (-M1 * S0 - (1 - M0) * S1) / det
        return w + a[None] + b[None] * t
    raise ValueError(f"unknown Lanczos weights {kind!r}")


def _gather(flat: torch.Tensor, w: int, iy: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
    return flat[:, iy * w + ix]


def bilinear_at(img: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(C, P) the image (C, h, w) sampled bilinearly at the positions (x, y) (P,), and (P,) whether the 2 x 2
    samples are all inside the frame (outside: NaN)."""
    C, h, w = img.shape
    flat = img.reshape(C, -1)
    x0f, y0f = torch.floor(x), torch.floor(y)
    fx, fy = (x - x0f).to(img.dtype), (y - y0f).to(img.dtype)
    x0, y0 = x0f.long(), y0f.long()
    ok = (x0 >= 0) & (x0 + 1 <= w - 1) & (y0 >= 0) & (y0 + 1 <= h - 1)
    x0c, y0c = x0.clamp(0, w - 2), y0.clamp(0, h - 2)
    v = (_gather(flat, w, y0c, x0c) * (1 - fx) + _gather(flat, w, y0c, x0c + 1) * fx) * (1 - fy) + \
        (_gather(flat, w, y0c + 1, x0c) * (1 - fx) + _gather(flat, w, y0c + 1, x0c + 1) * fx) * fy
    return torch.where(ok[None], v, torch.full_like(v, float("nan"))), ok


def lanczos_at(img: torch.Tensor, x: torch.Tensor, y: torch.Tensor, clamping: str = "siril",
               threshold: float = PCL_CLAMPING_THRESHOLD) -> tuple[torch.Tensor, torch.Tensor]:
    """(C, P) the image (C, h, w) sampled by Lanczos-4 at (x, y) (P,), and (P,) whether the 8 x 8 window is inside
    the frame (outside: NaN). ``clamping``: "pixinsight" applies PCL's clamping here; "siril" and "none" return
    OpenCV's plain Lanczos-4 (Siril's clamping needs the whole grid: ``resample``)."""
    C, h, w = img.shape
    flat = img.reshape(C, -1)
    x0f, y0f = torch.floor(x), torch.floor(y)
    fx, fy = (x - x0f).to(img.dtype), (y - y0f).to(img.dtype)
    x0, y0 = x0f.long(), y0f.long()
    ok = (x0 - N + 1 >= 0) & (x0 + N <= w - 1) & (y0 - N + 1 >= 0) & (y0 + N <= h - 1)
    x0c, y0c = x0.clamp(N - 1, w - 1 - N), y0.clamp(N - 1, h - 1 - N)
    pcl = clamping == "pixinsight"
    kind = "pcl" if pcl else "opencv"
    wx, wy = lanczos_weights(fx, kind), lanczos_weights(fy, kind)
    P = x.numel()
    if not pcl:
        acc = torch.zeros((C, P), device=img.device, dtype=img.dtype)
        for i in range(2 * N):
            row = torch.zeros_like(acc)
            iy = y0c + (i - N + 1)
            for j in range(2 * N):
                row += wx[j] * _gather(flat, w, iy, x0c + (j - N + 1))
            acc += wy[i] * row
    else:
        sp = torch.zeros((C, P), device=img.device, dtype=torch.float64)
        sn, wp, wn = torch.zeros_like(sp), torch.zeros_like(sp), torch.zeros_like(sp)
        for i in range(2 * N):
            iy = y0c + (i - N + 1)
            for j in range(2 * N):
                L2 = (wx[j] * wy[i]).double()[None].expand(C, P)
                s = _gather(flat, w, iy, x0c + (j - N + 1)).double() * L2
                neg = s < 0
                sn = torch.where(neg, sn - s, sn)
                wn = torch.where(neg, wn - L2, wn)
                sp = torch.where(neg, sp, sp + s)
                wp = torch.where(neg, wp, wp + L2)
        if threshold >= 0:
            r = sn / torch.where(sp == 0, torch.ones_like(sp), sp)
            rr = ((r - threshold) / (1 - threshold)).clamp(min=0)
            c = torch.where(r > threshold, 1 - rr * rr, torch.ones_like(r))
            sn2, wn2 = sn * c, wn * c
            acc = (sp - sn2) / (wp - wn2)
            acc = torch.where(r >= 1, sp / wp, acc)
            acc = torch.where(sp == 0, torch.zeros_like(acc), acc)
        else:
            acc = (sp - sn) / (wp - wn)
        acc = acc.to(img.dtype)
    return torch.where(ok[None], acc, torch.full_like(acc, float("nan"))), ok


def _cross_dilate(m: torch.Tensor) -> torch.Tensor:
    """Dilation of the boolean images m (C, h, w) by the 3 x 3 cross (OpenCV's 3 x 3 MORPH_ELLIPSE)."""
    k = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], device=m.device).view(1, 1, 3, 3)
    return F.conv2d(m.float()[:, None], k, padding=1)[:, 0] > 0


def resample(img: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, method: str = "lanczos4",
             clamping: str = "siril", threshold: float = PCL_CLAMPING_THRESHOLD,
             chunk: int = CHUNK) -> torch.Tensor:
    """(C, Ho, Wo): the image (C, h, w) sampled at the source positions x, y (Ho, Wo) of every output pixel, NaN
    where the interpolation window leaves the frame. See the module's docstring for the methods."""
    if method not in METHODS:
        raise ValueError(f"unknown interpolation {method!r}: one of {METHODS}")
    if clamping not in CLAMPINGS:
        raise ValueError(f"unknown clamping {clamping!r}: one of {CLAMPINGS}")
    C = img.shape[0]
    Ho, Wo = x.shape
    xf, yf = x.reshape(-1), y.reshape(-1)
    out = torch.empty((C, Ho * Wo), device=img.device, dtype=img.dtype)
    guide = torch.empty_like(out) if (method == "lanczos4" and clamping == "siril") else None
    for s in range(0, Ho * Wo, chunk):
        xs, ys = xf[s:s + chunk], yf[s:s + chunk]
        if method == "bilinear":
            out[:, s:s + chunk] = bilinear_at(img, xs, ys)[0]
            continue
        out[:, s:s + chunk] = lanczos_at(img, xs, ys, clamping, threshold)[0]
        if guide is not None:
            guide[:, s:s + chunk] = bilinear_at(img, xs, ys)[0]
    out = out.view(C, Ho, Wo)
    if guide is not None:
        guide = guide.view(C, Ho, Wo)
        low = _cross_dilate(out < CLAMPING_FACTOR * guide)
        out = torch.where(low & torch.isfinite(guide), guide, out)
    return out


def source_positions(coef, Ho: int, Wo: int, W: int, H: int, device="cpu", rows: tuple[int, int] | None = None):
    """(x, y) (Ho, Wo) float64: where the frame of registration polynomial ``coef`` (2, 10) (sensor W x H) sees each
    output pixel (j, i) = reference position (X = j, Y = i); ``rows``: only output rows [r0, r1)."""
    r0, r1 = rows if rows is not None else (0, Ho)
    Y, X = torch.meshgrid(torch.arange(r0, r1, device=device, dtype=torch.float64),
                          torch.arange(Wo, device=device, dtype=torch.float64), indexing="ij")
    return poly_inverse(coef, X, Y, W, H)


def to_reference(img: torch.Tensor, coef, W: int, H: int, *, Ho: int | None = None, Wo: int | None = None,
                 method: str = "lanczos4", clamping: str = "siril", threshold: float = PCL_CLAMPING_THRESHOLD,
                 chunk: int = CHUNK) -> torch.Tensor:
    """(C, Ho, Wo): a debayered frame (C, H, W) put on the reference grid (by default the sensor's, Ho x Wo = H x W)
    through its registration polynomial ``coef``; NaN where the frame does not cover."""
    Ho = H if Ho is None else Ho
    Wo = W if Wo is None else Wo
    x, y = source_positions(coef, Ho, Wo, W, H, img.device)
    return resample(img, x, y, method=method, clamping=clamping, threshold=threshold, chunk=chunk)
