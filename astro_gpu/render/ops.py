"""GPU image primitives for the rendering engine (torch). Tensors are (C, H, W) or (H, W), float32."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_KCACHE: dict = {}


def _gauss_kernel(sigma: float, device) -> torch.Tensor:
    key = (round(sigma, 4), str(device))
    if key not in _KCACHE:
        r = max(1, int(math.ceil(4 * sigma)))
        x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        _KCACHE[key] = k / k.sum()
    return _KCACHE[key]


def _as4(x: torch.Tensor):
    if x.dim() == 2:
        return x[None, None], 2
    if x.dim() == 3:
        return x[:, None], 3
    return x, 4


def _back(y: torch.Tensor, nd: int):
    if nd == 2:
        return y[0, 0]
    if nd == 3:
        return y[:, 0]
    return y


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur, kernel radius 4 sigma, mirror borders (like OpenCV for float images).

    Large sigmas are computed on a reduced grid (exact enough for smooth backgrounds, much faster).
    """
    if sigma <= 0:
        return x
    if sigma > 24:
        f = int(sigma // 8)
        y4, nd = _as4(x)
        h, w = y4.shape[-2:]
        small = F.avg_pool2d(F.pad(y4, (0, (-w) % f, 0, (-h) % f), mode="replicate"), f)
        out = _blur4(small, sigma / f)
        out = F.interpolate(out, scale_factor=f, mode="bilinear", align_corners=False)[..., :h, :w]
        return _back(out, nd)
    y4, nd = _as4(x)
    return _back(_blur4(y4, sigma), nd)


def _blur4(y: torch.Tensor, sigma: float) -> torch.Tensor:
    k = _gauss_kernel(sigma, y.device)
    r = (len(k) - 1) // 2
    h, w = y.shape[-2:]
    mode = "reflect" if r < min(h, w) else "replicate"
    y = F.pad(y, (r, r, 0, 0), mode=mode)
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    y = F.pad(y, (0, 0, r, r), mode=mode)
    return F.conv2d(y, k.view(1, 1, -1, 1))


def box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window, mirrored borders, O(1) per pixel (cumulative sums)."""
    y4, nd = _as4(x)
    h, w = y4.shape[-2:]
    mode = "reflect" if r < min(h, w) else "replicate"
    p = F.pad(y4, (r + 1, r, r + 1, r), mode=mode).double()
    c = p.cumsum(-1)
    c = c[..., 2 * r + 1:] - c[..., : -2 * r - 1]
    c = c.cumsum(-2)
    c = c[..., 2 * r + 1:, :] - c[..., : -2 * r - 1, :]
    return _back((c / (2 * r + 1) ** 2).float(), nd)


def guided(I: torch.Tensor, p: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """Guided filter (He et al. 2013)."""
    mI, mp = box_filter(I, r), box_filter(p, r)
    a = (box_filter(I * p, r) - mI * mp) / (box_filter(I * I, r) - mI * mI + eps)
    b = mp - a * mI
    return box_filter(a, r) * I + box_filter(b, r)


def median_filter(x: torch.Tensor, k: int = 5, rows: int = 512) -> torch.Tensor:
    """k x k median (2-D tensor), processed in row bands to bound memory."""
    h, w = x.shape
    r = k // 2
    p = F.pad(x[None, None], (r, r, r, r), mode="reflect")[0, 0]
    out = torch.empty_like(x)
    for y0 in range(0, h, rows):
        y1 = min(h, y0 + rows)
        band = p[y0:y1 + 2 * r]
        u = band.unfold(0, k, 1).unfold(1, k, 1)
        out[y0:y1] = u.reshape(y1 - y0, w, k * k).median(-1).values
    return out


def percentile(x: torch.Tensor, q, mask: torch.Tensor | None = None):
    """Percentile(s) exactly like numpy's default (linear interpolation), via one GPU sort."""
    v = x[mask] if mask is not None else x.reshape(-1)
    v = torch.sort(v).values
    n = v.numel()
    qs = [q] if isinstance(q, (int, float)) else list(q)
    out = []
    for qq in qs:
        pos = qq / 100 * (n - 1)
        i0 = int(math.floor(pos))
        i1 = min(i0 + 1, n - 1)
        out.append(float(v[i0] + (pos - i0) * (v[i1] - v[i0])))
    return out[0] if isinstance(q, (int, float)) else out


def robust_noise(x: torch.Tensor, mask: torch.Tensor | None = None, sigma: float = 2.0, sub: int = 7) -> float:
    hp = x - gaussian_blur(x, sigma)
    v = hp[::sub, ::sub]
    if mask is not None:
        v = v[mask[::sub, ::sub]]
    v = v.reshape(-1)
    return float(1.4826 * (v - v.median()).abs().median())


# ---------------------------------------------------------------------- colour
def luminance(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def srgb_oetf(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=1e-12) ** (1 / 2.4) - 0.055)


def srgb_eotf(v: torch.Tensor) -> torch.Tensor:
    v = v.clamp(min=0)
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


_M1 = torch.tensor([[0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = torch.tensor([[0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660]])


def _mat(m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,j...->i...", m.to(x.device, x.dtype), x)


def lin_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    lms = _mat(_M1, rgb)
    lms = torch.sign(lms) * lms.abs() ** (1 / 3)
    return _mat(_M2, lms)


def oklab_to_lin(lab: torch.Tensor) -> torch.Tensor:
    lms = _mat(torch.linalg.inv(_M2), lab)
    return _mat(torch.linalg.inv(_M1), lms ** 3)


def gamut_map_oklab(lin: torch.Tensor, iters: int = 14) -> torch.Tensor:
    """Bring out-of-gamut pixels into [0, 1] by reducing OKLab chroma at constant hue and lightness."""
    bad = (lin.min(0).values < 0) | (lin.max(0).values > 1)
    if not bool(bad.any()):
        return lin
    sub = lin[:, bad]
    lab = lin_to_oklab(sub)
    L = lab[0].clamp(0, 1)
    lo = torch.zeros_like(L)
    hi = torch.ones_like(L)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        t = oklab_to_lin(torch.stack([L, lab[1] * mid, lab[2] * mid]))
        ok = (t.min(0).values >= -1e-4) & (t.max(0).values <= 1 + 1e-4)
        lo = torch.where(ok, mid, lo)
        hi = torch.where(ok, hi, mid)
    out = lin.clone()
    out[:, bad] = oklab_to_lin(torch.stack([L, lab[1] * lo, lab[2] * lo])).clamp(0, 1)
    return out


def chroma_ok(disp: torch.Tensor, k: float, protect_dark: float) -> torch.Tensor:
    lab = lin_to_oklab(srgb_eotf(disp))
    w = ((lab[0] - protect_dark) / 0.25).clamp(0, 1)
    lab[1:] = lab[1:] * (1 + (k - 1) * w[None])
    return srgb_oetf(gamut_map_oklab(oklab_to_lin(lab)))


def color_preserving_stretch_ok(lin_rgb: torch.Tensor, L_disp: torch.Tensor) -> torch.Tensor:
    Y = luminance(lin_rgb)
    Yt = srgb_eotf(L_disp.clamp(0, 1))
    rgb = lin_rgb * (Yt / Y.clamp(min=1e-9))[None]
    return srgb_oetf(gamut_map_oklab(rgb))


def screen(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1 - (1 - a) * (1 - b)
