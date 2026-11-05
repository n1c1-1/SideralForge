"""GPU image primitives of the renderings (torch). Tensors are (C, H, W) or (H, W), float32."""
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


def percentile(x: torch.Tensor, q, mask: torch.Tensor | None = None):
    """Percentile(s) exactly like numpy's default (linear interpolation): one GPU sort, or on the CPU a selection of
    the order statistics needed (numpy's partition, O(n): the same values a sort puts there, several times faster)."""
    v = x[mask] if mask is not None else x.reshape(-1)
    n = v.numel()
    qs = [q] if isinstance(q, (int, float)) else list(q)
    pos = [qq / 100 * (n - 1) for qq in qs]
    idx = [(int(math.floor(p)), min(int(math.floor(p)) + 1, n - 1)) for p in pos]
    if v.device.type == "cpu" and v.dtype == torch.float32 and n > 4096 and not bool(torch.isnan(v).any()):
        import numpy as np
        v = torch.from_numpy(np.partition(v.detach().numpy(), sorted({k for pair in idx for k in pair})))
    else:
        v = torch.sort(v).values
    out = []
    for p, (i0, i1) in zip(pos, idx):              # the same float32 arithmetic on the same two values as before
        out.append(float(v[i0] + (p - i0) * (v[i1] - v[i0])))
    return out[0] if isinstance(q, (int, float)) else out


# ---------------------------------------------------------------------- colour
def luminance(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
