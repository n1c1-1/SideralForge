"""The registration polynomial in torch: a frame's sensor positions to the reference's (``poly_apply``, the cubic of
stages/register.py: x / (W/2) - 1, y / (H/2) - 1 -> reference position) and back (``poly_inverse``). core/resample.py
inverts it at every output pixel of the Classic stacking method.
"""
from __future__ import annotations

import numpy as np
import torch


def _poly(c, u, v):
    """The cubic of register.poly_terms with coefficients c (10,), on normalised coordinates u, v."""
    return (c[0] + c[1] * u + c[2] * v + c[3] * u * u + c[4] * u * v + c[5] * v * v + c[6] * u ** 3
            + c[7] * u * u * v + c[8] * u * v * v + c[9] * v ** 3)


def poly_apply(coef, x: torch.Tensor, y: torch.Tensor, W: int, H: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sensor positions (x, y) of a frame -> the reference's, by its registration polynomial ``coef`` (2, 10)
    (register.apply_poly, in torch)."""
    c = np.asarray(coef, np.float64)
    u, v = x / (W / 2) - 1.0, y / (H / 2) - 1.0
    return _poly([float(a) for a in c[0]], u, v), _poly([float(a) for a in c[1]], u, v)


def _poly_terms_np(x, y, W, H):
    u, v = x / (W / 2) - 1.0, y / (H / 2) - 1.0
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3], -1)


def poly_inverse(coef, X: torch.Tensor, Y: torch.Tensor, W: int, H: int, iters: int = 3, chunk: int = 1 << 21):
    """Reference positions (X, Y) -> the frame's sensor positions: an inverse cubic fitted on a grid, then Newton
    steps on the polynomial itself (float64; 1e-6 px after three steps), by chunks of ``chunk`` positions."""
    c = np.asarray(coef, np.float64)
    gx, gy = np.meshgrid(np.linspace(-0.25 * W, 1.25 * W, 25), np.linspace(-0.25 * H, 1.25 * H, 25))
    gx, gy = gx.ravel(), gy.ravel()
    T = _poly_terms_np(gx, gy, W, H)
    A = _poly_terms_np(T @ c[0], T @ c[1], W, H)
    inv = np.stack([np.linalg.lstsq(A, gx, rcond=None)[0], np.linalg.lstsq(A, gy, rcond=None)[0]])
    Xf, Yf = X.reshape(-1), Y.reshape(-1)
    xs, ys = [], []
    for s in range(0, Xf.numel(), chunk):
        x, y = _newton(c, inv, Xf[s:s + chunk].double(), Yf[s:s + chunk].double(), W, H, iters)
        xs.append(x)
        ys.append(y)
    if not xs:
        return X.clone(), Y.clone()
    return torch.cat(xs).reshape(X.shape).to(X.dtype), torch.cat(ys).reshape(Y.shape).to(Y.dtype)


def _newton(c, inv, Xd, Yd, W, H, iters):
    x, y = poly_apply(inv, Xd, Yd, W, H)
    cx, cy = [float(a) for a in c[0]], [float(a) for a in c[1]]
    for _ in range(iters):
        u, v = x / (W / 2) - 1.0, y / (H / 2) - 1.0
        rx, ry = Xd - _poly(cx, u, v), Yd - _poly(cy, u, v)
        # the Jacobian of the polynomial, per sensor pixel
        a = (cx[1] + 2 * cx[3] * u + cx[4] * v + 3 * cx[6] * u * u + 2 * cx[7] * u * v + cx[8] * v * v) * (2 / W)
        b = (cx[2] + cx[4] * u + 2 * cx[5] * v + cx[7] * u * u + 2 * cx[8] * u * v + 3 * cx[9] * v * v) * (2 / H)
        e = (cy[1] + 2 * cy[3] * u + cy[4] * v + 3 * cy[6] * u * u + 2 * cy[7] * u * v + cy[8] * v * v) * (2 / W)
        d = (cy[2] + cy[4] * u + 2 * cy[5] * v + cy[7] * u * u + 2 * cy[8] * u * v + 3 * cy[9] * v * v) * (2 / H)
        det = a * d - b * e
        x = x + (d * rx - b * ry) / det
        y = y + (a * ry - e * rx) / det
    return x, y

