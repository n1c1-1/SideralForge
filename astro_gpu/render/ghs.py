"""The generalised hyperbolic stretch (D. Payne and M. Cranfield, GHSastro; Siril's "Generalised Hyperbolic Stretch",
siril.org/tutorials/ghs; the equations of Siril's ``ght.c``): the transformation whose slope is D at the symmetry point
SP, shaped by the local intensity b, linear below the shadow protection LP and above the highlight protection HP with
the slope it has there. Shared by Post's GHS filter (client/postfilters/tone.py) and the layer's Faint tones slider
(render/simple.py)."""
from __future__ import annotations

import math

import torch


def _q(y: torch.Tensor, D: float, b: float) -> torch.Tensor:
    """The right branch of the GHS for y = x - SP >= 0: q(0) = 0, q'(0) = D (expm1 / log1p forms: no cancellation)."""
    if b == -1.0:
        return torch.log1p(y * D)
    if b < 0:
        B = -b
        return torch.expm1(torch.log1p(y * (B * D)) * ((B - 1) / B)) / (B - 1)
    if b == 0:
        return -torch.expm1(y * -D)
    return -torch.expm1(torch.log1p(y * (b * D)) * (-1.0 / b))


def _dq(y: float, D: float, b: float) -> float:
    """q'(y), y >= 0 (a number)."""
    if b < 0:
        return D * (1 + -b * D * y) ** (1.0 / b)
    if b == 0:
        return D * math.exp(-D * y)
    return D * (1 + b * D * y) ** (-(1 + b) / b)


def _t(x: torch.Tensor, D, b, SP, LP, HP) -> torch.Tensor:
    """The GHS before normalisation."""
    out = torch.where(x >= SP, _q((x - SP).clamp_(min=0), D, b), -_q((SP - x).clamp_(min=0), D, b))
    if LP < SP:
        tl = -float(_q(torch.tensor([SP - LP], dtype=torch.float64), D, b)[0])
        out = torch.where(x < LP, (x - LP) * _dq(SP - LP, D, b) + tl, out)
    if HP > SP:
        th = float(_q(torch.tensor([HP - SP], dtype=torch.float64), D, b)[0])
        out = torch.where(x > HP, (x - HP) * _dq(HP - SP, D, b) + th, out)
    return out


def ghs_params(p: dict) -> tuple[float, float, float, float, float, float]:
    """(D, b, SP, LP, HP, BP) from the filter's settings (stretch as ln(D + 1); LP <= SP <= HP enforced)."""
    D = math.expm1(float(p.get("stretch", 0.0)))
    sp = float(p.get("symmetry", 0.0))
    lp = min(float(p.get("shadows", 0.0)), sp)
    hp = max(float(p.get("highlights", 1.0)), sp)
    return D, float(p.get("local", 0.0)), sp, lp, hp, float(p.get("black", 0.0))


def ghs_unit(t: torch.Tensor, D: float, b: float, SP: float, LP: float, HP: float) -> torch.Tensor:
    """The GHS of values in 0..1, normalised so that 0 -> 0 and 1 -> 1."""
    if D <= 1e-9:
        return t
    t = t.clamp(0, 1)
    ends = _t(torch.tensor([0.0, 1.0], dtype=torch.float64), D, b, SP, LP, HP)
    t0, t1 = float(ends[0]), float(ends[1])
    return (_t(t, D, b, SP, LP, HP) - t0).mul_(1.0 / max(t1 - t0, 1e-30)).clamp_(0, 1)
