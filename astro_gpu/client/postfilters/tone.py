"""The stretches of Post, each a published method, the identity at its defaults.

* **Generalised hyperbolic stretch** (D. Payne and M. Cranfield, GHSastro; Siril's "Generalised Hyperbolic Stretch",
  siril.org/tutorials/ghs and processing/stretching; PixInsight's GHS). The stretch factor is given as ln(D + 1), as in
  Siril; b the local stretch intensity, SP the symmetry point, LP and HP the shadow and highlight protection points, BP a
  black point moved first (Siril's "linear stretch"). The transformation is the family whose slope is D at SP:
  T'(x) = D (1 + b D |x - SP|)^(-(1 + b) / b) for b > 0, D e^(-D |x - SP|) for b = 0, D (1 + |b| D |x - SP|)^(-1 / |b|)
  for b < 0 (b = -1: logarithmic), symmetric about SP, linear below LP and above HP with the slope it has there, and
  normalised so that 0 -> 0 and 1 -> 1 (the equations of Siril's ``ght.c``). Colour: each channel, or the luminance
  (human-weighted 0.2126 / 0.7152 / 0.0722, or even-weighted) with the colour ratios kept.
* **Asinh transformation** (Siril, processing/stretching; after Lupton et al. 2004, PASP 116, 133): as Siril's code
  runs it (checked against siril-cli 1.4), x' = max(0, (x - blackpoint) / (1 - blackpoint)), then
  pixel = x' * asinh(L' * stretch) / (L' * asinh(stretch)), with L' the even- or human-weighted luminance of x' (the
  colour ratios kept), each channel clipped to [0, 1] (Siril's clipping of this stretch).
* **Histogram transformation** (PixInsight HistogramTransformation, Siril's MTF): x' = (x - shadows) / (highlights -
  shadows), clipped to [0, 1], then the midtones transfer function MTF(m, x) = (m - 1) x / ((2m - 1) x - m).

The luminance-based ones can push a channel above 1: the pixel is then divided by its largest channel (the hue kept),
Siril's "Rescale" clipping mode (equal to siril-cli's to the 8-bit level).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .colour import LUM, f32

COLOUR_MODES = ["Independent channels", "Human-weighted luminance", "Even-weighted luminance"]
HT_CHANNELS = ["RGB/K", "R", "G", "B"]


# ---------------------------------------------------------------------------------------------- GHS
from ...render.ghs import _dq, _q, _t, ghs_params, ghs_unit  # noqa: E402,F401  (the engine's, shared)


def ghs_map(t: torch.Tensor, p: dict) -> torch.Tensor:
    """The GHS of values in a tensor (any shape)."""
    D, b, SP, LP, HP, BP = ghs_params(p)
    if BP > 0:
        t = ((t - BP) / max(1e-6, 1 - BP)).clamp_(0, 1)
    return ghs_unit(t, D, b, SP, LP, HP)


def is_ghs_identity(p: dict) -> bool:
    D, _b, _sp, _lp, _hp, BP = ghs_params(p)
    return D <= 1e-9 and BP <= 0


def ghs_curve(p: dict, xs: np.ndarray) -> np.ndarray:
    """The transfer function at the points xs (for the graph and the presets)."""
    return ghs_map(torch.from_numpy(np.asarray(xs, np.float64)), p).numpy()


# ---------------------------------------------------------------------------------------------- colour models
def _luma(t: torch.Tensor, mode: str) -> torch.Tensor:
    w = LUM if mode.startswith("Human") else (1 / 3, 1 / 3, 1 / 3)
    return t[..., 0] * w[0] + t[..., 1] * w[1] + t[..., 2] * w[2]


def _rescale(t: torch.Tensor) -> torch.Tensor:
    """Siril's "Rescale": a pixel whose largest channel passes 1 is divided by it."""
    return (t / t.amax(-1, keepdim=True).clamp(min=1.0)).clamp_(0, 1)


def by_luminance(x: np.ndarray, fn, mode: str) -> np.ndarray:
    """``fn`` (values -> values) applied to the luminance, every channel scaled by the same ratio."""
    t = torch.from_numpy(f32(x))
    L = _luma(t, mode).clamp(0, 1)
    L2 = fn(L.clone())
    k = torch.where(L > 1e-7, L2 / L.clamp(min=1e-7), torch.ones_like(L))
    return _rescale(t * k[..., None]).numpy()


def ghs(x: np.ndarray, p: dict) -> np.ndarray:
    if is_ghs_identity(p):
        return x
    mode = p.get("model", COLOUR_MODES[0])
    if mode == COLOUR_MODES[0]:
        return ghs_map(torch.from_numpy(f32(x)).clone(), p).numpy()
    return by_luminance(x, lambda L: ghs_map(L, p), mode)


# ---------------------------------------------------------------------------------------------- asinh
def asinh_factor(L: torch.Tensor, beta: float) -> torch.Tensor:
    """asinh(L beta) / (L asinh(beta)), its limit beta / asinh(beta) at L = 0, 1 at beta = 0."""
    if beta <= 1e-9:
        return torch.ones_like(L)
    a = math.asinh(beta)
    return torch.where(L > 1e-7, torch.asinh(L * beta) / (L.clamp(min=1e-7) * a), torch.full_like(L, beta / a))


def asinh(x: np.ndarray, p: dict) -> np.ndarray:
    beta, bp = float(p.get("stretch", 0.0)), float(p.get("black", 0.0))
    if beta <= 1e-9 and bp <= 0:
        return x
    t = torch.from_numpy(f32(x))
    if bp > 0:
        t = ((t - bp) / max(1e-6, 1 - bp)).clamp_(min=0)
    L = _luma(t, p.get("model", "Even-weighted luminance")).clamp(min=0)
    return (t * asinh_factor(L, beta)[..., None]).clamp_(0, 1).numpy()


def asinh_curve(p: dict, xs: np.ndarray) -> np.ndarray:
    """The transfer function of a grey pixel (for the graph)."""
    bp = float(p.get("black", 0.0))
    t = ((torch.from_numpy(np.asarray(xs, np.float64)) - bp) / max(1e-6, 1 - bp)).clamp(min=0)
    return (t * asinh_factor(t, float(p.get("stretch", 0.0)))).clamp(0, 1).numpy()


# ---------------------------------------------------------------------------------------------- histogram transformation
def mtf(m: float, t: torch.Tensor) -> torch.Tensor:
    """The midtones transfer function: MTF(m, 0) = 0, MTF(m, m) = 0.5, MTF(m, 1) = 1."""
    m = min(max(float(m), 1e-5), 1 - 1e-5)
    return (t * (m - 1)) / (t * (2 * m - 1) - m)


def ht_map(t: torch.Tensor, p: dict) -> torch.Tensor:
    s, m, h = float(p.get("shadows", 0.0)), float(p.get("midtones", 0.5)), float(p.get("highlights", 1.0))
    h = max(h, s + 1e-4)
    t = ((t - s) / (h - s)).clamp_(0, 1)
    return t if abs(m - 0.5) < 1e-9 else mtf(m, t)


def is_ht_identity(p: dict) -> bool:
    return float(p.get("shadows", 0.0)) <= 0 and float(p.get("highlights", 1.0)) >= 1 and \
        abs(float(p.get("midtones", 0.5)) - 0.5) < 1e-9


def histogram_transform(x: np.ndarray, p: dict) -> np.ndarray:
    if is_ht_identity(p):
        return x
    ch = p.get("channel", "RGB/K")
    t = torch.from_numpy(f32(x)).clone()
    if ch == "RGB/K":
        return ht_map(t, p).numpy()
    k = "RGB".index(ch)
    t[..., k] = ht_map(t[..., k].clone(), p)
    return t.numpy()


def ht_curve(p: dict, xs: np.ndarray) -> np.ndarray:
    return ht_map(torch.from_numpy(np.asarray(xs, np.float64)), p).numpy()
