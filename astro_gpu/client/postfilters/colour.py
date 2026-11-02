"""CIE L*a*b* and the SCNR of Post's filters.

* The layers of Post are display-referred sRGB values in [0, 1] (after the Output stretch). CIE L*a*b* (D65) is computed
  from them with the exact sRGB transfer function (IEC 61966-2-1) and OpenCV's linear-RGB to L*a*b* conversion (its
  sRGB path approximates the transfer by a spline: up to 1.3 8-bit levels off in a round trip; this one 0.07). Values
  are clipped to [0, 1] first, as in PixInsight and Siril.
* SCNR: Siril's "Remove green noise" (processing/colors, derived from PixInsight's SCNR): Average Neutral, Maximum
  Neutral, Maximum Mask and Additive Mask protection, with the amount and the "preserve lightness" option (the original
  CIE L* is put back, computed as Siril computes it, on the values taken as linear RGB: equal to siril-cli 1.4's).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

LUM = (0.2126, 0.7152, 0.0722)                         # Rec. 709 / sRGB luminance weights (Siril's "human" weights)


def f32(x: np.ndarray) -> np.ndarray:
    return x if x.dtype == np.float32 and x.flags.c_contiguous else np.ascontiguousarray(x, np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(f32(x)).clamp(0, 1)
    return torch.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055).pow_(2.4)).numpy()


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(f32(x)).clamp(0, 1)
    return torch.where(t <= 0.0031308, t * 12.92, t.pow(1 / 2.4).mul_(1.055).sub_(0.055)).numpy()


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (h, w, 3) in [0, 1] -> CIE L*a*b* D65 (L* 0-100)."""
    return cv2.cvtColor(srgb_to_linear(rgb), cv2.COLOR_LRGB2Lab)


def from_lab(lab: np.ndarray) -> np.ndarray:
    """CIE L*a*b* D65 -> sRGB in [0, 1] (out of gamut: clipped)."""
    return linear_to_srgb(cv2.cvtColor(f32(lab), cv2.COLOR_Lab2LRGB))


def lightness(rgb: np.ndarray) -> np.ndarray:
    """CIE L* (0-100) of an sRGB image."""
    return to_lab(rgb)[..., 0].copy()


def lab_linear(rgb: np.ndarray) -> np.ndarray:
    """CIE L*a*b* of the values taken as linear RGB (Siril's conversion)."""
    return cv2.cvtColor(np.clip(f32(rgb), 0, 1), cv2.COLOR_LRGB2Lab)


def with_lightness(rgb: np.ndarray, L: np.ndarray) -> np.ndarray:
    """The image with its CIE L* replaced by L (its a* and b* kept): "preserve lightness"."""
    lab = to_lab(rgb)
    lab[..., 0] = L
    return from_lab(lab)


SCNR_METHODS = ["Average neutral", "Maximum neutral", "Maximum mask", "Additive mask"]


def scnr(x: np.ndarray, method: str, amount: float, preserve: bool) -> np.ndarray:
    """SCNR, Siril's formulas: m = (R + B) / 2 (average neutral), max(R, B) (maximum neutral and mask), min(1, R + B)
    (additive mask). Neutral protections: G' = min(G, m), mixed with G by the amount (PixInsight's amount, 1 = Siril's);
    mask protections: G' = G (1 - a)(1 - m) + m G. ``preserve``: the original CIE L* is put back."""
    if amount <= 0:
        return x
    x = f32(x)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    if method == "Average neutral":
        g2 = g + amount * (np.minimum(g, (r + b) * 0.5) - g)
    elif method == "Maximum neutral":
        g2 = g + amount * (np.minimum(g, np.maximum(r, b)) - g)
    else:
        m = np.maximum(r, b) if method == "Maximum mask" else np.minimum(1.0, r + b)
        m = np.clip(m, 0, 1)
        g2 = g * (1 - amount) * (1 - m) + m * g
    y = x.copy()
    y[..., 1] = g2
    if preserve:
        lab = lab_linear(y)
        lab[..., 0] = lab_linear(x)[..., 0]
        y = cv2.cvtColor(lab, cv2.COLOR_Lab2LRGB)
    return y
