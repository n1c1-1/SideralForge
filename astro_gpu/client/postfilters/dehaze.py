"""Dehaze for night skies: the veil (residual sky glow, haze, a gradient left over) measured on a large-scale background
map, then taken off with the haze model, the white kept (issue #22).

The dark channel prior (K. He, J. Sun, X. Tang, "Single Image Haze Removal Using Dark Channel Prior", CVPR 2009 /
IEEE TPAMI 2011), the reference of daylight dehazing, does not hold on a night sky: the veil there is the darkest level,
not the brightest, and its "airlight" would be taken from the nebula's cores or the stars. The veil is measured instead
as the background of the astronomical source extractors (E. Bertin, S. Arnouts, "SExtractor", A&AS 117, 1996; its
documentation, Background): the image cut in meshes, each mesh's level its mode after an iterative 3-sigma clipping
(2.5 median - 1.5 mean, the median when the two differ by more than 30 % of the scatter), the grid of levels filtered by
a 3 x 3 median and interpolated by bicubic splines. It is then taken off with He's haze model I = J t + A (1 - t),
the airlight A the white (1): J = (I - h) / (1 - h), the veil h = Amount x the map, so that the white stays white and
the sky keeps (1 - Amount) of its level, never crushed (He's "a little haze kept", omega <= 0.95).

Each colour apart (a veil has a colour). Faint nebulosity wider than the meshes counts as veil: Scale is the key
control.
"""
from __future__ import annotations

import cv2
import numpy as np

#: the clipping of a mesh: iterations, sigmas; the share of the scatter beyond which the mode falls back to the median
CLIP_ITER, CLIP_SIGMA, MODE_TOLERANCE = 3, 3.0, 0.3
AMOUNT_MAX = 0.95


def mesh_levels(c: np.ndarray, m: int) -> np.ndarray:
    """The background level of each m x m mesh of one channel (h, w), SExtractor's: (gh, gw)."""
    h, w = c.shape
    gh, gw = max(1, -(-h // m)), max(1, -(-w // m))
    pad = np.pad(c, ((0, gh * m - h), (0, gw * m - w)), mode="reflect" if min(h, w) > 1 else "edge")
    a = pad.reshape(gh, m, gw, m).transpose(0, 2, 1, 3).reshape(gh, gw, m * m).astype(np.float64)
    for _ in range(CLIP_ITER):
        med = np.nanmedian(a, axis=2, keepdims=True)
        sd = np.nanstd(a, axis=2, keepdims=True)
        a = np.where(np.abs(a - med) <= CLIP_SIGMA * sd + 1e-12, a, np.nan)
    med, mean, sd = np.nanmedian(a, axis=2), np.nanmean(a, axis=2), np.nanstd(a, axis=2)
    mode = 2.5 * med - 1.5 * mean
    return np.where(np.abs(mean - med) > MODE_TOLERANCE * sd, med, mode)


def background_map(x: np.ndarray, m: int) -> np.ndarray:
    """The large-scale background of an image (h, w, 3): per channel, the mesh levels, a 3 x 3 median over the grid,
    then a bicubic interpolation back to the pixels (each level at its mesh's centre)."""
    h, w = x.shape[:2]
    out = np.empty_like(x, dtype=np.float32)
    for k in range(x.shape[2]):
        g = mesh_levels(x[..., k], m).astype(np.float32)
        if min(g.shape) >= 3:
            g = cv2.medianBlur(g, 3)
        gh, gw = g.shape
        # the grid at the mesh centres, then the pixels: resize with half-mesh alignment, cut to the image
        big = cv2.resize(g, (gw * m, gh * m), interpolation=cv2.INTER_CUBIC)
        out[..., k] = big[:h, :w]
    return out


def dehaze(x: np.ndarray, amount: float, scale_px: float) -> np.ndarray:
    """The image (h, w, 3, 0..1) with ``amount`` (0..AMOUNT_MAX) of its veil taken off; meshes of ``scale_px``
    pixels of this image."""
    a = float(np.clip(amount, 0.0, AMOUNT_MAX))
    if a <= 0:
        return x
    m = max(8, int(round(scale_px)))
    h = np.clip(a * background_map(x, m), 0.0, AMOUNT_MAX)
    return np.clip((x - h) / (1.0 - h), 0.0, 1.0).astype(np.float32)
