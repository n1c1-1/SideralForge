"""Clipping display of a rendering: where the image is pure black, pure white, or the nebula so bright that its
detail is lost. Shared by the heavy client's Final adjustment and the ComfyUI live render node."""
from __future__ import annotations

import numpy as np

BLACK = (60, 120, 255)      # blue: pure black
BURNT = (230, 170, 60)      # amber: nebula above 92 % (detail loss)
WHITE = (255, 60, 60)       # red: pure white
LEGEND = "Blue: pure black   Red: pure white   Amber: nebula above 92 % (detail loss)"


def clipping_rgba(final_u8: np.ndarray, starless_u8: np.ndarray) -> np.ndarray:
    """An RGBA overlay (H x W x 4, uint8) of the clipped pixels of an 8-bit rendering and its starless layer."""
    h, w, _ = final_u8.shape
    o = np.zeros((h, w, 4), np.uint8)
    black = final_u8.max(2) == 0
    white = final_u8.min(2) == 255
    sl = starless_u8.astype(np.float32) / 255
    burnt = (0.2126 * sl[..., 0] + 0.7152 * sl[..., 1] + 0.0722 * sl[..., 2]) > 0.92
    o[black] = (*BLACK, 255)
    o[burnt] = (*BURNT, 200)
    o[white] = (*WHITE, 255)
    return o


def with_clipping(final_u8: np.ndarray, starless_u8: np.ndarray) -> np.ndarray:
    """The rendering with its clipping overlay laid on it (8-bit RGB)."""
    o = clipping_rgba(final_u8, starless_u8)
    a = o[..., 3:].astype(np.float32) / 255
    return (final_u8 * (1 - a) + o[..., :3] * a + 0.5).astype(np.uint8)
