"""Clipping display (astro_gpu/render/clipping.py), shared by the heavy client and the ComfyUI live node."""
import numpy as np

from astro_gpu.render.clipping import BLACK, BURNT, WHITE, clipping_rgba, with_clipping


def test_clipping_colours():
    f = np.full((3, 3, 3), 100, np.uint8)
    f[0, 0] = 0                    # pure black
    f[0, 1] = 255                  # pure white
    f[0, 2] = (255, 255, 10)       # one channel low: not white
    s = np.zeros((3, 3, 3), np.uint8)
    s[1, 1] = 240                  # starless above 92 %: detail lost
    o = clipping_rgba(f, s)
    assert tuple(o[0, 0, :3]) == BLACK and tuple(o[0, 1, :3]) == WHITE
    assert o[0, 2, 3] == 0 and o[2, 2, 3] == 0
    assert tuple(o[1, 1, :3]) == BURNT and 0 < o[1, 1, 3] < 255


def test_with_clipping_only_changes_marked_pixels():
    f = np.full((2, 2, 3), 100, np.uint8)
    f[0, 0] = 0
    out = with_clipping(f, np.zeros_like(f))
    assert tuple(out[0, 0]) == BLACK and (out[1:] == 100).all()
