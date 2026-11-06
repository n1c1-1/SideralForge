"""The colour filter mosaic of a sensor: a 2 x 2 Bayer pattern or Fujifilm's 6 x 6 X-Trans one (issue #35).

A pattern is written as its colours row by row from the frame's top-left pixel: "RGGB" for a Bayer sensor, 36
letters for an X-Trans one (6 rows of 6, "RBGBRG" "GGRGGB" ...). Its period P (2 or 6) is the step of each position's
lattice: what walks the mosaic position by position (the flat's normalisation, the hot pixels, the fixed pattern,
the sky levels, the colour of each drizzle drop) takes it from here, so that both sensors go through the same code.

Monochrome (issue #43): a sensor without colour filters is the pattern ``MONO``, "GGGG": every sample one colour, so
that every walk of the mosaic above holds for it (the lattice of period 2, its four positions alike); the stack
deposits its samples in one channel and gives that channel to the three (``mono_rgb``): a grey image, which the
blocks after it read as any stack. A mono session's filter (Ha, OIII, L, R...) is the session's.

X-Trans: 20 green, 8 red and 8 blue samples in each 6 x 6 block, the three colours in every row and column and in
every 3 x 3 window (checked by the tests): a 3 x 3 neighbourhood holds each colour, so the bilinear demosaicing and
the half-resolution previews below work for it as for a Bayer pattern. Sources: Fujifilm's description of the X-Trans
filter array; LibRaw, which gives the colour of every pixel of the file (``raw_colors_visible``), the pattern read here.
"""
from __future__ import annotations

import numpy as np

CHANNELS = "RGB"
#: the four Bayer patterns
BAYER = ("RGGB", "BGGR", "GRBG", "GBRG")
#: a monochrome sensor (no colour filter array): every sample counts as one channel (green), see ``mono_rgb``
MONO = "GGGG"
#: the samples of each colour in a 6 x 6 X-Trans block
XTRANS_COUNTS = {"R": 8, "G": 20, "B": 8}


def period(cfa: str) -> int:
    """The side of the pattern's repeating block: 2 (Bayer) or 6 (X-Trans)."""
    n = len(cfa or "")
    if n == 4:
        return 2
    if n == 36:
        return 6
    raise ValueError(f"not a colour pattern: {cfa!r} (4 letters for a Bayer sensor, 36 for an X-Trans one)")


def is_xtrans(cfa) -> bool:
    return bool(cfa) and len(cfa) == 36


def is_mono(cfa) -> bool:
    return cfa == MONO


def mono_rgb(a: np.ndarray) -> np.ndarray:
    """A monochrome stack's image (3, H, W), its samples in the green channel: that channel in the three."""
    return np.repeat(np.asarray(a)[1:2], 3, axis=0)


def is_supported(cfa) -> bool:
    """A Bayer pattern, monochrome, or an X-Trans one (the three colours in their counts, each row and column holding
    all three)."""
    if cfa in BAYER or cfa == MONO:
        return True
    if not is_xtrans(cfa) or any(cfa.count(c) != n for c, n in XTRANS_COUNTS.items()):
        return False
    g = channels(cfa)
    return all(len(set(g[i])) == 3 and len(set(g[:, i])) == 3 for i in range(6))


def channels(cfa: str) -> np.ndarray:
    """(P, P) int: the colour (0 R, 1 G, 2 B) of each position of the pattern."""
    P = period(cfa)
    return np.array([CHANNELS.index(c) for c in cfa.upper()]).reshape(P, P)


def positions(cfa: str) -> list[tuple[int, int, int]]:
    """(row, column, colour) of each position k = P row + column of the pattern."""
    g = channels(cfa)
    P = g.shape[0]
    return [(k // P, k % P, int(g[k // P, k % P])) for k in range(P * P)]


def colour_map(cfa: str, H: int, W: int) -> np.ndarray:
    """(H, W) uint8: the colour of every sensor pixel."""
    g = channels(cfa).astype(np.uint8)
    P = g.shape[0]
    return np.tile(g, (-(-H // P), -(-W // P)))[:H, :W]


def plane(a, k: int, P: int):
    """The samples of position k (k = P row + column) of an (..., H, W) mosaic: a view."""
    i, j = divmod(k, P)
    return a[..., i::P, j::P]


def green_position(cfa: str) -> int:
    """The first position of the pattern holding a green sample."""
    return cfa.upper().index("G")


def superpixel(a: np.ndarray, cfa: str) -> np.ndarray:
    """(H // 2, W // 2, 3): each 2 x 2 cell one colour pixel. Bayer: the cell's own samples (the two greens averaged).
    X-Trans (a 2 x 2 cell does not hold the three colours): each colour the mean of its samples in the 3 x 3
    neighbourhood of every pixel, averaged over the cell - the same size, so that the previews and their coordinates
    are those of a Bayer sensor."""
    a = np.asarray(a, np.float32)
    H, W = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    if is_mono(cfa):                                 # a grey picture: the cell's mean in the three channels
        m = a[:H, :W].reshape(H // 2, 2, W // 2, 2).mean((1, 3))
        return np.repeat(m[:, :, None], 3, axis=2)
    if not is_xtrans(cfa):
        out = np.zeros((H // 2, W // 2, 3), np.float32)
        cnt = np.zeros(3, np.float32)
        for i, j, c in positions(cfa):
            out[:, :, c] += a[i:H:2, j:W:2]
            cnt[c] += 1
        return out / np.maximum(cnt, 1)
    import cv2
    col = colour_map(cfa, *a.shape)
    fin = np.isfinite(a)
    x = np.where(fin, a, 0).astype(np.float32)
    out = np.empty((H // 2, W // 2, 3), np.float32)
    for c in range(3):
        m = ((col == c) & fin).astype(np.float32)
        num, den = (cv2.boxFilter(z, -1, (3, 3), normalize=False, borderType=cv2.BORDER_REFLECT) for z in (x * m, m))
        if (den <= 0).any():                         # a frame's corner, where the mirrored window lacks the colour
            n5, d5 = (cv2.boxFilter(z, -1, (5, 5), normalize=False, borderType=cv2.BORDER_REFLECT) for z in (x * m, m))
            num, den = np.where(den > 0, num, n5), np.where(den > 0, den, d5)
        v = (num / np.maximum(den, 1e-6))[:H, :W]
        out[:, :, c] = v.reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    return out


def equalised(a: np.ndarray, cfa: str, valid_max: float | None = None) -> np.ndarray:
    """The mosaic with each colour's sky brought to the green's (its median offset to the green's taken off), for
    what bins it in 2 x 2 cells to find the stars. A Bayer cell holds every colour in the same counts, so its sum has
    no pattern and the frame is returned as it is; an X-Trans cell holds 1 to 3 greens: its sky would be a pattern of
    period 3 unless the colours are levelled first. An offset, not a gain: the pattern is the sky's levels, and a
    gain would raise the noise of the fainter colours (on an X-T3 frame of M31, the noise of the binned frame: 76 as
    it is, 72 levelled by offsets, 110 by gains, and fewer stars found). Values over ``valid_max`` (the marks of
    saturated samples) are left as they are."""
    if not is_xtrans(cfa):
        return a
    a = np.array(a, np.float32)
    col = colour_map(cfa, *a.shape)
    ok = np.isfinite(a) if valid_max is None else np.isfinite(a) & (a <= valid_max)
    sub = (slice(None, None, 5), slice(None, None, 5))      # a step prime to 6: every position of the pattern
    lv = []
    for c in range(3):
        v = a[sub][(col[sub] == c) & ok[sub]]
        lv.append(float(np.median(v)) if v.size else np.nan)
    if not all(np.isfinite(lv)):
        return a
    off = (np.array(lv, np.float32) - np.float32(lv[1]))[col]
    return np.where(ok, a - off, a)
