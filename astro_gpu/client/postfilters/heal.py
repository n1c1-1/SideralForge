"""Spot healing: a dust shadow, a satellite trail end, a blemish replaced by a patch of its neighbourhood, seamlessly.

A spot is ``[x, y, r, dx, dy]`` in full-resolution canvas pixels: the disc of radius ``r`` around (x, y) is replaced by
the disc at (x + dx, y + dy), its source, raccorded to the edge of the spot by Poisson editing (P. Perez, M. Gangnet
and A. Blake, "Poisson Image Editing", SIGGRAPH 2003; the principle of Photoshop's Healing Brush, T. Georgiev,
"Photoshop Healing Brush: a Tool for Seamless Cloning", ECCV 2004 workshop): the result keeps the source's detail (its
stars, its grain) and takes the level and colour of the spot's surroundings, so that a darker or lighter source leaves
no edge. A soft rim of ``FEATHER`` of the radius blends it into the image.

The source is chosen when the spot is made (``find_source``), as the spot removal of photo editors does, and stored
with it, so that the preview and the export are the same: around the spot, at 2.2 to 3.2 radii, the disc whose rim
looks most like the spot's rim (the patch search of PatchMatch, C. Barnes et al. 2009, on a ring of candidates), a
source holding a star brighter than the spot's surroundings penalised; the user may move it.
"""
from __future__ import annotations

import math

import numpy as np

#: the soft rim of a spot, as a share of its radius
FEATHER = 0.3
#: the candidate sources: distances in radii, and how many directions
SOURCE_DISTANCES, SOURCE_DIRECTIONS = (2.2, 2.7, 3.2), 24


def margin(spots) -> float:
    """Full-resolution pixels a region needs around itself so that every spot that touches it finds its source."""
    return max((r * (1 + FEATHER) + math.hypot(dx, dy) for _x, _y, r, dx, dy in spots), default=0.0)


def _disc(h: int, w: int, cx: float, cy: float, r: float):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.hypot(xx - cx, yy - cy)


def _laplace_fill(boundary_vals: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """The harmonic function on ``inside`` (bool, h x w) whose values on the pixels around it are those of
    ``boundary_vals`` (h x w x c): the membrane of Poisson cloning (Delta h = 0 inside, h = the boundary outside)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu
    h, w = inside.shape
    idx = -np.ones((h, w), np.int64)
    ys, xs = np.nonzero(inside)
    n = len(ys)
    idx[ys, xs] = np.arange(n)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    rhs = np.zeros((n, boundary_vals.shape[2]), np.float64)
    for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = np.clip(ys + oy, 0, h - 1), np.clip(xs + ox, 0, w - 1)
        j = idx[ny, nx]
        inner = j >= 0
        rows.append(np.arange(n)[inner])
        cols.append(j[inner])
        vals.append(np.full(int(inner.sum()), -1.0))
        rhs[~inner] += boundary_vals[ny[~inner], nx[~inner]]
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsc()
    sol = splu(A).solve(rhs)
    out = np.zeros_like(boundary_vals, dtype=np.float64)
    out[ys, xs] = sol
    return out


def heal_one(img: np.ndarray, x: float, y: float, r: float, dx: float, dy: float) -> None:
    """One spot, in place, on ``img`` (h x w x 3 float), coordinates in its pixels. A spot on the image's edge is
    healed on its part inside the image (and inside the source's reach); nothing when that is empty."""
    H, W = img.shape[:2]
    R = r * (1 + FEATHER)
    ix, iy = int(round(dx)), int(round(dy))
    x0, y0 = max(0, int(math.floor(x - R - 2)), -ix), max(0, int(math.floor(y - R - 2)), -iy)
    x1, y1 = min(W, int(math.ceil(x + R + 3)), W - ix), min(H, int(math.ceil(y + R + 3)), H - iy)
    sx0, sy0, sx1, sy1 = x0 + ix, y0 + iy, x1 + ix, y1 + iy
    if r < 1 or x1 - x0 < 3 or y1 - y0 < 3:
        return
    tgt = img[y0:y1, x0:x1].astype(np.float64)
    src = img[sy0:sy1, sx0:sx1].astype(np.float64)
    d = _disc(y1 - y0, x1 - x0, x - x0, y - y0, r)
    inside = d < R
    # Poisson cloning: the source plus the membrane that takes the difference (spot - source) on the rim to the inside
    diff = np.where(inside[..., None], 0.0, tgt - src)
    healed = src + _laplace_fill(diff, inside)
    # the soft rim: fully healed within r, back to the image at r (1 + FEATHER)
    t = np.clip((R - d) / max(R - r, 1e-6), 0, 1)
    a = (t * t * (3 - 2 * t))[..., None]
    img[y0:y1, x0:x1] = (a * healed + (1 - a) * tgt).astype(img.dtype)


def heal(x: np.ndarray, spots, scale: float, origin: tuple[int, int]) -> np.ndarray:
    """``x`` (h x w x 3) is a region of the canvas at ``scale`` (its pixels per full-resolution pixel) whose top-left
    corner is ``origin`` (its own pixels); every spot that touches it is healed, in the order made."""
    if not spots:
        return x
    out = np.array(x, np.float32, copy=True)
    ox, oy = origin
    H, W = out.shape[:2]
    for sx, sy, r, dx, dy in spots:
        cx, cy, rr = sx * scale - ox, sy * scale - oy, r * scale
        R = rr * (1 + FEATHER)
        if cx + R < 0 or cy + R < 0 or cx - R > W or cy - R > H:
            continue
        heal_one(out, cx, cy, rr, dx * scale, dy * scale)
    return out


def find_source(img: np.ndarray, x: float, y: float, r: float) -> tuple[float, float]:
    """(dx, dy) of the best source for a spot at (x, y), radius r, on ``img`` (h x w x 3, any scale, the same as the
    coordinates): the candidate disc whose rim matches the spot's rim best (mean squared difference of the rims at
    r to 1.4 r), a candidate holding pixels much brighter than the spot's rim (a star) penalised. (0, 0) when no
    candidate fits in the image. On the image's edge, the part of the rims inside the image; the source itself must be
    inside it."""
    ring_lo, ring_hi = r, r * 1.4
    S = int(math.ceil(ring_hi)) + 1
    a = np.pad(img.astype(np.float32), ((S, S), (S, S), (0, 0)), constant_values=np.nan)    # off the image: NaN
    H, W = a.shape[:2]
    yy, xx = np.mgrid[-S:S + 1, -S:S + 1]
    rad = np.hypot(xx, yy)
    ring, disc = (rad >= ring_lo) & (rad <= ring_hi), rad < r

    def patch(cx, cy):
        ix, iy = int(round(cx)) + S, int(round(cy)) + S
        if ix - S < 0 or iy - S < 0 or ix + S + 1 > W or iy + S + 1 > H:
            return None
        return a[iy - S:iy + S + 1, ix - S:ix + S + 1]
    t = patch(x, y)
    if t is None:
        return 0.0, 0.0
    t_ring = t[ring]
    seen = np.isfinite(t_ring).all(axis=1)                  # the spot's rim inside the image
    if seen.sum() < 8:
        return 0.0, 0.0
    level = np.median(t_ring[seen], axis=0)
    spread = float(np.median(np.abs(t_ring[seen] - level))) * 1.4826 + 1e-6
    best, best_d = None, (0.0, 0.0)
    for k in SOURCE_DISTANCES:
        for j in range(SOURCE_DIRECTIONS):
            ang = 2 * math.pi * j / SOURCE_DIRECTIONS
            dx, dy = k * r * math.cos(ang), k * r * math.sin(ang)
            c = patch(x + dx, y + dy)
            if c is None or not np.isfinite(c[disc]).all():     # the source must be inside the image
                continue
            cr = c[ring]
            both = seen & np.isfinite(cr).all(axis=1)
            if both.sum() < 8:
                continue
            score = float(np.mean((cr[both] - t_ring[both]) ** 2))
            star = np.clip(c[disc] - level - 4 * spread, 0, None)
            score += float(np.mean(star ** 2)) * 4
            if best is None or score < best:
                best, best_d = score, (dx, dy)
    return best_d
