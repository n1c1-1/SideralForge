"""Several images brought onto one of them (the Align images block): each image's own pixels resampled on the reference
image's grid, whatever its size, rotation or scale. It differs from Registration: finished images, possibly stretched and
of different instruments, not the frames of a night.

Both images with a plate solution in their FITS header: through the sky (fusion.reprojection_map). Otherwise through their
stars: the triangle matching of V. Valdes et al. (PASP 107, 1119, 1995) and E. Groth (AJ 91, 1244, 1986), with the triangles
of M. Beroiz, J. Cabral, B. Sanchez, "Astroalign" (Astronomy and Computing 32, 100384, 2020), then the cubic polynomial of
Registration on every star it matched (a similarity alone with too few of them). The positions come from sep, on each
image's own luminance: the stretch of an image does not move its stars.

The result has the reference's size: what overflows is cut, and where the image has no data is marked invalid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree

from ..core.stage import StageError
from .fusion import reprojection_map, warp
from .register import SPARSE_MIN_MATCHES, TRIANGLE_STARS, _fit_from, _matched, _similarity, apply_poly, triangle_similarities

#: the scale two images may differ by (a triangle fitted to a far smaller one maps every star onto one point)
MAX_SCALE = 0.9
#: stars read per image
MAX_STARS = 2500
#: the share of the stars that can be matched (those landing inside the image, of the shorter list) that must be, or the
#: two images are not the same field: a similarity of a very different scale gathers stars by chance
MIN_MATCHED = 0.25


def plate_solution(path: str | Path):
    """The plate solution in a FITS header, or None (any other file, or a FITS without one)."""
    if Path(path).suffix.lower() not in (".fits", ".fit", ".fts"):
        return None
    import warnings

    from astropy.io import fits
    from astropy.wcs import WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            wcs = WCS(fits.getheader(path)).celestial
        except Exception:  # noqa: BLE001 - a header astropy cannot read as a solution: none
            return None
    return wcs if wcs.has_celestial and wcs.wcs.ctype[0].strip() else None


def stars_of(a: np.ndarray) -> np.ndarray:
    """(x, y, flux) of an image's stars (3, H, W), the brightest first."""
    from .astrometry import detect
    g = np.ascontiguousarray(a.mean(0), np.float32)
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        o = detect(g)
    except Exception:  # noqa: BLE001 - sep refuses a flat image: no star in it
        return np.zeros((0, 3))
    cat = np.c_[o["x"], o["y"], o["flux"]]
    return cat[np.argsort(-cat[:, 2], kind="stable")][:MAX_STARS]


def _map_on_grid(fn, shape: tuple[int, int], step: int = 32):
    """(map_x, map_y) of a smooth mapping of the grid's pixels, computed on a coarse grid and interpolated."""
    H, W = shape
    ys = np.linspace(0, H - 1, max(4, int(np.ceil(H / step)) + 1))
    xs = np.linspace(0, W - 1, max(4, int(np.ceil(W / step)) + 1))
    gx, gy = np.meshgrid(xs, ys)
    ix, iy = fn(gx, gy)
    yy, xx = np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64)
    return (RectBivariateSpline(ys, xs, ix, kx=3, ky=3)(yy, xx).astype(np.float32),
            RectBivariateSpline(ys, xs, iy, kx=3, ky=3)(yy, xx).astype(np.float32))


def map_from_stars(ref: np.ndarray, cat: np.ndarray, shape: tuple[int, int], name: str, size: tuple[int, int]):
    """The map of the reference grid's pixels into the image (``size``: its own (H, W)), from their stars; (map, matched
    stars, residual px)."""
    H, W = shape
    if min(len(ref), len(cat)) < SPARSE_MIN_MATCHES:
        raise StageError(f"{name}: too few stars to align it. An image without its stars cannot be aligned by itself: "
                         "align the image that has them, then take them off (Stars/background after this block)")
    a, b = ref[:, :2], cat[:, :2]
    tree = cKDTree(b)
    s, t, _tri = triangle_similarities(a[:TRIANGLE_STARS], b[:TRIANGLE_STARS], max_scale=MAX_SCALE)
    za, zb = a[:, 0] + 1j * a[:, 1], b[:, 0] + 1j * b[:, 1]
    best, top = None, (0, 0.0)
    for s1, t1 in zip(s, t):
        z = s1 * za + t1
        i, _j, d = _matched(np.c_[z.real, z.imag], tree)
        score = (len(i), -float(np.sum(d ** 2)))
        if score > top:
            best, top = (s1, t1), score
    if best is not None:
        z = best[0] * za + best[1]
        inside = int(np.sum((z.real >= 0) & (z.real < size[1]) & (z.imag >= 0) & (z.imag < size[0])))
    if best is None or top[0] < SPARSE_MIN_MATCHES or top[0] < MIN_MATCHED * min(inside, len(cat)):
        raise StageError(f"{name}: its stars match none of the reference's: not the same field")
    s1, t1 = best
    for _ in range(3):
        z = s1 * za + t1
        i, j, d = _matched(np.c_[z.real, z.imag], tree)
        s1, t1 = _similarity(za[i], zb[j])
    z = s1 * za + t1
    fit = _fit_from(ref, cat, tree, W, H, np.c_[z.real, z.imag], 4.0)
    if fit is not None:
        coef, n, rms, _ratio = fit
        return _map_on_grid(lambda x, y: apply_poly(coef, x, y, W, H), shape), n, rms
    i, _j, d = _matched(np.c_[z.real, z.imag], tree)

    def similarity(x, y):
        m = s1 * (x + 1j * y) + t1
        return m.real, m.imag
    return _map_on_grid(similarity, shape), len(i), float(np.sqrt(np.mean(d ** 2)))


def align_onto(a: np.ndarray, path: str | Path, ref: np.ndarray, ref_path: str | Path, log=None):
    """The image ``a`` (3, H, W) on the grid of ``ref``: (the image, valid (H, W))."""
    H, W = ref.shape[-2:]
    name = Path(path).name
    wcs_a, wcs_r = plate_solution(path), plate_solution(ref_path)
    if wcs_a is not None and wcs_r is not None:
        wp = reprojection_map(wcs_r, wcs_a, (H, W))
        if log:
            log(f"{name}: aligned on {Path(ref_path).name} through their plate solutions")
    else:
        wp, n, rms = map_from_stars(stars_of(ref), stars_of(a), (H, W), name, a.shape[-2:])
        if log:
            log(f"{name}: aligned on {Path(ref_path).name} through {n} stars, residual {rms:.2f} px")
    out = np.stack([warp(c, wp) for c in a])
    valid = warp(np.ones(a.shape[-2:], np.float32), wp, nearest=True) > 0.5
    out[:, ~valid] = 0
    return out.astype(np.float32), valid
