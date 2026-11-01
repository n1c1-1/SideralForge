"""Putting renders of different stacks into one frame, each with its own project's files.

A render is a picture of the useful frame (``crop``) of its project's drizzled grid, at any scale. Two projects of the
same field have grids that differ by a few pixels (another stack, another crop), so a render must be mapped with the
alignment or the astrometric solution of the project that made it. judge.py mapped every render through the old
project's alignment: the new render came out shifted. Here:

* ``reference`` frame: the view is the user's reference image (every project aligned it itself: its ``align_ref.json``
  maps a reference pixel to a pixel of its own grid, a cubic polynomial);
* ``sky`` frame (no reference): the view is the first render's own pixels; another render reaches it through the sky:
  its own astrometric solution (wcs.fits) takes a grid pixel to the sky, the first one's brings it back. Two renders
  with the same solution (the same stack) meet at once, without resampling.

Pixel convention: grid and image pixels are indexes with the centre of pixel k at k (cv2.remap's); an image pixel i
of a render with frame (x0, x1) and width W covers the grid span x0 + i * f .. x0 + (i + 1) * f, f = (x1 - x0) / W.

A render larger than the view is first reduced with INTER_AREA (so that all renders are sampled alike, and none is
blurrier for being enlarged); then resampled with Lanczos.
"""
from __future__ import annotations

import filecmp
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from qc_addr import QCError, Source

ASPECT_TOLERANCE = 0.01          # an image whose proportions differ from its frame by more than this is refused
EDGE = 4                         # pixels at the edge of an image where a Lanczos resampling is not trusted
LATTICE = 16                     # the sky frame computes its map every 16 view pixels and interpolates


# ---------------------------------------------------------------------------------------------- images

def load_rgb(path: str | Path) -> np.ndarray:
    """An image as float32 (H, W, 3) in [0, 1]: 8 or 16 bit PNG / JPEG, TIFF (16 bit, float), grey or RGB(A)."""
    p = Path(path)
    if not p.is_file():
        raise QCError(f"no image: {p}")
    if p.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(p)
        if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[2] not in (3, 4):
            a = a.transpose(1, 2, 0)
    else:
        a = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise QCError(f"cannot read the image {p}")
        if a.ndim == 3:
            a = a[:, :, [2, 1, 0]]
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, 2)
    a = a[:, :, :3]
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    return np.clip(a.astype(np.float32), 0.0, 1.0)


def to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def undo_orientation(img: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """The image as captured, from an export turned k quarter turns counter-clockwise after a vertical mirror
    (render/export._orient)."""
    if k % 4:
        img = np.rot90(img, -k, axes=(0, 1))
    if flip:
        img = img[::-1]
    return np.ascontiguousarray(img)


@dataclass
class Geometry:
    """Where an image sits on its grid: ``extent`` (x0, x1, y0, y1) in grid pixels, its size in pixels."""

    extent: tuple[int, int, int, int]
    W: int
    H: int

    @property
    def fx(self) -> float:
        return (self.extent[1] - self.extent[0]) / self.W

    @property
    def fy(self) -> float:
        return (self.extent[3] - self.extent[2]) / self.H

    def to_image(self, gx, gy):
        """Grid index coordinates -> image index coordinates."""
        x0, _, y0, _ = self.extent
        return (gx + 0.5 - x0) / self.fx - 0.5, (gy + 0.5 - y0) / self.fy - 0.5

    def to_grid(self, ix, iy):
        x0, _, y0, _ = self.extent
        return x0 + (ix + 0.5) * self.fx - 0.5, y0 + (iy + 0.5) * self.fy - 0.5

    def resized(self, W: int, H: int) -> Geometry:
        return Geometry(self.extent, W, H)


def geometry_of(src: Source, shape: tuple[int, int]) -> Geometry:
    """The extent a render's image covers: the useful frame of its preparation, or the whole grid (the image of a
    layer exported uncropped). The proportions decide; neither fitting is an error: the image is not of this
    project's frame, or it was turned (give rot90 / flip)."""
    H, W = shape
    x0, x1, y0, y1 = src.crop
    ratio = lambda a, b: abs((W / H) / (a / b) - 1)      # noqa: E731
    if ratio(x1 - x0, y1 - y0) <= ASPECT_TOLERANCE:
        return Geometry((x0, x1, y0, y1), W, H)
    gw, gh = int(src.prep.get("width") or 0), int(src.prep.get("height") or 0)
    if gw and gh and ratio(gw, gh) <= ASPECT_TOLERANCE:
        return Geometry((0, gw, 0, gh), W, H)
    raise QCError(f"{src.address}: the image is {W} x {H} ({W / H:.3f}) but the useful frame is {x1 - x0} x {y1 - y0} "
                  f"({(x1 - x0) / (y1 - y0):.3f}): not the render of this workflow, or it is turned (rot90 / flip)")


@dataclass
class Render:
    src: Source
    img: np.ndarray            # (H, W, 3) float32, as captured
    geom: Geometry


def load_render(src: Source) -> Render:
    if src.image is None:
        raise QCError(f"{src.address}: no image to judge")
    img = undo_orientation(load_rgb(src.image), src.rot90, src.flip)
    return Render(src, img, geometry_of(src, img.shape[:2]))


# ---------------------------------------------------------------------------------------------- polynomial map

def _powers(t: np.ndarray) -> np.ndarray:
    return np.stack([t ** i for i in range(4)], 1)


def _cmat(coefs) -> np.ndarray:
    """The cubic terms of align_ref.json (a**i * b**j, i + j < 4, in the order of render/export.remap_to_reference) as
    a matrix indexed [j, i]."""
    m = np.zeros((4, 4))
    k = 0
    for i in range(4):
        for j in range(4 - i):
            m[j, i] = coefs[k]
            k += 1
    return m


def poly_dense(info: dict) -> tuple[np.ndarray, np.ndarray]:
    """The grid pixel (x, y) of each pixel of the reference, (Hr, Wr) float64 arrays."""
    Hr, Wr = info["ref_shape"]
    a = _powers(np.arange(Wr) / Wr * 2 - 1)
    b = _powers(np.arange(Hr) / Hr * 2 - 1)
    return b @ _cmat(info["cx"]) @ a.T, b @ _cmat(info["cy"]) @ a.T


def poly_points(info: dict, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Hr, Wr = info["ref_shape"]
    a, b = _powers(np.asarray(u, float) / Wr * 2 - 1), _powers(np.asarray(v, float) / Hr * 2 - 1)
    cx, cy = _cmat(info["cx"]), _cmat(info["cy"])
    return np.einsum("nj,ji,ni->n", b, cx, a), np.einsum("nj,ji,ni->n", b, cy, a)


def poly_inverse(info: dict, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The reference pixel (u, v) that maps to the grid pixel (x, y): Newton's iterations from the best affine map."""
    Hr, Wr = info["ref_shape"]
    gu, gv = np.meshgrid(np.linspace(0, Wr - 1, 9), np.linspace(0, Hr - 1, 9))
    gx, gy = poly_points(info, gu.ravel(), gv.ravel())
    A = np.c_[gx, gy, np.ones_like(gx)]
    cu, *_ = np.linalg.lstsq(A, gu.ravel(), rcond=None)
    cv_, *_ = np.linalg.lstsq(A, gv.ravel(), rcond=None)
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    u, v = np.c_[x, y, np.ones_like(x)] @ cu, np.c_[x, y, np.ones_like(x)] @ cv_
    h = 0.5
    for _ in range(8):
        fx, fy = poly_points(info, u, v)
        fxu, fyu = poly_points(info, u + h, v)
        fxv, fyv = poly_points(info, u, v + h)
        j11, j21, j12, j22 = (fxu - fx) / h, (fyu - fy) / h, (fxv - fx) / h, (fyv - fy) / h
        det = j11 * j22 - j12 * j21
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        du = (j22 * (x - fx) - j12 * (y - fy)) / det
        dv = (-j21 * (x - fx) + j11 * (y - fy)) / det
        u, v = u + du, v + dv
    return u, v


# ---------------------------------------------------------------------------------------------- lattice

def _dense(M: np.ndarray, W: int, H: int, step: int) -> np.ndarray:
    """A map computed on a lattice (rows v = k * step, columns u = k * step) as a dense (H, W) float32 array, bilinear."""
    u = np.arange(W) / step
    iu = np.minimum(u.astype(int), M.shape[1] - 2)
    fu = (u - iu)[None, :]
    Mu = M[:, iu] * (1 - fu) + M[:, iu + 1] * fu
    v = np.arange(H) / step
    iv = np.minimum(v.astype(int), M.shape[0] - 2)
    fv = (v - iv)[:, None]
    return (Mu[iv] * (1 - fv) + Mu[iv + 1] * fv).astype(np.float32)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def _wcs(path: Path):
    import warnings

    from astropy.io import fits
    from astropy.wcs import WCS, FITSFixedWarning
    with fits.open(path) as hdul, warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)         # a solution kept in a header-only file
        return WCS(hdul[0].header)


# ---------------------------------------------------------------------------------------------- the view

class View:
    """The frame the renders of a comparison are shown in. ``mode``: auto (the reference frame if every render has its
    project's alignment, else the sky), reference, or sky. In the sky frame the view is the image of render
    ``anchor``; in the reference frame, the reference image."""

    def __init__(self, renders: list[Render], mode: str = "auto", anchor: int = 0):
        if not renders:
            raise QCError("nothing to compare")
        self.renders = renders
        have_ref = all(r.src.align for r in renders)
        have_sky = all(r.src.wcs for r in renders)
        if mode == "auto":
            mode = "reference" if have_ref else "sky" if have_sky else ""
        if mode == "reference" and not have_ref:
            raise QCError("a render has no reference alignment: " + ", ".join(
                r.src.address for r in renders if not r.src.align))
        if mode == "sky" and not have_sky:
            raise QCError("a render has no astrometric solution: " + ", ".join(
                r.src.address for r in renders if not r.src.wcs))
        if not mode:
            raise QCError("no common frame: these renders have neither a reference alignment nor an astrometric "
                          "solution of their own project")
        self.kind, self.anchor = mode, anchor
        self._maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if mode == "reference":
            self.infos = [json.loads(r.src.align.read_text(encoding="utf-8")) for r in renders]
            shapes = {tuple(i["ref_shape"]) for i in self.infos}
            if len(shapes) > 1:
                raise QCError(f"the projects aligned references of different sizes: {sorted(shapes)}")
            self.H, self.W = self.infos[0]["ref_shape"]
        else:
            a = renders[anchor]
            self.H, self.W = a.img.shape[:2]
            self._wcs = {}

    # ------------------------------------------------------------------ view pixel -> grid pixel of render k
    def _wcs_of(self, k: int):
        if k not in self._wcs:
            self._wcs[k] = _wcs(self.renders[k].src.wcs)
        return self._wcs[k]

    def _same_sky(self, k: int) -> bool:
        return k == self.anchor or _same_file(self.renders[k].src.wcs, self.renders[self.anchor].src.wcs)

    def grid_of_view(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """For each view pixel, the grid pixel (x, y) of render k that lies there; float32 (H, W)."""
        if k in self._maps:
            return self._maps[k]
        if self.kind == "reference":
            gx, gy = (m.astype(np.float32) for m in poly_dense(self.infos[k]))
        else:
            ga = self.renders[self.anchor].geom
            if self._same_sky(k):
                ii, jj = np.meshgrid(np.arange(self.W, dtype=np.float64), np.arange(self.H, dtype=np.float64))
                gx, gy = ga.to_grid(ii, jj)                      # float64: the same stack meets without resampling
            else:
                wa, wk = self._wcs_of(self.anchor), self._wcs_of(k)
                us = np.arange(0, self.W + LATTICE, LATTICE, dtype=np.float64)
                vs = np.arange(0, self.H + LATTICE, LATTICE, dtype=np.float64)
                U, V = np.meshgrid(us, vs)
                ax, ay = ga.to_grid(U, V)
                world = wa.all_pix2world(np.c_[ax.ravel(), ay.ravel()], 0)
                px = wk.all_world2pix(world, 0)
                gx = _dense(px[:, 0].reshape(U.shape), self.W, self.H, LATTICE)
                gy = _dense(px[:, 1].reshape(U.shape), self.W, self.H, LATTICE)
        self._maps[k] = (gx, gy)
        return gx, gy

    def view_of_grid(self, k: int, x, y) -> tuple[np.ndarray, np.ndarray]:
        """The view pixel (u, v) of grid pixels (x, y) of render k (stars of its own list)."""
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        if self.kind == "reference":
            return poly_inverse(self.infos[k], x, y)
        if not self._same_sky(k):
            world = self._wcs_of(k).all_pix2world(np.c_[x, y], 0)
            pa = self._wcs_of(self.anchor).all_world2pix(world, 0)
            x, y = pa[:, 0], pa[:, 1]
        return self.renders[self.anchor].geom.to_image(x, y)

    # ------------------------------------------------------------------ pictures
    def remap(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Render k in the view: float32 (H, W, 3), and the mask of the pixels it covers (away from its edges)."""
        r = self.renders[k]
        gx, gy = self.grid_of_view(k)
        img, geom = r.img, r.geom
        ix, iy = geom.to_image(gx, gy)
        c = (slice(self.H // 4, 3 * self.H // 4), slice(self.W // 4, 3 * self.W // 4))
        per_view = float(np.median(np.abs(np.diff(ix[c], axis=1))))       # image pixels per view pixel
        if per_view > 1.5:                                               # reduce first: all renders sampled alike
            img = cv2.resize(img, (max(1, round(geom.W / per_view)), max(1, round(geom.H / per_view))),
                             interpolation=cv2.INTER_AREA)
            geom = geom.resized(img.shape[1], img.shape[0])
            ix, iy = geom.to_image(gx, gy)
        Hi, Wi = img.shape[:2]
        valid = (ix >= EDGE) & (ix <= Wi - 1 - EDGE) & (iy >= EDGE) & (iy <= Hi - 1 - EDGE)
        if img.shape[:2] == (self.H, self.W):
            jj, ii = np.mgrid[0:self.H, 0:self.W]
            if np.abs(ix - ii).max() < 1e-2 and np.abs(iy - jj).max() < 1e-2:
                return img.copy(), valid                                   # the same pixels: no resampling at all
        ix = np.nan_to_num(ix, nan=-1e6).astype(np.float32)
        iy = np.nan_to_num(iy, nan=-1e6).astype(np.float32)
        out = np.stack([cv2.remap(np.ascontiguousarray(img[:, :, c_]), ix, iy, cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT) for c_ in range(3)], -1)
        return np.clip(out, 0, 1), valid

    def reference_image(self) -> np.ndarray | None:
        """The user's reference image in the view (reference frame only), when its file is there."""
        if self.kind != "reference":
            return None
        ref = next((r.src.reference for r in self.renders if r.src.reference), None)
        if ref is None:
            return None
        img = load_rgb(ref)
        return img if img.shape[:2] == (self.H, self.W) else None

    def summary(self) -> dict:
        return {"frame": self.kind, "size": [self.W, self.H], "anchor": self.anchor if self.kind == "sky" else None}
