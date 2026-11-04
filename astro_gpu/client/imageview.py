"""The image viewer, the same everywhere (Frames, Calibration, Stacking, Denoise, Gradient, Output).

One widget, ``ImageView``, shows an image of any size (up to 6384 x 4272 RGB float, mono, or a raw Bayer mosaic) with:

* zoom (wheel, Fit, 1:1, presets), pan, a minimap, and a pixel readout (x, y, the values in ADU or 0-1, RA / Dec
  when the image has an astrometric solution);
* a display stretch as in Siril: automatic (channels linked or not), linear, midtones (MTF), asinh or histogram
  equalisation, black / mid / white handles on a live histogram, per channel or linked, "Neutral sky" for an image
  still in the camera's colours, Reset. The stretch is display only: the data are never changed;
* compare with the previous steps of the workflow: the caller gives named images, the view a selector; split slider
  (before / after) or side by side, with the same stretch on both if wanted. The compared image is drawn on the grid
  of the one shown, so that a star sits at the same place in both (``alignment``: by the placements the caller gives,
  ``ViewImage.place`` - drizzle scale, crop, half size -, else by the astrometric solutions), resampled for display
  only, and a line above the images says how;
* a sequence of frames: previous / next (arrow keys), a frame number, a film strip, an info band (the caller's
  numbers: time, exposure, ISO / gain, temperature, FWHM, roundness, transparency, kept or excluded) and Exclude /
  Include; frames load in worker threads, the neighbours ahead of time;
* Export… (FITS 32-bit float with the WCS and a header; TIFF 16-bit, PNG, JPEG: the data or the stretched view),
  Send to Post (a signal) and Copy image (the picture as shown, to paste elsewhere).

How it stays fast on the CPU: an image is prepared once in a worker thread (``prepare``: a pyramid of halves and a
histogram of a sample, about 0.15 s for the 6384 x 4272 stack). The view draws from the pyramid level nearest to the
screen's resolution; full resolution is touched only for the tiles in view at about 1:1 and above. Each 512 x 512
tile is quantized once to 16 bits and shown through the look-up tables of ``stretch.py``: moving a handle rebuilds
three tables and re-indexes the tiles in view, in parallel threads. The visible part is resampled to the screen's
pixels in one OpenCV call.

The views call (see ``ImageView`` for the whole list)::

    v = ImageView(post=True)                                  # Send to Post shown
    v.set_image(ViewImage(arr, "Stack", key=("stack", k), wcs=wcs, camera=True))
    v.set_compare_targets([ViewImage(arr0, "Calibrated"), ...])        # the previous steps
    v.set_sequence([ViewImage(path, name, info={...}, thumb=fn), ...], index=0)
    v.excludeToggled.connect(...); v.sendToPost.connect(...)
    SyncGroup([v1, v2], stretch=True)                         # two viewers share zoom, pan (and the stretch)

Classes: ``ImageView`` (the widget) draws on an ``ImageCanvas``, with a ``StretchPanel`` (its ``HistogramView``), an
``InfoBand`` and a ``FilmStrip``; a ``Loader`` turns each ``ViewImage`` into a ``Prepared`` image in worker threads and a
``Display`` shows it through a stretch; ``ImageExportDialog`` is Export…; ``SyncGroup`` links viewers. Post has its own
viewer (``viewer.Viewer``).
"""
from __future__ import annotations

import heapq
import itertools
import math
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QStandardPaths, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import stretch as sm
from .i18n import _, language, number
from .widgets import ACCENT, BAD, OK, WARN, ElidedLabel, label

TILE = 512
CANVAS_BG = QColor("#0f1115")
CANVAS_RGB = (15, 17, 21)               # the same, for a picture drawn where an aligned image has no pixel
CHANNEL_COLORS = ("#e06c6c", "#5fbf7f", "#5ea1ff")
MONO_COLOR = "#c8ccd4"
ZOOM_STEP = 1.25
ZOOM_PRESETS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MONO_FONT = "Consolas"
QSS = f"""
QFrame#ivbar {{ background: #1c1f26; border-bottom: 1px solid #2a2f38; }}
QFrame#ivread {{ background: #181b21; border-top: 1px solid #2a2f38; }}
QFrame#ivpanel {{ background: #1c1f26; border-left: 1px solid #2a2f38; }}
QFrame#ivseq {{ background: #1a1d23; border-top: 1px solid #2a2f38; }}
QFrame#ivsep {{ background: #353b47; }}
QFrame#ivalign {{ background: #1a1d23; border-bottom: 1px solid #2a2f38; }}
QLabel#ivalign {{ color: #9fc2ff; font-size: 8.5pt; }}
QLabel#ivcap {{ color: #7d8594; font-size: 8pt; }}
QLabel#ivval {{ color: #e8ebf0; font-size: 10pt; }}
QLabel#ivname {{ color: #ffffff; font-size: 10.5pt; font-weight: 600; }}
QLabel#ivread {{ color: #c9ced8; font-family: '{MONO_FONT}'; font-size: 9pt; }}
QPushButton#ivtool {{ padding: 3px 9px; }}
QPushButton#ivseg {{ padding: 3px 9px; border-radius: 0; }}
QPushButton#ivseg:checked {{ background: #2b3446; border-color: {ACCENT}; color: #ffffff; }}
QPushButton#ivstate {{ padding: 3px 12px; font-weight: 600; }}
QDoubleSpinBox#ivfield {{ min-width: 64px; }}
"""


# ---------------------------------------------------------------------------------------------- what the caller gives
@dataclass(eq=False)
class ViewImage:
    """One image for the viewer, as the caller describes it. Nothing is read until the viewer needs it (in a worker
    thread).

    ``source``: an array — (C, H, W) as the engine stores it, (H, W, C), (H, W) mono, or (H, W) Bayer mosaic with
    ``cfa`` — or a callable returning one, or a file path (.npy, .fits, .tif, .png, .jpg, or a camera raw).
    ``key``: the image's identity (the cache, Send to Post, the compare selector); default: the path, or the array.
    ``wcs``: an astropy WCS, a FITS header or the path of a wcs.fits: RA / Dec in the readout, and the FITS export.
    ``units``: "adu", "norm" (0-1), "count" (a count per pixel) or "" (a number without a unit); ``white``: the
    saturation level in those units (for the readout in the other unit; default 65535 ADU or 1). ``camera``: still
    in the camera's colours (Neutral sky on by default, unless ``neutral`` says otherwise); None: guessed from the sky
    (channel medians apart by more than 15 % and many noise sigmas: a camera's cast). ``neutral``: Neutral sky at
    first (a display option the user can tick or untick); None: as ``camera`` says (a mosaic: on).
    ``info``: a frame of a sequence, as the caller measured it: ``time`` (ISO), ``exposure`` (s), ``iso`` or
    ``gain``, ``temperature`` (°C), ``fwhm`` (px), ``fwhm_arcsec``, ``roundness`` (0-1), ``transparency`` (1 = 100 %),
    ``weight``, ``excluded`` (bool), ``reason`` (why the engine leaves it out), ``extra`` [(caption, text)].
    ``thumb``: a small (h, w, 3) uint8 picture or a callable returning one (the film strip). ``header``: FITS cards
    added on export."""

    source: Any
    name: str = ""
    key: Any = None
    wcs: Any = None
    units: str = "adu"
    white: float | None = None
    cfa: str = ""
    camera: bool | None = None
    info: dict = field(default_factory=dict)
    thumb: Any = None
    header: dict = field(default_factory=dict)
    #: already a picture for the eye (a final render, an engine preview): shown as it is, 0 to ``white`` linear, and
    #: read as levels without a unit
    stretched: bool = False
    #: Neutral sky at first: None follows ``camera``; False shows a camera-colour image as it is (its cast and all)
    neutral: bool | None = None
    #: what the image is, for its automatic stretch (``stretch.auto_for``): "" or "lights" (a light, a stack), "darks",
    #: "bias", "flat_darks", "flats", or "model" (a sky model, a gradient surface, a star flat). "Same stretch for all
    #: frames" keeps one stretch per kind, never across kinds.
    kind: str = ""
    #: where its pixels sit on a common grid, for the compare (``Placement``): two images placed on the same grid are
    #: drawn on each other by it; else by their astrometric solutions (``wcs``). A callable (w, h) -> Placement or None
    #: is called in the worker thread with the image's size.
    place: Any = None

    def identity(self):
        if self.key is not None:
            return self.key
        if isinstance(self.source, str | os.PathLike):
            return ("file", os.path.normcase(os.path.abspath(os.fspath(self.source))))
        return ("object", id(self.source))

    def white_level(self) -> float:
        if self.white:
            return float(self.white)
        return 1.0 if self.units == "norm" else 65535.0


@dataclass(frozen=True)
class Placement:
    """Where an image's pixels sit on a common grid (``ViewImage.place``). ``frame`` names the grid: two images placed
    on the same one are put on each other by their placements. ``m`` maps the image's coordinates to the grid's,
    X = m[0] x + m[1] y + m[2], Y = m[3] x + m[4] y + m[5], in pixels with the pixel edges on whole numbers (pixel i
    covers [i, i + 1)): a stack of drizzle scale s on the grid of its reference frame is (1/s, 0, 0, 0, 1/s, 0); a
    final image cut at (ox, oy) of that stack and shown at half size is, on the stack, (2, 0, ox, 0, 2, oy)."""

    frame: Any
    m: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    @classmethod
    def scaled(cls, frame, f: float = 1.0, ox: float = 0.0, oy: float = 0.0, fy: float | None = None) -> Placement:
        """X = ox + f x, Y = oy + fy y (fy: f by default)."""
        return cls(frame, (float(f), 0.0, float(ox), 0.0, float(f if fy is None else fy), float(oy)))

    def matrix(self) -> np.ndarray:
        a, b, c, d, e, f = self.m
        return np.array([[a, b, c], [d, e, f], [0.0, 0.0, 1.0]])

    def then(self, outer: Placement) -> Placement:
        """This placement, then ``outer`` (which places this one's grid on its own frame)."""
        return Placement(outer.frame, tuple(float(v) for v in (outer.matrix() @ self.matrix())[:2].ravel()))


@dataclass
class Alignment:
    """How a compared image is drawn on the grid of the image shown (display only; ``alignment``). ``T`` (2 x 3) maps a
    point of the shown image to the compared one's (pixel edges on whole numbers). ``how``: "grid" (both placed on one
    grid by the geometry of the results: drizzle scale, crop, half size), "sky" (by their astrometric solutions) or
    "width" (no common grid known: matched by their widths). ``residual``: for "sky", the largest distance, in pixels
    of the compared image, between the solutions and the affine map drawn."""

    T: np.ndarray
    how: str
    residual: float = 0.0
    #: for "sky": what the solutions add to the affine map (their distortions differ), sampled on an n x n grid of the
    #: shown image (x, y = linspace(0.5, size - 0.5, n)): (size, dx, dy), drawn by ``Display`` and read by ``apply``
    field: tuple | None = None

    def apply(self, x, y):
        t = self.T
        bx, by = t[0, 0] * x + t[0, 1] * y + t[0, 2], t[1, 0] * x + t[1, 1] * y + t[1, 2]
        if self.field is not None:
            rx, ry = self.field_at(np.asarray(x, float), np.asarray(y, float))
            bx, by = bx + rx, by + ry
            if np.ndim(bx) == 0:
                bx, by = float(bx), float(by)
        return bx, by

    def field_at(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The correction of the affine map at points of the shown image (bilinear between the grid's points)."""
        (w, h), fx, fy = self.field
        n = fx.shape[0]
        gx = np.clip((x - 0.5) / max(w - 1.0, 1e-9) * (n - 1), 0, n - 1)
        gy = np.clip((y - 0.5) / max(h - 1.0, 1e-9) * (n - 1), 0, n - 1)
        i0, j0 = np.minimum(np.floor(gy).astype(int), n - 2), np.minimum(np.floor(gx).astype(int), n - 2)
        ty, tx = gy - i0, gx - j0

        def lerp(f):
            return ((1 - ty) * ((1 - tx) * f[i0, j0] + tx * f[i0, j0 + 1])
                    + ty * ((1 - tx) * f[i0 + 1, j0] + tx * f[i0 + 1, j0 + 1]))
        return lerp(fx), lerp(fy)

    @property
    def scale(self) -> float:
        """Pixels of the compared image per pixel of the shown one."""
        return float(math.sqrt(abs(np.linalg.det(self.T[:, :2])))) or 1.0

    @property
    def angle(self) -> float:
        """The turn (degrees) of the compared image on the shown one."""
        return float(math.degrees(math.atan2(self.T[1, 0], self.T[0, 0])))

    def origin(self) -> tuple[float, float]:
        """Where the compared image's top-left corner falls on the shown one (its pixels)."""
        full = np.vstack([self.T, [0.0, 0.0, 1.0]])
        x, y, _w = np.linalg.solve(full, [0.0, 0.0, 1.0])
        return float(x), float(y)


def _identity(T: np.ndarray, tol: float = 1e-3) -> bool:
    return bool(np.allclose(T[:, :2], np.eye(2), atol=tol) and np.allclose(T[:, 2], 0.0, atol=0.02))


SKY_GRID = 17                           # the points of the shown image the solutions are read at (n x n)


def _sky_map(a: Prepared, b: Prepared) -> tuple[np.ndarray, float, tuple | None] | None:
    """The map from ``a`` to ``b`` their astrometric solutions give: the affine part (fitted on n x n points of ``a``),
    its largest error, and what the solutions add to it on that grid (their distortions differ: two programs' solutions
    of one field differ by a few pixels in the corners), None when under a tenth of a pixel. None when a solution cannot
    answer (no inverse, nothing in common)."""
    if not hasattr(b.wcs, "world_to_pixel_values") or not hasattr(a.wcs, "pixel_to_world_values"):
        return None
    n = SKY_GRID
    xs = np.linspace(0.5, a.width - 0.5, n)
    ys = np.linspace(0.5, a.height - 0.5, n)
    X, Y = (g.ravel() for g in np.meshgrid(xs, ys))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ra, dec = a.wcs.pixel_to_world_values(X - 0.5, Y - 0.5)
            bx, by = b.wcs.world_to_pixel_values(np.asarray(ra, float), np.asarray(dec, float))
        except Exception:  # noqa: BLE001 - a solution that cannot answer: no alignment by the sky
            return None
    bx, by = np.asarray(bx, float) + 0.5, np.asarray(by, float) + 0.5
    ok = np.isfinite(bx) & np.isfinite(by)
    if ok.sum() < 6:
        return None
    A = np.stack([X[ok], Y[ok], np.ones(int(ok.sum()))], 1)
    cx = np.linalg.lstsq(A, bx[ok], rcond=None)[0]
    cy = np.linalg.lstsq(A, by[ok], rcond=None)[0]
    res = float(np.hypot(A @ cx - bx[ok], A @ cy - by[ok]).max())
    field = None
    if res > 0.1 and ok.all():
        full = np.stack([X, Y, np.ones(X.size)], 1)
        field = ((float(a.width), float(a.height)), (bx - full @ cx).reshape(n, n).astype(np.float32),
                 (by - full @ cy).reshape(n, n).astype(np.float32))
    return np.array([cx, cy]), res, field


def alignment(a: Prepared, b: Prepared) -> Alignment | None:
    """How ``b`` (the compared image) is drawn on ``a`` (the one shown) so that a star sits at the same place in both:
    by their placements when both are on one grid (the geometry the results give: drizzle scale, crop, half size),
    else by their astrometric solutions when both have one, else matched by their widths (as the viewer always did).
    None: nothing to do (the same grid, the same size)."""
    pa, pb = a.place, b.place
    T, how, res, field = None, "", 0.0, None
    if pa is not None and pb is not None and pa.frame == pb.frame:
        T, how = (np.linalg.inv(pb.matrix()) @ pa.matrix())[:2], "grid"
    elif a.wcs is not None and b.wcs is not None:
        got = _sky_map(a, b)
        if got is not None:
            (T, res, field), how = got, "sky"
    if T is None:
        if a.size() == b.size():
            return None
        r = b.width / max(a.width, 1)
        T, how = np.array([[r, 0.0, 0.0], [0.0, r, 0.0]]), "width"
    if how != "width" and a.size() == b.size() and _identity(T) and res < 0.1:
        return None
    return Alignment(np.asarray(T, float), how, res, field)


RAW_SUFFIXES = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}


def read_file(path: str | os.PathLike) -> tuple[np.ndarray, str, Any]:
    """(array, Bayer pattern or "", WCS or None) of an image file (worker thread)."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".npy":
        return np.load(p), "", None
    if suf in (".fits", ".fit", ".fts"):
        import warnings

        from astropy.io import fits
        with fits.open(p, memmap=False) as hl:
            hdu = next((h for h in hl if getattr(h, "data", None) is not None), hl[0])
            data, hdr = np.asarray(hdu.data), hdu.header
        cfa = str(hdr.get("BAYERPAT", "")).strip().upper() if data.ndim == 2 else ""
        wcs = None
        if "CTYPE1" in hdr:
            from astropy.wcs import WCS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    wcs = WCS(hdr, naxis=2)
                except Exception:  # noqa: BLE001 - an unreadable solution: no sky readout
                    wcs = None
        return data, cfa, wcs
    if suf in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(p), "", None
    if suf in (".png", ".jpg", ".jpeg", ".bmp"):
        if suf == ".png":
            import cv2
            a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if a is not None and a.ndim == 3:
                a = a[:, :, 2::-1] if a.shape[2] >= 3 else a
            return a, "", None
        from PIL import Image
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB")), "", None
    if suf in RAW_SUFFIXES:
        from ..io.frames import read_cfa, read_info
        info = read_info(p)
        return read_cfa(p), info.cfa or "", None
    raise ValueError(f"cannot read {p.name}: unknown format")


def _wcs_of(w):
    """An astropy WCS from what the caller gave (a WCS, a header, a path, or a callable returning one of them: it is
    called in the worker thread), or None."""
    if callable(w) and not hasattr(w, "pixel_to_world_values"):
        try:
            w = w()
        except Exception:  # noqa: BLE001 - no solution: no sky readout
            return None
    if w is None:
        return None
    import warnings

    from astropy.wcs import WCS
    if isinstance(w, WCS) or hasattr(w, "pixel_to_world_values"):
        return w                                    # (an object with pixel_to_world_values only: the readout's)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if isinstance(w, str | os.PathLike):
                from astropy.io import fits
                return WCS(fits.getheader(w), naxis=2)
            return WCS(w, naxis=2)
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------------------------- a prepared image
_UID = itertools.count(1)


class _Lru:
    """A thread-safe least-recently-used store of arrays within a byte budget."""

    def __init__(self, budget: int):
        self.budget = budget
        self._d: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
            return v

    def put(self, key, value: np.ndarray):
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._d[key] = value
            self._bytes += value.nbytes
            while self._bytes > self.budget and len(self._d) > 1:
                _k, v = self._d.popitem(last=False)
                self._bytes -= v.nbytes

    def drop(self, pred):
        with self._lock:
            for k in [k for k in self._d if pred(k)]:
                self._bytes -= self._d.pop(k).nbytes

    def __len__(self):
        return len(self._d)


QTILES = _Lru(320 * 2 ** 20)            # the quantized tiles of every image shown (16 bits)
RTILES = _Lru(256 * 2 ** 20)            # the rendered tiles (8 bits), for the stretch of the moment


class Prepared:
    """An image ready to show: the data (read by the readout and the export), the display pyramid and the statistics.
    Made by ``prepare`` in a worker thread; read-only afterwards."""

    def __init__(self, vi: ViewImage, data: np.ndarray, levels: list[np.ndarray], factor: int, stats: sm.Stats,
                 cfa: str, wcs):
        self.uid = next(_UID)
        self.vi = vi
        self.key = vi.identity()
        self.name = vi.name
        self.data = data
        self.levels = levels
        self.factor = factor                         # image pixels per pixel of level 0 (2: a debayered mosaic)
        self.stats = stats
        self.cfa = cfa
        self.wcs = wcs
        if data.ndim == 2:
            self.height, self.width = data.shape
        else:
            self.height, self.width = data.shape[1:]
        self.channels = levels[0].shape[0]
        self.integer = bool(cfa or np.issubdtype(data.dtype, np.integer) or (
            isinstance(vi.source, np.ndarray) and np.issubdtype(vi.source.dtype, np.integer)))
        self.place: Placement | None = None          # where it sits on a common grid (``ViewImage.place``)

    # what the overlays read (``canvas.source.size()``)
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def nbytes(self) -> int:
        n = sum(lv.nbytes for lv in self.levels[1:]) + self.stats.hist.nbytes
        n += self.data.nbytes
        if self.levels[0] is not self.data and self.levels[0].base is not self.data:
            n += self.levels[0].nbytes
        return n

    def level_size(self, k: int) -> tuple[int, int]:
        lv = self.levels[k]
        return lv.shape[2], lv.shape[1]

    def tiles_of(self, k: int) -> tuple[int, int]:
        w, h = self.level_size(k)
        return -(-w // TILE), -(-h // TILE)

    def qtile(self, k: int, tx: int, ty: int) -> np.ndarray:
        """The quantized tile (C, th, tw) uint16 of level ``k`` (made once, kept while memory allows)."""
        key = (self.uid, k, tx, ty)
        q = QTILES.get(key)
        if q is None:
            lv = self.levels[k]
            blk = lv[:, ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE]
            q = sm.quantize(blk, self.stats.lo, self.stats.hi)
            QTILES.put(key, q)
        return q

    def value(self, x: int, y: int) -> tuple[list[float], str] | None:
        """The data at image pixel (x, y): the channel values, and the photosite's colour for a mosaic."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        d = self.data
        if d.ndim == 2:
            return [float(d[y, x])], sm.cfa_letter(self.cfa, x, y) if self.cfa else ""
        return [float(v) for v in d[:, y, x]], ""

    def release(self):
        QTILES.drop(lambda k: k[0] == self.uid)


def planar(arr: np.ndarray) -> np.ndarray:
    """An image array as (C, H, W) float32, C 1 or 3 (an alpha channel dropped; not copied when it already is)."""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[None]
    elif a.ndim == 3 and a.shape[0] not in (1, 3) and a.shape[2] in (1, 3, 4):
        a = a[:, :, :3].transpose(2, 0, 1) if a.shape[2] >= 3 else a.transpose(2, 0, 1)
    elif a.ndim != 3:
        raise ValueError(f"not an image: shape {a.shape}")
    if a.shape[0] not in (1, 3):
        a = a[:1] if a.shape[0] < 3 else a[:3]
    if a.dtype != np.float32 or not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a, dtype=np.float32)
    return a


def _read(vi: ViewImage) -> tuple[np.ndarray, str, Any]:
    src = vi.source
    cfa, wcs = vi.cfa, None
    if isinstance(src, str | os.PathLike):
        arr, fcfa, wcs = read_file(src)
        cfa = cfa or fcfa
    elif callable(src):
        arr = src()
    else:
        arr = src
    return np.asarray(arr), cfa, wcs


def _to_side(img: np.ndarray, side: int) -> np.ndarray:
    """An (h, w) or (h, w, 3) uint8 picture as (h, w, 3), reduced so that its long side is at most ``side``."""
    import cv2
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    h, w = img.shape[:2]
    f = side / max(h, w)
    if f < 1:
        img = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def quick_thumbnail(vi: ViewImage, side: int = 160) -> np.ndarray:
    """A thumbnail made straight from the source, for a film strip (worker thread): read, reduced, stretched
    automatically (neutral sky for a camera-colour image or a mosaic). ``ViewImage(..., thumb=partial(quick_thumbnail,
    vi))`` when the caller has no thumbnail of its own."""
    arr, cfa, _wcs = _read(vi)
    lv = sm.superpixel(arr, cfa) if arr.ndim == 2 and cfa else planar(arr)
    while max(lv.shape[1:]) > side * 2 and min(lv.shape[1:]) >= 4:
        lv = sm.reduce2(lv)
    stats = sm.compute_stats(lv, smooth=lv)
    camera = vi.camera if vi.camera is not None else (bool(arr.ndim == 2 and cfa) or sm.looks_camera(stats))
    st = sm.auto_for(stats, vi.kind, neutral=camera if sm.rule_of(vi.kind) == "light" else False)
    img = sm.render(sm.quantize(lv, stats.lo, stats.hi), sm.luts(st, stats))
    return _to_side(img, side)


def prepare(vi: ViewImage) -> Prepared:
    """Read (when needed) and prepare an image: the pyramid and the statistics (worker thread)."""
    arr, cfa, wcs = _read(vi)
    wcs = _wcs_of(vi.wcs) or wcs
    if arr.ndim == 2 and cfa:
        data = arr
        level0 = sm.superpixel(arr, cfa)
        factor = 2
    else:
        level0 = planar(arr)
        data = level0
        factor = 1
        cfa = ""
    levels = sm.pyramid(level0)
    stats = sm.compute_stats(level0, smooth=levels[-1])
    prep = Prepared(vi, data, levels, factor, stats, cfa, wcs)
    prep.place = _place_of(vi.place, prep.width, prep.height)
    return prep


def _place_of(place, w: int, h: int) -> Placement | None:
    """The placement the caller gave (a callable is called here, in the worker thread, with the image's size)."""
    if callable(place):
        try:
            place = place(w, h)
        except Exception:  # noqa: BLE001 - an unknown placement: the compare falls back on the solutions or the width
            return None
    return place if isinstance(place, Placement) else None


def thumbnail_of(prep: Prepared, st: sm.Stretch | None = None, side: int = 160) -> np.ndarray:
    """A small picture (h, w, 3) uint8 of a prepared image, its long side ``side``, stretched (automatic by default)."""
    st = st or default_stretch(prep)
    k = next((i for i, lv in enumerate(prep.levels) if max(lv.shape[1:]) <= max(side * 2, 64)), len(prep.levels) - 1)
    lv = prep.levels[k]
    img = sm.render(sm.quantize(lv, prep.stats.lo, prep.stats.hi), sm.luts(st, prep.stats))
    return _to_side(img, side)


def default_stretch(prep: Prepared, mode: str | None = None, linked: bool | None = None, neutral: bool | None = None,
                    vi: ViewImage | None = None) -> sm.Stretch:
    """The automatic stretch of an image, the one of its kind (``stretch.auto_for``); ``vi``: how the caller describes
    it now (default: as when it was read). ``mode``, ``linked``, ``neutral``: None for the kind's own."""
    v = vi or prep.vi
    if v.stretched and (mode or "mtf") == "mtf" and neutral is None:
        c = prep.channels
        white = v.white_level()
        return sm.Stretch("linear", np.zeros(c), np.full(c, white / 2), np.full(c, white), np.ones(c), np.zeros(c),
                          True, False, False)
    if neutral is None:
        if sm.rule_of(v.kind) != "light":
            neutral = False                     # a calibration frame or a model: no sky to make grey
        elif v.neutral is not None:
            neutral = bool(v.neutral)
        else:
            cam = v.camera
            neutral = bool(prep.cfa) or (cam if cam is not None else sm.looks_camera(prep.stats))
    return sm.auto_for(prep.stats, v.kind, mode, linked, neutral)


def kind_of(vi: ViewImage | None) -> str:
    """The kind of an image for "Same stretch for all frames" (one stretch per kind): its ``kind``, or "lights"."""
    k = str(vi.kind or "").lower() if vi is not None else ""
    return "lights" if k in ("", "light") else k


# ---------------------------------------------------------------------------------------------- the loader
class Unavailable(Exception):
    """Raised by a source that knows why its image is not there: the text (already in the language of the
    interface) is what the view shows, alone."""


class Loader(QObject):
    """Prepares images in worker threads, the most urgent first, and keeps the last ones (least recently used out,
    within a byte budget). Answers come as signals in the GUI thread. One loader is shared by default
    (``default_loader``): the same stack shown in two views is read once."""

    ready = Signal(object, object)            # key, Prepared
    failed = Signal(object, str)              # key, reason
    thumbReady = Signal(object, object)       # key, (h, w, 3) uint8

    def __init__(self, budget: int = 3 * 2 ** 30, threads: int = 2, parent=None):
        super().__init__(parent)
        self.budget = budget
        self._cache: OrderedDict = OrderedDict()
        self._bytes = 0
        self._cond = threading.Condition()
        self._heap: list = []
        self._queued: dict = {}                    # (kind, key) -> (priority, seq)
        self._running: set = set()
        self._seq = itertools.count()
        self._stop = False
        self.thumbs: dict = {}                     # key -> thumbnail
        self.unavailable: set = set()              # keys whose source raised Unavailable (its text is the message)
        self.loads = 0                             # images prepared (the tests count them)
        self.load_threads: set[int] = set()        # the threads that did it
        self._threads = [threading.Thread(target=self._run, daemon=True, name=f"imageview-loader-{i}")
                         for i in range(threads)]
        for t in self._threads:
            t.start()

    # ------------------------------------------------------------ GUI thread
    def cached(self, key) -> Prepared | None:
        with self._cond:
            p = self._cache.get(key)
            if p is not None:
                self._cache.move_to_end(key)
            return p

    def request(self, vi: ViewImage, priority: int = 0):
        """Prepare ``vi`` (0: now, 1, 2...: ahead of time). Answer: ``ready`` or ``failed``; nothing when cached."""
        self._submit("load", vi, priority)

    def request_thumb(self, vi: ViewImage, priority: int = 10):
        key = vi.identity()
        if key in self.thumbs:
            return
        if isinstance(vi.thumb, np.ndarray):
            self.thumbs[key] = np.ascontiguousarray(vi.thumb[:, :, :3]) if vi.thumb.ndim == 3 else vi.thumb
            self.thumbReady.emit(key, self.thumbs[key])
            return
        if callable(vi.thumb):
            self._submit("thumb", vi, priority)

    def forget_requests(self, keep=()):
        """Drop the waiting requests made ahead of time (priority above 0), but those of the keys in ``keep``."""
        keep = set(keep)
        with self._cond:
            for (kind, key), (pr, _s) in list(self._queued.items()):
                if pr > 0 and key not in keep and kind == "load":
                    del self._queued[(kind, key)]

    def busy(self) -> bool:
        with self._cond:
            return bool(self._queued or self._running)

    def wait(self, timeout: float = 60.0) -> bool:
        """Process events until every request is answered (tests and scripted walkthroughs)."""
        end = time.monotonic() + timeout
        app = QApplication.instance()
        while time.monotonic() < end:
            app.processEvents()
            if not self.busy():
                app.processEvents()
                return True
            time.sleep(0.003)
        return False

    def clear(self):
        with self._cond:
            for p in self._cache.values():
                p.release()
            self._cache.clear()
            self._bytes = 0
            self.thumbs.clear()

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def _submit(self, kind: str, vi: ViewImage, priority: int):
        key = vi.identity()
        with self._cond:
            if kind == "load" and key in self._cache:
                return
            if (kind, key) in self._running:
                return
            old = self._queued.get((kind, key))
            if old is not None and old[0] <= priority:
                return
            seq = next(self._seq)
            self._queued[(kind, key)] = (priority, seq)
            heapq.heappush(self._heap, (priority, seq, kind, key, vi))
            self._cond.notify()

    # ------------------------------------------------------------ worker threads
    def _next(self):
        with self._cond:
            while True:
                if self._stop:
                    return None
                while self._heap:
                    pr, seq, kind, key, vi = heapq.heappop(self._heap)
                    if self._queued.get((kind, key)) == (pr, seq):
                        del self._queued[(kind, key)]
                        self._running.add((kind, key))
                        return kind, key, vi
                self._cond.wait()

    def _run(self):
        while True:
            job = self._next()
            if job is None:
                return
            kind, key, vi = job
            try:
                if kind == "load":
                    prep = prepare(vi)
                    self.load_threads.add(threading.get_ident())
                    self.loads += 1
                    self._keep(key, prep)
                    self._emit(self.ready, key, prep)
                    if key not in self.thumbs and vi.thumb is None:
                        th = thumbnail_of(prep)
                        self.thumbs[key] = th
                        self._emit(self.thumbReady, key, th)
                else:
                    th = np.ascontiguousarray(np.asarray(vi.thumb())[:, :, :3])
                    self.thumbs[key] = th
                    self._emit(self.thumbReady, key, th)
            except Exception as e:  # noqa: BLE001 - every failure goes to the view as text
                if kind == "load":
                    if isinstance(e, Unavailable):          # the caller's own words (translated), shown alone
                        self.unavailable.add(key)
                        self._emit(self.failed, key, str(e))
                    else:
                        self.unavailable.discard(key)
                        self._emit(self.failed, key, f"{type(e).__name__}: {e}")
            finally:
                with self._cond:
                    self._running.discard((kind, key))

    def _keep(self, key, prep: Prepared):
        with self._cond:
            old = self._cache.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._cache[key] = prep
            self._bytes += prep.nbytes
            while self._bytes > self.budget and len(self._cache) > 1:
                _k, p = self._cache.popitem(last=False)
                self._bytes -= p.nbytes
                p.release()

    @staticmethod
    def _emit(sig, *args):
        try:
            sig.emit(*args)
        except RuntimeError:                          # the loader went with the application
            pass


_default_loader: Loader | None = None


def memory_budget() -> int:
    """The prepared images kept in memory: a fifth of the machine's memory, from 3 to 24 GB (a drizzled stack takes
    several hundred MB once prepared; with too small a budget a tab shown again is read from the disk again)."""
    try:
        import psutil
        total = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 - unknown machine: the smallest budget
        return 3 * 2 ** 30
    return int(min(max(total // 5, 3 * 2 ** 30), 24 * 2 ** 30))


def default_loader() -> Loader:
    global _default_loader
    if _default_loader is None:
        _default_loader = Loader(budget=memory_budget(), threads=max(2, min(4, (os.cpu_count() or 2) // 4)),
                                 parent=QApplication.instance())
    return _default_loader


# ---------------------------------------------------------------------------------------------- one image on screen
class Display:
    """A prepared image shown through one stretch: the rendered tiles and the picture of the visible part."""

    def __init__(self, prep: Prepared, st: sm.Stretch):
        self.uid = next(_UID)
        self.prep = prep
        self.version = 0
        self._comp = None
        self._thumb = None
        self.align: Alignment | None = None          # drawn on another image's grid (a compared image)
        self.frame: tuple[int, int] | None = None    # then the size of that image: the area it is drawn on
        self.stretch = st
        self.set_stretch(st)

    def set_alignment(self, al: Alignment | None, frame: tuple[int, int] | None = None):
        """Draw this image on the grid of another one (``al.T`` maps that grid onto this image), ``frame`` its size;
        None: on its own grid."""
        self.align = al
        self.frame = (int(frame[0]), int(frame[1])) if al is not None and frame is not None else None
        if self.frame is None:
            self.align = None
        self._comp = None
        self._thumb = None

    def size(self) -> tuple[int, int]:
        """The size of the area it is drawn on: its own, or the grid it is aligned on."""
        return self.frame if self.frame is not None else self.prep.size()

    def set_stretch(self, st: sm.Stretch):
        st = st.adapted(self.prep.channels)
        old = self.version
        self.stretch = st
        self.lut = sm.luts(st, self.prep.stats)
        self.version += 1
        self._comp = None
        self._thumb = None
        RTILES.drop(lambda k, u=self.uid, v=old: k[0] == u and k[1] <= v)

    def level_for(self, s_dev: float) -> int:
        """The pyramid level nearest the screen's resolution at ``s_dev`` device pixels per image pixel."""
        v = 1.0 / max(s_dev * self.prep.factor, 1e-9)
        k = int(round(math.log2(v))) if v > 1.0 else 0
        return max(0, min(k, len(self.prep.levels) - 1))

    def _tile(self, k: int, tx: int, ty: int) -> np.ndarray:
        key = (self.uid, self.version, k, tx, ty)
        t = RTILES.get(key)
        if t is None:
            t = sm.render(self.prep.qtile(k, tx, ty), self.lut)
            RTILES.put(key, t)
        return t

    def composite(self, k: int, tx0: int, ty0: int, tx1: int, ty1: int) -> np.ndarray:
        """The tiles [tx0, tx1) x [ty0, ty1) of level ``k`` side by side (made in parallel threads)."""
        key = (k, tx0, ty0, tx1, ty1, self.version)
        if self._comp is not None and self._comp[0] == key:
            return self._comp[1]
        coords = [(tx, ty) for ty in range(ty0, ty1) for tx in range(tx0, tx1)]
        missing = [c for c in coords if RTILES.get((self.uid, self.version, k) + c) is None]
        if len(missing) > 1:
            list(sm.pool("render").map(lambda c: self._tile(k, *c), missing))
        w, h = self.prep.level_size(k)
        x1, y1 = min(w, tx1 * TILE), min(h, ty1 * TILE)
        shape = (y1 - ty0 * TILE, x1 - tx0 * TILE) + ((3,) if self.prep.channels == 3 else ())
        out = np.empty(shape, np.uint8)
        for tx, ty in coords:
            t = self._tile(k, tx, ty)
            oy, ox = (ty - ty0) * TILE, (tx - tx0) * TILE
            out[oy:oy + t.shape[0], ox:ox + t.shape[1]] = t
        self._comp = (key, out)
        return out

    def view(self, s_dev: float, cx: float, cy: float, W: int, H: int, clip: tuple[int, int] | None = None):
        """The visible part of the image in device pixels: (picture, u0, v0), its top-left corner in the area, or
        None. ``s_dev``: device pixels per image pixel; (cx, cy): the image point at the centre of the W x H area;
        ``clip``: only the columns [u0, u1)."""
        if self.align is not None:
            return self._view_aligned(s_dev, cx, cy, W, H, clip)
        import cv2
        p = self.prep
        k = self.level_for(s_dev)
        F = p.factor * (1 << k)
        lw, lh = p.level_size(k)
        ex0, ex1 = W / 2 - cx * s_dev, W / 2 + (lw * F - cx) * s_dev
        ey0, ey1 = H / 2 - cy * s_dev, H / 2 + (lh * F - cy) * s_dev
        u0, u1 = max(0, math.floor(ex0)), min(W, math.ceil(ex1))
        v0, v1 = max(0, math.floor(ey0)), min(H, math.ceil(ey1))
        if clip is not None:
            u0, u1 = max(u0, clip[0]), min(u1, clip[1])
        if u1 <= u0 or v1 <= v0:
            return None
        lx0 = ((u0 - W / 2) / s_dev + cx) / F
        lx1 = ((u1 - W / 2) / s_dev + cx) / F
        ly0 = ((v0 - H / 2) / s_dev + cy) / F
        ly1 = ((v1 - H / 2) / s_dev + cy) / F
        ntx, nty = p.tiles_of(k)
        tx0 = max(0, int(math.floor(lx0 - 1)) // TILE)
        tx1 = min(ntx, int(math.ceil(lx1 + 1)) // TILE + 1)
        ty0 = max(0, int(math.floor(ly0 - 1)) // TILE)
        ty1 = min(nty, int(math.ceil(ly1 + 1)) // TILE + 1)
        comp = self.composite(k, tx0, ty0, tx1, ty1)
        s_lvl = s_dev * F
        bx = ((u0 + 0.5 - W / 2) / s_dev + cx) / F - 0.5 - tx0 * TILE
        by = ((v0 + 0.5 - H / 2) / s_dev + cy) / F - 0.5 - ty0 * TILE
        m = np.array([[1.0 / s_lvl, 0.0, bx], [0.0, 1.0 / s_lvl, by]])
        interp = cv2.INTER_NEAREST if s_lvl >= 0.9999 else cv2.INTER_LINEAR
        buf = cv2.warpAffine(comp, m, (u1 - u0, v1 - v0), flags=interp | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
        return buf, u0, v0

    def _view_aligned(self, s_dev: float, cx: float, cy: float, W: int, H: int, clip):
        """``view`` for an image drawn on another one's grid (``align``): (cx, cy) and the area are that grid's; each
        screen pixel takes this image's pixel ``align.T`` maps it on (bilinear below 1:1, display only). Where this
        image has no pixel the canvas's background shows."""
        import cv2
        p = self.prep
        fw, fh = self.frame
        T = self.align.T
        ex0, ex1 = W / 2 - cx * s_dev, W / 2 + (fw - cx) * s_dev
        ey0, ey1 = H / 2 - cy * s_dev, H / 2 + (fh - cy) * s_dev
        u0, u1 = max(0, math.floor(ex0)), min(W, math.ceil(ex1))
        v0, v1 = max(0, math.floor(ey0)), min(H, math.ceil(ey1))
        if clip is not None:
            u0, u1 = max(u0, clip[0]), min(u1, clip[1])
        if u1 <= u0 or v1 <= v0:
            return None
        lin = self.align.scale                                   # this image's pixels per pixel of the grid
        k = self.level_for(s_dev / lin)
        F = p.factor * (1 << k)
        ax = (u0 + 0.5 - W / 2) / s_dev + cx                     # the grid point of the first screen pixel
        ay = (v0 + 0.5 - H / 2) / s_dev + cy
        # screen pixel (u, v) -> grid (ax + u / s, ay + v / s) -> this image (T) -> its level-k pixel index
        m = np.array([[T[0, 0] / (s_dev * F), T[0, 1] / (s_dev * F), (T[0, 0] * ax + T[0, 1] * ay + T[0, 2]) / F - 0.5],
                      [T[1, 0] / (s_dev * F), T[1, 1] / (s_dev * F), (T[1, 0] * ax + T[1, 1] * ay + T[1, 2]) / F - 0.5]])
        du, dv = u1 - u0, v1 - v0
        corners = m @ np.array([[0, du, 0, du], [0, 0, dv, dv], [1, 1, 1, 1]], float)
        field = self.align.field
        pad = 1.0 + (max(float(np.abs(field[1]).max()), float(np.abs(field[2]).max())) / F if field is not None
                     else 0.0)
        lw, lh = p.level_size(k)
        ntx, nty = p.tiles_of(k)
        tx0 = max(0, int(math.floor(corners[0].min() - pad)) // TILE)
        tx1 = min(ntx, int(math.ceil(corners[0].max() + pad)) // TILE + 1)
        ty0 = max(0, int(math.floor(corners[1].min() - pad)) // TILE)
        ty1 = min(nty, int(math.ceil(corners[1].max() + pad)) // TILE + 1)
        if tx1 <= tx0 or ty1 <= ty0 or corners[0].max() < -1 or corners[1].max() < -1 or \
                corners[0].min() > lw or corners[1].min() > lh:
            bg = np.empty((dv, du, 3) if p.channels == 3 else (dv, du), np.uint8)
            bg[...] = CANVAS_RGB if p.channels == 3 else CANVAS_RGB[0]
            return bg, u0, v0                                    # nothing of this image here
        comp = self.composite(k, tx0, ty0, tx1, ty1)
        m[0, 2] -= tx0 * TILE
        m[1, 2] -= ty0 * TILE
        interp = cv2.INTER_NEAREST if s_dev * F / lin >= 0.9999 else cv2.INTER_LINEAR
        border = CANVAS_RGB if comp.ndim == 3 else CANVAS_RGB[0]
        if field is None:
            buf = cv2.warpAffine(comp, m, (du, dv), flags=interp | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=border)
            return buf, u0, v0
        # the solutions' own distortions: the affine map corrected pixel by pixel (bilinear between the grid's points)
        (fw_, fh_), fx, fy = field
        n = fx.shape[0]
        gx = np.clip((ax + np.arange(du) / s_dev - 0.5) / max(fw_ - 1.0, 1e-9) * (n - 1), 0, n - 1)
        gy = np.clip((ay + np.arange(dv) / s_dev - 0.5) / max(fh_ - 1.0, 1e-9) * (n - 1), 0, n - 1)
        mgx = np.ascontiguousarray(np.broadcast_to(gx.astype(np.float32)[None, :], (dv, du)))
        mgy = np.ascontiguousarray(np.broadcast_to(gy.astype(np.float32)[:, None], (dv, du)))
        rx = cv2.remap(fx, mgx, mgy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ry = cv2.remap(fy, mgx, mgy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        uu = np.arange(du, dtype=np.float32)[None, :]
        vv = np.arange(dv, dtype=np.float32)[:, None]
        mapx = (m[0, 0] * uu + m[0, 1] * vv + m[0, 2] + rx / F).astype(np.float32)
        mapy = (m[1, 0] * uu + m[1, 1] * vv + m[1, 2] + ry / F).astype(np.float32)
        buf = cv2.remap(comp, mapx, mapy, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
        return buf, u0, v0

    def release(self):
        """Forget the rendered tiles (the image is no longer shown)."""
        RTILES.drop(lambda k, u=self.uid: k[0] == u)
        self._comp = None

    def thumbnail(self, side: int) -> np.ndarray:
        if self._thumb is None or self._thumb[0] != side:
            if self.align is not None:                       # drawn on another grid: its picture there
                fw, fh = self.frame
                s = side / max(fw, fh)
                w, h = max(1, round(fw * s)), max(1, round(fh * s))
                img = np.empty((h, w, 3), np.uint8)
                img[...] = CANVAS_RGB
                got = self.view(s, fw / 2, fh / 2, w, h)
                if got is not None:
                    buf, u0, v0 = got
                    if buf.ndim == 2:
                        buf = np.repeat(buf[:, :, None], 3, axis=2)
                    img[v0:v0 + buf.shape[0], u0:u0 + buf.shape[1]] = buf
                self._thumb = (side, img)
            else:
                self._thumb = (side, thumbnail_of(self.prep, self.stretch, side))
        return self._thumb[1]


def to_qimage(buf: np.ndarray) -> QImage:
    """An (h, w, 3) or (h, w) uint8 array as a QImage over the same memory (keep the array while it is drawn)."""
    buf = np.ascontiguousarray(buf)
    h, w = buf.shape[:2]
    fmt = QImage.Format_RGB888 if buf.ndim == 3 else QImage.Format_Grayscale8
    return QImage(buf.data, w, h, buf.strides[0], fmt)


TAG_BG = QColor(0, 0, 0, 170)


def tag(p: QPainter, pt: QPointF, text: str, right: bool = False, color: str = "#ffffff", bg: QColor = TAG_BG):
    """A small label on the image (rounded, dark)."""
    m = p.fontMetrics()
    w = m.horizontalAdvance(text) + 14
    r = QRectF(pt.x() - w if right else pt.x(), pt.y(), w, m.height() + 6)
    p.setPen(Qt.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(r, 4, 4)
    p.setPen(QColor(color))
    p.drawText(r, Qt.AlignCenter, text)
    return r


# ---------------------------------------------------------------------------------------------- the canvas
class ImageCanvas(QWidget):
    """The drawing surface: one image (and, in split mode, the image it is compared with), zoom and pan, the minimap,
    overlays. ``scale`` is in logical pixels per image pixel; (cx, cy) is the image point at the centre."""

    viewChanged = Signal()
    hovered = Signal(float, float, int)        # image coordinates, 0: the image, 1: the compared one (split)
    left = Signal()
    rectDone = Signal(float, float, float, float)
    splitMoved = Signal(float)
    zoomed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(240, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display: Display | None = None
        self.compare: Display | None = None        # shown left of the split line
        self.split = 0.5
        self.split_names = ("", "")
        self.title = ""                             # the name tag (side by side)
        self.message = ""                           # centred: loading, failure
        self.busy = False                           # a small "Loading…" badge over the image shown
        self.scale = 1.0
        self.cx = self.cy = 0.0
        self.fitted = True                          # follows the size of the widget
        self.overlays: list = []                    # callables (painter, canvas)
        self.minimap_on = True
        self.mode = "pan"                           # pan | rect
        #: an editing tool (the Gradient block's samples and areas): press(canvas, x, y, button) -> bool (taken),
        #: move(canvas, x, y), release(canvas), hover(canvas, x, y), cursor (a Qt cursor), in image coordinates
        self.tool = None
        self.frame_ms: list[float] = []             # paint times (the measurements)
        self._drag = None
        self._rect = None
        self._mouse: QPointF | None = None

    # ------------------------------------------------------------ what the overlays read
    @property
    def source(self) -> Prepared | None:
        return self.display.prep if self.display is not None else None

    def _tag(self, p, pt, text, right=False):
        tag(p, pt, text, right)

    # ------------------------------------------------------------ images
    def set_display(self, d: Display | None, keep_view: bool = False):
        same = (self.display is not None and d is not None and self.display.size() == d.size())
        self.display = d
        if d is not None and not (keep_view and same):
            self.fit()
        self.update()

    def set_compare(self, d: Display | None):
        self.compare = d
        self.update()

    def ratio(self) -> float:
        """Compared image pixels per image pixel (images of different sizes are matched by their width)."""
        if self.display is None or self.compare is None:
            return 1.0
        return self.compare.size()[0] / max(self.display.size()[0], 1)

    # ------------------------------------------------------------ the view
    def fit_scale(self) -> float:
        if self.display is None:
            return 1.0
        w, h = self.display.size()
        return max(min((self.width() - 16) / w, (self.height() - 16) / h), 1e-4)

    def min_scale(self) -> float:
        return min(self.fit_scale() * 0.5, 0.05)

    def max_scale(self) -> float:
        return 32.0 / self.devicePixelRatioF()

    def fit(self):
        if self.display is None:
            return
        w, h = self.display.size()
        self.scale = self.fit_scale()
        self.cx, self.cy = w / 2, h / 2
        self.fitted = True
        self.viewChanged.emit()
        self.zoomed.emit()
        self.update()

    def zoom_to(self, scale: float, cx: float | None = None, cy: float | None = None):
        self.scale = float(np.clip(scale, self.min_scale(), self.max_scale()))
        if cx is not None:
            self.cx, self.cy = float(cx), float(cy)
        self.fitted = False
        self._clamp()
        self.viewChanged.emit()
        self.zoomed.emit()
        self.update()

    def zoom_by(self, k: float, at: QPointF | None = None):
        """Zoom by ``k`` keeping the image point under ``at`` (default: the centre) in place."""
        at = at or QPointF(self.width() / 2, self.height() / 2)
        ix, iy = self.w_to_img(at.x(), at.y())
        s = float(np.clip(self.scale * k, self.min_scale(), self.max_scale()))
        self.scale = s
        self.cx = ix - (at.x() - self.width() / 2) / s
        self.cy = iy - (at.y() - self.height() / 2) / s
        self.fitted = False
        self._clamp()
        self.viewChanged.emit()
        self.zoomed.emit()
        self.update()

    def zoom_percent(self) -> float:
        """The zoom as an image editor says it: 100 % is one image pixel per screen pixel."""
        return 100.0 * self.scale * self.devicePixelRatioF()

    def set_zoom_percent(self, pct: float, at: QPointF | None = None):
        self.zoom_by(pct / 100.0 / self.devicePixelRatioF() / self.scale, at)

    def pan_by(self, dx: float, dy: float):
        """Move the image by (dx, dy) logical pixels."""
        self.cx -= dx / self.scale
        self.cy -= dy / self.scale
        self.fitted = False
        self._clamp()
        self.viewChanged.emit()
        self.update()

    def _clamp(self):
        if self.display is None:
            return
        w, h = self.display.size()
        mx, my = self.width() / 2 / self.scale, self.height() / 2 / self.scale
        # wider than the view: the edges stop at the borders; narrower: centred
        self.cx = float(np.clip(self.cx, mx, w - mx)) if w > 2 * mx else w / 2
        self.cy = float(np.clip(self.cy, my, h - my)) if h > 2 * my else h / 2

    def img_to_w(self, x, y) -> QPointF:
        return QPointF((x - self.cx) * self.scale + self.width() / 2, (y - self.cy) * self.scale + self.height() / 2)

    def w_to_img(self, px, py) -> tuple[float, float]:
        return (px - self.width() / 2) / self.scale + self.cx, (py - self.height() / 2) / self.scale + self.cy

    def visible(self) -> tuple[float, float, float, float]:
        """The visible part of the image (x0, y0, x1, y1), image pixels."""
        if self.display is None:
            return 0, 0, 0, 0
        w, h = self.display.size()
        x0, y0 = self.w_to_img(0, 0)
        x1, y1 = self.w_to_img(self.width(), self.height())
        return max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)

    def whole_visible(self) -> bool:
        if self.display is None:
            return True
        w, h = self.display.size()
        x0, y0, x1, y1 = self.visible()
        return x0 <= 0.5 and y0 <= 0.5 and x1 >= w - 0.5 and y1 >= h - 0.5

    def resizeEvent(self, e):  # noqa: N802 - Qt's name
        if self.fitted:
            self.fit()
        super().resizeEvent(e)

    # ------------------------------------------------------------ drawing
    def paintEvent(self, e):  # noqa: N802
        t0 = time.perf_counter()
        p = QPainter(self)
        try:
            self._paint(p)
        finally:
            p.end()
        self.frame_ms.append(1000 * (time.perf_counter() - t0))
        if len(self.frame_ms) > 400:
            del self.frame_ms[:200]

    def _draw(self, p: QPainter, d: Display, s_dev: float, cx: float, cy: float, clip: QRectF | None = None):
        """One image over the whole area (the same resampling whatever the divider), drawn inside ``clip``."""
        dpr = self.devicePixelRatioF()
        W, H = max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr))
        res = d.view(s_dev, cx, cy, W, H)
        if res is None:
            return
        buf, u0, v0 = res
        qi = to_qimage(buf)
        qi.setDevicePixelRatio(dpr)
        if clip is not None:
            p.save()
            p.setClipRect(clip)
        p.drawImage(QPointF(u0 / dpr, v0 / dpr), qi)
        if clip is not None:
            p.restore()

    def _paint(self, p: QPainter):
        p.fillRect(self.rect(), CANVAS_BG)
        d = self.display
        if d is not None:
            dpr = self.devicePixelRatioF()
            s_dev = self.scale * dpr
            if self.compare is not None:
                r = self.ratio()
                xs = round(self.width() * self.split * dpr) / dpr
                self._draw(p, self.compare, s_dev / r, self.cx * r, self.cy * r, QRectF(0, 0, xs, self.height()))
                self._draw(p, d, s_dev, self.cx, self.cy, QRectF(xs, 0, self.width() - xs, self.height()))
            else:
                self._draw(p, d, s_dev, self.cx, self.cy)
            if s_dev >= 16:
                self._pixel_grid(p)
            for ov in self.overlays:
                ov(p, self)
            if self.compare is not None:
                self._split_line(p)
            if self.minimap_on and not self.whole_visible():
                self._minimap(p)
            if self.title:
                tag(p, QPointF(10, 10), self.title)
            if self.busy:
                tag(p, QPointF(self.width() - 10, 10), _("Loading…"), right=True, color="#ffd28a")
        if self._rect:
            a, b = self._rect
            p.setPen(QPen(QColor(WARN), 1.5, Qt.DashLine))
            p.setBrush(QColor(230, 162, 60, 40))
            p.drawRect(QRectF(self.img_to_w(*a), self.img_to_w(*b)).normalized())
        if self.message:
            f = QFont(self.font())
            f.setPointSizeF(11)
            p.setFont(f)
            m = p.fontMetrics()
            w = min(m.horizontalAdvance(self.message) + 40, self.width() - 20)
            r = QRectF((self.width() - w) / 2, (self.height() - m.height() * 3) / 2, w, m.height() * 3)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(28, 31, 38, 230))
            p.drawRoundedRect(r, 6, 6)
            p.setPen(QColor("#c9ced8"))
            p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap, self.message)

    def _pixel_grid(self, p: QPainter):
        x0, y0, x1, y1 = self.visible()
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        for x in range(int(x0), int(math.ceil(x1)) + 1):
            px = self.img_to_w(x, 0).x()
            p.drawLine(QPointF(px, self.img_to_w(0, y0).y()), QPointF(px, self.img_to_w(0, y1).y()))
        for y in range(int(y0), int(math.ceil(y1)) + 1):
            py = self.img_to_w(0, y).y()
            p.drawLine(QPointF(self.img_to_w(x0, 0).x(), py), QPointF(self.img_to_w(x1, 0).x(), py))

    def _split_line(self, p: QPainter):
        xs = self.width() * self.split
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(0, 0, 0, 120), 4))
        p.drawLine(QPointF(xs, 0), QPointF(xs, self.height()))
        p.setPen(QPen(QColor("#ffffff"), 1.5))
        p.drawLine(QPointF(xs, 0), QPointF(xs, self.height()))
        c = QPointF(xs, self.height() / 2)
        p.setBrush(QColor("#ffffff"))
        p.setPen(QPen(QColor(0, 0, 0, 140), 1))
        p.drawEllipse(c, 11, 11)
        p.setPen(QPen(QColor("#15171c"), 1.6))
        for sgn in (-1, 1):
            tri = QPolygonF([QPointF(xs + sgn * 3, c.y() - 4), QPointF(xs + sgn * 7, c.y()),
                             QPointF(xs + sgn * 3, c.y() + 4)])
            p.drawPolyline(tri)
        left, right = self.split_names
        if left:
            tag(p, QPointF(xs - 10, 10), "◂  " + left, right=True)
        if right:
            tag(p, QPointF(xs + 10, 10), right + "  ▸")
        p.setRenderHint(QPainter.Antialiasing, False)

    def minimap_rect(self) -> QRectF:
        w, h = self.display.size()
        mw, mh = 190.0, 130.0
        f = min(mw / w, mh / h)
        rw, rh = w * f, h * f
        return QRectF(self.width() - rw - 12, self.height() - rh - 12, rw, rh)

    def _minimap(self, p: QPainter):
        r = self.minimap_rect()
        dpr = self.devicePixelRatioF()
        th = self.display.thumbnail(int(max(r.width(), r.height()) * dpr))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(15, 17, 21, 220))
        p.drawRoundedRect(r.adjusted(-4, -4, 4, 4), 5, 5)
        qi = to_qimage(th)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(r, qi)
        w, h = self.display.size()
        x0, y0, x1, y1 = self.visible()
        f = r.width() / w
        v = QRectF(r.x() + x0 * f, r.y() + y0 * f, (x1 - x0) * f, (y1 - y0) * f)
        p.save()
        p.setClipRect(r)
        outside = QPainterPath()
        outside.addRect(r)
        inner = QPainterPath()
        inner.addRect(v)
        p.setBrush(QColor(0, 0, 0, 110))
        p.drawPath(outside.subtracted(inner))
        p.restore()
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(ACCENT), 1.5))
        p.drawRect(v)

    # ------------------------------------------------------------ interaction
    def _near_split(self, x: float) -> bool:
        return self.compare is not None and abs(x - self.width() * self.split) < 12

    def _in_minimap(self, pos: QPointF) -> bool:
        return (self.display is not None and self.minimap_on and not self.whole_visible()
                and self.minimap_rect().adjusted(-4, -4, 4, 4).contains(pos))

    def _minimap_goto(self, pos: QPointF):
        r = self.minimap_rect()
        w, _h = self.display.size()
        f = r.width() / w
        self.cx = (pos.x() - r.x()) / f
        self.cy = (pos.y() - r.y()) / f
        self.fitted = False
        self._clamp()
        self.viewChanged.emit()
        self.update()

    def wheelEvent(self, e):  # noqa: N802
        if self.display is None:
            return
        dy = e.angleDelta().y()
        if dy == 0:
            dy = e.pixelDelta().y() * 4
        if dy == 0:
            return
        self.zoom_by(ZOOM_STEP ** (dy / 120.0), e.position())
        e.accept()

    def mousePressEvent(self, e):  # noqa: N802
        pos = e.position()
        ix, iy = self.w_to_img(pos.x(), pos.y())
        if e.button() == Qt.LeftButton and self._near_split(pos.x()):
            self._drag = ("split",)
        elif e.button() == Qt.LeftButton and self._in_minimap(pos):
            self._drag = ("minimap",)
            self._minimap_goto(pos)
        elif e.button() == Qt.LeftButton and self.mode == "rect":
            self._rect = [(ix, iy), (ix, iy)]
            self._drag = ("rect",)
        elif self.tool is not None and e.button() != Qt.MiddleButton and self.tool.press(self, ix, iy, e.button()):
            self._drag = ("tool",)
        elif e.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._drag = ("pan", pos, self.cx, self.cy)
            self.setCursor(Qt.ClosedHandCursor)
        self.update()

    def mouseMoveEvent(self, e):  # noqa: N802
        pos = e.position()
        self._mouse = pos
        ix, iy = self.w_to_img(pos.x(), pos.y())
        d = self._drag
        if d is None:
            if self.compare is not None and pos.x() < self.width() * self.split:
                r = self.ratio()
                self.hovered.emit(ix * r, iy * r, 1)
            else:
                self.hovered.emit(ix, iy, 0)
            if self.tool is not None and not self._near_split(pos.x()) and not self._in_minimap(pos):
                self.tool.hover(self, ix, iy)
                self.setCursor(self.tool.cursor)
                self.update()
                return
            self.setCursor(Qt.SplitHCursor if self._near_split(pos.x()) else
                           Qt.PointingHandCursor if self._in_minimap(pos) else
                           Qt.CrossCursor if self.mode == "rect" else Qt.CrossCursor)
            return
        if d[0] == "split":
            self.split = float(np.clip(pos.x() / max(self.width(), 1), 0.02, 0.98))
            self.splitMoved.emit(self.split)
        elif d[0] == "minimap":
            self._minimap_goto(pos)
        elif d[0] == "rect":
            self._rect[1] = (ix, iy)
        elif d[0] == "tool":
            self.tool.move(self, ix, iy)
        elif d[0] == "pan":
            self.cx = d[2] - (pos.x() - d[1].x()) / self.scale
            self.cy = d[3] - (pos.y() - d[1].y()) / self.scale
            self.fitted = False
            self._clamp()
            self.viewChanged.emit()
        self.update()

    def mouseReleaseEvent(self, e):  # noqa: N802
        d = self._drag
        self._drag = None
        self.setCursor(Qt.CrossCursor if self.tool is None else self.tool.cursor)
        if d and d[0] == "tool" and self.tool is not None:
            self.tool.release(self)
        if d and d[0] == "rect" and self._rect:
            (ax, ay), (bx, by) = self._rect
            self._rect = None
            if abs(bx - ax) * self.scale > 6 and abs(by - ay) * self.scale > 6:
                self.rectDone.emit(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
        self.update()

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        if self.display is None or self.mode != "pan" or self.tool is not None or self._near_split(e.position().x()):
            return
        if self.fitted or self.scale <= self.fit_scale() * 1.01:
            self.set_zoom_percent(100.0, e.position())
        else:
            self.fit()

    def leaveEvent(self, e):  # noqa: N802
        self._mouse = None
        self.left.emit()
        super().leaveEvent(e)


# ---------------------------------------------------------------------------------------------- the histogram
def nice_ticks(a0: float, a1: float) -> list[float]:
    """Round values (1, 2, 5 x 10^n, and 0) between a0 and a1."""
    span = max(abs(a0), abs(a1), 1e-12)
    top = int(math.floor(math.log10(span)))
    out = {0.0}
    for e in range(top - 6, top + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            out.update((v, -v))
    return sorted(v for v in out if a0 <= v <= a1)


def short_number(v: float) -> str:
    """A tick label: 0, 0.002, 50, 1000, 20k (a decimal comma in French)."""
    a = abs(v)
    if a >= 10000:
        return number(v / 1000, 0) + "k"
    if a >= 1 or a == 0:
        return number(v, 0)
    d = max(0, -int(math.floor(math.log10(a))))
    return number(v, d)


def spaced_ticks(ticks: list[tuple[float, str]], width, left: float, right: float, gap: float = 8.0,
                 least: float = 34.0) -> list[tuple[float, str]]:
    """The tick labels (x, text) kept on an axis: inside [left, right], each at least ``least`` px from the previous
    one and never touching it (half widths plus ``gap``); ``width(text)``: its width in px. A fixed spacing let a dark's
    narrow range put "-100" against "100"."""
    out, last, last_half = [], -1e9, 0.0
    for x, text in ticks:
        half = width(text) / 2
        if x < left or x > right or x - last < max(least, last_half + half + gap):
            continue
        out.append((x, text))
        last, last_half = x, half
    return out


class HistogramView(QWidget):
    """The histogram of the data, with the black, mid and white handles and the curve of the stretch. The axis is a
    soft logarithm of the data (asinh): the sky, the faint nebula and the stars all get room, and the handles never
    pile up at the left as on a linear axis."""

    edited = Signal(str, float)                 # black | mid | white, value (after the neutral pre-scaling)
    released = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 150)
        self.setMouseTracking(True)
        self.stats: sm.Stats | None = None
        self.st: sm.Stretch | None = None
        self.channel = -1
        self.units = ""
        self._axis = None
        self._bins = None
        self._drag: str | None = None
        self._hover: float | None = None

    def set_state(self, stats: sm.Stats | None, st: sm.Stretch | None, channel: int = -1, units: str = ""):
        if stats is not self.stats or st is None or self.st is None or not (
                np.array_equal(st.gain, self.st.gain) and np.array_equal(st.shift, self.st.shift)):
            self._axis = self._bins = None
        self.stats, self.st, self.channel, self.units = stats, st, channel, units
        self.update()

    # ------------------------------------------------------------ geometry
    def plot_rect(self) -> QRectF:
        return QRectF(8, 6, self.width() - 16, self.height() - 6 - 36)

    def axis(self):
        """(a0, a1, s, A): u = asinh((y - a0) / s) / A, y in the units of the pre-scaled data."""
        if self._axis is None:
            t = sm.transformed(self.stats, self.st.gain, self.st.shift)
            a0 = max(float(t["lo"].min()), float((t["median"] - 10 * t["sigma"]).min()))
            a1 = float(t["hi"].max())
            if not a1 > a0:
                a1 = a0 + 1e-6
            s = max(float(t["sigma"].min()) * 2, (a1 - a0) * 1e-6)
            self._axis = (a0, a1, s, math.asinh((a1 - a0) / s))
        return self._axis

    def u_of(self, y):
        a0, _a1, s, A = self.axis()
        return np.arcsinh((np.asarray(y, np.float64) - a0) / s) / A

    def y_of(self, u):
        a0, _a1, s, A = self.axis()
        return a0 + s * np.sinh(np.asarray(u, np.float64) * A)

    def px_of(self, y) -> float:
        r = self.plot_rect()
        return r.x() + float(self.u_of(y)) * r.width()

    def bins(self, n: int) -> np.ndarray:
        """(C, n) counts of the data along the axis."""
        if self._bins is None or self._bins.shape[1] != n:
            st, vals = self.st, self.stats.values
            out = np.zeros((self.stats.channels, n))
            for k in range(self.stats.channels):
                u = self.u_of(vals * st.gain[k] + st.shift[k])
                ok = (u >= 0) & (u < 1)
                idx = (u[ok] * n).astype(np.int64)
                out[k] = np.bincount(idx, weights=self.stats.hist[k][ok], minlength=n)[:n]
            self._bins = out
        return self._bins

    # ------------------------------------------------------------ drawing
    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        try:
            self._paint(p)
        finally:
            p.end()

    def _paint(self, p: QPainter):
        p.fillRect(self.rect(), QColor("#1c1f26"))
        r = self.plot_rect()
        p.fillRect(r, QColor("#12141a"))
        p.setPen(QPen(QColor("#2a2f38"), 1))
        p.drawRect(r)
        if self.stats is None or self.st is None:
            return
        p.setRenderHint(QPainter.Antialiasing)
        n = max(32, int(r.width()))
        b = self.bins(n)
        a0, a1, _s, _A = self.axis()
        # the ticks
        f = QFont(self.font())
        f.setPointSizeF(7.5)
        p.setFont(f)
        fm = QFontMetricsF(f)
        for x, text in spaced_ticks([(self.px_of(v), short_number(v)) for v in nice_ticks(a0, a1)],
                                    fm.horizontalAdvance, r.x() + 2, r.right() - 2):
            p.setPen(QPen(QColor(255, 255, 255, 18), 1))
            p.drawLine(QPointF(x, r.y()), QPointF(x, r.bottom()))
            p.setPen(QColor("#6b7280"))
            p.drawText(QRectF(x - 30, r.bottom() + 20, 60, 14), Qt.AlignHCenter | Qt.AlignTop, text)
        # the counts (logarithmic)
        top = float(np.log1p(b.max())) or 1.0
        chans = range(b.shape[0])
        cols = [MONO_COLOR] if b.shape[0] == 1 else list(CHANNEL_COLORS)
        order = [k for k in chans if k != self.channel] + ([self.channel] if self.channel >= 0 else [])
        for k in order:
            hk = np.log1p(b[k]) / top
            path = QPainterPath(QPointF(r.x(), r.bottom()))
            xs = r.x() + (np.arange(n) + 0.5) * r.width() / n
            ys = r.bottom() - hk * (r.height() - 4)
            for x, y in zip(xs, ys):
                path.lineTo(QPointF(float(x), float(y)))
            path.lineTo(QPointF(r.right(), r.bottom()))
            c = QColor(cols[k])
            faint = self.channel >= 0 and k != self.channel
            c.setAlpha(28 if faint else 70)
            p.setBrush(c)
            c2 = QColor(cols[k])
            c2.setAlpha(90 if faint else 230)
            p.setPen(QPen(c2, 1))
            p.drawPath(path)
        # the curve of the stretch
        k = self.channel if self.channel >= 0 else (1 if self.st.channels == 3 else 0)
        m = 160
        us = np.linspace(0, 1, m)
        ys = self.y_of(us)
        t = np.clip((ys - self.st.black[k]) / max(self.st.white[k] - self.st.black[k], 1e-30), 0, 1)
        if self.st.mode == "histogram":
            vals = self.stats.values
            curve = sm.curves(self.st, self.stats)[k]
            yv = vals * self.st.gain[k] + self.st.shift[k]
            out = np.interp(ys, yv, curve)
        else:
            out = np.clip(sm._curve(t, self.st, k), 0, 1)
        path = QPainterPath()
        for i, (u, o) in enumerate(zip(us, out)):
            pt = QPointF(r.x() + u * r.width(), r.bottom() - o * (r.height() - 4))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 170), 1.3))
        p.drawPath(path)
        # the handles
        for which, col, style in (("black", "#0b0c0f", Qt.DashLine), ("mid", "#9aa3b2", Qt.DotLine),
                                  ("white", "#f0f2f5", Qt.DashLine)):
            if which == "mid" and self.st.mode in ("linear", "histogram"):
                continue
            x = self.px_of(self._value(which))
            if r.x() - 1 <= x <= r.right() + 1:
                p.setPen(QPen(QColor(255, 255, 255, 70), 1, style))
                p.drawLine(QPointF(x, r.y()), QPointF(x, r.bottom()))
            x = float(np.clip(x, r.x(), r.right()))
            y = r.bottom() + 3
            tri = QPolygonF([QPointF(x, y), QPointF(x - 6, y + 11), QPointF(x + 6, y + 11)])
            p.setBrush(QColor(col))
            p.setPen(QPen(QColor(ACCENT) if self._drag == which else QColor("#8b93a1"), 1.2))
            p.drawPolygon(tri)
        if self._hover is not None:
            f.setPointSizeF(8)
            p.setFont(f)
            v = float(self.y_of((self._hover - r.x()) / r.width()))
            txt = value_text(v, self.stats) + (" " + self.units if self.units else "")
            p.setPen(QColor("#c9ced8"))
            p.drawText(QRectF(r.x() + 4, r.y() + 2, r.width() - 8, 14), Qt.AlignRight | Qt.AlignTop, txt)

    def _value(self, which: str) -> float:
        arr = getattr(self.st, which)
        return float(arr[self.channel] if self.channel >= 0 else np.mean(arr))

    # ------------------------------------------------------------ interaction
    def _pick(self, x: float) -> str | None:
        handles = ["black", "white"] + ([] if self.st.mode in ("linear", "histogram") else ["mid"])
        r = self.plot_rect()
        pos = {h: float(np.clip(self.px_of(self._value(h)), r.x(), r.right())) for h in handles}
        best = min(handles, key=lambda h: (abs(pos[h] - x), 0 if h == "mid" else 1))
        if abs(pos[best] - x) > 12:
            return None
        if "mid" in pos and abs(pos["mid"] - pos["black"]) < 6 and best in ("mid", "black"):
            best = "black" if x < pos["black"] else "mid"
        return best

    def mousePressEvent(self, e):  # noqa: N802
        if self.st is None or e.button() != Qt.LeftButton:
            return
        self._drag = self._pick(e.position().x())
        self.update()

    def mouseMoveEvent(self, e):  # noqa: N802
        if self.st is None:
            return
        r = self.plot_rect()
        x = float(np.clip(e.position().x(), r.x(), r.right()))
        self._hover = x if r.adjusted(0, 0, 0, 20).contains(e.position()) else None
        if self._drag:
            self.edited.emit(self._drag, float(self.y_of((x - r.x()) / r.width())))
        else:
            self.setCursor(Qt.SizeHorCursor if self._pick(e.position().x()) else Qt.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, e):  # noqa: N802
        if self._drag:
            self._drag = None
            self.released.emit()
        self.update()

    def leaveEvent(self, e):  # noqa: N802
        self._hover = None
        self.update()
        super().leaveEvent(e)


def value_text(v: float, stats: sm.Stats | None = None, integer: bool = False) -> str:
    """A data value with as many decimals as its range needs (a decimal comma in French)."""
    if integer:
        return number(v, 0)
    span = (stats.hi - stats.lo) if stats is not None else max(abs(v), 1.0)
    d = int(np.clip(4 - math.floor(math.log10(max(span, 1e-12))), 0 if span >= 1e5 else 1, 6))
    return number(v, d)


# ---------------------------------------------------------------------------------------------- the stretch panel
class StretchPanel(QFrame):
    """The display stretch: Auto, the mode, linked channels, neutral sky, the channel edited, the histogram with its
    handles, the three points as numbers, Reset. Display only: ``changed`` gives the new ``Stretch``."""

    changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ivpanel")
        self.setFixedWidth(312)
        self.prep: Prepared | None = None
        self.vi: ViewImage | None = None
        self.units = "adu"
        self.st: sm.Stretch | None = None
        self.channel = -1
        self._busy = False
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(label(_("Stretch"), "h2"))
        head.addStretch(1)
        self.reset_b = QPushButton(_("Reset"))
        self.reset_b.setObjectName("link")
        self.reset_b.setToolTip(_("Back to the automatic stretch"))
        self.reset_b.clicked.connect(self.reset)
        head.addWidget(self.reset_b)
        v.addLayout(head)
        row = QHBoxLayout()
        self.auto_b = QPushButton(_("Auto"))
        self.auto_b.setCheckable(True)
        self.auto_b.setObjectName("toggle")
        self.auto_b.setToolTip(_("Black {k} σ under the sky, white at the 99.99th percentile, the sky at a quarter "
                                 "(A)").format(k=number(sm.SHADOWS, 1)))
        self.auto_b.toggled.connect(self._auto)
        row.addWidget(self.auto_b)
        self.mode_cb = QComboBox()
        for key, text in (("linear", _("Linear")), ("mtf", _("Midtones (MTF)")), ("asinh", _("Asinh")),
                          ("histogram", _("Histogram equalisation"))):
            self.mode_cb.addItem(text, key)
        self.mode_cb.currentIndexChanged.connect(self._mode)
        row.addWidget(self.mode_cb, 1)
        v.addLayout(row)
        row = QHBoxLayout()
        self.linked_c = QCheckBox(_("Linked channels"))
        self.linked_c.setToolTip(_("The same black, mid and white for the three channels"))
        self.linked_c.toggled.connect(self._linked)
        row.addWidget(self.linked_c)
        self.neutral_c = QCheckBox(_("Neutral sky"))
        self.neutral_c.setToolTip(_("For an image still in the camera's colours: each channel scaled so that the sky "
                                    "is grey and the stars white (display only)"))
        self.neutral_c.toggled.connect(self._neutral)
        row.addWidget(self.neutral_c)
        row.addStretch(1)
        v.addLayout(row)
        self.chan_row = QWidget()
        cr = QHBoxLayout(self.chan_row)
        cr.setContentsMargins(0, 0, 0, 0)
        cr.setSpacing(0)
        self.chan_btns = []
        grp = QButtonGroup(self)
        for i, t in enumerate((_("RGB"), "R", "G", "B")):
            b = QPushButton(t)
            b.setCheckable(True)
            b.setObjectName("ivseg")
            b.clicked.connect(lambda _x=False, c=i - 1: self.set_channel(c))
            grp.addButton(b)
            cr.addWidget(b)
            self.chan_btns.append(b)
        cr.addStretch(1)
        v.addWidget(self.chan_row)
        self.hist = HistogramView()
        self.hist.edited.connect(self._edited)
        v.addWidget(self.hist)
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        self.fields = {}
        for i, (which, text) in enumerate((("black", _("Black")), ("mid", _("Mid")), ("white", _("White")))):
            grid.addWidget(label(text, "ivcap"), 0, i)
            f = QDoubleSpinBox()                 # with its arrows: without them the client's style sheet hides the text
            f.setObjectName("ivfield")
            f.setLocale(self.locale())
            f.setKeyboardTracking(False)
            f.valueChanged.connect(lambda val, w=which: self._field(w, val))
            grid.addWidget(f, 1, i)
            self.fields[which] = f
        v.addLayout(grid)
        self.clip_l = label("", "small")
        v.addWidget(self.clip_l)
        v.addSpacing(4)
        v.addWidget(label(_("Statistics"), "h2"))
        self.table = QGridLayout()
        self.table.setHorizontalSpacing(10)
        self.table.setVerticalSpacing(2)
        self.cells: dict[tuple[int, int], QLabel] = {}
        rows = (_("Minimum"), _("Median"), _("Noise σ"), _("99.99 %"), _("Maximum"))
        for r, t in enumerate(rows):
            self.table.addWidget(label(t, "small"), r + 1, 0)
        for c, (name, col) in enumerate(zip(("R", "G", "B"), CHANNEL_COLORS)):
            hd = QLabel(name)
            hd.setAlignment(Qt.AlignRight)
            hd.setStyleSheet(f"color: {col}; font-weight: 600; font-size: 9pt;")
            self.table.addWidget(hd, 0, c + 1)
            self.cells[(-1, c)] = hd
            for r in range(len(rows)):
                cell = QLabel("")
                cell.setAlignment(Qt.AlignRight)
                cell.setStyleSheet(f"color: #c9ced8; font-family: '{MONO_FONT}'; font-size: 8.5pt;")
                self.table.addWidget(cell, r + 1, c + 1)
                self.cells[(r, c)] = cell
        self.table.setColumnStretch(0, 1)
        v.addLayout(self.table)
        self.stats_l = label("", "small")
        self.stats_l.setWordWrap(True)
        v.addWidget(self.stats_l)
        note = label(_("Display only: the data are not changed."), "reason")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        self.setEnabled(False)

    # ------------------------------------------------------------ state
    def set_image(self, prep: Prepared | None, st: sm.Stretch | None, vi: ViewImage | None = None):
        self.prep = prep
        self.vi = vi or (prep.vi if prep is not None else None)
        self.units = self.vi.units if self.vi is not None else "adu"
        if prep is None or st is None:
            self.st = None
            self.setEnabled(False)
            self.hist.set_state(None, None)
            return
        self.setEnabled(True)
        if prep.channels == 1:
            self.channel = -1
        self.auto_b.setToolTip(_(sm.auto_text(self.vi.kind if self.vi is not None else "")).format(
            k=number(sm.SHADOWS, 1)))
        self.set_stretch(st)

    def set_stretch(self, st: sm.Stretch):
        """Show a stretch (no signal)."""
        self.st = st.copy()
        self._busy = True
        try:
            colour = self.prep is not None and self.prep.channels == 3
            self.auto_b.setChecked(st.auto)
            self.mode_cb.setCurrentIndex(max(0, self.mode_cb.findData(st.mode)))
            self.linked_c.setChecked(st.linked)
            self.linked_c.setVisible(colour)
            self.neutral_c.setChecked(st.neutral)
            self.neutral_c.setVisible(colour)
            if st.linked or not colour:
                self.channel = -1
            self.chan_row.setVisible(colour and not st.linked)
            for i, b in enumerate(self.chan_btns):
                b.setChecked(i - 1 == self.channel)
            self._fill_fields()
        finally:
            self._busy = False
        self.hist.set_state(self.prep.stats if self.prep else None, self.st, self.channel, self._units())
        self._numbers()

    def _units(self) -> str:
        if self.prep is None:
            return ""
        return "ADU" if self.units == "adu" else ""

    def _fill_fields(self):
        stats = self.prep.stats
        span = stats.hi - stats.lo
        dec = int(np.clip(4 - math.floor(math.log10(max(span, 1e-12))), 1, 6))
        lo = min(stats.lo - span, float(np.min(self.st.black)))        # a model's black at zero shows as zero
        hi = max(stats.hi + span, float(np.max(self.st.white)))
        for which, f in self.fields.items():
            f.blockSignals(True)
            f.setDecimals(dec)
            f.setRange(lo, hi)
            f.setSingleStep(10 ** -dec * 10)
            arr = getattr(self.st, which)
            f.setValue(float(arr[self.channel] if self.channel >= 0 else np.mean(arr)))
            f.setEnabled(not (which == "mid" and self.st.mode in ("linear", "histogram")))
            f.blockSignals(False)

    def _numbers(self):
        if self.prep is None or self.st is None:
            return
        stats = self.prep.stats
        u = self._units()
        integer = self.prep.integer
        for c in range(3):
            shown = c < stats.channels
            hd = self.cells[(-1, c)]
            hd.setVisible(shown)
            if stats.channels == 1:
                hd.setText(_("Value"))
                hd.setStyleSheet(f"color: {MONO_COLOR}; font-weight: 600; font-size: 9pt;")
            else:                                         # a colour image after a mono one in the same viewer
                hd.setText("RGB"[c])
                hd.setStyleSheet(f"color: {CHANNEL_COLORS[c]}; font-weight: 600; font-size: 9pt;")
            for r, arr in enumerate((stats.vmin, stats.median, stats.sigma, stats.high, stats.vmax)):
                cell = self.cells[(r, c)]
                cell.setVisible(shown)
                if shown:
                    exact = integer and r in (0, 4)
                    cell.setText(value_text(float(arr[c]), None if exact else stats, exact))
        self.stats_l.setText(_("Data units: {u}").format(u=u) if u else
                             _("Data units: 0 to 1") if self.units == "norm" else "")
        lo_f, hi_f = sm.clipped(self.st, stats)
        self.clip_l.setText(_("Clipped: {b} % black · {w} % white").format(b=number(100 * lo_f, 2),
                                                                            w=number(100 * hi_f, 2)))

    def set_channel(self, c: int):
        self.channel = c
        for i, b in enumerate(self.chan_btns):
            b.setChecked(i - 1 == c)
        self._busy = True
        try:
            self._fill_fields()
        finally:
            self._busy = False
        self.hist.set_state(self.prep.stats, self.st, c, self._units())
        self._numbers()

    def _apply(self, st: sm.Stretch):
        self.set_stretch(st)
        self.changed.emit(st.copy())

    def _auto_for(self, mode=None, linked=None, neutral=None) -> sm.Stretch:
        """The automatic stretch of the image's kind, with the choices shown (or the ones given)."""
        st = self.st
        return sm.auto_for(self.prep.stats, self.vi.kind if self.vi is not None else "", mode or st.mode,
                           st.linked if linked is None else linked, st.neutral if neutral is None else neutral)

    # ------------------------------------------------------------ the controls
    def reset(self):
        if self.prep is None:
            return
        self.channel = -1
        self._apply(default_stretch(self.prep, vi=self.vi))

    def _auto(self, on: bool):
        if self._busy or self.st is None:
            return
        if on:
            self._apply(self._auto_for())
        else:
            st = self.st.copy()
            st.auto = False
            self._apply(st)

    def _mode(self, _i):
        if self._busy or self.st is None:
            return
        mode = self.mode_cb.currentData()
        if self.st.auto:
            self._apply(self._auto_for(mode=mode))
            return
        st = self.st.copy()
        st.mode = mode
        if mode in ("mtf", "asinh"):                      # a mid point that fits the new curve
            ref = self._auto_for(mode=mode)
            for k in range(st.channels):
                t = (st.mid[k] - st.black[k]) / max(st.white[k] - st.black[k], 1e-30)
                if not (0 < t < (0.5 if mode == "asinh" else 1)):
                    st.mid[k] = st.black[k] + (ref.mid[k] - ref.black[k]) / max(ref.white[k] - ref.black[k], 1e-30) \
                        * (st.white[k] - st.black[k])
        self._apply(st)

    def _linked(self, on: bool):
        if self._busy or self.st is None:
            return
        if self.st.auto:
            self._apply(self._auto_for(linked=on))
            return
        st = self.st.copy()
        st.linked = on
        if on:
            st.black[:] = st.black.mean()
            st.mid[:] = st.mid.mean()
            st.white[:] = st.white.max()
        self.channel = -1
        self._apply(st)

    def _neutral(self, on: bool):
        if self._busy or self.st is None:
            return
        self._apply(self._auto_for(neutral=on))

    def _edited(self, which: str, value: float):
        if self.st is None:
            return
        self._apply(sm.manual(self.st, which, value, self.channel, self.prep.stats))

    def _field(self, which: str, value: float):
        if self._busy or self.st is None:
            return
        self._apply(sm.manual(self.st, which, value, self.channel, self.prep.stats))


# ---------------------------------------------------------------------------------------------- a sequence of frames
def time_text(t: str) -> str:
    """The time of day of an ISO date and time ("2026-08-16T23:41:05" -> "23:41:05")."""
    if not t:
        return ""
    return t[11:19] if len(t) >= 19 and t[10] in "T " else t


def exposure_text(t) -> str:
    """An exposure time: 1/1000 s under a second, else 15 s (a decimal comma in French); "" when unknown."""
    if t is None:
        return ""
    t = float(t)
    if t < 1:
        return f"1/{round(1 / t)} s" if t > 0 else "0 s"
    txt = f"{t:g}"
    return (txt.replace(".", ",") if language() == "fr" else txt) + " s"


class SideScroll(QScrollArea):
    """A row that scrolls sideways when the window is too narrow for it, instead of making the window wider (the
    frame's facts and the viewer's tools are wider than a small screen). A scroll bar under the row only when it is
    needed."""

    def __init__(self, inner: QWidget, parent=None):
        super().__init__(parent)
        self.setWidget(inner)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(240)
        self.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.viewport().setAutoFillBackground(False)
        inner.setAutoFillBackground(False)
        self._fit()

    def _fit(self):
        inner = self.widget()
        need = inner.minimumSizeHint().width() > self.viewport().width()
        h = inner.sizeHint().height() + (self.horizontalScrollBar().sizeHint().height() if need else 0)
        if self.height() != h or self.minimumHeight() != h:
            self.setFixedHeight(h)

    def resizeEvent(self, e):  # noqa: N802 - Qt's name
        super().resizeEvent(e)
        self._fit()

    def sizeHint(self):  # noqa: N802 - Qt's name
        return self.widget().sizeHint()                  # as wide as the row wants when there is room


class InfoBand(QFrame):
    """What the caller knows of the frame shown: name, time, exposure, ISO / gain, temperature, FWHM, roundness,
    transparency, kept or excluded, and Exclude / Include."""

    excludeClicked = Signal()

    CELLS = ("time", "exposure", "iso", "temperature", "fwhm", "roundness", "transparency", "weight")

    def __init__(self, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(18)
        col = QVBoxLayout()
        col.setSpacing(0)
        self.pos_l = label("", "ivcap")
        col.addWidget(self.pos_l)
        self.name_l = ElidedLabel("", "ivname")
        self.name_l.setMinimumWidth(120)
        col.addWidget(self.name_l)
        h.addLayout(col, 2)
        self.cells: dict[str, tuple[QLabel, QLabel]] = {}
        caps = {"time": _("Time"), "exposure": _("Exposure"), "iso": _("ISO"), "temperature": _("Temperature"),
                "fwhm": "FWHM", "roundness": _("Roundness"), "transparency": _("Transparency"),
                "weight": _("Weight")}
        for k in self.CELLS:
            c = QVBoxLayout()
            c.setSpacing(0)
            cap = label(caps[k], "ivcap")
            val = label("–", "ivval")
            c.addWidget(cap)
            c.addWidget(val)
            h.addLayout(c)
            self.cells[k] = (cap, val)
        self.extra = QHBoxLayout()
        self.extra.setSpacing(18)
        h.addLayout(self.extra)
        h.addStretch(1)
        self.state = QLabel("")
        self.state.setAlignment(Qt.AlignCenter)
        h.addWidget(self.state)
        self.toggle = QPushButton(_("Exclude"))
        self.toggle.setObjectName("ivstate")
        self.toggle.setToolTip(_("Leave this frame out of the stack, or take it back (X)"))
        self.toggle.clicked.connect(self.excludeClicked.emit)
        h.addWidget(self.toggle)
        self.mode = "on"                          # on | off (no toggle, no state) | spec (◇: shown, disabled)
        self.spec_tip = ""
        # why the engine flags the frame shown, written out (not only in a tooltip): a line under the numbers
        self.why = QLabel("")
        self.why.setWordWrap(True)
        self.why.setTextFormat(Qt.PlainText)
        self.why.hide()

    def set_mode(self, mode: str, tip: str = ""):
        """What Exclude / Include does here: ``on``; ``off`` (masters: neither the toggle nor the state); ``spec``
        (◇ the engine cannot do it yet: the toggle disabled, ``tip`` says why)."""
        self.mode = mode if mode in ("on", "off", "spec") else "on"
        self.spec_tip = tip
        self.toggle.setVisible(self.mode != "off")
        self.state.setVisible(self.mode != "off")

    def set_frame(self, vi: ViewImage, index: int, total: int):
        info = vi.info or {}
        self.pos_l.setText(_("Frame {i} of {n}").format(i=index + 1, n=total))
        self.name_l.setText(vi.name)
        vals = {
            "time": time_text(info.get("time") or ""),
            "exposure": exposure_text(info.get("exposure")),
            "iso": (number(info["iso"], 0) if info.get("iso") is not None else
                    number(info["gain"], 0) if info.get("gain") is not None else ""),
            "temperature": number(info["temperature"], 1) + " °C" if info.get("temperature") is not None else "",
            "fwhm": (number(info["fwhm"], 2) + " px" + (f"  ·  {number(info['fwhm_arcsec'], 1)}″"
                                                        if info.get("fwhm_arcsec") is not None else ""))
            if info.get("fwhm") is not None else "",
            "roundness": number(info["roundness"], 2) if info.get("roundness") is not None else "",
            "transparency": number(100 * info["transparency"], 0) + " %" if info.get("transparency") is not None
            else "",
            "weight": number(info["weight"], 2) if info.get("weight") is not None else "",
        }
        self.cells["iso"][0].setText(_("Gain") if info.get("iso") is None and info.get("gain") is not None
                                     else _("ISO"))
        compact = bool(info.get("compact"))      # only the cells with a value (a master, a calibration frame)
        for k, (cap, val) in self.cells.items():
            val.setText(vals[k] or "–")
            val.setStyleSheet("" if vals[k] else "color: #5d6470;")
            show = bool(vals[k]) or (k != "weight" and not compact)
            cap.setVisible(show)
            val.setVisible(show)
        while self.extra.count():
            it = self.extra.takeAt(0)
            lay = it.layout()
            if lay is not None:
                while lay.count():
                    w = lay.takeAt(0).widget()
                    if w is not None:
                        w.hide()                  # not drawn again under the next frame's numbers before it goes
                        w.deleteLater()
        for cap_t, text in info.get("extra") or []:
            c = QVBoxLayout()
            c.setSpacing(0)
            c.addWidget(label(cap_t, "ivcap"))
            c.addWidget(label(text, "ivval"))
            self.extra.addLayout(c)
        excluded = bool(info.get("excluded"))
        reason = info.get("reason") or ""
        if excluded or reason:
            self.state.setText(_("Excluded") if excluded else _("Left out"))
            self.state.setToolTip(reason)
            self.state.setStyleSheet(f"color: {BAD}; border: 1px solid {BAD}; border-radius: 9px; padding: 2px 10px;")
        else:
            self.state.setText(_("Included"))
            self.state.setToolTip("")
            self.state.setStyleSheet(f"color: {OK}; border: 1px solid {OK}; border-radius: 9px; padding: 2px 10px;")
        flag = info.get("flag")                  # a quality flag of the engine: red (bad) or amber, the reason below
        col = BAD if info.get("flag_level") == "bad" else WARN
        if flag and not (excluded or reason):
            self.state.setText(str(flag))
            self.state.setToolTip(info.get("flag_reason") or "")
            self.state.setStyleSheet(f"color: {col}; border: 1px solid {col}; border-radius: 9px; "
                                     "padding: 2px 10px;")
        why = str(info.get("flag_reason") or "") if flag else ""
        self.why.setText(("⚠  " + why) if why else "")
        self.why.setStyleSheet(f"color: {col}; font-weight: 600;")
        self.why.setVisible(bool(why))
        self.toggle.setText(_("Include") if excluded else _("Exclude"))
        if self.mode == "spec":
            self.toggle.setText(self.toggle.text() + "  ◇")
            self.toggle.setEnabled(False)
            self.toggle.setToolTip("◇ " + (self.spec_tip or _("Not in the engine yet.")))
        else:
            self.toggle.setEnabled(True)
            self.toggle.setToolTip(_("Leave this frame out of the stack, or take it back (X)"))


class FilmStrip(QWidget):
    """The frames as thumbnails, the one shown framed, the excluded ones marked; click to show one."""

    picked = Signal(int)
    visibleChanged = Signal(int, int)

    CELL_W, CELL_H, GAP = 104, 70, 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.CELL_H + 22)
        self.setMouseTracking(True)
        self.n = 0
        self.current = 0
        self.thumbs: dict[int, QImage] = {}
        self.excluded: set[int] = set()
        self.flagged: dict[int, str] = {}           # frame -> the engine's flag: "bad" (red) or "warn" (amber)
        self.offset = 0.0
        self.bar = QScrollBar(Qt.Horizontal, self)
        self.bar.valueChanged.connect(self._scrolled)
        self._hover = -1

    def set_count(self, n: int):
        self.n = n
        self.thumbs.clear()
        self.excluded.clear()
        self.flagged.clear()
        self._layout_bar()
        self.update()

    def set_thumb(self, i: int, arr: np.ndarray):
        if 0 <= i < self.n:
            a = np.ascontiguousarray(arr)
            self.thumbs[i] = to_qimage(a).copy()
            self.update()

    def set_excluded(self, i: int, on: bool):
        (self.excluded.add if on else self.excluded.discard)(i)
        self.update()

    def set_flagged(self, i: int, level: str | None):
        """Mark a frame the engine flags: "bad" (a red frame and corner), "warn" (amber), None (no mark)."""
        if level in ("bad", "warn"):
            self.flagged[i] = level
        else:
            self.flagged.pop(i, None)
        self.update()

    def set_current(self, i: int):
        self.current = i
        x = i * (self.CELL_W + self.GAP)
        if x < self.offset or x + self.CELL_W > self.offset + self.width():
            self.bar.setValue(int(max(0, x - (self.width() - self.CELL_W) / 2)))
        self.update()

    def visible_range(self) -> tuple[int, int]:
        step = self.CELL_W + self.GAP
        i0 = int(self.offset // step)
        return max(0, i0), min(self.n, i0 + int(self.width() // step) + 2)

    def _layout_bar(self):
        total = self.n * (self.CELL_W + self.GAP)
        self.bar.setGeometry(0, self.height() - 10, self.width(), 10)
        self.bar.setRange(0, max(0, int(total - self.width())))
        self.bar.setPageStep(max(1, self.width()))
        self.bar.setSingleStep(self.CELL_W + self.GAP)
        self.bar.setVisible(total > self.width())

    def _scrolled(self, v):
        self.offset = float(v)
        self.visibleChanged.emit(*self.visible_range())
        self.update()

    def resizeEvent(self, e):  # noqa: N802
        self._layout_bar()
        self.visibleChanged.emit(*self.visible_range())
        super().resizeEvent(e)

    def wheelEvent(self, e):  # noqa: N802
        d = e.angleDelta().y() or e.angleDelta().x()
        self.bar.setValue(self.bar.value() - int(d / 120 * (self.CELL_W + self.GAP)))

    def _at(self, x: float) -> int:
        step = self.CELL_W + self.GAP
        i = int((x + self.offset) // step)
        return i if 0 <= i < self.n and (x + self.offset) % step <= self.CELL_W else -1

    def mousePressEvent(self, e):  # noqa: N802
        i = self._at(e.position().x())
        if i >= 0 and e.position().y() < self.CELL_H + 8:
            self.picked.emit(i)

    def mouseMoveEvent(self, e):  # noqa: N802
        i = self._at(e.position().x())
        if i != self._hover:
            self._hover = i
            self.update()

    def leaveEvent(self, e):  # noqa: N802
        self._hover = -1
        self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#16181e"))
        f = QFont(self.font())
        f.setPointSizeF(7.5)
        p.setFont(f)
        step = self.CELL_W + self.GAP
        i0, i1 = self.visible_range()
        for i in range(i0, i1):
            x = i * step - self.offset + 3
            cell = QRectF(x, 4, self.CELL_W, self.CELL_H)
            p.fillRect(cell, QColor("#0f1115"))
            qi = self.thumbs.get(i)
            if qi is not None:
                fw, fh = qi.width(), qi.height()
                k = min(cell.width() / fw, cell.height() / fh)
                tr = QRectF(cell.x() + (cell.width() - fw * k) / 2, cell.y() + (cell.height() - fh * k) / 2,
                            fw * k, fh * k)
                p.setRenderHint(QPainter.SmoothPixmapTransform)
                p.drawImage(tr, qi)
            ex = i in self.excluded
            fl = self.flagged.get(i)
            if fl is not None:                    # flagged by the engine: a coloured frame and a corner mark
                fc = QColor(BAD if fl == "bad" else WARN)
                p.setPen(QPen(fc, 2))
                p.setBrush(Qt.NoBrush)
                p.drawRect(cell.adjusted(1, 1, -1, -1))
                p.setPen(Qt.NoPen)
                p.setBrush(fc)
                p.drawPolygon(QPolygonF([QPointF(cell.x(), cell.y()), QPointF(cell.x() + 18, cell.y()),
                                         QPointF(cell.x(), cell.y() + 18)]))
                p.setPen(QColor("#ffffff"))
                p.drawText(QRectF(cell.x() + 1, cell.y(), 9, 10), Qt.AlignCenter, "!")
            if ex:
                p.fillRect(cell, QColor(15, 17, 21, 150))
                p.setPen(QPen(QColor(BAD), 2))
                p.drawLine(QPointF(cell.right() - 14, cell.y() + 4), QPointF(cell.right() - 4, cell.y() + 14))
                p.drawLine(QPointF(cell.right() - 4, cell.y() + 4), QPointF(cell.right() - 14, cell.y() + 14))
            if i == self.current:
                p.setPen(QPen(QColor(ACCENT), 2))
                p.setBrush(Qt.NoBrush)
                p.drawRect(cell.adjusted(-1, -1, 1, 1))
            elif i == self._hover:
                p.setPen(QPen(QColor("#8b93a1"), 1))
                p.setBrush(Qt.NoBrush)
                p.drawRect(cell)
            tag_r = QRectF(cell.x() + 3, cell.bottom() - 16, 30, 13)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 160))
            p.drawRoundedRect(tag_r, 3, 3)
            p.setPen(QColor(BAD) if ex else QColor("#d8dce4"))
            p.drawText(tag_r, Qt.AlignCenter, str(i + 1))
        p.end()


# ---------------------------------------------------------------------------------------------- the viewer
def _sep() -> QFrame:
    f = QFrame()
    f.setObjectName("ivsep")
    f.setFixedSize(1, 20)
    return f


class ImageView(QWidget):
    """The common image viewer. See the module's docstring; the API the views call:

    Images: ``set_image(ViewImage | array, keep_view=False, keep_stretch=False)``, ``image()``, ``prepared()``,
    ``clear()``. Compare: ``set_compare_targets([ViewImage, ...], current=None)``, ``set_compare(index | None)``,
    ``set_compare_mode("split" | "side")``, ``set_same_stretch(bool)``, ``compare_alignment()`` and
    ``alignment_note()`` (how the compared image was put on this one's grid). Sequence: ``set_sequence([ViewImage, ...],
    index=0)``, ``set_index(i)``, ``index()``, ``next_frame()``, ``previous_frame()``, ``set_frame_info(i, info)``,
    ``set_excluded(i, bool)``, ``set_lock_stretch(bool)``. Stretch: ``stretch()``, ``set_stretch(Stretch)``,
    ``auto_stretch()``, ``show_histogram(bool)``. View: ``fit()``, ``zoom_to(scale, cx, cy)``,
    ``set_zoom_percent(p)``, ``view_state()``, ``set_view_state(state)``. Actions: ``export_dialog()``,
    ``export(path, fmt, content, area)``, ``copy_view()``. Readout: ``pixel_text(x, y)``. Overlays: ``add_overlay(fn)``
    (fn(painter, canvas): ``canvas.img_to_w(x, y)``, ``canvas.scale``). Tests: ``wait(timeout)``.

    Signals: ``imageShown(key)``, ``viewChanged()``, ``stretchChanged(Stretch)``, ``hovered(x, y)``,
    ``frameChanged(i)``, ``excludeToggled(i, excluded)``, ``compareChanged(key | None)``, ``sendToPost(key)``,
    ``toasted(text)``, ``rectDone(x0, y0, x1, y1)``.
    """

    imageShown = Signal(object)
    viewChanged = Signal()
    stretchChanged = Signal(object)
    hovered = Signal(float, float)
    frameChanged = Signal(int)
    excludeToggled = Signal(int, bool)
    compareChanged = Signal(object)
    sendToPost = Signal(object)
    toasted = Signal(str)
    rectDone = Signal(float, float, float, float)

    def __init__(self, parent=None, *, loader: Loader | None = None, histogram: bool = True, post: bool = False,
                 export: bool = True, copy: bool = True, compare: bool = True):
        super().__init__(parent)
        self.setStyleSheet(QSS)
        self.loader = loader or default_loader()
        self.export_dir: str = ""                 # where Export… proposes to write (default: the Pictures folder)
        self._vi: ViewImage | None = None
        self._prep: Prepared | None = None
        self._main: Display | None = None
        self._st: sm.Stretch | None = None
        self._want = None                         # the key of the image asked for
        self._failed_key = None                   # the key of the last image that could not be read
        self._keep = (False, False)               # keep_view, keep_stretch of the image asked for
        self._targets: list[ViewImage] = []
        self._cmp_i: int | None = None
        self._cmp: Display | None = None
        self._cmp_mode = "split"
        self._same = False
        self._frames: list[ViewImage] = []
        self._index = 0
        self._lock = False                        # "Same stretch for all frames": off, each frame its own stretch
        self._locked: dict[str, sm.Stretch] = {}  # with it on: kind -> the stretch its frames are shown with
        self._st_kind: str | None = None          # the kind of the image the stretch was made for
        self._units = "adu"
        self._syncing = False
        self._shown_at: float | None = None
        self.times: dict[str, float] = {}         # measurements: open (s)
        self.loader.ready.connect(self._ready)
        self.loader.failed.connect(self._failed)
        self.loader.thumbReady.connect(self._thumb)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        # the tools in a row that scrolls sideways in a narrow window (they would widen it otherwise)
        v.addWidget(SideScroll(self._toolbar(post, export, copy, compare)))
        self.align_bar = QFrame()                  # one line: how the compared image was put on this one's grid
        self.align_bar.setObjectName("ivalign")
        ab = QHBoxLayout(self.align_bar)
        ab.setContentsMargins(12, 2, 12, 3)
        self.align_l = ElidedLabel("", "ivalign", Qt.ElideRight)
        ab.addWidget(self.align_l, 1)
        self.align_bar.hide()
        v.addWidget(self.align_bar)
        self._al: Alignment | None = None
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.area = QWidget()
        ah = QHBoxLayout(self.area)
        ah.setContentsMargins(0, 0, 0, 0)
        ah.setSpacing(2)
        self.canvas2 = ImageCanvas()               # the compared image, side by side
        self.canvas2.hide()
        self.canvas = ImageCanvas()
        ah.addWidget(self.canvas2, 1)
        ah.addWidget(self.canvas, 1)
        body.addWidget(self.area, 1)
        self.panel = StretchPanel()
        self.panel.changed.connect(self._panel_changed)
        # in a column that scrolls when the viewer is short (squeezed, its rows would run into each other)
        self.panel_box = QScrollArea()
        self.panel_box.setWidget(self.panel)
        self.panel_box.setWidgetResizable(True)
        self.panel_box.setFrameShape(QFrame.NoFrame)
        self.panel_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel_box.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.panel_box.setFixedWidth(self.panel.width() + self.panel_box.verticalScrollBar().sizeHint().width())
        self.panel_box.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.panel_box.viewport().setAutoFillBackground(False)
        self.panel.setVisible(histogram)
        self.panel_box.setVisible(histogram)
        body.addWidget(self.panel_box)
        v.addLayout(body, 1)
        v.addWidget(self._readout_bar())
        v.addWidget(self._sequence_bar())
        self.hist_b.setChecked(histogram)
        for c in (self.canvas, self.canvas2):
            c.hovered.connect(lambda x, y, which, cv=c: self._hover(cv, x, y, which))
            c.left.connect(self._left)
            c.viewChanged.connect(lambda cv=c: self._canvas_moved(cv))
            c.zoomed.connect(self._zoom_label)
        self.canvas.rectDone.connect(self.rectDone.emit)
        self._shortcuts()
        self._blink = QTimer(self)
        self._blink.setInterval(200)
        self._blink.timeout.connect(self._blink_tick)
        self._update_actions()

    # ------------------------------------------------------------ building
    def _toolbar(self, post, export, copy, compare) -> QFrame:
        bar = QFrame()
        bar.setObjectName("ivbar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(6)

        def tool(text, tip, slot, checkable=False):
            b = QPushButton(text)
            b.setObjectName("toggle" if checkable else "ivtool")
            b.setCheckable(checkable)
            b.setToolTip(tip)
            (b.toggled if checkable else b.clicked).connect(slot)
            h.addWidget(b)
            return b
        self.fit_b = tool(_("Fit"), _("The whole image (F)"), self.fit)
        self.one_b = tool("1:1", _("One image pixel per screen pixel (1)"), lambda: self.set_zoom_percent(100))
        self.zoom_b = QPushButton("100 %")
        self.zoom_b.setObjectName("ivtool")
        self.zoom_b.setToolTip(_("Zoom (wheel, + and −)"))
        self.zoom_b.setMinimumWidth(66)
        menu = QMenu(self.zoom_b)
        menu.addAction(_("Fit"), self.fit)
        for z in ZOOM_PRESETS:
            menu.addAction(f"{number(z * 100, 1 if z * 100 % 1 else 0)} %",
                           lambda zz=z: self.set_zoom_percent(zz * 100))
        self.zoom_b.setMenu(menu)
        h.addWidget(self.zoom_b)
        h.addSpacing(6)
        h.addWidget(_sep())
        h.addSpacing(6)
        self.cmp_box = QWidget()
        ch = QHBoxLayout(self.cmp_box)
        ch.setContentsMargins(0, 0, 0, 0)
        ch.setSpacing(6)
        ch.addWidget(label(_("Compare with"), "dim"))
        self.cmp_cb = QComboBox()
        self.cmp_cb.setMinimumWidth(150)
        self.cmp_cb.setMaximumWidth(330)
        # as wide as 26 characters: a long name (long in French) would make the toolbar, so the window, wider than
        # the screen; the list that opens shows the names whole
        self.cmp_cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmp_cb.setMinimumContentsLength(26)
        self.cmp_cb.currentIndexChanged.connect(self._cmp_picked)
        ch.addWidget(self.cmp_cb)
        self.cmp_opts = QWidget()                  # shown once an image is chosen
        oh = QHBoxLayout(self.cmp_opts)
        oh.setContentsMargins(0, 0, 0, 0)
        oh.setSpacing(6)
        seg = QHBoxLayout()
        seg.setSpacing(0)
        self.split_b = QPushButton(_("Split"))
        self.side_b = QPushButton(_("Side by side"))
        grp = QButtonGroup(self)
        for b, mode, tip in ((self.split_b, "split", _("Before on the left of the divider, after on the right")),
                             (self.side_b, "side", _("The two images next to each other, zoom and pan linked"))):
            b.setObjectName("ivseg")
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _x=False, m=mode: self.set_compare_mode(m))
            grp.addButton(b)
            seg.addWidget(b)
        self.split_b.setChecked(True)
        oh.addLayout(seg)
        self.same_c = QCheckBox(_("Same stretch"))
        self.same_c.setToolTip(_("Show the compared image with the same stretch as this one"))
        self.same_c.toggled.connect(self.set_same_stretch)
        oh.addWidget(self.same_c)
        ch.addWidget(self.cmp_opts)
        self.cmp_opts.hide()
        ch.addSpacing(6)
        ch.addWidget(_sep())
        h.addWidget(self.cmp_box)
        self.cmp_box.setVisible(False)
        self._compare_allowed = compare
        h.addSpacing(6)
        self.auto_b = tool(_("Auto-stretch"), _("Display only: the data stay linear (A)"), self._auto_toggled, True)
        self.hist_b = tool(_("Histogram"), _("The stretch and the histogram (H)"), self.show_histogram, True)
        h.addStretch(1)
        self.export_b = tool(_("Export…"), _("Write the image to a file: FITS, TIFF, PNG or JPEG (Ctrl+E)"),
                             self.export_dialog)
        self.export_b.setVisible(export)
        self.post_b = tool(_("Send to Post"), _("Use this image in Post"), self._send_post)
        self.post_b.setVisible(post)
        self.copy_b = tool(_("Copy image"), _("Copies the picture as it is shown here (its zoom, stretch and overlays) "
                                              "to the clipboard, to paste it into a message or another program "
                                              "(Ctrl+C)"), self.copy_view)
        self.copy_b.setVisible(copy)
        return bar

    def _readout_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("ivread")
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 3, 10, 3)
        h.setSpacing(10)
        self.readout = QLabel("")
        self.readout.setObjectName("ivread")
        self.readout.setTextFormat(Qt.PlainText)
        h.addWidget(self.readout, 1)
        self.size_l = label("", "small")
        h.addWidget(self.size_l)
        self.units_cb = QComboBox()
        self.units_cb.addItem("ADU", "adu")
        self.units_cb.addItem("0–1", "norm")
        self.units_cb.setToolTip(_("The values in the readout: ADU, or 0 to 1 (the saturation level is 1)"))
        self.units_cb.currentIndexChanged.connect(lambda _i: self._set_units(self.units_cb.currentData()))
        h.addWidget(self.units_cb)
        return bar

    def _sequence_bar(self) -> QFrame:
        self.seq = QFrame()
        self.seq.setObjectName("ivseq")
        v = QVBoxLayout(self.seq)
        v.setContentsMargins(10, 6, 10, 4)
        v.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.prev_b = QPushButton("◂")
        self.prev_b.setObjectName("ivtool")
        self.prev_b.setToolTip(_("Previous frame (←)"))
        self.prev_b.clicked.connect(self.previous_frame)
        row.addWidget(self.prev_b)
        self.num = QSpinBox()                     # with its arrows (see StretchPanel)
        self.num.setMinimum(1)
        self.num.setKeyboardTracking(False)
        self.num.setAlignment(Qt.AlignRight)
        self.num.setFixedWidth(72)
        self.num.setToolTip(_("Frame number"))
        self.num.valueChanged.connect(lambda n: self.set_index(n - 1))
        row.addWidget(self.num)
        self.total_l = label("", "dim")
        row.addWidget(self.total_l)
        self.next_b = QPushButton("▸")
        self.next_b.setObjectName("ivtool")
        self.next_b.setToolTip(_("Next frame (→)"))
        self.next_b.clicked.connect(self.next_frame)
        row.addWidget(self.next_b)
        self.blink_b = QPushButton("▶  " + _("Blink"))
        self.blink_b.setObjectName("toggle")
        self.blink_b.setCheckable(True)
        self.blink_b.setToolTip(_("Plays the frames at 5 frames/s (Space)"))
        self.blink_b.toggled.connect(self._blink_toggled)
        row.addWidget(self.blink_b)
        self.lock_c = QCheckBox(_("Same stretch for all frames"))
        self.lock_c.setToolTip(_("Keeps the stretch of the frame shown for the next frames of the same kind, so that "
                                 "blinking shows what changes. Off: each frame gets its own automatic stretch."))
        self.lock_c.setChecked(False)
        self.lock_c.toggled.connect(self.set_lock_stretch)
        row.addWidget(self.lock_c)
        row.addSpacing(14)
        self.band = InfoBand()
        self.band.excludeClicked.connect(self.toggle_excluded)
        row.addWidget(SideScroll(self.band), 1)       # never wider than the window: it scrolls sideways instead
        for w in (self.band.state, self.band.toggle):  # Included / Exclude stay in view, out of the scrolled row
            self.band.layout().removeWidget(w)
            row.addWidget(w)
        v.addLayout(row)
        v.addWidget(self.band.why)
        self.strip = FilmStrip()
        self.strip.picked.connect(self.set_index)
        self.strip.visibleChanged.connect(self._want_thumbs)
        v.addWidget(self.strip)
        self.seq.hide()
        return self.seq

    def _shortcuts(self):
        def sc(keys, fn):
            for k in keys:
                s = QShortcut(QKeySequence(k), self)
                s.setContext(Qt.WidgetWithChildrenShortcut)
                s.activated.connect(fn)
        sc(("F", "Ctrl+0"), self.fit)
        sc(("1", "Ctrl+1"), lambda: self.set_zoom_percent(100))
        sc(("2",), lambda: self.set_zoom_percent(200))
        sc(("+", "=", "Ctrl++"), lambda: self.canvas.zoom_by(ZOOM_STEP))
        sc(("-", "Ctrl+-"), lambda: self.canvas.zoom_by(1 / ZOOM_STEP))
        sc(("Right", "PgDown"), self.next_frame)
        sc(("Left", "PgUp"), self.previous_frame)
        sc(("Home",), lambda: self.set_index(0))
        sc(("End",), lambda: self.set_index(len(self._frames) - 1))
        sc(("X",), self.toggle_excluded)
        sc(("Space",), lambda: self.blink_b.toggle() if self._frames else None)
        sc(("A",), lambda: self.auto_b.toggle())
        sc(("H",), lambda: self.hist_b.toggle())
        sc(("Ctrl+C",), self.copy_view)
        sc(("Ctrl+E",), self.export_dialog)

    # ------------------------------------------------------------ images
    def set_image(self, image: ViewImage | np.ndarray, keep_view: bool = False, keep_stretch: bool = False):
        """Show an image (prepared in a worker thread; at once when the loader has it)."""
        vi = image if isinstance(image, ViewImage) else ViewImage(image)
        self._vi = vi
        self._want = vi.identity()
        self._keep = (keep_view, keep_stretch)
        self._shown_at = time.perf_counter()
        hit = self.loader.cached(self._want)
        if hit is not None:
            self._show(hit)
            return
        self.canvas.busy = self.canvas.display is not None
        self.canvas.message = "" if self.canvas.display is not None else _("Loading the image…")
        self.canvas.update()
        self.loader.request(vi, 0)

    def image(self) -> ViewImage | None:
        return self._vi

    def prepared(self) -> Prepared | None:
        return self._prep

    def clear(self):
        self._vi = self._prep = self._main = None
        self._want = None
        self.canvas.set_display(None)
        self.canvas.message = ""
        self.panel.set_image(None, None)
        self.readout.setText("")
        self.size_l.setText("")
        self._update_actions()

    def _ready(self, key, prep: Prepared):
        if key == self._want:
            self._show(prep)
        if self._cmp_i is not None and self._cmp_i < len(self._targets) and \
                self._targets[self._cmp_i].identity() == key:
            self._show_compare(prep)
        if self._frames:
            self._prefetch()

    def _failed(self, key, why: str):
        if key == self._want:
            self._failed_key = key
            self.canvas.busy = False
            self.canvas.message = why if key in self.loader.unavailable else \
                _("This image cannot be shown: {why}").format(why=why)
            if not self._frames:
                self.canvas.set_display(None)
            self.canvas.update()
        if self._cmp_i is not None and self._cmp_i < len(self._targets) and \
                self._targets[self._cmp_i].identity() == key:
            self._cmp_failed = key                  # the compared image cannot be read: said, and not waited for
            text = why if key in self.loader.unavailable else \
                _("This image cannot be shown: {why}").format(why=why)
            self.canvas2.message = text
            self.canvas2.update()
            self.toasted.emit(text)

    _cmp_failed = None

    def _show(self, prep: Prepared):
        """Show a prepared image. Its stretch: the automatic one of its kind, computed for it (the mode and the linked
        channels the user chose for the image before, when it was of the same kind); with "Same stretch for all
        frames", the stretch the frames of its kind were shown with (each kind its own: a dark's stretch never shows a
        flat)."""
        keep_view, keep_stretch = self._keep
        self._prep = prep
        old = self._st
        kind = kind_of(self._vi)
        same = old is not None and self._st_kind == kind
        locked = self._locked.get(kind) if self._frames and self._lock else None
        if locked is not None:
            st = locked.adapted(prep.channels)
        elif same and keep_stretch and not self._frames:
            st = old.adapted(prep.channels)
        elif same and old.auto:
            st = default_stretch(prep, old.mode, old.linked if prep.channels > 1 and old.channels > 1 else None, None,
                                 self._vi)
        else:
            st = default_stretch(prep, vi=self._vi)
        self._st = st
        self._st_kind = kind
        if self._frames and self._lock and locked is None:
            self._locked[kind] = st.copy()
        if self._main is not None:
            self._main.release()
        self._main = Display(prep, st)
        self.canvas.busy = False
        self.canvas.message = ""
        self.canvas.set_display(self._main, keep_view=keep_view or bool(self._frames))
        if self._cmp is not None:
            self._align()                           # the compared image on the grid of this one
        if self._cmp is not None and self._cmp_mode == "split":
            self.canvas.set_compare(self._cmp)
        self.panel.set_image(prep, st, self._vi)
        self._sync_auto_button()
        self._apply_same()
        self.size_l.setText(self._size_text(prep))
        self.units_cb.blockSignals(True)
        self._units = self._vi.units if self._vi.units in ("adu", "norm") else "adu"
        self.units_cb.setCurrentIndex(0 if self._units == "adu" else 1)
        self.units_cb.blockSignals(False)
        self._update_actions()
        self._names()
        self.canvas.repaint()
        if self._shown_at is not None:
            self.times["open"] = time.perf_counter() - self._shown_at
            self._shown_at = None
        self.imageShown.emit(prep.key)

    def _size_text(self, prep: Prepared) -> str:
        d = prep.data
        kind = (_("Bayer {cfa}").format(cfa=prep.cfa) if prep.cfa else
                _("mono") if prep.channels == 1 else _("RGB"))
        depth = (_("{n}-bit float").format(n=d.dtype.itemsize * 8) if np.issubdtype(d.dtype, np.floating)
                 else _("{n}-bit").format(n=d.dtype.itemsize * 8))
        return f"{prep.width} × {prep.height}  ·  {kind}  ·  {depth}"

    def _update_actions(self):
        has = self._prep is not None
        for b in (self.export_b, self.post_b, self.copy_b, self.fit_b, self.one_b, self.zoom_b, self.auto_b):
            b.setEnabled(has)
        if not self._post_ok:
            self.post_b.setEnabled(False)
        self.post_b.setToolTip(_("Use this image in Post") if self._post_ok else self._post_tip)
        self.cmp_box.setVisible(self._compare_allowed and bool(self._targets))

    # ------------------------------------------------------------ the stretch
    def stretch(self) -> sm.Stretch | None:
        return self._st.copy() if self._st is not None else None

    def set_stretch(self, st: sm.Stretch, emit: bool = False):
        """Show the image through ``st`` (a ``stretch.Stretch``, e.g. another view's)."""
        if self._main is None:
            self._st = st.copy()
            return
        self._st = st.adapted(self._main.prep.channels)
        self._main.set_stretch(self._st)
        self.panel.set_stretch(self._st)
        self._sync_auto_button()
        self._keep_locked()
        self._apply_same()
        self.canvas.update()
        self.canvas2.update()
        if emit:
            self.stretchChanged.emit(self._st.copy())

    def _keep_locked(self):
        """With "Same stretch for all frames", the stretch now shown is the one of the frames of its kind."""
        if self._frames and self._lock and self._st is not None and self._prep is not None:
            self._locked[kind_of(self._vi)] = self._st.copy()

    def _kind_auto(self, mode: str | None, linked: bool | None, neutral: bool | None) -> sm.Stretch:
        return sm.auto_for(self._prep.stats, self._vi.kind if self._vi is not None else "", mode, linked, neutral)

    def auto_stretch(self):
        """The automatic stretch of the image's kind again, with the Neutral sky choice the image had (after
        Auto-stretch off, which shows the data linear and plain, the choice made before it: once, turning Auto-stretch
        off and on again switched a stack shown with a neutral sky to its camera colours)."""
        if self._prep is not None:
            st = self._st
            neutral = st.neutral
            off = st.mode == "linear" and not st.auto           # after Auto-stretch off: the kind's own stretch
            if off and self._neutral_before is not None:
                neutral = self._neutral_before
            self.set_stretch(self._kind_auto(None if off else st.mode, None if off else st.linked, neutral), emit=True)

    _neutral_before: bool | None = None

    def set_neutral(self, on: bool):
        """Neutral sky on or off (display only): the automatic stretch with or without it."""
        if self._prep is None or self._st is None or bool(self._st.neutral) == bool(on):
            return
        st = self._st
        self.set_stretch(self._kind_auto(st.mode if st.mode != "linear" or st.auto else None, st.linked, bool(on)),
                         emit=True)

    def _panel_changed(self, st: sm.Stretch):
        self._st = st.adapted(self._main.prep.channels) if self._main else st
        if self._main is not None:
            self._main.set_stretch(self._st)
        self._sync_auto_button()
        self._keep_locked()
        self._apply_same()
        self.canvas.update()
        self.canvas2.update()
        self.stretchChanged.emit(self._st.copy())

    def _auto_toggled(self, on: bool):
        if self._prep is None or self._busy_auto:
            return
        if on:
            self.auto_stretch()
            return
        self._neutral_before = bool(self._st.neutral) if self._st is not None else None
        stats = self._prep.stats
        c = self._prep.channels
        st = sm.Stretch("linear", np.full(c, float(stats.vmin.min())), np.full(c, 0.5 * (stats.lo + stats.hi)),
                        np.full(c, float(stats.vmax.max())), np.ones(c), np.zeros(c), True, False, False)
        self.set_stretch(st, emit=True)

    _busy_auto = False

    def _sync_auto_button(self):
        self._busy_auto = True
        try:
            st = self._st
            self.auto_b.setChecked(bool(st is not None and (st.auto or st.mode != "linear")))
        finally:
            self._busy_auto = False

    def show_histogram(self, on: bool):
        self.panel.setVisible(bool(on))
        self.panel_box.setVisible(bool(on))
        if self.hist_b.isChecked() != bool(on):
            self.hist_b.setChecked(bool(on))

    # ------------------------------------------------------------ the view
    def fit(self):
        self.canvas.fit()
        if self.canvas2.isVisible():
            self.canvas2.fit()

    def zoom_to(self, scale: float, cx: float | None = None, cy: float | None = None):
        self.canvas.zoom_to(scale, cx, cy)

    def set_zoom_percent(self, pct: float):
        self.canvas.set_zoom_percent(pct)

    def view_state(self) -> dict:
        """Where the view is, independent of the image size: zoom relative to the image width, centre in fractions."""
        c = self.canvas
        if c.display is None:
            return {"fitted": True}
        w, h = c.display.size()
        out = {"fitted": c.fitted, "zoom": c.scale * w, "fx": c.cx / w, "fy": c.cy / h}
        pl = getattr(c.display, "place", None)
        if pl is not None:
            # the same point of the common grid (a final image is a crop of the stack at another scale)
            m = pl.m
            out["grid"] = (pl.frame, m[0] * c.cx + m[1] * c.cy + m[2], m[3] * c.cx + m[4] * c.cy + m[5],
                           c.scale / max(abs(m[0]), 1e-12))
        return out

    def set_view_state(self, state: dict):
        c = self.canvas
        pl = getattr(c.display, "place", None) if c.display is not None else None
        g = state.get("grid")
        if pl is not None and g is not None and g[0] == pl.frame and not state.get("fitted", True):
            # back from the common grid to this image's pixels
            inv = np.linalg.inv(pl.matrix())
            x, y, _one = inv @ np.array([g[1], g[2], 1.0])
            c.zoom_to(g[3] * abs(pl.m[0]), float(x), float(y))
            return
        _mirror_state(c, state)

    def _canvas_moved(self, cv: ImageCanvas):
        self._zoom_label()
        if self._syncing:
            return
        self._syncing = True
        try:
            if self.canvas2.isVisible():
                other = self.canvas2 if cv is self.canvas else self.canvas
                _mirror_state(other, _state_of(cv))
        finally:
            self._syncing = False
        if cv is self.canvas or self.canvas2.isVisible():
            self.viewChanged.emit()

    def _zoom_label(self):
        self.zoom_b.setText(number(self.canvas.zoom_percent(), 0 if self.canvas.zoom_percent() >= 10 else 1) + " %")

    # ------------------------------------------------------------ readout
    def _set_units(self, u: str):
        self._units = u

    def pixel_text(self, x: float, y: float, prep: Prepared | None = None, vi: ViewImage | None = None) -> str:
        """The readout of the image point (x, y): position, values (ADU or 0-1), RA / Dec."""
        if prep is None:
            prep, vi = self._prep, self._vi
        if prep is None:
            return ""
        xi, yi = int(math.floor(x)), int(math.floor(y))
        got = prep.value(xi, yi)
        if got is None:
            return ""
        vals, letter = got
        vi = vi or prep.vi
        if vi.stretched:                           # a picture for the eye: its levels, without a unit
            text = "   ".join(f"{c} {int(round(v))}" for c, v in zip("RGB", vals)) if len(vals) == 3 else \
                str(int(round(vals[0])))
            return f"x {xi:>5}  y {yi:>5}    {text}" + self._sky_part(prep, x, y)
        if vi.units == "count":                    # a count per pixel (a rejection map): whole numbers
            text = str(int(round(vals[0]))) if len(vals) == 1 else \
                "   ".join(f"{c} {int(round(v))}" for c, v in zip("RGB", vals))
            return f"x {xi:>5}  y {yi:>5}    {text}" + self._sky_part(prep, x, y)
        if vi.units not in ("adu", "norm"):        # a number without a unit (a coverage map)
            fmt = [value_text(v, prep.stats) for v in vals]
            text = "   ".join(f"{c} {v}" for c, v in zip("RGB", fmt)) if len(vals) == 3 else fmt[0]
            return f"x {xi:>5}  y {yi:>5}    {text}" + self._sky_part(prep, x, y)
        white = vi.white_level()
        if vi.units == "norm" and self._units == "adu":
            vals, unit = [v * white for v in vals], "ADU"
        elif vi.units != "norm" and self._units == "norm":
            vals, unit = [v / white for v in vals], ""
        else:
            unit = "ADU" if self._units == "adu" else ""
        integer = prep.integer and self._units == "adu" and vi.units != "norm"
        if self._units == "norm":
            fmt = [number(v, 5) for v in vals]
        else:
            fmt = [value_text(v, None if integer else prep.stats, integer) for v in vals]
        if len(vals) == 3:
            text = "   ".join(f"{c} {v}" for c, v in zip("RGB", fmt))
        elif letter:
            text = f"{letter} {fmt[0]}"
        else:
            text = fmt[0]
        return f"x {xi:>5}  y {yi:>5}    {text}" + (f" {unit}" if unit else "") + self._sky_part(prep, x, y)

    @staticmethod
    def _sky_part(prep: Prepared, x: float, y: float) -> str:
        if prep.wcs is None:
            return ""
        try:
            ra, dec = prep.wcs.pixel_to_world_values(x - 0.5, y - 0.5)
            if np.isfinite(ra) and np.isfinite(dec):
                return "    " + sky_text(float(ra), float(dec))
        except Exception:  # noqa: BLE001 - outside the solution
            pass
        return ""

    def _hover(self, cv: ImageCanvas, x: float, y: float, which: int):
        if cv is self.canvas2 or which == 1:
            prep = self._cmp.prep if self._cmp is not None else None
            vi = self._targets[self._cmp_i] if self._cmp_i is not None else None
            name = vi.name if vi is not None else ""
            if self._cmp is not None and self._cmp.align is not None:
                x, y = self._cmp.align.apply(x, y)  # the point of this grid -> the compared image's own pixel
        else:
            prep, vi = self._prep, self._vi
            name = ""
            self.hovered.emit(x, y)
        t = self.pixel_text(x, y, prep, vi) if prep is not None else ""
        self.readout.setText((name + "  ·  " if t and name else "") + t)

    def _left(self):
        self.readout.setText("")

    # ------------------------------------------------------------ compare
    def set_compare_targets(self, targets: list[ViewImage], current: int | None = None):
        """The images this one can be compared with (the previous steps of the workflow), by name."""
        self._targets = list(targets)
        self.cmp_cb.blockSignals(True)
        self.cmp_cb.clear()
        self.cmp_cb.addItem("—", None)
        for i, t in enumerate(self._targets):
            self.cmp_cb.addItem(t.name or _("Image {n}").format(n=i + 1), i)
        view = self.cmp_cb.view()
        view.setMinimumWidth(view.sizeHintForColumn(0) + 24)          # the list shows the names whole
        self.cmp_cb.blockSignals(False)
        self._update_actions()
        self.set_compare(current)

    def set_compare(self, index: int | None):
        if index is not None and not (0 <= index < len(self._targets)):
            index = None
        self._cmp_i = index
        self.cmp_cb.blockSignals(True)
        self.cmp_cb.setCurrentIndex(0 if index is None else index + 1)
        self.cmp_cb.setToolTip(self.cmp_cb.currentText() if index is not None else "")   # a name cut short, whole
        self.cmp_cb.blockSignals(False)
        if self._cmp is not None:
            self._cmp.release()
        self._cmp = None
        self._align()
        self.canvas.set_compare(None)
        self.canvas2.set_display(None)
        self.canvas2.setVisible(index is not None and self._cmp_mode == "side")
        self.cmp_opts.setVisible(index is not None)
        if index is not None:
            vi = self._targets[index]
            hit = self.loader.cached(vi.identity())
            if hit is not None:
                self._show_compare(hit)
            else:
                self.canvas2.message = _("Loading the image…")
                self.loader.request(vi, 0)
        self._names()
        self.compareChanged.emit(None if index is None else self._targets[index].identity())

    def _cmp_picked(self, i):
        self.set_compare(self.cmp_cb.itemData(i))

    def set_compare_mode(self, mode: str):
        self._cmp_mode = "side" if mode == "side" else "split"
        self.split_b.setChecked(self._cmp_mode == "split")
        self.side_b.setChecked(self._cmp_mode == "side")
        side = self._cmp_mode == "side" and self._cmp_i is not None
        self.canvas2.setVisible(side)
        if self._cmp is not None:
            if side:
                self.canvas.set_compare(None)
                self.canvas2.set_display(self._cmp)
                _mirror_state(self.canvas2, _state_of(self.canvas))
                QTimer.singleShot(0, lambda: _mirror_state(self.canvas2, _state_of(self.canvas)))
            else:
                self.canvas2.set_display(None)
                self.canvas.set_compare(self._cmp)
        self._names()

    def compare_mode(self) -> str:
        return self._cmp_mode

    def set_same_stretch(self, on: bool):
        self._same = bool(on)
        if self.same_c.isChecked() != self._same:
            self.same_c.setChecked(self._same)
        self._apply_same()

    def _same_possible(self) -> bool:
        """Same stretch only between images of one kind of values: a picture already stretched for the eye (the final
        image, 0-255) and linear data (a stack) never share one (the data would show black)."""
        vi = self._targets[self._cmp_i] if self._cmp_i is not None and self._cmp_i < len(self._targets) else None
        return vi is None or self._vi is None or bool(vi.stretched) == bool(self._vi.stretched)

    def _cmp_stretch(self, prep: Prepared) -> sm.Stretch:
        if self._same and self._st is not None and self._same_possible():
            return self._st.adapted(prep.channels)
        st = self._st
        vi = self._targets[self._cmp_i] if self._cmp_i is not None and self._cmp_i < len(self._targets) else None
        if st is None or kind_of(vi or prep.vi) != kind_of(self._vi):
            return default_stretch(prep, vi=vi)               # another kind of image: its own automatic stretch
        mode = st.mode if st.mode != "linear" else None
        # Neutral sky is the user's choice for both sides: "before" in the camera's colours stays as it is unless the
        # main picture is shown with a neutral sky (a mosaic and a mono picture keep their own default)
        neutral = None
        if prep.channels == 3 and self._prep is not None and self._prep.channels == 3 and \
                not (vi is not None and vi.stretched):
            neutral = bool(st.neutral) if (st.auto or st.mode != "linear") else bool(self._neutral_before)
            if self._cmp_neutral is not None:
                neutral = bool(self._cmp_neutral)
        return default_stretch(prep, mode, st.linked if prep.channels > 1 else True, neutral, vi)

    _cmp_neutral: bool | None = None

    def set_compare_neutral(self, on: bool | None):
        """Neutral sky for the compared image only (display only): True or False, or None to follow this one's."""
        self._cmp_neutral = None if on is None else bool(on)
        self._apply_same()

    def _apply_same(self):
        if self._cmp is not None:
            self._cmp.set_stretch(self._cmp_stretch(self._cmp.prep))
            self.canvas.update()
            self.canvas2.update()

    def _show_compare(self, prep: Prepared):
        if self._cmp is not None:
            self._cmp.release()
        self._cmp = Display(prep, self._cmp_stretch(prep))
        self._align()
        self.canvas2.message = ""
        if self._cmp_mode == "side":
            self.canvas2.set_display(self._cmp)
            self.canvas2.show()
            _mirror_state(self.canvas2, _state_of(self.canvas))
        else:
            self.canvas.set_compare(self._cmp)
        self._names()

    def _align(self):
        """Put the compared image on the grid of the one shown, so that a star sits at the same place in both (split
        and side by side; ``alignment``: the results' geometry, else the astrometric solutions), and say how in one line
        above the images. Display only: the compared image is resampled as it is drawn."""
        d = self._cmp
        al = None
        if d is not None and self._prep is not None:
            try:
                al = alignment(self._prep, d.prep)
            except Exception:  # noqa: BLE001 - a geometry that cannot be read: drawn as before, matched by width
                al = None
            d.set_alignment(al, self._prep.size())
        self._al = al
        text = self.alignment_note()
        self.align_l.setText(text)
        self.align_bar.setVisible(bool(text))
        if d is not None and self.canvas2.isVisible() and self.canvas2.display is d:
            self.canvas2.set_display(d)
            _mirror_state(self.canvas2, _state_of(self.canvas))
        self.canvas.update()
        self.canvas2.update()

    def compare_alignment(self) -> Alignment | None:
        """How the compared image is drawn on this one (None: the same grid, or no compare)."""
        return self._al if self._cmp is not None else None

    def alignment_note(self) -> str:
        """The line that says how the compared image was put on this one's grid ("" when nothing was done)."""
        al = self.compare_alignment()
        if al is None or self._cmp_i is None or self._cmp_i >= len(self._targets):
            return ""
        name = self._targets[self._cmp_i].name or _("the compared image")
        if al.how == "width":
            return _("Not aligned: no common grid is known for “{name}”; it is matched to this image by its "
                     "width").format(name=name)
        k = 1.0 / al.scale
        ox, oy = al.origin()
        bits = [_("× {k}").format(k=number(k, 3 if k < 0.1 else 2))] if abs(k - 1.0) > 5e-4 else []
        if abs(al.angle) >= 0.05:
            bits.append(_("turned {a}°").format(a=number(-al.angle, 2)))
        if abs(ox) >= 0.05 or abs(oy) >= 0.05:
            bits.append(_("shifted {dx}, {dy} px").format(dx=number(ox, 1), dy=number(oy, 1)))
        if al.field is not None:
            bits.append(_("their distortions followed, up to {d} px").format(d=number(al.residual, 1)))
        what = ", ".join(bits) or _("on the same pixels")
        if al.how == "sky":
            return _("⇄ Aligned for display by the astrometric solutions: “{name}” resampled onto this image "
                     "({what})").format(name=name, what=what)
        return _("⇄ Aligned for display by the geometry of the results (drizzle scale, crop, size): “{name}” "
                 "resampled onto this image ({what})").format(name=name, what=what)

    def _names(self):
        ok = self._same_possible()
        self.same_c.setEnabled(ok)
        self.same_c.setToolTip(_("Show the compared image with the same stretch as this one") if ok else _(
            "Not possible here: one image is already stretched for the eye (the final image, rendered by Post), the "
            "other is linear data. Each keeps its own stretch."))
        main = self._vi.name if self._vi is not None else ""
        other = self._targets[self._cmp_i].name if self._cmp_i is not None else ""
        self.canvas.split_names = (other, main)
        side = self._cmp_mode == "side" and self._cmp_i is not None
        self.canvas.title = main if side else ""
        self.canvas2.title = other if side else ""
        self.canvas.update()
        self.canvas2.update()

    # ------------------------------------------------------------ sequence
    def set_sequence(self, frames: list[ViewImage], index: int = 0):
        """Show a sequence of frames (paths or loaders, with the caller's ``info``)."""
        self._frames = list(frames)
        self._locked = {}
        self.seq.setVisible(bool(self._frames))
        if not self._frames:
            self._blink.stop()
            return
        self.strip.set_count(len(self._frames))
        for i, f in enumerate(self._frames):
            self.strip.set_excluded(i, bool((f.info or {}).get("excluded")))
            self.strip.set_flagged(i, (f.info or {}).get("flag_level") if (f.info or {}).get("flag") else None)
            th = self.loader.thumbs.get(f.identity())
            if th is not None:
                self.strip.set_thumb(i, th)
        self.num.blockSignals(True)
        self.num.setMaximum(len(self._frames))
        self.num.blockSignals(False)
        self.total_l.setText("/ " + str(len(self._frames)))
        self._index = -1
        self._st = None                           # the first frame shown gets its own automatic stretch
        self.set_index(index)
        QTimer.singleShot(0, lambda: self._want_thumbs(*self.strip.visible_range()))

    def frames(self) -> list[ViewImage]:
        return list(self._frames)

    def index(self) -> int:
        return self._index

    def set_index(self, i: int):
        if not self._frames:
            return
        i = int(np.clip(i, 0, len(self._frames) - 1))
        if i == self._index:
            return
        self._index = i
        f = self._frames[i]
        self.num.blockSignals(True)
        self.num.setValue(i + 1)
        self.num.blockSignals(False)
        self.band.set_frame(f, i, len(self._frames))
        self.strip.set_current(i)
        self.prev_b.setEnabled(i > 0)
        self.next_b.setEnabled(i < len(self._frames) - 1)
        self.set_image(f, keep_view=True, keep_stretch=self._lock)
        self._prefetch()
        self.frameChanged.emit(i)

    def next_frame(self):
        if self._frames:
            self.set_index(self._index + 1)

    def previous_frame(self):
        if self._frames:
            self.set_index(self._index - 1)

    def _prefetch(self):
        """The neighbours of the frame shown, ahead of time (the next ones first)."""
        n = len(self._frames)
        keys = []
        for pr, d in ((1, 1), (1, -1), (2, 2), (3, -2), (3, 3)):
            j = self._index + d
            if 0 <= j < n:
                vi = self._frames[j]
                keys.append(vi.identity())
                if self.loader.cached(vi.identity()) is None:
                    self.loader.request(vi, pr)
        self.loader.forget_requests(keep=keys)

    def set_frame_info(self, i: int, info: dict):
        if 0 <= i < len(self._frames):
            self._frames[i].info = dict(info)
            self.strip.set_excluded(i, bool(info.get("excluded")))
            self.strip.set_flagged(i, info.get("flag_level") if info.get("flag") else None)
            if i == self._index:
                self.band.set_frame(self._frames[i], i, len(self._frames))

    def set_excluded(self, i: int, excluded: bool):
        """Mark a frame kept or excluded (no signal: the caller's own change)."""
        if 0 <= i < len(self._frames):
            info = dict(self._frames[i].info or {})
            info["excluded"] = bool(excluded)
            self.set_frame_info(i, info)

    def set_exclude_mode(self, mode: str, tip: str = ""):
        """``on`` (default), ``off`` (no Exclude / Include, no state: masters) or ``spec`` (◇: the engine cannot
        leave such a frame out yet; the toggle is shown disabled with ``tip``)."""
        self.band.set_mode(mode, tip)
        if self._frames and 0 <= self._index < len(self._frames):
            self.band.set_frame(self._frames[self._index], self._index, len(self._frames))

    def set_post_enabled(self, on: bool, tip: str = ""):
        """Send to Post can be offered (the Post page takes images) or not: disabled, ``tip`` says why."""
        self._post_ok = bool(on)
        self._post_tip = tip
        self._update_actions()

    _post_ok = True
    _post_tip = ""

    def toggle_excluded(self):
        """Exclude / Include the frame shown: ``excludeToggled(index, excluded)``."""
        if not self._frames or self.band.mode != "on":
            return
        i = self._index
        now = not bool((self._frames[i].info or {}).get("excluded"))
        self.set_excluded(i, now)
        self.excludeToggled.emit(i, now)

    def set_lock_stretch(self, on: bool):
        """"Same stretch for all frames": on, the stretch shown is kept for the next frames of the same kind (each
        kind its own); off, each frame gets its own automatic stretch, the one shown now at once."""
        on = bool(on)
        changed = on != self._lock
        self._lock = on
        if self.lock_c.isChecked() != self._lock:
            self.lock_c.setChecked(self._lock)
        if not changed:
            return
        self._locked = {}
        if on:
            self._keep_locked()
        elif self._prep is not None and self._st is not None and self._st.auto:
            st = self._st
            self.set_stretch(default_stretch(self._prep, st.mode, st.linked if st.channels > 1 else None, None,
                                             self._vi), emit=True)

    def _want_thumbs(self, i0: int, i1: int):
        for i in range(max(0, i0 - 4), min(len(self._frames), i1 + 4)):
            f = self._frames[i]
            if f.thumb is not None and f.identity() not in self.loader.thumbs:
                self.loader.request_thumb(f, 10 + abs(i - self._index))

    def _thumb(self, key, arr):
        for i, f in enumerate(self._frames):
            if f.identity() == key:
                self.strip.set_thumb(i, arr)

    def _blink_toggled(self, on: bool):
        self.blink_b.setText(("❚❚  " if on else "▶  ") + _("Blink"))
        if on:
            self._blink.start()
        else:
            self._blink.stop()

    def _blink_tick(self):
        if not self._frames:
            self._blink.stop()
            return
        j = (self._index + 1) % len(self._frames)
        if self.loader.cached(self._frames[j].identity()) is None:
            self.loader.request(self._frames[j], 0)
            return
        if j == 0:
            self._index = -1
        self.set_index(j)

    # ------------------------------------------------------------ actions
    def _send_post(self):
        if self._prep is not None:
            self.sendToPost.emit(self._prep.key)

    def grab_view(self) -> QImage:
        return self.area.grab().toImage()

    def copy_view(self):
        if self._prep is None:
            return
        QGuiApplication.clipboard().setImage(self.grab_view())
        self.toasted.emit(_("Image copied to the clipboard, as shown: paste it where you want"))

    def visible_area(self) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.canvas.visible()
        return int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))

    def export(self, path: str | os.PathLike, fmt: str | None = None, content: str = "data",
               area: tuple[int, int, int, int] | None = None, wcs: bool = True) -> Path:
        """Write the image shown (in the GUI thread: the dialog runs it in a worker)."""
        if self._prep is None:
            raise RuntimeError("no image")
        return export_image(path, self._prep, self._st, fmt=fmt, content=content, area=area, with_wcs=wcs,
                            vi=self._vi)

    def export_dialog(self):
        if self._prep is None:
            return None
        dlg = ImageExportDialog(self, self.window())
        dlg.open()
        return dlg

    def add_overlay(self, fn):
        self.canvas.overlays.append(fn)
        self.canvas.update()

    #: the annotations the overlays draw, for Export… ((sky, show), or None): a callable the caller sets
    annotation_source = None

    def annotations(self):
        """(sky, show) the overlays draw (the sky grid, the named objects), or None: Export… offers to burn them in."""
        fn = self.annotation_source
        try:
            return fn() if callable(fn) else None
        except RuntimeError:
            return None

    # ------------------------------------------------------------ tests and scripts
    def wait(self, timeout: float = 60.0) -> bool:
        """Process events until the image asked for (and the compared one) are shown."""
        end = time.monotonic() + timeout
        app = QApplication.instance()
        while time.monotonic() < end:
            app.processEvents()
            done = self._want is None or (self._prep is not None and self._prep.key == self._want) or \
                self._failed_key == self._want
            cmp_done = self._cmp_i is None or self._cmp is not None or (
                self._cmp_i < len(self._targets) and self._cmp_failed == self._targets[self._cmp_i].identity())
            if done and cmp_done:
                app.processEvents()
                return True
            time.sleep(0.003)
        return False


def _state_of(c: ImageCanvas) -> dict:
    if c.display is None:
        return {"fitted": True}
    w, h = c.display.size()
    return {"fitted": c.fitted, "zoom": c.scale * w, "fx": c.cx / w, "fy": c.cy / h}


def _mirror_state(c: ImageCanvas, state: dict):
    """Put a canvas where another one is (images of other sizes: the same part of the field)."""
    if c.display is None:
        return
    if state.get("fitted", True) or "zoom" not in state:
        c.fit()
        return
    w, h = c.display.size()
    c.zoom_to(state["zoom"] / w, state["fx"] * w, state["fy"] * h)


class SyncGroup(QObject):
    """Viewers that move together: zoom and pan, and the stretch too when ``stretch`` ("same stretch on both")."""

    def __init__(self, views=(), stretch: bool = False, parent=None):
        super().__init__(parent)
        self.views: list[ImageView] = []
        self.stretch = bool(stretch)
        self._busy = False
        for v in views:
            self.add(v)

    def add(self, v: ImageView):
        if v in self.views:
            return
        self.views.append(v)
        v.viewChanged.connect(lambda vv=v: self._moved(vv))
        v.stretchChanged.connect(lambda st, vv=v: self._stretched(vv, st))

    def remove(self, v: ImageView):
        if v in self.views:
            self.views.remove(v)

    def set_stretch_shared(self, on: bool):
        self.stretch = bool(on)
        if on and self.views and self.views[0].stretch() is not None:
            self._stretched(self.views[0], self.views[0].stretch())

    def _moved(self, src: ImageView):
        if self._busy:
            return
        self._busy = True
        try:
            st = src.view_state()
            for v in self.views:
                if v is not src:
                    v.set_view_state(st)
        finally:
            self._busy = False

    def _stretched(self, src: ImageView, st):
        if self._busy or not self.stretch:
            return
        self._busy = True
        try:
            for v in self.views:
                if v is not src:
                    v.set_stretch(st)
        finally:
            self._busy = False


# ---------------------------------------------------------------------------------------------- export
FORMATS = {"fits": ("FITS 32-bit float", ".fits"), "tiff16": ("TIFF 16-bit", ".tif"),
           "png16": ("PNG 16-bit", ".png"), "png8": ("PNG 8-bit", ".png"), "jpeg": ("JPEG", ".jpg")}


def format_of(path: str | os.PathLike) -> str:
    suf = Path(path).suffix.lower()
    return {".fits": "fits", ".fit": "fits", ".fts": "fits", ".tif": "tiff16", ".tiff": "tiff16", ".png": "png16",
            ".jpg": "jpeg", ".jpeg": "jpeg"}.get(suf, "fits")


def slug(name: str) -> str:
    """A file name from a name: every character but letters, digits, "_", "." and "-" becomes "_"."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "image"


def export_image(path: str | os.PathLike, prep: Prepared, st: sm.Stretch | None = None, fmt: str | None = None,
                 content: str = "data", area: tuple[int, int, int, int] | None = None, with_wcs: bool = True,
                 header: dict | None = None, vi: ViewImage | None = None, annotations=None) -> Path:
    """Write a prepared image. FITS: the data as they are (32-bit float), with the WCS (shifted for a crop) and a
    header. TIFF 16-bit, PNG, JPEG: ``content`` "data" (ADU as 16-bit integers, 0-1 data scaled to the full range) or
    "view" (the stretch shown, at full precision). ``area``: (x0, y0, x1, y1) image pixels, else the whole image.
    ``annotations``: (sky, show) burned into the stretched view of a picture (``annotate.burn``: the sky grid, the
    named objects); never into a FITS file nor into the linear data."""
    path = Path(path)
    fmt = fmt or format_of(path)
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    vi = vi or prep.vi
    x0, y0, x1, y1 = area if area is not None else (0, 0, prep.width, prep.height)
    if prep.cfa:                                    # a mosaic is cut on whole 2 x 2 cells: the pattern stays
        x0, y0 = x0 // 2 * 2, y0 // 2 * 2
        x1, y1 = min(prep.width, -(-x1 // 2) * 2), min(prep.height, -(-y1 // 2) * 2)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(prep.width, int(x1)), min(prep.height, int(y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("empty area")
    if fmt == "fits":
        content = "data"
    if content == "view":
        st = st or default_stretch(prep, vi=vi)
        f = prep.factor
        lv = prep.levels[0][:, y0 // f:-(-y1 // f), x0 // f:-(-x1 // f)]
        img = sm.apply(lv, st.adapted(prep.channels), prep.stats)        # (C, h, w) 0..1
    else:
        d = prep.data
        img = d[y0:y1, x0:x1] if d.ndim == 2 else d[:, y0:y1, x0:x1]
    if fmt == "fits":
        _write_fits(path, img, prep, (x0, y0), with_wcs, header, vi)
        return path
    a = np.asarray(img, np.float32)
    if a.ndim == 3:
        a = a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0)
    if content == "view":
        unit = a
        if annotations is not None and a.ndim == 3:
            from .annotate import burn, shifted
            sky, show = annotations
            unit = burn(np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0), shifted(sky, x0, y0), show)
    elif vi.units == "norm":
        unit = a
    else:
        unit = a / 65535.0                          # ADU as they are: 16-bit integers
    unit = np.clip(np.nan_to_num(unit, nan=0.0), 0.0, 1.0)
    if fmt in ("tiff16", "png16"):
        out = (unit * 65535.0 + 0.5).astype(np.uint16)
    else:
        out = (unit * 255.0 + 0.5).astype(np.uint8)
    out = np.ascontiguousarray(out)
    if fmt == "tiff16":
        import tifffile
        tifffile.imwrite(path, out, photometric="rgb" if out.ndim == 3 else "minisblack", compression="zlib",
                         metadata=None)
    elif fmt == "png16":
        import cv2
        ok = cv2.imwrite(str(path), out[:, :, ::-1] if out.ndim == 3 else out)
        if not ok:
            raise OSError(f"cannot write {path}")
    else:
        from PIL import Image
        im = Image.fromarray(out)
        if fmt == "jpeg":
            im.save(path, quality=95)
        else:
            im.save(path)
    return path


def _write_fits(path: Path, img: np.ndarray, prep: Prepared, origin, with_wcs: bool, extra: dict | None,
                vi: ViewImage):
    import datetime as dt
    import warnings

    from astropy.io import fits
    a = np.asarray(img, np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    hdr = fits.Header()
    if with_wcs and prep.wcs is not None and hasattr(prep.wcs, "to_header"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = prep.wcs.deepcopy()
            w.wcs.crpix = [w.wcs.crpix[0] - origin[0], w.wcs.crpix[1] - origin[1]]
            hdr.update(w.to_header(relax=True))
    if vi.name:
        hdr["OBJECT"] = (vi.name[:68], "the image's name in astro_gpu")
    hdr["BUNIT"] = ("ADU" if vi.units == "adu" else "", "data units")
    if prep.cfa:
        hdr["BAYERPAT"] = (prep.cfa, "Bayer pattern of the mosaic")
        hdr["XBAYROFF"] = 0
        hdr["YBAYROFF"] = 0
    if origin != (0, 0):
        hdr["XORIGIN"] = (int(origin[0]), "x of the first column in the full image")
        hdr["YORIGIN"] = (int(origin[1]), "y of the first row in the full image")
    hdr["CREATOR"] = "astro_gpu"
    hdr["DATE"] = (dt.datetime.now().isoformat(timespec="seconds"), "file written")
    for k, v in {**(vi.header or {}), **(extra or {})}.items():
        key = str(k).upper()[:8]
        if key in ("HISTORY", "COMMENT"):
            for line in (v if isinstance(v, list | tuple) else [v]):
                hdr.add_history(str(line)) if key == "HISTORY" else hdr.add_comment(str(line))
        elif isinstance(v, str | int | float | bool):
            hdr[key] = v
    hdr.add_history("Written by the astro_gpu image viewer: the data as they are (no display stretch).")
    fits.PrimaryHDU(a, header=hdr).writeto(path, overwrite=True)


class ImageExportDialog(QDialog):
    """Export…: the format, what is written (the data or the stretched view), the area, the file."""

    finished_export = Signal(str, str)              # path, error ("" when written)

    def __init__(self, view: ImageView, parent=None):
        super().__init__(parent)
        self.view = view
        self.prep = view.prepared()
        self.vi = view.image() or self.prep.vi
        self.st = view.stretch()
        self.result_path: Path | None = None
        self.error = ""
        self.setWindowTitle(_("Export image"))
        self.setMinimumWidth(600)
        self.setStyleSheet(QSS)
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(10)
        v.addWidget(label(_("Export image"), "h1"))
        sub = label(f"{self.vi.name or _('Image')}  ·  {view._size_text(self.prep)}", "dim")
        v.addWidget(sub)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, 110)
        r = 0
        grid.addWidget(label(_("Format")), r, 0)
        self.fmt = QComboBox()
        for key, (name, _suf) in FORMATS.items():
            self.fmt.addItem(_(name), key)
        grid.addWidget(self.fmt, r, 1)
        r += 1
        grid.addWidget(label(_("Content")), r, 0, Qt.AlignTop)
        cb = QVBoxLayout()
        cb.setSpacing(4)
        self.c_data = QRadioButton(_("The data, as they are"))
        self.c_view = QRadioButton(_("The stretched view, as shown"))
        self.c_view.setChecked(True)
        cb.addWidget(self.c_data)
        cb.addWidget(self.c_view)
        grid.addLayout(cb, r, 1)
        r += 1
        grid.addWidget(label(_("Area")), r, 0, Qt.AlignTop)
        ab = QVBoxLayout()
        ab.setSpacing(4)
        self.a_all = QRadioButton(_("The whole image ({w} × {h})").format(w=self.prep.width, h=self.prep.height))
        x0, y0, x1, y1 = view.visible_area()
        self.area = (x0, y0, x1, y1)
        self.a_vis = QRadioButton(_("The visible part ({w} × {h})").format(w=x1 - x0, h=y1 - y0))
        self.a_all.setChecked(True)
        self.a_vis.setEnabled(not view.canvas.whole_visible())
        ab.addWidget(self.a_all)
        ab.addWidget(self.a_vis)
        grid.addLayout(ab, r, 1)
        r += 1
        self.c_wcs = QCheckBox(_("Astrometric solution (WCS) in the header"))
        self.c_wcs.setChecked(hasattr(self.prep.wcs, "to_header"))
        grid.addWidget(self.c_wcs, r, 1)
        r += 1
        # the annotations the view draws (the sky grid, the named objects), burned into a picture; never into a FITS
        self.ann = view.annotations() if callable(getattr(view, "annotations", None)) else None
        self.c_grid = QCheckBox(_("Sky grid"))
        self.c_names = QCheckBox(_("Object names"))
        self.ann_row = QWidget()
        ah = QHBoxLayout(self.ann_row)
        ah.setContentsMargins(0, 0, 0, 0)
        ah.addWidget(label(_("Include annotations:")))
        ah.addWidget(self.c_grid)
        ah.addWidget(self.c_names)
        ah.addStretch(1)
        if self.ann is not None:
            show = self.ann[1]
            self.c_grid.setChecked(bool(show.get("grid")) and bool(getattr(self.ann[0], "lines", None)))
            self.c_names.setChecked(bool(show.get("objects")) and bool(getattr(self.ann[0], "objects", None)))
            self.c_grid.setEnabled(bool(getattr(self.ann[0], "lines", None)))
            self.c_names.setEnabled(bool(getattr(self.ann[0], "objects", None)))
        self.ann_row.setVisible(self.ann is not None)
        grid.addWidget(self.ann_row, r, 1)
        r += 1
        grid.addWidget(label(_("File")), r, 0)
        fh = QHBoxLayout()
        self.dest = QLineEdit()
        self.browse = QPushButton("…")
        self.browse.setFixedWidth(34)
        self.browse.setToolTip(_("Choose the file"))
        self.browse.clicked.connect(self._browse)
        fh.addWidget(self.dest, 1)
        fh.addWidget(self.browse)
        grid.addLayout(fh, r, 1)
        v.addLayout(grid)
        self.note = label("", "small")
        self.note.setWordWrap(True)
        v.addWidget(self.note)
        self.status = label("", "small")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        v.addStretch(1)
        bb = QHBoxLayout()
        bb.addStretch(1)
        self.cancel_b = QPushButton(_("Cancel"))
        self.cancel_b.clicked.connect(self.reject)
        self.go = QPushButton(_("Export"))
        self.go.setObjectName("primary")
        self.go.setDefault(True)
        self.go.clicked.connect(self._go)
        bb.addWidget(self.cancel_b)
        bb.addWidget(self.go)
        v.addLayout(bb)
        folder = view.export_dir or QStandardPaths.writableLocation(QStandardPaths.PicturesLocation) or \
            str(Path.home())
        self.dest.setText((Path(folder) / (slug(self.vi.name or "image") + ".fits")).as_posix())
        self.fmt.currentIndexChanged.connect(self._changed)
        self.c_data.toggled.connect(self._changed)
        self.finished_export.connect(self._done)
        self._changed()

    def format_key(self) -> str:
        return self.fmt.currentData()

    def _changed(self, *_a):
        key = self.format_key()
        fits_ = key == "fits"
        self.c_view.setEnabled(not fits_)
        if fits_:
            self.c_data.setChecked(True)
        self.c_wcs.setEnabled(fits_ and hasattr(self.prep.wcs, "to_header"))
        self.c_wcs.setToolTip("" if fits_ else _("Only the FITS file has a header."))
        burn = not fits_ and not self.c_data.isChecked()
        self.ann_row.setEnabled(burn)
        self.ann_row.setToolTip("" if burn else _("Drawn into a picture of the stretched view only: a FITS file and "
                                                  "the linear data stay clean."))
        text = self.dest.text().strip()
        if text and Path(text).name:
            self.dest.setText(Path(text).with_suffix(FORMATS[key][1]).as_posix())
        if fits_:
            self.note.setText(_("FITS keeps the data as they are, 32-bit float: the display stretch is not applied."))
        elif self.c_data.isChecked():
            self.note.setText(_("The linear data: ADU as 16-bit integers (0–1 data over the whole range). They look "
                                "dark without a stretch.") if key != "jpeg" and key != "png8" else
                              _("The linear data in 8 bits: dark, and the faint levels are lost. For a picture, "
                                "export the stretched view."))
        else:
            self.note.setText(_("The image as the stretch shows it, computed at full resolution."))

    def _browse(self):
        key = self.format_key()
        name, suf = FORMATS[key]
        f, _flt = QFileDialog.getSaveFileName(self, _("Export image"), self.dest.text(), f"{_(name)} (*{suf})")
        if f:
            self.dest.setText(Path(f).with_suffix(suf).as_posix())

    def options(self) -> dict:
        show = {"grid": self.c_grid.isChecked() and self.c_grid.isEnabled(),
                "objects": self.c_names.isChecked() and self.c_names.isEnabled()}
        burn = self.ann is not None and self.ann_row.isEnabled() and (show["grid"] or show["objects"])
        return {"path": self.dest.text().strip(), "fmt": self.format_key(),
                "content": "data" if self.c_data.isChecked() else "view",
                "area": self.area if self.a_vis.isChecked() else None,
                "with_wcs": self.c_wcs.isEnabled() and self.c_wcs.isChecked(),
                "annotations": (self.ann[0], show) if burn else None}

    def _go(self):
        if self.result_path is not None:
            self.accept()
            return
        o = self.options()
        if not o["path"]:
            self.status.setText(_("Choose a file."))
            return
        self.go.setEnabled(False)
        self.status.setStyleSheet("")
        self.status.setText(_("Writing…"))
        prep, st, vi = self.prep, self.st, self.vi

        def job():
            try:
                p = export_image(o["path"], prep, st, fmt=o["fmt"], content=o["content"], area=o["area"],
                                 with_wcs=o["with_wcs"], vi=vi, annotations=o["annotations"])
                self.finished_export.emit(str(p), "")
            except Exception as e:  # noqa: BLE001 - shown in the dialog
                try:
                    self.finished_export.emit(o["path"], f"{type(e).__name__}: {e}")
                except RuntimeError:
                    pass
        self._thread = threading.Thread(target=job, daemon=True)
        self._thread.start()

    def _done(self, path: str, error: str):
        self.go.setEnabled(True)
        if error:
            self.error = error
            self.status.setText(_("The export failed: {why}").format(why=error))
            self.status.setStyleSheet(f"color: {WARN};")
            return
        self.result_path = Path(path)
        self.status.setText(_("Written: {path}").format(path=path))
        self.go.setText(_("Close"))
        self.cancel_b.setEnabled(False)
        self.view.toasted.emit(_("Exported to {path}").format(path=path))

    def wait(self, timeout: float = 60.0) -> bool:
        end = time.monotonic() + timeout
        app = QApplication.instance()
        while time.monotonic() < end:
            app.processEvents()
            t = getattr(self, "_thread", None)
            if t is not None and not t.is_alive():
                app.processEvents()
                return True
            time.sleep(0.01)
        return False


# ---------------------------------------------------------------------------------------------- helpers
def sky_text(ra: float, dec: float) -> str:
    """RA and Dec as the readout writes them: RA 02h42m06.6s   Dec +61°18′01″ (a decimal comma in French)."""
    tenths = round((ra % 360.0) / 15.0 * 36000) % (24 * 36000)     # tenths of a second of time, rounded once
    h, rest = divmod(tenths, 36000)
    m, t = divmod(rest, 600)
    sec = t / 10
    arcsec = round(abs(dec) * 3600)
    dd, rest = divmod(arcsec, 3600)
    dm, ds = divmod(rest, 60)
    sign = "+" if dec >= 0 else "−"
    sec_txt = number(sec, 1).rjust(4, "0")
    return f"RA {h:02d}h{m:02d}m{sec_txt}s   Dec {sign}{dd:02d}°{dm:02d}′{ds:02d}″"


__all__ = ["ViewImage", "ImageView", "SyncGroup", "Loader", "default_loader", "Prepared", "prepare", "Display",
           "ImageCanvas", "HistogramView", "StretchPanel", "InfoBand", "FilmStrip", "ImageExportDialog",
           "export_image", "sky_text", "read_file", "FORMATS"]
