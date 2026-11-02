"""The annotations of a solved picture: the RA / Dec grid and the named deep-sky objects (Messier, NGC, IC, Sharpless and
their common names, outlined at their size), drawn over the picture on the screen (Astrometry, Post) and burned into an
exported PNG / JPEG / TIFF. A FITS file never gets them: it keeps the data as they are, with the WCS.

* ``field_annotations(wcs, w, h, grid, names)`` (worker thread): the grid lines (``services/inspect.sky_grid``) and the
  objects of the field (``core/skyobjects.objects_in_field``) as ``Named`` items, in the picture's pixels: the centre,
  the outline as a polygon (the object's ellipse through the solution) and the label. The catalogue is the engine's
  (``core/skyobjects``: OpenNGC, Steinicke's Revised NGC/IC and the Sharpless catalogue, shipped with the package).
* ``paint_grid`` / ``paint_objects`` draw them with a QPainter on anything that has ``cx, cy, scale, width(),
  height()`` (the viewers' canvases, or ``Frame`` for a whole picture).
* ``burn(rgb, sky, show)``: the same drawing on an image array (H, W, 3, 0..1), for the export.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QPolygonF

#: the objects drawn at most (the biggest, and the ones with a Messier number or a common name, first)
MAX_OBJECTS = 150
#: an object smaller than this on the picture (pixels) is drawn only when it has a Messier number or a common name
MIN_PX = 6.0

GRID = QColor(94, 200, 224, 130)
GRID_TEXT = QColor(150, 220, 236)
LABEL_BG = QColor(15, 17, 21, 190)
#: outline colours by family of object (the label stays white)
KIND_COLORS = (("galaxy", QColor(255, 184, 107, 235)), ("dark", QColor(190, 190, 200, 220)),
               ("nebula", QColor(255, 128, 168, 235)), ("hii", QColor(255, 128, 168, 235)),
               ("remnant", QColor(255, 128, 168, 235)), ("cluster", QColor(143, 211, 255, 235)),
               ("", QColor(255, 128, 168, 235)))


@dataclass
class Named:
    """One object of the field on the picture: centre (x, y), outline (n, 2) in pixels (None: size unknown, a small
    circle is drawn), designation, English common name, kind."""

    x: float
    y: float
    outline: np.ndarray | None
    name: str
    common: str = ""
    kind: str = ""
    size_px: float = 0.0                      # the outline's largest extent, pixels
    aliases: tuple = ()

    @property
    def label(self) -> str:
        return f"{self.name} · {self.common}" if self.common else self.name

    def as_tuple(self) -> tuple:              # what SkyData.objects held before: (x, y, name, kind)
        return (self.x, self.y, self.label, self.kind)


@dataclass
class Annotations:
    """What ``field_annotations`` found: the grid lines, the objects, where the names come from and, when there are
    none, why (plain English, translated where shown)."""

    lines: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    source: str = ""
    why: str = ""


def colour_of(kind: str) -> QColor:
    k = (kind or "").lower()
    for word, col in KIND_COLORS:
        if word in k:
            return col
    return KIND_COLORS[-1][1]


def _important(o) -> bool:
    return o.name.startswith("M ") or bool(o.common_name)


def named_objects(wcs, w: int, h: int, objects=None) -> list[Named]:
    """The objects of the field as ``Named`` items in the pixels of a picture of w x h solved by ``wcs`` (worker
    thread). ``objects``: the engine's SkyObjects (default: ``objects_in_field``, which may raise
    ``CatalogueUnavailable``)."""
    from ..core import skyobjects as so
    objs = list(objects) if objects is not None else so.objects_in_field(wcs, w, h)
    if not objs:
        return []
    ra = np.array([o.ra for o in objs], float)
    dec = np.array([o.dec for o in objs], float)
    cx, cy = wcs.all_world2pix(ra, dec, 0, quiet=True) if hasattr(wcs, "all_world2pix") else wcs.wcs_world2pix(ra, dec, 0)
    cx, cy = np.asarray(cx, float), np.asarray(cy, float)
    major = np.array([o.major for o in objs], float)
    sized = np.nonzero(major > 0)[0]
    outlines: dict[int, np.ndarray] = {}
    if len(sized):
        ora, odec = so._outline(ra[sized], dec[sized], major[sized], np.array([objs[i].minor for i in sized], float),
                                np.array([objs[i].angle for i in sized], float))
        px, py = wcs.wcs_world2pix(ora.ravel(), odec.ravel(), 0)
        px, py = np.asarray(px).reshape(ora.shape), np.asarray(py).reshape(ora.shape)
        # the outline follows the distortion of the solution as its centre does
        dx, dy = cx[sized] - wcs.wcs_world2pix(ra[sized], dec[sized], 0)[0], \
            cy[sized] - wcs.wcs_world2pix(ra[sized], dec[sized], 0)[1]
        for k, i in enumerate(sized):
            pts = np.stack([px[k] + dx[k], py[k] + dy[k]], 1).astype(np.float32)
            if np.isfinite(pts).all():
                outlines[int(i)] = pts
    out = []
    for i, o in enumerate(objs):
        if not (np.isfinite(cx[i]) and np.isfinite(cy[i])):
            continue
        ol = outlines.get(i)
        size = float(np.ptp(ol, axis=0).max()) if ol is not None else 0.0
        out.append(Named(float(cx[i]), float(cy[i]), ol, o.name, o.common_name, o.kind, size, tuple(o.aliases)))
    keep = [n for n in out if n.size_px >= MIN_PX or n.name.startswith("M ") or n.common]
    keep.sort(key=lambda n: (not (n.name.startswith("M ") or n.common), -n.size_px))
    return keep[:MAX_OBJECTS]


def field_annotations(wcs, w: int, h: int, grid: bool = True, names: bool = True) -> Annotations:
    """The grid and the named objects of a picture of w x h pixels solved by ``wcs`` (worker thread). A catalogue that
    cannot be had leaves the objects empty with ``why`` set; the grid does not need it."""
    from ..core import skyobjects as so
    from .services.inspect import sky_grid
    out = Annotations()
    if wcs is None:
        out.why = "No astrometric solution for this image."
        return out
    if grid:
        try:
            out.lines = sky_grid(wcs, w, h)
        except (ValueError, ArithmeticError):
            out.lines = []
    if names:
        try:
            out.objects = named_objects(wcs, w, h)
            out.source = so.catalogue_source()
        except so.CatalogueUnavailable as e:
            out.why = str(e)
        except (ValueError, ArithmeticError) as e:          # a degenerate solution
            out.why = f"The objects cannot be placed: {e}"
    return out


# ---------------------------------------------------------------------------------------------- painting
class Frame:
    """A whole picture as a painting surface for the overlays: one picture pixel per device pixel, ``font_scale`` for
    the labels (an exported picture is much bigger than a screen)."""

    def __init__(self, w: int, h: int, font_scale: float = 1.0):
        self.w, self.h = int(w), int(h)
        self.cx, self.cy, self.scale = w / 2.0, h / 2.0, 1.0
        self.font_scale = float(font_scale)

    def width(self) -> int:
        return self.w

    def height(self) -> int:
        return self.h

    def img_to_w(self, x, y) -> QPointF:
        return QPointF(float(x), float(y))


def to_widget(pts: np.ndarray, v) -> np.ndarray:
    return (np.asarray(pts, np.float64) - (v.cx, v.cy)) * v.scale + (v.width() / 2, v.height() / 2)


def _font(p: QPainter, pt: float, v, bold: bool = False) -> QFontMetricsF:
    f = QFont(p.font())
    f.setPointSizeF(pt * getattr(v, "font_scale", 1.0))
    f.setBold(bold)
    p.setFont(f)
    return QFontMetricsF(f)


def label_at(w: np.ndarray, v, kind: str):
    """Where a grid line enters the picture, on its left edge for a parallel and its top edge for a meridian (else the
    other edges): the point its label hangs from. None when the line does not cross it."""
    W, H, m = v.width(), v.height(), 6.0 * getattr(v, "font_scale", 1.0)
    edges = (("x", m), ("y", m), ("x", W - m), ("y", H - m)) if kind == "dec" else \
        (("y", m), ("x", m), ("y", H - m), ("x", W - m))
    for axis, c in edges:
        a = w[:, 0] if axis == "x" else w[:, 1]
        o = w[:, 1] if axis == "x" else w[:, 0]
        lim = H if axis == "x" else W
        s = np.flatnonzero((a[:-1] - c) * (a[1:] - c) <= 0)
        for i in s:
            d = a[i + 1] - a[i]
            t = 0.0 if d == 0 else (c - a[i]) / d
            other = o[i] + t * (o[i + 1] - o[i])
            if m <= other <= lim - m:
                return (c, other) if axis == "x" else (other, c)
    return None


def paint_grid(p: QPainter, lines, v) -> None:
    """The RA / Dec grid with its labels on the edges (labels that would overlap are left out)."""
    fs = getattr(v, "font_scale", 1.0)
    p.setPen(QPen(GRID, 1.0 * fs))
    p.setBrush(Qt.NoBrush)
    labels = []
    for line in lines:
        for piece in line.pieces:
            w = to_widget(piece, v)
            p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in w]))
            at = label_at(w, v, line.kind)
            if at is not None:
                labels.append((line.label, at, line.kind))
    fm = _font(p, 8.5, v)
    taken: list[QRectF] = []
    for text, (x, y), kind in labels:
        tw, th = fm.horizontalAdvance(text) + 8 * fs, fm.height() + 2 * fs
        r = QRectF(x + 3 * fs, y - th / 2, tw, th) if kind == "dec" else QRectF(x - tw / 2, y + 3 * fs, tw, th)
        r.translate(max(0.0, 2 - r.left()) + min(0.0, v.width() - 2 - r.right()),
                    max(0.0, 2 - r.top()) + min(0.0, v.height() - 2 - r.bottom()))
        if any(r.intersects(t) for t in taken):
            continue
        taken.append(r)
        p.setPen(Qt.NoPen)
        p.setBrush(LABEL_BG)
        p.drawRoundedRect(r, 3 * fs, 3 * fs)
        p.setPen(GRID_TEXT)
        p.drawText(r, Qt.AlignCenter, text)


def paint_objects(p: QPainter, objects, v, max_labels: int = 80) -> None:
    """The named objects: each outline (its ellipse on the sky, through the solution) and a label above it, the
    labels that would overlap an earlier one left out (the objects come most important first)."""
    if not objects:
        return
    fs = getattr(v, "font_scale", 1.0)
    W, H = v.width(), v.height()
    fm = _font(p, 9.0, v, bold=False)
    taken: list[QRectF] = []
    n_labels = 0
    for o in objects:
        if not isinstance(o, Named):                   # an older (x, y, name, kind)
            o = Named(float(o[0]), float(o[1]), None, str(o[2]), "", str(o[3]) if len(o) > 3 else "")
        col = colour_of(o.kind)
        c = to_widget(np.array([[o.x, o.y]]), v)[0]
        if o.outline is not None:
            pts = to_widget(o.outline, v)
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            if x1 < -20 or y1 < -20 or x0 > W + 20 or y0 > H + 20:
                continue
            if max(x1 - x0, y1 - y0) < 7 * fs:          # smaller than a marker on the screen: the marker
                pts = None
        else:
            pts = None
        if pts is None:
            r = 7.0 * fs
            if not (-20 <= c[0] <= W + 20 and -20 <= c[1] <= H + 20):
                continue
            x0, y0, x1, y1 = c[0] - r, c[1] - r, c[0] + r, c[1] + r
        path = QPainterPath()
        if pts is not None:
            path.addPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
            path.closeSubpath()
        else:
            path.addEllipse(QPointF(c[0], c[1]), 7.0 * fs, 7.0 * fs)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 120), 3.2 * fs))          # a dark halo: the outline reads on bright nebulae
        p.drawPath(path)
        p.setPen(QPen(col, 1.4 * fs))
        p.drawPath(path)
        if n_labels >= max_labels:
            continue
        text = o.label
        tw, th = fm.horizontalAdvance(text) + 10 * fs, fm.height() + 4 * fs
        # above the outline, centred on the object; below it when there is no room above; on the object when the
        # outline is taller than the picture; always inside the picture
        lx = float(np.clip(c[0], max(x0, 0.0), min(x1, float(W)))) - tw / 2
        if y0 - th - 2 * fs >= 0:
            ly = y0 - th - 2 * fs
        elif y1 + th + 2 * fs <= H:
            ly = y1 + 2 * fs
        else:
            ly = float(np.clip(c[1], 0.0, H)) - th / 2
        r = QRectF(lx, ly, tw, th)
        r.translate(max(0.0, 2 - r.left()) + min(0.0, W - 2 - r.right()),
                    max(0.0, 2 - r.top()) + min(0.0, H - 2 - r.bottom()))
        if any(r.intersects(t) for t in taken):
            continue
        taken.append(r)
        n_labels += 1
        p.setPen(Qt.NoPen)
        p.setBrush(LABEL_BG)
        p.drawRoundedRect(r, 4 * fs, 4 * fs)
        p.setPen(QColor(255, 255, 255))
        p.drawText(r, Qt.AlignCenter, text)


def paint(p: QPainter, sky, v, show: dict) -> None:
    """The grid and the objects of ``sky`` (``Annotations`` or ``services/inspect.SkyData``), as ``show`` says
    ({"grid", "objects"} -> on)."""
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    try:
        if show.get("grid") and getattr(sky, "lines", None):
            paint_grid(p, sky.lines, v)
        if show.get("objects") and getattr(sky, "objects", None):
            paint_objects(p, sky.objects, v)
    finally:
        p.restore()


def burn(rgb: np.ndarray, sky, show: dict) -> np.ndarray:
    """The annotations drawn into a picture (H, W, 3) in 0..1 (or (3, H, W)): drawn once on a transparent layer, then
    laid over the picture at its own precision (a 16-bit TIFF keeps its 16 bits under the drawing)."""
    planar = rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[2] != 3
    a = np.moveaxis(rgb, 0, -1) if planar else rgb
    h, w = a.shape[:2]
    if not (show.get("grid") or show.get("objects")) or sky is None:
        return rgb
    layer = QImage(w, h, QImage.Format_RGBA8888_Premultiplied)
    layer.fill(0)
    p = QPainter(layer)
    # the labels at about the size they have on a screen showing the whole picture 1200 px wide
    paint(p, sky, Frame(w, h, font_scale=max(1.0, max(w, h) / 1200.0)), show)
    p.end()
    ptr = layer.constBits()
    lay = np.frombuffer(ptr, np.uint8, count=layer.sizeInBytes()).reshape(h, layer.bytesPerLine())[:, :4 * w]
    lay = lay.reshape(h, w, 4).astype(np.float32) / 255.0
    alpha = lay[:, :, 3:4]
    out = np.asarray(a, np.float32) * (1.0 - alpha) + lay[:, :, :3]       # premultiplied: colour already x alpha
    out = np.clip(out, 0.0, 1.0)
    return np.moveaxis(out, -1, 0) if planar else out


def shifted(sky, x0: float, y0: float):
    """The annotations of a crop whose first pixel is (x0, y0) of the picture they were made for."""
    if sky is None or (not x0 and not y0):
        return sky
    import copy
    from types import SimpleNamespace
    off = np.array([x0, y0], np.float32)
    lines = []
    for ln in getattr(sky, "lines", None) or []:
        n = copy.copy(ln)
        n.pieces = [pc - off for pc in ln.pieces]
        lines.append(n)
    objs = []
    for o in getattr(sky, "objects", None) or []:
        if isinstance(o, Named):
            objs.append(Named(o.x - x0, o.y - y0, None if o.outline is None else o.outline - off, o.name, o.common,
                              o.kind, o.size_px, o.aliases))
    return SimpleNamespace(lines=lines, objects=objs)


def crop_wcs(wcs, origin, size=None):
    """The solution of a crop of the grid (``origin``: the grid pixel of the crop's first pixel) and, with ``size``
    (w, h) smaller than the crop's (w0, h0) given as ``size=((w, h), (w0, h0))``, of a reduced copy of it."""
    if wcs is None:
        return None
    ox, oy = float(origin[0]), float(origin[1])
    w2 = wcs.deepcopy()
    w2.wcs.crpix = [w2.wcs.crpix[0] - ox, w2.wcs.crpix[1] - oy]
    if size is not None:
        (w, h), (w0, h0) = size
        f = w0 / float(w) if w else 1.0
        if abs(f - 1.0) > 1e-6:
            return ScaledSolution(w2, f)
    return w2


class ScaledSolution:
    """A solution for a picture reduced ``f`` times: pixel (x, y) of the picture is pixel ((x + 0.5) f - 0.5, ...) of
    the solved one. Only what the annotations ask (``all_world2pix``, ``wcs_world2pix``, ``all_pix2world``)."""

    def __init__(self, wcs, f: float):
        self.wcs, self.f = wcs, float(f)

    def _down(self, x, y):
        return (np.asarray(x, float) + 0.5) / self.f - 0.5, (np.asarray(y, float) + 0.5) / self.f - 0.5

    def all_world2pix(self, ra, dec, origin, **kw):
        return self._down(*self.wcs.all_world2pix(ra, dec, origin, **kw))

    def wcs_world2pix(self, ra, dec, origin):
        return self._down(*self.wcs.wcs_world2pix(ra, dec, origin))

    def all_pix2world(self, x, y, origin):
        return self.wcs.all_pix2world((np.asarray(x, float) + 0.5) * self.f - 0.5,
                                      (np.asarray(y, float) + 0.5) * self.f - 0.5, origin)


__all__ = ["Named", "Annotations", "named_objects", "field_annotations", "paint", "paint_grid", "paint_objects",
           "burn", "crop_wcs", "Frame"]
