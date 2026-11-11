"""Post's viewer (``Viewer``): zoom, pan, 100 %, before / after curtain, overlays, rectangle, brush and spot healing
tools.

It only asks its source for the pixels of the visible region, at full or half resolution (post3.PostSource);
``StaticSource`` is a fixed image. The other views show their pictures in ``imageview.ImageView``.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .i18n import _
from .layout import pinch_factor, wheel_gesture


class StaticSource:
    """A fixed image (H x W x 3 uint8)."""

    def __init__(self, arr: np.ndarray):
        self.arr = np.ascontiguousarray(arr)
        self.half = np.ascontiguousarray(arr[::2, ::2])
        self.version = 0

    def size(self):
        return self.arr.shape[1], self.arr.shape[0]

    def region(self, level, x0, y0, x1, y1):
        a = self.arr if level == 0 else self.half
        return a[y0:y1, x0:x1]


def to_qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class Viewer(QWidget):
    rectDone = Signal(float, float, float, float)        # image coordinates (full resolution of the source)
    stroke = Signal(float, float, bool)                  # x, y, first point of a stroke
    strokeEnd = Signal()
    toolPress = Signal(float, float, int, bool)          # spot healing: x, y, the button (Qt.MouseButton), Alt held
    toolMove = Signal(float, float)
    toolRelease = Signal()
    viewChanged = Signal()
    hovered = Signal(float, float)
    cropChanged = Signal()                               # the crop box was drawn or moved (let go)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)
        self.source = None
        self.compare = None                               # before image for the curtain
        self.curtain = 0.5
        self.scale = 1.0
        self.cx = self.cy = 0.0
        self.mode = "pan"                                 # pan | rect | brush | heal | crop
        self.crop_box = None                              # (x0, y0, x1, y1), image pixels: what is outside is not shown
        self.crop_edit = False                            # the Crop tool is on: outside only dimmed, with the handles
        self.crop_ratio = None                            # width / height of a fixed proportion
        self.brush_r = 30.0
        self.overlays = []                                # callables (painter, viewer)
        self.dim_outside = None                           # (x0, y0, x1, y1): darken outside (test zone)
        self.banner = ""
        self._cache = {}
        self._drag = None
        self._rect = None
        self._mouse = None
        self._fitted = False
        self._space = False                               # Space is held: a left drag pans, whatever the tool
        self._thumb = None                                # the minimap: (source, its version, the small picture)
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------ source and view
    def set_source(self, src, keep_view=False):
        same = self.source is not None and src is not None and self.source.size() == src.size()
        self.source = src
        self._cache.clear()
        if not (keep_view and same):
            self._fitted = False
            self.fit()
        self.update()

    def refresh(self):
        self._cache.clear()
        self.update()

    def fit(self):
        if not self.source:
            return
        w, h = self.source.size()
        x0, y0, x1, y1 = self.crop_box if self.crop_box and not self.crop_edit else (0, 0, w, h)
        self.scale = max(min((self.width() - 16) / (x1 - x0), (self.height() - 16) / (y1 - y0)), 0.02)
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self._fitted = False                              # stays fitted when the widget is resized
        self.viewChanged.emit()
        self.update()

    def zoom_to(self, scale, cx=None, cy=None):
        self.scale = float(np.clip(scale, 0.02, 8.0))
        if cx is not None:
            self.cx, self.cy = cx, cy
        self._fitted = True                               # a chosen view: kept on resize
        self.viewChanged.emit()
        self.update()

    def zoom_by(self, k, at: QPointF | None = None):
        """Zoom by ``k`` keeping the image point under ``at`` (default: the centre) in place."""
        if not self.source:
            return
        at = at or QPointF(self.width() / 2, self.height() / 2)
        ix, iy = self.w_to_img(at.x(), at.y())
        self.scale = float(np.clip(self.scale * k, 0.02, 8.0))
        self.cx = ix - (at.x() - self.width() / 2) / self.scale
        self.cy = iy - (at.y() - self.height() / 2) / self.scale
        self._fitted = True
        self.viewChanged.emit()
        self.update()

    def pan_by(self, dx, dy):
        """The image moved by (dx, dy) pixels of the widget (two fingers on a touchpad)."""
        if not self.source:
            return
        self.cx -= dx / self.scale
        self.cy -= dy / self.scale
        self._fitted = True
        self.viewChanged.emit()
        self.update()

    def img_to_w(self, x, y) -> QPointF:
        return QPointF((x - self.cx) * self.scale + self.width() / 2, (y - self.cy) * self.scale + self.height() / 2)

    def w_to_img(self, px, py):
        return (px - self.width() / 2) / self.scale + self.cx, (py - self.height() / 2) / self.scale + self.cy

    def resizeEvent(self, e):
        if not self._fitted:
            self.fit()
        super().resizeEvent(e)

    # ------------------------------------------------------------ drawing
    def _pixels(self, src, x0, y0, x1, y1):
        level = 1 if self.scale < 0.5 else 0
        f = 2 if level else 1
        w, h = src.size()
        key = (id(src), src.version, level)
        c = self._cache.get(key)
        if c and c[0] <= x0 and c[1] <= y0 and c[2] >= x1 and c[3] >= y1:
            return c
        mx, my = (x1 - x0) * 0.5, (y1 - y0) * 0.5
        ax0, ay0 = max(0, int(x0 - mx) // f * f), max(0, int(y0 - my) // f * f)
        ax1, ay1 = min(w, int(x1 + mx + f)), min(h, int(y1 + my + f))
        arr = src.region(level, ax0 // f, ay0 // f, -(-ax1 // f), -(-ay1 // f))
        c = (ax0, ay0, ax0 + arr.shape[1] * f, ay0 + arr.shape[0] * f, to_qimage(arr), f)
        self._cache = {k: v for k, v in self._cache.items() if k[0] != id(src)}
        self._cache[key] = c
        return c

    def _draw_src(self, p: QPainter, src, clip: QRectF | None = None):
        w, h = src.size()
        ix0, iy0 = self.w_to_img(0, 0)
        ix1, iy1 = self.w_to_img(self.width(), self.height())
        ix0, iy0, ix1, iy1 = max(0, ix0), max(0, iy0), min(w, ix1), min(h, iy1)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        cx0, cy0, cx1, cy1, qi, f = self._pixels(src, ix0, iy0, ix1, iy1)
        target = QRectF(self.img_to_w(cx0, cy0), self.img_to_w(cx1, cy1))
        p.save()
        if clip is not None:
            p.setClipRect(clip)
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale * f < 1.0)
        p.drawImage(target, qi)
        p.restore()

    def paintEvent(self, e):
        p = QPainter(self)
        try:
            self._paint(p)
        finally:
            p.end()

    def _paint(self, p: QPainter):
        p.fillRect(self.rect(), QColor("#0f1115"))
        if not self.source:                              # the first image is awaited: said (issue #69)
            self._paint_banner(p)
            return
        if self.compare is not None:
            xs = self.width() * self.curtain
            self._draw_src(p, self.compare, QRectF(0, 0, xs, self.height()))
            self._draw_src(p, self.source, QRectF(xs, 0, self.width() - xs, self.height()))
        else:
            self._draw_src(p, self.source)
        if self.dim_outside:
            x0, y0, x1, y1 = self.dim_outside
            r = QRectF(self.img_to_w(x0, y0), self.img_to_w(x1, y1))
            p.save()
            p.setClipRegion(self._outside(r))
            p.fillRect(self.rect(), QColor(15, 17, 21, 200))
            p.restore()
            p.setPen(QPen(QColor("#e6a23c"), 1.5, Qt.DashLine))
            p.drawRect(r)
        if self.crop_box:
            self._paint_crop(p)
        for ov in self.overlays:
            ov(p, self)
        if self.compare is not None:
            xs = self.width() * self.curtain
            p.setPen(QPen(QColor("#ffffff"), 2))
            p.drawLine(QPointF(xs, 0), QPointF(xs, self.height()))
            p.setBrush(QColor("#ffffff"))
            p.drawEllipse(QPointF(xs, self.height() / 2), 7, 7)
            self._tag(p, QPointF(xs - 12, 14), _("Before"), right=True)
            self._tag(p, QPointF(xs + 12, 14), _("After"))
        if self._rect:
            a, b = self._rect
            p.setPen(QPen(QColor("#e6a23c"), 1.5, Qt.DashLine))
            p.setBrush(QColor(230, 162, 60, 40))
            p.drawRect(QRectF(self.img_to_w(*a), self.img_to_w(*b)).normalized())
        if self.mode in ("brush", "heal") and self._mouse is not None:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 200), 1.2))
            p.drawEllipse(self._mouse, self.brush_r * self.scale, self.brush_r * self.scale)
        if not self.whole_visible():
            self._minimap(p)
        self._paint_banner(p)

    def _paint_crop(self, p: QPainter):
        r = QRectF(self.img_to_w(*self.crop_box[:2]), self.img_to_w(*self.crop_box[2:]))
        p.save()
        p.setClipRegion(self._outside(r))
        p.fillRect(self.rect(), QColor(15, 17, 21, 170 if self.crop_edit else 255))
        p.restore()
        if not self.crop_edit:
            return
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 90), 1))
        for k in (1, 2):                                 # the thirds
            x, y = r.left() + r.width() * k / 3, r.top() + r.height() * k / 3
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        p.setPen(QPen(QColor("#e6a23c"), 1.5))
        p.drawRect(r)
        p.setBrush(QColor("#e6a23c"))
        for pt in self._crop_points(r).values():
            p.drawRect(QRectF(pt.x() - 4, pt.y() - 4, 8, 8))

    @staticmethod
    def _crop_points(r: QRectF) -> dict:
        c = r.center()
        return {"nw": r.topLeft(), "n": QPointF(c.x(), r.top()), "ne": r.topRight(), "e": QPointF(r.right(), c.y()),
                "se": r.bottomRight(), "s": QPointF(c.x(), r.bottom()), "sw": r.bottomLeft(), "w": QPointF(r.left(), c.y())}

    def _crop_handle(self, pos: QPointF) -> str | None:
        """The handle under the mouse, "move" inside the box, None outside it."""
        if not self.crop_box:
            return None
        r = QRectF(self.img_to_w(*self.crop_box[:2]), self.img_to_w(*self.crop_box[2:]))
        for name, pt in self._crop_points(r).items():
            if abs(pos.x() - pt.x()) <= 8 and abs(pos.y() - pt.y()) <= 8:
                return name
        return "move" if r.contains(pos) else None

    def _crop_moved(self, handle: str, box, start, ix: float, iy: float):
        """The box while a handle is dragged, inside the image and at its fixed proportion if it has one."""
        W, H = self.source.size()
        x0, y0, x1, y1 = box
        if handle == "move":
            dx = float(np.clip(ix - start[0], -x0, W - x1))
            dy = float(np.clip(iy - start[1], -y0, H - y1))
            return x0 + dx, y0 + dy, x1 + dx, y1 + dy
        has_x, has_y = any(c in handle for c in "ew"), any(c in handle for c in "ns")
        ax = (x1 if "w" in handle else x0) if has_x else (x0 + x1) / 2     # what stays still
        ay = (y1 if "n" in handle else y0) if has_y else (y0 + y1) / 2
        sx, sy = (1 if ix >= ax else -1), (1 if iy >= ay else -1)
        w = abs(ix - ax) if has_x else x1 - x0
        h = abs(iy - ay) if has_y else y1 - y0
        maxw = (W - ax if sx > 0 else ax) if has_x else 2 * min(ax, W - ax)
        maxh = (H - ay if sy > 0 else ay) if has_y else 2 * min(ay, H - ay)
        if self.crop_ratio:
            r = self.crop_ratio
            if has_x and has_y:
                w = max(w, h * r)
            elif has_y:
                w = h * r
            h = w / r
            f = min(1.0, maxw / max(w, 1e-6), maxh / max(h, 1e-6))
            w, h = w * f, h * f
        else:
            w, h = min(w, maxw), min(h, maxh)
        a, b = sorted((ax, ax + sx * w)) if has_x else (ax - w / 2, ax + w / 2)
        c, d = sorted((ay, ay + sy * h)) if has_y else (ay - h / 2, ay + h / 2)
        return a, c, b, d

    def _paint_banner(self, p: QPainter):
        if not self.banner:
            return
        f = QFont(self.font())
        f.setBold(True)
        p.setFont(f)
        m = p.fontMetrics()
        w = m.horizontalAdvance(self.banner) + 24
        r = QRectF((self.width() - w) / 2, 10, w, m.height() + 10)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(230, 162, 60, 230))
        p.drawRoundedRect(r, 4, 4)
        p.setPen(QColor("#15171c"))
        p.drawText(r, Qt.AlignCenter, self.banner)

    # ------------------------------------------------------------ minimap (issue #44: as the Inspect viewer's)
    def extent(self) -> tuple:
        """What the picture is, in image pixels (x0, y0, x1, y1): the crop once applied, else the whole image (#144:
        the minimap showed the whole image over a cropped picture fitted to the view)."""
        w, h = self.source.size()
        return tuple(self.crop_box) if self.crop_box and not self.crop_edit else (0, 0, w, h)

    def whole_visible(self) -> bool:
        if not self.source:
            return True
        ex0, ey0, ex1, ey1 = self.extent()
        x0, y0 = self.w_to_img(0, 0)
        x1, y1 = self.w_to_img(self.width(), self.height())
        return x0 <= ex0 and y0 <= ey0 and x1 >= ex1 and y1 >= ey1

    def minimap_rect(self) -> QRectF:
        x0, y0, x1, y1 = self.extent()
        w, h = max(x1 - x0, 1), max(y1 - y0, 1)
        f = min(190.0 / w, 130.0 / h)
        return QRectF(self.width() - w * f - 12, self.height() - h * f - 12, w * f, h * f)

    def _in_minimap(self, pos: QPointF) -> bool:
        return bool(self.source) and not self.whole_visible() and \
            self.minimap_rect().adjusted(-4, -4, 4, 4).contains(pos)

    def _thumbnail(self) -> QImage:
        """The small picture of the minimap, cut once per version of the image and per crop."""
        src = self.source
        ext = self.extent()
        c = self._thumb
        if c is not None and c[0] is src and c[1] == src.version and c[3] == ext:
            return c[2]
        x0, y0, x1, y1 = (int(round(v / 2)) for v in ext)                  # (the half resolution's pixels)
        a = np.ascontiguousarray(src.region(1, x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)))
        step = max(1, int(max(a.shape[1], a.shape[0]) / 200))
        qi = to_qimage(a[::step, ::step])
        self._thumb = (src, src.version, qi, ext)
        return qi

    def _minimap(self, p: QPainter):
        r = self.minimap_rect()
        p.save()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(15, 17, 21, 220))
        p.drawRoundedRect(r.adjusted(-4, -4, 4, 4), 5, 5)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(r, self._thumbnail())
        ex0, ey0, ex1, ey1 = self.extent()
        f = r.width() / max(ex1 - ex0, 1)
        x0, y0 = self.w_to_img(0, 0)
        x1, y1 = self.w_to_img(self.width(), self.height())
        x0, y0, x1, y1 = max(x0, ex0), max(y0, ey0), min(x1, ex1), min(y1, ey1)
        v = QRectF(r.x() + (x0 - ex0) * f, r.y() + (y0 - ey0) * f, (x1 - x0) * f, (y1 - y0) * f)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#5ea1ff"), 1.5))
        p.drawRect(v)
        p.restore()

    def _minimap_goto(self, pos: QPointF):
        r = self.minimap_rect()
        ex0, ey0, ex1, _ey1 = self.extent()
        f = r.width() / max(ex1 - ex0, 1)
        self.cx = ex0 + (pos.x() - r.x()) / f
        self.cy = ey0 + (pos.y() - r.y()) / f
        self._fitted = True
        self.viewChanged.emit()
        self.update()

    def _outside(self, r: QRectF):
        from PySide6.QtGui import QRegion
        return QRegion(self.rect()).subtracted(QRegion(r.toRect()))

    def _tag(self, p, pt: QPointF, text, right=False):
        m = p.fontMetrics()
        w = m.horizontalAdvance(text) + 12
        r = QRectF(pt.x() - w if right else pt.x(), pt.y() - 2, w, m.height() + 4)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(r, 3, 3)
        p.setPen(QColor("#ffffff"))
        p.drawText(r, Qt.AlignCenter, text)

    # ------------------------------------------------------------ interaction
    def wheelEvent(self, e):
        if not self.source:
            return
        kind, a, b = wheel_gesture(e)                      # a touchpad's two fingers pan, its pinch (Ctrl) zooms (#44)
        if kind == "pan":
            self.pan_by(a, b)
        elif a:
            self.zoom_by(1.2 ** a, e.position())
        e.accept()

    def event(self, e):
        if e.type() == QEvent.ShortcutOverride and e.key() == Qt.Key_1 and e.modifiers() == Qt.ControlModifier:
            e.accept()                                     # 100 % here, not the window's "Graph"
            return True
        f = pinch_factor(e)
        if f is not None:                                  # a native pinch (macOS, Wayland)
            self.zoom_by(f, e.position())
            return True
        return super().event(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_1 and e.modifiers() == Qt.ControlModifier:
            self.zoom_to(1.0)
            return
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            return
        super().keyReleaseEvent(e)

    def _near_curtain(self, x):
        return self.compare is not None and abs(x - self.width() * self.curtain) < 10

    def mousePressEvent(self, e):
        pos = e.position()
        ix, iy = self.w_to_img(pos.x(), pos.y())
        if e.button() == Qt.LeftButton and self._in_minimap(pos):
            self._drag = ("minimap",)
            self._minimap_goto(pos)
        elif e.button() == Qt.LeftButton and self._space:
            self._drag = ("pan", pos, self.cx, self.cy)
        elif e.button() == Qt.LeftButton and self._near_curtain(pos.x()):
            self._drag = ("curtain",)
        elif e.button() == Qt.LeftButton and self.mode == "crop" and self.source:
            handle = self._crop_handle(pos)
            W, H = self.source.size()
            x, y = float(np.clip(ix, 0, W)), float(np.clip(iy, 0, H))
            box = self.crop_box if handle else (x, y, x, y)     # outside the box: a new one, from here
            self._drag = ("crop", handle or "se", box, (ix, iy), self.crop_box)
            self.crop_box = box
        elif e.button() == Qt.LeftButton and self.mode == "rect":
            self._rect = [(ix, iy), (ix, iy)]
            self._drag = ("rect",)
        elif e.button() == Qt.LeftButton and self.mode == "brush":
            self._drag = ("brush",)
            self.stroke.emit(ix, iy, True)
        elif e.button() == Qt.LeftButton and self.mode == "heal":
            self._drag = ("heal",)
            self.toolPress.emit(ix, iy, int(e.button().value), bool(e.modifiers() & Qt.AltModifier))
        else:
            self._drag = ("pan", pos, self.cx, self.cy)
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        ix, iy = self.w_to_img(pos.x(), pos.y())
        self._mouse = pos
        self.hovered.emit(ix, iy)
        d = self._drag
        if d is None and self._in_minimap(pos):
            self.setCursor(Qt.PointingHandCursor)
        elif d is None:
            self.setCursor(Qt.SplitHCursor if self._near_curtain(pos.x()) else
                           (Qt.CrossCursor if self.mode in ("rect", "crop") else
                            (Qt.BlankCursor if self.mode in ("brush", "heal") else Qt.OpenHandCursor)))
        elif d[0] == "minimap":
            self._minimap_goto(pos)
        elif d[0] == "curtain":
            self.curtain = float(np.clip(pos.x() / max(self.width(), 1), 0.02, 0.98))
        elif d[0] == "crop":
            self.crop_box = self._crop_moved(d[1], d[2], d[3], ix, iy)
        elif d[0] == "rect":
            self._rect[1] = (ix, iy)
        elif d[0] == "brush":
            self.stroke.emit(ix, iy, False)
        elif d[0] == "heal":
            self.toolMove.emit(ix, iy)
        elif d[0] == "pan":
            self.cx = d[2] - (pos.x() - d[1].x()) / self.scale
            self.cy = d[3] - (pos.y() - d[1].y()) / self.scale
            self._fitted = True
            self.viewChanged.emit()
        self.update()

    def mouseReleaseEvent(self, e):
        d = self._drag
        self._drag = None
        if d and d[0] == "crop":
            x0, y0, x1, y1 = self.crop_box
            if x1 - x0 < 8 or y1 - y0 < 8:               # a click, not a box: as it was
                self.crop_box = d[4]
            else:
                self.cropChanged.emit()
        elif d and d[0] == "rect" and self._rect:
            (ax, ay), (bx, by) = self._rect
            self._rect = None
            if abs(bx - ax) > 8 and abs(by - ay) > 8:
                self.rectDone.emit(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
        elif d and d[0] == "brush":
            self.strokeEnd.emit()
        elif d and d[0] == "heal":
            self.toolRelease.emit()
        self.update()

    def end_crop_drag(self):
        """The Crop tool left while its box is dragged (Escape): the release that follows changes nothing."""
        if self._drag and self._drag[0] == "crop":
            self._drag = None

    def leaveEvent(self, e):
        self._mouse = None
        self.update()

    def mouseDoubleClickEvent(self, e):
        if self.mode == "pan":
            self.fit()
