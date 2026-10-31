"""The image viewer: zoom, pan, 100 %, before / after curtain, overlays, rectangle and brush tools.

Kept in its own module on purpose: it is the one piece that could be rewritten natively later without touching
the rest. It only asks its source for the pixels of the visible region, at full or half resolution.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .i18n import _


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
    viewChanged = Signal()
    hovered = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)
        self.source = None
        self.compare = None                               # before image for the curtain
        self.curtain = 0.5
        self.scale = 1.0
        self.cx = self.cy = 0.0
        self.mode = "pan"                                 # pan | rect | brush
        self.brush_r = 30.0
        self.overlays = []                                # callables (painter, viewer)
        self.dim_outside = None                           # (x0, y0, x1, y1): darken outside (test zone)
        self.banner = ""
        self._cache = {}
        self._drag = None
        self._rect = None
        self._mouse = None
        self._fitted = False
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
        self.scale = max(min((self.width() - 16) / w, (self.height() - 16) / h), 0.02)
        self.cx, self.cy = w / 2, h / 2
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

    def show_rect(self, x0, y0, x1, y1):
        s = min((self.width() - 40) / max(x1 - x0, 1), (self.height() - 40) / max(y1 - y0, 1))
        self.zoom_to(s, (x0 + x1) / 2, (y0 + y1) / 2)
        self._fitted = True

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
        if not self.source:
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
        if self.mode == "brush" and self._mouse is not None:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 200), 1.2))
            p.drawEllipse(self._mouse, self.brush_r * self.scale, self.brush_r * self.scale)
        if self.banner:
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
        pos = e.position()
        ix, iy = self.w_to_img(pos.x(), pos.y())
        k = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        self.scale = float(np.clip(self.scale * k, 0.02, 8.0))
        self.cx = ix - (pos.x() - self.width() / 2) / self.scale
        self.cy = iy - (pos.y() - self.height() / 2) / self.scale
        self._fitted = True
        self.viewChanged.emit()
        self.update()

    def _near_curtain(self, x):
        return self.compare is not None and abs(x - self.width() * self.curtain) < 10

    def mousePressEvent(self, e):
        pos = e.position()
        ix, iy = self.w_to_img(pos.x(), pos.y())
        if e.button() == Qt.LeftButton and self._near_curtain(pos.x()):
            self._drag = ("curtain",)
        elif e.button() == Qt.LeftButton and self.mode == "rect":
            self._rect = [(ix, iy), (ix, iy)]
            self._drag = ("rect",)
        elif e.button() == Qt.LeftButton and self.mode == "brush":
            self._drag = ("brush",)
            self.stroke.emit(ix, iy, True)
        else:
            self._drag = ("pan", pos, self.cx, self.cy)
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        ix, iy = self.w_to_img(pos.x(), pos.y())
        self._mouse = pos
        self.hovered.emit(ix, iy)
        d = self._drag
        if d is None:
            self.setCursor(Qt.SplitHCursor if self._near_curtain(pos.x()) else
                           (Qt.CrossCursor if self.mode in ("rect",) else
                            (Qt.BlankCursor if self.mode == "brush" else Qt.OpenHandCursor)))
        elif d[0] == "curtain":
            self.curtain = float(np.clip(pos.x() / max(self.width(), 1), 0.02, 0.98))
        elif d[0] == "rect":
            self._rect[1] = (ix, iy)
        elif d[0] == "brush":
            self.stroke.emit(ix, iy, False)
        elif d[0] == "pan":
            self.cx = d[2] - (pos.x() - d[1].x()) / self.scale
            self.cy = d[3] - (pos.y() - d[1].y()) / self.scale
            self._fitted = True
            self.viewChanged.emit()
        self.update()

    def mouseReleaseEvent(self, e):
        d = self._drag
        self._drag = None
        if d and d[0] == "rect" and self._rect:
            (ax, ay), (bx, by) = self._rect
            self._rect = None
            if abs(bx - ax) > 8 and abs(by - ay) > 8:
                self.rectDone.emit(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
        elif d and d[0] == "brush":
            self.strokeEnd.emit()
        self.update()

    def leaveEvent(self, e):
        self._mouse = None
        self.update()

    def mouseDoubleClickEvent(self, e):
        if self.mode == "pan":
            self.fit()
