"""Reusable widgets: image viewer, auto-generated parameter form, help box, histogram, journal."""
from __future__ import annotations

import html
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import style


def to_qimage(rgb: np.ndarray) -> QImage:
    """(H, W, 3) uint8 or (3, H, W) float [0,1] -> QImage (copy)."""
    a = np.asarray(rgb)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[2] != 3:
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    a = np.ascontiguousarray(a)
    h, w, _ = a.shape
    return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# ---------------------------------------------------------------------- image viewer
class ImageView(QGraphicsView):
    """Zoom (wheel), pan (drag), double click = fit / 100 %, optional before/after split (drag the divider)."""

    viewChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setBackgroundBrush(QColor("#111215"))
        self.setFrameShape(QFrame.NoFrame)
        self.compare: QPixmap | None = None
        self.compare_label = ("", "")
        self.split = 0.5
        self._drag_split = False
        self._fitted = True
        self.overlay: QPixmap | None = None
        self.placeholder = "No image yet"

    def set_image(self, img: QImage | QPixmap | None, keep_view: bool = True, scale_factor: float = 1.0):
        """Show an image. ``scale_factor`` = full-resolution pixels per pixel of ``img`` (preview renders), so that
        scene coordinates are always full-resolution pixels and switching resolution does not move the view."""
        if img is None:
            self.item.setPixmap(QPixmap())
            self.viewport().update()
            return
        pm = img if isinstance(img, QPixmap) else QPixmap.fromImage(img)
        first = self.item.pixmap().isNull()
        self.item.setScale(scale_factor)
        self.item.setPixmap(pm)
        self.scene().setSceneRect(self.item.sceneBoundingRect())
        if first or not keep_view or self._fitted:
            self.fit()
        self.viewport().update()

    def set_compare(self, img: QImage | QPixmap | None, labels=("before", "after")):
        self.compare = None if img is None else (img if isinstance(img, QPixmap) else QPixmap.fromImage(img))
        self.compare_label = labels
        self.viewport().update()

    def set_overlay(self, img: QImage | None):
        self.overlay = None if img is None else QPixmap.fromImage(img)
        self.viewport().update()

    def fit(self):
        if self.item.pixmap().isNull():
            return
        self.fitInView(self.item.sceneBoundingRect(), Qt.KeepAspectRatio)
        self._fitted = True
        self.viewChanged.emit()

    def one_to_one(self):
        """100 %: one full-resolution image pixel per physical screen pixel."""
        self.resetTransform()
        s = 1.0 / self.devicePixelRatioF()
        self.scale(s, s)
        self._fitted = False
        self.viewChanged.emit()

    def zoom_factor(self) -> float:
        """Physical screen pixels per full-resolution image pixel."""
        return self.transform().m11() * self.devicePixelRatioF()

    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)
        self._fitted = False
        self.viewChanged.emit()

    def mouseDoubleClickEvent(self, e):
        if self._fitted:
            self.one_to_one()
            self.centerOn(self.mapToScene(e.position().toPoint()))
        else:
            self.fit()

    def _split_x(self) -> float | None:
        """Divider position in viewport pixels (a fraction of the window width: always visible when zoomed in)."""
        if self.compare is None or self.item.pixmap().isNull():
            return None
        return self.split * self.viewport().width()

    def mousePressEvent(self, e):
        sx = self._split_x()
        if sx is not None and abs(e.position().x() - sx) < 8 and e.button() == Qt.LeftButton:
            self._drag_split = True
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag_split:
            self.split = float(np.clip(e.position().x() / max(self.viewport().width(), 1), 0, 1))
            self.viewport().update()
            return
        sx = self._split_x()
        self.viewport().setCursor(Qt.SplitHCursor if sx is not None and abs(e.position().x() - sx) < 8
                                  else Qt.OpenHandCursor)
        super().mouseMoveEvent(e)
        self.viewChanged.emit()

    def mouseReleaseEvent(self, e):
        self._drag_split = False
        super().mouseReleaseEvent(e)

    def drawForeground(self, p: QPainter, rect: QRectF):
        r = self.item.sceneBoundingRect()
        if self.item.pixmap().isNull():
            p.save()
            p.resetTransform()
            p.setPen(QColor(style.FG2))
            p.drawText(self.viewport().rect(), Qt.AlignCenter, self.placeholder)
            p.restore()
            return
        if self.overlay is not None:
            p.drawPixmap(r, self.overlay, QRectF(self.overlay.rect()))
        if self.compare is not None:
            vx = self.split * self.viewport().width()
            x = self.mapToScene(int(vx), 0).x()                      # divider in scene coordinates
            xl = max(r.left(), min(x, r.right()))
            if xl > r.left():
                f = (xl - r.left()) / r.width()
                p.drawPixmap(QRectF(r.left(), r.top(), xl - r.left(), r.height()), self.compare,
                             QRectF(0, 0, self.compare.width() * f, self.compare.height()))
            p.save()
            p.resetTransform()
            p.setPen(QPen(QColor(style.ACCENT), 2))
            p.drawLine(int(vx), 0, int(vx), self.viewport().height())
            p.setPen(QColor(style.FG))
            p.drawText(int(vx) - 110, 22, 100, 20, Qt.AlignRight, self.compare_label[0])
            p.drawText(int(vx) + 10, 22, 100, 20, Qt.AlignLeft, self.compare_label[1])
            p.restore()

    def visible_fraction(self) -> tuple[float, float, float, float]:
        """Visible part of the image as fractions (x0, y0, x1, y1)."""
        r = self.item.sceneBoundingRect()
        v = self.mapToScene(self.viewport().rect()).boundingRect()
        if r.width() <= 0:
            return 0, 0, 1, 1
        return ((v.left() - r.left()) / r.width(), (v.top() - r.top()) / r.height(),
                (v.right() - r.left()) / r.width(), (v.bottom() - r.top()) / r.height())


# ---------------------------------------------------------------------- help box
class HelpBox(QTextBrowser):
    """Didactic explanation of the focused parameter (or of the selected stage)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setMinimumHeight(140)

    def show_param(self, d: dict, value=None, origin: str = "", reason: str = ""):
        rng = []
        for k, sym in (("ge", "≥"), ("gt", ">"), ("le", "≤"), ("lt", "<")):
            if d.get(k) is not None:
                rng.append(f"{sym} {d[k]}")
        unit = f" {d['unit']}" if d.get("unit") else ""
        parts = [f"<h3 style='margin:0'>{html.escape(d.get('label', d['name']))}</h3>",
                 f"<p style='color:{style.FG2};margin-top:2px'>{d['name']}"
                 + (f" &middot; range {', '.join(rng)}{unit}" if rng else "") + "</p>",
                 f"<p>{html.escape(d.get('help', ''))}</p>"]
        if d.get("tip"):
            parts.append(f"<p><b style='color:{style.ACCENT}'>When to change it</b><br>{html.escape(d['tip'])}</p>")
        if origin == "auto" and reason:
            parts.append(f"<p><b style='color:{style.OK}'>Automatic value</b> {html.escape(str(value))}{unit}: "
                         f"{html.escape(reason)}</p>")
        elif origin == "user":
            parts.append(f"<p><b style='color:{style.ACCENT}'>Your value</b> {html.escape(str(value))}{unit} "
                         f"(the reset button restores the automatic / default value)</p>")
        self.setHtml("".join(parts))

    def show_text(self, title: str, text: str):
        self.setHtml(f"<h3 style='margin:0'>{html.escape(title)}</h3><p>{html.escape(text)}</p>")


# ---------------------------------------------------------------------- parameter form
class _FloatSlider(QWidget):
    """Slider + spin box for a float with known bounds."""

    valueChanged = Signal(float)

    def __init__(self, lo, hi, step, decimals, parent=None):
        super().__init__(parent)
        self.lo, self.hi = lo, hi
        self.n = max(1, int(round((hi - lo) / step)))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumWidth(60)
        self.slider.setRange(0, self.n)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setFixedWidth(84)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self._lock = False
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, i):
        if self._lock:
            return
        v = self.lo + (self.hi - self.lo) * i / self.n
        self._lock = True
        self.spin.setValue(v)
        self._lock = False
        self.valueChanged.emit(float(self.spin.value()))

    def _from_spin(self, v):
        if self._lock:
            return
        self._lock = True
        self.slider.setValue(int(round((v - self.lo) / (self.hi - self.lo) * self.n)))
        self._lock = False
        self.valueChanged.emit(float(v))

    def setValue(self, v):
        self._lock = True
        self.spin.setValue(v)
        self.slider.setValue(int(round((v - self.lo) / (self.hi - self.lo) * self.n)))
        self._lock = False

    def value(self):
        return float(self.spin.value())


class ParamForm(QWidget):
    """Form generated from a parameter model. Emits ``changed(name, value)``; value ``None`` = reset."""

    changed = Signal(str, object)
    focused = Signal(str)

    def __init__(self, model_cls, values: dict, origins: dict | None = None, reasons: dict | None = None,
                 advanced: bool = False, sliders: bool = False, sections: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.model_cls = model_cls
        self.desc = {d["name"]: d for d in model_cls.describe()}
        self.values = dict(values)
        self.origins = origins or {}
        self.reasons = reasons or {}
        self.widgets: dict = {}
        self.badges: dict = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        groups: dict[str, QGridLayout] = {}
        order = sections or [None]
        boxes = {}
        for sec in order:
            if sec is None:
                g = QGridLayout()
                outer.addLayout(g)
            else:
                box = QGroupBox(sec)
                g = QGridLayout(box)
                boxes[sec] = box
                outer.addWidget(box)
            g.setColumnStretch(1, 1)
            g.setHorizontalSpacing(8)
            groups[sec] = g
        for name, d in self.desc.items():
            if d.get("level") == "internal":        # set by the project itself (stack.exclude): never a form field
                continue
            if d.get("level") == "advanced" and not advanced:
                continue
            sec = d.get("section") if sections else None
            g = groups.get(sec, groups[order[0]])
            row = g.rowCount()
            lab = QLabel(d.get("label", name))
            lab.setToolTip(d.get("help", ""))
            lab.setWordWrap(True)
            lab.setMinimumWidth(110)
            w = self._make_widget(name, d, self.values.get(name, d["default"]), sliders)
            badge = QLabel()
            reset = QPushButton("↺")
            reset.setFixedWidth(26)
            reset.setToolTip("Back to the automatic / default value")
            reset.clicked.connect(lambda _=False, n=name: self.changed.emit(n, None))
            self.badges[name] = (badge, reset)
            g.addWidget(lab, row, 0)
            g.addWidget(w, row, 1)
            g.addWidget(badge, row, 2)
            g.addWidget(reset, row, 3)
            self._update_badge(name)
        outer.addStretch(1)

    def _make_widget(self, name, d, value, sliders):
        t = d["type"]
        if t == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda v, n=name: self._emit(n, bool(v)))
        elif d.get("choices"):
            w = QComboBox()
            w.addItems(d["choices"])
            w.setCurrentText(str(value))
            w.currentTextChanged.connect(lambda v, n=name: self._emit(n, v))
        elif t == "int":
            w = QSpinBox()
            w.setRange(int(d.get("ge", d.get("gt", -10 ** 9))), int(d.get("le", d.get("lt", 10 ** 9))))
            w.setValue(int(value))
            w.valueChanged.connect(lambda v, n=name: self._emit(n, int(v)))
        elif t == "float" and sliders and d.get("ge") is not None and d.get("le") is not None:
            step = d.get("step") or (d["le"] - d["ge"]) / 100
            dec = max(0, min(4, -int(math.floor(math.log10(step))) if step < 1 else 0))
            w = _FloatSlider(d["ge"], d["le"], step, dec)
            w.setValue(float(value))
            w.valueChanged.connect(lambda v, n=name: self._emit(n, v))
        elif t == "float":
            w = QDoubleSpinBox()
            lo = d.get("ge", d.get("gt", -1e12))
            hi = d.get("le", d.get("lt", 1e12))
            w.setRange(lo if lo is not None else -1e12, hi if hi is not None else 1e12)
            step = d.get("step") or 0.1
            w.setDecimals(max(2, min(6, -int(math.floor(math.log10(step))) + 1 if step < 1 else 2)))
            w.setSingleStep(step)
            w.setValue(float(value))
            w.valueChanged.connect(lambda v, n=name: self._emit(n, float(v)))
        else:          # str or optional float
            w = QLineEdit("" if value is None else str(value))
            w.setPlaceholderText("automatic")
            opt_float = "float" in t
            w.editingFinished.connect(lambda n=name, ww=w, of=opt_float: self._emit(n, _parse(ww.text(), of)))
        w.installEventFilter(self)
        w.setProperty("param", name)
        for child in w.findChildren(QWidget):
            child.installEventFilter(self)
            child.setProperty("param", name)
        self.widgets[name] = w
        return w

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if ev.type() in (QEvent.FocusIn, QEvent.Enter) and obj.property("param"):
            self.focused.emit(obj.property("param"))
        return False

    def _emit(self, name, value):
        self.values[name] = value
        self.origins[name] = "user"
        self._update_badge(name)
        self.changed.emit(name, value)

    def _update_badge(self, name):
        badge, reset = self.badges[name]
        o = self.origins.get(name, "default")
        if o == "auto":
            badge.setText("auto")
            badge.setObjectName("badge_auto")
            badge.setToolTip(self.reasons.get(name, ""))
        elif o == "user":
            badge.setText("yours")
            badge.setObjectName("badge_user")
            badge.setToolTip("You changed this value")
        else:
            badge.setText("")
        badge.setStyleSheet("")
        reset.setVisible(o == "user")

    def set_values(self, values: dict, origins: dict | None = None):
        for n, v in values.items():
            w = self.widgets.get(n)
            if w is None:
                continue
            w.blockSignals(True)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(v))
            elif isinstance(w, (QSpinBox, QDoubleSpinBox, _FloatSlider)):
                w.setValue(v)
            elif isinstance(w, QLineEdit):
                w.setText("" if v is None else str(v))
            w.blockSignals(False)
            self.values[n] = v
        if origins:
            self.origins.update(origins)
        for n in self.badges:
            self._update_badge(n)


def _parse(text: str, opt_float: bool):
    text = text.strip()
    if opt_float:
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return text


def scroll(widget: QWidget) -> QScrollArea:
    s = QScrollArea()
    s.setWidget(widget)
    s.setWidgetResizable(True)
    s.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    s.setFrameShape(QFrame.NoFrame)
    return s


# ---------------------------------------------------------------------- histogram
class Histogram(QWidget):
    """RGB + luminance histogram (log counts) with clipping counters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.h = None
        self.clip = (0.0, 0.0)

    def set_image(self, rgb_u8: np.ndarray, valid: np.ndarray | None = None):
        a = rgb_u8 if valid is None else rgb_u8[valid]
        a = a.reshape(-1, 3)
        self.h = [np.bincount(a[:, c], minlength=256) for c in range(3)]
        y = (0.2126 * a[:, 0] + 0.7152 * a[:, 1] + 0.0722 * a[:, 2]).astype(np.uint8)
        self.h.append(np.bincount(y, minlength=256))
        n = max(len(a), 1)
        self.clip = (100 * float((a.max(1) == 0).mean()) if n else 0, 100 * float((a.min(1) == 255).mean()) if n else 0)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(style.BG2))
        if self.h is None:
            return
        w, hgt = self.width(), self.height() - 16
        top = max(float(np.log1p(max(hh.max() for hh in self.h))), 1.0)
        cols = [QColor(224, 90, 80, 150), QColor(110, 190, 110, 150), QColor(90, 150, 220, 150), QColor(230, 226, 220, 220)]
        for hh, col in zip(self.h, cols):
            p.setPen(QPen(col, 1.2))
            pts = [QPointF(i / 255 * w, hgt - np.log1p(hh[i]) / top * hgt) for i in range(256)]
            for a, b in zip(pts[:-1], pts[1:]):
                p.drawLine(a, b)
        p.setPen(QColor(style.FG2))
        p.drawText(4, self.height() - 3, "0")
        p.drawText(self.width() - 26, self.height() - 3, "255")


# ---------------------------------------------------------------------- journal
class JournalView(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)

    def add(self, text: str, level: str = "info"):
        col = {"error": style.ERR, "warning": style.WARN}.get(level, style.FG)
        self.appendHtml(f"<span style='color:{col}'>{html.escape(text)}</span>")


class Debouncer(QTimer):
    """Calls ``fn`` once, ``ms`` after the last ``poke()``."""

    def __init__(self, ms: int, fn, parent=None):
        super().__init__(parent)
        self.setSingleShot(True)
        self.setInterval(ms)
        self.timeout.connect(fn)

    def poke(self):
        self.start()
