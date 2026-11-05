"""The Curves editor of Post, and the graph of a stretch's transfer function.

As PixInsight's CurvesTransformation, Siril 1.4's Curves and Photoshop's Curves: a curve per channel over the live
histogram of the filter's input (sent by the render worker), control points added, moved and deleted by hand.

* Click on the graph: a point there; drag a point to move it (it stays between its neighbours; the end points move too,
  Photoshop's black and white points); Ctrl+click, a right-click or a drag out of the graph deletes it (PixInsight's and
  Photoshop's gestures); the arrow keys nudge the selected point by one 8-bit level (Shift: ten), Delete deletes it.
* Input and Output give the selected point's coordinates; Interpolation is the current curve's (Akima, cubic spline,
  linear); Presets apply a whole set of curves (Photoshop's, the astrophotography classics, the user's own); Save…
  keeps the curves as a preset of the user's; Import… reads a Photoshop .acv or a .json, Export… writes one.
* ``changed`` is emitted while a point moves (the view sends a draft), ``committed`` when it is let go (one undo step
  and the exact render).

Nothing is rendered here: the curve is drawn from its points (postfilters.curves.Spline), the histogram comes from the
worker.

Classes: ``CurvesEditor`` (the panel), ``CurveGraph`` (the plot it edits), ``FunctionGraph`` (the read-only graph of a
stretch's transfer function).
"""
from __future__ import annotations

import copy
import math

import numpy as np
from PySide6.QtCore import QLocale, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .i18n import _, language
from .postfilters import curves as pc
from .widgets import ACCENT, label

CH_COL = {"RGB/K": "#e8ebf0", "R": "#ff6b6b", "G": "#5fd17f", "B": "#6aa8ff", "L*": "#e8ebf0", "S": "#e0a8ff",
          "c*": "#ffb86b", "H": "#e8ebf0"}
CH_LABEL = {"RGB/K": "RGB"}                          # what the channel button shows (translated: RVB in French)


def ch_label(ch: str) -> str:
    return _(CH_LABEL[ch]) if ch in CH_LABEL else ch


HIST_COL = {"R": QColor(120, 34, 34), "G": QColor(34, 104, 46), "B": QColor(34, 56, 132)}
PLOT_H = 290
BAR = 9                                              # the gradient bars under and left of the plot
PICK = 9                                             # px: how near a point the mouse must be to take it
OUT = 26                                             # px out of the plot: a dragged point is deleted when let go
GROUPS = (("Classic tone curves", "classic"), ("Astrophotography", None))


def _hue_gradient(g: QLinearGradient):
    for k in range(7):
        g.setColorAt(k / 6, QColor.fromHsvF((k / 6) % 1.0, 0.85, 0.95))


class CurveGraph(QWidget):
    """The plot: the gradient bars, the histogram, the grid, the diagonal, the curve and its points. Read-only (a
    stretch's transfer function, ``func``) or editable (``curve``: the dict it edits in place)."""
    starting = Signal()                              # an edit begins (before the points change: the undo snapshot)
    moved = Signal()                                 # a point moves
    released = Signal()                              # a point was let go, added or deleted
    selected = Signal()                              # the selected point changed (the Input / Output fields)

    def __init__(self, parent=None, readonly=False):
        super().__init__(parent)
        self.readonly = readonly
        self.curve: dict | None = None
        self.func = None                             # (read-only) xs -> ys
        self.channel = "RGB/K"
        self.hist: dict | None = None
        self.log = False
        self.sel: int | None = None
        self._drag = False
        self._out = False
        self._hover: tuple[float, float] | None = None
        self.setMinimumHeight(PLOT_H if not readonly else 170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus if not readonly else Qt.NoFocus)

    # ------------------------------------------------------------ geometry
    def plot(self) -> QRectF:
        return QRectF(BAR + 6, 4, self.width() - BAR - 12, self.height() - BAR - 12)

    def to_w(self, x, y) -> QPointF:
        r = self.plot()
        return QPointF(r.left() + x * r.width(), r.bottom() - y * r.height())

    def to_c(self, px, py) -> tuple[float, float]:
        r = self.plot()
        return (px - r.left()) / r.width(), (r.bottom() - py) / r.height()

    # ------------------------------------------------------------ data
    def pts(self) -> list[list[float]]:
        return self.curve["pts"] if self.curve else [[0.0, 0.0], [1.0, 1.0]]

    def values(self, n=361) -> tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(0, 1, n)
        if self.func is not None:
            return xs, np.clip(np.asarray(self.func(xs), np.float64), 0, 1)
        return xs, pc.Spline(self.pts(), (self.curve or {}).get("interp", pc.INTERPS[0]))(xs)

    # ------------------------------------------------------------ drawing
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.plot()
        p.fillRect(self.rect(), QColor("#15171c"))
        p.fillRect(r, QColor("#101217"))
        self._bars(p, r)
        self._histogram(p, r)
        p.setPen(QPen(QColor("#2c323d"), 1))
        for k in range(1, 4):                         # Photoshop's 4 x 4 grid
            x = r.left() + r.width() * k / 4
            y = r.top() + r.height() * k / 4
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        p.setPen(QPen(QColor("#4a5160"), 1, Qt.DashLine))
        p.drawLine(self.to_w(0, 0), self.to_w(1, 1))
        p.setPen(QPen(QColor("#3a4150"), 1))
        p.drawRect(r)
        xs, ys = self.values()
        path = QPainterPath(self.to_w(xs[0], ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(self.to_w(x, y))
        p.setPen(QPen(QColor(CH_COL.get(self.channel, "#e8ebf0")), 2))
        p.drawPath(path)
        if not self.readonly:
            for i, (x, y) in enumerate(self.pts()):
                c = self.to_w(x, y)
                on = i == self.sel
                gone = on and self._out
                p.setPen(QPen(QColor("#e06c6c" if gone else (ACCENT if on else "#e8ebf0")), 1.5))
                p.setBrush(QBrush(QColor(ACCENT if on and not gone else "#101217")))
                p.drawRect(QRectF(c.x() - 4, c.y() - 4, 8, 8))
            if self._hover is not None and not self._drag:
                hx, hy = self._hover
                p.setPen(QPen(QColor("#8b93a1"), 1, Qt.DotLine))
                c = self.to_w(hx, hy)
                p.drawLine(QPointF(c.x(), r.bottom()), c)
        p.end()

    def _bars(self, p, r):
        gx = QLinearGradient(r.left(), 0, r.right(), 0)
        gy = QLinearGradient(0, r.bottom(), 0, r.top())
        if self.channel == "H":
            _hue_gradient(gx)
            _hue_gradient(gy)
        else:
            end = {"R": "#ff3b3b", "G": "#3bd16b", "B": "#3b7bff", "S": "#e05bd0",
                   "c*": "#ff9a3b"}.get(self.channel, "#ffffff")
            start = "#7a7a7a" if self.channel in ("S", "c*") else "#000000"
            for g in (gx, gy):
                g.setColorAt(0, QColor(start))
                g.setColorAt(1, QColor(end))
        p.fillRect(QRectF(r.left(), r.bottom() + 3, r.width(), BAR), QBrush(gx))
        p.fillRect(QRectF(r.left() - BAR - 3, r.top(), BAR, r.height()), QBrush(gy))

    def _histogram(self, p, r):
        h = self.hist
        if not h:
            return
        if self.channel in ("RGB/K", "R", "G", "B") or self.readonly:
            keys = ["R", "G", "B"] if self.channel == "RGB/K" or self.readonly else [self.channel]
        else:
            keys = [self.channel]
        arrs = [np.asarray(h[k], np.float64) for k in keys if k in h]
        if not arrs:
            return
        tr = (np.log1p if self.log else (lambda a: a))
        # the tallest bin but the two ends (the clipped black and white would flatten the rest), as Photoshop
        top = max(float(tr(a[1:-1]).max()) if len(a) > 2 else float(tr(a).max()) for a in arrs) or 1.0
        p.save()
        p.setPen(Qt.NoPen)
        if len(keys) > 1:                               # the channels added: grey where they overlap, as Photoshop
            p.setCompositionMode(QPainter.CompositionMode_Plus)
        for k, a in zip(keys, arrs):
            v = np.clip(tr(a) / top, 0, 1)
            poly = QPolygonF([QPointF(r.left(), r.bottom())])
            n = len(v)
            for i, y in enumerate(v):
                poly.append(QPointF(r.left() + r.width() * i / (n - 1), r.bottom() - y * r.height() * 0.96))
            poly.append(QPointF(r.right(), r.bottom()))
            p.setBrush(HIST_COL.get(k, QColor(78, 84, 96)) if len(keys) > 1 or k in "RGB" else QColor(78, 84, 96))
            p.drawPolygon(poly)
        p.restore()

    # ------------------------------------------------------------ editing
    def _nearest(self, pos) -> int | None:
        best, bi = PICK + 1, None
        for i, (x, y) in enumerate(self.pts()):
            c = self.to_w(x, y)
            d = math.hypot(c.x() - pos.x(), c.y() - pos.y())
            if d < best:
                best, bi = d, i
        return bi

    def delete(self, i: int) -> bool:
        pts = self.pts()
        if self.readonly or i is None or i in (0, len(pts) - 1) or len(pts) <= 2:
            return False
        self.starting.emit()
        del pts[i]
        self.sel = None
        self.update()
        self.selected.emit()
        self.released.emit()
        return True

    def add(self, x: float, y: float) -> int | None:
        pts = self.pts()
        if len(pts) >= pc.MAX_POINTS or any(abs(px - x) < pc.EPS for px, _py in pts):
            return None
        if x < pts[0][0] or x > pts[-1][0]:
            return None
        pts.append([x, y])
        pts.sort(key=lambda q: q[0])
        self.sel = next(i for i, q in enumerate(pts) if q[0] == x)
        self.selected.emit()
        return self.sel

    def set_point(self, i: int, x: float, y: float):
        pts = self.pts()
        lo = pts[i - 1][0] + pc.EPS if i > 0 else 0.0
        hi = pts[i + 1][0] - pc.EPS if i < len(pts) - 1 else 1.0
        pts[i][0] = float(min(max(x, lo), hi))
        pts[i][1] = float(min(max(y, 0.0), 1.0))
        self.update()

    def mousePressEvent(self, e):
        if self.readonly or self.curve is None:
            return
        self.setFocus()
        pos = e.position()
        i = self._nearest(pos)
        if e.button() == Qt.RightButton or (e.button() == Qt.LeftButton and e.modifiers() & Qt.ControlModifier):
            if i is not None:
                self.delete(i)
            return
        if e.button() != Qt.LeftButton:
            return
        self.starting.emit()
        if i is None:
            x, y = self.to_c(pos.x(), pos.y())
            if not (0 <= x <= 1 and -0.02 <= y <= 1.02):
                return
            i = self.add(x, min(max(y, 0.0), 1.0))
            if i is None:
                return
            self.moved.emit()
        self.sel = i
        self._drag, self._out = True, False
        self.selected.emit()
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._drag and self.sel is not None:
            r = self.plot()
            self._out = self.sel not in (0, len(self.pts()) - 1) and not r.adjusted(-OUT, -OUT, OUT, OUT).contains(pos)
            x, y = self.to_c(pos.x(), pos.y())
            self.set_point(self.sel, x, y)
            self.selected.emit()
            self.moved.emit()
            return
        x, _y = self.to_c(pos.x(), pos.y())
        if 0 <= x <= 1 and not self.readonly:
            sp = pc.Spline(self.pts(), (self.curve or {}).get("interp", pc.INTERPS[0]))
            self._hover = (x, float(sp(x)))
        else:
            self._hover = None
        self.update()

    def mouseReleaseEvent(self, e):
        if not self._drag:
            return
        self._drag = False
        if self._out and self.sel is not None:
            self._out = False
            self.delete(self.sel)
            return
        self.released.emit()
        self.update()

    def leaveEvent(self, e):
        self._hover = None
        self.update()

    def keyPressEvent(self, e):
        if self.readonly or self.sel is None or self.curve is None:
            return super().keyPressEvent(e)
        k = e.key()
        if k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete(self.sel)
            return
        step = (10 if e.modifiers() & Qt.ShiftModifier else 1) / 255
        d = {Qt.Key_Left: (-step, 0), Qt.Key_Right: (step, 0), Qt.Key_Up: (0, step), Qt.Key_Down: (0, -step)}.get(k)
        if d is None:
            return super().keyPressEvent(e)
        self.starting.emit()
        x, y = self.pts()[self.sel]
        self.set_point(self.sel, x + d[0], y + d[1])
        self.selected.emit()
        self.moved.emit()
        self.released.emit()


class CurvesEditor(QWidget):
    """The Curves tool's panel, editing ``curves`` (a layer's params["curves"]) in place."""
    starting = Signal()                              # an edit begins: the curves are about to change
    changed = Signal()                               # a draft: something moves
    committed = Signal()                             # the end of an edit: one undo step
    channelChanged = Signal(str)                     # the channel shown (the view keeps it across rebuilds)

    def __init__(self, curves: dict, hist: dict | None = None, parent=None, presets_path=None, channel="RGB/K"):
        super().__init__(parent)
        self.curves = curves
        self.presets_path = presets_path
        self.channel = "RGB/K"
        self._busy = False
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(5)
        row = QHBoxLayout()
        row.setSpacing(2)
        self.ch_buttons: dict[str, QPushButton] = {}
        for ch in pc.CHANNELS:
            b = QPushButton(ch_label(ch))
            b.setCheckable(True)
            b.setObjectName("chan")
            b.setToolTip(_(pc.CHANNEL_TIPS[ch]))
            b.setMinimumWidth(30)
            b.clicked.connect(lambda _c=False, c=ch: self.set_channel(c))
            row.addWidget(b)
            self.ch_buttons[ch] = b
        v.addLayout(row)
        self.graph = CurveGraph(self)
        self.graph.hist = hist
        v.addWidget(self.graph)
        io = QHBoxLayout()
        io.setSpacing(6)
        self.inp, self.out = QDoubleSpinBox(), QDoubleSpinBox()
        for sb in (self.inp, self.out):
            sb.setRange(0.0, 1.0)
            sb.setDecimals(3)
            sb.setSingleStep(1 / 255)
            sb.setLocale(QLocale(QLocale.French) if language() == "fr" else QLocale.c())
            sb.setFixedWidth(74)
            sb.editingFinished.connect(self._typed)
        io.addWidget(label(_("Input"), "dim"))
        io.addWidget(self.inp)
        io.addWidget(label(_("Output"), "dim"))
        io.addWidget(self.out)
        io.addStretch(1)
        self.interp = QComboBox()
        for it in pc.INTERPS:
            self.interp.addItem(_(it), it)
        self.interp.setToolTip(_("Interpolation of this curve: Akima (the usual default), the natural cubic spline, "
                                 "or straight segments"))
        self.interp.currentIndexChanged.connect(self._interp)
        io.addWidget(self.interp)
        v.addLayout(io)
        pr = QHBoxLayout()
        pr.setSpacing(4)
        pr.addWidget(label(_("Preset"), "dim"))
        self.preset = QComboBox()
        self.preset.setMinimumWidth(150)
        self.preset.activated.connect(self._preset_chosen)
        pr.addWidget(self.preset, 1)
        v.addLayout(pr)
        self.b_save = QPushButton(_("Save…"))
        self.b_save.setToolTip(_("Keep these curves as a preset of yours"))
        self.b_save.clicked.connect(lambda: self.save_preset())
        self.b_more = QPushButton("⋯")
        self.b_more.ensurePolished()                    # its width from the style's padding: never narrower than "⋯"
        self.b_more.setFixedWidth(max(28, self.b_more.sizeHint().width()))
        self.more = QMenu(self.b_more)
        self.a_delete = self.more.addAction(_("Delete this preset"))
        self.a_delete.triggered.connect(self.delete_preset)
        self.more.addSeparator()
        self.more.addAction(_("Import a curves file (.acv, .json)…")).triggered.connect(lambda: self.import_file())
        self.more.addAction(_("Export these curves (.acv, .json)…")).triggered.connect(lambda: self.export_file())
        self.b_more.setMenu(self.more)
        self.log = QCheckBox(_("Log histogram"))
        self.log.setChecked(False)
        self.log.toggled.connect(self._log)
        bt = QHBoxLayout()                              # (two short rows: the French is long)
        bt.setSpacing(4)
        bt.addWidget(self.b_save)
        bt.addWidget(self.b_more)
        bt.addStretch(1)
        bt.addWidget(self.log)
        v.addLayout(bt)
        self.b_reset = QPushButton(_("Reset channel"))
        self.b_reset.clicked.connect(self.reset_channel)
        self.b_reset_all = QPushButton(_("Reset all"))
        self.b_reset_all.clicked.connect(self.reset_all)
        rs = QHBoxLayout()
        rs.setSpacing(4)
        rs.addWidget(self.b_reset)
        rs.addWidget(self.b_reset_all)
        rs.addStretch(1)
        v.addLayout(rs)
        self.hint = label(_("Click: add a point · drag: move · Ctrl+click or drag out: delete · arrows: nudge"),
                          "small")
        self.hint.setWordWrap(True)
        v.addWidget(self.hint)
        self.setStyleSheet("QPushButton#chan { padding: 3px 4px; font-weight: 600; }"
                           f"QPushButton#chan:checked {{ background: #2b3446; border-color: {ACCENT}; color: #ffffff; }}")
        self.graph.starting.connect(self.starting.emit)
        self.graph.moved.connect(self._moved)
        self.graph.released.connect(self._released)
        self.graph.selected.connect(self._show_point)
        self.fill_presets()
        self.set_channel(channel if channel in pc.CHANNELS else "RGB/K")

    # ------------------------------------------------------------ channels
    def current(self) -> dict:
        """The current channel's curve (created as the identity on first use)."""
        c = self.curves.get(self.channel)
        if c is None:
            c = pc.identity()
            self.curves[self.channel] = c
        return c

    def set_channel(self, ch: str):
        if ch != self.channel:
            self.channelChanged.emit(ch)
        self.channel = ch
        self.graph.channel = ch
        self.graph.curve = self.current()
        self.graph.sel = None
        for c, b in self.ch_buttons.items():
            b.setChecked(c == ch)
            edited = c in self.curves and not pc.is_identity(self.curves[c])
            b.setText(ch_label(c) + (" •" if edited else ""))
            col = CH_COL[c] if c in ("R", "G", "B") else None
            b.setStyleSheet(f"color: {col};" if col else "")
        self._busy = True
        self.interp.setCurrentIndex(max(0, pc.INTERPS.index(self.current().get("interp", pc.INTERPS[0]))))
        self._busy = False
        self._show_point()
        self.graph.update()

    def set_hist(self, hist: dict | None):
        self.graph.hist = hist
        self.graph.update()

    def _marks(self):
        for c, b in self.ch_buttons.items():
            edited = c in self.curves and not pc.is_identity(self.curves[c])
            b.setText(ch_label(c) + (" •" if edited else ""))

    # ------------------------------------------------------------ edits
    def _moved(self):
        self._marks()
        self._to_custom()
        self.changed.emit()

    def _released(self):
        self._marks()
        self._show_point()
        self.committed.emit()

    def _show_point(self):
        i = self.graph.sel
        on = i is not None and i < len(self.graph.pts())
        self._busy = True
        for sb in (self.inp, self.out):
            sb.setEnabled(on)
        if on:
            x, y = self.graph.pts()[i]
            self.inp.setValue(x)
            self.out.setValue(y)
        self._busy = False

    def _typed(self):
        i = self.graph.sel
        if self._busy or i is None:
            return
        x, y = self.graph.pts()[i]
        if abs(self.inp.value() - x) < 5e-4 and abs(self.out.value() - y) < 5e-4:
            return
        self.starting.emit()
        self.graph.set_point(i, self.inp.value(), self.out.value())
        self._to_custom()
        self._released()

    def _interp(self, i):
        if self._busy:
            return
        self.starting.emit()
        self.current()["interp"] = self.interp.itemData(i)
        self.graph.update()
        self._to_custom()
        self.committed.emit()

    def _log(self, on):
        self.graph.log = on
        self.graph.update()

    def reset_channel(self):
        self.starting.emit()
        self.curves[self.channel] = pc.identity()
        self.set_channel(self.channel)
        self._to_custom()
        self.committed.emit()

    def reset_all(self):
        self.starting.emit()
        self.curves.clear()
        self.set_channel(self.channel)
        self._to_custom()
        self.committed.emit()

    def set_curves(self, curves: dict):
        """Replace every curve (a preset, a file): the channels it does not set go back to the identity."""
        self.starting.emit()
        self.curves.clear()
        self.curves.update(copy.deepcopy(pc.clean_curves(curves)))
        first = next((ch for ch, _c in pc.active(self.curves)), self.channel)
        self.set_channel(first)

    # ------------------------------------------------------------ presets
    def fill_presets(self, select: str | None = None):
        self.preset.blockSignals(True)
        self.preset.clear()
        self.preset.addItem(_("Custom"), None)
        self._presets: dict[str, dict] = {}
        built = pc.builtin_presets()
        lin = built[0]
        self.preset.addItem(_(lin["name"]), lin["name"])
        self.preset.setItemData(self.preset.count() - 1, _(lin["source"]), Qt.ToolTipRole)
        self._presets[lin["name"]] = lin
        for head, src in GROUPS:
            self._header(_(head))
            for p in built[1:]:
                if (p["source"] == pc.PS) == (src == "classic"):
                    self.preset.addItem("   " + _(p["name"]), p["name"])
                    self.preset.setItemData(self.preset.count() - 1, _(p["source"]), Qt.ToolTipRole)
                    self._presets[p["name"]] = p
        mine = pc.user_presets(self.presets_path)
        if mine:
            self._header(_("My presets"))
            for p in mine:
                key = "user:" + p["name"]
                self.preset.addItem("   " + p["name"], key)
                self._presets[key] = p
        if select is None:                              # the preset these curves are, if any (else Custom)
            mine = pc.clean_curves(self.curves)
            select = next((k for k, p in self._presets.items()
                           if pc.clean_curves(p["curves"]) == mine and (mine or k == "Linear")), None)
        i = self.preset.findData(select) if select else 0
        self.preset.setCurrentIndex(max(0, i))
        self.preset.blockSignals(False)
        self._preset_buttons()

    def _header(self, text):
        self.preset.addItem(text, "")
        it = self.preset.model().item(self.preset.count() - 1)
        it.setEnabled(False)

    def _preset_buttons(self):
        key = self.preset.currentData()
        self.a_delete.setEnabled(isinstance(key, str) and key.startswith("user:"))

    def _to_custom(self):
        if self.preset.currentIndex() != 0:
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(0)
            self.preset.blockSignals(False)
            self._preset_buttons()

    def _preset_chosen(self, i):
        key = self.preset.itemData(i)
        self._preset_buttons()
        if not key:
            return
        self.apply_preset(key)

    def apply_preset(self, key: str):
        """A preset by its name (a built-in one) or "user:<name>": its curves replace the current ones."""
        p = self._presets.get(key)
        if p is None:
            raise KeyError(key)
        self.set_curves(p["curves"])
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(max(0, self.preset.findData(key)))
        self.preset.blockSignals(False)
        self._preset_buttons()
        self.committed.emit()

    def save_preset(self, name: str | None = None) -> str | None:
        if name is None:
            from .dialogs import ask_name
            name = ask_name(self.window(), _("Save preset"), _("My curves"))
        if not name or not name.strip():
            return None
        pc.save_user_preset(name, self.curves, self.presets_path)
        self.fill_presets(select="user:" + name.strip())
        return name.strip()

    def delete_preset(self):
        key = self.preset.currentData()
        if isinstance(key, str) and key.startswith("user:"):
            pc.delete_user_preset(key[5:], self.presets_path)
            self.fill_presets()

    def import_file(self, path: str | None = None) -> bool:
        if path is None:
            path, _f = QFileDialog.getOpenFileName(self.window(), _("Import a curves file"), "",
                                                   _("Curves (*.acv *.json)"))
        if not path:
            return False
        try:
            cv = pc.read_preset_file(path)
        except (OSError, ValueError) as e:
            self.hint.setText(_("This file cannot be read: {why}").format(why=e))
            return False
        self.set_curves(cv)
        self._to_custom()
        self.committed.emit()
        return True

    def export_file(self, path: str | None = None) -> bool:
        if path is None:
            path, _f = QFileDialog.getSaveFileName(self.window(), _("Export these curves"), "curves.acv",
                                                   _("Curves files (*.acv);;Curves of this program (*.json)"))
        if not path:
            return False
        try:
            pc.write_preset_file(path, self.curves)
        except (OSError, ValueError) as e:
            self.hint.setText(_("This file cannot be written: {why}").format(why=e))
            return False
        return True


class FunctionGraph(CurveGraph):
    """A stretch's transfer function over the histogram of its input (GHS, arcsinh, histogram transformation)."""

    def __init__(self, func, hist=None, parent=None):
        super().__init__(parent, readonly=True)
        self.func = func
        self.hist = hist
        self.setMinimumHeight(170)
        self.setMaximumHeight(170)


def caption(text: str) -> QLabel:
    lb = label(text, "small")
    lb.setWordWrap(True)
    return lb
