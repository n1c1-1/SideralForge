"""The Post of mockup alt3: the image at the centre; on the right the layers, as in Photoshop, and the properties of
the selected layer or of its mask.

A row of the layers panel: eye, thumbnail, mask thumbnail, name, blend mode and opacity. A click on the thumbnail
shows the layer's settings; a click on the mask thumbnail shows its mask, made of parts as in Lightroom, with a red
overlay on the image. Filters are layers: they act on everything below them, through their mask.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox,
                               QPushButton, QScrollArea, QSlider, QStackedWidget, QVBoxLayout, QWidget)

from i18n import _, language
from postengine import (DIRECTIONS, FAMILIES, FILTERS, MASK_SOURCES, MODES, OPS, Engine, new_filter_layer, new_id,
                        new_mask, new_part, new_workflow_layer, normalize)
from viewer import Viewer, to_qimage
from widgets import ACCENT, WARN, clear, hline, label

FAMILY_COL = {"Tone": "#c9a35e", "Color": "#c46fa8", "Detail": "#5ea1ff", "Stars": "#e6d36a",
              "Combine": "#e06c6c"}
TW, TH = 54, 36                                          # thumbnail size
COPY, STARLESS, STARS = " (copy)", "starless", "stars"   # pieces of the layer names, translated where shown


def tr_name(n: str) -> str:
    """A layer's name as shown: the pieces it was built from (a filter's name, "stars", "(copy)") translated; a name
    typed by the user is shown as typed."""
    k = 0
    while n.endswith(COPY):
        n, k = n[:-len(COPY)], k + 1
    return " · ".join(_(b) for b in n.split(" · ")) + _(COPY) * k


def dec(s: str) -> str:
    """A decimal comma in French."""
    return s.replace(".", ",") if language() == "fr" else s


def with_unit(v: str, unit: str) -> str:
    if not unit:
        return v
    return v + ("" if unit == "°" or unit == "%" and language() != "fr" else " ") + unit


def pix(a: np.ndarray, w=TW, h=TH) -> QPixmap:
    a = np.ascontiguousarray(a)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    im = Image.fromarray(a).resize((w, h), Image.BILINEAR)
    return QPixmap.fromImage(to_qimage(np.asarray(im)))


def fmt(v, step):
    if step is not None and step >= 1:
        return f"{v:.0f}"
    return dec(f"{v:.2f}".rstrip("0").rstrip("."))


# ======================================================================== a continuous slider
class ContSlider(QWidget):
    """A slider that glides: `moved` while dragging (draft render), `done` when let go (full render). Double-click
    on the name: back to the default."""
    moved = Signal(float)
    done = Signal(float)

    def __init__(self, name, lo, hi, value, default, ends=None, unit="", step=None, show=None, parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.default, self.unit, self.step, self.show = lo, hi, default, unit, step, show
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 2, 0, 4)
        v.setSpacing(0)
        top = QHBoxLayout()
        self.name = QLabel(name)
        self.name.setToolTip(_("Double-click: default value"))
        self.val = label("", "small")
        top.addWidget(self.name)
        top.addStretch(1)
        top.addWidget(self.val)
        v.addLayout(top)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.s = QSlider(Qt.Horizontal)
        self.s.setRange(0, 1000)
        if ends:
            a, b = label(ends[0], "small"), label(ends[1], "small")
            row.addWidget(a)
            row.addWidget(self.s, 1)
            row.addWidget(b)
        else:
            row.addWidget(self.s, 1)
        v.addLayout(row)
        self.why = label("", "warn")
        self.why.setWordWrap(True)
        self.why.hide()
        v.addWidget(self.why)
        self.set_value(value)
        self.s.valueChanged.connect(self._changed)
        self.s.sliderReleased.connect(lambda: self.done.emit(self.value()))
        self.name.mouseDoubleClickEvent = lambda e: (self.set_value(self.default), self.done.emit(self.default))

    def value(self) -> float:
        x = self.lo + (self.hi - self.lo) * self.s.value() / 1000
        if self.step:
            x = round(x / self.step) * self.step
        return x

    def set_value(self, x):
        self.s.blockSignals(True)
        self.s.setValue(int(round(1000 * (x - self.lo) / ((self.hi - self.lo) or 1))))
        self.s.blockSignals(False)
        self._label()

    def _label(self):
        x = self.value()
        if self.show:
            self.val.setText(self.show(x))
        else:
            self.val.setText(with_unit(fmt(x, self.step), self.unit))
        changed = abs(x - self.default) > 1e-6 * max(1, abs(self.hi - self.lo))
        self.val.setStyleSheet(f"color: {WARN};" if changed else "")

    def _changed(self, _):
        self._label()
        self.moved.emit(self.value())
        if not self.s.isSliderDown():                       # keyboard or click on the groove
            self.done.emit(self.value())

    def set_available(self, ok, why=""):
        self.s.setEnabled(ok)
        self.name.setStyleSheet("" if ok else "color: #6b7280;")
        self.why.setText(why)
        self.why.setVisible(not ok and bool(why))


# ======================================================================== the layers panel
class LayerRow(QFrame):
    def __init__(self, panel, lay, thumb, mthumb, sel_layer, sel_mask, parent=None):
        super().__init__(parent)
        self.panel, self.lay = panel, lay
        self._press = None
        self.setObjectName("lrow")
        on = sel_layer or sel_mask
        self.setStyleSheet(f"QFrame#lrow {{ background: {'#2b3446' if on else 'transparent'}; border-radius: 4px; }}"
                           "QFrame#lrow:hover { background: #262b35; } QLabel { background: transparent; }")
        h = QHBoxLayout(self)
        h.setContentsMargins(4, 3, 6, 3)
        h.setSpacing(6)
        eye = QCheckBox()
        eye.setChecked(lay["visible"])
        eye.setToolTip(_("Show / hide layer"))
        eye.toggled.connect(lambda v: panel.set_visible(lay["id"], v))
        h.addWidget(eye)
        if lay["kind"] == "filter" and lay.get("target"):
            arrow = QLabel("↳")
            arrow.setStyleSheet(f"color: {ACCENT}; font-size: 13pt;")
            arrow.setFixedWidth(20)
            arrow.setToolTip(_("Applies only to the layer just below"))
            h.addSpacing(6)
            h.addWidget(arrow)
        self.t = QLabel()
        self.t.setPixmap(thumb)
        self.t.setFixedSize(TW + 4, TH + 4)
        self.t.setStyleSheet(f"border: 2px solid {ACCENT if sel_layer else 'transparent'};")
        self.t.setToolTip(_("Layer settings"))
        h.addWidget(self.t)
        self.m = None
        if mthumb is not None:
            self.m = QLabel()
            self.m.setPixmap(mthumb)
            self.m.setFixedSize(TW + 4, TH + 4)
            self.m.setStyleSheet(f"border: 2px solid {ACCENT if sel_mask else '#3a4150'};")
            self.m.setToolTip(_("Layer mask: white shows the layer, black hides it"))
            h.addWidget(self.m)
        v = QVBoxLayout()
        v.setSpacing(0)
        n = QLabel(tr_name(lay["name"]))
        n.setStyleSheet("font-weight: 600;" + ("" if lay["visible"] else " color: #8b93a1;"))
        v.addWidget(n)
        v.addWidget(label(panel.sub_text(lay), "small"))
        h.addLayout(v, 1)

    def _target(self, pos):
        return "mask" if self.m is not None and self.m.geometry().contains(pos) else "layer"

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = e.position().toPoint()
            self._dragging = False
        elif e.button() == Qt.RightButton:
            self.panel.layer_menu(self.lay, e.globalPosition().toPoint())

    def mouseMoveEvent(self, e):
        if self._press is not None and (e.position().toPoint() - self._press).manhattanLength() > 8:
            self._dragging = True
            self.panel.drag_move(self, e.globalPosition().toPoint())

    def mouseReleaseEvent(self, e):
        if self._press is None:
            return
        press, self._press = self._press, None
        if self._dragging:
            self.panel.drag_end(self, e.globalPosition().toPoint())
        else:                                            # a click: select (after the event, the rows are rebuilt)
            what = self._target(press)
            QTimer.singleShot(0, lambda: self.panel.select(self.lay["id"], what))

    def mouseDoubleClickEvent(self, e):
        self.panel.rename(self.lay)


class PostSource:
    """What the viewer shows: the composite (half resolution; a quarter, enlarged, while something moves), the
    region at full resolution when zoomed in, or the mask in grey."""

    def __init__(self, view):
        self.view = view
        self.version = 0
        self.img1 = None
        self.draft = False

    def size(self):
        return tuple(self.view.s.data["size"])

    def region(self, level, x0, y0, x1, y1):
        v = self.view
        if v.show_mask and v.mask1 is not None:
            m = (np.clip(v.mask1, 0, 1) * 255).astype(np.uint8)
            if level == 0:
                m = np.repeat(np.repeat(m, 2, 0), 2, 1)
            return np.repeat(m[y0:y1, x0:x1, None], 3, -1)
        if level == 1 or self.draft:
            if self.img1 is None:
                return np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
            if level == 1:
                return self.img1[y0:y1, x0:x1]
            a = self.img1[y0 // 2: -(-y1 // 2), x0 // 2: -(-x1 // 2)]
            a = np.repeat(np.repeat(a, 2, 0), 2, 1)
            return a[y0 % 2: y0 % 2 + (y1 - y0), x0 % 2: x0 % 2 + (x1 - x0)]
        return (v.eng.compose(0, x0, y0, x1, y1) * 255).astype(np.uint8)


class BeforeSource:
    """The before of the before/after: the bottom workflow layer, all sliders at the suggestion."""

    def __init__(self, view):
        self.view = view
        self.version = 0

    def size(self):
        return tuple(self.view.s.data["size"])

    def region(self, level, x0, y0, x1, y1):
        eng = self.view.eng
        base = next((l for l in reversed(eng.layers) if l["kind"] == "workflow" and eng.path(l["pid"])), None)
        if base is None:
            return np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        info = eng.s.layers[eng.path(base["pid"]).layer]
        return eng.store.get(info["base"], level)[y0:y1, x0:x1]


# ======================================================================== the view
class PostView(QWidget):
    goProcessing = Signal(int, str)
    goResults = Signal(int)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.s = session
        self.eng = Engine(session)
        self.sel = None                                 # (layer id, "layer" | "mask")
        self.part_sel = None                            # the mask part being edited (its id)
        self.show_mask = False
        self.mask1 = None
        self.undo, self.redo = [], []
        self._mthumbs = {}
        self._stroke_before = None
        self._last_pt = None
        self.sync_paths(initial=True)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(10, 6, 10, 6)
        self.b_fit, self.b_100 = QPushButton(_("Fit")), QPushButton(with_unit("100", "%"))
        self.zoom_lab = label("", "dim")
        self.b_cmp = QPushButton(_("Before / after"))
        self.b_cmp.setCheckable(True)
        self.b_cmp.setObjectName("toggle")
        self.b_cmp.setToolTip(_("Before: the bottom layer, suggested settings (C)"))
        self.b_mask = QPushButton(_("Show mask"))
        self.b_mask.setCheckable(True)
        self.b_mask.setObjectName("toggle")
        self.b_mask.setToolTip(_("Mask of the selected layer, in gray (M)"))
        self.b_undo, self.b_redo = QPushButton("↶"), QPushButton("↷")
        self.b_undo.setToolTip(_("Undo (Ctrl+Z)"))
        self.b_redo.setToolTip(_("Redo (Ctrl+Y)"))
        self.b_export = QPushButton(_("Export…"))
        for w in (self.b_fit, self.b_100, self.zoom_lab):
            bar.addWidget(w)
        bar.addStretch(1)
        for w in (self.b_cmp, self.b_mask, self.b_undo, self.b_redo, self.b_export):
            bar.addWidget(w)
        mv.addLayout(bar)
        self.src = PostSource(self)
        self.before = BeforeSource(self)
        self.viewer = Viewer()
        self.viewer.set_source(self.src)
        self.viewer.overlays.append(self._overlay)
        mv.addWidget(self.viewer, 1)
        self.hint = label("", "small")
        self.hint.setContentsMargins(10, 3, 10, 5)
        mv.addWidget(self.hint)
        root.addWidget(mid, 1)
        # ---- right: layers, then the properties of the selection
        right = QWidget()
        right.setObjectName("panel")
        right.setFixedWidth(400)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(10, 10, 10, 8)
        rv.setSpacing(6)
        rv.addWidget(label(_("Layers"), "h2"))
        self.lsc = QScrollArea()
        self.lsc.setWidgetResizable(True)
        self.lsc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lh = QWidget()
        self.lbox = QVBoxLayout(lh)
        self.lbox.setContentsMargins(0, 0, 4, 0)
        self.lbox.setSpacing(2)
        self.lsc.setWidget(lh)
        self.lsc.setMaximumHeight(370)
        rv.addWidget(self.lsc)
        self.drop_line = QFrame(lh)
        self.drop_line.setFixedHeight(2)
        self.drop_line.setStyleSheet(f"background: {ACCENT};")
        self.drop_line.hide()
        btns = QHBoxLayout()
        btns.setSpacing(4)
        self.b_filter = QPushButton(_("+ Filter"))
        self.b_filter.setToolTip(_("A filter layer above the selected layer"))
        fm = QMenu(self.b_filter)
        for fam in FAMILIES:
            sub = fm.addMenu(_(fam))
            for k, f in FILTERS.items():
                if f["family"] == fam:
                    sub.addAction(_(f["name"])).triggered.connect(lambda _c=False, kk=k: self.add_filter(kk))
        self.b_filter.setMenu(fm)
        self.b_dup = QPushButton(_("Duplicate"))
        self.b_dup.setToolTip("Ctrl+J")
        self.b_maskadd = QPushButton(_("Mask"))
        self.b_del = QPushButton(_("Delete"))
        for b in (self.b_filter, self.b_dup, self.b_maskadd, self.b_del):
            btns.addWidget(b)
        rv.addLayout(btns)
        rv.addWidget(hline())
        psc = QScrollArea()
        psc.setWidgetResizable(True)
        psc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ph = QWidget()
        self.pbox = QVBoxLayout(ph)
        self.pbox.setContentsMargins(0, 0, 14, 0)
        self.pbox.setSpacing(4)
        psc.setWidget(ph)
        rv.addWidget(psc, 1)
        root.addWidget(right)

        self.b_fit.clicked.connect(self.viewer.fit)
        self.b_100.clicked.connect(lambda: self.viewer.zoom_to(1.0))
        self.b_cmp.toggled.connect(self._compare)
        self.b_mask.toggled.connect(self._mask_view)
        self.b_undo.clicked.connect(self.do_undo)
        self.b_redo.clicked.connect(self.do_redo)
        self.b_export.clicked.connect(lambda: QMessageBox.information(
            self, _("Export"), _("Mockup: the real client exports the composite as 16-bit TIFF or JPEG, at full "
                                 "resolution.")))
        self.b_dup.clicked.connect(self.duplicate)
        self.b_maskadd.clicked.connect(self.mask_button)
        self.b_del.clicked.connect(self.delete)
        self.viewer.viewChanged.connect(lambda: self.zoom_lab.setText(
            _("Zoom {p}").format(p=with_unit(f"{self.viewer.scale * 100:.0f}", "%"))))
        self.viewer.stroke.connect(self._stroke)
        self.viewer.strokeEnd.connect(self._stroke_end)
        for seq, fn in (("Ctrl+Z", self.do_undo), ("Ctrl+Y", self.do_redo), ("C", self.b_cmp.toggle),
                        ("M", self.b_mask.toggle), ("F", self.viewer.fit), ("1", lambda: self.viewer.zoom_to(1.0)),
                        ("X", self.toggle_paint), ("[", lambda: self.brush_size(1 / 1.25)),
                        ("]", lambda: self.brush_size(1.25)), ("Ctrl+J", self.duplicate)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        self._draft_t = QTimer(self)
        self._draft_t.setSingleShot(True)
        self._draft_t.timeout.connect(self._render_draft)
        self._final_t = QTimer(self)
        self._final_t.setSingleShot(True)
        self._final_t.timeout.connect(self._render_final)
        self.s.pathsChanged.connect(self.sync_paths)
        if self.eng.layers:
            self.sel = (self.eng.layers[-1]["id"], "layer")
        self.rebuild()
        self._render_final()

    # ------------------------------------------------------------ layers from the workflows
    def sync_paths(self, initial=False):
        eng = self.eng
        known = {l["pid"] for l in eng.layers if l["kind"] == "workflow"}
        for p in self.s.paths:
            if p.rendered and p.pid not in known:
                eng.layers.insert(0, new_workflow_layer(p.pid, p.name))
        alive = {p.pid for p in self.s.paths}
        eng.layers = [l for l in eng.layers if l["kind"] != "workflow" or l["pid"] in alive]
        wl =[l for l in eng.layers if l["kind"] == "workflow"]
        if initial and wl:
            base = next((l for l in reversed(wl) if eng.path(l["pid"]).layer == "complet"), wl[-1])
            eng.layers.remove(base)
            eng.layers.append(base)
            base["visible"] = True
        elif wl and not any(l["visible"] for l in wl):
            wl[-1]["visible"] = True
        if not initial:
            if self.sel and not self.layer(self.sel[0]):
                self.sel = (eng.layers[-1]["id"], "layer") if eng.layers else None
            self.rebuild()
            self.request(final_only=True)

    def layer(self, lid):
        return next((l for l in self.eng.layers if l["id"] == lid), None)

    def index(self, lid):
        return next(i for i, l in enumerate(self.eng.layers) if l["id"] == lid)

    # ------------------------------------------------------------ rendering
    def request(self, final_only=False):
        if not final_only:
            self._draft_t.start(0)
        self._final_t.start(0 if final_only else 220)

    def _render(self, level):
        want = self.sel[0] if self.sel else -1
        img, m = self.eng.compose(level, want_mask=want)
        if level == 2:
            W, H = self.eng.size(1)
            img = np.repeat(np.repeat(img, 2, 0), 2, 1)[:H, :W]
            if m is not None:
                m = np.repeat(np.repeat(m, 2, 0), 2, 1)[:H, :W]
        self.src.img1 = (img * 255).astype(np.uint8)
        self.mask1 = m
        self._overlay_img = None
        self.src.version += 1
        self.viewer.refresh()

    def _render_draft(self):
        self.src.draft = True
        self._render(2)

    def _render_final(self):
        self._draft_t.stop()
        self.src.draft = False
        self._render(1)
        if self.sel and self.mask1 is not None:
            self._mthumbs[self.sel[0]] = pix((np.clip(self.mask1, 0, 1) * 255).astype(np.uint8))
            self._update_mask_thumb()

    def refresh(self):
        self.request(final_only=True)

    # ------------------------------------------------------------ the red overlay of the mask being edited
    def _overlay(self, p: QPainter, v: Viewer):
        if not self.sel or self.sel[1] != "mask" or self.mask1 is None or self.show_mask or self.b_cmp.isChecked():
            return
        if getattr(self, "_overlay_img", None) is None:
            m = np.clip(self.mask1, 0, 1)
            h, w = m.shape
            a = np.zeros((h, w, 4), np.uint8)
            a[..., 0], a[..., 1], a[..., 2] = 255, 70, 60
            a[..., 3] = (m * 120).astype(np.uint8)
            a = np.ascontiguousarray(a)
            self._overlay_img = QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        W, H = self.s.data["size"]
        p.drawImage(QRectF(v.img_to_w(0, 0), v.img_to_w(W, H)), self._overlay_img)

    # ------------------------------------------------------------ undo
    def push(self, before=None):
        self.undo.append(before if before is not None else self.eng.snapshot())
        self.redo.clear()
        self._undo_buttons()

    def _undo_buttons(self):
        self.b_undo.setEnabled(bool(self.undo))
        self.b_redo.setEnabled(bool(self.redo))

    def do_undo(self):
        if self.undo:
            self.redo.append(self.eng.snapshot())
            self.eng.layers = self.undo.pop()
            self._restored()

    def do_redo(self):
        if self.redo:
            self.undo.append(self.eng.snapshot())
            self.eng.layers = self.redo.pop()
            self._restored()

    def _restored(self):
        self.eng._layer_cache.clear()
        self.eng._brush.clear()
        if self.sel and not self.layer(self.sel[0]):
            self.sel = (self.eng.layers[-1]["id"], "layer") if self.eng.layers else None
        self.rebuild()
        self.request(final_only=True)

    # ------------------------------------------------------------ the layers panel
    def thumb(self, lay) -> QPixmap:
        if lay["kind"] == "filter":
            f = FILTERS[lay["filter"]]
            pm = QPixmap(TW, TH)
            pm.fill(QColor("#1d2129"))
            qp = QPainter(pm)
            qp.setPen(QColor(FAMILY_COL[f["family"]]))
            fo = QFont("Segoe UI", 15)
            fo.setItalic(True)
            fo.setBold(True)
            qp.setFont(fo)
            qp.drawText(pm.rect(), Qt.AlignCenter, "ƒ")
            qp.end()
            return pm
        p = self.eng.path(lay["pid"])
        info = self.s.layers[p.layer]
        rel = info["base"]
        if lay["part"] == "fond" and "nombre" in info["sliders"]:
            rel = info["sliders"]["nombre"][0]
        return pix(self.eng.store.get(rel, 2))

    def mask_thumb(self, lay):
        if lay["mask"] is None:
            return None
        pm = self._mthumbs.get(lay["id"])
        if pm is None:
            _, m = self.eng.compose(2, want_mask=lay["id"])
            m = np.ones((TH, TW), np.float32) if m is None else m
            pm = pix((np.clip(m, 0, 1) * 255).astype(np.uint8))
            self._mthumbs[lay["id"]] = pm
        return pm

    def _update_mask_thumb(self):
        for row in self.rows:
            if row.m is not None and row.lay["id"] in self._mthumbs:
                row.m.setPixmap(self._mthumbs[row.lay["id"]])

    def rebuild(self):
        clear(self.lbox)
        self.rows = []
        for lay in self.eng.layers:
            sl = self.sel is not None and self.sel[0] == lay["id"]
            row = LayerRow(self, lay, self.thumb(lay), self.mask_thumb(lay), sl and self.sel[1] == "layer",
                           sl and self.sel[1] == "mask")
            self.lbox.addWidget(row)
            self.rows.append(row)
        waiting = [p for p in self.s.paths if not p.rendered]
        for p in waiting:
            b = QPushButton(_('"{name}": not computed yet  →').format(name=tr_name(p.name)))
            b.setObjectName("link")
            b.clicked.connect(lambda _=False, pid=p.pid: self.goResults.emit(pid))
            self.lbox.addWidget(b)
        self.lbox.addStretch(1)
        self.lsc.setFixedHeight(min(370, 48 * (len(self.rows) + len(waiting) * 0.6) + 10))
        sel_row = next((r for r in self.rows if self.sel and r.lay["id"] == self.sel[0]), None)
        if sel_row is not None:
            QTimer.singleShot(0, lambda: self.lsc.ensureWidgetVisible(sel_row, 0, 10))
        lay = self.layer(self.sel[0]) if self.sel else None
        self.b_dup.setEnabled(lay is not None)
        self.b_del.setEnabled(lay is not None and len(self.eng.layers) > 1)
        self.b_maskadd.setEnabled(lay is not None)
        self.b_maskadd.setText(_("Mask") if lay is None or lay["mask"] is None else _("Mask") + " ✓")
        self.b_maskadd.setToolTip(_("Add a mask to the selected layer") if lay is None or lay["mask"] is None else
                                  _("Edit the mask of the selected layer"))
        self._undo_buttons()
        self.build_props()

    def select(self, lid, what):
        if self.sel == (lid, what):
            return
        self.sel = (lid, what)
        lay = self.layer(lid)
        if what == "mask" and lay["mask"] is not None:
            brush = next((p for p in lay["mask"]["parts"] if p["src"] == "pinceau"), None)
            self.part_sel = brush["id"] if brush else None
        else:
            self.part_sel = None
        self.rebuild()
        self.request(final_only=True)

    def set_visible(self, lid, v):
        self.push()
        self.layer(lid)["visible"] = v
        self.rebuild()
        self.request()

    def drag_move(self, row, gpos):
        y = self.lbox.parentWidget().mapFromGlobal(gpos).y()
        i = self._drop_index(y)
        ref = self.rows[min(i, len(self.rows) - 1)]
        yy = ref.geometry().top() - 2 if i < len(self.rows) else ref.geometry().bottom() + 1
        self.drop_line.setGeometry(0, yy, self.lbox.parentWidget().width(), 2)
        self.drop_line.show()
        self.drop_line.raise_()

    def _drop_index(self, y):
        for i, r in enumerate(self.rows):
            if y < r.geometry().center().y():
                return i
        return len(self.rows)

    def drag_end(self, row, gpos):
        if not self.drop_line.isVisible():
            return
        self.drop_line.hide()
        y = self.lbox.parentWidget().mapFromGlobal(gpos).y()
        i = self._drop_index(y)
        j = self.index(row.lay["id"])
        if i in (j, j + 1):
            return
        self.push()
        L = self.eng.layers
        lay = L.pop(j)
        k = i - 1 if i > j else i
        L.insert(k, lay)
        if lay["kind"] == "filter":                     # dropped on a layer's group: its filter; elsewhere: on all
            below = L[k + 1] if k + 1 < len(L) else None
            if below is None:
                lay["target"] = None
            elif below["kind"] == "workflow":
                lay["target"] = below["id"]
            else:
                lay["target"] = below.get("target")
        self.eng.layers = normalize(L)
        self.rebuild()
        self.request(final_only=True)

    def layer_menu(self, lay, gpos):
        m = QMenu(self)
        m.addAction(_("Duplicate")).triggered.connect(lambda: (self.select(lay["id"], "layer"), self.duplicate()))
        if lay["mask"] is None:
            m.addAction(_("Add mask")).triggered.connect(lambda: self.add_mask(lay))
        else:
            m.addAction(_("Delete mask")).triggered.connect(lambda: self.remove_mask(lay))
        if lay["kind"] == "workflow" and lay["part"] is None:
            p = self.eng.path(lay["pid"])
            if self.s.slider_state(p, "nombre")[0] and "nombre" in self.s.layers[p.layer]["sliders"]:
                m.addAction(_("Split stars and starless")).triggered.connect(lambda: self.split_stars(lay))
            m.addSeparator()
            m.addAction(_("Show workflow  →")).triggered.connect(lambda: self.goResults.emit(lay["pid"]))
        m.addAction(_("Rename")).triggered.connect(lambda: self.rename(lay))
        if len(self.eng.layers) > 1:
            m.addSeparator()
            m.addAction(_("Delete")).triggered.connect(lambda: (self.select(lay["id"], "layer"), self.delete()))
        m.exec(gpos)

    def rename(self, lay):
        from dialogs import ask_name
        old = tr_name(lay["name"])
        t = ask_name(self.window(), _("Rename layer"), old)
        if t and t != old:
            self.push()
            lay["name"] = t
            self.rebuild()

    # ------------------------------------------------------------ layer actions
    def _insert_above_sel(self, lay):
        i = self.index(self.sel[0]) if self.sel else 0
        self.eng.layers.insert(i, lay)
        self.eng.layers = normalize(self.eng.layers)
        self.sel = (lay["id"], "layer")
        self.part_sel = None

    def add_filter(self, key):
        """A new filter applies to the chosen layer (or to what the chosen filter applies to)."""
        cur = self.layer(self.sel[0]) if self.sel else None
        target = None
        if cur is not None:
            target = cur["id"] if cur["kind"] == "workflow" else cur.get("target")
        self.push()
        fl = new_filter_layer(key, target)
        if target is not None:                          # on top of that layer's own filters
            group = [l for l in self.eng.layers if l.get("target") == target]
            self.sel = ((group[0] if group else self.layer(target))["id"], "layer")
        self._insert_above_sel(fl)
        self.rebuild()
        self.request(final_only=True)

    def set_target(self, fl, target):
        self.push()
        old = fl.get("target")
        L = self.eng.layers
        L.remove(fl)
        fl["target"] = target
        if target is None:                              # leaves the group: just above the layer it was on
            top = next((i for i, l in enumerate(L) if l.get("target") == old or l["id"] == old), 0)
            L.insert(top, fl)
        else:
            L.insert(self.index(target), fl)
        self.eng.layers = normalize(L)
        self.rebuild()
        self.request(final_only=True)

    def _copy(self, lay):
        d = copy.deepcopy(lay)
        d["id"] = new_id()
        if d["mask"]:
            for part in d["mask"]["parts"]:
                part["id"] = new_id()
        return d

    def duplicate(self):
        """A copy above; a workflow layer comes with its own filters."""
        lay = self.layer(self.sel[0]) if self.sel else None
        if lay is None:
            return
        self.push()
        d = self._copy(lay)
        d["name"] = lay["name"] + COPY
        d["visible"] = True
        top = self.index(lay["id"])
        if lay["kind"] == "workflow":
            own = [l for l in self.eng.layers if l.get("target") == lay["id"]]
            copies = []
            for f in own:
                c = self._copy(f)
                c["target"] = d["id"]
                copies.append(c)
            if own:
                top = self.index(own[0]["id"])
            self.eng.layers[top:top] = copies + [d]
        else:
            self.eng.layers.insert(top, d)
        self.eng.layers = normalize(self.eng.layers)
        self.sel = (d["id"], "layer")
        self.part_sel = None
        self.rebuild()
        self.request(final_only=True)

    def delete(self):
        """A workflow layer goes with its own filters."""
        lay = self.layer(self.sel[0]) if self.sel else None
        others = [l for l in self.eng.layers if l["id"] != lay["id"] and l.get("target") != lay["id"]] if lay else []
        if lay is None or not any(l["kind"] == "workflow" for l in others) and lay["kind"] == "workflow":
            return
        self.push()
        i = self.index(lay["id"])
        self.eng.layers = normalize(others)
        L = self.eng.layers
        self.sel = (L[min(i, len(L) - 1)]["id"], "layer") if L else None
        self.rebuild()
        self.request(final_only=True)

    def mask_button(self):
        lay = self.layer(self.sel[0]) if self.sel else None
        if lay is None:
            return
        if lay["mask"] is None:
            self.add_mask(lay)
        else:
            self.select(lay["id"], "mask")

    def add_mask(self, lay):
        self.push()
        lay["mask"] = new_mask()
        self._mthumbs.pop(lay["id"], None)
        self.sel = None
        self.select(lay["id"], "mask")

    def remove_mask(self, lay):
        self.push()
        lay["mask"] = None
        self._mthumbs.pop(lay["id"], None)
        self.sel = (lay["id"], "layer")
        self.rebuild()
        self.request(final_only=True)

    def split_stars(self, lay):
        """The layer becomes two: its background without stars, and its stars alone, in Screen mode, above."""
        self.push()
        i = self.index(lay["id"])
        fond = dict(copy.deepcopy(lay), id=new_id(), part="fond", name=f"{lay['name']} · {STARLESS}")
        stars = dict(copy.deepcopy(lay), id=new_id(), part="etoiles", name=f"{lay['name']} · {STARS}", mode="Screen",
                     mask=None)
        self.eng.layers[i:i + 1] = [stars, fond]
        for f in self.eng.layers:                       # its own filters stay on the background
            if f.get("target") == lay["id"]:
                f["target"] = fond["id"]
        self.eng.layers = normalize(self.eng.layers)
        self.sel = (stars["id"], "layer")
        self.rebuild()
        self.request(final_only=True)

    # ------------------------------------------------------------ properties of the selection
    def build_props(self):
        clear(self.pbox)
        self.viewer.mode = "pan"
        lay = self.layer(self.sel[0]) if self.sel else None
        if lay is None:
            self.pbox.addStretch(1)
            return
        if self.sel[1] == "mask" and lay["mask"] is not None:
            self.mask_props(lay)
        else:
            self.layer_props(lay)
        self.pbox.addStretch(1)
        self._hint()

    def _group(self, text):
        g = label(text, "dim")
        g.setStyleSheet("color: #aab2bf; font-weight: 600; margin-top: 8px;")
        self.pbox.addWidget(g)

    def layer_props(self, lay):
        self.pbox.addWidget(label(tr_name(lay["name"]), "h2"))
        row = QHBoxLayout()
        row.addWidget(label(_("Blend mode"), "dim"))
        cb = QComboBox()
        for md in MODES:
            cb.addItem(_(md), md)
        cb.setCurrentIndex(MODES.index(lay["mode"]))
        cb.currentIndexChanged.connect(lambda i: self._set(lay, "mode", cb.itemData(i)))
        row.addWidget(cb, 1)
        self.pbox.addLayout(row)
        op = ContSlider(_("Opacity"), 0, 100, lay["opacity"], 100, unit="%", step=1)
        self._wire(op, lambda v: lay.__setitem__("opacity", v))
        self.pbox.addWidget(op)
        if lay["kind"] == "filter":
            f = FILTERS[lay["filter"]]
            self._group(_("Applies to"))
            i = self.index(lay["id"])
            below = next((l for l in self.eng.layers[i + 1:] if l["kind"] == "workflow"), None)
            tgt = self.layer(lay["target"]) if lay.get("target") else below
            r = QVBoxLayout()
            r.setSpacing(3)
            only = _('"{name}" only').format(name=tr_name(tgt["name"])) if tgt else None
            for txt, val in ((only, tgt["id"] if tgt else None), (_("Everything below"), None)):
                if txt is None:
                    continue
                b = QPushButton(txt)
                b.setObjectName("scope")
                b.setCheckable(True)
                b.setChecked(lay.get("target") == val)
                b.setStyleSheet("QPushButton#scope { padding: 4px 10px; text-align: left; }"
                                f"QPushButton#scope:checked {{ background: #2b3446; border-color: {ACCENT}; "
                                "color: #ffffff; font-weight: 600; }")
                b.clicked.connect(lambda _=False, t=val: self.set_target(lay, t) if lay.get("target") != t else None)
                r.addWidget(b)
            self.pbox.addLayout(r)
            self._group(_(f["name"]))
            for q in f["params"]:
                if q["type"] == "choice":
                    r = QHBoxLayout()
                    r.addWidget(QLabel(_(q["label"])))
                    c = QComboBox()
                    for ch in q["choices"]:
                        c.addItem(_(ch), ch)
                    c.setCurrentIndex(q["choices"].index(lay["params"][q["key"]]))
                    c.currentIndexChanged.connect(lambda i, k=q["key"], c=c: self._set_param(lay, k, c.itemData(i)))
                    r.addWidget(c, 1)
                    self.pbox.addLayout(r)
                else:
                    sl = ContSlider(_(q["label"]), q["lo"], q["hi"], lay["params"][q["key"]], q["default"],
                                    unit=q["unit"], step=q["step"])
                    self._wire(sl, lambda v, k=q["key"]: lay["params"].__setitem__(k, v))
                    self.pbox.addWidget(sl)
            return
        p = self.eng.path(lay["pid"])
        if lay["part"]:
            self.pbox.addWidget(label(_("Stars only, in Screen mode above the starless image.")
                                      if lay["part"] == "etoiles" else _("Starless image."), "small"))
        info = self.s.layers[p.layer]
        group = None
        for sd in self.s.data["sliders"]:
            if sd["group"] != group:
                group = sd["group"]
                self._group(_(group))
            pos = lay["sliders"].get(sd["key"], 2.0)
            sl = ContSlider(_(sd["name"]), 0, 4, pos, 2, ends=[_(e) for e in sd["ends"]] if sd["ends"] else None,
                            show=lambda x: _("suggested") if abs(x - 2) < 0.02 else dec(f"{x - 2:+.1f}"))
            ok, why, need = self.s.slider_state(p, sd["key"])
            if ok and sd["key"] not in info["sliders"]:
                ok, why = False, "not prepared in the mockup for this layer"
            sl.set_available(ok, _(why) if why else why)
            self._wire(sl, lambda v, k=sd["key"]: self._slider(lay, k, v))
            self.pbox.addWidget(sl)

    def _slider(self, lay, k, v):
        if abs(v - 2) < 0.02:
            lay["sliders"].pop(k, None)
        else:
            lay["sliders"][k] = v

    def _wire(self, sl: ContSlider, apply):
        """A slider: draft render while it moves, full render and one undo step when it is let go."""
        state = {"before": None}

        def moved(v):
            if state["before"] is None:
                state["before"] = self.eng.snapshot()
            apply(v)
            self.request()

        def done(v):
            if state["before"] is None:
                state["before"] = self.eng.snapshot()
            apply(v)
            self.push(state["before"])
            state["before"] = None
            self.request(final_only=True)
            QTimer.singleShot(0, self._refresh_rows)
        sl.moved.connect(moved)
        sl.done.connect(done)

    def sub_text(self, lay) -> str:
        """Under a layer's name: what a filter applies to, then mode and opacity."""
        bits = []
        if lay["kind"] == "filter":
            t = self.layer(lay["target"]) if lay.get("target") else None
            bits.append(_('on "{name}"').format(name=tr_name(t["name"])) if t else _("on everything below"))
        bits.append(f"{_(lay['mode'])} · {with_unit(format(lay['opacity'], '.0f'), '%')}")
        if lay["kind"] == "workflow" and self.eng.path(lay["pid"]) is not None and \
                self.eng.path(lay["pid"]).provisional:
            bits.append(_("preview"))
        return " · ".join(bits)

    def _refresh_rows(self):
        """The row texts (mode, opacity) without rebuilding the properties under the mouse."""
        for row in self.rows:
            lab = row.findChildren(QLabel)
            if lab:
                lab[-1].setText(self.sub_text(row.lay))

    def _set(self, lay, key, value):
        self.push()
        lay[key] = value
        self._refresh_rows()
        self.request(final_only=True)

    def _set_param(self, lay, key, value):
        self.push()
        lay["params"][key] = value
        self.request(final_only=True)

    # ------------------------------------------------------------ the mask
    def mask_props(self, lay):
        m = lay["mask"]
        top = QHBoxLayout()
        top.addWidget(label(_('Mask of "{name}"').format(name=tr_name(lay["name"])), "h2"), 1)
        rm = QPushButton(_("Delete"))
        rm.clicked.connect(lambda: self.remove_mask(lay))
        top.addWidget(rm)
        self.pbox.addLayout(top)
        if not m["parts"]:
            self.pbox.addWidget(label(_("Empty: the layer shows everywhere."), "small"))
        for j, part in enumerate(m["parts"]):
            self.pbox.addWidget(self.part_box(lay, part, j))
        add = QPushButton(_("+ Add to mask"))
        am = QMenu(add)
        for src, name in MASK_SOURCES.items():
            am.addAction(_(name)).triggered.connect(lambda _c=False, s=src: self.add_part(lay, s))
        add.setMenu(am)
        self.pbox.addWidget(add, 0, Qt.AlignLeft)
        self._group(_("Whole mask"))
        fe = ContSlider(_("Feather"), 0, 60, m["feather"], 0, unit="px", step=1)
        self._wire(fe, lambda v: m.__setitem__("feather", v))
        self.pbox.addWidget(fe)
        de = ContSlider(_("Density"), 0, 100, m["density"], 100, unit="%", step=1)
        self._wire(de, lambda v: m.__setitem__("density", v))
        self.pbox.addWidget(de)
        inv = QCheckBox(_("Invert"))
        inv.setChecked(m["invert"])
        inv.toggled.connect(lambda v: (self.push(), m.__setitem__("invert", v), self.request(final_only=True)))
        self.pbox.addWidget(inv)
        if self.part_sel and any(p["id"] == self.part_sel and p["src"] == "pinceau" for p in m["parts"]):
            self.viewer.mode = "brush"
            part = next(p for p in m["parts"] if p["id"] == self.part_sel)
            self.viewer.brush_r = part["size"] / 2

    def part_box(self, lay, part, j):
        on = part["id"] == self.part_sel
        f = QFrame()
        f.setObjectName("part")
        f.setStyleSheet(f"QFrame#part {{ background: #20242c; border: 1px solid {ACCENT if on else '#353b47'}; "
                        "border-radius: 6px; } QLabel { background: transparent; }")
        v = QVBoxLayout(f)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(2)
        top = QHBoxLayout()
        cb = QComboBox()
        for k, t in OPS.items():
            cb.addItem(_(t), k)
        cb.setCurrentIndex(list(OPS).index(part["op"]))
        cb.setFixedWidth(110)
        cb.currentIndexChanged.connect(lambda i: self._part_set(part, "op", cb.itemData(i)))
        top.addWidget(cb)
        t = QLabel(_(MASK_SOURCES[part["src"]]))
        t.setStyleSheet("font-weight: 600;")
        top.addWidget(t, 1)
        x = QPushButton("✕")
        x.setFixedSize(22, 22)
        x.setStyleSheet("padding: 0;")
        x.setToolTip(_("Remove from mask"))
        x.clicked.connect(lambda: self.remove_part(lay, part))
        top.addWidget(x)
        v.addLayout(top)
        src = part["src"]

        def sl(name, key, lo, hi, unit="", step=1):
            s = ContSlider(_(name), lo, hi, part[key], new_part(src, False)[key], unit=unit, step=step)
            self._wire(s, lambda val: part.__setitem__(key, val))
            v.addWidget(s)

        if src == "etoiles":
            sl("Size around stars", "taille", 0, 24, "px")
        elif src == "lum":
            sl("From", "lo", 0, 100, "%")
            sl("To", "hi", 0, 100, "%")
        elif src == "couleur":
            sl("Hue", "teinte", 0, 359, "°")
            sl("Width", "largeur", 10, 120, "°")
        elif src == "degrade":
            r = QHBoxLayout()
            r.addWidget(QLabel(_("Direction")))
            c = QComboBox()
            for d in DIRECTIONS:
                c.addItem(_(d), d)
            c.setCurrentIndex(DIRECTIONS.index(part["direction"]))
            c.currentIndexChanged.connect(lambda i: self._part_set(part, "direction", c.itemData(i)))
            r.addWidget(c, 1)
            v.addLayout(r)
            sl("Position", "position", 0, 100, "%")
            sl("Softness", "douceur", 1, 100, "%")
        elif src == "pinceau":
            r = QHBoxLayout()
            for txt, val in (("Paint", True), ("Erase", False)):
                b = QPushButton(_(txt))
                b.setCheckable(True)
                b.setChecked(part["paint"] == val)
                b.setObjectName("face")
                b.clicked.connect(lambda _=False, vv=val: self._brush_mode(part, vv))
                r.addWidget(b)
            r.addStretch(1)
            cl = QPushButton(_("Clear all"))
            cl.clicked.connect(lambda: self._part_set(part, "strokes", []))
            r.addWidget(cl)
            v.addLayout(r)
            sz = ContSlider(_("Size"), 4, 400, part["size"], 60, unit="px", step=1)
            sz.moved.connect(lambda val: (part.__setitem__("size", val), setattr(self.viewer, "brush_r", val / 2)))
            sz.done.connect(lambda val: (part.__setitem__("size", val), setattr(self.viewer, "brush_r", val / 2)))
            v.addWidget(sz)
            hd = ContSlider(_("Hardness"), 0, 100, part["hardness"], 50, unit="%", step=1)
            hd.done.connect(lambda val: part.__setitem__("hardness", val))
            v.addWidget(hd)
            if not on:
                go = QPushButton(_("Paint on image"))
                go.clicked.connect(lambda: self._edit_part(part))
                v.addWidget(go, 0, Qt.AlignLeft)
        f.mousePressEvent = lambda e: self._edit_part(part) if part["src"] == "pinceau" else None
        return f

    def _edit_part(self, part):
        self.part_sel = part["id"]
        self.build_props()

    def _brush_mode(self, part, paint):
        part["paint"] = paint
        self.part_sel = part["id"]
        self.build_props()

    def toggle_paint(self):
        lay = self.layer(self.sel[0]) if self.sel else None
        if not lay or not lay["mask"]:
            return
        part = next((p for p in lay["mask"]["parts"] if p["id"] == self.part_sel and p["src"] == "pinceau"), None)
        if part:
            self._brush_mode(part, not part["paint"])

    def brush_size(self, k):
        lay = self.layer(self.sel[0]) if self.sel else None
        part = next((p for p in (lay["mask"]["parts"] if lay and lay["mask"] else [])
                     if p["id"] == self.part_sel and p["src"] == "pinceau"), None)
        if part:
            part["size"] = float(np.clip(part["size"] * k, 4, 400))
            self.viewer.brush_r = part["size"] / 2
            self.build_props()

    def add_part(self, lay, src):
        self.push()
        part = new_part(src, not lay["mask"]["parts"])
        lay["mask"]["parts"].append(part)
        self.part_sel = part["id"] if src == "pinceau" else None
        self.build_props()
        self.request(final_only=True)

    def remove_part(self, lay, part):
        self.push()
        lay["mask"]["parts"] = [p for p in lay["mask"]["parts"] if p["id"] != part["id"]]
        if self.part_sel == part["id"]:
            self.part_sel = None
        self.build_props()
        self.request(final_only=True)

    def _part_set(self, part, key, value):
        self.push()
        part[key] = value
        if key == "strokes":
            self.eng._brush = {k: v for k, v in self.eng._brush.items() if k[0] != part["id"]}
        self.request(final_only=True)

    def _stroke(self, x, y, first):
        lay = self.layer(self.sel[0]) if self.sel else None
        part = next((p for p in (lay["mask"]["parts"] if lay and lay["mask"] else []) if p["id"] == self.part_sel),
                    None)
        if part is None or part["src"] != "pinceau":
            return
        if first:
            self._stroke_before = self.eng.snapshot()
            self._last_pt = None
        rad = part["size"] / 2
        pts = [(x, y)]
        if self._last_pt is not None:                    # fill the gaps of a fast stroke
            lx, ly = self._last_pt
            n = int(math.hypot(x - lx, y - ly) / max(rad / 3, 1))
            pts = [(lx + (x - lx) * k / (n + 1), ly + (y - ly) * k / (n + 1)) for k in range(1, n + 1)] + pts
        for px, py in pts:
            part["strokes"].append((px, py, rad, part["hardness"], part["paint"]))
        self._last_pt = (x, y)
        self.request()

    def _stroke_end(self):
        if self._stroke_before is not None:
            self.push(self._stroke_before)
            self._stroke_before = None
        self.request(final_only=True)

    # ------------------------------------------------------------ view toggles
    def _compare(self, on):
        self.viewer.compare = self.before if on else None
        self.viewer.update()
        self._hint()

    def _mask_view(self, on):
        self.show_mask = on
        self.src.version += 1
        self.viewer.refresh()
        self._hint()

    def _hint(self):
        if self.viewer.mode == "brush":
            t = _("Brush: drag to paint · X paint / erase · [ ] size · M show mask")
        elif self.sel and self.sel[1] == "mask":
            t = _("Red: where the layer shows · M: mask in gray")
        elif self.b_cmp.isChecked():
            t = _("Drag the divider · Before: the bottom layer, suggested settings")
        else:
            t = ""
        self.hint.setText(t)


class PostPage3(QWidget):
    """The Post, or what is missing for it."""
    goWorkflow = Signal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.s = session
        self.pages = QStackedWidget()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.pages)
        self.empty = QWidget()
        ev = QVBoxLayout(self.empty)
        ev.setAlignment(Qt.AlignCenter)
        ev.addWidget(label(_("No layers yet"), "h1"), 0, Qt.AlignHCenter)
        ev.addWidget(label(_("Each computed workflow becomes a layer here."), "dim"), 0, Qt.AlignHCenter)
        b = QPushButton(_("Go to graph  →"))
        b.setObjectName("link")
        b.clicked.connect(self.goWorkflow.emit)
        ev.addWidget(b, 0, Qt.AlignHCenter)
        self.pages.addWidget(self.empty)
        self.view = PostView(session)
        self.pages.addWidget(self.view)
        session.pathsChanged.connect(self.update_page)
        self.update_page()

    def update_page(self):
        self.pages.setCurrentIndex(1 if any(p.rendered for p in self.s.paths) else 0)

    def refresh(self):
        self.update_page()
        if self.pages.currentIndex() == 1:
            self.view.refresh()
