"""The Post of the client: the image at the centre; on the right the layers, as in Affinity Photo, and the properties of
the selected layer or of its mask.

The layers panel follows Affinity Photo's model. A pixel layer (a workflow's output, or a stage's image sent from
Inspect) shows its own filters as children, indented under it: they apply to it alone and move with it. A filter that
applies to everything below is a row of its own, like a layer, above what it affects. Drag a filter onto a layer to nest
it, between two rows to make it apply to everything below (an indented line: it joins that layer's children); the line,
the frame and a short label say where it goes. Read from the bottom up, the panel is the order of the composition.

A row: eye, the ▾ of a layer's children, thumbnail, mask thumbnail, name, blend mode and opacity. A click on the
thumbnail shows the layer's settings; a click on the mask thumbnail shows its mask, made of parts as in Lightroom, with a
red overlay on the image.

Nothing here renders. Every image comes from the render worker (services/render.py): a draft while a slider or the
brush moves (an eighth of the image on the CPU, a quarter on a GPU), on release the exact quarter first on the CPU then
the half resolution (the screen), and the visible region at full resolution when zoomed in. A newer request replaces an
older one. The layers are saved in the project (``project.post["layers"]``).

    page = PostPage3(model)              # model: the ProjectModel
    page.goWorkflow                      # "Go to graph →"
    page.view.goResults(pid)             # '"X": not computed yet →' and "Show workflow →": Inspect on that Output
    page.export(parent)                  # File > Export image… and the Export… button
    page.add_image_layer(name, source, key)   # Inspect: a stage's image as a layer
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .i18n import _, language, maybe, plural
from .postengine import (
    DIRECTIONS,
    FAMILIES,
    FILTERS,
    IMAGE_SLIDERS,
    MASK_SOURCES,
    MAX_LEVEL,
    MODES,
    OPS,
    PART_SETTINGS,
    THUMB,
    children,
    is_pixel,
    load_layers,
    move,
    new_filter_layer,
    new_id,
    new_image_layer,
    new_mask,
    new_part,
    new_workflow_layer,
    normalize,
    rows,
    save_layers,
    snapshot,
    tree,
)
from .services.render import CPU_DRAFT_LEVEL, DRAFT_LEVEL, HALF_LEVEL, RenderWorker, Request, default_device_pref
from .viewer import Viewer, to_qimage
from .widgets import ACCENT, WARN, clear, hline, label

FAMILY_COL = {"Tone": "#c9a35e", "Color": "#c46fa8", "Detail": "#5ea1ff", "Stars": "#e6d36a",
              "Combine": "#e06c6c"}
TW, TH = THUMB                                           # thumbnail size
COPY, STARLESS, STARS = " (copy)", "starless", "stars"   # pieces of the layer names, translated where shown
NEEDS_STARS = "Needs a layer with the Stars block"
REGION_MAX = 12_000_000                                  # pixels of the region rendered at full resolution
REGION_WAIT_MS = 150                                     # after the view stops moving
INDENT = 24                                              # a child row's indent (px)
QUARTER_SCALE = 0.26                                     # zoomed out to this, the quarter-resolution image fills the screen
TOGGLE_W = 16                                            # the ▾ / ▸ of a layer's children


def tr_name(n: str, image: bool = False) -> str:
    """A layer's name as shown: the pieces the program built it from (a filter's name, "stars", "starless", "(copy)")
    translated; the name of a workflow, or a name typed by the user, as it is (the graph shows it so). ``image``: an
    image layer, named by Inspect in English (``tune.TuneView._send``): every piece the catalogue has is translated
    (a file name stays as it is)."""
    k = 0
    while n.endswith(COPY):
        n, k = n[:-len(COPY)], k + 1
    if image:
        return _pieces(n, maybe) + _(COPY) * k
    known = {f["name"] for f in FILTERS.values()} | {STARS, STARLESS}
    return " · ".join(_(b) if b in known else b for b in n.split(" · ")) + _(COPY) * k


def layer_name(lay: dict) -> str:
    """The name of a layer as shown (``tr_name``)."""
    return tr_name(lay["name"], lay.get("kind") == "image")


def _pieces(n: str, fn) -> str:
    """``fn`` applied to each piece of a name made of pieces joined by "·" (the separators kept as they are)."""
    parts = re.split(r"(\s*·\s*)", n)
    return "".join(p if i % 2 else fn(p) for i, p in enumerate(parts))



def dec(s: str) -> str:
    """A decimal comma in French."""
    return s.replace(".", ",") if language() == "fr" else s


def with_unit(v: str, unit: str) -> str:
    if not unit:
        return v
    return v + ("" if unit == "°" or unit == "%" and language() != "fr" else " ") + unit


def pix(a: np.ndarray, w=TW, h=TH) -> QPixmap:
    """A pixmap of a uint8 image (h, w) or (h, w, 3), at the size of a thumbnail."""
    a = np.ascontiguousarray(a)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    if a.shape[:2] != (h, w):
        import cv2
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    return QPixmap.fromImage(to_qimage(a))


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
    """A row of the layers panel. ``depth`` 1: a filter nested in the pixel layer above it (indented, with a guide
    line); ``kids``: how many children a pixel layer has (its ▾ / ▸)."""

    def __init__(self, panel, lay, thumb, mthumb, sel_layer, sel_mask, depth=0, kids=0, collapsed=False, parent=None):
        super().__init__(parent)
        self.panel, self.lay, self.depth = panel, lay, depth
        self._press = None
        self._dragging = False
        self.setObjectName("lrow")
        on = sel_layer or sel_mask
        self.setStyleSheet(f"QFrame#lrow {{ background: {'#2b3446' if on else 'transparent'}; border-radius: 4px; "
                           "border: 2px solid transparent; } QFrame#lrow:hover { background: #262b35; } "
                           f"QFrame#lrow[drop=\"true\"] {{ border: 2px solid {ACCENT}; background: #22314a; }} "
                           "QLabel { background: transparent; }")
        h = QHBoxLayout(self)
        h.setContentsMargins(4, 3, 6, 3)
        h.setSpacing(6)
        eye = QCheckBox()
        eye.setChecked(lay["visible"])
        eye.setToolTip(_("Show / hide layer"))
        eye.toggled.connect(lambda v: panel.set_visible(lay["id"], v))
        h.addWidget(eye)
        self.toggle = None
        if depth:                                       # a child: indented, a guide line down from its layer
            guide = QFrame()
            guide.setFixedWidth(2)
            guide.setStyleSheet("background: #4a5366;")
            h.addSpacing(TOGGLE_W // 2 - 1)
            h.addWidget(guide)
            h.addSpacing(INDENT - TOGGLE_W // 2 - 1)
        elif kids:
            self.toggle = QToolButton()
            self.toggle.setText("▸" if collapsed else "▾")
            self.toggle.setAutoRaise(True)
            self.toggle.setFixedWidth(TOGGLE_W)
            self.toggle.setStyleSheet("QToolButton { border: none; color: #aab2bf; padding: 0; }")
            self.toggle.setToolTip(_("Show / hide its filters"))
            self.toggle.clicked.connect(lambda: panel.toggle_children(lay["id"]))
            h.addWidget(self.toggle)
        else:
            h.addSpacing(TOGGLE_W)
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
        self.n = QLabel(layer_name(lay))
        self.n.setStyleSheet("font-weight: 600;" + ("" if lay["visible"] else " color: #8b93a1;"))
        v.addWidget(self.n)
        self.sub = label(panel.sub_text(lay), "small")
        v.addWidget(self.sub)
        h.addLayout(v, 1)
        if depth:
            tgt = panel.layer(lay.get("target"))
            self.setToolTip(_('Applies to "{name}" only').format(name=layer_name(tgt)) if tgt else "")
        elif lay["kind"] == "filter":
            self.setToolTip(_("Applies to everything below"))

    def set_drop(self, on: bool, text: str = ""):
        """The frame of a drop that nests the dragged filter in this layer; ``text`` (what the drop does) shows in place of
        the row's mode and opacity while it lasts."""
        if bool(self.property("drop")) == on and (not on or self.sub.text() == text):
            return
        self.setProperty("drop", on)
        self.style().unpolish(self)
        self.style().polish(self)
        self.sub.setText(text if on else self.panel.sub_text(self.lay))
        self.sub.setStyleSheet(f"color: {ACCENT}; font-weight: 600;" if on else "")

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
            if not self._dragging:
                self._dragging = True
                self.panel.drag_start(self)
            self.panel.drag_move(self, e.globalPosition().toPoint())

    def mouseReleaseEvent(self, e):
        if self._press is None:
            return
        press, self._press = self._press, None
        if self._dragging:
            self._dragging = False
            self.panel.drag_end(self, e.globalPosition().toPoint())
        else:                                            # a click: select (after the event, the rows are rebuilt)
            what = self._target(press)
            QTimer.singleShot(0, lambda: self.panel.select(self.lay["id"], what))

    def mouseDoubleClickEvent(self, e):
        self.panel.rename(self.lay)


def _upsample(a: np.ndarray, k: int = 2) -> np.ndarray:
    return np.repeat(np.repeat(a, k, 0), k, 1)


class PostSource:
    """What the viewer shows: the composite (half resolution; the quarter, enlarged, from a draft that is newer), the
    region at full resolution when zoomed in (from the worker; the half image enlarged until it arrives), or the mask in
    grey. Everything is already rendered: the viewer asks in the GUI thread and nothing is computed here."""

    def __init__(self, view):
        self.view = view
        self.version = 0
        self.img1 = None                                  # uint8 (H/2, W/2, 3) and the request it answers
        self.seq1 = 0
        self.img2 = None                                  # the draft, (H/4, W/4, 3)
        self.seq2 = 0
        self.level2 = 2                                   # its resolution: 2 a quarter, 3 an eighth
        self.full = None                                  # (box, uint8 array) at full resolution
        self.canvas = None                                # (W, H)

    def size(self):
        return self.canvas or (2, 2)

    def _half(self) -> np.ndarray | None:
        """The image at half resolution to show: the newest of the half render and the (enlarged) draft."""
        if self.img2 is not None and self.seq2 > self.seq1:
            return _upsample(self.img2, 2 ** (self.level2 - 1))[:self.canvas[1] // 2, :self.canvas[0] // 2]
        return self.img1

    def region(self, level, x0, y0, x1, y1):
        v = self.view
        if v.show_mask and v.mask1 is not None:
            m = (np.clip(v.mask1, 0, 1) * 255).astype(np.uint8)
            if level == 0:
                m = _upsample(m)
            return np.repeat(m[y0:y1, x0:x1, None], 3, -1)
        a = self._half()
        if a is None:
            return np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        if level == 1:
            return a[y0:y1, x0:x1]
        b = _upsample(a[y0 // 2: -(-y1 // 2), x0 // 2: -(-x1 // 2)])
        out = b[y0 % 2: y0 % 2 + (y1 - y0), x0 % 2: x0 % 2 + (x1 - x0)]
        if self.full is not None:
            (bx0, by0, bx1, by1), arr = self.full
            ix0, iy0, ix1, iy1 = max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1)
            if ix1 > ix0 and iy1 > iy0:
                out = np.array(out)
                out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = arr[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
        return out


class BeforeSource:
    """The before of the before/after: the bottom workflow layer, all sliders at the suggestion (rendered by the
    worker, at half resolution; enlarged when zoomed in)."""

    def __init__(self, view):
        self.view = view
        self.version = 0
        self.img1 = None

    def size(self):
        return self.view.src.size()

    def region(self, level, x0, y0, x1, y1):
        a = self.img1
        if a is None:
            return np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        if level == 1:
            return a[y0:y1, x0:x1]
        b = _upsample(a[y0 // 2: -(-y1 // 2), x0 // 2: -(-x1 // 2)])
        return b[y0 % 2: y0 % 2 + (y1 - y0), x0 % 2: x0 % 2 + (x1 - x0)]


# ======================================================================== the view
class PostView(QWidget):
    goProcessing = Signal(int, str)
    goResults = Signal(int)
    rendered = Signal(object)                             # a Result was shown (tests, the status line)

    def __init__(self, session, parent=None, *, worker: RenderWorker | None = None, persist: bool = True):
        super().__init__(parent)
        self.setObjectName("root")
        self.s = session
        self.persist = persist
        cpu = getattr(session, "device", None) == "cpu" or getattr(session, "gpu", True) is False
        self.worker = worker or RenderWorker(getattr(session, "p", None),
                                             device="cpu" if cpu else default_device_pref())
        self.layers: list[dict] = []
        self.sel = None                                 # (layer id, "layer" | "mask")
        self.part_sel = None                            # the mask part being edited (its id)
        self.show_mask = False
        self.mask1 = None                               # the selected layer's mask at half resolution (float)
        self.undo, self.redo = [], []
        self._mthumbs: dict[int, QPixmap] = {}
        self._lthumbs: dict[int, tuple] = {}            # layer id -> (signature, pixmap)
        self._overlay_img = None
        self._stroke_before = None
        self._last_pt = None
        self._notes: list[str] = []
        self._device = ""
        self._last_seq = {"draft": 0, "half": 0, "region": 0, "before": 0, "refine": 0}
        self._saved = ""
        self.draft_ms: list[float] = []
        self.draft_level = CPU_DRAFT_LEVEL if cpu else DRAFT_LEVEL
        self._collapsed: set[int] = set()                # the pixel layers whose children are hidden in the panel
        self._half_due = False                           # (CPU) the exact quarter is shown, the half not rendered
        self._missing_images: set[str] = set()           # image layers' keys whose pixels the worker did not find
        self._drop = None                                # the drop the drag shows (see drop_at)
        self.first_image = False
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
        self.viewer.overlays.append(self._overlay)
        self.viewer.banner = _("Rendering…")
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
        head = QHBoxLayout()
        head.addWidget(label(_("Layers"), "h2"))
        head.addStretch(1)
        # while a row is dragged between two others: what the drop does, beside the panel's title (a drop onto a layer
        # says it in that layer's row)
        self.drop_text = QLabel()
        self.drop_text.setStyleSheet(f"color: {ACCENT}; font-weight: 600;")
        self.drop_text.hide()
        head.addWidget(self.drop_text)
        rv.addLayout(head)
        self.lsc = QScrollArea()
        self.lsc.setWidgetResizable(True)
        self.lsc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lh = QWidget()
        self.lbox = QVBoxLayout(lh)
        self.lbox.setContentsMargins(0, 0, 4, 12)          # (room for a drop line after the last row)
        self.lbox.setSpacing(2)
        self.lsc.setWidget(lh)
        self.lsc.setMaximumHeight(370)
        rv.addWidget(self.lsc)
        self.drop_line = QFrame(lh)
        self.drop_line.setFixedHeight(3)
        self.drop_line.setStyleSheet(f"background: {ACCENT}; border-radius: 1px;")
        self.drop_line.hide()
        self.drop_dot = QFrame(lh)                       # the start of the line, as in Affinity Photo
        self.drop_dot.setFixedSize(9, 9)
        self.drop_dot.setStyleSheet(f"background: #1b1f27; border: 2px solid {ACCENT}; border-radius: 4px;")
        self.drop_dot.hide()
        btns = QHBoxLayout()
        btns.setSpacing(4)
        self.b_filter = QPushButton(_("+ Filter"))
        self.b_filter.setToolTip(_("A filter layer above the selected layer"))
        self.filter_menu = QMenu(self.b_filter)
        self.b_filter.setMenu(self.filter_menu)
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
        self.b_export.clicked.connect(lambda: self.export_dialog())
        self.b_dup.clicked.connect(self.duplicate)
        self.b_maskadd.clicked.connect(self.mask_button)
        self.b_del.clicked.connect(self.delete)
        self.viewer.viewChanged.connect(self._view_changed)
        self.viewer.stroke.connect(self._stroke)
        self.viewer.strokeEnd.connect(self._stroke_end)
        # Ctrl+Z / Ctrl+Y are the window's Edit > Undo / Redo, which come here while Post is shown (two shortcuts on
        # one key would cancel each other)
        for seq, fn in (("C", self.b_cmp.toggle), ("M", self.b_mask.toggle), ("F", self.viewer.fit),
                        ("1", lambda: self.viewer.zoom_to(1.0)), ("X", self.toggle_paint),
                        ("[", lambda: self.brush_size(1 / 1.25)), ("]", lambda: self.brush_size(1.25)),
                        ("Ctrl+J", self.duplicate)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        self._draft_t = QTimer(self)
        self._draft_t.setSingleShot(True)
        self._draft_t.timeout.connect(self._send_draft)
        self._final_t = QTimer(self)
        self._final_t.setSingleShot(True)
        self._final_t.timeout.connect(self._send_final)
        self._region_t = QTimer(self)
        self._region_t.setSingleShot(True)
        self._region_t.timeout.connect(self._send_region)
        self._save_t = QTimer(self)                      # the composition into the project, just after an edit (push)
        self._save_t.setSingleShot(True)
        self._save_t.timeout.connect(self._save)
        self.worker.rendered.connect(self._on_result)
        self.worker.failed.connect(self._on_failed)
        self.worker.deviceChanged.connect(self._on_device)
        if self.worker.device:                           # the worker may have said it before we listened
            self._on_device(self.worker.device)
        self.s.pathsChanged.connect(self.sync_paths)
        self.destroyed.connect(self.worker.stop if worker is None else (lambda *_a: None))
        if self.layers:
            self.sel = (self.layers[-1]["id"], "layer")
        self.build_filter_menu()
        self.rebuild()
        self._send_final()

    # ------------------------------------------------------------ the model
    def path(self, pid):
        """The model's workflow of a layer (a WorkflowView), or None."""
        return self.s.view(pid)

    def shutdown(self):
        self.flush()
        self.worker.stop()

    def flush(self):
        """The last edit into the project now (before the project is saved and closed)."""
        if self._save_t.isActive():
            self._save_t.stop()
            self._save()

    # ------------------------------------------------------------ layers from the workflows
    def sync_paths(self, initial=False):
        paths = list(self.s.paths)
        alive = {p.pid for p in paths}
        if initial:
            post = getattr(getattr(self.s, "p", None), "post", None) or {}
            self.layers = load_layers(post.get("layers"), alive)
            self._saved = json.dumps(save_layers(self.layers), sort_keys=True)
        known = {lay["pid"] for lay in self.layers if lay["kind"] == "workflow"}
        fresh = not known
        for p in paths:
            if p.rendered and p.pid not in known:
                self.layers.insert(0, new_workflow_layer(p.pid, p.name))
        self.layers = [lay for lay in self.layers if lay["kind"] != "workflow" or lay["pid"] in alive]
        self.layers = normalize(self.layers)
        wl = [lay for lay in self.layers if lay["kind"] == "workflow"]
        if wl and fresh:                                # the first layers: "Full" (else the bottom one) is the base
            names = {lay["id"]: self.path(lay["pid"]).name for lay in wl}
            base = next((lay for lay in reversed(wl) if names[lay["id"]] == "Full"),
                        next((lay for lay in reversed(wl) if names[lay["id"]].startswith("Full")), wl[-1]))
            self.layers.remove(base)
            self.layers.append(base)
            base["visible"] = True
        elif wl and not any(lay["visible"] for lay in wl):
            wl[-1]["visible"] = True
        if not initial:
            if self.sel and not self.layer(self.sel[0]):
                self.sel = (self.layers[-1]["id"], "layer") if self.layers else None
            self.rebuild()
            self.worker.refresh()
            self.request(final_only=True)                  # (nothing is saved here: the layers a workflow brings are
                                                           # worked out again from the workflows, only edits are kept)

    def layer(self, lid):
        return next((lay for lay in self.layers if lay["id"] == lid), None)

    def index(self, lid):
        return next(i for i, lay in enumerate(self.layers) if lay["id"] == lid)

    def _save(self):
        """The composition goes into the project (its definition file is written by the model, as for any edit)."""
        p = getattr(self.s, "p", None)
        if not self.persist or p is None or not hasattr(p, "post"):
            return
        data = save_layers(self.layers)
        text = json.dumps(data, sort_keys=True)
        if text == self._saved:
            return
        self._saved = text
        p.post = {**(p.post or {}), "layers": data}
        edited = getattr(self.s, "edited", None)
        if edited is not None:
            edited()

    # ------------------------------------------------------------ rendering (in the worker)
    def refresh(self):
        self.request(final_only=True)

    def request(self, final_only=False):
        """Something changed: a draft now (while it moves), the screen's image 220 ms after the last change."""
        self.src.full = None
        if not final_only:
            self._draft_t.start(0)
        self._final_t.start(0 if final_only else 220)

    def _selected_wants(self):
        want = (self.sel[0],) if self.sel else ()
        return want, bool(self.sel and self.sel[1] == "mask")

    def _send_draft(self):
        want, overlay = self._selected_wants()
        self.worker.request(Request(kind="draft", layers=snapshot(self.layers), level=self.draft_level, want=want,
                                    overlay=overlay))

    def _quarter_enough(self) -> bool:
        """On the CPU, when the screen shows the image at a quarter or less: the exact quarter is the picture."""
        cpu = (self._device or self.worker.device) == "cpu"
        return cpu and self.src.canvas is not None and self.viewer.scale <= QUARTER_SCALE

    def _send_final(self):
        self._draft_t.stop()
        want, overlay = self._selected_wants()
        cpu = (self._device or self.worker.device) == "cpu"
        if self._quarter_enough():
            # the CPU, zoomed out: the exact quarter (about a second) is what the screen shows; the half comes when the
            # view is zoomed in past a quarter
            self._half_due = True
            self.worker.request(Request(kind="refine", layers=snapshot(self.layers), level=HALF_LEVEL + 1, want=want,
                                        thumbs=True, overlay=overlay))
        else:
            # on the CPU the exact quarter comes first, then the half
            self._half_due = False
            self.worker.request(Request(kind="half", layers=snapshot(self.layers), level=HALF_LEVEL, want=want,
                                        thumbs=True, overlay=overlay, refine=cpu))
        if self.b_cmp.isChecked():
            self._send_before()
        need = tuple(self._stale_thumbs())
        if need:
            self.worker.request(Request(kind="thumbs", layers=snapshot(self.layers), need_thumbs=need))

    def _send_before(self):
        self.worker.request(Request(kind="before", layers=snapshot(self.layers), level=HALF_LEVEL))

    def _stale_thumbs(self):
        for lay in self.layers:
            if is_pixel(lay) and self._lthumbs.get(lay["id"], (None,))[0] != self._thumb_sig(lay):
                yield lay["id"]

    @staticmethod
    def _thumb_sig(lay):
        return (lay.get("pid"), lay.get("key"), lay.get("part"), json.dumps(lay["sliders"], sort_keys=True),
                json.dumps(lay.get("base"), sort_keys=True, default=str), lay.get("linked"))

    def _view_changed(self):
        self.zoom_lab.setText(_("Zoom {p}").format(p=with_unit(f"{self.viewer.scale * 100:.0f}", "%")))
        if self._half_due and self.viewer.scale > QUARTER_SCALE:     # zoomed in past the quarter: the half now
            self._half_due = False
            self._final_t.start(0)
        self._region_t.start(REGION_WAIT_MS)

    def visible_box(self):
        """The part of the image to render at full resolution when zoomed in: what the viewer shows with a quarter more
        on each side, or None (zoomed out: the half image is enough)."""
        v = self.viewer
        W, H = self.src.size()
        if self.src.canvas is None or v.scale < 0.5:
            return None
        ix0, iy0 = v.w_to_img(0, 0)
        ix1, iy1 = v.w_to_img(v.width(), v.height())
        ix0, iy0, ix1, iy1 = max(0.0, ix0), max(0.0, iy0), min(float(W), ix1), min(float(H), iy1)
        if ix1 - ix0 < 2 or iy1 - iy0 < 2:
            return None
        mx, my = (ix1 - ix0) / 4, (iy1 - iy0) / 4
        box = [max(0, ix0 - mx), max(0, iy0 - my), min(W, ix1 + mx), min(H, iy1 + my)]
        if (box[2] - box[0]) * (box[3] - box[1]) > REGION_MAX:
            box = [ix0, iy0, ix1, iy1]
        return int(box[0]), int(box[1]), min(W, int(math.ceil(box[2]))), min(H, int(math.ceil(box[3])))

    def _send_region(self):
        box = self.visible_box()
        if box is None:
            return
        f = self.src.full
        if f is not None and f[0][0] <= box[0] and f[0][1] <= box[1] and f[0][2] >= box[2] and f[0][3] >= box[3]:
            return
        if self.src.img1 is None:
            return
        self.worker.request(Request(kind="region", layers=snapshot(self.layers), level=0, box=box))

    def _on_result(self, res):
        if res.kind == "inspect":
            return                                              # Inspect's final image, rendered on the same worker
        if res.seq < self._last_seq.get(res.kind, 0):
            return                                              # an answer older than one already shown
        if res.kind != "thumbs":
            self._last_seq[res.kind] = res.seq
        if res.canvas is not None:
            size = (res.canvas.W, res.canvas.H)
            if self.src.canvas != size:
                self.src.canvas = size
                self.viewer.set_source(self.src)
        if res.kind == "thumbs":
            self._take_layer_thumbs(res)
            return
        if res.kind in ("half", "refine") and set(res.missing) != self._missing_images:
            self._missing_images = set(res.missing)
            self._refresh_rows()
        if res.rgb is None:
            self.viewer.banner = _("No layers yet")
            self.viewer.update()
            return
        self.viewer.banner = ""
        self.first_image = True
        if res.kind in ("draft", "half", "refine"):
            if res.kind in ("draft", "refine"):
                self.src.img2, self.src.seq2, self.src.level2 = res.rgb, res.seq, res.level
                if res.kind == "draft":
                    self.draft_ms = (self.draft_ms + [res.ms])[-4:]
                    self._adapt_draft()
            else:
                self.src.img1, self.src.seq1 = res.rgb, res.seq
            if res.kind == "refine" and res.mask_thumbs:
                self._take_mask_thumbs(res)
            if res.kind == "refine" and res.layer_thumbs:
                self._take_layer_thumbs(res)
            sel = self.sel[0] if self.sel else None
            m = res.masks.get(sel) if sel is not None else None
            self.mask1 = None if m is None else (_upsample(m, 2 ** (res.level - 1)) if res.level > 1 else m)
            self._set_overlay(res.overlay if self.sel and self.sel[1] == "mask" else None)
            self.src.version += 1
            if res.kind == "half":
                self._take_mask_thumbs(res)
                self._take_layer_thumbs(res)
                if self.viewer.scale >= 0.5:
                    self._region_t.start(60)
            self.viewer.refresh()
        elif res.kind == "region":
            self.src.full = (res.box, res.rgb)
            self.src.version += 1
            self.viewer.refresh()
        elif res.kind == "before":
            self.before.img1 = res.rgb
            self.before.version += 1
            self.viewer.update()
        if res.notes:                                           # the engine fell back to the CPU: said until it changes
            self._notes = sorted(res.notes)
        self._hint()
        self.rendered.emit(res)

    def _adapt_draft(self):
        """The draft must stay within its budget (150 ms of the engine): on a slow machine it drops a level (to an
        eighth of the image at most), and goes back up to its device's level when that is fast again."""
        from .services.render import BUDGET
        if len(self.draft_ms) < 2:
            return
        avg = sum(self.draft_ms[-2:]) / 2
        base = CPU_DRAFT_LEVEL if self._device == "cpu" else DRAFT_LEVEL
        if self.draft_level < MAX_LEVEL and avg > 2.5 * 1000 * BUDGET["draft"]:
            self.draft_level, self.draft_ms = self.draft_level + 1, []
        elif self.draft_level > base and avg < 0.3 * 1000 * BUDGET["draft"]:
            self.draft_level, self.draft_ms = self.draft_level - 1, []

    def _on_failed(self, seq, kind, text):
        if kind in ("draft", "half", "region", "before", "start"):
            first = text.splitlines()[0] if text else "?"
            self.viewer.banner = first[:160]
            self.viewer.update()

    def _on_device(self, dev):
        self._device, self._notes = dev, []
        if dev == "cpu":
            self.draft_level = max(self.draft_level, CPU_DRAFT_LEVEL)

    def _take_mask_thumbs(self, res):
        for lid, t in res.mask_thumbs.items():
            self._mthumbs[lid] = pix(t)
        self._update_mask_thumb()

    def _take_layer_thumbs(self, res):
        changed = False
        for lid, t in res.layer_thumbs.items():
            lay = self.layer(lid)
            if lay is not None:
                self._lthumbs[lid] = (self._thumb_sig(lay), pix(t))
                changed = True
        if changed:
            self._update_layer_thumbs()

    def _set_overlay(self, a):
        if a is None:
            self._overlay_img = None
            return
        a = np.ascontiguousarray(a)
        h, w = a.shape[:2]
        self._overlay_img = QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()

    # ------------------------------------------------------------ the red overlay of the mask being edited
    def _overlay(self, p: QPainter, v: Viewer):
        if not self.sel or self.sel[1] != "mask" or self._overlay_img is None or self.show_mask or \
                self.b_cmp.isChecked() or self.src.canvas is None:
            return
        W, H = self.src.size()
        p.drawImage(QRectF(v.img_to_w(0, 0), v.img_to_w(W, H)), self._overlay_img)

    # ------------------------------------------------------------ undo
    def push(self, before=None):
        """One undo step: the composition before a change. The change itself follows in the same call (or came just
        before it, for a slider let go), so the composition is saved when the event loop comes back, with the change."""
        self.undo.append(before if before is not None else snapshot(self.layers))
        self.redo.clear()
        self._undo_buttons()
        self._save_t.start(0)

    def _undo_buttons(self):
        self.b_undo.setEnabled(bool(self.undo))
        self.b_redo.setEnabled(bool(self.redo))

    def do_undo(self):
        if self.undo:
            self.redo.append(snapshot(self.layers))
            self.layers = self.undo.pop()
            self._restored()

    def do_redo(self):
        if self.redo:
            self.undo.append(snapshot(self.layers))
            self.layers = self.redo.pop()
            self._restored()

    def _restored(self):
        if self.sel and not self.layer(self.sel[0]):
            self.sel = (self.layers[-1]["id"], "layer") if self.layers else None
        self.rebuild()
        self.request(final_only=True)
        self._save()

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
        hit = self._lthumbs.get(lay["id"])
        if hit is not None:
            return hit[1]
        pm = QPixmap(TW, TH)
        pm.fill(QColor("#1d2129"))
        return pm

    def mask_thumb(self, lay):
        if lay["mask"] is None:
            return None
        pm = self._mthumbs.get(lay["id"])
        if pm is None:
            pm = pix(np.full((TH, TW), 255, np.uint8))             # an empty mask shows the layer everywhere
            self._mthumbs[lay["id"]] = pm
        return pm

    def _update_mask_thumb(self):
        for row in self.rows:
            if row.m is not None and row.lay["id"] in self._mthumbs:
                row.m.setPixmap(self._mthumbs[row.lay["id"]])

    def _update_layer_thumbs(self):
        for row in self.rows:
            if is_pixel(row.lay) and row.lay["id"] in self._lthumbs:
                row.t.setPixmap(self._lthumbs[row.lay["id"]][1])

    def build_filter_menu(self):
        """+ Filter: a menu by family; the filters that need stars or the H-alpha map are off without them."""
        self.filter_menu.clear()
        has_stars = self.has_stars()
        for fam in FAMILIES:
            sub = self.filter_menu.addMenu(_(fam))
            for k, f in FILTERS.items():
                if f["family"] == fam:
                    a = sub.addAction(_(f["name"]))
                    a.triggered.connect(lambda _c=False, kk=k: self.add_filter(kk))
                    if f.get("needs") == "stars" and not has_stars:
                        a.setEnabled(False)
                        a.setToolTip(_(NEEDS_STARS))

    def has_stars(self) -> bool:
        """Whether a layer with the Stars block is in the composition (the star mask and the star filters need it)."""
        for lay in self.layers:
            if lay["kind"] == "workflow":
                p = self.path(lay["pid"])
                if p is not None and self.s.slider_state(p, "stars_number")[0]:
                    return True
        return False

    def rebuild(self):
        clear(self.lbox)
        self.rows = []
        self._collapsed &= {lay["id"] for lay in self.layers}
        if self.sel and (lay := self.layer(self.sel[0])) is not None and lay.get("target") in self._collapsed:
            self._collapsed.discard(lay["target"])       # the selected filter is shown
        for lay, depth in rows(self.layers, self._collapsed):
            sl = self.sel is not None and self.sel[0] == lay["id"]
            row = LayerRow(self, lay, self.thumb(lay), self.mask_thumb(lay), sl and self.sel[1] == "layer",
                           sl and self.sel[1] == "mask", depth, len(children(self.layers, lay["id"])) if is_pixel(lay)
                           else 0, lay["id"] in self._collapsed)
            self.lbox.addWidget(row)
            self.rows.append(row)
        waiting = [p for p in self.s.paths if not p.rendered]
        for p in waiting:
            b = QPushButton(_('"{name}": not computed yet  →').format(name=tr_name(p.name)))
            b.setObjectName("link")
            b.clicked.connect(lambda _=False, pid=p.pid: self.goResults.emit(pid))
            self.lbox.addWidget(b)
        self.lbox.addStretch(1)
        # as tall as the rows really are (styled first: a row is about 52 px, not a fixed guess; they are not shown yet,
        # so the layout would not count them), up to 382 px: a list that fits never opens scrolled, its first row cut
        kids = [w for w in (self.lbox.itemAt(i).widget() for i in range(self.lbox.count())) if w is not None]
        self.lsc.widget().ensurePolished()
        m = self.lbox.contentsMargins()
        want = sum(w.sizeHint().height() for w in kids) + self.lbox.spacing() * len(kids) + m.top() + m.bottom()
        self.lsc.setFixedHeight(int(min(382, want + 2 * self.lsc.frameWidth())))
        sel_row = next((r for r in self.rows if self.sel and r.lay["id"] == self.sel[0]), None)
        if sel_row is not None:
            QTimer.singleShot(0, lambda: self.lsc.ensureWidgetVisible(sel_row, 0, 10))
        lay = self.layer(self.sel[0]) if self.sel else None
        self.b_dup.setEnabled(lay is not None)
        self.b_del.setEnabled(lay is not None and (not is_pixel(lay) or sum(map(is_pixel, self.layers)) > 1))
        self.b_maskadd.setEnabled(lay is not None)
        self.b_maskadd.setText(_("Mask") if lay is None or lay["mask"] is None else _("Mask") + " ✓")
        self.b_maskadd.setToolTip(_("Add a mask to the selected layer") if lay is None or lay["mask"] is None else
                                  _("Edit the mask of the selected layer"))
        self._undo_buttons()
        self.build_filter_menu()
        self.build_props()

    def select(self, lid, what):
        if self.sel == (lid, what):
            return
        self.sel = (lid, what)
        lay = self.layer(lid)
        if what == "mask" and lay["mask"] is not None:
            brush = next((p for p in lay["mask"]["parts"] if p["src"] == "brush"), None)
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

    def toggle_children(self, lid):
        """▾ / ▸: show or hide a layer's filters in the panel (the image does not change). A hidden filter that was
        selected leaves the selection to its layer."""
        self._collapsed ^= {lid}
        cur = self.layer(self.sel[0]) if self.sel else None
        if lid in self._collapsed and cur is not None and cur.get("target") == lid:
            self.sel, self.part_sel = (lid, "layer"), None
        self.rebuild()

    # ---- drag and drop (Affinity Photo's): onto a layer nests, between rows places; the line says at which depth
    def drag_start(self, row):
        eff = QGraphicsOpacityEffect(row)
        eff.setOpacity(0.45)
        row.setGraphicsEffect(eff)
        for r in self.rows:                              # a layer is dragged with its children
            if r.lay.get("target") == row.lay["id"] and r.depth:
                e = QGraphicsOpacityEffect(r)
                e.setOpacity(0.45)
                r.setGraphicsEffect(e)
        self.lsc.viewport().setCursor(Qt.ClosedHandCursor)

    def drop_at(self, lay, x: float, y: float) -> dict | None:
        """Where a row dropped at (x, y) of the panel goes: {parent, index, row (nest) | y, x0 (line), text}. A filter
        over the middle of a pixel layer's row nests in it (on top of its children); between two rows it goes there -
        among a layer's children when the line is indented (the gap sits in a group, or the pointer is to the right
        after a group), else as a row of its own that applies to everything below. A pixel layer goes between top-level
        rows only."""
        rs = self.rows
        if not rs:
            return None
        is_f = lay["kind"] == "filter"
        mine = {lay["id"]} | ({c["id"] for c in children(self.layers, lay["id"])} if is_pixel(lay) else set())
        gap = len(rs) if y > rs[-1].geometry().bottom() else 0
        for i, r in enumerate(rs):
            g = r.geometry()
            if g.top() <= y <= g.bottom():
                rel = (y - g.top()) / max(1, g.height())
                if is_f and is_pixel(r.lay) and r.lay["id"] != lay.get("target", -1) and 0.28 <= rel <= 0.72:
                    return {"parent": r.lay["id"], "index": 0, "row": i,
                            "text": _('Applies to "{name}" only').format(name=layer_name(r.lay))}
                if is_f and is_pixel(r.lay) and 0.28 <= rel <= 0.72:
                    return None                         # onto its own layer: where it is
                gap = i if rel < 0.5 else i + 1
                break
        above = rs[gap - 1] if gap > 0 else None
        below = rs[gap] if gap < len(rs) else None
        indent_x = self.rows[0].geometry().left() + 60
        parent = None
        if below is not None and below.depth:           # inside a group
            parent = below.lay["target"]
        elif above is not None and is_f and x >= indent_x and (above.depth or is_pixel(above.lay)) and \
                (above.depth or above.lay["id"] not in self._collapsed):
            parent = above.lay["target"] if above.depth else above.lay["id"]
        if parent is not None and (not is_f or parent in mine):
            # a pixel layer is never nested: the line goes after the group
            while gap < len(rs) and rs[gap].depth:
                gap += 1
            above = rs[gap - 1] if gap > 0 else None
            below = rs[gap] if gap < len(rs) else None
            parent = None
        if parent is not None:
            kids = [c["id"] for c in children(self.layers, parent) if c["id"] not in mine]
            before = [r.lay["id"] for r in rs[:gap] if r.depth and r.lay.get("target") == parent and
                      r.lay["id"] not in mine]
            index = len([k for k in kids if k in before])
            text = _('Applies to "{name}" only').format(name=layer_name(self.layer(parent)))
            depth = 1
        else:
            index = len([r for r in rs[:gap] if not r.depth and r.lay["id"] not in mine])
            text = _("Applies to everything below") if is_f else ""
            depth = 0
        line_y = below.geometry().top() - 2 if below is not None else rs[-1].geometry().bottom() + 1
        x0 = rs[0].geometry().left() + (34 + INDENT if depth else 30)
        return {"parent": parent, "index": index, "y": line_y, "x0": x0, "text": text}

    def drag_move(self, row, gpos):
        holder = self.lbox.parentWidget()
        vp = self.lsc.viewport()
        y_vp = vp.mapFromGlobal(gpos).y()               # near an edge of the list: it scrolls
        sb = self.lsc.verticalScrollBar()
        if y_vp < 24:
            sb.setValue(sb.value() - 14)
        elif y_vp > vp.height() - 24:
            sb.setValue(sb.value() + 14)
        pt = holder.mapFromGlobal(gpos)
        d = self.drop_at(row.lay, pt.x(), pt.y())
        self._drop = d
        for r in self.rows:
            nest = d is not None and d.get("row") is not None and r is self.rows[d["row"]]
            r.set_drop(nest, d["text"] if nest else "")
        self.drop_line.hide()
        self.drop_dot.hide()
        self.drop_text.hide()
        if d is None:
            self.lsc.viewport().setCursor(Qt.ForbiddenCursor)
            return
        self.lsc.viewport().setCursor(Qt.ClosedHandCursor)
        w = holder.width()
        if d.get("row") is None:
            self.drop_line.setGeometry(d["x0"], d["y"] - 1, w - d["x0"] - 4, 3)
            self.drop_dot.move(d["x0"] - 7, d["y"] - 4)
            self.drop_line.show()
            self.drop_dot.show()
            self.drop_line.raise_()
            self.drop_dot.raise_()
            if d["text"]:
                self.drop_text.setText(d["text"])
                self.drop_text.show()

    def drag_cancel(self):
        """The drag ends without a drop: the hints go, nothing moves."""
        self._drop = None
        self.drop_line.hide()
        self.drop_dot.hide()
        self.drop_text.hide()
        self.lsc.viewport().unsetCursor()
        for r in self.rows:
            r.set_drop(False)
            r.setGraphicsEffect(None)

    def drag_end(self, row, gpos):
        d = self._drop
        self.drag_cancel()
        if d is not None:
            if d["parent"] is not None:
                self._collapsed.discard(d["parent"])
            self.move(row.lay["id"], d["parent"], d["index"])

    def move(self, lid, parent=None, index=0) -> bool:
        """Move a layer as a drop does (postengine.move): one undo step, nothing when it lands where it was."""
        before = snapshot(self.layers)
        new = move(self.layers, lid, parent, index)
        if [(x["id"], x.get("target")) for x in new] == [(x["id"], x.get("target")) for x in before]:
            return False
        self.push(before)
        self.layers = new
        self.rebuild()
        self.request(final_only=True)
        return True

    def scope_choices(self, fl) -> list[tuple[str, int | None]]:
        """What a filter can apply to: (text, target): the layer it is nested in, else the pixel layer just under its
        row; and everything below."""
        tgt = self.layer(fl.get("target")) if fl.get("target") else None
        if tgt is None:
            t = tree(self.layers)
            i = next(k for k, (x, _kids) in enumerate(t) if x["id"] == fl["id"])
            tgt = next((x for x, _kids in t[i + 1:] if is_pixel(x)), None)
        out = [(_('"{name}" only').format(name=layer_name(tgt)), tgt["id"])] if tgt else []
        return out + [(_("Everything below"), None)]

    def layer_menu(self, lay, gpos):
        m = QMenu(self)
        m.addAction(_("Duplicate")).triggered.connect(lambda: (self.select(lay["id"], "layer"), self.duplicate()))
        if lay["mask"] is None:
            m.addAction(_("Add mask")).triggered.connect(lambda: self.add_mask(lay))
        else:
            m.addAction(_("Delete mask")).triggered.connect(lambda: self.remove_mask(lay))
        if lay["kind"] == "filter":
            m.addSeparator()
            for _txt, val in self.scope_choices(lay):
                a = m.addAction(_("Applies to everything below") if val is None else
                                _('Applies to "{name}" only').format(name=layer_name(self.layer(val))))
                a.setCheckable(True)
                a.setChecked(lay.get("target") == val)
                a.triggered.connect(lambda _c=False, t=val: self.set_target(lay, t) if lay.get("target") != t else None)
            m.addSeparator()
        if lay["kind"] == "workflow" and lay["part"] is None:
            p = self.path(lay["pid"])
            if p is not None and self.s.slider_state(p, "stars_number")[0]:
                m.addAction(_("Split stars and starless")).triggered.connect(lambda: self.split_stars(lay))
            m.addSeparator()
            m.addAction(_("Show workflow  →")).triggered.connect(lambda: self.goResults.emit(lay["pid"]))
        m.addAction(_("Rename")).triggered.connect(lambda: self.rename(lay))
        if not is_pixel(lay) or sum(map(is_pixel, self.layers)) > 1:
            m.addSeparator()
            m.addAction(_("Delete")).triggered.connect(lambda: (self.select(lay["id"], "layer"), self.delete()))
        self._menu = m                                   # (the tests and the screenshots read it)
        m.exec(gpos)

    def rename(self, lay, new: str | None = None):
        old = layer_name(lay)
        if new is None:
            from .dialogs import ask_name
            new = ask_name(self.window(), _("Rename layer"), old)
        if new and new != old:
            self.push()
            lay["name"] = new
            self.rebuild()

    # ------------------------------------------------------------ layer actions
    def add_image_layer(self, name: str, source, key: str) -> dict:
        """A stage's image as a layer, on top: stretched automatically, blendable, maskable (see PostPage3). Its name
        is shown translated where the catalogue has it (Inspect gives it in English); an image Inspect showed with a
        neutral sky (``source.neutral``: still in the camera's colours) has its channels stretched apart, so that its
        sky is neutral here too."""
        self.worker.set_image(key, source)
        self._missing_images.discard(key)
        self.push()
        lay = new_image_layer(key, name)
        if getattr(source, "neutral", False):
            lay["linked"] = False
        self.layers.insert(0, lay)
        self.layers = normalize(self.layers)
        self.sel = (lay["id"], "layer")
        self.part_sel = None
        self.rebuild()
        self.request(final_only=True)
        return lay

    def _insert_above_sel(self, lay):
        i = self.index(self.sel[0]) if self.sel else 0
        self.layers.insert(i, lay)
        self.layers = normalize(self.layers)
        self.sel = (lay["id"], "layer")
        self.part_sel = None

    def add_filter(self, key):
        """A new filter is nested in the chosen pixel layer, on top of its children (or goes where the chosen filter
        is: in its layer, or above it on everything below)."""
        cur = self.layer(self.sel[0]) if self.sel else None
        target = None
        if cur is not None:
            target = cur["id"] if is_pixel(cur) else cur.get("target")
        self.push()
        fl = new_filter_layer(key, target)
        if target is not None:                          # on top of that layer's own filters
            group = [lay for lay in self.layers if lay.get("target") == target]
            self.sel = ((group[0] if group else self.layer(target))["id"], "layer")
        self._insert_above_sel(fl)
        self.rebuild()
        self.request(final_only=True)
        return fl

    def set_target(self, fl, target):
        """Nest a filter in a pixel layer (on top of its children), or take it out: then it is the row just above that
        layer and applies to everything below."""
        if target is None:
            old = fl.get("target")
            t = [x["id"] for x, _kids in tree(self.layers)]
            idx = t.index(old) if old in t else t.index(fl["id"])
        else:
            idx = 0
        self._collapsed.discard(target)
        self.move(fl["id"], target, idx)

    @staticmethod
    def _copy(lay):
        d = snapshot([lay])[0]
        d["id"] = new_id()
        if d["mask"]:
            for part in d["mask"]["parts"]:
                part["id"] = new_id()
        return d

    def duplicate(self):
        """A copy above; a pixel layer comes with its own filters."""
        lay = self.layer(self.sel[0]) if self.sel else None
        if lay is None:
            return
        self.push()
        d = self._copy(lay)
        d["name"] = lay["name"] + COPY
        d["visible"] = True
        top = self.index(lay["id"])
        if is_pixel(lay):
            own = [x for x in self.layers if x.get("target") == lay["id"]]
            copies = []
            for f in own:
                c = self._copy(f)
                c["target"] = d["id"]
                copies.append(c)
            if own:
                top = self.index(own[0]["id"])
            self.layers[top:top] = copies + [d]
        else:
            self.layers.insert(top, d)
        self.layers = normalize(self.layers)
        self.sel = (d["id"], "layer")
        self.part_sel = None
        self.rebuild()
        self.request(final_only=True)

    def delete(self):
        """A pixel layer goes with its own filters; the last pixel layer stays."""
        lay = self.layer(self.sel[0]) if self.sel else None
        others = [x for x in self.layers if x["id"] != lay["id"] and x.get("target") != lay["id"]] if lay else []
        if lay is None or not any(is_pixel(x) for x in others) and is_pixel(lay):
            return
        self.push()
        i = self.index(lay["id"])
        self.layers = normalize(others)
        L = self.layers
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
        base = snapshot([lay])[0]
        fond = dict(base, id=new_id(), part="starless", name=f"{lay['name']} · {STARLESS}")
        stars = dict(snapshot([lay])[0], id=new_id(), part="stars", name=f"{lay['name']} · {STARS}", mode="Screen",
                     mask=None)
        if fond["mask"]:
            for part in fond["mask"]["parts"]:
                part["id"] = new_id()
        self.layers[i:i + 1] = [stars, fond]
        for f in self.layers:                           # its own filters stay on the background
            if f.get("target") == lay["id"]:
                f["target"] = fond["id"]
        self.layers = normalize(self.layers)
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
        self.pbox.addWidget(label(layer_name(lay), "h2"))
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
            r = QVBoxLayout()
            r.setSpacing(3)
            for txt, val in self.scope_choices(lay):
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
        if lay["kind"] == "image":
            self.image_props(lay)
            return
        p = self.path(lay["pid"])
        if lay["part"]:
            self.pbox.addWidget(label(_("Stars only, in Screen mode above the starless image.")
                                      if lay["part"] == "stars" else _("Starless image."), "small"))
        bases = self.bases_of(p)
        if bases:
            r = QHBoxLayout()
            r.addWidget(label(_("Rendering"), "dim"))
            c = QComboBox()
            c.addItem(_("Suggested"), None)
            for name in bases:
                c.addItem(name, name)
            c.setCurrentIndex(max(0, c.findData(lay.get("base"))))
            c.currentIndexChanged.connect(lambda i, c=c: self._set(lay, "base", c.itemData(i)))
            r.addWidget(c, 1)
            self.pbox.addLayout(r)
        group = None
        for sd in self.s.slider_infos(p) if p is not None else []:
            if sd.group != group:
                group = sd.group
                self._group(_(group))
            pos = lay["sliders"].get(sd.key, 2.0)
            sl = ContSlider(_(sd.name), 0, 4, pos, 2, ends=[_(e) for e in sd.ends] if sd.ends else None,
                            show=lambda x: _("suggested") if abs(x - 2) < 0.02 else dec(f"{x - 2:+.1f}"))
            ok, why, _need = self.s.slider_state(p, sd.key)         # the model answers in the language of the screen
            sl.set_available(ok, why)
            self._wire(sl, lambda v, k=sd.key: self._slider(lay, k, v))
            self.pbox.addWidget(sl)

    def image_props(self, lay):
        """An image layer: where it comes from, its automatic stretch (linked or each channel) and its three
        sliders."""
        ok = self.worker_has_image(lay)
        self.pbox.addWidget(label(_("A stage's image, stretched automatically.") if ok else
                                  _("This image is no longer in the project's results."), "small" if ok else "warn"))
        r = QHBoxLayout()
        r.addWidget(label(_("Channels"), "dim"))
        c = QComboBox()
        c.addItem(_("Linked"), True)
        c.addItem(_("Each channel"), False)
        c.setToolTip(_("Each channel: every color stretched apart, which neutralizes the background"))
        c.setCurrentIndex(0 if lay.get("linked", True) else 1)
        c.currentIndexChanged.connect(lambda i, c=c: self._set(lay, "linked", c.itemData(i)))
        r.addWidget(c, 1)
        self.pbox.addLayout(r)
        self._group(_("Stretch"))
        for key, name, ends in IMAGE_SLIDERS:
            pos = lay["sliders"].get(key, 2.0)
            sl = ContSlider(_(name), 0, 4, pos, 2, ends=[_(e) for e in ends],
                            show=lambda x: _("suggested") if abs(x - 2) < 0.02 else dec(f"{x - 2:+.1f}"))
            self._wire(sl, lambda v, k=key: self._slider(lay, k, v))
            self.pbox.addWidget(sl)

    def worker_has_image(self, lay) -> bool:
        """Whether an image layer's pixels are there (the last image the worker drew had it; until then, yes)."""
        return lay["key"] not in self._missing_images

    def bases_of(self, p) -> list[str]:
        """The named renderings of the project (faithful, artistic…), when it has some."""
        rb = getattr(getattr(self.s, "p", None), "render_bases", None) or {}
        return list(rb)

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
                state["before"] = snapshot(self.layers)
            apply(v)
            self.request()

        def done(v):
            if state["before"] is None:
                state["before"] = snapshot(self.layers)
            apply(v)
            self.push(state["before"])
            state["before"] = None
            self.request(final_only=True)
            QTimer.singleShot(0, self._refresh_rows)
        sl.moved.connect(moved)
        sl.done.connect(done)

    def sub_text(self, lay) -> str:
        """Under a layer's name: mode and opacity; for a filter of its own row, that it applies to everything below (a
        nested one says it by its place)."""
        bits = []
        if lay["kind"] == "filter" and not lay.get("target"):
            bits.append(_("on everything below"))
        bits.append(f"{_(lay['mode'])} · {with_unit(format(lay['opacity'], '.0f'), '%')}")
        if lay["kind"] == "workflow" and self.is_preview(lay["pid"]):
            bits.append(_("preview"))
        if lay["kind"] == "image":
            bits.append(_("image") if self.worker_has_image(lay) else _("not found"))
        if is_pixel(lay) and lay["id"] in self._collapsed:
            n = len(children(self.layers, lay["id"]))
            bits.append(plural(n, "{n} filter", "{n} filters"))
        return " · ".join(bits)

    def is_preview(self, pid) -> bool:
        """The workflow's image is still an approximation: its output is Outdated and the layer shows the earlier one."""
        p = self.path(pid)
        return p is not None and (bool(getattr(p, "provisional", False)) or (p.rendered and not p.computed))

    def _refresh_rows(self):
        """The row texts (mode, opacity) without rebuilding the properties under the mouse."""
        for row in self.rows:
            row.sub.setText(self.sub_text(row.lay))

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
        top.addWidget(label(_('Mask of "{name}"').format(name=layer_name(lay)), "h2"), 1)
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
            a = am.addAction(_(name))
            a.triggered.connect(lambda _c=False, s=src: self.add_part(lay, s))
            if src == "stars" and not self.has_stars():
                a.setEnabled(False)
                a.setToolTip(_(NEEDS_STARS))
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
        if self.part_sel and any(p["id"] == self.part_sel and p["src"] == "brush" for p in m["parts"]):
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
        default = new_part(src)

        def sl(key, name, lo, hi, unit=""):
            s = ContSlider(_(name), lo, hi, part[key], default[key], unit=unit, step=1)
            self._wire(s, lambda val: part.__setitem__(key, val))
            v.addWidget(s)

        for key, name, lo, hi, unit in PART_SETTINGS.get(src, []):
            if src == "gradient" and key == "position":                 # the direction comes first
                r = QHBoxLayout()
                r.addWidget(QLabel(_("Direction")))
                c = QComboBox()
                for d in DIRECTIONS:
                    c.addItem(_(d), d)
                c.setCurrentIndex(DIRECTIONS.index(part["direction"]))
                c.currentIndexChanged.connect(lambda i, c=c: self._part_set(part, "direction", c.itemData(i)))
                r.addWidget(c, 1)
                v.addLayout(r)
            sl(key, name, lo, hi, unit)
        if src == "brush":
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
        f.mousePressEvent = lambda e: self._edit_part(part) if part["src"] == "brush" else None
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
        part = next((p for p in lay["mask"]["parts"] if p["id"] == self.part_sel and p["src"] == "brush"), None)
        if part:
            self._brush_mode(part, not part["paint"])

    def brush_size(self, k):
        lay = self.layer(self.sel[0]) if self.sel else None
        part = next((p for p in (lay["mask"]["parts"] if lay and lay["mask"] else [])
                     if p["id"] == self.part_sel and p["src"] == "brush"), None)
        if part:
            part["size"] = float(np.clip(part["size"] * k, 4, 400))
            self.viewer.brush_r = part["size"] / 2
            self.build_props()

    def add_part(self, lay, src):
        self.push()
        part = new_part(src)
        lay["mask"]["parts"].append(part)
        self.part_sel = part["id"] if src == "brush" else None
        self.build_props()
        self.request(final_only=True)
        return part

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
        self.request(final_only=True)

    def _stroke(self, x, y, first):
        lay = self.layer(self.sel[0]) if self.sel else None
        part = next((p for p in (lay["mask"]["parts"] if lay and lay["mask"] else []) if p["id"] == self.part_sel),
                    None)
        if part is None or part["src"] != "brush":
            return
        if first:
            self._stroke_before = snapshot(self.layers)
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
        if on:
            self._send_before()
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
        if self._notes:
            note = "; ".join(_(n) for n in self._notes)
            t = note + (" · " + t if t else "")
        self.hint.setText(t)

    # ------------------------------------------------------------ export
    def export_dialog(self, parent=None):
        """The Export dialog on the composition as it is now (File > Export image…, or Export… in the bar)."""
        from .exportdlg import ExportDialog
        d = ExportDialog(self, parent or self.window())
        d.exec()
        d.deleteLater()


class PostPage3(QWidget):
    """The Post, or what is missing for it."""
    goWorkflow = Signal()

    def __init__(self, session, parent=None, *, worker: RenderWorker | None = None, persist: bool = True):
        super().__init__(parent)
        self.setObjectName("root")
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
        self.view = PostView(session, worker=worker, persist=persist)
        self.pages.addWidget(self.view)
        session.pathsChanged.connect(self.update_page)
        self.update_page()

    def update_page(self):
        shown = any(p.rendered for p in self.s.paths) or any(lay["kind"] == "image" for lay in self.view.layers)
        self.pages.setCurrentIndex(1 if shown else 0)

    def add_image_layer(self, name: str, source, key: str) -> dict:
        """Inspect sends a stage's image to Post: a stack, a gradient-removed image, a denoised image. It becomes a
        normal layer on top of the others: stretched automatically (Sky, Contrast, Saturation; channels linked or each
        apart), with a blend mode, an opacity, a mask and filters of its own.

        ``source``: the linear image - an array (H, W), (H, W, 3) or (3, H, W), a file (.npy, .fits, .tif, .png) or a
        function returning one (called in the render worker, never in the GUI thread); a function with ``neutral =
        True`` (``services/inspect.post_source``) is an image still in the camera's colours: its channels are
        stretched apart. ``key``: the stage result it
        is, saved in the project instead of the pixels; when the project is opened again the image is found from it:
        "<stage>/<store key>[/<array>]" is read from the project's store (``store/<stage>/<store key>/``, the array's
        name with or without .npy; without it, the stage's main image), any other key through ``image_resolver``.
        Returns the new layer (selected)."""
        lay = self.view.add_image_layer(name, source, key)
        self.update_page()
        return lay

    def set_image_resolver(self, fn):
        """key -> source (as ``add_image_layer``'s) for the image layers of a composition opened again whose key is not
        a store path; None when it is not there. Called in the render worker."""
        self.view.worker.set_image_resolver(fn)
        if any(lay["kind"] == "image" for lay in self.view.layers):
            self.view.request(final_only=True)           # (a first image may have been drawn without them)

    def refresh(self):
        self.update_page()
        if self.pages.currentIndex() == 1:
            self.view.worker.refresh()
            self.view.refresh()

    def export(self, parent=None):
        """File > Export image…"""
        if self.pages.currentIndex() == 1:
            self.view.export_dialog(parent)

    def shutdown(self):
        self.view.shutdown()
