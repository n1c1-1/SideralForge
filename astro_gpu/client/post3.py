"""The Post of the client: the image at the centre; on the right the layers, as in Affinity Photo, and the properties of
the selected layer or of its mask.

The layers panel follows Affinity Photo's model. A pixel layer (a workflow's output, one of its Stars block's images,
or a stage's image sent from Inspect) shows its own filters as children, indented under it: they apply to it alone and
move with it. A filter that applies to everything below is a row of its own, like a layer, above what it affects. Drag a
filter onto a layer to nest it, between two rows to make it apply to everything below (an indented line: it joins that
layer's children); the line, the frame and a short label say where it goes. Read from the bottom up, the panel is the
order of the composition.

A row: eye, the ▾ of a layer's children, thumbnail, mask thumbnail, name, blend mode and opacity. A click on the
thumbnail shows the layer's settings; a click on the mask thumbnail shows its mask, made of parts as in Lightroom, with a
red overlay on the image.

Nothing here renders. Every image comes from the render worker (services/render.py): a draft while a slider or the
brush moves (an eighth of the image on the CPU, a quarter on a GPU), on release the exact quarter first on the CPU then
the half resolution (the screen), and the visible region at full resolution when zoomed in. A newer request replaces an
older one. The layers are saved in the project (``project.post["layers"]``) through the model, which the graph edits
too (the Stars block's images sent to Post, the Stars and background blocks and their settings): a change made there
comes back here (``postChanged``).

A workflow with the Stars block comes in as a Stars and background layer (its starless image and its stars, each
stretched, the stars put back in Screen: render/stars.py), with the star sliders; the others as their Output block's
image.

    page = PostPage3(model)              # model: the ProjectModel
    page.goWorkflow                      # "Go to graph →"
    page.view.goResults(pid)             # '"X": not computed yet →' and "Show workflow →": Inspect on that Output
    page.export(parent)                  # File > Export image… and the Export… button
    page.add_image_layer(name, source, key)   # Inspect: a stage's image as a layer

Classes: ``PostPage3`` holds the ``PostView`` (the viewer, the layers panel of ``LayerRow`` rows, the properties with
their ``ContSlider`` sliders and the Curves editor); ``PostSource`` and ``BeforeSource`` feed the viewer.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
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
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..render.sliders import STARS as STARS_GROUP
from ..render.sliders import group_title
from . import prefs
from .curveseditor import CurvesEditor, FunctionGraph, caption
from .i18n import _, language, log_detail, maybe, plain_error, plural
from .icons import draw as draw_icon
from .icons import icon_only, pixmap, set_icon
from .imageview import ZOOM_PRESETS, ZOOM_STEP
from .layout import Dock, Sections
from .model import POST_PART_NAMES
from .postengine import (
    BLENDS,
    DIRECTIONS,
    FAMILIES,
    FILTERS,
    IMAGE_SLIDERS,
    MASK_SOURCES,
    MAX_LEVEL,
    MODES,
    OLD_NAMES,
    OPS,
    PART_SETTINGS,
    SUBS,
    THUMB,
    children,
    is_group,
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
from .widgets import ACCENT, WARN, ElidedLabel, clear, hline, label

NL = "\n"                                                   # a line break in a tooltip

FAMILY_COL = {"Stretch": "#c9a35e", "Contrast": "#d98a4e", "Color": "#c46fa8", "Detail": "#5ea1ff",
              "Noise": "#62c2b1", "Combine": "#e06c6c"}
TW, TH = THUMB                                           # thumbnail size
COPY = " (copy)"                                         # a piece of the layer names, translated where shown
#: what the Stars block's images are, under their name in the properties
PART_NOTES = {"starless": "The Stars/background block's starless image, stretched like the whole image.",
              "stars": "The Stars/background block's stars alone, on black: in Screen mode above the background.",
              "combined": "The starless image and the stars, recombined. The same settings as its block in the "
                          "graph."}
REGION_MAX = 12_000_000                                  # pixels of the region rendered at full resolution
REGION_WAIT_MS = 150                                     # after the view stops moving
INDENT = 24                                              # a child row's indent (px)
QUARTER_SCALE = 0.26                                     # zoomed out to this, the quarter-resolution image fills the screen
TOGGLE_W = 16                                            # the ▾ / ▸ of a layer's children


def tr_name(n: str, image: bool = False) -> str:
    """A layer's name as shown: the pieces the program built it from (a filter's name, the Stars block's images,
    "(copy)") translated; the name of a workflow, or a name typed by the user, as it is (the graph shows it so). ``image``: an
    image layer, named by Inspect in English (``tune.TuneView._send``): every piece the catalogue has is translated
    (a file name stays as it is)."""
    k = 0
    while n.endswith(COPY):
        n, k = n[:-len(COPY)], k + 1
    if image:
        return _pieces(n, maybe) + _(COPY) * k
    known = {f["name"] for f in FILTERS.values()} | set(OLD_NAMES) | set(POST_PART_NAMES.values())

    def piece(b):                                 # "Combine layers 2": a numbered copy of a Split image's name
        m = re.fullmatch(r"(.+) (\d+)", b)
        if b not in known and m and m.group(1) in POST_PART_NAMES.values():
            return _(m.group(1)) + " " + m.group(2)
        return _(b) if b in known else b
    return " · ".join(piece(b) for b in n.split(" · ")) + _(COPY) * k


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


def offset_text(x: float) -> str:
    """A rendering slider's value, as Lightroom shows it: its offset from the suggested setting (position 2), 0 there,
    with a true minus sign."""
    return "0" if abs(x - 2) < 0.02 else dec(f"{x - 2:+.1f}").replace("-", "−")


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
    on the slider or on its name: back to the default (Lightroom's, issue #16)."""
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
        self.setToolTip(_("Double-click: default value"))
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
        self.name.mouseDoubleClickEvent = lambda e: self.reset()
        self.s.installEventFilter(self)

    def reset(self):
        """Back to the default value (a double-click)."""
        if not self.s.isEnabled():
            return
        self.set_value(self.default)
        self.moved.emit(self.default)
        self.done.emit(self.default)

    def eventFilter(self, obj, e):                                   # noqa: N802 - Qt's name
        if obj is self.s and e.type() == QEvent.MouseButtonDblClick and e.button() == Qt.LeftButton:
            self.reset()
            return True
        return super().eventFilter(obj, e)

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
#: the right panel's width: by default, and the limits of the user's drag
PANEL_W, PANEL_MIN, PANEL_MAX = 400, 320, 760


class EyeBox(QCheckBox):
    """A layer's visibility, drawn as an eye: a check box underneath (its clicks, keys and signals as any check box's)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setCursor(Qt.PointingHandCursor)

    def hitButton(self, pos):                             # noqa: N802 - Qt's name
        return self.rect().contains(pos)

    def paintEvent(self, e):                              # noqa: N802
        p = QPainter(self)
        on = self.isChecked()
        p.drawPixmap(2, 2, pixmap("eye" if on else "eye_off", 16, "normal" if on else "dim", self.devicePixelRatioF()))
        p.end()


class LayerRow(QFrame):
    """A row of the layers panel. ``depth`` 1: a filter nested in the pixel layer above it (indented, with a guide
    line), or one of a Stars and background layer's two images (``kind`` "sub", issue #3); 2: a filter of one of those
    images; ``kids``: how many children a pixel layer has (its ▾ / ▸)."""

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
        compact = lay["kind"] in ("filter", "sub")      # issue #5: the filters, many, on one short line each
        h.setContentsMargins(4, 1 if compact else 3, 6, 1 if compact else 3)
        h.setSpacing(6)
        eye = EyeBox()
        eye.setChecked(lay["visible"])
        eye.setToolTip(_("Show / hide layer"))
        if lay["kind"] == "sub":
            eye.setToolTip(_("Show / hide this image in its Combine layers layer"))
            eye.toggled.connect(lambda v: panel.set_shown(lay["id"], lay["sub"], v))
        else:
            eye.toggled.connect(lambda v: panel.set_visible(lay["id"], v))
        h.addWidget(eye)
        self.toggle = None
        if depth:                                       # a child: indented, a guide line down from its layer
            h.addSpacing(INDENT * (depth - 1))
            guide = QFrame()
            guide.setFixedWidth(2)
            guide.setStyleSheet("background: #4a5366;")
            h.addSpacing(TOGGLE_W // 2 - 1)
            h.addWidget(guide)
            h.addSpacing(INDENT - TOGGLE_W // 2 - 1)
        if kids:                                        # issue #21: down expanded, right collapsed; none without
            key = (lay["id"], lay["sub"]) if lay["kind"] == "sub" else lay["id"]
            self.toggle = QToolButton()
            icon_only(self.toggle, "chevron_right" if collapsed else "chevron_down", _("Show / hide what is under it"), 16, "dim")
            self.toggle.setAutoRaise(True)
            self.toggle.setFixedSize(TOGGLE_W, TOGGLE_W)
            self.toggle.setStyleSheet("QToolButton { border: none; padding: 0; }")
            self.toggle.clicked.connect(lambda: panel.toggle_children(key))
            h.addWidget(self.toggle)
        else:
            h.addSpacing(TOGGLE_W)
        self.t = QLabel()
        if compact:
            thumb = thumb.scaled(TW * 2 // 3, TH * 2 // 3, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.t.setFixedSize(TW * 2 // 3 + 4, TH * 2 // 3 + 4)
        else:
            self.t.setFixedSize(TW + 4, TH + 4)
        self.t.setPixmap(thumb)
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
        self.n = ElidedLabel(layer_name(lay), mode=Qt.ElideRight)     # a long name never widens the row
        self.n.setMinimumWidth(40)
        self.n.setStyleSheet("font-weight: 600;" + ("" if lay["visible"] else " color: #8b93a1;"))
        v.addWidget(self.n)
        self.sub = label(panel.sub_text(lay), "small")
        v.addWidget(self.sub)
        # a filter at its usual blend (Normal, 100 %, no mask) says nothing more: one line
        self.sub.setVisible(lay["kind"] != "sub" and not (compact and lay.get("target") and lay["mode"] == "Normal"
                                                          and lay["opacity"] >= 100 and mthumb is None))
        h.addLayout(v, 1)
        if lay["kind"] == "sub":
            self.setToolTip(_("The filters nested here apply to this image only, before the stars are put back on "
                              "the starless image"))
        elif lay.get("sub"):
            self.setToolTip(_('Applies to "{name}" only').format(name=_(POST_PART_NAMES[lay["sub"]])))
        elif depth:
            tgt = panel.layer(lay.get("target"))
            self.setToolTip(_('Applies to "{name}" only').format(name=layer_name(tgt)) if tgt else "")
        elif lay["kind"] == "filter":
            self.setToolTip(_("On all below"))

    def set_drop(self, on: bool, text: str = ""):
        """The frame of a drop that nests the dragged filter in this layer; ``text`` (what the drop does) shows in place of
        the row's mode and opacity while it lasts."""
        if bool(self.property("drop")) == on and (not on or self.sub.text() == text):
            return
        self.setProperty("drop", on)
        if on:
            self.sub.setVisible(True)                   # a compact row says what the drop does too
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
        elif e.button() == Qt.RightButton and self.lay["kind"] != "sub":
            self.panel.layer_menu(self.lay, e.globalPosition().toPoint())

    def mouseMoveEvent(self, e):
        if self.lay["kind"] == "sub":                    # a sub-layer stays in its layer
            return
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
            what = self.lay["sub"] if self.lay["kind"] == "sub" else self._target(press)
            QTimer.singleShot(0, lambda: self.panel.select(self.lay["id"], what))

    def mouseDoubleClickEvent(self, e):
        if self.lay["kind"] != "sub":
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
#: a step of Post's Undo that is a change of a workflow (its Output's channels): the workflows' Undo takes it back
MODEL_STEP = object()


class PostView(QWidget):
    is_post_page = True                                   # its changes go to its own Undo, not the graph's (2.1)
    goProcessing = Signal(int, str)
    goResults = Signal(int)
    panelChosen = Signal(str)                             # issue #44: the panel pinned or folded by the user
    rendered = Signal(object)                             # a Result was shown (tests, the status line)
    toasted = Signal(str)                                 # what a change did beyond Post (the window's status line)
    layerAsked = Signal(int)                              # Open in Post of a layer (the page lists it when not computed)
    openImages = Signal()                                 # images asked as layers
    alignAsked = Signal()                                 # Aligner: the Align images block, its Inspect
    runAsked = Signal()                                   # an image's kind changed: it is computed again

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
        self.heal_size = 80.0                           # Spot healing: the next spot's diameter (px)
        self._heal_stroke = None                        # the stroke being painted: (its points, the layers before)
        self._heal_src = None                           # Alt+click: the source set by hand (x, y), then its offset
        self._heal_off = None                           # (dx, dy) of the strokes since, as Photoshop's Aligned
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
        self._restoring = False                          # a step of the workflows' Undo being undone/redone here
        self._hists: dict[int, dict] = {}                # filter id -> the histograms of its input (the last render)
        self.curves_editor: CurvesEditor | None = None   # the Curves panel shown, or the graph of a stretch
        self._curve_channel: dict[int, str] = {}         # filter id -> the channel its Curves panel showed last
        self.function_graph: FunctionGraph | None = None
        self.first_image = False
        self.sync_paths(initial=True)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        # the right panel, as the docks of the usual image editors: its width is the user's (its left edge dragged,
        # kept in the settings); the double chevron at the top of the panel folds it into a thin strip at the window's
        # edge, whose chevron opens it again (kept too)
        self.split = QSplitter(Qt.Horizontal)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self._build_centre())
        self.right = self._build_panel()
        self.split.addWidget(self.right)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        self.split.setSizes([10_000, int(prefs.get("post_panel_width") or PANEL_W)])
        self.split.splitterMoved.connect(self._panel_moved)
        root.addWidget(self.split)
        # issue #44: docked, or the strip of icons (a click opens it over the viewer) or hidden, as the other pages
        self.dock = Dock(self, self.right, self._dock_panel, lambda: self.right.setParent(self),
                         lambda: [(_("Layers"), self._show_layers, "layers")]
                         + (self.folds.items() or [(_("Properties"), self._show_props, "adjust")]),
                         _("Layers"), role="section")
        root.addWidget(self.dock.strip)
        self.strip = self.dock.strip
        self.dock.chosen.connect(lambda st: self.panelChosen.emit(st))
        self.fold_b.clicked.connect(self.dock.fold)
        self._show_panel(prefs.get("post_panel_shown") is not False)
        self._connect(own_worker=worker is None)
        if self.layers:
            self.sel = (self.layers[-1]["id"], "layer")
        self.build_filter_menu()
        self.rebuild()
        self._send_final()

    def _build_centre(self) -> QWidget:
        """The centre: the bar of view tools, the viewer and the hint line under it."""
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(10, 6, 10, 6)
        self.top_bar = bar
        self.b_fit = icon_only(QPushButton(), "fit", _("Fit (F)"), 16)
        self.b_100 = icon_only(QPushButton(), "actual_size", _("Actual size (1)"), 16)
        self.zoom_lab = QPushButton(with_unit("100", "%"))   # the zoom, and its presets (issue #44: as Inspect's viewer)
        self.zoom_lab.setToolTip(_("Zoom (wheel, + and −)"))
        self.zoom_menu = QMenu(self.zoom_lab)
        self.zoom_menu.addAction(_("Fit"), lambda: self.viewer.fit())
        for z in ZOOM_PRESETS:
            self.zoom_menu.addAction(with_unit(f"{z * 100:g}", "%"), lambda zz=z: self.viewer.zoom_to(zz))
        self.zoom_lab.setMenu(self.zoom_menu)
        self.b_cmp = QPushButton(_("Before / after"))
        self.b_cmp.setCheckable(True)
        self.b_cmp.setObjectName("toggle")
        self.b_cmp.setToolTip(_("Before: the bottom layer, suggested settings (C)"))
        self.b_mask = QPushButton(_("Show mask"))
        self.b_mask.setCheckable(True)
        self.b_mask.setObjectName("toggle")
        self.b_mask.setToolTip(_("Mask of the selected layer, in gray (M)"))
        # the annotations of the astrometric solution over the image (and, if wanted, burned into an export)
        self.ann_show = {"grid": False, "objects": False}
        self._ann = None                                 # annotate.Annotations of the canvas shown
        self._ann_key = None
        self._ann_token = None
        self._canvas = None                              # the last result's canvas (size and origin on the grid)
        self.b_grid = QPushButton(_("Sky grid"))
        self.b_grid.setToolTip(_("The RA / Dec grid of the astrometric solution"))
        self.b_names = QPushButton(_("Object names"))
        self.b_names.setToolTip(_("The known deep-sky objects of the field (Messier, NGC, IC, Sharpless and their "
                                  "common names), outlined at their size. Export… can burn them into the picture."))
        for b in (self.b_grid, self.b_names):
            b.setCheckable(True)
            b.setObjectName("toggle")
        self.b_undo = icon_only(QPushButton(), "undo", _("Undo (Ctrl+Z)"), 16)
        self.b_redo = icon_only(QPushButton(), "redo", _("Redo (Ctrl+Y)"), 16)
        self.b_export = set_icon(QPushButton(_("Export…")), "export", 16)
        for w in (self.b_fit, self.b_100, self.zoom_lab):
            bar.addWidget(w)
        bar.addStretch(1)
        for w in (self.b_grid, self.b_names, self.b_cmp, self.b_mask, self.b_undo, self.b_redo, self.b_export):
            bar.addWidget(w)
        mv.addLayout(bar)
        self.src = PostSource(self)
        self.before = BeforeSource(self)
        self.viewer = Viewer()
        self.viewer.overlays.append(self._overlay)
        self.viewer.overlays.append(self._ann_overlay)
        self.viewer.overlays.append(self._heal_overlay)
        self.viewer.banner = _("Rendering…")
        mv.addWidget(self.viewer, 1)
        self.hint = label("", "small")
        self.hint.setContentsMargins(10, 3, 10, 5)
        mv.addWidget(self.hint)
        return mid

    def _build_panel(self) -> QWidget:
        """The right side: the layers, their buttons, then the properties of the selection."""
        right = QWidget()
        right.setObjectName("panel")
        right.setMinimumWidth(PANEL_MIN)
        right.setMaximumWidth(PANEL_MAX)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(10, 10, 10, 8)
        rv.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label(_("Layers"), "h2"))
        head.addStretch(1)
        fold = QToolButton()
        icon_only(fold, "fold_right", _("Fold the panel"), 16, "dim")
        fold.setAutoRaise(True)
        fold.setFixedSize(20, 18)
        fold.setStyleSheet("QToolButton { border: none; padding: 0; } "
                           "QToolButton:hover { background: #2b3446; border-radius: 3px; }")
        self.fold_b = fold
        # while a row is dragged between two others: what the drop does, beside the panel's title (a drop onto a layer
        # says it in that layer's row)
        self.drop_text = QLabel()
        self.drop_text.setStyleSheet(f"color: {ACCENT}; font-weight: 600;")
        self.drop_text.hide()
        head.addWidget(self.drop_text)
        head.addWidget(fold)
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
        # the layers' buttons, as the reference editors': icons with their full names in the tooltips
        self.b_open = icon_only(QPushButton(), "block_image", _("Open images as layers"), 16, box=28)
        self.b_open.clicked.connect(self.openImages.emit)
        self.b_align = icon_only(QPushButton(), "block_alignment", _("Align the image layers"), 16, box=28)
        self.b_align.clicked.connect(self.alignAsked.emit)
        self.b_filter = icon_only(QPushButton(), "add", _("Add a filter layer above the selected layer"), 16, box=28)
        self.filter_menu = QMenu(self.b_filter)
        self.b_filter.setMenu(self.filter_menu)
        self.b_dup = icon_only(QPushButton(), "duplicate", _("Duplicate (Ctrl+J)"), 16, box=28)
        self.b_maskadd = icon_only(QPushButton(), "mask", _("Add a mask to the selected layer"), 16, box=28)
        self.b_del = icon_only(QPushButton(), "delete", _("Delete the selected layer"), 16, box=28)
        for b in (self.b_open, self.b_align, self.b_filter, self.b_dup, self.b_maskadd, self.b_del):
            btns.addWidget(b)
        btns.addStretch(1)
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
        self.psc = psc
        self.folds = Sections(self.pbox, psc)            # its groups fold, as the block panel's sections (issue #44)
        return right

    def _connect(self, own_worker: bool):
        """The buttons, the shortcuts, the timers and the signals of the worker and the model."""
        self.b_fit.clicked.connect(self.viewer.fit)
        self.b_100.clicked.connect(lambda: self.viewer.zoom_to(1.0))
        self.b_cmp.toggled.connect(self._compare)
        self.b_mask.toggled.connect(self._mask_view)
        self.b_grid.toggled.connect(lambda on: self.set_annotations(grid=on))
        self.b_names.toggled.connect(lambda on: self.set_annotations(objects=on))
        self.b_undo.clicked.connect(self.do_undo)
        self.b_redo.clicked.connect(self.do_redo)
        self.b_export.clicked.connect(lambda: self.export_dialog())
        self.b_dup.clicked.connect(self.duplicate)
        self.b_maskadd.clicked.connect(self.mask_button)
        self.b_del.clicked.connect(self.delete)
        self.viewer.viewChanged.connect(self._view_changed)
        self.viewer.stroke.connect(self._stroke)
        self.viewer.strokeEnd.connect(self._stroke_end)
        self.viewer.toolPress.connect(self._heal_press)
        self.viewer.toolMove.connect(self._heal_move)
        self.viewer.toolRelease.connect(self._heal_release)
        # Ctrl+Z / Ctrl+Y are the window's Edit > Undo / Redo, which come here while Post is shown (two shortcuts on
        # one key would cancel each other)
        for seq, fn in (("C", self.b_cmp.toggle), ("M", self.b_mask.toggle), ("F", self.viewer.fit),
                        ("Ctrl+0", self.viewer.fit), ("1", lambda: self.viewer.zoom_to(1.0)),
                        ("+", lambda: self.viewer.zoom_by(ZOOM_STEP)), ("=", lambda: self.viewer.zoom_by(ZOOM_STEP)),
                        ("Ctrl++", lambda: self.viewer.zoom_by(ZOOM_STEP)),
                        ("-", lambda: self.viewer.zoom_by(1 / ZOOM_STEP)),
                        ("Ctrl+-", lambda: self.viewer.zoom_by(1 / ZOOM_STEP)), ("X", self.toggle_paint),
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
        self._main_seq = 0                               # the newest draft or release request (the exact view's, Q1)
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
        if hasattr(self.s, "postChanged"):
            self.s.postChanged.connect(self._post_changed)
            self.s.post_shown = lambda: save_layers(self.layers)
            self.destroyed.connect(lambda *_a, s=self.s: setattr(s, "post_shown", None))
        self.destroyed.connect(self.worker.stop if own_worker else (lambda *_a: None))

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
        removed = getattr(self.s, "post_removed", lambda: set())()
        for p in paths:
            # every workflow's image comes, except one the user took out (issues #10, #52: it does not come back)
            stars = p.opts.get("etoiles") == "retirer"
            out = (p.pid, "combined" if stars else None) in removed and getattr(p, "end", None) is None
            channel = getattr(self.s, "feeds_fusion", None) is not None and self.s.feeds_fusion(p)   # #47: the fusion
            channel = channel or (getattr(self.s, "gives_post", None) is not None and not self.s.gives_post(p))
            if not out and p.rendered and p.pid not in known and not channel:   # with Stars: its Combine
                # a block's image sent to Post (a branch made for it) is shown at once, as the Split images are
                sent = getattr(p, "end", None) is not None or getattr(self.s, "is_sent_branch", lambda q: False)(p) or \
                    getattr(self.s, "is_plain_image", lambda q: False)(p)
                self.layers.insert(0, new_workflow_layer(
                    p.pid, p.name + (" · " + POST_PART_NAMES["combined"] if stars else ""), bool(sent),
                    part="combined" if stars else None))
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

    def _post_changed(self, source):
        """The graph changed the composition (a layer sent to Post, a Stars and background block, its settings):
        taken in as one undo step."""
        if source is self:
            return
        post = getattr(getattr(self.s, "p", None), "post", None) or {}
        text = json.dumps(post.get("layers") or [], sort_keys=True)
        # what is shown, not what was saved last: a branch removed takes its layer from the panel without a save, and
        # its Ctrl+Z puts back the very composition saved before (the layer with its settings, not a new one)
        if text == self._saved and text == json.dumps(save_layers(self.layers), sort_keys=True):
            return
        if not getattr(self, "_restoring", False):      # (a step of the workflows' Undo undone here: no step of its own)
            self.undo.append(snapshot(self.layers))
            self.redo.clear()
        old = {lay["id"] for lay in self.layers}
        self.layers = load_layers(post.get("layers"), {p.pid for p in self.s.paths})
        self._saved = json.dumps(save_layers(self.layers), sort_keys=True)
        added = [lay["id"] for lay in self.layers if lay["id"] not in old]
        if added:                                   # a layer sent from the graph: selected, as a pasted layer is
            self.sel = (added[0], "layer")
        elif self.sel and not self.layer(self.sel[0]):
            self.sel = (self.layers[0]["id"], "layer") if self.layers else None
        self.sync_paths()                           # the layers the workflows give and the composition lacks yet

    def select_layer(self, lid):
        """Show this layer's settings (the graph's Open in Post); not computed yet, the page's list shows it."""
        if self.layer(lid) is not None:
            self.select(lid, "layer")
        if lid is not None:
            self.layerAsked.emit(lid)

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
        if hasattr(self.s, "set_post_layers"):
            self.s.set_post_layers(data, self)
            return
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

    def _hist_ids(self) -> tuple:
        """The selected filter, when its panel shows the histogram of its input (Curves, the stretches)."""
        lay = self.layer(self.sel[0]) if self.sel and self.sel[1] == "layer" else None
        if lay is None or lay["kind"] != "filter" or not FILTERS[lay["filter"]].get("hist"):
            return ()
        return (lay["id"],)

    def _send_draft(self):
        want, overlay = self._selected_wants()
        self._main_seq = self.worker.request(Request(kind="draft", layers=snapshot(self.layers),
                                                     level=self.draft_level, want=want, overlay=overlay))

    def _exact_wanted(self, res) -> bool:
        """After the release's image (GUI exploration 4, Q1 and Q4): the fitted view made again from the full
        resolution, when the screen shows the image reduced (zoomed in, the region is exact). The reduced image is not
        the file reduced: a filter sees its noise already averaged (a strong Saturation, the 1 px detail sliders), and
        so does the stretch."""
        if res.seq != self._main_seq or self.viewer.scale >= 0.5:
            return False
        return res.kind == "half" or (res.kind == "refine" and self._half_due)

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
            self._main_seq = self.worker.request(Request(kind="refine", layers=snapshot(self.layers),
                                                         level=HALF_LEVEL + 1, want=want, thumbs=True, overlay=overlay,
                                                         hists=self._hist_ids()))
        else:
            # on the CPU the exact quarter comes first, then the half
            self._half_due = False
            self._main_seq = self.worker.request(Request(kind="half", layers=snapshot(self.layers), level=HALF_LEVEL,
                                                         want=want, thumbs=True, overlay=overlay, refine=cpu,
                                                         hists=self._hist_ids()))
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
        if res.kind == "exact":                                 # the release's image from the full resolution (Q1)
            if res.tag == self._main_seq and res.rgb is not None:     # nothing asked since its release
                self.src.img1, self.src.seq1 = res.rgb, res.seq
                self.src.version += 1
                self.viewer.refresh()
                self.rendered.emit(res)
            return
        if res.seq < self._last_seq.get(res.kind, 0):
            return                                              # an answer older than one already shown
        if res.kind != "thumbs":
            self._last_seq[res.kind] = res.seq
        if res.canvas is not None:
            self._canvas = res.canvas
            size = (res.canvas.W, res.canvas.H)
            if self.src.canvas != size:
                self.src.canvas = size
                self.viewer.set_source(self.src)
            if self.ann_show["grid"] or self.ann_show["objects"]:
                self._want_annotations()
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
        if res.hists:
            self._hists.update(res.hists)
            self._show_hist()
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
            if self._exact_wanted(res):
                self.worker.request(Request(kind="exact", layers=snapshot(self.layers), level=HALF_LEVEL,
                                            tag=res.seq))
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

    def _show_hist(self):
        """The histogram of the selected filter's input into its panel."""
        h = self._hists.get(self.sel[0]) if self.sel else None
        if h is None:
            return
        if self.curves_editor is not None:
            self.curves_editor.set_hist(h)
        if self.function_graph is not None:
            self.function_graph.hist = h
            self.function_graph.update()

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
            first = _("Image not computed: {why}").format(why=plain_error(text))
            if first != self.viewer.banner:                  # said once; the program's error in the Log only
                log_detail(self.s, text)
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

    # ------------------------------------------------------------ the annotations of the solution
    def set_annotations(self, grid: bool | None = None, objects: bool | None = None):
        """The sky grid and the named objects over the image, on or off (Export… proposes the same)."""
        if grid is not None:
            self.ann_show["grid"] = bool(grid)
        if objects is not None:
            self.ann_show["objects"] = bool(objects)
        for b, k in ((self.b_grid, "grid"), (self.b_names, "objects")):
            if b.isChecked() != self.ann_show[k]:
                b.blockSignals(True)
                b.setChecked(self.ann_show[k])
                b.blockSignals(False)
        if self.ann_show["grid"] or self.ann_show["objects"]:
            self._want_annotations()
        self.viewer.update()

    def _ann_pid(self):
        """The workflow whose solution the canvas uses: the bottom workflow layer (as the export does)."""
        return next((lay["pid"] for lay in reversed(self.layers) if lay["kind"] == "workflow"), None)

    def _want_annotations(self):
        c, pid = self._canvas, self._ann_pid()
        if c is None or pid is None:
            return
        key = (pid, c.W, c.H, tuple(c.origin))
        if key == self._ann_key:
            return
        try:
            from .services import inspect as si
            sv = si.service_of(self.s)
        except Exception:  # noqa: BLE001 - a session without the inspect service (tests): no annotations
            return
        if not getattr(self, "_ann_wired", False):
            sv.jobReady.connect(self._ann_ready)
            self._ann_wired = True
        self._ann_key = key
        self._ann_token = sv.request_job("post.sky", si.post_annotations, pid, c.W, c.H, tuple(c.origin))

    def _ann_ready(self, name: str, token: int, result):
        if name != "post.sky" or token != self._ann_token:
            return
        self._ann = result
        why = getattr(result, "why", "")
        n = len(getattr(result, "objects", None) or [])
        self.b_names.setToolTip(_("The known deep-sky objects of the field (Messier, NGC, IC, Sharpless and their "
                                  "common names), outlined at their size. Export… can burn them into the picture.")
                                + (NL + plural(n, "{n} named object", "{n} named objects") if n else "")
                                + (NL + _(why) if why else ""))
        self.viewer.update()

    def _ann_overlay(self, p: QPainter, v: Viewer):
        if self._ann is None or self.show_mask or not (self.ann_show["grid"] or self.ann_show["objects"]):
            return
        from .annotate import paint
        paint(p, self._ann, v, self.ann_show)

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
        can = getattr(self.s, "can_undo", None)          # with none of Post's, the workflows' (do_undo)
        self.b_undo.setEnabled(bool(self.undo) or bool(can and can()))
        can = getattr(self.s, "can_redo", None)
        self.b_redo.setEnabled(bool(self.redo) or bool(can and can()))

    def _set_workflow_param(self, p, stage, name, value):
        """A setting of the workflow shown here (its Output's channels): a step of Post's Undo too (2.4), undone by the
        workflows' Undo."""
        self.s.set_param(p, stage, name, value)
        self.undo.append(MODEL_STEP)
        self.redo.clear()
        self._undo_buttons()

    def do_undo(self) -> bool:
        """Post's last change undone; with none left, the workflows' (a "To Post" made in the graph, say): one Undo
        for the whole window. True when it was the workflows' step."""
        if self.undo and self.undo[-1] is not MODEL_STEP:
            self.redo.append(snapshot(self.layers))
            self.layers = self.undo.pop()
            self._restored()
            return False
        if self.undo:
            self.undo.pop()
        return self._model_step(getattr(self.s, "undo", None), self.redo)

    def do_redo(self) -> bool:
        if self.redo and self.redo[-1] is not MODEL_STEP:
            self.undo.append(snapshot(self.layers))
            self.layers = self.redo.pop()
            self._restored()
            return False
        if self.redo:
            self.redo.pop()
        return self._model_step(getattr(self.s, "redo", None), self.undo)

    def _model_step(self, fn, other: list) -> bool:
        """A step of the workflows' Undo taken back or redone from here: the composition it puts back is not a step
        of Post's."""
        if fn is None:
            return False
        self._restoring = True
        try:
            done = fn()
            if done:
                other.append(MODEL_STEP)
        finally:
            self._restoring = False
        self.rebuild()
        self._undo_buttons()
        return done

    def _restored(self):
        if self.sel and not self.layer(self.sel[0]):
            self.sel = (self.layers[-1]["id"], "layer") if self.layers else None
        self.rebuild()
        self.request(final_only=True)
        self._save()

    # ------------------------------------------------------------ the layers panel
    def thumb(self, lay) -> QPixmap:
        if lay["kind"] == "filter":
            pm = QPixmap(TW, TH)
            pm.fill(QColor("#1d2129"))
            qp = QPainter(pm)
            draw_icon(qp, QRectF((TW - 20) / 2, (TH - 20) / 2, 20, 20), "filter_layer", "dim", 20)
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
        """+ Filter: a menu by family."""
        self.filter_menu.clear()
        for fam in FAMILIES:
            sub = self.filter_menu.addMenu(_(fam))
            sub.setToolTipsVisible(True)
            for k, f in FILTERS.items():
                if f["family"] == fam and not f.get("hidden"):
                    a = sub.addAction(_(f["name"]))
                    a.setToolTip(_(f["source"]))
                    a.triggered.connect(lambda _c=False, kk=k: self.add_filter(kk))

    def rebuild(self, props: bool = True):
        """The layers' rows built again; ``props``: the properties panel too (not for a visibility toggle, issue #24:
        its scroll and an open curves editor were lost)."""
        clear(self.lbox)
        self.rows = []
        ids = {lay["id"] for lay in self.layers}
        self._collapsed = {c for c in self._collapsed if (c[0] if isinstance(c, tuple) else c) in ids}
        if self.sel and (lay := self.layer(self.sel[0])) is not None and lay.get("target"):
            self._collapsed -= {lay["target"], (lay["target"], lay.get("sub"))}     # the selected filter is shown
        for lay, depth in rows(self.layers, self._collapsed):
            sl = self.sel is not None and self.sel[0] == lay["id"]
            if lay["kind"] == "sub" or (sl and self.sel[1] in SUBS):      # a sub-layer is selected as (its layer, it)
                sl = sl and self.sel[1] == lay.get("sub", "layer")
                if lay["kind"] == "sub":
                    n = sum(1 for f in children(self.layers, lay["id"]) if f.get("sub") == lay["sub"])
                    row = LayerRow(self, lay, self.thumb(lay), None, sl, False, depth, n,
                                   (lay["id"], lay["sub"]) in self._collapsed)
                else:
                    row = LayerRow(self, lay, self.thumb(lay), None, sl, False, depth)
                self.lbox.addWidget(row)
                self.rows.append(row)
                continue
            row = LayerRow(self, lay, self.thumb(lay), self.mask_thumb(lay), sl and self.sel[1] == "layer",
                           sl and self.sel[1] == "mask", depth,
                           len(children(self.layers, lay["id"])) + 2 * is_group(lay) if is_pixel(lay) else 0,
                           lay["id"] in self._collapsed)
            self.lbox.addWidget(row)
            self.rows.append(row)
        waiting = [p for p in self.s.paths if not p.rendered]
        for p in waiting:
            b = QPushButton(_('"{name}": to compute  →').format(name=tr_name(p.name)))
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
            # scrolled to the selected row up and down only: the list never stays scrolled sideways (it has no
            # horizontal bar, so its left column, the visibility boxes, could not be brought back)
            QTimer.singleShot(0, lambda: (self.lsc.ensureWidgetVisible(sel_row, 0, 10),
                                          self.lsc.horizontalScrollBar().setValue(0)))
        lay = self.layer(self.sel[0]) if self.sel else None
        self.b_dup.setEnabled(lay is not None)
        why = self.s.align_problem() if hasattr(self.s, "align_problem") else "needs 2 images"     # greyed, and why
        self.b_align.setEnabled(not why)
        self.b_align.setToolTip(_("Align the image layers") if not why else _("Align the image layers") + ": " + why)
        self.b_del.setEnabled(lay is not None and (not is_pixel(lay) or sum(map(is_pixel, self.layers)) > 1))
        # greyed, it says why (control round): the last image layer stays, or nothing chosen
        self.b_del.setToolTip(_("Delete the selected layer") if self.b_del.isEnabled() else
                              _("Choose a layer first") if lay is None else
                              _("The last image layer stays: Post needs an image (hide it with its eye)"))
        self.b_maskadd.setEnabled(lay is not None)
        masked = lay is not None and lay["mask"] is not None      # the accent: the selected layer has its mask
        icon_only(self.b_maskadd, "mask", _("Edit the mask of the selected layer") if masked else
                  _("Add a mask to the selected layer"), 16, "selected" if masked else "normal", box=28)
        self._undo_buttons()
        self.build_filter_menu()
        if props:
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
        self.rebuild(props=False)
        self.request()

    def set_shown(self, lid, sub, v):
        """One of a Stars and background layer's two images shown or not in it (issue #3)."""
        self.push()
        self.layer(lid)["shown"][sub] = v
        self.rebuild(props=False)
        self.request()

    def toggle_children(self, key):
        """The arrow: show or hide the rows under a layer (``key`` its id), or under one of a Stars and background
        layer's two images (``key`` (its id, the image)); the image does not change. A hidden filter that was selected
        leaves the selection to the row it is under."""
        self._collapsed ^= {key}
        cur = self.layer(self.sel[0]) if self.sel else None
        if key in self._collapsed and cur is not None and cur["kind"] == "filter":
            lid, sub = key if isinstance(key, tuple) else (key, None)
            if cur.get("target") == lid and (sub is None or cur.get("sub") == sub):
                self.sel, self.part_sel = (lid, sub or "layer"), None
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
                if is_f and r.lay["kind"] == "sub" and 0.28 <= rel <= 0.72:
                    return {"parent": r.lay["target"], "index": 0, "row": i, "sub": r.lay["sub"],
                            "text": _('Applies to "{name}" only').format(name=_(POST_PART_NAMES[r.lay["sub"]]))}
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
            if is_group(self.layer(parent)):              # between the filters of one of its two images: in it
                near = below if below is not None and below.depth == 2 else \
                    above if above is not None and (above.depth == 2 or above.lay["kind"] == "sub") else None
                sub = near.lay.get("sub") if near is not None else None
                if sub:
                    text, depth = _('Applies to "{name}" only').format(name=_(POST_PART_NAMES[sub])), 2
                return {"parent": parent, "index": index, "sub": sub, "text": text,
                        "y": below.geometry().top() - 2 if below is not None else rs[-1].geometry().bottom() + 1,
                        "x0": rs[0].geometry().left() + 30 + 4 + INDENT * depth}
        else:
            index = len([r for r in rs[:gap] if not r.depth and r.lay["id"] not in mine])
            text = _("On all below") if is_f else ""
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
            moved = self.move(row.lay["id"], d["parent"], d["index"])
            if row.lay["kind"] == "filter" and d["parent"] is not None and is_group(self.layer(d["parent"])):
                self._set_sub(self.layer(row.lay["id"]), d.get("sub"), push=not moved)

    def _set_sub(self, fl, sub, push=True):
        """A filter of a Stars and background layer put on one of its two images (``sub``), or on the two together."""
        if fl.get("sub") == sub:
            return
        if push:
            self.push()
        fl["sub"] = sub
        self.rebuild()
        self.request(final_only=True)

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
        if tgt is not None and is_group(tgt):           # or one of its two images (issue #3)
            out += [(_('"{name}" only').format(name=_(POST_PART_NAMES[sub])), (tgt["id"], sub)) for sub in SUBS]
        return out + [(_("All below"), None)]

    @staticmethod
    def scope_of(fl):
        """What a filter applies to, as ``scope_choices`` says it."""
        return (fl["target"], fl["sub"]) if fl.get("sub") else fl.get("target")

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
                a = m.addAction(_("On all below") if val is None else _("Applies to {what}").format(
                    what=_txt))
                a.setCheckable(True)
                a.setChecked(self.scope_of(lay) == val)
                a.triggered.connect(lambda _c=False, t=val: self.set_target(lay, t) if self.scope_of(lay) != t else None)
            m.addSeparator()
        if lay["kind"] == "workflow":
            m.addSeparator()
            m.addAction(_("Show workflow  →")).triggered.connect(lambda: self.goResults.emit(lay["pid"]))
            q = self.path(lay["pid"])
            if getattr(self.s, "is_plain_image", lambda x: False)(q):      # told from its file when it can be
                a = m.addAction(_("Already stretched"))
                a.setCheckable(True)
                a.setChecked(bool(q.source.get("stretched")))
                a.toggled.connect(lambda on, x=q: self.set_stretched(x, on))
        m.addAction(_("Rename")).triggered.connect(lambda: self.rename(lay))
        if not is_pixel(lay) or sum(map(is_pixel, self.layers)) > 1:
            m.addSeparator()
            m.addAction(_("Delete")).triggered.connect(lambda: (self.select(lay["id"], "layer"), self.delete()))
        m.exec(gpos)

    def set_stretched(self, q, stretched: bool):
        """An image's kind said by the user: computed again."""
        self.s.set_image_file(q, stretched=stretched)
        self.runAsked.emit()

    def _dock_panel(self):
        self.split.addWidget(self.right)
        self.split.setStretchFactor(1, 0)
        self.split.setSizes([10_000, int(prefs.get("post_panel_width") or PANEL_W)])

    def _show_layers(self):
        self.lsc.ensureVisible(0, 0)

    def _show_props(self):
        self.psc.ensureVisible(0, 0)

    def _show_panel(self, on: bool):
        self.dock.set_state("docked" if on else "icons")
        prefs.set("post_panel_shown", bool(on))

    def set_layout(self, state: str, solo: bool = False):
        """Issue #44: docked | icons | hidden (the panel's state the window chose for this size range), and its groups
        one at a time on a small window."""
        self.folds.set_solo(solo)
        self.dock.set_state(state)

    def set_compact(self, on: bool):
        self.top_bar.setContentsMargins(*((8, 2, 8, 2) if on else (10, 6, 10, 6)))

    def _panel_moved(self, *_a):
        if self.right.isVisible():
            prefs.set("post_panel_width", int(self.right.width()))

    def rename(self, lay, new: str | None = None):
        old = layer_name(lay)
        if new is None:
            from .dialogs import ask_name
            new = ask_name(self.window(), _("Rename layer"), old)
        if not new or new == old:
            return
        p = self.path(lay["pid"]) if lay["kind"] == "workflow" and lay.get("part") is None else None
        if p is not None and lay.get("name") == p.name and hasattr(self.s, "name_problem"):
            # the layer named after its workflow: the workflow (and its wire in the graph) take the name too, its
            # layers follow (control round); one step of the workflows' Undo, as a branch removed with its layer
            why = self.s.name_problem(new, pid=p.pid)
            if why:
                self.toasted.emit(_("Name refused: {why}").format(why=why))
                return
            self._save_t.stop()
            self._save()                                  # the layers shown into the project, then renamed there
            self._restoring = True
            try:
                self.s.rename(p.pid, new)
                self._post_changed(None)                  # and taken back here with their new names
            finally:
                self._restoring = False
            self.undo.append(MODEL_STEP)
            self.redo.clear()
            self._undo_buttons()
            self.toasted.emit(_("“{old}” renamed “{name}”: its workflow too · Ctrl+Z").format(old=old, name=new))
            return
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
        target = sub = None
        if cur is not None:
            target = cur["id"] if is_pixel(cur) else cur.get("target")
            sub = self.sel[1] if self.sel[1] in SUBS else None if is_pixel(cur) else cur.get("sub")
        self.push()
        fl = new_filter_layer(key, target, sub)
        same = sum(1 for lay in self.layers if lay["kind"] == "filter" and lay.get("filter") == key)
        if same:                                        # a second Curves is "Curves · 2" (issue #15: no twin names)
            fl["name"] = f"{fl['name']} · {same + 1}"
        if target is not None:                          # on top of that layer's own filters
            group = [lay for lay in self.layers if lay.get("target") == target]
            self.sel = ((group[0] if group else self.layer(target))["id"], "layer")
        self._insert_above_sel(fl)
        self.rebuild()
        self.request(final_only=True)
        return fl

    def set_target(self, fl, target):
        """Nest a filter in a pixel layer (on top of its children), or in one of a Stars and background layer's two
        images (``target`` (its id, the image)), or take it out: then it is the row just above that layer and applies
        to everything below."""
        sub = None
        if isinstance(target, tuple):
            target, sub = target
        if target is not None and target == fl.get("target"):
            self._set_sub(fl, sub)
            return
        if target is None:
            old = fl.get("target")
            t = [x["id"] for x, _kids in tree(self.layers)]
            idx = t.index(old) if old in t else t.index(fl["id"])
        else:
            idx = 0
        self._collapsed.discard(target)
        moved = self.move(fl["id"], target, idx)
        if sub:
            self._set_sub(self.layer(fl["id"]), sub, push=not moved)

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
        p = self.path(lay["pid"]) if lay["kind"] == "workflow" and lay.get("part") is None else None
        if p is not None and not any(x.get("pid") == p.pid for x in others if x["kind"] == "workflow") and (
                getattr(p, "end", None) is not None or getattr(self.s, "is_sent_branch", lambda q: False)(p)):
            # a block's image sent to Post: its branch was made only for it, it goes with its layer, as when its wire
            # is taken away in the graph (one step of the workflows' Undo, which brings both back)
            self.flush()
            self._restoring = True
            try:
                self.s.remove_path(self.s.paths.index(p))
            finally:
                self._restoring = False
            if hasattr(self.s, "log"):
                self.s.log('Workflow "{name}" deleted', name=p.name)
            self.undo.append(MODEL_STEP)
            self.redo.clear()
            self.toasted.emit(_('Layer "{name}" deleted: its branch went with it · Ctrl+Z').format(name=layer_name(lay)))
            self._undo_buttons()
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

    # ------------------------------------------------------------ properties of the selection
    def build_props(self):
        clear(self.pbox)
        self.folds.reset()
        self.curves_editor = self.function_graph = None
        self.viewer.mode = "pan"
        lay = self.layer(self.sel[0]) if self.sel else None
        if lay is None:
            self.pbox.addStretch(1)
            return
        if self.sel[1] == "mask" and lay["mask"] is not None:
            self.mask_props(lay)
        elif self.sel[1] in SUBS:
            self.pbox.addWidget(label(_(POST_PART_NAMES[self.sel[1]]), "h2"))
            self.pbox.addWidget(label(_('Image of "{name}". The filters added while it is selected apply to it only, '
                                        "before the stars are put back on the starless image.").format(
                                            name=layer_name(lay)), "small", wrap=True))
        else:
            self.layer_props(lay)
        self.pbox.addStretch(1)
        self.folds.finish(f"{lay['kind']}:{lay.get('filter', '')}:{self.sel[1]}")     # issue #44: as the block panel's
        self.dock.refresh_icons()
        self._hint()

    def _group(self, text):
        g = self.folds.title(text)
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
            # what it applies to is its place in the layers (issue #26): moved there, or with its right-click menu
            self._group(_(f["name"]))
            self.pbox.addWidget(caption(_(f["source"])))
            refresh = lambda: None                                          # noqa: E731
            if f.get("editor") == "spots":
                self.heal_props(lay)
            if f.get("editor") == "curves":
                self.curves_editor = ed = CurvesEditor(lay["params"]["curves"], self._hists.get(lay["id"]),
                                                       channel=self._curve_channel.get(lay["id"], "RGB/K"))
                ed.channelChanged.connect(lambda ch, i=lay["id"]: self._curve_channel.__setitem__(i, ch))
                self._wire_editor(ed)
                self.pbox.addWidget(ed)
            elif f.get("graph"):
                self.function_graph = g = FunctionGraph(lambda xs, fn=f["graph"]: fn(lay["params"], xs),
                                                        self._hists.get(lay["id"]))
                self.pbox.addWidget(g)
                refresh = g.update
            for q in f["params"]:
                if q["type"] in ("curve", "spots"):
                    continue
                if q["type"] == "bool":
                    cb = QCheckBox(_(q["label"]))
                    cb.setChecked(bool(lay["params"][q["key"]]))
                    cb.toggled.connect(lambda v, k=q["key"]: self._set_param(lay, k, bool(v)))
                    self.pbox.addWidget(cb)
                elif q["type"] == "choice":
                    r = QHBoxLayout()
                    r.addWidget(QLabel(_(q["label"])))
                    c = QComboBox()
                    for ch in q["choices"]:
                        c.addItem(_(ch), ch)
                    c.setCurrentIndex(q["choices"].index(lay["params"][q["key"]]))
                    c.currentIndexChanged.connect(lambda i, k=q["key"], c=c: (self._set_param(lay, k, c.itemData(i)),
                                                                              refresh()))
                    r.addWidget(c, 1)
                    self.pbox.addLayout(r)
                else:
                    sl = ContSlider(_(q["label"]), q["lo"], q["hi"], lay["params"][q["key"]], q["default"],
                                    unit=q["unit"], step=q["step"])
                    if q.get("help"):                  # what the short name leaves out
                        tip = _(q["help"]) + NL + _("Double-click: default value")
                        sl.setToolTip(tip)
                        sl.name.setToolTip(tip)
                    self._wire(sl, lambda v, k=q["key"]: (lay["params"].__setitem__(k, v), refresh()))
                    self.pbox.addWidget(sl)
            return
        if lay["kind"] == "image":
            self.image_props(lay)
            return
        p = self.path(lay["pid"])
        part = lay.get("part")
        if part:
            self.pbox.addWidget(label(_(PART_NOTES[part]), "small", wrap=True))
        if p is not None and hasattr(self.s, "param_value") and hasattr(self.s, "set_param"):
            # issue #49: the Output block's setting, here (the block is in the wire): its channels stretched linked
            # the same control, the same name as in the Output's panel of the graph (one name, one kind of control)
            c = QCheckBox(_("Linked channels"))
            c.setToolTip(_("Unticked: every color stretched apart, which neutralizes the background (the "
                           "workflow's output is prepared again)"))
            try:
                linked = self.s.param_value(p, "render_simple", "linked")
            except Exception:  # noqa: BLE001 - a value the worker has not read yet
                linked = True
            c.setChecked(linked in (None, True))
            c.toggled.connect(lambda on, q=p: self._set_workflow_param(q, "render_simple", "linked", bool(on)))
            self.pbox.addWidget(c)
            self.linked_box = c
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
        infos = self.s.slider_infos(p, part) if p is not None else []
        if part:            # the Stars block's images: the sliders they have, the stars first (what they are for)
            infos = sorted((x for x in infos if x.available), key=lambda x: x.group != STARS_GROUP)
        for sd in infos:
            if sd.group != group:
                group = sd.group
                self._group(_(group_title(group, part)))
                if group == STARS_GROUP and part == "combined":
                    r = QHBoxLayout()
                    r.addWidget(label(_("Star blend mode"), "dim"))
                    c = QComboBox()
                    for b in BLENDS:
                        c.addItem(_(b), b)
                    c.setToolTip(_("How the stars go back on the starless image: Screen (the usual way), Linear "
                                   "dodge (add) or Lighten"))
                    c.setCurrentIndex(max(0, BLENDS.index(lay.get("blend") or BLENDS[0])))
                    c.currentIndexChanged.connect(lambda i, c=c: self._set(lay, "blend", c.itemData(i)))
                    r.addWidget(c, 1)
                    self.pbox.addLayout(r)
            pos = lay["sliders"].get(sd.key, 2.0)
            sl = ContSlider(_(sd.name), 0, 4, pos, 2, ends=[_(e) for e in sd.ends] if sd.ends else None,
                            show=offset_text)
            ok, why, _need = self.s.slider_state(p, sd.key, part)   # the model answers in the language of the screen
            sl.set_available(ok, why)
            self._wire(sl, lambda v, k=sd.key: self._slider(lay, k, v))
            self.pbox.addWidget(sl)

    def image_props(self, lay):
        """An image layer: where it comes from, its automatic stretch (linked or each channel) and its three
        sliders."""
        ok = self.worker_has_image(lay)
        self.pbox.addWidget(label(_("A stage's image, stretched automatically.") if ok else
                                  _("This image is no longer in the project's results."), "small" if ok else "warn"))
        # the same control, the same name as the Output block's in the graph (one name, one kind of control)
        c = QCheckBox(_("Linked channels"))
        c.setToolTip(_("Unticked: every color stretched apart, which neutralizes the background"))
        c.setChecked(bool(lay.get("linked", True)))
        c.toggled.connect(lambda on: self._set(lay, "linked", bool(on)))
        self.pbox.addWidget(c)
        self.linked_box = c
        self._group(_("Stretch"))
        for key, name, ends in IMAGE_SLIDERS:
            pos = lay["sliders"].get(key, 2.0)
            sl = ContSlider(_(name), 0, 4, pos, 2, ends=[_(e) for e in ends],
                            show=offset_text)
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

    def _wire_editor(self, ed: CurvesEditor):
        """The Curves panel: a draft while a point moves; on release the exact render and one undo step (the state
        before is taken when the edit starts, before the points change)."""
        state = {"before": None}

        def starting():
            if state["before"] is None:
                state["before"] = snapshot(self.layers)

        def committed():
            self.push(state["before"] if state["before"] is not None else snapshot(self.layers))
            state["before"] = None
            self.request(final_only=True)
        ed.starting.connect(starting)
        ed.changed.connect(self.request)
        ed.committed.connect(committed)

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
        add = set_icon(QPushButton(_("Add to mask")), "add", 16)
        am = QMenu(add)
        for src, name in MASK_SOURCES.items():
            a = am.addAction(_(name))
            a.triggered.connect(lambda _c=False, s=src: self.add_part(lay, s))
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
        x = icon_only(QPushButton(), "remove", _("Remove from mask"), 16, box=22)
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
        if self.viewer.mode == "heal":
            self.heal_size = float(np.clip(self.heal_size * k, 6, 1200))
            self.viewer.brush_r = self.heal_size / 2
            self.build_props()
            return
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

    # ------------------------------------------------------------ spot healing
    def heal_props(self, lay):
        """The Spot healing filter's panel (Photoshop's Spot Healing Brush, issue #20): the brush size, what is
        healed, and the source set by hand (Alt+click) when there is one."""
        self.viewer.mode = "heal"
        self.viewer.brush_r = self.heal_size / 2
        sz = ContSlider(_("Size"), 6, 1200, self.heal_size, 80, unit="px", step=1)
        sz.moved.connect(lambda v: (setattr(self, "heal_size", float(v)), setattr(self.viewer, "brush_r", v / 2)))
        sz.done.connect(lambda v: (setattr(self, "heal_size", float(v)), setattr(self.viewer, "brush_r", v / 2)))
        self.pbox.addWidget(sz)
        r = QHBoxLayout()
        n = len(lay["params"]["spots"])
        r.addWidget(label(_("{n} spots healed").format(n=n) if n != 1 else _("1 spot healed"), "small"))
        r.addStretch(1)
        cl = QPushButton(_("Remove all"))
        cl.setEnabled(n > 0)
        cl.clicked.connect(lambda: self._heal_set(lay, []))
        r.addWidget(cl)
        self.pbox.addLayout(r)
        src = QHBoxLayout()
        manual = self._heal_src is not None or self._heal_off is not None
        src.addWidget(label(_("Source: set by hand (Alt+click)") if manual else
                            _("Source: found around each stroke · Alt+click: set it by hand"), "small", wrap=True), 1)
        if manual:
            au = QPushButton(_("Automatic"))
            au.setToolTip(_("The source found around each stroke again"))
            au.clicked.connect(lambda: (setattr(self, "_heal_src", None), setattr(self, "_heal_off", None),
                                        self.build_props()))
            src.addWidget(au)
        self.pbox.addLayout(src)

    def _heal_layer(self):
        lay = self.layer(self.sel[0]) if self.sel and self.sel[1] == "layer" else None
        return lay if lay is not None and lay["kind"] == "filter" and lay["filter"] == "heal" else None

    def _heal_set(self, lay, spots):
        self.push()
        lay["params"]["spots"] = spots
        self.request(final_only=True)
        self.build_props()

    def _heal_source(self, x, y, r, avoid=()):
        """The source of a spot: the best match around it on the image shown (its half resolution), as Photoshop's
        Spot Healing Brush; ``avoid``: the stroke's other spots (x, y, r), no source on them (a trail is not healed
        from itself); beside the spot when the image is not there yet."""
        from .postfilters.heal import find_source
        W, H = self.src.size()
        R = int(math.ceil(r * 4.8)) + 4
        x0, y0 = max(0, int(x - R) // 2), max(0, int(y - R) // 2)
        x1, y1 = min(W // 2, int(x + R) // 2 + 1), min(H // 2, int(y + R) // 2 + 1)
        a = self.src.region(1, x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
        if a is not None and a.size and a.max() > 0:
            dx, dy = find_source(a.astype(np.float32) / 255.0, x / 2 - x0, y / 2 - y0, r / 2,
                                 avoid=[(ax / 2 - x0, ay / 2 - y0, ar / 2) for ax, ay, ar in avoid])
            if dx or dy:
                return 2 * dx, 2 * dy
        return 2.5 * r, 0.0

    def _heal_press(self, x, y, button, alt):
        """Photoshop's Spot Healing Brush: a press starts a stroke (healed when the button is let go); Alt+click
        sets the source by hand (Photoshop's Healing Brush) for the strokes that follow."""
        lay = self._heal_layer()
        if lay is None or button != int(Qt.LeftButton.value):
            return
        if alt:
            self._heal_src, self._heal_off = (x, y), None
            self.build_props()
            self.viewer.update()
            return
        if self._heal_src is not None:                          # the first stroke after Alt+click fixes the offset
            self._heal_off = (self._heal_src[0] - x, self._heal_src[1] - y)
            self._heal_src = None
        self._heal_stroke = ([(x, y)], snapshot(self.layers))
        self.viewer.update()

    def _heal_move(self, x, y):
        if self._heal_layer() is None or self._heal_stroke is None:
            return
        pts = self._heal_stroke[0]
        step = max(1.0, self.heal_size / 2 * 0.5)              # a spot every half radius along the stroke
        lx, ly = pts[-1]
        d = math.hypot(x - lx, y - ly)
        for k in range(1, int(d // step) + 1):
            pts.append((lx + (x - lx) * k * step / d, ly + (y - ly) * k * step / d))
        self.viewer.update()

    def _heal_release(self):
        """The stroke healed: a spot every half radius along it, each from a source found around it (not on the
        stroke) or from the source set by hand."""
        lay = self._heal_layer()
        st, self._heal_stroke = self._heal_stroke, None
        if lay is None or st is None:
            self.viewer.update()
            return
        pts, before = st
        r = self.heal_size / 2
        avoid = [(px, py, r) for px, py in pts]
        new = []
        for x, y in pts:
            dx, dy = self._heal_off if self._heal_off is not None else self._heal_source(x, y, r, avoid)
            new.append([round(x, 1), round(y, 1), round(r, 1), round(dx, 1), round(dy, 1)])
        self.push(before)
        lay["params"]["spots"] = lay["params"]["spots"] + new
        self.request(final_only=True)
        self.build_props()
        self.viewer.update()

    def _heal_overlay(self, p: QPainter, v: Viewer):
        """While painting, the stroke in light grey (as Photoshop's), and the source set by hand as a small cross;
        nothing is left on the image afterwards."""
        lay = self._heal_layer()
        if lay is None or self.b_cmp.isChecked():
            return
        if self._heal_stroke is not None:
            rr = self.heal_size / 2 * v.scale
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 70))
            for x, y in self._heal_stroke[0]:
                p.drawEllipse(v.img_to_w(x, y), rr, rr)
        cross = None
        if self._heal_src is not None:
            cross = v.img_to_w(*self._heal_src)
        elif self._heal_off is not None and self._heal_stroke is not None:
            x, y = self._heal_stroke[0][-1]
            cross = v.img_to_w(x + self._heal_off[0], y + self._heal_off[1])
        if cross is not None:
            p.setPen(QPen(QColor("#ffffff"), 1.5))
            p.drawLine(QPointF(cross.x() - 7, cross.y()), QPointF(cross.x() + 7, cross.y()))
            p.drawLine(QPointF(cross.x(), cross.y() - 7), QPointF(cross.x(), cross.y() + 7))

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
        elif self.viewer.mode == "heal":
            t = _("Spot healing: click or paint over a defect, healed when released · [ ] size · Alt+click: the "
                  "source by hand · Ctrl+Z: undo · middle button: move the image")
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
    openImages = Signal()

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
        # nothing computed yet: the layers the graph counts (its Post block's "N layers"), named, each "not computed
        # yet", so that the page and the block say the same thing
        self.empty_title = label(_("No layers yet"), "h1")
        ev.addWidget(self.empty_title, 0, Qt.AlignHCenter)
        self.empty_note = label(_("Each computed workflow becomes a layer here."), "dim")
        ev.addWidget(self.empty_note, 0, Qt.AlignHCenter)
        self.empty_list = QVBoxLayout()
        self.empty_list.setSpacing(2)
        ev.addLayout(self.empty_list)
        o = set_icon(QPushButton(_("Open images…")), "block_image", 16, "on_accent")
        o.setObjectName("primary")
        o.setToolTip(_("Each image becomes a layer, and a block in the graph"))
        o.clicked.connect(self.openImages.emit)
        ev.addWidget(o, 0, Qt.AlignHCenter)
        b = QPushButton(_("Graph  →"))
        b.setObjectName("link")
        b.clicked.connect(self.goWorkflow.emit)
        ev.addWidget(b, 0, Qt.AlignHCenter)
        self.pages.addWidget(self.empty)
        self.view = PostView(session, worker=worker, persist=persist)
        self.pages.addWidget(self.view)
        self._asked = None                               # the layer Open in Post asked for, shown in the list
        self.view.layerAsked.connect(self._ask)
        # bound to this page's own methods, never a lambda: Qt drops them with the page (control round: a page
        # rebuilt by a language switch still answered the project's changes, a Python error at the next Ctrl+Z)
        session.pathsChanged.connect(self.update_page)
        if hasattr(session, "postChanged"):
            session.postChanged.connect(self._post_changed)
        self.update_page()

    def _post_changed(self, _source):
        self.update_page()

    def _ask(self, lid):
        self._asked = lid
        self.update_page()

    def coming_layers(self) -> list[tuple[str, int | None]]:
        """The layers Post will have, top first, as the graph counts them: (name as saved, workflow id)."""
        try:
            from .blocks import Graph
            return list(Graph(self.s).post_layers)
        except Exception:  # noqa: BLE001 - a model without the graph's surface: its workflows' images
            return [(p.name, p.pid) for p in self.s.paths]

    def update_page(self):
        shown = any(p.rendered for p in self.s.paths) or any(lay["kind"] == "image" for lay in self.view.layers)
        self.pages.setCurrentIndex(1 if shown else 0)
        if shown:
            return
        clear(self.empty_list)
        coming = self.coming_layers()
        self.empty_title.setText(_("Nothing computed yet") if coming else _("No layers yet"))
        self.empty_note.setText(_("The layers Post will have once their workflows are computed:") if coming else
                                _("Each computed workflow becomes a layer here."))
        asked = self.view.layer(self._asked) if self._asked is not None else None
        if asked is None and self._asked is not None and hasattr(self.s, "post_layer"):
            asked = self.s.post_layer(self._asked)
        for name, _pid in coming:
            lb = label(_("{name}  ·  not computed yet").format(name=tr_name(name)), "dim")
            if asked is not None and asked.get("name") == name:     # Open in Post of this layer's block
                lb.setStyleSheet(f"color: {ACCENT}; font-weight: 600;")
            self.empty_list.addWidget(lb, 0, Qt.AlignHCenter)

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
