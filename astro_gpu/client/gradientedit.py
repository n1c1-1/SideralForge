"""The Gradient block's samples and excluded areas, edited on the image in Inspect (``GradientEditor``).

The usual tools of the background extractions: the samples of the polynomial and of the RBF on a generated grid, then
placed, moved and removed by hand (left click adds or moves, right click removes, as in Siril), outlined in cyan when
the model uses them and in red when it leaves them out (as DynamicBackgroundExtraction); the areas left out of every
model painted with a brush or drawn as ellipses, under a red veil at 50 % (Photoshop's quick mask). The brush is Post's:
the same size and paint / erase controls, the same keys ([ ] size, X paint / erase), the line of the gestures under the
image. Space held, the middle button or (with the brush) a right drag move the image, as in Post and Photoshop; M
hides what is drawn over the image; Ctrl+Z / Ctrl+Y undo and redo each edit.

What the edits give is seen on a reduced copy of the stack (about a million pixels, whatever the sensor), with the
stack before them behind the curtain: one computation at a time in a thread of its own, after the mouse is let go and
0.3 s without another edit, the last edit only (the ones in between are skipped); the automatic model, slower, is
previewed on request. Nothing is stored until Apply: the samples and areas then become the block's settings
(``sample_points``, ``exclude_ellipses``, ``exclude_brush``), the block is to compute again. The edits not applied are
kept per workflow while Inspect is open.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .i18n import _
from .widgets import label

#: the preview's copy of the stack holds about this many pixels
PREVIEW_PIXELS = 1_000_000
#: the preview starts this long after the last edit
DEBOUNCE_MS = 300
#: a click this close to a sample (screen px) takes it
PICK_PX = 9
USED, UNUSED, OUTLINE = QColor("#56c8f5"), QColor("#ff4d55"), QColor(255, 255, 255, 220)
VEIL = (229, 72, 77, 128)                      # the excluded areas: red, 50 %
FIELDS = ("sample_points", "exclude_ellipses", "exclude_brush")
UNDO_MAX = 50


class _Job(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        self.fn()


class GradientEditor(QObject):
    """Edits the samples and excluded areas of the Gradient block of the workflow shown, on the main viewer of
    Inspect. ``attach(view, p)`` when the block's picture is shown, ``detach()`` otherwise; ``controls(p)`` the
    panel's section. It is the canvas's ``tool`` while a tool is on."""

    _done = Signal(int, object)                # (request number, result of the thread)

    def __init__(self, tune):
        super().__init__(tune)
        self.tv, self.s = tune, tune.s
        self.view = None
        self.p = None
        self.mode: str | None = None           # None (looking), "points", "brush", "ellipse"
        self.paint = True
        self.size = 60.0                       # brush diameter, stack pixels
        self.shown = True                      # what is drawn over the image (M)
        self.work: dict[str, list] = {f: [] for f in FIELDS}
        self.dirty = False
        self.undo_l: list = []
        self.redo_l: list = []
        self._kept: dict[int, tuple] = {}      # workflow -> its edits not applied
        self.gen = 0
        self.small = None                      # {"pid", "arr", "valid", "f", "W", "H"}
        self.samples = None                    # {"x", "y", "used", "manual", "box"} in stack pixels
        self.result = None                     # the last preview (3, h, w)
        self.busy = False
        self._pending = None                   # the request waiting for the one running
        self._hover = None
        self._grab = None
        self._ell = None
        self._last = None
        self._stroke0 = None                   # where the brush stroke in progress starts in exclude_brush
        self._veil = None                      # (key, QImage)
        self._ui: dict = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self.preview)
        self._done.connect(self._arrived)

    # ------------------------------------------------------------ the workflow and the viewer
    def _method(self) -> str:
        return str(self.s.param_value(self.p, "gradient", "method") or "polynomial")

    def _switch(self, p):
        """Another workflow: the edits of the one left are kept, those of ``p`` come back (or its settings)."""
        if self.p is not None:
            self._kept[self.p.pid] = (self.work, self.dirty, self.undo_l, self.redo_l)
        self.p = p
        if p.pid in self._kept:
            self.work, self.dirty, self.undo_l, self.redo_l = self._kept.pop(p.pid)
        else:
            self.work = {f: [list(x) for x in (self.s.param_value(p, "gradient", f) or [])] for f in FIELDS}
            self.dirty, self.undo_l, self.redo_l = False, [], []
        self.result = None
        self._veil = None
        self.samples = self._computed_samples()
        if self.small is not None and self.small["pid"] != p.pid:
            self.small = None

    def _computed_samples(self):
        from ..api import stage_outputs
        try:
            out = stage_outputs(self.s.p, self.p.pid, "gradient")
            z = np.load(f"{out.dir}/samples.npz") if out is not None else None
        except (OSError, ValueError, KeyError):
            return None
        if z is None or "box" not in z.files:
            return None
        return {"x": z["x"], "y": z["y"], "used": z["used"].astype(bool), "manual": bool(z["manual"]),
                "box": float(z["box"])}

    def attach(self, view, p):
        """The Gradient block's picture is shown in ``view`` (its tab built again: the edits carry on)."""
        if self.p is None or p.pid != self.p.pid:
            self.set_mode(None)
            self._switch(p)
        self.view = view
        view.add_overlay(self._paint)
        view.canvas.tool = self if self.mode else None
        self._hint()
        if self.result is not None:
            self._show(self.result)

    def detach(self):
        if self.view is not None:
            try:
                self.view.canvas.tool = None
                self.view.tool_hint.hide()
            except RuntimeError:
                pass
        self.view = None

    # ------------------------------------------------------------ the panel's section
    def controls(self, p) -> QWidget:
        """Samples and excluded areas: the tools, the brush, the grid, undo, Apply / Discard, the preview's state.
        Built whole (the brush's row is there in every mode, enabled with the brush): a tool switched on or off never
        moves the panel."""
        if self.p is None or p.pid != self.p.pid:
            self._switch(p)
        from .post3 import ContSlider
        box = QFrame()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 4, 0, 6)
        v.setSpacing(6)
        method = self._method()
        sampled = method in ("polynomial", "rbf")
        self._ui = {}
        tools = QHBoxLayout()
        for key, text, tip in (
                ("points", "Samples", "Left click: add or move a sample · right click: remove it"),
                ("brush", "Brush", "Paint the areas left out of the sky model"),
                ("ellipse", "Ellipse", "Drag an ellipse around an object to leave it out")):
            if key == "brush":
                tools.addSpacing(8)
                tools.addWidget(label(_("Exclude:"), "small"))
            b = QPushButton(_(text))
            b.setObjectName("face")
            b.setCheckable(True)
            b.setChecked(self.mode == key)
            on = key != "points" or sampled
            b.setEnabled(on)
            b.setToolTip(_(tip) if on else _("The automatic method uses no samples: it fits every pixel that does "
                                            "not look like an object"))
            b.clicked.connect(lambda _c=False, k=key: self.set_mode(None if self.mode == k else k))
            tools.addWidget(b)
            self._ui[key] = b
        tools.addStretch(1)
        v.addLayout(tools)
        r = QHBoxLayout()
        for text, val in (("Paint", True), ("Erase", False)):
            b = QPushButton(_(text))
            b.setObjectName("face")
            b.setCheckable(True)
            b.setChecked(self.paint == val)
            b.clicked.connect(lambda _c=False, vv=val: self._set_paint(vv))
            r.addWidget(b)
            self._ui["paint" if val else "erase"] = b
        r.addStretch(1)
        v.addLayout(r)
        sz = ContSlider(_("Size"), 4, 1200, self.size, 60, unit="px", step=1)
        sz.moved.connect(lambda val: setattr(self, "size", float(val)))
        sz.done.connect(lambda val: setattr(self, "size", float(val)))
        v.addWidget(sz)
        self._ui["size"] = sz
        row = QHBoxLayout()
        if sampled:
            g = QPushButton(_("Generate the grid"))
            g.setToolTip(_("Samples on a regular grid, the brighter areas left out (Samples per side, Object "
                           "rejection); then move or remove them by hand"))
            g.clicked.connect(self.generate)
            row.addWidget(g)
        for key, text, tip, fn in (("undo", "↶", "Undo the last edit (Ctrl+Z)", self.undo),
                                   ("redo", "↷", "Redo (Ctrl+Y)", self.redo)):
            b = QPushButton(text)
            b.setToolTip(_(tip))
            b.clicked.connect(fn)
            row.addWidget(b)
            self._ui[key] = b
        clr = QPushButton(_("Remove all…"))
        clr.setToolTip(_("Remove the placed samples and the excluded areas: back to the automatic grid (undo brings "
                         "them back)"))
        clr.clicked.connect(self.clear)
        row.addWidget(clr)
        if method == "automatic":
            pv = QPushButton(_("Preview"))
            pv.setToolTip(_("The automatic model on a reduced copy (a few seconds)"))
            pv.clicked.connect(lambda: self.preview(force=True))
            row.addWidget(pv)
        row.addStretch(1)
        v.addLayout(row)
        act = QHBoxLayout()
        ap = QPushButton(_("Apply"))
        ap.setToolTip(_("Keep these samples and areas as the block's settings (the block is then to compute)"))
        ap.clicked.connect(self.apply)
        ds = QPushButton(_("Discard"))
        ds.setToolTip(_("Back to the samples and areas of the block's settings"))
        ds.clicked.connect(self.discard)
        act.addWidget(ap)
        act.addWidget(ds)
        act.addStretch(1)
        v.addLayout(act)
        self._ui["apply"], self._ui["discard"] = ap, ds
        st = label("", "small", wrap=True)
        v.addWidget(st)
        self._ui["status"] = st
        self._sync()
        return box

    def _status(self) -> str:
        bits = []
        if self.work["sample_points"]:
            bits.append(_("{n} samples placed").format(n=len(self.work["sample_points"])))
        n = len(self.work["exclude_ellipses"]) + (1 if any(d[3] for d in self.work["exclude_brush"]) else 0)
        if n:
            bits.append(_("{n} excluded areas").format(n=n) if n > 1 else _("1 excluded area"))
        if self.dirty:
            bits.append(_("not applied: Apply to keep them"))
        return " · ".join(bits)

    def _hint(self):
        """The gestures of the tool on, on the line under the image (as Post's)."""
        if self.view is None:
            return
        t = {"points": _("Samples: left click add or move · right click remove · Space or middle button: move the "
                         "image · M hide · Ctrl+Z undo"),
             "brush": _("Brush: drag to paint · X paint / erase · [ ] size · right drag: move the image · M hide · "
                        "Ctrl+Z undo"),
             "ellipse": _("Ellipse: drag around the object · right click inside: remove · Space: move the image · "
                          "M hide · Ctrl+Z undo")}.get(self.mode, "")
        try:
            self.view.tool_hint.setText(t)
            self.view.tool_hint.setVisible(bool(t))
        except RuntimeError:
            self.view = None

    def _sync(self):
        """The panel's widgets as the editor is (without building the panel again)."""
        try:
            for k in ("points", "brush", "ellipse"):
                if k in self._ui:
                    self._ui[k].setChecked(self.mode == k)
            for k, on in (("paint", self.paint), ("erase", not self.paint)):
                if k in self._ui:
                    self._ui[k].setChecked(on)
                    self._ui[k].setEnabled(self.mode == "brush")
            if "size" in self._ui:
                self._ui["size"].setEnabled(self.mode == "brush")
            if "status" in self._ui:
                self._ui["status"].setText(self._status())
            for k in ("apply", "discard"):
                if k in self._ui:
                    self._ui[k].setEnabled(self.dirty)
            if "undo" in self._ui:
                self._ui["undo"].setEnabled(bool(self.undo_l))
                self._ui["redo"].setEnabled(bool(self.redo_l))
        except RuntimeError:                   # the panel was built again
            self._ui = {}

    # ------------------------------------------------------------ modes
    def set_mode(self, mode: str | None):
        self.mode = mode
        if self.view is not None:
            try:
                c = self.view.canvas
                c.tool = self if mode else None
                if mode:
                    c.setFocus()               # the keys of the tool go to the image
                c.update()
            except RuntimeError:
                self.view = None
        if mode == "points" and not self.work["sample_points"]:
            self.generate()                    # Siril's way: a grid first, then edited by hand
        elif mode and self.result is None and self._method() != "automatic":
            self.preview()
        elif mode and self.small is None:
            self._run("load")
        self._hint()
        self._sync()

    def _set_paint(self, on: bool):
        self.paint = on
        self._sync()

    def key(self, e) -> bool:
        """The keys of the tool, from the canvas (it has the focus while a tool is on)."""
        k, mods = e.key(), e.modifiers()
        if k == Qt.Key_Z and mods & Qt.ControlModifier:
            self.redo() if mods & Qt.ShiftModifier else self.undo()
        elif k == Qt.Key_Y and mods & Qt.ControlModifier:
            self.redo()
        elif k == Qt.Key_BracketLeft and self.mode == "brush":
            self._resize(1 / 1.25)
        elif k == Qt.Key_BracketRight and self.mode == "brush":
            self._resize(1.25)
        elif k == Qt.Key_X and self.mode == "brush":
            self._set_paint(not self.paint)
        elif k == Qt.Key_M:
            self.shown = not self.shown
            if self.view is not None:
                self.view.canvas.update()
        elif k == Qt.Key_Escape:
            self.set_mode(None)
        else:
            return False
        return True

    def _resize(self, k: float):
        self.size = float(np.clip(self.size * k, 4, 1200))
        if "size" in self._ui:
            try:
                self._ui["size"].set_value(self.size)
            except (RuntimeError, AttributeError):
                pass
        if self.view is not None:
            self.view.canvas.update()

    # ------------------------------------------------------------ the edits and their undo
    def _push(self):
        self.undo_l.append(copy.deepcopy(self.work))
        del self.undo_l[:-UNDO_MAX]
        self.redo_l.clear()

    def undo(self):
        if self.undo_l:
            self.redo_l.append(copy.deepcopy(self.work))
            self.work = self.undo_l.pop()
            self._edited()

    def redo(self):
        if self.redo_l:
            self.undo_l.append(copy.deepcopy(self.work))
            self.work = self.redo_l.pop()
            self._edited()

    def _edited(self):
        self.dirty = True
        self._veil = None
        if self._method() != "automatic":
            self._timer.start()
        self._sync()
        if self.view is not None:
            self.view.canvas.update()

    def clear(self):
        self._push()
        self.work = {f: [] for f in FIELDS}
        self._edited()

    def generate(self):
        """Siril's Generate: a grid of samples, the brighter areas left out; they become placed samples."""
        self._run("grid")

    def apply(self):
        from .model import NotInEngine
        try:
            for f in FIELDS:
                self.s.set_param(self.p, "gradient", f, self.work[f] or None)
        except (ValueError, NotInEngine) as e:
            self.tv.toasted.emit(str(e))
            return
        self.dirty, self.undo_l, self.redo_l = False, [], []
        self.set_mode(None)
        self.result = None
        self.tv.request_refresh()

    def discard(self):
        self.set_mode(None)
        self._kept.pop(self.p.pid, None)
        pid = self.p
        self.p = None
        self._switch(pid)
        self.tv.request_refresh()

    # ------------------------------------------------------------ the tool (ImageCanvas.tool)
    @property
    def cursor(self):
        return Qt.BlankCursor if self.mode == "brush" else Qt.CrossCursor

    def _f(self, canvas) -> float:
        """Displayed image pixels per stack pixel."""
        W = self.small["W"] if self.small else self._width()
        if canvas.display is None:
            return 1.0
        return canvas.display.size()[0] / max(W, 1)

    def _pick(self, canvas, sx, sy) -> int | None:
        pts = self.work["sample_points"]
        if not pts:
            return None
        f = self._f(canvas)
        d = [math.hypot((x - sx) * f * canvas.scale, (y - sy) * f * canvas.scale) for x, y in pts]
        i = int(np.argmin(d))
        return i if d[i] <= PICK_PX else None

    def press(self, canvas, ix, iy, button) -> bool:
        f = self._f(canvas)
        sx, sy = ix / f, iy / f
        if self.mode == "points":
            i = self._pick(canvas, sx, sy)
            if button == Qt.RightButton:
                if i is not None:
                    self._push()
                    self.work["sample_points"].pop(i)
                    self._edited()
                return True
            if button == Qt.LeftButton:
                self._push()
                if i is None:
                    self.work["sample_points"].append([sx, sy])
                    i = len(self.work["sample_points"]) - 1
                self._grab = i
                return True
        elif self.mode == "brush" and button == Qt.LeftButton:
            self._push()
            self._last = None
            self._stroke0 = len(self.work["exclude_brush"])
            self._dab(sx, sy)
            return True
        elif self.mode == "ellipse":
            if button == Qt.RightButton:
                for j, (cx, cy, rx, ry) in enumerate(self.work["exclude_ellipses"]):
                    if ((sx - cx) / max(rx, 1)) ** 2 + ((sy - cy) / max(ry, 1)) ** 2 <= 1:
                        self._push()
                        self.work["exclude_ellipses"].pop(j)
                        self._edited()
                        break
                return True
            if button == Qt.LeftButton:
                self._ell = [(sx, sy), (sx, sy)]
                return True
        return False

    def move(self, canvas, ix, iy):
        f = self._f(canvas)
        sx, sy = ix / f, iy / f
        self._hover = (ix, iy)
        if self.mode == "points" and self._grab is not None:
            self.work["sample_points"][self._grab] = [sx, sy]
        elif self.mode == "brush":
            self._dab(sx, sy)
        elif self.mode == "ellipse" and self._ell:
            self._ell[1] = (sx, sy)
        canvas.update()

    def release(self, canvas):
        if self.mode == "points" and self._grab is not None:
            self._grab = None
            self._edited()
        elif self.mode == "brush":
            self._last = None
            self._stroke0 = None
            self._edited()
        elif self.mode == "ellipse" and self._ell:
            (ax, ay), (bx, by) = self._ell
            self._ell = None
            k = self._f(canvas) * canvas.scale
            if abs(bx - ax) * k > 6 and abs(by - ay) * k > 6:
                self._push()
                self.work["exclude_ellipses"].append([(ax + bx) / 2, (ay + by) / 2, abs(bx - ax) / 2, abs(by - ay) / 2])
                self._edited()
        canvas.update()

    def hover(self, canvas, ix, iy):
        self._hover = (ix, iy)

    def _dab(self, x, y):
        r = self.size / 2
        pts = [(x, y)]
        if self._last is not None:                       # fill the gaps of a fast stroke (as Post's brush)
            lx, ly = self._last
            n = int(math.hypot(x - lx, y - ly) / max(r / 3, 1))
            pts = [(lx + (x - lx) * k / (n + 1), ly + (y - ly) * k / (n + 1)) for k in range(1, n + 1)] + pts
        for px, py in pts:
            self.work["exclude_brush"].append([float(px), float(py), float(r), 1.0 if self.paint else 0.0])
        self._last = (x, y)

    # ------------------------------------------------------------ what is drawn over the image
    def _params(self, work=None):
        from ..stages.gradient import GradientParams
        vals = {}
        for name in GradientParams.model_fields:
            v = self.s.param_value(self.p, "gradient", name)
            if v is not None:
                vals[name] = v
        vals.update(work if work is not None else self.work)
        return GradientParams(**vals)

    def _paint(self, painter, canvas):
        if self.p is None or canvas.display is None:
            return
        f = self._f(canvas)
        dw, dh = canvas.display.size()
        if self.shown:
            self._paint_veil(painter, canvas, f, dw, dh)
            self._paint_samples(painter, canvas, f)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(OUTLINE, 1.2, Qt.DashLine))
            for cx, cy, rx, ry in self.work["exclude_ellipses"]:
                painter.drawEllipse(canvas.img_to_w(cx * f, cy * f), rx * f * canvas.scale, ry * f * canvas.scale)
        if self._ell:
            (ax, ay), (bx, by) = self._ell
            painter.setPen(QPen(OUTLINE, 1.5, Qt.DashLine))
            painter.setBrush(QColor(*VEIL))
            painter.drawEllipse(QRectF(canvas.img_to_w(ax * f, ay * f), canvas.img_to_w(bx * f, by * f)).normalized())
        if self.mode == "brush" and self._hover is not None:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1.2))
            r = self.size / 2 * f * canvas.scale
            painter.drawEllipse(canvas.img_to_w(*self._hover), r, r)
        from .imageview import tag
        if self.result is not None and self.mode is not None or self.busy:
            text = (_("Computing the preview…") if self.busy else
                    _("Preview on a reduced copy · Apply or Discard"))
            tag(painter, QPointF(10, canvas.height() - 28), text)     # bottom left: the curtain's names are on top

    def _paint_veil(self, painter, canvas, f, dw, dh):
        brush = self.work["exclude_brush"]
        done = brush if self._stroke0 is None else brush[:self._stroke0]
        if self.work["exclude_ellipses"] or done:
            from ..stages.gradient import exclusion_mask
            k = min(1.0, 1500 / max(dw, 1))
            key = (dw, dh, len(done), str(self.work["exclude_ellipses"]), id(self.work))
            if self._veil is None or self._veil[0] != key:
                work = {**self.work, "exclude_brush": done}
                m = exclusion_mask(self._params(work), int(dh * k), int(dw * k), f * k)
                rgba = np.zeros((*m.shape, 4), np.uint8)
                rgba[m] = VEIL
                img = QImage(rgba.data, m.shape[1], m.shape[0], 4 * m.shape[1], QImage.Format_RGBA8888).copy()
                self._veil = (key, img)
            painter.drawImage(QRectF(canvas.img_to_w(0, 0), canvas.img_to_w(dw, dh)), self._veil[1])
        if self._stroke0 is not None:          # the stroke in progress, drawn as it goes (the veil follows on release)
            painter.setPen(Qt.NoPen)
            for x, y, r, paint in brush[self._stroke0:]:
                painter.setBrush(QColor(*VEIL) if paint else QColor(0, 0, 0, 90))
                painter.drawEllipse(canvas.img_to_w(x * f, y * f), r * f * canvas.scale, r * f * canvas.scale)

    def _paint_samples(self, painter, canvas, f):
        if self._method() == "automatic":
            return
        s = self.samples
        pts = self.work["sample_points"]
        if pts and (s is None or not s.get("manual") or len(s["used"]) != len(pts) or self.dirty):
            used = (np.asarray(s["used"], bool) if s is not None and s.get("manual") and len(s["used"]) == len(pts)
                    else np.ones(len(pts), bool))
            s = {"x": np.array([q[0] for q in pts]), "y": np.array([q[1] for q in pts]), "used": used,
                 "box": (s or {}).get("box", 16.0)}
        if s is None:
            return
        half = max(3.0, s["box"] / 2 * f * canvas.scale)
        for x, y, u in zip(s["x"], s["y"], s["used"]):
            c = canvas.img_to_w(x * f, y * f)
            col = USED if u else UNUSED
            painter.setPen(QPen(col, 1.6 if u else 2.2))
            painter.setBrush(Qt.NoBrush if u else QColor(255, 77, 85, 70))
            painter.drawRect(QRectF(c.x() - half, c.y() - half, 2 * half, 2 * half))

    def _width(self) -> int:
        import json

        from ..api import stage_outputs
        try:
            out = stage_outputs(self.s.p, self.p.pid, "gradient")
            return int(json.loads(open(f"{out.dir}/gradient.json", encoding="utf-8").read())["width"])
        except (OSError, ValueError, KeyError, AttributeError):
            return 1

    # ------------------------------------------------------------ the preview (one thread at a time)
    def preview(self, force: bool = False):
        if self.p is None or (self._method() == "automatic" and not force):
            return
        self._run("preview")

    def _run(self, what: str):
        """One computation at a time: a request made while one runs waits, replaced by the next (only the last
        counts); the answer of an older edit is dropped."""
        if self.p is None:
            return
        self.gen += 1
        if self.busy:
            self._pending = what if what != "preview" or self._pending != "grid" else "grid"
            return
        gen, params, pid = self.gen, self._params(), self.p.pid
        small = self.small if self.small and self.small["pid"] == pid else None
        project = self.s.p
        self.busy = True
        if self.view is not None:
            self.view.canvas.update()

        def work():
            try:
                sm = small or _reduced_stack(project, pid)
                out = {"small": sm, "what": what}
                if what == "grid":
                    from ..stages.gradient import exclusion_mask, grid_points
                    ex = exclusion_mask(params, *sm["valid"].shape, sm["f"])
                    ok = sm["valid"] & ~ex if ex is not None else sm["valid"]
                    pts, kept, _box = grid_points(sm["arr"], ok, params.samples, params.object_clip)
                    out["grid"] = [[float(x) / sm["f"], float(y) / sm["f"]] for x, y in pts[kept]]
                elif what == "preview" or (what == "load" and params.method != "automatic"):
                    out.update(_preview(sm, params))
                self._done.emit(gen, out)
            except Exception as e:  # noqa: BLE001 - said in the panel, the editor goes on
                self._done.emit(gen, {"error": str(e)})
        QThreadPool.globalInstance().start(_Job(work))

    def _arrived(self, gen: int, out: dict):
        self.busy = False
        if out.get("small") is not None:
            self.small = out["small"]
        if "error" in out:
            self.tv.toasted.emit(out["error"])
        elif "grid" in out:
            self._push()
            self.work["sample_points"] = out["grid"]
            self._edited()
        elif "result" in out and gen == self.gen:
            self.result = out["result"]
            self.samples = out["samples"]
            self._show(self.result)
        if self._pending is not None:
            what, self._pending = self._pending, None
            self._run(what)
        elif gen != self.gen and out.get("what") == "preview":
            self.preview()                     # edits came during the computation: the last one
        self._sync()
        if self.view is not None:
            self.view.canvas.update()

    def _show(self, res):
        from .imageview import ViewImage
        if self.view is None or self.small is None:
            return
        try:
            v = self.view
            vi = ViewImage(res, _("Preview: gradient removed"), key=("gradient-preview", self.p.pid, self.gen),
                           camera=True, neutral=False)
            before = ViewImage(self.small["arr"], _("Stack"), key=("gradient-small", self.p.pid), camera=True,
                               neutral=False)
            v.set_image(vi, keep_view=True, keep_stretch=True)
            v.set_compare_targets([before], 0)        # the stack behind the curtain, as before the edits
            v.set_same_stretch(True)
            v.canvas.tool = self if self.mode else None
        except RuntimeError:
            self.view = None


def _reduced_stack(project, pid: int) -> dict:
    """The stack of the workflow reduced to about PREVIEW_PIXELS (block means), its covered pixels."""
    from ..api import stage_outputs
    out = stage_outputs(project, pid, "stack")
    if out is None:
        raise ValueError(_("The stack is not computed"))
    arr = np.load(f"{out.dir}/drz_full.npy", mmap_mode="r")
    wt = np.asarray(np.load(f"{out.dir}/drz_weight.npy", mmap_mode="r")[1], np.float32)
    H, W = arr.shape[1:]
    k = max(1, int(math.ceil(math.sqrt(H * W / PREVIEW_PIXELS))))
    h, w = H // k, W // k
    small = np.stack([np.asarray(arr[c, :h * k, :w * k], np.float32).reshape(h, k, w, k).mean(axis=(1, 3))
                      for c in range(3)])
    pos = wt[wt > 0]
    valid = (wt > 0.6 * float(np.median(pos))) if pos.size else wt > 0
    valid = valid[:h * k, :w * k].reshape(h, k, w, k).mean(axis=(1, 3)) > 0.99
    return {"pid": pid, "arr": small, "valid": valid, "f": 1.0 / k, "W": W, "H": H}


def _preview(sm: dict, params) -> dict:
    """The stack with the model's gradient removed (its level kept), on the reduced copy, and its samples."""
    from ..stages.gradient import read_grid, sky_model_of
    m = sky_model_of(sm["arr"], sm["valid"], params, f=sm["f"])
    h, w = sm["arr"].shape[1:]
    g = read_grid(m["grid"], m["info"], h, w)
    lvl = np.array([np.median(g[c][sm["valid"]]) for c in range(3)], np.float32)[:, None, None]
    return {"result": (sm["arr"] - g + lvl).astype(np.float32), "samples": m["samples"]}
