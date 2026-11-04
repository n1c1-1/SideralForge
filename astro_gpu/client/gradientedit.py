"""The Gradient block's samples and excluded areas, edited on the image in Inspect (``GradientEditor``).

The usual tools of the background extractions: the samples of the polynomial and of the RBF on a generated grid, then
placed, moved and removed by hand (left click adds or moves, right click removes, as in Siril), outlined in cyan when
the model uses them and in red when it leaves them out (as DynamicBackgroundExtraction), in white while not computed;
the areas left out of every model painted with a brush or drawn as ellipses, under a red veil at 50 % (Photoshop's
quick mask). The brush is Post's: the same size and paint / erase controls, the same keys ([ ] size, X paint / erase),
the line of the gestures under the image. Space held, the middle button or (with the brush) a right drag move the
image, as in Post and Photoshop; M or Marks hides what is drawn over the image.

Each edit is a setting of the block, saved as soon as the mouse is let go (``sample_points``, ``exclude_ellipses``,
``exclude_brush``, in the workflows the panel's changes go to), as Post's mask and Lightroom's: one step of the
window's Undo (Ctrl+Z), the block Outdated (its previous result shown, Run to here), the result it had back with an
Undo, without computing. What the edits give is seen on a reduced copy of the stack (about a million pixels, whatever
the sensor), with the stack before them behind the curtain: one computation at a time in a thread of its own, 0.3 s
after the last edit, the last edit only (the ones in between are skipped); the automatic model, slower, is previewed on
request. No sample placed: the block chooses its own at each run (the polynomial: small blocks of sky all over the
field, the RBF: the generated grid), drawn as the last run used them; Generate the grid places editable ones.
"""
from __future__ import annotations

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
#: an ellipse's handles: their size, and how far above the ellipse the rotation one is (screen px)
HANDLE_PX, ROT_PX = 5, 26
#: the keys of the tools: the image takes them before the window's and Inspect's shortcuts (Escape goes back to the
#: graph otherwise)
TOOL_KEYS = (Qt.Key_Escape, Qt.Key_Delete, Qt.Key_Backspace, Qt.Key_M, Qt.Key_X, Qt.Key_BracketLeft,
             Qt.Key_BracketRight)


def _ell(e) -> tuple[float, float, float, float, float]:
    """An ellipse of exclude_ellipses: centre, half axes, angle (degrees; an earlier one has none)."""
    return float(e[0]), float(e[1]), float(e[2]), float(e[3]), float(e[4]) if len(e) > 4 else 0.0


def _local(e, x, y) -> tuple[float, float]:
    """(x, y) in the frame of the ellipse: along its first axis, along its second."""
    cx, cy, _rx, _ry, a = _ell(e)
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    dx, dy = x - cx, y - cy
    return dx * c + dy * s, -dx * s + dy * c


def _world(e, u, v) -> tuple[float, float]:
    cx, cy, _rx, _ry, a = _ell(e)
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return cx + u * c - v * s, cy + u * s + v * c


def _inside(e, x, y) -> bool:
    _cx, _cy, rx, ry, _a = _ell(e)
    u, v = _local(e, x, y)
    return (u / max(rx, 1e-6)) ** 2 + (v / max(ry, 1e-6)) ** 2 <= 1


class _Job(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        self.fn()


class GradientEditor(QObject):
    """Edits the samples and excluded areas of the Gradient block of the workflow shown, on the main viewer of
    Inspect. ``attach(view, p)`` when the block's picture is shown, ``detach()`` otherwise; ``controls(p)`` the
    panel's section. It is the canvas's ``tool`` while a tool is on. ``gesture``: the mouse is down on the image
    (Inspect is not built again under it)."""

    _done = Signal(int, object)                # (request number, result of the thread)

    def __init__(self, tune):
        super().__init__(tune)
        self.tv, self.s = tune, tune.s
        self.view = None
        self.p = None
        self.mode: str | None = None           # None (looking), "points", "brush", "ellipse"
        self.paint = True
        self.size = 60.0                       # brush diameter, stack pixels
        self.shown = True                      # what is drawn over the image (M, Marks)
        self.work: dict[str, list] = {f: [] for f in FIELDS}
        self.done = False                      # the block is computed with these settings
        self.density = None                    # points of the generated grid along the shorter side (None: the setting)
        self.gesture = False
        self.gen = 0
        self.small = None                      # {"pid", "arr", "valid", "f", "W", "H"}
        self.samples = None                    # {"x", "y", "used", "manual", "box"} in stack pixels
        self.result = None                     # the last preview (3, h, w)
        self.busy = False
        self._dims = None                      # (pid, W, H) of the stack
        self._pending = None                   # the request waiting for the one running
        self._hover = None
        self._grab = None
        self._ell = None                       # the ellipse being drawn (two corners, stack px)
        self._sel = None                       # the ellipse selected (its handles shown)
        self._edit_ell = None                  # (what is dragged: move / rx / ry / rot, the start)
        self._last = None
        self._stroke0 = None                   # where the brush stroke in progress starts in exclude_brush
        self._extra = None                     # the click that asked for the grid: a sample added to it
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

    def _settings(self) -> dict[str, list]:
        return {f: [[float(x) for x in e] for e in (self.s.param_value(self.p, "gradient", f) or [])] for f in FIELDS}

    def _load(self, p):
        """The edits as the workflow's settings have them (another workflow, an Undo, a run, the project opened
        again); the samples drawn: the preview's, else the ones the last run used."""
        if self.p is None or p.pid != self.p.pid:
            if self.p is not None:
                self.set_mode(None)
            self.p = p
            self.result, self.samples, self._sel, self.density, self._veil = None, None, None, None, None
            if self.small is not None and self.small["pid"] != p.pid:
                self.small = None
        new = self._settings()
        if new != self.work:                   # an Undo, or the settings changed elsewhere
            self.work = new
            self._veil = None
            if self._sel is not None and self._sel >= len(new["exclude_ellipses"]):
                self._sel = None
            if self.result is not None and self._method() != "automatic":
                self._timer.start()
        self.done = self.s.state(p, "gradient") == "done"
        if self.done:
            self.result = None                 # the computed result is what is shown
        if self.result is None:
            self.samples = self._computed_samples()

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
        """The Gradient block's picture is shown in ``view`` (its tab built again: the tool stays on)."""
        self._load(p)
        self.view = view
        view.add_overlay(self._paint)
        view.canvas.tool = self if self.mode else None
        self._hint()
        if self.result is not None:
            self._show(self.result)

    def detach(self):
        # a gesture cut by a rebuild (a refresh asked before the mouse went down): its release will not come
        self.gesture = False
        self._grab = self._stroke0 = self._ell = self._edit_ell = self._last = None
        if self.view is not None:
            try:
                self.view.canvas.tool = None
                self.view.tool_hint.hide()
            except RuntimeError:
                pass
        self.view = None

    # ------------------------------------------------------------ the panel's section
    def controls(self, p) -> QWidget:
        """Samples and excluded areas: the tools, the brush, the grid, undo, Run to here, the state. Built whole (the
        brush's row is there in every mode, enabled with the brush): a tool switched on or off never moves the panel."""
        self._load(p)
        from PySide6.QtWidgets import QSpinBox

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
            g.setToolTip(_("Editable samples on a regular grid, the brighter areas left out (Object rejection), in "
                           "place of the ones the block chooses at each run; then move or remove them by hand"))
            g.clicked.connect(self.generate)
            row.addWidget(g)
            dens = QSpinBox()
            dens.setRange(4, 64)
            dens.setValue(self._density())
            dens.setSuffix(" " + _("per side"))
            dens.setToolTip(_("Samples of the grid along the shorter side of the image"))
            t = QTimer(box)
            t.setSingleShot(True)
            t.setInterval(500)
            t.timeout.connect(lambda: self._set_density(dens.value()))
            dens.valueChanged.connect(lambda _v: t.start())
            row.addWidget(dens)
            row.addStretch(1)
            v.addLayout(row)
            row = QHBoxLayout()                # the grid on its line, the undo and removal on the next
        for key, text, tip, fn in (("undo", "↶", "Undo (Ctrl+Z)", self.undo), ("redo", "↷", "Redo (Ctrl+Y)", self.redo)):
            b = QPushButton(text)
            b.setToolTip(_(tip))
            b.clicked.connect(fn)
            row.addWidget(b)
            self._ui[key] = b
        clr = QPushButton(_("Remove all"))
        clr.setToolTip(_("Remove the placed samples and the excluded areas: the block chooses its samples again "
                         "(Undo brings them back)"))
        clr.clicked.connect(self.clear)
        row.addWidget(clr)
        if method == "automatic":
            pv = QPushButton(_("Preview"))
            pv.setToolTip(_("The automatic model on a reduced copy (a few seconds)"))
            pv.clicked.connect(lambda: self.preview(force=True))
            row.addWidget(pv)
        row.addStretch(1)
        mk = QPushButton(_("Marks"))
        mk.setObjectName("face")
        mk.setCheckable(True)
        mk.setChecked(self.shown)
        mk.setToolTip(_("Show the samples and the excluded areas over the image (M)"))
        mk.toggled.connect(self._set_shown)
        row.addWidget(mk)
        self._ui["marks"] = mk
        v.addLayout(row)
        if not self.done and self.s.state(p, "gradient") != "running":
            run = self.tv._run_button(p, "fond")     # the edits are saved; computing them is the block's run
            v.addWidget(run, 0, Qt.AlignLeft)
        st = label("", "small", wrap=True)
        v.addWidget(st)
        self._ui["status"] = st
        self._sync()
        return box

    def _status(self) -> str:
        bits = []
        pts, s = self.work["sample_points"], self.samples
        if self._method() == "automatic":
            pass
        elif pts:
            bits.append(_("{n} samples placed").format(n=len(pts)))
        elif s is not None and not s.get("manual"):
            bits.append(_("No sample placed: the block chooses its own at each run ({n} of sky the last time)").format(
                n=int(np.sum(s["used"]))))
        else:
            bits.append(_("No sample placed: the block chooses its own at each run"))
        n = len(self.work["exclude_ellipses"]) + (1 if any(d[3] for d in self.work["exclude_brush"]) else 0)
        if n:
            bits.append(_("{n} excluded areas").format(n=n) if n > 1 else _("1 excluded area"))
        if not self.done:
            bits.append(_("saved: the block is to compute again"))
        return " · ".join(bits)

    def _hint(self):
        """The gestures of the tool on, on the line under the image (as Post's)."""
        if self.view is None:
            return
        t = {"points": _("Samples: left click add or move (none placed: the grid first) · right click remove · Space "
                         "or middle button: move the image · M hide · Ctrl+Z undo"),
             "brush": _("Brush: drag to paint · X paint / erase · [ ] size · right drag: move the image · M hide · "
                        "Ctrl+Z undo"),
             "ellipse": _("Ellipse: drag to draw · click inside: select and move · handles: size and rotation · "
                          "Delete or right click inside: remove · Space: move the image · Ctrl+Z undo")}.get(self.mode, "")
        try:
            self.view.tool_hint.setText(t)
            self.view.tool_hint.setVisible(bool(t))
        except RuntimeError:
            self.view = None

    def _sync(self):
        """The panel's widgets as the editor is (without building the panel again)."""
        if self.p is None:
            return
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
            if "marks" in self._ui:
                self._ui["marks"].setChecked(self.shown)
            if "status" in self._ui:
                self._ui["status"].setText(self._status())
            if "undo" in self._ui:
                self._ui["undo"].setEnabled(self.s.can_undo())
                self._ui["redo"].setEnabled(self.s.can_redo())
        except RuntimeError:                   # the panel was built again
            self._ui = {}

    # ------------------------------------------------------------ modes
    def set_mode(self, mode: str | None):
        self.mode = mode
        self._sel = None
        if self.view is not None:
            try:
                c = self.view.canvas
                c.tool = self if mode else None
                if mode:
                    c.setFocus()               # the keys of the tool go to the image
                c.update()
            except RuntimeError:
                self.view = None
        if mode and self.result is None and not self.done and self._method() != "automatic":
            self.preview()                     # Outdated: what the settings give, at once
        elif mode and self.small is None:
            self._run("load")
        self._hint()
        self._sync()

    def _density(self) -> int:
        return int(self.density or self.s.param_value(self.p, "gradient", "samples") or 16)

    def _set_density(self, n: int):
        """The grid's points along the shorter side, then a new grid (an edit: Undo brings the one before back)."""
        if n != self._density():
            self.density = int(n)
            self.generate()

    def _set_paint(self, on: bool):
        self.paint = on
        self._sync()

    def _set_shown(self, on: bool):
        self.shown = bool(on)
        self._sync()
        if self.view is not None:
            self.view.canvas.update()

    def wants(self, e) -> bool:
        """Whether the image takes this key before the shortcuts (the canvas asks, ShortcutOverride)."""
        return e.key() in TOOL_KEYS and not e.modifiers() & (Qt.ControlModifier | Qt.AltModifier)

    def key(self, e) -> bool:
        """The keys of the tool, from the canvas (it has the focus while a tool is on). Ctrl+Z / Ctrl+Y are the
        window's Undo / Redo: each edit is one of its steps."""
        k = e.key()
        if e.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
            return False
        if k == Qt.Key_BracketLeft and self.mode == "brush":
            self._resize(1 / 1.25)
        elif k == Qt.Key_BracketRight and self.mode == "brush":
            self._resize(1.25)
        elif k == Qt.Key_X and self.mode == "brush":
            self._set_paint(not self.paint)
        elif k == Qt.Key_M:
            self._set_shown(not self.shown)
        elif k in (Qt.Key_Delete, Qt.Key_Backspace) and self.mode == "ellipse" and self._sel is not None:
            self.work["exclude_ellipses"].pop(self._sel)
            self._sel = None
            self._commit()
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

    # ------------------------------------------------------------ the edits: settings of the block
    def _commit(self):
        """The edits into the block's settings (in the workflows the panel's changes go to): one step of the window's
        Undo; the block is then Outdated. Inspect is built again (as for any setting), the tool stays on."""
        self._veil = None
        vals = {f: self.work[f] for f in FIELDS}
        vals["exclude_brush"] = [[round(x, 1), round(y, 1), round(r, 1), float(d)] for x, y, r, d in vals["exclude_brush"]]
        vals["sample_points"] = [[round(x, 1), round(y, 1)] for x, y in vals["sample_points"]]
        if vals == self._settings():
            self._sync()
            return
        self.work = vals
        paths = self.s.scope_paths()
        if self.p.pid not in [q.pid for q in paths]:
            paths = [self.p]
        from .model import NotInEngine
        try:
            self.s.apply_change(paths, {"kind": "params",
                                        "items": [("gradient", f, vals[f] or None) for f in FIELDS]})
        except (ValueError, NotInEngine) as e:
            self.tv.toasted.emit(_(str(e)))
            self.work = self._settings()
        self.done = self.s.state(self.p, "gradient") == "done"
        if self._method() != "automatic":
            self._timer.start()
        self._sync()
        if self.view is not None:
            self.view.canvas.update()

    def undo(self):
        w = self.tv.window()
        (w.undo if hasattr(w, "undo") else self.s.undo)()

    def redo(self):
        w = self.tv.window()
        (w.redo if hasattr(w, "redo") else self.s.redo)()

    def clear(self):
        self.work = {f: [] for f in FIELDS}
        self._sel = None
        self._commit()

    def generate(self):
        """Siril's Generate: a grid of samples, the brighter areas left out; they become placed samples."""
        self._run("grid")

    # ------------------------------------------------------------ the tool (ImageCanvas.tool)
    @property
    def cursor(self):
        return Qt.BlankCursor if self.mode == "brush" else Qt.CrossCursor

    def _stack_size(self) -> tuple[int, int]:
        """(W, H) of the stack: the reduced copy's, else read from the stack (the block may not be computed)."""
        if self.small is not None and self.small["pid"] == self.p.pid:
            return self.small["W"], self.small["H"]
        if self._dims is None or self._dims[0] != self.p.pid:
            from ..api import stage_outputs
            try:
                out = stage_outputs(self.s.p, self.p.pid, "stack")
                H, W = np.load(f"{out.dir}/drz_full.npy", mmap_mode="r").shape[1:]
            except (OSError, ValueError, AttributeError):
                return 1, 1
            self._dims = (self.p.pid, int(W), int(H))
        return self._dims[1], self._dims[2]

    def _f(self, canvas) -> float:
        """Displayed image pixels per stack pixel."""
        if canvas.display is None:
            return 1.0
        return canvas.display.size()[0] / max(self._stack_size()[0], 1)

    def _pick(self, canvas, sx, sy) -> int | None:
        pts = self.work["sample_points"]
        if not pts:
            return None
        f = self._f(canvas)
        d = [math.hypot((x - sx) * f * canvas.scale, (y - sy) * f * canvas.scale) for x, y in pts]
        i = int(np.argmin(d))
        return i if d[i] <= PICK_PX else None

    def press(self, canvas, ix, iy, button) -> bool:
        took = self._press(canvas, ix, iy, button)
        self.gesture = took and button == Qt.LeftButton
        return took

    def _press(self, canvas, ix, iy, button) -> bool:
        f = self._f(canvas)
        sx, sy = ix / f, iy / f
        if self.mode == "points":
            i = self._pick(canvas, sx, sy)
            if button == Qt.RightButton:
                if i is not None:
                    self.work["sample_points"].pop(i)
                    self._commit()
                return True
            if button == Qt.LeftButton:
                if not self.work["sample_points"]:
                    self._extra = [sx, sy]     # none placed: the grid first, this sample with it
                    self.generate()
                    return True
                if i is None:
                    self.work["sample_points"].append([sx, sy])
                    i = len(self.work["sample_points"]) - 1
                self._grab = i
                return True
        elif self.mode == "brush" and button == Qt.LeftButton:
            self._last = None
            self._stroke0 = len(self.work["exclude_brush"])
            self._dab(sx, sy)
            return True
        elif self.mode == "ellipse":
            ells = self.work["exclude_ellipses"]
            if button == Qt.RightButton:
                for j in reversed(range(len(ells))):
                    if _inside(ells[j], sx, sy):
                        ells.pop(j)
                        self._sel = None
                        self._commit()
                        break
                return True
            if button == Qt.LeftButton:
                # the handles of the selected ellipse, then inside an ellipse (select and move), else a new one
                h = self._handle(canvas, ix, iy)
                if h is not None:
                    self._edit_ell = (h, list(ells[self._sel]), (sx, sy))
                    return True
                hit = next((j for j in reversed(range(len(ells))) if _inside(ells[j], sx, sy)), None)
                if hit is not None:
                    self._sel = hit
                    self._edit_ell = ("move", list(ells[hit]), (sx, sy))
                    canvas.update()
                    return True
                self._sel = None
                self._ell = [(sx, sy), (sx, sy)]
                return True
        return False

    def _handles(self, canvas, f) -> dict[str, tuple[float, float]]:
        """The selected ellipse's handles, in screen px: its four axis ends and the rotation one above it."""
        if self._sel is None or self._sel >= len(self.work["exclude_ellipses"]):
            return {}
        e = self.work["exclude_ellipses"][self._sel]
        _cx, _cy, rx, ry, _a = _ell(e)
        k = f * canvas.scale
        out = {}
        for name, (u, v) in (("rx", (rx, 0)), ("-rx", (-rx, 0)), ("ry", (0, ry)), ("-ry", (0, -ry)),
                             ("rot", (0, -ry - ROT_PX / max(k, 1e-9)))):
            x, y = _world(e, u, v)
            p = canvas.img_to_w(x * f, y * f)
            out[name] = (p.x(), p.y())
        return out

    def _handle(self, canvas, ix, iy) -> str | None:
        f = self._f(canvas)
        p = canvas.img_to_w(ix, iy)
        for name, (hx, hy) in self._handles(canvas, f).items():
            if math.hypot(p.x() - hx, p.y() - hy) <= HANDLE_PX + 4:
                return name
        return None

    def move(self, canvas, ix, iy):
        f = self._f(canvas)
        sx, sy = ix / f, iy / f
        self._hover = (ix, iy)
        if self.mode == "points" and self._grab is not None:
            self.work["sample_points"][self._grab] = [sx, sy]
        elif self.mode == "brush" and self._stroke0 is not None:
            self._dab(sx, sy)
        elif self.mode == "ellipse" and self._edit_ell is not None:
            what, e0, (x0, y0) = self._edit_ell
            cx, cy, rx, ry, a = _ell(e0)
            if what == "move":
                e = [cx + sx - x0, cy + sy - y0, rx, ry, a]
            elif what == "rot":                  # the handle above the ellipse follows the mouse around its centre
                e = [cx, cy, rx, ry, math.degrees(math.atan2(sy - cy, sx - cx)) + 90.0]
            else:
                u, v = _local(e0, sx, sy)
                e = [cx, cy, max(abs(u), 2.0), ry, a] if what.endswith("rx") else [cx, cy, rx, max(abs(v), 2.0), a]
            self.work["exclude_ellipses"][self._sel] = e
            self._veil = None
        elif self.mode == "ellipse" and self._ell:
            self._ell[1] = (sx, sy)
        canvas.update()

    def release(self, canvas):
        self.gesture = False
        edited = False
        if self.mode == "points" and self._grab is not None:
            self._grab = None
            edited = True
        elif self.mode == "brush" and self._stroke0 is not None:
            self._last = None
            self._stroke0 = None
            edited = True
        elif self.mode == "ellipse" and self._edit_ell is not None:
            self._edit_ell = None
            edited = True
        elif self.mode == "ellipse" and self._ell:
            (ax, ay), (bx, by) = self._ell
            self._ell = None
            k = self._f(canvas) * canvas.scale
            if abs(bx - ax) * k > 6 and abs(by - ay) * k > 6:
                self.work["exclude_ellipses"].append([(ax + bx) / 2, (ay + by) / 2, abs(bx - ax) / 2, abs(by - ay) / 2,
                                                      0.0])
                self._sel = len(self.work["exclude_ellipses"]) - 1     # selected: its handles at once
                edited = True
        if edited:
            self._commit()
        if self.tv._dirty:                     # a refresh held back while the mouse was down
            self.tv.request_refresh()
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
            for j, e in enumerate(self.work["exclude_ellipses"]):
                cx, cy, rx, ry, a = _ell(e)
                sel = j == self._sel and self.mode == "ellipse"
                painter.save()
                painter.translate(canvas.img_to_w(cx * f, cy * f))
                painter.rotate(a)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(OUTLINE, 1.6 if sel else 1.2, Qt.SolidLine if sel else Qt.DashLine))
                painter.drawEllipse(QPointF(0, 0), rx * f * canvas.scale, ry * f * canvas.scale)
                painter.restore()
            hs = self._handles(canvas, f) if self.mode == "ellipse" else {}
            if hs:
                painter.setPen(QPen(OUTLINE, 1.2))
                painter.drawLine(QPointF(*hs["-ry"]), QPointF(*hs["rot"]))
                painter.setBrush(QColor("#15171c"))
                for name, (hx, hy) in hs.items():
                    if name == "rot":
                        painter.drawEllipse(QPointF(hx, hy), HANDLE_PX + 1, HANDLE_PX + 1)
                    else:
                        painter.drawRect(QRectF(hx - HANDLE_PX, hy - HANDLE_PX, 2 * HANDLE_PX, 2 * HANDLE_PX))
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
        if self.result is not None or self.busy:      # the reduced preview is never taken for the result
            text = _("Computing the preview…") if self.busy else _("Reduced preview · not computed yet")
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
        """The placed samples, coloured as the last computation (run or preview) used them, white while not computed;
        none placed: the ones the block chose at its last run."""
        if self._method() == "automatic":
            return
        s, pts = self.samples, self.work["sample_points"]
        if pts:
            P = np.asarray(pts, float).reshape(-1, 2)
            same = (s is not None and s.get("manual") and len(s["x"]) == len(P)
                    and np.allclose(np.c_[s["x"], s["y"]], P, atol=0.6))
            used = list(s["used"]) if same else [None] * len(P)
            W, H = self._stack_size()
            box = s["box"] if s is not None else max(4, int(min(W, H) / self._density() / 2))
            rows = zip(P[:, 0], P[:, 1], used)
        elif s is not None and not s.get("manual"):
            box, rows = s["box"], zip(s["x"], s["y"], s["used"])
        else:
            return
        half = max(3.0, box / 2 * f * canvas.scale)
        for x, y, u in rows:
            c = canvas.img_to_w(x * f, y * f)
            col = OUTLINE if u is None else USED if u else UNUSED
            painter.setPen(QPen(col, 2.2 if u is False else 1.6))
            painter.setBrush(QColor(255, 77, 85, 70) if u is False else Qt.NoBrush)
            painter.drawRect(QRectF(c.x() - half, c.y() - half, 2 * half, 2 * half))

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
        gen, params, pid, n = self.gen, self._params(), self.p.pid, self._density()
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
                    pts, kept, _box = grid_points(sm["arr"], ok, n, params.object_clip)
                    out["grid"] = [[float(x) / sm["f"], float(y) / sm["f"]] for x, y in pts[kept]]
                elif what == "preview":
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
            self.work["sample_points"] = out["grid"] + ([self._extra] if self._extra else [])
            self._extra = None
            self._commit()
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
