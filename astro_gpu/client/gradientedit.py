"""The Gradient block's samples and excluded areas, edited on the image in Inspect (``GradientEditor``).

The usual tools of the background extractions: the samples of the polynomial and of the RBF placed, moved and removed
by hand (left click adds or moves, right click removes, as in Siril), shown green when the model uses them and red when
it leaves them out (as DynamicBackgroundExtraction); the areas left out of every model painted with a brush or drawn
as ellipses, under a red veil at 50 % (Photoshop's quick mask). The brush is Post's: the same size and paint / erase
controls, the same keys ([ ] size, X paint / erase).

What the edits give is seen at once on a reduced copy of the stack (about a million pixels, whatever the sensor):
computed in a thread of its own, after the mouse is let go and 0.3 s without another edit, only the last edit counted;
the automatic model, slower, is previewed on request. Nothing is stored until Apply: the samples and areas then become
the block's settings (``sample_points``, ``exclude_ellipses``, ``exclude_brush``), the block is to compute again, and
Undo takes them back as any setting.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QObject, QRectF, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPen, QShortcut
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .i18n import _
from .widgets import ACCENT, label

#: the preview's copy of the stack holds about this many pixels
PREVIEW_PIXELS = 1_000_000
#: the preview starts this long after the last edit
DEBOUNCE_MS = 300
#: a click this close to a sample (screen px) takes it
PICK_PX = 9
USED, UNUSED = QColor("#5fbf7f"), QColor("#e5484d")
VEIL = (229, 72, 77, 128)                      # the excluded areas: red, 50 %
FIELDS = ("sample_points", "exclude_ellipses", "exclude_brush")


class _Job(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        self.fn()


class GradientEditor(QObject):
    """Edits the samples and excluded areas of the Gradient block of the workflow shown, on the main viewer of
    Inspect. ``attach(view, p)`` when the block's picture is shown, ``detach()`` otherwise; ``controls(p)`` the
    panel's section."""

    _done = Signal(int, object)                # (edit number, result of the thread)

    def __init__(self, tune):
        super().__init__(tune)
        self.tv, self.s = tune, tune.s
        self.view = None
        self.p = None
        self.mode: str | None = None           # None (looking), "points", "brush", "ellipse"
        self.paint = True
        self.size = 60.0                       # brush diameter, stack pixels
        self.work: dict[str, list] = {f: [] for f in FIELDS}
        self.dirty = False
        self.gen = 0
        self.small = None                      # {"pid", "arr", "valid", "f", "W", "H"}
        self.samples = None                    # {"x", "y", "used", "manual", "box"} in stack pixels
        self.result = None                     # the last preview (3, h, w)
        self.busy = False
        self._hover = None                     # image point under the mouse
        self._grab = None                      # the sample being moved
        self._ell = None                       # the ellipse being drawn (two corners, stack px)
        self._last = None                      # the last brush dab
        self._veil = None                      # (key, QImage)
        self._ui: dict = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self.preview)
        self._done.connect(self._arrived)
        for keys, fn in (("[", lambda: self._resize(1 / 1.25)), ("]", lambda: self._resize(1.25)),
                         ("X", self._toggle_paint), ("Escape", lambda: self.set_mode(None))):
            sc = QShortcut(QKeySequence(keys), tune)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda f=fn: self.mode is not None and f())

    # ------------------------------------------------------------ the workflow and the viewer
    def _method(self) -> str:
        return str(self.s.param_value(self.p, "gradient", "method") or "polynomial")

    def _load(self, p):
        """The stored samples and areas of ``p`` (a new workflow, or after Apply / Discard)."""
        self.p = p
        self.work = {f: [list(x) for x in (self.s.param_value(p, "gradient", f) or [])] for f in FIELDS}
        self.dirty = False
        self.result = None
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
        """The Gradient block's picture is shown in ``view`` (its tab built again: the edits carry on)."""
        if self.p is None or p.pid != self.p.pid:
            self.set_mode(None, refresh=False)
            self._load(p)
        self.view = view
        view.add_overlay(self._paint)
        view.canvas.tool = self if self.mode else None
        if self.result is not None:
            self._show(self.result)

    def detach(self):
        if self.view is not None:
            try:
                self.view.canvas.tool = None
            except RuntimeError:
                pass
        self.view = None

    # ------------------------------------------------------------ the panel's section
    def controls(self, p) -> QWidget:
        """Samples and areas: the tools, the brush, the grid, Apply / Discard, what the preview is doing."""
        if self.p is None or p.pid != self.p.pid:
            self._load(p)
        box = QFrame()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 4, 0, 6)
        v.setSpacing(6)
        method = self._method()
        sampled = method in ("polynomial", "rbf")
        tools = QHBoxLayout()
        self._ui = {}
        for key, text, tip, on in (
                ("points", "Samples", "Left click: add or move a sample · right click: remove it", sampled),
                ("brush", "Brush", "Paint the areas left out of the sky model · X paint / erase · [ ] size", True),
                ("ellipse", "Ellipse", "Drag an ellipse around an object to leave it out · right click inside: remove",
                 True)):
            b = QPushButton(_(text))
            b.setObjectName("face")
            b.setCheckable(True)
            b.setChecked(self.mode == key)
            b.setEnabled(on)
            b.setToolTip(_(tip))
            b.clicked.connect(lambda _c=False, k=key: self.set_mode(None if self.mode == k else k))
            tools.addWidget(b)
            self._ui[key] = b
        tools.addStretch(1)
        v.addLayout(tools)
        if self.mode == "brush":
            from .post3 import ContSlider
            r = QHBoxLayout()
            for text, val in (("Paint", True), ("Erase", False)):
                b = QPushButton(_(text))
                b.setObjectName("face")
                b.setCheckable(True)
                b.setChecked(self.paint == val)
                b.clicked.connect(lambda _c=False, vv=val: self._set_paint(vv))
                r.addWidget(b)
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
        clr = QPushButton(_("Clear all"))
        clr.setToolTip(_("Remove the placed samples (back to the automatic grid) and the excluded areas"))
        clr.clicked.connect(self.clear)
        row.addWidget(clr)
        if method == "automatic":
            pv = QPushButton(_("Preview"))
            pv.setToolTip(_("The automatic model on a reduced copy (a few seconds)"))
            pv.clicked.connect(lambda: self.preview(force=True))
            row.addWidget(pv)
        row.addStretch(1)
        v.addLayout(row)
        st = label(self._status(), "small", wrap=True)
        v.addWidget(st)
        self._ui["status"] = st
        act = QHBoxLayout()
        ap = QPushButton(_("Apply"))
        ap.setEnabled(self.dirty)
        ap.setToolTip(_("Keep these samples and areas as the block's settings (the block is then to compute)"))
        ap.clicked.connect(self.apply)
        ds = QPushButton(_("Discard"))
        ds.setEnabled(self.dirty)
        ds.clicked.connect(self.discard)
        act.addWidget(ap)
        act.addWidget(ds)
        act.addStretch(1)
        v.addLayout(act)
        self._ui["apply"], self._ui["discard"] = ap, ds
        return box

    def _status(self) -> str:
        n = len(self.work["exclude_ellipses"]) + (1 if self.work["exclude_brush"] else 0)
        bits = []
        if self.work["sample_points"]:
            bits.append(_("{n} samples placed").format(n=len(self.work["sample_points"])))
        if n:
            bits.append(_("areas excluded"))
        state = (_("preview being computed…") if self.busy else
                 _("preview on a reduced copy: Apply keeps it") if self.result is not None and self.dirty else
                 _("not applied") if self.dirty else "")
        hint = {"points": _("Left click: add or move · right click: remove"),
                "brush": _("Drag to paint · X paint / erase · [ ] size"),
                "ellipse": _("Drag an ellipse · right click inside: remove")}.get(self.mode, "")
        return " · ".join(x for x in (", ".join(bits), state, hint) if x)

    def _sync(self):
        """The panel's widgets as the editor is (without rebuilding the panel)."""
        try:
            for k in ("points", "brush", "ellipse"):
                if k in self._ui:
                    self._ui[k].setChecked(self.mode == k)
            if "status" in self._ui:
                self._ui["status"].setText(self._status())
            for k in ("apply", "discard"):
                if k in self._ui:
                    self._ui[k].setEnabled(self.dirty)
        except RuntimeError:                   # the panel was built again
            self._ui = {}

    # ------------------------------------------------------------ modes
    def set_mode(self, mode: str | None, refresh: bool = True):
        was = self.mode
        self.mode = mode
        if self.view is not None:
            try:
                self.view.canvas.tool = self if mode else None
                self.view.canvas.update()
            except RuntimeError:
                self.view = None
        if mode and self.small is None:
            self._load_small()
        elif mode and self.result is None and self._method() != "automatic":
            self.preview()
        if refresh and (mode == "brush") != (was == "brush"):
            self.tv.panel.refresh(self.tv.panel.node)        # the brush's controls come and go
        else:
            self._sync()

    def _set_paint(self, on: bool):
        self.paint = on
        self.tv.panel.refresh(self.tv.panel.node)

    def _toggle_paint(self):
        if self.mode == "brush":
            self._set_paint(not self.paint)

    def _resize(self, k: float):
        if self.mode == "brush":
            self.size = float(np.clip(self.size * k, 4, 1200))
            if "size" in self._ui:
                try:
                    self._ui["size"].set_value(self.size)
                except (RuntimeError, AttributeError):
                    pass
            if self.view is not None:
                self.view.canvas.update()

    # ------------------------------------------------------------ the edits
    def _edited(self):
        self.dirty = True
        self._veil = None
        if self._method() != "automatic":
            self._timer.start()
        self._sync()
        if self.view is not None:
            self.view.canvas.update()

    def clear(self):
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
        self.dirty = False
        self.set_mode(None, refresh=False)
        self.result = None
        self.tv.request_refresh()

    def discard(self):
        self.set_mode(None, refresh=False)
        self._load(self.p)
        self.tv.request_refresh()

    # ------------------------------------------------------------ the tool (ImageCanvas.tool)
    @property
    def cursor(self):
        return Qt.BlankCursor if self.mode == "brush" else Qt.CrossCursor

    def _f(self, canvas) -> float:
        """Displayed image pixels per stack pixel."""
        W = self.small["W"] if self.small else None
        if W is None or canvas.display is None:
            return 1.0
        return canvas.display.size()[0] / W

    def _pick(self, canvas, sx, sy) -> int | None:
        pts = self.work["sample_points"]
        if not pts:
            return None
        f = self._f(canvas)
        d = [math.hypot((x - sx) * f * canvas.scale, (y - sy) * f * canvas.scale) for x, y in pts]
        i = int(np.argmin(d))
        return i if d[i] <= PICK_PX else None

    def _manual(self):
        """The first edit of the samples: the automatic ones become placed ones (Siril: generate, then edit)."""
        if not self.work["sample_points"] and self.samples is not None and not self.samples["manual"]:
            u = np.asarray(self.samples["used"], bool)
            self.work["sample_points"] = [[float(x), float(y)] for x, y in
                                          zip(np.asarray(self.samples["x"])[u], np.asarray(self.samples["y"])[u])]

    def press(self, canvas, ix, iy, button) -> bool:
        f = self._f(canvas)
        sx, sy = ix / f, iy / f
        if self.mode == "points":
            i = self._pick(canvas, sx, sy)
            if button == Qt.RightButton:
                if i is not None:
                    self.work["sample_points"].pop(i)
                    self._edited()
                return True
            if button == Qt.LeftButton:
                self._manual()
                i = self._pick(canvas, sx, sy)
                if i is None:
                    self.work["sample_points"].append([sx, sy])
                    i = len(self.work["sample_points"]) - 1
                self._grab = i
                return True
        elif self.mode == "brush" and button == Qt.LeftButton:
            self._last = None
            self._dab(sx, sy)
            return True
        elif self.mode == "ellipse":
            if button == Qt.RightButton:
                for j, (cx, cy, rx, ry) in enumerate(self.work["exclude_ellipses"]):
                    if ((sx - cx) / max(rx, 1)) ** 2 + ((sy - cy) / max(ry, 1)) ** 2 <= 1:
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
            self._edited()
        elif self.mode == "ellipse" and self._ell:
            (ax, ay), (bx, by) = self._ell
            self._ell = None
            if abs(bx - ax) * self._f(canvas) * canvas.scale > 6 and abs(by - ay) * self._f(canvas) * canvas.scale > 6:
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
        self._veil = None

    # ------------------------------------------------------------ what is drawn over the image
    def _params(self):
        from ..stages.gradient import GradientParams
        vals = {}
        for name in GradientParams.model_fields:
            v = self.s.param_value(self.p, "gradient", name)
            if v is not None:
                vals[name] = v
        vals.update(self.work)
        return GradientParams(**vals)

    def _paint(self, painter, canvas):
        if self.p is None or canvas.display is None:
            return
        f = self._f(canvas) if self.small else canvas.display.size()[0] / max(self._width(), 1)
        dw, dh = canvas.display.size()
        if self.work["exclude_ellipses"] or self.work["exclude_brush"]:
            from ..stages.gradient import exclusion_mask
            k = min(1.0, 1500 / max(dw, 1))
            key = (dw, dh, len(self.work["exclude_brush"]), str(self.work["exclude_ellipses"]))
            if self._veil is None or self._veil[0] != key:
                m = exclusion_mask(self._params(), int(dh * k), int(dw * k), f * k)
                rgba = np.zeros((*m.shape, 4), np.uint8)
                rgba[m] = VEIL
                img = QImage(rgba.data, m.shape[1], m.shape[0], 4 * m.shape[1], QImage.Format_RGBA8888).copy()
                self._veil = (key, img)
            painter.drawImage(QRectF(canvas.img_to_w(0, 0), canvas.img_to_w(dw, dh)), self._veil[1])
        s = self.samples
        if self.dirty or self.mode == "points":
            pts = self.work["sample_points"]
            if pts:
                used = self._used_of(pts)
                s = {"x": np.array([q[0] for q in pts]), "y": np.array([q[1] for q in pts]), "used": used,
                     "box": (self.samples or {}).get("box", 16.0)}
        if s is not None and self._method() != "automatic":
            half = max(3.0, s["box"] / 2 * f * canvas.scale)
            for x, y, u in zip(s["x"], s["y"], s["used"]):
                c = canvas.img_to_w(x * f, y * f)
                col = USED if u else UNUSED
                painter.setPen(QPen(col, 1.4))
                fill = QColor(col)
                fill.setAlpha(50)
                painter.setBrush(fill)
                painter.drawRect(QRectF(c.x() - half, c.y() - half, 2 * half, 2 * half))
        if self._ell:
            (ax, ay), (bx, by) = self._ell
            painter.setPen(QPen(QColor(ACCENT), 1.5, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QRectF(canvas.img_to_w(ax * f, ay * f), canvas.img_to_w(bx * f, by * f)).normalized())
        if self.mode == "brush" and self._hover is not None:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1.2))
            r = self.size / 2 * f * canvas.scale
            painter.drawEllipse(canvas.img_to_w(*self._hover), r, r)

    def _used_of(self, pts) -> np.ndarray:
        """Which placed samples the model can use: those of the last preview, the new ones counted as used until the
        next."""
        s = self.samples
        if s is not None and s.get("manual") and len(s["used"]) == len(pts):
            return np.asarray(s["used"], bool)
        return np.ones(len(pts), bool)

    def _width(self) -> int:
        from ..api import stage_outputs
        try:
            out = stage_outputs(self.s.p, self.p.pid, "gradient")
            import json
            return int(json.loads(open(f"{out.dir}/gradient.json", encoding="utf-8").read())["width"])
        except (OSError, ValueError, KeyError, AttributeError):
            return 1

    # ------------------------------------------------------------ the preview (a thread of its own)
    def _load_small(self):
        self._run("load")

    def preview(self, force: bool = False):
        if self.p is None or (self._method() == "automatic" and not force):
            return
        self._run("preview")

    def _run(self, what: str):
        """Work in a thread: only the last edit counts (an older answer is dropped)."""
        if self.p is None:
            return
        self.gen += 1
        gen = self.gen
        params = self._params()
        pid = self.p.pid
        small = self.small if self.small and self.small["pid"] == pid else None
        project = self.s.p
        self.busy = True
        self._sync()

        def work():
            try:
                sm = small or _reduced_stack(project, pid)
                out = {"small": sm}
                if what == "grid":
                    from ..stages.gradient import grid_points
                    pts, kept, _box = grid_points(sm["arr"], sm["valid"], params.samples, params.object_clip)
                    out["grid"] = [[float(x) / sm["f"], float(y) / sm["f"]] for x, y in pts[kept]]
                elif what == "preview" or (what == "load" and params.method != "automatic"):
                    out.update(_preview(sm, params))
                self._done.emit(gen, out)
            except Exception as e:  # noqa: BLE001 - said in the panel, the editor goes on
                self._done.emit(gen, {"error": str(e)})
        QThreadPool.globalInstance().start(_Job(work))

    def _arrived(self, gen: int, out: dict):
        if out.get("small") is not None:
            self.small = out["small"]
        if gen != self.gen:                    # an older edit's answer
            return
        self.busy = False
        if "error" in out:
            self.tv.toasted.emit(out["error"])
        if "grid" in out:
            self.work["sample_points"] = out["grid"]
            self._edited()
            return
        if "result" in out:
            self.result = out["result"]
            self.samples = out["samples"]
            self._show(self.result)
        self._sync()

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
            v.set_compare_targets([before], None)
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
