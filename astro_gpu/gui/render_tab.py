"""Final adjustment tab: interactive GPU rendering with presets, snapshots, comparison, clipping view and export."""
from __future__ import annotations

import json
import time

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..render.settings import PRESETS, SECTIONS, RenderSettings
from . import style
from .widgets import Debouncer, HelpBox, Histogram, ImageView, ParamForm, scroll, to_qimage

OK_SYM, WARN_SYM = "✔", "⚠"


class RenderWorker(QObject):
    """Owns the GPU engine in its own thread. Requests are coalesced: only the latest one is rendered."""

    ready = Signal(int, object, float, object)        # seq, {"final","starless","stars"} uint8 arrays, scale, info
    failed = Signal(str)
    loaded = Signal(object)                            # engine metadata
    exportProgress = Signal(float, str)
    exported = Signal(object)

    def __init__(self, prep_dir, calibrated_white, device="cuda:0"):
        super().__init__()
        self.prep_dir, self.calibrated_white, self.device = prep_dir, calibrated_white, device
        self.engine = None
        self.pending = None
        self._scheduled = False

    @Slot()
    def load(self):
        try:
            from ..render.engine import RenderEngine
            self.engine = RenderEngine(self.prep_dir, device=self.device, calibrated_white=self.calibrated_white)
            self.loaded.emit({"H": self.engine.H, "W": self.engine.W, "crop": self.engine.crop, "SC": self.engine.SC})
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"cannot load the rendering layers: {e}")

    @Slot(int, dict, float)
    def request(self, seq, values, scale):
        # requests queued while rendering are all delivered before the scheduled drain: only the last one survives
        self.pending = (seq, values, scale)
        if not self._scheduled:
            self._scheduled = True
            QTimer.singleShot(0, self._drain)

    def _drain(self):
        import torch
        self._scheduled = False
        if self.pending is None or self.engine is None:
            return
        seq, values, scale = self.pending
        self.pending = None
        try:
            t0 = time.time()
            lay = self.engine.render(RenderSettings(**values), scale)
            x0, x1, y0, y1 = self.engine.crop
            f = int(round(1 / lay.scale))
            crop = np.s_[:, y0 // f:y1 // f, x0 // f:x1 // f]
            out = {k: (getattr(lay, k)[crop].clamp(0, 1) * 255 + 0.5).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                   for k in ("final", "starless", "stars")}
            info = dict(lay.info)
            info["render_ms"] = int(1000 * (time.time() - t0))
            if torch.cuda.is_available():
                info["gpu_mb"] = int(torch.cuda.max_memory_allocated() / 2 ** 20)
            self.ready.emit(seq, out, lay.scale, info)
        except Exception as e:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()[-800:]}")

    @Slot(object, dict, dict)
    def export(self, project, values, export_values):
        from ..render.export import ExportSettings, export
        try:
            meta = export(project, self.engine, RenderSettings(**values), ExportSettings(**export_values),
                          lambda f, m: self.exportProgress.emit(f, m))
            self.exported.emit(meta)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"export failed: {e}")


class RenderTab(QWidget):
    _request = Signal(int, dict, float)
    _load = Signal()
    _export = Signal(object, dict, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.advanced = False
        self.values: dict = {}
        self.history: list[dict] = []
        self.future: list[dict] = []
        self.snapshot = None
        self.images: dict = {}
        self.info: dict = {}
        self.cur_scale = 0.0
        self.seq = 0
        self.thread = None
        self.worker = None
        self.meta = None
        self.ref_img = None
        self.prog = None
        self._export_name = ""

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Horizontal)
        lay.addWidget(split)

        # ---------------- left: controls
        left = QWidget()
        lv = QVBoxLayout(left)
        row = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS))
        apply_p = QPushButton("Apply preset")
        apply_p.clicked.connect(self.apply_preset)
        row.addWidget(self.preset, 1)
        row.addWidget(apply_p)
        lv.addLayout(row)
        row2 = QHBoxLayout()
        self.undo_b, self.redo_b = QPushButton("↶ Undo"), QPushButton("↷ Redo")
        self.undo_b.clicked.connect(self.undo)
        self.redo_b.clicked.connect(self.redo)
        self.undo_b.setShortcut("Ctrl+Z")
        self.redo_b.setShortcut("Ctrl+Y")
        snap = QPushButton("Snapshot")
        snap.setToolTip("Keep the current rendering as 'before' for the split comparison")
        snap.clicked.connect(self.take_snapshot)
        reset = QPushButton("Reset all")
        reset.clicked.connect(lambda: self.set_all({}))
        for b in (self.undo_b, self.redo_b, snap, reset):
            row2.addWidget(b)
        lv.addLayout(row2)
        self.form_host = QVBoxLayout()
        lv.addLayout(self.form_host, 1)
        split.addWidget(left)

        # ---------------- centre: viewer
        centre = QWidget()
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.layer_group = QButtonGroup(self)
        for i, (k, lab) in enumerate((("final", "Final"), ("starless", "Starless"), ("stars", "Stars"))):
            b = QToolButton()
            b.setText(lab)
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.setProperty("layer", k)
            self.layer_group.addButton(b, i)
            bar.addWidget(b)
        self.layer_group.buttonClicked.connect(lambda _: self.show_current())
        bar.addSpacing(12)
        self.compare = QComboBox()
        self.compare.addItems(["No comparison", "Compare with snapshot", "Compare with reference image"])
        self.compare.currentIndexChanged.connect(lambda _: self.show_current())
        bar.addWidget(self.compare)
        self.clip_b = QToolButton()
        self.clip_b.setText("Show clipping")
        self.clip_b.setCheckable(True)
        self.clip_b.setToolTip("Blue: pure black   Red: pure white   Amber: nebula above 92 % (detail loss)")
        self.clip_b.toggled.connect(lambda _: self.show_current())
        bar.addWidget(self.clip_b)
        bar.addStretch(1)
        fit_b, one_b = QToolButton(), QToolButton()
        fit_b.setText("Fit")
        one_b.setText("100 %")
        bar.addWidget(fit_b)
        bar.addWidget(one_b)
        self.status = QLabel("")
        self.status.setObjectName("muted")
        bar.addWidget(self.status)
        cv.addLayout(bar)
        self.view = ImageView()
        self.view.placeholder = "Run the processing up to 'Rendering preparation' to adjust the final image"
        fit_b.clicked.connect(self.view.fit)
        one_b.clicked.connect(self.view.one_to_one)
        fit_b.setShortcut("F")
        one_b.setShortcut("1")
        fit_b.setToolTip("Fit the image in the window (F)")
        one_b.setToolTip("One image pixel per screen pixel (1)")
        cv.addWidget(self.view, 1)
        split.addWidget(centre)

        # ---------------- right: histogram, checks, help, export
        right = QWidget()
        rv = QVBoxLayout(right)
        self.hist = Histogram()
        rv.addWidget(self.hist)
        self.checks = QLabel("")
        self.checks.setWordWrap(True)
        self.checks.setTextFormat(Qt.RichText)
        rv.addWidget(self.checks)
        self.help = HelpBox()
        self.help.show_text("Final adjustment", "Every slider re-renders the image on the GPU from the calibrated "
                            "layers. Hover or click a setting to read what it does. The preview is computed at reduced "
                            "resolution while you move a slider, then at full resolution when you zoom in.")
        rv.addWidget(self.help, 1)
        exp = QPushButton("Export...")
        exp.setObjectName("primary")
        exp.clicked.connect(self.export_dialog)
        rv.addWidget(exp)
        split.addWidget(right)
        split.setSizes([430, 900, 300])

        self.debounce = Debouncer(60, self.render_preview, self)
        self.refine = Debouncer(700, self.render_refine, self)
        self.saver = Debouncer(1500, self.save_settings, self)
        self.view.viewChanged.connect(self.refine.poke)
        self.form = None
        self.form_scroll = None

    # ------------------------------------------------------------------ project
    def set_project(self, project, advanced: bool):
        self.project = project
        self.advanced = advanced
        self._stop_worker()
        self.values = dict(project.config.render.get("settings", {}))
        self.history, self.future = [], []
        self.images = {}
        self.ref_img = None
        self.view.set_image(None)
        self._build_form()
        prep = project.stage_dir("render_prep") / "manifest.json"
        if not prep.exists() or json.loads(prep.read_text()).get("status") != "done":
            self.status.setText("not ready")
            return
        col = project.stage_dir("color") / "manifest.json"
        white = json.loads(col.read_text())["params"].get("white_temp", 5778.0) if col.exists() else 5778.0
        self.thread = QThread()
        self.worker = RenderWorker(project.stage_dir("render_prep"), white, project.config.devices.get("main", "cuda:0"))
        self.worker.moveToThread(self.thread)
        self._request.connect(self.worker.request)
        self._load.connect(self.worker.load)
        self._export.connect(self.worker.export)
        self.worker.ready.connect(self.on_ready)
        self.worker.failed.connect(self.on_failed)
        self.worker.loaded.connect(self.on_loaded)
        self.worker.exportProgress.connect(self.on_export_progress)
        self.worker.exported.connect(self.on_exported)
        self.thread.start()
        self.status.setText("loading layers...")
        self._load.emit()

    def _stop_worker(self):
        if self.thread:
            for sig in (self._request, self._load, self._export):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            self.thread.quit()
            self.thread.wait(5000)
            self.thread = None
            self.worker = None

    def on_loaded(self, meta):
        self.meta = meta
        self.render_preview()

    def on_failed(self, msg):
        if self.prog:
            self.prog.close()
            self.prog = None
        self.status.setText("error")
        QMessageBox.warning(self, "Rendering", msg)

    def set_advanced(self, adv: bool):
        self.advanced = adv
        self._build_form()

    def _build_form(self):
        if self.form_scroll is not None:
            self.form_host.removeWidget(self.form_scroll)
            self.form_scroll.hide()
            self.form_scroll.setParent(None)
            self.form_scroll.deleteLater()
        s = RenderSettings(**self.values)
        self.form = ParamForm(RenderSettings, s.model_dump(), {k: "user" for k in self.values}, advanced=self.advanced,
                              sliders=True, sections=SECTIONS)
        self.form.changed.connect(self.on_change)
        self.form.focused.connect(self.on_focus)
        self.form_scroll = scroll(self.form)
        self.form_host.addWidget(self.form_scroll)

    # ------------------------------------------------------------------ editing
    def on_focus(self, name):
        d = self.form.desc[name]
        self.help.show_param(d, self.form.values.get(name), "user" if name in self.values else "")

    def on_change(self, name, value):
        self.history.append(dict(self.values))
        self.future.clear()
        if value is None:
            self.values.pop(name, None)
            self.form.set_values({name: RenderSettings.model_fields[name].default}, {name: "default"})
        else:
            self.values[name] = value
        self.on_focus(name)
        self.debounce.poke()
        self.saver.poke()

    def set_all(self, values: dict):
        self.history.append(dict(self.values))
        self.values = dict(values)
        self._build_form()
        self.debounce.poke()
        self.saver.poke()

    def apply_preset(self):
        self.set_all(dict(PRESETS[self.preset.currentText()]))

    def undo(self):
        if self.history:
            self.future.append(dict(self.values))
            self.values = self.history.pop()
            self._build_form()
            self.debounce.poke()
            self.saver.poke()

    def redo(self):
        if self.future:
            self.history.append(dict(self.values))
            self.values = self.future.pop()
            self._build_form()
            self.debounce.poke()
            self.saver.poke()

    def take_snapshot(self):
        if "final" in self.images:
            self.snapshot = (dict(self.values), self.images["final"].copy(), self.cur_scale)
            self.compare.setCurrentIndex(1)

    def save_settings(self):
        if self.project:
            self.project.config.render["settings"] = dict(self.values)
            self.project.save()

    # ------------------------------------------------------------------ rendering
    def _proxy_scale(self) -> float:
        if not self.meta:
            return 1 / 3
        vw = max(self.view.viewport().width(), 400)
        w = self.meta["crop"][1] - self.meta["crop"][0]
        f = int(np.clip(np.floor(w / (1.5 * vw)), 1, 4))
        return 1.0 / f

    def render_preview(self):
        if not self.worker or not self.meta:
            return
        self.seq += 1
        self.status.setText("rendering...")
        self._request.emit(self.seq, dict(self.values), self._proxy_scale())
        self.refine.poke()

    def render_refine(self):
        """Full resolution when the preview is magnified on screen (zoomed in)."""
        if not self.worker or not self.meta or not self.images or self.cur_scale >= 1.0:
            return
        if self.view.zoom_factor() / max(self.cur_scale, 1e-6) > 1.05:
            self.seq += 1
            self.status.setText("refining at full resolution...")
            self._request.emit(self.seq, dict(self.values), 1.0)

    def on_ready(self, seq, out, scale, info):
        self.images = out
        self.cur_scale = scale
        self.info = info
        res = "full resolution" if scale >= 1 else f"preview 1/{int(round(1 / scale))}"
        self.status.setText(f"{res} · {info.get('render_ms', 0)} ms")
        self.show_current()
        self.update_checks(info)

    def show_current(self):
        if not self.images:
            return
        b = self.layer_group.checkedButton()
        key = b.property("layer") if b else "final"
        img = self.images[key]
        self.view.set_image(to_qimage(img), keep_view=True, scale_factor=1.0 / max(self.cur_scale, 1e-6))
        self.hist.set_image(img[4:-4, 4:-4])
        mode = self.compare.currentIndex()
        if mode == 1 and self.snapshot is not None:
            snap = self.snapshot[1]
            if snap.shape != img.shape:
                import cv2
                snap = cv2.resize(snap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            self.view.set_compare(to_qimage(snap), ("snapshot", "current"))
        elif mode == 2:
            ref = self.reference_image(img.shape)
            self.view.set_compare(to_qimage(ref) if ref is not None else None, ("reference", "current"))
        else:
            self.view.set_compare(None)
        if self.clip_b.isChecked():
            self.view.set_overlay(clipping_overlay(self.images["final"], self.images["starless"]))
        else:
            self.view.set_overlay(None)

    def reference_image(self, shape):
        """The reference image resampled into the geometry of the current rendering (cropped stack frame)."""
        if self.ref_img is not None and self.ref_img[0] == shape:
            return self.ref_img[1]
        p = self.project.stage_dir("reference") / "align_ref.json"
        if not p.exists():
            return None
        info = json.loads(p.read_text())
        if not info.get("reference"):
            return None
        import cv2
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        ref = np.asarray(Image.open(info["reference"]).convert("RGB"))
        Hr, Wr = info["ref_shape"]
        # inverse of the reference -> stack polynomial, fitted on a grid
        uu, vv = np.meshgrid(np.linspace(0, Wr - 1, 60), np.linspace(0, Hr - 1, 40))
        a, b = uu / Wr * 2 - 1, vv / Hr * 2 - 1
        terms = [a ** i * b ** j for i in range(4) for j in range(4 - i)]
        X = sum(c * t for c, t in zip(info["cx"], terms))
        Y = sum(c * t for c, t in zip(info["cy"], terms))
        H, W = self.meta["H"], self.meta["W"]
        Xn, Yn = X.ravel() / W * 2 - 1, Y.ravel() / H * 2 - 1
        T = np.stack([Xn ** i * Yn ** j for i in range(4) for j in range(4 - i)], 1)
        cu = np.linalg.lstsq(T, uu.ravel(), rcond=None)[0]
        cvv = np.linalg.lstsq(T, vv.ravel(), rcond=None)[0]
        x0, x1, y0, y1 = self.meta["crop"]
        h, w = shape[:2]
        sx, sy = (x1 - x0) / w, (y1 - y0) / h
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
        Xd, Yd = x0 + (gx + 0.5) * sx - 0.5, y0 + (gy + 0.5) * sy - 0.5
        Xn, Yn = Xd / W * 2 - 1, Yd / H * 2 - 1
        Tg = [Xn ** i * Yn ** j for i in range(4) for j in range(4 - i)]
        mu = sum(c * t for c, t in zip(cu, Tg)).astype(np.float32)
        mv = sum(c * t for c, t in zip(cvv, Tg)).astype(np.float32)
        out = cv2.remap(ref, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.ref_img = (shape, out)
        return out

    def update_checks(self, info):
        b = 4                                       # preview edge pixels mix in the uncovered border: ignored
        img = self.images["final"][b:-b, b:-b].astype(np.float32) / 255
        sl = self.images["starless"][b:-b, b:-b].astype(np.float32) / 255
        black = 100 * float((img.max(2) <= 0).mean())
        Y = 0.2126 * sl[..., 0] + 0.7152 * sl[..., 1] + 0.0722 * sl[..., 2]
        burnt = 100 * float((Y > 0.92).mean())
        sky = 255 * float(np.percentile(0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2], 30))
        st = info.get("stars", {})

        def mark(ok):
            sym = OK_SYM if ok else WARN_SYM
            return f"<span style='color:{style.OK if ok else style.WARN}'>{sym}</span>"
        white = 100 * float((img.min(2) >= 1).mean())
        rows = [f"{mark(black < 0.01)} pure black {black:.3f} %   {mark(white < 0.01)} pure white {white:.3f} %",
                f"{mark(burnt < 0.3)} nebula above 92 %: {burnt:.2f} %",
                f"&nbsp;&nbsp;&nbsp;sky (30th percentile) {sky:.0f}/255"]
        if st:
            rows.append(f"&nbsp;&nbsp;&nbsp;{st.get('drawn', 0)} stars drawn, G=10 Ø {st.get('d10', 0):.1f} px, "
                        f"G=7 Ø {st.get('d7', 0):.1f} px")
        self.checks.setText("<br>".join(rows))

    # ------------------------------------------------------------------ export
    def export_dialog(self):
        if not self.worker or not self.meta:
            return
        from ..render.export import ExportSettings
        dlg = QDialog(self)
        dlg.setWindowTitle("Export")
        dlg.resize(560, 420)
        v = QVBoxLayout(dlg)
        form = ParamForm(ExportSettings, ExportSettings().model_dump(), advanced=True)
        vals: dict = {}
        form.changed.connect(lambda n, val: vals.__setitem__(n, val))
        help_ = HelpBox()
        form.focused.connect(lambda n: help_.show_param(form.desc[n], form.values.get(n)))
        v.addWidget(form)
        v.addWidget(help_)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Export")
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        e = ExportSettings(**{k: val for k, val in vals.items() if val is not None})
        self.prog = QProgressDialog("Exporting...", None, 0, 100, self)
        self.prog.setWindowModality(Qt.WindowModal)
        self.prog.show()
        self._export_name = e.name
        self.save_settings()
        self._export.emit(self.project, dict(self.values), e.model_dump())

    def on_export_progress(self, f, msg):
        if self.prog:
            self.prog.setValue(int(100 * f))
            self.prog.setLabelText(msg)

    def on_exported(self, meta):
        if self.prog:
            self.prog.close()
            self.prog = None
        warn = meta["qa"].get("warnings") or []
        QMessageBox.information(self, "Export done",
                                f"{len(meta['files'])} files written to\n{self.project.results / self._export_name}\n\n"
                                + ("Quality checks passed." if not warn else "Check: " + ", ".join(warn)))


def clipping_overlay(final_u8: np.ndarray, starless_u8: np.ndarray) -> QImage:
    h, w, _ = final_u8.shape
    o = np.zeros((h, w, 4), np.uint8)
    black = final_u8.max(2) == 0
    white = final_u8.min(2) == 255
    sl = starless_u8.astype(np.float32) / 255
    burnt = (0.2126 * sl[..., 0] + 0.7152 * sl[..., 1] + 0.0722 * sl[..., 2]) > 0.92
    o[black] = (60, 120, 255, 255)
    o[burnt] = (230, 170, 60, 200)
    o[white] = (255, 60, 60, 255)
    return QImage(np.ascontiguousarray(o).data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
