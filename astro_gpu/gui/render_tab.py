"""Final adjustment tab: interactive GPU rendering with presets, snapshots, comparison, clipping view and export."""
from __future__ import annotations

import json
import time

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..render.clipping import LEGEND as CLIP_LEGEND
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
            from ..render.engine import StarRemovalEngines
            self.engine = StarRemovalEngines(self.prep_dir, device=self.device, calibrated_white=self.calibrated_white)
            self.loaded.emit({"H": self.engine.H, "W": self.engine.W, "crop": self.engine.crop, "SC": self.engine.SC,
                              "layered": self.engine.layered,
                              "fine_colour": "chroma_fine" in getattr(self.engine, "_host", {})})
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
    variantRequested = Signal(str)        # a view of another processing variant was chosen
    compareRequested = Signal(list)       # (label, variant, settings) of the views to compare

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
        self.current_view = ""
        self._pending_view = None
        self._last_view = ""                  # the view shown before the current one (B: back to it)
        self._restore_view = None             # zoom and place to keep when another processing is loaded
        self.view_cache: dict = {}            # view -> (final image, scale): instant switching

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
        # views: named renderings of this image (processing + rendering settings), one button each; no single
        # rendering suits every part of an image - keep one good view per zone, flip between them, export them
        vb = QHBoxLayout()
        lab = QLabel("Views:")
        lab.setToolTip("A view = the processing (main or a variant, see the Methods tab) and these rendering "
                       "settings. Click a view to show it at the same place; Left / Right: previous / next view; "
                       "B: back to the view shown before (flip to compare). Right click a view: update, rename, "
                       "export, delete.")
        vb.addWidget(lab)
        self.view_rows = QVBoxLayout()           # one row per base view: it and its alternatives
        self.view_rows.setSpacing(2)
        vb.addLayout(self.view_rows, 1)
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        new_b = QToolButton()
        new_b.setText("+ New view")
        new_b.setToolTip("Save the current settings and processing as a new view")
        new_b.clicked.connect(lambda: self.save_view(new=True))
        vb.addWidget(new_b)
        cmp_b = QToolButton()
        cmp_b.setText("Side by side...")
        cmp_b.setToolTip("Render the same zone of the views at 100 %, side by side")
        cmp_b.clicked.connect(self.compare_views)
        vb.addWidget(cmp_b)
        vb.addStretch(1)
        self.view_state = QLabel("")
        self.view_state.setObjectName("muted")
        vb.addWidget(self.view_state)
        cv.addLayout(vb)
        for key, fn in ((Qt.Key_Left, lambda: self._step_view(-1)), (Qt.Key_Right, lambda: self._step_view(1)),
                        (Qt.Key_B, self._back_view)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
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
        self.clip_b.setToolTip(CLIP_LEGEND)
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
        self.values = RenderSettings.from_saved(project.config.render.get("settings", {})).model_dump(
            exclude_defaults=True)
        self.history, self.future = [], []
        self.images = {}
        self.ref_img = None
        if self._pending_view and not self.view.item.pixmap().isNull():
            vc = self.view.viewport().rect().center()
            self._restore_view = (self.view.transform(), self.view.mapToScene(vc), self.view._fitted)
        self.view.set_image(None)
        self.view_cache.clear()
        self._init_views()
        if self._pending_view and self._pending_view in self._views():
            self.values = dict(self._views()[self._pending_view]["settings"])
            self.current_view = self._pending_view
        self._pending_view = None
        self._fill_views()
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
        if not meta.get("layered", True):
            QMessageBox.information(self, "Rendering", "These rendering layers were prepared by an older version: "
                                    "'Fine detail', 'Faint area smoothing', 'Fine colour', 'OIII tones' "
                                    "and 'Blue / reflection tones' have no effect until the 'Rendering preparation' "
                                    "stage is run again "
                                    "(Processing tab, about a minute).")
        elif not meta.get("fine_colour", True):
            self.status.setText("'Fine colour' needs the 'Star removal and denoising' stage to be run again")
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
        if self.project:
            self._fill_views()

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

    # ------------------------------------------------------------------ views
    def _views(self) -> dict:
        return self.project.config.render.setdefault("views", {}) if self.project else {}

    def _init_views(self):
        """A project's first views: its delivered renderings and their alternatives that need no computing."""
        if self.project:
            from ..render.views import ensure_default_views
            ensure_default_views(self.project)

    def _view_modified(self) -> bool:
        v = self._views().get(self.current_view)
        return v is not None and (dict(v["settings"]) != dict(self.values)
                                  or v.get("variant", "") != self.project.config.active_variant)

    def _fill_views(self):
        while self.view_rows.count():
            row = self.view_rows.takeAt(0)
            lay_ = row.layout()
            while lay_ is not None and lay_.count():
                w = lay_.takeAt(0).widget()
                if w is not None:
                    if isinstance(w, QToolButton):
                        self.view_group.removeButton(w)
                    w.deleteLater()
        known = set(self.project.config.variants) if self.project else set()
        views = self._views()
        for key, names in self._view_groups().items():
            row = QHBoxLayout()
            row.setSpacing(4)
            head = QLabel(f"<b>{key}</b>")
            head.setMinimumWidth(90)
            row.addWidget(head)
            for name in names:
                self._view_button(row, name, views[name], known, key)
            row.addStretch(1)
            self.view_rows.addLayout(row)
        self._fill_state()

    def _view_groups(self) -> dict[str, list[str]]:
        """Rows of the bar: a view 'X - something' goes with the view 'X'; the others start their own row."""
        views = self._views()
        groups: dict[str, list[str]] = {}
        for name in views:
            base, _, suffix = name.partition(" - ")
            groups.setdefault(base if suffix and base in views else name, []).append(name)
        return groups

    def _view_button(self, row, name, v, known, key):
        var = v.get("variant", "")
        b = QToolButton()
        b.setCheckable(True)
        short = "as delivered" if name == key else name[len(key) + 3:] if name.startswith(key + " - ") else name
        b.setText(short + ("  *" if name == self.current_view and self._view_modified() else ""))
        lost = bool(var) and var not in known
        b.setToolTip(f"{name}\nProcessing: {var or 'main'}{' (deleted)' if lost else ''}\n"
                     + (f"{v.get('note')}\n" if v.get("note") else "")
                     + "Right click: update with the current settings, rename, export, delete")
        b.setEnabled(not lost)
        b.setChecked(name == self.current_view)
        b.clicked.connect(lambda _=False, n=name: self.show_view(n))
        b.setContextMenuPolicy(Qt.CustomContextMenu)
        b.customContextMenuRequested.connect(lambda pos, n=name, bb=b: self._view_menu(n, bb.mapToGlobal(pos)))
        self.view_group.addButton(b)
        row.addWidget(b)

    def _fill_state(self):
        if not self.current_view:
            self.view_state.setText("current settings (no view)")
        elif self._view_modified():
            self.view_state.setText(f"'{self.current_view}' modified: right click it to update it")
        else:
            self.view_state.setText(f"processing: {self.project.config.active_variant or 'main'}")

    def show_view(self, name):
        if name not in self._views():
            return
        v = self._views()[name]
        var = v.get("variant", "")
        if var and var not in self.project.config.variants:
            QMessageBox.warning(self, "View", f"The processing '{var}' of this view was deleted.")
            self._fill_views()
            return
        if name != self.current_view:
            self._last_view = self.current_view
        if var != self.project.config.active_variant:
            self._pending_view = name          # applied once the other processing is loaded, at the same place
            self.variantRequested.emit(var)
            return
        self.current_view = name
        if name in self.view_cache:            # instant: the last image of this view, refreshed just after
            img, sc = self.view_cache[name]
            self.images = dict(self.images, final=img) if self.images else {"final": img, "starless": img,
                                                                             "stars": img}
            self.cur_scale = sc
            self.show_current()
        self.history.append(dict(self.values))
        self.values = dict(v["settings"])
        self._build_form()
        self.debounce.poke()
        self.saver.poke()
        self._fill_views()

    def _step_view(self, d):
        views = self._views()
        names = [n for row in self._view_groups().values() for n in row       # the order of the bar, row by row
                 if not views[n].get("variant") or views[n]["variant"] in self.project.config.variants]
        if not names:
            return
        i = names.index(self.current_view) if self.current_view in names else (-1 if d > 0 else 0)
        self.show_view(names[(i + d) % len(names)])

    def _back_view(self):
        if self._last_view and self._last_view in self._views():
            self.show_view(self._last_view)

    def _view_menu(self, name, pos):
        m = QMenu(self)
        up = m.addAction("Update with the current settings and processing")
        rn = m.addAction("Rename...")
        ex = m.addAction("Export...")
        m.addSeparator()
        dl = m.addAction("Delete")
        a = m.exec(pos)
        if a is up:
            self._views()[name].update(variant=self.project.config.active_variant, settings=dict(self.values))
            self.project.save()
            self.current_view = name
            self._fill_views()
        elif a is rn:
            new, ok = QInputDialog.getText(self, "Rename view", "New name:", text=name)
            new = new.strip() if ok else ""
            if new and new != name and new not in self._views():
                views = self._views()
                self.project.config.render["views"] = {(new if k == name else k): val for k, val in views.items()}
                self.project.save()
                if self.current_view == name:
                    self.current_view = new
                self.view_cache.pop(name, None)
                self._fill_views()
        elif a is ex:
            if name != self.current_view:
                self.show_view(name)
            if name == self.current_view:
                self.export_dialog()
        elif a is dl:
            if QMessageBox.question(self, "Delete view", f"Delete the view '{name}'? (Its exported images are "
                                    "kept.)") == QMessageBox.Yes:
                del self._views()[name]
                self.project.save()
                self.view_cache.pop(name, None)
                if self.current_view == name:
                    self.current_view = ""
                self._fill_views()

    def save_view(self, new: bool = False):
        if not self.project:
            return
        name, ok = QInputDialog.getText(self, "New view", "Name of the view:",
                                        text="" if new else (self.current_view or ""))
        name = name.strip() if ok else ""
        if not name:
            return
        if name in self._views() and QMessageBox.question(
                self, "New view", f"Replace the view '{name}'?") != QMessageBox.Yes:
            return
        from datetime import datetime
        self._views()[name] = {"variant": self.project.config.active_variant, "settings": dict(self.values),
                               "created": datetime.now().isoformat(timespec="seconds")}
        self.project.save()
        if name != self.current_view:
            self._last_view = self.current_view
        self.current_view = name
        self._fill_views()

    def compare_views(self):
        known = set(self.project.config.variants) if self.project else set()
        entries = [(name, v.get("variant", ""), dict(v["settings"])) for name, v in self._views().items()
                   if not v.get("variant") or v["variant"] in known]
        if len(entries) < 2:
            QMessageBox.information(self, "Compare views", "Save at least two views first (Save view...).")
            return
        self.save_settings()
        self.compareRequested.emit(entries)

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
        if self.current_view and not self._view_modified():
            self.view_cache[self.current_view] = (out["final"], scale)
            while len(self.view_cache) > 8:
                self.view_cache.pop(next(iter(self.view_cache)))
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
        if self._restore_view is not None:          # same zoom and place as the view shown before
            t, c, fitted = self._restore_view
            self._restore_view = None
            if not fitted:
                self.view.setTransform(t)
                self.view.centerOn(c)
                self.view._fitted = False
        self.hist.set_image(_sample(img[4:-4, 4:-4]))
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
        # statistics on a fixed-size sample of pixels: on the whole image (several million pixels at full resolution)
        # they blocked the interface for up to seconds after every change
        img = _sample(self.images["final"][b:-b, b:-b]).astype(np.float32) / 255
        sl = _sample(self.images["starless"][b:-b, b:-b]).astype(np.float32) / 255
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
        v_ = self.project.config.active_variant
        init = ExportSettings().model_dump()
        if self.current_view:        # a view is exported under its own name
            import re
            init["name"] = re.sub(r"[^A-Za-z0-9_-]+", "_", self.current_view).strip("_") or init["name"]
        elif v_:                     # a variant's export does not overwrite main's results
            init["name"] = f"{init['name']}_{self.project.config.variants[v_]['slug']}"
        form = ParamForm(ExportSettings, init, advanced=True)
        vals: dict = {"name": init["name"]} if (v_ or self.current_view) else {}
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


def _sample(img: np.ndarray, n: int = 250_000) -> np.ndarray:
    """Regular subsample of an image with about n pixels (statistics for display: histogram, checks)."""
    h, w = img.shape[:2]
    step = max(1, int(np.sqrt(h * w / n)))
    return img[::step, ::step]


def clipping_overlay(final_u8: np.ndarray, starless_u8: np.ndarray) -> QImage:
    from ..render.clipping import clipping_rgba
    o = clipping_rgba(final_u8, starless_u8)
    h, w, _ = o.shape
    return QImage(np.ascontiguousarray(o).data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
