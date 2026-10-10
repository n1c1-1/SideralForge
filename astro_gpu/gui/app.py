"""Main window: project, pipeline stages with their parameters and results, journal with progress and ETA."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QColor, QImage
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..core.project import Project
from ..core.runner import Pipeline
from ..stages import all_stages
from . import style
from .render_tab import RenderTab
from .runner import RunController, stage_history_seconds
from .widgets import HelpBox, ImageView, JournalView, ParamForm, scroll


def _fmt_dur(s: float | None) -> str:
    if s is None:
        return ""
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


class StagePage(QWidget):
    """Right-hand side of the processing tab: what the selected stage does, its parameters and its results."""

    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        v = QVBoxLayout(self)
        self.title = QLabel("")
        self.title.setObjectName("h1")
        self.state = QLabel("")
        self.state.setWordWrap(True)
        v.addWidget(self.title)
        v.addWidget(self.state)
        row = QHBoxLayout()
        self.run_here = QPushButton("Run up to this stage")
        self.rerun = QPushButton("Recompute this stage")
        self.rerun.setToolTip("Force this stage (and what depends on it) to be computed again")
        self.frames_b = QPushButton("Frames...")
        self.frames_b.setToolTip("Every light frame with its weight, transparency, star size and exclusion reason")
        self.frames_b.clicked.connect(self.show_frames)
        row.addWidget(self.run_here)
        row.addWidget(self.rerun)
        row.addWidget(self.frames_b)
        v.addLayout(row)
        self.form_host = QVBoxLayout()
        v.addLayout(self.form_host, 3)
        self.metrics = QTableWidget(0, 2)
        self.metrics.setHorizontalHeaderLabels(["Result", "Value"])
        self.metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.metrics.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.metrics.setWordWrap(True)
        self.metrics.verticalHeader().setVisible(False)
        self.metrics.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.metrics, 2)
        self.help = HelpBox()
        v.addWidget(self.help, 3)
        self.form = None
        self.form_scroll = None
        self.stage = None
        self.run_here.clicked.connect(lambda: self.main.start_run(until=self.stage.name))
        self.rerun.clicked.connect(lambda: self.main.start_run(until=self.stage.name, force=[self.stage.name]))

    def show_stage(self, stage, status, pipeline: Pipeline, advanced: bool):
        self.stage = stage
        self.title.setText(stage.title)
        col = style.STATE_COLOURS.get(status.state, style.FG2)
        txt = f"<span style='color:{col}'>{style.STATE_SYMBOLS.get(status.state, '')} {status.state}</span>"
        if status.reason:
            txt += f" &middot; {status.reason}"
        if status.duration:
            txt += f" &middot; took {_fmt_dur(status.duration)}"
        if stage.network:
            txt += " &middot; needs internet"
        self.state.setText(txt)
        p, origin, auto = pipeline.effective_params(stage)
        if self.form_scroll is not None:
            self.form_host.removeWidget(self.form_scroll)
            self.form_scroll.hide()
            self.form_scroll.setParent(None)
            self.form_scroll.deleteLater()
        self.form = ParamForm(stage.Params, p.model_dump(), origin, {k: a.reason for k, a in auto.items()},
                              advanced=advanced)
        self.form.changed.connect(self.on_change)
        self.form.focused.connect(self.on_focus)
        self.form_scroll = scroll(self.form)
        self.form_host.addWidget(self.form_scroll)
        self.help.show_text(stage.title, stage.description)
        self.frames_b.setVisible((pipeline.project.stage_dir(stage.name) / "frames.json").exists()
                                 and stage.name == "stack")
        self.show_metrics(pipeline)

    def show_metrics(self, pipeline):
        m = json.loads((pipeline.project.stage_dir(self.stage.name) / "manifest.json").read_text()) \
            if (pipeline.project.stage_dir(self.stage.name) / "manifest.json").exists() else {}
        rows = list(m.get("metrics", {}).items())
        if m.get("error"):
            rows.insert(0, ("error", m["error"]))
        self.metrics.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.metrics.setItem(i, 0, QTableWidgetItem(k.replace("_", " ")))
            it = QTableWidgetItem(v if isinstance(v, str) else json.dumps(v))
            if k == "error":
                it.setForeground(QColor(style.ERR))
            self.metrics.setItem(i, 1, it)

    def on_focus(self, name):
        d = self.form.desc[name]
        self.help.show_param(d, self.form.values.get(name), self.form.origins.get(name, ""), self.form.reasons.get(name, ""))

    def show_frames(self):
        from PySide6.QtWidgets import QDialog
        rows = json.loads((self.main.project.stage_dir("stack") / "frames.json").read_text())
        dlg = QDialog(self)
        dlg.setWindowTitle("Light frames")
        dlg.resize(900, 640)
        v = QVBoxLayout(dlg)
        n_ex = sum(1 for r in rows if r["excluded"])
        lab = QLabel(f"{len(rows)} frames, {len(rows) - n_ex} used, {n_ex} excluded. Weight = (transparency / noise)^2 x "
                     "sharpness, relative to the best frame. Click a column title to sort.")
        lab.setWordWrap(True)
        v.addWidget(lab)
        t = QTableWidget(len(rows), 6)
        t.setHorizontalHeaderLabels(["File", "Time", "Weight", "Transparency", "FWHM (px)", "Excluded because"])
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        for i, r in enumerate(rows):
            vals = [Path(r["path"]).name, (r.get("time") or "")[11:19], r["weight"], r["transparency"], r["fwhm"],
                    r["excluded"]]
            for j, val in enumerate(vals):
                it = QTableWidgetItem()
                if isinstance(val, float):
                    it.setData(Qt.DisplayRole, round(val, 3))
                else:
                    it.setText(str(val))
                if r["excluded"]:
                    it.setForeground(QColor(style.WARN))
                t.setItem(i, j, it)
        t.setSortingEnabled(True)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        v.addWidget(t)
        dlg.exec()

    def on_change(self, name, value):
        proj = self.main.project
        if value is None:
            proj.clear_override(self.stage.name, name)
        else:
            proj.set_override(self.stage.name, name, value)
        self.main.refresh_status(keep_page=True)
        p, origin, auto = self.main.pipeline.effective_params(self.stage)
        self.form.set_values({name: getattr(p, name)}, {name: origin[name]})
        self.on_focus(name)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"astro_GPU {__version__}")
        self.resize(1600, 980)
        self.settings = QSettings("astro_GPU", "astro_GPU")
        self.project: Project | None = None
        self.pipeline: Pipeline | None = None
        self.advanced = bool(self.settings.value("advanced", False, type=bool))
        self.statuses = {}
        self.cur_stage_t0: dict[str, float] = {}
        self.runner = RunController(self)
        self.runner.event.connect(self.on_event)
        self.runner.runningChanged.connect(self.on_running)

        # ---------------- toolbar
        tb = self.addToolBar("main")
        tb.setMovable(False)
        for text, fn, tip, key in (("New project", self.new_project, "Create a project from your frame folders", "Ctrl+N"),
                                   ("Open...", self.open_project, "Open an existing project folder", "Ctrl+O")):
            a = QAction(text, self)
            a.setToolTip(f"{tip} ({key})")
            a.setShortcut(key)
            a.triggered.connect(fn)
            tb.addAction(a)
        self.recent = QToolButton()
        self.recent.setText("Recent ▾")
        self.recent.setPopupMode(QToolButton.InstantPopup)
        tb.addWidget(self.recent)
        tb.addSeparator()
        self.run_a = QAction("▶ Run / resume", self)
        self.run_a.setToolTip("Compute every stage that is not up to date (an interrupted run continues where it stopped)")
        self.run_a.triggered.connect(lambda: self.start_run())
        self.run_a.setShortcut("F5")
        self.stop_a = QAction("■ Stop", self)
        self.stop_a.setToolTip("Stop at the next checkpoint; the run can be resumed later")
        self.stop_a.triggered.connect(self.runner.stop)
        self.stop_a.setShortcut("Shift+F5")
        tb.addAction(self.run_a)
        tb.addAction(self.stop_a)
        tb.addSeparator()
        self.adv_b = QToolButton()
        self.adv_b.setText("Advanced settings")
        self.adv_b.setCheckable(True)
        self.adv_b.setChecked(self.advanced)
        self.adv_b.setToolTip("Show every parameter (basic mode shows the few that matter most)")
        self.adv_b.toggled.connect(self.set_advanced)
        tb.addWidget(self.adv_b)
        guide = QAction("Guide", self)
        guide.setShortcut("F1")
        guide.setToolTip("How to use astro_GPU (F1)")
        guide.triggered.connect(self.show_guide)
        tb.addAction(guide)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.gpu_box = QComboBox()
        self.gpu_box.setToolTip("GPU used by the processing and the rendering of this project")
        for i, name in _gpus():
            self.gpu_box.addItem(name, f"cuda:{i}")
        if self.gpu_box.count() == 0:
            self.gpu_box.addItem("CPU (no CUDA GPU found)", "cpu")
        self.gpu_box.currentIndexChanged.connect(self.set_device)
        tb.addWidget(self.gpu_box)

        # ---------------- stages dock
        self.stage_list = QListWidget()
        self.stage_list.setMinimumWidth(250)
        self.stage_list.currentRowChanged.connect(self.on_stage_selected)
        d1 = QDockWidget("Pipeline", self)
        d1.setWidget(self.stage_list)
        d1.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, d1)

        # ---------------- central tabs
        self.tabs = QTabWidget()
        proc = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        prow = QHBoxLayout()
        self.preview_box = QComboBox()
        self.preview_box.currentIndexChanged.connect(self.show_preview)
        prow.addWidget(QLabel("Preview:"))
        prow.addWidget(self.preview_box, 1)
        lv.addLayout(prow)
        self.viewer = ImageView()
        self.viewer.placeholder = "Previews of the selected stage appear here"
        lv.addWidget(self.viewer, 1)
        proc.addWidget(left)
        self.page = StagePage(self)
        proc.addWidget(self.page)
        proc.setSizes([950, 560])
        self.tabs.addTab(proc, "Processing")
        self.render_tab = RenderTab()
        self.tabs.addTab(self.render_tab, "Final adjustment")
        self.tabs.currentChanged.connect(self.on_tab)
        self.setCentralWidget(self.tabs)

        # ---------------- journal dock
        jw = QWidget()
        jv = QVBoxLayout(jw)
        jv.setContentsMargins(6, 4, 6, 4)
        r1 = QHBoxLayout()
        self.cur_label = QLabel("idle")
        self.cur_bar = QProgressBar()
        self.cur_bar.setRange(0, 1000)
        self.all_label = QLabel("")
        self.all_bar = QProgressBar()
        self.all_bar.setRange(0, 1000)
        r1.addWidget(self.cur_label, 2)
        r1.addWidget(self.cur_bar, 3)
        r1.addSpacing(16)
        r1.addWidget(self.all_label, 2)
        r1.addWidget(self.all_bar, 3)
        jv.addLayout(r1)
        self.journal = JournalView()
        jv.addWidget(self.journal)
        d2 = QDockWidget("Journal", self)
        d2.setWidget(jw)
        d2.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.BottomDockWidgetArea, d2)
        self.resizeDocks([d2], [230], Qt.Vertical)
        self.resizeDocks([d1], [270], Qt.Horizontal)
        self._update_recent()
        self.on_running(False)

    # ------------------------------------------------------------------ projects
    def new_project(self):
        from .project_dialog import NewProjectDialog
        dlg = NewProjectDialog(self)
        if dlg.exec() and dlg.project:
            self.load_project(dlg.project.root)
            self.start_run(until="inventory")

    def open_project(self):
        d = QFileDialog.getExistingDirectory(self, "Open a project folder (containing project.json)")
        if d:
            self.load_project(d)

    def load_project(self, root):
        try:
            p = Project(root)
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Open project", str(e))
            return
        self.project = p
        self.pipeline = Pipeline(p, all_stages())
        self.setWindowTitle(f"astro_GPU {__version__} - {p.config.name}")
        from ..core.project import remember
        remember(p)
        self._update_recent()
        self.journal.clear()
        dev = p.config.devices.get("main")
        i = self.gpu_box.findData(dev) if dev else 0
        self.gpu_box.blockSignals(True)
        self.gpu_box.setCurrentIndex(max(i, 0))
        self.gpu_box.blockSignals(False)
        self.runner.attach(p)
        for e in self.runner.history():
            self._journal_line(e)
        self.refresh_status()
        self.stage_list.setCurrentRow(0)
        self.render_tab.set_project(p, self.advanced)

    def show_guide(self):
        from PySide6.QtWidgets import QDialog, QTextBrowser
        p = Path(__file__).resolve().parents[1] / "resources" / "GUIDE.md"
        dlg = QDialog(self)
        dlg.setWindowTitle("astro_GPU guide")
        dlg.resize(820, 760)
        v = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setOpenExternalLinks(True)
        tb.setMarkdown(p.read_text(encoding="utf-8") if p.exists() else "Guide not found.")
        v.addWidget(tb)
        dlg.exec()

    def set_device(self, _=None):
        if self.project:
            self.project.config.devices["main"] = self.gpu_box.currentData()
            self.project.save()

    def _update_recent(self):
        """Recently used projects, whether they were opened here or run from the command line."""
        from PySide6.QtWidgets import QMenu

        from ..core.project import recent_projects
        m = QMenu(self)
        items = recent_projects()
        known = {Path(r).resolve() for _, r in items}
        for r in self.settings.value("recent", [], type=list):          # list kept by older versions
            if Path(r, "project.json").exists() and Path(r).resolve() not in known:
                items.append((Path(r).name, r))
        for name, r in items[:20]:
            a = m.addAction(f"{name}    {r}")
            a.setToolTip(r)
            a.triggered.connect(lambda _=False, rr=r: self.load_project(rr))
        if not items:
            m.addAction("(no project yet)").setEnabled(False)
        self.recent.setMenu(m)

    # ------------------------------------------------------------------ status
    def refresh_status(self, keep_page: bool = False):
        if not self.pipeline:
            return
        sts = self.pipeline.status()
        self.statuses = {s.name: s for s in sts}
        row = self.stage_list.currentRow()
        self.stage_list.blockSignals(True)
        self.stage_list.clear()
        for s in sts:
            it = QListWidgetItem(f"{style.STATE_SYMBOLS.get(s.state, '')}  {s.title}"
                                 + (f"   {_fmt_dur(s.duration)}" if s.state == "done" and s.duration else ""))
            it.setForeground(QColor(style.STATE_COLOURS.get(s.state, style.FG)))
            it.setToolTip(s.reason or s.state)
            self.stage_list.addItem(it)
        self.stage_list.setCurrentRow(max(row, 0))
        self.stage_list.blockSignals(False)
        if not keep_page:
            self.on_stage_selected(self.stage_list.currentRow())
        self._update_overall()

    def on_stage_selected(self, row):
        if not self.pipeline or row < 0:
            return
        st = self.pipeline.stages[row]
        self.page.show_stage(st, self.statuses[st.name], self.pipeline, self.advanced)
        self._fill_previews(st.name)

    def _fill_previews(self, stage):
        self.preview_box.blockSignals(True)
        self.preview_box.clear()
        d = self.project.stage_dir(stage) / "previews"
        files = sorted(d.glob("*.jpg"), key=lambda p: p.stat().st_mtime) if d.exists() else []
        for f in files:
            self.preview_box.addItem(f.stem.replace("_", " "), str(f))
        self.preview_box.blockSignals(False)
        if files:
            self.preview_box.setCurrentIndex(len(files) - 1)
            self.show_preview()
        else:
            self.viewer.set_image(None)

    def show_preview(self):
        p = self.preview_box.currentData()
        if p:
            self.viewer.set_image(QImage(p), keep_view=False)

    def set_advanced(self, v):
        self.advanced = v
        self.settings.setValue("advanced", v)
        self.on_stage_selected(self.stage_list.currentRow())
        self.render_tab.set_advanced(v)

    def on_tab(self, i):
        if i == 1 and self.project and self.render_tab.worker is None:
            self.render_tab.set_project(self.project, self.advanced)

    # ------------------------------------------------------------------ running
    def start_run(self, until=None, force=None):
        if not self.project:
            return
        if self.runner.is_running():
            QMessageBox.information(self, "Run", "A run is already in progress for this project.")
            return
        todo = self.pipeline.plan(until, tuple(force or ()))
        if not todo:
            self.journal.add("everything is up to date")
            return
        net = [s for s in todo if self.pipeline.by_name[s].network]
        if net:
            self.journal.add(f"stages needing internet: {', '.join(net)}")
        self.plan = todo
        self.runner.start(until, force)

    def on_running(self, running):
        self.run_a.setEnabled(not running)
        self.stop_a.setEnabled(running)
        if running:
            self.cur_label.setText("running...")
        else:
            self.cur_label.setText("idle")
            self.cur_bar.setValue(0)
            if self.project:
                self.refresh_status(keep_page=True)

    def on_event(self, e):
        t = e.get("type")
        st = e.get("stage", "")
        if t == "progress":
            title = self.pipeline.by_name[st].title if st in self.pipeline.by_name else st
            eta = e.get("eta")
            self.cur_label.setText(f"{title}: {e.get('msg', '')}" + (f"  (ETA {_fmt_dur(eta)})" if eta else ""))
            self.cur_bar.setValue(int(1000 * e.get("frac", 0)))
            self._cur_eta = eta
            self._update_overall()
            return
        self._journal_line(e)
        if t in ("stage_start", "stage_end", "run_end"):
            if t == "stage_start":
                self.cur_stage_t0[st] = e["t"]
                self.cur_bar.setValue(0)
                self._cur_eta = None
            self.refresh_status(keep_page=True)
            cur = self.page.stage
            if cur is not None and cur.name == st:
                self.page.show_metrics(self.pipeline)
            if t == "run_end" and self.tabs.currentIndex() == 1 or (t == "stage_end" and st == "render_prep"):
                self.render_tab.set_project(self.project, self.advanced)
        elif t == "preview":
            cur = self.page.stage
            if cur is not None and cur.name == st:
                self._fill_previews(st)
        elif t == "metric":
            cur = self.page.stage
            if cur is not None and cur.name == st:
                self.page.show_metrics(self.pipeline)

    def _journal_line(self, e):
        t = e.get("type")
        ts = time.strftime("%H:%M:%S", time.localtime(e.get("t", time.time())))
        st = e.get("stage", "")
        if t == "log":
            self.journal.add(f"{ts}  [{st}] {e.get('msg')}", e.get("level", "info"))
        elif t == "stage_start":
            self.journal.add(f"{ts}  ▶ {st} started")
        elif t == "stage_end":
            lvl = "error" if e.get("status") == "failed" else ("warning" if e.get("status") != "done" else "info")
            self.journal.add(f"{ts}  {style.STATE_SYMBOLS.get(e.get('status'), '')} {st} {e.get('status')}"
                             + (f" in {_fmt_dur(e.get('duration'))}" if e.get("duration") else ""), lvl)
        elif t == "run_start":
            self.journal.add(f"{ts}  run: {', '.join(e.get('stages', []))}")
        elif t == "run_end":
            self.journal.add(f"{ts}  run {e.get('status')}")

    def _update_overall(self):
        """Overall progress and ETA: remaining stages estimated from their past durations on this machine."""
        if not self.pipeline:
            return
        todo = [s for s, v in self.statuses.items() if v.state not in ("done",)]
        if not self.runner.is_running() or not todo:
            self.all_label.setText(f"{sum(1 for v in self.statuses.values() if v.state == 'done')}/{len(self.statuses)} "
                                   "stages up to date")
            self.all_bar.setValue(int(1000 * sum(1 for v in self.statuses.values() if v.state == "done")
                                      / max(len(self.statuses), 1)))
            return
        est = 0.0
        for s in todo:
            if self.statuses[s].state == "running":
                est += getattr(self, "_cur_eta", None) or 0
            else:
                est += stage_history_seconds(s) or 120
        done = sum(1 for v in self.statuses.values() if v.state == "done")
        self.all_label.setText(f"Overall: {done}/{len(self.statuses)} stages, about {_fmt_dur(est)} left")
        self.all_bar.setValue(int(1000 * done / max(len(self.statuses), 1)))

    def closeEvent(self, e):
        if self.runner.is_running():
            self.journal.add("the run continues in the background; reopen the project to follow it")
        self.render_tab._stop_worker()
        super().closeEvent(e)


def _gpus() -> list[tuple[int, str]]:
    try:
        import torch
        if torch.cuda.is_available():
            out = []
            for i in range(torch.cuda.device_count()):
                pr = torch.cuda.get_device_properties(i)
                out.append((i, f"{pr.name} ({pr.total_memory / 2 ** 30:.0f} GB)"))
            return sorted(out, key=lambda t: -torch.cuda.get_device_properties(t[0]).total_memory)
    except Exception:  # noqa: BLE001
        pass
    return []


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("astro_GPU")
    app.setStyle("Fusion")
    app.setPalette(style.palette())
    app.setStyleSheet(style.QSS)
    w = MainWindow()
    w.show()
    if argv:
        w.load_project(Path(argv[0]))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
