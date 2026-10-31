"""The main window of the client.

Home: the workspace and its projects. A project has three modes over the same blocks:
* Graph: the workflows as a graph of blocks, the tools on top, the selected block's settings on the right;
* Inspect: the image of one block, large, the workflow as a strip, the same settings with the measurements;
* Post: the workflows' layers, masks and filter layers.
Settings hold the application preferences; the language takes effect at once (the menus, Home and the project screen
are built again). Spec mode shows, disabled and marked ◇, what the engine does not have yet.

The shell (this module) opens, creates and lists v2 projects (core/project2) and never writes into a project unless
the user creates it or saves it elsewhere. The views come from the later packages; ``READY`` says which are wired,
and the actions of the others stay disabled. Mount points, for P13:
* ``graph_page()``    P9b: GraphView with the toolbar of optional blocks, and the block Panel (model.ProjectModel);
* ``inspect_page()``  P10: TuneView (Inspect), with FramesView and SourcesView for the Frames block;
* ``post_page()``     P12: PostPage3 (Post is the Inspect view of the Post block) and ``export()``;
* ``launch()``, ``run_to_here()``, ``stop_run()``: P11 (services/run: plan dialog, RunFollower, the Log console);
* ``import_frames(folder)``: P10 (Import folder); ``offer_import(root)``: P6 (a v1 project);
* ``undo()``, ``redo()``: P9b (the workflows' undo stack); ``clear_cache()``: P5 (api.clear_unused).

    python -m astro_gpu.client [project folder]
"""
from __future__ import annotations

import html
import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, Qt, QTranslator
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..core import project2
from ..core.newproject import guess_names
from ..core.project import Target
from . import prefs
from .home import HomeView
from .i18n import _, language, set_language
from .logview import LogPanel
from .services.worker import ApiWorker, Machine, machine_info
from .settingsdlg import SettingsDialog
from .widgets import NEW, QSS, WARN, label

EXTRA_QSS = ("QSlider:horizontal { margin-left: 6px; margin-right: 6px; }"
             "QMenuBar { background: #1c1f26; } QMenuBar::item:selected { background: #2b3446; }")
HELP = """<h2>{title}</h2>
<p>{p1}</p><p>{p2}</p>"""

#: the views and services of the later packages, by what they bring; P13 turns each on as it wires it
READY = {
    "workflows": False,      # P9b: graph, block panel, undo / redo
    "inspect": False,        # P10: Inspect, Frames
    "frames": False,         # P10: Import folder, Import folder as new project
    "run": False,            # P11: Run all, Run to selected block, Stop
    "post": False,           # P12: Post, Export image
    "import_v1": False,      # P6: import of a project of the previous version
    "cache": False,          # P5: Clear cache of closed projects
}

_qt_translator: QTranslator | None = None


def apply_language(code: str):
    """The language of the texts, of Qt's own buttons (Yes, No…) and of the numbers (a decimal comma in French)."""
    global _qt_translator
    set_language(code)
    lang = language()
    QLocale.setDefault(QLocale(QLocale.French, QLocale.France) if lang == "fr"
                       else QLocale(QLocale.English, QLocale.UnitedStates))
    app = QCoreApplication.instance()
    if app is None:
        return
    if _qt_translator is not None:
        app.removeTranslator(_qt_translator)
        _qt_translator = None
    if lang != "en":
        t = QTranslator(app)
        if t.load(f"qtbase_{lang}", QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
            app.installTranslator(t)
            _qt_translator = t


def machine_text(machine: Machine | None, device: str) -> tuple[str, bool]:
    """The machine named in the top bar, and whether a GPU computes (orange without one)."""
    if machine is None:
        return "", True
    if device == "cpu":
        return _("CPU only"), False
    if not machine.gpus:
        return _("CPU only (no GPU)"), False
    g = machine.gpu(device) or max(machine.gpus, key=lambda x: x.memory_gb)
    return g.name, True


class OpenProject:
    """The open project as the shell knows it: its folder, its definition and its history (for the Log). P13 hands
    the views the ProjectModel built on it."""

    def __init__(self, root: Path, definition: project2.Project2):
        self.root = Path(root)
        self.definition = definition
        self.history: list[tuple] = []

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def target(self) -> str:
        return self.definition.target.name or ""

    def log(self, template: str, **args):
        """A history line: an English template and its arguments, translated when shown."""
        self.history.append((time.strftime("%H:%M:%S"), template, args))


class Placeholder(QWidget):
    """A mode whose view is not mounted yet."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        v = QVBoxLayout(self)
        v.addStretch(1)
        t = label(title, "dim")
        t.setAlignment(Qt.AlignCenter)
        v.addWidget(t)
        v.addStretch(1)


class Main(QMainWindow):
    def __init__(self, worker: ApiWorker | None = None):
        super().__init__()
        self.worker = worker or ApiWorker(self)
        self.worker.done.connect(self._worker_done)
        self.project: OpenProject | None = None
        self.s = None                                # the ProjectModel of the views (P9b), wired by P13
        self.machine: Machine | None = None
        self.home: HomeView | None = None
        self.proj_screen: QWidget | None = None
        self.mode = 0
        self.screens = QStackedWidget()
        self.setCentralWidget(self.screens)
        self.status = label("", "small")
        self.statusBar().addWidget(self.status, 1)
        self.spec_note = label("", "small")
        self.statusBar().addPermanentWidget(self.spec_note)
        self.b_log = QPushButton(_("Log"))
        self.b_log.setCheckable(True)
        self.b_log.setObjectName("toggle")
        self.b_log.toggled.connect(self.toggle_log)
        self.statusBar().addPermanentWidget(self.b_log)
        self.statusBar().setStyleSheet("QStatusBar { background: #1c1f26; border-top: 1px solid #2a2f38; }")
        self.build_menus()
        self.show_home()
        self.worker.submit("machine", machine_info)

    # ------------------------------------------------------------ menus
    def build_menus(self):
        mb = self.menuBar()
        for m in mb.findChildren(QMenu, options=Qt.FindDirectChildrenOnly):
            m.deleteLater()                          # with their actions; later, as one of them may be running
        mb.clear()
        has = self.project is not None
        spec = prefs.spec()

        def act(menu, text, fn=None, key=None, checkable=False, enabled=True, new=False):
            a = QAction(text + ("  ◇" if new else ""), menu)
            if key:
                a.setShortcut(QKeySequence(key))
            a.setCheckable(checkable)
            if fn is not None:
                a.triggered.connect(lambda _checked=False, f=fn: f())     # not the checked flag as an argument
            a.setEnabled(enabled and not new)
            if new:
                a.setToolTip("◇ " + _("Not in the engine yet."))
            menu.addAction(a)
            return a

        f = mb.addMenu(_("File"))
        act(f, _("Home"), self.show_home, "Ctrl+H")
        act(f, _("New project…"), self.new_project, "Ctrl+N")
        act(f, _("Open project…"), self.open_other, "Ctrl+O")
        rec = f.addMenu(_("Recent projects"))
        for r in prefs.recent():
            text = r["name"] + (f" · {r['target']}" if r.get("target") else "")
            a = act(rec, text, lambda _x=False, path=r["path"]: self.open_project(path))
            a.setToolTip(r["path"])
        rec.setEnabled(bool(prefs.recent()))
        rec.setToolTipsVisible(True)
        f.addSeparator()
        act(f, _("Save"), self.save, "Ctrl+S", enabled=has)
        act(f, _("Save as…"), self.save_as, enabled=has)
        f.addSeparator()
        act(f, _("Import folder…"), self.add_sources, "Ctrl+I", enabled=has and READY["frames"])
        act(f, _("Export image…"), self.export, "Ctrl+E", enabled=has and READY["post"])
        if spec:
            act(f, _("Export project as script…"), new=True)
        f.addSeparator()
        act(f, _("Close project"), self.close_project, "Ctrl+W", enabled=has)
        act(f, _("Quit"), self.close, "Ctrl+Q")
        e = mb.addMenu(_("Edit"))
        self.a_undo = act(e, _("Undo"), self.undo, "Ctrl+Z", enabled=has and READY["workflows"])
        self.a_redo = act(e, _("Redo"), self.redo, "Ctrl+Y", enabled=has and READY["workflows"])
        e.addSeparator()
        act(e, _("Settings…"), self.open_settings, "Ctrl+,")
        v = mb.addMenu(_("View"))
        act(v, _("Graph"), lambda: self.set_mode(0), "Ctrl+1", enabled=has)
        act(v, _("Inspect"), lambda: self.set_mode(1), "Ctrl+2", enabled=has)
        act(v, _("Post"), lambda: self.set_mode(2), "Ctrl+3", enabled=has)
        v.addSeparator()
        act(v, _("Log"), lambda: self.b_log.toggle(), "Ctrl+L")
        self.a_spec = act(v, _("Spec mode"), None, None, True)
        self.a_spec.setChecked(spec)
        self.a_spec.toggled.connect(self.set_spec)
        r = mb.addMenu(_("Run"))
        act(r, _("Run all"), self.launch, "F5", enabled=has and READY["run"])
        act(r, _("Run to selected block"), self.run_to_here, "Shift+F5", enabled=has and READY["run"])
        r.addSeparator()
        self.a_stop = act(r, _("Stop"), self.stop_run, enabled=has and READY["run"])
        h = mb.addMenu(_("Help"))
        act(h, _("About"), self.about, "F1")
        if spec:
            act(h, _("Glossary"), new=True)
        for m in (f, e, v, r, h):
            m.setToolTipsVisible(True)

    # ------------------------------------------------------------ Home
    def show_home(self, last=None):
        if self.home is None:
            self.home = HomeView(self.worker, can_import=READY["frames"], last=last)
            self.home.openProject.connect(self.open_project)
            self.home.createProject.connect(self.create_project)
            self.home.importFolder.connect(self.import_folder_as_new)
            self.screens.insertWidget(0, self.home)
        else:
            self.home.rescan()
        self.screens.setCurrentWidget(self.home)
        self.statusBar().clearMessage()
        self.setWindowTitle("astro_gpu · " + _("Home"))
        self.status.setText(prefs.get("workspace"))
        self.update_spec_note()

    def new_project(self):
        self.show_home()
        self.home.new_project()

    def open_other(self):
        self.show_home()
        self.home.open_other()

    def create_project(self, root: str, name: str, target: str) -> bool:
        """A new v2 project in ``root``, opened. Its calibrated frames go to <cache folder>/<project folder>."""
        try:
            p = project2.create(root, name, Target(name=target))
            cache = prefs.get("cache_dir")
            if cache:
                p.scratch_dir = (Path(cache) / Path(root).name).as_posix()
                project2.save(p)
        except (OSError, ValueError, project2.LegacyProject) as e:
            QMessageBox.warning(self, _("New project"), _("The project cannot be created: {why}").format(why=e))
            return False
        return self.open_project(root, created=True)

    def import_folder_as_new(self):
        """Import folder as new project: the folder of the lights gives the name and the target, then its frames
        are read into the new project."""
        folder = QFileDialog.getExistingDirectory(self, _("Import folder as new project…"), prefs.get("workspace"))
        if not folder:
            return
        g = guess_names(folder)
        got = self.home.ask_new_project(g["name"], g["target"])
        if got and self.create_project(*got):
            self.import_frames(folder)

    # ------------------------------------------------------------ a project
    def open_project(self, root, created: bool = False) -> bool:
        root = Path(root)
        if self.project is not None and root.as_posix().lower() == self.project.root.as_posix().lower():
            self.show_project()
            return True
        try:
            d = project2.load(root)
        except project2.LegacyProject:
            self.offer_import(root)
            return False
        except FileNotFoundError:
            QMessageBox.warning(self, _("Open project"), _("No project in this folder: {folder}").format(
                folder=root.as_posix()))
            prefs.drop_recent(root.as_posix())
            self.build_menus()
            return False
        except (OSError, ValueError, KeyError, TypeError) as e:
            QMessageBox.warning(self, _("Open project"), _("This project cannot be opened: {why}").format(why=e))
            return False
        self.close_project(go_home=False)
        self.project = OpenProject(root, d)
        self.project.log("Project created" if created else "Project opened")
        prefs.add_recent(root.as_posix(), d.name, d.target.name or "")
        self.mode = 0
        self.build_menus()
        self.build_project()
        return True

    def show_project(self):
        if self.proj_screen is not None:
            self.screens.setCurrentWidget(self.proj_screen)
            self.refresh()

    def close_project(self, go_home=True):
        """Save what is pending and close the project."""
        if self.project is not None:
            flush = getattr(self.s, "flush", None)
            if flush is not None:
                flush()
            self.project = None
            self.s = None
            if self.proj_screen is not None:
                self.screens.removeWidget(self.proj_screen)
                self.proj_screen.deleteLater()
                self.proj_screen = None
            self.build_menus()
        if go_home:
            self.show_home()

    def build_project(self):
        """The project screen, built again for a new project or a change of language."""
        old = self.proj_screen
        if old is not None:
            self.screens.removeWidget(old)
            old.deleteLater()
        screen = QWidget()
        screen.setObjectName("root")
        sv = QVBoxLayout(screen)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(0)
        bar = QFrame()
        bar.setObjectName("panel")
        bar.setStyleSheet("QFrame#panel { border-bottom: 1px solid #2a2f38; }")
        h = QHBoxLayout(bar)
        h.setContentsMargins(12, 0, 12, 0)
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 11pt;")
        h.addWidget(self.title)
        h.addSpacing(30)
        self.mode_btns = []
        g = QButtonGroup(screen)
        for i, name in enumerate((_("Graph"), _("Inspect"), _("Post"))):
            b = QPushButton(name)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.clicked.connect(lambda _x=False, k=i: self.set_mode(k))
            g.addButton(b)
            h.addWidget(b)
            self.mode_btns.append(b)
        h.addStretch(1)
        self.machine_lb = label("", "small")
        h.addWidget(self.machine_lb)
        h.addSpacing(12)
        self.bt_run = QPushButton(_("Run"))
        self.bt_run.setObjectName("primary")
        self.bt_run.clicked.connect(self.launch)
        h.addWidget(self.bt_run)
        bar.setFixedHeight(46)
        sv.addWidget(bar)
        self.modes = QStackedWidget()
        sv.addWidget(self.modes, 1)
        self.logp = LogPanel(self.project)
        self.logp.setVisible(self.b_log.isChecked())
        sv.addWidget(self.logp)
        # the modes: mount points of the views
        self.modes.addWidget(self.graph_page())
        self.modes.addWidget(self.inspect_page())
        self.modes.addWidget(self.post_page())
        self.proj_screen = screen
        self.screens.addWidget(screen)
        self.screens.setCurrentWidget(screen)
        self.set_mode(self.mode)

    def graph_page(self) -> QWidget:
        """Graph: the toolbar of optional blocks, the GraphView and the block Panel (P9b)."""
        return Placeholder(_("Graph"))

    def inspect_page(self) -> QWidget:
        """Inspect: TuneView, with Frames (P10)."""
        return Placeholder(_("Inspect"))

    def post_page(self) -> QWidget:
        """Post: PostPage3 (P12). In the views, Post is the Inspect view of the Post block."""
        return Placeholder(_("Post"))

    def set_mode(self, i: int):
        """0 Graph, 1 Inspect the chosen block, 2 Post."""
        if self.project is None:
            return
        if self.screens.currentWidget() is not self.proj_screen:
            self.screens.setCurrentWidget(self.proj_screen)
        self.mode = i
        self.modes.setCurrentIndex(i)
        self.mode_btns[i].setChecked(True)
        self.refresh()

    # ------------------------------------------------------------ refresh
    def project_name(self) -> str:
        pr = self.project
        name = "<b>" + html.escape(pr.name) + "</b>"
        return name + ("  ·  " + html.escape(pr.target) if pr.target else "")

    def refresh(self):
        pr = self.project
        if pr is None or self.proj_screen is None:
            return
        d = pr.definition
        self.title.setText(self.project_name())
        self.setWindowTitle("astro_gpu · " + pr.name)
        text, gpu = machine_text(self.machine, prefs.get("device"))
        self.machine_lb.setText(text)
        self.machine_lb.setStyleSheet("" if gpu else f"color: {WARN};")
        g = self.machine.gpu(prefs.get("device")) if self.machine is not None else None
        self.machine_lb.setToolTip(f"{g.device} · {g.memory_gb:.0f} GB" if g is not None else "")
        frames = bool(d.sessions)
        self.bt_run.setText(_("Run"))
        self.bt_run.setEnabled(READY["run"] and frames)
        self.bt_run.setToolTip("" if frames else _("No frames yet"))
        n = len(d.workflows)
        self.status.setText(_("{n} workflow(s)").format(n=n) if n else "")
        if self.logp.isVisible():
            self.logp.refresh()
        self.update_spec_note()

    def update_spec_note(self):
        self.spec_note.setText(_("Spec mode · ◇ not in the engine yet") + "   " if prefs.spec() else "")
        self.spec_note.setStyleSheet(f"color: {NEW};")

    def _worker_done(self, channel, _n, result):
        if channel == "machine":
            self.machine = result
            self.refresh()

    # ------------------------------------------------------------ files
    def save(self):
        """Edits are saved as they are made; Save writes what is still pending."""
        flush = getattr(self.s, "flush", None)
        if flush is not None:
            flush()
        self.flash(_("Project saved"))

    def save_as(self):
        """The definition of the project in another folder (its results stay where they are), then opened."""
        if self.project is None:
            return
        save = getattr(self.s, "flush", None)
        if save is not None:
            save()
        d = QFileDialog.getExistingDirectory(self, _("Save as…"), prefs.get("workspace"))
        if not d:
            return
        root = Path(d)
        if (root / project2.FILE).exists() or (root / project2.V1_FILE).exists():
            QMessageBox.warning(self, _("Save as…"), _("This folder already holds a project."))
            return
        old = self.project.definition
        p = project2.Project2.from_dict(old.to_dict())
        for s in p.sessions:
            for kind in project2.KINDS:
                setattr(s, kind, [old.abspath(x) for x in getattr(s, kind)])
        p.reference_image = old.reference_path()
        cache = prefs.get("cache_dir")
        p.scratch_dir = (Path(cache) / root.name).as_posix() if cache else ""
        try:
            project2.save(p, root)
        except (OSError, ValueError, project2.LegacyProject) as e:
            QMessageBox.warning(self, _("Save as…"), _("The project cannot be saved: {why}").format(why=e))
            return
        self.open_project(root)

    def offer_import(self, root: Path):
        """A project of the previous version: it opens once imported (P6); its folder is never written."""
        box = self.import_box(root)
        box.exec()
        box.deleteLater()

    def import_box(self, root: Path) -> QMessageBox:
        return QMessageBox(QMessageBox.Information, _("Open project"), _(
            "“{name}” was made with the previous version. Import it into the workspace to open it here; its folder "
            "is never changed.").format(name=Path(root).name), QMessageBox.Ok, self)

    # ------------------------------------------------------------ mount points of the later packages
    def add_sources(self):
        """File > Import folder…: frames into the open project (P10)."""

    def import_frames(self, folder: str):
        """The frames of a folder into the open project, after Import folder as new project (P10)."""

    def export(self):
        """File > Export image… (P12)."""

    def launch(self):
        """Run all (P11)."""

    def run_to_here(self):
        """Run to the selected block (P11)."""

    def stop_run(self):
        """Stop the run (P11): the STOP file; the next run resumes."""

    def undo(self):
        """Undo a change of the workflows (P9b)."""

    def redo(self):
        """Redo a change of the workflows (P9b)."""

    def clear_cache(self):
        """Clear the unused results of the closed projects, after a confirmation (P5)."""

    # ------------------------------------------------------------ settings, language, spec mode, log
    def open_settings(self, page: str | None = None):
        d = SettingsDialog(self.worker, self.machine, can_clear=READY["cache"], parent=self)
        if page:
            d.show_page(page)
        d.languageChanged.connect(self.change_language)
        d.deviceChanged.connect(self.set_device)
        d.specChanged.connect(self.set_spec)
        d.workspaceChanged.connect(lambda _w: self.home.rescan() if self.home is not None else None)
        d.clearCache.connect(self.clear_cache)
        d.exec()
        d.deleteLater()

    def change_language(self, code: str):
        apply_language(code)
        self.build_menus()
        self.b_log.setText(_("Log"))
        on_home = self.home is not None and self.screens.currentWidget() is self.home
        last = None
        if self.home is not None:
            last = self.home.last
            self.screens.removeWidget(self.home)
            self.home.deleteLater()
            self.home = None
        self.show_home(last)
        if self.project is not None:
            self.build_project()
            if on_home:
                self.show_home()

    def set_device(self, device: str):
        prefs.set("device", device)
        self.refresh()

    def set_spec(self, on: bool):
        if on != prefs.spec():
            prefs.set("spec_mode", on)
        self.build_menus()
        self.update_spec_note()
        self.refresh()

    def toggle_log(self, on):
        logp = getattr(self, "logp", None)
        if logp is not None and self.project is not None:
            logp.setVisible(on)
            if on:
                logp.refresh()

    def flash(self, text):
        self.statusBar().showMessage(text, 4000)

    def about(self):
        d = self.about_dialog()
        d.exec()
        d.deleteLater()

    def about_dialog(self) -> QDialog:
        d = QDialog(self)
        d.setWindowTitle(_("About"))
        lv = QVBoxLayout(d)
        t = QLabel(HELP.format(
            title=f"astro_gpu {__version__}",
            p1=_("Deep-sky images from a camera's raw frames: calibration, registration, stacking, astrometry, "
                 "color calibration on Gaia, denoise, stars and gradient, then the final image in Post. Each "
                 "workflow keeps its own settings; a block shared by several workflows is computed once."),
            p2=_("Shortcuts: Ctrl+1 Graph, Ctrl+2 Inspect, Ctrl+3 Post, Enter opens the selected block, Esc back "
                 "to the graph, Ctrl+Z / Ctrl+Y undo / redo, F5 run.")))
        t.setWordWrap(True)
        t.setTextFormat(Qt.RichText)
        lv.addWidget(t)
        lv.addStretch(1)
        b = QPushButton(_("Close"))
        b.clicked.connect(d.accept)
        lv.addWidget(b, 0, Qt.AlignRight)
        d.resize(640, 300)
        return d

    def closeEvent(self, e):
        flush = getattr(self.s, "flush", None)
        if flush is not None:
            flush()
        self.worker.stop()
        super().closeEvent(e)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    prefs.load()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("astro_gpu")
    app.setStyleSheet(QSS + EXTRA_QSS)
    apply_language(prefs.get("language"))

    def show_error(t, v, tb):
        QMessageBox.critical(None, "astro_gpu", "".join(traceback.format_exception(t, v, tb))[-3000:])

    sys.excepthook = show_error
    w = Main()
    w.resize(1680, 1000)
    w.showMaximized()
    if argv:
        w.open_project(argv[0])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
