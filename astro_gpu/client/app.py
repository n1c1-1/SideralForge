"""The main window of the client.

Home: the workspace and its projects. A project has three modes over the same blocks:
* Graph: the workflows as a graph of blocks, the tools on top, the selected block's settings on the right;
* Inspect: the image of one block, large, the workflow as a strip, the same settings with the measurements;
* Post: the workflows' layers, masks and filter layers (Post is the Inspect view of the Post block).
Settings hold the application preferences; the language takes effect at once (the menus, Home and the project screen
are built again). Spec mode shows, disabled and marked ◇, what the engine does not have yet.

The shell opens, creates and lists v2 projects (core/project2) and never writes into a project unless the user edits
it or saves it elsewhere; a project of an older version (v1) is imported first (previous.py), its folder untouched.
Once a project is open the window holds, for it:
* ``s``: the ProjectModel (model.py) every view reads and edits; it saves the definition 300 ms after an edit;
* ``run``: the RunController (services/run.py): the plan dialog, the detached run, its progress and the Log console;
* ``render``: the RenderWorker of Post (services/render.py), made when Post is first shown and kept while the project
  is open, so that a rebuilt screen (a change of language) does not load the engines again.
The views are built again with the project screen: ``graph`` (GraphPage), ``tune`` (TuneView, holding Frames and the
Post page). Classes: ``Main`` (the window), ``OpenProject`` (the open project as the shell knows it), ``PostHolder``
(makes the Post page the first time it is shown).

    python -m astro_gpu.client [project folder]
"""
from __future__ import annotations

import html
import sys
import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, Qt, QTimer, QTranslator
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

from .. import __version__, api
from ..core import project2
from ..core.newproject import guess_names
from ..core.project import Target
from . import APP_NAME, prefs
from .gcguard import GuiGarbageCollector
from .home import HomeView
from .i18n import _, language, plural, set_language
from .logview import LogPanel
from .services.run import duration_text
from .services.worker import ApiWorker, Machine, machine_info
from .settingsdlg import SettingsDialog
from .widgets import NEW, QSS, WARN, label

EXTRA_QSS = ("QSlider:horizontal { margin-left: 6px; margin-right: 6px; }"
             "QMenuBar { background: #1c1f26; } QMenuBar::item:selected { background: #2b3446; }")
HELP = """<h2>{title}</h2>
<p>{p1}</p><p>{p2}</p>"""

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
    """The open project as the shell knows it: its folder, its definition and its history (for the Log). The model of
    the views (``Main.s``) shares the definition and the history."""

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
    """A centred dim note in an empty place (tests stand it in for a page)."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        v = QVBoxLayout(self)
        v.addStretch(1)
        t = label(title, "dim")
        t.setAlignment(Qt.AlignCenter)
        v.addWidget(t)
        v.addStretch(1)


class PostHolder(QWidget):
    """Post's place in Inspect. The page is made the first time it is shown: its engines load then, not when the
    project opens (on the CPU, a layer's engine takes seconds)."""

    def __init__(self, make, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        self._make = make
        self.page = None
        self._v = QVBoxLayout(self)
        self._v.setContentsMargins(0, 0, 0, 0)

    def ensure(self):
        if self.page is None:
            self.page = self._make()
            self._v.addWidget(self.page)
        return self.page

    def showEvent(self, e):                                          # noqa: N802 - Qt's name
        super().showEvent(e)
        self.ensure()


class Main(QMainWindow):
    def __init__(self, worker: ApiWorker | None = None):
        super().__init__()
        self.worker = worker or ApiWorker(self)
        self.worker.done.connect(self._worker_done)
        self.worker.failed.connect(self._worker_failed)
        self.project: OpenProject | None = None
        self.s = None                                # the ProjectModel of the views
        self.run = None                              # the RunController of the project
        self.render = None                           # the RenderWorker of Post (made when Post is first shown)
        self.machine: Machine | None = None
        self.home: HomeView | None = None
        self.proj_screen: QWidget | None = None
        self.graph = self.tune = self.frames = self.browser = self.steps = self.post_holder = None
        self.mode = 0
        self._last_step = None                       # the block Inspect showed before Post
        self._pending = False
        self._run_widget = None
        self._thumb_of = None                        # the Post image last left as the project's Home picture
        self._flash = ""                             # a word said for a few seconds in place of the status line
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._flash_end)
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
        act(f, _("Import folder…"), self.add_sources, "Ctrl+I", enabled=has)
        self.a_export = act(f, _("Export image…"), self.export, "Ctrl+E", enabled=has)
        if spec:
            act(f, _("Export project as script…"), new=True)
        f.addSeparator()
        act(f, _("Close project"), self.close_project, "Ctrl+W", enabled=has)
        act(f, _("Quit"), self.close, "Ctrl+Q")
        e = mb.addMenu(_("Edit"))
        self.a_undo = act(e, _("Undo"), self.undo, "Ctrl+Z", enabled=has)
        self.a_redo = act(e, _("Redo"), self.redo, "Ctrl+Y", enabled=has)
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
        self.a_run = act(r, _("Run all"), self.launch, "F5", enabled=has)
        self.a_here = act(r, _("Run to selected block"), self.run_to_here, "Shift+F5", enabled=has)
        r.addSeparator()
        self.a_stop = act(r, _("Stop"), self.stop_run, enabled=False)
        h = mb.addMenu(_("Help"))
        act(h, _("About"), self.about, "F1")
        if spec:
            act(h, _("Glossary"), new=True)
        for m in (f, e, v, r, h):
            m.setToolTipsVisible(True)
        self._update_actions()

    def _update_actions(self):
        """What the menus allow now: Undo / Redo, Stop, Export."""
        s = self.s
        if s is None:
            return
        try:
            self.a_undo.setEnabled(s.can_undo() or self._post_shown())
            self.a_redo.setEnabled(s.can_redo() or self._post_shown())
            self.a_stop.setEnabled(self.run is not None and self.run.running)
            self.a_export.setEnabled(any(p.rendered for p in s.paths))
            self.a_here.setEnabled(s.path is not None and s.sel_step not in ("sources", "final"))
        except RuntimeError:                         # the menus are being rebuilt
            pass

    # ------------------------------------------------------------ Home
    def show_home(self, last=None):
        if self.project is not None:
            self._leave_thumbnail()                  # Home shows the open project as Post last showed it
        if self.home is None:
            self.home = HomeView(self.worker, can_import=True, last=last)
            self.home.openProject.connect(self.open_project)
            self.home.createProject.connect(self.create_project)
            self.home.importFolder.connect(self.import_folder_as_new)
            self.screens.insertWidget(0, self.home)
        else:
            self.home.rescan()
        self.screens.setCurrentWidget(self.home)
        self._flash = ""
        self._flash_timer.stop()
        self.setWindowTitle(f"{APP_NAME} · " + _("Home"))
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
    def _device(self) -> tuple[bool, str | None]:
        """(a GPU computes, the device asked: None for automatic) from Settings > Compute and the machine."""
        dev = prefs.get("device") or "auto"
        if dev == "cpu" or (self.machine is not None and not self.machine.gpus):
            return False, "cpu"
        return True, (None if dev == "auto" else dev)

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
        if self.close_project(go_home=False) is False:              # unsaved edits kept: Cancel
            return False
        self.project = OpenProject(root, d)
        self.project.log("Project created" if created else "Project opened")
        prefs.add_recent(root.as_posix(), d.name, d.target.name or "")
        self.make_model()
        self.mode = 0
        self._last_step = None
        self.build_menus()
        self.build_project()
        self.run.attach()                            # a run that goes on (started before, or elsewhere): follow it
        return True

    def make_model(self):
        """The model, the run controller and their signals, once per project."""
        from .model import ProjectModel
        from .services.run import RunController
        gpu, device = self._device()
        s = ProjectModel(self.project.definition, self.worker, history=self.project.history, gpu=gpu, device=device,
                         readonly=bool(getattr(self, "readonly", False)))      # a test harness may ask: never write
        self.s = s
        for sig in (s.pathsChanged, s.currentChanged, s.costChanged, s.sourcesChanged, s.readyChanged):
            sig.connect(self.request_refresh)
        s.dirtyChanged.connect(lambda _d: self._title())
        self.run = RunController(s, window=self, console=lambda: getattr(self, "logp", None))
        self.run.runningChanged.connect(lambda _on: self.request_refresh())
        self.run.updated.connect(self._run_updated)
        self.run.finished.connect(self._run_finished)
        self.run.logRequested.connect(lambda: self.b_log.setChecked(True))
        self._run_widget = self.run.status_widget()
        sb = self.statusBar()                       # the run's progress and Stop, left of the Log toggle
        sb.removeWidget(self.b_log)
        sb.addPermanentWidget(self._run_widget)
        sb.addPermanentWidget(self.b_log)
        self.b_log.show()

    def render_worker(self):
        """Post's render worker: one per project, kept while it is open (its engines stay loaded)."""
        if self.render is None:
            from .services.render import RenderWorker, default_device_pref
            gpu, _device = self._device()
            self.render = RenderWorker(self.s.p, device=default_device_pref() if gpu else "cpu")
        return self.render

    def _leave_thumbnail(self):
        """The composition Post last showed becomes the project's picture on Home (``cache/thumb.jpg``, written by the
        worker)."""
        holder = self.post_holder
        page = holder.page if holder is not None else None
        img = getattr(getattr(getattr(page, "view", None), "src", None), "img1", None)
        if img is None or self.s is None or self.s.readonly or self.project is None:
            return
        if img is self._thumb_of:
            return                                   # written already
        self._thumb_of = img
        from .home import THUMB
        self.worker.submit(f"thumb:{self.project.root}", _write_thumb, self.project.root / THUMB, img.copy())

    def show_project(self):
        if self.proj_screen is not None:
            self.screens.setCurrentWidget(self.proj_screen)
            self.refresh()

    def ask_save(self) -> str:
        """The project has edits not saved: "save", "discard" or "cancel", as the user answers (offscreen, a test
        harness: "save", as before Save existed)."""
        if QApplication.platformName() == "offscreen":
            return "save"
        box = QMessageBox(QMessageBox.Warning, _("Unsaved changes"),
                          _('Save the changes made to "{name}"?').format(name=self.project.name), parent=self)
        box.setInformativeText(_("Without saving, the changes since the last save are lost (the computed results "
                                 "stay)."))
        b_save = box.addButton(_("Save"), QMessageBox.AcceptRole)
        b_drop = box.addButton(_("Don't save"), QMessageBox.DestructiveRole)
        box.addButton(_("Cancel"), QMessageBox.RejectRole)
        box.setDefaultButton(b_save)
        box.exec()
        return "save" if box.clickedButton() is b_save else "discard" if box.clickedButton() is b_drop else "cancel"

    def close_project(self, go_home=True) -> bool:
        """Close the project, asking first when it has edits not saved (a run in progress goes on: opening the
        project follows it again). False when the user cancels."""
        save = True
        if self.project is not None:
            self._flush_post()
            if self.s is not None and self.s.dirty:
                answer = self.ask_save()
                if answer == "cancel":
                    return False
                save = answer == "save"
        if self.project is not None:
            if self.run is not None:
                self.run.close()
                if self._run_widget is not None:
                    self.statusBar().removeWidget(self._run_widget)
                    self._run_widget.deleteLater()
                    self._run_widget = None
                self.run.deleteLater()
                self.run = None
            self._leave_thumbnail()
            self._flush_post()
            if self.s is not None:
                from .services.inspect import _services
                sv = _services.get(self.s)
                self.s.close(save)
                if sv is not None:
                    sv.stop()
            if self.render is not None:
                self.render.stop()
                self.render = None
            self.project = None
            self.s = None
            if self.proj_screen is not None:
                self.screens.removeWidget(self.proj_screen)
                self.proj_screen.deleteLater()
                self.proj_screen = None
            self.graph = self.tune = self.frames = self.browser = self.post_holder = None
            self.build_menus()
        if go_home:
            self.show_home()
        return True

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
        # the modes: Graph, and Inspect (which holds Frames and Post)
        self.modes.addWidget(self.graph_page())
        self.modes.addWidget(self.inspect_page())
        self.proj_screen = screen
        self.screens.addWidget(screen)
        self.screens.setCurrentWidget(screen)
        if self.run is not None:
            self.run.retranslate()                   # the console of the last run, in the new Log panel
        self.set_mode(self.mode)

    def graph_page(self) -> QWidget:
        """Graph: the toolbar of optional blocks, the GraphView and the block Panel."""
        from .graph import GraphPage
        from .steps import Steps
        self.steps = Steps(self.s)
        page = GraphPage(self.s, self.steps)
        page.openBlock.connect(lambda: self.set_mode(2 if self.s.sel_step == "final" else 1))
        page.openPost.connect(lambda: self.set_mode(2))
        page.openLayer.connect(self.open_layer)
        page.addSources.connect(self.add_sources)
        page.addSession.connect(self.add_session)
        page.variantMade.connect(self._variant_made)
        page.changed.connect(self.request_refresh)
        self.graph = page
        if self.run is not None:
            self.run.watch_graph(page.graph)
        return page

    def inspect_page(self) -> QWidget:
        """Inspect: TuneView, with Frames (FramesView and its browser) and the Post page."""
        from .frames import FramesView
        from .sources import SourcesView
        from .tune import TuneView
        s = self.s
        self.browser = SourcesView(s)
        self.frames = FramesView(s, self.browser)
        self.post_holder = PostHolder(self.post_page)
        tune = TuneView(s, self.steps, self.frames, self.post_holder)
        tune.back.connect(lambda: self.set_mode(0))
        tune.pick.connect(self._strip_pick)
        tune.workflow.connect(lambda _i: self.request_refresh())
        tune.runHere.connect(self.run_to_here)
        tune.openPost.connect(lambda: self.set_mode(2))
        tune.removeTool.connect(self._remove_tool)
        tune.toasted.connect(self.flash)
        tune.can_post = self.browser.can_post = lambda: True    # Post takes Inspect's images as layers
        tune.sendToPost.connect(self._send_to_post)
        self.browser.sendToPost.connect(self._send_to_post)
        self.frames.imported.connect(self.request_refresh)
        self.frames.stackMode.connect(lambda m: s.set_stack_mode(m))
        self.browser.goChain.connect(lambda k: self.select(step=k) if s.path is not None else None)
        self.tune = tune
        return tune

    def post_page(self) -> QWidget:
        """Post: PostPage3, on the project's render worker (made the first time Post is shown)."""
        from .post3 import PostPage3
        from .services.inspect import resolve_key
        page = PostPage3(self.s, worker=self.render_worker())
        page.set_image_resolver(lambda key, p=self.s.p: resolve_key(p, key))   # the layers Inspect sent, reopened
        page.goWorkflow.connect(lambda: self.set_mode(0))
        page.view.goResults.connect(self._go_results)
        page.view.goProcessing.connect(lambda pid, k: (self.select(index=self.s.index_of(pid), step=k),
                                                       self.set_mode(1)))
        return page

    @property
    def post(self):
        """The Post page, made now if it was not yet."""
        return self.post_holder.ensure() if self.post_holder is not None else None

    def _flush_post(self):
        """Post's last edit into the project's definition (it is saved just after an edit, when the event loop comes
        back: a close or a rebuild may come first)."""
        page = self.post_holder.page if self.post_holder is not None else None
        view = getattr(page, "view", None)
        if view is not None and hasattr(view, "flush"):
            try:
                view.flush()
            except RuntimeError:                     # gone with the window
                pass

    def _send_to_post(self, name: str, source, key):
        """Send to Post of a viewer of Inspect: the image becomes a layer of Post, which is shown."""
        self.post.add_image_layer(name, source, key)
        self.set_mode(2)

    def open_layer(self, lid: int):
        """Post, with this layer selected (a Stars and background block's Open in Post)."""
        self.set_mode(2)
        if self.post is not None:
            self.post.view.select_layer(lid)

    def _post_shown(self) -> bool:
        return self.s is not None and self.mode == 2 and self.post_holder is not None and \
            self.post_holder.page is not None

    def select(self, index=None, step=None):
        """Select a workflow and a block (the scope follows the block, as a click on the graph does)."""
        if self.graph is not None:
            self.graph.select(index, step)
        else:
            self.s.select(index, step)

    def set_mode(self, i: int):
        """0 Graph, 1 Inspect the chosen block, 2 Post (Inspect of the Post block)."""
        s = self.s
        if self.project is None:
            return
        if self.screens.currentWidget() is not self.proj_screen:
            self.screens.setCurrentWidget(self.proj_screen)
        was_post = self.mode == 2 and s is not None and s.sel_step == "final"
        if s is not None and i == 1 and s.sel_step == "composition":
            # a Stars and background block is a layer of Post: Inspect shows it there
            self.open_layer(s.sel_layer) if s.sel_layer is not None else self.set_mode(2)
            return
        if s is not None:
            if i == 2:
                if s.sel_step not in ("final", "composition"):
                    self._last_step = s.sel_step
                s.scope = None
                s.pending_change = None
                if s.sel_step != "final":
                    s.select(step="final")
            elif i == 1 and s.sel_step == "final":
                self.select(step=self._last_step or ("post" if s.path is not None else "sources"))
        self.mode = i
        self.modes.setCurrentIndex(0 if i == 0 else 1)
        self.mode_btns[i].setChecked(True)
        if i == 2 and not was_post and self.post_holder is not None and self.post_holder.page is not None:
            self.post_holder.page.refresh()
        if i == 0 and self.graph is not None and getattr(self.graph, "graph", None) is not None:
            self.graph.graph.rebuild()             # the arrows of the images sent to Post meanwhile
        self.refresh()

    def _strip_pick(self, k: str):
        if k == "final":
            self.set_mode(2)
        else:
            self.request_refresh()

    def _go_results(self, pid: int):
        """'Show workflow →' of a layer: Inspect on that workflow's Output."""
        self._last_step = "post"
        self.select(index=self.s.index_of(pid), step="post")
        self.set_mode(1)

    def _variant_made(self, pid: int, orig: int):
        """A variant opens in Inspect compared with its original."""
        if self.tune is not None:
            self.tune.cmp_pid = orig

    def _remove_tool(self):
        if self.graph is not None:
            self.graph.remove_tool(self.graph.current_node(), None)

    # ------------------------------------------------------------ refresh
    def _title(self):
        """The window's title: the project's name, a star when it has edits not saved (issue #1)."""
        if self.project is not None:
            star = " *" if self.s is not None and self.s.dirty else ""
            self.setWindowTitle(f"{APP_NAME} · {self.project.name}{star}")

    def project_name(self) -> str:
        pr = self.project
        name = "<b>" + html.escape(pr.name) + "</b>"
        return name + ("  ·  " + html.escape(pr.target) if pr.target else "")

    def request_refresh(self, *_a):
        if not self._pending:
            self._pending = True
            QTimer.singleShot(0, self.refresh)

    def refresh(self):
        self._pending = False
        pr = self.project
        if pr is None or self.proj_screen is None:
            return
        s = self.s
        self.title.setText(self.project_name())
        self._title()
        text, gpu = machine_text(self.machine, prefs.get("device"))
        self.machine_lb.setText(text)
        self.machine_lb.setStyleSheet("" if gpu else f"color: {WARN};")
        g = self.machine.gpu(prefs.get("device")) if self.machine is not None else None
        self.machine_lb.setToolTip(f"{g.device} · {g.memory_gb:.0f} GB" if g is not None else "")
        btn = 0 if self.mode == 0 else (2 if s is not None and s.sel_step == "final" else 1)
        if self.mode != btn:
            self.mode = btn
        self.mode_btns[btn].setChecked(True)
        self.update_run_button()
        if self.logp.isVisible():
            self.logp.refresh()
        self.update_spec_note()
        self._update_actions()

    def update_run_button(self):
        """The Run button (Run · the time of Run all) and the status line (workflows, what is left to run)."""
        s = self.s
        frames = bool(s is not None and s.sources_loaded)
        running = self.run is not None and self.run.running
        n = len(s.paths) if s is not None else 0
        cost = s.cost() if s is not None and s.ready and s.paths else None
        todo = cost is not None and cost.jobs > 0
        if running:
            self.bt_run.setText(_("Running · {pct} %").format(pct=int(round(100 * self.run.progress()))))
            self.bt_run.setEnabled(True)
            self.bt_run.setToolTip(_("Show the run"))
        else:
            self.bt_run.setText(_("Run") + (f"  ·  {duration_text(cost.seconds, cost.measured)}" if todo else ""))
            self.bt_run.setEnabled(frames and todo)
            self.bt_run.setToolTip("" if todo else (_("No frames yet") if not frames else
                                                    _("Nothing to compute: all up to date.") if cost is not None
                                                    else ""))
        self.a_run.setEnabled(frames and (todo or running))
        if not n:
            line = _("No workflow yet") if s is not None and s.ready else ""
        else:
            line = plural(n, "{n} workflow", "{n} workflows")
            if s.error:
                line += "  ·  " + _("cannot be read")
            elif not s.ready:
                line += "  ·  " + _("reading the project…")
            elif todo:
                line += "  ·  " + _("to run: {t}").format(t=duration_text(cost.seconds, cost.measured))
            elif cost is not None:
                line += "  ·  " + _("all up to date")
        self._line = line
        self.status.setText(self._flash or line)

    def _run_updated(self):
        if self.run is not None and self.run.running and self.proj_screen is not None:
            self.update_run_button()

    def _run_finished(self, status: str, text: str):
        self.flash(text)
        if self.render is not None:
            self.render.refresh()
        self.request_refresh()

    def update_spec_note(self):
        self.spec_note.setText(_("Spec mode · ◇ not in the engine yet") + "   " if prefs.spec() else "")
        self.spec_note.setStyleSheet(f"color: {NEW};")

    def _worker_done(self, channel, _n, result):
        if channel == "machine":
            self.machine = result
            if self.s is not None:
                self.s.set_machine(*self._device())
            self.refresh()
        elif channel == "cache.usage":
            self._confirm_clear(result)
        elif channel == "cache.clear":
            freed, n = result
            self.flash(plural(n, "{size} freed in {n} project", "{size} freed in {n} projects",
                              size=_gb_text(freed)))

    def _worker_failed(self, channel, _n, text):
        if channel.startswith("cache."):
            QMessageBox.warning(self, _("Clear cache"), _("The cache cannot be cleared: {why}").format(
                why=(text or "?").splitlines()[0]))

    # ------------------------------------------------------------ files
    def save(self):
        """Save: the edits written into the project (nothing is written before, issue #1)."""
        self._flush_post()
        if self.s is not None:
            self.s.flush()
        self.flash(_("Project saved"))

    def save_as(self):
        """The definition of the project in another folder (its results stay where they are), then opened."""
        if self.project is None:
            return
        if self.s is not None:
            self.s.flush()
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
        """A project made with an older version: it opens once imported into the workspace; its folder is never
        written."""
        d = self.import_dialog(root)
        ok = d.exec()
        new = d.result_root
        d.release()
        d.deleteLater()
        if ok and new is not None:
            if self.open_project(new):
                self.project.log('Imported from "{folder}"', folder=Path(root).as_posix())
                self.flash(_("Project imported: {name}").format(name=self.project.name))

    def import_dialog(self, root: Path):
        from .previous import PreviousVersionDialog
        return PreviousVersionDialog(root, self)

    def import_box(self, root: Path) -> QMessageBox:
        """A note that the folder holds a project of an older version (tests check its text)."""
        return QMessageBox(QMessageBox.Information, _("Open project"), _(
            "“{name}” was made with an older version. Import it into the workspace to open it here; its folder "
            "is never changed.").format(name=Path(root).name), QMessageBox.Ok, self)

    # ------------------------------------------------------------ frames, runs, export, undo
    def add_sources(self):
        """File > Import folder…: frames into the open project."""
        if self.s is None:
            return
        from .services.inspect import service_of
        from .sources import import_folder
        import_folder(self.s, self, service=service_of(self.s))

    def add_session(self):
        """Add session…: a folder of frames as a new session."""
        if self.s is None:
            return
        from .services.inspect import service_of
        from .sources import import_folder
        import_folder(self.s, self, new_session=True, service=service_of(self.s))

    def import_frames(self, folder: str):
        """The frames of a folder into the open project, after Import folder as new project."""
        if self.s is None:
            return
        from .services.inspect import service_of
        from .sources import import_folder
        if import_folder(self.s, self, folder=folder, service=service_of(self.s)):
            self.select(step="sources")
            self.set_mode(1)

    def export(self):
        """File > Export image… (the Export dialog of Post, on the composition of its layers)."""
        if self.s is None:
            return
        if not any(p.rendered for p in self.s.paths):
            self.flash(_("Nothing to export yet: no workflow is computed."))
            return
        self.post.export(self)

    def launch(self):
        """Run all (F5): the plan dialog, or the progress of the run in progress."""
        if self.run is not None:
            self.run.launch()

    def run_to_here(self):
        """Run to the selected block (Shift+F5)."""
        if self.run is not None:
            self.run.run_to_here()

    def stop_run(self):
        """Stop the run: the job in progress keeps its checkpoints; the next run resumes."""
        if self.run is not None:
            self.run.stop()

    def undo(self):
        """Undo: in Post, the composition's last change; elsewhere the workflows' last change."""
        if self._post_shown():
            self.post_holder.page.view.do_undo()
        elif self.s is not None and self.s.undo():
            self.s.log("Undone")
            self.flash(_("Undone"))

    def redo(self):
        if self._post_shown():
            self.post_holder.page.view.do_redo()
        elif self.s is not None and self.s.redo():
            self.s.log("Redone")
            self.flash(_("Redone"))

    def clear_cache(self):
        """Clear the unused results of the closed projects of the workspace, after a confirmation (read in the
        worker: it walks the stores)."""
        open_root = self.project.root if self.project is not None else None
        self.worker.submit("cache.usage", _closed_usage, prefs.get("workspace"), open_root)

    def _confirm_clear(self, found):
        total = sum(b for _r, b in found)
        if not total:
            QMessageBox.information(self, _("Clear cache"), _("The closed projects hold no unused results."))
            return
        text = plural(len(found), "{size} of results no workflow uses, in {n} closed project. Clear them?",
                      "{size} of results no workflow uses, in {n} closed projects. Clear them?",
                      size=_gb_text(total))
        if QMessageBox.question(self, _("Clear cache"), text) == QMessageBox.Yes:
            self.worker.submit("cache.clear", _clear_closed, [r for r, b in found if b])

    # ------------------------------------------------------------ settings, language, spec mode, log
    def open_settings(self, page: str | None = None):
        d = SettingsDialog(self.worker, self.machine, can_clear=True, parent=self)
        if page:
            d.show_page(page)
        d.languageChanged.connect(self.change_language)
        d.deviceChanged.connect(self.set_device)
        d.specChanged.connect(self.set_spec)
        d.workspaceChanged.connect(lambda _w: self.home.rescan() if self.home is not None else None)
        d.homeChanged.connect(lambda: self.home.rescan() if self.home is not None else None)
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
            self._flush_post()                       # Post is made again from the project: its last edit first
            self.build_project()
            if on_home:
                self.show_home()

    def set_device(self, device: str):
        prefs.set("device", device)
        if self.s is not None:
            self.s.set_machine(*self._device())
        if self.render is not None:                  # Post picks its device again before its next render
            from .services.render import default_device_pref
            self.render.pref = default_device_pref() if self._device()[0] else "cpu"
            self.render._busy_at, self.render._last_busy = 0.0, "?"
        self.refresh()

    def set_spec(self, on: bool):
        if on != prefs.spec():
            prefs.set("spec_mode", on)
        self.build_menus()
        self.update_spec_note()
        if self.graph is not None:
            self.graph.request_refresh()
        if self.tune is not None:
            self.tune.request_refresh()
        self.refresh()

    def toggle_log(self, on):
        logp = getattr(self, "logp", None)
        if logp is not None and self.project is not None:
            logp.setVisible(on)
            if on:
                logp.refresh()

    def flash(self, text):
        """A word for four seconds, in place of the status line (the status bar's own message would leave the line
        under it when the bar is laid out again meanwhile)."""
        self._flash = text
        self.status.setText(text)
        self._flash_timer.start(4000)

    def _flash_end(self):
        self._flash = ""
        if self.project is not None and self.proj_screen is not None and \
                self.screens.currentWidget() is self.proj_screen:
            self.status.setText(getattr(self, "_line", ""))
        else:
            self.status.setText(prefs.get("workspace"))

    def about(self):
        d = self.about_dialog()
        d.exec()
        d.deleteLater()

    def about_dialog(self) -> QDialog:
        d = QDialog(self)
        d.setWindowTitle(_("About"))
        lv = QVBoxLayout(d)
        t = QLabel(HELP.format(
            title=f"{APP_NAME} {__version__}",
            p1=_("Deep-sky images from a camera's raw frames: calibration, registration, stacking, astrometry, "
                 "color calibration on Gaia, denoise, stars and gradient, then the final image in Post. Each "
                 "workflow keeps its own settings; a block shared by several workflows is computed once."),
            p2=_("Shortcuts: Ctrl+1 Graph, Ctrl+2 Inspect, Ctrl+3 Post, Enter opens the selected block, Esc back "
                 "to the graph, Ctrl+Z / Ctrl+Y undo / redo, F5 run.")))
        t.setWordWrap(True)
        t.setTextFormat(Qt.RichText)
        lv.addWidget(t)
        from PySide6.QtWidgets import QTextBrowser

        from .credits import data_sources_html
        src = QTextBrowser()                         # the data the product reads, credited (credits.py)
        src.setObjectName("data_sources")
        src.setOpenExternalLinks(True)
        src.setHtml(data_sources_html())
        lv.addWidget(src, 1)
        b = QPushButton(_("Close"))
        b.clicked.connect(d.accept)
        lv.addWidget(b, 0, Qt.AlignRight)
        d.resize(760, 640)
        return d

    def closeEvent(self, e):
        if self.close_project(go_home=False) is False:              # unsaved edits kept: Cancel
            e.ignore()
            return
        self.worker.wait(5)                          # what closing the project left to write (Home's thumbnail)
        self.worker.stop()
        super().closeEvent(e)


def _gb_text(n_bytes: int) -> str:
    from .i18n import number
    gb = n_bytes / 1e9
    return number(gb, 1 if gb >= 0.1 else 2) + " " + _("GB")


def _write_thumb(path: Path, rgb) -> None:
    """A small JPEG of a composition (worker thread): what Home shows of the project."""
    from PIL import Image
    im = Image.fromarray(rgb)
    im.thumbnail((640, 400))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    im.save(tmp, format="JPEG", quality=88)
    tmp.replace(path)


def _closed_usage(workspace: str, open_root) -> list[tuple[str, int]]:
    """(project folder, unused bytes) of the projects of the workspace, the open one aside (worker thread)."""
    out = []
    for info in api.list_projects(workspace):
        root = Path(info.root)
        if open_root is not None and root.resolve() == Path(open_root).resolve():
            continue
        if info.running or info.error:
            continue
        try:
            u = api.cache_usage(api.open_project(root))
        except Exception:  # noqa: BLE001 - a project that cannot be read is left alone
            continue
        out.append((str(root), sum(i.bytes for i in u.unused if not i.resumable)))
    return out


def _clear_closed(roots: list[str]) -> tuple[int, int]:
    freed = 0
    for r in roots:
        freed += api.clear_unused(api.open_project(r))
    return freed, len(roots)


def _warm_up():
    """What the first render would otherwise load while the user waits (core/determinism.warm_up)."""
    try:
        from ..core.determinism import warm_up
        warm_up()
    except Exception:  # noqa: BLE001 - a warm-up only: the render loads it anyway
        pass


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    prefs.load()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app._gc = GuiGarbageCollector(app)                  # cycles collected on the GUI thread only (gcguard.py)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(QSS + EXTRA_QSS)
    apply_language(prefs.get("language"))

    def show_error(t, v, tb):
        QMessageBox.critical(None, APP_NAME, "".join(traceback.format_exception(t, v, tb))[-3000:])

    sys.excepthook = show_error
    threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
    w = Main()
    w.resize(1680, 1000)
    w.showMaximized()
    if argv:
        w.open_project(argv[0])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
