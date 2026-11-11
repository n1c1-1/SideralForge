"""The main window of the client.

Home: the recent projects and the sections of folders. A project has three modes over the same blocks:
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
Post page; made the first time Inspect is shown or reached). Classes: ``Main`` (the window), ``OpenProject`` (the open
project as the shell knows it), ``PageHolder`` (makes a page, Inspect or Post, the first time it is shown).

    python -m astro_gpu.client [project folder]
"""
from __future__ import annotations

import html
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, Qt, QTimer, QTranslator
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, api
from ..core import project2
from ..core.project import Target
from . import APP_MOTTO, APP_NAME, drafts, prefs
from . import layout as lay
from .gcguard import GuiGarbageCollector
from .home import HomeView, ask_project_name, forget_state
from .i18n import _, language, plain_error, plural, set_language
from .icons import app_icon
from .icons import icon as make_icon
from .icons import IconLabel, icon_only, set_icon
from .logview import LogPanel
from .services.run import duration_text
from .services.worker import ApiWorker, Machine, machine_info
from .settingsdlg import SettingsDialog, auto_device, device_items
from .widgets import NEW, QSS, WARN, cap_dialog, clear, label, size_within

EXTRA_QSS = ("QSlider:horizontal { margin-left: 6px; margin-right: 6px; }"
             "QMenuBar { background: #1c1f26; } QMenuBar::item:selected { background: #2b3446; }")
HELP = """<h2>{title}</h2>
<p><i>{motto}</i></p><p>{p1}</p><p>{p2}</p>"""

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


def no_block(w) -> bool:
    """A workflow with no block placed after its source (GUI exploration 6.5: it is not "up to date", it has nothing
    to compute); a Combine channels computes itself, a workflow made before the free graph has its blocks; an image is
    computed to go to Post as it is, whether stretched or linear."""
    from ..core.project2 import is_fusion
    from ..core.workflow import source_of
    return w.blocks is not None and not is_fusion(w) and source_of(w) not in ("stretched", "image") and \
        all(b.get("block") == "output" for b in w.blocks)


def images_only(s) -> bool:
    """Whether every workflow starts from an Image block (no Session): what is missing then is its file, not "lights
    or image" (GUI exploration round 2: the "Already stretched image" template)."""
    srcs = [q.source for q in s.paths if not (q.source and "fusion" in q.source)]
    return bool(srcs) and all(isinstance(x, dict) and "path" in x for x in srcs)


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


class PageHolder(QWidget):
    """A page's place, the page made the first time it is shown or reached (``ensure``), not when the project opens:
    Post in Inspect (its engines: on the CPU, a layer's engine takes seconds), Inspect in the modes (#115: its viewers,
    frames page and browser froze the screen 0.2 s as a project opened, hidden)."""

    def __init__(self, make, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        self._make = make
        self._making = False                         # what the page's making reaches for meanwhile is not made twice
        self.page = None
        self._v = QVBoxLayout(self)
        self._v.setContentsMargins(0, 0, 0, 0)

    def ensure(self):
        if self.page is None and not self._making:
            self._making = True
            try:
                page = self._make()
            finally:
                self._making = False
            self.page = page
            self._v.addWidget(page)
        return self.page

    def showEvent(self, e):                                          # noqa: N802 - Qt's name
        super().showEvent(e)
        self.ensure()


#: an edit is copied for the recovery after a crash this long after it (ms)
RECOVERY_MS = 3000


class Banner(QFrame):
    """A failure said without stopping the user: a strip over the window, until it is closed or replaced; ``ask``: with
    what can be done about it."""

    def __init__(self):
        super().__init__()
        self.setObjectName("banner")
        self.setStyleSheet("QFrame#banner { background: #3a2f1c; border-bottom: 1px solid #5a4a26; }")
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 5, 6, 5)
        h.addWidget(IconLabel("warning", 16, "warning"))
        self.text = label("", "", wrap=True)
        h.addWidget(self.text, 1)
        self.actions_box = QHBoxLayout()
        h.addLayout(self.actions_box)
        self.close_b = icon_only(QToolButton(), "remove", _("Close"))
        self.close_b.setAutoRaise(True)
        self.close_b.clicked.connect(self.hide)
        h.addWidget(self.close_b)
        self.hide()

    def say(self, text: str):
        self.ask(text)

    def ask(self, text: str, *actions):
        """``actions``: (words, what they do); each closes the strip first."""
        clear(self.actions_box)
        for words, fn in actions:
            b = QPushButton(words)
            b.clicked.connect(lambda _x=False, f=fn: (self.hide(), f()))
            self.actions_box.addWidget(b)
        self.close_b.setVisible(not actions)
        self.text.setText(text)
        self.show()


class Main(QMainWindow):
    def __init__(self, worker: ApiWorker | None = None):
        super().__init__()
        self.worker = worker or ApiWorker(self)
        self.worker.done.connect(self._worker_done)
        from .services.selftest import job as card_test_job
        card_test_job().changed.connect(self._card_tested, Qt.QueuedConnection)
        self.worker.failed.connect(self._worker_failed)
        self.project: OpenProject | None = None
        self.s = None                                # the ProjectModel of the views
        self._recovery_mine = False                  # the recovery copy of this project is this session's
        self._recovery_timer = QTimer(self)
        self._recovery_timer.setSingleShot(True)
        self._recovery_timer.setInterval(RECOVERY_MS)
        self._recovery_timer.timeout.connect(self._write_recovery)
        drafts.clean_orphans()
        self.run = None                              # the RunController of the project
        self.render = None                           # the RenderWorker of Post (made when Post is first shown)
        self.machine: Machine | None = None
        self.home: HomeView | None = None
        from .services import extracts
        # a catalogue's download ends (it goes on with Settings closed): the blocks waiting for it can run
        extracts.job().finished.connect(self.sky_changed, Qt.QueuedConnection)
        self.proj_screen: QWidget | None = None
        self._inspect_holder: PageHolder | None = None   # Inspect's place in the modes (its views: ``tune``…)
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
        self.banner = Banner()
        root = QWidget()
        rv = QVBoxLayout(root)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        rv.addWidget(self.banner)
        rv.addWidget(self.screens, 1)
        self.setCentralWidget(root)
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
        self.setMinimumSize(*lay.MIN_WINDOW)
        self.size_range = None                       # issue #44: the window's size range (layout.py)
        self.compact = False
        self.build_menus()
        # Tab: focus goes on in a field, as in a form; elsewhere it hides the panels (a reference editor's habit)
        for seq, fn in (("Tab", self._tab), ("Shift+Tab", self._shift_tab), ("Backtab", self._shift_tab),
                        ("Shift+Backtab", self._shift_tab)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(fn)
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

        def act(menu, text, fn=None, key=None, checkable=False, enabled=True, new=False, role=None):
            a = QAction(text + ("  ◇" if new else ""), menu)
            if role:
                a.setIcon(make_icon(role, 16))
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
        act(f, _("Home"), self.show_home, "Ctrl+H", role="home")
        act(f, _("New project"), self.new_project, "Ctrl+N", role="new")
        act(f, _("Open images…"), self.open_images, role="open")
        act(f, _("Open project…"), self.open_other, "Ctrl+O", role="open")
        rec = f.addMenu(_("Recent projects"))
        for r in prefs.recent():
            text = r["name"] + (f" · {r['target']}" if r.get("target") else "")
            a = act(rec, text, lambda _x=False, path=r["path"]: self.open_project(path))
            a.setToolTip(r["path"])
        rec.setEnabled(bool(prefs.recent()))
        rec.setToolTipsVisible(True)
        f.addSeparator()
        act(f, _("Save"), self.save, "Ctrl+S", enabled=has, role="save")
        act(f, _("Save as…"), self.save_as, enabled=has, role="save")
        f.addSeparator()
        self.a_export = act(f, _("Export image…"), self.export, "Ctrl+E", enabled=has, role="export")
        if spec:
            act(f, _("Export as script…"), new=True)
        f.addSeparator()
        act(f, _("Close project"), self.close_project, "Ctrl+W", enabled=has, role="close_project")
        act(f, _("Quit"), self.close, "Ctrl+Q", role="quit")
        e = mb.addMenu(_("Edit"))
        self.a_undo = act(e, _("Undo"), self.undo, "Ctrl+Z", enabled=has, role="undo")
        self.a_redo = act(e, _("Redo"), self.redo, "Ctrl+Y", enabled=has, role="redo")
        self.a_redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        e.addSeparator()
        act(e, _("Settings…"), self.open_settings, "Ctrl+,", role="settings")
        v = mb.addMenu(_("View"))
        act(v, _("Graph"), lambda: self.set_mode(0), "Ctrl+1", enabled=has, role="graph")
        act(v, _("Inspect"), lambda: self.set_mode(1), "Ctrl+2", enabled=has, role="inspect")
        act(v, _("Post"), lambda: self.set_mode(2), "Ctrl+3", enabled=has, role="layers")
        act(v, _("Report"), lambda: self.set_mode(3), "Ctrl+4", enabled=has)
        v.addSeparator()
        self.a_hide = act(v, _("Hide panels") + "\tTab", lambda: self.toggle_panels("hidden"), None, True, enabled=has)
        self.a_hide_r = act(v, _("Hide right") + "\t" + QKeySequence("Shift+Tab").toString(QKeySequence.NativeText),
                            lambda: self.toggle_panels("right_hidden"), None, True, enabled=has)   # "Maj+Tab" in French
        self.a_max = act(v, _("Maximise"), lambda: self.toggle_panels("max"), "Ctrl+Space", True, enabled=has)
        act(v, _("Reset layout"), self.reset_layout, enabled=has, role="reset")
        v.addSeparator()
        act(v, _("Log"), lambda: self.b_log.toggle(), "Ctrl+L", role="log")
        self.a_spec = act(v, _("Spec mode"), None, None, True)
        self.a_spec.setChecked(spec)
        self.a_spec.toggled.connect(self.set_spec)
        r = mb.addMenu(_("Run"))
        self.a_run = act(r, _("Run all"), self.launch, "F5", enabled=has, role="run")
        self.a_here = act(r, _("Run to here"), self.run_to_here, "Shift+F5", enabled=has, role="run")   # Inspect's button's name
        r.addSeparator()
        self.a_stop = act(r, _("Stop"), self.stop_run, enabled=False, role="stop")
        h = mb.addMenu(_("Help"))
        act(h, _("About"), self.about, "F1", role="about")
        if spec:
            act(h, _("Glossary"), new=True)
        for m in (f, e, v, r, h):
            m.setToolTipsVisible(True)
        self._update_actions()
        self._sync_layout_marks()

    def _update_actions(self):
        """What the menus allow now: Undo / Redo, Stop, Export."""
        s = self.s
        if s is None:
            return
        try:
            self.a_undo.setEnabled(s.can_undo() or self._post_shown())
            self.a_redo.setEnabled(s.can_redo() or self._post_shown())
            self.a_stop.setEnabled(self.run is not None and self.run.running)
            self.a_stop.setToolTip(_("Stop the run: what is computed is kept") if self.a_stop.isEnabled()
                                   else _("No run in progress"))             # greyed: why (round 2)
            self.a_export.setEnabled(any(p.rendered for p in s.paths))
            from .services.run import run_here_refusal
            why = run_here_refusal(s, s.sel_step)            # a Session block is not computed: greyed, and why (#29)
            self.a_here.setEnabled(not why)
            self.a_here.setToolTip(_(why) if why else _("Run what the selected block needs, up to it"))
        except RuntimeError:                         # the menus are being rebuilt
            pass

    # ------------------------------------------------------------ small screens (issue #44)
    def resizeEvent(self, e):                                        # noqa: N802 - Qt's name
        super().resizeEvent(e)
        c = lay.size_class(self.width(), self.height(), self.size_range)
        if c != self.size_range:
            self.size_range = c
            self.apply_layout()

    def apply_layout(self):
        """What the window's size range and the user's choices say: the look, the panels, the bars."""
        cls = self.size_range or lay.LARGE
        comp = lay.compact(cls)
        if comp != self.compact:
            self.compact = comp
            self.setStyleSheet(lay.COMPACT_QSS if comp else "")
        st = lay.state(cls)
        full = bool(st["max"])
        right = "hidden" if st["hidden"] or st["right_hidden"] or full else st["right"]
        left = "hidden" if st["hidden"] or full else st["left"]
        if self.project is not None and self.proj_screen is not None:
            self.proj_bar.setFixedHeight(36 if comp else 46)
            self.proj_bar.setVisible(not full)
            self.statusBar().setVisible(not full)
            self.logp.setVisible(self.b_log.isChecked() and not full)
            if self.graph is not None:
                self.graph.set_layout(right, left)
            self._layout_inspect()
            self._layout_post()
        self._sync_layout_marks()

    def _layout_inspect(self):
        """Inspect's right panel, band of blocks and look, as ``apply_layout`` says (Inspect made after it: when it is
        first shown)."""
        cls = self.size_range or lay.LARGE
        st = lay.state(cls)
        full = bool(st["max"])
        if self._tune is not None:
            right = "hidden" if st["hidden"] or st["right_hidden"] or full else st["right"]
            self._tune.set_layout(right, not st["hidden"] and not full, lay.compact(cls))
        if self._browser is not None:
            self._browser.set_compact(lay.effective(cls) == lay.SMALL)

    def _sync_layout_marks(self):
        """The View menu's check marks: Hide panels, Hide right, Maximise."""
        st = lay.state(self.size_range or lay.LARGE)
        for a, on in ((getattr(self, "a_hide", None), st["hidden"]), (getattr(self, "a_hide_r", None), st["right_hidden"]),
                      (getattr(self, "a_max", None), st["max"])):
            if a is not None:
                a.setChecked(bool(on))

    def _layout_post(self):
        """Post's two sides: the user's choice for this size range, else the layers as the graph's palette (automatic
        on a small window) and the settings docked; Tab hides both, Shift+Tab the settings."""
        page = self._post_holder.page if self._post_holder is not None else None
        view = getattr(page, "view", None)
        if view is None:
            return
        cls = self.size_range or lay.LARGE
        st = lay.state(cls)
        gone = st["hidden"] or st["max"]
        left = "hidden" if gone else st.get("post_left") or ("auto" if st["left"] == "auto" else "docked")
        right = "hidden" if gone or st["right_hidden"] else st.get("post_right") or "docked"
        view.set_layout(left, right)
        view.set_compact(lay.compact(cls))

    def _tab(self):
        fw = QApplication.focusWidget()
        if self.project is None or isinstance(fw, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)) or \
                (isinstance(fw, QComboBox) and fw.isEditable()):
            self.focusNextPrevChild(True)
        else:
            self.toggle_panels("hidden")

    def _shift_tab(self):
        fw = QApplication.focusWidget()
        if self.project is None or isinstance(fw, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)) or \
                (isinstance(fw, QComboBox) and fw.isEditable()):
            self.focusNextPrevChild(False)
        else:
            self.toggle_panels("right_hidden")

    def toggle_panels(self, key: str):
        """Tab (hidden), Shift+Tab (right_hidden), Ctrl+Space (max): a toggle, remembered for this size range."""
        if self.project is None:
            return
        cls = self.size_range or lay.LARGE
        lay.remember(cls, **{key: not lay.state(cls)[key]})
        self.apply_layout()

    def _side_chosen(self, side: str, st: str):
        """A panel pinned or folded by the user: his choice from now on, for this size range."""
        lay.remember(self.size_range or lay.LARGE, **{side: st})
        self.apply_layout()

    def _post_panel_chosen(self, side: str, st: str):
        lay.remember(self.size_range or lay.LARGE, **{f"post_{side}": st})
        self.apply_layout()

    def _show_sides(self):
        """The arrow of a hidden side: the panels come back (Tab pressed again)."""
        lay.remember(self.size_range or lay.LARGE, hidden=False, right_hidden=False)
        self.apply_layout()

    def reset_layout(self):
        lay.reset()
        self.apply_layout()
        view = getattr(self.post, "view", None) if self.post is not None else None
        if view is not None:                             # the widths too, at once (the night test pass: kept until
            view._fit_split()                            # the next docking)

    # ------------------------------------------------------------ Home
    def show_home(self, last=None):
        QTimer.singleShot(0, self.offer_recovery)
        if self.project is not None:
            self._leave_thumbnail()                  # Home shows the open project as Post last showed it
        if self.home is None:
            self.home = HomeView(self.worker, last=last)
            self.home.openProject.connect(self.open_project)
            self.home.newProject.connect(self.new_project)
            self.home.openImages.connect(self.open_images)
            self.screens.insertWidget(0, self.home)
        else:
            self.home.rescan()
        self.screens.setCurrentWidget(self.home)
        self._flash = ""
        self._flash_timer.stop()
        self.setWindowTitle(f"{APP_NAME} · " + _("Home"))
        self.status.setText("")
        self.update_spec_note()

    def new_project(self):
        """The graph at once: the project has no name until its first Save or Run (``name_project``)."""
        self.create_project(drafts.new_root().as_posix(), _("Untitled"))

    def open_other(self):
        self.show_home()
        self.home.open_other()

    def create_project(self, root: str, name: str, target: str = "", details: dict | None = None) -> bool:
        """A new v2 project in ``root``, opened. Its calibrated frames go to <cache folder>/<project folder>.
        ``details``: the target's coordinates and the capture data New project gives (issue #38)."""
        try:
            d = details or {}
            p = project2.create(root, name, Target(name=target, ra=d.get("ra"), dec=d.get("dec"),
                                                   focal_mm=d.get("focal_mm"), pixel_um=d.get("pixel_um")))
            cache = prefs.get("cache_dir")
            if cache:
                p.scratch_dir = (Path(cache) / Path(root).name).as_posix()
            # issue #43: each session calibrated with its own frames (with one session, the same as before)
            p.stack_mode = "per_session"
            project2.save(p)
        except (OSError, ValueError, project2.LegacyProject) as e:
            self.banner.say(_("The project cannot be created: {why}").format(why=e))
            return False
        return self.open_project(root, created=True)

    # ------------------------------------------------------------ a project
    def _device(self) -> tuple[bool, str | None]:
        """(a GPU computes, the device asked) from Settings > Compute and the machine. Automatic is the GPU it picks
        (GUI exploration 6.8: its durations were the CPU's); None while the machine is not known yet."""
        dev = prefs.get("device") or "auto"
        if dev == "auto":
            dev = auto_device(self.machine) or "auto"
        if dev == "cpu" or (self.machine is not None and not self.machine.gpus):
            return False, "cpu"
        return True, (None if dev == "auto" else dev)

    def open_project(self, root, created: bool = False, recovered=None) -> bool:
        """``recovered``: the definition found after a crash, in place of the file."""
        root = Path(root)
        if self.project is not None and root.as_posix().lower() == self.project.root.as_posix().lower():
            self.show_project()
            return True
        try:
            d = recovered if recovered is not None else project2.load(root)
        except project2.LegacyProject:
            self.offer_import(root)
            return False
        except FileNotFoundError:
            self.banner.say(_("No project in this folder: {folder}").format(folder=root.as_posix()))
            return False                                  # still in the recents: its disk may only be unplugged
        except (OSError, ValueError, KeyError, TypeError):   # said plainly, never the raw error (7.5)
            self.banner.say(_("This project cannot be opened: its file is damaged, or it is not a project file of this "
                              "program."))
            return False
        if self.close_project(go_home=False) is False:              # unsaved edits kept: Cancel
            return False
        self.project = OpenProject(root, d)
        forget_state(root)                               # Home reads its state again once it is closed
        self.project.log("Project created" if created else "Project opened")
        if not drafts.is_untitled(root):
            prefs.add_recent(root.as_posix(), d.name, d.target.name or "")
        self._recovery_mine = recovered is not None
        try:
            self.make_model()
            self.mode = 0
            self._last_step = None
            self.build_menus()
            self.build_project()
        except Exception as e:  # noqa: BLE001 - the night test pass: a project that failed to build stayed half open,
            import logging      # invisible, the next gestures acting on it; closed, said, the detail in the output
            logging.getLogger(__name__).exception("the project could not be shown")
            self.close_project(go_home=True, discard=True)
            self.banner.say(_("This project cannot be shown: {why}").format(why=plain_error(f"{type(e).__name__}: {e}")))
            return False
        self.run.attach()                            # a run that goes on (started before, or elsewhere): follow it
        self.run.before_run = self._named_for_run
        if recovered is not None:
            self.s.edited()                              # it waits for Save, as before the crash
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
        s.dirtyChanged.connect(self._saved_or_not)
        s.editMade.connect(self._recovery_timer.start)
        self.run = RunController(s, window=self, console=lambda: getattr(self, "logp", None))
        self.run.runningChanged.connect(lambda _on: self.request_refresh())
        # Inspect's Run to here greys while a run is in progress (issue #33)
        self.run.runningChanged.connect(lambda _on: getattr(self._tune, "request_refresh", lambda: None)())  # if made
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
        holder = self._post_holder
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
        harness: "save", as before Save existed; nothing asked of a project without a name)."""
        if QApplication.platformName() == "offscreen":
            return "discard" if drafts.is_untitled(self.project.root) else "save"
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

    def close_project(self, go_home=True, discard=False) -> bool:
        """Close the project, asking first when it has edits not saved (a run in progress goes on: opening the
        project follows it again). False when the user cancels. ``discard``: closed without its edits, no question
        (Save As wrote them elsewhere)."""
        save = not discard
        if self.project is not None and not discard:
            self._flush_post()
            if self.s is not None and self.s.dirty:
                answer = self.ask_save()
                if answer == "cancel":
                    return False
                save = answer == "save"
                if save and drafts.is_untitled(self.project.root) and not self.name_project():
                    return False
                if save and self.s.changed_on_disk():  # another window's work is not replaced: closing waits
                    self.banner.say(_("Another window of the program saved this project since it was opened here: "
                                      "nothing is replaced. Save as… keeps this window's work in another folder."))
                    return False
        if self.project is not None:
            self._recovery_timer.stop()
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
                # taken out: the service holds its model, so the weak key never let either go (GUI exploration 4: a
                # project closed stayed in memory with its workers and image cache, ≈ 10 MB a reopening)
                sv = _services.pop(self.s, None)
                self.s.close(save)
                if sv is not None:
                    sv.stop()
            if self.render is not None:
                self.render.stop()
                self.render = None
            if self._recovery_mine:
                drafts.clear_recovery(self.project.root)
            drafts.discard(self.project.root)            # a project without a name: nothing left behind
            forget_state(self.project.root)              # nor a state Home was still reading as it opened
            self.project = None
            self.s = None
            if self.proj_screen is not None:
                self.screens.removeWidget(self.proj_screen)
                self.proj_screen.deleteLater()
                self.proj_screen = None
            self.report = self.graph = self.tune = self.frames = self.browser = self.steps = self.post_holder = None   # (steps held
            self.build_menus()                                                       # the closed model, GUI expl. 4)
            self._inspect_holder = None
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
        for i, name in enumerate((_("Graph"), _("Inspect"), _("Post"), _("Report"))):
            b = QPushButton(name)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.clicked.connect(lambda _x=False, k=i: self.set_mode(k))
            g.addButton(b)
            h.addWidget(b)
            self.mode_btns.append(b)
        h.addStretch(1)
        self.device_cb = QComboBox()                  # where the run computes: a GPU of the machine, or the CPU
        self.device_cb.setToolTip(_("Where the run computes"))
        self.device_cb.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.device_cb.activated.connect(lambda _i: self.set_device(self.device_cb.currentData()))
        h.addWidget(self.device_cb)
        h.addSpacing(6)
        self.bt_run = set_icon(QPushButton(_("Run")), "run", 16, "on_accent")
        self.bt_run.setObjectName("primary")
        self.bt_run.clicked.connect(self.launch)
        h.addWidget(self.bt_run)
        bar.setFixedHeight(46)
        self.proj_bar = bar
        sv.addWidget(bar)
        self.modes = QStackedWidget()
        sv.addWidget(self.modes, 1)
        self.logp = LogPanel(self.project)
        self.logp.setVisible(self.b_log.isChecked())
        sv.addWidget(self.logp)
        # the modes: Graph, and Inspect (which holds Frames and Post), made the first time it is shown or reached
        self.modes.addWidget(self.graph_page())
        self.tune = self.frames = self.browser = self.post_holder = None     # a rebuilt screen's: made again
        self._inspect_holder = PageHolder(self._make_inspect)
        self.modes.addWidget(self._inspect_holder)
        self.modes.addWidget(self.report_page())
        self.proj_screen = screen
        self.screens.addWidget(screen)
        self.screens.setCurrentWidget(screen)
        if self.run is not None:
            self.run.retranslate()                   # the console of the last run, in the new Log panel
        self.set_mode(self.mode)
        self.apply_layout()

    def report_page(self) -> QWidget:
        """Report: the measures of a workflow, read-only (issue #81)."""
        from .report import ReportPage
        self.report = ReportPage(self.s, self.steps)
        self.report.said.connect(self.flash)
        return self.report

    def graph_page(self) -> QWidget:
        """Graph: the toolbar of optional blocks, the GraphView and the block Panel."""
        from .graph import GraphPage
        from .steps import Steps
        self.steps = Steps(self.s)
        page = GraphPage(self.s, self.steps)
        page.openBlock.connect(lambda: self.set_mode(2 if self.s.sel_step == "final" else 1))
        page.openPost.connect(lambda: self.set_mode(2))
        page.openLayer.connect(self.open_layer)
        page.addSession.connect(self.add_session)
        page.variantMade.connect(self._variant_made)
        page.changed.connect(self.request_refresh)
        page.sideChosen.connect(self._side_chosen)
        page.sidesShown.connect(self._show_sides)
        self.graph = page
        if self.run is not None:
            self.run.watch_graph(page.graph)
        return page

    def _make_inspect(self) -> QWidget:
        page = self.inspect_page()
        self._layout_inspect()                       # the window's layout, applied before Inspect was made
        return page

    def _inspect(self):
        """Inspect made now if it was not yet (what reaches for its views: a test, Add session, Post)."""
        if self._inspect_holder is not None and self.project is not None:
            self._inspect_holder.ensure()

    # Inspect's views, made with it (``inspect_page``): None while no project is open
    @property
    def tune(self):
        self._inspect()
        return self._tune

    @tune.setter
    def tune(self, v):
        self._tune = v

    @property
    def frames(self):
        self._inspect()
        return self._frames

    @frames.setter
    def frames(self, v):
        self._frames = v

    @property
    def browser(self):
        self._inspect()
        return self._browser

    @browser.setter
    def browser(self, v):
        self._browser = v

    @property
    def post_holder(self):
        self._inspect()
        return self._post_holder

    @post_holder.setter
    def post_holder(self, v):
        self._post_holder = v

    def inspect_page(self) -> QWidget:
        """Inspect: TuneView, with Frames (FramesView and its browser) and the Post page."""
        from .frames import FramesView
        from .sources import SourcesView
        from .tune import TuneView
        s = self.s
        self.browser = SourcesView(s)
        self.browser.toast.connect(self.flash)
        self.frames = FramesView(s, self.browser)
        self.post_holder = PageHolder(self.post_page)
        tune = TuneView(s, self.steps, self.frames, self.post_holder)
        tune.back.connect(lambda: self.set_mode(0))
        tune.pick.connect(self._strip_pick)
        tune.workflow.connect(lambda _i: self.request_refresh())
        tune.runHere.connect(self.run_to_here)
        tune.openPost.connect(lambda: self.set_mode(2))
        tune.removeTool.connect(self._remove_tool)
        tune.toasted.connect(self.flash)
        tune.sideChosen.connect(lambda st: self._side_chosen("right", st))
        tune.sidesShown.connect(self._show_sides)
        tune.can_post = self.browser.can_post = lambda: True    # Post takes Inspect's images as layers
        # the Project block chosen in the graph (a click, its double-click, Enter): its form (issue #89)
        tune.project_open = lambda: self.graph is not None and self.graph.project_open()
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
        page.openImages.connect(self.open_images)
        page.view.openImages.connect(self.open_images)
        page.view.runAsked.connect(self.launch)
        page.view.alignAsked.connect(self.align_layers)
        page.view.goResults.connect(self._go_results)
        page.view.card_source = self.report              # the night card, for its Export
        page.view.toasted.connect(self.flash)            # what a change of Post did beyond Post (a branch removed)
        page.view.goProcessing.connect(lambda pid, k: (self.select(index=self.s.index_of(pid), step=k),
                                                       self.set_mode(1)))
        page.view.panelChosen.connect(self._post_panel_chosen)
        page.view.dock.arrow.clicked.connect(self._show_sides)
        page.view.ldock.arrow.clicked.connect(self._show_sides)
        QTimer.singleShot(0, self._layout_post)
        return page

    @property
    def post(self):
        """The Post page, made now if it was not yet."""
        return self.post_holder.ensure() if self.post_holder is not None else None

    def _flush_post(self):
        """Post's last edit into the project's definition (it is saved just after an edit, when the event loop comes
        back: a close or a rebuild may come first)."""
        page = self._post_holder.page if self._post_holder is not None else None
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
        return self.s is not None and self.mode == 2 and self._post_holder is not None and \
            self._post_holder.page is not None

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
        self.modes.setCurrentIndex(0 if i == 0 else 2 if i == 3 else 1)
        self.mode_btns[i].setChecked(True)
        if i == 3:
            self.report.refresh()
        if i == 2 and not was_post and self._post_holder is not None and self._post_holder.page is not None:
            self._post_holder.page.refresh()
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
        self.update_device_combo()
        btn = self.mode if self.mode in (0, 3) else (2 if s is not None and s.sel_step == "final" else 1)
        if self.mode != btn:
            self.mode = btn
        self.mode_btns[btn].setChecked(True)
        self.update_run_button()
        if self.logp.isVisible():
            self.logp.refresh()
        self.update_spec_note()
        self._update_actions()

    def update_device_combo(self):
        """The combo left of Run: the machine's GPUs and the CPU, the one the run computes on chosen (Automatic: the
        GPU it picks); orange on the CPU; not while a run goes (it keeps the device it started on)."""
        cb = self.device_cb
        dev = prefs.get("device") or "auto"
        m = self.machine
        items = device_items(m, dev)                   # the same list as Settings > Compute (GUI exploration 6.9)
        cb.blockSignals(True)
        if [(cb.itemText(i), cb.itemData(i)) for i in range(cb.count())] != [(t, d) for t, d, _tip in items]:
            cb.clear()
            for t, d, tip in items:
                cb.addItem(t, d)
                cb.setItemData(cb.count() - 1, tip, Qt.ToolTipRole)
        _text, gpu = machine_text(m, dev)
        cb.setCurrentIndex(max(cb.findData(dev), 0))
        cb.blockSignals(False)
        cb.setStyleSheet("" if gpu else f"QComboBox {{ color: {WARN}; }}")

    def update_run_button(self):
        """The Run button (Run · the time of Run all) and the status line (workflows, what is left to run)."""
        s = self.s
        frames = bool(s is not None and s.has_data())          # lights, or an Image's file
        running = self.run is not None and self.run.running
        n = len(s.paths) if s is not None else 0
        cost = s.cost() if s is not None and s.ready and s.paths else None
        todo = cost is not None and cost.jobs > 0
        empty = s is not None and bool(s.paths) and all(no_block(w) for w in s.p.workflows)
        self.device_cb.setEnabled(not running)
        self.device_cb.setToolTip(_("A run computes on the device it started on") if running else
                                  _("Where the run computes"))
        if running:
            self.bt_run.setText(_("Running · {pct} %").format(pct=int(round(100 * self.run.progress()))))
            self.bt_run.setEnabled(True)
            self.bt_run.setToolTip(_("Show the run"))
        else:
            why = self.run_refusal(cost)
            # no duration on a refused Run (round 2: "≈ 3 h 03" with the lights' folder gone, "≈ 10 min" on a fusion
            # that cannot run)
            self.bt_run.setText(_("Run") + (f"  ·  {duration_text(cost.seconds, cost.measured)}"
                                            if todo and frames and not empty and not why else ""))
            self.bt_run.setEnabled(todo and not why)
            self.bt_run.setToolTip(why)                       # a greyed Run says why (6.7, #15)
        self.a_run.setEnabled(frames and not empty and (todo or running))
        self.a_run.setToolTip(_("Run every workflow, up to the review pauses") if self.a_run.isEnabled() else
                              self.run_refusal(cost) or _("All up to date"))       # greyed: why (round 2)
        if not n:
            line = _("No workflow yet") if s is not None and s.ready else ""
        else:
            line = plural(n, "{n} workflow", "{n} workflows")
            if s.error:
                line += "  ·  " + _("cannot be read")
            elif not s.ready:
                line += "  ·  " + _("reading the project…")
            elif not frames:                              # no estimate without frames (issue #47)
                line += "  ·  " + (_("Image: choose its file") if images_only(s) else
                                   _("nothing to compute: add lights or image"))
            elif empty:                                   # never "all up to date" with nothing placed (6.5, #13)
                # its Image or Session is a block on screen: what is missing is a block after it (round 3)
                line += "  ·  " + (_("nothing after Session: place a block") if not any(
                                       q.source is not None for q in s.paths) else
                                   _("nothing after the sources: place a block"))
            elif running:                                 # the run says its own time left (issue #67)
                pass
            elif todo and self.run_refusal(cost):         # refused: said with its reason, never a time to run
                line += "  ·  " + self.run_refusal(cost)  # (round 2; report shots: the reason only in a tooltip)
            elif todo:
                line += "  ·  " + _("to run: {t}").format(t=duration_text(cost.seconds, cost.measured))
            elif cost is not None:
                line += "  ·  " + _("all up to date")
        self._line = line
        self.status.setText(self._flash or line)

    def run_refusal(self, cost=None) -> str:
        """Why Run cannot run now, translated ('' when it can): one place for the Run button's tooltip (GUI exploration
        6.5, 6.7, #13, #15). The fusions that cannot run (the model's ``run_problems``, when it has it) are said
        too."""
        s = self.s
        if s is None or not s.ready:
            return ""
        if not s.paths:
            return _("No workflow: place a block")
        if not s.has_data():
            return _("Image: choose its file") if images_only(s) else _("Nothing to compute: add lights or image")
        if all(no_block(w) for w in s.p.workflows):
            return _("No block: place one")
        problems = getattr(s, "run_problems", lambda: [])()
        if problems:
            from .runs import refusal_text
            return refusal_text(problems)
        if cost is not None and not cost.jobs:
            return _("All up to date")
        return ""

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
            self.test_card()
        elif channel == "cache.usage":
            self._confirm_clear(result)
        elif channel == "cache.clear":
            freed, n = result
            self.flash(plural(n, "{size} freed in {n} project", "{size} freed in {n} projects",
                              size=_gb_text(freed)))

    def _worker_failed(self, channel, _n, text):
        if channel.startswith("cache."):
            self.banner.say(_("The cache cannot be cleared: {why}").format(why=plain_error(text)))

    # ------------------------------------------------------------ files
    def save(self):
        """Save: the edits written into the project (nothing is written before, issue #1), a field still being typed
        in included (its focus taken away: it applies what it holds, as Enter would)."""
        self._commit_typing()
        self._flush_post()
        if self.project is not None and drafts.is_untitled(self.project.root):
            self.name_project()
            return
        if self.s is not None and not self.s.flush():
            self.banner.say(_("Another window of the program saved this project since it was opened here: nothing is "
                              "replaced. Save as… keeps this window's work in another folder."))
            return
        self.flash(_("Project saved"))

    def _suggested_name(self) -> str:
        """The name a project without one is proposed: the name typed in its form (issue #89), else its target, else the
        folder of its images."""
        s = self.s
        if s.name and s.name not in ("Untitled", _("Untitled")):
            return s.name
        if s.target_state().name:
            return s.target_state().name
        for ses in s.p.sessions:
            for e in ses.lights:
                f = Path(s.p.abspath(e))
                return f.parent.name if f.suffix else f.name
        for w in s.p.workflows:
            if w.source and w.source.get("path"):
                return Path(w.source["path"]).stem
        return ""

    def name_project(self) -> bool:
        """A project without a name is given its name and folder, once: at its first Save or Run (its results will be
        written there, and Save as does not move them). False when cancelled."""
        got = ask_project_name(self, self._suggested_name())
        if got is None:
            return False
        root, name = got
        p, old = self.project.definition, self.project.root
        if self.s.path is not None:
            p.current = self.s.path.pid
        for ses in p.sessions:
            for kind in project2.KINDS:
                setattr(ses, kind, [p.abspath(x) for x in getattr(ses, kind)])
        p.name = name
        cache = prefs.get("cache_dir")
        p.scratch_dir = (Path(cache) / root.name).as_posix() if cache else ""
        try:
            project2.save(p, root)
        except (OSError, ValueError, project2.LegacyProject) as e:
            self.banner.say(_("The project cannot be saved: {why}").format(why=e))
            return False
        self.project.root = root
        self.s.relocated(root)
        self.s.flush()
        drafts.clear_recovery(old)
        drafts.discard(old)
        prefs.add_recent(root.as_posix(), name, p.target.name or "")
        self.title.setText(self.project_name())
        self._title()
        self.flash(_("Project saved"))
        return True

    def open_images(self):
        """Images as layers of Post, one Image workflow each (the graph follows); from nothing, the project is made
        first. Computed at once: only then does Post have them."""
        from .gallery import ask_files
        files = ask_files(self, {"images": 1})
        if not files:
            return
        if self.project is None:
            self.new_project()
        if self.project is not None and self.graph.place_images(files) is not None:
            self.set_mode(2)
            self._launch_when_ready()

    def align_layers(self):
        """Post's Aligner: the Align images block in the graph, its Inspect, where Run to here computes it."""
        if self.graph is not None and self.graph.align_images():
            self.set_mode(1)

    def _launch_when_ready(self):
        if self.s is None:
            return
        if self.s.ready:
            self.launch()
        else:
            QTimer.singleShot(50, self._launch_when_ready)

    def _named_for_run(self) -> bool:
        return self.project is None or not drafts.is_untitled(self.project.root) or self.name_project()

    # ------------------------------------------------------------ the recovery after a crash
    def _write_recovery(self):
        if self.s is not None and self.s.dirty:
            drafts.write_recovery(self.project.root, self.project.name, self.project.definition.to_dict())
            self._recovery_mine = True

    def _saved_or_not(self, dirty: bool):
        # saved: the copy of this project goes, this session's or one a crash left (the night test pass: the project
        # opened from Home, saved, and the older copy was still offered: Reopen would have taken the work back)
        if not dirty and self.project is not None:
            self._recovery_timer.stop()
            drafts.clear_recovery(self.project.root)
            self._recovery_mine = False

    def offer_recovery(self):
        """What a crash left: said on Home, kept until the user chooses."""
        found = drafts.recoveries() if self.project is None else []
        if not found:
            return
        rec = found[0]
        self.banner.ask(_("Unsaved work found: “{name}” ({when})").format(name=rec.get("name") or "", when=rec.get("saved")),
                        (_("Reopen"), lambda: self.reopen_recovered(rec)), (_("Ignore"), lambda: self.ignore_recovered(rec)))

    def reopen_recovered(self, rec: dict):
        root = Path(rec["root"])
        if not drafts.is_untitled(root) and not (root / project2.FILE).exists():
            self.banner.say(_("Its folder is not found: {folder}").format(folder=root.as_posix()))
            return
        try:
            d = project2.Project2.from_dict(rec["definition"], root)
            root.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError, KeyError, TypeError):
            drafts.clear_recovery(root)
            self.banner.say(_("This work cannot be reopened: its copy is damaged."))
            return
        self.open_project(root, recovered=d)

    def ignore_recovered(self, rec: dict):
        drafts.clear_recovery(rec["root"])
        drafts.discard(rec["root"])
        self.offer_recovery()

    @staticmethod
    def _commit_typing():
        w = QApplication.focusWidget()
        if isinstance(w, (QLineEdit, QAbstractSpinBox)):
            w.clearFocus()                               # editingFinished: what is typed applies before the save

    def save_as(self):
        """The definition of the project, its edits included, in another folder (its results stay where they are),
        then opened in place of the project, which keeps what it had on disk (as Save As does in reference apps:
        cancelled, nothing is written; GUI exploration 4)."""
        if self.project is None:
            return
        self._commit_typing()
        self._flush_post()                               # Post's last edit into the definition (not written)
        d = QFileDialog.getExistingDirectory(self, _("Save as…"), prefs.last_folder())
        prefs.remember_folder(d)
        if not d:
            return
        root = Path(d)
        if (root / project2.FILE).exists() or (root / project2.V1_FILE).exists():
            self.banner.say(_("This folder already holds a project."))
            return
        old = self.project.definition
        if self.s is not None and self.s.path is not None:
            old.current = self.s.path.pid                # the copy opens on the workflow shown
        p = project2.Project2.from_dict(old.to_dict())
        p.name = root.name                               # two projects of one name on Home otherwise (GUI exploration 4, Q8)
        for s in p.sessions:
            for kind in project2.KINDS:
                setattr(s, kind, [old.abspath(x) for x in getattr(s, kind)])
        p.reference_image = old.reference_path()
        cache = prefs.get("cache_dir")
        p.scratch_dir = (Path(cache) / root.name).as_posix() if cache else ""
        try:
            project2.save(p, root)
        except (OSError, ValueError, project2.LegacyProject) as e:
            self.banner.say(_("The project cannot be saved: {why}").format(why=e))
            return
        self.close_project(go_home=False, discard=True)  # its edits went into the copy: no question
        self.open_project(root)

    def offer_import(self, root: Path):
        """A project made with an older version: it opens once imported as a new project, in a folder the user
        chooses; its own folder is never written."""
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

    # ------------------------------------------------------------ frames, runs, export, undo
    def add_session(self):
        """Add a session…: a new empty session, opened in Inspect, where its files are added (issue #43)."""
        s = self.s
        if s is None:
            return
        # "+ Session": it joins the flow it is drawn with (else the one shown), or one of its own (decision 1)
        sid = s.new_session(getattr(getattr(self.graph, "graph", None), "plus_session_pid", None))
        self.flash(_("Session “{name}” added · its files in Inspect").format(name=s.session(sid).name))
        self.select(step=f"session@{sid}" if s.separate_sessions() else "sources")
        if self.browser is not None:
            self.browser.show_session(sid)
            self.frames.show_page(1)
        self.set_mode(1)

    def export(self):
        """File > Export image… (the Export dialog of Post, on the composition of its layers)."""
        if self.s is None:
            return
        if not any(p.rendered for p in self.s.paths):
            self.flash(_("Nothing to export: no workflow computed"))
            return
        self.post.export(self)

    def launch(self):
        """Run all (F5, the Run button): the run starts at once, its window following it (issue #51); without
        frames, the window says why nothing runs."""
        if self.s is not None and not self.s.has_data():
            self.flash(_("Nothing to compute: add lights or image"))
            return
        if self.s is not None and self.s.paths and all(no_block(w) for w in self.s.p.workflows):
            self.flash(_("No block: place one"))     # an empty workflow is not run
            return
        if self.run is not None:
            self.run.launch(start=True)

    def run_to_here(self):
        """Run to the selected block (Shift+F5): as F5 without data, nothing opens and why is said (round 2: a window
        announced a disk need with no frames)."""
        s = self.s
        busy = self.run is not None and self.run.running
        if s is not None and not busy and s.path is not None and not s.has_data(s.path):
            self.flash(_("Nothing to compute: add lights or image"))
            return
        if self.run is not None:
            self.run.run_to_here()

    def stop_run(self):
        """Stop the run: the job in progress keeps its checkpoints; the next run resumes."""
        if self.run is not None:
            self.run.stop()

    def undo(self):
        """Undo: in Post, the composition's last change; elsewhere the workflows' last change."""
        if self._post_shown():
            if self.post_holder.page.view.do_undo():
                self._say_restored("Undone", "Undone: {what}")
        elif self.s is not None and self.s.undo():
            self._say_restored("Undone", "Undone: {what}")

    def redo(self):
        if self._post_shown():
            if self.post_holder.page.view.do_redo():
                self._say_restored("Redone", "Redone: {what}")
        elif self.s is not None and self.s.redo():
            self._say_restored("Redone", "Redone: {what}")

    def _say_restored(self, plain: str, said: str):
        """The Undo / Redo message and Log line: what was undone or redone, in the words of its step."""
        what = getattr(self.s, "restored", "")
        if what:
            self.s.log(said, what=what)
            self.flash(_(said).format(what=what))
        else:
            self.s.log(plain)
            self.flash(_(plain))

    def clear_cache(self):
        """Clear the unused results of the closed projects Home knows (the recent ones and those of its sections),
        after a confirmation (read in the worker: it walks the stores)."""
        from .home import section_rows
        open_root = self.project.root if self.project is not None else None
        self.worker.submit("cache.usage", _closed_usage, section_rows(), [r["path"] for r in prefs.recent()],
                           open_root)

    def _confirm_clear(self, found):
        total = sum(b for _r, b in found)
        if not total:
            self.flash(_("The closed projects hold no unused results."))
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
        d.homeChanged.connect(lambda: self.home.rescan() if self.home is not None else None)
        d.clearCache.connect(self.clear_cache)
        d.densityChanged.connect(self.apply_layout)
        d.exec()
        d.deleteLater()
        self.sky_changed()                               # a catalogue installed there: planned again at once

    def sky_changed(self, *_a):
        """The data of the sky on this computer may have changed (an extract installed, a survey fetched): the blocks
        that could not run for lack of it, and Home's line, are said again (core/sky_data.py)."""
        if self.s is not None and hasattr(self.s, "sky_changed"):
            self.s.sky_changed()
        if self.home is not None and hasattr(self.home, "sky_changed"):
            self.home.sky_changed()

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

    def test_card(self):
        """The card test once in the background (first launch, a changed card or driver): never in the tests."""
        if "PYTEST_CURRENT_TEST" in os.environ:
            return
        from .services.selftest import job
        pref = prefs.get("device") or "auto"
        job().start(pref, auto_device(self.machine) if pref == "auto" else pref)

    def _card_tested(self):
        """A new profile from the card test: Post follows it at its next render (a run reads it when it starts)."""
        if self.render is not None:
            from .services.render import default_device_pref
            self.render.pref = default_device_pref() if self._device()[0] else "cpu"
            self.render._busy_at, self.render._last_busy = 0.0, "?"

    def set_device(self, device: str):
        prefs.set("device", device)
        self.test_card()
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
        if self._tune is not None:
            self._tune.request_refresh()
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
            self.status.setText("")

    def about(self):
        d = self.about_dialog()
        d.exec()
        d.deleteLater()

    def about_dialog(self) -> QDialog:
        d = QDialog(self)
        cap_dialog(d)
        d.setWindowTitle(_("About"))
        lv = QVBoxLayout(d)
        t = QLabel(HELP.format(
            title=f"{APP_NAME} {__version__}",
            motto=APP_MOTTO,
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
        src.setStyleSheet('QTextBrowser { background: #12141a; border: 1px solid #353b47; }')
        src.document().setDefaultStyleSheet('a { color: #5ea1ff; } th, td { padding-right: 8px; }')
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


def _closed_usage(rows: list[dict], recent: list[str], open_root) -> list[tuple[str, int]]:
    """(project folder, unused bytes) of the projects Home knows (the recent ones, those of the sections: issue #31,
    no workspace), the open one aside (worker thread)."""
    from .home import known_projects
    out = []
    for root in known_projects(rows, recent):
        info = api.project_summary(root)
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
    """What the first opening and the first render would otherwise load while the user waits, loaded while Home is
    shown (#115): the stage graph (every stage's module: the keys of any project need it, seconds on a laptop), the
    names of the catalogue (the Project block's target, read in the GUI thread), the FITS reader (the estimates read
    one frame's header), then core/determinism.warm_up."""
    try:
        from ..core.store import key_rules
        key_rules()
        from ..core.skyobjects import names_index
        names_index()
        from astropy.io import fits  # noqa: F401 - loaded here, read by the worker
    except Exception:  # noqa: BLE001 - a warm-up only: the opening loads them anyway
        pass
    try:
        from ..core.determinism import warm_up
        warm_up()
    except Exception:  # noqa: BLE001 - a warm-up only: the render loads it anyway
        pass


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    f = prefs.settings_file()
    first = f is not None and not f.exists()
    prefs.load()
    if first:                                           # a first launch speaks the system's language
        prefs.set("language", prefs.system_language())
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app._gc = GuiGarbageCollector(app)                  # cycles collected on the GUI thread only (gcguard.py)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(app_icon())
    if sys.platform == "win32":                         # the taskbar shows this program's icon, not Python's
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_NAME)
    app.setStyleSheet(QSS + EXTRA_QSS)
    apply_language(prefs.get("language"))

    def show_error(t, v, tb):
        box = QMessageBox(QMessageBox.Critical, APP_NAME, _("The program hit an unexpected error. The details below "
                                                            "help to find it."))
        box.setDetailedText("".join(traceback.format_exception(t, v, tb))[-3000:])
        box.exec()

    sys.excepthook = show_error
    threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
    w = Main()
    size_within(w, 1680, 1000)                          # a laptop: no taller than its screen (issue #44)
    w.showMaximized()
    if argv:
        w.open_project(argv[0])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
