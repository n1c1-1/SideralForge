"""Mockup alt3 of the heavy client.

Home: the workspace and its projects. A project has three modes over the same blocks:
* Graph: the workflows as a graph of blocks, the tools on top, the selected block's settings on the right;
* Inspect: the image of one block, large, the workflow as a strip, the same settings with the measurements;
* Post: the workflows' layers, masks and filter layers.
Settings hold the application preferences; the language takes effect at once. Spec mode shows what the engine still
has to build. A Demo menu switches projects and machines.

    C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe mockup/alt3/app/app3.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import project  # noqa: E402,F401  (puts alt2's modules on the path, after alt3's)

from PySide6.QtCore import QLocale, QMimeData, QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QAction, QActionGroup, QDrag, QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDialog, QFrame, QHBoxLayout, QLabel,  # noqa: E402
                               QMainWindow, QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

import prefs  # noqa: E402
from blocks import COLUMNS, TOOL_COLUMN, Graph, block_title  # noqa: E402
from dialogs import NewWorkflowDialog, ask_name  # noqa: E402
from exportdlg import ExportDialog  # noqa: E402
from frames import FramesView  # noqa: E402
from graph import GraphView  # noqa: E402
from home import HomeView  # noqa: E402
from i18n import _, set_language  # noqa: E402
from logview import LogPanel  # noqa: E402
from model import DATA, TOOLS, PathDef, hm  # noqa: E402
from panel import Panel  # noqa: E402
from post3 import PostPage3  # noqa: E402
from project import MACHINES, SCENARIOS3, Project  # noqa: E402
from runs import run_blocks  # noqa: E402
from settingsdlg import SettingsDialog  # noqa: E402
from sources import ImportDialog, SourcesView  # noqa: E402
from steps import Steps  # noqa: E402
from tune import TuneView  # noqa: E402
from widgets import ACCENT, QSS, WARN, clear, label  # noqa: E402

EXTRA_QSS = ("QSlider:horizontal { margin-left: 6px; margin-right: 6px; }"
             "QMenuBar { background: #1c1f26; } QMenuBar::item:selected { background: #2b3446; }")
TOOLBOX = ["debruitage", "deconv", "etoiles"]           # the optional blocks; the others are always there
HELP = """<h2>{title}</h2>
<p>{p1}</p><p>{p2}</p><p>{p3}</p>"""


class ToolCard(QFrame):
    """A tool: dragged onto a workflow of the graph, or added to the current workflow by a double-click."""

    def __init__(self, win, key, state, text, parent=None):
        super().__init__(parent)
        self.win, self.key, self.state = win, key, state
        self._press = None
        on = state == "in"
        self.setObjectName("tool")
        self.setCursor(Qt.ArrowCursor if on else Qt.OpenHandCursor)
        self.setStyleSheet(f"QFrame#tool {{ background: {'transparent' if on else '#232831'}; border: 1px "
                           f"{'dashed #3a4150' if on else 'solid #353b47'}; border-radius: 6px; }}"
                           f"QFrame#tool:hover {{ border-color: {'#3a4150' if on else ACCENT}; }}"
                           " QLabel { background: transparent; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 8, 6)
        v.setSpacing(1)
        t = QLabel(("⋮⋮  " if not on else "") + block_title(None, TOOL_COLUMN[key]))
        t.setStyleSheet("font-weight: 600;" + (" color: #6b7280;" if on else ""))
        v.addWidget(t)
        v.addWidget(label(text, "small"))
        self.setToolTip(_(TOOLS[key]["what"]))
        self.setMinimumWidth(140)

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._press is None or not (e.buttons() & Qt.LeftButton):
            return
        if (e.position().toPoint() - self._press).manhattanLength() < QApplication.startDragDistance():
            return
        self._press = None
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText("tool:" + self.key)
        drag.setMimeData(mime)
        drag.setPixmap(self.grab())
        drag.setHotSpot(QPoint(20, 14))
        drag.exec(Qt.CopyAction)

    def mouseDoubleClickEvent(self, e):
        if self.state != "in" and self.win.s.path is not None:
            self.win.add_tool(self.key, self.win.s.path.pid)


class Main(QMainWindow):
    def __init__(self, scenario=None):
        super().__init__()
        self.s = None
        self._pending = False
        self._gpu = prefs.get("device") != "cpu"
        self.undo_stack, self.redo_stack = [], []
        self._restoring = False
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
        if scenario:
            self.set_scenario(scenario)

    # ------------------------------------------------------------ menus
    def build_menus(self):
        mb = self.menuBar()
        mb.clear()

        def act(menu, text, fn, key=None, checkable=False):
            a = QAction(text, self)
            if key:
                a.setShortcut(QKeySequence(key))
            a.setCheckable(checkable)
            a.triggered.connect(fn)
            menu.addAction(a)
            return a

        f = mb.addMenu(_("File"))
        act(f, _("Home"), self.show_home, "Ctrl+H")
        act(f, _("New project…"), lambda: self.home.new_project(), "Ctrl+N")
        act(f, _("Open project…"), lambda: self.home.open_other(), "Ctrl+O")
        rec = f.addMenu(_("Recent projects"))
        for pr in project.PROJECTS_DEMO:
            act(rec, f"{pr['name']} · {pr['target']}", lambda _x=False, p=pr: self.home.open_card(p))
        f.addSeparator()
        act(f, _("Save"), lambda: self.flash(_("Project saved")), "Ctrl+S")
        act(f, _("Save as…"), lambda: self.flash(_("Project saved")))
        f.addSeparator()
        act(f, _("Import folder…"), self.add_sources, "Ctrl+I")
        act(f, _("Export image…"), self.export, "Ctrl+E")
        act(f, _("Export project as script…"), lambda: self.flash(_("Script written: the project as a "
                                                                   "reproducible recipe")))
        f.addSeparator()
        act(f, _("Close project"), self.show_home, "Ctrl+W")
        act(f, _("Quit"), self.close, "Ctrl+Q")
        e = mb.addMenu(_("Edit"))
        self.a_undo = act(e, _("Undo"), self.undo, "Ctrl+Z")
        self.a_redo = act(e, _("Redo"), self.redo, "Ctrl+Y")
        e.addSeparator()
        act(e, _("Settings…"), self.open_settings, "Ctrl+,")
        v = mb.addMenu(_("View"))
        act(v, _("Graph"), lambda: self.set_mode(0), "Ctrl+1")
        act(v, _("Inspect"), lambda: self.set_mode(1), "Ctrl+2")
        act(v, _("Post"), lambda: self.set_mode(2), "Ctrl+3")
        v.addSeparator()
        act(v, _("Log"), lambda: self.b_log.toggle(), "Ctrl+L")
        self.a_spec = act(v, _("Spec mode"), self.toggle_spec, None, True)
        self.a_spec.setChecked(prefs.spec())
        r = mb.addMenu(_("Run"))
        act(r, _("Run all"), self.launch, "F5")
        act(r, _("Run to selected block"), self.run_to_here, "Shift+F5")
        d = mb.addMenu(_("Demo"))
        ag = QActionGroup(self)
        for key, name in SCENARIOS3.items():
            a = act(d, _(name), lambda _x=False, k=key: self.set_scenario(k), checkable=True)
            ag.addAction(a)
        d.addSeparator()
        act(d, _("Add a second night"), self.add_session)
        d.addSeparator()
        mg = QActionGroup(self)
        for gpu, name in MACHINES.items():
            a = act(d, _("Machine: {name}").format(name=_(name)), lambda _x=False, g=gpu: self.set_machine(g),
                    checkable=True)
            a.setChecked(gpu == self._gpu)
            mg.addAction(a)
        h = mb.addMenu(_("Help"))
        act(h, _("About this mockup"), self.help, "F1")
        act(h, _("Glossary"), self.glossary)

    # ------------------------------------------------------------ screens
    def show_home(self):
        if getattr(self, "home", None) is None:
            self.home = HomeView()
            self.home.openProject.connect(self.set_scenario)
            self.home.newProject.connect(lambda name: self.set_scenario("neuf", name))
            self.screens.insertWidget(0, self.home)
        self.screens.setCurrentWidget(self.home)
        self.statusBar().clearMessage()
        self.setWindowTitle("astro_gpu · " + _("Home"))
        self.status.setText(prefs.get("workspace"))
        self.update_spec_note()

    def set_scenario(self, key, name=None):
        self.s = Project(key)
        self.s.gpu = self._gpu
        if name and key == "neuf":
            self.s.display_name = name
        self.undo_stack, self.redo_stack = [], []
        self._last_snap = None
        self.build_project()

    def build_project(self):
        """The project screen, built again for a new project or a change of language."""
        old = getattr(self, "proj_screen", None)
        if old is not None:
            self.screens.removeWidget(old)
            old.deleteLater()
        s = self.s
        self.steps = Steps(s)
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
        g = QButtonGroup(self)
        for i, name in enumerate((_("Graph"), _("Inspect"), _("Post"))):
            b = QPushButton(name)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.clicked.connect(lambda _x=False, k=i: self.set_mode(k))
            g.addButton(b)
            h.addWidget(b)
            self.mode_btns.append(b)
        h.addStretch(1)
        self.machine = label("", "small")
        h.addWidget(self.machine)
        h.addSpacing(12)
        self.bt_run = QPushButton(_("Run"))
        self.bt_run.setObjectName("primary")
        self.bt_run.clicked.connect(self.launch)
        h.addWidget(self.bt_run)
        bar.setFixedHeight(46)
        sv.addWidget(bar)
        self.modes = QStackedWidget()
        sv.addWidget(self.modes, 1)
        self.logp = LogPanel(s)
        self.logp.setVisible(self.b_log.isChecked())
        sv.addWidget(self.logp)
        # Graph
        page = QWidget()
        ph = QHBoxLayout(page)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.setSpacing(0)
        centre = QWidget()
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        tools = QFrame()
        tools.setObjectName("tools")
        tools.setStyleSheet("QFrame#tools { background: #1c1f26; border-bottom: 1px solid #2a2f38; }")
        self.tl = QHBoxLayout(tools)
        self.tl.setContentsMargins(12, 8, 12, 8)
        self.tl.setSpacing(8)
        cv.addWidget(tools)
        self.graph = GraphView(s, self.steps)
        cv.addWidget(self.graph, 1)
        ph.addWidget(centre, 1)
        self.panel = Panel(s, self.steps)
        ph.addWidget(self.panel)
        self.modes.addWidget(page)
        # Inspect and Post
        self.src = SourcesView(s)
        self.frames = FramesView(s, self.src)
        self.post = PostPage3(s)
        try:
            self.post.view.b_export.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.post.view.b_export.clicked.connect(self.export)
        self.tune = TuneView(s, self.steps, self.frames, self.post)
        self.modes.addWidget(self.tune)
        self.post.goWorkflow.connect(lambda: self.set_mode(0))
        self.post.view.goProcessing.connect(lambda pid, k: (self.select(index=self.index_of(pid), step=k),
                                                            self.set_mode(1)))
        self.post.view.goResults.connect(lambda pid: (self.select(index=self.index_of(pid), step="post"),
                                                      self.set_mode(1)))
        self._last_step = None
        # wiring
        gv = self.graph
        gv.blockClicked.connect(self.select_node)
        gv.blockOpened.connect(lambda n: (self.select_node(n), self.set_mode(2 if n.key == "final" else 1)))
        gv.newWorkflow.connect(lambda: self.new_workflow())
        gv.toolDropped.connect(self.add_tool)
        gv.removeTool.connect(self.remove_tool)
        gv.workflowMenu.connect(self.workflow_action)
        gv.sendToPost.connect(self.send_to_post)
        for pn in (self.panel, self.tune.panel):
            pn.changed.connect(self.request_refresh)
            pn.openTune.connect(lambda: self.set_mode(1))
            pn.openPost.connect(lambda: self.set_mode(2))
            pn.variantMade.connect(self.variant_made)
            pn.addSources.connect(self.add_sources)
            pn.addSession.connect(self.add_session)
            pn.renamed.connect(self.rename)
            pn.fondChanged.connect(lambda v: [self.add_tool(v, q.pid) for q in self.s.scope_paths()])
            pn.removeTool.connect(lambda: self.remove_tool(self.current_node(), None))
        self.frames.addSources.connect(self.add_sources)
        self.frames.addSession.connect(self.add_session)
        self.frames.stackMode.connect(lambda m: (self.s.set_stack_mode(m), self.request_refresh()))
        self.src.goChain.connect(lambda k: self.select(step=k) if self.s.path is not None else None)
        self.tune.back.connect(lambda: self.set_mode(0))
        self.tune.pick.connect(lambda k: self.select(step=k))
        self.tune.workflow.connect(lambda i: self.select(index=i))
        for sig in (s.pathsChanged, s.currentChanged, s.costChanged, s.sourcesChanged):
            sig.connect(self.request_refresh)
        ret = QShortcut(QKeySequence("Return"), self.graph)
        ret.setContext(Qt.WidgetWithChildrenShortcut)
        ret.activated.connect(lambda: self.set_mode(2 if self.s.sel_step == "final" else 1))
        esc = QShortcut(QKeySequence("Escape"), self.tune)
        esc.setContext(Qt.WidgetWithChildrenShortcut)
        esc.activated.connect(lambda: self.set_mode(0))
        self.proj_screen = screen
        self.screens.addWidget(screen)
        self.screens.setCurrentWidget(screen)
        self.mode = 0
        self.set_mode(0)

    # ------------------------------------------------------------ refresh
    def request_refresh(self):
        if not self._pending:
            self._pending = True
            QTimer.singleShot(0, self.refresh)

    def project_name(self):
        s = self.s
        if getattr(s, "display_name", None) and not s.sources_loaded:
            return f"<b>{s.display_name}</b>"
        if s.sources_loaded or s.paths:
            return f"<b>{s.data['project']['name']}</b>  ·  {s.data['project']['target']}"
        return "<b>" + _("New project") + "</b>"

    def refresh(self):
        self._pending = False
        s = self.s
        if s is None or getattr(self, "proj_screen", None) is None:
            return
        if s.path is None and s.paths:
            s.current = 0
        self.title.setText(self.project_name())
        self.machine.setText(_(MACHINES[s.gpu]))
        self.machine.setStyleSheet("" if s.gpu else f"color: {WARN};")
        self.setWindowTitle("astro_gpu · " + (s.data["project"]["name"] if s.sources_loaded else _("New project")))
        if self.mode == 0:
            self.graph.rebuild()
            g = self.graph.graph
            self.panel.refresh(self.current_node(g))
            self.build_tools()
        else:
            g = Graph(s, self.steps)
            self.graph.graph = g
            self.tune.refresh(g)
        on_post = self.mode == 1 and s.sel_step == "final"
        if on_post and not getattr(self, "_on_post", False):
            self.post.refresh()
        self._on_post = on_post
        btn = 0 if self.mode == 0 else (2 if s.sel_step == "final" else 1)
        self.mode_btns[btn].setChecked(True)
        items = s.all_pending() if s.paths else []
        sec = sum(s.seconds(x) for _p, x in items)
        self.bt_run.setText(_("Run") + (f"  ·  {hm(sec)}" if items else ""))
        self.bt_run.setEnabled(bool(items) and s.sources_loaded)
        self.bt_run.setToolTip("" if s.sources_loaded else _("No frames yet"))
        last = getattr(self, "_last_sec", None)
        if last is not None and s is getattr(self, "_last_s", None) and sec - last >= 30 and s.paths and \
                len(s.paths) == getattr(self, "_last_n", 0):
            self.graph.show_toast(_("+ {t} to run").format(t=hm(sec - last)))
        self._last_sec, self._last_s, self._last_n = sec, s, len(s.paths)
        n = len(s.paths)
        self.status.setText(_("{n} workflow(s)").format(n=n) + "  ·  " + (
            _("to run: {t}").format(t=hm(sec)) if items else _("all up to date")) if n else "")
        self.record()
        if self.logp.isVisible():
            self.logp.refresh()
        self.update_spec_note()

    def update_spec_note(self):
        self.spec_note.setText(_("Spec mode · ◇ not in the engine yet · demo data") + "   " if prefs.spec() else "")

    def current_node(self, g=None):
        g = g or Graph(self.s, self.steps)
        k = self.s.sel_step
        if k == "sources":
            return g.root
        if k == "final":
            return g.final
        return g.node_of(self.s.path, k)

    def build_tools(self):
        clear(self.tl)
        s = self.s
        p = s.path
        head = QVBoxLayout()
        head.setSpacing(0)
        head.addWidget(label(_("Add step"), "h2"))
        head.addWidget(label(_('for "{name}"').format(name=p.name) if p is not None else "", "small"))
        self.tl.addLayout(head)
        self.tl.addSpacing(6)
        end = getattr(p, "end", None) if p is not None else None
        for k in TOOLBOX:
            full = ""
            if p is None:
                state, text = "off", ""
            elif end and COLUMNS.index(TOOL_COLUMN[k]) > COLUMNS.index(end):
                state, text = "in", _("after the output")
            elif s.has_tool(p.opts, k):
                state, text = "in", _("already there")
            else:
                new, _notes = s.with_tool(p.opts, k)
                dt = s.opts_seconds(new) - s.opts_seconds(p.opts)
                state, text = "add", (f"+{hm(dt)}" if dt >= 30 else _("same duration"))
            card = ToolCard(self, k, state, text)
            card.setEnabled(p is not None)
            self.tl.addWidget(card)
        self.tl.addStretch(1)

    # ------------------------------------------------------------ undo / redo of the graph
    def snapshot(self):
        s = self.s
        return ([(p.pid, p.name, copy.deepcopy(p.opts), copy.deepcopy(p.params), getattr(p, "end", None),
                  getattr(p, "variant_of", None), p.layer, p.provisional, p.rendered) for p in s.paths],
                s.stack_mode)

    def record(self):
        if self._restoring:
            return
        snap = self.snapshot()
        if getattr(self, "_last_snap", None) is None:
            self._last_snap = snap
            return
        if snap != self._last_snap:
            self.undo_stack.append(self._last_snap)
            self.redo_stack.clear()
            self._last_snap = snap

    def restore(self, snap):
        s = self.s
        rows, s.stack_mode = snap
        s.paths = []
        for pid, name, opts, params, end, var, layer, prov, rendered in rows:
            p = PathDef(pid, name, copy.deepcopy(opts), layer, provisional=prov, params=copy.deepcopy(params),
                        rendered=rendered)
            if end:
                p.end = end
            p.variant_of = var
            s.paths.append(p)
        s.current = min(max(s.current, 0), len(s.paths) - 1)
        s._sync()
        self._restoring = True
        self._last_snap = snap
        self.refresh()
        self._restoring = False

    def undo(self):
        if self.s is not None and self.undo_stack:
            self.redo_stack.append(self.snapshot())
            self.restore(self.undo_stack.pop())
            self.flash(_("Undone"))

    def redo(self):
        if self.s is not None and self.redo_stack:
            self.undo_stack.append(self.snapshot())
            self.restore(self.redo_stack.pop())
            self.flash(_("Redone"))

    # ------------------------------------------------------------ selection and modes
    def select_node(self, node):
        s = self.s
        s.pending_change = None
        if node.key not in ("sources", "final") and node.paths:
            if s.path not in node.paths:
                s.current = s.paths.index(node.paths[0])
            s.scope = [q.pid for q in node.paths]
        else:
            s.scope = None
        s.sel_step = node.key
        self.request_refresh()

    def select(self, index=None, step=None):
        s = self.s
        s.pending_change = None
        if index is not None:
            s.current = index
        if step is not None:
            s.sel_step = step
        node = self.current_node()
        s.scope = [q.pid for q in node.paths] if node is not None and node.key not in ("sources", "final") else None
        self.request_refresh()

    def set_mode(self, i):
        """0 Graph, 1 Inspect the chosen block, 2 Post (Inspect of the Post block)."""
        s = self.s
        if s is None:
            return
        if self.screens.currentWidget() is not self.proj_screen:
            self.screens.setCurrentWidget(self.proj_screen)
        if i == 2:
            if s.sel_step != "final":
                self._last_step = s.sel_step
            s.sel_step = "final"
            s.scope = None
        elif i == 1 and s.sel_step == "final":
            s.sel_step = self._last_step or ("post" if s.path is not None else "sources")
        self.mode = 0 if i == 0 else 1
        self.modes.setCurrentIndex(self.mode)
        self.refresh()

    def variant_made(self, pid, orig):
        """The variant is shown at the same block, side by side with the workflow it comes from."""
        self.tune.cmp_pid = orig
        self.select(index=self.index_of(pid))
        q = self.s.paths[self.index_of(pid)]
        self.graph.show_toast(_('Variant "{name}": it shares everything up to this block').format(name=q.name))

    def set_machine(self, gpu):
        """Demo: the same project on a machine with or without a GPU (durations and methods follow)."""
        self._gpu = gpu
        prefs.set("device", "auto" if gpu else "cpu")
        if self.s is not None:
            self.s.set_machine(gpu)
            self.graph.show_toast(_("Machine: {name}").format(name=_(MACHINES[gpu])))
            self.request_refresh()

    def index_of(self, pid):
        return next(i for i, q in enumerate(self.s.paths) if q.pid == pid)

    def send_to_post(self, node):
        s = self.s
        if not node.paths:
            return
        p = s.path if s.path in node.paths else node.paths[0]
        name = {"empilement": "Raw stack"}.get(node.key, f"{block_title(p, node.key)} output")
        q = s.add_exit(p, node.key, name)
        s.log(f'Layer "{q.name}" from {block_title(p, node.key)}')
        self.select(index=s.paths.index(q), step="post")
        self.graph.show_toast(_('Layer "{name}": output of {block}, sent to Post').format(
            name=q.name, block=block_title(p, node.key)))

    # ------------------------------------------------------------ actions
    def add_sources(self):
        if self.s is None:
            return
        d = ImportDialog(self.src.raws, self)
        if d.exec():
            self.s.load_sources()

    def add_session(self):
        if self.s is not None and self.s.sources_loaded:
            if self.s.add_session():
                self.flash(_("Session 2026-08-19 added: 240 lights, 30 flats"))
            else:
                self.flash(_("Demo: this project already has its two nights"))

    def new_workflow(self, start=None):
        d = NewWorkflowDialog(self.s, start, self)
        if d.exec():
            p = self.s.add_path(d.name.text().strip() or _("New workflow"), d.opts, d.params)
            self.s.log(f'Workflow "{p.name}" created')
            self.select(index=self.s.paths.index(p), step="post")

    def add_tool(self, key, pid):
        s = self.s
        p = next((q for q in s.paths if q.pid == pid), None)
        if p is None or s.has_tool(p.opts, key):
            return
        new, notes = s.with_tool(p.opts, key)
        if new.get("debruitage") and not p.opts.get("debruitage") and not s.gpu:
            new["dn"] = "affine"                         # without a GPU: pretrained, fine-tuned
            notes = notes + [_("Denoise: pretrained + fine-tuned (no GPU)")]
        s.set_opts(p, new)
        s.log(f'{block_title(p, TOOL_COLUMN[key])} added to "{p.name}"')
        s.current = s.paths.index(p)
        s.sel_step = TOOL_COLUMN.get(key, key)
        node = Graph(s, self.steps).node_of(p, s.sel_step)
        s.scope = [q.pid for q in node.paths] if node is not None else [p.pid]
        if notes:
            self.graph.show_toast(" · ".join(_(n) for n in notes))
        self.request_refresh()

    def remove_tool(self, node, pids):
        s = self.s
        if node is None:
            return
        key = node.key
        for p in [q for q in node.paths if pids is None or q.pid in pids]:
            new, notes = s.without_tool(p.opts, key)
            if new is None:
                self.graph.show_toast(_('{block} cannot be removed from "{name}": {why}').format(
                    block=block_title(p, key), name=p.name, why=_(notes[0])))
                continue
            s.set_opts(p, new)
            s.log(f'{block_title(p, key)} removed from "{p.name}"')
            if notes:
                self.graph.show_toast(" · ".join(_(n) for n in notes))
        s.sel_step = node.parent.key if node.parent is not None else "sources"
        self.request_refresh()

    def rename(self, pid, name):
        p = next(q for q in self.s.paths if q.pid == pid)
        p.name = name
        self.request_refresh()

    def workflow_action(self, action, pid):
        s = self.s
        p = next(q for q in s.paths if q.pid == pid)
        if action == "rename":
            t = ask_name(self, _("Rename"), p.name)
            if t:
                self.rename(pid, t)
        elif action == "duplicate":
            self.new_workflow(p)
        elif action == "delete":
            if QMessageBox.question(self, _("Delete"), _('Delete the workflow "{name}"?').format(name=p.name)) \
                    == QMessageBox.Yes:
                s.remove_path(s.paths.index(p))
                s.log(f'Workflow "{p.name}" deleted')
                s.sel_step = "sources"
                self.request_refresh()

    def launch(self):
        if self.s is not None:
            run_blocks(self.s, self.steps, self.s.all_pending(), _("Run project"), self, show_done=True)

    def run_to_here(self):
        s = self.s
        if s is None or s.path is None or s.sel_step in ("sources", "final"):
            return
        from tune import run_stages
        items = s.pending_upto(s.path, run_stages(s, self.steps, s.path, s.sel_step))
        run_blocks(s, self.steps, items, _('Run "{name}" to {block}').format(
            name=s.path.name, block=block_title(s.path, s.sel_step)), self)

    def export(self):
        if self.s is not None:
            ExportDialog(self.s.data["project"]["name"], self).exec()

    # ------------------------------------------------------------ settings, language, spec mode, log
    def open_settings(self):
        d = SettingsDialog(self)
        d.languageChanged.connect(self.change_language)
        d.deviceChanged.connect(self.set_machine)
        d.specChanged.connect(lambda on: (self.a_spec.setChecked(on), self.request_refresh()))
        d.exec()

    def change_language(self, code):
        set_language(code)
        self.build_menus()
        self.b_log.setText(_("Log"))
        if getattr(self, "home", None) is not None:
            on_home = self.screens.currentWidget() is self.home
            self.screens.removeWidget(self.home)
            self.home.deleteLater()
            self.home = None
            self.show_home()
            if not on_home and self.s is not None:
                self.screens.setCurrentWidget(self.proj_screen)
        if self.s is not None:
            on_proj = self.screens.currentWidget() is getattr(self, "proj_screen", None)
            self.build_project()
            if not on_proj:
                self.show_home()

    def toggle_spec(self, on):
        prefs.set("spec_mode", on)
        self.request_refresh()

    def toggle_log(self, on):
        if getattr(self, "logp", None) is not None:
            self.logp.setVisible(on)
            if on:
                self.logp.refresh()

    def flash(self, text):
        self.statusBar().showMessage(text, 4000)

    def help(self):
        d = QDialog(self)
        d.setWindowTitle(_("About this mockup"))
        lv = QVBoxLayout(d)
        t = QLabel(HELP.format(
            title=_("Mockup alt3"),
            p1=_("A mockup, not the software: nothing is computed. Images, measurements, durations and settings come "
                 "from the Heart project, prepared once with the real engine."),
            p2=_("Demo menu: a new project or the existing one, a second night, a machine with or without a GPU. "
                 "Spec mode (View menu) marks with ◇ what the engine does not have yet."),
            p3=_("Shortcuts: Ctrl+1 Graph, Ctrl+2 Inspect, Ctrl+3 Post, Enter opens the selected block, Esc back "
                 "to the graph, Ctrl+Z / Ctrl+Y undo / redo, F5 run.")))
        t.setWordWrap(True)
        t.setTextFormat(Qt.RichText)
        lv.addWidget(t)
        d.resize(640, 320)
        d.exec()

    def glossary(self):
        p = Path(__file__).resolve().parents[1] / "GLOSSARY.md"
        d = QDialog(self)
        d.setWindowTitle(_("Glossary"))
        lv = QVBoxLayout(d)
        from PySide6.QtWidgets import QTextBrowser
        tb = QTextBrowser()
        tb.setMarkdown(p.read_text(encoding="utf-8"))
        lv.addWidget(tb)
        d.resize(760, 640)
        d.exec()


def main():
    import traceback
    prefs.load()
    set_language(prefs.get("language"))
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS + EXTRA_QSS)
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))

    def show_error(t, v, tb):
        QMessageBox.critical(None, "alt3", "".join(traceback.format_exception(t, v, tb))[-3000:])

    sys.excepthook = show_error
    if not (DATA / "data.json").exists():
        QMessageBox.warning(None, "alt3", "The alt2 images are not prepared (mockup/alt2/build).")
        return
    w = Main()
    w.resize(1680, 1000)
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
