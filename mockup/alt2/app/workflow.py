"""Workflow view.

On the left, the project as a tree: the steps common to all the workflows once, then one branch per workflow with
its own steps; "+ ajouter ici" where a tool can go; the Outils drawer under the tree for dragging tools in.
On the right, three header levels, each with its own buttons (the workflow, the step, the Settings / Result tabs).
A workflow is renamed by a double-click on its name, or from its "⋯" menu.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QMimeData, QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea, QStackedWidget, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from dialogs import NewWorkflowDialog, RunDialog, RunPlanDialog, run
from model import TOOLS, TRUNK, PathDef, Session, hm
from results import DetachedResult, ResultPane, state_chip
from settings import SettingRow, fmt, step_settings
from steps import BY_KEY, STEPS, Node, Steps, fnum, kv
from widgets import ACCENT, OK, WARN, Box, clear, hline, label, para

TOOL_STEP = {"debruitage": "debruitage", "etoiles": "etoiles", "appris": "appris", "halpha": "fond", "simple": "fond"}
TRUNK_STEPS = ["calibration", "alignement", "selection", "empilement", "demi_piles"]


# ======================================================================== small pieces
class InlineName(QWidget):
    """A name renamed in place by a double-click (Enter to keep, Échap to cancel)."""
    renamed = Signal(str)

    def __init__(self, text, style="font-weight: 600;", parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.lb = QLabel(text)
        self.lb.setStyleSheet(style + " background: transparent;")
        self.lb.setAttribute(Qt.WA_TransparentForMouseEvents)  # the double-click reaches this widget
        self.setToolTip("Double-clic pour renommer")
        self.ed = QLineEdit(text)
        self.ed.setVisible(False)
        self.ed.editingFinished.connect(self._done)
        h.addWidget(self.lb)
        h.addWidget(self.ed)

    def mouseDoubleClickEvent(self, e):
        self.start()

    def start(self):
        self.lb.setVisible(False)
        self.ed.setText(self.lb.text())
        self.ed.setVisible(True)
        self.ed.setFocus()
        self.ed.selectAll()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self.ed.isVisible():
            self.ed.setText(self.lb.text())
            self._done()
        else:
            super().keyPressEvent(e)

    def _done(self):
        if not self.ed.isVisible():
            return
        t = self.ed.text().strip()
        self.ed.setVisible(False)
        self.lb.setVisible(True)
        if t and t != self.lb.text():
            self.lb.setText(t)
            self.renamed.emit(t)


def menu_button(parent=None) -> QToolButton:
    b = QToolButton(parent)
    b.setText("⋯")
    b.setToolTip("Renommer, dupliquer, supprimer")
    b.setPopupMode(QToolButton.InstantPopup)
    b.setStyleSheet("QToolButton { border: none; color: #aab2bf; font-size: 13pt; padding: 0 6px; }"
                    "QToolButton:hover { color: #ffffff; background: #323846; border-radius: 4px; }"
                    "QToolButton::menu-indicator { image: none; }")
    return b


class ToolCard(QFrame):
    """A tool of the Outils drawer: dragged onto the tree, or added by a double-click."""

    def __init__(self, view, key, state, text, parent=None):
        super().__init__(parent)
        self.view, self.key, self.state = view, key, state
        self._press = None
        on = state == "in"
        self.setObjectName("tool")
        self.setCursor(Qt.ArrowCursor if on else Qt.OpenHandCursor)
        self.setStyleSheet(f"QFrame#tool {{ background: {'transparent' if on else '#232831'}; border: 1px "
                           f"{'dashed #3a4150' if on else 'solid #353b47'}; border-radius: 6px; }}"
                           "QFrame#tool:hover { border-color: %s; } QLabel { background: transparent; }"
                           % ("#3a4150" if on else ACCENT))
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 5, 8, 5)
        grip = QLabel("⋮⋮")
        grip.setStyleSheet("color: #6b7280;" if not on else "color: transparent;")
        h.addWidget(grip)
        v = QVBoxLayout()
        v.setSpacing(0)
        t = QLabel(TOOLS[key]["title"])
        t.setStyleSheet("font-weight: 600;" + (" color: #8b93a1;" if on else ""))
        v.addWidget(t)
        st = label(("✓ " if on else "") + text, "small", wrap=True)
        st.setStyleSheet(f"color: {OK if on else '#c9b37e'}; font-size: 9pt;")
        v.addWidget(st)
        h.addLayout(v, 1)
        self.setToolTip(TOOLS[key]["what"])

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self.state == "in" or self._press is None or not (e.buttons() & Qt.LeftButton):
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
        view = self.view
        view.dropped = None
        drag.exec(Qt.CopyAction)
        key, view.dropped = view.dropped, None
        view.show_drop(None)
        if key:                                            # out of the drag loop before rebuilding the drawer
            QTimer.singleShot(0, lambda: view.add_tool(key))

    def mouseDoubleClickEvent(self, e):
        if self.state != "in":
            self.view.add_tool(self.key)


class TreeHost(QWidget):
    """The tree accepts the dropped tools anywhere: each one goes to its own place in the chosen workflow."""

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.setAcceptDrops(True)

    def _key(self, e):
        t = e.mimeData().text() if e.mimeData().hasText() else ""
        return t[5:] if t.startswith("tool:") else None

    def dragEnterEvent(self, e):
        k = self._key(e)
        p = self.view.s.path
        if k and p is not None and not self.view.s.has_tool(p.opts, k):
            e.acceptProposedAction()
            self.view.show_drop(k)
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self.view.show_drop(None)

    def dropEvent(self, e):
        self.view.dropped = self._key(e)
        e.acceptProposedAction()


class StepRow(QFrame):
    clicked = Signal(str)

    def __init__(self, view, p: PathDef, b, selected, first, last, indent=0):
        super().__init__()
        self.key = b["key"]
        st = view.steps
        state = st.state(p, b)
        auto = b["kind"] == "auto"
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("blk")
        self.setStyleSheet(("QFrame#blk { background: #2b3446; border: 1px solid #3d4a63; border-radius: 6px; }"
                            if selected else
                            "QFrame#blk { background: transparent; border: 1px solid transparent; border-radius: 6px; }"
                            "QFrame#blk:hover { background: #232831; }") + " QLabel { background: transparent; }")
        h = QHBoxLayout(self)
        h.setContentsMargins(2 + indent, 2, 6, 2)
        h.setSpacing(6)
        h.addWidget(Node(state, first, last, dim=auto))
        v = QVBoxLayout()
        v.setSpacing(0)
        top = QHBoxLayout()
        top.setSpacing(6)
        t = QLabel(st.title(p, b))
        t.setStyleSheet("font-weight: 600;" + (" color: #aab2bf; font-weight: 500;" if auto else ""))
        top.addWidget(t)
        top.addStretch(1)
        txt, col = st.state_text(p, b)
        if state == "stale":
            txt = "à recalculer"
        s_ = QLabel(txt)
        s_.setStyleSheet(f"color: {col}; font-size: 9pt;")
        top.addWidget(s_)
        v.addLayout(top)
        summ = st.summary(p, b)
        if summ:
            v.addWidget(label(summ, "small"))
        h.addLayout(v, 1)
        if b["kind"] == "tool":
            x = QPushButton("✕")
            x.setFixedSize(20, 20)
            x.setStyleSheet("QPushButton { padding: 0; background: transparent; border: none; color: #6b7280; }"
                            "QPushButton:hover { color: #e06c6c; }")
            x.setToolTip("Retirer du workflow")
            x.clicked.connect(lambda: view.remove_tool(b["tool"]))
            h.addWidget(x, 0, Qt.AlignTop)

    def mousePressEvent(self, e):
        self.clicked.emit(self.key)


class WorkflowRow(QFrame):
    """A workflow in the tree: unfolds on a click, renamed by a double-click on its name, ⋯ for its menu."""
    clicked = Signal(int)

    def __init__(self, view, i, p: PathDef, current):
        super().__init__()
        self.i = i
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("wfr")
        self.setStyleSheet(("QFrame#wfr { background: #222833; border-radius: 6px; }" if current else
                            "QFrame#wfr { background: transparent; border-radius: 6px; } QFrame#wfr:hover "
                            "{ background: #20242c; }") + " QLabel { background: transparent; }")
        h = QHBoxLayout(self)
        h.setContentsMargins(4, 4, 2, 4)
        h.setSpacing(6)
        arrow = QLabel("▾" if current else "▸")
        arrow.setStyleSheet("color: #aab2bf;")
        arrow.setFixedWidth(12)
        h.addWidget(arrow)
        self.name = InlineName(p.name, "font-weight: 600; font-size: 10.5pt;")
        self.name.renamed.connect(lambda t: view.rename(self.i, t))
        h.addWidget(self.name)
        h.addStretch(1)
        todo = view.s.pending(p) + view.s.trunk_pending()
        if todo:
            t = label(f"à calculer · {hm(sum(view.s.seconds(x) for x in todo))}", "small")
            t.setStyleSheet(f"color: {WARN if p.rendered else '#8b93a1'}; font-size: 9pt;")
        else:
            t = label("calculé", "small")
            t.setStyleSheet(f"color: {OK}; font-size: 9pt;")
        h.addWidget(t)
        mb = menu_button()
        m = QMenu(mb)
        m.addAction("Renommer").triggered.connect(lambda: self.name.start())
        m.addAction("Dupliquer").triggered.connect(lambda: view.new_workflow(p))
        m.addSeparator()
        m.addAction("Supprimer").triggered.connect(lambda: view.remove(self.i))
        mb.setMenu(m)
        h.addWidget(mb)

    def mousePressEvent(self, e):
        self.clicked.emit(self.i)


class TabKey(QObject):
    """Tab switches the Settings / Result tabs, except while typing in a field."""

    def __init__(self, view):
        super().__init__(view)
        self.view = view

    def eventFilter(self, obj, e):
        if e.type() == QEvent.KeyPress and e.key() == Qt.Key_Tab and not e.modifiers() and self.view.isVisible():
            fw = QApplication.focusWidget()
            typing = isinstance(fw, (QAbstractSpinBox, QLineEdit, QComboBox, QPlainTextEdit)) or (
                fw is not None and isinstance(fw.parent(), QAbstractSpinBox))
            if not typing and (fw is None or fw.window() is self.view.window()) and self.view.s.path is not None:
                self.view.set_face(1 - self.view.face)
                return True
        return False


# ======================================================================== the view
class WorkflowView(QWidget):
    openSources = Signal(int)
    openPost = Signal()

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.s = session
        self.steps = Steps(session)
        self.show_expert = False
        self.drop_tool = None
        self.dropped = None
        self._last = None
        self._dirty = False
        self.face = 0
        self._seen = (None, None, None)                  # (workflow, step, state) at the last refresh
        self.detached = None
        self.drawer_open = False
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        # ---- left: the project tree and the Outils drawer
        left = QWidget()
        left.setObjectName("panel")
        left.setFixedWidth(390)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 6, 8)
        lv.setSpacing(4)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.host = TreeHost(self)
        self.tl = QVBoxLayout(self.host)
        self.tl.setContentsMargins(0, 0, 6, 0)
        self.tl.setSpacing(1)
        sc.setWidget(self.host)
        lv.addWidget(sc, 1)
        self.drawer = QVBoxLayout()
        self.drawer.setSpacing(4)
        lv.addLayout(self.drawer)
        root.addWidget(left)
        # ---- right
        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)
        # page 0: no workflow yet
        self.empty = QWidget()
        self.ev = QVBoxLayout(self.empty)
        self.ev.setAlignment(Qt.AlignCenter)
        self.pages.addWidget(self.empty)
        # page 1: workflow header, step header, tabs
        body = QWidget()
        bv = QVBoxLayout(body)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(0)
        wh = QWidget()
        wh.setObjectName("wfhead")
        wh.setStyleSheet("QWidget#wfhead { background: #1a1d23; border-bottom: 1px solid #2a2f38; }")
        self.wfhead = QHBoxLayout(wh)
        self.wfhead.setContentsMargins(18, 8, 14, 8)
        bv.addWidget(wh)
        sh = QWidget()
        self.sthead = QHBoxLayout(sh)
        self.sthead.setContentsMargins(18, 12, 14, 8)
        self.st_left = QHBoxLayout()
        self.sthead.addLayout(self.st_left)
        self.sthead.addStretch(1)
        self.detach_b = QPushButton("⧉  Détacher le résultat")
        self.detach_b.setCheckable(True)
        self.detach_b.setObjectName("toggle")
        self.detach_b.setToolTip("Le résultat dans une fenêtre à part, qui suit l'étape choisie (second écran)")
        self.detach_b.toggled.connect(self.toggle_detach)
        self.sthead.addWidget(self.detach_b)
        self.sthead.addSpacing(8)
        self.st_run = QHBoxLayout()
        self.sthead.addLayout(self.st_run)
        bv.addWidget(sh)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("faces")
        tw = QWidget()
        tv = QVBoxLayout(tw)
        tv.setContentsMargins(14, 0, 14, 10)
        tv.addWidget(self.tabs)
        bv.addWidget(tw, 1)
        # tab 0: settings
        f0 = QWidget()
        sv = QVBoxLayout(f0)
        sv.setContentsMargins(16, 10, 6, 6)
        sv.setSpacing(6)
        self.notice = QVBoxLayout()
        sv.addLayout(self.notice)
        ssc = QScrollArea()
        ssc.setWidgetResizable(True)
        ssc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        shost = QWidget()
        self.sl = QVBoxLayout(shost)
        self.sl.setContentsMargins(0, 0, 14, 4)
        self.sl.setSpacing(2)
        ssc.setWidget(shost)
        sv.addWidget(ssc, 1)
        self.tabs.addTab(f0, "Réglages")
        # tab 1: the result
        self.result = ResultPane(session)
        self.result.editStep.connect(lambda: self.set_face(0))
        self.result.openSources.connect(self.openSources.emit)
        self.result.openPost.connect(self.openPost.emit)
        self.tabs.addTab(self.result, "Résultat")
        self.tabs.setTabToolTip(0, "Les réglages de l'étape (touche Tab pour basculer)")
        self.tabs.setTabToolTip(1, "Le résultat de l'étape (touche Tab pour basculer)")
        self.tabs.currentChanged.connect(self.set_face)
        self.pages.addWidget(body)

        for sig in (session.pathsChanged, session.currentChanged, session.sourcesChanged):
            sig.connect(self.request)
        session.costChanged.connect(self._cost_changed)
        self.tabkey = TabKey(self)
        QApplication.instance().installEventFilter(self.tabkey)
        self.refresh()

    # ------------------------------------------------------------ refresh
    def request(self):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    def showEvent(self, e):
        if self._dirty:
            self.refresh()
        super().showEvent(e)

    def _cost_changed(self):
        if not self.isVisible():
            self._dirty = True
            return
        if self.face == 1 and self.s.sel_step == "selection":
            self.build_tree()
            self.build_heads()
            self.result.cost_changed()
        else:
            self.refresh()

    @property
    def path(self) -> PathDef | None:
        return self.s.path

    def refresh(self):
        self._dirty = False
        if self.path is None and self.s.paths:           # a workflow was just added, before it is selected
            self.s.current = len(self.s.paths) - 1
        p = self.path
        if p is not None:
            b = BY_KEY.get(self.s.sel_step)
            if b is None or not self.steps.included(p, b):
                keys = [x["key"] for x in STEPS]
                i = keys.index(self.s.sel_step) if self.s.sel_step in keys else 0
                while i > 0 and not self.steps.included(p, STEPS[i]):
                    i -= 1
                self.s.sel_step = STEPS[i]["key"]
        self.build_tree()
        self.build_drawer()
        if p is None:
            self.build_empty()
            self.pages.setCurrentIndex(0)
            return
        self.pages.setCurrentIndex(1)
        b = BY_KEY[self.s.sel_step]
        state = self.steps.state(p, b)
        seen = (p.pid, b["key"], state)
        if seen[:2] != self._seen[:2]:
            self.face = self.auto_face(state)            # a new step: the tab follows its state
        elif self._seen[2] == "pending" and state in ("done", "shared") and self.detached is None:
            self.face = 1                                # just computed: show it
        self._seen = seen
        self.build_heads()
        self.build_settings()
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(self.face)
        self.tabs.blockSignals(False)
        if self.face == 1:
            self.result.refresh()
        else:
            self.result._dirty = True

    def auto_face(self, state) -> int:
        if self.detached is not None:
            return 0
        return 0 if state == "pending" else 1

    def set_face(self, i):
        if self.path is None:
            return
        self.face = i
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(i)
        self.tabs.blockSignals(False)
        if i == 1:
            self.result.refresh()

    def build_empty(self):
        clear(self.ev)
        self.ev.addWidget(label("Aucun workflow dans ce projet", "h1"), 0, Qt.AlignHCenter)
        self.ev.addWidget(para("Un workflow va de la calibration au calque pour le post. Chaque workflow calculé "
                               "donne un calque en post."), 0, Qt.AlignHCenter)
        self.ev.addSpacing(10)
        b = QPushButton("+ Nouveau workflow")
        b.setObjectName("primary")
        b.clicked.connect(lambda: self.new_workflow())
        self.ev.addWidget(b, 0, Qt.AlignHCenter)
        if not self.s.sources_loaded:
            self.ev.addSpacing(16)
            self.ev.addWidget(label("Pas encore de sources : le workflow pourra se créer, pas se lancer.", "warn"),
                              0, Qt.AlignHCenter)
            g = QPushButton("Ajouter les sources  →")
            g.setObjectName("link")
            g.clicked.connect(lambda: self.openSources.emit(-1))
            self.ev.addWidget(g, 0, Qt.AlignHCenter)

    # ------------------------------------------------------------ the project tree
    def build_tree(self):
        clear(self.tl)
        s = self.s
        p = self.path
        n = len(s.paths)
        if s.paths:
            head = label(f"Commun aux {n} workflows" if n > 1 else "Étapes communes", "h2")
            head.setContentsMargins(4, 2, 0, 2)
            self.tl.addWidget(head)
            trunk = [BY_KEY[k] for k in TRUNK_STEPS
                     if k != "demi_piles" or any(q.opts["debruitage"] for q in s.paths)]
            for j, b in enumerate(trunk):
                ctx = p if p is not None and self.steps.included(p, b) else next(
                    q for q in s.paths if self.steps.included(q, b))
                row = StepRow(self, ctx, b, b["key"] == s.sel_step, j == 0, j == len(trunk) - 1)
                row.clicked.connect(self.select_step)
                self.tl.addWidget(row)
            self.tl.addSpacing(10)
            for i, q in enumerate(s.paths):
                wr = WorkflowRow(self, i, q, i == s.current)
                wr.clicked.connect(self.select_workflow)
                self.tl.addWidget(wr)
                if i == s.current:
                    self.build_branch(q)
                    self.tl.addSpacing(6)
        nb = QPushButton("+ Nouveau workflow")
        nb.setObjectName("link")
        nb.setStyleSheet("padding: 6px 4px;")
        nb.clicked.connect(lambda: self.new_workflow())
        self.tl.addWidget(nb, 0, Qt.AlignLeft)
        self.tl.addStretch(1)

    def anchors(self, p):
        """Where each tool not in the workflow would go: {step key it goes before: [tools]}."""
        order = [b["key"] for b in STEPS]
        rows = [b["key"] for b in self.steps.steps_of(p)]
        out = {}
        for k in TOOLS:
            if self.s.has_tool(p.opts, k):
                continue
            at = TOOL_STEP[k]
            before = "fond" if at == "fond" else next((r for r in rows if order.index(r) > order.index(at)), None)
            out.setdefault(before, []).append(k)
        return out

    def build_branch(self, p):
        rows = [b for b in self.steps.steps_of(p) if b["phase"] >= 4]
        anchors = self.anchors(p)
        drop_before = None
        if self.drop_tool:
            drop_before = next((k for k, ts in anchors.items() if self.drop_tool in ts), None)
        phase = None
        for j, b in enumerate(rows):
            if b["phase"] != phase:
                phase = b["phase"]
                ph = label("Traitement linéaire" if phase == 4 else "Post", "small")
                ph.setContentsMargins(26, 4, 0, 0)
                self.tl.addWidget(ph)
            if b["key"] == drop_before:
                self.tl.addWidget(self._placeholder())
            elif b["key"] in anchors:
                self.tl.addWidget(self._add_point(p, anchors[b["key"]]))
            row = StepRow(self, p, b, b["key"] == self.s.sel_step, j == 0, j == len(rows) - 1, indent=14)
            row.clicked.connect(self.select_step)
            self.tl.addWidget(row)

    def _add_point(self, p, tools):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(42, 0, 8, 0)
        b = QPushButton("+ ajouter ici")
        b.setObjectName("addpt")
        b.setStyleSheet("QPushButton#addpt { background: transparent; border: 1px dashed #3a4150; color: #7ea6e0;"
                        " text-align: left; padding: 1px 10px; font-size: 9pt; border-radius: 4px; }"
                        f"QPushButton#addpt:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}")
        h.addWidget(b, 1)
        b.setToolTip("Peut aller ici : " + ", ".join(TOOLS[k]["title"] for k in tools))
        m = QMenu(b)
        for k in tools:
            a = m.addAction(f"{TOOLS[k]['title']}   ·   {self.s.tool_status(p.opts, k)[1]}")
            a.setToolTip(TOOLS[k]["what"])
            a.triggered.connect(lambda _=False, kk=k: self.add_tool(kk))
        m.setToolTipsVisible(True)
        b.setMenu(m)
        return w

    def _placeholder(self):
        k = self.drop_tool
        f = QFrame()
        f.setObjectName("ph")
        f.setStyleSheet(f"QFrame#ph {{ border: 2px dashed {ACCENT}; border-radius: 6px; background: #1d2633; "
                        "margin-left: 14px; } QLabel { background: transparent; }")
        v = QVBoxLayout(f)
        v.setContentsMargins(30, 6, 8, 6)
        t = label(f"{TOOLS[k]['title']} ira ici", "")
        t.setStyleSheet(f"color: {ACCENT}; font-weight: 600;")
        v.addWidget(t)
        _, notes = self.s.with_tool(self.path.opts, k)
        if notes:
            v.addWidget(label(" · ".join(notes), "small", wrap=True))
        return f

    def show_drop(self, key):
        if key == self.drop_tool:
            return
        self.drop_tool = key
        self.build_tree()

    def select_step(self, key):
        s = self.s
        b = BY_KEY[key]
        if s.path is not None and not self.steps.included(s.path, b):
            i = next((i for i, q in enumerate(s.paths) if self.steps.included(q, b)), s.current)
            s.select(index=i, step=key)
        else:
            s.select(step=key)

    def select_workflow(self, i):
        if i != self.s.current:
            self.s.select(index=i)

    # ------------------------------------------------------------ the Outils drawer
    def build_drawer(self):
        clear(self.drawer)
        p = self.path
        if p is None:
            return
        self.drawer.addWidget(hline())
        b = QPushButton(("▾  " if self.drawer_open else "▸  ") + f"Outils à ajouter à « {p.name} »")
        b.setObjectName("link")
        b.setStyleSheet("color: #d8dce4; font-weight: 600; padding: 4px 2px;")
        b.clicked.connect(self._toggle_drawer)
        self.drawer.addWidget(b, 0, Qt.AlignLeft)
        if not self.drawer_open:
            return
        for k in TOOLS:
            state, text = self.s.tool_status(p.opts, k)
            self.drawer.addWidget(ToolCard(self, k, state, text))

    def _toggle_drawer(self):
        self.drawer_open = not self.drawer_open
        self.build_drawer()

    # ------------------------------------------------------------ headers: workflow, step
    def build_heads(self):
        p = self.path
        b = BY_KEY[self.s.sel_step]
        clear(self.wfhead)
        n = len(self.s.paths)
        if b["phase"] < 4:
            self.wfhead.addWidget(label(f"Commun aux {n} workflows" if n > 1 else "Étapes communes", "h2"))
            self.wfhead.addSpacing(10)
            self.wfhead.addWidget(label("Un changement ici vaut pour tous les workflows.", "small"))
            self.wfhead.addStretch(1)
        else:
            nm = InlineName(p.name, "font-size: 12pt; font-weight: 600;")
            nm.renamed.connect(lambda t: self.rename(self.s.current, t))
            self.wfhead.addWidget(nm)
            mb = menu_button()
            m = QMenu(mb)
            m.addAction("Renommer").triggered.connect(lambda: nm.start())
            m.addAction("Dupliquer").triggered.connect(lambda: self.new_workflow(p))
            m.addSeparator()
            m.addAction("Supprimer").triggered.connect(lambda: self.remove(self.s.current))
            mb.setMenu(m)
            self.wfhead.addWidget(mb)
            total = self.s.trunk_seconds() + self.s.path_seconds(p)
            self.wfhead.addSpacing(10)
            self.wfhead.addWidget(label(f"{hm(total)} depuis les poses brutes", "small"))
            self.wfhead.addStretch(1)
            al = QPushButton("Tous les réglages…")
            al.clicked.connect(self.open_all)
            self.wfhead.addWidget(al)
            items = self.workflow_items(p)
            sec = sum(self.s.seconds(x) for _, x in items)
            rb = QPushButton(f"Lancer « {p.name} »" + (f"  ·  {hm(sec)}" if items else ""))
            rb.setToolTip("Calcule ce workflow, avec les étapes communes qui manquent")
            rb.setEnabled(bool(items) and self.s.sources_loaded)
            rb.clicked.connect(lambda: self.launch({p.pid}, f"Lancer « {p.name} »"))
            self.wfhead.addWidget(rb)
        clear(self.st_left)
        clear(self.st_run)
        state = self.steps.state(p, b)
        self.st_left.addWidget(label(self.steps.title(p, b), "h1"))
        self.st_left.addWidget(state_chip(state))
        kind = {"fixed": "", "tool": "étape facultative", "auto": "vient avec une autre étape",
                "slot": "deux variantes"}[b["kind"]]
        if kind:
            self.st_left.addSpacing(6)
            self.st_left.addWidget(label(kind, "small"))
        items = self.s.pending_upto(p, self.steps.run_stages(p, b))
        if items:
            stale = any(self.s.state(q, x) == "stale" for q, x in items)
            sec = sum(self.s.seconds(x) for _, x in items)
            rb = QPushButton(f"{'Recalculer' if stale else 'Lancer'} jusqu'ici  ·  {hm(sec)}")
            rb.setObjectName("primary")
            rb.setToolTip("Calcule cette étape et ce qu'elle demande : " +
                          ", ".join(self.steps.step_titles(p, [x for _, x in items])))
            if not self.s.sources_loaded:
                rb.setEnabled(False)
                rb.setToolTip("Ajoute d'abord des sources")
            rb.clicked.connect(lambda: run(self.s, items, self.window(), f"Lancer jusqu'à : {self.steps.title(p, b)}"))
            self.st_run.addWidget(rb)
        else:
            ok = label("à jour", "small")
            ok.setToolTip("Cette étape et ce qu'elle demande sont calculés")
            self.st_run.addWidget(ok)

    def workflow_items(self, p):
        return [(None, x) for x in self.s.trunk_pending()] + [(p, x) for x in self.s.pending(p)]

    def launch(self, pids, title):
        d = RunPlanDialog(self.s, self.steps, pids, title, self.window())
        if d.exec() and d.items():
            RunDialog(self.s, d.items(), title, self.window()).exec()

    # ------------------------------------------------------------ detached result
    def toggle_detach(self, on):
        if on and self.detached is None:
            self.detached = DetachedResult(self.s, self.window())
            self.detached.closed.connect(self._detached_closed)
            self.detached.editStep.connect(lambda: (self.window().raise_(), self.window().activateWindow()))
            self.detached.show()
            self.set_face(0)
        elif not on and self.detached is not None:
            d, self.detached = self.detached, None
            d.close()

    def _detached_closed(self):
        self.detached = None
        self.detach_b.blockSignals(True)
        self.detach_b.setChecked(False)
        self.detach_b.blockSignals(False)

    # ------------------------------------------------------------ workflows
    def new_workflow(self, start=None):
        d = NewWorkflowDialog(self.s, start, self.window())
        if d.exec():
            p = self.s.add_path(d.name.text().strip() or "Nouveau workflow", d.opts, d.params)
            self.s.select(index=self.s.paths.index(p), step="astrometrie")

    def rename(self, i, name):
        if 0 <= i < len(self.s.paths) and name:
            self.s.paths[i].name = name
            QTimer.singleShot(0, self.s.pathsChanged.emit)   # after the name editor has finished

    def remove(self, i):
        p = self.s.paths[i]
        if QMessageBox.question(self.window(), "Supprimer", f"Supprimer le workflow « {p.name} » et son calque ?") \
                == QMessageBox.Yes:
            self.s.remove_path(i)

    # ------------------------------------------------------------ changes
    def _was_run(self, p) -> bool:
        return p.rendered or any((p.pid, st) in self.s.ever for st in self.s.path_stages(p))

    def add_tool(self, key):
        p = self.path
        if p is None or self.s.has_tool(p.opts, key):
            return
        new, notes = self.s.with_tool(p.opts, key)
        self.s.sel_step = TOOL_STEP[key]
        self.change_opts(new, notes, f"Ajouté : {TOOLS[key]['title']}")

    def remove_tool(self, key):
        p = self.path
        new, notes = self.s.without_tool(p.opts, key)
        if new is None:
            QMessageBox.information(self, "Retirer", f"{TOOLS[key]['title']} ne peut pas être retiré : {notes[0]}.")
            return
        self.change_opts(new, notes, f"Retiré : {TOOLS[key]['title']}")

    def change_opts(self, new, notes, what):
        p = self.path
        if new == p.opts:
            return
        self._last = dict(kind="opts", pid=p.pid, old=dict(p.opts), new=new, notes=notes, what=what,
                          was_run=self._was_run(p), step=self.s.sel_step)
        self.s.set_opts(p, new)

    def set_setting(self, r, value):
        """A setting changed from its row, the all-settings page or the text."""
        p = self.path
        target = self.s.trunk_params if r["stage"] in TRUNK else p.params
        old = target.get(r["stage"], {}).get(r["name"], r["suggested"])
        if value == old:
            return
        self._last = dict(kind="param", pid=p.pid, stage=r["stage"], name=r["name"], suggested=r["suggested"],
                          old=old, new=value, notes=[],
                          what=f"{r['label']} : {fmt(old, r['type'], r.get('choices'))} → "
                               f"{fmt(value, r['type'], r.get('choices'))}",
                          was_run=self._was_run(p) or (r["stage"] in TRUNK and bool(self.s.ever)),
                          step=self.s.sel_step)
        self.s.set_param(p, r["stage"], r["name"], value, r["suggested"])

    def _revert(self):
        ch = self._last
        self._last = None
        p = next((q for q in self.s.paths if q.pid == ch["pid"]), None)
        if p is None:
            return
        if ch["kind"] == "opts":
            self.s.set_opts(p, ch["old"])
        else:
            self.s.set_param(p, ch["stage"], ch["name"], ch["old"], ch["suggested"])

    def _branch(self):
        """Keep the workflow as it was; the change goes into a copy."""
        ch = self._last
        self._last = None
        p = next(q for q in self.s.paths if q.pid == ch["pid"])
        if ch["kind"] == "opts":
            self.s.set_opts(p, ch["old"])
            opts, params = ch["new"], p.params
        else:
            self.s.set_param(p, ch["stage"], ch["name"], ch["old"], ch["suggested"])
            params = {k: dict(v) for k, v in p.params.items()}
            params.setdefault(ch["stage"], {})[ch["name"]] = ch["new"]
            opts = p.opts
        np_ = self.s.add_path(self.s.unique_name(f"{p.name} (variante)"), opts, params)
        self.s.select(index=self.s.paths.index(np_))

    def notice_box(self):
        ch = self._last
        p = self.path
        if not ch or p is None or (ch["pid"] != p.pid and not (ch["kind"] == "param" and ch["stage"] in TRUNK)):
            return None
        here = BY_KEY[self.s.sel_step]
        if ch["kind"] == "opts" and here["phase"] < 4 or ch["kind"] == "param" and ch["step"] != here["key"]:
            return None                                  # the notice stays with what it is about
        box = Box()
        box.v.addWidget(label(f"<b>{ch['what']}</b>", wrap=True))
        for n_ in ch["notes"]:
            box.v.addWidget(label("• " + n_, "small", wrap=True))
        row = QHBoxLayout()
        if ch["kind"] == "param" and ch["stage"] in TRUNK:
            items = self.s.all_pending()
            sec = sum(self.s.seconds(x) for _, x in items)
            box.v.addWidget(label(f"Étape commune : le changement vaut pour les {len(self.s.paths)} workflows."
                                  + (f" À recalculer : {hm(sec)}." if ch["was_run"] and items else ""), wrap=True))
        else:
            pend = self.s.pending(p)
            stale = [x for x in pend if self.s.state(p, x) == "stale"]
            if ch["was_run"] and pend:
                names = self.steps.step_titles(p, stale)
                box.v.addWidget(label(f"« {p.name} » a déjà tourné. "
                                      + (f"À recalculer : {', '.join(names)}. " if stale else "")
                                      + f"À calculer en tout : {hm(sum(self.s.seconds(x) for x in pend))}.", wrap=True))
                br = QPushButton("En faire une copie à part")
                br.setToolTip(f"« {p.name} » reste tel qu'il était ; le changement part dans un nouveau workflow")
                br.clicked.connect(self._branch)
                row.addWidget(br)
            elif ch["was_run"]:
                box.v.addWidget(label("Rien à calculer : ces étapes sont déjà calculées par un autre workflow.",
                                      "small", wrap=True))
            twin = next((q for q in self.s.paths if q.pid != p.pid and q.opts == p.opts and q.params == p.params),
                        None)
            if twin:
                box.v.addWidget(label(f"Ce workflow est maintenant identique à « {twin.name} ».", "warn", wrap=True))
        undo = QPushButton("Annuler")
        undo.clicked.connect(self._revert)
        ok = QPushButton("OK")
        ok.clicked.connect(lambda: (setattr(self, "_last", None), self.build_settings()))
        row.addWidget(undo)
        row.addWidget(ok)
        row.addStretch(1)
        box.v.addLayout(row)
        return box

    # ------------------------------------------------------------ the Settings tab
    def build_settings(self):
        p = self.path
        b = BY_KEY[self.s.sel_step]
        clear(self.notice)
        nb = self.notice_box()
        if nb:
            self.notice.addWidget(nb)
        clear(self.sl)
        self.sl.addWidget(label(b["what"], "dim", wrap=True))
        k = b["key"]
        if b["kind"] == "tool":
            self.sl.addWidget(self.remove_box(p, b))
        elif b["kind"] == "slot":
            self.sl.addWidget(self.slot_box(p))
        elif k == "demi_piles":
            users = [q.name for q in self.s.paths if q.opts["debruitage"]]
            self.sl.addWidget(label(f"Pour le débruitage de : {', '.join(users)}.", "small", wrap=True))
        elif k == "sans_etoiles":
            self.sl.addWidget(label("Vient avec la mesure des étoiles.", "small", wrap=True))
        if k == "calibration":
            self.calibration_box()
        if k == "selection":
            self.selection_settings()
        if k == "post":
            self.sl.addWidget(label("Ce que ce workflow donnera en post", "h2"))
            self.sl.addWidget(kv(self.steps.facts(p, "post")))
            go = QPushButton("Ouvrir en post  →")
            go.setObjectName("link")
            go.setEnabled(p.rendered)
            go.clicked.connect(self.openPost.emit)
            self.sl.addWidget(go, 0, Qt.AlignLeft)
        self.settings_list(p, b)
        self.sl.addStretch(1)
        self.sl.addWidget(label(self.steps.state_sentence(p, b), "small", wrap=True))

    def settings_list(self, p, b):
        groups = step_settings(self.s, p, b["key"])
        allr = [r for _, rs in groups for r in rs]
        if not allr:
            return
        n_exp = sum(r["level"] == "expert" for r in allr)
        n_sys = sum(r["level"] == "system" for r in allr)
        n_new = sum(r["kind"] == "new" for r in allr)
        head = QHBoxLayout()
        head.addWidget(label("Réglages", "h2"))
        bits = [f"{len(allr)} réglages"]
        if n_exp:
            bits.append(f"{n_exp} en mode expert")
        if n_sys:
            bits.append(f"{n_sys} système")
        if n_new:
            bits.append(f"{n_new} à créer dans le moteur")
        head.addSpacing(8)
        head.addWidget(label(" · ".join(bits), "small"))
        head.addStretch(1)
        ex = QCheckBox("Expert")
        ex.setChecked(self.show_expert)
        ex.setToolTip("Montrer aussi les réglages fins et, à part, les réglages système")
        ex.toggled.connect(self._expert)
        head.addWidget(ex)
        self.sl.addSpacing(6)
        self.sl.addLayout(head)
        system = []
        for title, rs in groups:
            shown = [r for r in rs if r["level"] == "normal" or (self.show_expert and r["level"] == "expert")]
            system += [r for r in rs if r["level"] == "system"]
            if not shown:
                continue
            g = label(title, "dim")
            g.setStyleSheet("color: #aab2bf; font-weight: 600; margin-top: 8px;")
            self.sl.addWidget(g)
            self.sl.addWidget(hline())
            for r in shown:
                row = SettingRow(r)
                row.changed.connect(self.set_setting)
                self.sl.addWidget(row)
        if self.show_expert and system:
            g = label("Réglages système : vitesse, mémoire, reproductibilité", "dim")
            g.setStyleSheet("color: #aab2bf; font-weight: 600; margin-top: 12px;")
            self.sl.addWidget(g)
            self.sl.addWidget(hline())
            for r in system:
                row = SettingRow(r)
                row.changed.connect(self.set_setting)
                self.sl.addWidget(row)
        hidden = n_exp + n_sys
        if not self.show_expert and hidden:
            more = QPushButton(f"+ {hidden} réglage{'s' if hidden > 1 else ''} en mode expert")
            more.setObjectName("link")
            more.clicked.connect(lambda: self._expert(True))
            self.sl.addWidget(more, 0, Qt.AlignLeft)

    def _expert(self, v):
        self.show_expert = v
        self.build_settings()

    def remove_box(self, p, b):
        key = b["tool"]
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 2, 0, 4)
        new, notes = self.s.without_tool(p.opts, key)
        row = QHBoxLayout()
        rb = QPushButton("Retirer du workflow")
        if new is None:
            rb.setEnabled(False)
            row.addWidget(rb)
            row.addWidget(label(f"Impossible : {notes[0]}.", "warn", wrap=True), 1)
        else:
            rb.clicked.connect(lambda: self.remove_tool(key))
            row.addWidget(rb)
            dt = self.s.opts_seconds(p.opts) - self.s.opts_seconds(new)
            lost = [self.s.sliders[x]["name"] for x in self.s.sliders
                    if x in self.s.available_sliders(p.opts) and x not in self.s.available_sliders(new)]
            t = f"−{hm(dt)} de calcul" + (f" ; en post, tu perdrais : {', '.join(lost)}" if lost else "")
            row.addWidget(label(t + ".", "small", wrap=True), 1)
        v.addLayout(row)
        for n_ in (notes if new is not None else []):
            v.addWidget(label("• " + n_, "small", wrap=True))
        return w

    def slot_box(self, p):
        other = "simple" if p.opts["fond"] == "halpha" else "halpha"
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 2, 0, 4)
        row = QHBoxLayout()
        rb = QPushButton(f"Remplacer par : {TOOLS[other]['title']}")
        rb.clicked.connect(lambda: self.add_tool(other))
        row.addWidget(rb)
        row.addWidget(label(self.s.tool_status(p.opts, other)[1] + ".", "small", wrap=True), 1)
        v.addLayout(row)
        if p.opts["fond"] == "simple":
            v.addWidget(label("Fond simple : étape à créer dans le moteur (≈ 1 min).", "small", wrap=True))
        return w

    def calibration_box(self):
        if not self.s.sources_loaded:
            self.sl.addWidget(label("Aucune source pour l'instant.", "warn"))
        else:
            inv = self.s.data["inventory"]
            self.sl.addWidget(kv([("Poses", f"{inv['n_lights']}"), ("Darks", f"{inv['n_darks']}"),
                                  ("Flats", f"{inv['n_flats']}"), ("Offsets", "aucun"), ("Darks de flat", "aucun")]))
        sb = QPushButton("Voir les sources  →")
        sb.setObjectName("link")
        sb.clicked.connect(lambda: self.openSources.emit(-1))
        self.sl.addWidget(sb, 0, Qt.AlignLeft)

    def selection_settings(self):
        t, f = self.s.thresholds_now()
        st, sf = self.s.suggested_thresholds()
        sp = {q["name"]: q for q in self.s.stages["stack"]["params"]}
        g = label("Seuils", "dim")
        g.setStyleSheet("color: #aab2bf; font-weight: 600; margin-top: 8px;")
        self.sl.addWidget(g)
        self.sl.addWidget(hline())
        for name, val, sug, unit, lo, hi, step, which, reason in (
                ("Transparence minimale", t, st, "% des meilleures poses", 0, 100, 1, "t",
                 sp["min_transparency"]["reason"]),
                ("Taille maximale des étoiles", f, sf, "px", 1, 20, 0.1, "f", "2 × la taille médiane (4,0 px)")):
            w = QFrame()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 6, 0, 6)
            v.setSpacing(2)
            row = QHBoxLayout()
            n_ = QLabel(name)
            n_.setStyleSheet("font-weight: 600;")
            row.addWidget(n_, 1)
            e = QDoubleSpinBox()
            e.setDecimals(0 if which == "t" else 1)
            e.setRange(lo, hi)
            e.setSingleStep(step)
            e.setValue(val)
            e.setFixedWidth(110)
            e.editingFinished.connect(lambda ed=e, wh=which: self.s.set_selection(
                tmin=ed.value() if wh == "t" else None, fmax=ed.value() if wh == "f" else None))
            row.addWidget(e)
            row.addWidget(label(unit, "small"))
            v.addLayout(row)
            if abs(val - sug) > 1e-6:
                v.addWidget(label(f"modifié · suggéré : {fnum(sug, 0 if which == 't' else 1)}", "warn"))
            else:
                v.addWidget(label(f"suggéré : {reason}", "reason", wrap=True))
            self.sl.addWidget(w)
        self.sl.addWidget(label(f"Exclues à la main : {len(self.s.frames_excluded)}.", "small"))
        ready = self.s.state(None, "register") == "done"
        row = QHBoxLayout()
        gb = QPushButton("Régler sur le graphe des poses")
        gb.setEnabled(ready)
        gb.setToolTip("Onglet Résultat" if ready else "Le graphe demande l'alignement calculé")
        gb.clicked.connect(lambda: self.set_face(1))
        row.addWidget(gb)
        r = QPushButton("↺  Revenir à la sélection suggérée")
        r.clicked.connect(lambda: self.s.set_selection(st, sf, set()))
        row.addWidget(r)
        row.addStretch(1)
        self.sl.addLayout(row)

    # ------------------------------------------------------------ all settings
    def open_all(self):
        from allsettings import AllSettingsDialog
        d = AllSettingsDialog(self.s, self, self.window())
        self._all_dialog = d
        d.exec()
