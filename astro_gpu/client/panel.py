"""The panel of the selected block, on the right in Graph and Inspect.

One line per setting. What there is to know about a setting (its suggestion and why, its range, its use, a tip)
shows under the list when the setting is clicked or changed. When the block is shared by several workflows, the panel
says which ones a change goes to; a change that would throw away computed results offers Replace or Try as variant.
Blocks with several methods show them as cards: what each does, where it runs, how long on this machine.

The settings come from the model's cache of the engine's ``describe`` (the worker fills it): the panel shows what it has
and builds itself again when the rest arrives. The ◇ methods and settings (not in the engine yet) show in Spec mode
only, disabled.
"""
from __future__ import annotations

import html

from PySide6.QtCore import QLocale, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import texts
from ..core import workflow as wfl
from . import prefs
from .blocks import BY_KEY, STATE_WORD, block_state, block_title, fmt_int, stages_of  # noqa: F401
from .i18n import _, all_workflows, hm, language, number
from .model import NotInEngine, method_of, method_value, methods_of
from .settings import RangeBar, SettingRow, fmt, needed_stages, selection_rule_text, step_settings
from .settings import loading as settings_loading
from .widgets import ACCENT, BAD, NEW, WARN, clear, hline, label

STATE_COLOR = {"done": "#5fbf7f", "pending": "#8b93a1", "stale": WARN, "running": ACCENT, "failed": BAD,
               "interrupted": WARN, "skipped": "#8b93a1"}
OPTIONAL_BLOCKS = ("debruitage", "etoiles", "deconv")
EDITOR_W = 104                      # every editor of a row has this width...
BACK_W = 22                         # ...then the unit column, then the ↺ button (its room kept when hidden)
UNIT_MAX = 76                       # a longer unit wraps on two lines


def fnum(x, d=1) -> str:
    return number(x, d)


def kv(rows, pad=14) -> QLabel:
    """Two columns of text: a name and its value."""
    html = "<table cellspacing='0' cellpadding='3'>" + "".join(
        f"<tr><td style='color:#8b93a1; padding-right:{pad}px' valign='top'>{k}</td><td>{v}</td></tr>"
        for k, v in rows) + "</table>"
    lb = QLabel(html)
    lb.setTextFormat(Qt.RichText)
    lb.setWordWrap(True)
    return lb


def unit_column(units) -> int:
    """The width of the unit column of a panel: its longest unit (a longer one than UNIT_MAX wraps), 0 without any.
    The same width on every row puts every editor at the same x."""
    units = [u for u in units if u]
    if not units:
        return 0
    probe = label("", "small")
    probe.ensurePolished()
    return min(UNIT_MAX, max(probe.fontMetrics().horizontalAdvance(u) for u in units) + 6)


def unit_label(text: str, width: int) -> QLabel:
    u = label(text or "", "small")
    u.setFixedWidth(width)
    u.ensurePolished()
    u.setWordWrap(bool(text) and u.fontMetrics().horizontalAdvance(text) > width)
    u.setToolTip(text or "")
    return u


def name_label(text: str) -> QLabel:
    """The name of a setting: on two lines when the row has no room for it (long French names), never cut."""
    n = QLabel(text)
    n.setWordWrap(True)
    return n


def retain(w: QWidget, shown: bool) -> QWidget:
    """Hidden, the widget keeps its room, so that the columns after it do not move."""
    sp = w.sizePolicy()
    sp.setRetainSizeWhenHidden(True)
    w.setSizePolicy(sp)
    w.setVisible(shown)
    return w


def spec_mark(row_layout, new):
    """The ◇ mark of what the engine does not have yet: only in Spec mode."""
    if new and prefs.spec():
        mark = QLabel("◇")
        mark.setStyleSheet(f"color: {NEW};")
        mark.setToolTip(_("Not in the engine yet"))
        row_layout.addWidget(mark)


class CompactRow(SettingRow):
    """A setting on one line: name, value, back to the suggestion."""
    focused = Signal(dict)

    def __init__(self, r: dict, unit_w: int | None = None, parent=None):
        QFrame.__init__(self, parent)
        self.r = r
        if unit_w is None:
            unit_w = unit_column([r.get("unit")])
        self.setObjectName("row")
        self.setStyleSheet("QFrame#row { background: transparent; border-radius: 4px; }"
                           "QFrame#row:hover { background: #20242c; } QLabel { background: transparent; }")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        h = QHBoxLayout()
        h.setContentsMargins(4, 2, 2, 2)
        h.setSpacing(6)
        outer.addLayout(h)
        name = name_label(r["label"])
        name.setStyleSheet(f"color: {WARN};" if r["changed"] else "")
        h.addWidget(name, 1)
        spec_mark(h, r["kind"] == "new")
        ed = self._editor()
        back = r["changed"] and not r.get("locked")
        if isinstance(ed, QComboBox):                 # a choice: its own line, full width, never truncated
            ed.setToolTip(ed.currentText() if ed.isEnabled() else ed.toolTip())
            line = QHBoxLayout()
            line.setContentsMargins(4, 0, 2, 4)
            line.addWidget(ed, 1)
            line.addWidget(self._back(r, back))
            outer.addLayout(line)
            self.setToolTip(r.get("help") or "")
            return
        ed.setFixedWidth(EDITOR_W)
        h.addWidget(ed)
        h.addWidget(unit_label(r.get("unit") or "", unit_w))
        h.addWidget(self._back(r, back))
        self.setToolTip(r.get("help") or "")

    def _back(self, r, shown: bool) -> QPushButton:
        rb = QPushButton("↺")
        rb.setFixedSize(BACK_W, BACK_W)
        rb.setStyleSheet("padding: 0;")
        rb.setToolTip(_("Back to the suggested value: {v}").format(v=fmt(r["suggested"], r["type"], r.get("choices"))))
        rb.clicked.connect(lambda: self.changed.emit(self.r, None))
        return retain(rb, shown)

    def mousePressEvent(self, e):                       # noqa: N802 - Qt's name
        self.focused.emit(self.r)
        super().mousePressEvent(e)


def durations_text(times, here: str, estimated: bool = False) -> str:
    """The duration of a method on each device it runs on, as rich text: "GPU ≈ 8 min · CPU ≈ 23 h", the device this
    machine uses in bold. ``times``: {"gpu"|"cpu": (seconds, measured)}, or None while the worker has not read them.
    ``estimated``: every figure is an estimate (the spec's methods the engine does not have yet)."""
    if times is None:
        return "…"
    parts = []
    here = here if here in times else "cpu"                  # a method that only runs on the CPU runs there
    for dev in ("gpu", "cpu"):
        if dev not in times:
            continue
        sec, measured = times[dev]
        t = html.escape(f"{dev.upper()} {'' if measured and not estimated else '≈ '}{hm(sec)}")
        parts.append(f"<b style='color:#e6edf7'>{t}</b>" if dev == here else t)
    return " · ".join(parts)


class MethodCard(QFrame):
    """One way of computing a block: its name, what it does, how long it takes on each device it runs on (the device
    this machine uses in bold). ``times``: see ``durations_text``; ``here``: "gpu" or "cpu"."""
    clicked = Signal()

    def __init__(self, m, on, ok, times, here, why="", parent=None):
        super().__init__(parent)
        self.ok = ok
        self.setObjectName("meth")
        self.setCursor(Qt.PointingHandCursor if ok and not on else Qt.ArrowCursor)
        border = ACCENT if on else "#353b47"
        self.setStyleSheet(f"QFrame#meth {{ background: {'#2b3446' if on else '#20242c'}; border: 1px solid "
                           f"{border}; border-radius: 6px; }} QLabel {{ background: transparent; }}")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 6)
        v.setSpacing(1)
        n = QLabel(("● " if on else "○ ") + _(m["name"]))
        n.setWordWrap(True)
        n.setStyleSheet("font-weight: 600;" + ("" if ok else " color: #6b7280;"))
        v.addWidget(n)
        top = QHBoxLayout()
        top.setSpacing(6)
        spec_mark(top, m.get("new"))
        d = label(durations_text(times, here, bool(m.get("new"))), "small", wrap=True)
        d.setTextFormat(Qt.RichText)
        d.setToolTip(_("Duration on each device; in bold, the one this machine uses · ≈ an estimate"))
        top.addWidget(d, 1)
        v.addLayout(top)
        v.addWidget(label(_(m["what"]), "small", wrap=True))
        if why:
            v.addWidget(label(why, "warn", wrap=True))

    def mousePressEvent(self, e):                       # noqa: N802 - Qt's name
        if self.ok:
            self.clicked.emit()


class InfoDot(QLabel):
    """A small "i" in a circle: what it is about shows in its tooltip."""

    def __init__(self, tip: str, parent=None):
        super().__init__("i", parent)
        self.setObjectName("infodot")
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(15, 15)
        self.setStyleSheet("QLabel#infodot { color: #b9c2cf; border: 1px solid #6b7280; border-radius: 7px; "
                           "font-size: 8pt; font-weight: 600; font-family: Georgia, serif; padding: 0; }")
        self.setToolTip(tip)
        self.setCursor(Qt.WhatsThisCursor)


class StackMethodCard(QFrame):
    """A stacking method: its name, what it does; the engine's advice as a discreet "Recommended" outline in the accent
    colour, with an "i" whose tooltip says why. ``clicked`` asks for this method (the advice never chooses it)."""
    clicked = Signal()

    def __init__(self, key: str, on: bool, recommended: bool = False, why: str = "", enabled: bool = True,
                 parent=None):
        super().__init__(parent)
        from .settings import method_text
        self.key, self.on, self.recommended, self.ok = key, on, recommended, enabled and not on
        self.setObjectName("meth")
        self.setCursor(Qt.PointingHandCursor if self.ok else Qt.ArrowCursor)
        border = ACCENT if on else "#353b47"
        self.setStyleSheet(f"QFrame#meth {{ background: {'#2b3446' if on else '#20242c'}; border: 1px solid "
                           f"{border}; border-radius: 6px; }} QLabel {{ background: transparent; }}")
        name, what = method_text(key)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 6)
        v.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(6)
        n = QLabel(("● " if on else "○ ") + name)
        n.setStyleSheet("font-weight: 600;" + ("" if enabled else " color: #6b7280;"))
        top.addWidget(n)
        top.addStretch(1)
        self.tag = self.info = None
        if recommended:
            self.tag = QLabel(_("Recommended"))
            self.tag.setObjectName("rec")
            self.tag.setStyleSheet(f"QLabel#rec {{ color: {ACCENT}; border: 1px solid {ACCENT}; border-radius: 8px; "
                                   "padding: 0px 7px; font-size: 8.5pt; }")
            top.addWidget(self.tag)
            if why:
                self.info = InfoDot(why)
                top.addWidget(self.info)
        v.addLayout(top)
        if what:
            v.addWidget(label(what, "small", wrap=True))

    def mousePressEvent(self, e):                       # noqa: N802 - Qt's name
        if self.ok:
            self.clicked.emit()


class Details(QFrame):
    """Everything about one setting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("det")
        self.setStyleSheet("QFrame#det { background: #191c22; border-top: 1px solid #2a2f38; } "
                           "QLabel { background: transparent; }")
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(12, 10, 12, 10)
        self.v.setSpacing(4)
        self.hide()

    def show_setting(self, r):
        clear(self.v)
        self.v.addWidget(label(r["label"], "h2"))
        if r["type"] in ("int", "float") and r.get("min") is not None and r.get("max") is not None:
            self.v.addWidget(RangeBar(r["min"], r["max"], r["suggested"], r["value"]))
        sug = fmt(r["suggested"], r["type"], r.get("choices"))
        self.v.addWidget(label(_("Suggested: {v}").format(v=sug) + (f" · {r['reason']}" if r.get("reason") else ""),
                               "reason", wrap=True))
        if r.get("help"):
            self.v.addWidget(label(r["help"], "small", wrap=True))
        if r.get("tip"):
            t = label(r["tip"], "small", wrap=True)
            t.setStyleSheet("color: #b9c2cf; font-size: 9pt;")
            self.v.addWidget(t)
        if r.get("drift"):
            self.v.addWidget(label(_("The current result used {v}: the suggested value changed since.").format(
                v=fmt(r.get("result_value"), r["type"], r.get("choices"))), "warn", wrap=True))
        if r.get("locked"):
            self.v.addWidget(label(_("Set by the options of the workflow."), "small", wrap=True))
        if r["kind"] == "new" and prefs.spec():
            self.v.addWidget(label("◇ " + _("Not in the engine yet."), "small"))
        self.show()


class Panel(QWidget):
    changed = Signal()                  # a setting changed (the window rebuilds the graph)
    openTune = Signal()                 # "Browse frames" and the like
    openPost = Signal()
    addSources = Signal()
    addSession = Signal()
    renamed = Signal(int, str)
    removeTool = Signal()
    variantMade = Signal(int, int)      # new workflow, the one it is a variant of
    toast = Signal(str)                 # a word to say (what came with a change, why it was refused)

    def __init__(self, project, steps=None, tune=False, parent=None):
        super().__init__(parent)
        self.s, self.steps, self.tune = project, steps, tune
        self.expert = False
        self.focus_key = None
        self.node = None
        self._unit_w = 0                    # the unit column of the rows shown
        self.setObjectName("panel")
        self.setFixedWidth(390)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        head = QWidget()
        self.hv = QVBoxLayout(head)
        self.hv.setContentsMargins(14, 12, 14, 8)
        self.hv.setSpacing(6)
        v.addWidget(head)
        v.addWidget(hline())
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        self.bv = QVBoxLayout(host)
        self.bv.setContentsMargins(10, 8, 10, 8)
        self.bv.setSpacing(1)
        sc.setWidget(host)
        v.addWidget(sc, 1)
        self.details = Details()
        v.addWidget(self.details)
        self._later = QTimer(self)
        self._later.setSingleShot(True)
        self._later.setInterval(0)
        self._later.timeout.connect(self._rebuild)
        project.paramsChanged.connect(self._params_arrived)
        for sig in (project.selectionChanged, project.readyChanged):
            sig.connect(self._again)

    def _again(self, *_a):
        """What the worker was asked for arrived: build the panel again (once, whatever the count of answers)."""
        if self.node is not None:
            self._later.start()

    def _params_arrived(self, wid: int, stage: str):
        """Settings arrived: only the panel that shows them builds itself again."""
        node = self.node
        if node is None or node.key in ("sources", "final") or not node.paths:
            return
        if any(q.pid == wid for q in node.paths) and stage in needed_stages(self.s, node.paths[0], node.key):
            self._later.start()

    def _rebuild(self):
        """Not while a setting is being typed: the editor would go away under the user's fingers."""
        fw = QApplication.focusWidget()
        if fw is not None and self.isAncestorOf(fw) and isinstance(fw, QLineEdit | QAbstractSpinBox):
            self._later.start(400)
            return
        self.refresh(self.node)

    # ------------------------------------------------------------ build
    def refresh(self, node):
        clear(self.hv)
        clear(self.bv)
        self.details.hide()
        self.node = node
        if node is None:
            return
        k = node.key
        if k == "sources":
            self.sources()
        elif k == "final":
            self.hv.addWidget(label(_("Post"), "h1"))
            rows = []
            for q in self.s.paths:
                src = _("from {block}").format(block=block_title(q, q.end)) if q.end else ""
                st = _("Up to date") if q.rendered and q.computed else (_("Outdated") if q.rendered else _("Pending"))
                rows.append((html.escape(q.name), " · ".join(x for x in (st, src) if x)))
            if rows:
                self.bv.addWidget(self.group_title(_("Layers")))
                self.bv.addWidget(kv(rows, pad=10))
                self.bv.addSpacing(8)
                b = QPushButton(_("Open Post"))
                b.setObjectName("primary")
                b.setEnabled(any(q.rendered for q in self.s.paths))
                b.clicked.connect(self.openPost.emit)
                self.bv.addWidget(b, 0, Qt.AlignLeft)
            self.bv.addStretch(1)
        else:
            self.block(node)

    def sources(self):
        s = self.s
        self.hv.addWidget(label(_("Frames"), "h1"))
        if not s.sources_loaded:
            b = QPushButton(_("Import folder…"))
            b.setObjectName("primary")
            b.clicked.connect(self.addSources.emit)
            self.bv.addWidget(b, 0, Qt.AlignLeft)
            self.bv.addStretch(1)
            return
        self.bv.addWidget(self.group_title(_("Sessions")))
        for ses in s.sessions:
            darks = str(ses["darks"]) if ses["darks"] else _("none")
            self.bv.addWidget(kv([(html.escape(ses["night"]), _("{lights} lights · {flats} flats · darks: {darks}").format(
                lights=ses["lights"], flats=ses["flats"], darks=darks))], pad=10))
            if ses.get("rotation"):
                self.bv.addWidget(label(_("Camera rotated {deg}°: this session uses its own flats").format(
                    deg=fnum(ses["rotation"])), "warn", wrap=True))
        self.bv.addSpacing(6)
        self.bv.addWidget(kv([(_("Excluded by hand"), str(len(s.frames_excluded))),
                              (_("Stacking"), _("all sessions together") if s.stack_mode == "together" else
                               _("one stack per session"))]))
        self.bv.addSpacing(8)
        row = QHBoxLayout()
        if not self.tune:
            b = QPushButton(_("Open frames"))
            b.clicked.connect(self.openTune.emit)
            row.addWidget(b)
        a = QPushButton(_("Add session…"))
        a.clicked.connect(self.addSession.emit)
        row.addWidget(a)
        row.addStretch(1)
        self.bv.addLayout(row)
        self.bv.addStretch(1)

    def block(self, node):
        s = self.s
        p = s.path if s.path in node.paths else node.paths[0]
        k = node.key
        top = QHBoxLayout()
        if k == "post":
            ed = QLineEdit(p.name)
            ed.setStyleSheet("QLineEdit { font-size: 13pt; font-weight: 600; background: transparent; border: none; "
                             "border-bottom: 1px dashed #3a4150; border-radius: 0; } "
                             f"QLineEdit:focus {{ border-bottom: 1px solid {ACCENT}; }}")
            ed.setToolTip(_("Workflow name"))
            ed.editingFinished.connect(lambda: ed.text().strip() and ed.text().strip() != p.name
                                       and self.renamed.emit(p.pid, ed.text().strip()))
            top.addWidget(ed, 1)
        else:
            top.addWidget(label(block_title(p, k), "h1"))
            top.addStretch(1)
        st = self.graph_state(node)
        chip = QLabel(_(STATE_WORD[st]))
        chip.setStyleSheet(f"color: {STATE_COLOR[st]}; font-size: 9pt;")
        top.addWidget(chip)
        if k in OPTIONAL_BLOCKS:
            x = QPushButton(_("Remove"))
            x.clicked.connect(self.removeTool.emit)
            top.addWidget(x)
        self.hv.addLayout(top)
        # which workflows a change goes to
        if node.shared:
            self.hv.addWidget(label(_("Changes apply to:"), "dim"))
            box = QVBoxLayout()
            box.setSpacing(2)
            allp = [q.pid for q in node.paths]
            cur = s.scope if s.scope and set(s.scope) <= set(allp) else allp
            choices = [(all_workflows(len(node.paths)), allp)] + \
                      [(_('"{name}" only').format(name=q.name), [q.pid]) for q in node.paths]
            if len(choices) > 4:                       # many workflows: a list instead of a column of buttons
                cb = QComboBox()
                for text, pids in choices:
                    cb.addItem(text, pids)
                cb.setCurrentIndex(next((i for i, (_t, ps) in enumerate(choices) if sorted(ps) == sorted(cur)), 0))
                cb.currentIndexChanged.connect(lambda i: self._scope(cb.itemData(i)))
                box.addWidget(cb)
                choices = []
            for text, pids in choices:
                b = QPushButton(text)
                b.setObjectName("scope")
                b.setCheckable(True)
                b.setChecked(sorted(cur) == sorted(pids))
                b.setStyleSheet("QPushButton#scope { text-align: left; padding: 4px 10px; }"
                                f"QPushButton#scope:checked {{ background: #2b3446; border-color: {ACCENT}; "
                                "color: #ffffff; font-weight: 600; }")
                b.clicked.connect(lambda _x=False, ps=pids: self._scope(ps))
                box.addWidget(b)
            self.hv.addLayout(box)
        elif k != "post":
            self.hv.addWidget(label(p.name, "small"))
        self.variant_row(p)
        self.choice_box(node, p)
        if k == "fond" and prefs.spec():
            note = label("◇ " + _("Gradient before Color, from the stack and the catalogue: in a later version."),
                         "small", wrap=True)
            note.setStyleSheet(f"color: {NEW};")
            self.hv.addWidget(note)
        # measurements, in Inspect
        if self.tune and self.steps is not None and st != "pending":
            facts = self.steps.facts(p, k)
            if facts:
                self.bv.addWidget(self.group_title(_("Measurements")))
                self.bv.addWidget(kv(facts, pad=10))
                self.bv.addSpacing(6)
        if k in ("fond", "debruitage", "etoiles", "deconv"):
            self.methods(p, k)
        groups = self._groups(p, k)
        units = [r.get("unit") for _t, rs in groups for r in rs]          # hidden rows too: the column stays put
        if k == "selection":
            units += [_("% of the best"), "px", _("× median")]
        self._unit_w = unit_column(units)
        extra = 0
        if k == "selection":
            groups, extra = self.selection_rule(p, groups)
            self.selection(p)
        if k == "empilement" and len(s.sessions) > 1:
            self.stack_sessions()
        if k == "empilement":
            self.stack_methods(p)
        if k == "post" and p.end:
            self.bv.addWidget(kv([(_("Output of"), block_title(p, p.end))]))
        self.settings(p, k, groups, extra)
        self.bv.addStretch(1)

    def stack_sessions(self):
        """Several nights: one stack of all (one per night is ◇: the engine stacks all the sessions together)."""
        self.bv.addWidget(self.group_title(_("Sessions")))
        row = QHBoxLayout()
        for text, mode in ((_("All together"), "together"), (_("One per session"), "per_session")):
            if mode == "per_session" and not prefs.spec():
                continue
            b = QPushButton(text + ("  ◇" if mode == "per_session" else ""))
            b.setCheckable(True)
            b.setObjectName("face")
            b.setChecked(self.s.stack_mode == mode)
            b.setEnabled(mode == "together")
            if mode == "per_session":
                b.setToolTip("◇ " + _("Not in the engine yet."))
            row.addWidget(b)
        row.addStretch(1)
        self.bv.addLayout(row)

    # ------------------------------------------------------------ replace or try as a variant
    def variant_row(self, p):
        """A variant says what it is a variant of, and is kept or discarded from here."""
        src = self.s.view(p.variant_of) if p.variant_of is not None else None
        if src is None:
            return
        row = QHBoxLayout()
        row.addWidget(label(_('Variant of "{name}"').format(name=src.name), "dim"), 1)
        keep = QPushButton(_("Keep"))
        keep.setToolTip(_('Replaces "{name}" and takes its name').format(name=src.name))
        keep.clicked.connect(lambda: (self.s.keep_variant(p), self.changed.emit()))
        drop = QPushButton(_("Discard"))
        drop.clicked.connect(lambda: (self.s.drop_variant(p), self.changed.emit()))
        row.addWidget(keep)
        row.addWidget(drop)
        self.hv.addLayout(row)

    def choice_box(self, node, p):
        ch = self.s.pending_change
        if not ch or ch["step"] != node.key:
            return
        s = self.s
        paths = [q for q in s.paths if q.pid in ch["pids"]]
        f = QFrame()
        f.setObjectName("choice")
        f.setStyleSheet(f"QFrame#choice {{ background: #2a2316; border: 1px solid {WARN}; border-radius: 6px; }}"
                        " QLabel { background: transparent; }")
        v = QVBoxLayout(f)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)
        v.addWidget(label(f"<b>{html.escape(ch['what'])}</b>", wrap=True))
        c = s.cost_of_change(paths, ch)
        rep = QPushButton(_("Replace") + f"  ·  {'' if c.measured else '≈ '}{hm(c.seconds)}")
        rep.setToolTip(_("The current results of {names} are computed again").format(
            names=", ".join(q.name for q in paths)))
        rep.clicked.connect(self._replace)
        v.addWidget(rep)
        cv = s.cost_of_change([p], ch)
        var = QPushButton(_("Try as variant") + f"  ·  {'' if cv.measured else '≈ '}{hm(cv.seconds)}")
        var.setObjectName("primary")
        var.setToolTip(_('"{name}" stays as it is; the variant shares everything up to this block').format(
            name=p.name))
        var.clicked.connect(lambda: self._variant(p))
        v.addWidget(var)
        cancel = QPushButton(_("Cancel"))
        cancel.clicked.connect(self._cancel)
        v.addWidget(cancel, 0, Qt.AlignLeft)
        self.hv.addWidget(f)

    def propose(self, stage, name, value, suggested, what, short):
        """A change of setting: made at once when nothing computed is lost; otherwise Replace, or Try as variant."""
        self._propose(dict(kind="param", stage=stage, name=name, value=value, suggested=suggested), what, short)

    def propose_opt(self, key, value, what, short):
        """A change of option, such as a method."""
        self._propose(dict(kind="opt", key=key, value=value), what, short)

    def _propose(self, ch, what, short):
        s = self.s
        paths = s.scope_paths()
        try:
            if s.throws_away(paths, ch):
                s.pending_change = dict(ch, what=what, short=short, step=self.node.key, pids=[q.pid for q in paths])
            else:
                s.pending_change = None
                notes = s.apply_change(paths, ch)
                s.log("{what} · {names}", what=what, names=", ".join(q.name for q in paths))
                if notes:
                    self.toast.emit(" · ".join(_(n) for n in notes))
        except NotInEngine:
            s.pending_change = None
            self.toast.emit("◇ " + _("Not in the engine yet."))
        except (ValueError, wfl.OptionRefused) as e:
            s.pending_change = None
            self.toast.emit(_(str(e)))
        self.changed.emit()

    def _replace(self):
        ch, self.s.pending_change = self.s.pending_change, None
        paths = [q for q in self.s.paths if q.pid in ch["pids"]]
        try:
            notes = self.s.apply_change(paths, ch)
        except (ValueError, wfl.OptionRefused) as e:
            self.toast.emit(_(str(e)))
            self.changed.emit()
            return
        self.s.log("{what} · replaced in {names}", what=ch["what"], names=", ".join(q.name for q in paths))
        if notes:
            self.toast.emit(" · ".join(_(n) for n in notes))
        self.changed.emit()

    def _variant(self, p):
        ch, self.s.pending_change = self.s.pending_change, None
        try:
            q = self.s.add_variant(p, ch, f"{p.name} · {ch['short']}")
        except (ValueError, wfl.OptionRefused) as e:
            self.toast.emit(_(str(e)))
            self.changed.emit()
            return
        self.s.log('Variant "{name}" created', name=q.name)
        self.variantMade.emit(q.pid, p.pid)

    # ------------------------------------------------------------ the method of a block
    def methods(self, p, k):
        """The ways of computing this block, as cards: what it does, where it runs, how long on this machine."""
        s = self.s
        from .model import METHOD_OPT
        cur = method_of(p, k)
        self.bv.addWidget(self.group_title(_("Method")))
        opt = METHOD_OPT[k]
        spec = prefs.spec()
        for m in methods_of(k):
            if m["new"] and not spec:
                continue
            on = m["key"] == cur
            ok = (s.gpu or m["runs"] != "gpu") and not m["new"]
            times = s.method_times(p, k, m["key"])          # read by the worker: None until it has
            why = ""
            if m["new"]:
                why = "◇ " + _("Not in the engine yet.")
            elif not ok:
                t = hm(times["cpu"][0]) if times and "cpu" in times else "…"
                why = _("Needs an NVIDIA GPU (≈ {t} on the CPU).").format(t=t)
                if on:
                    why = _("Computed earlier on a GPU; running it again here takes ≈ {t}.").format(t=t)
            card = MethodCard(m, on, ok, times, "gpu" if s.gpu else "cpu", why)
            if not on and ok:
                card.clicked.connect(lambda mk=m["key"], nm=m["name"], sh=m["short"]: self.propose_opt(
                    opt, method_value(k, mk), _("Method: {name}").format(name=_(nm)), _(sh)))
            self.bv.addWidget(card)

    def stack_methods(self, p):
        """The stacking methods the engine offers, as cards; the engine's advice marks one "Recommended" (with its
        reason under an "i"), and never changes the method by itself. Below, only the settings of the method shown."""
        from .settings import STACK_METHOD, method_advice, method_text, stack_methods
        keys = stack_methods(self.s, p)
        if len(keys) < 2:
            return
        info = self.s.param_info(p, *STACK_METHOD)
        advice = method_advice(self.s, p)
        self.bv.addWidget(self.group_title(_("Method")))
        for key in keys:
            rec = advice is not None and advice[0] == key
            card = StackMethodCard(key, key == info.value, rec, advice[1] if rec else "", not info.locked)
            if card.ok:
                name = method_text(key)[0]
                card.clicked.connect(lambda k_=key, nm=name: self.propose(
                    STACK_METHOD[0], STACK_METHOD[1], k_, None, _("Stacking method: {name}").format(name=nm),
                    nm.lower()))
            self.bv.addWidget(card)

    def _cancel(self):
        self.s.pending_change = None
        self.changed.emit()

    def graph_state(self, node):
        states = {block_state(self.s, q, node.key) for q in node.paths}
        from .blocks import STATE_ORDER
        return next((w for w in STATE_ORDER if w in states), "pending")

    def _scope(self, pids):
        self.s.scope = list(pids)
        if self.s.path is None or self.s.path.pid not in pids:
            self.s.current = self.s.index_of(pids[0])
        self.changed.emit()

    def group_title(self, text):
        g = label(text, "dim")
        g.setStyleSheet("color: #aab2bf; font-weight: 600; margin-top: 8px; margin-bottom: 2px;")
        return g

    def selection_rule(self, p, groups):
        """The Selection block's top: the rule that decides which frames are left out, in plain words, and its setting
        rows. Returns the other groups, and how many Advanced rows were drawn here (for the Advanced settings count)."""
        title = _("Rule")
        rule = next((rs for t, rs in groups if t == title), [])
        rest = [(t, rs) for t, rs in groups if t != title]
        if not rule and settings_loading(self.s, p, "selection"):
            return groups, 0
        self.bv.addWidget(self.group_title(title))
        self.bv.addWidget(label(selection_rule_text(self.s, p), "small", wrap=True))
        for r in rule:
            if r["level"] == "normal" or (self.expert and r["level"] == "expert"):
                self.add_row(r)
        return rest, sum(r["level"] != "normal" for r in rule)

    def selection(self, p):
        s = self.s
        t, f = s.thresholds_now()
        px = s.thresholds_unit() == "px"
        live = s.thresholds_active(p)
        self.bv.addWidget(self.group_title(_("Thresholds")))
        if not live:
            why = label(_("These thresholds act only when the frame selection is “{rule}”.").format(
                rule=texts.choice_label("thresholds", language())), "small", wrap=True)
            why.setObjectName("thresholds_reason")
            self.bv.addWidget(why)
        for name, val, unit, lo, hi, step, which in (
                (_("Minimum transparency"), t, _("% of the best"), 0, 100, 1, "t"),
                (_("Maximum FWHM") if px else _("Maximum star size"), f, "px" if px else _("× median"),
                 1 if px else 1, 20 if px else 5, 0.1, "f")):
            row = QHBoxLayout()                        # the columns of a setting row (CompactRow)
            row.setContentsMargins(4, 2, 2, 2)
            row.setSpacing(6)
            row.addWidget(name_label(name), 1)
            e = QDoubleSpinBox()
            e.setLocale(QLocale())                     # a decimal comma in French
            e.setDecimals(0 if which == "t" else 1)
            e.setRange(lo, hi)
            e.setSingleStep(step)
            e.setValue(val)
            e.setFixedWidth(EDITOR_W)
            e.setEnabled(live)
            e.editingFinished.connect(lambda ed=e, wh=which: self._sel(ed.value(), wh))
            row.addWidget(e)
            row.addWidget(unit_label(unit, self._unit_w))
            room = QWidget()                               # where the ↺ of a setting row is
            room.setFixedSize(BACK_W, BACK_W)
            row.addWidget(room)
            self.bv.addLayout(row)
        s.request_selection()
        if s.state(p, "register") == "done":
            self.bv.addWidget(label(_("{kept} of {n} frames kept").format(kept=s.kept_frames(),
                                                                         n=s.selection_total()), "small"))
        else:
            self.bv.addWidget(label(_("The count appears once registration is computed."), "small", wrap=True))

    def _sel(self, value, which):
        t, f = self.s.thresholds_now()
        if which == "t" and abs(value - t) > 1e-9:
            self.propose_selection(value, f)
        elif which == "f" and abs(value - f) > 1e-9:
            self.propose_selection(t, value)

    def propose_selection(self, tmin, fmax):
        m = self.s.fwhm_median()
        ratio = round(fmax / m if m else fmax, 4)
        ch = dict(kind="params", items=[("stack", "min_transparency", round(tmin / 100, 4)),
                                        ("stack", "max_fwhm_ratio", ratio)])
        unit = "px" if m else "× median"
        self._propose(ch, _("Selection: transparency ≥ {t} %, FWHM ≤ {f} {unit}").format(
            t=f"{tmin:.0f}", f=fnum(fmax), unit=unit),
            _("selection {t} % / {f} {unit}").format(t=f"{tmin:.0f}", f=fnum(fmax), unit=unit))

    def _groups(self, p, k) -> list:
        """[(group title, [resolved settings])] of the block, with the half-stacks, the neural removal and the
        starless image where they belong."""
        s = self.s
        groups = list(step_settings(s, p, k))
        if k == "empilement" and p.opts["debruitage"]:
            groups += [(_("Half-stacks") if g == _("Half-stacks") else _("Half-stacks") + f" · {g}", rs)
                       for g, rs in step_settings(s, p, "demi_piles")]
        if k == "etoiles":
            if p.opts.get("appris"):
                groups += [(_("Neural removal") + f" · {g}", rs) for g, rs in step_settings(s, p, "appris")]
            groups += [(_("Starless image") + f" · {g}", rs) for g, rs in step_settings(s, p, "sans_etoiles")]
        return groups

    def settings(self, p, k, groups=None, extra_hidden=0):
        s = self.s
        groups = self._groups(p, k) if groups is None else groups
        if settings_loading(s, p, k) and not groups:
            err = next((s.params_error(p, st) for st in needed_stages(s, p, k) if s.params_error(p, st)), "")
            self.bv.addWidget(label(_("The settings cannot be read: {why}").format(why=err) if err else
                                    _("Loading the settings…"), "dim", wrap=True))
            return
        hidden = extra_hidden
        system = []
        for title, rs in groups:
            shown = [r for r in rs if r["level"] == "normal" or (self.expert and r["level"] == "expert")]
            hidden += sum(r["level"] != "normal" for r in rs)
            system += [r for r in rs if r["level"] == "system"]
            if not shown:
                continue
            self.bv.addWidget(self.group_title(title))
            for r in shown:
                self.add_row(r)
        if self.expert and system:
            self.bv.addWidget(self.group_title(_("System: speed and memory")))
            for r in system:
                self.add_row(r)
        if hidden:
            self.bv.addSpacing(8)
            ex = QCheckBox(_("Advanced settings ({n})").format(n=hidden))
            ex.setChecked(self.expert)
            ex.toggled.connect(self._expert)
            self.bv.addWidget(ex)

    def add_row(self, r):
        row = CompactRow(r, self._unit_w)
        row.changed.connect(self.set_setting)
        row.focused.connect(self._focus)
        self.bv.addWidget(row)
        if r["key"] == self.focus_key:
            self.details.show_setting(r)

    def _focus(self, r):
        self.focus_key = r["key"]
        self.details.show_setting(r)

    def _expert(self, on):
        self.expert = on
        self.refresh(self.node)

    def set_setting(self, r, value):
        """A setting was edited (value None: back to the suggestion)."""
        if r["kind"] == "new":
            return
        self.focus_key = r["key"]
        old = fmt(r["value"], r["type"], r.get("choices"))
        new = fmt(r["suggested"] if value is None else value, r["type"], r.get("choices"))
        self.propose(r["stage"], r["name"], value, r["suggested"],
                     _("{label}: {old} → {new}").format(label=r["label"], old=old, new=new),
                     f"{r['label'].lower()} {new}")
