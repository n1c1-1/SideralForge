"""Runs: the plan dialog, the progress of a run block by block, its recap, the progress drawn on the graph and the
status bar widget (the service that launches and follows a run is services/run.py).

The dialog has one row per block in graph order (the block, the workflows sharing it, the duration, a progress bar),
"Already up to date: N blocks", "Duration: t · disk: N GB", Cancel and Run. Once Run is pressed the same rows show the
run (a mark and a coloured bar per block), Close leaves the run going on (the progress stays in the status bar and on
the graph), and Stop ends it so that the next run goes on from there. At the end the rows keep their final state and a
recap says what was computed, what failed and which blocks were not run because of it, in plain words.

When the plan stops at a breakpoint ("Review pause") the dialog says so before the run ("Stops after
Calibration for your review"), and once the run is over a yellow box asks for the review: Continue (the results are
marked as checked, the run goes on to the next pause or the end) and Open in Inspect. With nothing to compute before
the review (the block is done already) the dialog opens on that box.

The dialog is not modal. ``RunDialog`` reads a ``RunTrack`` (services/run.py) through the controller and redraws when
it changes. ``RunStatus`` is the status bar widget, ``RunOverlay`` the progress drawn over the graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QEvent, QObject, QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .blocks import ENGINE_BLOCK, PAUSE_COL, Graph, Node, block_title, node_title
from .i18n import _, all_workflows, hm, plural
from .icons import IconText, draw, set_icon
from .services.run import (
    DONE,
    FAILED,
    INTERRUPTED,
    NOT_STARTED,
    RUNNING,
    SKIPPED,
    WAITING,
    JobState,
    RunTrack,
    duration_text,
    engine_text,
    pause_text,
    pause_titles,
    size_text,
    stage_title,
)
from .widgets import ACCENT, BAD, OK, WARN, Box, ElidedLabel, cap_dialog, clear, label

GREY = "#5d6470"
#: waiting: to compute, an orange circle (issue #27)
STATE_COLOR = {WAITING: WARN, RUNNING: ACCENT, DONE: OK, FAILED: BAD, INTERRUPTED: WARN, SKIPPED: GREY,
               NOT_STARTED: GREY}
STATE_ICON = {WAITING: ("waiting", "warning"), RUNNING: ("running", "selected"), DONE: ("ok", "ok"), FAILED: ("failed", "error"),
              INTERRUPTED: ("stop", "warning"), SKIPPED: ("skipped", "dim"), NOT_STARTED: ("skipped", "dim")}
STATE_TIP = {WAITING: "Waiting", RUNNING: "Running", DONE: "Done", FAILED: "Failed", INTERRUPTED: "Stopped",
             SKIPPED: "Not run", NOT_STARTED: "Not run"}
ROW_H = 28
WARN_NOTE = "⚠ "                  # a note of the plan that opens with it: drawn with the warning icon
#: why nothing can be computed yet (the status bar's words; exploration 6.7 and #36)
NO_DATA = "Nothing to compute: add lights or image"
ROWS_MAX_H = 440
#: the recap's lines scroll past this height (GUI exploration 4)
RECAP_MAX_H = 170


# ---------------------------------------------------------------------------------------------- the rows
@dataclass
class Row:
    """A line of the dialog: one block of the graph, the jobs (stage keys) the run computes in it."""

    title: str
    who: list[str]
    keys: list[str]
    block: str = ""                                          # the views' block key
    pids: set = field(default_factory=set)
    everyone: bool = False                                   # every workflow of the project shares it
    uptodate: bool = False                                   # nothing to compute in it (issue #27: shown, ticked)
    echo: bool = False                                       # Selection: it has no job, it follows its Registration

    def who_text(self) -> str:
        """The workflows sharing the block, as the panel's scope says them: "all 4 workflows", or their names."""
        if self.everyone and len(self.who) > 1:
            return all_workflows(len(self.who))
        return ", ".join(self.who)


def view_block(block: str) -> str | None:
    """The views' key of an engine block of a job: a graph's later block (issue #47) "gradient~2" → "fond~2", a
    session's "calibration@2" → "calibration@2" (exploration 6.1); None for a block the views do not have."""
    b, sep, n = block.partition("~")
    b, at, sid = b.partition("@")
    if b not in ENGINE_BLOCK:
        return None
    return ENGINE_BLOCK[b] + (f"~{n}" if sep else "") + (f"@{sid}" if at else "")


def _node(g, v, bk):
    """The node of the graph a job's block is, or None when the graph does not show it for that workflow (a block not
    placed: a session's Calibration nobody placed)."""
    if g is None or v is None or not bk:
        return None
    node = g.node_of(v, bk)
    if node is None and bk == "calibration":
        # a workflow of one session whose Calibration is drawn beside its session (a mono project's Ha, OIII: round
        # 2, they had no row): the engine computes it as the workflow's own
        node = next((n for pair in getattr(g, "sessions", []) for n in pair
                     if n.key == "calibration" and v.pid in {q.pid for q in n.paths}), None)
        return node
    return node if node is not None and v in node.paths else None


def row_title(model, node: Node, v) -> str:
    """A row's name: the block as the graph names it ("Gradient 2", the Split block that holds its Output, exploration
    4.1 and 6.2), a session's Calibration with its session ("Calibration · Session 2", 6.1), a workflow's Output with
    the workflow's name."""
    if node.key == "post" and not node.handle:
        return _('Output "{name}"').format(name=v.name if v is not None else "?")
    text = node_title(model, node)
    if node.sid is not None and node.key != "session":
        text += " · " + (model.session(node.sid).name or _("Session {n}").format(n=node.sid))
    return text


def build_rows(model, jobs, names: dict[int, str] | None = None, with_done: bool = False) -> list[Row]:
    """One row per block of the graph that has something to run, in the graph's order. ``jobs``: runner2.Job or
    JobState. The workflows named are those that share the result (``job.wf``). ``with_done``: the blocks already
    up to date too, in their place (``uptodate``; issue #27: the dialog said "3 blocks" without saying which)."""
    names = names or {}
    g = Graph(model) if model.paths else None
    order = {id(n): i for i, n in enumerate(g.nodes)} if g is not None else {}
    jobs = list(jobs)
    views = [model.view(j.run_wf) or next((model.view(w) for w in j.wf if model.view(w) is not None), None)
             for j in jobs]
    keys = [view_block(j.block) for j in jobs]
    nodes = [_node(g, v, bk) for v, bk in zip(views, keys)]
    # exploration 6.3: a block the engine computes but the graph does not show (Calibration and Registration not
    # placed) is part of the first later block of its workflow that the graph shows (Stacking), not a row of its own
    for i, j in enumerate(jobs):
        if nodes[i] is None and g is not None and views[i] is not None:
            nodes[i] = next((nodes[k] for k in range(i + 1, len(jobs))
                             if nodes[k] is not None and set(jobs[k].wf) & set(j.wf)), None)
    found: dict = {}
    for i, j in enumerate(jobs):
        v, bk, node = views[i], keys[i], nodes[i]
        rid = id(node) if node is not None else (j.block, tuple(sorted(j.wf)))
        if rid not in found:
            if node is not None:
                title = row_title(model, node, v)         # as the graph names it ("Gradient 1", left to right)
            elif bk == "post":
                title = _('Output "{name}"').format(name=v.name if v is not None else names.get(j.run_wf, "?"))
            elif bk:
                title = block_title(v, bk)
            else:
                title = stage_title(model, j.stage)
            pos = order.get(id(node), len(order) + i)
            found[rid] = (pos, Row(title, [], [], (node.ref if node is not None else bk) or "", set()))
        row = found[rid][1]
        row.keys.append(j.key)
        row.pids |= set(j.wf)
    if g is not None:
        # Selection has no job of its own: the Registration before it computes what it shows (round 2: placed after
        # Registration it had no row); its row follows that Registration's
        for nd in g.nodes:
            if nd.key != "selection" or id(nd) in found or nd.cut or not nd.paths:
                continue
            pids = {q.pid for q in nd.paths}
            reg = next((r for _p, r in found.values() if r.block == "alignement" and r.pids & pids), None)
            if reg is not None:
                found[id(nd)] = (order.get(id(nd), 0), Row(row_title(model, nd, nd.paths[0]), [], list(reg.keys),
                                                           nd.ref, set(reg.pids & pids), echo=True))
    if with_done and g is not None:
        for nd in g.nodes:
            # a Session block is not computed (exploration #10): never a row; nor a block switched off (round 2: it
            # said "up to date" though nothing is computed in it)
            if id(nd) in found or nd.key in ("sources", "final", "composition", "session") or not nd.paths or nd.cut:
                continue
            if g.state(nd) != "done":
                continue
            found[id(nd)] = (order.get(id(nd), 0), Row(row_title(model, nd, nd.paths[0]), [], [], nd.ref,
                                                         {q.pid for q in nd.paths}, uptodate=True))
    out = []
    for _pos, row in sorted(found.values(), key=lambda t: t[0]):
        ids = [p.pid for p in model.paths if p.pid in row.pids]
        ids += sorted(w for w in row.pids if w not in ids)
        row.who = [] if row.block == "post" else [(model.view(w).name if model.view(w) is not None
                                                   else names.get(w, f"#{w}")) for w in ids]
        row.everyone = bool(row.who) and {p.pid for p in model.paths} <= set(ids)
        out.append(row)
    return out


def row_state(states: list[str]) -> str:
    """The state of a block from the states of its jobs."""
    s = set(states)
    if not s:
        return WAITING
    for k in (RUNNING, FAILED, INTERRUPTED):
        if k in s:
            return k
    if s == {DONE}:
        return DONE
    if s <= {SKIPPED, NOT_STARTED}:
        return SKIPPED if SKIPPED in s else NOT_STARTED
    return RUNNING if DONE in s else WAITING


def refusal_text(problems: list[tuple[str, str]]) -> str:
    """Why Run is refused, in one line: the first workflow and its reason ("HOO: Ha, OIII: needs Astrometry"), and how
    many more."""
    if not problems:
        return ""
    name, why = problems[0]
    text = _("Run refused · {name}: {why}").format(name=name, why=why)
    return text + (" " + plural(len(problems) - 1, "(+{n} other)", "(+{n} others)") if len(problems) > 1 else "")


# ---------------------------------------------------------------------------------------------- widgets
class Mark(QWidget):
    """The state of a block as a small drawing: a tick, a cross, a dot, a square, a dash."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = WAITING
        self.setFixedSize(16, 16)

    def set_state(self, state: str):
        if state != self.state:
            self.state = state
            self.update()

    def paintEvent(self, e):                                      # noqa: N802
        p = QPainter(self)
        draw(p, QRectF(0, 0, 16, 16), *STATE_ICON[self.state])
        p.end()


class RunRow(QFrame):
    """A block: its state, name, the workflows sharing it, its duration, its progress bar."""

    def __init__(self, row: Row, parent=None):
        super().__init__(parent)
        self.row = row
        self.setFixedHeight(ROW_H)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        self.mark = Mark()
        h.addWidget(self.mark)
        self.name = ElidedLabel(row.title, mode=Qt.ElideRight)
        h.addWidget(self.name, 1)
        self.who = ElidedLabel(row.who_text(), "small", mode=Qt.ElideRight)
        self.who.setMaximumWidth(280)
        self.who.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.who.setToolTip("\n".join(row.who))
        h.addWidget(self.who)
        self.dur = label("", "small")
        self.dur.setFixedWidth(84)
        self.dur.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(self.dur)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedWidth(120)
        self.bar.setTextVisible(False)
        h.addWidget(self.bar)
        self._color = None

    def show_state(self, state: str, frac: float, text: str, tip: str):
        self.mark.set_state(state)
        self.bar.setValue(int(round(1000 * frac)))
        self.dur.setText(text)
        col = STATE_COLOR[state] if state != WAITING else ACCENT
        if col != self._color:
            self._color = col
            self.bar.setStyleSheet(f"QProgressBar::chunk {{ background: {col}; border-radius: 2px; }}")
        self.setToolTip(tip)
        faded = state in (SKIPPED, NOT_STARTED)
        self.name.setStyleSheet("color: #8b93a1;" if faded else "")


# ---------------------------------------------------------------------------------------------- the dialog
class RunDialog(QDialog):
    """The plan of a run, then the run, then its recap. ``plan``: a runner2.Plan; ``controller``: services/run's."""

    def __init__(self, model, plan, title="Run", show_done: bool = False, controller=None, parent=None,
                 track: RunTrack | None = None, names: dict[int, str] | None = None):
        super().__init__(parent)
        self.s, self.plan, self.c = model, plan, controller
        self.heading, self.show_done = title, show_done
        self.names = dict(names or {})
        self.setMinimumWidth(740)
        cap_dialog(self)
        self.setModal(False)
        self._track = track
        self.root = QVBoxLayout(self)
        self.body: QWidget | None = None
        self.rows: list[RunRow] = []
        self.rebuild()
        if controller is not None:
            controller.updated.connect(self.refresh)

    # ------------------------------------------------------------ what it shows
    @property
    def track(self) -> RunTrack | None:
        """The run this dialog is about: the controller's, once Run was pressed for this plan."""
        t = self.c.track if self.c is not None else self._track
        return t if t is not None and getattr(t, "plan", None) is self.plan else None

    def phase(self) -> str:
        t = self.track
        return "plan" if t is None else ("end" if t.finished else "run")

    def rebuild(self):
        """Build the dialog again (a change of language)."""
        if self.body is not None:
            self.root.removeWidget(self.body)
            self.body.hide()
            self.body.deleteLater()
        self.body = QWidget()
        self.root.addWidget(self.body)
        v = QVBoxLayout(self.body)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        head = self.heading() if callable(self.heading) else self.heading
        self.setWindowTitle(head)
        title = ElidedLabel(head, "h1", mode=Qt.ElideRight)          # a long name: "…" and the whole in its tooltip
        title.setMinimumWidth(0)
        v.addWidget(title)
        self.warn = QVBoxLayout()
        self.warn.setSpacing(2)
        v.addLayout(self.warn)
        self.rows_data = (build_rows(self.s, self.plan.jobs, self.names, with_done=self.show_done)
                          if self.plan is not None else [])
        self.inner = QWidget()
        iv = QVBoxLayout(self.inner)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.setSpacing(0)
        self.rows = []
        for r in self.rows_data:
            w = RunRow(r)
            iv.addWidget(w)
            self.rows.append(w)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.inner)
        h = min(len(self.rows) * ROW_H + 2, ROWS_MAX_H)
        self.scroll.setFixedHeight(h)
        self.scroll.setVisible(bool(self.rows))
        v.addWidget(self.scroll)
        self.done_note = label("", "small")
        self.done_note.setStyleSheet(f"color: {OK}; font-size: 9pt;")
        v.addWidget(self.done_note)
        self.pause_note = label("", None, wrap=True)                   # "Stops after Calibration for your review"
        self.pause_note.setStyleSheet(f"color: {PAUSE_COL};")
        v.addWidget(self.pause_note)
        self.pause_more = label("", "small", wrap=True)                # which workflows, which pauses come after
        v.addWidget(self.pause_more)
        self.pause_off = label("", "small", wrap=True)                 # pauses on blocks switched off: no stop
        v.addWidget(self.pause_off)
        self.pause_off.hide()
        self.hint = label("", "small", wrap=True)
        v.addWidget(self.hint)
        self.recap = Box(OK)
        # its lines scroll past a few lines (GUI exploration 4: 72 blocks not started ran out of the box, cut mid-name)
        self.recap_area = QScrollArea()
        self.recap_area.setWidgetResizable(True)
        self.recap_area.setFrameShape(QFrame.NoFrame)
        self.recap_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.recap_area.setMaximumHeight(RECAP_MAX_H)
        self.recap_area.setStyleSheet("QScrollArea { background: transparent; }")
        lines = QWidget()
        lines.setObjectName("recaplines")
        lines.setStyleSheet("QWidget#recaplines { background: transparent; }")
        self.recap_area.viewport().setAutoFillBackground(False)
        self.recap_lines = QVBoxLayout(lines)
        self.recap_lines.setContentsMargins(0, 0, 0, 0)
        self.recap_lines.setSpacing(6)
        self.recap_area.setWidget(lines)
        self.recap.v.addWidget(self.recap_area)
        self.recap.hide()
        v.addWidget(self.recap)
        self.review = Box(PAUSE_COL)                                   # the pause: what to do now
        self.review_text = label("", None, wrap=True)
        self.review_more = label("", "small", wrap=True)
        self.review.v.addWidget(self.review_text)
        self.review.v.addWidget(self.review_more)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.b_inspect = QPushButton(_("Open in Inspect"))
        self.b_inspect.clicked.connect(self._inspect)
        self.b_continue = QPushButton(_("Continue"))
        self.b_continue.setObjectName("primary")
        self.b_continue.clicked.connect(self._continue)
        buttons.addWidget(self.b_inspect)
        buttons.addWidget(self.b_continue)
        self.review.v.addLayout(buttons)
        self.review.hide()
        v.addWidget(self.review)
        v.addSpacing(8)
        foot = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(1)
        self.summary = label("", "h2")
        self.detail = label("", "small")
        col.addWidget(self.summary)
        col.addWidget(self.detail)
        foot.addLayout(col, 1)
        self.b_log = QPushButton(_("Show log"))
        self.b_log.setObjectName("link")
        self.b_log.clicked.connect(lambda: self.c.logRequested.emit() if self.c is not None else None)
        foot.addWidget(self.b_log)
        foot.addSpacing(8)
        self.b_cancel = QPushButton(_("Cancel"))
        self.b_cancel.clicked.connect(self.close)
        self.b_stop = set_icon(QPushButton(_("Stop")), "stop", 16)
        self.b_stop.clicked.connect(self._stop)
        self.b_run = set_icon(QPushButton(_("Run")), "run", 16, "on_accent")
        self.b_run.setObjectName("primary")
        self.b_run.clicked.connect(self.start)
        self.b_done = QPushButton(_("Close"))
        self.b_done.setObjectName("primary")
        self.b_done.clicked.connect(self.close)
        for b in (self.b_cancel, self.b_stop, self.b_run, self.b_done):
            foot.addWidget(b)
        v.addLayout(foot)
        self.adjustSize()
        self.refresh()

    # ------------------------------------------------------------ actions
    def start(self):
        """Run: the plan goes to the service (a worker launches the process)."""
        if self.c is None or not self.plan:
            return
        if self.c.begin(self.plan):
            self.refresh()

    def _stop(self):
        if self.c is not None:
            self.c.stop()
        self.refresh()

    def _continue(self):
        """Continue: the review is done, the run goes on (the controller makes the next plan and starts it)."""
        if self.c is not None:
            self.c.continue_run()

    def _inspect(self):
        if self.c is not None:
            self.c.open_in_inspect()

    # ------------------------------------------------------------ the pause
    @property
    def pause(self):
        """The review this dialog's plan stops at (model.Pause), or None."""
        return self.c.pause if self.c is not None and self.c.plan is self.plan else None

    def paused(self) -> bool:
        """Whether the dialog asks for a review now: the run ended at its breakpoint, or the block is computed already
        and there is nothing to run before it."""
        pa, t = self.pause, self.track
        if pa is None:
            return False
        if t is None:
            return not (self.plan is not None and self.plan.jobs)
        return t.finished and t.status == "done"

    def _pause_lines(self) -> tuple[str, str]:
        """(what the plan says about the pause, its second line): the block it stops after; the workflows that pause
        here when not all of them do, and the later pauses."""
        pa = self.pause
        if pa is None:
            return "", ""
        stops = pa.all_stops()
        pids = [pid for pid, _k in stops]
        # each later pause under its own workflow when the run stops in several (GUI exploration: the other
        # workflow's pause was listed as "after")
        named = len({k for _p, k in stops}) > 1 or any(pid not in pids for pid, _k in pa.later)
        later = pause_titles(self.s, None, [tuple(x) for x in pa.later], named=named)   # the same block in two
        more = plural(len(later), "+{n} pause after: {blocks}", "+{n} pauses after: {blocks}",   # workflows: once
                      blocks=", ".join(later)) if later else ""
        one = len({k for _p, k in stops}) == 1
        names = [q.name for q in self.s.paths if q.pid in pids]
        who = plural(len(names), "Workflow: {names}", "Workflows: {names}", names=", ".join(names)) \
            if one and names and len(names) < len(self.s.paths) else ""
        held = [q.name for q in self.s.paths if q.pid in (pa.waiting or [])]
        wait = _("Waiting for the review: {names}").format(names=", ".join(held)) if held else ""
        return (_("Review pause after {block}").format(block=pause_text(self.s, pa)),
                " · ".join(x for x in (who, more, wait) if x))

    def _off_pauses(self) -> str:
        """The pauses this run goes through because their block is switched off: kept, they wait for it to be on again
        (GUI exploration round 3: such a pause was announced in the graph and the run passed it silently)."""
        waiting = getattr(self.s, "waiting_breakpoints", None)
        if waiting is None:
            return ""
        scope = getattr(self.c, "scope", None) if self.c is not None else None
        paths = [q for q in self.s.paths if scope is None or q.pid in scope]
        stops = waiting(paths)
        if not stops:
            return ""
        return _("No stop at {blocks}: switched off, the pause waits for it to be on again (D)").format(
            blocks=pause_text(self.s, None, stops))

    def finish_now(self):
        """For scripted walkthroughs: wait until the run is over."""
        import time

        from PySide6.QtWidgets import QApplication
        end = time.time() + 600
        while self.phase() != "end" and time.time() < end:
            QApplication.processEvents()
            time.sleep(0.02)

    # ------------------------------------------------------------ redraw
    def _rows_state(self):
        t = self.track
        data = self.has_data()
        for w in self.rows:
            r = w.row
            if r.uptodate:                                   # nothing to compute: a green tick
                w.show_state(DONE, 1.0, _("up to date"), _("All up to date"))
                continue
            jobs = [t.by_key[k] for k in r.keys if t is not None and k in t.by_key] if t is not None else []
            plan_jobs = {j.key: j for j in self.plan.jobs}
            est = sum(plan_jobs[k].est_s for k in r.keys if k in plan_jobs)
            measured = all(plan_jobs[k].measured for k in r.keys if k in plan_jobs)
            if not jobs:
                # the block and its time, never the engine's stages (exploration #30); no time without data (6.7)
                text = duration_text(est, measured) if est and data and not r.echo else ""
                tip = _("{block}: {t}").format(block=r.title, t=text) if text else r.title
                w.show_state(WAITING, 0.0, text, tip)
                continue
            state = row_state([j.state for j in jobs])
            tot = sum(RunTrack.weight(j) for j in jobs)
            frac = sum(RunTrack.weight(j) * t.share(j, True) for j in jobs) / tot
            if r.echo:                                       # Selection: its Registration's state, no time of its own
                w.show_state(state, frac, "", self._tip(jobs, state))
                continue
            if state == DONE:
                real = sum(j.duration or 0.0 for j in jobs)
                text = hm(real) if any(not j.reused for j in jobs) else "—"
            elif state in (SKIPPED, NOT_STARTED):
                text = "—"
            else:
                text = duration_text(est, measured) if est else ""
            w.show_state(state, frac, text, self._tip(jobs, state))

    def _tip(self, jobs: list[JobState], state: str) -> str:
        """The state of a block, and why it failed: in the block's words, not the engine's stages (exploration #30)."""
        lines = [_(STATE_TIP[state])]
        for j in jobs:
            if j.state == FAILED and j.error:
                lines.append(engine_text(j.error.strip().splitlines()[0][:300]))
        return "\n".join(lines)

    def refused(self) -> list[tuple[str, str]]:
        """(workflow, why) of what refuses this plan (the controller's ``refusal``)."""
        if self.c is not None and hasattr(self.c, "refusal"):
            return self.c.refusal(self.plan)
        return getattr(self.s, "run_problems", lambda: [])()

    def no_data_text(self) -> str:
        """Why nothing can be computed: the workflows without their data named (round 2: "add lights or image" while
        a channel had its image and the other had none)."""
        if self.plan is not None and self.plan.jobs:
            views = {j.run_wf: self.s.view(j.run_wf) for j in self.plan.jobs}
            names = [v.name for v in views.values() if v is not None and not self.s.has_data(v)]
            if names and len(names) < len(self.s.paths):
                return _("No data in {names}: add lights or image").format(names=", ".join(names))
        return _(NO_DATA)

    def has_data(self) -> bool:
        """Whether the workflows of the plan have their data (lights, or the Image's file): no time is quoted and Run is
        greyed without it (exploration 6.7)."""
        if self.plan is None or not self.plan.jobs:
            return self.s.has_data()
        views = {j.run_wf: self.s.view(j.run_wf) for j in self.plan.jobs}
        views = [v for v in views.values() if v is not None]
        return all(self.s.has_data(v) for v in views) if views else self.s.has_data()

    def _file_warnings(self) -> list[str]:
        """The files that may be mistakes in the sessions the plan reads (issue #58, exploration 6.10: said on the
        Session block and in its panel, and before the run)."""
        if self.plan is None or not hasattr(self.s, "file_warnings"):
            return []
        from .frames import type_warning_text
        wids = {w for j in self.plan.jobs for w in j.wf}
        sids: list[int] = []
        for s in self.s.p.sessions:
            if any(self.s.view(w) is not None and self.s.view(w).source is None and
                   s in self.s.p.sessions_of(w) for w in wids) and s.id not in sids:
                sids.append(s.id)
        out = [_("{session}: {text}").format(session=self.s.p.session(sid).name, text=type_warning_text(w))
               for sid in sids for w in self.s.file_warnings(sid)]
        from .frames import colour_note_text
        notes = {sid: getattr(self.s, "colour_note", lambda _i: None)(sid) for sid in sids}    # issue #57: said before
        return out + [_("{session}: {text}").format(session=self.s.p.session(sid).name, text=colour_note_text(n))
                      for sid, n in notes.items() if n]

    def warnings(self, sync: bool = False) -> list[str]:
        """What the plan says before it runs: the workflows left out and why, the disk too small, the files that may be
        mistakes. Run (F5) opens this window first when there is one (GUI exploration round 2: they were never seen).
        ``sync``: the drive's free room is read now (it is asked to the worker otherwise)."""
        out = []
        for wid, why in (getattr(self.plan, "skipped", None) or {}).items():
            v = self.s.view(wid)
            out.append(_('"{name}" not run: {why}').format(
                name=v.name if v is not None else self.names.get(wid, "?"), why=_(why)))
        if self.plan is not None and self.plan.jobs and self.has_data():      # no disk need without data (6.7)
            free = self.c.free_gb() if self.c is not None else None
            if free is None and sync and self.c is not None:
                try:
                    from .services.run import _free_job
                    free = self.c._free = _free_job(str(self.s.p.root))
                except (OSError, AttributeError, TypeError):
                    free = None
            need = self.plan.disk_gb
            if free is not None and need > free:
                out.append(_("Disk: {need} GB needed, {free} GB free").format(need=size_text(need), free=size_text(free)))
        out += [WARN_NOTE + text for text in self._file_warnings()]
        return out

    def set_plan(self, plan):
        """The graph changed while the window was open: the new plan, drawn again (the controller made it)."""
        self.plan = plan
        self.rebuild()

    def refresh(self, *_a):
        if self.body is None:
            return
        ph, t = self.phase(), self.track
        self._rows_state()
        n = len(self.plan.jobs) if self.plan is not None else 0
        # notes of the plan
        clear(self.warn)
        if ph == "plan":
            for text in self.warnings():
                self.warn.addWidget(IconText("warning", text[len(WARN_NOTE):], "warning", "warn") if text.startswith(WARN_NOTE)
                                    else label(text, "warn", wrap=True))
                self.warn.itemAt(self.warn.count() - 1).widget().show()
        data = self.has_data()
        paused = self.paused()
        stops, more = self._pause_lines()
        self.pause_note.setVisible(bool(stops) and ph in ("plan", "run") and not paused)
        self.pause_note.setText(stops)
        self.pause_more.setVisible(bool(stops) and bool(more) and ph in ("plan", "run") and not paused)
        self.pause_more.setText(more)
        off = self._off_pauses() if ph == "plan" and not paused else ""
        self.pause_off.setVisible(bool(off))
        self.pause_off.setText(off)
        self.review.setVisible(paused)
        if paused:
            self.review_text.setText(_("Review pause: {block} · Inspect, then Continue").format(
                block=pause_text(self.s, self.pause)))
            self.review_more.setText(more or _("Continue goes on with the run."))
            self.b_continue.setEnabled(self.c is not None and not self.c.running)
        # the notes under the rows
        n_done = 0                                           # the up-to-date blocks are rows of their own (#27)
        self.done_note.setText(_("Up to date: {n} blocks").format(n=n_done) if n_done else "")
        self.done_note.setVisible(bool(n_done))
        self.hint.setVisible(ph == "run")
        self.hint.setText(_("Closing does not stop the run"))
        self.recap.setVisible(ph == "end")
        if ph == "end":
            self._fill_recap(t)
        # the footer
        self.summary.setStyleSheet("")
        self.detail.setText("")
        if ph == "plan" and paused:
            self.summary.setText(_("Awaiting review"))
            self.summary.setStyleSheet(f"color: {PAUSE_COL};")
        elif ph == "plan" and n and not data:                # no figures without data, and why (exploration 6.7)
            self.summary.setText(_("Nothing to compute"))
            self.detail.setText(self.no_data_text())
        elif ph == "plan":
            tot = self.plan.seconds if self.plan is not None else 0.0
            self.summary.setText(_("Duration: {t} · disk: {gb} GB").format(
                t=duration_text(tot, self.plan.measured), gb=size_text(self.plan.disk_gb)) if n else _("Nothing to run"))
        elif ph == "run":
            self._fill_progress(t)
        else:
            self._fill_end(t)
        stopping = self.c is not None and self.c.phase == "stopping"
        self.b_cancel.setText(_("Cancel") if ph == "plan" and not paused else _("Close"))
        self.b_cancel.setVisible(ph != "end")
        self.b_stop.setVisible(ph == "run")
        self.b_stop.setEnabled(self.c is not None and self.c.phase in ("starting", "running"))   # starting too (round 2)
        self.b_stop.setText(_("Stopping…") if stopping else _("Stop"))
        self.b_run.setVisible(ph == "plan" and not paused)
        # what refuses the plan (a fusion, data that cannot be computed: 1.1), among the workflows it computes; a plan
        # that could not be made (round 2: "Nothing to run" hid it)
        refused = self.refused() if ph == "plan" and not paused else []
        error = self.c.plan_error if self.c is not None and self.c.plan is self.plan and ph == "plan" else ""
        if (refused or error) and ph == "plan" and not paused:
            self.summary.setText(error or refusal_text(refused))
            self.summary.setStyleSheet(f"color: {BAD};")
            self.detail.setText("")
        busy = self.c is not None and self.c.running
        self.b_run.setEnabled(bool(n) and data and not refused and not error and not busy)
        # a greyed Run says why (exploration 6.7)
        self.b_run.setToolTip("" if self.b_run.isEnabled() else
                              _("A run is in progress") if busy else error if error else
                              refusal_text(refused) if refused else self.no_data_text() if n and not data else
                              _("Nothing to run: what this plan needs is up to date"))
        self.b_stop.setToolTip("" if self.b_stop.isEnabled() else _("Stopping…") if stopping else "")
        # Enter: the button that leads (round 2: the default one was a hidden "Open in Inspect")
        lead = (self.b_continue if paused and ph != "run" else self.b_run if ph == "plan" else
                self.b_done if ph == "end" else self.b_cancel)
        for b in (self.b_inspect, self.b_continue, self.b_cancel, self.b_stop, self.b_run, self.b_done):
            b.setAutoDefault(False)
            b.setDefault(b is lead)
        self.b_done.setVisible(ph == "end")
        self.b_log.setVisible(ph == "end" and t is not None and (t.count()[FAILED] > 0 or t.lost or t.status == "failed"))

    def _up_to_date(self) -> int:
        """The blocks of the graph that are done ("Already up to date")."""
        if not self.s.paths:
            return 0
        g = Graph(self.s)
        return sum(1 for nd in g.nodes if nd.key not in ("sources", "final") and nd.paths and g.state(nd) == "done")

    def _fill_progress(self, t: RunTrack):
        if self.c is not None and self.c.phase == "starting" or not t.running:
            self.summary.setText(_("Starting…"))
            return
        elapsed, left = t.elapsed(), t.remaining()
        self.summary.setText(_("Running · {t}").format(t=hm(elapsed)) + (
            " · " + _("≈ {t} left").format(t=hm(left)) if left > 30 else ""))
        j = t.current()
        if j is not None:
            pct = int(round(100 * j.frac))
            self.detail.setText(f"{stage_title(self.s, j.stage)} · {pct} %" +
                                (f" · {engine_text(j.msg)}" if j.msg else ""))

    def _fill_end(self, t: RunTrack):
        color = {"done": OK, "failed": BAD, "interrupted": WARN}[t.status]
        text = {"done": _("Done in {t}").format(t=hm(t.elapsed())),
                "failed": _("Failed"), "interrupted": _("Run stopped")}[t.status]
        if t.never_started:                                  # 6.4: it did not fail, it never began
            text = _("Run not started")
        if self.paused():
            color, text = PAUSE_COL, _("Review pause")
        self.summary.setText(text)
        self.summary.setStyleSheet(f"color: {color};")

    # ------------------------------------------------------------ the recap
    def _fill_recap(self, t: RunTrack):
        """What happened, in plain words: computed, failed, not run because of what, stopped."""
        box = self.recap
        while self.recap_lines.count():
            it = self.recap_lines.takeAt(0)
            if it.widget() is not None:                    # hidden at once: a rebuild never shows it twice (#35)
                it.widget().hide()
                it.widget().deleteLater()
        c = t.counts or t.count()
        lines: list[tuple[str, str]] = []                          # (text, colour)
        by_row = [(r, [t.by_key[k] for k in r.keys if k in t.by_key]) for r in self.rows_data if not r.echo]
        failed = [(r, js) for r, js in by_row if any(j.state == FAILED for j in js)]
        stopped = [r for r, js in by_row if any(j.state == INTERRUPTED for j in js)]
        notrun = [(r, js) for r, js in by_row if js and all(j.state in (SKIPPED, NOT_STARTED) for j in js)]
        col = {"done": OK, "interrupted": WARN}.get(t.status, BAD)
        named = self._named()
        if t.status == "done":
            lines.append((plural(t.block_count()["computed"], "{n} block computed · {t}",
                                 "{n} blocks computed · {t}", t=hm(t.elapsed())), OK))
        elif t.never_started:                              # 6.4: why, in plain words (the details are in the Log)
            from .services.run import start_failure
            lines.append((_("Run not started: {why}").format(why=_(start_failure(t.error))), BAD))
            names = [v.name for v in self.s.paths if v.rendered and v.pid in self._plan_wf()]
            if names:
                lines.append((_("Ready in Post: {names}").format(names=", ".join(names)), ""))
        elif t.lost:
            lines.append((_("Run ended unexpectedly · results kept"), WARN))
        elif t.status == "interrupted":
            lines.append((_("Stopped · results kept"), WARN))
            if stopped:
                lines.append((_("Stopped during {blocks}").format(blocks=", ".join(named[id(r)] for r in stopped)), ""))
        listed = set()
        for r, js in failed:
            bad = next(j for j in js if j.state == FAILED)
            why = (bad.error or t.error or "").strip()
            why = why.splitlines()[0][:240] if why else ""
            lines.append((_("{block} failed: {why}").format(block=named[id(r)], why=engine_text(why)) if why
                          else _("{block} failed.").format(block=named[id(r)]), BAD))
            blocked = [x for x, jx in notrun if any(bad.key in {k.key for k in t.culprits(j)}
                                                      for j in jx if j.state == SKIPPED)]
            listed |= {id(x) for x in blocked}
            if blocked:
                lines.append((_("Not run ({block} failed): {blocks}").format(
                    block=named[id(r)], blocks=", ".join(named[id(x)] for x in blocked)), ""))
        other = [r for r, js in notrun if id(r) not in listed]
        if other and t.status != "done":
            started = any(j.state == SKIPPED for r, js in notrun if r in other for j in js)
            lines.append(((_("Not run (earlier failure): {blocks}") if started
                           else _("Not started: {blocks}")).format(blocks=", ".join(named[id(r)] for r in other)), ""))
        if failed and c[DONE]:
            lines.append((_("Independent blocks computed"), ""))
        if t.status == "failed" and not failed and t.error and not t.never_started:
            from .services.run import start_failure
            lines.append((_(start_failure(t.error)), BAD))     # never the worker's raw words (6.4)
        box.setStyleSheet(f"QFrame#box {{ background: #23211c; border: 1px solid {col}; border-radius: 6px; }}"
                          "QFrame#box QLabel { background: transparent; }")
        for text, color in lines:
            lb = label(text, None, wrap=True)
            if color:
                lb.setStyleSheet(f"color: {color};")
            self.recap_lines.addWidget(lb)
            lb.show()
        # as tall as its lines at the dialog's width, up to RECAP_MAX_H; past it, a scroll bar
        width = max(self.recap_area.viewport().width(), self.width() - 60, 200)
        h = self.recap_lines.totalHeightForWidth(width)
        self.recap_area.setMinimumHeight(min(h if h > 0 else self.recap_lines.sizeHint().height(), RECAP_MAX_H))

    def _named(self) -> dict[int, str]:
        """The rows' names in the recap: a name two rows share says its workflows (#34: "Stacking, Stacking")."""
        titles = [r.title for r in self.rows_data]
        return {id(r): (f"{r.title} ({r.who_text()})" if titles.count(r.title) > 1 and r.who else r.title)
                for r in self.rows_data}

    def _plan_wf(self) -> set:
        out = set()
        for j in self.plan.jobs:
            out |= set(j.wf)
        return out

    def closeEvent(self, e):                                       # noqa: N802
        if self.c is not None and self.phase() == "plan":              # a plan nobody ran: forgotten
            if self.c.dialog is self:
                self.c.dialog = None
            self.deleteLater()
        super().closeEvent(e)


# ---------------------------------------------------------------------------------------------- the status bar
class RunStatus(QFrame):
    """What a run in progress looks like in the status bar: its state, a progress bar, Details and Stop. Hidden when
    no run is going."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.c = controller
        self.setObjectName("runstatus")
        self.h = QHBoxLayout(self)
        self.h.setContentsMargins(0, 0, 8, 0)
        self.h.setSpacing(8)
        self._build()
        controller.updated.connect(self.refresh)
        self.refresh()

    def _build(self):
        clear(self.h)
        self.text = label("", "small")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedSize(120, 10)
        self.bar.setTextVisible(False)
        self.b_details = QPushButton(_("Details"))
        self.b_details.setObjectName("link")
        self.b_details.clicked.connect(self.c.show_dialog)
        self.b_stop = set_icon(QPushButton(_("Stop")), "stop", 16)
        self.b_stop.setStyleSheet("padding: 1px 10px;")
        self.b_stop.clicked.connect(self.c.stop)
        for w in (self.text, self.bar, self.b_details, self.b_stop):
            self.h.addWidget(w)

    def rebuild(self):
        self._build()
        self.refresh()

    def refresh(self, *_a):
        on = self.c.running
        self.setVisible(on)
        if on:
            self.text.setText(self.c.status_text())
            self.bar.setValue(int(1000 * self.c.progress()))
            self.b_stop.setEnabled(self.c.phase in ("starting", "running"))     # starting too: done once started
            self.b_stop.setToolTip(_("Stop the run: what is computed is kept") if self.b_stop.isEnabled()
                                   else _("Stopping…"))


# ---------------------------------------------------------------------------------------------- on the graph
class RunOverlay(QWidget):
    """The progress of the running blocks, drawn over the graph: a thin bar along the foot of each block. It follows
    the view (zoom, scroll, rebuilds) and lets every click through. The graph itself is untouched."""

    def __init__(self, view, controller):
        super().__init__(view)
        self.view, self.c = view, controller
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        view.installEventFilter(self)
        controller.updated.connect(self.update)
        view.horizontalScrollBar().valueChanged.connect(lambda _v: self.update())
        view.verticalScrollBar().valueChanged.connect(lambda _v: self.update())
        view.scene().changed.connect(lambda _r: self.update())
        self._place()
        self.show()

    def eventFilter(self, obj: QObject, e: QEvent) -> bool:        # noqa: N802
        if obj is self.view and e.type() in (QEvent.Resize, QEvent.Show):
            self._place()
        return False

    def _place(self):
        self.setGeometry(self.view.viewport().geometry())
        self.raise_()

    def fraction_of(self, node: Node) -> float | None:
        """How far the running jobs of a block are, or None when nothing of it runs."""
        t = self.c.track
        if t is None or not self.c.running:
            return None
        pids = {p.pid for p in node.paths}
        jobs = [j for j in t.jobs if view_block(j.block) == node.ref and pids & set(j.wf)]     # "fond~2" too (4.2)
        if not jobs or not any(j.state == RUNNING for j in jobs):
            return None
        tot = sum(RunTrack.weight(j) for j in jobs)
        return sum(RunTrack.weight(j) * t.share(j) for j in jobs) / tot

    def paintEvent(self, e):                                       # noqa: N802
        if not self.c.running:
            return
        try:
            from .graph import BH, BW
        except ImportError:
            BW, BH = 124, 54
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        k = self.view.transform().m11()
        for it in self.view.scene().items():
            node = getattr(it, "node", None)
            if not isinstance(node, Node) or node.key in ("sources", "final") and not node.paths:
                continue
            f = self.fraction_of(node)
            if f is None:
                continue
            r = self.view.mapFromScene(it.mapRectToScene(QRectF(0, 0, getattr(it, "w", BW), BH))).boundingRect()
            inset, hh = 10 * k, max(3.0, 3.5 * k)
            track = QRectF(r.left() + inset, r.bottom() - 7 * k - hh / 2, r.width() - 2 * inset, hh)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#12141a"))
            p.drawRoundedRect(track, hh / 2, hh / 2)
            p.setBrush(QColor(ACCENT))
            p.drawRoundedRect(QRectF(track.left(), track.top(), max(track.width() * f, hh), hh), hh / 2, hh / 2)


# ---------------------------------------------------------------------------------------------- run to a stage
def run_blocks(s, steps, items, title: str, parent, show_done: bool = False, controller=None):
    """The plan dialog for (workflow, stage) pairs as ``pending_upto`` gives them (tests call it). The plan is that of
    the block of the last stage, in the first workflow. Prefer ``RunController.run_to_here()``."""
    if not items or controller is None:
        return None
    from .blocks import BY_KEY
    p, stage = items[-1]
    key = next((k for k, b in BY_KEY.items() if stage in b["stages"] and b["kind"] != "auto"), None)
    if key is None:
        return None
    return controller.open_plan(s.plan_to_block(p, key), title, show_done)
