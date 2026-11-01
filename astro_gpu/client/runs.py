"""Runs: the plan dialog, the progress of a run block by block, its recap, the progress drawn on the graph and the
status bar widget (the service that launches and follows a run is services/run.py).

The dialog is the mockup's Run dialog (one row per block in graph order: the block, the workflows sharing it, the
duration, a progress bar; "Already up to date: N blocks"; "Duration: t · disk: N GB"; Cancel, Run) carried on: once Run
is pressed the same rows show the run (a mark and a coloured bar per block), Close leaves the run going on (the
progress stays in the status bar and on the graph), and Stop ends it so that the next run goes on from there. At the
end the rows keep their final state and a recap says what was computed, what failed and which blocks were not run
because of it, in plain words.

The dialog is not modal. ``RunDialog`` reads a ``RunTrack`` (services/run.py) through the controller and redraws when
it changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
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

from .blocks import ENGINE_BLOCK, Graph, Node, block_title
from .i18n import _, hm
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
    size_text,
)
from .widgets import ACCENT, BAD, OK, WARN, Box, ElidedLabel, clear, label

GREY = "#5d6470"
STATE_COLOR = {WAITING: "#3a4150", RUNNING: ACCENT, DONE: OK, FAILED: BAD, INTERRUPTED: WARN, SKIPPED: GREY,
               NOT_STARTED: GREY}
STATE_TIP = {WAITING: "Waiting", RUNNING: "Running", DONE: "Done", FAILED: "Failed", INTERRUPTED: "Stopped",
             SKIPPED: "Not run", NOT_STARTED: "Not run"}
ROW_H = 28
ROWS_MAX_H = 440


# ---------------------------------------------------------------------------------------------- the rows
@dataclass
class Row:
    """A line of the dialog: one block of the graph, the jobs (stage keys) the run computes in it."""

    title: str
    who: list[str]
    keys: list[str]
    block: str = ""                                          # the views' block key
    pids: set = field(default_factory=set)


def build_rows(model, jobs, names: dict[int, str] | None = None) -> list[Row]:
    """One row per block of the graph that has something to run, in the graph's order. ``jobs``: runner2.Job or
    JobState. The workflows named are those that share the result (``job.wf``)."""
    names = names or {}
    g = Graph(model, None) if model.paths else None
    order = {id(n): i for i, n in enumerate(g.nodes)} if g is not None else {}
    found: dict = {}
    for i, j in enumerate(jobs):
        bk = ENGINE_BLOCK.get(j.block)
        v = model.view(j.run_wf) or next((model.view(w) for w in j.wf if model.view(w) is not None), None)
        node = g.node_of(v, bk) if (g is not None and v is not None and bk) else None
        rid = id(node) if node is not None else (j.block, tuple(sorted(j.wf)))
        if rid not in found:
            if bk == "post":
                title = _('Output "{name}"').format(name=v.name if v is not None else names.get(j.run_wf, "?"))
            elif bk:
                title = block_title(v, bk)
            else:
                title = model.title(j.stage)
            pos = order.get(id(node), len(order) + i)
            found[rid] = (pos, Row(title, [], [], bk or "", set()))
        row = found[rid][1]
        row.keys.append(j.key)
        row.pids |= set(j.wf)
    out = []
    for _pos, row in sorted(found.values(), key=lambda t: t[0]):
        ids = [p.pid for p in model.paths if p.pid in row.pids]
        ids += sorted(w for w in row.pids if w not in ids)
        row.who = [] if row.block == "post" else [(model.view(w).name if model.view(w) is not None
                                                   else names.get(w, f"#{w}")) for w in ids]
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
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor(STATE_COLOR[self.state])
        pen = QPen(c, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        s = self.state
        if s == DONE:
            p.drawPolyline([QPointF(3, 8.5), QPointF(6.5, 12), QPointF(13, 4.5)])
        elif s == FAILED:
            p.drawLine(QPointF(4, 4), QPointF(12, 12))
            p.drawLine(QPointF(12, 4), QPointF(4, 12))
        elif s == RUNNING:
            p.setBrush(c)
            p.drawEllipse(QPointF(8, 8), 3.6, 3.6)
        elif s == INTERRUPTED:
            p.setBrush(c)
            p.drawRect(QRectF(4.5, 4.5, 7, 7))
        elif s in (SKIPPED, NOT_STARTED):
            p.drawLine(QPointF(4.5, 8), QPointF(11.5, 8))
        else:
            p.setPen(QPen(c, 1.4))
            p.drawEllipse(QPointF(8, 8), 3.2, 3.2)


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
        self.who = ElidedLabel(", ".join(row.who), "small", mode=Qt.ElideRight)
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
        v.addWidget(label(head, "h1"))
        self.warn = QVBoxLayout()
        self.warn.setSpacing(2)
        v.addLayout(self.warn)
        self.rows_data = build_rows(self.s, self.plan.jobs, self.names) if self.plan is not None else []
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
        self.hint = label("", "small", wrap=True)
        v.addWidget(self.hint)
        self.recap = Box(OK)
        self.recap.hide()
        v.addWidget(self.recap)
        v.addSpacing(8)
        foot = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(1)
        self.summary = label("", "h2")
        self.detail = label("", "small")
        col.addWidget(self.summary)
        col.addWidget(self.detail)
        foot.addLayout(col, 1)
        self.b_log = QPushButton(_("Show the log"))
        self.b_log.setObjectName("link")
        self.b_log.clicked.connect(lambda: self.c.logRequested.emit() if self.c is not None else None)
        foot.addWidget(self.b_log)
        foot.addSpacing(8)
        self.b_cancel = QPushButton(_("Cancel"))
        self.b_cancel.clicked.connect(self.close)
        self.b_stop = QPushButton(_("Stop"))
        self.b_stop.clicked.connect(self._stop)
        self.b_run = QPushButton(_("Run"))
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
        for w in self.rows:
            r = w.row
            jobs = [t.by_key[k] for k in r.keys if t is not None and k in t.by_key] if t is not None else []
            plan_jobs = {j.key: j for j in self.plan.jobs}
            est = sum(plan_jobs[k].est_s for k in r.keys if k in plan_jobs)
            measured = all(plan_jobs[k].measured for k in r.keys if k in plan_jobs)
            if not jobs:
                tip = "\n".join(f"{self.s.title(plan_jobs[k].stage)}: "
                                f"{duration_text(plan_jobs[k].est_s, plan_jobs[k].measured)}"
                                for k in r.keys if k in plan_jobs)
                w.show_state(WAITING, 0.0, duration_text(est, measured) if est else "", tip)
                continue
            state = row_state([j.state for j in jobs])
            tot = sum(RunTrack.weight(j) for j in jobs)
            frac = sum(RunTrack.weight(j) * t.share(j, True) for j in jobs) / tot
            if state == DONE:
                real = sum(j.duration or 0.0 for j in jobs)
                text = hm(real) if any(not j.reused for j in jobs) else "—"
            elif state in (SKIPPED, NOT_STARTED):
                text = "—"
            else:
                text = duration_text(est, measured) if est else ""
            w.show_state(state, frac, text, self._tip(jobs, state))

    def _tip(self, jobs: list[JobState], state: str) -> str:
        lines = [_(STATE_TIP[state])]
        for j in jobs:
            what = self.s.title(j.stage)
            if j.state == FAILED and j.error:
                lines.append(f"{what}: {j.error[:300]}")
            elif len(jobs) > 1:
                lines.append(f"{what}: {_(STATE_TIP[j.state])}")
        return "\n".join(lines)

    def refresh(self, *_a):
        if self.body is None:
            return
        ph, t = self.phase(), self.track
        self._rows_state()
        n = len(self.plan.jobs) if self.plan is not None else 0
        # notes of the plan
        clear(self.warn)
        if ph == "plan":
            for wid, why in (getattr(self.plan, "skipped", None) or {}).items():
                v = self.s.view(wid)
                self.warn.addWidget(label(_('"{name}" is not run: {why}').format(
                    name=v.name if v is not None else self.names.get(wid, "?"), why=_(why)), "warn", wrap=True))
                self.warn.itemAt(self.warn.count() - 1).widget().show()
            free = self.c.free_gb() if self.c is not None else None
            need = self.plan.disk_gb if self.plan is not None else 0.0
            if free is not None and need > free:
                self.warn.addWidget(label(_("Not enough free disk space: {need} GB needed, {free} GB free on the "
                                            "project's drive.").format(need=size_text(need), free=size_text(free)),
                                          "warn", wrap=True))
                self.warn.itemAt(self.warn.count() - 1).widget().show()
        # the notes under the rows
        n_done = self._up_to_date() if ph == "plan" and self.show_done else 0
        self.done_note.setText(_("Already up to date: {n} blocks").format(n=n_done) if n_done else "")
        self.done_note.setVisible(bool(n_done))
        self.hint.setVisible(ph == "run")
        self.hint.setText(_("The run goes on if this window is closed: its progress stays in the status bar and on "
                            "the graph."))
        self.recap.setVisible(ph == "end")
        if ph == "end":
            self._fill_recap(t)
        # the footer
        self.summary.setStyleSheet("")
        self.detail.setText("")
        if ph == "plan":
            tot = self.plan.seconds if self.plan is not None else 0.0
            self.summary.setText(_("Duration: {t} · disk: {gb} GB").format(
                t=duration_text(tot, self.plan.measured), gb=size_text(self.plan.disk_gb)) if n else _("Nothing to run"))
        elif ph == "run":
            self._fill_progress(t)
        else:
            self._fill_end(t)
        stopping = self.c is not None and self.c.phase == "stopping"
        self.b_cancel.setText(_("Cancel") if ph == "plan" else _("Close"))
        self.b_cancel.setVisible(ph != "end")
        self.b_stop.setVisible(ph == "run")
        self.b_stop.setEnabled(self.c is not None and self.c.phase == "running")
        self.b_stop.setText(_("Stopping…") if stopping else _("Stop"))
        self.b_run.setVisible(ph == "plan")
        self.b_run.setEnabled(bool(n) and self.s.sources_loaded and (self.c is None or not self.c.running))
        self.b_done.setVisible(ph == "end")
        self.b_log.setVisible(ph == "end" and t is not None and (t.count()[FAILED] > 0 or t.lost or t.status == "failed"))

    def _up_to_date(self) -> int:
        """The blocks of the graph that are done (the mockup's "Already up to date")."""
        if not self.s.paths:
            return 0
        g = Graph(self.s, None)
        return sum(1 for nd in g.nodes if nd.key not in ("sources", "final") and nd.paths and g.state(nd) == "done")

    def _fill_progress(self, t: RunTrack):
        if self.c is not None and self.c.phase == "starting" or not t.running:
            self.summary.setText(_("Starting the run…"))
            return
        elapsed, left = t.elapsed(), t.remaining()
        self.summary.setText(_("Running · {t}").format(t=hm(elapsed)) + (
            " · " + _("≈ {t} left").format(t=hm(left)) if left > 30 else ""))
        j = t.current()
        if j is not None:
            from .services.run import engine_text
            pct = int(round(100 * j.frac))
            self.detail.setText(f"{self.s.title(j.stage)} · {pct} %" + (f" · {engine_text(j.msg)}" if j.msg else ""))

    def _fill_end(self, t: RunTrack):
        color = {"done": OK, "failed": BAD, "interrupted": WARN}[t.status]
        text = {"done": _("Done in {t}").format(t=hm(t.elapsed())),
                "failed": _("Failed"), "interrupted": _("Run stopped")}[t.status]
        self.summary.setText(text)
        self.summary.setStyleSheet(f"color: {color};")

    # ------------------------------------------------------------ the recap
    def _title_of(self, key: str) -> str:
        r = next((x for x in self.rows_data if key in x.keys), None)
        return r.title if r is not None else key

    def _fill_recap(self, t: RunTrack):
        """What happened, in plain words: computed, failed, not run because of what, stopped."""
        box = self.recap
        while box.v.count():
            it = box.v.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        c = t.counts or t.count()
        lines: list[tuple[str, str]] = []                          # (text, colour)
        by_row = [(r, [t.by_key[k] for k in r.keys if k in t.by_key]) for r in self.rows_data]
        failed = [(r, js) for r, js in by_row if any(j.state == FAILED for j in js)]
        stopped = [r for r, js in by_row if any(j.state == INTERRUPTED for j in js)]
        notrun = [(r, js) for r, js in by_row if js and all(j.state in (SKIPPED, NOT_STARTED) for j in js)]
        col = {"done": OK, "interrupted": WARN}.get(t.status, BAD)
        if t.status == "done":
            lines.append((_("{n} stages computed in {t}.").format(n=c["computed"], t=hm(t.elapsed())), OK))
            names = [v.name for v in self.s.paths if v.rendered and v.pid in self._plan_wf()]
            if names:
                lines.append((_("Layers ready in Post: {names}").format(names=", ".join(names)), ""))
        elif t.lost:
            lines.append((_("The run stopped without warning: its process ended. What was computed is kept; the next "
                            "run goes on from the last checkpoint."), WARN))
        elif t.status == "interrupted":
            lines.append((_("Stopped. What was computed is kept; the next run goes on from where this one stopped."),
                          WARN))
            if stopped:
                lines.append((_("Stopped during {blocks}.").format(blocks=", ".join(r.title for r in stopped)), ""))
        listed = set()
        for r, js in failed:
            bad = next(j for j in js if j.state == FAILED)
            why = (bad.error or t.error or "").strip()
            why = why.splitlines()[0][:240] if why else ""
            lines.append((_("{block} failed: {why}").format(block=r.title, why=why) if why
                          else _("{block} failed.").format(block=r.title), BAD))
            blocked = [x for x, jx in notrun if any(bad.key in {k.key for k in t.culprits(j)}
                                                      for j in jx if j.state == SKIPPED)]
            listed |= {id(x) for x in blocked}
            if blocked:
                lines.append((_("Not run because {block} failed: {blocks}.").format(
                    block=r.title, blocks=", ".join(x.title for x in blocked)), ""))
        other = [r for r, js in notrun if id(r) not in listed]
        if other and t.status != "done":
            started = any(j.state == SKIPPED for r, js in notrun if r in other for j in js)
            lines.append(((_("Not run because an earlier block failed: {blocks}.") if started
                           else _("Not started: {blocks}.")).format(blocks=", ".join(r.title for r in other)), ""))
        if failed and c[DONE]:
            lines.append((_("The blocks that did not depend on it were computed."), ""))
        if t.status == "failed" and not failed and t.error:
            lines.append((t.error, BAD))
        box.setStyleSheet(f"QFrame#box {{ background: #23211c; border: 1px solid {col}; border-radius: 6px; }}"
                          "QFrame#box QLabel { background: transparent; }")
        for text, color in lines:
            lb = label(text, None, wrap=True)
            if color:
                lb.setStyleSheet(f"color: {color};")
            box.v.addWidget(lb)
            lb.show()

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
        self.b_stop = QPushButton(_("Stop"))
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
            self.b_stop.setEnabled(self.c.phase == "running")


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
        jobs = [j for j in t.jobs if ENGINE_BLOCK.get(j.block) == node.key and pids & set(j.wf)]
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
            r = self.view.mapFromScene(it.mapRectToScene(QRectF(0, 0, BW, BH))).boundingRect()
            inset, hh = 10 * k, max(3.0, 3.5 * k)
            track = QRectF(r.left() + inset, r.bottom() - 7 * k - hh / 2, r.width() - 2 * inset, hh)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#12141a"))
            p.drawRoundedRect(track, hh / 2, hh / 2)
            p.setBrush(QColor(ACCENT))
            p.drawRoundedRect(QRectF(track.left(), track.top(), max(track.width() * f, hh), hh), hh / 2, hh / 2)


# ---------------------------------------------------------------------------------------------- the mockup's door
def run_blocks(s, steps, items, title: str, parent, show_done: bool = False, controller=None):
    """The mockup's ``run_blocks(s, steps, items, title, parent)``: ``items`` are (workflow, stage) pairs as
    ``pending_upto`` gives them. The plan is that of the block of the last stage, in the first workflow. Prefer
    ``RunController.run_to_here()``."""
    if not items or controller is None:
        return None
    from .blocks import BY_KEY
    p, stage = items[-1]
    key = next((k for k, b in BY_KEY.items() if stage in b["stages"] and b["kind"] != "auto"), None)
    if key is None:
        return None
    return controller.open_plan(s.plan_to_block(p, key), title, show_done)
