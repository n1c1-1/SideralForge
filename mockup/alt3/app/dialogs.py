"""Dialogs: a new workflow (a name and a starting point, nothing else), the simulated run, renaming.

Displayed texts are English and go through _() at display time (French: locale/fr/dialogs.json)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from i18n import _
from model import TEMPLATES, Session, hm
from widgets import ACCENT, clear, label

# displayed name of each starting point, by TEMPLATES key (the keys stay as they are in the model)
TEMPLATE_NAME = {
    "Vierge": "Classic stack",
    "Classique + débruitage": "Classic + denoise",
    "Complet": "Full",
}
TEMPLATE_TEXT = {
    "Vierge": "The fixed steps only: calibration, registration, stacking, astrometry, color, simple background. "
              "A classic stack.",
    "Classique + débruitage": "The same, plus denoise on two half-stacks.",
    "Complet": "Denoise, star measurement, starless image, background anchored on H-alpha.",
}


class StartCard(QFrame):
    clicked = Signal(int)

    def __init__(self, i, title, text, dur, parent=None):
        super().__init__(parent)
        self.i = i
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("card")
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(2)
        top = QHBoxLayout()
        t = QLabel(_(title))
        t.setStyleSheet("font-weight: 600; background: transparent;")
        top.addWidget(t, 1)
        d = label(dur, "small")
        top.addWidget(d)
        v.addLayout(top)
        if text:
            v.addWidget(label(_(text), "small", wrap=True))
        self.set_selected(False)

    def set_selected(self, on):
        self.setStyleSheet(f"QFrame#card {{ border: 1px solid {ACCENT if on else '#2e333d'}; border-radius: 6px; "
                           f"background: {'#232a36' if on else 'transparent'}; }} QLabel {{ background: transparent; }}")

    def mousePressEvent(self, e):
        self.clicked.emit(self.i)

    def mouseDoubleClickEvent(self, e):
        self.clicked.emit(self.i)
        w = self.window()
        if isinstance(w, QDialog):
            w.accept()


class NewWorkflowDialog(QDialog):
    def __init__(self, session: Session, start=None, parent=None):
        super().__init__(parent)
        self.s = session
        self.setWindowTitle(_("New workflow"))
        self.setMinimumWidth(620)
        v = QVBoxLayout(self)
        v.setSpacing(8)
        v.addWidget(label(_("New workflow"), "h1"))
        row = QHBoxLayout()
        row.addWidget(label(_("Name"), "dim"))
        self.name = QLineEdit()
        row.addWidget(self.name, 1)
        v.addLayout(row)
        v.addSpacing(4)
        v.addWidget(label(_("Start from"), "h2"))
        self.choices = []                                  # (default name, opts, params)
        self.cards = []
        host = QWidget()
        hv = QVBoxLayout(host)
        hv.setContentsMargins(0, 0, 0, 0)
        hv.setSpacing(6)
        for name, opts in TEMPLATES.items():
            shown = TEMPLATE_NAME.get(name, name)
            self._card(hv, shown, TEMPLATE_TEXT.get(name, ""), f"≈ {hm(session.opts_seconds(opts))}",
                       (_("New workflow") if name == "Vierge" else _(shown), opts, {}))
        if session.paths:
            hv.addSpacing(4)
            hv.addWidget(label(_("Copy of a project workflow, with its settings"), "dim"))
            for p in session.paths:
                self._card(hv, _("Copy of “{name}”").format(name=p.name), "",
                           f"≈ {hm(session.opts_seconds(p.opts))}",
                           (_("{name} (copy)").format(name=p.name), p.opts, p.params))
        hv.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(host)
        sc.setMinimumHeight(min(520, 90 + 66 * len(self.cards)))
        v.addWidget(sc, 1)
        bb = QHBoxLayout()
        bb.addStretch(1)
        cancel = QPushButton(_("Cancel"))
        ok = QPushButton(_("Create workflow"))
        ok.setObjectName("primary")
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self.accept)
        bb.addWidget(cancel)
        bb.addWidget(ok)
        v.addLayout(bb)
        first = 0
        if start is not None:
            first = len(TEMPLATES) + session.paths.index(start)
        self.pick(first)

    def _card(self, layout, title, text, dur, choice):
        c = StartCard(len(self.cards), title, text, dur)
        c.clicked.connect(self.pick)
        layout.addWidget(c)
        self.cards.append(c)
        self.choices.append(choice)

    def pick(self, i):
        self.cur = i
        for c in self.cards:
            c.set_selected(c.i == i)
        self.name.setText(self.s.unique_name(self.choices[i][0]))

    @property
    def opts(self):
        return dict(self.choices[self.cur][1])

    @property
    def params(self):
        return {k: dict(v) for k, v in self.choices[self.cur][2].items()}


class RunDialog(QDialog):
    """Simulated computation: the real durations, played in a few seconds."""

    def __init__(self, session: Session, items, title="Run", parent=None):
        super().__init__(parent)
        self.s = session
        self.items = items
        self.setWindowTitle(_(title))
        self.setMinimumWidth(560)
        v = QVBoxLayout(self)
        total = sum(session.seconds(st) for _st, st in items)
        v.addWidget(label(_(title), "h1"))
        v.addWidget(label(_("Estimated real time on this PC: {t}. The mockup plays it in a few seconds; nothing is "
                            "actually computed.").format(t=hm(total)), "dim", wrap=True))
        self.bars = []
        for p, st in items:
            row = QHBoxLayout()
            row.addWidget(label(f"{_(session.title(st))}  ·  {p.name if p else _('common to all workflows')}"), 1)
            row.addWidget(label(hm(session.seconds(st)), "small"))
            b = QProgressBar()
            b.setRange(0, 100)
            b.setFixedWidth(160)
            b.setTextVisible(False)
            row.addWidget(b)
            v.addLayout(row)
            self.bars.append(b)
        self.close_b = QPushButton(_("Close"))
        self.close_b.setEnabled(False)
        self.close_b.clicked.connect(self.accept)
        v.addWidget(self.close_b)
        w = np.array([max(session.seconds(st), 30) for _st, st in items], float)
        self.ticks = np.maximum((w / w.sum() * 90).astype(int), 3)
        self.i, self.k = 0, 0
        self.t = QTimer(self)
        self.t.timeout.connect(self._tick)
        self.t.start(40)

    def _tick(self):
        if self.i >= len(self.bars):
            self.t.stop()
            self.s.mark_stages(self.items)
            self.close_b.setEnabled(True)
            self.close_b.setText(_("Done: close"))
            return
        self.k += 1
        self.bars[self.i].setValue(int(100 * self.k / self.ticks[self.i]))
        if self.k >= self.ticks[self.i]:
            self.i, self.k = self.i + 1, 0

    def finish_now(self):
        """For the automatic walkthrough: skip the animation."""
        self.t.stop()
        for b in self.bars:
            b.setValue(100)
        self.i = len(self.bars)
        self._tick()


def run(session: Session, items, parent, title="Run"):
    if items:
        RunDialog(session, items, title, parent).exec()


def ask_name(parent, title, current) -> str | None:
    d = QDialog(parent)
    d.setWindowTitle(_(title))
    v = QVBoxLayout(d)
    e = QLineEdit(current)
    v.addWidget(e)
    b = QPushButton(_("OK"))
    b.setObjectName("primary")
    b.clicked.connect(d.accept)
    v.addWidget(b, 0, Qt.AlignRight)
    if d.exec() and e.text().strip():
        return e.text().strip()
    return None


class RunPlanDialog(QDialog):
    """What a run will compute, before it starts: the common steps once, then what each workflow adds. One box per
    workflow; a step computed the same way in two workflows is computed once, and it says so."""

    def __init__(self, session: Session, steps, checked=None, title="Run project", parent=None):
        super().__init__(parent)
        self.s = session
        self.steps = steps
        self.setWindowTitle(_(title))
        self.setMinimumWidth(640)
        v = QVBoxLayout(self)
        v.addWidget(label(_(title), "h1"))
        self.common = QVBoxLayout()
        v.addLayout(self.common)
        v.addSpacing(6)
        self.boxes = {}
        self.lines = {}
        for p in session.paths:
            row = QHBoxLayout()
            cb = QCheckBox(p.name)
            cb.setStyleSheet("font-weight: 600;")
            cb.setChecked(checked is None or p.pid in checked)
            cb.toggled.connect(self.update_plan)
            row.addWidget(cb)
            row.addStretch(1)
            dur = label("", "small")
            row.addWidget(dur)
            v.addLayout(row)
            text = label("", "small", wrap=True)
            text.setContentsMargins(26, 0, 0, 4)
            text.setTextFormat(Qt.RichText)
            v.addWidget(text)
            self.boxes[p.pid] = cb
            self.lines[p.pid] = (dur, text)
        v.addSpacing(8)
        bb = QHBoxLayout()
        self.total = label("", "h2")
        bb.addWidget(self.total)
        bb.addStretch(1)
        cancel = QPushButton(_("Cancel"))
        cancel.clicked.connect(self.reject)
        self.go = QPushButton(_("Run"))
        self.go.setObjectName("primary")
        self.go.clicked.connect(self.accept)
        bb.addWidget(cancel)
        bb.addWidget(self.go)
        v.addLayout(bb)
        self.update_plan()

    def plan(self):
        """(common stages, {pid: own stages}, {pid: [(stage, workflow computing it)]}) for the checked workflows."""
        sel = [p for p in self.s.paths if self.boxes[p.pid].isChecked()]
        common = self.s.trunk_pending() if sel else []
        own, shared, seen = {}, {}, {}
        for p in self.s.paths:
            own[p.pid], shared[p.pid] = [], []
            if p not in sel:
                continue
            for st in self.s.pending(p):
                k = self.s.stage_key(p, st)
                if k in seen:
                    shared[p.pid].append((st, seen[k]))
                else:
                    seen[k] = p
                    own[p.pid].append(st)
        return sel, common, own, shared

    def items(self):
        sel, common, own, _shared = self.plan()
        return [(None, st) for st in common] + [(p, st) for p in sel for st in own[p.pid]]

    def _titles(self, p, stages):
        return ", ".join(self.steps.step_titles(p, stages))

    def update_plan(self):
        sel, common, own, shared = self.plan()
        clear(self.common)
        ref = sel[0] if sel else (self.s.paths[0] if self.s.paths else None)
        n = len(self.s.paths)
        head = QHBoxLayout()
        head.addWidget(label(_("Common to the {n} workflows").format(n=n) if n > 1 else _("Common steps"), "h2"))
        head.addStretch(1)
        sec = sum(self.s.seconds(x) for x in common)
        head.addWidget(label(hm(sec) if common else "", "small"))
        self.common.addLayout(head)
        if not sel:
            t = _("No workflow checked.")
        elif common:
            t = _("{steps}: computed once for all the workflows.").format(steps=self._titles(ref, common))
        else:
            t = _("Already computed.")
        lb = label(t, "small", wrap=True)
        lb.setContentsMargins(26, 0, 0, 0)
        self.common.addWidget(lb)
        for p in self.s.paths:
            dur, text = self.lines[p.pid]
            if p not in sel:
                dur.setText("")
                text.setText("<span style='color:#6b7280'>" + _("Not run.") + "</span>")
                continue
            mine = own[p.pid]
            sec_p = sum(self.s.seconds(x) for x in mine)
            dur.setText(hm(sec_p) if mine else "")
            parts = []
            if mine:
                parts.append(self._titles(p, mine))
            for st, q in shared[p.pid]:
                parts.append("<span style='color:#a395ff'>" + _("{steps}: computed once with “{name}”").format(
                    steps=self._titles(p, [st]), name=q.name) + "</span>")
            if not parts:
                parts.append("<span style='color:#5fbf7f'>" + _("Nothing to compute: all up to date.") + "</span>")
            text.setText(" · ".join(dict.fromkeys(parts)))
        tot = sum(self.s.seconds(x) for _p, x in self.items())
        self.total.setText(_("Total time: {t}").format(t=hm(tot)) if tot else _("Nothing to compute"))
        self.go.setText(_("Run · {t}").format(t=hm(tot)) if tot else _("Run"))
        self.go.setEnabled(bool(tot))
