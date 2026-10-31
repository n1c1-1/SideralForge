"""The run, shown block by block, the same before and during the (simulated) computation."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout

from blocks import Graph, block_title
from i18n import _
from model import hm
from steps import STEPS
from widgets import OK, label


def block_key(steps, p, stage) -> str:
    if stage in ("starless", "star_net"):
        return "etoiles"
    for b in STEPS:
        if b["kind"] != "auto" and steps.included(p, b) and stage in steps.stages_of(p, b):
            return b["key"]
    return "post"


def group(s, steps, items):
    """[(node, [(p, stage)])] in the graph's order."""
    g = Graph(s, steps)
    order = {id(n): i for i, n in enumerate(g.nodes)}
    groups = {}
    for p, st in items:
        n = g.node_of(p, block_key(steps, p, st))
        groups.setdefault(id(n), (n, []))[1].append((p, st))
    return g, sorted(groups.values(), key=lambda t: order.get(id(t[0]), 0))


class RunBlocks(QDialog):
    def __init__(self, s, steps, items, title="Run", show_done=False, parent=None):
        super().__init__(parent)
        self.s, self.items = s, items
        self.setWindowTitle(title)
        self.setMinimumWidth(620)
        v = QVBoxLayout(self)
        v.addWidget(label(title, "h1"))
        g, groups = group(s, steps, items)
        self.rows = []
        for n, its in groups:
            p = n.paths[0] if n.paths else its[0][0]
            row = QHBoxLayout()
            row.addWidget(QLabel(block_title(p, n.key) if n.key != "post" else _('Output "{name}"').format(
                name=p.name)), 1)
            who = "" if n.key == "post" else (", ".join(q.name for q in n.paths) if n.shared else p.name)
            w = label(who, "small")
            row.addWidget(w)
            sec = sum(s.seconds(x) for _, x in its)
            d = label(hm(sec), "small")
            d.setFixedWidth(60)
            d.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(d)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFixedWidth(120)
            bar.setTextVisible(False)
            row.addWidget(bar)
            v.addLayout(row)
            self.rows.append((bar, max(sec, 30.0)))
        if show_done:
            done = sum(1 for n in g.nodes if n.key not in ("sources", "final") and n.paths and g.state(n) == "done")
            if done:
                ok = label(_("Already up to date: {n} blocks").format(n=done), "small")
                ok.setStyleSheet(f"color: {OK}; font-size: 9pt;")
                v.addWidget(ok)
        tot = sum(s.seconds(x) for _, x in items)
        bb = QHBoxLayout()
        gb = round(sum(s.data["stages"].get(x, {}).get("gb", 0) for _p, x in items), 1)
        bb.addWidget(label(_("Duration: {t} · disk: {gb} GB").format(t=hm(tot), gb=gb) if tot else
                           _("Nothing to run"), "h2"))
        bb.addStretch(1)
        self.cancel = QPushButton(_("Cancel"))
        self.cancel.clicked.connect(self.reject)
        self.go = QPushButton(_("Run"))
        self.go.setObjectName("primary")
        self.go.setEnabled(bool(items) and s.sources_loaded)
        self.go.clicked.connect(self.start)
        bb.addWidget(self.cancel)
        bb.addWidget(self.go)
        v.addSpacing(8)
        v.addLayout(bb)
        self.t = QTimer(self)
        self.t.timeout.connect(self._tick)

    def start(self):
        if self.go.text() == _("Close"):
            self.accept()
            return
        self.go.setEnabled(False)
        self.cancel.setEnabled(False)
        w = np.array([sec for _b, sec in self.rows], float)
        self.ticks = np.maximum((w / w.sum() * 90).astype(int), 3) if len(w) else []
        self.i, self.k = 0, 0
        self.t.start(40)

    def _tick(self):
        if self.i >= len(self.rows):
            self.t.stop()
            self.s.mark_stages(self.items)
            if hasattr(self.s, "log"):
                self.s.log(_("Run: {n} stages · {t}").format(n=len(self.items), t=hm(sum(
                    self.s.seconds(x) for _p, x in self.items))))
            self.go.setText(_("Close"))
            self.go.setEnabled(True)
            return
        self.k += 1
        self.rows[self.i][0].setValue(int(100 * self.k / self.ticks[self.i]))
        if self.k >= self.ticks[self.i]:
            self.i, self.k = self.i + 1, 0

    def finish_now(self):
        """For the automatic walkthrough."""
        if not self.t.isActive():
            self.start()
        self.t.stop()
        for bar, _sec in self.rows:
            bar.setValue(100)
        self.i = len(self.rows)
        self._tick()


def run_blocks(s, steps, items, title, parent, show_done=False):
    if items:
        RunBlocks(s, steps, items, title, show_done, parent).exec()
