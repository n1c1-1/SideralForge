"""Log: the history of the project (runs, changes, variants) and the console of the last run, at the bottom of the
window."""
from __future__ import annotations

from PySide6.QtWidgets import (QAbstractItemView, QFrame, QHBoxLayout, QPlainTextEdit, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from i18n import _
from widgets import label

CONSOLE_DEMO = """[stack] 721 frames · drizzle ×1.5 · pixfrac 0.70
[stack] pass 1: weighted mean · 721 / 721
[stack] pass 2: rejection (κ 5.0σ) · 1.8 % of pixels rejected
[stack] half-stacks A / B: 360 / 361 frames
[stack] done in 33 min 52 s · 6384 × 4272 px · 2.1 GB"""


class LogPanel(QFrame):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.s = project
        self.setObjectName("log")
        self.setStyleSheet("QFrame#log { background: #1c1f26; border-top: 1px solid #2a2f38; }")
        self.setFixedHeight(200)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 6)
        top = QHBoxLayout()
        top.addWidget(label(_("Log"), "h2"))
        top.addStretch(1)
        ex = QPushButton(_("Export log…"))
        top.addWidget(ex)
        v.addLayout(top)
        h = QHBoxLayout()
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels([_("Time"), _("Event")])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setStyleSheet("QTableWidget { background: #15171c; gridline-color: #2a2f38; } "
                                 "QHeaderView::section { background: #23272f; border: none; padding: 3px; }")
        h.addWidget(self.table, 3)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background: #12141a; font-family: Consolas; font-size: 9pt;")
        self.console.setPlainText(CONSOLE_DEMO)
        h.addWidget(self.console, 2)
        v.addLayout(h)

    def refresh(self):
        rows = list(reversed(self.s.history))
        self.table.setRowCount(len(rows))
        for i, (t, text) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(t))
            self.table.setItem(i, 1, QTableWidgetItem(text))
        self.table.resizeColumnToContents(0)
