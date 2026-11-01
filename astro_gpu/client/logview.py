"""Log: the history of the project (changes, variants, runs), newest first, and the console of the last run, at the
bottom of the window (View > Log, Ctrl+L).

The history is the project's ``history``: (time, text) rows, or (time, English template, arguments) rows translated
when shown, so that a change of language applies to them too. The console is fed by the run service
(``set_console``, ``append_console``).
"""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .i18n import _
from .widgets import label

#: the arguments of a history line that are words of the interface (a block's name), translated when shown, so that a
#: line written in one language reads in the other
WORD_ARGS = ("block",)


def event_text(row) -> tuple[str, str]:
    """(time, text) of a history row, translated."""
    if len(row) >= 3:
        t, template, args = row[0], row[1], dict(row[2] or {})
        for k in WORD_ARGS:
            if isinstance(args.get(k), str):
                args[k] = _(args[k])
        return t, _(template).format(**args)
    return row[0], row[1]


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
        self.b_export = QPushButton(_("Export log…"))
        self.b_export.clicked.connect(self.export_log)
        top.addWidget(self.b_export)
        v.addLayout(top)
        h = QHBoxLayout()
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels([_("Time"), _("Event")])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setStyleSheet("QTableWidget { background: #15171c; gridline-color: #2a2f38; } "
                                 "QHeaderView::section { background: #23272f; border: none; padding: 3px; }")
        h.addWidget(self.table, 3)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText(_("No run yet"))
        self.console.setStyleSheet("background: #12141a; font-family: Consolas; font-size: 9pt;")
        h.addWidget(self.console, 2)
        v.addLayout(h)
        self.refresh()

    def rows(self) -> list[tuple[str, str]]:
        return [event_text(r) for r in reversed(getattr(self.s, "history", None) or [])]

    def refresh(self):
        rows = self.rows()
        self.table.setRowCount(len(rows))
        for i, (t, text) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(t))
            self.table.setItem(i, 1, QTableWidgetItem(text))
        self.table.resizeColumnToContents(0)

    # ------------------------------------------------------------ the console of the run
    def set_console(self, text: str):
        self.console.setPlainText(text)

    def append_console(self, text: str):
        self.console.appendPlainText(text.rstrip("\n"))

    # ------------------------------------------------------------ export
    def write(self, path: str | Path):
        """The history, then the console, as a text file."""
        lines = [f"{t}\t{text}" for t, text in self.rows()]
        console = self.console.toPlainText().strip()
        if console:
            lines += ["", console]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def export_log(self):
        root = getattr(self.s, "root", None)
        name = f"log_{time.strftime('%Y%m%d_%H%M')}.txt"
        start = (Path(root) / "logs" / name) if root else Path(name)
        f, _x = QFileDialog.getSaveFileName(self, _("Export log…"), start.as_posix(), _("Text files (*.txt)"))
        if f:
            self.write(f)
