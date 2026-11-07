"""Dialogs: renaming (``ask_name``, ``name_dialog``). A new workflow is made in the template gallery (gallery.py).

Displayed texts are English and go through _() at display time.
"""
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout

from .i18n import _


def ask_name(parent, title, current) -> str | None:
    """The name typed, "" when OK was pressed on an empty name (the caller says it is refused); None: cancelled."""
    d, e = name_dialog(parent, title, current)
    if d.exec():
        return e.text().strip()
    return None


def name_dialog(parent, title, current) -> tuple[QDialog, QLineEdit]:
    d = QDialog(parent)
    d.setWindowTitle(_(title))
    d.setMinimumWidth(360)
    v = QVBoxLayout(d)
    e = QLineEdit(current)
    e.selectAll()
    v.addWidget(e)
    bb = QHBoxLayout()
    bb.addStretch(1)
    c = QPushButton(_("Cancel"))
    c.clicked.connect(d.reject)
    b = QPushButton(_("OK"))
    b.setObjectName("primary")
    b.setDefault(True)
    b.clicked.connect(d.accept)
    bb.addWidget(c)
    bb.addWidget(b)
    v.addLayout(bb)
    return d, e
