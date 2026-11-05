"""Dialogs: a new workflow (a name and a starting point, nothing else), renaming.

Displayed texts are English and go through _() at display time.

The New workflow dialog works on the project model (model.py): its ``paths`` (each with ``name``, ``opts``,
``params``), ``opts_seconds(opts)`` and ``unique_name(name)``; the starting points are the model's ``TEMPLATES``
({name: opts}) unless a dict is given.

Classes and functions: ``NewWorkflowDialog`` (its starting points are ``StartCard`` items), ``ask_name`` and
``name_dialog`` (Rename).
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget

from .blocks import TRUNK
from .i18n import _, hm
from .widgets import ACCENT, label

# displayed name of each starting point, by its key in model.TEMPLATES
TEMPLATE_NAME = {
    "Vierge": "Classic stack",
    "Classique + débruitage": "Classic + denoise",
    "Complet": "Full",
}
TEMPLATE_TEXT = {
    "Classic stack": "The fixed steps only: calibration, registration, stacking, astrometry, color, simple background. "
                     "A classic stack.",
    "Classic + denoise": "The same, plus denoise on two half-stacks.",
    "Full": "The same as Classic + denoise, plus the stars taken off: the starless image and the stars, recombined in "
            "Post.",
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
        t = QLabel(title)
        t.setStyleSheet("font-weight: 600; background: transparent;")
        top.addWidget(t, 1)
        d = label(dur, "small")
        top.addWidget(d)
        v.addLayout(top)
        if text:
            v.addWidget(label(text, "small", wrap=True))
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


#: the template a new workflow starts from unless the user picks another (the one the defaults were checked with)
RECOMMENDED = "Full"


class NewWorkflowDialog(QDialog):
    def __init__(self, session, start=None, parent=None, templates: dict | None = None):
        super().__init__(parent)
        self.s = session
        if templates is None:
            from .model import TEMPLATES as templates
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
        # a new workflow joins the current one's pre-processing (its settings of Calibration to Stacking): its
        # duration is quoted with them; a copy, with all of its settings
        cur = getattr(session, "path", None)
        trunk = {k: v for k, v in (cur.params if cur is not None else {}).items() if k in TRUNK}
        recommended = 0
        for i, (key, opts) in enumerate(templates.items()):
            shown = TEMPLATE_NAME.get(key, key)
            best = shown == RECOMMENDED                    # issue #11: the recommended one said, and chosen first
            if best:
                recommended = i
            self._card(hv, _(shown) + ("  ·  " + _("Recommended") if best else ""), _(TEMPLATE_TEXT.get(shown, "")),
                       f"≈ {hm(session.opts_seconds(opts, trunk))}", (_(shown), opts, {}))
        if session.paths:
            hv.addSpacing(4)
            hv.addWidget(label(_("Copy of a project workflow, with its settings"), "dim"))
            for p in session.paths:
                self._card(hv, _("Copy of “{name}”").format(name=p.name), "",
                           f"≈ {hm(session.opts_seconds(p.opts, p.params))}",
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
        ok.setDefault(True)
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self.accept)
        bb.addWidget(cancel)
        bb.addWidget(ok)
        v.addLayout(bb)
        first = recommended
        if start is not None:
            first = len(templates) + session.paths.index(start)
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


def ask_name(parent, title, current) -> str | None:
    d, e = name_dialog(parent, title, current)
    if d.exec() and e.text().strip():
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
