"""Home: the workspace and its projects. New project, open, import a folder as a new project, recent projects."""
from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import (QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

import prefs
from i18n import _
from model import DATA
from project import PROJECTS_DEMO
from viewer import to_qimage
from widgets import ACCENT, label

TW, TH = 260, 150


def thumb(proj) -> QPixmap:
    if proj.get("thumb"):
        im = Image.open(DATA / "img" / "complet_base.jpg").convert("RGB")
        w, h = im.size
        cw, ch = (w, int(w * TH / TW)) if w * TH / TW <= h else (int(h * TW / TH), h)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((TW, TH))
        return QPixmap.fromImage(to_qimage(np.asarray(im)))
    pm = QPixmap(TW, TH)
    pm.fill(QColor("#14161b"))
    p = QPainter(pm)
    rng = np.random.default_rng(abs(hash(proj["name"])) % 2 ** 32)
    for _i in range(160):
        a = int(rng.integers(40, 220))
        p.setPen(QColor(a, a, a))
        p.drawPoint(int(rng.integers(0, TW)), int(rng.integers(0, TH)))
    f = QFont("Segoe UI", 11)
    p.setFont(f)
    p.setPen(QColor("#6b7280"))
    p.drawText(pm.rect(), Qt.AlignCenter, _("No preview"))
    p.end()
    return pm


class ProjectCard(QFrame):
    clicked = Signal(dict)

    def __init__(self, proj, parent=None):
        super().__init__(parent)
        self.proj = proj
        self.setObjectName("pcard")
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("QFrame#pcard { background: #1c1f26; border: 1px solid #2e333d; border-radius: 8px; }"
                           f"QFrame#pcard:hover {{ border-color: {ACCENT}; }} QLabel {{ background: transparent; }}")
        self.setFixedWidth(TW + 20)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(3)
        t = QLabel()
        t.setPixmap(thumb(proj))
        v.addWidget(t)
        n = QLabel(proj["name"])
        n.setStyleSheet("font-weight: 600; font-size: 11pt;")
        v.addWidget(n)
        v.addWidget(label(proj["target"], "dim"))
        v.addWidget(label(_("{n} night(s) · {h} · {d}").format(n=proj["nights"], h=proj["integration"],
                                                             d=proj["date"]), "small"))
        v.addWidget(label(_(proj["state"]), "small"))

    def mousePressEvent(self, e):
        self.clicked.emit(self.proj)


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_("New project"))
        self.setMinimumWidth(520)
        v = QVBoxLayout(self)
        v.addWidget(label(_("New project"), "h1"))
        for text, ph in ((_("Name"), "NGC 7000"), (_("Target"), _("optional: filled in by astrometry"))):
            row = QHBoxLayout()
            lb = label(text, "dim")
            lb.setFixedWidth(80)
            row.addWidget(lb)
            e = QLineEdit()
            e.setPlaceholderText(ph)
            row.addWidget(e, 1)
            v.addLayout(row)
            if text == _("Name"):
                self.name = e
        row = QHBoxLayout()
        lb = label(_("Folder"), "dim")
        lb.setFixedWidth(80)
        row.addWidget(lb)
        row.addWidget(label(prefs.get("workspace") + "/…", "small"), 1)
        v.addLayout(row)
        bb = QHBoxLayout()
        bb.addStretch(1)
        c = QPushButton(_("Cancel"))
        c.clicked.connect(self.reject)
        ok = QPushButton(_("Create"))
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        bb.addWidget(c)
        bb.addWidget(ok)
        v.addLayout(bb)


class HomeView(QWidget):
    openProject = Signal(str)          # demo scenario
    newProject = Signal(str)           # name

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        left = QWidget()
        left.setObjectName("panel")
        left.setFixedWidth(300)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(18, 20, 18, 18)
        lv.setSpacing(8)
        lv.addWidget(label(_("Workspace"), "h2"))
        self.ws = label(prefs.get("workspace"), "dim")
        self.ws.setWordWrap(True)
        lv.addWidget(self.ws)
        ch = QPushButton(_("Change…"))
        ch.clicked.connect(self.change_ws)
        lv.addWidget(ch, 0, Qt.AlignLeft)
        lv.addSpacing(14)
        for text, fn, primary in ((_("New project…"), self.new_project, True),
                                  (_("Import folder as new project…"), self.import_folder, False),
                                  (_("Open project…"), self.open_other, False)):
            b = QPushButton(text)
            if primary:
                b.setObjectName("primary")
            b.clicked.connect(fn)
            lv.addWidget(b)
        lv.addStretch(1)
        lv.addWidget(label(_("{n} projects in the workspace").format(n=len(PROJECTS_DEMO)), "small"))
        root.addWidget(left)
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(24, 20, 24, 20)
        mv.addWidget(label(_("Recent projects"), "h1"))
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        host = QWidget()
        grid = QGridLayout(host)
        grid.setSpacing(16)
        grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        for i, pr in enumerate(PROJECTS_DEMO):
            c = ProjectCard(pr)
            c.clicked.connect(self.open_card)
            grid.addWidget(c, i // 3, i % 3)
        sc.setWidget(host)
        mv.addWidget(sc, 1)
        root.addWidget(mid, 1)

    def change_ws(self):
        d = QFileDialog.getExistingDirectory(self, _("Workspace"), prefs.get("workspace"))
        if d:
            prefs.set("workspace", d)
            self.ws.setText(d)

    def open_card(self, proj):
        from PySide6.QtWidgets import QMessageBox
        if proj.get("scenario"):
            self.openProject.emit(proj["scenario"])
        else:
            QMessageBox.information(self, proj["name"], _("Demo: only HeartSoul has data in this mockup."))

    def new_project(self):
        d = NewProjectDialog(self.window())
        if d.exec():
            self.newProject.emit(d.name.text().strip() or "NGC 7000")

    def import_folder(self):
        self.newProject.emit("HeartSoul")

    def open_other(self):
        f, _x = QFileDialog.getOpenFileName(self, _("Open project"), prefs.get("workspace"),
                                            _("Projects (*.astroproj)"))
        if f:
            self.openProject.emit("existant")
