"""Results tab: what this project is, and which rendered image to look at."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

# the two renders every project ends with (astro-gpu render --preset faithful / artistic); other names are
# experiments or older versions
DELIVERABLES = {
    "artistic": ("Artistic - look at this one first",
                 "Sharpening with the measured star shape, stronger stretch and local contrast, judged by eye."),
    "faithful": ("Faithful - only what the data prove",
                 "Measured colours, no display sharpening, detail as measured by the denoiser."),
}


class ResultsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.project = None
        v = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 16px; font-weight: 600;")
        v.addWidget(self.title)
        v.addWidget(QLabel("About this project (saved in project.json, shown in the Recent menu):"))
        self.desc = QPlainTextEdit()
        self.desc.setPlaceholderText("Instrument, frames, what was done, what to look at...")
        self.desc.setMaximumHeight(90)
        v.addWidget(self.desc)
        row = QHBoxLayout()
        self.save_btn = QPushButton("Save description")
        self.save_btn.clicked.connect(self.save_description)
        row.addWidget(self.save_btn)
        self.open_btn = QPushButton("Open results folder")
        self.open_btn.clicked.connect(lambda: self._open(self.project.results) if self.project else None)
        row.addWidget(self.open_btn)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        row.addWidget(self.refresh_btn)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(QLabel("Rendered images (results/<name>/<name>.jpg):"))
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.inner = QWidget()
        self.grid = QGridLayout(self.inner)
        self.grid.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.inner)
        v.addWidget(self.scroll, 1)

    # ------------------------------------------------------------------
    def set_project(self, project):
        self.project = project
        self.title.setText(f"{project.config.name}  -  {project.root}")
        self.desc.setPlainText(project.config.description)
        self.refresh()

    def save_description(self):
        if self.project is None:
            return
        self.project.config.description = self.desc.toPlainText().strip()
        self.project.save()
        from ..core.project import remember
        remember(self.project)

    @staticmethod
    def _open(path: Path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def refresh(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        if self.project is None:
            return
        rows = []
        for d in sorted(self.project.results.glob("*")):
            jpg = d / f"{d.name}.jpg"
            if not d.is_dir() or not jpg.exists():
                continue
            date = ""
            meta = d / f"{d.name}_settings.json"
            try:
                date = json.loads(meta.read_text(encoding="utf-8")).get("date", "")[:16].replace("T", " ")
            except (OSError, ValueError):
                pass
            kind = DELIVERABLES.get(d.name.lower())
            rows.append((0 if kind else 1, d.name, jpg, date, kind))
        rows.sort()
        if not rows:
            self.grid.addWidget(QLabel("No render yet: use the Final adjustment tab (presets Faithful and Artistic) "
                                       "or 'astro-gpu render PROJECT --preset artistic'."), 0, 0)
            return
        for r, (_, name, jpg, date, kind) in enumerate(rows):
            thumb = QLabel()
            pm = QPixmap(str(jpg))
            if not pm.isNull():
                thumb.setPixmap(pm.scaled(260, 170, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            thumb.setFrameShape(QFrame.StyledPanel)
            self.grid.addWidget(thumb, r, 0)
            head, note = kind if kind else (name, "Experiment or earlier version.")
            lab = QLabel(f"<b>{head}</b><br>{note}<br><span style='color: gray'>{name}  {date}</span>")
            lab.setWordWrap(True)
            lab.setTextFormat(Qt.RichText)
            self.grid.addWidget(lab, r, 1)
            b1 = QPushButton("Open image")
            b1.clicked.connect(lambda _=False, p=jpg: self._open(p))
            b2 = QPushButton("Open folder")
            b2.clicked.connect(lambda _=False, p=jpg.parent: self._open(p))
            bb = QWidget()
            bl = QVBoxLayout(bb)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.addWidget(b1)
            bl.addWidget(b2)
            bl.addStretch(1)
            self.grid.addWidget(bb, r, 2)
        self.grid.setColumnStretch(1, 1)
