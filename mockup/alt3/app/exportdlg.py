"""Export: the final image (and, if wanted, each layer, the masks and the workflows' outputs), in a chosen format."""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                               QPushButton, QVBoxLayout)

import prefs
from i18n import _
from widgets import label

FORMATS = ["TIFF 16-bit", "TIFF 32-bit float", "FITS 32-bit float", "XISF 32-bit float", "PNG 16-bit", "JPEG"]
PROFILES = ["sRGB", "Adobe RGB (1998)", "Display P3", "Linear sRGB"]
SIZES = ["Full resolution", "Web (2048 px)", "Web (4096 px)"]


class ExportDialog(QDialog):
    def __init__(self, name="HeartSoul", parent=None):
        super().__init__(parent)
        self.setWindowTitle(_("Export"))
        self.setMinimumWidth(560)
        v = QVBoxLayout(self)
        v.addWidget(label(_("Export"), "h1"))

        def row(text, w):
            h = QHBoxLayout()
            lb = QLabel(text)
            lb.setFixedWidth(150)
            h.addWidget(lb)
            h.addWidget(w, 1)
            v.addLayout(h)
            return w

        fmt = QComboBox()
        fmt.addItems(FORMATS)
        row(_("Format"), fmt)
        prof = QComboBox()
        prof.addItems(PROFILES)
        row(_("Color profile"), prof)
        size = QComboBox()
        size.addItems([_(s) for s in SIZES])
        row(_("Size"), size)
        dest = QLineEdit(f"{prefs.get('workspace')}/{name}/export/{name}.tif")
        row(_("File"), dest)
        v.addWidget(label(_("Include"), "h2"))
        for text, on in ((_("Astrometric solution (WCS) in the header"), True),
                         (_("Processing history in the header"), True),
                         (_("Each layer as a separate file"), False), (_("The masks"), False),
                         (_("Each workflow's output, linear (FITS)"), False)):
            c = QCheckBox(text)
            c.setChecked(on)
            v.addWidget(c)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.hide()
        v.addWidget(self.bar)
        bb = QHBoxLayout()
        bb.addStretch(1)
        c = QPushButton(_("Cancel"))
        c.clicked.connect(self.reject)
        self.go = QPushButton(_("Export"))
        self.go.setObjectName("primary")
        self.go.clicked.connect(self._export)
        bb.addWidget(c)
        bb.addWidget(self.go)
        v.addLayout(bb)

    def _export(self):
        if self.go.text() == _("Close"):
            self.accept()
            return
        self.bar.show()
        self.go.setEnabled(False)
        self._t = QTimer(self)

        def tick():
            self.bar.setValue(self.bar.value() + 5)
            if self.bar.value() >= 100:
                self._t.stop()
                self.go.setText(_("Close"))
                self.go.setEnabled(True)
        self._t.timeout.connect(tick)
        self._t.start(30)
