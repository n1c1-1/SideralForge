"""New project dialog: input folders, target, working folders - with sensible guesses from the lights folder."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.project import Inputs, Project, Target, list_frames

GENERIC = {"raw", "raws", "raw2", "signal", "lights", "light", "brut", "bruts", "ok", "capture", "captures", "images",
           "serie1", "serie2", "series", "fits", "arw", "data"}
GUESSES = {"darks": ("dark", "darks", "Dark", "Darks", "DARK"), "flats": ("flat", "flats", "Flat", "Flats", "FLAT"),
           "bias": ("bias", "offset", "offsets", "Bias", "Offset"),
           "flat_darks": ("flatdark", "flat_dark", "flat_darks", "darkflat", "dark_flat", "FlatDarks")}


MONTHS = {"jan", "janv", "janvier", "january", "fev", "fevr", "fevrier", "february", "feb", "mar", "mars", "march",
          "avr", "avril", "april", "apr", "mai", "may", "juin", "june", "jun", "juil", "juillet", "july", "jul",
          "aout", "august", "aug", "sept", "sep", "septembre", "september", "oct", "octobre", "october", "nov",
          "novembre", "november", "dec", "decembre", "december"}
NOISE_WORDS = {"astro", "photo", "photos", "session", "nuit", "night", "test"}


def target_guess(folder_name: str) -> str:
    """Object name from a folder name: drop dates, numbers, month names and generic words (NGC7822 is kept)."""
    import unicodedata
    words = [w for w in re.split(r"[_\s-]+", folder_name) if w]
    keep = []
    for w in words:
        plain = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode().lower()
        if re.fullmatch(r"[0-9.]+", plain) or plain in MONTHS or plain in NOISE_WORDS:
            continue
        keep.append(w)
    return " ".join(keep)


class FolderList(QWidget):
    def __init__(self, placeholder: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setMaximumHeight(58)
        self.list.setToolTip(placeholder)
        add, rem = QPushButton("Add..."), QPushButton("Remove")
        col = QVBoxLayout()
        col.addWidget(add)
        col.addWidget(rem)
        lay.addWidget(self.list, 1)
        lay.addLayout(col)
        add.clicked.connect(self._add)
        rem.clicked.connect(lambda: [self.list.takeItem(self.list.row(i)) for i in self.list.selectedItems()])
        self.on_change = None

    def _add(self):
        d = QFileDialog.getExistingDirectory(self, "Choose a folder")
        if d:
            self.add(d)
            if self.on_change:
                self.on_change()

    def add(self, d: str):
        # Windows paths are case-insensitive: 'dark', 'Dark' and 'DARK' are the same folder
        key = os.path.normcase(os.path.realpath(d))
        if key not in {os.path.normcase(os.path.realpath(f)) for f in self.folders()}:
            self.list.addItem(str(Path(d)))

    def folders(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.resize(760, 640)
        self.project: Project | None = None
        v = QVBoxLayout(self)
        intro = QLabel("Choose the folder of your light frames: the other folders, the project name and the working "
                       "folders are guessed and can be changed. Only colour (Bayer) camera RAW or FITS frames are "
                       "supported in this version.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        v.addWidget(intro)

        g1 = QGroupBox("Frames")
        f1 = QFormLayout(g1)
        self.lists = {k: FolderList(k) for k in ("lights", "darks", "flats", "bias", "flat_darks")}
        self.lists["lights"].on_change = self._guess
        f1.addRow("Light frames *", self.lists["lights"])
        f1.addRow("Darks", self.lists["darks"])
        f1.addRow("Flats", self.lists["flats"])
        f1.addRow("Bias / offsets", self.lists["bias"])
        f1.addRow("Flat darks", self.lists["flat_darks"])
        self.count = QLabel("")
        self.count.setObjectName("muted")
        f1.addRow("", self.count)
        v.addWidget(g1)

        g2 = QGroupBox("Target (helps the astrometry; optional with FITS files that contain the pointing)")
        f2 = QFormLayout(g2)
        self.target = QLineEdit()
        self.target.setPlaceholderText("e.g. NGC 7822, IC 1805, Heart Nebula")
        res = QPushButton("Resolve")
        res.clicked.connect(self._resolve)
        row = QHBoxLayout()
        row.addWidget(self.target, 1)
        row.addWidget(res)
        f2.addRow("Object name", row)
        self.ra, self.dec = QLineEdit(), QLineEdit()
        self.ra.setPlaceholderText("degrees (optional)")
        self.dec.setPlaceholderText("degrees (optional)")
        f2.addRow("Centre RA", self.ra)
        f2.addRow("Centre Dec", self.dec)
        self.focal = QLineEdit()
        self.focal.setPlaceholderText("mm, with reducer (optional: makes the astrometry immediate)")
        f2.addRow("Focal length", self.focal)
        v.addWidget(g2)

        g3 = QGroupBox("Project")
        f3 = QFormLayout(g3)
        self.name = QLineEdit()
        self.folder = self._path_row(f3, "Project folder *", True,
                                     "Results, intermediate files and the journal go here (a few GB).")
        f3.insertRow(0, "Name *", self.name)
        self.scratch = self._path_row(f3, "Fast scratch folder", True,
                                      "Calibrated frames cache (large). Prefer an internal SSD: it is read many times.")
        self.reference = self._path_row(f3, "Reference image", False,
                                        "Optional: your own earlier processing of these data, for side by side "
                                        "comparison and same-framing exports.")
        self.space = QLabel("")
        self.space.setObjectName("muted")
        f3.addRow("", self.space)
        v.addWidget(g3)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Create project")
        bb.button(QDialogButtonBox.Ok).setObjectName("primary")
        bb.accepted.connect(self._create)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _path_row(self, form, label, folder: bool, tip: str) -> QLineEdit:
        e = QLineEdit()
        e.setToolTip(tip)
        b = QPushButton("Browse...")
        row = QHBoxLayout()
        row.addWidget(e, 1)
        row.addWidget(b)

        def pick():
            if folder:
                d = QFileDialog.getExistingDirectory(self, label)
            else:
                d, _ = QFileDialog.getOpenFileName(self, label, "", "Images (*.png *.tif *.tiff *.jpg *.jpeg)")
            if d:
                e.setText(d)
                self._update_space()
        b.clicked.connect(pick)
        form.addRow(label, row)
        hint = QLabel(tip)
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        form.addRow("", hint)
        return e

    def _guess(self):
        lights = self.lists["lights"].folders()
        if not lights:
            return
        lp = Path(lights[0])
        parent = lp.parent
        for k, names in GUESSES.items():
            for nm in names:
                for base in (parent, lp):
                    c = base / nm
                    if c.is_dir() and list_frames([str(c)]):
                        self.lists[k].add(str(c))
        # the meaningful name is the first ancestor that is not a generic capture folder name
        named = lp
        while named.name.lower() in GENERIC and named.parent != named:
            named = named.parent
        if not self.name.text():
            self.name.setText(named.name)
        if not self.folder.text():
            self.folder.setText(str(named / "astro_gpu"))
        if not self.scratch.text():
            self.scratch.setText(str(Path.home() / "astro_gpu_scratch" / (self.name.text() or "project")))
        if not self.target.text():
            self.target.setText(target_guess(named.name))
        self._update_space()

    def _update_space(self):
        lights = list_frames(self.lists["lights"].folders())
        counts = {k: len(list_frames(w.folders())) for k, w in self.lists.items()}
        self.count.setText(", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items() if v))
        if lights and self.scratch.text():
            try:
                from ..io.frames import read_info
                inf = read_info(lights[0])
                need = len(lights) * inf.width * inf.height * 2 / 1e9
                root = Path(self.scratch.text()).anchor or str(Path.home())
                free = shutil.disk_usage(root).free / 1e9
                self.space.setText(f"Calibrated cache: about {need:.1f} GB on the scratch disk ({free:.0f} GB free)."
                                   + ("  Not enough space!" if need > free * 0.9 else ""))
            except Exception as e:  # noqa: BLE001
                self.space.setText(f"(could not read the first frame: {e})")

    def _resolve(self):
        try:
            from ..stages.astrometry import resolve_name
            ra, dec = resolve_name(self.target.text())
            self.ra.setText(f"{ra:.5f}")
            self.dec.setText(f"{dec:.5f}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Resolve", str(e))

    def _create(self):
        if not self.lists["lights"].folders() or not self.folder.text() or not self.name.text():
            QMessageBox.warning(self, "New project", "Light frames, a project folder and a name are required.")
            return
        root = Path(self.folder.text())
        if (root / "project.json").exists():
            QMessageBox.warning(self, "New project", "This folder already contains a project: open it instead.")
            return

        def fnum(e):
            try:
                return float(e.text()) if e.text().strip() else None
            except ValueError:
                return None
        inputs = Inputs(**{k: w.folders() for k, w in self.lists.items()})
        self.project = Project.create(root, self.name.text(), inputs,
                                      Target(self.target.text(), fnum(self.ra), fnum(self.dec), fnum(self.focal)),
                                      self.scratch.text() or None, self.reference.text())
        self.accept()


__all__ = ["NewProjectDialog", "Qt"]
