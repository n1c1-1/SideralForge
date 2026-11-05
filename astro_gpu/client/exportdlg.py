"""Export: the final image (and, if wanted, each layer and the masks), in a chosen format.

The image is the whole composition of Post at full resolution (each layer rendered by the engine on the whole grid),
composed in the render worker. Files go into the folder of the File line and nowhere else (the layers and the masks
into its subfolders ``layers`` and ``masks``); a folder inside the project's results is refused. The FITS file has
the astrometric solution (WCS) and the processing history in its header. What the engine does not have yet (XISF,
other colour profiles, the linear outputs) is shown in Spec mode only, disabled, marked ◇. The dialog is ``ExportDialog``.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from . import prefs
from .i18n import _, plural
from .imageview import slug
from .postengine import snapshot
from .services.render import ExportJob
from .widgets import WARN, label

#: display name -> the engine's format key (None: not in the engine yet)
FORMATS = {"TIFF 16-bit": "tiff16", "TIFF 32-bit float": "tiff32f", "FITS 32-bit float": "fits32f",
           "XISF 32-bit float": None, "PNG 16-bit": "png16", "JPEG": "jpeg"}
PROFILES = {"sRGB": True, "Adobe RGB (1998)": False, "Display P3": False, "Linear sRGB": False}
SIZES = {"Full resolution": 0, "Web (2048 px)": 2048, "Web (4096 px)": 4096}
#: the file each format gives (render/layers.export_rgb): name + suffix
SUFFIX = {"tiff16": "_16bit.tif", "tiff32f": "_32bit_float.tif", "png16": ".png", "jpeg": ".jpg", "fits32f": ".fits"}
KNOWN_ENDINGS = tuple(sorted(list(SUFFIX.values()) + [".tiff", ".jpeg", ".fit"], key=len, reverse=True))


def split_file(text: str) -> tuple[str, str]:
    """The File line as (folder, base name): the name loses the ending the format adds."""
    p = Path(text.strip())
    name = p.name
    for end in KNOWN_ENDINGS:
        if name.lower().endswith(end):
            name = name[:-len(end)]
            break
    return (p.parent.as_posix() if text.strip() else ""), name


def inside(folder: Path, other: Path) -> bool:
    try:
        folder.resolve().relative_to(other.resolve())
        return True
    except (ValueError, OSError):
        return False


class ExportDialog(QDialog):
    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.worker = view.worker
        s = view.s
        self.pname = getattr(s, "name", "") or "image"
        root = getattr(s, "root", None)
        self.root = Path(root) if root else None
        self.files: list[str] = []
        self.error = ""
        self.running = False
        spec = prefs.spec()
        self.setWindowTitle(_("Export"))
        self.setMinimumWidth(640)
        v = QVBoxLayout(self)
        v.addWidget(label(_("Export"), "h1"))

        def row(text, w, extra=None):
            h = QHBoxLayout()
            lb = QLabel(text)
            lb.setFixedWidth(150)
            h.addWidget(lb)
            h.addWidget(w, 1)
            if extra is not None:
                h.addWidget(extra)
            v.addLayout(h)
            return w

        self.fmt = QComboBox()
        self.fmt.setModel(QStandardItemModel(self.fmt))
        for name, key in FORMATS.items():
            if key is None and not spec:
                continue
            self.fmt.addItem(_(name) + ("  ◇" if key is None else ""), key)
            if key is None:
                self._disable(self.fmt, self.fmt.count() - 1)
        row(_("Format"), self.fmt)
        self.prof = QComboBox()
        self.prof.setModel(QStandardItemModel(self.prof))
        for name, ok in PROFILES.items():
            if not ok and not spec:
                continue
            self.prof.addItem(_(name) + ("" if ok else "  ◇"), name)
            if not ok:
                self._disable(self.prof, self.prof.count() - 1)
        row(_("Color profile"), self.prof)
        self.size_cb = QComboBox()
        for name, px in SIZES.items():
            self.size_cb.addItem(_(name), px)
        row(_("Size"), self.size_cb)
        self.dest = QLineEdit(self.default_file("tiff16"))
        self.browse = QPushButton("…")
        self.browse.setFixedWidth(max(30, self.browse.sizeHint().width()))
        self.browse.setToolTip(_("Choose the folder"))
        row(_("File"), self.dest, self.browse)
        self.warn = label("", "warn")
        self.warn.setWordWrap(True)
        v.addWidget(self.warn)
        v.addWidget(label(_("Include"), "h2"))
        self.c_wcs = self._check(v, _("Astrometric solution (WCS) in the header"), True)
        self.c_hist = self._check(v, _("Processing history (a file next to the image; in the header too for FITS)"),
                                  True)
        self.c_layers = self._check(v, _("Each layer as a separate file"), False)
        self.c_masks = self._check(v, _("The masks"), False)
        # the annotations of Post's view, drawn into the picture (PNG, JPEG, TIFF): handy to show what the image is
        show = dict(getattr(view, "ann_show", None) or {})
        ah = QHBoxLayout()
        ah.addWidget(QLabel(_("Include annotations:")))
        self.c_grid = QCheckBox(_("Sky grid"))
        self.c_grid.setChecked(bool(show.get("grid")))
        self.c_names = QCheckBox(_("Object names"))
        self.c_names.setChecked(bool(show.get("objects")))
        ah.addWidget(self.c_grid)
        ah.addWidget(self.c_names)
        ah.addStretch(1)
        v.addLayout(ah)
        self.spec_boxes = []
        if spec:
            for text in (_("Each workflow's output, linear (FITS)"),):
                c = self._check(v, text + "  ◇", False)
                c.setEnabled(False)
                c.setToolTip("◇ " + _("Not in the engine yet."))
                self.spec_boxes.append(c)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 100)
        self.bar.hide()
        v.addWidget(self.bar)
        self.status = label("", "small")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        bb = QHBoxLayout()
        bb.addStretch(1)
        self.cancel = QPushButton(_("Cancel"))
        self.cancel.clicked.connect(self.reject)
        self.go = QPushButton(_("Export"))
        self.go.setObjectName("primary")
        self.go.clicked.connect(self._export)
        bb.addWidget(self.cancel)
        bb.addWidget(self.go)
        v.addLayout(bb)
        self.fmt.currentIndexChanged.connect(self._format_changed)
        self.dest.textChanged.connect(self._check_dest)
        self.dest.textChanged.connect(self.dest.setToolTip)
        self.browse.clicked.connect(self._browse)
        self.worker.progress.connect(self._progress)
        self.worker.exported.connect(self._done)
        self._format_changed()

    # ------------------------------------------------------------ the fields
    @staticmethod
    def _disable(combo: QComboBox, i: int):
        it = combo.model().item(i)
        if it is not None:
            it.setFlags(it.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)

    @staticmethod
    def _check(layout, text, on) -> QCheckBox:
        c = QCheckBox(text)
        c.setChecked(on)
        layout.addWidget(c)
        return c

    def default_file(self, fmt: str) -> str:
        base = self.root if self.root is not None else Path(prefs.get("workspace")) / slug(self.pname)
        name = slug(self.pname)
        return (base / "export" / name / (name + SUFFIX[fmt])).as_posix()

    def format_key(self) -> str:
        return self.fmt.currentData() or "tiff16"

    def _format_changed(self, *_a):
        key = self.format_key()
        folder, name = split_file(self.dest.text())
        self.dest.setText(f"{folder}/{name}{SUFFIX[key]}" if folder else name + SUFFIX[key])
        # an option the format does not take shows unticked (greyed), and comes back as the user left it
        fits = key == "fits32f"
        for c, on in ((self.c_wcs, fits), (self.c_grid, not fits), (self.c_names, not fits)):
            if c.isEnabled() and not on:
                c.setProperty("wish", c.isChecked())
            if on and not c.isEnabled():
                c.setChecked(bool(c.property("wish")) if c.property("wish") is not None else c.isChecked())
            elif not on:
                c.setChecked(False)
            c.setEnabled(on)
        self.c_wcs.setToolTip("" if fits else _("Only the FITS file has a header."))
        for c in (self.c_grid, self.c_names):
            c.setToolTip(_("A FITS file stays clean: its header holds the astrometric solution.") if key == "fits32f"
                         else _("Drawn into the picture, as Post shows them."))
        self._check_dest()

    def _browse(self):
        folder, name = split_file(self.dest.text())
        d = QFileDialog.getExistingDirectory(self, _("Choose the folder"), folder or prefs.get("workspace"))
        if d:
            self.dest.setText(f"{Path(d).as_posix()}/{name or slug(self.pname)}{SUFFIX[self.format_key()]}")

    def problem(self) -> str:
        """Why the export cannot start (shown under the File line), or ''."""
        folder, name = split_file(self.dest.text())
        if not folder or not name:
            return _("Choose a folder and a name.")
        if self.root is not None and inside(Path(folder), self.root / "store"):
            return _("This folder is inside the project's results: choose another.")
        return ""

    def _check_dest(self, *_a):
        why = self.problem()
        folder, name = split_file(self.dest.text())
        replaces = (not why and folder and (Path(folder) / (name + SUFFIX[self.format_key()])).exists())
        self.warn.setText(why or (_("A file of this name exists: it will be replaced.") if replaces else ""))
        self.warn.setStyleSheet(f"color: {WARN};")
        if not self.running:
            self.go.setEnabled(not why)

    # ------------------------------------------------------------ the job
    def job(self) -> ExportJob:
        folder, name = split_file(self.dest.text())
        return ExportJob(layers=snapshot(self.view.layers), folder=folder, name=name, format=self.format_key(),
                         size=int(self.size_cb.currentData() or 0), wcs=self.c_wcs.isEnabled() and self.c_wcs.isChecked(),
                         history=self.c_hist.isChecked(), each_layer=self.c_layers.isChecked(),
                         masks=self.c_masks.isChecked(), project_name=self.pname,
                         annotations={"grid": self.c_grid.isEnabled() and self.c_grid.isChecked(),
                                      "objects": self.c_names.isEnabled() and self.c_names.isChecked()})

    def _export(self):
        if self.go.text() == _("Close"):
            self.accept()
            return
        self.start()

    def start(self):
        """Hand the export to the render worker (the images are composed there, not here)."""
        why = self.problem()
        if why:
            self._check_dest()
            return
        self.running = True
        self.error, self.files = "", []
        self.status.setText("")
        self.bar.setValue(0)
        self.bar.show()
        self.go.setEnabled(False)
        self.dest.setEnabled(False)
        self.worker.export(self.job())

    def _progress(self, f, _msg):
        if self.running:
            self.bar.setValue(int(100 * f))

    def _done(self, files, error):
        if not self.running:
            return
        self.running = False
        self.dest.setEnabled(True)
        self.go.setEnabled(True)
        if error:
            self.error = error
            self.bar.hide()
            self.status.setText(_("Export stopped.") if error == "stopped"
                                else _("The export failed: {why}").format(why=error))
            self.status.setStyleSheet(f"color: {WARN};")
            return
        self.files = files
        self.bar.setValue(100)
        folder, _name = split_file(self.dest.text())
        self.status.setText(plural(len(files), "{n} file written to {folder}", "{n} files written to {folder}",
                                   folder=folder))
        self.status.setStyleSheet("")
        self.go.setText(_("Close"))
        self.cancel.setEnabled(False)

    def reject(self):
        if self.running:
            self.worker.cancel_export()
            return
        self._disconnect()
        super().reject()

    def accept(self):
        self._disconnect()
        super().accept()

    def _disconnect(self):
        for sig, slot in ((self.worker.progress, self._progress), (self.worker.exported, self._done)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
