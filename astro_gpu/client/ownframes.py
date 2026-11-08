"""A Calibration block placed in parallel: its own darks, flats, bias and dark flats, in its panel of Inspect. Each kind
follows its Session until it is changed here; the graph's panel only opens Inspect, the one place where they are chosen.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QListWidgetItem, QPushButton, QWidget

from . import prefs
from .i18n import _
from .icons import icon_only

KINDS = (("darks", "Darks"), ("flats", "Flats"), ("bias", "Bias"), ("flat_darks", "Dark flats"))


def own_frames(panel, node, p) -> None:
    """The section of a panel; nothing for a Calibration that reads its Session."""
    s = panel.s
    sid = s.own_calibration_night(p, node.sid)
    if sid is None:
        return
    if not panel.tune:
        b = QPushButton(_("Own frames…"))
        b.setToolTip(_("Opens the block in Inspect: its darks, flats and bias, added or taken out file by file"))
        b.clicked.connect(panel.openTune.emit)
        panel.bv.addWidget(b, 0, Qt.AlignLeft)
        panel.bv.addSpacing(6)
        return
    paths = s.same_calibration(p, sid)
    for kind, title in KINDS:
        _kind(panel, paths, sid, kind, title)
    panel.bv.addSpacing(8)


def _kind(panel, paths, sid: int, kind: str, title: str) -> None:
    from .sources import FileList
    from .widgets import label
    s = panel.s
    files = s.calibration_files(paths[0], sid, kind)
    follows = s.calibration_follows(paths[0], sid, kind)
    panel.bv.addWidget(panel.group_title(_(title) + "  ·  " + (str(len(files)) if files else _("none"))))
    if follows:
        panel.bv.addWidget(label(_("as Session"), "dim"))

    def out(chosen, k=kind):
        s.remove_calibration_files(paths, sid, k, chosen)
    if files:
        lst = FileList(out)
        for f in files:
            it = QListWidgetItem()
            it.setData(Qt.UserRole, f)
            lst.addItem(it)
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(2, 0, 2, 0)
            name = label(Path(f).name)
            name.setToolTip(f)
            h.addWidget(name, 1)
            b = icon_only(QPushButton(), "remove", _("Take this file out of this Calibration (the file is not touched)"))
            b.setFixedSize(26, 20)
            b.setStyleSheet("QPushButton { padding: 0px; }")
            b.clicked.connect(lambda _c=False, ff=f: out([ff]))
            h.addWidget(b)
            it.setSizeHint(row.sizeHint())
            lst.setItemWidget(it, row)
        lst.setFixedHeight(min(len(files), 8) * (lst.sizeHintForRow(0) or 24) + 6)
        panel.bv.addWidget(lst)
    bar = QHBoxLayout()
    add = QPushButton(_("Add files…"))
    add.clicked.connect(lambda _c=False, k=kind: _add(panel, paths, sid, k))
    bar.addWidget(add)
    if not follows:
        back = QPushButton(_("Use Session"))
        back.setToolTip(_("Back to the Session's files of this kind"))
        back.clicked.connect(lambda _c=False, k=kind: s.follow_session(paths, sid, k))
        bar.addWidget(back)
    bar.addStretch(1)
    panel.bv.addLayout(bar)


def _add(panel, paths, sid: int, kind: str) -> None:
    from ..core.project import COLOUR_EXT, FITS_EXT, RAW_EXT
    pats = " ".join(f"*{e}" for e in sorted(RAW_EXT | FITS_EXT | COLOUR_EXT | {".fz"}))
    got, _f = QFileDialog.getOpenFileNames(panel, _("Add files…"), prefs.last_folder(),
                                           _("Frames ({patterns})").format(patterns=pats) + ";;" + _("All files (*)"))
    if got:
        prefs.remember_folder(str(Path(got[0]).parent))
        panel.s.add_calibration_files(paths, sid, kind, got)
