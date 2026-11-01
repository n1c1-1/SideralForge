"""Browse: the lights and calibration frames of the project, their calibration matching, and a raw frames browser (grid,
viewer with filmstrip and blink, sortable list) where frames are kept or excluded; and Import folder.

The thumbnails and the frame shown large are made by the worker from the raw files (``api.frame_thumb``: one stretch for
all the lights of a session, so that a hazy frame looks darker when blinking). Nothing here reads a file.

Displayed texts are English and go through _() at display time (French: locale/fr/inspect.json).
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import api
from . import prefs
from .frames import fexp, night_text
from .i18n import _, number
from .services.inspect import FrameItem, ProjectFrames, add_scan, service_of
from .viewer import StaticSource, Viewer
from .widgets import NEW, OK, WARN, clear, hline, label, para

# (key, title, section): plain English, translated where shown
GROUPS = [("lights", "Lights", "Lights"), ("darks", "Darks", "Calibration"), ("flats", "Flats", "Calibration"),
          ("bias", "Bias", "Calibration"), ("flat_darks", "Dark flats", "Calibration")]
KIND_WORD = {"lights": "lights", "darks": "darks", "flats": "flats", "bias": "bias", "flat_darks": "dark flats"}
KIND_COUNT = {"lights": "{n} lights", "darks": "{n} darks", "flats": "{n} flats", "bias": "{n} bias",
              "flat_darks": "{n} dark flats"}


def fnum(x, d=1) -> str:
    return number(x, d)


class NumItem(QTableWidgetItem):
    """A table cell sorted by its number, shown with the decimal separator of the language."""

    def __init__(self, value, text=None):
        super().__init__(text if text is not None else str(value))
        self.v = value

    def __lt__(self, other):
        return self.v < getattr(other, "v", 0)


class GroupRow(QFrame):
    clicked = Signal(str)

    def __init__(self, key, title, sub, selected, empty, parent=None):
        super().__init__(parent)
        self.key = key
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(("QFrame { background: #2b3446; border-radius: 4px; }" if selected else
                            "QFrame { background: transparent; border-radius: 4px; } QFrame:hover "
                            "{ background: #232831; }") + " QLabel { background: transparent; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 5, 8, 5)
        v.setSpacing(0)
        a = QLabel(title)
        a.setStyleSheet("font-weight: 600;" + (" color: #6b7280;" if empty else ""))
        b = QLabel(sub)
        b.setObjectName("small")
        b.setWordWrap(True)
        v.addWidget(a)
        v.addWidget(b)

    def mousePressEvent(self, e):                                    # noqa: N802 - Qt's name
        self.clicked.emit(self.key)


def pixmap(arr: np.ndarray) -> QPixmap:
    a = np.ascontiguousarray(arr)
    h, w = a.shape[:2]
    return QPixmap.fromImage(QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy())


_PLACEHOLDER: list = []


def placeholder() -> QIcon:
    """A dark thumbnail: the lists measure their items from it, before the real thumbnails come."""
    if not _PLACEHOLDER:
        pm = QPixmap(192, 128)
        pm.fill(QColor("#232831"))
        _PLACEHOLDER.append(QIcon(pm))
    return _PLACEHOLDER[0]


def join_and(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " " + _("and") + " " + parts[-1]


# ---------------------------------------------------------------------------------------------- Import folder
class ImportDialog(QDialog):
    """Reads a folder (the lights, and the darks, flats and bias found next to it): a progress bar, then what was found
    and the sessions it makes. The scan is the worker's; Cancel stops it."""

    def __init__(self, model, folder: str, new_session: bool = False, parent=None, service=None):
        super().__init__(parent)
        self.s, self.folder, self.new_session = model, folder, new_session
        self.service = service or service_of(model)
        self.scan = None
        self.setWindowTitle(_("Add session") if new_session else _("Add folder"))
        self.setMinimumWidth(560)
        v = QVBoxLayout(self)
        v.addWidget(label(_("Add session") if new_session else _("Add folder"), "h1"))
        v.addWidget(label(_("Folder: {folder}").format(folder=folder), "dim", wrap=True))
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        v.addWidget(self.bar)
        self.cur = label(_("Reading the files…"), "small")
        v.addWidget(self.cur)
        self.found = label("", "", wrap=True)
        self.found.setTextFormat(Qt.RichText)
        v.addWidget(self.found)
        self.groups = label("", "small", wrap=True)
        v.addWidget(self.groups)
        self.warn = label("", "warn", wrap=True)
        v.addWidget(self.warn)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton(_("Cancel"))
        cancel.clicked.connect(self._cancel)
        self.ok = QPushButton(_("Add to project"))
        self.ok.setObjectName("primary")
        self.ok.setEnabled(False)
        self.ok.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(self.ok)
        v.addLayout(row)
        sv = self.service
        sv.scanProgress.connect(self._progress)
        sv.scanDone.connect(self._done)
        sv.scanFailed.connect(self._failed)
        self.token = sv.scan(folder)

    def _cancel(self):
        self.service.cancel_scan(self.token)
        self.reject()

    def _progress(self, token: int, f: float, text: str):
        if token != self.token:
            return
        self.bar.setValue(int(f * 1000))
        self.cur.setText(_("Reading the files: {pct} %").format(pct=int(100 * f)))

    def _failed(self, token: int, why: str):
        if token != self.token:
            return
        self.cur.setText("")
        self.warn.setText(_("The folder cannot be read: {why}").format(why=why))

    def _done(self, token: int, scan):
        if token != self.token:
            return
        self.scan = scan
        n = {k: len(scan.frames.get(k, [])) for k, _t, _s in GROUPS}
        total = sum(n.values())
        self.bar.setValue(1000)
        self.cur.setText(_("{n} files read.").format(n=total))
        if not n["lights"]:
            self.warn.setText(_("No light frames in this folder."))
            return
        parts = [_(KIND_COUNT[k]).format(n=n[k]) for k, _t, _s in GROUPS if n[k]]
        missing = [_(KIND_WORD[k]) for k in ("bias", "flat_darks") if not n[k]]
        txt = _("<b>Found</b>: {parts}.").format(parts=join_and(parts))
        if missing:
            txt += " " + _("No {what}.").format(what=", ".join(missing))
        self.found.setText(txt)
        gs = api.group_sessions(scan.frames["lights"], self.service.criteria)
        names = " · ".join(_("{name} ({n} lights)").format(name=g.name, n=g.count) for g in gs)
        self.groups.setText(_("{n} session(s): {names}").format(n=len(gs), names=names)
                            if not self.new_session else _("One session: {name}").format(name=scan.name))
        if scan.errors:
            self.warn.setText(_("{n} files could not be read.").format(n=len(scan.errors)))
        self.ok.setEnabled(True)

    def finish_now(self):
        """Wait for the scan (tests and scripted walkthroughs)."""
        self.service.wait(120)


def import_folder(model, parent=None, folder: str | None = None, new_session: bool = False, service=None) -> bool:
    """Import folder… (File menu, the Frames block, Import folder as new project) and Add session…: ask for a folder when none
    is given, read it, and on Add to project put its frames into the project, which is saved and read again."""
    svc = service or service_of(model)
    if folder is None:
        folder = QFileDialog.getExistingDirectory(parent, _("Add session") if new_session else _("Add folder"),
                                                  prefs.get("workspace"))
        if not folder:
            return False
    norm = os.path.normcase(os.path.normpath(folder))
    for ses in model.p.sessions:
        if any(os.path.normcase(os.path.normpath(model.p.abspath(x))) == norm for x in ses.lights):
            QMessageBox.information(parent, _("Add folder"), _("This folder is already in the project."))
            return False
    d = ImportDialog(model, folder, new_session, parent, svc)
    ok = d.exec()
    scan = d.scan
    d.deleteLater()
    if not ok or scan is None:
        return False
    if new_session:
        made = [api.add_session(model.p, name=scan.name)]
        api.add_frames(model.p, scan, session=made[0].id)
    else:
        made = add_scan(model.p, scan, svc.criteria)
    model.log('Folder "{folder}" imported · {lights} lights', folder=folder, lights=len(scan.frames.get("lights", [])))
    touch = getattr(model, "_touch", None)
    if touch is not None:
        touch()                                       # the model saves the definition 300 ms after
    model.load_sources()
    return bool(made)


# ---------------------------------------------------------------------------------------------- Browse
class SourcesView(QWidget):
    goChain = Signal(str)
    imported = Signal()

    def __init__(self, project, parent=None, service=None):
        super().__init__(parent)
        self.s = project
        self.service = service or service_of(project)
        self.data: ProjectFrames | None = None
        self.items: list[FrameItem] = []
        self.group = "lights"
        self.mode = 1                                      # 0 grid, 1 viewer, 2 list
        self.cur = 0
        self.sort = "time"
        self._icons: dict[str, list[QListWidgetItem]] = {}
        self._group_token = 0
        self._frame_token = 0
        self._loaded_group = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.pages = QStackedWidget()
        outer.addWidget(self.pages)
        self.pages.addWidget(self._empty_page())
        main = QWidget()
        self.pages.addWidget(main)
        root = QHBoxLayout(main)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        # ---- left: groups and calibration matching
        left = QWidget()
        left.setObjectName("panel")
        left.setFixedWidth(300)
        self.lv = QVBoxLayout(left)
        self.lv.setContentsMargins(10, 12, 10, 10)
        root.addWidget(left)
        # ---- centre: browser
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(10, 6, 10, 6)
        self.mode_btns = []
        g = QButtonGroup(self)
        for i, t in enumerate(("Grid", "Viewer", "List")):
            b = QPushButton(_(t))
            b.setCheckable(True)
            b.setObjectName("toggle")
            b.clicked.connect(lambda _x=False, k=i: self.set_mode(k))
            g.addButton(b)
            bar.addWidget(b)
            self.mode_btns.append(b)
        bar.addSpacing(12)
        bar.addWidget(label(_("Sort by"), "dim"))
        self.sort_cb = QComboBox()
        self.sort_cb.currentIndexChanged.connect(self._sort_changed)
        bar.addWidget(self.sort_cb)
        self.blink = QPushButton("▶  " + _("Blink"))
        self.blink.setCheckable(True)
        self.blink.setObjectName("toggle")
        self.blink.setToolTip(_("Plays the frames at 5 frames/s in the viewer (Space)"))
        self.blink.toggled.connect(self._blink)
        bar.addWidget(self.blink)
        bar.addStretch(1)
        self.count = label("", "dim")
        bar.addWidget(self.count)
        mv.addLayout(bar)
        self.info = label("", "small")
        self.info.setContentsMargins(12, 0, 12, 4)
        mv.addWidget(self.info)
        self.stack = QStackedWidget()
        # grid
        self.grid = QListWidget()
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setIconSize(QSize(176, 118))
        self.grid.setGridSize(QSize(196, 160))
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setMovement(QListView.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setSelectionMode(QAbstractItemView.SingleSelection)
        self.grid.itemDoubleClicked.connect(lambda it: (self._goto(it.data(Qt.UserRole)), self.set_mode(1)))
        self.grid.itemClicked.connect(lambda it: self._goto(it.data(Qt.UserRole)))
        self.stack.addWidget(self.grid)
        # viewer + filmstrip
        vw = QWidget()
        vv = QVBoxLayout(vw)
        vv.setContentsMargins(0, 0, 0, 0)
        vv.setSpacing(0)
        self.viewer = Viewer()
        vv.addWidget(self.viewer, 1)
        nav = QHBoxLayout()
        nav.setContentsMargins(10, 4, 10, 4)
        self.prev_b, self.next_b = QPushButton("◀"), QPushButton("▶")
        self.prev_b.setToolTip(_("Previous frame (←)"))
        self.next_b.setToolTip(_("Next frame (→)"))
        self.prev_b.clicked.connect(lambda: self.step(-1))
        self.next_b.clicked.connect(lambda: self.step(1))
        self.keep_b = QPushButton()
        self.keep_b.setToolTip(_("Keep or exclude this frame (X)"))
        self.keep_b.clicked.connect(self.toggle_keep)
        self.zone_b = QPushButton(_("Draw test area") + "  ◇")      # ◇: Preview on a region, Spec mode only
        self.zone_b.setToolTip("◇ " + _("Not in the engine yet."))
        self.zone_b.setEnabled(False)
        nav.addWidget(self.prev_b)
        nav.addWidget(self.next_b)
        nav.addWidget(self.keep_b)
        nav.addStretch(1)
        nav.addWidget(self.zone_b)
        vv.addLayout(nav)
        self.strip = QListWidget()
        self.strip.setViewMode(QListView.IconMode)
        self.strip.setFlow(QListView.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setIconSize(QSize(120, 80))
        self.strip.setGridSize(QSize(128, 100))
        self.strip.setFixedHeight(118)
        self.strip.setMovement(QListView.Static)
        self.strip.setUniformItemSizes(True)
        self.strip.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.strip.itemClicked.connect(lambda it: self._goto(it.data(Qt.UserRole)))
        vv.addWidget(self.strip)
        self.stack.addWidget(vw)
        # list
        self.table = QTableWidget()
        self.table.setStyleSheet("QTableWidget { background: #1c1f26; gridline-color: #2a2f38; } "
                                 "QHeaderView::section { background: #23272f; border: none; padding: 4px; }")
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.cellDoubleClicked.connect(lambda r, c: (self._goto(self.table.item(r, 0).data(Qt.UserRole)),
                                                           self.set_mode(1)))
        self.table.cellClicked.connect(lambda r, c: self._table_click(r, c))
        self.stack.addWidget(self.table)
        mv.addWidget(self.stack, 1)
        root.addWidget(mid, 1)
        # ---- right: properties
        right = QWidget()
        right.setObjectName("panel")
        right.setFixedWidth(320)
        self.rv = QVBoxLayout(right)
        self.rv.setContentsMargins(12, 12, 12, 12)
        root.addWidget(right)

        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.step(1))
        self._table_timer = QTimer(self)                   # the list is built again once the levels came
        self._table_timer.setSingleShot(True)
        self._table_timer.setInterval(250)
        self._table_timer.timeout.connect(lambda: self.fill_table(self.order()) if self.items else None)
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(120)
        self._thumb_timer.timeout.connect(self._want_thumbs)
        for bar_ in (self.grid.verticalScrollBar(), self.strip.horizontalScrollBar()):
            bar_.valueChanged.connect(lambda _v: self._thumb_timer.start())
        for seq, fn in (("Left", lambda: self.step(-1)), ("Right", lambda: self.step(1)), ("X", self.toggle_keep),
                        ("Space", self.blink.toggle)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        sv = self.service
        sv.groupReady.connect(self._group_ready)
        sv.framesReady.connect(self._frames_ready)
        sv.thumbReady.connect(self._thumb)
        sv.frameReady.connect(self._frame_ready)
        sv.statsReady.connect(self._stats_ready)
        self._fill_sort()
        self.set_mode(1)

    # ------------------------------------------------------------ the empty project
    def _empty_page(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setAlignment(Qt.AlignCenter)
        v.addWidget(label(_("No frames yet"), "h1"), 0, Qt.AlignHCenter)
        v.addWidget(para(_("Add the folder of the lights. Darks, flats and bias are searched next to it and sorted by "
                           "type; their match with the lights (exposure, ISO, levels) is checked and reported. Each "
                           "frame can then be browsed and excluded.")), 0, Qt.AlignHCenter)
        v.addSpacing(10)
        b = QPushButton(_("Add folder…"))
        b.setObjectName("primary")
        b.clicked.connect(self.import_folder)
        v.addWidget(b, 0, Qt.AlignHCenter)
        return w

    def import_folder(self):
        if import_folder(self.s, self.window(), service=self.service):
            self.imported.emit()

    def add_session(self):
        if import_folder(self.s, self.window(), new_session=True, service=self.service):
            self.imported.emit()

    # ------------------------------------------------------------ data
    def refresh(self):
        """Show what the project holds; ask the worker for the headers and the frames of the group."""
        loaded = self.s.sources_loaded
        self.pages.setCurrentIndex(1 if loaded else 0)
        self.zone_b.setVisible(prefs.spec() and self.group == "lights")
        if not loaded:
            return
        wid = self.s.path.pid if self.s.path is not None else None
        self._group_token = self.service.request_group(wid, self.group)
        self.service.request_frames(wid)
        self.build_left()
        self.build_right()

    def _frames_ready(self, data: ProjectFrames):
        self.data = data
        if self.pages.currentIndex() == 1 and self.isVisibleTo(self.window()):
            self.build_left()
            self.build_right()

    def _group_ready(self, kind: str, token: int, items: list):
        if kind != self.group or token != self._group_token:
            return
        same = [it.path for it in items] == [it.path for it in self.items]
        self.items = items
        self.cur = min(self.cur, max(len(items) - 1, 0))
        if same and self.grid.count() == len(items):
            self._marks()
            self.fill_table(self.order())
            self.update_count()
            self.build_left()
            self.build_right()
            return
        self.fill()
        self._goto(self.cur)
        if self.group != "lights":
            self.service.request_stats_batch([x.path for x in items])

    def order(self):
        it = self.items
        idx = list(range(len(it)))
        if self.sort == "fwhm":
            idx.sort(key=lambda i: it[i].fwhm if it[i].fwhm is not None else 0.0)
        elif self.sort == "transparency":
            idx.sort(key=lambda i: it[i].transparency if it[i].transparency is not None else 0.0)
        elif self.sort == "level":
            idx.sort(key=lambda i: it[i].level if it[i].level is not None else 0.0)
        else:
            idx.sort(key=lambda i: (it[i].time or "", it[i].name))
        return idx

    @property
    def p90(self) -> float:
        tr = [x.transparency for x in self.items if x.transparency is not None and np.isfinite(x.transparency)]
        return float(np.percentile(tr, 90)) if tr else 1.0

    def black(self) -> float:
        return float(self.data.camera.get("black") or 0) if self.data is not None else 0.0

    def excluded(self, i) -> bool:
        """Left out of the stack: by hand, or by the thresholds of Selection."""
        return bool(self.items[i].excluded or self.items[i].status)

    def by_hand(self, i) -> bool:
        return bool(self.items[i].excluded)

    # ------------------------------------------------------------ left panel
    def count_of(self, key: str) -> int:
        if self.data is not None:
            return self.data.counts.get(key, 0)
        return sum(ses.get({"lights": "lights", "darks": "darks", "flats": "flats", "bias": "bias",
                            "flat_darks": "dark_flats"}[key], 0) for ses in self.s.sessions)

    def build_left(self):
        clear(self.lv)
        section = None
        d = self.data
        row0 = d.sessions[0] if d is not None and d.sessions else None
        for key, title, sec in GROUPS:
            if sec != section:
                section = sec
                if sec == "Calibration":
                    self.lv.addSpacing(8)
                self.lv.addWidget(label(_(sec), "h2"))
            n = self.count_of(key)
            if key == "lights":
                sub = _("{n} lights · {exp} · ISO {iso} · night of {day}/{month}").format(
                    n=n, exp=fexp(row0.exposure) if row0 else "–", iso=int(row0.iso) if row0 and row0.iso else "–",
                    day=night_text(row0.first)[:2].strip() if row0 and row0.first else "–",
                    month=row0.first[5:7] if row0 and row0.first else "–")
            elif n:
                folder = (d.folders.get(key) or [""])[0] if d is not None else ""
                sub = _("{n} frames · {folder}").format(n=n, folder=folder)
            else:
                sub = _("none · optional")
            row = GroupRow(key, f"{_(title)} · {n}" if n else _(title), sub, key == self.group, not n)
            row.clicked.connect(self.select_group)
            self.lv.addWidget(row)
            if key == "lights":
                b = QPushButton(_("+ Add lights (another night)…"))
                b.setObjectName("link")
                b.clicked.connect(self.add_session)
                self.lv.addWidget(b)
        self.lv.addSpacing(10)
        self.lv.addWidget(hline())
        self.lv.addWidget(label(_("Calibration matching"), "h2"))
        for ok, text in self.checks():
            row = QHBoxLayout()
            ic = QLabel("✓" if ok is True else ("!" if ok is False else "i"))
            ic.setFixedWidth(14)
            ic.setStyleSheet(f"color: {OK if ok is True else (WARN if ok is False else '#8b93a1')}; font-weight: 700;")
            row.addWidget(ic, 0, Qt.AlignTop)
            row.addWidget(label(text, "small", wrap=True), 1)
            self.lv.addLayout(row)
        if prefs.spec():
            row = QHBoxLayout()
            ic = QLabel("◇")
            ic.setFixedWidth(14)
            ic.setStyleSheet(f"color: {NEW}; font-weight: 700;")
            row.addWidget(ic, 0, Qt.AlignTop)
            t = label(_("Library masters, sensor temperature and rotation checks."), "small", wrap=True)
            t.setEnabled(False)
            t.setToolTip("◇ " + _("Not in the engine yet."))
            row.addWidget(t, 1)
            self.lv.addLayout(row)
        self.lv.addStretch(1)

    def checks(self):
        """(ok, text) of the calibration matching: True a good sign, False a thing to look at, None a note."""
        d = self.data
        out = []
        if d is None:
            return [(None, _("Reading the frames…"))]
        ref = d.sessions[0] if d.sessions else None
        if d.darks and ref is not None:
            same = d.darks.get("exposure") == ref.exposure and d.darks.get("iso") == ref.iso
            t = (_("Darks: exposure {exp}, ISO {iso}, same as the lights.") if same else
                 _("Darks: exposure {exp}, ISO {iso}, different from the lights."))
            out.append((same, t.format(exp=fexp(d.darks.get("exposure")), iso=int(d.darks.get("iso") or 0))))
            if d.dark_level is not None:
                out.append((True, _("Darks: median level {level} ADU, nominal black level {black} ADU.").format(
                    level=fnum(d.dark_level, 0), black=fnum(self.black(), 0))))
        if d.flats:
            pct = d.flat_pct
            n = d.counts.get("flats", 0)
            if pct is not None:
                out.append((None, _("Flats: {n} frames at {pct} % of saturation, exposure {exp}.").format(
                    n=n, pct=fnum(pct, 0), exp=fexp(d.flats.get("exposure")))))
            else:
                out.append((None, _("Flats: {n} frames, exposure {exp}.").format(n=n, exp=fexp(d.flats.get("exposure")))))
        if not d.counts.get("bias") and not d.counts.get("flat_darks"):
            out.append((None, _("No bias, no dark flats: the flats are corrected with the camera black level "
                                "({black} ADU).").format(black=int(self.black() or 0))))
        n_ex = len(self.s.frames_excluded)
        if n_ex:
            t = (_("{n} frame excluded by hand: stacking to recompute.") if n_ex == 1 else
                 _("{n} frames excluded by hand: stacking to recompute."))
            out.append((False, t.format(n=n_ex)))
        return out

    def select_group(self, key):
        if not self.count_of(key):
            QMessageBox.information(self, _("Browse"), _("No frames in this group. Use Import folder… to add some."))
            return
        self.group = key
        self.cur = 0
        self.items = []
        self.blink.setChecked(False)
        self._fill_sort()
        self.build_left()
        self.fill()
        self.refresh()

    def _fill_sort(self):
        self.sort_cb.blockSignals(True)
        self.sort_cb.clear()
        opts = [("time", "time")]
        if self.group == "lights":
            opts += [("fwhm", "FWHM"), ("transparency", "transparency")]
        else:
            opts.append(("level", "level"))
        for k, t in opts:
            self.sort_cb.addItem(t if k == "fwhm" else _(t), k)             # FWHM stays FWHM, as in the tables
        self.sort = "time"
        self.sort_cb.blockSignals(False)

    # ------------------------------------------------------------ browser
    def _item_text(self, i):
        f = self.items[i]
        if self.group == "lights":
            tr = f"{100 * f.transparency / self.p90:.0f} %" if f.transparency is not None else "–"
            t = f"{f.name}\n{(fnum(f.fwhm) + ' px') if f.fwhm is not None else '–'} · {tr}"
        else:
            t = f"{f.name}\n{(fnum(f.level, 0) + ' ADU') if f.level is not None else (f.time[:10] if f.time else '')}"
        return ("✕ " if self.excluded(i) else "") + t

    def fill(self):
        idx = self.order()
        self._icons = {}
        for w in (self.grid, self.strip):
            w.clear()
            for i in idx:
                it = QListWidgetItem(placeholder(), self._item_text(i) if w is self.grid else "")
                it.setData(Qt.UserRole, i)
                it.setToolTip(self._item_text(i).replace("\n", " · "))
                if self.excluded(i):
                    it.setForeground(QColor("#e06c6c"))
                th = self.service.thumb_cache.get(self.items[i].path)
                if th is not None:
                    it.setIcon(QIcon(pixmap(th)))
                else:
                    self._icons.setdefault(self.items[i].path, []).append(it)
                w.addItem(it)
        self.fill_table(idx)
        self.update_count()
        self._thumb_timer.start()

    def _marks(self):
        """The excluded frames are shown as such, in place."""
        for w in (self.grid, self.strip):
            for r in range(w.count()):
                it = w.item(r)
                i = it.data(Qt.UserRole)
                it.setForeground(QColor("#e06c6c") if self.excluded(i) else QColor("#d8dce4"))
                if w is self.grid:
                    it.setText(self._item_text(i))

    def _thumb(self, path: str, arr):
        for it in self._icons.pop(path, []):
            it.setIcon(QIcon(pixmap(arr)))

    def _want_thumbs(self):
        """Thumbnails for the frames in view and around the current one (the worker makes them in this order)."""
        if not self.items or self.pages.currentIndex() != 1:
            return
        idx = self.order()
        w = self.grid if self.mode == 0 else self.strip
        c = w.indexAt(QPoint(w.viewport().width() // 2, w.viewport().height() // 2)).row()
        pos = c if c >= 0 else (idx.index(self.cur) if self.cur in idx else 0)
        lo, hi = max(0, pos - 20), min(len(idx), pos + 60)
        order = sorted(range(lo, hi), key=lambda r: abs(r - pos))
        cur_pos = idx.index(self.cur) if self.cur in idx else pos
        want = [self.items[idx[r]].path for r in [cur_pos] + order]
        self.service.request_thumbs(list(dict.fromkeys(want)), balance=self.group == "lights")

    def fill_table(self, idx):
        it = self.items
        if self.group == "lights":
            cols = ["File", "Time", "FWHM (px)", "Transparency (%)", "Weight", "Level (ADU)", "Kept"]
        else:
            cols = ["File", "Date", "Level (ADU)", "Noise (ADU)", "Kept"]
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([_(c) for c in cols])
        self.table.setRowCount(len(idx))
        for r, i in enumerate(idx):
            f = it[i]
            out = self.excluded(i)
            keep = _("no") if out else _("yes")
            if self.group == "lights":
                tr = 100 * f.transparency / self.p90 if f.transparency is not None else 0
                cells = [QTableWidgetItem(f.name), QTableWidgetItem((f.time or "").replace("T", " ")[:19]),
                         NumItem(f.fwhm or 0, fnum(f.fwhm, 2) if f.fwhm is not None else "–"),
                         NumItem(tr, f"{tr:.0f}" if f.transparency is not None else "–"),
                         NumItem(f.weight or 0, fnum(f.weight, 2) if f.weight is not None else "–"),
                         NumItem(f.level or 0, fnum(f.level) if f.level is not None else "–"),
                         QTableWidgetItem(keep)]
            else:
                st = self.service.stats.get(f.path)
                cells = [QTableWidgetItem(f.name), QTableWidgetItem((f.time or "")[:10]),
                         NumItem(st["level"] if st else 0, fnum(self.black() + st["level"], 0) if st else "–"),
                         NumItem(st["noise"] if st else 0, fnum(st["noise"], 2) if st else "–"),
                         QTableWidgetItem(keep)]
            cells[0].setData(Qt.UserRole, i)
            for c, c_item in enumerate(cells):
                if out:
                    c_item.setForeground(QColor("#e06c6c"))
                self.table.setItem(r, c, c_item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def _table_click(self, r, c):
        i = self.table.item(r, 0).data(Qt.UserRole)
        self._goto(i)
        if c == self.table.columnCount() - 1:
            self.toggle_keep()

    def update_count(self):
        n_ex = sum(self.excluded(i) for i in range(len(self.items)))
        noun = {"lights": "lights", "darks": "darks", "flats": "flats"}.get(self.group, "frames")
        self.count.setText(_("{n} {noun} · {x} excluded").format(n=len(self.items), noun=_(noun), x=n_ex))

    def _sort_changed(self, k):
        self.sort = self.sort_cb.itemData(k) or "time"
        self.fill()
        self._goto(self.cur)

    def set_mode(self, k):
        self.mode = k
        self.mode_btns[k].setChecked(True)
        self.stack.setCurrentIndex(k)
        self._goto(self.cur)
        self._thumb_timer.start()

    def step(self, d):
        idx = self.order()
        if not idx:
            return
        pos = idx.index(self.cur) if self.cur in idx else 0
        self._goto(idx[(pos + d) % len(idx)])

    def _goto(self, i):
        if not self.items:
            return
        self.cur = int(min(max(int(i), 0), len(self.items) - 1))
        f = self.items[self.cur]
        if self.mode == 1:
            self._frame_token = self.service.request_frame(f.path, 1280, self.group == "lights")
            self._prefetch()
        for w in (self.grid, self.strip):
            for r in range(w.count()):
                if w.item(r).data(Qt.UserRole) == self.cur:
                    w.setCurrentRow(r)
                    w.scrollToItem(w.item(r))
                    break
        d = self.data
        row0 = d.sessions[0] if d is not None and d.sessions else None
        if self.group == "lights":
            tr = f"{100 * f.transparency / self.p90:.0f}" if f.transparency is not None else "–"
            t = _("{file} · {time} · {exp} · ISO {iso} · FWHM {fwhm} px · transparency {transp} % of the best · "
                  "weight {weight}").format(
                file=f.name, time=(f.time or "")[11:16], exp=fexp(row0.exposure) if row0 else "–",
                iso=int(row0.iso) if row0 and row0.iso else "–", fwhm=fnum(f.fwhm, 2) if f.fwhm is not None else "–",
                transp=tr, weight=fnum(f.weight, 2) if f.weight is not None else "–")
        else:
            st = self.service.stats.get(f.path)
            t = (_("{file} · {date} · level {level} ADU · noise {noise} ADU").format(
                file=f.name, date=(f.time or "")[:10], level=fnum(self.black() + st["level"], 0),
                noise=fnum(st["noise"], 2)) if st else f"{f.name} · {(f.time or '')[:10]}")
        self.info.setText(t + (" · " + _("EXCLUDED") if self.by_hand(self.cur) else ""))
        self.keep_b.setText(_("Keep this frame") if self.by_hand(self.cur) else _("Exclude this frame  (X)"))
        self.keep_b.setEnabled(self.group == "lights" and f.session is not None)
        if self.group != "lights":
            self.keep_b.setToolTip("◇ " + _("Not in the engine yet.") if prefs.spec() else "")
            self.keep_b.setVisible(prefs.spec())
        else:
            self.keep_b.setVisible(True)
        if self.group != "lights":
            self.service.request_stats(f.path)
        self.build_right()

    def _prefetch(self):
        """The frames the blink will show next, made in the background."""
        idx = self.order()
        if not idx or self.cur not in idx:
            return
        pos = idx.index(self.cur)
        nxt = [self.items[idx[(pos + k) % len(idx)]].path for k in range(1, 9)]
        self.service.prefetch_frames(nxt, balance=self.group == "lights")

    def _frame_ready(self, token: int, path: str, arr):
        if token != self._frame_token or self.mode != 1:
            return
        self.viewer.set_source(StaticSource(arr), keep_view=True)

    def _stats_ready(self, path: str, st: dict):
        if not self.items:
            return
        if path == self.items[min(self.cur, len(self.items) - 1)].path:
            self.build_right()
            self._update_info_stats()
        if self.group != "lights":
            self._table_timer.start()

    def _update_info_stats(self):
        f = self.items[self.cur]
        st = self.service.stats.get(f.path)
        if st and self.group != "lights":
            self.info.setText(_("{file} · {date} · level {level} ADU · noise {noise} ADU").format(
                file=f.name, date=(f.time or "")[:10], level=fnum(self.black() + st["level"], 0),
                noise=fnum(st["noise"], 2)))

    def toggle_keep(self):
        if not self.items or self.group != "lights":
            return
        f = self.items[self.cur]
        if f.session is None:
            return
        want = not f.excluded
        self.s.set_excluded([(f.session, f.name)], want)
        touch = getattr(self.s, "_touch", None)
        if touch is not None:
            touch()                                       # the model does not mark an exclusion as an edit
        f.excluded = want
        self._marks()
        self.fill_table(self.order())
        self.update_count()
        self.build_left()
        self._goto(self.cur)

    def _blink(self, on):
        if on:
            self.set_mode(1)
            self.timer.start(200)
        else:
            self.timer.stop()

    # ------------------------------------------------------------ right panel
    def build_right(self):
        clear(self.rv)
        d = self.data
        self.rv.addWidget(label(_("Properties"), "h2"))
        rows = []
        row0 = d.sessions[0] if d is not None and d.sessions else None
        if self.group == "lights" and d is not None and row0 is not None:
            cam = d.camera
            n = d.counts.get("lights", 0)
            hours = n * (row0.exposure or 0) / 3600
            rows = [(_("Camera"), cam.get("camera") or "–"),
                    (_("Sensor"), _("{w} × {h} px, {cfa} pattern").format(w=cam.get("width"), h=cam.get("height"),
                                                                          cfa=cam.get("cfa"))),
                    (_("Exposure"), fexp(row0.exposure)), ("ISO", f"{int(row0.iso)}" if row0.iso else "–"),
                    (_("First frame"), (min((r.first for r in d.sessions if r.first), default="") or "–")
                     .replace("T", " ")[:16]),
                    (_("Last frame"), (max((r.last for r in d.sessions if r.last), default="") or "–")
                     .replace("T", " ")[:16]),
                    (_("Integration"), f"{fnum(hours)} h ({n} × {fexp(row0.exposure)})"),
                    (_("Black / white level"), f"{int(cam.get('black') or 0)} / {int(cam.get('white') or 0)} ADU")]
        elif d is not None and self.items:
            first = self.items[0]
            info = d.darks if self.group == "darks" else (d.flats if self.group == "flats" else {})
            rows = [(_("Frames"), str(len(self.items))), (_("Exposure"), fexp(info.get("exposure")) if info else "–"),
                    ("ISO", str(int(info["iso"])) if info and info.get("iso") else "–"),
                    (_("Taken on"), (first.time or "")[:10] or "–")]
        folder = (d.folders.get(self.group) or [""])[0] if d is not None and self.group != "lights" else \
            (self.s.p.abspath(self.s.p.sessions[0].lights[0]) if self.s.p.sessions and self.s.p.sessions[0].lights
             else "")
        rows.append((_("Folder"), folder))
        self.rv.addWidget(self._kv(rows))
        self.rv.addSpacing(8)
        self.rv.addWidget(hline())
        if self.items:
            f = self.items[min(self.cur, len(self.items) - 1)]
            self.rv.addWidget(label(_("Shown frame: {file}").format(file=f.name), "h2"))
            st = self.service.stats.get(f.path)
            rows = []
            if self.group == "lights":
                if f.level is not None:
                    rows.append((_("Median level"), _("{level} ADU above black").format(level=fnum(f.level))))
                rows += [(_("Time"), (f.time or "")[11:16]),
                         ("FWHM", f"{fnum(f.fwhm, 2)} px" if f.fwhm is not None else "–"),
                         (_("Transparency"), _("{pct} % of the best lights").format(
                             pct=f"{100 * f.transparency / self.p90:.0f}") if f.transparency is not None else "–"),
                         (_("Weight in the stack"), fnum(f.weight, 2) if f.weight is not None else "–")]
            elif st:
                rows.append((_("Median level"), _("{level} ADU (nominal black {black} ADU)").format(
                    level=fnum(self.black() + st["level"], 0), black=fnum(self.black(), 0)) if self.group == "darks"
                    else _("{level} ADU above black").format(level=fnum(st["level"]))))
                if self.group == "flats":
                    rows.append((_("Exposure level"), _("{pct} % of saturation").format(pct=fnum(st["pct"]))))
                else:
                    rows.append((_("Noise"), f"{fnum(st['noise'], 2)} ADU"))
            else:
                rows.append((_("Median level"), "…"))
            rows.append((_("State"), _("excluded") if self.by_hand(self.cur) else
                         (_("left out: {why}").format(why=_(f.status)) if f.status else _("kept"))))
            self.rv.addWidget(self._kv(rows))
        if self.group == "lights":
            self.rv.addSpacing(8)
            self.rv.addWidget(label(_("FWHM and transparency are measured at registration. To exclude by thresholds "
                                      "rather than one by one:"), "small", wrap=True))
            b = QPushButton(_("Frame selection by thresholds  →"))
            b.setObjectName("link")
            b.clicked.connect(lambda: self.goChain.emit("selection"))
            self.rv.addWidget(b)
        self.rv.addStretch(1)

    @staticmethod
    def _kv(rows) -> QLabel:
        import html
        table = "<table cellspacing='0' cellpadding='2'>" + "".join(
            f"<tr><td style='color:#8b93a1; padding-right:10px'>{k}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in rows) + "</table>"
        lb = QLabel(table)
        lb.setTextFormat(Qt.RichText)
        lb.setWordWrap(True)
        return lb


__all__ = ["SourcesView", "ImportDialog", "import_folder", "GROUPS"]
