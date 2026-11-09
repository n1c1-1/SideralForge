"""The files of a session: its lights and calibration frames kind by kind, added and taken out here one by one from
anywhere on the disk (nothing is guessed from a folder's name: each file goes where the user puts it), their calibration
matching, and a raw frames browser (grid, the common viewer with its frame sequence, sortable list) where frames are kept
or excluded.

The grid's thumbnails are made by the worker from the raw files (``api.frame_thumb``: one stretch for all the lights of
a session, so that a hazy frame looks darker when blinking). The viewer (``imageview.ImageView``) reads the raw frame
shown, and its neighbours ahead of time, in its own threads: zoom, stretch, the pixel readout, the info band (time,
exposure, ISO, temperature, FWHM, roundness, transparency, weight, kept or excluded), Exclude / Include, Blink. A
calibration frame left out by hand is left out of its master (``services/inspect.set_calibration_excluded``). Nothing
here reads a file.

Classes: ``SourcesView`` (the files and the browser), ``GroupRow`` (a kind of frames), ``FileList`` (its files).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import icons as ic
from . import prefs
from .frames import colour_note_text, fexp, night_text, type_warning_text
from .i18n import _, hours_text, number, plural
from .imageview import ImageView, ViewImage, quick_thumbnail
from .services.inspect import FrameItem, ProjectFrames, light_thumb, service_of, set_calibration_excluded
from .services.worker import norm_path
from .widgets import ACCENT, NEW, WARN, ElidedLabel, clear, hline, label, para

# (key, title, section): plain English, translated where shown
GROUPS = [("lights", "Lights", "Lights"), ("darks", "Darks", "Calibration"), ("flats", "Flats", "Calibration"),
          ("bias", "Bias", "Calibration"), ("flat_darks", "Dark flats", "Calibration")]
COUNT_KEY = {"lights": "lights", "darks": "darks", "flats": "flats", "bias": "bias", "flat_darks": "dark_flats"}
#: under this exposure (s), lights look like flats: a light of the deep sky lasts seconds, a flat a fraction of one
FLAT_LIKE_S = 0.1


class NumItem(QTableWidgetItem):
    """A table cell sorted by its number, shown with the decimal separator of the language."""

    def __init__(self, value, text=None):
        super().__init__(text if text is not None else str(value))
        self.v = value

    def __lt__(self, other):
        return self.v < getattr(other, "v", 0)


def local_paths(mime) -> list[str]:
    """The files or folders of a drop from the Explorer."""
    return [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]


class GroupRow(QFrame):
    """A kind of frames of the session: its name, what it holds and its +; a click shows it, files dropped on it are
    added to it."""
    clicked = Signal(str)
    dropped = Signal(str, list)                        # kind, the files or folders

    def __init__(self, key, title, sub, selected, empty, fill_menu=None, parent=None):
        super().__init__(parent)
        self.key = key
        self.setCursor(Qt.PointingHandCursor)
        self.setAcceptDrops(True)
        self.setStyleSheet(("QFrame { background: #2b3446; border-radius: 4px; }" if selected else
                            "QFrame { background: transparent; border-radius: 4px; } QFrame:hover "
                            "{ background: #232831; }") + " QLabel { background: transparent; }")
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 5, 4, 5)
        v = QVBoxLayout()
        v.setSpacing(0)
        a = QLabel(title)
        a.setStyleSheet("font-weight: 600;" + (" color: #6b7280;" if empty else ""))
        b = QLabel(sub)
        b.setObjectName("small")
        b.setWordWrap(True)
        v.addWidget(a)
        v.addWidget(b)
        h.addLayout(v, 1)
        if fill_menu is not None:
            add = ic.icon_only(QToolButton(), "add", _("Add files or a folder; they can also be dropped here"), 16)
            add.setAutoRaise(True)
            add.setStyleSheet("QToolButton::menu-indicator { image: none; }")
            menu = QMenu(add)
            fill_menu(menu)
            add.setMenu(menu)
            add.setPopupMode(QToolButton.InstantPopup)
            h.addWidget(add, 0, Qt.AlignTop)

    def mousePressEvent(self, e):                                    # noqa: N802 - Qt's name
        self.clicked.emit(self.key)

    def dragEnterEvent(self, e):                                     # noqa: N802 - Qt's name
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):                                          # noqa: N802 - Qt's name
        paths = local_paths(e.mimeData())
        if paths:
            self.dropped.emit(self.key, paths)


class FileList(QListWidget):
    """The files of a kind: several can be chosen (Ctrl, Shift, Ctrl+A), Del takes them out."""

    def __init__(self, remove):
        super().__init__()
        self.remove = remove
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setStyleSheet("QListWidget { background: #1c1f26; border: none; }")

    def chosen(self) -> list[str]:
        return [it.data(Qt.UserRole) for it in self.selectedItems() if it.data(Qt.UserRole)]

    def keyPressEvent(self, e):                                       # noqa: N802 - Qt's name
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.chosen():
            self.remove(self.chosen())
            return
        super().keyPressEvent(e)


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


def frame_filter() -> str:
    """The file dialog's filter: the frames the engine reads (raw, FITS, and the frames already in colour, #57)."""
    from ..core.project import COLOUR_EXT, FITS_EXT, RAW_EXT
    pats = " ".join(f"*{e}" for e in sorted(RAW_EXT | FITS_EXT | COLOUR_EXT | {".fz"}))
    return _("Frames ({patterns})").format(patterns=pats) + ";;" + _("All files (*)")


# ---------------------------------------------------------------------------------------------- the files and the browser
class SourcesView(QWidget):
    goChain = Signal(str)
    toast = Signal(str)                                    # what an addition or a removal did
    sendToPost = Signal(str, object, object)               # name, source, key (Send to Post of the viewer)

    def __init__(self, project, parent=None, service=None):
        super().__init__(parent)
        self.s = project
        self.service = service or service_of(project)
        self.data: ProjectFrames | None = None
        self.items: list[FrameItem] = []
        self.group = "lights"
        self.sid: int | None = None                        # the session shown
        self._step = None                                  # the block opened when it was last looked at
        self.file_list: FileList | None = None             # the files of the kind shown
        self.mode = 1                                      # 0 grid, 1 viewer, 2 list
        self.cur = 0
        self.sort = "time"
        self.can_post = None                               # callable: Post takes images (the window sets it)
        self._icons: dict[str, list[QListWidgetItem]] = {}
        self._group_token = 0
        self._reading = False                              # the frames of the kind shown are being read
        self._seq_order: list[int] = []                    # the viewer's sequence: indices of ``items``
        self._syncing = False
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
        left = QScrollArea()                               # a long list of warnings never makes the window taller
        left.setWidgetResizable(True)
        left.setFrameShape(QFrame.NoFrame)
        left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left.setFixedWidth(300)
        self._left_w = left
        inner = QWidget()
        inner.setObjectName("panel")
        left.setWidget(inner)
        self.lv = QVBoxLayout(inner)
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
        self.zone_b = QPushButton(_("Draw test area") + "  ◇")      # ◇: Preview on a region, Spec mode only
        self.zone_b.setToolTip("◇ " + _("Not in the engine yet."))
        self.zone_b.setEnabled(False)
        bar.addWidget(self.zone_b)
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
        # the common viewer, with the frames as its sequence (film strip, info band, Exclude / Include, Blink)
        self.viewer = ImageView(post=True, histogram=False)
        self.viewer.frameChanged.connect(self._viewer_frame)
        self.viewer.excludeToggled.connect(self._viewer_exclude)
        self.viewer.sendToPost.connect(self._send)
        self.stack.addWidget(self.viewer)
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
        self.stack.addWidget(self._drop_page())
        mv.addWidget(self.stack, 1)
        root.addWidget(mid, 1)
        # ---- right: properties
        right = QWidget()
        right.setObjectName("panel")
        right.setFixedWidth(320)
        self._right_w = right
        self.rv = QVBoxLayout(right)
        self.rv.setContentsMargins(12, 12, 12, 12)
        root.addWidget(right)

        self._table_timer = QTimer(self)                   # the list is built again once the levels came
        self._table_timer.setSingleShot(True)
        self._table_timer.setInterval(250)
        self._table_timer.timeout.connect(lambda: self.fill_table(self.order()) if self.items else None)
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(120)
        self._thumb_timer.timeout.connect(self._want_thumbs)
        self.grid.verticalScrollBar().valueChanged.connect(lambda _v: self._thumb_timer.start())
        sv = self.service
        sv.groupReady.connect(self._group_ready)
        sv.framesReady.connect(self._frames_ready)
        sv.thumbReady.connect(self._thumb)
        sv.statsReady.connect(self._stats_ready)
        self.setAcceptDrops(True)
        self._fill_sort()
        self.set_mode(1)

    # ------------------------------------------------------------ the empty project
    def set_compact(self, on: bool):
        """Issue #44: on a small window the two side columns narrow, the viewer between them keeps its room."""
        self._left_w.setFixedWidth(240 if on else 300)
        self._right_w.setFixedWidth(260 if on else 320)
        p = self                                       # hidden, its page would keep its minimum width: the layouts up
        while p is not None:                           # to the window (which that width would hold above 1024) are told
            if p.layout() is not None:
                p.layout().invalidate()
            p = p.parentWidget()

    def _empty_page(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setAlignment(Qt.AlignCenter)
        v.addWidget(label(_("No session yet"), "h1"), 0, Qt.AlignHCenter)
        v.addWidget(para(_("A session holds the files of one night: its lights, darks, flats and bias, from wherever "
                           "they are on the disk.")), 0, Qt.AlignHCenter)
        v.addSpacing(10)
        b = QPushButton(_("Add session…"))
        b.setObjectName("primary")
        b.clicked.connect(self.add_session)
        v.addWidget(b, 0, Qt.AlignHCenter)
        return w

    def _drop_page(self):
        """What an empty kind shows, in the place of its frames: where its files go."""
        w = QWidget()
        o = QVBoxLayout(w)
        o.setContentsMargins(24, 24, 24, 24)
        self.drop_zone = QFrame()
        self.drop_zone.setObjectName("dropzone")
        self._drop_hot(False)
        v = QVBoxLayout(self.drop_zone)
        v.setAlignment(Qt.AlignCenter)
        v.addWidget(label(_("Drop your files here"), "h1"), 0, Qt.AlignHCenter)
        self.drop_kind = label("", "dim")
        v.addWidget(self.drop_kind, 0, Qt.AlignHCenter)
        v.addSpacing(10)
        row = QHBoxLayout()
        row.addStretch(1)
        a = QPushButton(_("Add files…"))
        a.setObjectName("primary")
        a.clicked.connect(lambda: self.pick_files(self.group))
        b = QPushButton(_("Add folder…"))
        b.clicked.connect(lambda: self.pick_folder(self.group))
        row.addWidget(a)
        row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        o.addWidget(self.drop_zone)
        return w

    def _drop_hot(self, on: bool):
        self.drop_zone.setStyleSheet("QFrame#dropzone { border: 2px dashed " + (ACCENT if on else "#3a4150") +
                                     "; border-radius: 8px; } QLabel { border: none; }")

    def _show_zone(self):
        """An empty kind shows where its files go; the grid, the viewer and the list come back with its first files."""
        self.current_session()                             # taken away meanwhile (Ctrl+Z): another one is shown
        empty = self.sid is not None and not self.held(self.group)
        self.stack.setCurrentIndex(3 if empty else self.mode)
        for b in self.mode_btns + [self.sort_cb]:
            b.setEnabled(not empty)
        self.info.setVisible(self.mode != 1 and not empty)
        if empty:
            title = next(t for k, t, _s in GROUPS if k == self.group)
            self.drop_kind.setText(_("{kind} of “{session}”").format(kind=_(title), session=self._session().name))

    # ------------------------------------------------------------ the files of the session
    def _session(self):
        return self.s.p.session(self.sid)

    def current_session(self):
        """The session shown: the block just opened, else the one chosen here, else the first."""
        st = self.s.sel_step or ""
        if st != self._step:
            self._step = st
            if st.startswith("session@"):
                self._set_session(int(st.partition("@")[2]))
        ids = [x.id for x in self.s.p.sessions]
        if self.sid not in ids:
            self._set_session(ids[0] if ids else None)

    def _set_session(self, sid):
        if sid != self.sid:
            self.sid, self.items, self.cur = sid, [], 0

    def show_session(self, sid: int, kind: str = "lights"):
        """Show a session's files of a kind."""
        self._set_session(sid)
        if self.s.separate_sessions():
            self.s.select(None, f"session@{sid}")
            self._step = self.s.sel_step
        self.select_group(kind)

    def add_session(self):
        """A new session, without frames: shown here, where they are added."""
        self.show_session(self.s.new_session())

    def add_files(self, kind: str, paths: list) -> int:
        """Files or folders into a kind of the session shown (a folder is a shortcut for its frames); said, and shown."""
        if self.sid is None:
            return 0
        notes: list = []                                   # what was not added, said (GUI exploration 7.5)
        n = self.s.add_session_entries(self.sid, kind, paths, notes)
        said = [_(t).format(name=nm) for t, nm in notes[:2]] + (["…"] if len(notes) > 2 else [])
        if n:
            said.insert(0, plural(n, "{n} file added to “{session}” · Ctrl+Z", "{n} files added to “{session}” · Ctrl+Z",
                                  session=self._session().name))
            self.select_group(kind)
        self.toast.emit(" · ".join(said) if said else _("Nothing added: no frame in what was dropped"))
        return n

    def pick_files(self, kind: str):
        got, _f = QFileDialog.getOpenFileNames(self.window(), _("Add files…"), prefs.last_folder(), frame_filter())
        if got:
            prefs.remember_folder(got[0])
            self.add_files(kind, got)

    def pick_folder(self, kind: str):
        d = QFileDialog.getExistingDirectory(self.window(), _("Add folder…"), prefs.last_folder())
        if d:
            prefs.remember_folder(d)
            self.add_files(kind, [d])

    def files_out(self, kind: str, files: list):
        """Files taken out of the session (the files themselves are never touched): one undo step, said."""
        s, name = self.s, self._session().name
        with s.batch():
            done = s.remove_session_files(self.sid, kind, files)
        if done:
            self.toast.emit(plural(len(files), "{n} file out of “{session}” · file kept · Ctrl+Z",
                                   "{n} files out of “{session}” · files kept · Ctrl+Z", session=name))

    def locate(self, kind: str, folder: str, entries: list[str]):
        """A folder of the session not found: where it is now, chosen (a reference photo app's "find missing folder",
        GUI exploration 4); it takes the old one's place in the list, its files found in it by their names. One undo
        step; a folder without frames chosen: nothing changes, said."""
        start = next((p for p in Path(folder).parents if p.is_dir()), Path(prefs.last_folder()))
        d = self.ask_folder(_("Where is “{name}” now?").format(name=Path(folder).name), start.as_posix())
        if not d:
            return
        prefs.remember_folder(d)
        whole = any(not Path(e).suffix for e in entries)
        new = [Path(d).as_posix()] if whole else [(Path(d) / Path(e).name).as_posix() for e in entries
                                                  if (Path(d) / Path(e).name).is_file()]
        notes: list = [] if new else [("“{name}” holds no frames (camera raw, FITS, JPEG, PNG or TIFF): not added.",
                                       Path(d).name)]
        s = self.s
        with s.batch():
            if new and s.add_session_entries(self.sid, kind, new, notes):
                for e in entries:
                    s.remove_session_entry(self.sid, kind, e)
                notes = []
                self.toast.emit(_("“{name}” found again in “{session}” · Ctrl+Z").format(
                    name=Path(d).name, session=self._session().name))
        if notes:
            self.toast.emit(" · ".join(_(t).format(name=nm) for t, nm in notes))

    def ask_folder(self, title: str, start: str) -> str:
        return QFileDialog.getExistingDirectory(self, title, start)

    def anomalies(self, kind: str) -> list[str]:
        """What may be a mistake in a kind's files, said beside it as long as it holds (issue #58): a file not read or
        of another type, the same files in two kinds, and lights with the exposure of flats."""
        out = [type_warning_text(w) for w in self.s.file_warnings(self.sid) if w["kind"] == kind]
        row = self._row()
        if kind == "lights" and row is not None and row.exposure is not None and row.exposure < FLAT_LIKE_S:
            out.append(_("Lights at {exp}: the exposure of flats. Right row?").format(exp=fexp(row.exposure)))
        return out

    def _flagged(self, kind: str) -> set[str]:
        """The files a warning points at (lower-case paths): those of a type that is not read, or that differs."""
        from ..core.project import file_type
        out: set[str] = set()
        for w in self.s.file_warnings(self.sid):
            if w["kind"] == kind and w["key"] in ("unread", "mixed"):
                out |= {f.lower() for f in self.s.session_files(self.sid, kind) if file_type(f) == w["type"]}
        return out

    def not_found(self, kind: str) -> tuple[dict[str, list[str]], list[str]]:
        """What a kind lists and the disk does not give: ({folder not found: its entries}, the files gone from a folder
        still there; these only once the worker has read the kind)."""
        p, folders = self.s.p, {}
        for e in getattr(self._session(), kind):
            pth = Path(p.abspath(e))
            folders.setdefault((pth.parent if pth.suffix else pth).as_posix(), []).append(e)
        read = {norm_path(f) for f in self.s.session_files(self.sid, kind)}
        gone, files = {}, []
        for folder, entries in folders.items():
            if not Path(folder).exists():
                gone[folder] = entries
            elif read:
                files += [e for e in entries if Path(e).suffix and norm_path(p.abspath(e)) not in read]
        return gone, files

    def _fill_add_menu(self, kind: str, menu: QMenu):
        """One menu per kind: its files, a folder's, another session's."""
        menu.addAction(_("Add files…"), lambda k=kind: self.pick_files(k))
        menu.addAction(_("Add folder…"), lambda k=kind: self.pick_folder(k))
        others = [o for o in self.s.p.sessions if o.id != self.sid and getattr(o, kind)] if kind != "lights" else []
        if others:
            menu.addSeparator()
        for o in others:
            menu.addAction(_("Use the frames of “{name}”").format(name=o.name),
                           lambda k=kind, oid=o.id: self.s.use_from_session(oid, self.sid, k))

    def _file_box(self, kind: str, was: tuple):
        """The files of the kind shown, under its row, by folder when there are several; ``was``: the selection and the
        scrolling of the list before it was built again."""
        files = self.s.session_files(self.sid, kind)
        flagged = self._flagged(kind)
        by: dict[str, list[str]] = {}
        for f in files:
            by.setdefault(Path(f).parent.as_posix(), []).append(f)
        lst = FileList(lambda chosen, k=kind: self.files_out(k, chosen))
        for folder, fs in by.items():
            if len(by) > 1:
                head = QListWidgetItem(f"{Path(folder).name} · {len(fs)}")
                head.setFlags(Qt.NoItemFlags)
                head.setToolTip(folder)
                lst.addItem(head)
            for f in fs:
                it = QListWidgetItem(Path(f).name)
                it.setData(Qt.UserRole, f)
                it.setToolTip(f)
                if f.lower() in flagged:
                    it.setForeground(QColor(WARN))
                lst.addItem(it)
                it.setSelected(f in was[0])
        lst.setFixedHeight(min(lst.count(), 10) * (lst.sizeHintForRow(0) or 20) + 6)
        lst.verticalScrollBar().setValue(was[1])
        lst.currentItemChanged.connect(self._file_picked)
        self.lv.addWidget(lst)
        self.file_list = lst
        b = ic.set_icon(QPushButton(_("Remove")), "remove", 16)
        b.clicked.connect(lambda _c=False, k=kind: self.files_out(k, lst.chosen()))

        def selection():
            on = bool(lst.chosen())
            b.setEnabled(on)
            b.setToolTip(_("Take the chosen files out of the session (the files are not touched)") if on else
                         _("Choose files in the list first"))
        lst.itemSelectionChanged.connect(selection)
        selection()
        add = QPushButton(_("Add…"))                       # the "+" of the row, with its name (issue #61)
        add.setToolTip(_("Add files or a folder; they can also be dropped here"))
        menu = QMenu(add)
        self._fill_add_menu(kind, menu)
        add.setMenu(menu)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(b)
        h.addWidget(add)
        h.addStretch(1)
        self.lv.addWidget(row)

    def _file_picked(self, it, _prev=None):
        """A file chosen in the list: shown in the viewer."""
        path = it.data(Qt.UserRole) if it is not None else None
        i = next((n for n, x in enumerate(self.items) if norm_path(x.path) == norm_path(path)), None) if path else None
        if i is not None:
            self._goto(i)

    def _not_found_rows(self, kind: str):
        """What a kind lists and the disk does not give: said, with what can be done about it."""
        gone, files = self.not_found(kind)
        for folder, entries in gone.items():
            r = QHBoxLayout()
            line = ic.IconText("folder_missing", Path(folder).name + " · " + _("folder not found"), "warning", "warn")
            line.setToolTip(folder + "\n" + _("Not found (moved, renamed, or its disk unplugged): click to choose where "
                                              "it is now"))
            r.addWidget(line, 1)
            b = QPushButton(_("Locate…"))
            b.clicked.connect(lambda _c=False, k=kind, d=folder, ens=list(entries): self.locate(k, d, ens))
            r.addWidget(b, 0, Qt.AlignTop)
            self.lv.addLayout(r)
        if files:
            r = QHBoxLayout()
            r.addWidget(ic.IconText("warning", plural(len(files), "{n} file not found: moved or deleted?",
                                                      "{n} files not found: moved or deleted?"), "warning", "warn"), 1)
            b = QPushButton(_("Take out"))
            b.clicked.connect(lambda _c=False, k=kind, fs=list(files): self.files_out(k, fs))
            r.addWidget(b, 0, Qt.AlignTop)
            self.lv.addLayout(r)

    def dragEnterEvent(self, e):                                     # noqa: N802 - Qt's name
        if e.mimeData().hasUrls() and self.sid is not None:
            self._drop_hot(True)
            e.acceptProposedAction()

    def dragLeaveEvent(self, e):                                     # noqa: N802 - Qt's name
        self._drop_hot(False)

    def dropEvent(self, e):                                          # noqa: N802 - Qt's name
        """Dropped anywhere but on a kind's row: into the kind shown."""
        self._drop_hot(False)
        paths = local_paths(e.mimeData())
        if paths:
            self.add_files(self.group, paths)

    # ------------------------------------------------------------ data
    def refresh(self):
        """Show the session's files; ask the worker for the headers and the frames of the group."""
        loaded = bool(self.s.p.sessions)
        self.pages.setCurrentIndex(1 if loaded else 0)
        self.zone_b.setVisible(prefs.spec() and self.group == "lights")
        if not loaded:
            return
        self.current_session()
        wid = self.s.path.pid if self.s.path is not None else None
        self._group_token = self.service.request_group(wid, self.group, self.sid)
        self._reading = True
        self.service.request_frames(wid)
        self.build_left()
        self.build_right()
        self._show_zone()
        self.update_count()

    def _frames_ready(self, data: ProjectFrames):
        self.data = data
        self.current_session()
        if self.pages.currentIndex() == 1 and self.isVisibleTo(self.window()):
            self.build_left()
            self.build_right()

    def _group_ready(self, kind: str, token: int, items: list):
        if kind != self.group or token != self._group_token:
            return
        self._reading = False
        same = [it.path for it in items] == [it.path for it in self.items]
        self.items = items
        self._show_zone()
        self.cur = min(self.cur, max(len(items) - 1, 0))
        if same and self.grid.count() == len(items):
            self._marks()
            self.fill_table(self.order())
            self.update_count()
            for pos, i in enumerate(self._seq_order):
                if i < len(items):
                    self.viewer.set_frame_info(pos, self._info(i))
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
        """The files of a kind in the session shown."""
        ses = next((x for x in self.s.sessions if x["id"] == self.sid), None)
        return ses.get(COUNT_KEY[key], 0) if ses is not None else 0

    def held(self, key: str) -> int:
        """What a kind has to show: its files, and those left out by hand (to take them back)."""
        return self.count_of(key) + len(self._session().left_out.get(key, []))

    def _row(self):
        """What the headers say of the session shown (None until the worker has read them)."""
        d = self.data
        return next((r for r in d.sessions if r.id == self.sid), None) if d is not None else None

    def _sub(self, key: str, n: int, row) -> str:
        """The line under a kind's name: what it holds."""
        if not n:
            return _("none yet") if key == "lights" else _("none · optional")
        if key == "lights" and row is not None and row.first:
            return _("{n} lights · {exp} · ISO {iso} · night of {day}/{month}").format(
                n=n, exp=fexp(row.exposure), iso=int(row.iso) if row.iso else "–", day=night_text(row.first)[:2].strip(),
                month=row.first[5:7])
        folders = {Path(f).parent.name for f in self.s.session_files(self.sid, key)}
        if len(folders) == 1:
            return _("{n} frames · {folder}").format(n=n, folder=folders.pop())
        return _("{n} frames · {f} folders").format(n=n, f=len(folders))

    def build_left(self):
        old = self.file_list
        was = (set(old.chosen()), old.verticalScrollBar().value()) if old is not None else (set(), 0)
        clear(self.lv)
        self.file_list = None
        if self.sid is None:
            return
        sessions = self.s.p.sessions
        if len(sessions) > 1:
            cb = QComboBox()
            for x in sessions:
                cb.addItem(x.name, x.id)
            cb.setCurrentIndex(max(0, cb.findData(self.sid)))
            cb.currentIndexChanged.connect(lambda _i, c=cb: self.show_session(c.currentData(), self.group))
            cb.setToolTip(_("The session whose files are shown"))
            self.lv.addWidget(cb)
        else:
            self.lv.addWidget(label(self._session().name, "h2"))
        note = self.s.colour_note(self.sid)
        if note:                                           # issue #57: the limit of frames already in colour, said
            self.lv.addWidget(ic.IconText("info", colour_note_text(note), "dim", "colour_note"))
        section = None
        row0 = self._row()
        for key, title, sec in GROUPS:
            if sec != section:
                section = sec
                if sec == "Calibration":
                    self.lv.addSpacing(8)
                self.lv.addWidget(label(_(sec), "h2"))
            n = self.count_of(key)
            row = GroupRow(key, f"{_(title)} · {n}" if n else _(title), self._sub(key, n, row0), key == self.group, not n,
                           partial(self._fill_add_menu, key))
            row.clicked.connect(self.select_group)
            row.dropped.connect(self.add_files)
            self.lv.addWidget(row)
            shared = self.s.shared_with(self.sid, key) if key != "lights" else []
            if shared:
                names = ", ".join(self.s.session(o).name for o in shared)
                self.lv.addWidget(ic.IconText("link", _("shared with {names}: one master").format(names=names), "dim", "small"))
            for text in self.anomalies(key):               # beside its kind, as long as it holds
                self.lv.addWidget(ic.IconText("warning", text, "warning", "warn"))
            self._not_found_rows(key)
            if key == self.group and n:
                self._file_box(key, was)
        self.lv.addSpacing(10)
        self.lv.addWidget(hline())
        self.lv.addWidget(label(_("Calibration matching"), "h2"))
        for ok, text in self.checks():
            row = QHBoxLayout()
            mark = ic.IconLabel("ok" if ok is True else "warning" if ok is False else "info", 16,
                                "ok" if ok is True else "warning" if ok is False else "dim")
            row.addWidget(mark, 0, Qt.AlignTop)
            row.addWidget(label(text, "small", wrap=True), 1)
            self.lv.addLayout(row)
        if prefs.spec():
            row = QHBoxLayout()
            mark = QLabel("◇")
            mark.setFixedWidth(16)
            mark.setStyleSheet(f"color: {NEW}; font-weight: 700;")
            row.addWidget(mark, 0, Qt.AlignTop)
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
        ref = self._row()
        if d.darks and ref is not None:
            same = d.darks.get("exposure") == ref.exposure and d.darks.get("iso") == ref.iso
            t = (_("Darks: exposure {exp}, ISO {iso}, same as the lights.") if same else
                 _("Darks: exposure {exp}, ISO {iso}, different from the lights."))
            out.append((same, t.format(exp=fexp(d.darks.get("exposure")), iso=int(d.darks.get("iso") or 0))))
            if d.dark_level is not None:
                out.append((True, _("Darks: median level {level} ADU, nominal black level {black} ADU.").format(
                    level=number(d.dark_level, 0), black=number(self.black(), 0))))
        if d.flats:
            pct = d.flat_pct
            n = d.counts.get("flats", 0)
            if pct is not None:
                out.append((None, _("Flats: {n} frames at {pct} % of saturation, exposure {exp}.").format(
                    n=n, pct=number(pct, 0), exp=fexp(d.flats.get("exposure")))))
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
        """Show a kind: its frames, or where its files go."""
        self.group = key
        self.cur = 0
        self.items = []
        self.viewer.blink_b.setChecked(False)
        self._fill_sort()
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
            t = f"{f.name}\n{(number(f.fwhm) + ' px') if f.fwhm is not None else '–'} · {tr}"
        else:
            t = f"{f.name}\n{(number(f.level, 0) + ' ADU') if f.level is not None else (f.time[:10] if f.time else '')}"
        return t

    def fill(self):
        idx = self.order()
        self._icons = {}
        self.grid.clear()
        for i in idx:
            it = QListWidgetItem(placeholder(), self._item_text(i))
            it.setData(Qt.UserRole, i)
            it.setToolTip(self._item_text(i).replace("\n", " · "))
            if self.excluded(i):
                it.setForeground(QColor("#e06c6c"))
            if self.service.thumb_cache.get(self.items[i].path) is not None:
                it.setIcon(self._item_icon(i))
            else:
                self._icons.setdefault(self.items[i].path, []).append(it)
            self.grid.addItem(it)
        self.fill_table(idx)
        self.update_count()
        self._fill_viewer(idx)
        self._thumb_timer.start()

    # ------------------------------------------------------------ the viewer's sequence
    def _info(self, i: int) -> dict:
        """The info band of a frame: what its header and the engine say."""
        f = self.items[i]
        info = {"time": f.time, "exposure": f.exposure, "iso": f.iso, "temperature": f.temperature}
        if self.group == "lights":
            info.update(fwhm=f.fwhm, roundness=f.roundness, weight=f.weight,
                        transparency=(f.transparency / self.p90) if f.transparency is not None else None,
                        excluded=bool(f.excluded), reason=_(f.status) if f.status and not f.excluded else "")
            if f.level is not None:
                info["extra"] = [(_("Sky level"), number(f.level, 0) + " ADU")]
        else:
            st = self.service.stats.get(f.path)
            info.update(excluded=bool(f.excluded), compact=True)
            info["extra"] = [(_("Median level"), number(self.black() + st["level"], 0) + " ADU"),
                             (_("Noise σ"), number(st["noise"], 1) + " ADU")] if st else [(_("Median level"), "…")]
            if st and self.group == "flats":
                info["extra"].append((_("Exposure level"), number(st["pct"], 0) + " %"))
            if f.flag:
                info["flag"], info["flag_reason"] = f.flag, f.reason
        return info

    def _fill_viewer(self, idx):
        """The frames of the group, in the order of the sort, as the viewer's sequence."""
        self._seq_order = list(idx)
        if not idx:
            self.viewer.set_sequence([])
            self.viewer.clear()
            return
        lights = self.group == "lights"
        cfa = (self.data.camera.get("cfa") if self.data is not None else "") or ""
        frames = []
        for i in idx:
            f = self.items[i]
            # the kind of frame: its automatic stretch (a dark centred on its level, a flat its shape)
            vi = ViewImage(f.path, f.name, camera=True, cfa=f.cfa or cfa, white=f.white, info=self._info(i),
                           kind=self.group)
            vi.thumb = light_thumb(self.s.p, f.path) if lights else \
                partial(quick_thumbnail, ViewImage(f.path, cfa=f.cfa or cfa, camera=True, kind=self.group))
            frames.append(vi)
        self.viewer.set_exclude_mode("on")
        ok = False
        try:
            ok = bool(self.can_post()) if callable(self.can_post) else False
        except Exception:  # noqa: BLE001
            ok = False
        self.viewer.set_post_enabled(ok, "" if ok else _("Post does not take images yet: this comes with its image "
                                                         "layers."))
        pos = idx.index(self.cur) if self.cur in idx else 0
        self._syncing = True
        try:
            self.viewer.set_sequence(frames, pos)
        finally:
            self._syncing = False

    def _viewer_frame(self, pos: int):
        """The viewer moved to another frame (arrows, film strip, number, blink)."""
        if self._syncing or not (0 <= pos < len(self._seq_order)):
            return
        self.cur = self._seq_order[pos]
        self._select_in_lists()
        self.build_right()
        if self.group != "lights":
            self.service.request_stats(self.items[self.cur].path)

    def _viewer_exclude(self, pos: int, excluded: bool):
        if not (0 <= pos < len(self._seq_order)):
            return
        i = self._seq_order[pos]
        self.cur = i
        self._set_excluded(i, excluded)

    def _send(self, key):
        """Send to Post: the frame shown becomes a layer of Post (debayered, read in Post's worker)."""
        from .services.inspect import post_key, post_source, shown_neutral
        vi = self.viewer.image()
        if vi is not None:
            self.sendToPost.emit(vi.name, post_source(vi, shown_neutral(self.viewer)), post_key(vi, None))

    def _marks(self):
        """The excluded frames are shown as such, in place."""
        for r in range(self.grid.count()):
            it = self.grid.item(r)
            i = it.data(Qt.UserRole)
            it.setForeground(QColor("#e06c6c") if self.excluded(i) else QColor("#d8dce4"))
            it.setText(self._item_text(i))
            if self.service.thumb_cache.get(self.items[i].path) is not None:
                it.setIcon(self._item_icon(i))

    def _item_icon(self, i: int) -> QIcon:
        """A frame's thumbnail, an excluded one with its cross."""
        th = self.service.thumb_cache.get(self.items[i].path)
        if th is None:
            return placeholder()
        pm = pixmap(th)
        return QIcon(ic.stamped(pm, "remove", "error") if self.excluded(i) else pm)

    def _thumb(self, path: str, arr):
        for it in self._icons.pop(path, []):
            i = it.data(Qt.UserRole)
            it.setIcon(self._item_icon(i) if self.service.thumb_cache.get(path) is not None else QIcon(pixmap(arr)))

    def _want_thumbs(self):
        """Thumbnails of the grid for the frames in view and around the current one (the worker makes them in this
        order)."""
        if not self.items or self.pages.currentIndex() != 1 or self.mode != 0:
            return
        idx = self.order()
        w = self.grid
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
                         NumItem(f.fwhm or 0, number(f.fwhm, 2) if f.fwhm is not None else "–"),
                         NumItem(tr, f"{tr:.0f}" if f.transparency is not None else "–"),
                         NumItem(f.weight or 0, number(f.weight, 2) if f.weight is not None else "–"),
                         NumItem(f.level or 0, number(f.level) if f.level is not None else "–"),
                         QTableWidgetItem(keep)]
            else:
                st = self.service.stats.get(f.path)
                cells = [QTableWidgetItem(f.name), QTableWidgetItem((f.time or "")[:10]),
                         NumItem(st["level"] if st else 0, number(self.black() + st["level"], 0) if st else "–"),
                         NumItem(st["noise"] if st else 0, number(st["noise"], 2) if st else "–"),
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
        waiting = self._reading and not self.items and self.sid is not None and bool(self.held(self.group))
        self.viewer.canvas.message = _("Reading the frames…") if waiting else ""     # never a zero that is not one (#60)
        self.viewer.canvas.update()
        if waiting:
            self.count.setText(_("Reading the frames…"))
            return
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
        self._show_zone()                                # the viewer has its own info band
        self._goto(self.cur)
        self._thumb_timer.start()

    def step(self, d):
        idx = self.order()
        if not idx:
            return
        pos = idx.index(self.cur) if self.cur in idx else 0
        self._goto(idx[(pos + d) % len(idx)])

    def _select_in_lists(self):
        for r in range(self.grid.count()):
            if self.grid.item(r).data(Qt.UserRole) == self.cur:
                self.grid.setCurrentRow(r)
                self.grid.scrollToItem(self.grid.item(r))
                break
        self._info_line()

    def _goto(self, i):
        if not self.items:
            return
        self.cur = int(min(max(int(i), 0), len(self.items) - 1))
        if self.cur in self._seq_order:
            self._syncing = True
            try:
                self.viewer.set_index(self._seq_order.index(self.cur))
            finally:
                self._syncing = False
        self._select_in_lists()
        if self.group != "lights":
            self.service.request_stats(self.items[self.cur].path)
        self.build_right()

    def _info_line(self):
        """One line on the frame chosen, above the grid and the list (the viewer has its info band)."""
        f = self.items[self.cur]
        d = self.data
        row0 = self._row() if d is not None else None
        if self.group == "lights":
            tr = f"{100 * f.transparency / self.p90:.0f}" if f.transparency is not None else "–"
            exp = f.exposure if f.exposure is not None else (row0.exposure if row0 else None)
            iso = f.iso if f.iso is not None else (row0.iso if row0 else None)
            t = _("{file} · {time} · {exp} · ISO {iso} · FWHM {fwhm} px · transparency {transp} % of the best · "
                  "weight {weight}").format(
                file=f.name, time=(f.time or "")[11:16], exp=fexp(exp), iso=int(iso) if iso else "–",
                fwhm=number(f.fwhm, 2) if f.fwhm is not None else "–",
                transp=tr, weight=number(f.weight, 2) if f.weight is not None else "–")
        else:
            st = self.service.stats.get(f.path)
            t = (_("{file} · {date} · level {level} ADU · noise {noise} ADU").format(
                file=f.name, date=(f.time or "")[:10], level=number(self.black() + st["level"], 0),
                noise=number(st["noise"], 2)) if st else f"{f.name} · {(f.time or '')[:10]}")
        self.info.setText(t + (" · " + _("EXCLUDED") if self.by_hand(self.cur) else ""))

    def _stats_ready(self, path: str, st: dict):
        if not self.items:
            return
        if path == self.items[min(self.cur, len(self.items) - 1)].path:
            self.build_right()
            self._info_line()
        if self.group != "lights":
            for pos, i in enumerate(self._seq_order):
                if i < len(self.items) and self.items[i].path == path:
                    self.viewer.set_frame_info(pos, self._info(i))
                    break
            self._table_timer.start()

    def toggle_keep(self):
        """Exclude or take back the frame chosen (the list's Kept column)."""
        if not self.items:
            return
        f = self.items[self.cur]
        self._set_excluded(self.cur, not f.excluded)

    def _set_excluded(self, i: int, want: bool):
        """A light frame: by hand, for the whole project. A calibration frame: left out of its master."""
        f = self.items[i]
        if f.session is None:
            return
        if self.group == "lights":
            self.s.set_excluded([(f.session, f.name)], want)          # saved by the model, as any edit
            f.excluded = want
        else:
            if not set_calibration_excluded(self.s, f, want, self.items):
                pos = self._seq_order.index(i) if i in self._seq_order else -1
                if pos >= 0:
                    self.viewer.set_excluded(pos, f.excluded)
                return
            f.excluded = want
        self._marks()
        self.fill_table(self.order())
        self.update_count()
        if i in self._seq_order:
            self.viewer.set_frame_info(self._seq_order.index(i), self._info(i))
        self.build_left()
        self.build_right()

    # ------------------------------------------------------------ right panel
    def build_right(self):
        clear(self.rv)
        d = self.data
        self.rv.addWidget(label(_("Properties"), "h2"))
        rows = []
        row0 = self._row()
        if self.group == "lights" and d is not None and row0 is not None and row0.lights:
            cam = d.camera
            n = row0.lights
            hours = n * (row0.exposure or 0) / 3600
            rows = [(_("Camera"), cam.get("camera") or "–"),
                    (_("Sensor"), _("{w} × {h} px, {cfa} pattern").format(w=cam.get("width"), h=cam.get("height"),
                                                                          cfa=cam.get("cfa"))),
                    (_("Exposure"), fexp(row0.exposure)), ("ISO", f"{int(row0.iso)}" if row0.iso else "–"),
                    (_("First frame"), (row0.first or "–").replace("T", " ")[:16]),
                    (_("Last frame"), (row0.last or "–").replace("T", " ")[:16]),
                    (_("Integration"), f"{hours_text(hours)} ({n} × {fexp(row0.exposure)})"),
                    (_("Black / white level"), f"{int(cam.get('black') or 0)} / {int(cam.get('white') or 0)} ADU")]
        elif d is not None and self.items:
            first = self.items[0]
            info = d.darks if self.group == "darks" else (d.flats if self.group == "flats" else {})
            rows = [(_("Frames"), str(len(self.items))), (_("Exposure"), fexp(info.get("exposure")) if info else "–"),
                    ("ISO", str(int(info["iso"])) if info and info.get("iso") else "–"),
                    (_("Taken on"), (first.time or "")[:10] or "–")]
        folder = Path(self.items[min(self.cur, len(self.items) - 1)].path).parent.as_posix() if self.items else ""
        if folder:
            rows.append((_("Folder"), folder))
        self.rv.addWidget(self._kv(rows))
        self.rv.addSpacing(8)
        self.rv.addWidget(hline())
        if self.items:
            f = self.items[min(self.cur, len(self.items) - 1)]
            # a long file name is shortened in the middle (its whole name in the tooltip), never cut at the end
            self.rv.addWidget(ElidedLabel(_("Shown frame: {file}").format(file=f.name), "h2"))
            st = self.service.stats.get(f.path)
            rows = []
            if self.group == "lights":
                if f.level is not None:
                    rows.append((_("Median level"), _("{level} ADU above black").format(level=number(f.level))))
                rows += [(_("Time"), (f.time or "")[11:16]),
                         ("FWHM", f"{number(f.fwhm, 2)} px" if f.fwhm is not None else "–"),
                         (_("Transparency"), _("{pct} % of the best lights").format(
                             pct=f"{100 * f.transparency / self.p90:.0f}") if f.transparency is not None else "–"),
                         (_("Weight in the stack"), number(f.weight, 2) if f.weight is not None else "–")]
            elif st:
                rows.append((_("Median level"), _("{level} ADU (nominal black {black} ADU)").format(
                    level=number(self.black() + st["level"], 0), black=number(self.black(), 0)) if self.group == "darks"
                    else _("{level} ADU above black").format(level=number(st["level"]))))
                if self.group == "flats":
                    rows.append((_("Exposure level"), _("{pct} % of saturation").format(pct=number(st["pct"]))))
                else:
                    rows.append((_("Noise"), f"{number(st['noise'], 2)} ADU"))
            else:
                rows.append((_("Median level"), "…"))
            rows.append((_("State"), _("excluded") if self.by_hand(self.cur) else
                         (_("left out: {why}").format(why=_(f.status)) if f.status else _("kept"))))
            self.rv.addWidget(self._kv(rows))
        if self.group == "lights":
            self.rv.addSpacing(8)
            self.rv.addWidget(label(_("FWHM and transparency are measured at registration. To exclude by thresholds "
                                      "rather than one by one:"), "small", wrap=True))
            b = QPushButton(_("Select by thresholds  →"))
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


__all__ = ["SourcesView", "GROUPS"]
