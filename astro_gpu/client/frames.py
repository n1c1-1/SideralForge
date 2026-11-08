"""Frames: the sessions of the project (one per night), the health check of the data set, and the frame browser.

An imported folder is sorted by itself: one session per night (noon to noon), grouped by exposure and ISO or gain as the
Grouping dialog says. Darks, flats and bias are the project's: the engine makes one master of each kind. The sessions
table and the health checks come from the headers of the frames and from the engine's checks, read by the worker
(``services/inspect.project_frames``).

The checks the engine does not make yet (the master library, the sensor temperature, a rotated camera) show in Spec mode
only, marked ◇.

Classes: ``FramesView`` (the Sessions page and the browser of sources.py), ``GroupingDialog``.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import api, texts
from . import prefs
from .i18n import _, language, number, plural
from .imageview import exposure_text
from .services.inspect import ProjectFrames, service_of
from .widgets import NEW, OK, WARN, cap_dialog, clear, label

MONTHS = {"en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          "fr": ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]}


def fexp(t) -> str:
    """An exposure time: 1/1000 s under a second, else 15 s; "–" when unknown."""
    return "–" if t is None else exposure_text(t)


def night_text(first: str) -> str:
    """The night a frame belongs to (noon to noon): "16 → 17 Aug 2026"."""
    try:
        t = datetime.fromisoformat(first)
    except (TypeError, ValueError):
        return "–"
    d0 = (t - timedelta(hours=12)).date()
    d1 = d0 + timedelta(days=1)
    month = MONTHS.get(language(), MONTHS["en"])
    if d0.month == d1.month:
        return f"{d0.day} → {d1.day} {month[d1.month - 1]} {d1.year}"
    return f"{d0.day} {month[d0.month - 1]} → {d1.day} {month[d1.month - 1]} {d1.year}"


#: the kinds of frames in a sentence ("dark flats": one name everywhere, #49)
KIND_WORDS = {"lights": "lights", "darks": "darks", "flats": "flats", "bias": "bias", "flat_darks": "dark flats"}


def type_warning_text(w: dict) -> str:
    """A file that may be a mistake (core.project.type_warnings, issue #58), in the interface's language."""
    kind = _(KIND_WORDS.get(w.get("kind", ""), w.get("kind", "")))
    if w["key"] == "unread":
        return (_("{name} is not a frame the engine reads here (camera raw or FITS; a JPEG, PNG or TIFF only with lights "
                  "of its kind): added by mistake?").format(name=w["name"])
                if w["n"] == 1 else _("{n} {type} files are not frames the engine reads here (camera raw or FITS; a JPEG, "
                                      "PNG or TIFF only with lights of its kind)").format(n=w["n"], type=w["type"]))
    if w["key"] == "mixed":
        return plural(w["n"], "{n} {type} file among {of} {main} {kind}: added by mistake?",
                      "{n} {type} files among {of} {main} {kind}: added by mistake?",
                      type=w["type"], of=w["of"], main=w["main"], kind=kind)
    if w["key"] == "wrong_kind":
        return _("The {kind} are {type} files, the lights {main}: they cannot calibrate them").format(
            kind=kind, type=w["type"], main=w["main"])
    return _("The {kind} are {type} files, the lights {main}").format(kind=kind, type=w["type"], main=w["main"])


def colour_note_text(note: dict) -> str:
    """The limit of frames already in colour (issue #57), said plainly: ``note`` is io.frames.colour_session's."""
    if note.get("stretched"):
        return _("{type} frames already in colour: no debayering, no Bayer drizzle. They are not linear data (the "
                 "camera or a program already stretched and balanced them): dark and flat correction are "
                 "approximate.").format(type=note["type"])
    return _("{type} frames already in colour: no debayering, no Bayer drizzle. If they were stretched they are not "
             "linear data: dark and flat correction are then approximate.").format(type=note["type"])


def check_text(c, data: ProjectFrames | None = None) -> str:
    """A health check of the engine (api.Check) in the language of the interface; the numbers come from its arguments."""
    a = c.args or {}
    if c.key == "ok":
        return _("Nothing to worry about")
    if c.key == "no_lights":
        return _("No light frames yet")
    if c.key == "no_bias":
        black = (data.camera.get("black") if data is not None else None)
        if black:
            return _("No bias: flats are calibrated with the camera black level ({black} ADU).").format(
                black=int(black))
        return _("No bias: flats are calibrated with the camera black level.")
    if c.key == "file_types":
        return f"{a.get('session', '')} : " + type_warning_text(a) if language() == "fr" else \
            f"{a.get('session', '')}: " + type_warning_text(a)
    if c.key == "colour_frames":                          # issue #57: the limit of frames already in colour
        return f"{a.get('session', '')} : " + colour_note_text(a) if language() == "fr" else \
            f"{a.get('session', '')}: " + colour_note_text(a)
    if c.key == "unreadable":                             # GUI exploration 6.10: a light that cannot be read
        return _("{session}: {name} cannot be read (damaged, or not a frame): take it out of the session.").format(
            session=a.get("session", ""), name=a.get("name", ""))
    if c.key == "filters_mixed":                         # two filters in one workflow's stack
        return _("{workflow}: frames of {filters} would be stacked together; give each filter its own "
                 "workflow.").format(workflow=a.get("workflow", ""), filters=", ".join(a.get("filters", [])))
    if c.key == "flats_merged":
        return _("The flats of {n} sessions are merged into one master flat.").format(n=a.get("n", 0))
    if c.key == "disk":
        t = _("Disk: about {gb} GB needed for the calibrated frames; {free} GB free on the cache drive.").format(
            gb=number(a.get("need_gb", 0)), free=a.get("free_gb", 0))
        return t + (" " + _("Not enough space!") if c.level == "warn" else "")
    if c.key == "fwhm":
        return _("Star size in {session}: {fwhm} px").format(session=a.get("session", ""), fwhm=number(a.get("fwhm", 0)))
    return texts.translate(c.text, language())


def check_lines(data: ProjectFrames | None) -> list[tuple[str, str]]:
    """The health check of the data set: (level, text), level ok | warn | info. A session whose stars are more than 10 %
    larger than the best one's is said so (its frames weigh less)."""
    if data is None:
        return []
    out: list[tuple[str, str]] = []
    fwhm: dict[str, float] = {}
    for c in data.checks:
        if c.key == "fwhm":
            fwhm[c.args["session"]] = float(c.args["fwhm"])
            continue
        out.append((c.level, check_text(c, data)))
    if len(fwhm) > 1:
        best = min(fwhm.values())
        for name, f in fwhm.items():
            if f > best * 1.1:
                out.append(("info", _("Session {name}: FWHM {f} px against {b} px — its frames weigh less.").format(
                    name=name, f=number(f), b=number(best))))
    return out


#: what the engine does not check yet (Spec mode only)
NEW_CHECKS = ["Darks: a master of the library is matched by exposure, ISO and temperature.",
              "Sensor temperature more than 2 °C from the master dark: the dark is scaled.",
              "Camera rotated: its session uses its own flats."]


def grouping_sentence(c) -> str:
    parts = [_("night (noon to noon)")] + ([_("exposure")] if c.exposure else []) + ([_("ISO / gain")] if c.iso else [])
    what = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " " + _("and") + " " + parts[-1]
    return _("Grouped by {what}.").format(what=what)


class GroupingDialog(QDialog):
    """How frames are grouped into sessions: by night (noon to noon), exposure, ISO or gain. The choice holds for the
    folders imported next. The other criteria (filter, temperature, binning, rotation, object) are not in the engine yet."""

    def __init__(self, criteria=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_("Grouping"))
        c = criteria or api.GroupingCriteria()
        v = QVBoxLayout(self)
        self.setMinimumWidth(340)
        cap_dialog(self)
        v.addWidget(label(_("Group frames by"), "h2"))
        self.boxes: dict[str, QCheckBox] = {}
        rows = [("night", _("Night (noon to noon)"), c.night, True), ("exposure", _("Exposure"), c.exposure, True),
                ("iso", _("ISO / gain"), c.iso, True)]
        if prefs.spec():
            rows += [("filter", _("Filter"), True, False), ("temp", _("Sensor temperature (± 2 °C)"), True, False),
                     ("binning", _("Binning"), True, False), ("rotation", _("Camera rotation"), True, False),
                     ("object", _("Object (FITS OBJECT keyword)"), False, False)]
        for key, text, on, real in rows:
            b = QCheckBox(text + ("" if real else "  ◇"))
            b.setChecked(on)
            if not real:
                b.setEnabled(False)
                b.setToolTip("◇ " + _("Not in the engine yet."))
            v.addWidget(b)
            self.boxes[key] = b
        v.addSpacing(4)
        v.addWidget(label(_("The frames of a folder are sorted this way when it is imported."), "small", wrap=True))
        v.addSpacing(6)
        ok = QPushButton(_("OK"))
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        v.addWidget(ok, 0, Qt.AlignRight)

    def criteria(self):
        return api.GroupingCriteria(night=self.boxes["night"].isChecked(), exposure=self.boxes["exposure"].isChecked(),
                                    iso=self.boxes["iso"].isChecked())


class FramesView(QWidget):
    """The Frames block, inspected: sessions and health check, or the frame browser. Import folder… and Add session… run
    the import themselves (``sources.import_folder``); ``imported`` follows a successful one."""
    imported = Signal()
    stackMode = Signal(str)

    def __init__(self, project, browser, parent=None, service=None):
        super().__init__(parent)
        self.s = project
        self.service = service or service_of(project)
        self.data: ProjectFrames | None = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QFrame()
        bar.setObjectName("fbar")
        bar.setStyleSheet("QFrame#fbar { background: #1c1f26; border-bottom: 1px solid #2a2f38; }")
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 6, 10, 6)
        g = QButtonGroup(self)
        self.tabs = []
        for i, t in enumerate((_("Sessions"), _("Browse"))):
            b = QPushButton(t)
            b.setObjectName("face")
            b.setCheckable(True)
            b.clicked.connect(lambda _x=False, k=i: self.show_page(k))
            g.addButton(b)
            h.addWidget(b)
            self.tabs.append(b)
        h.addStretch(1)
        self.b_import = QPushButton(_("Import folder…"))
        self.b_import.clicked.connect(self.import_folder)
        self.b_add = QPushButton(_("Add session…"))
        self.b_add.clicked.connect(self.add_session)
        self.b_edit = QPushButton(_("Change frames…"))
        self.b_edit.setToolTip(_("Choose again the lights, darks, flats and bias of the selected session (or "
                                 "double-click its row)"))
        self.b_edit.clicked.connect(lambda: self.edit_session())
        self.b_group = QPushButton(_("Grouping…"))
        self.b_group.clicked.connect(self.grouping)
        self.table = None
        for b in (self.b_import, self.b_add, self.b_edit, self.b_group):
            h.addWidget(b)
        v.addWidget(bar)
        self.pages = QStackedWidget()
        v.addWidget(self.pages, 1)
        sp = QWidget()
        self.sv = QVBoxLayout(sp)
        self.sv.setContentsMargins(16, 12, 16, 12)
        self.sv.setSpacing(8)
        # the page scrolls: a long health check (a folder of unreadable files) never makes the window taller than the
        # screen (GUI exploration 2: 16 warnings took a 768 window to 920 px, and kept it there)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(sp)
        self.pages.addWidget(scroll)
        self.browser = browser
        self.pages.addWidget(browser)
        self.service.framesReady.connect(self._data_ready)
        self.show_page(0)

    def import_folder(self):
        from .sources import import_folder
        if import_folder(self.s, self.window(), service=self.service):
            self.imported.emit()

    def add_session(self):
        from .sources import import_folder
        if import_folder(self.s, self.window(), new_session=True, service=self.service):
            self.imported.emit()

    def edit_session(self, row: int | None = None):
        """Change frames…: the import dialog on the selected session's frames (the first when none is selected)."""
        from .sources import edit_session_frames
        sess = self.s.p.sessions
        if not sess:
            return
        if row is None:
            sel = self.table.selectionModel().selectedRows() if self.table is not None else []
            row = sel[0].row() if sel else 0
        if edit_session_frames(self.s, sess[row].id, self.window(), service=self.service):
            self.imported.emit()

    def grouping(self):
        d = GroupingDialog(self.service.criteria, self.window())
        if d.exec():
            self.service.criteria = d.criteria()
            self._build()
        d.deleteLater()

    def show_page(self, i):
        self.pages.setCurrentIndex(i)
        self.tabs[i].setChecked(True)
        if i == 0:
            self.refresh()
        elif hasattr(self.browser, "refresh"):
            self.browser.refresh()

    def refresh(self):
        """Build the page from what is known, and ask the worker for the headers and the checks."""
        if self.pages.currentIndex() == 1:
            if hasattr(self.browser, "refresh"):
                self.browser.refresh()
            return
        s = self.s
        if s.sources_loaded:
            self.service.request_frames(s.path.pid if s.path is not None else None)
        self._build()

    def _data_ready(self, data: ProjectFrames):
        self.data = data
        if self.pages.currentIndex() == 0 and self.isVisibleTo(self.window()):
            self._build()

    # ------------------------------------------------------------ the sessions page
    def _rows(self):
        """The rows of the table: what the worker has read, else what the model counts."""
        if self.data is not None and len(self.data.sessions) == len(self.s.sessions):
            return self.data.sessions
        return None

    def _build(self):
        clear(self.sv)
        s = self.s
        self.b_add.setEnabled(s.sources_loaded)
        self.b_edit.setEnabled(s.sources_loaded and bool(s.p.sessions))
        if not s.sources_loaded:
            self.sv.addStretch(1)
            self.sv.addWidget(label(_("No frames yet"), "h1"), 0, Qt.AlignHCenter)
            b = QPushButton(_("Import folder…"))
            b.setObjectName("primary")
            b.clicked.connect(self.import_folder)
            self.sv.addWidget(b, 0, Qt.AlignHCenter)
            self.sv.addStretch(2)
            return
        rows = self._rows()
        self.sv.addWidget(label(_("Sessions"), "h2"))
        cols = [_("Session"), _("Night"), _("Lights"), _("Exposure"), _("ISO / gain"), _("Temp."), _("Filter"),
                _("Flats"), _("Darks"), _("Bias"), "FWHM", _("Kept frames")]
        sess = s.sessions
        t = QTableWidget(len(sess), len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setStyleSheet("QTableWidget { background: #1c1f26; gridline-color: #2a2f38; } "
                        "QHeaderView::section { background: #23272f; border: none; padding: 4px; }")
        osc = bool(self.data and self.data.camera.get("cfa"))
        for r, ses in enumerate(sess):
            vals = self._cells(ses, rows[r] if rows is not None else None, osc)
            for c, val in enumerate(vals):
                it = QTableWidgetItem(val)
                if 2 <= c < len(vals) - 1:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r, c, it)
        t.resizeColumnsToContents()
        for c in range(len(cols)):
            t.setColumnWidth(c, t.columnWidth(c) + 18)
        t.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        t.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        t.setFixedHeight(t.horizontalHeader().height() + sum(t.rowHeight(r) for r in range(len(sess))) + 6)
        t.setFrameShape(QFrame.NoFrame)
        t.setFixedWidth(sum(t.columnWidth(c) for c in range(len(cols))))
        t.cellDoubleClicked.connect(lambda r, _c: self.edit_session(r))
        self.table = t
        self.sv.addWidget(t, 0, Qt.AlignLeft)
        self.sv.addWidget(label(grouping_sentence(self.service.criteria), "small"))
        self.sv.addSpacing(6)
        self.sv.addWidget(label(_("Health check"), "h2"))
        lines = check_lines(self.data)
        if self.data is None:
            self.sv.addWidget(label(_("Reading the frames…"), "dim"))
        for lvl, text in lines:
            self._check_row(lvl, text)
        if prefs.spec():
            for text in NEW_CHECKS:
                row = QHBoxLayout()
                mark = QLabel("◇")
                mark.setFixedWidth(18)
                mark.setStyleSheet(f"color: {NEW}; font-weight: 700;")
                row.addWidget(mark, 0, Qt.AlignTop)
                lb = label(_(text), "", wrap=True)
                lb.setEnabled(False)
                lb.setToolTip("◇ " + _("Not in the engine yet."))
                row.addWidget(lb, 1)
                self.sv.addLayout(row)
        if len(sess) > 1:
            self.sv.addSpacing(6)
            self.sv.addWidget(label(_("Stacking"), "h2"))
            row = QHBoxLayout()
            g = QButtonGroup(self)
            for text, tip, mode in ((_("All together"), _("All sessions together"), "together"),
                                    (_("One per session"), _("One stack per session"), "per_session")):
                if mode == "per_session" and not prefs.spec():
                    continue                                   # ◇: the engine stacks all the sessions together
                b = QPushButton(text + ("  ◇" if mode == "per_session" else ""))     # compact names (decision 12)
                b.setToolTip(tip)
                b.setObjectName("face")
                b.setCheckable(True)
                b.setChecked(s.stack_mode == mode)
                if mode == "per_session":
                    b.setEnabled(False)
                    b.setToolTip("◇ " + _("Not in the engine yet."))
                b.clicked.connect(lambda _x=False, m=mode: self.stackMode.emit(m))
                g.addButton(b)
                row.addWidget(b)
            row.addStretch(1)
            self.sv.addLayout(row)
        self.sv.addStretch(1)

    @staticmethod
    def _cells(ses: dict, row, osc: bool) -> list[str]:
        """One line of the table: what the headers say (``row``, once read), else what the model counts (``ses``)."""
        def count(n):
            return str(n) if n else _("none")
        src = row if row is not None else None
        # the session's filter (exploration 6.10: the column always said "–"); a colour camera's: OSC
        flt = (src.filter if src is not None else ses.get("filter", "")) or ""
        flt = "" if flt == "OSC" else flt
        lights = src.lights if src else ses["lights"]
        kept = src.kept if src is not None and src.kept is not None else ses["kept"]
        return [ses["name"], night_text(src.first) if src else "–", str(lights),
                fexp(src.exposure) if src else "–", str(int(src.iso)) if src and src.iso else "–",
                f"{number(src.temperature)} °C" if src and src.temperature is not None else "–",
                flt or ("OSC" if osc else "–"), count(src.flats if src else ses["flats"]),
                count(src.darks if src else ses["darks"]), count(src.bias if src else ses["bias"]),
                f"{number(src.fwhm)} px" if src and src.fwhm else "–", f"{kept} / {lights}"]

    def _check_row(self, lvl: str, text: str):
        row = QHBoxLayout()
        mark = QLabel({"ok": "✓", "warn": "⚠", "info": "i"}.get(lvl, "i"))
        mark.setFixedWidth(18)
        mark.setStyleSheet(f"color: {OK if lvl == 'ok' else WARN if lvl == 'warn' else '#8b93a1'}; font-weight: 700;")
        row.addWidget(mark, 0, Qt.AlignTop)
        row.addWidget(label(text, "", wrap=True), 1)
        self.sv.addLayout(row)
