"""Settings (application preferences): General, Folders & disk, Compute, Catalogs & astrometry, File formats, Master
library, Display. Kept in prefs; the language, the device and Spec mode take effect at once. Folders & disk also holds
the sections of Home (a name, a folder and a description each: home.section_rows).

What the engine does not have yet is shown only in Spec mode, disabled and marked ◇, and hidden otherwise: Temporary
files, Cache limit, CPU threads, Memory limit, Benchmark, ASTAP, astrometry.net, the survey cache folder, and the File
formats and Master library pages. The dialog is ``SettingsDialog``.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import gaia_extracts as ge
from . import prefs
from .i18n import LANGUAGES, _, maybe, number
from .services import extracts
from .services.worker import Machine, disk_usage
from .widgets import label, spec_mark

# (key, title, ◇ page)
PAGES = [("general", "General", False), ("folders", "Folders & disk", False), ("compute", "Compute", False),
         ("catalogs", "Catalogs & astrometry", False), ("formats", "File formats", True),
         ("masters", "Master library", True), ("display", "Display", False)]
LABEL_MIN = 200


def not_yet() -> str:
    return "◇ " + _("Not in the engine yet.")


class SettingsDialog(QDialog):
    languageChanged = Signal(str)
    deviceChanged = Signal(str)                      # auto | cuda:<n> | cpu
    specChanged = Signal(bool)
    workspaceChanged = Signal(str)
    homeChanged = Signal()                           # the sections of Home (Folders page)
    clearCache = Signal()

    def __init__(self, worker=None, machine: Machine | None = None, can_clear: bool = False, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.machine = machine
        self.can_clear = can_clear
        self.resize(900, 600)
        self.h = QHBoxLayout(self)
        self.nav = self.pages = None
        self.keys: list[str] = []
        self.bars: dict[str, tuple[QProgressBar, QLabel]] = {}
        self.ex: dict = {}                           # the widgets of the offline extracts (Catalogs & astrometry)
        if worker is not None:
            worker.done.connect(self._done)
        j = extracts.job()                           # the download goes on when this window is closed
        j.progress.connect(self._ex_progress, Qt.QueuedConnection)
        j.finished.connect(self._ex_finished, Qt.QueuedConnection)
        self.build()

    # ------------------------------------------------------------ building
    def build(self, page: str | None = None):
        """Build (or build again, after a change of language or of Spec mode) every page."""
        while self.h.count():
            w = self.h.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self.setWindowTitle(_("Settings"))
        self.spec = prefs.spec()
        self.bars = {}
        self.sec_rows, self.sec_ws = [], None
        self.nav = QListWidget()
        self.nav.setFixedWidth(200)
        self.h.addWidget(self.nav)
        self.pages = QStackedWidget()
        self.h.addWidget(self.pages, 1)
        self.keys = []
        for key, title, new in PAGES:
            if new and not self.spec:
                continue
            item = QListWidgetItem(_(title) + ("  ◇" if new else ""))
            if new:
                item.setForeground(QColor("#8b93a1"))
                item.setToolTip(not_yet())
            self.nav.addItem(item)
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(16, 8, 8, 8)
            v.setSpacing(10)
            self._labels: list[QLabel] = []
            self._page_new = new
            v.addWidget(label(spec_mark(_(title)) if new else _(title), "h1"))
            if new:
                v.addWidget(label(not_yet(), "small"))
            getattr(self, key)(v)
            v.addStretch(1)
            width = max([LABEL_MIN] + [lb.sizeHint().width() + 12 for lb in self._labels if not lb.isHidden()])
            for lb in self._labels:
                lb.setFixedWidth(width)
            if key == "folders":                     # the sections of Home make it the longest page: it scrolls
                sc = QScrollArea()
                sc.setWidgetResizable(True)
                sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                sc.setWidget(w)
                w = sc
            self.pages.addWidget(w)
            self.keys.append(key)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(self.keys.index(page) if page in self.keys else 0)
        self.measure_disks()

    def current(self) -> str:
        return self.keys[self.nav.currentRow()] if self.nav is not None and self.keys else "general"

    def show_page(self, key: str):
        if key in self.keys:
            self.nav.setCurrentRow(self.keys.index(key))

    def rebuild(self):
        self.build(self.current())

    # ------------------------------------------------------------ rows
    def _is_new(self, new: bool) -> bool:
        return new or self._page_new

    def row(self, v, text, w, unit="", new=False):
        """A setting on one line: name, editor, unit. A ◇ setting: only in Spec mode, disabled."""
        h = QHBoxLayout()
        own = new and not self._page_new
        lb = QLabel(spec_mark(text) if own else text)
        self._labels.append(lb)
        h.addWidget(lb)
        h.addWidget(w, 1)
        parts = [lb, w]
        if unit:
            u = label(unit, "small")
            h.addWidget(u)
            parts.append(u)
        self._mark(parts, new)
        v.addLayout(h)
        return w

    def _mark(self, widgets, new):
        if not self._is_new(new):
            return
        for w in widgets:
            w.setEnabled(False)
            w.setToolTip(not_yet())
            if not self.spec:
                w.setVisible(False)

    def path_row(self, v, text, key, new=False, placeholder="", changed=None):
        e = QLineEdit(prefs.get(key) or "")
        e.setPlaceholderText(placeholder)
        b = QPushButton(_("Browse…"))
        h = QHBoxLayout()
        h.setSpacing(6)
        h.setContentsMargins(0, 0, 0, 0)
        box = QWidget()
        box.setLayout(h)
        h.addWidget(e, 1)
        h.addWidget(b)

        def store(value):
            if value != prefs.get(key):
                prefs.set(key, value)
                if changed is not None:
                    changed(value)

        def pick():
            d = QFileDialog.getExistingDirectory(self, text, e.text())
            if d:
                e.setText(d)
                store(d)
        e.editingFinished.connect(lambda: store(e.text().strip()))
        b.clicked.connect(pick)
        self.row(v, text, box, new=new)
        return e

    def combo(self, options, current, key=None):
        """A choice: options are (English text, value, ◇). ◇ values appear only in Spec mode, disabled; when that
        leaves one value, it is shown as a plain text."""
        real = [o for o in options if not o[2]]
        if len(real) == 1 and len(options) > 1 and not self.spec:
            return QLabel(_(real[0][0]))
        c = QComboBox()
        for text, value, new in options:
            if new and not self.spec:
                continue
            c.addItem(_(text) + ("  ◇" if new else ""), value)
            if new:
                it = c.model().item(c.count() - 1)
                it.setEnabled(False)
                it.setToolTip(not_yet())
        j = c.findData(current)
        c.setCurrentIndex(max(j, 0))
        if key:
            c.currentIndexChanged.connect(lambda _i: prefs.set(key, c.currentData()))
        return c

    def spin(self, lo, hi, key, double=False):
        s = QDoubleSpinBox() if double else QSpinBox()
        s.setRange(lo, hi)
        s.setValue(prefs.get(key))
        s.valueChanged.connect(lambda x: prefs.set(key, x))
        return s

    # ------------------------------------------------------------ pages
    def general(self, v):
        lang = QComboBox()
        for code, name in LANGUAGES.items():              # each language in its own name
            lang.addItem(name, code)
        lang.setCurrentIndex(max(lang.findData(prefs.get("language")), 0))
        lang.currentIndexChanged.connect(lambda _i: self._lang(lang.currentData()))
        self.row(v, _("Language"), lang)
        spec = QCheckBox(_("Show what the engine still has to build (◇)"))
        spec.setChecked(self.spec)
        spec.toggled.connect(self._spec)
        self.row(v, _("Spec mode"), spec)

    def folders(self, v):
        self.path_row(v, _("Workspace"), "workspace", changed=self._workspace)
        self.path_row(v, _("Cache (calibrated frames)"), "cache_dir",
                      placeholder=_("In the project folder"), changed=lambda _x: self.measure_disks())
        self.path_row(v, _("Temporary files"), "temp_dir", new=True)
        self.row(v, _("Cache limit"), self.spin(10, 4000, "cache_limit_gb"), _("GB"), new=True)
        v.addWidget(label(_("Disk use"), "h2"))
        for key, name, new in (("cache", _("Cache"), False), ("workspace", _("Workspace"), False),
                               ("temp", _("Temporary"), True)):
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            text = label("", "small")
            text.setMinimumWidth(120)
            text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            h = QWidget()
            hl = QHBoxLayout(h)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(bar, 1)
            hl.addWidget(text)
            self.row(v, name, h, new=new)
            self.bars[key] = (bar, text)
        clean = QPushButton(_("Clear cache of closed projects"))
        clean.setEnabled(self.can_clear)
        clean.clicked.connect(lambda: self.clearCache.emit())
        v.addWidget(clean, 0, Qt.AlignLeft)
        self.home_sections(v)

    def home_sections(self, v):
        """The sections of Home: a row each (name, folder, description); the workspace's row takes its folder from
        the Workspace setting above and cannot be removed."""
        from .home import section_rows
        v.addSpacing(6)
        v.addWidget(label(_("Home sections"), "h2"))
        v.addWidget(label(_("Each section of Home lists the projects of one folder and of its sub-folders."), "small",
                          wrap=True))
        self.sec_rows = []
        for k, r in enumerate(section_rows()):
            box = QFrame()
            box.setObjectName("secrow")
            box.setStyleSheet("QFrame#secrow { background: #23272f; border-radius: 6px; }")
            g = QGridLayout(box)
            g.setContentsMargins(8, 8, 8, 8)
            g.setHorizontalSpacing(6)
            g.setVerticalSpacing(6)
            name = QLineEdit(maybe(r["label"]))
            name.setFixedWidth(190)
            folder = QLineEdit(r["folder"])
            browse = QPushButton(_("Browse…"))
            rm = QPushButton(_("Remove"))
            note = QLineEdit(maybe(r["note"]))
            for e, text in ((name, _("Name")), (folder, _("Folder")), (note, _("Description"))):
                e.setPlaceholderText(text)
                e.setToolTip(text)
                e.setCursorPosition(0)
            if r["workspace"]:
                why = _("The workspace, set above: New project creates its folders there.")
                folder.setEnabled(False)
                browse.setEnabled(False)
                rm.setEnabled(False)
                for w in (folder, browse, rm):
                    w.setToolTip(why)
                self.sec_ws = folder
            self.sec_rows.append({"row": r, "name": name, "folder": folder, "note": note})
            for e in (name, folder, note):
                e.editingFinished.connect(self._save_sections)
            browse.clicked.connect(lambda _x=False, e=folder: self._pick_section(e))
            rm.clicked.connect(lambda _x=False, i=k: self._remove_section(i))
            g.addWidget(name, 0, 0)
            g.addWidget(folder, 0, 1)
            g.addWidget(browse, 0, 2)
            g.addWidget(rm, 0, 3)
            g.addWidget(note, 1, 0, 1, 4)
            g.setColumnStretch(1, 1)
            v.addWidget(box)
        add = QPushButton(_("Add a section"))
        add.clicked.connect(self._add_section)
        v.addWidget(add, 0, Qt.AlignLeft)

    def compute(self, v):
        m = self.machine
        options = [("Automatic", "auto", False)]
        dev = prefs.get("device")
        c = self.combo(options, dev)
        if m is not None:
            for g in m.gpus:
                c.addItem(g.name, g.device)
                c.setItemData(c.count() - 1, f"{g.device} · {g.memory_gb:.0f} GB", Qt.ToolTipRole)
        elif dev.startswith("cuda:"):
            c.addItem(dev, dev)
        c.addItem(_("CPU only"), "cpu")
        c.setCurrentIndex(max(c.findData(dev), 0))
        c.currentIndexChanged.connect(lambda _i: self._device(c.currentData()))
        self.row(v, _("Device"), c)
        cores = (m.cores if m is not None and m.cores else os.cpu_count()) or 1
        th = QSpinBox()
        th.setRange(0, cores)
        th.setSpecialValueText(_("Automatic ({n} of {total})").format(n=max(cores - 2, 1), total=cores))
        th.setValue(min(prefs.get("threads"), cores))
        th.valueChanged.connect(lambda x: prefs.set("threads", x))
        self.row(v, _("CPU threads"), th, new=True)
        total = round(m.memory_gb) if m is not None and m.memory_gb else None      # None: not measured yet
        mem = self.spin(4, max(total, 4) if total else 1024, "memory_limit_gb")
        self.row(v, _("Memory limit"), mem, _("GB of {t} GB").format(t=total) if total else _("GB"), new=True)
        head = label(spec_mark(_("Benchmark")), "h2")
        v.addWidget(head)
        text = label(_("Measures this machine on a small data set, so that the durations shown are real."),
                     "small", wrap=True)
        v.addWidget(text)
        b = QPushButton(_("Run benchmark"))
        bar = QProgressBar()
        bar.setTextVisible(False)
        r = QHBoxLayout()
        r.addWidget(b)
        r.addWidget(bar, 1)
        v.addLayout(r)
        res = label(_("Not run yet: the durations shown are those measured on the reference machine."), "small",
                    wrap=True)
        v.addWidget(res)
        self._mark([head, text, b, bar, res], True)

    def catalogs(self, v):
        self.offline_extracts(v)
        self.row(v, _("Plate solver"), self.combo([("Internal (Gaia)", "internal", False), ("ASTAP", "astap", True),
                                                   ("astrometry.net (local)", "astrometry.net", True)],
                                                  prefs.get("solver"), "solver"))
        self.path_row(v, _("Survey cache (H-alpha)"), "survey_cache", new=True)
        self.survey_offline(v)

    def survey_offline(self, v):
        """Issue #39: the H-alpha survey's tiles of the recent projects' fields fetched ahead, so that their Surveys
        stage (and any new stack of the same fields) works offline. Off the GUI thread; said when done."""
        v.addWidget(label(_("H-alpha survey offline"), "h2"))
        v.addWidget(label(_("The survey is read tile by tile and each tile is kept: a field already seen works "
                            "offline. Download fetches the tiles of the recent projects' solved fields now."),
                          "small", wrap=True))
        r = QHBoxLayout()
        b = QPushButton(_("Download for offline use"))
        st = label("", "small", wrap=True)
        r.addWidget(b)
        r.addWidget(st, 1)
        v.addLayout(r)
        state = {}

        def done():
            if "result" not in state:
                return
            t.stop()
            b.setEnabled(True)
            res = state["result"]
            if isinstance(res, Exception):
                st.setText(_("Not downloaded: {why}").format(why=str(res)[:200]))
                return
            st.setText(_("{fields} fields, {tiles} tiles in the cache").format(fields=res["fields"], tiles=res["tiles"])
                       + (("; " + _("not read: {why}").format(why="; ".join(res["failed"][:3]))) if res["failed"]
                          else ""))

        def go():
            import threading

            from ..stages.surveys import prefetch_field
            roots = [r["path"] for r in prefs.recent()]
            b.setEnabled(False)
            st.setText(_("Downloading the tiles of {n} projects…").format(n=len(roots)))
            state.clear()

            def work():
                try:
                    state["result"] = prefetch_field(roots)
                except Exception as e:  # noqa: BLE001 - said in the dialog
                    state["result"] = e
            threading.Thread(target=work, daemon=True).start()
            t.start(500)
        t = QTimer(b)
        t.timeout.connect(done)
        b.clicked.connect(go)

    def offline_extracts(self, v):
        """The Gaia DR3 extracts the engine reads offline: their folder, what is installed, the two downloads (run
        off the GUI thread by services.extracts, only when a button is pressed)."""
        v.addWidget(label(_("Offline Gaia extracts"), "h2"))
        v.addWidget(label(_("Star positions find where the image points on the sky; star spectra calibrate the colours. "
                            "Without them, Gaia stars are fetched online and kept in the cache."), "small", wrap=True))
        where = QLineEdit(str(ge.folder()))
        where.setReadOnly(True)
        where.setCursorPosition(0)
        opn = QPushButton(_("Open folder"))
        opn.clicked.connect(self._ex_open)
        box = QWidget()
        hl = QHBoxLayout(box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(where, 1)
        hl.addWidget(opn)
        self.row(v, _("Folder"), box)
        self.ex = {"status": {}, "button": {}}
        buttons = QHBoxLayout()
        for kind, name, button in ((ge.ASTRO, _("Star positions"), _("Download star positions ({size})")),
                                   (ge.XP, _("Star spectra"), _("Download star spectra ({size})"))):
            st = label("", "small", wrap=True)
            self.row(v, name, st)
            gb = f"{number(ge.DOWNLOAD_BYTES[kind] / 1e9)} {_('GB')}"
            b = QPushButton(button.format(size=gb))
            b.clicked.connect(lambda _x=False, k=kind: self._ex_start(k))
            buttons.addWidget(b)
            self.ex["status"][kind], self.ex["button"][kind] = st, b
        buttons.addStretch(1)
        v.addLayout(buttons)
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setTextVisible(False)
        cancel = QPushButton(_("Cancel"))
        cancel.clicked.connect(lambda: extracts.job().cancel())
        run = QWidget()
        hl = QHBoxLayout(run)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(bar, 1)
        hl.addWidget(cancel)
        v.addWidget(run)
        msg = label("", "small", wrap=True)
        v.addWidget(msg)
        self.ex.update(bar=bar, run=run, msg=msg)
        v.addSpacing(6)
        j = extracts.job()
        run.setVisible(j.running())
        if j.running():
            self._ex_progress(*j.last)
        self._ex_refresh()

    def _ex_status_text(self, st: ge.Status) -> str:
        if st.state == "installed":
            if st.outside is not None:
                return _("Found in another program's data folder: {path}").format(path=st.outside)
            return _("Installed")
        if st.state == "partial":
            if st.files and st.total > 1:
                return _("Partly downloaded: {n} of {total} files").format(n=st.files, total=st.total)
            return _("Partly downloaded: {size}").format(size=ge.human(st.part_bytes))
        return _("Not installed")

    def _ex_refresh(self):
        """The state of each extract and which buttons can be pressed."""
        busy = extracts.job().running()
        try:
            for kind, lab in self.ex.get("status", {}).items():
                st = ge.status(kind)
                lab.setText(self._ex_status_text(st))
                done = st.state == "installed" and st.own
                b = self.ex["button"][kind]
                b.setEnabled(not busy and not done)
                b.setToolTip(_("Already installed.") if done else "")
        except RuntimeError:                         # the page was built again meanwhile
            pass

    def _ex_start(self, kind):
        if extracts.job().start(kind):
            self.ex["msg"].setText("")
            self.ex["bar"].setValue(0)
            self.ex["run"].setVisible(True)
            self._ex_progress(*extracts.job().last)
            self._ex_refresh()

    def _ex_open(self):
        try:
            ge.folder().mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(ge.folder())))

    @Slot(int, int, str)
    def _ex_progress(self, done, total, text):
        try:
            self.ex["bar"].setValue(int(1000 * done / max(total, 1)))
            self.ex["msg"].setText(text)
        except (KeyError, RuntimeError):             # no such page, or built again meanwhile
            pass

    @Slot(str, str)
    def _ex_finished(self, kind, outcome):
        try:
            self.ex["run"].setVisible(False)
            self.ex["msg"].setText(
                _("Done.") if not outcome else
                _("Cancelled. What was downloaded is kept: the next download goes on from there.")
                if outcome == "cancelled" else outcome)
        except (KeyError, RuntimeError):
            pass
        self._ex_refresh()

    def formats(self, v):
        self.row(v, _("Intermediate files"), self.combo([("FITS", "FITS", False), ("XISF", "XISF", False)],
                                                        prefs.get("format"), "format"))
        self.row(v, _("Bit depth"), self.combo([("32-bit float", "32-bit float", False),
                                                ("16-bit integer", "16-bit integer", False)], prefs.get("bits"),
                                               "bits"))
        self.row(v, _("Compression"), self.combo([("None", "None", False), ("Rice (FITS)", "Rice (FITS)", False),
                                                  ("Gzip", "Gzip", False)], prefs.get("compression"), "compression"))
        keep = QCheckBox(_("Keep the calibrated and registered frames after stacking"))
        keep.setChecked(bool(prefs.get("keep_frames")))
        keep.toggled.connect(lambda x: prefs.set("keep_frames", x))
        self.row(v, _("Intermediate frames"), keep)

    def masters(self, v):
        self.path_row(v, _("Library folder"), "masters_dir")
        self.row(v, _("Temperature tolerance"), self.spin(0.5, 10, "master_tolerance_temp", double=True), "°C")
        self.row(v, _("Maximum age of a dark"), self.spin(7, 3650, "master_tolerance_days"), _("days"))
        cols = [_("Type"), _("Exposure"), _("ISO"), _("Sensor"), _("Frames"), _("Date"), _("Used here")]
        tab = QTableWidget(0, len(cols))
        tab.setHorizontalHeaderLabels(cols)
        tab.verticalHeader().setVisible(False)
        tab.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tab.setStyleSheet("QTableWidget { background: #1c1f26; gridline-color: #2a2f38; } "
                          "QHeaderView::section { background: #23272f; border: none; padding: 4px; }")
        tab.resizeColumnsToContents()
        tab.horizontalHeader().setStretchLastSection(True)
        tab.setMinimumHeight(140)
        v.addWidget(tab)
        b = QPushButton(_("Build masters from a folder…"))
        v.addWidget(b, 0, Qt.AlignLeft)
        self._mark([tab, b], True)

    def display(self, v):
        st = QCheckBox(_("Auto-stretch linear images by default"))
        st.setChecked(bool(prefs.get("auto_stretch")))
        st.toggled.connect(lambda x: prefs.set("auto_stretch", x))
        self.row(v, _("Linear images"), st)
        self.row(v, _("Angular units"), self.combo([("arcsec", "arcsec", False), ("px", "px", False)],
                                                   prefs.get("units"), "units"))

    # ------------------------------------------------------------ disk use (in the worker)
    def measure_disks(self):
        if self.worker is None or not self.bars:
            return
        self.worker.submit("settings.disks", disk_usage, {"cache": prefs.get("cache_dir"),
                                                          "workspace": prefs.get("workspace"),
                                                          "temp": prefs.get("temp_dir")})

    def _done(self, channel, _n, result):
        if channel != "settings.disks":
            return
        for key, (bar, text) in self.bars.items():
            u = result.get(key)
            try:
                if u is None:
                    bar.setValue(0)
                    text.setText(_("Not found"))
                else:
                    bar.setValue(int(1000 * u[0] / max(u[1], 1e-9)))
                    text.setText(_("{u} GB of {t} GB").format(u=round(u[0]), t=round(u[1])))
            except RuntimeError:                     # the page was built again meanwhile
                return

    # ------------------------------------------------------------ actions
    def _lang(self, code):
        prefs.set("language", code)
        self.languageChanged.emit(code)
        QTimer.singleShot(0, self.rebuild)

    def _spec(self, on):
        prefs.set("spec_mode", on)
        self.specChanged.emit(on)
        QTimer.singleShot(0, self.rebuild)

    def _device(self, dev):
        prefs.set("device", dev)
        self.deviceChanged.emit(dev)

    def _workspace(self, folder):
        self.measure_disks()
        if getattr(self, "sec_ws", None) is not None:
            try:
                self.sec_ws.setText(folder)
            except RuntimeError:                     # the page was built again meanwhile
                pass
        self.workspaceChanged.emit(folder)

    # ------------------------------------------------------------ the sections of Home
    def _section_values(self) -> list[dict]:
        """The rows as edited: a name or a description left as shown (translated) keeps its stored English."""
        rows = []
        for w in self.sec_rows:
            r = dict(w["row"])
            name, note = w["name"].text().strip(), w["note"].text().strip()
            r["label"] = r["label"] if name == maybe(r["label"]) else name
            r["note"] = r["note"] if note == maybe(r["note"]) else note
            if not r["workspace"]:
                r["folder"] = w["folder"].text().strip()
            rows.append(r)
        return rows

    def _save_sections(self, rows: list[dict] | None = None):
        from .home import save_rows, section_rows
        rows = self._section_values() if rows is None else rows
        if rows != section_rows():
            save_rows(rows)
            self.homeChanged.emit()

    def _pick_section(self, e: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, _("Folder"), e.text())
        if d:
            e.setText(d)
            self._save_sections()

    def _remove_section(self, k: int):
        rows = self._section_values()
        if 0 <= k < len(rows) and not rows[k]["workspace"]:
            del rows[k]
            self._save_sections(rows)
            QTimer.singleShot(0, self.rebuild)

    def _add_section(self):
        rows = self._section_values() + [{"label": "", "folder": "", "note": "", "open": True, "workspace": False}]
        self._save_sections(rows)
        QTimer.singleShot(0, self.rebuild)
