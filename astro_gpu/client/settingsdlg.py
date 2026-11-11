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
from .icons import icon
from .services import extracts
from .services.selftest import job as card_test_job
from .services.worker import Machine, disk_usage
from .widgets import cap_dialog, label, spec_mark

# (key, title, ◇ page)
PAGES = [("general", "General", False), ("folders", "Folders & disk", False), ("compute", "Compute", False),
         ("catalogs", "Catalogs & astrometry", False), ("formats", "File formats", True),
         ("masters", "Master library", True), ("display", "Display", False)]
#: the pages' icons in the settings' list
PAGE_ROLES = {"general": "adjust", "folders": "folder", "compute": "compute", "catalogs": "catalogue", "formats": "file",
              "masters": "layers", "display": "display"}
LABEL_MIN = 200


def not_yet() -> str:
    return "◇ " + _("Not in the engine yet.")


def auto_device(machine: Machine | None) -> str | None:
    """The device Automatic computes on: the GPU with the most memory, the CPU without a GPU; None while the machine is
    not known yet."""
    if machine is None:
        return None
    g = max(machine.gpus, key=lambda x: x.memory_gb) if machine.gpus else None
    return g.device if g is not None else "cpu"


def device_items(machine: Machine | None, dev: str = "auto") -> list[tuple[str, str, str]]:
    """(text, device, tooltip) of the devices a run can compute on, translated: Automatic (it says what it picks), the
    machine's GPUs, the CPU. Settings > Compute and the combo left of Run list the same (GUI exploration 6.9)."""
    auto = auto_device(machine)
    g = machine.gpu(auto) if machine is not None and auto else None
    out = [(_("Automatic"), "auto", _("The graphics card with the most memory: {name}").format(name=g.name) if g else
            _("The processor: this computer has no graphics card the engine can use") if auto == "cpu" else "")]
    out += [(x.name + (f" ({_('beta')})" if x.beta else ""), x.device,
             _("Graphics card, {gb} GB of memory").format(gb=f"{x.memory_gb:.0f}")
             + (" · " + _("Beta: this kind of card is not tested by the program's authors; the card test decides "
                          "which blocks use it") if x.beta else ""))
            for x in (machine.gpus if machine else [])]
    if machine is None and dev.startswith("cuda:"):
        out.append((dev, dev, ""))
    out.append((_("CPU only"), "cpu", _("The processor: much slower")))
    return out


class SettingsDialog(QDialog):
    languageChanged = Signal(str)
    deviceChanged = Signal(str)                      # auto | cuda:<n> | cpu
    specChanged = Signal(bool)
    homeChanged = Signal()                           # the sections of Home (Folders page)
    clearCache = Signal()
    densityChanged = Signal()                        # the compact look was set (issue #44)

    def __init__(self, worker=None, machine: Machine | None = None, can_clear: bool = False, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.machine = machine
        self.can_clear = can_clear
        self.resize(900, 600)
        cap_dialog(self)
        self.h = QHBoxLayout(self)
        self.nav = self.pages = None
        self.keys: list[str] = []
        self.bars: dict[str, tuple[QProgressBar, QLabel]] = {}
        self.ex: dict = {}                           # the widgets of the offline extracts (Catalogs & astrometry)
        self.ct: dict = {}                           # the widgets of the card test (Compute)
        if worker is not None:
            worker.done.connect(self._done)
        j = extracts.job()                           # the download goes on when this window is closed
        j.progress.connect(self._ex_progress, Qt.QueuedConnection)
        j.finished.connect(self._ex_finished, Qt.QueuedConnection)
        card_test_job().changed.connect(self._ct_refresh, Qt.QueuedConnection)
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
        self.sec_rows = []
        self.nav = QListWidget()
        self.nav.setFixedWidth(200)
        self.h.addWidget(self.nav)
        self.pages = QStackedWidget()
        self.h.addWidget(self.pages, 1)
        self.keys = []
        for key, title, new in PAGES:
            if new and not self.spec:
                continue
            item = QListWidgetItem(icon(PAGE_ROLES[key], 16), _(title) + ("  ◇" if new else ""))
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
            # every page scrolls (the Folders page is the longest; issue #44: a small window's dialog is capped)
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
        spec = QCheckBox(_("Show ◇"))                      # compact names (decision 12): the sentence in the tooltip
        spec.setToolTip(_("Show what the engine still has to build (◇)"))
        spec.setChecked(self.spec)
        spec.toggled.connect(self._spec)
        self.row(v, _("Spec mode"), spec)
        dens = QComboBox()                                # issue #44: how the window adapts to its size
        dens.setToolTip(_("Auto: compact and icons on a small window · Never: today's layout on any screen"))
        for text, key in (("Auto", "auto"), ("Always compact", "compact"), ("Never compact", "comfortable")):
            dens.addItem(_(text), key)
        dens.setCurrentIndex(max(dens.findData(prefs.get("layout_density") or "auto"), 0))
        dens.currentIndexChanged.connect(lambda _i: self._density(dens.currentData()))
        self.row(v, _("Interface density"), dens)

    def _density(self, key: str):
        prefs.set("layout_density", key)
        self.densityChanged.emit()

    def folders(self, v):
        self.path_row(v, _("Cache (calibrated frames)"), "cache_dir",
                      placeholder=_("In the project folder"), changed=lambda _x: self.measure_disks())
        self.path_row(v, _("Temporary files"), "temp_dir", new=True)
        limit = self.spin(10, 4000, "cache_limit_gb")
        limit.valueChanged.connect(lambda _x: self.measure_disks())      # the Cache bar is read against it
        self.row(v, _("Cache limit"), limit, _("GB"), new=True)
        v.addWidget(label(_("Disk use"), "h2"))
        for key, name, new in (("cache", _("Cache"), False), ("temp", _("Temporary"), True)):
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
        clean = QPushButton(_("Clear cache"))
        clean.setToolTip(_("Clear cache of closed projects"))
        clean.setEnabled(self.can_clear)
        clean.clicked.connect(lambda: self.clearCache.emit())
        v.addWidget(clean, 0, Qt.AlignLeft)
        self.home_sections(v)

    def home_sections(self, v):
        """The sections of Home, under the recent projects: a row each (name, folder, description)."""
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
        add = QPushButton(_("Add section"))
        add.clicked.connect(self._add_section)
        v.addWidget(add, 0, Qt.AlignLeft)

    def compute(self, v):
        m = self.machine
        dev = prefs.get("device") or "auto"
        c = QComboBox()
        for text, data, tip in device_items(m, dev):           # the same list as the combo left of Run (6.9)
            c.addItem(text, data)
            c.setItemData(c.count() - 1, tip, Qt.ToolTipRole)
        c.setCurrentIndex(max(c.findData(dev), 0))
        c.currentIndexChanged.connect(lambda _i: self._device(c.currentData()))
        self.row(v, _("Device"), c)
        self.card_test(v)
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

    def card_test(self, v):
        """Whether each heavy block keeps the graphics card here (core/selftest.py), in plain words, and the button that
        tests it again. The test runs in the background (services/selftest.py)."""
        v.addWidget(label(_("Card test"), "h2"))
        status = label("", "small", wrap=True)
        v.addWidget(status)
        rows, names = {}, []
        for stage, name in ((("stack", _("Stack")), ("denoise", _("Denoise")), ("star_removal", _("Stars")),
                             ("post", _("Post")))):
            val = label("", "small")
            self.row(v, name, val)
            rows[stage] = val
            names.append(self._labels[-1])
        b = QPushButton(_("Test again"))
        b.setToolTip(_("Run the card test again now, in the background"))
        b.clicked.connect(self._ct_start)
        v.addWidget(b, 0, Qt.AlignLeft)
        self.ct = {"status": status, "rows": rows, "button": b, "names": names}
        self._ct_refresh()

    def _ct_device(self) -> tuple[str, str | None]:
        """(the setting, the device it names), None when there is no card to test."""
        pref = prefs.get("device") or "auto"
        dev = auto_device(self.machine) if pref == "auto" else pref
        return pref, dev if dev not in (None, "cpu") else None

    def _ct_start(self):
        pref, dev = self._ct_device()
        if dev is not None:
            card_test_job().start(pref, dev, force=True)

    @Slot()
    def _ct_refresh(self):
        if not self.ct:
            return
        try:
            self._ct_show()
        except RuntimeError:                         # the page was built again meanwhile
            pass

    def _ct_show(self):
        from ..core import selftest as st
        pref, dev = self._ct_device()
        job = card_test_job()
        profile = st.profile_of(prefs.get("compute_profiles"), pref) if dev is not None else None
        self.ct["button"].setEnabled(dev is not None and not job.running())
        # nothing to test on the processor: the section is its sentence, the button greyed says why (the night test
        # pass: four empty lines, and a greyed button whose tip did not say why)
        self.ct["button"].setToolTip(_("Run the card test again now, in the background") if dev is not None else
                                     _("No graphics card computes here: nothing to test"))
        for w in [*self.ct["names"], *self.ct["rows"].values()]:
            w.setVisible(dev is not None)
        if job.running():
            text = _("Testing the card…")
        elif dev is None:
            text = _("The processor computes: nothing to test")
        elif profile is None:
            text = (_("The card is in use: tested when it is free") if job.note == "busy" else _("Not tested yet"))
        else:
            text = _("{card} · driver {driver} · tested {when}").format(card=profile.get("card", ""),
                                                                        driver=profile.get("driver") or "?",
                                                                        when=profile.get("when", ""))
        self.ct["status"].setText(text)
        for stage, lab in self.ct["rows"].items():
            lab.setText("")
            lab.setToolTip("")
            if profile is None:
                continue
            ops = [r for r in (profile.get("ops") or {}).values() if r.get("stage") == stage]
            if (profile.get("stages") or {}).get(stage) == "cpu":
                short = {"wrong": _("wrong result"), "slower": _("slower"), "error": _("card error")}
                bad = next((r for r in ops if r["verdict"] not in ("ok", "absent")), {})
                lab.setText(_("CPU · {why}").format(why=short.get(bad.get("verdict"), _("card error"))))
                lab.setToolTip(_(st.stage_reason(profile, stage)) + (f" ({bad.get('error')})" if bad.get("error") else ""))
                continue
            timed = [r for r in ops if r.get("card_s") and r.get("cpu_s")]
            if timed:
                ratio = min(r["cpu_s"] / r["card_s"] for r in timed)
                lab.setText(_("GPU · {n} x faster").format(n=number(ratio, 1)))
                lab.setToolTip(_("Card {card} s · processor {cpu} s").format(
                    card=number(sum(r["card_s"] for r in timed), 2), cpu=number(sum(r["cpu_s"] for r in timed), 2)))
            else:
                lab.setText(_("GPU · not tested"))

    def catalogs(self, v):
        self.offline_extracts(v)
        self.asterion_status(v)
        self.row(v, _("Plate solver"), self.combo([("Internal (Gaia)", "internal", False),
                                                   ("External solver (installed separately)", "astap", True),
                                                   ("External all-sky index solver (installed separately)",
                                                    "astrometry.net", True)],
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
        b = QPushButton(_("Download (offline)"))
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

    def asterion_status(self, v):
        """Asterion, the plate solver of Astrometry's first solution (core/asterion.py): the program and its star index,
        installed or not, and where. No install button yet: how they reach a computer is not decided."""
        from ..core import asterion
        st = asterion.status()
        v.addWidget(label(_("Asterion plate solver"), "h2"))
        v.addWidget(label(_("Gives Astrometry its first solution in a second, at any scale; without it, the field is "
                            "found by FLOAT's own search, minutes when the focal length is unknown."), "small", wrap=True))
        if st.program is None:
            prog = _("Not installed")
        elif not (st.ok or st.missing):
            prog = _("Cannot run: {why}").format(why=st.why)
        else:
            prog = _("Version {version}").format(version=st.version) + f" · {st.program}"
        self.row(v, _("Program"), label(prog, "small", wrap=True))
        cat = asterion.find_catalogs(program=st.program)
        idx = f"{', '.join(asterion.indexes(cat))} · {cat}" if cat is not None else _("Not installed")
        self.row(v, _("Star index"), label(idx, "small", wrap=True))
        v.addSpacing(6)

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
        for kind, name, button, tip in ((ge.ASTRO, _("Star positions"), _("Download positions ({size})"),
                                         _("Download star positions ({size})")),
                                        (ge.XP, _("Star spectra"), _("Download spectra ({size})"),
                                         _("Download star spectra ({size})"))):
            st = label("", "small", wrap=True)
            self.row(v, name, st)
            gb = f"{number(ge.DOWNLOAD_BYTES[kind] / 1e9)} {_('GB')}"
            b = QPushButton(button.format(size=gb))
            b.setToolTip(tip.format(size=gb))
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
        keep = QCheckBox(_("Keep them"))                   # the row says which: "Intermediate frames"
        keep.setToolTip(_("Keep calibrated + registered frames"))
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
        b = QPushButton(_("Import masters…"))
        b.setToolTip(_("Masters from folder…"))
        v.addWidget(b, 0, Qt.AlignLeft)
        self._mark([tab, b], True)

    def display(self, v):
        st = QCheckBox(_("Auto-stretch"))                  # the row says which: "Linear images"
        st.setToolTip(_("Auto-stretch linear images by default"))
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
                    continue
                own, free = u
                if key == "cache":                   # the cache against its limit, never the whole drive (exploration 2)
                    limit = float(prefs.get("cache_limit_gb") or 1)
                    bar.setValue(min(int(1000 * own / max(limit, 1e-9)), 1000))
                    text.setText(_("{u} GB of {t} GB").format(u=number(own, 1), t=round(limit)))
                    bar.setToolTip(_("What the cache holds, against its limit; {f} GB free on its drive").format(
                        f=round(free)))
                else:                                # the temporary files against the room their drive leaves
                    bar.setValue(min(int(1000 * own / max(own + free, 1e-9)), 1000))
                    text.setText(_("{u} GB · {f} GB free").format(u=number(own, 1), f=round(free)))
                    bar.setToolTip(_("What the temporary files take, against the room left on their drive"))
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
        self._ct_refresh()

    # ------------------------------------------------------------ the sections of Home
    def _section_values(self) -> list[dict]:
        """The rows as edited: a name or a description left as shown (translated) keeps its stored English."""
        rows = []
        for w in self.sec_rows:
            r = dict(w["row"])
            name, note = w["name"].text().strip(), w["note"].text().strip()
            r["label"] = r["label"] if name == maybe(r["label"]) else name
            r["note"] = r["note"] if note == maybe(r["note"]) else note
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
        if 0 <= k < len(rows):
            del rows[k]
            self._save_sections(rows)
            QTimer.singleShot(0, self.rebuild)

    def _add_section(self):
        rows = self._section_values() + [{"label": "", "folder": "", "note": "", "open": True, "only": ""}]
        self._save_sections(rows)
        QTimer.singleShot(0, self.rebuild)
