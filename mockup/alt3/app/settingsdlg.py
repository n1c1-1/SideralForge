"""Settings (application preferences): General, Folders & disk, Compute, Catalogs & astrometry, File formats, Master
library, Display. Kept in prefs; the language and the device take effect at once."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QProgressBar, QPushButton, QSpinBox,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

import prefs
from i18n import LANGUAGES, _
from project import MASTERS_DEMO
from widgets import label

PAGES = ["General", "Folders & disk", "Compute", "Catalogs & astrometry", "File formats", "Master library", "Display"]


def row(form, text, w, unit=""):
    h = QHBoxLayout()
    lb = QLabel(text)
    lb.setFixedWidth(200)
    h.addWidget(lb)
    h.addWidget(w, 1)
    if unit:
        h.addWidget(label(unit, "small"))
    form.addLayout(h)
    return w


def combo(items, current, key=None, values=None):
    c = QComboBox()
    for i, t in enumerate(items):
        c.addItem(_(t), values[i] if values else t)
    j = c.findData(current)
    c.setCurrentIndex(max(j, 0))
    if key:
        c.currentIndexChanged.connect(lambda _i: prefs.set(key, c.currentData()))
    return c


def path_row(form, text, key):
    h = QHBoxLayout()
    lb = QLabel(text)
    lb.setFixedWidth(200)
    h.addWidget(lb)
    e = QLineEdit(prefs.get(key))
    e.editingFinished.connect(lambda: prefs.set(key, e.text()))
    h.addWidget(e, 1)
    b = QPushButton(_("Browse…"))

    def pick():
        d = QFileDialog.getExistingDirectory(None, text, e.text())
        if d:
            e.setText(d)
            prefs.set(key, d)
    b.clicked.connect(pick)
    h.addWidget(b)
    form.addLayout(h)


class SettingsDialog(QDialog):
    languageChanged = Signal(str)
    deviceChanged = Signal(bool)          # True: GPU
    specChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_("Settings"))
        self.resize(900, 600)
        h = QHBoxLayout(self)
        self.nav = QListWidget()
        self.nav.setFixedWidth(200)
        for p in PAGES:
            self.nav.addItem(_(p))
        h.addWidget(self.nav)
        self.pages = QStackedWidget()
        h.addWidget(self.pages, 1)
        for build in (self.general, self.folders, self.compute, self.catalogs, self.formats, self.masters,
                      self.display):
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(16, 8, 8, 8)
            v.setSpacing(10)
            build(v)
            v.addStretch(1)
            self.pages.addWidget(w)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)

    # ------------------------------------------------------------ pages
    def general(self, v):
        v.addWidget(label(_("General"), "h1"))
        lang = combo(list(LANGUAGES.values()), prefs.get("language"), values=list(LANGUAGES))
        lang.currentIndexChanged.connect(lambda _i: self._lang(lang.currentData()))
        row(v, _("Language"), lang)
        spec = QCheckBox(_("Show what the engine still has to build (◇), estimated durations and demo notes"))
        spec.setChecked(prefs.spec())
        spec.toggled.connect(self._spec)
        row(v, _("Spec mode"), spec)

    def folders(self, v):
        v.addWidget(label(_("Folders & disk"), "h1"))
        path_row(v, _("Workspace"), "workspace")
        path_row(v, _("Cache (calibrated frames)"), "cache_dir")
        path_row(v, _("Temporary files"), "temp_dir")
        lim = QSpinBox()
        lim.setRange(10, 4000)
        lim.setValue(prefs.get("cache_limit_gb"))
        lim.valueChanged.connect(lambda x: prefs.set("cache_limit_gb", x))
        row(v, _("Cache limit"), lim, "GB")
        v.addWidget(label(_("Disk use"), "h2"))
        for name, used, total in ((_("Cache"), 186, 412 + 186), (_("Workspace"), 954, 1863), (_("Temporary"), 3, 412)):
            r = QHBoxLayout()
            lb = QLabel(name)
            lb.setFixedWidth(200)
            r.addWidget(lb)
            bar = QProgressBar()
            bar.setRange(0, total)
            bar.setValue(used)
            bar.setTextVisible(False)
            r.addWidget(bar, 1)
            r.addWidget(label(_("{u} GB of {t} GB").format(u=used, t=total), "small"))
            v.addLayout(r)
        clean = QPushButton(_("Clear cache of closed projects"))
        v.addWidget(clean, 0, Qt.AlignLeft)

    def compute(self, v):
        v.addWidget(label(_("Compute"), "h1"))
        dev = combo(["Automatic", "NVIDIA RTX 4060 Ti", "CPU only"], prefs.get("device"),
                    values=["auto", "gpu0", "cpu"])
        dev.currentIndexChanged.connect(lambda _i: self._device(dev.currentData()))
        row(v, _("Device"), dev)
        th = QSpinBox()
        th.setRange(0, 32)
        th.setSpecialValueText(_("Automatic (14 of 16)"))
        th.setValue(prefs.get("threads"))
        th.valueChanged.connect(lambda x: prefs.set("threads", x))
        row(v, _("CPU threads"), th)
        mem = QSpinBox()
        mem.setRange(4, 128)
        mem.setValue(prefs.get("memory_limit_gb"))
        mem.valueChanged.connect(lambda x: prefs.set("memory_limit_gb", x))
        row(v, _("Memory limit"), mem, _("GB of 32 GB"))
        v.addWidget(label(_("Benchmark"), "h2"))
        v.addWidget(label(_("Measures this machine on a small data set, so that the durations shown are real."),
                          "small", wrap=True))
        self.bench_bar = QProgressBar()
        self.bench_bar.setTextVisible(False)
        self.bench_res = label(self._bench_text(), "small", wrap=True)
        b = QPushButton(_("Run benchmark"))
        b.clicked.connect(self._bench)
        r = QHBoxLayout()
        r.addWidget(b)
        r.addWidget(self.bench_bar, 1)
        v.addLayout(r)
        v.addWidget(self.bench_res)

    def catalogs(self, v):
        v.addWidget(label(_("Catalogs & astrometry"), "h1"))
        row(v, _("Gaia DR3"), combo(["Local copy", "Online (VizieR)"], prefs.get("gaia"), "gaia",
                                    ["local", "online"]))
        path_row(v, _("Gaia DR3 folder"), "gaia_path")
        dl = QPushButton(_("Download Gaia DR3 extract (12 GB)…"))
        v.addWidget(dl, 0, Qt.AlignLeft)
        row(v, _("Plate solver"), combo(["Internal (Gaia)", "ASTAP", "astrometry.net (local)"], prefs.get("solver"),
                                        "solver", ["internal", "astap", "astrometry.net"]))
        path_row(v, _("Survey cache (H-alpha)"), "survey_cache")

    def formats(self, v):
        v.addWidget(label(_("File formats"), "h1"))
        row(v, _("Intermediate files"), combo(["FITS", "XISF"], prefs.get("format"), "format"))
        row(v, _("Bit depth"), combo(["32-bit float", "16-bit integer"], prefs.get("bits"), "bits"))
        row(v, _("Compression"), combo(["None", "Rice (FITS)", "Gzip"], prefs.get("compression"), "compression"))
        keep = QCheckBox(_("Keep the calibrated and registered frames after stacking"))
        keep.setChecked(True)
        row(v, _("Intermediate frames"), keep)

    def masters(self, v):
        v.addWidget(label(_("Master library"), "h1"))
        path_row(v, _("Library folder"), "masters_dir")
        t = QDoubleSpinBox()
        t.setRange(0.5, 10)
        t.setValue(prefs.get("master_tolerance_temp"))
        t.valueChanged.connect(lambda x: prefs.set("master_tolerance_temp", x))
        row(v, _("Temperature tolerance"), t, "°C")
        d = QSpinBox()
        d.setRange(7, 3650)
        d.setValue(prefs.get("master_tolerance_days"))
        d.valueChanged.connect(lambda x: prefs.set("master_tolerance_days", x))
        row(v, _("Maximum age of a dark"), d, _("days"))
        cols = [_("Type"), _("Exposure"), _("ISO"), _("Sensor"), _("Frames"), _("Date"), _("Used here")]
        tab = QTableWidget(len(MASTERS_DEMO), len(cols))
        tab.setHorizontalHeaderLabels(cols)
        tab.verticalHeader().setVisible(False)
        tab.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tab.setStyleSheet("QTableWidget { background: #1c1f26; gridline-color: #2a2f38; } "
                          "QHeaderView::section { background: #23272f; border: none; padding: 4px; }")
        for r_, m in enumerate(MASTERS_DEMO):
            for c, val in enumerate((_(m["kind"]), m["exposure"], str(m["iso"]), m["temp"], str(m["frames"]),
                                     m["date"], "✓" if m["used"] else "")):
                tab.setItem(r_, c, QTableWidgetItem(val))
        tab.resizeColumnsToContents()
        tab.horizontalHeader().setStretchLastSection(True)
        v.addWidget(tab)
        b = QPushButton(_("Build masters from a folder…"))
        v.addWidget(b, 0, Qt.AlignLeft)

    def display(self, v):
        v.addWidget(label(_("Display"), "h1"))
        st = QCheckBox(_("Auto-stretch linear images by default"))
        st.setChecked(prefs.get("auto_stretch"))
        st.toggled.connect(lambda x: prefs.set("auto_stretch", x))
        row(v, _("Linear images"), st)
        row(v, _("Angular units"), combo(["arcsec", "px"], prefs.get("units"), "units"))

    # ------------------------------------------------------------ actions
    def _lang(self, code):
        prefs.set("language", code)
        self.languageChanged.emit(code)

    def _spec(self, on):
        prefs.set("spec_mode", on)
        self.specChanged.emit(on)

    def _device(self, dev):
        prefs.set("device", dev)
        self.deviceChanged.emit(dev != "cpu")

    def _bench_text(self):
        b = prefs.get("benchmark")
        if not b:
            return _("Not run yet: the durations shown are those measured on the reference machine.")
        return _("Last run {date}: GPU {g}× and CPU {c}× the reference machine.").format(**b)

    def _bench(self):
        self.bench_bar.setValue(0)
        self._t = QTimer(self)

        def tick():
            self.bench_bar.setValue(self.bench_bar.value() + 4)
            if self.bench_bar.value() >= 100:
                self._t.stop()
                import time
                prefs.set("benchmark", {"date": time.strftime("%Y-%m-%d %H:%M"), "g": "1.00", "c": "0.92"})
                self.bench_res.setText(self._bench_text())
        self._t.timeout.connect(tick)
        self._t.start(40)
