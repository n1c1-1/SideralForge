"""Frames: the sessions of the project (one per night), how the calibration frames are matched to them, the health
check of the data set, and the frame browser.

An imported folder is sorted by itself: one session per night (noon to noon), then grouped by exposure, gain, filter
and temperature. Darks and bias come from the session or from the master library; flats belong to their session.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QDialog, QFrame, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from i18n import _
from steps import fnum
from widgets import OK, WARN, clear, label


def checks(s) -> list[tuple[str, str]]:
    """The health check of the data set: (level, text), level ok | warn | info."""
    out = []
    if not s.sessions:
        return out
    temps = [ses["temp"] for ses in s.sessions]
    out.append(("ok", _("Darks: library master 15 s · ISO 5000 · 20 °C (28 frames, 17 Aug 2026)")))
    for ses in s.sessions:
        d = abs(ses["temp"] - 20.0)
        if d > 2:
            out.append(("warn", _("Session {name}: sensor at {t} °C, master dark at 20 °C ({d} °C apart; tolerance "
                                  "2 °C). Dark optimization will scale it.").format(
                name=ses["name"], t=fnum(ses["temp"]), d=fnum(d))))
    out.append(("info", _("No bias: flats are calibrated with the camera black level (512 ADU).")))
    for ses in s.sessions:
        if ses.get("rotation"):
            out.append(("warn", _("Session {name}: camera rotated {deg}° — its lights use its own flats.").format(
                name=ses["name"], deg=fnum(ses["rotation"]))))
    if len(s.sessions) > 1:
        best = min(ses["fwhm"] for ses in s.sessions)
        for ses in s.sessions:
            if ses["fwhm"] > best * 1.1:
                out.append(("info", _("Session {name}: FWHM {f} px against {b} px — its frames weigh less.").format(
                    name=ses["name"], f=fnum(ses["fwhm"]), b=fnum(best))))
    n = sum(ses["lights"] for ses in s.sessions)
    out.append(("ok", _("Disk: about {gb} GB needed for calibrated frames and drizzle ×1.5; {free} GB free on the "
                        "cache drive.").format(gb=round(n * 0.26), free=412)))
    if len(set(temps)) == 1:
        pass
    return out


class GroupingDialog(QDialog):
    """How frames are grouped into sessions and matched with their calibration frames."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_("Grouping"))
        v = QVBoxLayout(self)
        v.addWidget(label(_("Group frames by"), "h2"))
        for text, on in ((_("Night (noon to noon)"), True), (_("Exposure"), True), (_("ISO / gain"), True),
                         (_("Filter"), True), (_("Sensor temperature (± 2 °C)"), True), (_("Binning"), True),
                         (_("Camera rotation"), True), (_("Object (FITS OBJECT keyword)"), False)):
            c = QCheckBox(text)
            c.setChecked(on)
            v.addWidget(c)
        v.addSpacing(6)
        ok = QPushButton(_("OK"))
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        v.addWidget(ok, 0, Qt.AlignRight)


class FramesView(QWidget):
    """The Frames block, inspected: sessions and matching, or the frame browser."""
    addSources = Signal()
    addSession = Signal()
    stackMode = Signal(str)

    def __init__(self, project, browser, parent=None):
        super().__init__(parent)
        self.s = project
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
        self.b_import.clicked.connect(self.addSources.emit)
        self.b_add = QPushButton(_("Add session…"))
        self.b_add.clicked.connect(self.addSession.emit)
        gr = QPushButton(_("Grouping…"))
        gr.clicked.connect(lambda: GroupingDialog(self.window()).exec())
        for b in (self.b_import, self.b_add, gr):
            h.addWidget(b)
        v.addWidget(bar)
        self.pages = QStackedWidget()
        v.addWidget(self.pages, 1)
        sp = QWidget()
        self.sv = QVBoxLayout(sp)
        self.sv.setContentsMargins(16, 12, 16, 12)
        self.sv.setSpacing(8)
        self.pages.addWidget(sp)
        self.browser = browser
        self.pages.addWidget(browser)
        self.show_page(0)

    def show_page(self, i):
        self.pages.setCurrentIndex(i)
        self.tabs[i].setChecked(True)
        if i == 0:
            self.refresh()

    def refresh(self):
        clear(self.sv)
        s = self.s
        self.b_add.setEnabled(s.sources_loaded)
        if not s.sources_loaded:
            self.sv.addStretch(1)
            self.sv.addWidget(label(_("No frames yet"), "h1"), 0, Qt.AlignHCenter)
            b = QPushButton(_("Import folder…"))
            b.setObjectName("primary")
            b.clicked.connect(self.addSources.emit)
            self.sv.addWidget(b, 0, Qt.AlignHCenter)
            self.sv.addStretch(2)
            return
        self.sv.addWidget(label(_("Sessions"), "h2"))
        cols = [_("Session"), _("Night"), _("Lights"), _("Exposure"), _("ISO / gain"), _("Temp."), _("Filter"),
                _("Flats"), _("Darks"), _("Bias"), "FWHM", _("Kept")]
        t = QTableWidget(len(s.sessions), len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setStyleSheet("QTableWidget { background: #1c1f26; gridline-color: #2a2f38; } "
                        "QHeaderView::section { background: #23272f; border: none; padding: 4px; }")
        for r, ses in enumerate(s.sessions):
            vals = [ses["name"], ses["night"], str(ses["lights"]), f"{ses['exposure']} s", str(ses["iso"]),
                    f"{fnum(ses['temp'])} °C", ses["filter"], str(ses["flats"]),
                    _("library") if ses["darks"] == "library" else str(ses["darks"]),
                    _("none") if ses["bias"] == "none" else str(ses["bias"]), f"{fnum(ses['fwhm'])} px",
                    f"{ses['kept']} / {ses['lights']}"]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(val)
                if c >= 2:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r, c, it)
        t.resizeColumnsToContents()
        for c in range(len(cols)):
            t.setColumnWidth(c, t.columnWidth(c) + 18)
        t.horizontalHeader().setStretchLastSection(True)
        t.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        t.setFixedHeight(t.horizontalHeader().height() + sum(t.rowHeight(r) for r in range(len(s.sessions))) + 6)
        self.sv.addWidget(t)
        self.sv.addWidget(label(_("Grouped by night (noon to noon), exposure, ISO, filter, sensor temperature, "
                                  "binning and rotation."), "small"))
        self.sv.addSpacing(6)
        self.sv.addWidget(label(_("Health check"), "h2"))
        for lvl, text in checks(s):
            row = QHBoxLayout()
            mark = QLabel({"ok": "✓", "warn": "⚠", "info": "i"}[lvl])
            mark.setFixedWidth(18)
            mark.setStyleSheet(f"color: {OK if lvl == 'ok' else WARN if lvl == 'warn' else '#8b93a1'}; "
                               "font-weight: 700;")
            row.addWidget(mark, 0, Qt.AlignTop)
            row.addWidget(label(text, "", wrap=True), 1)
            self.sv.addLayout(row)
        if len(s.sessions) > 1:
            self.sv.addSpacing(6)
            self.sv.addWidget(label(_("Stacking"), "h2"))
            row = QHBoxLayout()
            g = QButtonGroup(self)
            for text, mode in ((_("All sessions together"), "together"), (_("One stack per session"), "per_session")):
                b = QPushButton(text)
                b.setObjectName("face")
                b.setCheckable(True)
                b.setChecked(s.stack_mode == mode)
                b.clicked.connect(lambda _x=False, m=mode: self.stackMode.emit(m))
                g.addButton(b)
                row.addWidget(b)
            row.addStretch(1)
            self.sv.addLayout(row)
        self.sv.addStretch(1)
