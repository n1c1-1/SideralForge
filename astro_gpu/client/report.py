"""The Report tab: the measures of a workflow as a scientific report (issue #81), and the night card to post with an
image.

A document, not a dashboard: a white page, numbered sections, tables with thin rules, plain black figures with their
captions. Nothing is measured or computed here: the tables are the blocks' own facts (``Steps.facts``). The figures are
drawn for the report; Inspect keeps its interactive charts. ``report_data`` reads the files and draws the figures, in
the worker thread. A report holds no path and no folder name: it is made to be shared.
"""
from __future__ import annotations

import html
import io
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from PySide6.QtCore import QMarginsF, QSizeF, Qt, QUrl, Signal
from PySide6.QtGui import QFont, QImage, QPainter, QPageSize, QPdfWriter, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import APP_NAME
from .blocks import STATE_WORD, block_state
from .charts import LINES, RegistrationData, registration_data
from .i18n import _, hm, language, number

#: the sections that are blocks of the chain, in its order: (block key, title)
BLOCKS = (("calibration", "Calibration"), ("alignement", "Registration"), ("selection", "Selection"),
          ("empilement", "Stacking"), ("astrometrie", "Astrometry"), ("fond", "Gradient"), ("couleur", "Color"))
PAGE_W = 794                                             # the page on the screen, px (A4 at 96 dpi)
FIG_W = 640                                              # a figure in the page, px
#: what the files do not say: entered with a click, kept in the project
ENTERED = ("instrument", "mount", "filter", "location", "date")
#: the blocks' figures a reader of the report has no use for, and the star size on 2 x 2 super-pixels, which overestimates
#: undersampled stars: the report's sizes are the Moffat fit's
HIDDEN = ("Star FWHM range", "Cache size", "Flat minimum")
#: the words of the shooting details, in the language of the forum whatever the language of the screen
WORDS = {"en": dict(camera="Camera", instrument="Instrument", mount="Mount", filter="Filter", location="Location",
                    date="Date", frames="Frames", darks="Darks", flats="Flats", field="Field", sampling="Sampling",
                    fwhm="Median FWHM", kept="Frames kept", processing="Processing"),
         "fr": dict(camera="Appareil", instrument="Instrument", mount="Monture", filter="Filtre", location="Lieu",
                    date="Date", frames="Poses", darks="Darks", flats="Flats", field="Champ",
                    sampling="Échantillonnage", fwhm="FWHM médiane", kept="Poses gardées", processing="Traitement")}
#: a value that names a place on a disk: never in a report
PATHS = re.compile(r"[A-Za-z]:[\\/]|\\\\|(^|\s)/[^\s]+/")
STYLE = {"font.family": "DejaVu Serif", "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
         "ytick.major.width": 0.6, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
         "ytick.right": True, "axes.labelsize": 8, "legend.frameon": False, "legend.fontsize": 7}
INK, GREY = "#000000", "#808080"
#: the sensor's three curves: dark, thin, printable
RGB = ("#a02c2c", "#2c7a3f", "#27508f")


@dataclass
class NightFigure:
    """A session's night: its name, the frames it holds, the figure of the report and the card to post (PNG files)."""

    name: str
    idx: list
    figure: bytes
    card: bytes


@dataclass
class ReportData:
    """What the report reads from the files. The arrays have one value per light frame. ``settings``: [(block title,
    setting, value)] of the done stages."""

    reg: RegistrationData | None = None
    sessions: list = field(default_factory=list)         # [(name, frame indices)]
    kept: np.ndarray | None = None
    fwhm: np.ndarray | None = None                       # the Moffat FWHM of each frame, sensor px
    roundness: np.ndarray | None = None
    sky: np.ndarray | None = None                        # the sky level of each frame, ADU
    exposure: np.ndarray | None = None                   # s
    spectral: object = None
    settings: list = field(default_factory=list)
    nights: list = field(default_factory=list)           # [NightFigure]
    drift: bytes | None = None
    response: bytes | None = None
    astro: dict = field(default_factory=dict)            # the Astrometry block's metrics
    thumb: bytes | None = None                           # the picture Post last showed


def shareable(v) -> bool:
    """Whether a setting's value can go in a report: a plain value that names no place on a disk."""
    return isinstance(v, (bool, int, float, str)) and not PATHS.search(str(v))


# ------------------------------------------------------------ reading (worker thread)
def _settings(project, wf, stages) -> list:
    from .. import api
    from ..stages.v2 import V2_STAGES
    known = {st.name: st for st in V2_STAGES()}
    rows = []
    for stage in stages:
        st = known.get(stage.split("~")[0].split("@")[0])
        try:
            out = api.stage_outputs(project, wf, stage) if st is not None else None
        except (OSError, ValueError, KeyError):
            out = None
        if out is None:
            continue
        for key, value in out.params.items():
            info = st.Params.model_fields.get(key)
            if info is not None and shareable(value):
                rows.append((st.title, (info.json_schema_extra or {}).get("label", key), value, value != info.default))
    return rows


def _sky_levels(project, wf, n: int) -> np.ndarray | None:
    """The sky level of each frame: calibrate's statistics, those of the sessions together when they are apart."""
    from .. import api
    for stage in ("sessions", "calibrate"):
        try:
            out = api.stage_outputs(project, wf, stage)
        except (OSError, ValueError, KeyError):
            continue
        f = Path(out.dir) / "frame_stats.npy" if out is not None else None
        if f is not None and f.is_file():
            st = np.load(f)
            if len(st) == n:
                return np.asarray(st[:, :, 0], float).mean(1)
    return None


def _exposures(project, wf, table) -> np.ndarray:
    from .. import api
    out = np.full(len(table.rows), np.nan)
    inv = api.stage_outputs(project, wf, "inventory")
    f = Path(inv.dir) / "frames.json" if inv is not None else None
    if f is not None and f.is_file():
        lights = json.loads(f.read_text(encoding="utf-8")).get("lights") or []
        for i, r in enumerate(table.rows):
            if r.index < len(lights) and lights[r.index].get("exposure"):
                out[i] = float(lights[r.index]["exposure"])
    return out


def report_data(project, wf, stages=(), title: str = "") -> ReportData:
    """Worker thread. What is not computed yet is left empty: its section says so."""
    from .. import api
    d = ReportData(settings=_settings(project, wf, stages))
    try:
        d.reg = registration_data(project, wf)
        table = api.frame_table(project, wf)
    except (api.NotDone, OSError, KeyError, ValueError):
        d.reg = None
    if d.reg is not None:
        n = d.reg.n
        by: dict = {}
        for i, r in enumerate(table.rows):
            by.setdefault(r.session, []).append(i)
        names = {s.id: s.name for s in project.sessions}
        d.sessions = [(names.get(sid, ""), idx) for sid, idx in by.items()]
        d.kept = np.array([not r.status and not r.excluded for r in table.rows], bool)
        with api.stage_array(project, wf, "register", "reg.npz") as z:
            if "fwhm_moffat" in z.files:
                d.fwhm = np.array(z["fwhm_moffat"][:n], float)
            if "roundness_moffat" in z.files:
                d.roundness = np.array(z["roundness_moffat"][:n], float)
        d.sky = _sky_levels(project, wf, n)
        d.exposure = _exposures(project, wf, table)
    try:
        d.spectral = api.spectral_response(project, wf)
    except Exception:  # noqa: BLE001 - a diagnostic that cannot be made: its figure is left out
        d.spectral = None
    try:
        ast = api.stage_outputs(project, wf, "astrometry")
    except (OSError, ValueError, KeyError):
        ast = None
    d.astro = dict(ast.metrics) if ast is not None else {}
    thumb = Path(project.root) / "cache" / "thumb.jpg" if project.root is not None else None
    if thumb is not None and thumb.is_file():
        d.thumb = thumb.read_bytes()
    draw_figures(d, title)
    return d


# ------------------------------------------------------------ the figures
def _figure(rows: int, height: float):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(6.4, height), dpi=200)
    FigureCanvasAgg(fig)
    return fig, fig.subplots(rows, 1, sharex=True, squeeze=False)[:, 0]


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def elapsed(reg: RegistrationData, idx: list) -> tuple[np.ndarray, bool]:
    """The x axis of a session: minutes since its first frame (never the clock itself: a camera's may be wrong), or the
    frame numbers when the times are not known. True: minutes."""
    t = reg.secs[idx]
    if len(idx) > 1 and bool(np.isfinite(t).all()):
        return (t - t.min()) / 60.0, True
    return np.arange(len(idx)) + 1.0, False


def size_of(d: ReportData) -> tuple[np.ndarray, float, str]:
    """The star size of each frame (the Moffat fit's when there is one), what turns it into the unit shown, the unit."""
    if d.fwhm is not None and d.reg.scale is not None:
        return d.fwhm, float(d.reg.scale), '"'
    return (d.fwhm if d.fwhm is not None else d.reg.fwhm), 1.0, " px"


def card_rows(d: ReportData, idx: list) -> list:
    """The figures of a session."""
    kept = d.kept[idx]
    exp = d.exposure[idx] if d.exposure is not None else np.full(len(idx), np.nan)
    rows = [(_("Frames kept"), f"{int(kept.sum())} / {len(idx)}")]
    if np.isfinite(exp[kept]).any():
        rows.append((_("Total exposure"), hm(float(np.nansum(exp[kept])))))
    size, k, unit = size_of(d)
    s = size[idx][kept]
    s = s[np.isfinite(s)]
    if len(s):
        for name, v in ((_("Median FWHM"), np.median(s)), (_("Best FWHM"), s.min()), (_("Worst FWHM"), s.max())):
            rows.append((name, f"{number(float(v * k), 1)}{unit}"))
    if d.roundness is not None and np.isfinite(d.roundness[idx][kept]).any():
        rows.append((_("Median roundness"), number(float(np.nanmedian(d.roundness[idx][kept])), 2)))
    return rows


def _night(d: ReportData, idx: list, heading: str = "", foot: str = "") -> bytes:
    """A session: star size, transparency and sky level on one axis; the frames left out as open circles."""
    import matplotlib
    r, kept = d.reg, d.kept[idx]
    x, minutes = elapsed(r, idx)
    size, k, unit = size_of(d)
    panels = [(_("FWHM") + f" ({unit.strip()})", size[idx] * k), (_("Transparency") + " (%)", r.transp[idx])]
    if d.sky is not None:
        panels.append((_("Sky level (ADU)"), d.sky[idx]))
    with matplotlib.rc_context(STYLE):
        fig, axes = _figure(len(panels), 1.25 * len(panels) + 0.9 + (0.45 if heading else 0))
        for ax, (name, y) in zip(axes, panels):
            ax.plot(x[kept], y[kept], "o", ms=2.2, color=INK, mew=0)
            if (~kept).any():
                ax.plot(x[~kept], y[~kept], "o", ms=3, mfc="none", mec=GREY, mew=0.6)
            ax.set_ylabel(name)
            ax.grid(color="0.9", lw=0.4)
            ax.set_axisbelow(True)
        axes[-1].set_xlabel(_("Minutes since the first frame") if minutes else _("Frame number"))
        if heading:
            fig.suptitle(heading, x=0.08, ha="left", fontsize=10)
        if foot:
            fig.text(0.08, 0.012, foot, fontsize=7.5, va="bottom")
        fig.tight_layout(rect=(0, 0.09 if foot else 0, 1, 0.97 if heading else 1))
        return _png(fig)


def _drift(reg: RegistrationData) -> bytes | None:
    if not np.isfinite(reg.dx).any():
        return None
    import matplotlib
    x = np.arange(reg.n) + 1.0
    k, unit = (float(reg.scale), '"') if reg.scale is not None else (1.0, "px")
    with matplotlib.rc_context(STYLE):
        fig, (a, b) = _figure(2, 3.2)
        a.plot(x, reg.dx * k, "-", color=INK, lw=0.8, label="x")
        a.plot(x, reg.dy * k, "--", color=GREY, lw=0.8, label="y")
        a.set_ylabel(_("Offset") + f" ({unit})")
        a.legend(loc="best")
        b.plot(x, reg.rms, "o", ms=2.2, color=INK, mew=0)
        b.set_ylabel(_("Residual (px)"))
        b.set_xlabel(_("Frame number"))
        for ax in (a, b):
            ax.grid(color="0.9", lw=0.4)
            ax.set_axisbelow(True)
        fig.tight_layout()
        return _png(fig)


def _response(r) -> bytes:
    import matplotlib
    with matplotlib.rc_context(STYLE):
        fig, (a,) = _figure(1, 2.5)
        for curve, col, name in zip(r.S, RGB, ("R", "G", "B")):
            a.plot(r.lam, np.maximum(curve, 0.0), "-", color=col, lw=0.9, label=name)
        for lam0, name in LINES:
            a.axvline(lam0, color=GREY, lw=0.4, ls=":")
            a.text(lam0, 1.07, name, fontsize=6, ha="center", va="bottom", color=GREY)
        a.set_ylim(0.0, 1.05)
        a.set_xlabel(_("Wavelength (nm)"))
        a.set_ylabel(_("Relative response"))
        a.legend(loc="center right")
        fig.tight_layout()
        return _png(fig)


def draw_figures(d: ReportData, title: str = ""):
    """The report's figures and the cards. Agg only: no widget, so that the worker thread can do it."""
    if d.reg is not None:
        for name, idx in d.sessions:
            rows = [f"{a}: {b}" for a, b in card_rows(d, idx)]
            foot = "     ".join(rows[:3]) + chr(10) + "     ".join(rows[3:])
            head = " · ".join(x for x in (_("Night card"), title, name) if x)
            d.nights.append(NightFigure(name, idx, _night(d, idx), _night(d, idx, head, foot)))
        d.drift = _drift(d.reg)
    if d.spectral is not None:
        d.response = _response(d.spectral)


# ------------------------------------------------------------ the shooting details
def _dec(x: float, digits: int, lang: str) -> str:
    t = f"{x:.{digits}f}"
    return t.replace(".", ",") if lang == "fr" else t


def _whole(v):
    """A whole number without its ".0"."""
    return int(v) if isinstance(v, float) and v.is_integer() else v


def shot_rows(d: ReportData, info: dict, entered: dict, lang: str) -> list:
    """The shooting details as (key, word, value), in the order of a forum post. ``entered``: what the user gave; a date
    only from there, a camera's clock may be wrong. An empty value is one nobody knows yet."""
    w = WORDS[lang]
    rows = [("camera", w["camera"], " · ".join(x for x in (str(info.get("camera") or ""), (
        f"ISO {_whole(info['iso_gain'])}" if info.get("iso_gain") else "")) if x))]
    focal = d.astro.get("focal_estimate_mm")
    rows.append(("instrument", w["instrument"], entered.get("instrument", "")))
    rows.append(("focal", "Focale" if lang == "fr" else "Focal length", f"{float(focal):.0f} mm" if focal else ""))
    rows += [(k, w[k], entered.get(k, "")) for k in ("mount", "filter", "location", "date")]
    if d.exposure is not None and d.kept is not None and np.isfinite(d.exposure[d.kept]).any():
        exp = d.exposure[d.kept]
        exp = exp[np.isfinite(exp)]
        groups = " + ".join(f"{int((np.round(exp) == t).sum())} × {t:g} s" for t in sorted(set(np.round(exp).tolist())))
        rows.append(("frames", w["frames"], f"{groups} ({hm(float(exp.sum()))})"))
    for k, n in (("darks", info.get("n_darks")), ("flats", info.get("n_flats"))):
        if n:
            rows.append((k, w[k], str(n)))
    fld = d.astro.get("field_deg")
    if fld:
        rows.append(("field", w["field"], f"{_dec(float(fld[0]), 1, lang)}° × {_dec(float(fld[1]), 1, lang)}°"))
    if d.reg is not None and d.reg.scale is not None:
        rows.append(("sampling", w["sampling"], f'{_dec(float(d.reg.scale), 1, lang)}"/px'))
    if d.reg is not None and d.kept is not None:
        size, k, unit = size_of(d)
        s = size[d.kept]
        s = s[np.isfinite(s)]
        if len(s):
            rows.append(("fwhm", w["fwhm"], f"{_dec(float(np.median(s) * k), 1, lang)}{unit}"))
        rows.append(("kept", w["kept"], f"{int(d.kept.sum())}/{len(d.kept)}"))
    rows.append(("processing", w["processing"], APP_NAME))
    return rows


def shot_text(target: str, rows: list) -> str:
    """The text to paste in a forum: what is known, nothing else."""
    lines = [target] if target else []
    return chr(10).join(lines + [f"{word} : {v}" if word != v else v for _k, word, v in rows if v])


# ------------------------------------------------------------ the document
def _table(rows, widths=("46%", "54%")) -> str:
    if not rows:
        return ""
    body = "".join(f"<tr><td width='{widths[0]}'>{a}</td><td width='{widths[1]}'>{b}</td></tr>" for a, b in rows)
    return f"<hr/><table width='100%' cellspacing='0' cellpadding='2'>{body}</table><hr/>"


class ReportPage(QWidget):
    """The tab: a white page, centred, and its two exports."""

    loaded = Signal()                                    # the data of the workflow shown have been read

    def __init__(self, session, steps, parent=None):
        super().__init__(parent)
        self.s, self.steps = session, steps
        self.service = steps.service
        self.data: ReportData | None = None
        self._token = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.setContentsMargins(10, 8, 10, 8)
        self.wf = QComboBox()
        self.wf.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.wf.activated.connect(lambda _i: self.load())
        bar.addWidget(self.wf)
        bar.addStretch(1)
        self.copied = QLabel("")
        bar.addWidget(self.copied)
        self.lang = QComboBox()
        self.lang.addItem("Français", "fr")
        self.lang.addItem("English", "en")
        self.lang.setCurrentIndex(max(0, self.lang.findData(language())))
        self.lang.setToolTip(_("The language of the text to paste, whatever the language of the screen"))
        bar.addWidget(self.lang)
        self.b_copy = QPushButton(_("Copy details"))
        self.b_copy.setToolTip(_("The shooting details as text, to paste in a forum"))
        self.b_copy.clicked.connect(self.copy_details)
        bar.addWidget(self.b_copy)
        self.b_png = QPushButton(_("Export PNG…"))
        self.b_pdf = QPushButton(_("Export PDF…"))
        self.b_png.clicked.connect(self._ask_png)
        self.b_pdf.clicked.connect(self._ask_pdf)
        bar.addWidget(self.b_png)
        bar.addWidget(self.b_pdf)
        v.addLayout(bar)
        row = QHBoxLayout()
        row.addStretch(1)
        self.view = QTextBrowser()
        self.view.setFixedWidth(PAGE_W + 24)
        self.view.setOpenLinks(False)
        self.view.setOpenExternalLinks(False)
        self.view.anchorClicked.connect(lambda url: self.enter(url.path()))
        self.view.setStyleSheet("QTextBrowser { background: #ffffff; color: #000000; border: none; }")
        row.addWidget(self.view)
        row.addStretch(1)
        v.addLayout(row, 1)
        self.service.jobReady.connect(self._ready)
        self.service.jobFailed.connect(self._failed)
        steps.changed.connect(self._facts_changed)

    # ------------------------------------------------------------ data
    def path(self):
        return next((q for q in self.s.paths if q.pid == self.wf.currentData()), None)

    def title(self) -> str:
        return self.s.target or self.s.name

    def refresh(self):
        """Shown: the workflows of the project, and what they have computed."""
        cur = self.wf.currentData()
        self.wf.clear()
        for q in self.s.paths:
            self.wf.addItem(q.name, q.pid)
        self.wf.setCurrentIndex(max(0, self.wf.findData(cur)))
        self.load()

    def load(self):
        p = self.path()
        if p is None:
            self.data = None
            self.build()
            return
        self._token = self.service.request_job("report", report_data, p.pid, tuple(self.s.all_stages(p)), self.title())

    def _ready(self, name, token, result):
        if name == "report" and token == self._token:
            self.data = result
            self.build()
            self.loaded.emit()

    def _failed(self, name, token, why):
        if name == "report" and token == self._token:
            self.data = ReportData()
            self.build()
            self.loaded.emit()

    def _facts_changed(self):
        if self.data is not None and self.isVisible():
            self.build()

    # ------------------------------------------------------------ the report
    def entered(self) -> dict:
        return self.s.p.acquisition

    def enter(self, key: str):
        """A click on a field the files do not give: what the user says, kept in the project."""
        name = dict(zip(ENTERED, ("Instrument", "Mount", "Filter", "Location", "Date")))[key]
        text, ok = QInputDialog.getText(self, _(name), _(name), text=self.entered().get(key, ""))
        if not ok:
            return
        if text.strip():
            self.entered()[key] = text.strip()
        else:
            self.entered().pop(key, None)
        if hasattr(self.s, "edited"):
            self.s.edited()
        at = self.view.verticalScrollBar().value()
        self.build()
        self.view.verticalScrollBar().setValue(at)

    def details(self) -> str:
        """The text the Copy button gives, in the language chosen."""
        if self.data is None or self.data.reg is None:
            return ""
        return shot_text(self.title(), shot_rows(self.data, self.s.info(), self.entered(), self.lang.currentData()))

    def copy_details(self):
        text = self.details()
        QApplication.clipboard().setText(text)
        self.copied.setText(_("Copied") if text else _("Nothing computed yet: run the workflow first."))

    def build(self):
        self.copied.setText("")
        self.view.setDocument(self.document())

    def document(self, exporting: bool = False) -> QTextDocument:
        self._exporting = exporting                      # a paper has no field to click
        doc = QTextDocument(self)
        font = QFont("Times New Roman", 11)
        font.setStyleHint(QFont.Serif)
        doc.setDefaultFont(font)
        doc.setDocumentMargin(36)
        doc.setDefaultStyleSheet("h1 { font-size: 18pt; font-weight: normal; } h2 { font-size: 12.5pt; } "
                                 "h3 { font-size: 11pt; font-style: italic; font-weight: normal; } "
                                 ".cap { font-size: 9.5pt; } .state { color: #8a2a2a; font-style: italic; }")
        self._n_fig = 0
        self._doc = doc
        doc.setHtml(self._html())
        return doc

    def _figure(self, png: bytes, caption: str) -> str:
        self._n_fig += 1
        url = f"fig{self._n_fig}"
        self._doc.addResource(QTextDocument.ImageResource, QUrl(url), QImage.fromData(png))
        return (f"<p align='center'><img src='{url}' width='{FIG_W}'/></p>"
                f"<p class='cap'><b>{_('Figure')} {self._n_fig}.</b> {html.escape(caption)}</p>")

    def _html(self) -> str:
        d, p = self.data, self.path()
        many = p is not None and len(self.s.paths) > 1   # a project of one workflow: its name says nothing
        head = (f"<h1 align='center'>{_('Processing report')}</h1>"
                f"<p align='center'>{html.escape(self.title())}"
                f"{' &mdash; ' + html.escape(p.name) if many else ''}<br/>"
                f"<i>{APP_NAME}, {date.today().isoformat()}</i></p>")
        if d is not None and d.thumb:
            self._doc.addResource(QTextDocument.ImageResource, QUrl("thumb"), QImage.fromData(d.thumb))
            head += "<p align='center'><img src='thumb' width='440'/></p>"
        if d is None or p is None or d.reg is None:
            return head + f"<p>{_('Nothing computed yet: run the workflow first.')}</p>"
        parts, n = [head, self._observation(d)], 1
        keys = self.s.block_keys(p)
        for key, title in BLOCKS:
            if key in keys:
                n += 1
                parts.append(self._block(n, p, key, title, d))
        parts.append(self._recipe(n + 1, keys, d))
        parts.append(self._appendix(n + 2, d))
        return "".join(parts)

    def _observation(self, d: ReportData) -> str:
        rows = []
        for key, word, value in shot_rows(d, self.s.info(), self.entered(), language() if language() in WORDS else "en"):
            if key in ENTERED and not self._exporting:   # a click to give it, or to change it
                link = html.escape(value) if value else _("click to enter")
                value = f"<a href='edit:{key}'>{link}</a>"
            elif value:
                value = html.escape(value)
            if value:
                rows.append((word, value))
        out = [f"<h2>1. {_('Observation')}</h2>", _table(rows)]
        for night in d.nights:
            if night.name:
                out.append(f"<h3>{html.escape(night.name)}</h3>")
            out.append(_table(card_rows(d, night.idx)))
            out.append(self._figure(night.figure, _("Star size (measured FWHM, Moffat fit), transparency relative to the "
                                                    "best frames and sky level of each frame. Open circles: frames left "
                                                    "out.")))
        return "".join(out)

    def _block(self, n: int, p, key: str, title: str, d: ReportData) -> str:
        state = block_state(self.s, p, key)
        out = [f"<h2>{n}. {_(title)}</h2>"]
        if state != "done":
            out.append(f"<p class='state'>{_(STATE_WORD.get(state, state))}</p>")
        out.append(_table([r for r in self.steps.facts(p, key) if r[0] not in [_(h) for h in HIDDEN]]))
        if key == "alignement" and d.drift is not None:
            out.append(self._figure(d.drift, _("Offset of each frame from the reference frame, and alignment residual.")))
        if key == "couleur" and d.response is not None:
            out.append(self._figure(d.response, _("Approximate spectral response of the sensor, rebuilt from {n} stars of "
                                                  "the field.").format(n=d.spectral.stars)))
        return "".join(out)

    @staticmethod
    def _value(value) -> str:
        text = _("Yes") if value is True else _("No") if value is False else \
            number(value, 3).rstrip("0").rstrip(",.") if isinstance(value, float) else str(value)
        return html.escape(text)

    def _recipe(self, n: int, keys, d: ReportData) -> str:
        """The chain, and what was chosen in it: the settings that differ from the defaults."""
        names = dict(BLOCKS)
        chain = " &rarr; ".join(html.escape(_(names[k])) for k in keys if k in names)
        rows = [(f"{html.escape(_(block))} · {html.escape(_(name))}", self._value(value))
                for block, name, value, changed in d.settings if changed]
        return (f"<h2>{n}. {_('Recipe')}</h2><p>{chain}</p>" +
                (_table(rows) if rows else f"<p>{_('All the settings are the defaults.')}</p>"))

    def _appendix(self, n: int, d: ReportData) -> str:
        out = [f"<h2>{n}. {_('Appendix: all the settings')}</h2>"]
        last, rows = None, []
        for block, name, value, _changed in d.settings:
            if block != last:
                if rows:
                    out.append(_table(rows))
                out.append(f"<h3>{html.escape(_(block))}</h3>")
                rows, last = [], block
            rows.append((html.escape(_(name)), self._value(value)))
        out.append(_table(rows))
        return "".join(out)

    # ------------------------------------------------------------ export
    def _ask_png(self):
        path, _f = QFileDialog.getSaveFileName(self, _("Export PNG…"), "report.png", "PNG (*.png)")
        if path:
            self.save_png(path)

    def _ask_pdf(self):
        path, _f = QFileDialog.getSaveFileName(self, _("Export PDF…"), "report.pdf", "PDF (*.pdf)")
        if path:
            self.save_pdf(path)

    def save_png(self, path):
        doc = self.document(exporting=True)
        doc.setTextWidth(PAGE_W)
        size = doc.size().toSize()
        img = QImage(size, QImage.Format_RGB32)
        img.fill(Qt.white)
        q = QPainter(img)
        doc.drawContents(q)
        q.end()
        img.save(str(path), "PNG")

    def save_pdf(self, path):
        w = QPdfWriter(str(path))
        w.setPageSize(QPageSize(QPageSize.A4))
        w.setResolution(96)
        w.setPageMargins(QMarginsF(0, 0, 0, 0))
        doc = self.document(exporting=True)
        doc.setPageSize(QSizeF(w.width(), w.height()))
        doc.print_(w)

    # ------------------------------------------------------------ the night card, for Post's Export
    def card_ready(self) -> bool:
        return self.data is not None and bool(self.data.nights)

    def save_cards(self, folder, name: str) -> list[Path]:
        """One PNG per session, next to the exported image."""
        out = []
        many = len(self.data.nights) > 1
        for i, night in enumerate(self.data.nights, 1):
            f = Path(folder) / (f"{name}_night_card_{i}.png" if many else f"{name}_night_card.png")
            f.write_bytes(night.card)
            out.append(f)
        return out
