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
from PySide6.QtCore import QBuffer, QEvent, QIODevice, QMarginsF, QSizeF, Qt, QUrl, Signal
from PySide6.QtGui import QFont, QImage, QPageSize, QPainter, QPdfWriter, QTextDocument
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import APP_NAME
from .blocks import STATE_WORD, block_state
from .charts import LINES, RegistrationData, registration_data
from .. import texts
from ..core.params import ADVANCED
from ..core.project2 import ACQUISITION
from .i18n import _, language, number
from .widgets import put_back, typed

#: the sections that are blocks of the chain, in its order: (block key, title)
BLOCKS = (("calibration", "Calibration"), ("alignement", "Registration"), ("selection", "Selection"),
          ("empilement", "Stacking"), ("astrometrie", "Astrometry"), ("fond", "Gradient"), ("debruitage", "Denoise"),
          ("couleur", "Color"), ("etoiles", "Stars/background"))
PAGE_W = 794                                             # the page on the screen, px (A4 at 96 dpi)
FOOT = 40                                                # the foot of a page, px
FIG_W = 640                                              # a figure in the page, px
#: what each block does, in one sentence, at the head of its section
INTRO = {"calibration": "Removes what comes from the sensor and the optics: its dark signal and the vignetting.",
         "alignement": "Brings every frame onto one reference frame, by matching their stars.",
         "selection": "Decides which frames are kept and how much each one counts.",
         "empilement": "Combines the frames kept into one image, more signal and less noise than any of them.",
         "astrometrie": "Finds where the image is on the sky, from its stars.",
         "fond": "Removes the smooth gradient of the sky left by light pollution, the Moon and the vignetting.",
         "debruitage": "Reduces the noise of the stacked image.",
         "couleur": "Sets the colours from stars whose real colours are known.",
         "etoiles": "Separates the stars from the sky, to process them apart."}
ZOOM_STEP = 1.15                                         # the reader's zoom, per step
#: what the files do not say: entered with a click, kept in the project (the Project block's form in Inspect edits the
#: same, issue #89)
ENTERED = ACQUISITION
#: the blocks' figures a reader of the report has no use for, and the star size on 2 x 2 super-pixels, which overestimates
#: undersampled stars: the report's sizes are the Moffat fit's
HIDDEN = ("Star FWHM range", "Cache size", "Flat minimum", "Integration", "Stacked exposure", "Drift")
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
    metrics: dict = field(default_factory=dict)          # {stage: its metrics} of the stages the head quotes
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
        if st is not None and st.name == "inventory":    # what was read, not a choice
            continue
        try:
            out = api.stage_outputs(project, wf, stage) if st is not None else None
        except (OSError, ValueError, KeyError):
            out = None
        if out is None:
            continue
        for key, value in out.params.items():
            info = st.Params.model_fields.get(key)
            if info is None or not shareable(value) or key == "workers":
                continue
            extra = info.json_schema_extra or {}
            changed = value != info.default
            if changed or extra.get("level") != ADVANCED:   # the rest changes nothing a reader needs
                rows.append((st.title, extra.get("label", key), value, changed))
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
    for stage in HEAD_STAGES:
        try:
            out = api.stage_outputs(project, wf, stage)
        except (OSError, ValueError, KeyError):
            out = None
        if out is not None:
            d.metrics[stage] = dict(out.metrics)
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
    if language() == "fr":                               # a decimal comma on the axes too (the night test pass)
        from matplotlib.ticker import FuncFormatter, ScalarFormatter
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                if type(axis.get_major_formatter()) is ScalarFormatter:
                    axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}".replace(".", ",").replace("-", "−")))
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


def exact_time(seconds: float) -> str:
    """A duration to the second: 7 min 30 s, never rounded to something another line contradicts."""
    s = int(round(seconds))
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    if h:
        return f"{h} h {m:02d}"
    return f"{m} min {s:02d} s" if s else f"{m} min"


def size_text(v: float, k: float, unit: str, digits_of) -> str:
    """A star size, in arcseconds with its pixels beside it: 10.7" alone looks like a terrible sky at 6.4"/px."""
    return f"{digits_of(v * k)}{unit} ({digits_of(v)} px)" if unit == '"' else f"{digits_of(v)}{unit}"


def card_rows(d: ReportData, idx: list) -> list:
    """The figures of a session."""
    kept = d.kept[idx]
    exp = d.exposure[idx] if d.exposure is not None else np.full(len(idx), np.nan)
    rows = [(_("Frames kept"), f"{int(kept.sum())} / {len(idx)}")]
    if np.isfinite(exp[kept]).any():
        rows.append((_("Total exposure"), exact_time(float(np.nansum(exp[kept])))))
    size, k, unit = size_of(d)
    s = size[idx][kept]
    s = s[np.isfinite(s)]
    if len(s):
        rows.append((_("Median FWHM"), size_text(float(np.median(s)), k, unit, lambda x: number(x, 1))))
        rows.append((_("Best FWHM"), f"{number(float(s.min() * k), 1)}{unit}"))
        rows.append((_("Worst FWHM"), f"{number(float(s.max() * k), 1)}{unit}"))
    if d.roundness is not None and np.isfinite(d.roundness[idx][kept]).any():
        rows.append((_("Median roundness"), number(float(np.nanmedian(d.roundness[idx][kept])), 2)))
    return rows


def _night(d: ReportData, idx: list, heading: str = "", foot: str = "") -> bytes:
    """A session: star size, transparency and sky level on one axis; the frames left out as open circles."""
    import matplotlib
    r, kept = d.reg, d.kept[idx]
    x, minutes = elapsed(r, idx)
    size, k, unit = size_of(d)
    panels = [(_("FWHM") + f" ({unit.strip()})", size[idx] * k), (_("Relative signal (%)"), r.transp[idx])]
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
        for i, (lam0, name) in enumerate(LINES):
            a.axvline(lam0, color=GREY, lw=0.4, ls=":")
            a.text(lam0, 1.07 + 0.08 * (i % 2), name, fontsize=6, ha="center", va="bottom", color=GREY)  # close lines: staggered
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
            colon = " : " if language() == "fr" else ": "      # French typography (the night test pass)
            rows = [f"{a}{colon}{b}" for a, b in card_rows(d, idx)]
            foot = "     ".join(rows[:3]) + chr(10) + "     ".join(rows[3:])
            shown = name if len(d.sessions) > 1 else ""  # a project of one session: its name says nothing
            head = " · ".join(x for x in (_("Night card"), title, shown) if x)
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
        rows.append(("frames", w["frames"], f"{groups} ({exact_time(float(exp.sum()))})"))
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
            rows.append(("fwhm", w["fwhm"], size_text(float(np.median(s)), k, unit, lambda x: _dec(x, 1, lang))))
        rows.append(("kept", w["kept"], f"{int(d.kept.sum())}/{len(d.kept)}"))
    rows.append(("processing", w["processing"], APP_NAME))
    return rows


#: the stages whose figures the head of the report quotes
HEAD_STAGES = ("inventory", "register", "stack", "catalogs", "gradient", "denoise", "color")


def sky_position(ra: float, dec: float) -> str:
    """RA and Dec as an observer reads them: 03h47m04s +24°03′36″."""
    h = (ra % 360) / 15
    t = int(round(h * 3600))
    hh, mm, ss = t // 3600 % 24, t // 60 % 60, t % 60
    sign, a = ("−" if dec < 0 else "+"), abs(dec)
    u = int(round(a * 3600))
    return f"{hh:02d}h{mm:02d}m{ss:02d}s {sign}{u // 3600:02d}°{u // 60 % 60:02d}′{u % 60:02d}″"


def scale_text(arcsec: float, lang: str) -> str:
    """A scale on the sky as the first page writes it, wherever it is (#137): 1.94″/px."""
    return f"{_dec(arcsec, 2, lang)}″/px"


def highlights(d: ReportData, lang: str) -> list:
    """The figures of the head (the night test pass: a picture to post that says what others do not): [(value, what,
    how it was measured)], each measured on this session's frames; none when it was not."""
    m, a = d.metrics, d.astro
    dec = lambda x, n=1: _dec(float(x), n, lang)          # noqa: E731
    scale = float(a.get("scale_arcsec") or (d.reg.scale if d.reg is not None and d.reg.scale else 0) or 0)
    out = []
    if a.get("focal_estimate_mm") and a.get("stars_used"):
        res = f"{dec(float(a['residual_px']) * scale, 2)}″" if a.get("residual_px") is not None and scale else "–"
        out.append((f"{float(a['focal_estimate_mm']):.0f} mm", _("Focal length, measured"),
                    _("from the solved sky: {n} stars, residual {res}").format(n=int(a["stars_used"]), res=res)))
    if d.reg is not None and d.kept is not None:
        size, k, unit = size_of(d)
        sz = size[d.kept]
        sz = sz[np.isfinite(sz)]
        if len(sz) and unit == '"':
            out.append((f"{dec(float(np.median(sz)) * k)}″", _("Star FWHM, measured"),
                        _("median star width (Moffat fit) of {n} frames, best {best}″").format(
                            n=len(sz), best=dec(float(sz.min()) * k))))
        elif len(sz):                                    # no scale on the sky (#135): measured all the same, in pixels
            how = _("median star width (Moffat fit) of {n} frames, best {best} px; in pixels: the scale on the sky is "
                    "not known") if d.fwhm is not None else \
                _("median star width of {n} frames, best {best} px; in pixels: the scale on the sky is not known")
            out.append((f"{dec(float(np.median(sz)))} px", _("Star FWHM, measured"),
                        how.format(n=len(sz), best=dec(float(sz.min())))))
    reg = m.get("register") or {}
    inv = m.get("inventory") or {}
    if reg.get("drift_px") and scale:
        drift = float(reg["drift_px"]) * scale
        how = _("how far the field moved on the sensor during the session (tracking and deliberate shifts)")
        try:
            from datetime import datetime
            mins = (datetime.fromisoformat(inv["last_time"]) - datetime.fromisoformat(inv["first_time"])).total_seconds() / 60
            if mins > 0:
                how = _("how far the field moved on the sensor in {t} (tracking and deliberate shifts): {rate}″ a minute").format(
                    t=exact_time(mins * 60), rate=dec(drift / mins))
        except (KeyError, TypeError, ValueError):
            pass
        out.append((f"{drift:.0f}″", _("Field drift"), how))
    ch = (m.get("stack") or {}).get("channel_offsets") or {}
    if ch.get("applied") and ch.get("centre_px"):
        r, b = (float(np.hypot(*ch["centre_px"][c])) for c in ("R", "B"))
        out.append((f"{dec(max(r, b), 2)} px", _("Colours offset, corrected"),
                    _("red {r} px and blue {b} px from green, measured on {n} stars").format(
                        r=dec(r, 2), b=dec(b, 2), n=int(ch.get("stars") or 0))))
    dn = m.get("denoise") or {}
    if dn.get("noise_reduction"):
        out.append((f"÷ {dec(dn['noise_reduction'])}", _("Sky noise"),
                    _("measured on the sky before and after Denoise")))
    gr = m.get("gradient") or {}
    if gr.get("gradient_pct"):
        out.append((f"{dec(gr['gradient_pct'])} %", _("Sky gradient, removed"),
                    _("spread of the sky across the field, against its level, modelled on {n} samples").format(
                        n=int(gr.get("samples_used") or 0))))
    cat = m.get("catalogs") or {}
    if cat.get("stars_deep"):
        out.append((f"{int(cat['stars_deep']):,}".replace(",", " "), _("Gaia DR3 stars in the field"),
                    _("{n} of them with their measured spectrum").format(n=int(cat.get("xp_spectra") or 0))))
    if d.spectral is not None and getattr(d.spectral, "stars", 0):
        out.append((str(int(d.spectral.stars)), _("Sensor response, rebuilt"),
                    _("stars whose spectra rebuilt the sensor's colour response")))
    if reg.get("transparency_p10_p90"):
        lo, hi = (float(x) for x in reg["transparency_p10_p90"])
        if hi > 0:
            out.append((f"{100 * (hi - lo) / hi:.0f} %", _("Transparency changes"),
                        _("signal of the stars from frame to frame (tenth to ninetieth percentile)")))
    return out


def shot_text(target: str, rows: list) -> str:
    """The text to paste in a forum: what is known, nothing else."""
    lines = [target] if target else []
    colon = " : " if rows and rows[0][1] == WORDS["fr"]["camera"] else ": "
    return chr(10).join(lines + [f"{word}{colon}{v}" for _k, word, v in rows if v])


# ------------------------------------------------------------ the document
def _table(rows, widths=("38%", "62%")) -> str:
    if not rows:
        return ""
    body = "".join(f"<tr><td width='{widths[0]}'>{a}</td><td width='{widths[1]}'><b>{b}</b></td></tr>" for a, b in rows)
    return f"<hr/><table width='100%' cellspacing='0' cellpadding='3'>{body}</table><hr/>"


def _ruled(head, rows, widths) -> str:
    """A table as journals set them: three rules (above, under the head, below), no other line, nothing bold, small.
    The head and the body are two tables of the same columns between rules (Qt draws a cell's border on all sides)."""
    if not rows:
        return ""
    open_ = "<table width='100%' cellspacing='0' cellpadding='2' style='font-size:9.5pt;'>"
    top = "".join(f"<td width='{w}'><i>{h}</i></td>" for w, h in zip(widths, head))
    body = "".join("<tr>" + "".join(f"<td width='{w}'>{c}</td>" for w, c in zip(widths, r)) + "</tr>" for r in rows)
    return f"<hr/>{open_}<tr>{top}</tr></table><hr/>{open_}{body}</table><hr/>"


def _meanings() -> dict:
    """What each figure of the blocks means, by the name it is shown under."""
    lang = language()
    return {html.escape(texts.metric_label(k, lang)): texts.translate(m.meaning, lang) for k, m in texts.METRICS.items()}


def _explained(rows) -> str:
    """A block's figures, each with what it means: a reader should not need the manual."""
    if not rows:
        return ""
    known = _meanings()
    body = "".join(f"<tr><td width='30%'>{a}</td><td width='20%'><b>{b}</b></td>"
                   f"<td width='50%' class='why'>{html.escape(known.get(a, '')) or '&nbsp;'}</td></tr>" for a, b in rows)
    return f"<hr/><table width='100%' cellspacing='0' cellpadding='3'>{body}</table><hr/>"


class ReportPage(QWidget):
    """The tab: on the left what is entered and what can be pasted, on the right the report itself, in a PDF reader: what
    is seen is what is exported, and it zooms."""

    loaded = Signal()                                    # the data of the workflow shown have been read
    said = Signal(str)                                   # a word for the status line (an export written)

    def __init__(self, session, steps, parent=None):
        super().__init__(parent)
        self.s, self.steps = session, steps
        self.service = steps.service
        self.data: ReportData | None = None
        self._token = None
        self._buffer = None
        self._given: dict = {}                           # {field: the text it was last given} (``put_back``, #131)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._side())
        root.addLayout(self._reader(), 1)
        self.service.jobReady.connect(self._ready)
        self.service.jobFailed.connect(self._failed)
        steps.changed.connect(self._facts_changed)
        if hasattr(session, "pathsChanged"):             # Undo of a shooting detail while the Report is shown (#89)
            session.pathsChanged.connect(self._entered_changed)

    def _side(self) -> QWidget:
        side = QFrame()
        side.setFixedWidth(330)
        side.setObjectName("panel")
        v = QVBoxLayout(side)
        v.setContentsMargins(12, 10, 12, 12)
        self.wf = QComboBox()
        self.wf.activated.connect(lambda _i: self.load())
        v.addWidget(self.wf)
        title = QLabel(_("Shooting details"))
        title.setObjectName("h2")
        v.addWidget(title)
        hint = QLabel(_("What the files do not say. Kept in the project."))
        hint.setObjectName("small")
        hint.setWordWrap(True)
        v.addWidget(hint)
        form = QFormLayout()
        self.fields: dict[str, QLineEdit] = {}
        for key, name in zip(ENTERED, ("Instrument", "Mount", "Filter", "Location", "Date")):
            e = QLineEdit()
            e.editingFinished.connect(lambda k=key: self.set_field(k))
            form.addRow(_(name), e)
            self.fields[key] = e
        v.addLayout(form)
        row = QHBoxLayout()
        self.lang = QComboBox()
        self.lang.addItem("Français", "fr")
        self.lang.addItem("English", "en")
        self.lang.setCurrentIndex(max(0, self.lang.findData(language())))
        self.lang.setToolTip(_("The language of the text to paste, whatever the language of the screen"))
        self.lang.currentIndexChanged.connect(lambda _i: self._show_text())
        row.addWidget(self.lang)
        self.b_copy = QPushButton(_("Copy text"))
        self.b_copy.setToolTip(_("The shooting details as text, to paste in a forum"))
        self.b_copy.clicked.connect(self.copy_details)
        row.addWidget(self.b_copy, 1)
        v.addLayout(row)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("QPlainTextEdit { background: #14161b; color: #e8ebf0; border: 1px solid #2a2f38; }")
        v.addWidget(self.text, 1)
        self.part = QComboBox()
        self.part.setToolTip(_("A part of the report, as a picture to paste in a forum"))
        v.addWidget(self.part)
        self.b_card = QPushButton(_("Copy image"))
        self.b_card.setToolTip(_("The part chosen above, as a picture"))
        self.b_card.clicked.connect(self.copy_image)
        v.addWidget(self.b_card)
        self.copied = QLabel("")
        self.copied.setObjectName("small")
        v.addWidget(self.copied)
        return side

    def _reader(self) -> QVBoxLayout:
        v = QVBoxLayout()
        v.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.setContentsMargins(10, 6, 10, 6)
        self.b_out, self.b_in, self.b_fit = QPushButton("−"), QPushButton("+"), QPushButton(_("Fit width"))
        self.b_out.clicked.connect(lambda: self.zoom_by(1 / ZOOM_STEP))
        self.b_in.clicked.connect(lambda: self.zoom_by(ZOOM_STEP))
        self.b_fit.clicked.connect(self.fit_width)
        for b in (self.b_out, self.b_in, self.b_fit):
            bar.addWidget(b)
        self.zoom_lab = QLabel("")
        bar.addWidget(self.zoom_lab)
        bar.addStretch(1)
        self.b_png = QPushButton(_("Export PNG…"))
        self.b_pdf = QPushButton(_("Export PDF…"))
        self.b_png.clicked.connect(self._ask_png)
        self.b_pdf.clicked.connect(self._ask_pdf)
        bar.addWidget(self.b_png)
        bar.addWidget(self.b_pdf)
        v.addLayout(bar)
        self.pdf = QPdfDocument(self)
        self.view = QPdfView(self)
        self.view.setDocument(self.pdf)
        self.view.setPageMode(QPdfView.PageMode.MultiPage)
        self.view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        self.view.viewport().installEventFilter(self)
        v.addWidget(self.view, 1)
        return v

    # ------------------------------------------------------------ the reader's zoom
    def zoom_by(self, k: float):
        self.view.setZoomFactor(self._zoom() * k)
        self.view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.zoom_lab.setText(f"{self.view.zoomFactor() * 100:.0f} %")

    def _zoom(self) -> float:
        """The scale shown; fitted to the width, Qt keeps no factor: it is worked out from the page."""
        if self.view.zoomMode() == QPdfView.ZoomMode.Custom:
            return self.view.zoomFactor()
        points = self.pdf.pagePointSize(0).width() if self.pdf.pageCount() else 0
        return (self.view.viewport().width() - 20) / (points * 96 / 72) if points else 1.0

    def fit_width(self):
        self.view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        self.zoom_lab.setText("")

    def eventFilter(self, obj, e):                                   # noqa: N802 - Qt's name
        if obj is self.view.viewport() and e.type() == QEvent.Wheel and e.modifiers() & Qt.ControlModifier:
            self.zoom_by(ZOOM_STEP if e.angleDelta().y() > 0 else 1 / ZOOM_STEP)
            return True
        return super().eventFilter(obj, e)

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
        self.wf.setVisible(self.wf.count() > 1)          # a project of one workflow: nothing to choose
        self._fill_entered()                             # at once: typed in the Project block's form meanwhile (#89)
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

    # ------------------------------------------------------------ what the user gives, what can be pasted
    def entered(self) -> dict:
        return self.s.p.acquisition

    def set_field(self, key: str):
        """A field left: what it holds is kept in the project (an empty one is forgotten), by the model as the Project
        block's form keeps it (issue #89): Undo takes it back, Save writes it. Only a value the user changed: the one it
        was given, the form having changed it since, is not written back (#131)."""
        text = self.fields[key].text().strip()
        if not typed(self.fields[key], self._given) or text == self.entered().get(key, ""):
            return
        self.s.set_shooting_detail(key, text)
        self.build()

    def details(self) -> str:
        """The text to paste, in the language chosen."""
        if self.data is None or self.data.reg is None:
            return ""
        return shot_text(self.title(), shot_rows(self.data, self.s.info(), self.entered(), self.lang.currentData()))

    def _show_text(self):
        self.text.setPlainText(self.details())

    def copy_details(self):
        text = self.details()
        QApplication.clipboard().setText(text)
        self.copied.setText(_("Copied") if text else _("Nothing computed yet: run the workflow first."))

    def copy_image(self):
        """The part chosen, as a picture: the night card, or a section of the report as it is printed."""
        i = self.part.currentIndex()
        if i < 0:
            self.copied.setText(_("Nothing computed yet: run the workflow first."))
            return
        QApplication.clipboard().setImage(self._part_image(i))
        self.copied.setText(_("Copied"))

    def _part_image(self, i: int) -> QImage:
        night, section = self._parts[i]
        return QImage.fromData(night.card) if night is not None else self._picture([section], 2.0)

    # ------------------------------------------------------------ the report
    def _fill_entered(self):
        for key, e in self.fields.items():               # the one being typed in keeps what is typed (#131)
            put_back(e, self.entered().get(key, ""), self._given)

    def _entered_changed(self):
        """The project changed (an Undo, the Project block's form): the fields and the report follow what is kept."""
        if self.isVisible() and any(e.text().strip() != self.entered().get(k, "") for k, e in self.fields.items()):
            self.build()

    def build(self):
        self.copied.setText("")
        self._fill_entered()
        self._show_text()
        self._sections_now = self.sections()
        self._fill_parts()
        buf = QBuffer(self)
        buf.open(QIODevice.ReadWrite)
        self._write_pdf(buf)
        buf.seek(0)
        self.pdf.load(buf)
        if self._buffer is not None:
            self._buffer.deleteLater()
        self._buffer = buf

    def _fill_parts(self):
        """What Copy image offers, [(night, section)] one of them None, in the order the report is read (#140): its first
        page (the session card), the night cards, then the sections; the pieces without a title (the appendix's) are
        not offered."""
        was = self.part.currentText()
        self.part.clear()
        nights = self.data.nights if self.data is not None else []
        titled = [s for s in self._sections_now if s[0]]
        self._parts = [(None, s) for s in titled[:1]] + [(n, None) for n in nights] + [(None, s) for s in titled[1:]]
        for night, section in self._parts:
            self.part.addItem(section[0] if night is None else
                              _("Night card") + (f" · {night.name}" if len(nights) > 1 else ""))
        self.part.setCurrentIndex(max(0, self.part.findText(was)))
        self.b_card.setEnabled(self.part.count() > 0)

    def _new_doc(self, body: str) -> QTextDocument:
        # no parent: gone with its last reference (the night test pass: owned by the page, each showing of the report
        # kept its documents and their figures, about 15 MB, for good)
        doc = QTextDocument()
        font = QFont("Times New Roman", 12)
        font.setStyleHint(QFont.Serif)
        doc.setDefaultFont(font)
        doc.setDocumentMargin(36)
        doc.setDefaultStyleSheet("h1 { font-size: 20pt; font-weight: normal; } h2 { font-size: 14pt; } "
                                 "h3 { font-size: 12pt; font-style: italic; font-weight: normal; } "
                                 ".cap { font-size: 10.5pt; } .state { color: #8a2a2a; font-style: italic; } "
                                 ".why { color: #444444; font-size: 10.5pt; } .intro { font-style: italic; }")
        for url, img in self._images.items():
            doc.addResource(QTextDocument.ImageResource, QUrl(url), img)
        doc.setHtml(body)
        doc.setTextWidth(PAGE_W)
        return doc

    def document(self) -> QTextDocument:
        """The whole report as one document (its text; the pages are made by ``_write_pdf``)."""
        return self._new_doc("".join(body for _title, body in self.sections()))

    def _figure(self, png: bytes, caption: str) -> str:
        self._n_fig += 1
        url = f"fig{self._n_fig}"
        self._images[url] = QImage.fromData(png)
        return (f"<p align='center'><img src='{url}' width='{FIG_W}'/></p>"
                f"<p class='cap'><b>{_('Figure')} {self._n_fig}.</b> {html.escape(caption)}</p>")

    def sections(self) -> list:
        """[(title, html)]: the pieces the pages are made of, none of them cut, each one a picture to paste. An empty
        title: a piece that is not offered on its own."""
        self._n_fig = 0
        self._images: dict[str, QImage] = {}
        d, p = self.data, self.path()
        many = p is not None and len(self.s.paths) > 1   # a project of one workflow: its name says nothing
        version = self.s.post_version_label() if hasattr(self.s, "post_version_label") else ""
        head = self._head(d, p, many, version)
        if d is None or p is None or d.reg is None:
            return [("", head + f"<p>{_('Nothing computed yet: run the workflow first.')}</p>")]
        out = [(_("Session card"), head)] + self._observation(d)
        n, keys = 0, self.s.block_keys(p)                 # the observation is the head's Table 1: blocks from 1
        for key, title in BLOCKS:
            if key in keys:
                n += 1
                out.append((f"{n}. {_(title)}", self._block(n, p, key, title, d)))
        out.append((f"{n + 1}. {_('Recipe')}", self._recipe(n + 1, keys, d)))
        return out + self._appendix(n + 2, d)

    def _head(self, d, p, many: bool, version: str) -> str:
        """The first piece, set as a journal article opens: the target and where it is, the final image (Figure 1),
        then the program named in one small line, and the tables of the observation and of the figures measured on
        the session with how each was measured (``highlights``): that part can be posted as a picture on its own."""
        from PySide6.QtCore import QRectF
        from PySide6.QtSvg import QSvgRenderer

        from .icons import APP_ICON
        logo = QImage(48, 48, QImage.Format_ARGB32_Premultiplied)
        logo.fill(Qt.transparent)
        q = QPainter(logo)
        q.setRenderHint(QPainter.Antialiasing)
        QSvgRenderer(str(APP_ICON)).render(q, QRectF(0, 0, 48, 48))
        q.end()
        self._images["logo"] = logo
        lang = language() if language() in WORDS else "en"
        small = "font-size:9pt;"
        sub = [html.escape(x) for x in (p.name if many and p is not None else "", version) if x]
        a = d.astro if d is not None else {}
        where = []
        if a.get("ra") is not None and a.get("dec") is not None:
            where.append(sky_position(float(a["ra"]), float(a["dec"])) + " (ICRS)")
        if a.get("field_deg"):
            where.append(_("field {w}° × {h}°").format(w=_dec(float(a["field_deg"][0]), 1, lang),
                                                       h=_dec(float(a["field_deg"][1]), 1, lang)))
        if a.get("scale_arcsec"):
            where.append(scale_text(float(a["scale_arcsec"]), lang))
        out = (f"<p align='center' style='font-size:14pt;'>{html.escape(self.title())}</p>"
               + (f"<p align='center' style='{small}'>{' &mdash; '.join(sub)}</p>" if sub else "")
               + (f"<p align='center' style='{small}'>{html.escape(' · '.join(where))}</p>" if where else ""))
        if d is not None and d.thumb:
            self._images["thumb"] = QImage.fromData(d.thumb)
            out += ("<p align='center'><img src='thumb' width='560'/></p>"
                    f"<p align='center' style='{small}'>{_('Figure 1. The final image, as Post last showed it.')}</p>")
        # the program, in one small line just above the tables (a picture of them takes it along)
        out += (f"<p style='font-size:8pt; color:#555555;'><img src='logo' width='12' height='12' "
                f"style='vertical-align: middle;'/>&nbsp;{APP_NAME} &middot; {_('Processing report')} &middot; "
                f"{date.today().isoformat()}</p>")
        if d is None or d.reg is None:
            return out
        measured = highlights(d, lang)
        # the focal length and the star size are in Table 2, with how they were measured: not twice; the field is in the
        # line under the title, the object's place on the sky, and so is the scale (#137): the sensor's sampling is in
        # Table 1 only when it is another figure (a drizzled image's pixels are smaller) or the only one (no plate
        # solving: the focal length and the pixel size give it), written the same way
        twice = ({"focal", "fwhm"} if measured else set()) | {"field"}
        written = {"sampling": scale_text(float(d.reg.scale), lang)} if d.reg.scale else {}
        if written.get("sampling") in where:
            twice.add("sampling")
        rows = [(html.escape(word), html.escape(written.get(key, value)))
                for key, word, value in shot_rows(d, self.s.info(), self.entered(), lang)
                if value and key != "processing" and key not in twice]
        out += (f"<p style='{small}'>{_('Table 1. Acquisition.')}</p>"
                + _ruled((_("Quantity"), _("Value")), rows, ("40%", "60%")))
        if measured:
            out += (f"<p style='{small}'>{_('Table 2. Figures measured on the frames of this session.')}</p>"
                    + _ruled((_("Quantity"), _("Value"), _("How it is measured")),
                             [(html.escape(t), html.escape(v), html.escape(h)) for v, t, h in measured],
                             ("27%", "15%", "58%")))
        return out

    def _observation(self, d: ReportData) -> list:
        out = []                                         # (its table opens the report, in the session card)
        for night in d.nights:
            body = f"<h3>{html.escape(night.name)}</h3>" if len(d.nights) > 1 else ""
            if len(d.nights) > 1 or d.roundness is not None:
                body += _table(card_rows(d, night.idx))
            body += self._figure(night.figure, _("Star size (measured FWHM, Moffat fit), signal of each frame relative to "
                                                 "the best ones (100 = the best tenth; it can go a little above) and sky "
                                                 "level after calibration. Open circles: frames left out."))
            out.append((_("The night, frame by frame") + (f" · {night.name}" if len(d.nights) > 1 else ""), body))
        return out

    def _block(self, n: int, p, key: str, title: str, d: ReportData) -> str:
        state = block_state(self.s, p, key)
        out = [f"<h2>{n}. {_(title)}</h2>", f"<p class='intro'>{_(INTRO[key])}</p>"]
        if state != "done":
            out.append(f"<p class='state'>{_(STATE_WORD.get(state, state))}</p>")
        hidden = [html.escape(_(h)) for h in HIDDEN]
        out.append(_explained([r for r in self.steps.facts(p, key) if r[0] not in hidden]))
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

    def _appendix(self, n: int, d: ReportData) -> list:
        """The settings of each block, a piece each. Of the two stacking methods, only the one that ran."""
        classic = any(str(v) == "classic" for _b, name, v, _c in d.settings if name == "Stacking method")
        other = "(drizzle)" if classic else "(Classic)"
        by: dict = {}
        for block, name, value, _changed in d.settings:
            if other not in name:
                by.setdefault(block, []).append((html.escape(_(name)), self._value(value)))
        out = []
        for i, (block, rows) in enumerate(by.items()):
            head = f"<h2>{n}. {_('Appendix: settings')}</h2>" if i == 0 else ""
            out.append(("", f"{head}<h3>{html.escape(_(block))}</h3>{_table(rows)}"))
        return out

    # ------------------------------------------------------------ export
    def _ask_png(self):
        self._export(_("Export PNG…"), "png", "PNG (*.png)", self.save_png)

    def _ask_pdf(self):
        self._export(_("Export PDF…"), "pdf", "PDF (*.pdf)", self.save_pdf)

    def _export(self, title: str, ext: str, kinds: str, write):
        """The file asked (in the project's folder at first, named after it), written with the wait shown, then said
        (the night test pass: 7 s without a sign, the file proposed in the program's own folder)."""
        from . import prefs
        from .imageview import slug
        root = getattr(self.s, "root", None)
        folder = Path(root) if root else Path(prefs.last_folder())
        path, _f = QFileDialog.getSaveFileName(self, title, (folder / f"{slug(self.title())}_report.{ext}").as_posix(),
                                               kinds)
        if not path:
            return
        prefs.remember_folder(path)
        self.said.emit(_("Writing {name}…").format(name=Path(path).name))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            write(path)
        finally:
            QApplication.restoreOverrideCursor()
        self.said.emit(_("Exported to {path}").format(path=path))

    def _picture(self, parts: list, scale: float = 1.0) -> QImage:
        docs = [self._new_doc(body) for _title, body in parts]
        h = sum(d.size().height() for d in docs)
        img = QImage(int(PAGE_W * scale), int(h * scale), QImage.Format_RGB32)
        img.fill(Qt.white)
        q = QPainter(img)
        q.scale(scale, scale)
        for d in docs:
            d.drawContents(q)
            q.translate(0, d.size().height())
        q.end()
        return img

    def save_png(self, path):
        self._picture(self.sections()).save(str(path), "PNG")

    def _write_pdf(self, device):
        """The pages: a piece that does not fit goes whole to the next one; a foot on each."""
        w = QPdfWriter(device)
        w.setPageSize(QPageSize(QPageSize.A4))
        w.setResolution(96)
        w.setPageMargins(QMarginsF(0, 0, 0, 0))
        H = w.height() - FOOT
        q = QPainter(w)
        page, y = 1, 0.0

        def foot():
            q.setPen(Qt.gray)
            q.drawText(36, w.height() - 22, f"{self.title()} · {APP_NAME} · {page}")
        for _title, body in self.sections():
            doc = self._new_doc(body)
            doc.setTextWidth(w.width())
            h = doc.size().height()
            if y and y + h > H:
                foot()
                w.newPage()
                page, y = page + 1, 0.0
            q.save()
            q.translate(0, y)
            doc.drawContents(q)
            q.restore()
            y += h
        foot()
        q.end()

    def save_pdf(self, path):
        self._write_pdf(str(path))

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
