"""The charts of the client, interactive (zoom, pan, values under the cursor, CSV and PNG export; see ``chartview``).

* ``FramesChart``: Selection. Star size and transparency of every frame, the thresholds dragged, the frames clicked
  (``api.frame_table``). Shift + drag draws a rectangle of frames to exclude or keep.
* ``SpectralChart``, ``ColorLinesChart``: Color. An approximation of the sensor's curves rebuilt from the field's
  stars (``api.spectral_response``), and the colour the camera gives to each wavelength.
* Registration (``registration_data`` reads the stage's results, ``registration_tabs`` builds the charts):
  ``QualityChart`` (star size, roundness, stars, transparency of each frame), ``DriftChart`` (the offset of each
  frame from the reference, in pixels or arcseconds), ``TrackChart`` (x against y, coloured by time),
  ``DistributionChart`` (histograms of star size and transparency).

Reading the project is ``registration_data``'s only job and it belongs in a worker thread (as the other loads of the
Inspect view); everything else only draws.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QMenu

from . import prefs
from .chartview import (
    AMBER,
    BLUE,
    GREEN,
    RED,
    VIOLET,
    ChartView,
    clock,
    num,
    to_seconds,
)
from .i18n import _, number

KEEP, OUT, HAND = BLUE, RED, AMBER
UNALIGNED = QColor("#8b93a1")                  # a frame that did not align: no transparency, shown at the bottom


# ============================================================================================ Selection
class FramesChart(ChartView):
    """Star size and transparency of every frame, Siril / PixInsight style: thresholds dragged, points clicked.

    ``rows``: the frame table's rows (name, time, fwhm, transparency, status, excluded). Thresholds: ``tmin`` in % of the
    best frames, ``fmax`` in px. ``hand``: the indices excluded by hand. The mouse: the wheel zooms the frames axis, a
    drag pans, a click picks the frame under the cursor (``picked``), the grips on the left move the thresholds
    (``changed`` when dropped), Shift + drag draws a rectangle of frames to exclude or keep by hand.

    ``thresholds``: whether the thresholds act (the stack's "thresholds" selection rule). Without, there are no
    grips and no threshold lines, and a frame is left out for the reason the table gives (the rule's), or by hand."""
    changed = Signal()
    picked = Signal(int)

    def __init__(self, rows, tmin: float, fmax: float, parent=None, *, thresholds: bool = True):
        super().__init__("", parent, zoom="x", readout="x", pickable=True, export_name="frames")
        self.setMinimumHeight(300)
        self.rows = list(rows)
        self.names = [r.name for r in self.rows]
        self.times = [r.time or "" for r in self.rows]
        self.fwhm = np.array([r.fwhm for r in self.rows], float)
        tr = np.array([r.transparency for r in self.rows], float)
        self.aligned = np.array([r.status != "not aligned" for r in self.rows], bool)
        ok = tr[self.aligned & np.isfinite(tr)]
        self.p90 = float(np.percentile(ok, 90)) if len(ok) else 1.0     # the engine's reference: the best frames
        self.tr = 100 * tr / self.p90                                    # % of the best frames
        self.n = len(self.rows)
        self.tmin, self.fmax = float(tmin), float(fmax)                  # thresholds (%, px)
        self.thresholds = bool(thresholds)
        # without thresholds: the frames the rule leaves out (not the ones excluded by hand, which ``hand`` holds)
        self.ruled = np.array([bool(r.status) and not r.excluded for r in self.rows], bool)
        self.sugg = (self.tmin, self.fmax)
        self.hand = {i for i, r in enumerate(self.rows) if r.excluded}
        self.status = [r.status for r in self.rows]
        self._current = None
        secs = [to_seconds(t) for t in self.times]
        self.secs = np.array([np.nan if s is None else s for s in secs], float)
        self.has_time = self.n > 1 and bool(np.isfinite(self.secs).all())
        self.xmode = "frame"
        if not self.n:
            return
        self.p_f = self.add_panel(_("Star size (FWHM, px)"), "px", digits=1)
        self.p_t = self.add_panel(_("Transparency (% of the best frames)"), "%", digits=0)
        x = self._xvals()
        self.set_x(_("Frame number"), kind="int")
        self.x_short = _("Frame")
        self._series = {}
        for key, panel, vals in (("f", self.p_f, self.fwhm), ("t", self.p_t, self.tr)):
            for state, name, color in (("keep", _("kept"), KEEP), ("out", _("excluded"), OUT),
                                       ("hand", _("excluded by hand"), HAND), ("na", _("not aligned"), UNALIGNED)):
                self._series[key, state] = self.add_scatter(name, x, vals, panel=panel, color=color, size=6.5,
                                                            readout=False, export=False)
        self.d_f = self.add_data("FWHM", x, self.fwhm, panel=self.p_f, unit="px", digits=2)
        # issue #37: each frame says why it is left out (the readout's Reason), not only by its colour
        reasons = [_(r.status) if r.status else (_("excluded by hand") if r.excluded else _("kept")) for r in self.rows]
        self.d_t = self.add_data(_("Transparency"), x, self.tr, panel=self.p_t, unit="%", digits=0,
                                 extra={_("Frame"): self.names, _("Time"): self.times, _("Reason"): reasons})
        f = self.fwhm[np.isfinite(self.fwhm)]
        t = self.tr[np.isfinite(self.tr) & self.aligned]
        f0, f1 = (min(f.min(), self.fmax), max(f.max(), self.fmax)) if f.size else (0.0, 1.0)
        t0, t1 = (min(t.min(), self.tmin), max(t.max(), self.tmin, 100)) if t.size else (0.0, 100.0)
        if not self.thresholds:
            f0, f1 = (f.min(), f.max()) if f.size else (0.0, 1.0)
            t0, t1 = (t.min(), max(t.max(), 100)) if t.size else (0.0, 100.0)
        self.p_f.home_y = (f0 * 0.95, f1 * 1.05)
        self.p_t.home_y = (t0 * 0.95, t1 * 1.03)
        self.h_f = self.h_t = None
        if self.thresholds:
            self.h_f = self.add_hline("fmax", self.fmax, panel=self.p_f, color=OUT, draggable=True,
                                      label=lambda v: _("threshold {v} px  (drag)").format(v=number(v, 1)))
            self.h_t = self.add_hline("tmin", self.tmin, panel=self.p_t, color=OUT, draggable=True,
                                      label=lambda v: _("threshold {v} %  (drag)").format(v=f"{v:.0f}"))
        if self.has_time:
            self.add_choice([("frame", _("Frame number")), ("time", _("Time"))], "frame", self.set_xmode)
        self.enable_lasso(True)
        self.lineMoved.connect(self._line_moved)
        self.lineReleased.connect(lambda *_a: self.changed.emit())
        self.pointClicked.connect(self._clicked)
        self.rubberBand.connect(self._band)
        if self.thresholds:
            self.add_action(_("Restore the suggested thresholds"), self.reset_thresholds)
        self._classify()
        self.fit()

    # ------------------------------------------------------------ the frames
    def excluded(self) -> np.ndarray:
        """Left out by the thresholds, by hand, or because the frame did not align."""
        ex = ((self.fwhm > self.fmax) | (self.tr < self.tmin) if self.thresholds else self.ruled) | ~self.aligned
        for i in self.hand:
            ex[i] = True
        return ex

    def kept(self) -> int:
        return int(self.n - self.excluded().sum())

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, i):
        self._current = i
        self.update()

    def _xvals(self) -> np.ndarray:
        return self.secs if self.xmode == "time" else np.arange(self.n) + 1.0

    def _classify(self):
        """Colour the points by what the thresholds and the hand say, in both panels."""
        ex = self.excluded()
        hand = np.zeros(self.n, bool)
        for i in self.hand:
            hand[i] = True
        x = self._xvals()
        for key, vals in (("f", self.fwhm), ("t", self.tr)):
            na = ~self.aligned
            for state, m in (("keep", ~ex), ("out", ex & ~hand & ~na), ("hand", hand & ~na), ("na", na)):
                y = vals[m]
                if state == "na" and key == "t":            # no transparency without alignment: on the bottom line
                    y = np.full(int(m.sum()), self.p_t.home_y[0] if self.p_t.home_y else 0.0)
                self._series[key, state].set_data(x[m], y)
        self.update()

    def set_xmode(self, mode: str):
        """The x axis: frame number or time of the frame."""
        if mode == "time" and not self.has_time:
            return
        self.xmode = mode
        x = self._xvals()
        if mode == "time":
            first, last = self.times[0][:10], self.times[-1][:10]
            self.set_x(_("Time"), kind="time", dates=first if first == last else f"{first} → {last}")
        else:
            self.set_x(_("Frame number"), kind="int")
        for t in (self.d_f, self.d_t):
            t.set_data(x, t.y)
        self._classify()
        self.fit()

    # ------------------------------------------------------------ the handles, the clicks
    def _line_moved(self, name: str, v: float):
        if name == "fmax":
            self.fmax = float(v)
        else:
            self.tmin = float(min(max(v, 0.0), 100.0))
            self.h_t.value = self.tmin
        self._classify()

    def _clicked(self, trace, i: int):
        if 0 <= i < self.n:
            self.current = i
            self.picked.emit(i)

    def _band(self, rect: QRectF, where):
        inside = set()
        x = self._xvals()
        for panel, vals in ((self.p_f, self.fwhm), (self.p_t, self.tr)):
            px, py = panel.fx(x), panel.fy(vals)
            for i in range(self.n):
                if np.isfinite(vals[i]) and rect.contains(QPointF(float(px[i]), float(py[i]))):
                    inside.add(i)
        if not inside:
            return
        m = QMenu(self)
        a1 = m.addAction(_("Exclude these {n} frames").format(n=len(inside)))
        a2 = m.addAction(_("Keep these {n} frames").format(n=len(inside)))
        act = self.run_menu(m, where)
        if act is not None and act.text() == a1.text():
            self.hand |= inside
        elif act is not None and act.text() == a2.text():
            self.hand -= inside
        else:
            return
        self._classify()
        self.changed.emit()

    def reset_thresholds(self):
        if not self.thresholds:
            return
        self.tmin, self.fmax = self.sugg
        self.h_t.value, self.h_f.value = self.tmin, self.fmax
        self._classify()
        self.changed.emit()

    def reset(self):
        """The suggested thresholds, and no frame excluded by hand."""
        self.hand.clear()
        if not self.thresholds:
            self._classify()
            self.changed.emit()
            return
        self.reset_thresholds()

    # ------------------------------------------------------------ the readout, the ring on the picked frame
    def readout_lines(self, h):
        if not h.items:
            return "", []
        i = next((it[1] for it in h.items if it[1] is not None), None)
        if i is None:
            return "", []
        ex = bool(self.excluded()[i])
        if i in self.hand:
            state, col = _("excluded by hand"), HAND
        elif ex:
            state, col = _("excluded"), OUT
        else:
            state, col = _("kept"), KEEP
        head = f"{_('Frame')} {i + 1} · {self.names[i]}"
        lines = [(None, _("Time"), self.times[i][11:16] or "–"),
                 (BLUE, "FWHM", f"{num(self.fwhm[i], 2)} px"),
                 (AMBER, _("Transparency"), f"{num(self.tr[i], 0)} %"),
                 (col, _("State"), state)]
        return head, lines

    def paint_overlay(self, p):
        i = self._current
        if i is None or not (0 <= i < self.n):
            return
        x = self._xvals()[i]
        p.setPen(QPen(QColor("#ffffff"), 1.6))
        p.setBrush(Qt.NoBrush)
        for panel, vals in ((self.p_f, self.fwhm), (self.p_t, self.tr)):
            if np.isfinite(vals[i]) and np.isfinite(x):
                px, py = float(panel.fx(x)), float(panel.fy(vals[i]))
                if panel.plot.contains(QPointF(px, py)):
                    p.drawEllipse(QPointF(px, py), 7, 7)

    # ------------------------------------------------------------ what the views and the tests read
    def _panels(self):
        """The two plots (top: star size, bottom: transparency), in the widget's coordinates."""
        return self.p_f.plot, self.p_t.plot

    def _ranges(self):
        """The values the plots show now: ((low, high) of the star size, (low, high) of the transparency)."""
        return (self.p_f.y0, self.p_f.y1), (self.p_t.y0, self.p_t.y1)

    def _xy(self, r: QRectF, i, v, lo, hi) -> QPointF:
        """Where frame ``i`` with value ``v`` is in the plot ``r``, whose vertical range is ``lo`` to ``hi``."""
        panel = self.p_t if abs(r.top() - self.p_t.plot.top()) < 0.5 else self.p_f
        return QPointF(float(panel.fx(self._xvals()[i])), r.bottom() - (v - lo) / (hi - lo) * r.height())


# ============================================================================================ Color
CHANNELS = ((QColor("#ff6b6b"), "Red"), (QColor("#5fd38d"), "Green"), (QColor("#5ea1ff"), "Blue"))
LINES = ((486.1, "Hβ"), (500.7, "[OIII]"), (656.3, "Hα"))           # the emission lines drawn (nm)


class SpectralChart(ChartView):
    """An approximation of the sensor's response, rebuilt from the Gaia spectra of the field's stars
    (``core.spectral``): the three channels' curves relative to the strongest, and the emission lines of the nebulae.
    Approximate: star spectra are broad, a narrow feature (the H-alpha line) can be well off."""

    def __init__(self, r, parent=None):
        super().__init__(_("Approximate sensor response, rebuilt from {n} stars of the field").format(n=r.stars),
                         parent, zoom="xy", readout="x", export_name="spectral_response")
        self.setMinimumHeight(260)
        self.r = r
        self.x_pad = 0.0                                          # the curves are known from lam[0] to lam[-1]
        self.add_panel(_("Relative response"), "", digits=None)
        self.set_x(_("Wavelength (nm)"), kind="num", digits=0)
        for c, (col, en) in enumerate(CHANNELS):
            self.add_line(_(en), r.lam, np.maximum(r.S[c], 0.0), color=col, width=2.2, interp=True, digits=2)
        for lam0, name in LINES:
            self.add_vline(lam0, name)
        self.set_home(0, y=(0.0, 1.08))


class ColorLinesChart(ChartView):
    """The colour the camera gives to each wavelength, from the approximate curves: the share of each channel in the
    signal of a pure line. The emission lines of the nebulae (Hβ, [OIII], Hα) sit where their colour is read."""

    def __init__(self, r, parent=None):
        super().__init__(_("Colour the camera gives to each wavelength"), parent, zoom="xy", readout="x",
                         export_name="line_colours")
        self.setMinimumHeight(260)
        self.r = r
        share = r.shares()
        self.x_pad = 0.0
        self.add_panel(_("Share of the signal (%)"), "%", digits=0)
        self.set_x(_("Wavelength (nm)"), kind="num", digits=0)
        for c, (col, en) in enumerate(CHANNELS):
            self.add_line(_(en), r.lam, share[c], color=col, width=2.2, interp=False, digits=0, unit="%")
        for lam0, name in LINES:
            self.add_vline(lam0, name)
        self.set_home(0, y=(0.0, 100.0))


def color_tabs(r) -> list[tuple[str, ChartView]]:
    """The charts of the Color block as (tab title, widget). ``r``: ``api.spectral_response``. No I/O."""
    return [(_("Approximate spectral response"), SpectralChart(r)), (_("Colour of the lines"), ColorLinesChart(r))]


# ============================================================================================ Registration
@dataclass
class RegistrationData:
    """What the Registration charts draw, one value per light frame (NaN: not measured or not aligned)."""

    names: list
    iso: list                    # the time of each frame as written ("" when unknown)
    secs: np.ndarray             # the same in seconds since 1970 (the clock as written), NaN when unknown
    fwhm: np.ndarray             # px
    transp: np.ndarray           # % of the best frames (90th percentile of the aligned frames)
    nmatch: np.ndarray           # stars matched with the reference frame
    nstars: np.ndarray           # stars detected
    rms: np.ndarray              # alignment residual, px
    aligned: np.ndarray
    dx: np.ndarray               # offset of the stars from the reference frame, at the centre of the frame, px
    dy: np.ndarray
    ref: int                     # the reference frame
    scale: float | None          # arcsec per sensor pixel (None: unknown)
    roundness: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def has_time(self) -> bool:
        return self.n > 1 and bool(np.isfinite(self.secs).all())

    @property
    def dates(self) -> str:
        first, last = (self.iso[0] or "")[:10], (self.iso[-1] or "")[:10]
        return first if first == last else f"{first} → {last}"


def registration_data(project, wf) -> RegistrationData:
    """Read what the Registration charts draw. ``project``: the api's project, ``wf``: the workflow (id or object).
    Reads files: call it from the worker thread. Raises ``api.NotDone`` when registration is not computed.

    The frame table gives the names, times, star size and transparency; the register stage's ``reg.npz`` the matched
    and detected stars, the residual and the polynomial that maps each frame on the reference: its constant term is
    where the centre of the frame falls on the reference, so the stars of the frame sit that far from the
    reference's (``coef`` is 0-based in the sensor's pixels; the sensor's size is the calibration cache's).
    The scale comes from the Astrometry stage when it is done, else from the focal length and the pixel size."""
    from .. import api
    table = api.frame_table(project, wf)
    with api.stage_array(project, wf, "register", "reg.npz") as z:
        reg = {k: np.array(z[k]) for k in z.files}
    rows = table.rows
    n = len(rows)
    width = height = None
    out = api.stage_outputs(project, wf, "calibrate")
    if out is not None:
        f = Path(out.dir) / "cache.json"
        if f.is_file():
            cache = json.loads(f.read_text(encoding="utf-8"))
            width, height = float(cache["width"]), float(cache["height"])
    scale = None
    ast = api.stage_outputs(project, wf, "astrometry")
    if ast is not None and ast.metrics.get("sensor_scale_arcsec"):
        scale = float(ast.metrics["sensor_scale_arcsec"])
    inv_dir = api.stage_outputs(project, wf, "inventory")
    if inv_dir is not None and (width is None or scale is None):
        f = Path(inv_dir.dir) / "frames.json"
        if f.is_file():
            first = (json.loads(f.read_text(encoding="utf-8")).get("lights") or [{}])[0]
            width = width or (float(first["width"]) if first.get("width") else None)
            height = height or (float(first["height"]) if first.get("height") else None)
            if scale is None and first.get("focal_mm") and first.get("pixel_um"):
                scale = 206.265 * float(first["pixel_um"]) / float(first["focal_mm"])
    rms = np.array(reg["rms"][:n], float)
    aligned = np.isfinite(rms) & np.array([r.status != "not aligned" for r in rows], bool)
    coef = np.array(reg["coef"][:n], float)
    if width is not None and height is not None:
        dx = np.where(aligned, width / 2.0 - coef[:, 0, 0], np.nan)
        dy = np.where(aligned, height / 2.0 - coef[:, 1, 0], np.nan)
    else:
        dx = dy = np.full(n, np.nan)
    tr = np.array([r.transparency for r in rows], float)
    ok = tr[aligned & np.isfinite(tr)]
    p90 = float(np.percentile(ok, 90)) if len(ok) else 1.0
    secs = np.array([np.nan if (s := to_seconds(r.time)) is None else s for r in rows], float)
    roundness = None
    if "roundness" in reg:
        roundness = np.array(reg["roundness"][:n], float)
    elif "ellipticity" in reg:
        roundness = 1.0 - np.array(reg["ellipticity"][:n], float)
    return RegistrationData(
        names=[r.name for r in rows], iso=[r.time or "" for r in rows], secs=secs,
        fwhm=np.array([r.fwhm for r in rows], float), transp=np.where(aligned, 100 * tr / p90, np.nan),
        nmatch=np.where(aligned, np.array(reg["nmatch"][:n], float), np.nan),
        nstars=np.array(reg["nstars"][:n], float), rms=np.where(aligned, rms, np.nan), aligned=aligned,
        dx=dx, dy=dy, ref=int(reg["ref"]), scale=scale, roundness=roundness)


class _ByFrame(ChartView):
    """A chart with one value per frame: the x axis is the frame number or the time (a choice in the title row)."""

    def _setup_x(self, d: RegistrationData, series: list):
        """``series``: [(trace, y values)] whose x follows the choice."""
        self.d = d
        self._series = series
        self.xmode = "time" if d.has_time else "frame"
        self._set_axis()
        x = self._xvals()
        for t, y in series:
            t.set_data(x, y)
        if d.has_time:
            self.add_choice([("frame", _("Frame number")), ("time", _("Time"))], self.xmode, self.set_xmode)

    def _xvals(self) -> np.ndarray:
        return self.d.secs if self.xmode == "time" else np.arange(self.d.n) + 1.0

    def _set_axis(self):
        if self.xmode == "time":
            self.set_x(_("Time"), kind="time", dates=self.d.dates)
        else:
            self.set_x(_("Frame number"), kind="int")
        self.x_short = _("Frame")

    def set_xmode(self, mode: str):
        if mode == "time" and not self.d.has_time:
            return
        self.xmode = mode
        self._set_axis()
        x = self._xvals()
        for t, y in self._series:
            t.set_data(x, y)
        self.fit()

    def _extra(self) -> dict:
        d = self.d
        notes = [_("reference frame") if i == d.ref else "" for i in range(d.n)]
        return {_("Frame"): d.names, _("Time"): d.iso, _("Note"): notes}


class QualityChart(_ByFrame):
    """The quality of each frame: star size, roundness (when the stage measures it), stars and transparency."""

    def __init__(self, d: RegistrationData, parent=None):
        super().__init__(_("Quality of the frames"), parent, zoom="x", readout="x", export_name="registration_quality")
        self.setMinimumHeight(360)
        x = np.arange(d.n) + 1.0
        series = []
        self.d = d
        extra = self._extra()
        show = (_("Frame"), _("Note"))

        def add(panel, name, y, color, unit, digits, **kw):
            t = self.add_scatter(name, x, y, panel=panel, color=color, size=6, unit=unit, digits=digits, alpha=0.9,
                                 extra=extra, show_extra=show, **kw)
            series.append((t, y))
            return t

        p = self.add_panel(_("Star size (FWHM, px)"), "px", digits=1)
        add(p, "FWHM", d.fwhm, BLUE, "px", 2)
        if d.roundness is not None:
            p = self.add_panel(_("Roundness (1 = round)"), "", digits=2)
            add(p, _("Roundness"), d.roundness, GREEN, "", 2)
        p = self.add_panel(_("Stars per frame"), "", digits=0, zero=True)
        add(p, _("Stars detected"), d.nstars, QColor("#6e7a99"), "", 0)
        add(p, _("Stars matched"), d.nmatch, VIOLET, "", 0)
        p = self.add_panel(_("Transparency (% of the best frames)"), "%", digits=0)
        add(p, _("Transparency"), d.transp, AMBER, "%", 0)
        self._setup_x(d, series)
        self.fit()


class _Offsets(_ByFrame):
    """What the drift charts share: the offsets in pixels or in arcseconds, a choice in the title row."""

    def _unit_choice(self, d: RegistrationData):
        self.unit = "px"
        if d.scale:
            self.unit = "arcsec" if prefs.get("units") == "arcsec" else "px"
            self.add_choice([("px", "px"), ("arcsec", _("arcsec"))], self.unit, self.set_unit)

    def _k(self, unit: str) -> float:
        return self.d.scale if unit == "arcsec" and self.d.scale else 1.0

    def _alt(self):
        """What the readout adds in brackets: the other unit."""
        if not self.d.scale:
            return None
        return (self.d.scale, "arcsec", 1) if self.unit == "px" else (1.0 / self.d.scale, "px", 1)

    def set_unit(self, unit: str):
        raise NotImplementedError


class DriftChart(_Offsets):
    """Where the stars of each frame are against the reference frame: the x and y offsets of the frame's centre, in
    pixels (the sensor's) or arcseconds (the sky's), against time or frame number. The mount's drift, its periodic
    error, a dither or a meridian flip show here."""

    def __init__(self, d: RegistrationData, parent=None):
        super().__init__(_("Drift from the reference frame"), parent, zoom="x", readout="x", export_name="drift")
        self.setMinimumHeight(300)
        self.d = d
        self._unit_choice(d)
        k = self._k(self.unit)
        self.add_panel(_("Offset ({unit})").format(unit=self._unit_name()), "", digits=None)
        x = np.arange(d.n) + 1.0
        extra = self._extra()
        show = (_("Frame"), _("Note"))
        self.t_x = self.add_line(_("x offset"), x, d.dx * k, color=BLUE, points=True, size=5, digits=2,
                                 unit=self._unit_name(), alt=self._alt(), extra=extra, show_extra=show)
        self.t_y = self.add_line(_("y offset"), x, d.dy * k, color=AMBER, points=True, size=5, digits=2,
                                 unit=self._unit_name(), alt=self._alt(), extra=extra, show_extra=show)
        self._setup_x(d, [(self.t_x, d.dx * k), (self.t_y, d.dy * k)])
        self.fit()

    def _unit_name(self) -> str:
        return _("arcsec") if self.unit == "arcsec" else "px"

    def set_unit(self, unit: str):
        if unit == self.unit:
            return
        old = self._k(self.unit)
        self.unit = unit
        k = self._k(unit)
        for t, base in ((self.t_x, self.d.dx), (self.t_y, self.d.dy)):
            t.unit, t.alt = self._unit_name(), self._alt()
            t.set_data(t.x, base * k)
        self._series = [(self.t_x, self.d.dx * k), (self.t_y, self.d.dy * k)]
        self.panels[0].title = _("Offset ({unit})").format(unit=self._unit_name())
        self.scale_axes(1.0, k / old)


class TrackChart(_Offsets):
    """The offsets against each other (x against y, a pixel is the same in both): the path of the drift, with the
    reference frame at the origin and the time as colour. Dithering makes a cloud, a drift a line, periodic error and
    field rotation loops and arcs."""

    MIN_PLOT_H = 120.0                  # a pixel the same in x and y: lower, the path was a thin strip (issue #132)

    def __init__(self, d: RegistrationData, parent=None, left_out: dict | None = None):
        """``left_out``: in Selection, the frames the stack leaves out (name -> reason), circled in red on the path."""
        super().__init__(_("Drift path (x / y)"), parent, zoom="xy", readout="xy", snap=18.0,
                         export_name="drift_path")
        self.setMinimumHeight(300)
        self.d = d
        self.left_out = left_out or {}
        self._unit_choice(d)
        k = self._k(self.unit)
        self.p = self.add_panel("", "", digits=None, equal=True)
        by_time = d.has_time
        color = d.secs if by_time else np.arange(d.n, dtype=float)
        ok = np.isfinite(d.dx) & np.isfinite(d.dy)
        self.t_trail = self.add_line(_("path"), d.dx * k, d.dy * k, color=QColor("#4a5163"), width=1.0, legend=False,
                                     readout=False, export=False)
        self.t_pts = self.add_scatter(_("y offset"), d.dx * k, d.dy * k, color=BLUE, size=7, c=color, legend=False,
                                      unit=self._unit_name(), digits=2, extra=self._extra(),
                                      show_extra=(_("Frame"), _("Note")))
        ref = d.ref if 0 <= d.ref < d.n and ok[d.ref] else None
        if ref is not None:
            self.add_scatter(_("reference frame"), [d.dx[ref] * k], [d.dy[ref] * k], color=QColor("#ffffff"), size=11,
                             readout=False, export=False)
        out = [i for i, name in enumerate(d.names) if name in self.left_out and ok[i]]
        if out:
            self.add_scatter(_("excluded"), d.dx[out] * k, d.dy[out] * k, color=OUT, size=13, legend=True,
                             unit=self._unit_name(), digits=2,
                             extra={_("Frame"): [d.names[i] for i in out],
                                    _("Reason"): [self.left_out[d.names[i]] for i in out]},
                             show_extra=(_("Frame"), _("Reason")))
        if by_time and ok.any():
            self.colorbar = (_("Time"), clock(color[ok].min()), clock(color[ok].max()))
        elif ok.any():
            lo, hi = np.flatnonzero(ok)[[0, -1]]
            self.colorbar = (_("Frame"), str(lo + 1), str(hi + 1))
        self._titles()
        self.fit()

    def _unit_name(self) -> str:
        return _("arcsec") if self.unit == "arcsec" else "px"

    def _titles(self):
        u = self._unit_name()
        self.set_x(_("x offset"), u, kind="num")
        self.p.title = _("y offset ({unit})").format(unit=u)

    def set_unit(self, unit: str):
        if unit == self.unit:
            return
        old = self._k(self.unit)
        self.unit = unit
        k = self._k(unit)
        self.t_trail.set_data(self.d.dx * k, self.d.dy * k)
        self.t_pts.set_data(self.d.dx * k, self.d.dy * k)
        self.t_pts.unit = self._unit_name()
        for t in self.traces:
            if t.name in (_("reference frame"), _("excluded")):
                t.set_data(t.x * k / old, t.y * k / old)
                if t.name == _("excluded"):
                    t.unit = self._unit_name()
        self._titles()
        self.scale_axes(k / old, k / old)

    def readout_lines(self, h):
        if not h.items:
            return "", []
        t, i, _y = h.items[0]
        d = self.d
        k = self._k(self.unit)
        u = self._unit_name()
        alt = self._alt()

        def val(v):
            s = f"{num(v, 2)} {u}"
            return s + (f" ({num(v * alt[0], alt[2])} {alt[1]})" if alt else "")
        head = f"{_('Frame')} {i + 1} · {d.names[i]}"
        lines = [(None, _("Time"), (d.iso[i] or "")[11:16] or "–"), (BLUE, _("x offset"), val(d.dx[i] * k)),
                 (AMBER, _("y offset"), val(d.dy[i] * k))]
        if i == d.ref:
            lines.append((None, _("reference frame"), ""))
        return head, lines


class DistributionChart(ChartView):
    """How the frames spread: a histogram of the star size or of the transparency (a choice in the title row), with
    the median marked. The bars count frames."""

    def __init__(self, d: RegistrationData, parent=None):
        super().__init__(_("Distribution of the frames"), parent, zoom="x", readout="x", export_name="distribution")
        self.setMinimumHeight(300)
        self.d = d
        self.what = ""
        self.add_panel(_("Frames"), "", digits=0, zero=True)
        self.add_choice([("fwhm", "FWHM"), ("transp", _("Transparency"))], "fwhm", self.show_what)
        self.show_what("fwhm")

    def show_what(self, what: str):
        d = self.d
        self.what = what
        vals, color = (d.fwhm, BLUE) if what == "fwhm" else (d.transp, AMBER)
        self.clear()
        v = vals[np.isfinite(vals)]
        if what == "fwhm":
            self.set_x("FWHM", "px", kind="num")
        else:
            self.set_x(_("Transparency (% of the best frames)"), "", kind="num")
        if v.size:
            edges = np.histogram_bin_edges(v, bins=min(max(int(np.sqrt(v.size)) + 2, 5), 40))
            counts, _e = np.histogram(v, bins=edges)
            self.add_histogram(_("Frames"), edges, counts, color=color, digits=0, unit=_("frames"), legend=False)
            self.add_vline(float(np.median(v)), _("median"), color="#8b93a1")
        self.fit()


def registration_tabs(d: RegistrationData) -> list[tuple[str, ChartView]]:
    """The charts of the Registration block as (tab title, widget), in this order: Quality, Drift, Path, Distribution.
    No I/O: ``d`` is ``registration_data``'s. The drift charts need the offsets, so they are left out of a project whose
    sensor size is unknown."""
    tabs = [(_("Quality"), QualityChart(d))]
    if np.isfinite(d.dx).any():
        tabs += [(_("Drift"), DriftChart(d)), (_("Drift path (x / y)"), TrackChart(d))]
    tabs.append((_("Distribution"), DistributionChart(d)))
    return tabs
