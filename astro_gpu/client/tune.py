"""Inspect: one block, large, every result of it examinable.

At the top, the current workflow as a strip of blocks; in the middle, the chosen block's tabs; on the right, the same
panel as in Graph, with the measurements. Every picture is shown in the common viewer (``imageview.ImageView``: zoom,
stretch with its histogram, pixel readout in ADU and sky coordinates, compare with the previous steps, Export, Send to
Post); the charts are the interactive ones (``charts``); the frames are sequences with their numbers (info band,
Exclude / Include).

What each block shows (the tabs are ``Steps.visuals``):

* Calibration: the masters, every dark, flat, bias and dark flat (left out by hand, or taken back), the calibrated
  lights compared with the raw;
* Registration: the reference frame, the charts (quality, drift with its path under it, drift path, distribution) and
  the frame table;
* Selection: the thresholds chart and the frames;
* Stacking: the stack compared with a calibrated frame and the half-stacks, the rejection map, the coverage, the
  weights;
* Astrometry: the stack with the RA / Dec grid and the catalogue stars, then the survey;
* Color: before / after, the sensor response charts, the per-star fit;
* Denoise: compared with the stack, the same stretch on both; Stars; Gradient: the result compared with the earlier
  steps, the sky model, the gradient, the star flat; Output: the final image (the render worker's) compared with the
  earlier steps.

The data of a block (masters, tables, charts) are read by the worker (``services/inspect``: ``request_job``); the
pictures are read in the viewer's own threads (their sources are callables). Nothing here reads a file. An Outdated
block shows the result it had ("previous result"); a Pending one says so, with Run to here. What the engine does not
give yet says so: "Not available yet: recompute <block> to get it".

Mounting (the window's ``inspect_page``): ``steps = Steps(model)``; ``browser = SourcesView(model)``;
``frames = FramesView(model, browser)``; ``tune = TuneView(model, steps, frames, post_page)``. The view refreshes itself
from the model's signals while it is visible. Its signals: ``back`` (the Graph button, Esc), ``runHere`` (Run to the
selected block of the current workflow), ``openPost``, ``removeTool`` (Remove on the panel of an optional block: the
graph's ``remove_tool(node, None)``), ``toasted(text)``, ``sendToPost(name, source, key)`` (Send to Post of a viewer:
the window gives it to Post's ``add_image_layer``; ``can_post``, a callable the window sets, says whether Post takes
it). Importing is the views' own: ``sources.import_folder(model, parent, folder=None, new_session=False)``.
"""
from __future__ import annotations

import html
import os
import re
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import texts
from . import prefs
from .blocks import ENGINE_BLOCK, Graph, block_state, block_title, chain, stages_of
from .charts import (
    AMBER,
    BLUE,
    RED,
    DistributionChart,
    DriftChart,
    FramesChart,
    QualityChart,
    TrackChart,
    color_tabs,
)
from .chartview import ChartView
from .i18n import _, hm, hours_text, language, maybe, number, plural
from .imageview import ImageView, ViewImage, quick_thumbnail
from .panel import STATE_COLOR, STATE_WORD, Panel
from .services import inspect as si
from .services.inspect import CATALOGUE_G_MAX, VisualSpec
from .steps import CAMERA_NOTE, COVERAGE_DRIZZLE_NOTE, MODEL_COLOURS_NOTE, Steps
from .widgets import ACCENT, WARN, clear, label

STATE_COL = dict(STATE_COLOR, pending="#5d6470")      # the strip: a block not computed yet is a dark grey


class Strip(QFrame):
    """The current workflow, block after block."""
    picked = Signal(str)
    back = Signal()
    workflow = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("strip")
        self.setStyleSheet("QFrame#strip { background: #1c1f26; border-bottom: 1px solid #2a2f38; }")
        self.h = QHBoxLayout(self)
        self.h.setContentsMargins(10, 8, 10, 8)
        self.h.setSpacing(4)

    def build(self, s, g: Graph):
        clear(self.h)
        bk = QPushButton("◂  " + _("Graph"))
        bk.setToolTip(_("Back to the graph (Esc)"))
        bk.clicked.connect(self.back.emit)
        self.h.addWidget(bk)
        self.h.addSpacing(10)
        p = s.path
        if s.sel_step == "final":
            self.h.addWidget(label(_("Post") + "  ·  " + _("the workflows' layers, combined"), "h2"))
            self.h.addStretch(1)
            return
        if p is not None:
            cb = QComboBox()
            for q in s.paths:
                cb.addItem(q.name, q.pid)
            cb.setCurrentIndex(s.current)
            cb.setMinimumWidth(200)
            cb.currentIndexChanged.connect(lambda i: self.workflow.emit(i))
            self.h.addWidget(cb)
            self.h.addSpacing(10)
        keys = ["sources"] + (chain(g.steps, p) if p is not None else [])
        for i, k in enumerate(keys):
            if i:
                a = QLabel("›")
                a.setStyleSheet("color: #6b7280; background: transparent;")
                self.h.addWidget(a)
            node = g.node_of(p, k)
            st = g.state(node) if node is not None else "pending"
            b = QPushButton(block_title(None, k) if k == "post" else block_title(p, k))
            sel = k == s.sel_step
            b.setStyleSheet(f"QPushButton {{ border-left: 3px solid {STATE_COL.get(st, '#5d6470')}; padding: 4px 8px; "
                            f"background: {'#2b3446' if sel else '#22262e'}; "
                            f"border-top: 1px solid {ACCENT if sel else '#353b47'}; "
                            f"border-right: 1px solid {ACCENT if sel else '#353b47'}; "
                            f"border-bottom: 1px solid {ACCENT if sel else '#353b47'}; "
                            f"{'font-weight: 600; color: #ffffff;' if sel else ''} }}")
            b.setToolTip(_(STATE_WORD[st]))
            b.clicked.connect(lambda _x=False, kk=k: self.picked.emit(kk))
            self.h.addWidget(b)
        if p is not None:
            a = QLabel("›")
            a.setStyleSheet("color: #6b7280; background: transparent;")
            self.h.addWidget(a)
            b = QPushButton(_("Post"))
            b.setStyleSheet("QPushButton { padding: 4px 8px; }")
            b.clicked.connect(lambda: self.picked.emit("final"))
            self.h.addWidget(b)
        self.h.addStretch(1)


# ---------------------------------------------------------------------------------------------- small pieces
class NumItem(QTableWidgetItem):
    """A table cell sorted by its number."""

    def __init__(self, value, text: str):
        super().__init__(text)
        self.v = value if value is not None and np.isfinite(value) else -np.inf
        self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def __lt__(self, other):
        return self.v < getattr(other, "v", -np.inf)


def make_table(headers: list[str], rows: list[list], digits: dict[int, int] | None = None,
               red: list[bool] | None = None) -> QTableWidget:
    """A sortable table: a cell is text, or a number shown with ``digits[column]`` decimals (– when unknown)."""
    digits = digits or {}
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setStyleSheet("QTableWidget { background: #1c1f26; gridline-color: #2a2f38; } "
                    "QHeaderView::section { background: #23272f; border: none; padding: 4px; }")
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            if c in digits:
                ok = v is not None and np.isfinite(float(v))
                it = NumItem(float(v) if ok else None, number(float(v), digits[c]) if ok else "–")
            else:
                it = QTableWidgetItem("" if v is None else str(v))
            if red is not None and red[r]:
                it.setForeground(QColor("#e06c6c"))
            t.setItem(r, c, it)
    t.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)       # the rows as given until a header is clicked
    t.setSortingEnabled(True)
    t.resizeColumnsToContents()
    for c in range(len(headers)):
        t.setColumnWidth(c, t.columnWidth(c) + 14)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def colour_index(v) -> float:
    """One number for a star's colour: the value itself, a difference of two, or −2.5 log10(B / R) of three
    fluxes (a B − R index)."""
    a = np.atleast_1d(np.asarray(v, float)).ravel()
    if a.size == 1:
        return float(a[0])
    if a.size == 2:
        return float(a[0] - a[1])
    if a[0] > 0 and a[2] > 0:
        return float(-2.5 * np.log10(a[2] / a[0]))
    return float("nan")


CHANNEL_COLORS = (QColor("#ff6b6b"), QColor("#5fd38d"), QColor("#5ea1ff"))


def per_channel(v) -> np.ndarray:
    """(3,) values of R, G, B from a number, a sequence, or None (NaN)."""
    if v is None:
        return np.full(3, np.nan)
    a = np.atleast_1d(np.asarray(v, float)).ravel()
    return np.resize(a, 3) if a.size else np.full(3, np.nan)


def weight_rows(b) -> list[dict]:
    """One record per frame of a stack: name, time, weight, normalisation offset and scale (R, G, B, or None), FWHM,
    transparency, and the reason it is left out ('' kept)."""
    tab = {r.name: r for r in (b.frames.rows if b.frames is not None else [])}
    out = []
    d = b.diag
    if d is not None and getattr(d, "weights", None):
        left = dict(getattr(d, "excluded", None) or {})
        why = {name: exclusion_text(ex) for name, ex in (getattr(d, "exclusions", None) or {}).items()}
        for name, w, off, sc in d.weights:
            r = tab.get(name)
            # the stack's own verdict: a frame with a weight was stacked (the frame table may follow newer settings)
            st = "" if w is not None and w > 0 else \
                (left.get(name) or (r.status if r is not None else "") or "left out")
            out.append(dict(name=name, time=(r.time if r is not None else "") or "", weight=w, offset=off, scale=sc,
                            fwhm=r.fwhm if r is not None else None,
                            transparency=r.transparency if r is not None else None,
                            status=st, why=(why.get(name) or _(st)) if st else ""))
        return out
    for r in tab.values():
        out.append(dict(name=r.name, time=r.time or "", weight=r.weight, offset=None, scale=None, fwhm=r.fwhm,
                        transparency=r.transparency, status=r.status, why=_(r.status) if r.status else ""))
    return out


def rejection_rate_text(d) -> str:
    """The share of rejected pixels of a stack next to the usual range (0.1 to 0.5 %), as rich text: in amber when
    outside it; with the Classic method, the rejection its rule chose and why. '' when the stack does not say."""
    rej = dict(getattr(d, "rejection", None) or {})
    pct = rej.get("rejected_pct")
    if pct is None:
        pct = getattr(d, "rejected_pct", None)
    if pct is None:
        return ""
    lo, hi = (rej.get("reference_pct") or [0.1, 0.5])[:2]
    pct = float(pct)
    # a tiny share keeps its first figures (0.0010 %, as the Measurements say), never "0.00 %"
    out = _("Rejected pixels: {pct} %").format(pct=number(pct, 1 if pct >= 10 else 2 if pct >= 0.01 or pct == 0
                                                       else 4))
    guide = _("Usual range: {lo} to {hi} %").format(lo=number(lo, 1), hi=number(hi, 1))
    if pct < lo:
        guide += " · " + _("below it: few outliers were found")
    elif pct > hi:
        guide += " · " + _("above it: many pixels were rejected")
    colour = "#5fbf7f" if lo <= pct <= hi else WARN
    text = f"<span style='color:{colour}'>{html.escape(out, quote=False)}</span>   ·   {html.escape(guide, quote=False)}"
    if rej.get("classic"):
        how = texts.choice_label(str(rej["classic"]), language())
        rt = rej.get("reason_template") or {}
        why = texts.format_reason(str(rt.get("template") or ""), rt.get("args"), language()) if rt.get("template") \
            else ""
        text += "<br>" + html.escape(_("Rejection: {how}").format(how=how), quote=False) + (
            f" ({html.escape(why, quote=False)})" if why else "")
    return text


def coverage_text(d, drizzle: bool) -> str:
    """The variation of the coverage of each colour (std / median) against DrizzlePac's 20 %, as rich text: in amber
    above it. '' when the stack does not say."""
    cov = dict(getattr(d, "coverage", None) or {})
    var = cov.get("variation") or {}
    if not var or not drizzle:
        return ""
    limit = 100 * float((cov.get("estimate") or {}).get("limit") or 0.2)
    parts = []
    for c, name in (("R", _("red")), ("G", _("green")), ("B", _("blue"))):
        x = var.get(c)
        if x is None:
            continue
        x = 100 * float(x)
        colour = WARN if x > limit else "#5fbf7f"
        parts.append(f"{html.escape(name, quote=False)} <span style='color:{colour}'>{number(x, 0)} %</span>")
    if not parts:
        return ""
    head = _("Coverage variation (std / median): {parts}").format(parts=" · ".join(parts))
    return head + "   ·   " + html.escape(_("even under {limit} %").format(limit=number(limit, 0)), quote=False) + \
        "<br>" + html.escape(_(COVERAGE_DRIZZLE_NOTE), quote=False)


def exclusion_text(reasons) -> str:
    """Why a frame was left out of the stack, in the language shown: the engine's reasons ([{"code", "template",
    "args"}], ``StackDiag.exclusions``) formatted and joined."""
    lang = language()
    return "; ".join(texts.format_reason(str(x.get("template") or ""), x.get("args"), lang)
                     for x in reasons or [] if x.get("template"))


class WeightsChart(ChartView):
    """The weight the stack gave each frame and, once the engine gives them, its normalisation: the offset (the
    frame's sky at the field centre, ADU) and the scale that bring each channel of the frame to the reference's
    level."""

    def __init__(self, rows: list[dict], parent=None):
        super().__init__(_("Weight of each frame in the stack"), parent, zoom="x", readout="x", export_name="weights")
        self.setMinimumHeight(300)
        n = len(rows)
        x = np.arange(n) + 1.0
        w = np.array([_f(r["weight"]) for r in rows])
        kept = np.array([not r["status"] for r in rows], bool)
        extra = {_("Frame"): [r["name"] for r in rows], _("Time"): [r["time"] for r in rows]}
        p0 = self.add_panel(_("Weight"), "", digits=2, zero=True)
        self.add_scatter(_("kept"), x[kept], w[kept], panel=p0, color=BLUE, size=7, readout=False, export=False)
        if (~kept).any():
            self.add_scatter(_("left out"), x[~kept], w[~kept], panel=p0, color=RED, size=7, readout=False,
                             export=False)
        self.add_data(_("Weight"), x, w, panel=p0, digits=3, extra=extra, show_extra=(_("Frame"),))
        if any(r["scale"] is not None for r in rows):
            sc = np.stack([per_channel(r["scale"]) for r in rows]) if rows else np.zeros((0, 3))
            off = np.stack([per_channel(r["offset"]) for r in rows]) if rows else np.zeros((0, 3))
            p1 = self.add_panel(_("Normalisation scale"), "", digits=3)
            p2 = self.add_panel(_("Normalisation offset (ADU)"), "ADU", digits=0)
            for c, name in enumerate((_("Red"), _("Green"), _("Blue"))):
                self.add_line(name, x, sc[:, c], panel=p1, color=CHANNEL_COLORS[c], points=True, size=4, digits=4)
                self.add_line(name, x, off[:, c], panel=p2, color=CHANNEL_COLORS[c], points=True, size=4, digits=1,
                              unit="ADU", legend=False)
        self.set_x(_("Frame number"), kind="int")
        self.x_short = _("Frame")
        self.fit()


def star_colour(s) -> tuple[float, float, float, float]:
    """(catalogue colour, measured colour, residual, G) of a star of the colour fit: the catalogue's Gaia BP − RP
    (else the x of its chromaticity), the camera's −2.5 log10(B / R) of its measured fractions."""
    bp = getattr(s, "bp_rp", None)
    cat = float(bp) if bp is not None else colour_index(getattr(s, "catalogue", None) or [np.nan])
    meas = colour_index(s.measured) if getattr(s, "measured", None) is not None else float("nan")
    res = _f(s.residual) if getattr(s, "residual", None) is not None else float("nan")
    return cat, meas, res, _f(getattr(s, "g", np.nan))


class ColorFitChart(ChartView):
    """The photometric colour fit, star by star, as in a PCC / SPCC plot: the camera's colour of each star against
    its catalogue colour (Gaia BP − RP), the stars the fit used in blue and the rejected ones in red; below, the
    residual of each used star (its chromaticity distance to the catalogue after the fit)."""

    def __init__(self, fit, parent=None):
        super().__init__(_("Star colours: the camera against the catalogue"), parent, zoom="xy", readout="xy",
                         export_name="colour_fit")
        self.setMinimumHeight(360)
        stars = list(getattr(fit, "stars", None) or [])
        vals = np.array([star_colour(s) for s in stars], float).reshape(-1, 4)
        cat, meas, res, g = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]
        used = np.array([bool(s.used) for s in stars], bool)
        reason = [_(str(getattr(s, "reason", "") or "")) for s in stars]     # the colour stage's STAR_REASONS
        p0 = self.add_panel(_("Camera colour (−2.5 log B / R)"), "", digits=2)
        self.add_scatter(_("used"), cat[used], meas[used], panel=p0, color=BLUE, size=4.5, alpha=0.7, digits=3,
                         extra={"G": list(g[used])})
        if (~used).any():
            self.add_scatter(_("rejected"), cat[~used], meas[~used], panel=p0, color=RED, size=4.5, alpha=0.8,
                             digits=3, extra={_("Reason"): [r for r, u in zip(reason, used) if not u]},
                             show_extra=(_("Reason"),))
        p1 = self.add_panel(_("Residual after the fit"), "", digits=3, weight=0.6, zero=True)
        self.add_scatter(_("residual"), cat[used], res[used], panel=p1, color=AMBER, size=4, alpha=0.7, digits=4)
        self.set_x(_("Catalogue colour (Gaia BP − RP)"), kind="num")
        self.fit()


# ---------------------------------------------------------------------------------------------- the view
class TuneView(QWidget):
    back = Signal()
    pick = Signal(str)                    # a block of the strip was chosen
    workflow = Signal(int)                # a workflow of the strip's list was chosen
    runHere = Signal()                    # Run to here: the selected block of the current workflow
    openPost = Signal()
    removeTool = Signal()                 # Remove, on the panel of an optional block
    toasted = Signal(str)                 # a word to say (what came with a change, why it was refused)
    sendToPost = Signal(str, object, object)       # name, source (what the viewer reads), key

    def __init__(self, project, steps, sources_view, post_page, parent=None, service=None):
        super().__init__(parent)
        steps = steps if steps is not None else Steps(project, service)
        self.s, self.steps = project, steps
        self.service = service or steps.service
        self.tab_of: dict[str, str] = {}                  # block -> the tab shown last
        self.bg_of: dict[str, str] = {}                   # block -> the image chosen behind the solution (Astrometry)
        self.cmp_pid = None                               # a workflow to compare with (a variant and its original)
        self.picked = None                                # the frame picked on the chart (an index of the frame table)
        self.show_hist = True                             # the viewers' stretch panel
        self.sky_show = {"grid": True, "stars": True, "objects": True}      # what the solution draws over a picture
        self._splits: dict[str, list[int]] = {}           # block / tab -> the sizes of its two parts (chart, image)
        self._cal_filter: dict[str, str] = {}             # calibration tab -> "all" | "flagged" (the suspicious frames)
        self._note_row = None                             # the line under the tabs, where a picture's options go
        self.can_post = None                              # callable: Post takes images (the window sets it)
        self._pending = False
        self._dirty = True
        self.views: list[ImageView] = []                  # the viewers of the tab shown
        self.view: ImageView | None = None                # the main one
        self.chart: FramesChart | None = None
        self.table = None                                 # the frame table of the Selection chart
        self.sel_frames: list[ViewImage] = []
        self._cur = None                                  # (workflow, block, VisualSpec) of the tab shown
        self._data: OrderedDict = OrderedDict()           # (job, keys) -> what the worker read
        self._asked: dict[str, tuple] = {}                # job -> (token, (job, keys))
        self._kept: dict[tuple, tuple] = {}               # (block, tab) -> (image key, view state, stretch)
        self._render_token = None
        self._render_spec = None                          # the final image the tab asked for (its own or a background)
        self._render_geom = None                          # (grid, origin, size) of the final image shown
        self._table_token = None
        self._seq_items: list = []                        # the frames of the sequence shown (FrameItem)
        self._pool: list[ImageView] = []                  # viewers kept for the next tabs
        self._conns: dict[int, list] = {}                 # viewer -> the connections of the tab shown
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.strip = Strip()
        self.strip.back.connect(self.back.emit)
        self.strip.picked.connect(self._strip_pick)
        self.strip.workflow.connect(self._strip_workflow)
        v.addWidget(self.strip)
        self.pages = QStackedWidget()
        v.addWidget(self.pages, 1)
        body = QWidget()
        h = QHBoxLayout(body)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(10, 8, 10, 6)
        mv.setSpacing(6)
        self.head = QHBoxLayout()
        mv.addLayout(self.head)
        self.center = QVBoxLayout()
        self.center.setSpacing(6)
        mv.addLayout(self.center, 1)
        h.addWidget(mid, 1)
        self.panel = Panel(project, steps, tune=True)
        h.addWidget(self.panel)
        self.pages.addWidget(body)
        self.sources = sources_view
        self.pages.addWidget(sources_view)
        self.post = post_page
        self.pages.addWidget(post_page)
        pn = self.panel
        pn.changed.connect(self.request_refresh)
        pn.openPost.connect(self.openPost.emit)
        pn.variantMade.connect(self._variant_made)
        pn.addSources.connect(self._import)
        pn.addSession.connect(self._add_session)
        pn.renamed.connect(lambda pid, name: self.s.rename(pid, name))
        pn.toast.connect(self.toasted.emit)
        pn.removeTool.connect(self.removeTool.emit)
        for sig in (project.pathsChanged, project.currentChanged, project.costChanged, project.sourcesChanged,
                    project.readyChanged):
            sig.connect(self.request_refresh)
        steps.changed.connect(self._facts_arrived)
        sv = self.service
        sv.imageReady.connect(self._image_ready)
        sv.imageFailed.connect(self._image_failed)
        sv.tableReady.connect(self._table_ready)
        sv.previewReady.connect(self._preview_ready)
        sv.jobReady.connect(self._job_ready)
        sv.jobFailed.connect(self._job_failed)
        sv.statsReady.connect(self._stats_ready)
        esc = QShortcut(QKeySequence("Escape"), self)
        esc.setContext(Qt.WidgetWithChildrenShortcut)
        esc.activated.connect(self.back.emit)

    # ------------------------------------------------------------ selection (what the strip does)
    def select(self, index=None, step=None):
        s = self.s
        s.pending_change = None
        s.select(index, step)
        g = Graph(s, self.steps)
        node = g.node_of(s.path, s.sel_step) if s.sel_step not in ("sources", "final") else None
        s.scope = [q.pid for q in node.paths] if node is not None else None

    def _strip_pick(self, k: str):
        self.select(step=k)
        self.pick.emit(k)

    def _strip_workflow(self, i: int):
        self.select(index=i)
        self.workflow.emit(i)

    def _import(self):
        from .sources import import_folder
        import_folder(self.s, self.window(), service=self.service)

    def _add_session(self):
        from .sources import import_folder
        import_folder(self.s, self.window(), new_session=True, service=self.service)

    def _variant_made(self, pid, orig):
        """The variant is shown at the same block, side by side with the workflow it comes from."""
        self.cmp_pid = orig
        self.select(index=self.s.index_of(pid))
        q = self.s.view(pid)
        if q is not None:
            self.toasted.emit(_('Variant "{name}": it shares everything up to this block').format(name=q.name))

    # ------------------------------------------------------------ refresh
    def request_refresh(self, *_a):
        self._dirty = True
        if self.isVisible() and not self._pending:
            self._pending = True
            QTimer.singleShot(0, self.refresh)

    def showEvent(self, e):                                          # noqa: N802 - Qt's name
        super().showEvent(e)
        if self._dirty:
            self.request_refresh()

    def _facts_arrived(self):
        if self.pages.currentIndex() == 0 and self.panel.node is not None:
            self.panel.refresh(self.panel.node)

    def _remember(self):
        """Where the user left the picture of the tab shown (zoom, position, stretch), kept while the view is built
        again and when the tab is shown again."""
        if self._cur is None or self.view is None:
            return
        try:
            img = self.view.image()
            if img is not None and self.view.prepared() is not None:
                _p, k, spec = self._cur
                self._kept[(k, spec.key)] = (img.identity(), self.view.view_state(), self.view.stretch())
            self.show_hist = self.view.panel.isVisibleTo(self.view)
        except RuntimeError:                                   # the viewer went with a rebuild
            pass

    def refresh(self, g: Graph | None = None):
        self._pending = False
        self._dirty = False
        s = self.s
        g = g or Graph(s, self.steps)
        self.strip.build(s, g)
        if s.sel_step == "sources":
            self.pages.setCurrentIndex(1)
            if hasattr(self.sources, "refresh"):
                self.sources.refresh()
            return
        if s.sel_step == "final":
            self.pages.setCurrentIndex(2)
            return
        self.pages.setCurrentIndex(0)
        p = s.path
        node = g.node_of(p, s.sel_step)
        self.panel.refresh(node)
        self._remember()
        clear(self.head)
        self._clear_center()
        if node is None or p is None:
            return
        k = s.sel_step
        state = block_state(s, p, k)
        self.head.addWidget(label(f"{block_title(p, k)}  ·  {p.name}" if k != "post" else p.name, "h1"))
        not_done = state in ("pending", "running", "failed", "interrupted", "skipped") or \
            (state == "stale" and not self._has_previous(p, k))
        # the calibration frames can be looked at (and the wrong ones left out) before Calibration is computed: its
        # darks, flats and bias tabs show; its masters and calibrated lights say what is missing
        early = not_done and k == "calibration" and self.s.sources_loaded and state != "running"
        if not_done and not early:
            self.head.addStretch(1)
            self.pending(p, k, state)
            return
        self._early = early
        if state == "stale":
            self._previous_tag(p, k)
        if k == "selection":
            self.head.addStretch(1)
            self.selection_center(p)
            return
        vis = self.visuals(p, k)
        if not vis:
            self.head.addStretch(1)
            self.center.addWidget(label(_("No image for this block."), "dim"), 0, Qt.AlignCenter)
            return
        keys = [x.key for x in vis]
        want = self.tab_of.get(k)
        i = keys.index(want) if want in keys and not vis[keys.index(want)].spec else 0
        if len(vis) > 1:
            self.head.addSpacing(12)
            tb = QTabBar()
            tb.setDrawBase(False)
            tb.setExpanding(False)
            # every tab stays in view when the row is short of room (its name shortened), never behind scroll arrows:
            # the last tabs of Registration (Drift path, Distribution, Frames) went out of sight on a narrow window
            tb.setUsesScrollButtons(False)
            tb.setElideMode(Qt.ElideRight)
            for sp in vis:
                n = tb.addTab(_(sp.caption) + ("  ◇" if sp.spec else ""))
                if sp.spec:
                    tb.setTabEnabled(n, False)
                    tb.setTabToolTip(n, "◇ " + _("Not in the engine yet."))
                else:
                    tb.setTabToolTip(n, _(sp.caption))          # its whole name, when the row shortens it
            tb.setCurrentIndex(i)
            tb.currentChanged.connect(lambda j, vv=vis, kk=k: self._tab(kk, vv[j].key))
            self.head.addWidget(tb)
        self.head.addStretch(1)
        self._cur = (p, k, vis[i])
        self._fill()

    def _clear_center(self):
        for v in list(self.views):
            self._release(v)
        clear(self.center)
        self.views, self.view, self.chart, self.table = [], None, None, None
        self.sel_frames, self._seq_items = [], []
        self._render_spec = self._render_geom = None

    # ------------------------------------------------------------ what the walkthrough (tools/) asks of the earlier view
    @property
    def img_tab(self) -> int:
        """The index of the tab shown (the earlier view's ``img_tab``)."""
        if self._cur is None:
            return 0
        p, k, spec = self._cur
        keys = [v.key for v in self.visuals(p, k)]
        return keys.index(spec.key) if spec.key in keys else 0

    @img_tab.setter
    def img_tab(self, i: int):
        p, k = self.s.path, self.s.sel_step
        vis = self.visuals(p, k) if p is not None and k not in ("sources", "final", "selection") else []
        if vis:
            self.tab_of[k] = vis[min(max(int(i), 0), len(vis) - 1)].key

    @property
    def _slots(self) -> list:
        """The earlier view's picture slots: ``image`` is set once the viewer shows its picture (or says why not)."""
        class Slot:
            def __init__(self, v):
                self.viewer = v
                self.image = v.prepared() or (v.canvas.message or None if v._want is not None and
                                              v._failed_key == v._want else None)
                self.note = self                    # its "Loading…" note, until the picture is there
                self.stack = self

            def text(self):
                return "" if self.image is not None else "Loading"

            def currentWidget(self):  # noqa: N802 - the earlier slot's QStackedLayout
                return self.note
        return [Slot(v) for v in self.views]

    def visuals(self, p, k) -> list[VisualSpec]:
        """The tabs of a block; the ◇ ones (not in the engine yet) only in Spec mode."""
        return [v for v in self.steps.visuals(p, k) if not v.spec or prefs.spec()]

    def _has_previous(self, p, k) -> bool:
        """Whether an Outdated block has a result to show: one of its pictures' stages is done or has its previous one."""
        stages = ["register"] if k == "selection" else [v.stage for v in self.steps.visuals(p, k) if not v.spec]
        return any(self.steps.key_of(p, x, self.steps.which_for(p, x)) for x in stages)

    def _previous_tag(self, p, k):
        tag = QLabel(_("previous result"))
        tag.setStyleSheet(f"color: {WARN}; border: 1px solid {WARN}; border-radius: 9px; padding: 1px 8px;")
        tag.setToolTip(_("This block is Outdated: what it shows is the result it had."))
        self.head.addWidget(tag)
        self.head.addWidget(self._run_button(p, k))

    def _tab(self, k: str, key: str):
        self._remember()
        self.tab_of[k] = key
        self.refresh()

    # ------------------------------------------------------------ the tab shown
    def _fill(self):
        """Build the content of the tab shown (again, when the worker brings its data)."""
        if self._cur is None:
            return
        p, k, spec = self._cur
        self._clear_center()
        self._note_row = None
        note = spec.note or (CAMERA_NOTE if spec.camera and spec.kind in ("array", "removed") else "")
        if note:                                            # what this picture is, when its role is not obvious
            row = QHBoxLayout()
            row.setSpacing(14)
            text = _(note) + (" " + _(MODEL_COLOURS_NOTE) if spec.display == "model" else "")   # what its colours mean
            nt = label(text, "small", wrap=True)
            nt.setContentsMargins(2, 0, 2, 0)
            row.addWidget(nt, 1)
            self.center.addLayout(row)
            self._note_row = row
        kind = spec.kind
        if kind in ("array", "preview", "color", "gradient", "removed", "calibrated"):
            self._image_tab(p, k, spec)
        elif kind == "render":
            self._render_tab(p, k, spec)
        elif kind in ("masters", "calframes", "callights"):
            self._calibration_tab(p, spec)
        elif kind in ("regref", "regchart", "regtable"):
            self._registration_tab(p, spec)
        elif kind in ("rejection", "weights", "coverage"):
            self._stack_tab(p, spec)
        elif kind in ("chart", "colorfit"):
            self._color_tab(p, spec)
        else:
            self._message(_("No image for this block."))

    def _message(self, text: str, warn: bool = False):
        lb = label(text, "warn" if warn else "dim", wrap=True)
        lb.setAlignment(Qt.AlignCenter)
        self.center.addStretch(1)
        self.center.addWidget(lb)
        self.center.addStretch(2)

    def _missing_line(self, text: str) -> QLabel:
        """A line on what this result does not hold (amber: the user can recompute to get it)."""
        lb = label("◌  " + text, "warn", wrap=True)
        return lb

    def _card(self, title: str, lines: list[str], more: str = ""):
        """A missing diagnostic, said plainly in the middle of the tab: what, why and how to get it."""
        box = QFrame()
        box.setObjectName("misscard")
        box.setStyleSheet("QFrame#misscard { background: #1c1f26; border: 1px solid #3a4150; border-radius: 8px; }"
                          " QLabel { background: transparent; }")
        box.setFixedWidth(640)                     # a fixed width: the wrapped lines get their height
        v = QVBoxLayout(box)
        v.setContentsMargins(22, 18, 22, 18)
        v.setSpacing(8)
        v.addWidget(label(title, "h2"))
        for line in lines:
            v.addWidget(label(line, "warn", wrap=True))
        if more:
            v.addWidget(label(more, "dim", wrap=True))
        for i in range(v.count()):                 # styled now, so that the wrapped lines are measured right
            w_ = v.itemAt(i).widget()
            if w_ is not None:
                w_.ensurePolished()
        box.setFixedHeight(v.totalHeightForWidth(640) + 8)
        self.center.addStretch(1)
        self.center.addWidget(box, 0, Qt.AlignHCenter)
        self.center.addStretch(2)

    def _loading(self):
        self._message(_("Reading the results…"))

    # ------------------------------------------------------------ the worker's data
    def _need(self, name: str, keys: tuple, fn, *args):
        """What the worker read for a block (None while it reads it: the tab is built again when it comes). An
        error comes back as ``_Failed``."""
        dk = (name, keys)
        if dk in self._data:
            self._data.move_to_end(dk)
            return self._data[dk]
        asked = self._asked.get(name)
        if asked is None or asked[1] != dk:
            self._asked[name] = (self.service.request_job(name, fn, *args), dk)
        return None

    def _job_ready(self, name: str, token: int, result):
        asked = self._asked.get(name)
        if asked is None or asked[0] != token:
            return
        del self._asked[name]
        self._data[asked[1]] = result
        while len(self._data) > 16:
            self._data.popitem(last=False)
        if name == "sky":
            self._add_sky(result)
        elif name in ("lights", "registration") and self._cur is None and self.s.sel_step == "selection":
            self._build_selection()
        elif self.isVisible() and self.pages.currentIndex() == 0:
            self._remember()
            self._fill()

    def _job_failed(self, name: str, token: int, why: str):
        asked = self._asked.get(name)
        if asked is None or asked[0] != token:
            return
        self._job_ready(name, token, _Failed(why))

    def _keys(self, p, *stages) -> tuple:
        return (p.pid,) + tuple((x, self.steps.which_for(p, x), self.steps.key_of(p, x, self.steps.which_for(p, x)))
                                for x in stages)

    def _sources_sig(self) -> tuple:
        """The sessions' frame lists and the frames excluded by hand (in memory): what the frame data depend on."""
        P = self.s.p
        return (tuple((s.id, *(tuple(getattr(s, kk)) for kk in ("lights", "darks", "flats", "bias", "flat_darks")))
                      for s in P.sessions), tuple(sorted(tuple(x) for x in P.excluded)))

    # ------------------------------------------------------------ viewers
    def _post_state(self) -> tuple[bool, str]:
        try:
            ok = bool(self.can_post()) if callable(self.can_post) else False
        except Exception:  # noqa: BLE001
            ok = False
        return ok, "" if ok else _("Post does not take images yet: this comes with its image layers.")

    def _new_view(self, k: str, spec: VisualSpec | None, main: bool = True,
                  histogram: bool | None = None) -> ImageView:
        """A viewer for the tab: one kept from an earlier tab when there is one (building a viewer costs the GUI
        thread a fifth of a second), else a new one. What a tab connects goes through ``_on`` and is let go with the
        viewer (``_release``)."""
        v = self._pool.pop() if self._pool else self._make_view()
        v.setMinimumHeight(320)
        v.show_histogram(self.show_hist if histogram is None else histogram)
        ok, tip = self._post_state()
        v.set_post_enabled(ok, tip)
        kept = self._kept.get((k, spec.key)) if spec is not None else None
        if kept is not None:
            self._on(v, v.imageShown, lambda key, vv=v, kp=kept: self._restore(vv, key, kp))
        carry = self.__dict__.pop("_carry_view", None)
        if carry is not None and spec is not None and spec.backgrounds and not carry.get("fitted", True):
            once = [carry]

            def keep_view(_key, vv=v):
                try:
                    if once:
                        vv.set_view_state(once.pop())
                except RuntimeError:
                    pass
            self._on(v, v.imageShown, keep_view)
        self.views.append(v)
        if main:
            self.view = v
        v.show()
        return v

    def _make_view(self) -> ImageView:
        v = ImageView(post=True)
        v.setMinimumHeight(320)
        v.toasted.connect(self.toasted.emit)
        v.sendToPost.connect(lambda key, vv=v: self._send(vv, key))
        return v

    def _on(self, v: ImageView, signal, slot):
        """Connect a signal of a viewer for the tab shown (let go when the viewer goes back to the pool)."""
        signal.connect(slot)
        self._conns.setdefault(id(v), []).append((signal, slot))

    def _release(self, v: ImageView):
        """A viewer of the tab shown goes back to the pool: out of the layout, its tab's connections, sequence,
        comparison and overlays let go, its picture cleared."""
        for sig, slot in self._conns.pop(id(v), []):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        try:
            self.center.removeWidget(v)
            v.hide()
            if isinstance(v.parentWidget(), QSplitter):      # out of a split tab before the splitter goes
                v.setParent(self)
                v.hide()
            v.annotation_source = None
            v.set_compare_neutral(None)
            v.blink_b.setChecked(False)
            v.set_sequence([])
            v.set_compare(None)
            v.set_compare_targets([], None)
            v.set_compare_mode("split")
            v.set_same_stretch(False)
            v.canvas.overlays.clear()
            v.set_exclude_mode("on")
            v.clear()
            v._st = None
        except RuntimeError:                                  # gone with the window
            return
        if v not in self._pool:
            self._pool.append(v)

    @staticmethod
    def _restore(v: ImageView, key, kept):
        ident, state, st = kept
        try:
            if key == ident:
                if st is not None:
                    v.set_stretch(st)
                v.set_view_state(state)
        except RuntimeError:
            pass

    def _send(self, v: ImageView, key):
        """Send to Post: the image shown becomes a layer of Post (its source read in Post's worker, its key kept in
        the project: ``services/inspect.post_key``). The tab's own picture is named by its English caption, which
        Post shows translated (the layer keeps one name whatever the language); a camera-colour image shown with a
        neutral sky keeps it in Post."""
        vi = v.image()
        p = self.s.path
        if vi is None:
            return
        spec = self._cur[2] if self._cur is not None else None
        name = spec.caption if spec is not None and vi.name == _(spec.caption) else vi.name
        self.sendToPost.emit(name, si.post_source(vi, si.shown_neutral(v)),
                             si.post_key(vi, p.pid if p is not None else None))

    def _targets(self, p, k: str, spec: VisualSpec) -> list[ViewImage]:
        """What a picture is compared with: the previous steps of the workflow, then the same picture of the other
        workflows."""
        out = []
        for j, key in enumerate(spec.compare):
            if key == "calibrated_ref":
                b = self._need("stack", self._keys(p, "stack", "register"), si.stack_bundle, p.pid)
                vi = self._calibrated_vi(p, b.ref_path, b.ref_name) if isinstance(b, si.StackBundle) and \
                    b.ref_path else None
                if vi is not None:                 # the frame the stack grid is drawn on: compared on the same sky
                    vi.place = si.reference_frame_place(self.s.p, p.pid, self.steps.which_for(p, "stack"))
            else:
                sp = self.steps.picture(p, key)
                vi = self.steps.view_image(p, sp) if sp is not None else None
                if vi is not None and sp.kind == "preview":      # an engine picture: shown as it is
                    vi.stretched, vi.white, vi.units = True, 255.0, ""
            if vi is not None:
                if j < len(spec.compare_names) and spec.compare_names[j]:
                    vi.name = _(spec.compare_names[j])
                out.append(vi)
        for q in self.s.paths:
            if q.pid == p.pid or k not in chain(self.steps, q):
                continue
            vi = self.steps.view_image(q, spec, prefix=q.name)
            if vi is not None and all(vi.identity() != t.identity() for t in out) and \
                    vi.identity() != (self.steps.view_image(p, spec) or vi).identity():
                vi.info = {"workflow": q.pid}
                out.append(vi)
        return out

    def _background(self, p, k: str, spec: VisualSpec) -> VisualSpec:
        """The picture a tab with a choice of background shows: the one chosen if the workflow still has it computed, else
        the tab's own (the stack)."""
        want = self.bg_of.get(k)
        return next((b for b in self.steps.backgrounds(p) if b.key == want), spec)

    def _image_tab(self, p, k: str, spec: VisualSpec):
        shown = self._background(p, k, spec) if spec.backgrounds else spec
        if shown.kind == "render":                        # the final image, over which the solution is drawn
            self._sky_bar(spec, p, shown)
            self._render_tab(p, k, shown, spec)
            return
        vi = self.steps.view_image(p, shown)
        if vi is None:
            self._message(si.not_computed(shown.stage))
            return
        if shown.kind == "preview":
            vi.stretched, vi.white, vi.units = True, 255.0, ""
        if spec.sky:
            self._sky_bar(spec, p, shown)
        v = self._new_view(k, spec)
        if self._note_row is not None and shown.kind != "preview" and shown.display != "model":   # a model: its own colours
            if shown.camera:                              # camera colours: Neutral sky, an explicit display option
                self._note_row.addWidget(self._neutral_box(v))
            elif shown.kind == "color" and not spec.backgrounds:      # before / after: the option is the before's
                self._note_row.addWidget(self._neutral_box(v, compare=True))
        if spec.sky:
            self._on(v, v.imageShown, lambda _key, pp=p, vv=v: self._want_sky(pp, vv))
        v.set_image(vi)
        if spec.backgrounds:                              # one picture with its solution: nothing to compare with
            v.set_compare_targets([], None)
            self.center.addWidget(v, 1)
            return
        targets = self._targets(p, k, spec)
        cur, side = None, False
        if self.cmp_pid is not None:
            cur = next((i for i, t in enumerate(targets) if (t.info or {}).get("workflow") == self.cmp_pid), None)
            side = cur is not None
        if cur is None and spec.open_compare and targets:
            cur = 0
        if side:
            v.set_compare_mode("side")
        v.set_compare_targets(targets, cur)
        if spec.same_stretch:
            v.set_same_stretch(True)
        self.center.addWidget(v, 1)

    def _neutral_box(self, v: ImageView, compare: bool = False) -> QCheckBox:
        """Neutral sky as an explicit display option of a picture in the camera's colours (``compare``: of the
        before of a before / after). It follows the stretch panel's own Neutral sky."""
        c = QCheckBox(_("Neutral sky on the before (display only)") if compare else _("Neutral sky (display only)"))
        c.setToolTip(_("Each channel scaled on the screen so that the sky is grey and the stars white. The data stay "
                       "in the camera's colours."))
        if compare:
            c.toggled.connect(lambda on, vv=v: vv.set_compare_neutral(bool(on)))
            return c
        c.toggled.connect(lambda on, vv=v: vv.set_neutral(bool(on)))

        def sync(*_a, cc=c, vv=v):
            try:
                st = vv.stretch()
                if st is not None and cc.isChecked() != bool(st.neutral):
                    cc.blockSignals(True)
                    cc.setChecked(bool(st.neutral))
                    cc.blockSignals(False)
            except RuntimeError:                       # the box went with a rebuild
                pass
        self._on(v, v.stretchChanged, sync)
        self._on(v, v.imageShown, sync)
        return c

    # ------------------------------------------------------------ the sky over the stack
    def _sky_bar(self, spec: VisualSpec, p=None, shown: VisualSpec | None = None):
        row = QHBoxLayout()
        row.setSpacing(10)
        if spec.backgrounds and p is not None:            # the image behind the solution, as many as the workflow has
            row.addWidget(label(_("Background image:"), "dim"))
            row.addWidget(self._background_box(p, shown or spec))
            row.addSpacing(14)
        row.addWidget(label(_("Over the image:"), "dim"))
        for key, text, tip in (("grid", _("Sky grid"), _("The RA / Dec grid of the astrometric solution")),
                               ("stars", _("Catalogue stars"),
                                _("Gaia stars brighter than magnitude {g}, where the solution puts them")
                                .format(g=number(CATALOGUE_G_MAX, 0))),
                               ("objects", _("Name the objects"),
                                _("The known deep-sky objects of the field (Messier, NGC, IC, Sharpless and their "
                                  "common names), outlined at their size. Export… can burn them into a picture."))):
            b = QCheckBox(text)
            b.setChecked(self.sky_show.get(key, True))
            b.setToolTip(tip)
            b.toggled.connect(lambda on, kk=key: self._sky(kk, on))
            row.addWidget(b)
        self.sky_note = label("", "small")
        row.addSpacing(8)
        row.addWidget(self.sky_note)
        row.addStretch(1)
        self.center.addLayout(row)

    def _background_box(self, p, shown: VisualSpec) -> QComboBox:
        """The list of the images the solution can be drawn over (those the workflow has computed, in its order)."""
        cb = QComboBox()
        cb.setMinimumWidth(190)
        cb.setToolTip(_("The image the grid and the stars are drawn over"))
        for b in self.steps.backgrounds(p):
            cb.addItem(_(b.caption) + ("  ·  " + _("previous result") if b.previous else ""), b.key)
        cb.setCurrentIndex(max(cb.findData(shown.key), 0))
        cb.activated.connect(lambda i, c=cb, kk=self._cur[1]: self._pick_background(kk, c.itemData(i)))
        return cb

    def _pick_background(self, k: str, key: str):
        self._remember()
        # the zoom and the position stay where they were: every background but the final image is the stack grid, so
        # the view (zoom relative to the width, centre in fractions) is the same place of the sky
        try:
            self._carry_view = self.view.view_state() if self.view is not None else None
        except RuntimeError:
            self._carry_view = None
        self.bg_of[k] = key
        self.refresh()

    def _want_sky(self, p, v: ImageView):
        prep = v.prepared()
        if prep is None or self._cur is None:
            return
        w, h = prep.size()
        which = self.steps.which_for(p, "astrometry")
        keys = self._keys(p, "astrometry", "catalogs") + (w, h)
        if self._render_spec is not None:                 # the final image: a part of the stack grid, reduced
            if self._render_geom is None:
                return
            grid, origin, _size = self._render_geom
            got = self._need("sky", keys + (tuple(grid), tuple(origin)), si.final_sky_bundle, p.pid, which, grid, origin,
                             w, h)
        else:
            got = self._need("sky", keys, si.sky_bundle, p.pid, which, w, h)
        if got is not None:
            self._add_sky(got)

    def _add_sky(self, sky):
        v = self.view
        if v is None or self._cur is None or not self._cur[2].sky:
            return
        try:
            if isinstance(sky, si.SkyData):
                ov = self.steps.overlay(self._cur[1], self._cur[2], sky, self.sky_show)
                if ov is not None and not any(isinstance(o, type(ov)) for o in v.canvas.overlays):
                    v.add_overlay(ov)
                v.annotation_source = lambda s=sky: (s, dict(self.sky_show))
                n = len(sky.stars) if sky.stars is not None else 0
                text = _("{n} catalogue stars, {m} grid lines").format(n=n, m=len(sky.lines))
                tip = ""
                if sky.objects:
                    text += "  ·  " + plural(len(sky.objects), "{n} named object", "{n} named objects")
                    src = " + ".join(_(x) for x in re.sub(r"\s*\([^)]*\)", "", sky.source or "").split(" + ")
                                     if x)
                    if src:
                        text += "  ·  " + _("names: {source}").format(source=src)
                    tip = sky.source
                elif sky.why:
                    text += "  ·  " + _("No names: {why}").format(why=_(sky.why))
                self.sky_note.setText(text)
                self.sky_note.setToolTip(tip)
            else:
                self.sky_note.setText(_("No astrometric solution for this image."))
        except RuntimeError:
            pass

    def _sky(self, key: str, on: bool):
        self.sky_show[key] = on
        for v in self.views:
            v.canvas.update()
            v.canvas2.update()

    # ------------------------------------------------------------ the final image (Output)
    def _render_tab(self, p, k: str, spec: VisualSpec, tab: VisualSpec | None = None):
        """The final image ``spec``, in the tab ``tab`` (default: its own; Astrometry draws its solution over it)."""
        which = self.steps.which_for(p, spec.stage)
        key = self.steps.key_of(p, spec.stage, which)
        hit = self.service.cache.get((spec.stage, key, spec.name, which)) if key else None
        tab = tab or spec
        self._render_spec = spec
        v = self._new_view(k, tab)
        self.center.addWidget(v, 1)
        if tab.sky:
            self._on(v, v.imageShown, lambda _key, pp=p, vv=v: self._want_sky(pp, vv))
        if hit is not None:
            self._show_render(hit)
            return
        v.canvas.message = _("Rendering the final image…")
        v.canvas.update()
        self.service.use_renderer(self._render_worker())
        self._render_token = self.service.request_image(0, p.pid, spec, which)

    def _render_worker(self):
        """The project's render worker, the one Post renders with (the window makes it when asked): the final image of
        Inspect and Post then share the engines and the images. None outside a window that has one (the service makes
        its own)."""
        make = getattr(self.window(), "render_worker", None)
        try:
            return make() if callable(make) else None
        except Exception:  # noqa: BLE001 - no worker from the window: Inspect renders with its own
            return None

    def _image_ready(self, slot: int, token: int, im):
        if slot != 0 or token != self._render_token or self._cur is None or self._render_spec is None:
            return
        self._show_render(im)

    def _image_failed(self, slot: int, token: int, why: str):
        if slot != 0 or token != self._render_token or self.view is None:
            return
        self.view.canvas.message = _("This image is not available: {why}").format(why=why)
        self.view.canvas.update()

    def _show_render(self, im):
        p, k, tab = self._cur
        spec = self._render_spec or tab
        v = self.view
        if v is None:
            return
        self._render_geom = (im.grid, im.origin, im.size) if im.grid else None
        which = self.steps.which_for(p, spec.stage)
        wcs = si.final_wcs_source(self.s.p, p.pid, which, im.grid, im.origin, im.size) if im.grid else None
        # where it sits: the render's crop of the stack grid at half size (compared on the same sky)
        place = si.render_place(self.s.p, p.pid, self.steps.which_for(p, "stack"), im.grid, im.origin, im.size) \
            if im.grid else None
        vi = ViewImage(im.stretched.arr, _("Final image"), key=("inspect", "render") + tuple(im.key), wcs=wcs,
                       camera=False, stretched=True, white=255.0, units="", place=place)
        v.set_image(vi)
        if tab.backgrounds:                               # the solution's picture: nothing to compare with
            v.set_compare_targets([], None)
            return
        targets = self._targets(p, k, spec)
        v.set_compare_mode("side")
        v.set_compare_targets(targets, None)

    # ------------------------------------------------------------ two parts of a tab, the height shared by hand
    def _split(self, key: str, top: QWidget, bottom: QWidget, default=(2, 5)) -> QSplitter:
        """Two parts of a tab one above the other (the chart and the image, the chart and its table, the drift and its
        path) with a handle to share the height between them: sometimes the image, sometimes the curves. Where the user
        leaves the handle is kept for that block and tab, in this session and in the settings."""
        sp = QSplitter(Qt.Vertical)
        sp.setObjectName("tabsplit")
        sp.setChildrenCollapsible(False)
        sp.setHandleWidth(9)
        sp.setStyleSheet("QSplitter#tabsplit::handle { background: #2a2f38; margin: 3px 40%; border-radius: 1px; } "
                         "QSplitter#tabsplit::handle:hover { background: #4d5a70; }")
        sp.addWidget(top)
        sp.addWidget(bottom)
        sizes = self._splits.get(key) or (prefs.get("inspect_splits") or {}).get(key)
        if not (isinstance(sizes, list | tuple) and len(sizes) == 2 and all(isinstance(x, int | float) and x > 0
                                                                           for x in sizes)):
            sizes = default
        tot = float(sum(sizes))
        sp.setSizes([int(1000 * sizes[0] / tot), int(1000 * sizes[1] / tot)])
        sp.handle(1).setToolTip(_("Drag to share the height between the two"))
        sp.splitterMoved.connect(lambda *_a, ss=sp, kk=key: self._split_moved(kk, ss))
        self.center.addWidget(sp, 1)
        return sp

    def _split_moved(self, key: str, sp: QSplitter):
        a, b = sp.sizes()
        if a <= 0 or b <= 0:
            return
        self._splits[key] = [int(1000 * a / (a + b)), int(1000 * b / (a + b))]
        t = self.__dict__.get("_split_t")
        if t is None:
            t = self._split_t = QTimer(self)
            t.setSingleShot(True)
            t.setInterval(700)
            t.timeout.connect(self._save_splits)
        t.start()

    def _save_splits(self):
        prefs.set("inspect_splits", {**(prefs.get("inspect_splits") or {}), **self._splits})

    # ------------------------------------------------------------ Calibration
    def _calibrated_vi(self, p, path: str, name: str, info: dict | None = None, cfa: str = "") -> ViewImage | None:
        which = self.steps.which_for(p, "calibrate")
        key = self.steps.key_of(p, "calibrate", which)
        if not key or not path:
            return None
        return ViewImage(si.calframe_source(self.s.p, p.pid, path, which), _("{file}, calibrated").format(file=name),
                         key=("inspect", "calframe", key, path), cfa=cfa or self._cfa(p), camera=True,
                         info=dict(info or {}))

    def _cfa(self, p) -> str:
        data = self._data.get(("calibration", self._cal_keys(p)))
        return data.cfa if isinstance(data, si.CalibrationData) and data.cfa else "RGGB"

    def _cal_keys(self, p) -> tuple:
        return self._keys(p, "calibrate", "inventory") + (self._sources_sig(),)

    _early = False

    def _calibration_tab(self, p, spec: VisualSpec):
        if self._early and spec.kind in ("masters", "callights"):
            self.pending(p, "calibration", block_state(self.s, p, "calibration"))
            return
        which = self.steps.which_for(p, "calibrate")
        data = self._need("calibration", self._cal_keys(p), si.calibration_data, p.pid, which, self.service.headers)
        if data is None:
            self._loading()
            return
        if isinstance(data, _Failed):
            self._message(_("The calibration cannot be read: {why}").format(why=data.why), warn=True)
            return
        # the engine's word on each frame: it reads every raw frame the first time, so it is asked when the frames
        # are looked at (the tab shows without it first); the masters' tab shows its notes once it is there
        if spec.kind == "calframes":
            rep = self._need("calreport", self._cal_keys(p), si.calibration_report, p.pid, which)
        else:
            rep = self._data.get(("calreport", self._cal_keys(p)))
        self._report = rep if isinstance(rep, si.CalibrationReport) else None
        if si.diag("calibration_frames") is None or isinstance(rep, _Failed):
            self._report_state = "none"
        else:
            self._report_state = "ok" if self._report is not None else "wait"
        if spec.kind == "masters":
            self._masters(p, spec, data)
        elif spec.kind == "calframes":
            self._calframes(p, spec, data)
        else:
            self._callights(p, spec, data)

    _report = None
    _report_state = "none"

    def _masters(self, p, spec: VisualSpec, data: si.CalibrationData):
        if not data.masters:
            self._message(si.not_computed("calibrate"))
            return
        which = self.steps.which_for(p, "calibrate")
        key = self.steps.key_of(p, "calibrate", which)
        first = {kind: (items[0] if items else None) for kind, items in data.groups.items()}
        frames = []
        for mk, caption, st in data.masters:
            kind = {"master_dark": "darks", "master_flat": "flats", "master_bias": "bias",
                    "master_flat_dark": "flat_darks"}.get(mk)
            n = len([x for x in data.groups.get(kind, []) if not x.excluded]) if kind else 0
            f0 = first.get(kind) if kind else None
            norm = mk in ("master_flat", "badmask")
            extra = []
            if kind:
                extra.append((_("Frames"), str(n)))
            if mk == "badmask":
                if st.get("bad_pct") is not None:
                    extra.append((_("Defective pixels"), number(st["bad_pct"], 2) + " %"))
            elif st:
                unit = "" if norm else " ADU"
                d = 3 if norm else 1
                extra += [(_("Median"), number(st["median"], d) + unit),
                          (_("Noise σ"), number(st["noise"], d + 1) + unit)]
                if mk == "master_flat":
                    extra.append((_("Darkest corner"), number(100 * st["min"] / max(st["median"], 1e-9), 0) + " %"))
            info = {"compact": True, "extra": extra}
            if f0 is not None:
                info.update(exposure=f0.exposure, iso=f0.iso, temperature=f0.temperature)
            frames.append(ViewImage(si.array_source(self.s.p, p.pid, "calibrate", mk, which), _(caption),
                                    key=("inspect", "master", key, mk), cfa="" if mk == "badmask" else data.cfa,
                                    camera=mk != "badmask", units="norm" if norm else "adu",
                                    white=1.0 if norm else None, info=info, stretched=mk == "badmask",
                                    kind=kind or "mask"))       # each master the automatic stretch of its kind
        v = self._new_view("calibration", spec)
        v.set_exclude_mode("off")
        v.set_sequence(frames, 0)
        self._notes()
        self.center.addWidget(v, 1)

    def _notes(self):
        """The engine's remarks on the calibration (in the interface's language when it has them)."""
        rep = self._report
        if rep is None:
            return
        lang = language()
        for n in rep.notes[:4]:
            if lang == "en" or texts.has_translation(n, lang):
                self.center.addWidget(label(texts.translate(n, lang), "small", wrap=True))

    def _engine_frame(self, it):
        """The engine's CalFrame of a frame (its statistics and quality flag), or None."""
        rep = self._report
        if rep is None:
            return None
        return rep.frames.get(os.path.normcase(os.path.normpath(it.path)))

    def _frame_extra(self, it) -> list[tuple[str, str]]:
        """The numbers of a calibration frame for the info band: the engine's statistics when it gives them, else
        the level and noise the worker measured on the raw."""
        cf = self._engine_frame(it)
        st = dict(getattr(cf, "stats", None) or {}) if cf is not None else {}
        extra = []
        if st:
            def num(k):
                v = st.get(k)
                return float(v) if isinstance(v, int | float) and np.isfinite(v) else None
            if num("median") is not None:
                extra.append((_("Median level"), number(num("median"), 0) + " ADU"))
            if num("noise") is not None:
                extra.append((_("Noise σ"), number(num("noise"), 1) + " ADU"))
            if it.kind == "flats":
                if num("level_pct") is not None:
                    extra.append((_("Exposure level"), number(num("level_pct"), 0) + " %"))
                if num("corner_pct") is not None:
                    extra.append((_("Corners"), number(num("corner_pct"), 0) + " %"))
            else:
                if num("hot_pixels") is not None:
                    extra.append((_("Hot pixels"), number(num("hot_pixels"), 0)))
                if num("gradient") is not None:
                    extra.append((_("Centre − edge"), number(num("gradient"), 1) + " ADU"))
            if (num("saturated_pct") or 0) >= 0.005:
                extra.append((_("Saturated"), number(num("saturated_pct"), 2) + " %"))
            return extra
        sw = self.service.stats.get(it.path)
        if sw:
            extra.append((_("Median level"), number(sw["level"] + (sw.get("black") or 0), 0) + " ADU"))
            extra.append((_("Noise σ"), number(sw["noise"], 1) + " ADU"))
            if it.kind == "flats":
                extra.append((_("Exposure level"), number(sw["pct"], 0) + " %"))
        else:
            extra.append((_("Median level"), "…"))
        return extra

    def _flag_level(self, it) -> str:
        """The engine's flag of a calibration frame: "bad" (not a frame of its kind: a dark with stars), "warn" (unlike
        the others of its kind), or "" (nothing to say, or not measured yet)."""
        cf = self._engine_frame(it)
        lvl = str(getattr(cf, "flag", "ok") or "ok") if cf is not None else "ok"
        return "" if lvl == "ok" else ("bad" if lvl == "bad" else "warn")

    def _cal_info(self, it) -> dict:
        info = {"time": it.time, "exposure": it.exposure, "iso": it.iso, "temperature": it.temperature,
                "excluded": it.excluded, "compact": True, "extra": self._frame_extra(it)}
        lvl = self._flag_level(it)
        if lvl:
            info["flag"] = _("Suspicious")
            info["flag_level"] = lvl
            info["flag_reason"] = frame_reason(self._engine_frame(it), it.kind, language())
        return info

    def _calframes(self, p, spec: VisualSpec, data: si.CalibrationData):
        items = data.groups.get(spec.name) or []
        if not items:
            self._no_frames(spec.name)
            return
        flagged = [it for it in items if self._flag_level(it)]
        mode = self._cal_filter.get(spec.key, "all")
        if not flagged:
            mode = "all"
        shown = flagged if mode == "flagged" else items
        frames = []
        for it in shown:
            vi = ViewImage(it.path, it.name, cfa=it.cfa or data.cfa, camera=True, white=it.white,
                           info=self._cal_info(it), kind=spec.name)
            vi.thumb = partial(quick_thumbnail, ViewImage(it.path, cfa=it.cfa or data.cfa, camera=True,
                                                          kind=spec.name))
            frames.append(vi)
        v = self._new_view("calibration", spec)
        self._on(v, v.excludeToggled, lambda i, ex, its=shown, grp=items: self._cal_exclude(its, i, ex, grp))
        self._seq_items = shown
        self.sel_frames = frames
        n_out = sum(1 for x in items if x.excluded)
        n_bad = sum(1 for x in flagged if self._flag_level(x) == "bad")
        line = _("{n} frames, {x} left out by hand. Exclude (X) leaves the frame shown out of the master: Calibration "
                 "is then computed again.").format(n=len(items), x=n_out)
        if self._report_state == "wait":
            line += "  " + _("Measuring every frame for its quality flag…")
        elif self._report_state == "ok":
            if not flagged:
                line += "  " + _("No frame stands out from the others.")
            elif n_bad:
                line += "  " + plural(len(flagged), "{n} suspicious frame: red, not a frame of this kind at all; "
                                                    "amber, unlike the others.",
                                      "{n} suspicious frames: red, not frames of this kind at all; amber, unlike the "
                                      "others.")
            else:
                line += "  " + plural(len(flagged), "{n} suspicious frame (amber): unlike the others of its kind.",
                                      "{n} suspicious frames (amber): unlike the others of their kind.")
        else:
            line += "  " + _("Quality flags: {why}").format(why=si.not_yet("calibrate"))
        self.center.addWidget(label(line, "small", wrap=True))
        if flagged:
            self.center.addLayout(self._flag_row(spec, items, flagged, mode))
        self.center.addWidget(v, 1)
        v.set_sequence(frames, min(self._seq_index.get((spec.key, mode), 0), len(frames) - 1))
        self._on(v, v.frameChanged, lambda i, kk=(spec.key, mode): self._seq_index.__setitem__(kk, i))
        if self._report is None:
            self.service.request_stats_batch([x.path for x in items])

    def _flag_row(self, spec: VisualSpec, items, flagged, mode: str) -> QHBoxLayout:
        """All the frames or the suspicious ones only, and Exclude the suspicious frames in one click."""
        row = QHBoxLayout()
        row.setSpacing(0)
        grp = QButtonGroup(self)
        for key, text in (("all", _("All frames ({n})").format(n=len(items))),
                          ("flagged", _("Suspicious ({n})").format(n=len(flagged)))):
            b = QPushButton(text)
            b.setObjectName("face")
            b.setCheckable(True)
            b.setChecked(key == mode)
            b.clicked.connect(lambda _x=False, kk=key, sk=spec.key: self._set_cal_filter(sk, kk))
            grp.addButton(b)
            row.addWidget(b)
        row.addSpacing(14)
        todo = [it for it in flagged if not it.excluded]
        ex = QPushButton(_("Exclude the suspicious frames ({n})").format(n=len(todo)))
        ex.setEnabled(bool(todo))
        ex.setToolTip(_("Leaves every suspicious frame out of the master in one go: Calibration is then computed "
                        "again. Include takes a frame back; a folder left empty comes back with Add a folder."))
        ex.clicked.connect(lambda _x=False, its=items: self.exclude_suspicious(its))
        row.addWidget(ex)
        row.addStretch(1)
        return row

    def _set_cal_filter(self, key: str, mode: str):
        self._cal_filter[key] = mode
        self._remember()
        self._fill()

    def exclude_suspicious(self, items) -> dict:
        """Every suspicious frame of the tab left out of its master, in one change. A session left with no
        frame of the kind (all its "darks" were frames of the sky) is said, and Add a folder is offered."""
        todo = [it for it in items if self._flag_level(it) and not it.excluded and it.session is not None]
        if not todo:
            return {"done": 0, "emptied": []}
        kind = todo[0].kind
        res = self.s.leave_out_calibration([(it.session, it.kind, it.path) for it in todo],
                                           [(g.session, g.kind, g.path, g.excluded) for g in items
                                            if g.session is not None])
        msg = plural(res["done"], "{n} suspicious frame left out of the master: Calibration is to compute again.",
                     "{n} suspicious frames left out of the master: Calibration is to compute again.")
        if res["emptied"]:
            msg += " " + no_frames_text(kind)
        self._cal_filter.clear()
        self.toasted.emit(msg)
        return res

    def _no_frames(self, kind: str):
        """No frame of a kind in the project (none was given, or all were left out): what Calibration does then, and
        Add a folder."""
        title, line = NO_FRAMES.get(kind, ("No frames", "No frames of this kind."))
        box = QFrame()
        box.setObjectName("misscard")
        box.setStyleSheet("QFrame#misscard { background: #1c1f26; border: 1px solid #3a4150; border-radius: 8px; }"
                          " QLabel { background: transparent; }")
        box.setFixedWidth(640)
        v = QVBoxLayout(box)
        v.setContentsMargins(22, 18, 22, 18)
        v.setSpacing(10)
        v.addWidget(label(_(title), "h2"))
        v.addWidget(label(_(line), "", wrap=True))
        b = QPushButton(_(ADD_FOLDER.get(kind, "Add a folder…")))
        b.setObjectName("primary")
        b.clicked.connect(lambda _x=False, kk=kind: self.add_calibration_folder(kk))
        v.addWidget(b, 0, Qt.AlignLeft)
        for i in range(v.count()):                 # styled now, so that the wrapped lines are measured right
            w_ = v.itemAt(i).widget()
            if w_ is not None:
                w_.ensurePolished()
        box.setFixedHeight(v.totalHeightForWidth(640) + 8)
        self.center.addStretch(1)
        self.center.addWidget(box, 0, Qt.AlignHCenter)
        self.center.addStretch(2)

    def add_calibration_folder(self, kind: str, folder: str | None = None) -> bool:
        """Add a folder of darks (flats, bias, dark flats) to the sessions that have none of that kind (to every
        session when all have some): Calibration is then computed again."""
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self.window(), _(ADD_FOLDER.get(kind, "Add a folder…")),
                                                      prefs.get("workspace"))
            if not folder:
                return False
        P = self.s.p
        targets = [x.id for x in P.sessions if not getattr(x, kind)] or [x.id for x in P.sessions]
        done = [sid for sid in targets if self.s.add_calibration_folder(sid, kind, folder)]
        if done:
            self.toasted.emit(_("Folder added: Calibration is to compute again."))
        return bool(done)

    @property
    def _seq_index(self) -> dict:
        d = self.__dict__.setdefault("_seq_pos", {})
        return d

    def _cal_exclude(self, items, i: int, excluded: bool, group=None):
        it = items[i]
        if not si.set_calibration_excluded(self.s, it, excluded, group if group is not None else items):
            if self.view is not None:
                self.view.set_excluded(i, not excluded)
            self.toasted.emit(_("The last frame of a folder cannot be left out."))
            return
        self.toasted.emit((_("{file} left out of the master: Calibration is to compute again.") if excluded else
                           _("{file} taken back into the master.")).format(file=it.name))

    def _stats_ready(self, path: str, _st: dict):
        v = self.view
        if v is None or not self._seq_items or self._cur is None or self._cur[2].kind != "calframes":
            return
        for i, it in enumerate(self._seq_items):
            if it.path == path:
                try:
                    v.set_frame_info(i, self._cal_info(it))
                except RuntimeError:
                    pass
                return

    def _light_info(self, it, p90: float) -> dict:
        info = {"time": it.time, "exposure": it.exposure, "iso": it.iso, "temperature": it.temperature,
                "fwhm": it.fwhm, "roundness": it.roundness, "weight": it.weight,
                "transparency": (it.transparency / p90) if it.transparency is not None and p90 else None,
                "excluded": bool(it.excluded), "reason": _(it.status) if it.status and not it.excluded else ""}
        if it.level is not None:
            info["extra"] = [(_("Sky level"), number(it.level, 0) + " ADU")]
        return info

    @staticmethod
    def _p90(items) -> float:
        tr = [x.transparency for x in items if x.transparency is not None and np.isfinite(x.transparency)]
        return float(np.percentile(tr, 90)) if tr else 1.0

    def _lights(self, p):
        return self._need("lights", self._keys(p, "register", "inventory", "stack") + (self._sources_sig(),),
                          si.lights_of, p.pid, self.service.headers)

    def _callights(self, p, spec: VisualSpec, data: si.CalibrationData):
        items = self._lights(p)
        if items is None:
            self._loading()
            return
        if isinstance(items, _Failed) or not items:
            self._message(_("No light frames yet"))
            return
        p90 = self._p90(items)
        frames = []
        for it in items:
            vi = self._calibrated_vi(p, it.path, it.name, self._light_info(it, p90), it.cfa or data.cfa)
            if vi is None:
                self._message(si.not_computed("calibrate"))
                return
            vi.thumb = si.light_thumb(self.s.p, it.path)
            frames.append(vi)
        if not data.cache:
            self.center.addWidget(label(_("The calibrated frames are not on disk: each one is calibrated again here "
                                          "from the masters of this result, with the engine's formula."),
                                        "small", wrap=True))
        v = self._new_view("calibration", spec)
        self._seq_items = items
        self.sel_frames = frames
        self._on(v, v.frameChanged, lambda i, vv=v, its=items: self._raw_target(vv, its, i))
        self._on(v, v.frameChanged, lambda i, kk=spec.key: self._seq_index.__setitem__(kk, i))
        self._on(v, v.excludeToggled, lambda i, ex, its=items: self._light_exclude(its, i, ex))
        self.center.addWidget(v, 1)
        start = min(self._seq_index.get(spec.key, 0), len(frames) - 1)
        v.set_sequence(frames, start)
        self._raw_target(v, items, start)

    def _raw_target(self, v: ImageView, items, i: int):
        """The calibrated light is compared with the same frame, raw."""
        if not (0 <= i < len(items)):
            return
        it = items[i]
        raw = ViewImage(it.path, _("{file}, raw").format(file=it.name), cfa=it.cfa, camera=True, white=it.white)
        cur = v._cmp_i
        v.set_compare_targets([raw], cur)

    def _light_exclude(self, items, i: int, excluded: bool):
        it = items[i]
        if it.session is None:
            return
        it.excluded = excluded
        self.s.set_excluded([(it.session, it.name)], excluded)       # saved by the model, as any edit

    # ------------------------------------------------------------ Registration
    def _registration_tab(self, p, spec: VisualSpec):
        b = self._need("registration", self._keys(p, "register", "calibrate", "astrometry", "inventory"),
                       si.registration_bundle, p.pid)
        if b is None:
            self._loading()
            return
        if isinstance(b, _Failed):
            self._message(si.not_yet("register") if "NotDone" in b.why else
                          _("The registration cannot be read: {why}").format(why=b.why), warn="NotDone" not in b.why)
            return
        d = b.data
        if spec.kind == "regref":
            vi = self._calibrated_vi(p, b.ref_path, b.ref_name)
            if vi is None:
                self._message(si.not_computed("calibrate"))
                return
            v = self._new_view("alignement", spec)
            v.set_image(vi)
            raw = ViewImage(b.ref_path, _("{file}, raw").format(file=b.ref_name), camera=True)
            v.set_compare_targets([raw], None)
            if 0 <= d.ref < d.n:
                ns = d.nstars[d.ref]
                self.center.addWidget(label(_("{file} · FWHM {fwhm} px · {n} stars detected").format(
                    file=b.ref_name, fwhm=number(d.fwhm[d.ref], 2), n=int(ns) if np.isfinite(ns) else "–"), "small"))
            self.center.addWidget(v, 1)
            return
        if spec.kind == "regchart":
            if spec.name in ("drift", "track") and not np.isfinite(d.dx).any():
                self._message(_("The offsets of the frames cannot be computed: the size of the sensor is not known "
                                "(no inventory, calibration or stack result gives it). Recompute Registration."),
                              warn=True)
                return
            if spec.name == "drift":
                # the drift against time and, under it, its path (x against y, coloured by time): the path of the
                # session is seen with the drift, not only in its own tab
                self._split("alignement/drift", DriftChart(d), TrackChart(d), (1, 1))
                return
            make = {"quality": QualityChart, "track": TrackChart, "distribution": DistributionChart}[spec.name]
            ch = make(d)
            self.center.addWidget(ch, 1)
            if spec.name == "quality" and d.roundness is None:
                for line in si.missing_of(b.table, "roundness") or [_("Roundness: {why}").format(
                        why=si.not_yet("register"))]:
                    self.center.addWidget(self._missing_line(line))
            return
        self._registration_table(b)

    def _registration_table(self, b: si.RegistrationBundle):
        d = b.data
        if b.table:
            heads = [_("Frame"), _("Time"), _("x offset (px)"), _("y offset (px)"), _("Rotation (°)"), _("Scale"),
                     _("Stars matched"), _("Stars detected"), _("FWHM (px)"), _("Roundness"), _("Residual (px)"),
                     _("Transparency (%)"), _("Aligned")]
            rows, red = [], []
            for i, r in enumerate(b.table):
                tr = d.transp[i] if i < d.n else np.nan
                rows.append([r.name + ("  ★" if getattr(r, "reference", False) else ""), (r.time or "")[11:19],
                             r.dx, r.dy, r.rotation, r.scale, r.nmatch if r.used else None, r.nstars or None, r.fwhm,
                             r.roundness, getattr(r, "rms", None), tr, _("yes") if r.used else _("no")])
                red.append(not r.used)
            t = make_table(heads, rows, {2: 2, 3: 2, 4: 3, 5: 5, 6: 0, 7: 0, 8: 2, 9: 3, 10: 3, 11: 0}, red)
            self.center.addWidget(label(_("★ the reference frame."), "small"))
            for line in si.missing_of(b.table):
                self.center.addWidget(self._missing_line(line))
        else:
            heads = [_("Frame"), _("Time"), _("x offset (px)"), _("y offset (px)"), _("Stars matched"),
                     _("Stars detected"), _("FWHM (px)"), _("Residual (px)"), _("Transparency (%)"), _("Aligned")]
            rows, red = [], []
            for i in range(d.n):
                rows.append([d.names[i] + ("  ★" if i == d.ref else ""), (d.iso[i] or "")[11:19], d.dx[i], d.dy[i],
                             d.nmatch[i], d.nstars[i], d.fwhm[i], d.rms[i], d.transp[i],
                             _("yes") if d.aligned[i] else _("no")])
                red.append(not d.aligned[i])
            t = make_table(heads, rows, {2: 2, 3: 2, 4: 0, 5: 0, 6: 2, 7: 3, 8: 0}, red)
            self.center.addWidget(label(_("★ the reference frame. Rotation, scale and roundness: {why}").format(
                why=si.not_yet("register")), "small", wrap=True))
        self.center.addWidget(t, 1)

    # ------------------------------------------------------------ Stacking
    def _stack_tab(self, p, spec: VisualSpec):
        b = self._need("stack", self._keys(p, "stack", "register"), si.stack_bundle, p.pid)
        if b is None:
            self._loading()
            return
        if isinstance(b, _Failed):
            self._message(_("The stack cannot be read: {why}").format(why=b.why), warn=True)
            return
        if spec.kind == "rejection":
            self._rejection(p, spec, b)
            return
        if spec.kind == "coverage":
            self._coverage(p, spec, b)
            return
        rows = weight_rows(b)
        if not rows:
            self._message(si.not_yet("stack"))
            return
        n_left = sum(1 for r in rows if r["status"])
        head = _("{kept} of {n} frames kept").format(kept=len(rows) - n_left, n=len(rows))
        if n_left:
            head += " · " + _("{n} left out, each with its reason").format(n=n_left)
        self.center.addWidget(label(head, "small", wrap=True))
        ch = WeightsChart(rows)
        ch.setMinimumHeight(160)
        norm = any(r["scale"] is not None for r in rows)
        # the state (kept, or why left out) right after the frame's name and time: seen without scrolling
        heads = [_("Frame"), _("Time"), _("State"), _("Weight"), _("FWHM (px)"), _("Transparency (%)")]
        if norm:
            heads += [_("Scale R"), _("Scale G"), _("Scale B"), _("Offset R (ADU)"), _("Offset G (ADU)"),
                      _("Offset B (ADU)")]
        tr = [r["transparency"] for r in rows if r["transparency"] is not None and np.isfinite(r["transparency"])]
        p90 = float(np.percentile(tr, 90)) if tr else 1.0
        data_rows, red = [], []
        for r in rows:
            sc, off = per_channel(r["scale"]), per_channel(r["offset"])
            t_ = r["transparency"]
            row = [r["name"], (r["time"] or "")[11:19], r.get("why") or _(r["status"]) if r["status"] else _("kept"),
                   r["weight"], r["fwhm"], 100 * t_ / p90 if t_ is not None and np.isfinite(t_) else None]
            if norm:
                row += [sc[0], sc[1], sc[2], off[0], off[1], off[2]]
            data_rows.append(row)
            red.append(bool(r["status"]))
        digits = {3: 3, 4: 2, 5: 0} | ({6: 4, 7: 4, 8: 4, 9: 1, 10: 1, 11: 1} if norm else {})
        t = make_table(heads, data_rows, digits, red)
        t.setMinimumHeight(120)
        self._split("empilement/weights", ch, t, (3, 2))
        d = b.diag
        bits = []
        if d is None:
            bits.append(_("Normalisation: {why}").format(why=si.not_yet("stack")))
        if d is not None and getattr(d, "noise", None):
            bits.append(_("Noise of the stack: {v}").format(v=_noise_text(d.noise)))
        if d is not None and getattr(d, "rejected_on_grid_pct", None) is not None:
            bits.append(_("Rejected samples on the stack: {v} %").format(v=number(float(d.rejected_on_grid_pct), 2)))
        if bits:
            self.center.addWidget(label("   ·   ".join(bits), "small", wrap=True))
        if d is not None and not norm:
            for line in si.missing_of(d, "normalisation"):
                self.center.addWidget(self._missing_line(line))

    def _rejection(self, p, spec: VisualSpec, b: si.StackBundle):
        d = b.diag
        if d is None or (getattr(d, "rejection_low", None) is None and getattr(d, "rejection_high", None) is None):
            self._card(_("Rejection map"), si.missing_of(d, "rejection") or [si.not_yet("stack")],
                       _("Once Stacking is computed again, the map shows where pixels were rejected below (dark: cold "
                         "pixels, a dust shadow) and above (bright: satellite and plane trails, cosmic rays, hot "
                         "pixels) the other frames."))
            return
        key = self.steps.key_of(p, "stack", self.steps.which_for(p, "stack"))
        imgs = []
        for which, cap in (("rejection_low", _("Rejected low")), ("rejection_high", _("Rejected high"))):
            src = getattr(d, which, None)
            if src is not None:
                imgs.append(ViewImage(src, cap, key=("inspect", which, key), camera=False, units="count"))
        row = QHBoxLayout()
        row.setSpacing(0)
        v = self._new_view("empilement", spec)
        grp = QButtonGroup(self)
        for i, vi in enumerate(imgs):
            bt = QPushButton(vi.name)
            bt.setObjectName("face")
            bt.setCheckable(True)
            bt.setChecked(i == len(imgs) - 1)
            bt.clicked.connect(lambda _x=False, img=vi, vv=v: vv.set_image(img, keep_view=True))
            grp.addButton(bt)
            row.addWidget(bt)
        f = int(getattr(d, "rejection_factor", 1) or 1)
        row.addSpacing(12)
        row.addWidget(label(_("Rejected samples per pixel, on the stack grid reduced {f} times.").format(f=f) if f > 1
                            else _("Rejected samples per pixel of the stack."), "small"))
        row.addStretch(1)
        self.center.addLayout(row)
        rate = rejection_rate_text(d)
        if rate:
            lb = label(rate, "small", wrap=True)
            lb.setTextFormat(Qt.RichText)
            self.center.addWidget(lb)
        v.set_image(imgs[-1])
        if len(imgs) == 2:
            v.set_compare_targets([imgs[0]], None)
        self.center.addWidget(v, 1)

    def _coverage(self, p, spec: VisualSpec, b: si.StackBundle):
        """How evenly the frames cover the stack, colour by colour (the drizzle's red and blue first: one pixel in four
        of the sensor), with the variation of each colour against DrizzlePac's 20 %; a classic stack, one map."""
        import dataclasses
        d = b.diag
        method = (getattr(d, "switches", None) or {}).get("method") or "drizzle"
        drizzle = method == "drizzle"
        chans = ((0, "Red"), (2, "Blue"), (1, "Green")) if drizzle else ((1, "Coverage"),)
        imgs = []
        for c, cap in chans:                      # the picture's name is translated where it is shown
            vi = self.steps.view_image(p, dataclasses.replace(spec, kind="array", channel=c, shown=cap))
            if vi is not None:
                imgs.append((vi, _(cap)))
        if not imgs:
            self._message(si.not_yet("stack"))
            return
        v = self._new_view("empilement", spec)
        if len(imgs) > 1:
            row = QHBoxLayout()
            row.setSpacing(0)
            grp = QButtonGroup(self)
            for i, (vi, cap) in enumerate(imgs):
                bt = QPushButton(cap)
                bt.setObjectName("face")
                bt.setCheckable(True)
                bt.setChecked(i == 0)
                bt.clicked.connect(lambda _x=False, img=vi, vv=v: vv.set_image(img, keep_view=True))
                grp.addButton(bt)
                row.addWidget(bt)
            row.addStretch(1)
            self.center.addLayout(row)
        var = coverage_text(d, drizzle)
        if var:
            lb = label(var, "small", wrap=True)
            lb.setTextFormat(Qt.RichText)
            self.center.addWidget(lb)
        v.set_image(imgs[0][0])
        if len(imgs) > 1:
            v.set_compare_targets([vi for vi, _c in imgs[1:]], None)
        self.center.addWidget(v, 1)

    # ------------------------------------------------------------ Color
    def _color_tab(self, p, spec: VisualSpec):
        which = self.steps.which_for(p, "color")
        if spec.kind == "chart":
            b = self._need("color", self._keys(p, "color"), si.color_bundle, p.pid, which)
            if b is None:
                self._loading()
                return
            if isinstance(b, _Failed):
                self._message(_("The colour result cannot be read: {why}").format(why=b.why), warn=True)
                return
            ch = color_tabs(b.curves.lam, b.curves.S)[int(spec.name or 0)][1]
            self.center.addWidget(ch, 1)
            return
        if si.diag("color_fit") is None:
            self._message(si.not_yet("color"))
            return
        fit = self._need("colorfit", self._keys(p, "color", "catalogs", "stack"), si.color_fit_of, p.pid, which)
        if fit is None:
            self._message(_("Measuring the catalogue stars on the stack…"))
            return
        if isinstance(fit, _Failed) or not getattr(fit, "stars", None):
            self._message(si.not_yet("color"))
            return
        ch = ColorFitChart(fit)
        ch.setMinimumHeight(180)
        stars = list(fit.stars)
        heads = [_("x (px)"), _("y (px)"), "G", _("Gaia BP − RP"), _("Camera colour"), _("Residual"), _("Used"),
                 _("Reason")]
        rows, red = [], []
        for s in stars[:4000]:
            cat, meas, res, g = star_colour(s)
            why = str(getattr(s, "reason", "") or "")
            rows.append([s.x, s.y, g, cat, meas, res, _("yes") if s.used else _("no"), _(why) if why else ""])
            red.append(not s.used)
        t = make_table(heads, rows, {0: 0, 1: 0, 2: 2, 3: 3, 4: 3, 5: 4}, red)
        t.setMinimumHeight(120)
        self._split("couleur/fit", ch, t, (3, 2))
        st = dict(getattr(fit, "stats", None) or {})
        bits = []
        if st.get("used") is not None and st.get("stars") is not None:
            bits.append(_("{u} of {n} stars used").format(u=st["used"], n=st["stars"]))
        if st.get("chromaticity_residual") is not None:
            bits.append(_("median residual {v}").format(v=number(float(st["chromaticity_residual"]), 4)))
        if st.get("residual_p90") is not None:
            bits.append(_("90 % of the stars under {v}").format(v=number(float(st["residual_p90"]), 4)))
        if bits:
            self.center.addWidget(label("   ·   ".join(bits), "small"))

    # ------------------------------------------------------------ a block not computed
    def _run_button(self, p, k) -> QPushButton:
        s = self.s
        plan = s.plan_to_block(p, k)
        rb = QPushButton(_("Run to here") + "  ·  " + approx(plan.seconds, plan.measured))
        titles = []
        for j in plan.jobs:
            key = ENGINE_BLOCK.get(j.block)
            t = block_title(p, key) if key else j.block
            if t not in titles:
                titles.append(t)
        rb.setToolTip(", ".join(titles))
        rb.setEnabled(s.sources_loaded and bool(plan.jobs))
        rb.clicked.connect(self.runHere.emit)
        return rb

    def pending(self, p, k, state="pending"):
        s = self.s
        plan = s.plan_to_block(p, k)
        box = QVBoxLayout()
        box.setAlignment(Qt.AlignCenter)
        if state in ("pending", "stale"):
            box.addWidget(label(_("Not computed yet"), "h2"), 0, Qt.AlignHCenter)
        else:
            box.addWidget(label(_(STATE_WORD[state]), "h2"), 0, Qt.AlignHCenter)
            st = next((s.state_info(p, x) for x in stages_of(s, p, k) if s.state(p, x) == state), None)
            if st is not None and st.reason:
                box.addWidget(label(_(st.reason), "dim"), 0, Qt.AlignHCenter)
        box.addSpacing(8)
        rb = self._run_button(p, k)
        rb.setObjectName("primary")
        box.addWidget(rb, 0, Qt.AlignHCenter)
        if prefs.spec():
            pv = QPushButton(_("Preview on a region") + "  ·  " + approx(max(plan.seconds / 12, 30), False) + "  ◇")
            pv.setToolTip("◇ " + _("Not in the engine yet."))
            pv.setEnabled(False)
            box.addWidget(pv, 0, Qt.AlignHCenter)
        if not s.sources_loaded:
            box.addWidget(label(_("No frames yet."), "warn"), 0, Qt.AlignHCenter)
        self.center.addStretch(1)
        self.center.addLayout(box)
        self.center.addStretch(2)

    # ------------------------------------------------------------ Selection: the frames chart and the frames
    def _thresholds(self) -> tuple[float, float]:
        """The thresholds the chart shows: the settings', or the ones just dragged while Replace / Try as variant is asked."""
        s = self.s
        t, f = s.thresholds_now()
        ch = s.pending_change
        if ch and ch.get("kind") == "params" and ch.get("step") == "selection":
            for _stage, name, v in ch["items"]:
                if name == "min_transparency":
                    t = 100 * float(v)
                elif name == "max_fwhm_ratio":
                    f = float(v) * (s.fwhm_median() or 1.0)
        return t, f

    def selection_center(self, p):
        self._cur = None
        self._loading()
        self._table_token = self.service.request_table(p.pid)
        self._table_wid = p.pid
        self._table_got = None
        self._lights(p)

    def _table_ready(self, wid: int, token: int, table):
        if self.pages.currentIndex() != 0 or self.s.sel_step != "selection" or wid != getattr(self, "_table_wid", None) \
                or token != self._table_token:
            return
        self._table_got = table
        self._build_selection()

    def _build_selection(self):
        table = getattr(self, "_table_got", None)
        p = self.s.path
        if table is None or p is None or self.s.sel_step != "selection":
            return
        if isinstance(table, str):
            self._clear_center()
            self._message(_("The frames chart appears once registration is computed."))
            return
        items = self._lights(p)
        if items is None:
            return                                          # the frames' headers come next
        self._clear_center()
        s = self.s
        self.table = table
        t, f = self._thresholds()
        ch = FramesChart(table.rows, t, f, thresholds=s.thresholds_active())
        ch.sugg = s.suggested_thresholds()
        ch.current = self.picked if self.picked is not None and self.picked < ch.n else None
        ch.setMinimumHeight(140)
        ch.changed.connect(lambda c=ch: self._chart_changed(c))
        ch.picked.connect(self._pick)
        ch.setToolTip(_("Thresholds: drag · frame: click") if ch.thresholds else _("Frame: click"))
        self.chart = ch
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(4)
        track = self._selection_track(p, table)
        if track is not None:
            # the tracking of the night (x against y, Lissajous) beside the frames, the left-out ones circled
            row = QSplitter(Qt.Horizontal)
            row.addWidget(ch)
            row.addWidget(track)
            row.setStretchFactor(0, 1)
            row.setStretchFactor(1, 1)
            tl.addWidget(row, 1)
        else:
            tl.addWidget(ch, 1)
        self.kept_label = label("", "dim")
        tl.addWidget(self.kept_label)
        self._kept_n = (table.kept, table.total)
        self._kept_text()
        if ch.thresholds and (abs(t - 100 * table.min_transparency) > 1e-6
                              or abs(f - table.max_fwhm_ratio * (s.fwhm_median() or 1.0)) > 1e-6):
            self._ask_preview(ch)                  # the thresholds shown are not the stack's: what do they keep?
        by = {it.path: it for it in items} if isinstance(items, list) else {}
        p90 = ch.p90
        frames = []
        for r in table.rows:
            it = by.get(r.path)
            info = {"time": r.time or "", "fwhm": r.fwhm, "weight": r.weight,
                    "transparency": r.transparency / p90 if np.isfinite(r.transparency) else None,
                    "excluded": bool(r.excluded), "reason": _(r.status) if r.status and not r.excluded else ""}
            if it is not None:
                info.update(exposure=it.exposure, iso=it.iso, temperature=it.temperature, roundness=it.roundness)
            vi = ViewImage(r.path, r.name, camera=True, cfa=it.cfa if it is not None else "",
                           white=it.white if it is not None else None, info=info)
            vi.thumb = si.light_thumb(s.p, r.path)
            frames.append(vi)
        self.sel_frames = frames
        v = self._new_view("selection", None, histogram=False)
        v.setMinimumHeight(220)
        self._on(v, v.frameChanged, self._frame_shown)
        self._on(v, v.excludeToggled, self._seq_exclude)
        self._split("selection", top, v, (2, 5))
        v.set_sequence(frames, self.picked if self.picked is not None and self.picked < len(frames) else 0)

    def _selection_track(self, p, table):
        """The x / y tracking chart of Selection (``TrackChart``), the frames the stack leaves out circled with their
        reason; a plain line when the offsets are not known. None while Registration's data is read (the selection is
        built again when it comes)."""
        b = self._need("registration", self._keys(p, "register", "calibrate", "astrometry", "inventory"),
                       si.registration_bundle, p.pid)
        if b is None or isinstance(b, _Failed):
            return None
        d = b.data
        if not np.isfinite(d.dx).any():
            return label(_("The x / y tracking appears once the offsets of the frames are known (Registration)."),
                         "dim")
        out = {r.name: _(r.status) for r in table.rows if r.status}         # status: the reason, '' when kept
        ch = TrackChart(d, left_out=out)
        ch.title = ""                          # its title on its own line: the chart's head is full beside the frames
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(label(_("Tracking x / y (Lissajous)"), "h2"))
        bl.addWidget(ch, 1)
        return box

    def _ask_preview(self, c):
        """The engine counts the frames the dragged thresholds keep (selection_preview, in the worker)."""
        m = self.s.fwhm_median() or 1.0
        self._preview_token = self.service.request_preview(self.s.path.pid, c.tmin / 100, c.fmax / m)

    def _preview_ready(self, token: int, kept: int, total: int):
        if token == getattr(self, "_preview_token", None):
            self._kept_n = (kept, total)
            self._kept_text()

    def _kept_text(self):
        if self.chart is None or getattr(self, "kept_label", None) is None:
            return
        try:
            s = self.s
            kept, n = self._kept_n
            exp = s.info().get("exposure_s") or 0
            self.kept_label.setText(_("{kept} of {n} frames kept · {t}").format(
                kept=kept, n=n, t=hours_text(kept * exp / 3600)))
        except RuntimeError:                                  # the label went with a rebuild
            pass

    def _chart_changed(self, c):
        """Frames excluded by hand count for the whole project; thresholds go through the panel (Replace or Try as
        variant, when computed results would be lost)."""
        if self.table is not None:
            now = {(r.session, r.name) for r in self.table.rows if r.excluded}
            want = {(self.table.rows[i].session, self.table.rows[i].name) for i in c.hand}
            if now != want:
                self._set_hand(want - now, True)
                self._set_hand(now - want, False)
        t, f = self._thresholds()
        if c.thresholds and (abs(c.tmin - t) > 1e-6 or abs(c.fmax - f) > 1e-6):
            self._ask_preview(c)
            self.panel.propose_selection(c.tmin, c.fmax)

    def _set_hand(self, pairs, excluded: bool):
        pairs = [x for x in pairs if x[0] is not None]
        if pairs:
            self.s.set_excluded(pairs, excluded)                 # saved by the model, as any edit

    def _pick(self, i):
        """A frame clicked on the chart: shown in the viewer below."""
        self.picked = i
        if self.view is not None and self.sel_frames and 0 <= i < len(self.sel_frames):
            self.view.set_index(i)

    def _frame_shown(self, i: int):
        self.picked = i
        if self.chart is not None:
            try:
                self.chart.current = i
            except RuntimeError:
                pass

    def _seq_exclude(self, i: int, excluded: bool):
        """Exclude / Include in the viewer: by hand, for the whole project (the chart follows)."""
        if self.table is None or not (0 <= i < len(self.table.rows)):
            return
        r = self.table.rows[i]
        if r.session is None:
            return
        if self.chart is not None:
            (self.chart.hand.add if excluded else self.chart.hand.discard)(i)
            self.chart._classify()
        self._set_hand([(r.session, r.name)], excluded)


#: what Calibration does without frames of a kind, and the button that adds a folder of them (English, translated)
NO_FRAMES = {
    "darks": ("No darks", "No session of this project has darks now. Calibration subtracts the master bias, or the "
                          "camera's black level, instead, and finds the hot pixels on the lights. Add a folder of real "
                          "darks (the cap on, same exposure, ISO and temperature as the lights) to correct the "
                          "thermal signal."),
    "flats": ("No flats", "No session of this project has flats: the vignetting and the dust shadows stay in the "
                          "image. Add a folder of flats taken with the same optics and focus."),
    "bias": ("No bias", "No bias frames: the flats are calibrated with their dark flats, or the camera's black level."),
    "flat_darks": ("No dark flats", "No dark flats: the flats are calibrated with the bias, or the camera's black "
                                    "level."),
}
ADD_FOLDER = {"darks": "Add a darks folder…", "flats": "Add a flats folder…", "bias": "Add a bias folder…",
              "flat_darks": "Add a dark flats folder…"}


def no_frames_text(kind: str) -> str:
    """One sentence for a session left without frames of a kind."""
    return {"darks": _("A session has no darks any more: Calibration uses the master bias, or the camera's black "
                       "level, instead. Add a darks folder in the Darks tab."),
            "flats": _("A session has no flats any more: the vignetting stays. Add a flats folder in the Flats tab.")
            }.get(kind, _("A session has no frames of this kind any more."))


#: the engine's checks of a calibration frame (core/diagnostics.flag_frames) as templates: the same English as the
#: engine writes, with fields, so that the French is a translation with the numbers in it ({kind}: the frame's kind)
CHECK_TEMPLATES = {
    "exposure": "exposure {value:g} s, the other {plural} {others:g} s",
    "temperature": "taken at {value:.0f} °C, the other {plural} at {others:.0f} °C",
    "level": "its level ({value:.0f} % of full scale) differs from the other flats ({others:.0f} %): the light changed?",
    "sky": "its sky level ({value:.0f} ADU) differs from the other lights ({others:.0f} ADU): moon, twilight, clouds "
           "or dew?",
    "median": "its level ({value:.1f} ADU) differs from the other {plural} ({others:.1f} ADU)",
    "median_dark": "its level ({value:.1f} ADU) differs from the other {plural} ({others:.1f} ADU): another exposure, "
                   "ISO or temperature?",
    "hot_pixels": "{value} hot pixels, the other {plural} about {others:.0f}: a warmer sensor?",
    "gradient_brighter": "the centre is {absv:.1f} ADU brighter than the corners, the other {plural} {others:+.1f} ADU: "
                         "heat or light reaching the sensor?",
    "gradient_darker": "the centre is {absv:.1f} ADU darker than the corners, the other {plural} {others:+.1f} ADU: "
                       "heat or light reaching the sensor?",
    "noise": "its noise ({value:.1f} ADU) is {ratio:.1f}x that of the other {plural} ({others:.1f} ADU)",
    "noise_alone": "its noise is {value:.1f} ADU",
    "saturated": "{value:.2f} % of its pixels are saturated",
    "underexposed": "its level is only {value:.0f} % of full scale (aim for 30-60 %)",
    "overexposed": "its level is {value:.0f} % of full scale: close to saturation, where the sensor is no longer linear",
    "stars": "contains stars ({value:.0f} found): a frame of the sky, not a {one} (was the cap on?)",
    "stars_flat": "contains stars ({value:.0f} found): a frame of the sky, not a flat",
    "sky_level": "its level is {value:.0f} ADU above the camera's black level ({black:.0f} ADU), {ratio:.0f}x its "
                 "noise: light reached the sensor, not a {one}",
}
KIND_ONE = {"darks": "dark", "flats": "flat", "bias": "bias frame", "flat_darks": "flat dark", "lights": "light"}
KIND_PLURAL = {"darks": "darks", "flats": "flats", "bias": "bias frames", "flat_darks": "flat darks",
               "lights": "lights"}


def check_template(check: dict, kind: str) -> tuple[str, dict] | None:
    """(English template with the kind's words in it, its numbers) of one check of the engine, or None when the
    check is not one of these."""
    key = check.get("key")
    v, o = check.get("value"), check.get("others")
    args = {"value": v, "others": o}
    if key == "median":
        name = "median_dark" if kind in ("darks", "flat_darks") else "median"
    elif key == "gradient":
        if not isinstance(v, int | float):
            return None
        name = "gradient_brighter" if v > 0 else "gradient_darker"
        args["absv"] = abs(float(v))
    elif key == "noise":
        if isinstance(o, int | float) and o > 0 and isinstance(v, int | float):
            name, args["ratio"] = "noise", float(v) / float(o)
        else:
            name = "noise_alone"
    elif key == "stars":
        name = "stars_flat" if kind == "flats" else "stars"
    elif key == "sky_level":
        nz = check.get("noise")
        if not (isinstance(v, int | float) and isinstance(nz, int | float) and nz > 0):
            return None
        name = "sky_level"
        args.update(black=check.get("black"), ratio=float(v) / float(nz))
    elif key in CHECK_TEMPLATES:
        name = key
    else:
        return None
    t = CHECK_TEMPLATES[name].replace("{plural}", KIND_PLURAL.get(kind, kind)).replace("{one}", KIND_ONE.get(kind, kind))
    if any(a is None for f_, a in args.items() if "{" + f_ in t):
        return None
    return t, args


def frame_reason(cf, kind: str, lang: str) -> str:
    """Why the engine flags a calibration frame, in the interface's language: each check from its template (the
    numbers written as the language writes them), else the engine's own sentence (English when it has no
    translation)."""
    if cf is None:
        return ""
    parts = []
    for c in getattr(cf, "checks", None) or []:
        got = check_template(c, kind) if isinstance(c, dict) else None
        if got is None:
            txt = str(c.get("text", "")) if isinstance(c, dict) else str(c)
            parts.append(texts.translate(txt, lang))
            continue
        try:
            parts.append(texts.format_reason(got[0], got[1], lang))
        except (KeyError, ValueError, TypeError):
            parts.append(str(c.get("text", "")))
    text = "; ".join(x for x in parts if x) or texts.translate(str(getattr(cf, "reason", "") or ""), lang)
    return text[:1].upper() + text[1:] if text else _("The engine flags this frame against the others of its kind.")


class _Failed:
    """What the worker could not read, and why (its first line)."""

    def __init__(self, why: str):
        self.why = why


def _noise_text(v) -> str:
    if isinstance(v, dict):
        return " · ".join(f"{k} {_stat(x)} ADU" for k, x in v.items())
    if isinstance(v, list | tuple | np.ndarray):
        return " / ".join(_stat(x) for x in v)
    return _stat(v)


def _stat(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    return number(x, 3 if abs(x) < 10 else 1)


def approx(seconds: float, measured: bool) -> str:
    """A duration, with ≈ when it is an estimate ("≤ 1 min" is one already)."""
    t = hm(seconds)
    return t if measured or t.startswith("≤") else "≈ " + t


def sky_text(ra: float, dec: float) -> str:
    """RA and Dec as the readout writes them: 02h42m06.6s +61°18′01″ (a decimal comma in French)."""
    from .imageview import sky_text as iv_sky_text
    return iv_sky_text(ra, dec)


__all__ = ["TuneView", "Strip", "make_table", "WeightsChart", "ColorFitChart", "sky_text", "approx"]
