"""Interactive charts of the client, on PySide6.QtCharts.

``ChartView`` is one widget for every chart of the client: stacked panels that share their x axis, each with its
y axis, line / scatter / histogram series, and:

* the wheel zooms (over the plot: both axes, or x only for time series; over an axis: that axis; Shift: x only;
  Ctrl: y only), a drag pans, a double click (or the "Reset zoom" button, which shows when zoomed) goes back;
* a crosshair snaps to the data and a box reads the values under the cursor, with their units;
* the legend toggles series (a click on its name);
* an "Export CSV…" button in the title row, and the context menu, export the data as CSV (a ";" and decimal commas
  in French, so that a French spreadsheet opens it right); the menu also exports the picture as PNG and holds the
  actions a chart adds;
* horizontal handles (``add_hline(..., draggable=True)``) are dragged with the mouse (the thresholds of Selection).

The axes of QtCharts are kept hidden: they give the series their scale, and the ticks, the labels (nice steps, the
language's decimal comma, clock times for a time axis), the grid and the titles are drawn here, so that the panels line
up and every number goes through ``i18n.number`` as the rest of the client does. Theme: the colours of ``widgets.py``.

The mouse events are the widget's own (the QGraphicsView that draws the chart lets them through), so a test or a
parent sends them to the chart itself.
"""
from __future__ import annotations

import math
import re
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PySide6.QtCharts import QAreaSeries, QChart, QLineSeries, QScatterSeries, QValueAxis
from PySide6.QtCore import QEvent, QMargins, QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QImage, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
    QPushButton,
    QToolTip,
    QWidget,
)

from .i18n import _, language, number

# ---------------------------------------------------------------------------------------------- theme (widgets.py)
BG = QColor("#181b21")                      # the charts of the client have always been on this
PLOT = QColor("#15181e")
GRID = QColor("#2a2f38")
FRAME = QColor("#3a4150")
TEXT = QColor("#d8dce4")
DIM = QColor("#8b93a1")
CROSS = QColor("#7d8696")
TIP_BG, TIP_LINE = QColor("#262a33"), QColor("#3a4150")        # the tooltips of the style sheet
BLUE, AMBER, GREEN, VIOLET, RED = (QColor(c) for c in ("#5ea1ff", "#e6a23c", "#5fbf7f", "#a395ff", "#e06c6c"))
#: a sequential ramp that stays readable on the dark background: light grows with the value
RAMP = [QColor(c) for c in ("#3f6fe0", "#7a63d8", "#c25aa6", "#ee7d6a", "#f7b955", "#fff0a8")]
EPOCH = datetime(1970, 1, 1)
CHART_PAD = 9.5                              # what QtCharts adds around its plot area (measured, never assumed)

_last_dir: list[str] = []                    # the folder of the last export, for this session only


# ---------------------------------------------------------------------------------------------- numbers and ticks
def num(v: float, digits: int) -> str:
    """A number for the screen: the language's decimal comma, a true minus sign, no "-0,0"."""
    if not np.isfinite(v):
        return "–"
    if abs(v) < 0.5 * 10.0 ** -digits:
        v = 0.0
    return number(v, digits).replace("-", "−")


def nice_step(span: float, target: int) -> float:
    """The 1-2-5 step that gives at most ``target`` ticks over ``span``."""
    if not span > 0 or not np.isfinite(span):
        return 1.0
    raw = span / max(target, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * mag >= raw * 0.9999:
            return m * mag
    return 10 * mag


def digits_of(step: float) -> int:
    """The decimals a tick label needs to tell the ticks of this step apart."""
    return max(0, int(-math.floor(math.log10(step) + 1e-9)))


def tick_values(lo: float, hi: float, step: float) -> list[float]:
    k0, k1 = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    if k1 - k0 > 200:
        return []
    return [round(k * step, 12) for k in range(k0, k1 + 1)]


TIME_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800,
              604800)


def time_step(span: float, target: int) -> float:
    raw = span / max(target, 1)
    for s in TIME_STEPS:
        if s >= raw:
            return float(s)
    return float(TIME_STEPS[-1])


def clock(v: float, seconds: bool = False) -> str:
    """A time given in seconds since 1970 as the camera wrote it (no time zone is applied): "23:41"."""
    d = EPOCH + timedelta(seconds=float(v))
    return d.strftime("%H:%M:%S" if seconds else "%H:%M")


def to_seconds(iso: str | None) -> float | None:
    """An ISO time of a frame as seconds since 1970 (the clock as written), None when it cannot be read."""
    if not iso:
        return None
    try:
        return (datetime.fromisoformat(str(iso)).replace(tzinfo=None) - EPOCH).total_seconds()
    except ValueError:
        return None


def ramp_color(t: float) -> QColor:
    """The colour of the ramp at t in [0, 1]."""
    t = min(max(float(t), 0.0), 1.0) * (len(RAMP) - 1)
    i = min(int(t), len(RAMP) - 2)
    f = t - i
    a, b = RAMP[i], RAMP[i + 1]
    return QColor(round(a.red() + f * (b.red() - a.red())), round(a.green() + f * (b.green() - a.green())),
                  round(a.blue() + f * (b.blue() - a.blue())))


def _snap(v: float) -> float:
    return math.floor(v) + 0.5


def _same(a, b) -> bool:
    """Two ranges (x0, x1, y0, y1) that are the same view, up to rounding."""
    tol = 1e-9 * (abs(b[1] - b[0]) + abs(b[3] - b[2])) + 1e-300
    return all(abs(u - v) <= tol for u, v in zip(a, b))


def _slug(text: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", text.strip(), flags=re.UNICODE).strip("_")
    return s or "chart"


# ---------------------------------------------------------------------------------------------- data
class Trace:
    """One series of a chart: its data, how it is drawn, and what the readout and the CSV say about it.

    ``kind``: line | scatter | hist | data (not drawn: it only gives the readout and the CSV their values).
    ``c``: a value for each point of a scatter, mapped on the colour ramp. ``edges``: the bin edges of a histogram
    (``x`` are the centres, ``y`` the counts). ``extra``: more columns for the CSV, {name: sequence}; those named in
    ``show_extra`` are in the readout. ``alt``: (factor, unit, digits) another unit the readout adds in brackets.
    """

    def __init__(self, view: ChartView, panel: Panel, name: str, kind: str, x, y, color: QColor, *, unit: str = "",
                 digits: int = 2, legend=True, size: float = 6.0, width: float = 1.8, points: bool = False,
                 interp: bool = False, extra: dict | None = None, show_extra: tuple = (), readout: bool = True,
                 export: bool = True, c=None, edges=None, alt=None, alpha: float = 1.0):
        self.view, self.panel, self.name, self.kind = view, panel, name, kind
        self.color = QColor(color)
        self.unit, self.digits = unit, digits
        self.legend = (name if legend is True else legend) or None
        self.size, self.width, self.points, self.interp = size, width, points, interp
        self.extra = {k: list(v) for k, v in (extra or {}).items()}
        self.show_extra = tuple(show_extra)
        self.readout, self.export = readout, export
        self.alt, self.alpha = alt, alpha
        self.visible = True
        self.qt: list = []
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.c = None if c is None else np.asarray(c, float)
        self.edges = None if edges is None else np.asarray(edges, float)
        self._build()
        self.set_data(self.x, self.y, self.c, self.edges)

    # ------------------------------------------------------------ the QtCharts series
    def _new(self, s):
        ch = self.panel.chart
        ch.addSeries(s)
        s.attachAxis(self.panel.ax)
        s.attachAxis(self.panel.ay)
        self.qt.append(s)
        return s

    def _scatter(self, color: QColor, size: float):
        s = QScatterSeries()
        s.setMarkerShape(QScatterSeries.MarkerShapeCircle)
        s.setMarkerSize(size)
        s.setColor(color)
        s.setPen(QPen(Qt.NoPen))
        s.setOpacity(self.alpha)
        return self._new(s)

    def _build(self):
        if self.kind == "line":
            s = QLineSeries()
            pen = QPen(self.color, self.width)
            pen.setJoinStyle(Qt.RoundJoin)
            pen.setCapStyle(Qt.RoundCap)
            s.setPen(pen)
            s.setOpacity(self.alpha)
            self._new(s)
            if self.points:
                self._scatter(self.color, self.size)
        elif self.kind == "scatter":
            bins = 1 if self.c is None else 24
            for b in range(bins):
                self._scatter(self.color if self.c is None else ramp_color((b + 0.5) / bins), self.size)
        elif self.kind == "hist":
            up = QLineSeries()
            up.setPen(QPen(self.color, 1.6))
            area = QAreaSeries(up)
            area.setPen(QPen(self.color, 1.6))
            fill = QColor(self.color)
            fill.setAlpha(90)
            area.setBrush(QBrush(fill))
            self._up = up
            self.panel.chart.addSeries(area)
            area.attachAxis(self.panel.ax)
            area.attachAxis(self.panel.ay)
            self.qt.append(area)

    def set_data(self, x, y, c=None, edges=None):
        """New values for the series (the colours and the bins with them); the view is not refitted."""
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        if c is not None:
            self.c = np.asarray(c, float)
        if edges is not None:
            self.edges = np.asarray(edges, float)
        ok = np.isfinite(self.x) & np.isfinite(self.y)
        if self.kind == "line":
            self.qt[0].replaceNp(self.x[ok], self.y[ok])
            if self.points:
                self.qt[1].replaceNp(self.x[ok], self.y[ok])
        elif self.kind == "scatter":
            if self.c is None:
                self.qt[0].replaceNp(self.x[ok], self.y[ok])
            else:
                cv = self.c
                fin = ok & np.isfinite(cv)
                idx = np.zeros(len(cv), int)
                if fin.any():
                    lo, hi = float(cv[fin].min()), float(cv[fin].max())
                    idx = np.clip((np.nan_to_num(cv - lo) / ((hi - lo) or 1.0) * len(self.qt)).astype(int),
                                  0, len(self.qt) - 1)
                for b, series in enumerate(self.qt):
                    m = fin & (idx == b)
                    series.replaceNp(self.x[m], self.y[m])
        elif self.kind == "hist" and self.edges is not None and len(self.edges) > 1:
            e, n = self.edges, np.nan_to_num(self.y)
            px = np.repeat(e, 2)
            py = np.concatenate([[0.0], np.repeat(n, 2), [0.0]])
            self._up.replaceNp(px, py)
        self.view._data_changed()

    def set_visible(self, v: bool):
        self.visible = bool(v)
        for s in self.qt:
            s.setVisible(self.visible)

    # ------------------------------------------------------------ what the readout and the CSV need
    def finite(self) -> np.ndarray:
        return np.isfinite(self.x) & np.isfinite(self.y)

    def y_text(self, v: float) -> str:
        t = num(v, self.digits) + (f" {self.unit}" if self.unit else "")
        if self.alt is not None:
            k, u, d = self.alt
            t += f" ({num(v * k, d)} {u})"
        return t


class Panel:
    """A plot of a chart: its QChart (the series), the scale, the ticks, the handles and the lines drawn over it."""

    def __init__(self, view: ChartView, title: str, unit: str, weight: float, digits: int | None, zero: bool,
                 equal: bool):
        self.view, self.title, self.unit, self.weight, self.digits = view, title, unit, weight, digits
        self.zero, self.equal = zero, equal                      # the y axis starts at 0; one pixel is one unit on both
        self.chart = QChart()
        self.chart.setBackgroundVisible(False)
        self.chart.setPlotAreaBackgroundVisible(False)
        self.chart.legend().hide()
        self.chart.setMargins(QMargins(0, 0, 0, 0))
        self.chart.setAnimationOptions(QChart.NoAnimation)
        self.ax, self.ay = QValueAxis(), QValueAxis()
        self.chart.addAxis(self.ax, Qt.AlignBottom)
        self.chart.addAxis(self.ay, Qt.AlignLeft)
        self.ax.setVisible(False)
        self.ay.setVisible(False)
        self.ax.setRange(0, 1)
        self.ay.setRange(0, 1)
        view._canvas.scene().addItem(self.chart)
        self.x0, self.x1, self.y0, self.y1 = 0.0, 1.0, 0.0, 1.0
        self.home = self.home_raw = (0.0, 1.0, 0.0, 1.0)         # what Reset shows (home_raw: before equal axes)
        self.home_y: tuple[float, float] | None = None          # set by the chart: else fitted to the data
        self.plot = QRectF(0, 0, 10, 10)
        self.traces: list[Trace] = []
        self.hlines: list[HLine] = []
        self.yticks: list[tuple[float, str]] = []

    # ------------------------------------------------------------ the scale
    def fx(self, x):
        return self.plot.left() + (np.asarray(x, float) - self.x0) / ((self.x1 - self.x0) or 1.0) * self.plot.width()

    def fy(self, y):
        return self.plot.bottom() - (np.asarray(y, float) - self.y0) / ((self.y1 - self.y0) or 1.0) * self.plot.height()

    def vx(self, px: float) -> float:
        return self.x0 + (px - self.plot.left()) / (self.plot.width() or 1.0) * (self.x1 - self.x0)

    def vy(self, py: float) -> float:
        return self.y0 + (self.plot.bottom() - py) / (self.plot.height() or 1.0) * (self.y1 - self.y0)

    def apply(self):
        self.ax.setRange(self.x0, self.x1)
        self.ay.setRange(self.y0, self.y1)

    def y_digits(self, step: float) -> int:
        """Decimals of a y label: at least ``digits``, more when a zoom makes the ticks finer than that."""
        return max(self.digits or 0, digits_of(step))


@dataclass
class HLine:
    """A horizontal line over a panel, with a label; ``draggable``: the mouse moves it (``lineMoved``, ``lineReleased``)."""
    name: str
    panel: Panel
    value: float
    color: QColor
    label: object = None                       # callable(value) -> str
    draggable: bool = False
    width: float = 1.6


@dataclass
class VLine:
    x: float
    panel: Panel
    label: str = ""
    color: QColor = field(default_factory=lambda: QColor("#6b7280"))


@dataclass
class _Hover:
    panel: Panel
    pos: QPointF
    items: list                                # [(trace, index or None, y)]
    x: float
    y: float
    snap_x: float | None = None                # where the crosshair sits when it is on a sample


class _Canvas(QGraphicsView):
    """The QGraphicsView that draws the charts and lets every mouse event through to the ChartView."""

    def __init__(self, owner: ChartView):
        super().__init__(owner)
        self.owner = owner
        self.setScene(QGraphicsScene(self))
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.viewport().setAttribute(Qt.WA_TransparentForMouseEvents)

    def drawBackground(self, p, rect):                                   # noqa: N802 - Qt's name
        self.owner._paint_back(p)

    def drawForeground(self, p, rect):                                   # noqa: N802
        self.owner._paint_front(p)


BUTTON_QSS = ("QPushButton { background: #232831; border: 1px solid #353b47; border-radius: 3px; padding: 1px 9px;"
              " font-size: 9pt; color: #d8dce4; }"
              "QPushButton:hover { background: #2b3446; }"
              "QPushButton:checked { background: #2b3446; border-color: #5ea1ff; color: #ffffff; }"
              "QPushButton:disabled { color: #5d6470; border-color: #2a2f39; }")


class ChartView(QWidget):
    """An interactive chart. Build it with ``add_panel`` / ``set_x`` / ``add_line`` / ``add_scatter`` /
    ``add_histogram`` / ``add_hline`` / ``add_vline``, then ``fit()``."""

    viewChanged = Signal()
    lineMoved = Signal(str, float)               # a handle is being dragged (name, value)
    lineReleased = Signal(str, float)            # and dropped
    pointClicked = Signal(object, int)           # (trace, index) of the sample clicked
    rubberBand = Signal(QRectF, QPoint)          # Shift + drag: the rectangle (widget coordinates), the mouse (screen)

    HEAD = 30                                    # the title / legend row

    def __init__(self, title: str = "", parent=None, *, zoom: str = "xy", readout: str = "x", pickable: bool = False,
                 snap: float | None = None, export_name: str = ""):
        super().__init__(parent)
        self.title = title
        self.zoom_axes = zoom                    # what the wheel zooms over the plot: "xy" | "x"
        self.readout_mode = readout              # "x": the samples at the cursor's x · "xy": the sample nearest to it
        self.pickable = pickable
        self.snap_px = snap if snap is not None else (44.0 if readout == "x" else 16.0)
        self.export_name = export_name
        self.x_title, self.x_unit, self.x_kind, self.x_digits = "", "", "num", None   # num | int | time
        self.x_pad = 0.03                                                         # room left of and right of the data
        self.x_short = ""                                                         # "Frame": "Frame 12" in the readout
        self.x_dates = ""                                                         # shown with a time axis
        self.zero_x = False
        self.panels: list[Panel] = []
        self.traces: list[Trace] = []
        self.vlines: list[VLine] = []
        self.colorbar: tuple[str, str, str] | None = None                         # caption, low text, high text
        self.actions: list[tuple[str, object]] = []
        self.controls: list[QWidget] = []
        self._hover: _Hover | None = None
        self._drag = None
        self._lasso = None
        self._lasso_on = False
        self._legend_hits: list[tuple[QRectF, str]] = []
        self._title_rect = QRectF()
        self._left = 40.0
        self._export = False
        self._legend: dict[str, list[Trace]] = {}
        self.xticks: list[tuple[float, str]] = []
        self._ctrl_left = 0.0
        self.setMouseTracking(True)
        self.setMinimumSize(280, 200)
        self._canvas = _Canvas(self)
        self._reset_btn = QPushButton(_("Reset zoom"), self)
        self._reset_btn.setStyleSheet(BUTTON_QSS)
        self._reset_btn.setCursor(Qt.PointingHandCursor)
        self._reset_btn.clicked.connect(self.reset_view)
        self._reset_btn.hide()
        # the data of the chart to a spreadsheet: always in sight (the context menu has it too, and PNG)
        self._csv_btn = QPushButton(_("Export CSV…"), self)
        self._csv_btn.setStyleSheet(BUTTON_QSS)
        self._csv_btn.setCursor(Qt.PointingHandCursor)
        self._csv_btn.setToolTip(_("The chart's data as a CSV file, for a spreadsheet (right click: PNG too)"))
        self._csv_btn.clicked.connect(lambda: self.export_csv())

    # ------------------------------------------------------------ building
    def add_panel(self, title: str = "", unit: str = "", weight: float = 1.0, digits: int | None = None,
                  zero: bool = False, equal: bool = False) -> Panel:
        """A plot below the others; ``zero``: its y axis starts at 0, ``equal``: one pixel is the same in x and y.
        A chart that never calls it has one plot without a title (panel 0)."""
        p = Panel(self, title, unit, weight, digits, zero, equal)
        self.panels.append(p)
        return p

    def set_x(self, title: str = "", unit: str = "", kind: str = "num", digits: int | None = None, dates: str = "",
              zero: bool = False):
        """The shared x axis: ``kind`` num | int (whole numbers: frame numbers) | time (seconds since 1970)."""
        self.x_title, self.x_unit, self.x_kind, self.x_digits, self.x_dates = title, unit, kind, digits, dates
        self.zero_x = zero
        self.update()

    def _panel(self, panel) -> Panel:
        if isinstance(panel, Panel):
            return panel
        while len(self.panels) <= panel:
            self.add_panel()
        return self.panels[panel]

    def _add(self, panel, name, kind, x, y, color, **kw) -> Trace:
        p = self._panel(panel)
        t = Trace(self, p, name, kind, x, y, color, **kw)
        p.traces.append(t)
        self.traces.append(t)
        self._legend_rebuild()
        return t

    def add_line(self, name, x, y, *, panel=0, color=BLUE, **kw) -> Trace:
        return self._add(panel, name, "line", x, y, color, **kw)

    def add_scatter(self, name, x, y, *, panel=0, color=BLUE, **kw) -> Trace:
        return self._add(panel, name, "scatter", x, y, color, **kw)

    def add_data(self, name, x, y, *, panel=0, color=BLUE, **kw) -> Trace:
        """Values that are not drawn (the series that draw them are others): the readout and the CSV have them."""
        kw.setdefault("legend", False)
        return self._add(panel, name, "data", x, y, color, **kw)

    def add_histogram(self, name, edges, counts, *, panel=0, color=BLUE, **kw) -> Trace:
        edges = np.asarray(edges, float)
        kw.setdefault("unit", "")
        t = self._add(panel, name, "hist", 0.5 * (edges[:-1] + edges[1:]), counts, color, edges=edges, **kw)
        self._panel(panel).zero = True
        return t

    def add_hline(self, name, value, *, panel=0, color=RED, label=None, draggable=False, width=1.6) -> HLine:
        h = HLine(name, self._panel(panel), float(value), QColor(color), label, draggable, width)
        h.panel.hlines.append(h)
        self.update()
        return h

    def add_vline(self, x, label="", *, panel=0, color=None) -> VLine:
        v = VLine(float(x), self._panel(panel), label, QColor(color) if color else QColor("#6b7280"))
        self.vlines.append(v)
        self.update()
        return v

    def add_choice(self, options, current, callback) -> QButtonGroup:
        """A row of exclusive buttons in the title row (options: [(key, text)]); ``callback(key)`` when one is picked."""
        g = QButtonGroup(self)
        g.setExclusive(True)
        for key, text in options:
            b = QPushButton(text, self)
            b.setCheckable(True)
            b.setStyleSheet(BUTTON_QSS)
            b.setCursor(Qt.PointingHandCursor)
            b.setChecked(key == current)
            b.setProperty("key", key)
            g.addButton(b)
            self.controls.append(b)
            b.clicked.connect(lambda _c=False, k=key: callback(k))
            b.show()
        self._layout()
        return g

    def add_action(self, text: str, callback):
        """An entry the context menu has above the exports."""
        self.actions.append((text, callback))

    def clear(self):
        """Remove every series, line and panel's content (the panels stay)."""
        for t in self.traces:
            for s in t.qt:
                t.panel.chart.removeSeries(s)
        for p in self.panels:
            p.traces.clear()
            p.hlines.clear()
        self.traces.clear()
        self.vlines.clear()
        self._hover = None
        self._legend_rebuild()
        self.update()

    # ------------------------------------------------------------ the scale
    def _data_changed(self):
        self.update()

    def fit(self):
        """Fit every panel to its data (and to the handles): the view the Reset button goes back to."""
        xs = [t.x[t.finite()] for t in self.traces if t.kind != "hist" or t.edges is None]
        xs += [np.asarray(t.edges, float) for t in self.traces if t.kind == "hist" and t.edges is not None]
        xs = [a for a in xs if a.size]
        if xs:
            lo, hi = min(float(a.min()) for a in xs), max(float(a.max()) for a in xs)
            pad = 0.5 if self.x_kind == "int" else (hi - lo) * self.x_pad or (1.0 if self.x_pad else 0.0)
            if self.x_kind == "time" and hi == lo:
                pad = 60.0
            lo, hi = lo - pad, hi + pad
            if self.zero_x:
                lo = min(lo, 0.0)
        else:
            lo, hi = 0.0, 1.0
        for p in self.panels:
            if p.home_y is not None:
                ylo, yhi = p.home_y
            else:
                ys = [t.y[t.finite()] for t in p.traces if t.kind != "hist" or t.edges is None]
                ys += [t.y for t in p.traces if t.kind == "hist"]
                ys = [np.asarray(a, float) for a in ys if len(a)]
                ys = [a[np.isfinite(a)] for a in ys]
                ys = [a for a in ys if a.size]
                vals = [h.value for h in p.hlines]
                if ys:
                    ylo = min(float(a.min()) for a in ys)
                    yhi = max(float(a.max()) for a in ys)
                else:
                    ylo, yhi = 0.0, 1.0
                if vals:
                    ylo, yhi = min(ylo, min(vals)), max(yhi, max(vals))
                span = yhi - ylo or max(abs(yhi), 1.0) * 0.1
                ylo, yhi = ylo - 0.06 * span, yhi + 0.08 * span
                if p.zero:
                    ylo = 0.0
            p.home = p.home_raw = (lo, hi, ylo, yhi)
            p.x0, p.x1, p.y0, p.y1 = p.home
            p.apply()
        self._relayout()

    def set_home(self, panel=0, *, y: tuple | None = None):
        """Fix the y range of a panel that ``fit`` and Reset go back to (else it is fitted to the data)."""
        self._panel(panel).home_y = None if y is None else (float(y[0]), float(y[1]))
        self.fit()

    def reset_view(self):
        for p in self.panels:
            p.x0, p.x1, p.y0, p.y1 = p.home
            p.apply()
        self._relayout()
        self.viewChanged.emit()

    def is_zoomed(self) -> bool:
        return any(not _same((p.x0, p.x1, p.y0, p.y1), p.home) for p in self.panels)

    def set_range(self, panel=0, x=None, y=None):
        """Show this part of the data (the x range is the shared one)."""
        p = self._panel(panel)
        if x is not None:
            for q in self.panels:
                q.x0, q.x1 = float(x[0]), float(x[1])
        if y is not None:
            p.y0, p.y1 = float(y[0]), float(y[1])
        self._equalize()
        for q in self.panels:
            q.apply()
        self._relayout()
        self.viewChanged.emit()

    def scale_axes(self, fx: float = 1.0, fy: float = 1.0, panel=0):
        """The data of a panel were multiplied by fx / fy (a unit change): the view and its home follow."""
        for q in self.panels:
            if fx != 1.0:
                q.x0, q.x1 = q.x0 * fx, q.x1 * fx
                q.home_raw = (q.home_raw[0] * fx, q.home_raw[1] * fx) + q.home_raw[2:]
                q.home = (q.home[0] * fx, q.home[1] * fx) + q.home[2:]
        p = self._panel(panel)
        if fy != 1.0:
            p.y0, p.y1 = p.y0 * fy, p.y1 * fy
            p.home_raw = p.home_raw[:2] + (p.home_raw[2] * fy, p.home_raw[3] * fy)
            p.home = p.home[:2] + (p.home[2] * fy, p.home[3] * fy)
            if p.home_y:
                p.home_y = (p.home_y[0] * fy, p.home_y[1] * fy)
        for q in self.panels:
            q.apply()
        self._relayout()

    @staticmethod
    def _eq_box(p: Panel, box):
        """The range that makes a pixel the same in x and y, around the centre of ``box`` (it only grows)."""
        x0, x1, y0, y1 = box
        s = max((x1 - x0) / p.plot.width(), (y1 - y0) / p.plot.height())
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        return (cx - s * p.plot.width() / 2, cx + s * p.plot.width() / 2,
                cy - s * p.plot.height() / 2, cy + s * p.plot.height() / 2)

    def _equalize(self):
        """After a zoom: one pixel is the same in x and y on the panels that ask for it (the track of the drift)."""
        for p in self.panels:
            if p.equal and p.plot.width() >= 20 and p.plot.height() >= 20:
                p.x0, p.x1, p.y0, p.y1 = self._eq_box(p, (p.x0, p.x1, p.y0, p.y1))

    def _sync_home(self):
        """The home of the equal-axes panels follows the shape of their plot; an unzoomed view follows its home."""
        for p in self.panels:
            if not (p.equal and p.plot.width() >= 20 and p.plot.height() >= 20):
                p.home = p.home_raw
                continue
            unzoomed = _same((p.x0, p.x1, p.y0, p.y1), p.home)
            p.home = self._eq_box(p, p.home_raw)
            p.x0, p.x1, p.y0, p.y1 = p.home if unzoomed else self._eq_box(p, (p.x0, p.x1, p.y0, p.y1))

    def _clamp(self, p: Panel, lo, hi, which: int):
        """Keep a range inside the home range (a little leeway; a lot for the panels with equal axes)."""
        h0, h1 = p.home[2 * which], p.home[2 * which + 1]
        full = h1 - h0
        if not p.equal and hi - lo >= full:                      # zoomed all the way out: the home view itself
            return h0, h1
        leeway = (0.5 if p.equal else 0.04) * full
        span = min(hi - lo, full * (1.6 if p.equal else 1.0))
        span = max(span, full * 1e-4)
        c = 0.5 * (lo + hi)
        lo, hi = c - span / 2, c + span / 2
        if lo < h0 - leeway:
            lo, hi = h0 - leeway, h0 - leeway + span
        if hi > h1 + leeway:
            lo, hi = h1 + leeway - span, h1 + leeway
        return lo, hi

    # ------------------------------------------------------------ layout
    def _font(self) -> QFont:
        self.ensurePolished()
        f = QFont(self.font())
        f.setPointSizeF(9.0)
        return f

    def _xticks(self, p: Panel) -> list[tuple[float, str]]:
        span = p.x1 - p.x0
        n = max(2, int(p.plot.width() / 78))
        if self.x_kind == "time":
            step = time_step(span, n)
            vals = tick_values(p.x0, p.x1, step)
            return [(v, clock(v, step < 60) if step < 172800 else (EPOCH + timedelta(seconds=v)).strftime("%Y-%m-%d"))
                    for v in vals]
        step = max(nice_step(span, n), 1.0) if self.x_kind == "int" else nice_step(span, n)
        d = 0 if self.x_kind == "int" else (self.x_digits if self.x_digits is not None else digits_of(step))
        return [(v, num(v, d)) for v in tick_values(p.x0, p.x1, step)]

    def _yticks(self, p: Panel) -> list[tuple[float, str]]:
        step = nice_step(p.y1 - p.y0, max(2, int(p.plot.height() / 30)))
        d = p.y_digits(step)
        return [(v, num(v, d)) for v in tick_values(p.y0, p.y1, step)]

    def _head_h(self) -> float:
        return max(self.HEAD, QFontMetricsF(self._font()).height() + 14.0)       # the title row (Export CSV… is there)

    def resizeEvent(self, e):                                            # noqa: N802
        super().resizeEvent(e)
        self._canvas.setGeometry(self.rect())
        self._canvas.scene().setSceneRect(0, 0, self.width(), self.height())
        self._layout()

    def showEvent(self, e):                                              # noqa: N802
        super().showEvent(e)
        self._layout()

    def _relayout(self):
        self._layout()

    def _layout(self):
        """Place the panels and the controls; the width the y labels need decides the left margin (two passes)."""
        W, H = self.width(), self.height()
        if W < 40 or H < 40 or not self.panels:
            return
        fm = QFontMetricsF(self._font())
        lh = fm.height()
        head = self._head_h()
        bottom = lh + 10 + ((lh + 6) if self.x_title else 0) + 4
        bands = [(lh + 8) if p.title else 6.0 for p in self.panels]
        gap = 6.0
        avail = max(H - head - bottom - sum(bands) - gap * (len(self.panels) - 1), 40.0 * len(self.panels))
        wsum = sum(p.weight for p in self.panels) or 1.0
        right = 16.0
        for _pass in range(2):
            y = head
            for p, band in zip(self.panels, bands):
                y += band
                h = avail * p.weight / wsum
                want = QRectF(self._left, y, max(W - self._left - right, 20.0), h)
                ch = p.chart
                ch.setGeometry(want.adjusted(-CHART_PAD, -CHART_PAD, CHART_PAD, CHART_PAD))
                ch.layout().activate()
                pa = ch.plotArea()
                g = ch.geometry()
                p.plot = QRectF(g.x() + pa.x(), g.y() + pa.y(), pa.width(), pa.height())
                y += h + gap
            self._sync_home()
            for p in self.panels:
                p.apply()
                p.yticks = self._yticks(p)
            need = max([fm.horizontalAdvance(t) for p in self.panels for _v, t in p.yticks] + [14.0]) + 14.0
            need = max(need, 36.0)
            if abs(need - self._left) < 0.5:
                break
            self._left = need
        for p in self.panels:
            p.yticks = self._yticks(p)
        self.xticks = self._xticks(self.panels[-1])
        self._place_controls(W)
        self.update()

    def _place_controls(self, W: int):
        """Export CSV… at the right end of the title row, the choice buttons at its left (in the order they were
        added), the reset button at their left."""
        x, y = W - 12.0, 3
        self._csv_btn.adjustSize()
        x -= self._csv_btn.width()
        self._csv_btn.move(int(x), y)
        self._csv_btn.raise_()
        x -= 8
        for b in reversed(self.controls):
            b.adjustSize()
            x -= b.width()
            b.move(int(x), y)
            b.raise_()
            x -= 2
        if self.controls:
            x -= 8
        self._reset_btn.adjustSize()
        zoomed = self.is_zoomed()
        self._reset_btn.setVisible(zoomed)
        if zoomed:
            x -= self._reset_btn.width()
            self._reset_btn.move(int(x), y)
            self._reset_btn.raise_()
            x -= 8
        self._ctrl_left = x

    # ------------------------------------------------------------ the legend
    def _legend_rebuild(self):
        order: dict[str, list[Trace]] = {}
        for t in self.traces:
            if t.legend:
                order.setdefault(t.legend, []).append(t)
        self._legend = order

    def toggle_series(self, key: str, visible: bool | None = None):
        ts = self._legend.get(key, [])
        if not ts:
            return
        v = (not ts[0].visible) if visible is None else visible
        for t in ts:
            t.set_visible(v)
        self.update()

    # ------------------------------------------------------------ painting
    def _paint_back(self, p: QPainter):
        p.fillRect(QRectF(0, 0, self.width(), self.height()), BG)
        for pn in self.panels:
            r = pn.plot
            p.fillRect(r, PLOT)
            p.setPen(QPen(GRID, 1, Qt.DotLine))
            for v, _t in pn.yticks:
                y = _snap(pn.fy(v))
                if r.top() <= y <= r.bottom():
                    p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            for v, _t in self.xticks:
                x = _snap(float(pn.fx(v)))
                if r.left() <= x <= r.right():
                    p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(QPen(FRAME, 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(_snap(r.left()), _snap(r.top()), round(r.width()), round(r.height())))

    def _paint_front(self, p: QPainter):
        if not self.panels:
            return
        f = self._font()
        p.setFont(f)
        fm = QFontMetricsF(f)
        lh = fm.height()
        for pn in self.panels:                                  # y labels, panel titles
            r = pn.plot
            p.setPen(DIM)
            for v, t in pn.yticks:
                y = pn.fy(v)
                if r.top() - 2 <= y <= r.bottom() + 2:
                    p.drawText(QRectF(0, y - lh / 2, r.left() - 7, lh), int(Qt.AlignRight | Qt.AlignVCenter), t)
            if pn.title:
                p.setPen(TEXT)
                p.drawText(QRectF(r.left(), r.top() - lh - 4, r.width(), lh + 2), int(Qt.AlignLeft | Qt.AlignVCenter),
                           fm.elidedText(pn.title, Qt.ElideRight, r.width()))
        last = self.panels[-1]
        r = last.plot
        p.setPen(DIM)
        for v, t in self.xticks:                # x labels under the last panel
            x = float(last.fx(v))
            if r.left() - 2 <= x <= r.right() + 2:
                p.drawText(QRectF(x - 40, r.bottom() + 4, 80, lh), int(Qt.AlignHCenter | Qt.AlignTop), t)
        if self.x_title:
            title = self.x_title + (f" ({self.x_unit})" if self.x_unit else "")
            if self.x_dates:
                title += " · " + self.x_dates
            p.drawText(QRectF(r.left(), r.bottom() + lh + 8, r.width(), lh + 2), int(Qt.AlignHCenter | Qt.AlignTop),
                       fm.elidedText(title, Qt.ElideRight, r.width()))
        self._paint_lines(p, fm)
        self.paint_overlay(p)
        if not self._export:
            self._paint_hover(p, fm)
        self._paint_head(p, fm)
        if self._lasso:
            a, b = self._lasso
            p.setPen(QPen(AMBER, 1, Qt.DashLine))
            p.setBrush(QColor(230, 162, 60, 40))
            p.drawRect(QRectF(a, b).normalized())

    def paint_overlay(self, p: QPainter):
        """For the charts that draw their own marks over the series (called with the font set)."""

    def _paint_lines(self, p: QPainter, fm: QFontMetricsF):
        right, row = -1e9, 0
        for v in sorted(self.vlines, key=lambda v: v.x):
            pn = v.panel
            x = float(pn.fx(v.x))
            if not (pn.plot.left() <= x <= pn.plot.right()):
                continue
            p.setPen(QPen(v.color, 1, Qt.DashLine))
            p.drawLine(QPointF(_snap(x), pn.plot.top()), QPointF(_snap(x), pn.plot.bottom()))
            if v.label:
                w = fm.horizontalAdvance(v.label)
                row = row + 1 if x - w / 2 < right + 4 else 0           # labels that would touch go one line lower
                right = x + w / 2
                p.setPen(DIM)
                p.drawText(QRectF(x - 40, pn.plot.top() + 2 + row * fm.height(), 80, fm.height()), int(Qt.AlignCenter),
                           v.label)
        for pn in self.panels:
            for h in pn.hlines:
                y = float(pn.fy(h.value))
                if not (pn.plot.top() - 1 <= y <= pn.plot.bottom() + 1):
                    continue
                p.setPen(QPen(h.color, h.width, Qt.DashLine))
                p.drawLine(QPointF(pn.plot.left(), y), QPointF(pn.plot.right(), y))
                if h.draggable:                                  # the grip, on the left edge
                    p.setPen(Qt.NoPen)
                    p.setBrush(h.color)
                    p.drawRoundedRect(QRectF(pn.plot.left() - 5, y - 6, 10, 12), 2, 2)
                if h.label:
                    txt = h.label(h.value)
                    w = fm.horizontalAdvance(txt) + 4
                    ty = y - fm.height() - 2 if y - fm.height() - 2 > pn.plot.top() else y + 3
                    p.setPen(h.color)
                    p.drawText(QRectF(pn.plot.right() - w - 4, ty, w, fm.height()), int(Qt.AlignRight | Qt.AlignVCenter),
                               txt)

    def _x_text(self, v: float, digits: int = 1) -> str:
        if self.x_kind == "time":
            return clock(v, True)
        if self.x_kind == "int":
            return num(v, 0)
        return num(v, self.x_digits if self.x_digits is not None else digits)

    def _paint_hover(self, p: QPainter, fm: QFontMetricsF):
        h = self._hover
        if h is None:
            return
        pn = h.panel
        r = pn.plot
        p.save()
        p.setClipRect(r.adjusted(-1, -1, 1, 1))
        p.setPen(QPen(CROSS, 1, Qt.DotLine))
        cx = float(pn.fx(h.snap_x)) if h.snap_x is not None else h.pos.x()
        for q in self.panels:                                    # the vertical line crosses every panel (shared x)
            p.setClipRect(q.plot.adjusted(-1, -1, 1, 1))
            p.drawLine(QPointF(_snap(cx), q.plot.top()), QPointF(_snap(cx), q.plot.bottom()))
        p.setClipRect(r.adjusted(-1, -1, 1, 1))
        if self.readout_mode == "xy" or h.snap_x is None:
            p.drawLine(QPointF(r.left(), _snap(h.pos.y())), QPointF(r.right(), _snap(h.pos.y())))
        p.restore()
        for t, i, y in h.items:                                  # a ring on every sample read
            if t.kind in ("hist",):
                continue
            x = t.x[i] if i is not None else h.x
            q = t.panel
            px, py = float(q.fx(x)), float(q.fy(y))
            if not (q.plot.left() - 1 <= px <= q.plot.right() + 1 and q.plot.top() - 1 <= py <= q.plot.bottom() + 1):
                continue
            p.setPen(QPen(QColor("#ffffff"), 1.6))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(px, py), 5.5, 5.5)
        # the cursor's values on the axes
        last = self.panels[-1]
        xv = h.snap_x if h.snap_x is not None else h.x
        self._tag(p, fm, self._x_text(xv, max(1, digits_of(nice_step(pn.x1 - pn.x0, 40)))),
                  QPointF(min(max(cx, last.plot.left() + 20), last.plot.right() - 20), last.plot.bottom() + 4),
                  "top")
        if self.readout_mode == "xy" or h.snap_x is None:
            d = pn.y_digits(nice_step(pn.y1 - pn.y0, 40))
            self._tag(p, fm, num(h.y, d), QPointF(r.left() - 3, h.pos.y()), "left")
        header, lines = self.readout_lines(h)
        if header or lines:
            self._paint_box(p, fm, h.pos, header, lines)

    def _tag(self, p, fm, text, at: QPointF, side: str):
        w, hh = fm.horizontalAdvance(text) + 10, fm.height() + 3
        box = QRectF(at.x() - w / 2, at.y(), w, hh) if side == "top" else QRectF(at.x() - w, at.y() - hh / 2, w, hh)
        box.moveLeft(min(max(box.left(), 1.0), self.width() - w - 1.0))               # always inside the widget
        p.setPen(QPen(TIP_LINE, 1))
        p.setBrush(TIP_BG)
        p.drawRoundedRect(box, 3, 3)
        p.setPen(TEXT)
        p.drawText(box, int(Qt.AlignCenter), text)

    def _paint_box(self, p, fm, pos: QPointF, header: str, lines: list):
        rows = [(c, a, b) for c, a, b in lines]
        wl = max([fm.horizontalAdvance(a) for _c, a, _b in rows] + [0.0])
        wv = max([fm.horizontalAdvance(b) for _c, _a, b in rows] + [0.0])
        fb = QFont(p.font())
        fb.setBold(True)
        wh = QFontMetricsF(fb).horizontalAdvance(header) if header else 0.0
        w = max(wh, 14 + wl + (14 if wv else 0) + wv) + 18
        lh = fm.height() + 3
        hgt = (lh + 4 if header else 0) + lh * len(rows) + 12
        x, y = pos.x() + 16, pos.y() + 16
        if x + w > self.width() - 4:
            x = pos.x() - w - 16
        if y + hgt > self.height() - 4:
            y = pos.y() - hgt - 16
        box = QRectF(max(x, 4), max(y, 4), w, hgt)
        p.setPen(QPen(TIP_LINE, 1))
        p.setBrush(QColor(38, 42, 51, 238))
        p.drawRoundedRect(box, 5, 5)
        yy = box.top() + 6
        if header:
            p.setPen(TEXT)
            f = p.font()
            f.setBold(True)
            p.setFont(f)
            p.drawText(QRectF(box.left() + 9, yy, w - 18, lh), int(Qt.AlignLeft | Qt.AlignVCenter), header)
            f.setBold(False)
            p.setFont(f)
            yy += lh + 4
        for c, a, b in rows:
            if c is not None:
                p.setPen(Qt.NoPen)
                p.setBrush(c)
                p.drawEllipse(QPointF(box.left() + 14, yy + lh / 2 - 0.5), 3.5, 3.5)
            p.setPen(DIM if b else TEXT)
            p.drawText(QRectF(box.left() + 24, yy, wl + 2, lh), int(Qt.AlignLeft | Qt.AlignVCenter), a)
            if b:
                p.setPen(TEXT)
                p.drawText(QRectF(box.right() - 9 - wv - 2, yy, wv + 2, lh), int(Qt.AlignRight | Qt.AlignVCenter), b)
            yy += lh

    def _paint_head(self, p: QPainter, fm: QFontMetricsF):
        """The title at the left, the legend and the colour bar at the right of the controls."""
        self._legend_hits = []
        self._title_rect = QRectF()
        W = self.width()
        x_right = (self._ctrl_left or W - 12.0) - 4.0
        if not (self.title or self._legend or self.colorbar):
            return
        y0 = 4.0
        hh = self._head_h() - 8.0
        items = []
        for key, ts in self._legend.items():
            w = 10 + 6 + fm.horizontalAdvance(key)
            items.append((key, ts, w))
        if self.colorbar:
            cap, lo, hi = self.colorbar
            wcb = fm.horizontalAdvance(lo) + 6 + 84 + 6 + fm.horizontalAdvance(hi) + 10 + fm.horizontalAdvance(cap)
            x = x_right - wcb
            p.setPen(DIM)
            p.drawText(QRectF(x, y0, fm.horizontalAdvance(cap), hh), int(Qt.AlignLeft | Qt.AlignVCenter), cap)
            x += fm.horizontalAdvance(cap) + 10
            p.drawText(QRectF(x, y0, fm.horizontalAdvance(lo) + 2, hh), int(Qt.AlignLeft | Qt.AlignVCenter), lo)
            x += fm.horizontalAdvance(lo) + 6
            bar = QRectF(x, y0 + hh / 2 - 4, 84, 8)
            g = QLinearGradient(bar.left(), 0, bar.right(), 0)
            for k in range(len(RAMP)):
                g.setColorAt(k / (len(RAMP) - 1), RAMP[k])
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(g))
            p.drawRoundedRect(bar, 3, 3)
            x += 84 + 6
            p.setPen(DIM)
            p.drawText(QRectF(x, y0, fm.horizontalAdvance(hi) + 2, hh), int(Qt.AlignLeft | Qt.AlignVCenter), hi)
            x_right -= wcb + 16
        x = x_right
        for key, ts, w in reversed(items):
            x -= w
            on = ts[0].visible
            rect = QRectF(x - 4, y0, w + 8, hh)
            self._legend_hits.append((rect, key))
            col = ts[0].color
            p.setPen(QPen(col, 1.6))
            p.setBrush(col if on else Qt.NoBrush)
            p.drawEllipse(QRectF(x, y0 + hh / 2 - 4.5, 9, 9))
            p.setPen(TEXT if on else QColor("#5d6470"))
            p.drawText(QRectF(x + 16, y0, w - 14, hh), int(Qt.AlignLeft | Qt.AlignVCenter), key)
            x -= 16
        self._legend_hits.reverse()                                  # left to right, as they read
        if self.title:
            p.setPen(TEXT)
            f = p.font()
            f.setBold(True)
            p.setFont(f)
            fb = QFontMetricsF(f)
            avail = max(x - 8 - 14, 40)
            shown = fb.elidedText(self.title, Qt.ElideRight, avail)
            self._title_rect = QRectF(14, y0, fb.horizontalAdvance(shown), hh)
            p.drawText(QRectF(14, y0, avail, hh), int(Qt.AlignLeft | Qt.AlignVCenter), shown)
            f.setBold(False)
            p.setFont(f)

    # ------------------------------------------------------------ the readout
    def _panel_at(self, pos: QPointF) -> Panel | None:
        for p in self.panels:
            if p.plot.contains(pos):
                return p
        return None

    def _probe(self, pos: QPointF) -> _Hover | None:
        pn = self._panel_at(pos)
        if pn is None:
            return None
        cx, cy = pn.vx(pos.x()), pn.vy(pos.y())
        items: list = []
        snap_x = None
        if self.readout_mode == "x":
            for t in self.traces:
                if not (t.visible and t.readout) or t.kind in ("hist",):
                    continue
                ok = t.finite()
                if not ok.any():
                    continue
                if t.interp:
                    xs, ys = t.x[ok], t.y[ok]
                    if xs.min() <= cx <= xs.max():
                        order = np.argsort(xs)
                        items.append((t, None, float(np.interp(cx, xs[order], ys[order]))))
                    continue
                px = t.panel.fx(t.x)
                d = np.where(ok, np.abs(px - pos.x()), np.inf)
                i = int(np.argmin(d))
                if d[i] <= self.snap_px:
                    items.append((t, i, float(t.y[i])))
            sampled = [it for it in items if it[1] is not None]
            if sampled:
                snap_x = float(sampled[0][0].x[sampled[0][1]])
            for t in pn.traces:                                    # a histogram: the bin under the cursor
                if t.visible and t.kind == "hist" and t.edges is not None:
                    i = int(np.searchsorted(t.edges, cx)) - 1
                    if 0 <= i < len(t.y):
                        items.append((t, i, float(t.y[i])))
        else:
            best = None
            for t in pn.traces:
                if not (t.visible and t.readout) or t.kind in ("hist",):
                    continue
                ok = t.finite()
                if not ok.any():
                    continue
                d = np.hypot(t.panel.fx(t.x) - pos.x(), t.panel.fy(t.y) - pos.y())
                d = np.where(ok, d, np.inf)
                i = int(np.argmin(d))
                if d[i] <= self.snap_px and (best is None or d[i] < best[0]):
                    best = (float(d[i]), t, i)
            if best is not None:
                _d, t, i = best
                items.append((t, i, float(t.y[i])))
        return _Hover(pn, pos, items, cx, cy, snap_x)

    def readout_lines(self, h: _Hover):
        """(header, [(colour, label, value)]) of the box that follows the cursor; charts with their own words override."""
        if not h.items:
            return "", []
        if self.readout_mode == "x":
            first = h.items[0]
            xv = first[0].x[first[1]] if first[1] is not None else h.x
            if first[0].kind == "hist" and first[0].edges is not None:
                i = first[1]
                e = first[0].edges
                head = f"{num(e[i], 2)} – {num(e[i + 1], 2)}" + (f" {first[0].panel.unit}" if first[0].panel.unit else "")
            elif self.x_kind == "int" and self.x_short:
                head = f"{self.x_short} {num(xv, 0)}"
            else:
                head = self._x_text(xv, 2) + (f" {self.x_unit}" if self.x_unit and self.x_kind != "time" else "")
            for k in first[0].show_extra:
                if first[1] is not None and k in first[0].extra and str(first[0].extra[k][first[1]]):
                    head += " · " + str(first[0].extra[k][first[1]])
            lines = []
            for t, _i, y in h.items:
                if t.kind == "hist":
                    lines.append((t.color, t.name, num(y, 0)))
                else:
                    lines.append((t.color, t.name, t.y_text(y)))
            return head, lines
        t, i, y = h.items[0]
        head_parts = [str(t.extra[k][i]) for k in t.show_extra if k in t.extra and i is not None
                      and str(t.extra[k][i])]
        xt = f"{self.x_title or 'x'}: {self._x_text(float(t.x[i]), 2)}" + (f" {self.x_unit}" if self.x_unit else "")
        return " · ".join(head_parts), [(t.color, xt, ""), (t.color, t.panel.title or t.name, t.y_text(y))]

    # ------------------------------------------------------------ mouse
    def _hline_at(self, pos: QPointF) -> HLine | None:
        for pn in self.panels:
            if pn.plot.adjusted(-8, -6, 4, 6).contains(pos):
                for h in pn.hlines:
                    if h.draggable and abs(float(pn.fy(h.value)) - pos.y()) < 7:
                        return h
        return None

    def _legend_at(self, pos: QPointF) -> str | None:
        for r, key in self._legend_hits:
            if r.contains(pos):
                return key
        return None

    def enable_lasso(self, on: bool = True):
        """Shift + drag draws a rectangle and emits ``rubberBand`` when it is dropped."""
        self._lasso_on = on

    def mousePressEvent(self, e):                                        # noqa: N802
        pos = e.position()
        if e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)
        key = self._legend_at(pos)
        if key is not None:
            self.toggle_series(key)
            return
        h = self._hline_at(pos)
        if h is not None:
            self._drag = ("line", h)
            self._hover = None
            return
        if self._lasso_on and e.modifiers() & Qt.ShiftModifier:
            self._drag = ("lasso", pos)
            self._lasso = (pos, pos)
            return
        pn = self._panel_at(pos)
        if pn is not None:
            self._drag = ("pan", pos, {id(q): (q.x0, q.x1, q.y0, q.y1) for q in self.panels}, pn, False)

    def mouseMoveEvent(self, e):                                         # noqa: N802
        pos = e.position()
        d = self._drag
        if d and d[0] == "line":
            h = d[1]
            pn = h.panel
            v = pn.vy(min(max(pos.y(), pn.plot.top()), pn.plot.bottom()))
            h.value = float(v)
            self.lineMoved.emit(h.name, h.value)
            self.update()
        elif d and d[0] == "lasso":
            self._lasso = (d[1], pos)
            self.update()
        elif d and d[0] == "pan":
            _k, start, ranges, pn, moved = d
            if not moved and (pos - start).manhattanLength() < 4:
                return
            self._drag = ("pan", start, ranges, pn, True)
            self.setCursor(Qt.ClosedHandCursor)
            self._hover = None
            x0, x1, y0, y1 = ranges[id(pn)]
            dx = -(pos.x() - start.x()) / (pn.plot.width() or 1.0) * (x1 - x0)
            dy = (pos.y() - start.y()) / (pn.plot.height() or 1.0) * (y1 - y0)
            for q in self.panels:
                qx0, qx1 = ranges[id(q)][:2]
                q.x0, q.x1 = self._clamp(q, qx0 + dx, qx1 + dx, 0)
            qy0, qy1 = ranges[id(pn)][2:]
            pn.y0, pn.y1 = self._clamp(pn, qy0 + dy, qy1 + dy, 1)
            self._after_zoom()
        else:
            self._hover = self._probe(pos)
            self.setCursor(Qt.SizeVerCursor if self._hline_at(pos) is not None else
                           (Qt.PointingHandCursor if self._legend_at(pos) is not None or
                            (self.pickable and self._hover is not None and self._hover.items) else Qt.ArrowCursor))
            self.update()

    def mouseReleaseEvent(self, e):                                      # noqa: N802
        d, self._drag = self._drag, None
        pos = e.position()
        if d and d[0] == "line":
            self.lineReleased.emit(d[1].name, d[1].value)
        elif d and d[0] == "lasso":
            a, b = self._lasso or (pos, pos)
            self._lasso = None
            r = QRectF(a, b).normalized()
            if r.width() > 6 and r.height() > 6:
                self.rubberBand.emit(r, e.globalPosition().toPoint())
        elif d and d[0] == "pan":
            if d[4]:
                self.setCursor(Qt.ArrowCursor)
            elif self.pickable:
                hv = self._probe(pos)
                if hv is not None and hv.items:
                    t, i, _y = hv.items[0]
                    if i is not None:
                        self.pointClicked.emit(t, int(i))
        self.update()

    def mouseDoubleClickEvent(self, e):                                  # noqa: N802
        self._drag = None
        self.reset_view()

    def leaveEvent(self, e):                                             # noqa: N802
        self._hover = None
        self.update()
        super().leaveEvent(e)

    def wheelEvent(self, e):                                             # noqa: N802
        pos = e.position()
        steps = e.angleDelta().y() / 120.0
        if not steps:
            return
        f = 0.8 ** steps
        mods = e.modifiers()
        pn = self._panel_at(pos)
        axis = None                                              # None: the plot's rule · "x" · "y"
        last = self.panels[-1]
        if pn is None:
            for q in self.panels:                                # over the labels of an axis: that axis
                if QRectF(0, q.plot.top(), q.plot.left(), q.plot.height()).contains(pos):
                    pn, axis = q, "y"
            if pn is None and QRectF(last.plot.left(), last.plot.bottom(), last.plot.width(), 40).contains(pos):
                pn, axis = last, "x"
        if pn is None:
            return
        if axis is None:
            if mods & Qt.ShiftModifier:
                axis = "x"
            elif mods & Qt.ControlModifier:
                axis = "y"
            else:
                axis = "x" if self.zoom_axes == "x" else "xy"
        if "x" in axis:
            c = pn.vx(pos.x())
            for q in self.panels:
                lo, hi = c - (c - q.x0) * f, c + (q.x1 - c) * f
                q.x0, q.x1 = self._clamp(q, lo, hi, 0)
        if "y" in axis:
            c = pn.vy(pos.y())
            lo, hi = c - (c - pn.y0) * f, c + (pn.y1 - c) * f
            pn.y0, pn.y1 = self._clamp(pn, lo, hi, 1)
        self._hover = self._probe(pos)
        self._after_zoom()
        e.accept()

    def _after_zoom(self):
        self._equalize()
        for p in self.panels:
            p.apply()
            p.yticks = self._yticks(p)
        self.xticks = self._xticks(self.panels[-1])
        wide = max([QFontMetricsF(self._font()).horizontalAdvance(t) for p in self.panels for _v, t in p.yticks] + [14])
        if abs(max(wide + 14, 36.0) - self._left) > 0.5:
            self._layout()
        else:
            self._place_controls(self.width())                     # the Reset button shows when the view is not home
        self.update()
        self.viewChanged.emit()

    def event(self, e):
        """No tooltip over the plot (the readout is there): one over the handles and the legend only."""
        if e.type() == QEvent.ToolTip:
            pos = QPointF(e.pos())
            if self._hline_at(pos) is not None and self.toolTip():
                QToolTip.showText(e.globalPos(), self.toolTip(), self)
            elif self._legend_at(pos) is not None:
                QToolTip.showText(e.globalPos(), _("Click to show or hide"), self)
            else:
                QToolTip.hideText()
            e.accept()
            return True
        return super().event(e)

    # ------------------------------------------------------------ export
    def run_menu(self, menu: QMenu, global_pos):
        """Show a menu at the mouse and return the action chosen (None: closed): the one place that blocks."""
        return menu.exec(global_pos)

    def build_menu(self) -> QMenu:
        """The context menu, built and not shown: a test (or a caller) reads and triggers its actions without exec."""
        m = QMenu(self)
        for text, cb in self.actions:
            m.addAction(text).triggered.connect(lambda _c=False, f=cb: f())
        if self.actions:
            m.addSeparator()
        a = m.addAction(_("Reset zoom"))
        a.setEnabled(self.is_zoomed())
        a.triggered.connect(self.reset_view)
        m.addSeparator()
        m.addAction(_("Export CSV…")).triggered.connect(lambda: self.export_csv())
        m.addAction(_("Export PNG…")).triggered.connect(lambda: self.export_png())
        return m

    def contextMenuEvent(self, e):                                       # noqa: N802
        self.run_menu(self.build_menu(), e.globalPos())

    def _default_path(self, ext: str) -> str:
        base = _slug(self.export_name or self.title or "chart") + "_" + _time.strftime("%Y%m%d")
        folder = _last_dir[0] if _last_dir else str(Path.home())
        return str(Path(folder) / f"{base}.{ext}")

    def csv_text(self) -> str:
        """The data of every series as text: "x" and a column per series when they share their x values, else a
        block of columns per series. French: ";" and decimal commas (a French spreadsheet opens it as it is)."""
        fr = language() == "fr"
        sep = ";" if fr else ","

        def cell(v) -> str:
            if isinstance(v, (float, np.floating)):
                if not np.isfinite(v):
                    return ""
                s = f"{float(v):.9g}"
                return s.replace(".", ",") if fr else s
            s = "" if v is None else str(v)
            return f'"{s}"' if sep in s or '"' in s or "\n" in s else s

        def xcell(v) -> str:
            if self.x_kind == "time" and np.isfinite(v):
                return (EPOCH + timedelta(seconds=float(v))).strftime("%Y-%m-%d %H:%M:%S")
            return cell(v)

        ts = [t for t in self.traces if t.export]
        if not ts:
            return ""
        xname = self.x_title or "x"
        if self.x_unit and self.x_kind != "time":
            xname += f" ({self.x_unit})"

        def yname(t: Trace) -> str:
            return t.name + (f" ({t.unit})" if t.unit else "")

        wide = all(t.kind != "hist" for t in ts) and all(
            len(t.x) == len(ts[0].x) and np.allclose(t.x, ts[0].x, equal_nan=True) for t in ts)
        rows: list[list[str]] = []
        if wide:
            extras: dict[str, list] = {}
            for t in ts:
                for k, v in t.extra.items():
                    if len(v) == len(t.x) and k != self.x_title:         # the time of a frame is the x column already
                        extras.setdefault(k, v)
            head = [xname] + list(extras) + [yname(t) for t in ts]
            for i in range(len(ts[0].x)):
                rows.append([xcell(ts[0].x[i])] + [cell(v[i]) for v in extras.values()] + [cell(t.y[i]) for t in ts])
        else:
            cols: list[tuple[str, list[str]]] = []
            for t in ts:
                if t.kind == "hist" and t.edges is not None:
                    cols += [(f"{t.name} · {_('from')}", [cell(v) for v in t.edges[:-1]]),
                             (f"{t.name} · {_('to')}", [cell(v) for v in t.edges[1:]]),
                             (f"{t.name} · {_('count')}", [cell(v) for v in t.y])]
                    continue
                cols.append((f"{t.name} · {xname}", [xcell(v) for v in t.x]))
                cols += [(f"{t.name} · {k}", [cell(x) for x in v]) for k, v in t.extra.items() if len(v) == len(t.x)]
                cols.append((f"{t.name} · {_('value')}" + (f" ({t.unit})" if t.unit else ""), [cell(v) for v in t.y]))
            head = [c[0] for c in cols]
            for i in range(max(len(c[1]) for c in cols)):
                rows.append([c[1][i] if i < len(c[1]) else "" for c in cols])
        return "\r\n".join(sep.join(cell(h) for h in head) if k == 0 else sep.join(r)
                           for k, r in enumerate([head] + rows)) + "\r\n"

    def export_csv(self, path: str | None = None) -> str | None:
        """Write the data as CSV; without ``path`` the user chooses the file. Returns the file written."""
        if path is None:
            path, _f = QFileDialog.getSaveFileName(self, _("Export CSV…"), self._default_path("csv"),
                                                   _("CSV files (*.csv)"))
        if not path:
            return None
        Path(path).write_text(self.csv_text(), encoding="utf-8-sig" if language() == "fr" else "utf-8", newline="")
        _last_dir[:] = [str(Path(path).parent)]
        return str(path)

    def render_image(self, scale: float = 2.0) -> QImage:
        """The chart as a picture (without the cursor), ``scale`` times the size on screen."""
        img = QImage(int(self.width() * scale), int(self.height() * scale), QImage.Format_ARGB32_Premultiplied)
        img.setDevicePixelRatio(scale)
        img.fill(BG)
        saved, self._hover, self._export = self._hover, None, True
        was = self._reset_btn.isVisibleTo(self)
        self._reset_btn.hide()
        self._csv_btn.hide()
        try:
            self.render(img)
        finally:
            self._hover, self._export = saved, False
            self._reset_btn.setVisible(was)
            self._csv_btn.show()
        return img

    def export_png(self, path: str | None = None) -> str | None:
        if path is None:
            path, _f = QFileDialog.getSaveFileName(self, _("Export PNG…"), self._default_path("png"),
                                                   _("PNG images (*.png)"))
        if not path:
            return None
        self.render_image().save(str(path), "PNG")
        _last_dir[:] = [str(Path(path).parent)]
        return str(path)

    def update(self):                                                    # noqa: D102 - Qt's name
        """Repaint the chart: the QGraphicsView inside has its own viewport to refresh."""
        super().update()
        canvas = self.__dict__.get("_canvas")
        if canvas is not None:
            canvas.viewport().update()

    def sizeHint(self):                                                  # noqa: N802
        return QSize(680, 380)
