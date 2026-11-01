"""The interactive charts (client/chartview.py, client/charts.py), offscreen.

Synthetic data for the widget's behaviour (zoom, pan, readout, legend, CSV, PNG); the test project
F:/astro_essais/v2/heart_a7s_p10 (read only) for the Registration and Color charts. ASTRO_GPU_SETTINGS points to a
temporary file in every test, and the user's settings file is checked untouched. The screenshots go to
$TEMP/a2_shots (ASTRO_GPU_SHOTS_A2 names another folder), English and French.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")          # the offscreen platform has no fonts else

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent, QFontMetricsF, QImage, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QPushButton  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import charts, i18n, prefs  # noqa: E402
from astro_gpu.client import chartview as cv  # noqa: E402
from astro_gpu.client.i18n import _  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
F_PROJECT = Path("F:/astro_essais/v2/heart_a7s_p10")
HAVE_PROJECT = (F_PROJECT / "project.astroproj").is_file()
needs_project = pytest.mark.skipif(not HAVE_PROJECT, reason="the heart_a7s_p10 v2 test project is not on this machine")


def _user_settings() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "astro_gpu" / "settings.json"


def _fingerprint(p: Path):
    return (p.stat().st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest()) if p.exists() else None


def _snapshot(folder: Path):
    """What the charts must not change in the test project: names, sizes and dates of its small files."""
    return sorted((p.relative_to(folder).as_posix(), p.stat().st_mtime_ns, p.stat().st_size)
                  for p in folder.rglob("*") if p.is_file() and p.stat().st_size < 5_000_000)


USER_SETTINGS = _user_settings()
USER_BEFORE = _fingerprint(USER_SETTINGS)


def shots_dir() -> Path:
    d = Path(os.environ.get("ASTRO_GPU_SHOTS_A2") or Path(os.environ.get("TEMP", ".")) / "a2_shots")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def qapp():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS)
    return a


SHOWN: list = []


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, qapp):
    # nothing in this file may block on a modal menu or file dialog (offscreen, nobody to close them)
    monkeypatch.setattr(cv.ChartView, "run_menu", lambda self, menu, pos: None)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    yield {"tmp": tmp_path}
    while SHOWN:                                       # one window at a time: the mouse is the topmost one's
        SHOWN.pop().close()
    i18n.set_language("en")
    assert _fingerprint(USER_SETTINGS) == USER_BEFORE, "the user's settings file was touched"


@pytest.fixture(scope="module")
def project_data(qapp):
    """The Registration data of the test project, read once; the project is checked unchanged at the end."""
    if not HAVE_PROJECT:
        pytest.skip("the test project is not on this machine")
    before = _snapshot(F_PROJECT)
    p = api.open_project(F_PROJECT)
    yield p, charts.registration_data(p, 1)
    assert _snapshot(F_PROJECT) == before, "the charts changed the test project"


def pump():
    QApplication.processEvents()


def show(w, size=(900, 520)):
    while SHOWN:
        SHOWN.pop().close()
    SHOWN.append(w)
    w.resize(*size)
    w.show()
    w.grab()                                          # paints once: the legend and the title know where they are
    pump()
    return w


def at(w, fx: float, fy: float, panel: int = 0) -> QPoint:
    """A point of the plot of ``panel``, as fractions of its width and height."""
    r = w.panels[panel].plot
    return QPoint(round(r.left() + fx * r.width()), round(r.top() + fy * r.height()))


def move(w, pos: QPoint):
    """The mouse moves to ``pos`` over the chart (no button down)."""
    pos = QPointF(pos)
    QApplication.sendEvent(w, QMouseEvent(QEvent.MouseMove, pos, QPointF(w.mapToGlobal(pos.toPoint())), Qt.NoButton,
                                          Qt.NoButton, Qt.NoModifier))
    pump()


def wheel(w, pos: QPoint, steps: float = 1.0, mods=Qt.NoModifier):
    ev = QWheelEvent(QPointF(pos), QPointF(w.mapToGlobal(pos)), QPoint(0, 0), QPoint(0, int(120 * steps)),
                     Qt.NoButton, mods, Qt.NoScrollPhase, False)
    QApplication.sendEvent(w, ev)
    pump()


def drag(w, a: QPoint, b: QPoint, mods=Qt.NoModifier):
    QTest.mousePress(w, Qt.LeftButton, mods, a)
    QTest.mouseMove(w, QPoint((a.x() + b.x()) // 2, (a.y() + b.y()) // 2))
    QTest.mouseMove(w, b)
    QTest.mouseRelease(w, Qt.LeftButton, mods, b)
    pump()


def synthetic(zoom="xy", n=60, **kw) -> cv.ChartView:
    v = cv.ChartView("Test chart", zoom=zoom, export_name="synthetic", **kw)
    p = v.add_panel("Level (u)", "u", digits=1)
    v.set_x("Time", "s", kind="num")
    x = np.arange(n) * 10.0
    v.add_line("sine", x, 10 + 5 * np.sin(x / 50), panel=p, color=cv.BLUE, unit="u", digits=2)
    v.add_scatter("noise", x, 10 + np.cos(x / 70) * 3, panel=p, color=cv.AMBER, unit="u", digits=2)
    v.fit()
    return show(v, (800, 420))


def two_panels() -> cv.ChartView:
    """Two plots that share a clock axis: 80 frames, 15 s apart, from 21:30."""
    v = cv.ChartView("Two panels")
    p0 = v.add_panel("Top (px)", "px", digits=1)
    p1 = v.add_panel("Bottom (%)", "%", digits=0)
    t0 = datetime(2026, 8, 17, 21, 30)
    secs = np.array([(t0 + timedelta(seconds=15 * i) - cv.EPOCH).total_seconds() for i in range(80)])
    v.set_x(_("Time"), kind="time", dates="2026-08-17")
    v.add_scatter("top", secs, 4 + np.sin(np.arange(80) / 9) * 0.3, panel=p0, color=cv.BLUE, unit="px", digits=2)
    v.add_line("bottom", secs, 90 + np.arange(80) * 0.1, panel=p1, color=cv.AMBER, unit="%", digits=1)
    v.fit()
    return show(v, (800, 480))


# ---------------------------------------------------------------------------------------------- numbers and ticks
def test_ticks_are_nice_and_numbers_follow_the_language():
    assert cv.nice_step(10, 5) == 2 and cv.nice_step(7, 5) == 2 and cv.nice_step(0.9, 5) == 0.2
    assert cv.nice_step(480, 8) == 100 and cv.nice_step(0, 4) == 1.0
    assert cv.tick_values(0.35, 1.65, 0.5) == [0.5, 1.0, 1.5]
    assert cv.digits_of(0.5) == 1 and cv.digits_of(0.05) == 2 and cv.digits_of(2) == 0 and cv.digits_of(100) == 0
    assert cv.num(2.5, 1) == "2.5" and cv.num(-0.0004, 2) == "0.00" and cv.num(-3.26, 1) == "\u22123.3"
    i18n.set_language("fr")
    assert cv.num(2.5, 1) == "2,5" and cv.num(-12.34, 2) == "\u221212,34" and cv.num(float("nan"), 1) == "–"
    t = (datetime(2026, 8, 17, 23, 41, 5) - cv.EPOCH).total_seconds()
    assert cv.clock(t) == "23:41" and cv.clock(t, True) == "23:41:05"
    assert cv.to_seconds("2026-08-17T23:41:05") == t and cv.to_seconds("") is None and cv.to_seconds("soon") is None
    steps = [cv.time_step(span, 8) for span in (40, 900, 3 * 3600, 9 * 3600, 3 * 86400)]
    assert steps == sorted(steps) and steps[2] == 1800 and steps[3] == 7200


def test_the_labels_of_the_axes_never_touch_and_have_the_decimals_of_their_step():
    for lang in ("en", "fr"):
        i18n.set_language(lang)
        v = two_panels()
        fm = QFontMetricsF(v._font())
        px = [v.panels[-1].fx(t) for t, _l in v.xticks]
        w = max(fm.horizontalAdvance(label) for _t, label in v.xticks)
        assert len(px) >= 3 and min(np.diff(px)) > w + 6, (lang, v.xticks)
        assert all(re.fullmatch(r"\d\d:\d\d", label) for _t, label in v.xticks)            # 21:30, 21:35...
        for p in v.panels:
            ys = [p.fy(t) for t, _l in p.yticks]
            assert len(ys) >= 3 and min(abs(np.diff(ys))) >= fm.height() + 2
            assert all(fm.horizontalAdvance(label) < p.plot.left() - 6 for _t, label in p.yticks)
        v.close()
    i18n.set_language("fr")
    v = two_panels()
    labels = [lab for _t, lab in v.panels[0].yticks]
    assert all("," in lab and "." not in lab for lab in labels), labels                    # 3,8  4,0  4,2: a decimal comma
    for _i in range(12):                                                                    # zoom in a lot, y only
        wheel(v, at(v, 0.5, 0.5), 1.0, Qt.ControlModifier)
    labels = [lab for _t, lab in v.panels[0].yticks]
    assert len(set(labels)) == len(labels) and any(len(lab.split(",")[-1]) >= 2 for lab in labels), labels


# ---------------------------------------------------------------------------------------------- zoom, pan, reset
def test_the_wheel_zooms_around_the_cursor_and_a_double_click_goes_back():
    v = synthetic()
    p = v.panels[0]
    home = (p.x0, p.x1, p.y0, p.y1)
    assert not v.is_zoomed() and not v._reset_btn.isVisibleTo(v)
    c = at(v, 0.25, 0.5)
    under = p.vx(c.x())
    wheel(v, c, 2)
    assert v.is_zoomed() and v._reset_btn.isVisibleTo(v)
    assert p.x1 - p.x0 < 0.7 * (home[1] - home[0]) and p.y1 - p.y0 < 0.7 * (home[3] - home[2])
    assert abs(p.vx(c.x()) - under) < 1e-6 * (home[1] - home[0])               # the value under the cursor stays there
    zoomed = (p.x0, p.x1, p.y0, p.y1)
    wheel(v, c, 1, Qt.ShiftModifier)                                           # x only
    assert p.x1 - p.x0 < zoomed[1] - zoomed[0] and (p.y0, p.y1) == zoomed[2:]
    zoomed = (p.x0, p.x1, p.y0, p.y1)
    wheel(v, c, 1, Qt.ControlModifier)                                         # y only
    assert p.y1 - p.y0 < zoomed[3] - zoomed[2] and (p.x0, p.x1) == zoomed[:2]
    wheel(v, c, -40)                                                           # out and out: it stops at the data
    assert (p.x0, p.x1, p.y0, p.y1) == home and not v.is_zoomed()
    wheel(v, c, 3)
    QTest.mouseDClick(v, Qt.LeftButton, Qt.NoModifier, c)
    assert (p.x0, p.x1, p.y0, p.y1) == home and not v.is_zoomed()
    wheel(v, c, 3)
    v._reset_btn.click()
    assert (p.x0, p.x1, p.y0, p.y1) == home and not v._reset_btn.isVisibleTo(v)
    wheel(v, QPoint(int(p.plot.left()) - 20, int(p.plot.center().y())), 2)         # over the labels of an axis: that axis
    assert (p.x0, p.x1) == home[:2] and p.y1 - p.y0 < home[3] - home[2]
    v.reset_view()
    wheel(v, QPoint(int(p.plot.center().x()), int(p.plot.bottom()) + 12), 2)
    assert (p.y0, p.y1) == home[2:] and p.x1 - p.x0 < home[1] - home[0]


def test_a_time_series_zooms_in_time_only_and_its_plots_share_the_time_axis():
    v = two_panels()
    v.zoom_axes = "x"
    top, bot = v.panels
    y0 = (top.y0, top.y1, bot.y0, bot.y1)
    wheel(v, at(v, 0.5, 0.5, 1), 3)
    assert (top.y0, top.y1, bot.y0, bot.y1) == y0
    assert top.x0 == bot.x0 and top.x1 == bot.x1 and v.is_zoomed()
    assert top.plot.left() == bot.plot.left() and top.plot.right() == bot.plot.right()
    wheel(v, at(v, 0.5, 0.5, 0), 2, Qt.ControlModifier)                         # Ctrl: the plot under the cursor, in y
    assert top.y1 - top.y0 < y0[1] - y0[0] and (bot.y0, bot.y1) == y0[2:]
    for _i in range(10):                                                         # the labels get finer: minutes, seconds
        wheel(v, at(v, 0.5, 0.5, 0), 1)
    assert any(re.fullmatch(r"\d\d:\d\d:\d\d", lab) for _t, lab in v.xticks), v.xticks


def test_a_drag_pans_and_stays_on_the_data():
    v = synthetic()
    p = v.panels[0]
    wheel(v, at(v, 0.5, 0.5), 4)
    x0, x1, y0 = p.x0, p.x1, p.y0
    a = at(v, 0.5, 0.5)
    drag(v, a, a + QPoint(100, 30))
    shift = 100 / p.plot.width() * (x1 - x0)
    assert abs((x0 - p.x0) - shift) < 1e-6 * (x1 - x0) and p.y0 > y0           # the data followed the mouse
    assert abs((p.x1 - p.x0) - (x1 - x0)) < 1e-9 * (x1 - x0)                   # panned, not zoomed
    drag(v, a, a + QPoint(700, 0))
    drag(v, a, a + QPoint(700, 0))
    assert p.x0 >= p.home[0] - 0.05 * (p.home[1] - p.home[0]) - 1e-9           # it cannot leave the data behind
    v.reset_view()
    assert not v.is_zoomed()
    before = (p.x0, p.x1, p.y0, p.y1)                                          # a click without a move does not pan
    QTest.mouseClick(v, Qt.LeftButton, Qt.NoModifier, a)
    assert (p.x0, p.x1, p.y0, p.y1) == before


# ---------------------------------------------------------------------------------------------- the readout
def test_the_cursor_reads_the_samples_with_their_units_and_the_language_s_decimal_comma():
    v = two_panels()
    move(v, at(v, 0.4, 0.5))
    h = v._hover
    assert h is not None and len(h.items) == 2 and h.snap_x is not None          # both plots' samples at that time
    head, lines = v.readout_lines(h)
    assert re.fullmatch(r"\d\d:\d\d:\d\d", head)
    assert [ln[1] for ln in lines] == ["top", "bottom"]
    assert lines[0][2].endswith(" px") and lines[1][2].endswith(" %") and "." in lines[0][2]
    top = v.traces[0]
    i = h.items[0][1]
    assert abs(top.y[i] - float(lines[0][2].split()[0])) < 0.006
    assert abs(float(h.snap_x) - top.x[i]) < 1e-9                                # the crosshair sits on the sample
    i18n.set_language("fr")
    head, lines = v.readout_lines(v._probe(QPointF(at(v, 0.4, 0.5))))
    assert "," in lines[0][2] and "." not in lines[0][2], lines
    move(v, QPoint(2, 2))                                                        # outside every plot: nothing
    assert v._hover is None


def test_a_curve_is_read_between_its_samples():
    v = cv.ChartView("curve", readout="x")
    v.add_panel("v", "")
    v.set_x("lambda", "nm")
    v.add_line("c", [400, 410, 420], [0.0, 1.0, 0.5], interp=True, digits=2)
    v.fit()
    show(v)
    pn = v.panels[0]
    h = v._probe(QPointF(float(pn.fx(405.0)), pn.plot.center().y()))
    assert h.snap_x is None and abs(h.items[0][2] - 0.5) < 1e-9 and abs(h.x - 405.0) < 0.2


def test_the_legend_shows_and_hides_the_series_and_the_readout_follows():
    v = synthetic()
    assert [k for _r, k in v._legend_hits] == ["sine", "noise"]
    rect = dict((k, r) for r, k in v._legend_hits)["noise"]
    noise = v.traces[1]
    assert noise.visible and noise.qt[0].isVisible()
    QTest.mouseClick(v, Qt.LeftButton, Qt.NoModifier, rect.center().toPoint())
    assert not noise.visible and not noise.qt[0].isVisible()
    move(v, at(v, 0.3, 0.5))
    assert [it[0].name for it in v._hover.items] == ["sine"]
    assert not v.is_zoomed()                                                     # the click was not a pan
    v.toggle_series("noise")
    assert noise.visible
    assert [it[0].name for it in v._probe(QPointF(at(v, 0.3, 0.5))).items] == ["sine", "noise"]


def test_the_handles_are_dragged_and_report_where_they_are_dropped():
    v = two_panels()
    seen, dropped = [], []
    h = v.add_hline("limit", 95.0, panel=1, color=cv.RED, draggable=True, label=lambda x: f"limit {x:.0f}")
    v.lineMoved.connect(lambda n, x: seen.append((n, x)))
    v.lineReleased.connect(lambda n, x: dropped.append((n, x)))
    v.fit()
    bot = v.panels[1]
    start = QPoint(int(bot.plot.left()) + 40, round(float(bot.fy(95.0))))
    dest = QPoint(start.x(), round(float(bot.fy(92.0))))
    assert v._hline_at(QPointF(start)) is h
    drag(v, start, dest)
    assert len(dropped) == 1 and dropped[0][0] == "limit" and abs(h.value - 92.0) < 0.2 and len(seen) >= 2
    assert not v.is_zoomed()                                                     # the handle was taken, the view not panned
    drag(v, dest, QPoint(dest.x(), dest.y() + 600))                              # it cannot go below the plot
    assert h.value >= bot.y0 - 1e-9


# ---------------------------------------------------------------------------------------------- export
def test_csv_has_one_column_per_series_when_they_share_x_and_the_language_s_format(tmp_path):
    v = two_panels()
    en = v.csv_text()
    rows = en.strip().split("\r\n")
    assert rows[0] == "Time,top (px),bottom (%)" and len(rows) == 81
    first = rows[1].split(",")
    assert first[0] == "2026-08-17 21:30:00" and abs(float(first[1]) - v.traces[0].y[0]) < 1e-6
    f = v.export_csv(str(tmp_path / "e.csv"))
    assert Path(f).read_bytes()[:3] != b"\xef\xbb\xbf" and Path(f).read_bytes().decode("utf-8") == en
    assert b"\r\n" in Path(f).read_bytes()
    i18n.set_language("fr")
    v = two_panels()
    fr = v.csv_text()
    r = fr.strip().split("\r\n")
    assert r[0] == "Heure;top (px);bottom (%)" and "," in r[2] and ";" in r[2] and "." not in r[2].split(";", 1)[1]
    raw = Path(v.export_csv(str(tmp_path / "f.csv"))).read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf" and raw.decode("utf-8-sig") == fr           # a French spreadsheet reads it as it is


def test_csv_of_series_with_their_own_x_and_of_histograms_is_in_blocks():
    v = cv.ChartView("blocks")
    v.add_panel("a", "")
    v.set_x("x", "u")
    v.add_line("one", [1, 2, 3], [10, 20, 30], unit="m", extra={"note": ["a", "b,c", "d"]})
    v.add_line("two", [1, 2], [5, 6])
    v.add_histogram("h", [0, 1, 2, 3], [4, 5, 6])
    show(v)
    rows = v.csv_text().strip().split("\r\n")
    assert rows[0] == "one · x (u),one · note,one · value (m),two · x (u),two · value,h · from,h · to,h · count"
    assert len(rows) == 4
    assert rows[1] == "1,a,10,1,5,0,1,4" and rows[2] == '2,"b,c",20,2,6,1,2,5' and rows[3] == "3,d,30,,,2,3,6"


def test_png_export_is_the_chart_at_twice_the_size_without_the_cursor(tmp_path):
    v = synthetic()
    move(v, at(v, 0.5, 0.5))
    assert v._hover is not None
    f = v.export_png(str(tmp_path / "c.png"))
    img = QImage(f)
    assert (img.width(), img.height()) == (1600, 840)
    assert v._hover is not None                                                  # the screen keeps its cursor
    assert len({img.pixelColor(x, y).name() for x in range(0, 1600, 37) for y in range(0, 840, 29)}) > 8
    cross = v.panels[0].plot.center()                                            # where the crosshair was: no white ring
    ring = [img.pixelColor(int(2 * cross.x() + dx), int(2 * cross.y() + dy)).name() for dx in range(-14, 15) for dy in
            range(-14, 15) if 9 < (dx * dx + dy * dy) ** 0.5 < 12]
    assert "#ffffff" not in ring


def test_the_context_menu_exports_and_the_dialogs_name_a_default_file(monkeypatch, tmp_path):
    v = synthetic()
    v.add_action("Do the thing", lambda: None)
    asked, shown = [], []
    m = v.build_menu()                                                           # built, never shown: nothing can block
    assert [a.text() for a in m.actions() if not a.isSeparator()] == ["Do the thing", "Reset zoom", "Export CSV…",
                                                                      "Export PNG…"]
    assert not next(a for a in m.actions() if a.text() == "Reset zoom").isEnabled()
    # the right click goes through the same menu (the one place that would block is replaced)
    monkeypatch.setattr(cv.ChartView, "run_menu", lambda self, menu, pos: shown.append(menu))
    QApplication.sendEvent(v, QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(120, 120), v.mapToGlobal(QPoint(120, 120))))
    assert len(shown) == 1 and [a.text() for a in shown[0].actions()] == [a.text() for a in m.actions()]

    def fake(parent, title, start, filt):
        asked.append((title, Path(start).name, filt))
        return str(tmp_path / ("out." + ("csv" if "csv" in filt else "png"))), ""
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake))
    next(a for a in m.actions() if a.text() == "Export CSV…").trigger()
    next(a for a in m.actions() if a.text() == "Export PNG…").trigger()
    assert (tmp_path / "out.csv").is_file() and (tmp_path / "out.png").is_file()
    assert asked[0][1].startswith("synthetic_") and asked[0][1].endswith(".csv") and asked[1][1].endswith(".png")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a: ("", "")))      # cancelled
    assert v.export_csv() is None and v.export_png() is None
    i18n.set_language("fr")
    w = synthetic()
    assert [a.text() for a in w.build_menu().actions() if not a.isSeparator()] == [
        "Réinitialiser le zoom", "Exporter en CSV…", "Exporter en PNG…"]


# ---------------------------------------------------------------------------------------------- Selection
@dataclass
class Row:
    name: str
    time: str | None
    fwhm: float
    transparency: float
    status: str = ""
    excluded: bool = False


def frame_rows(n=30):
    rng = np.random.default_rng(7)
    t0 = datetime(2026, 8, 17, 21, 30)
    return [Row(f"DSC{2000 + i}.ARW", (t0 + timedelta(seconds=30 * i)).isoformat(),
                float(3.8 + 0.5 * (i > 20) + rng.normal(0, 0.1)), float(1.0 + 0.2 * (i < 8) + rng.normal(0, 0.03)),
                "not aligned" if i == 11 else "") for i in range(n)]


def frames_chart(by_hand=()) -> charts.FramesChart:
    rows = frame_rows(30)
    for i in by_hand:
        rows[i].excluded = True
    return show(charts.FramesChart(rows, 70.0, 4.0), (900, 520))


def test_the_frames_chart_keeps_its_old_surface():
    ch = frames_chart()
    assert ch.n == 30 and abs(ch.tmin - 70.0) < 1e-9 and ch.fmax == 4.0 and ch.sugg == (70.0, 4.0) and ch.hand == set()
    assert ch.p90 > 0 and len(ch.tr) == 30 and ch.current is None
    ex = ch.excluded()
    assert ex[11] and ex.sum() == ch.n - ch.kept()                              # the frame that did not align is out
    assert (ch.fwhm[ex & ch.aligned] > 4.0).sum() >= 1
    top, bot = ch._panels()
    (f0, f1), (t0, t1) = ch._ranges()
    assert top.top() < top.bottom() < bot.top() < bot.bottom() and f0 < 4.0 < f1 and t0 < 70.0 < t1
    pt = ch._xy(bot, 0, 90.0, t0, t1)
    assert bot.left() < pt.x() < bot.right() and abs(pt.y() - (bot.bottom() - (90 - t0) / (t1 - t0) * bot.height())) < 1e-6
    ch.current = 5
    assert ch.current == 5
    empty = show(charts.FramesChart([], 60.0, 2.0), (400, 300))
    assert empty.n == 0 and empty.kept() == 0


def test_the_thresholds_are_dragged_with_the_mouse_and_the_chart_says_once_when_they_are_dropped():
    ch = frames_chart()
    changed = []
    ch.changed.connect(lambda: changed.append((ch.tmin, ch.fmax)))
    kept0 = ch.kept()
    top, bot = ch._panels()
    (f0, f1), (t0, t1) = ch._ranges()
    start = ch._xy(bot, 0, ch.tmin, t0, t1)
    dest = ch._xy(bot, 0, 95.0, t0, t1)
    x = int(bot.left() + 30)
    QTest.mousePress(ch, Qt.LeftButton, Qt.NoModifier, QPoint(x, int(start.y())))
    QTest.mouseMove(ch, QPoint(x, int((start.y() + dest.y()) / 2)))
    assert changed == []                                                          # nothing is announced while it moves
    QTest.mouseMove(ch, QPoint(x, int(dest.y())))
    QTest.mouseRelease(ch, Qt.LeftButton, Qt.NoModifier, QPoint(x, int(dest.y())))
    assert len(changed) == 1 and 93 < ch.tmin < 97 and abs(changed[0][0] - ch.tmin) < 1e-9
    assert ch.kept() < kept0
    kept = int((~ch.excluded()).sum())
    assert sum(len(t.x) for k, t in ch._series.items() if k == ("t", "keep")) == kept          # the colours follow
    s2 = ch._xy(top, 0, ch.fmax, f0, f1)                                          # the star size handle
    d2 = ch._xy(top, 0, 3.9, f0, f1)
    QTest.mousePress(ch, Qt.LeftButton, Qt.NoModifier, QPoint(x, int(s2.y())))
    QTest.mouseMove(ch, QPoint(x, int(d2.y())))
    QTest.mouseRelease(ch, Qt.LeftButton, Qt.NoModifier, QPoint(x, int(d2.y())))
    assert len(changed) == 2 and 3.8 < ch.fmax < 4.0
    ch.reset_thresholds()
    assert (ch.tmin, ch.fmax) == (70.0, 4.0) and len(changed) == 3 and abs(ch.h_t.value - 70.0) < 1e-9


def test_a_click_picks_the_frame_and_a_shift_drag_excludes_or_keeps_the_frames_inside(monkeypatch):
    ch = frames_chart()
    picked, changed = [], []
    ch.picked.connect(picked.append)
    ch.changed.connect(lambda: changed.append(1))
    (f0, f1), _t = ch._ranges()
    top, _bot = ch._panels()
    pt = ch._xy(top, 6, ch.fwhm[6], f0, f1)
    QTest.mouseClick(ch, Qt.LeftButton, Qt.NoModifier, QPoint(round(pt.x()), round(pt.y())))
    assert picked == [6] and ch.current == 6
    head, lines = ch.readout_lines(ch._probe(pt))
    assert head == "Frame 7 · DSC2006.ARW" and [ln[1] for ln in lines] == ["Time", "FWHM", "Transparency", "State"]
    assert lines[0][2] == "21:33" and lines[1][2].endswith(" px") and lines[3][2] in ("kept", "excluded")
    # Shift + drag around frames 3 to 5 (indices 2..4): the menu offers to exclude them, and the chart tells it
    xs = [ch._xy(top, i, f0, f0, f1).x() for i in range(8)]
    a = QPoint(round((xs[1] + xs[2]) / 2), round(top.top() + 2))
    b = QPoint(round((xs[4] + xs[5]) / 2), round(top.bottom() - 2))
    monkeypatch.setattr(cv.ChartView, "run_menu",
                        lambda self, m, pos: next(x for x in m.actions() if x.text().startswith("Exclude")))
    drag(ch, a, b, Qt.ShiftModifier)
    assert ch.hand == {2, 3, 4} and changed == [1]
    assert all(ch.excluded()[i] for i in (2, 3, 4)) and not ch.is_zoomed()
    monkeypatch.setattr(cv.ChartView, "run_menu",
                        lambda self, m, pos: next(x for x in m.actions() if x.text().startswith("Keep")))
    drag(ch, a, b, Qt.ShiftModifier)
    assert ch.hand == set() and changed == [1, 1]
    monkeypatch.setattr(cv.ChartView, "run_menu", lambda self, m, pos: None)       # the menu closed without a choice
    drag(ch, a, b, Qt.ShiftModifier)
    assert ch.hand == set() and changed == [1, 1]
    ch.hand = {1}
    ch.reset()
    assert ch.hand == set() and (ch.tmin, ch.fmax) == (70.0, 4.0)


def test_the_frames_chart_has_a_time_axis_and_exports_its_frames():
    ch = frames_chart(by_hand=(3,))
    assert ch.hand == {3} and ch.has_time and ch.xmode == "frame"
    assert [x for _x, x in ch.xticks][:3] == ["5", "10", "15"]
    ch.set_xmode("time")
    assert ch.xmode == "time" and ch.x_kind == "time" and "2026-08-17" in ch.x_dates
    assert all(re.fullmatch(r"\d\d:\d\d(:\d\d)?", lab) for _t, lab in ch.xticks)
    assert np.allclose(ch.d_f.x, ch.secs)
    ch.set_xmode("frame")
    csv = ch.csv_text().strip().split("\r\n")
    assert csv[0] == "Frame number,Frame,Time,FWHM (px),Transparency (%)" and len(csv) == 31
    assert csv[1].split(",")[:3] == ["1", "DSC2000.ARW", "2026-08-17T21:30:00"]
    rows = frame_rows(10)                                                         # no times: no time axis
    rows[4].time = None
    plain = show(charts.FramesChart(rows, 70.0, 4.0))
    assert not plain.has_time and not plain.controls


# ---------------------------------------------------------------------------------------------- Color
def curves():
    lam = np.arange(400, 801, 10.0)

    def g(c, w):
        return np.exp(-0.5 * ((lam - c) / w) ** 2)
    return lam, np.stack([5e17 * g(600, 25) + 3e17 * g(670, 12), 4e17 * g(560, 30), 2e17 * g(480, 40)])


def test_the_spectral_chart_draws_the_curves_and_reads_them_between_the_samples():
    lam, S = curves()
    ch = show(charts.SpectralChart(lam, S), (900, 420))
    assert np.array_equal(ch.lam, lam) and np.array_equal(ch.S, S)
    assert [t.name for t in ch.traces] == ["Red", "Green", "Blue"] and len(ch.vlines) == 3
    assert max(t.y.max() for t in ch.traces) == 1.0                                # relative to the strongest
    pn = ch.panels[0]
    h = ch._probe(QPointF(float(pn.fx(605.0)), pn.plot.center().y()))
    assert abs(h.items[0][2] - 0.5 * (ch.traces[0].y[20] + ch.traces[0].y[21])) < 1e-9
    assert (pn.x0, pn.x1) == (400.0, 800.0) and pn.y1 > 1.0
    img = ch.grab().toImage()
    assert len({img.pixelColor(x, y).name() for x in range(60, 880, 7) for y in range(40, 360, 5)}) > 8
    tabs = charts.color_tabs(lam, S)
    assert [t for t, _w in tabs] == ["Spectral response", "Colour of the lines"]
    lines = show(tabs[1][1], (900, 420))
    share = np.nansum([t.y for t in lines.traces], axis=0)
    assert np.allclose(share[~np.isnan(lines.traces[0].y)], 100.0)               # the three shares make the whole signal
    assert np.isnan(lines.traces[0].y[-1])                                        # where the sensor sees nothing: no share


# ---------------------------------------------------------------------------------------------- Registration
@needs_project
def test_the_registration_data_come_from_the_stage_outputs(project_data):
    p, d = project_data
    assert d.n == 40 and d.has_time and d.ref == 20 and abs(d.scale - 6.3686) < 1e-4
    with np.load(Path(api.stage_outputs(p, 1, "register").dir) / "reg.npz") as z:
        coef, nmatch, nstars = z["coef"], z["nmatch"], z["nstars"]
    cache = json.loads((Path(api.stage_outputs(p, 1, "calibrate").dir) / "cache.json").read_text())
    assert np.allclose(d.dx, cache["width"] / 2 - coef[:, 0, 0]) and np.allclose(d.dy, cache["height"] / 2 - coef[:, 1, 0])
    assert abs(d.dx[d.ref]) < 1.5 and abs(d.dy[d.ref]) < 1.5                      # the reference is where it is
    assert np.array_equal(d.nmatch, nmatch) and np.array_equal(d.nstars, nstars)
    assert d.roundness is None                                                    # the stage does not measure it yet
    assert np.nanpercentile(d.transp, 90) == pytest.approx(100.0)
    assert d.dates == "2026-08-17" and d.iso[0] == "2026-08-17T00:25:13" and np.all(np.diff(d.secs) > 0)


@needs_project
def test_the_registration_charts_and_their_unit_and_time_choices(project_data):
    _p, d = project_data
    tabs = charts.registration_tabs(d)
    assert [t for t, _w in tabs] == ["Quality", "Drift", "Drift path", "Distribution"]
    quality, drift, track, dist = (show(w, (900, 520)) for _t, w in tabs)
    # quality: FWHM, stars, transparency (no roundness yet), time by default, frames on request
    assert [p.title for p in quality.panels] == ["Star size (FWHM, px)", "Stars per frame", "Transparency (% of the best frames)"]
    assert quality.xmode == "time" and quality.x_kind == "time"
    assert [b.text() for b in quality.controls] == ["Frame number", "Time"]
    quality.set_xmode("frame")
    assert quality.x_kind == "int" and np.array_equal(quality.traces[0].x, np.arange(40) + 1.0)
    move(quality, at(quality, 0.3, 0.3))
    head, lines = quality.readout_lines(quality._hover)
    assert head.startswith("Frame ") and ".ARW" in head and lines[0][1] == "FWHM"
    # drift: the user's unit first (arcsec by default), pixels on request, the other unit in brackets
    assert drift.unit == "arcsec" and [b.text() for b in drift.controls[:2]] == ["px", "arcsec"]
    tx = drift.t_x
    assert np.allclose(tx.y, d.dx * d.scale) and "arcsec" in drift.panels[0].title
    y0, y1 = drift.panels[0].y0, drift.panels[0].y1
    drift.set_unit("px")
    assert np.allclose(tx.y, d.dx) and tx.unit == "px" and "px" in drift.panels[0].title
    assert abs(drift.panels[0].y0 - y0 / d.scale) < 1e-6 and abs(drift.panels[0].y1 - y1 / d.scale) < 1e-6
    move(drift, at(drift, 0.3, 0.5))
    head, lines = drift.readout_lines(drift._hover)
    assert re.fullmatch(r"\d\d:\d\d:\d\d · DSC\d+\.ARW", head), head
    assert re.fullmatch(r"-?[\d.]+ px \(-?[\d.]+ arcsec\)", lines[0][2].replace("\u2212", "-")), lines
    drift.set_unit("arcsec")
    assert drift.unit == "arcsec" and np.allclose(tx.y, d.dx * d.scale)
    csv = drift.csv_text().strip().split("\r\n")
    assert csv[0].split(",")[0] == "Time" and "x offset (arcsec)" in csv[0] and len(csv) == 41
    # path: x against y with a pixel the same on both axes, time as colour, the reference frame marked
    p0 = track.panels[0]
    assert track.colorbar is not None and track.colorbar[0] == "Time" and len(track.t_pts.qt) == 24
    for size in ((900, 520), (640, 560), (1000, 400)):
        track.resize(*size)
        pump()
        assert abs((p0.x1 - p0.x0) / p0.plot.width() - (p0.y1 - p0.y0) / p0.plot.height()) < 1e-6 * (p0.x1 - p0.x0)
    wheel(track, at(track, 0.5, 0.5), 2)
    assert track.is_zoomed()
    assert abs((p0.x1 - p0.x0) / p0.plot.width() - (p0.y1 - p0.y0) / p0.plot.height()) < 1e-6 * (p0.x1 - p0.x0)
    track.reset_view()
    assert not track.is_zoomed()
    assert track.unit == "arcsec"
    h = track._probe(QPointF(float(p0.fx(d.dx[5] * d.scale)), float(p0.fy(d.dy[5] * d.scale))))
    head, lines = track.readout_lines(h)
    assert head.startswith("Frame 6 · ") and lines[1][1] == "x offset" and "px" in lines[1][2]
    track.set_unit("px")                                                          # the unit rescales data, axes and view
    assert np.allclose(track.t_pts.x, d.dx) and "px" in p0.title and track.x_unit == "px"
    assert not track.is_zoomed()
    # distribution: the bars count the frames
    assert dist.what == "fwhm" and dist.traces[0].y.sum() == 40
    dist.show_what("transp")
    assert dist.traces[0].y.sum() == 40 and dist.what == "transp" and len(dist.vlines) == 1
    move(dist, at(dist, 0.3, 0.3))
    head, lines = dist.readout_lines(dist._hover)
    assert "–" in head and lines[0][1] == "Frames" and lines[0][2].isdigit()


def test_the_charts_leave_out_what_the_data_cannot_give():
    d = charts.RegistrationData(["a", "b"], ["", ""], np.full(2, np.nan), np.array([3.0, 3.1]), np.array([100.0, 90]),
                                np.array([10.0, 11]), np.array([20.0, 20]), np.array([0.3, 0.4]), np.ones(2, bool),
                                np.full(2, np.nan), np.full(2, np.nan), 0, None)
    tabs = charts.registration_tabs(d)
    assert [t for t, _w in tabs] == ["Quality", "Distribution"]                   # no drift without the sensor's size
    assert not d.has_time and not tabs[0][1].controls
    for _t, w in tabs:
        show(w, (600, 400))


# ---------------------------------------------------------------------------------------------- French, shots
def readable(w: cv.ChartView):
    """Nothing overlaps: the labels of the axes, the legend, the title and the buttons."""
    w.grab()
    fm = QFontMetricsF(w._font())
    last = w.panels[-1]
    px = [float(last.fx(t)) for t, _l in w.xticks]
    wmax = max([fm.horizontalAdvance(label) for _t, label in w.xticks] + [0])
    assert len(px) < 2 or min(np.diff(px)) > wmax + 4, w.xticks
    for p in w.panels:
        ys = [float(p.fy(t)) for t, _l in p.yticks]
        assert len(ys) < 2 or min(abs(np.diff(ys))) >= fm.height()
        assert all(fm.horizontalAdvance(label) <= p.plot.left() - 4 for _t, label in p.yticks)
    rects = [r for r, _k in w._legend_hits] + [QRectF(b.geometry()) for b in w.controls]
    if w._reset_btn.isVisibleTo(w):
        rects.append(QRectF(w._reset_btn.geometry()))
    if not w._title_rect.isEmpty():
        rects.append(w._title_rect)
    rects.sort(key=lambda r: r.left())
    for a, b in zip(rects, rects[1:]):
        assert a.right() <= b.left() + 1, (a, b)
    assert all(r.right() <= w.width() and r.left() >= 0 for r in rects)


def every_chart(pd):
    lam, S = curves()
    out = [("frames", frames_chart(by_hand=(3,)))]
    if pd is not None:
        p, d = pd
        out += [(n, w) for n, (_t, w) in zip(("quality", "drift", "path", "distribution"), charts.registration_tabs(d))]
        sp = api.spectral_response(p, 1)
        lam, S = sp.lam, sp.S
    out += [(n, w) for n, (_t, w) in zip(("spectral", "lines"), charts.color_tabs(lam, S))]
    return out


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_every_chart_in_both_languages_has_its_french_readable_axes_and_a_screenshot(lang):
    pd = None
    if HAVE_PROJECT:
        p = api.open_project(F_PROJECT)
        pd = (p, charts.registration_data(p, 1))
    i18n.set_language(lang)
    made = every_chart(pd)
    assert len(made) == (7 if pd else 3)
    for name, w in made:
        show(w, (900, 520))
        pn = w.panels[0]
        pos = QPoint(int(pn.plot.left() + 0.35 * pn.plot.width()), int(pn.plot.top() + 0.5 * pn.plot.height()))
        if name == "path":
            pos = QPoint(int(pn.fx(w.t_pts.x[8])), int(pn.fy(w.t_pts.y[8])))
        move(w, pos)
        assert w._hover is not None and w._hover.items, name
        head, lines = w.readout_lines(w._hover)
        assert lines, name
        readable(w)
        f = shots_dir() / f"{name}_{lang}.png"
        w.grab().save(str(f))
        assert f.stat().st_size > 5000
        wheel(w, pos, 2)                                                           # a zoomed shot, with its Reset button
        readable(w)
        w.grab().save(str(shots_dir() / f"{name}_zoomed_{lang}.png"))
        w.reset_view()
        if lang == "fr":
            labels = [lab for p_ in w.panels for _t, lab in p_.yticks] + [lab for _t, lab in w.xticks]
            labels += [x for ln in lines for x in (ln[1], ln[2])] + [head]
            labels += [b.text() for b in w.findChildren(QPushButton)]
            assert not [t for t in labels if re.search(r"\d\.\d", t)], (name, labels)
    assert not i18n.missing(), sorted(i18n.missing())


def test_the_modules_import_only_what_the_views_may_and_every_text_is_translated():
    allowed = {"__future__", "os", "sys", "threading", "collections", "dataclasses", "pathlib", "typing", "weakref",
               "numpy", "PySide6", "PIL", "astropy", "datetime", "json", "html", "warnings", "traceback", "re"}
    cat: dict[str, str] = {}
    for f in sorted((ROOT / "astro_gpu" / "client" / "locale" / "fr").glob("*.json")):
        cat.update(json.loads(f.read_text(encoding="utf-8")))
    for name in ("charts", "chartview"):
        tree = ast.parse((ROOT / "astro_gpu" / "client" / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if name == "charts" and isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert node.module.split(".")[0] in allowed, node.module
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_" and node.args \
                    and isinstance(node.args[0], ast.Constant):
                assert node.args[0].value in cat, (name, node.lineno, node.args[0].value)
    mine = json.loads((ROOT / "astro_gpu" / "client" / "locale" / "fr" / "charts.json").read_text(encoding="utf-8"))
    assert len(mine) > 30 and all(isinstance(v, str) and v for v in mine.values())


def test_a_chart_is_a_plain_widget_that_takes_the_mouse_itself():
    """The QGraphicsView inside lets the events through: a parent, a test or a layout talks to the chart."""
    v = synthetic()
    assert v._canvas.testAttribute(Qt.WA_TransparentForMouseEvents) and v.hasMouseTracking()
    assert v.childAt(at(v, 0.5, 0.5)) in (None, v)
    assert v.sizeHint().width() > 300 and v.minimumSize().height() >= 200
