"""P10: Inspect and Frames, offscreen, on F:/astro_essais/v2/heart_a7s_p10 (a copy of the v2 test project with adopted
results: all 15 stages of Full done).

The model is read-only (nothing is saved into the project). The window is the real shell (``client/app.Main``) with the
views of P10 mounted in its three mount points, as P13 will do. ASTRO_GPU_SETTINGS points to a temporary file in every
test; the user's settings file is checked untouched. ASTRO_GPU_SHOTS names the folder of the screenshots (default:
$TEMP/p10_shots).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")                 # the GPUs are the runs' (DESIGN §6.1): these tests use the CPU
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")      # the offscreen platform has no fonts else

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPoint, Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabBar,
    QTableWidget,
    QWidget,
)

from astro_gpu import api, texts  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.charts import FramesChart, SpectralChart  # noqa: E402
from astro_gpu.client.frames import FramesView, GroupingDialog, check_lines, night_text  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services import inspect as si  # noqa: E402
from astro_gpu.client.sources import ImportDialog, SourcesView, import_folder  # noqa: E402
from astro_gpu.client.steps import Steps  # noqa: E402
from astro_gpu.client.tune import Histogram, TuneView, sky_text  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
F_PROJECT = Path("F:/astro_essais/v2/heart_a7s_p10")
DATA = Path("F:/astro_essais/donnees/heart_a7s")
pytestmark = pytest.mark.skipif(not (F_PROJECT / "project.astroproj").is_file(),
                                reason="the heart_a7s_p10 v2 test project is not on this machine")


def _user_settings() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "astro_gpu" / "settings.json"


def _fingerprint(p: Path):
    return (p.stat().st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest()) if p.exists() else None


def _snapshot(folder: Path):
    """The definition and the manifests of the test project (its store is big: names, sizes and dates of the small files)."""
    return sorted((p.relative_to(folder).as_posix(), p.stat().st_mtime_ns, p.stat().st_size)
                  for p in folder.rglob("*") if p.is_file() and (p.suffix in (".json", ".astroproj", ".jpg")))


USER_SETTINGS = _user_settings()
USER_BEFORE = _fingerprint(USER_SETTINGS)
PROJECT_BEFORE = _snapshot(F_PROJECT) if F_PROJECT.is_dir() else None


def shots_dir() -> Path:
    d = Path(os.environ.get("ASTRO_GPU_SHOTS") or Path(os.environ.get("TEMP", ".")) / "p10_shots")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def qapp():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    appmod.apply_language("en")
    shown = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, _k=kind, **k: shown.append((_k, a[1:3])))
    yield {"tmp": tmp_path, "shown": shown}
    appmod.apply_language("en")
    assert _fingerprint(USER_SETTINGS) == USER_BEFORE, "the user's settings file was touched"


def _events():
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def settle(win, n=8, timeout=120.0):
    """Process events until the workers are idle and the views have nothing left to build."""
    end = time.monotonic() + timeout
    for _i in range(n):
        win.worker.wait(30)
        win.tune.service.wait(max(end - time.monotonic(), 1))
        _events()
        time.sleep(0.02)
    win.worker.wait(30)
    win.tune.service.wait(max(end - time.monotonic(), 1))
    _events()


class Harness(appmod.Main):
    """The shell with the views of P10 in the Inspect mount point (Graph and Post stay placeholders)."""

    def __init__(self, readonly=True, **kw):
        self.readonly = readonly
        super().__init__(**kw)

    def graph_page(self):
        if self.s is None or self.s.p is not self.project.definition:           # a change of language keeps the model
            self.s = ProjectModel(self.project.definition, self.worker, readonly=self.readonly)
        return appmod.Placeholder("Graph")

    def inspect_page(self):
        s = self.s
        self.steps = Steps(s)
        self.browser = SourcesView(s)
        self.frames = FramesView(s, self.browser)
        self.tune = TuneView(s, self.steps, self.frames, appmod.Placeholder("Post"))
        self.tune.back.connect(lambda: self.set_mode(0))
        return self.tune

    def close_project(self, go_home=True):
        super().close_project(go_home)


def open_window(root: Path, size=(1680, 1000), readonly=True) -> Harness:
    w = Harness(readonly=readonly)
    w.resize(*size)
    w.show()
    w.open_project(root)
    assert w.s.ready or w.worker.wait(180)
    for _i in range(200):                                            # the model warms up in the worker
        if w.s.ready:
            break
        w.worker.wait(5)
        time.sleep(0.05)
    assert w.s.ready, w.s.error
    return w


def inspect(win: Harness, step: str, index: int | None = None):
    win.s.select(index=index, step=step)
    win.set_mode(1)
    settle(win)


def shot(win, name: str, lang: str = "en") -> Path:
    settle(win)
    out = shots_dir() / f"{name}_{lang}.png"
    win.grab().save(str(out))
    assert out.stat().st_size > 5000
    return out


@pytest.fixture(scope="module")
def module_win(qapp):
    """One window on the test project for the read-only tests (the load of an image is not repeated)."""
    prefs_dir = Path(os.environ.get("TEMP", ".")) / "p10_settings"
    prefs_dir.mkdir(exist_ok=True)
    os.environ["ASTRO_GPU_SETTINGS"] = str(prefs_dir / "settings.json")
    prefs.load()
    w = open_window(F_PROJECT)
    yield w
    w.close()
    os.environ.pop("ASTRO_GPU_SETTINGS", None)


@pytest.fixture()
def win(module_win, env):
    w = module_win
    appmod.apply_language("en")
    prefs.set("spec_mode", False)
    w.s.select(index=0, step="empilement")
    w.s.pending_change = None
    w.tune.cmp_pid, w.tune.img_tab, w.tune.show_hist = None, 0, False
    w.set_mode(1)
    settle(w)
    return w


def texts_of(root: QWidget) -> list[str]:
    out = []
    for w in [root] + root.findChildren(QWidget):
        if not w.isVisibleTo(root) and w is not root:
            continue
        if isinstance(w, QLabel):
            out.append(re.sub(r"<[^>]+>", "", w.text()))
        elif isinstance(w, QAbstractButton):
            out.append(w.text())
        elif isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())]
        elif isinstance(w, QTabBar):
            out += [w.tabText(i) for i in range(w.count())]
        elif isinstance(w, QTableWidget):
            out += [w.horizontalHeaderItem(c).text() for c in range(w.columnCount()) if w.horizontalHeaderItem(c)]
            out += [w.item(r, c).text() for r in range(min(w.rowCount(), 60)) for c in range(w.columnCount())
                    if w.item(r, c) is not None]
        elif isinstance(w, QListWidget):
            out += [w.item(i).text() for i in range(min(w.count(), 60))]
        if w.toolTip():
            out.append(w.toolTip())
    return [t.strip() for t in out if t and t.strip()]


# ---------------------------------------------------------------------------------------------- Inspect
def test_the_real_stack_shows_with_auto_stretch_histogram_and_catalogue_metrics(win):
    t = win.tune
    assert win.s.sel_step == "empilement" and t.pages.currentIndex() == 0
    sl = t._slots[0]
    assert sl.image is not None and sl.viewer is not None
    im = sl.image
    assert im.size == (6384, 4272) and im.lin is not None and im.lin.shape == (3, 4272, 6384)
    # the display is the stretch: the sky sits at a quarter of the range, the stars reach the top
    a = im.stretched.arr[::8, ::8]
    assert 40 < np.median(a) < 110 and a.max() > 240
    assert np.median(im.linear.arr[::8, ::8]) < np.median(a) / 2            # linear: dark, as stored
    # the toggle changes the display, never the data
    before = im.lin[:, 100, 100].copy()
    on = [b for b in t.findChildren(QPushButton) if b.text() == "Auto-stretch"][0]
    assert on.isChecked()
    on.setChecked(False)
    assert not t.stretched and sl.viewer.source.size() == (6384, 4272)
    assert sl.viewer.source.region(0, 0, 0, 4, 4).shape == (4, 4, 3)
    assert np.array_equal(im.lin[:, 100, 100], before)
    on.setChecked(True)
    # the histogram: three channels, log counts, of what is shown
    hb = [b for b in t.findChildren(QPushButton) if b.text() == "Histogram"][0]
    hb.setChecked(True)
    settle(win)
    h = t.findChildren(Histogram)[0]
    assert h.h is not None and len(h.h) == 3 and h.isVisibleTo(t)
    assert abs(float(h.h[0].max()) - float(np.log1p(im.hist["stretched"][0]).max())) < 1e-6
    # the measurements come from the manifest, through the metric catalogue
    rows = dict(win.steps.facts(win.s.path, "empilement"))
    assert texts.metric_label("frames_used") in rows and rows[texts.metric_label("frames_used")] == "40"
    assert rows[texts.metric_label("grid")] == "6384 × 4272 px"
    assert rows[texts.metric_label("rejected_pct")] == "1.821 %"
    panel_text = texts_of(t.panel)
    assert "Measurements" in panel_text and any(texts.metric_label("frames_used") in t for t in panel_text)
    shot(win, "inspect_stack")


def test_the_pixel_readout_gives_adu_and_sky(win):
    t = win.tune
    im = t._slots[0].image
    t._hover(0, 1500.4, 900.7)
    txt = t.readout.text()
    r, g, b = (int(round(float(c))) for c in im.lin[:, 900, 1500])
    assert f"x 1500  y 900   R {r}  G {g}  B {b} ADU" in txt
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra, dec = api.pixel_to_sky(win.s.p, 1, 1500.5, 900.5)
    assert sky_text(ra, dec) in txt and "RA 02h" in txt and "Dec +62°" in txt
    t._hover(0, -5, 10)
    assert t.readout.text() == ""
    appmod.apply_language("fr")
    assert "06,6s" in sky_text(40.5, 61.0) or "," in sky_text(40.50001, 61.0)
    appmod.apply_language("en")
    assert "." in sky_text(40.50001, 61.0)
    t.show_hist = True
    t.refresh()
    settle(win)
    t._hover(0, 1500.4, 900.7)
    shot(win, "inspect_stack_hist_readout")


# ---------------------------------------------------------------------------------------------- the other blocks
def test_each_block_shows_its_real_picture(win):
    """Every block of Full has a picture from its done results: the arrays are stretched, the previews as they are."""
    want = {"calibration": ("preview", 3), "alignement": ("preview", 1), "empilement": ("array", 1),
            "astrometrie": ("array", 2), "couleur": ("chart", 2), "debruitage": ("preview", 2),
            "etoiles": ("array", 3), "fond": ("preview", 2), "post": ("preview", 1)}
    for step, (kind, n) in want.items():
        inspect(win, step)
        t = win.tune
        vis = t.visuals(win.s.path, step)
        assert len(vis) == n and vis[0].kind == kind, step
        sl = t._slots[0]
        assert sl.image is not None, step
        if kind == "chart":
            assert isinstance(sl.content, SpectralChart)
        else:
            assert sl.viewer is not None and sl.image.size[0] > 100
            assert (sl.image.lin is not None) == (kind == "array")
    # a block with several pictures has tabs, the pictures change with the tab
    inspect(win, "etoiles")
    tb = win.tune.findChildren(QTabBar)[0]
    assert [tb.tabText(i) for i in range(tb.count())] == ["Starless image", "Star model", "Measured shape of a star"]
    tb.setCurrentIndex(1)
    settle(win)
    assert win.tune._slots[0].image.spec.key == "star_model"
    assert win.tune.img_tab == 1
    win.tune.img_tab = 0


def test_the_color_chart_draws_the_fitted_curves(win):
    inspect(win, "couleur")
    ch = win.tune._slots[0].content
    assert isinstance(ch, SpectralChart) and ch.S.shape[0] == 3 and len(ch.lam) == ch.S.shape[1]
    sp = api.spectral_response(win.s.p, 1)
    assert np.allclose(ch.S, sp.S) and np.allclose(ch.lam, sp.lam)
    ch.resize(900, 400)
    img = ch.grab().toImage()
    colors = {img.pixelColor(x, y).name() for x in range(60, 880, 7) for y in range(40, 360, 5)}
    assert len(colors) > 8                                         # the three curves and the grid, not a blank widget
    rows = dict(win.steps.facts(win.s.path, "couleur"))
    assert rows[texts.metric_label("stars")] == "2879"
    shot(win, "inspect_color")


def test_the_selection_chart_updates_the_kept_count_through_selection_preview(win):
    inspect(win, "selection")
    t = win.tune
    ch = t.chart
    assert isinstance(ch, FramesChart) and ch.n == 40 and t.table.total == 40
    assert abs(ch.tmin - 60.0) < 1e-6 and abs(ch.fmax - win.s.thresholds_now()[1]) < 1e-6
    assert ch.kept() == 40
    # drag the transparency threshold up: the engine's preview gives the count, the chart agrees
    from PySide6.QtTest import QTest
    top, bot = ch._panels()
    (f0, f1), (t0, t1) = ch._ranges()
    start = ch._xy(bot, 0, ch.tmin, t0, t1)
    dest = ch._xy(bot, 0, 90.0, t0, t1)
    QTest.mousePress(ch, Qt.LeftButton, Qt.NoModifier, QPoint(int(bot.left() + 30), int(start.y())))
    QTest.mouseMove(ch, QPoint(int(bot.left() + 30), int(dest.y())))
    QTest.mouseRelease(ch, Qt.LeftButton, Qt.NoModifier, QPoint(int(bot.left() + 30), int(dest.y())))
    settle(win)
    assert 88 < ch.tmin < 92
    want = api.selection_preview(win.s.p, 1, ch.tmin / 100, ch.fmax / win.s.fwhm_median()).kept
    assert want < 40 and ch.kept() == want
    assert t.kept_label.text().startswith(f"{want} of 40 frames kept")
    # the stack is computed: Replace or Try as variant is offered, the chart keeps the dragged threshold meanwhile
    assert win.s.pending_change is not None and win.s.pending_change["step"] == "selection"
    assert any(b.text().startswith("Try as variant") for b in t.panel.findChildren(QPushButton))
    assert abs(t.chart.tmin - ch.tmin) < 1.0
    win.s.pending_change = None
    shot(win, "inspect_selection")


def test_a_frame_is_picked_on_the_chart_and_excluded_by_hand_for_the_whole_project(win):
    inspect(win, "selection")
    t = win.tune
    s = win.s
    before = {st: s.stage_key(s.path, st) for st in ("inventory", "calibrate", "register", "stack", "color")}
    t._pick(3)
    settle(win)
    row = t.table.rows[3]
    assert t.picked == 3 and t._frame_token and t.pick_view.source is not None         # the frame, made by the worker
    btn = [b for b in t.findChildren(QPushButton) if b.text() == "Exclude"][0]
    btn.click()
    settle(win)
    assert (row.session, row.name) in s.frames_excluded
    after = {st: s.stage_key(s.path, st) for st in before}
    assert after["stack"] != before["stack"]                            # Exclude changes the Stacking key
    assert all(after[k] == before[k] for k in ("inventory", "calibrate", "register"))
    assert s.state(s.path, "stack") != "done"
    assert t.table.rows[3].excluded and t.chart.hand == {3}
    assert any(b.text() == "Keep" for b in t.findChildren(QPushButton))
    s.set_excluded([(row.session, row.name)], False)                    # the shared window goes back as it was
    settle(win)
    assert s.stage_key(s.path, "stack") == before["stack"] and s.state(s.path, "stack") == "done"


def test_a_block_not_computed_says_so_and_offers_run_to_here(win):
    s = win.s
    q = s.add_path("Classic stack", {"debruitage": False, "etoiles": "garder", "appris": False, "fond": "simple"})
    try:
        settle(win)
        s.select(index=s.index_of(q.pid), step="fond")
        win.set_mode(1)
        settle(win)
        t = win.tune
        texts_ = texts_of(t)
        assert "Not computed yet" in texts_ and any(x.startswith("Run to here") for x in texts_)
        run = [b for b in t.findChildren(QPushButton) if b.text().startswith("Run to here")][0]
        assert run.toolTip()
        got = []
        t.runHere.connect(lambda: got.append(1))
        run.click()
        assert got == [1]
        assert not any("Preview on a region" in x for x in texts_)       # ◇: Spec mode only
        shot(win, "inspect_pending")
        prefs.set("spec_mode", True)
        t.refresh()
        settle(win)
        assert any("Preview on a region" in x for x in texts_of(t))
        prefs.set("spec_mode", False)
    finally:
        s.remove_path(s.index_of(q.pid))
        s.select(index=0, step="empilement")
        settle(win)


def test_compare_shows_two_workflows_side_by_side_in_sync(win):
    s = win.s
    q = s.add_path("Classic stack", {"debruitage": False, "etoiles": "garder", "appris": False, "fond": "simple"})
    try:
        s.select(index=0, step="empilement")
        win.set_mode(1)
        settle(win)
        t = win.tune
        cmp_ = [c for c in t.findChildren(QComboBox) if c.itemText(0) == "—"][0]
        assert cmp_.isEnabled() and cmp_.findText("Classic stack") > 0
        cmp_.setCurrentIndex(cmp_.findText("Classic stack"))
        settle(win)
        assert t.cmp_pid == q.pid and len(t.viewers) == 2
        assert "same block" in " ".join(texts_of(t))                    # Stacking is shared by the two
        a, b = t.viewers
        assert a.source.size() == b.source.size()
        a.zoom_to(1.0, 3000, 2000)
        assert b.scale == 1.0 and (b.cx, b.cy) == (3000, 2000)         # zoom and pan follow
        b.zoom_to(0.5, 1000, 1000)
        assert a.scale == 0.5 and (a.cx, a.cy) == (1000, 1000)
        assert t._slots[0].image is t._slots[1].image                    # one picture in memory for a shared block
        shot(win, "inspect_compare")
        # a block that differs: Gradient (the other workflow has not computed it)
        s.select(step="fond")
        settle(win)
        assert "Not computed yet." in texts_of(t)
        assert "same block" not in " ".join(texts_of(t))
    finally:
        win.tune.cmp_pid = None
        s.remove_path(s.index_of(q.pid))
        s.select(index=0, step="empilement")
        settle(win)


def test_no_disk_access_in_the_gui_thread(win):
    """A debug hook counts what the GUI thread opens, lists or writes under the project and its raws: the views ask the
    worker for every image, table, thumbnail and scan."""
    inspect(win, "calibration")
    sv = win.tune.service
    with si.GuiDiskWatch([F_PROJECT, DATA]) as w:
        # the hook sees the GUI thread: a deliberate read is counted, a read in another thread is not
        with open(F_PROJECT / "project.astroproj", "rb") as f:
            f.read(1)
        assert w.count == 1
        w.hits.clear()
        import threading
        th = threading.Thread(target=lambda: open(F_PROJECT / "project.astroproj", "rb").close())
        th.start()
        th.join()
        assert w.count == 0
        req0 = sv.requests
        for step in ("alignement", "selection", "empilement", "astrometrie", "couleur", "debruitage", "etoiles", "fond",
                     "post", "sources"):
            inspect(win, step)
        win.frames.show_page(1)
        settle(win)
        win.browser.set_mode(0)
        win.browser.select_group("darks")
        win.browser.select_group("lights")
        settle(win)
        win.tune.show_hist = True
        win.tune.select(step="empilement")
        win.tune.refresh()
        settle(win)
        win.tune._stretch(False)
        win.tune._hover(0, 100.5, 100.5)
        win.tune._stretch(True)
        win.tune.show_hist = False
        win.frames.show_page(0)
        settle(win)
        where = list(w.where)
    # who asked: the innermost call of the client; the settings panel and the model (P9b) are not the views of P10
    mine = [(path, fr[-3:]) for path, fr in where
            if fr and fr[-1].split(":")[0] in ("tune.py", "frames.py", "sources.py", "steps.py", "charts.py", "inspect.py")]
    assert mine == [], mine[:5]
    assert sv.requests > req0                                            # the worker did the reading


# ---------------------------------------------------------------------------------------------- Frames
def test_frames_shows_the_sessions_the_health_checks_and_the_counts(win):
    from PySide6.QtWidgets import QTableWidget
    inspect(win, "sources")
    f = win.frames
    assert win.tune.pages.currentIndex() == 1 and f.data is not None
    t = f.pages.widget(0).findChildren(QTableWidget)[0]
    hdr = [t.horizontalHeaderItem(i).text() for i in range(t.columnCount())]
    assert hdr == ["Session", "Night", "Lights", "Exposure", "ISO / gain", "Temp.", "Filter", "Flats", "Darks", "Bias",
                   "FWHM", "Kept"]
    row = {h: t.item(0, i).text() for i, h in enumerate(hdr)}
    assert row["Lights"] == "40" and row["Flats"] == "15" and row["Darks"] == "15" and row["Bias"] == "none"
    assert row["Exposure"] == "15 s" and row["ISO / gain"] == "5000" and row["Night"] == "16 → 17 Aug 2026"
    assert row["FWHM"] == "4.0 px" and row["Kept"] == "40 / 40" and row["Temp."] == "–"
    lines = check_lines(f.data)
    assert ("ok", "Nothing to worry about") in lines
    assert any(lv == "info" and tx.startswith("No bias: flats are calibrated with the camera black level (512 ADU)")
               for lv, tx in lines)
    assert any(lv == "ok" and tx.startswith("Disk: about ") and "GB free on the cache drive" in tx for lv, tx in lines)
    assert not any("◇" in x for x in texts_of(f.pages.widget(0)))        # the checks the engine lacks: Spec mode only
    prefs.set("spec_mode", True)
    f._build()
    settle(win)
    assert sum(1 for x in texts_of(f.pages.widget(0)) if "master" in x and x.startswith(("Darks", "Sensor"))) == 2
    prefs.set("spec_mode", False)
    assert night_text("2026-08-17T00:25:13") == "16 → 17 Aug 2026"
    assert night_text("2026-08-17T14:00:00") == "17 → 18 Aug 2026"
    shot(win, "frames_sessions")


def shot_dialog(d, name, lang="en"):
    QApplication.processEvents()
    d.grab().save(str(shots_dir() / f"{name}_{lang}.png"))


def test_the_grouping_dialog_holds_the_criteria_of_the_engine(win):
    d = GroupingDialog(api.GroupingCriteria(night=True, exposure=False, iso=True), win)
    d.show()
    assert set(d.boxes) == {"night", "exposure", "iso"}                  # the others are ◇: Spec mode only
    assert d.boxes["night"].isChecked() and not d.boxes["exposure"].isChecked()
    d.boxes["exposure"].setChecked(True)
    c = d.criteria()
    assert (c.night, c.exposure, c.iso) == (True, True, True)
    prefs.set("spec_mode", True)
    d2 = GroupingDialog(None, win)
    assert {"filter", "temp", "binning", "rotation", "object"} <= set(d2.boxes)
    assert all(not d2.boxes[k].isEnabled() and "◇" in d2.boxes[k].text() for k in ("filter", "temp", "object"))
    prefs.set("spec_mode", False)
    d3 = GroupingDialog(None, win)
    d3.show()
    shot_dialog(d3, "grouping")
    sv = win.tune.service
    old = sv.criteria
    sv.criteria = c
    inspect(win, "sources")
    win.frames._build()
    settle(win)
    assert "night (noon to noon), exposure and ISO / gain" in " ".join(texts_of(win.frames.pages.widget(0)))
    sv.criteria = old
    for x in (d, d2, d3):
        x.close()


def test_the_import_buttons_run_the_import(win, monkeypatch):
    import astro_gpu.client.sources as srcmod
    calls = []
    monkeypatch.setattr(srcmod, "import_folder", lambda *a, **k: calls.append(bool(k.get("new_session"))) or True)
    inspect(win, "sources")
    got = []
    win.frames.imported.connect(lambda: got.append(1))
    win.frames.b_import.click()
    win.frames.b_add.click()
    assert calls == [False, True] and got == [1, 1]
    win.tune.panel.addSources.emit()                                  # the panel of the Frames block, in Inspect
    win.tune.panel.addSession.emit()
    assert calls == [False, True, False, True]

def test_two_sessions_merge_their_flats_and_offer_the_stacking_mode(win):
    """A second night with flats: the engine's check says the flats merge, and the Stacking modes show."""
    s = win.s
    p = s.p
    extra = api.add_session(p, lights=[str(DATA / "lights")], flats=[str(DATA / "flats")], name="Session 2")
    try:
        s.reload_sources()
        win.worker.wait(120)
        settle(win)
        assert len(s.sessions) == 2
        inspect(win, "sources")
        f = win.frames
        lines = check_lines(f.data)
        assert any(lv == "warn" and "The flats of 2 sessions are merged into one master flat" in tx for lv, tx in lines)
        assert "Stacking" in texts_of(f.pages.widget(0))
        assert "All sessions together" in texts_of(f.pages.widget(0))
        assert "One stack per session" not in " ".join(texts_of(f.pages.widget(0)))        # ◇
        prefs.set("spec_mode", True)
        f._build()
        settle(win)
        b = [x for x in f.findChildren(QPushButton) if x.text().startswith("One stack per session")][0]
        assert not b.isEnabled() and "◇" in b.text()
        prefs.set("spec_mode", False)
        shot(win, "frames_two_sessions")
    finally:
        p.sessions.remove(extra)
        p.validate()
        s.reload_sources()
        win.worker.wait(120)
        settle(win)
        assert len(s.sessions) == 1


# ---------------------------------------------------------------------------------------------- import, Outdated
def _scratch_project(folder: Path) -> Path:
    """A copy of the test project's definition whose store is a junction onto the real one: the results are read, and
    what the model writes (state.json, the definition) lands in the temporary folder."""
    import subprocess
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(F_PROJECT / "project.astroproj", folder / "project.astroproj")
    for name in ("store", "inputs"):
        if (F_PROJECT / name).is_dir() and not (folder / name).exists():
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(folder / name), str(F_PROJECT / name)],
                               capture_output=True, text=True)
            if r.returncode:
                pytest.skip(f"no directory junction here: {r.stderr.strip()}")
    return folder


@pytest.fixture()
def scratch(tmp_path):
    folder = _scratch_project(tmp_path / "heart")
    p = api.open_project(folder)
    api.add_workflow(p, "Classic stack", "Classic stack", None)
    api.add_workflow(p, "Full + neural stars", "Full + neural stars", {"denoise": {"iterations": 6000}})
    api.save(p)
    w = open_window(folder, readonly=False)
    yield w
    w.s.flush()
    w.close()


def test_an_outdated_block_shows_its_previous_result(scratch):
    w = scratch
    s = w.s
    full = s.paths[0]
    assert s.state(full, "stack") == "done"
    s.set_param(full, "stack", "kappa", 4.0)                              # Stacking and what follows are Outdated now
    settle(w)
    assert s.state(full, "stack") == "stale" and s.previous_key(full, "stack")
    inspect(w, "empilement", index=0)
    t = w.tune
    im = t._slots[0].image
    assert im is not None and im.key[-1] == "previous" and im.size == (6384, 4272)
    texts_ = texts_of(t)
    assert "previous result" in texts_ and any(x.startswith("Run to here") for x in texts_)
    assert "Outdated" in texts_ or any("Outdated" in x for x in texts_)
    # the measurements are the previous result's, read from its manifest
    assert dict(w.steps.facts(full, "empilement"))[texts.metric_label("frames_used")] == "40"
    shot(w, "inspect_previous")
    run = [b for b in t.findChildren(QPushButton) if b.text().startswith("Run to here")][0]
    got = []
    t.runHere.connect(lambda: got.append(1))
    run.click()
    assert got == [1]
    # each picture has the result its own stage has: Calibration is up to date, Denoise (downstream) shows its previous one
    inspect(w, "calibration", index=0)
    assert w.tune._slots[0].image.key[-1] == "current" and "previous result" not in texts_of(w.tune)
    inspect(w, "debruitage", index=0)
    assert w.tune._slots[0].image is not None and "previous result" in texts_of(w.tune)
    # a block never computed has no previous result: it says "Not computed yet"
    inspect(w, "fond", index=1)
    assert "Not computed yet" in texts_of(w.tune)
    s.set_param(full, "stack", "kappa", None)                             # back to the suggestion: Up to date again
    settle(w)
    assert s.state(full, "stack") == "done"


def test_a_variant_is_shown_compared_with_its_original(scratch):
    w = scratch
    s = w.s
    full = s.paths[0]
    ch = {"kind": "param", "stage": "stack", "name": "kappa", "value": 3.0, "suggested": 5.0}
    q = s.add_variant(full, ch, "Full · kappa 3")
    settle(w)
    w.tune._variant_made(q.pid, full.pid)
    inspect(w, "alignement")
    t = w.tune
    assert t.cmp_pid == full.pid and s.path is q
    assert "same block" in " ".join(texts_of(t)) and len(t.viewers) == 2       # Registration is shared
    assert t._slots[0].image is t._slots[1].image


def test_importing_the_heart_folder_gives_the_counts_the_sessions_and_the_thumbnails(env, monkeypatch, tmp_path):
    root = tmp_path / "ws" / "New"
    project2.create(root, "New", Target(name="IC 1805"))
    w = open_window(root, readonly=False)
    try:
        inspect(w, "sources")
        assert "No frames yet" in texts_of(w.frames.pages.widget(0))
        shots = shots_dir()

        def fake_exec(self):
            end = time.monotonic() + 120
            while self.scan is None and time.monotonic() < end:
                QApplication.processEvents()
                time.sleep(0.01)
            assert self.scan is not None and self.ok.isEnabled()
            QApplication.processEvents()
            self.grab().save(str(shots / f"import_dialog_{i18n.language()}.png"))
            self.accept()
            return 1
        monkeypatch.setattr(ImportDialog, "exec", fake_exec)
        assert import_folder(w.s, w, str(DATA / "lights"), service=w.tune.service)
        for _i in range(200):
            if w.s.ready and w.s.sources_loaded and len(w.s.sessions) == 1 and w.s.sessions[0]["lights"]:
                break
            w.worker.wait(5)
            time.sleep(0.05)
        settle(w)
        ses = w.s.sessions
        assert len(ses) == 1 and (ses[0]["lights"], ses[0]["darks"], ses[0]["flats"]) == (40, 15, 15)
        assert w.s.history[-1][1].startswith('Folder "{folder}" imported')
        # saved: the definition holds the session, so the project opens again with it
        w.s.flush()
        d = project2.load(root)
        assert len(d.sessions) == 1 and d.sessions[0].lights and d.sessions[0].darks and d.sessions[0].flats
        # the Frames page, from the headers
        inspect(w, "sources")
        from PySide6.QtWidgets import QTableWidget
        t = w.frames.pages.widget(0).findChildren(QTableWidget)[0]
        row = [t.item(0, i).text() for i in range(t.columnCount())]
        assert row[2:5] == ["40", "15 s", "5000"] and row[7:10] == ["15", "15", "none"] and row[11] == "40 / 40"
        # the thumbnails of the browser come from the worker
        w.frames.show_page(1)
        settle(w)
        time.sleep(1.0)
        settle(w)
        b = w.browser
        assert len(b.items) == 40 and b.grid.count() == 40
        assert len(w.tune.service.thumb_cache) >= 20
        have = [b.grid.item(i) for i in range(b.grid.count()) if i < 20]
        assert all(not it.icon().isNull() for it in have)
        assert b.viewer.source is not None and b.viewer.source.size()[0] == 1280
        shot(w, "frames_browse_import")
        # Add session…: everything goes into a session of its own
        assert import_folder(w.s, w, str(DATA / "flats"), new_session=True, service=w.tune.service)
        assert len(w.s.p.sessions) == 2
        # the same folder twice: said, not added
        assert not import_folder(w.s, w, str(DATA / "lights"), service=w.tune.service)
        assert len(w.s.p.sessions) == 2 and any(k == "information" for k, _a in env["shown"])
    finally:
        w.s.flush()
        w.close()


def test_the_frame_chart_and_the_browser_keep_and_exclude_frames(win):
    inspect(win, "sources")
    win.frames.show_page(1)
    settle(win)
    b = win.browser
    s = win.s
    assert b.group == "lights" and len(b.items) == 40
    # sort by star size, the worst frame last
    b.sort_cb.setCurrentIndex(b.sort_cb.findData("fwhm"))
    idx = b.order()
    assert b.items[idx[0]].fwhm <= b.items[idx[-1]].fwhm
    b.sort_cb.setCurrentIndex(b.sort_cb.findData("time"))
    # X excludes the shown frame for the whole project, X again keeps it
    b._goto(5)
    row = b.items[5]
    key0 = s.stage_key(s.path, "stack")
    b.toggle_keep()
    settle(win)
    assert (row.session, row.name) in s.frames_excluded and b.items[5].excluded
    assert s.stage_key(s.path, "stack") != key0
    assert "1 lights" not in b.count.text() and b.count.text().endswith("1 excluded")
    b.toggle_keep()
    settle(win)
    assert not s.frames_excluded and s.stage_key(s.path, "stack") == key0
    # Blink plays the frames
    b.blink.setChecked(True)
    assert b.timer.isActive() and b.mode == 1
    b.blink.setChecked(False)
    assert not b.timer.isActive()
    # the calibration frames: darks, with the level and the noise measured
    b.select_group("darks")
    settle(win)
    time.sleep(1.5)
    settle(win)
    assert b.group == "darks" and len(b.items) == 15
    assert any(win.tune.service.stats.get(it.path) for it in b.items)
    assert not b.keep_b.isVisible()                                        # ◇ for the calibration frames
    b.select_group("lights")
    settle(win)


# ---------------------------------------------------------------------------------------------- the units of the service
def test_the_automatic_stretch_puts_the_sky_at_a_quarter():
    rng = np.random.default_rng(1)
    sky = rng.normal(200.0, 12.0, (3, 600, 800)).astype(np.float32)
    sky[:, 100:120, 100:120] = 12000.0                                   # a star
    sp = si.stretch_params(sky)
    s8, l8, _ = si.stretch_image(sky)
    assert s8.shape == (600, 800, 3) and s8.dtype == np.uint8
    assert all(abs(np.median(s8[:, :, c]) / 255 - 0.25) < 0.02 for c in range(3))
    assert s8[105, 105].min() >= 250 and l8[105, 105].min() >= 250
    assert np.median(l8) < np.median(s8) / 3                             # linear: dark
    assert np.allclose(si.mtf(0.3, np.array([0.0, 0.3, 1.0])), [0.0, 0.5, 1.0])
    assert (sp["black"] < 200).all() and (sp["white"] > 5000).all()
    # the data are untouched, nan and inf are shown black / white, not raised
    bad = sky.copy()
    bad[0, 5, 5], bad[1, 6, 6] = np.nan, np.inf
    assert si.render8(bad, 150, 12000, 0.5).shape == (600, 800, 3)
    # the half-resolution copy is an average, not a decimation
    a = np.zeros((4, 4, 3), np.uint8)
    a[::2, ::2] = 200
    assert (si.reduce2(a) == 50).all()
    assert si.PixelSource(np.zeros((5, 7, 3), np.uint8)).half.shape == (3, 4, 3)
    h = si.histogram8(s8)
    assert h.shape == (3, 256) and h[0].sum() == s8[::2, ::2, 0].size


def test_the_image_cache_forgets_the_oldest_within_its_budget():
    c = si.ImageCache(budget=2500)
    spec = si.VisualSpec("a", "A", "array", "stack", "x")
    for i in range(4):
        c.put(si.ShownImage(("stack", f"k{i}", "x", "current"), spec, nbytes=1000))
    assert len(c) == 2 and c.get(("stack", "k0", "x", "current")) is None
    assert c.get(("stack", "k3", "x", "current")) is not None
    c.get(("stack", "k2", "x", "current"))                               # most recently used now
    c.put(si.ShownImage(("stack", "k4", "x", "current"), spec, nbytes=1000))
    assert c.get(("stack", "k2", "x", "current")) is not None and c.get(("stack", "k3", "x", "current")) is None


# ---------------------------------------------------------------------------------------------- the French, the look
def cut_texts(root: QWidget) -> list[str]:
    """Labels, buttons and check boxes narrower than their text, and scroll areas whose content is wider than them (their
    horizontal bar is off: what passes the right edge is cut)."""
    out = []
    for sa in root.findChildren(QScrollArea):
        inner = sa.widget()
        if sa.isVisibleTo(root) and inner is not None and sa.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff:
            if inner.minimumSizeHint().width() > sa.viewport().width() + 1:
                out.append(f"scroll area: content {inner.minimumSizeHint().width()} px > {sa.viewport().width()} px")
    for w in root.findChildren(QWidget):
        if not w.isVisibleTo(root):
            continue
        if isinstance(w, QLabel) and not w.wordWrap() and w.text() and (w.pixmap() is None or w.pixmap().isNull()):
            if w.width() + 1 < w.sizeHint().width() and w.objectName() != "dim":
                out.append(w.text())
        elif isinstance(w, (QPushButton, QCheckBox)) and w.text():
            if w.width() + 1 < w.sizeHint().width():
                out.append(w.text())
    return out


NAMES = {"Full", "Classic stack", "Classic + denoise", "Full + neural stars", "Raw stack", "Session 1", "heart_a7s",
         "IC 1805", "OSC", "ILCE-7S", "Heart", "FWHM"}


def untranslated(texts_) -> list[str]:
    """Texts shown in English although a French translation exists (the names the user typed are never translated)."""
    cat = i18n._cat
    return [t for t in texts_ if t in cat and cat[t] != t and t not in NAMES]


def decimal_points(texts_) -> list[str]:
    """A number written with a decimal point in the French interface (paths, dates, times and versions are not numbers)."""
    bad = []
    for t in texts_:
        if re.search(r"[/\\]|\d{4}-\d\d-\d\d|\d:\d\d|\.ARW|\.json|\.fits", t):
            continue
        if re.search(r"(?<![\w.])\d+\.\d+(?![\w.])", t):
            bad.append(t)
    return bad


def test_french_translates_every_screen_with_decimal_commas(module_win, env):
    w = module_win
    prefs.set("spec_mode", True)
    w.change_language("fr")
    seen: list[str] = []
    try:
        i18n._missing.clear()
        for step in ("calibration", "alignement", "selection", "empilement", "astrometrie", "couleur", "debruitage",
                     "etoiles", "fond", "post", "sources"):
            inspect(w, step)
            if step == "empilement":
                w.tune.show_hist = True
                w.tune.refresh()
                settle(w)
                w.tune._hover(0, 1500.4, 900.7)
            seen += texts_of(w)
            w.tune.show_hist = False
            if step in ("astrometrie", "etoiles", "debruitage"):
                tb = w.tune.findChildren(QTabBar)[0]
                tb.setCurrentIndex(1)
                settle(w)
                seen += texts_of(w)
                w.tune.img_tab = 0
        w.frames.show_page(1)
        settle(w)
        time.sleep(0.5)
        for mode in (0, 1, 2):
            w.browser.set_mode(mode)
            settle(w)
            seen += texts_of(w)
        w.browser.select_group("darks")
        settle(w)
        seen += texts_of(w)
        w.browser.select_group("lights")
        w.frames.show_page(0)
        settle(w)
        d = GroupingDialog(None, w)
        d.show()
        seen += texts_of(d)
        d.close()
        assert not untranslated(seen), sorted(set(untranslated(seen)))
        assert not decimal_points(seen), sorted(set(decimal_points(seen)))
        mine = {t for t in i18n.missing()}
        assert not mine, sorted(mine)
        # numbers: the readout, the thresholds, the measurements, the table
        joined = " ".join(seen)
        assert "06,6s" in joined or "RA 02h" in joined
        assert "7,9 px" in joined and "0,17 h" in joined and "4,0 px" in joined
        assert "16 → 17 août 2026" in joined and "aucun" in joined
    finally:
        prefs.set("spec_mode", False)
        w.change_language("en")


def test_the_shots_in_english_and_french_have_no_cut_text(module_win, env):
    w = module_win
    cut = {}
    for lang in ("en", "fr"):
        w.change_language(lang)
        w.resize(1680, 1000)
        w.show()

        def take(name):
            f = shot(w, name, lang)
            c = cut_texts(w)
            if c:
                cut[f"{name}_{lang}"] = c
            return f

        w.s.pending_change = None
        inspect(w, "empilement", index=0)
        take("inspect_stack")
        w.tune.show_hist = True
        w.tune.refresh()
        settle(w)
        w.tune._hover(0, 1500.4, 900.7)
        take("inspect_stack_hist_readout")
        w.tune.show_hist = False
        inspect(w, "debruitage")
        take("inspect_denoise")
        inspect(w, "etoiles")
        take("inspect_stars")
        inspect(w, "couleur")
        take("inspect_color")
        inspect(w, "selection")
        w.tune._pick(10)
        settle(w)
        take("inspect_selection")
        inspect(w, "sources")
        take("frames_sessions")
        d = GroupingDialog(None, w)
        d.show()
        shot_dialog(d, "grouping", lang)
        d.close()
        w.frames.show_page(1)
        w.browser.set_mode(1)
        settle(w)
        time.sleep(1.0)
        settle(w)
        take("frames_browse")
        w.browser.set_mode(0)
        settle(w)
        take("frames_browse_grid")
        w.browser.set_mode(2)
        settle(w)
        take("frames_browse_list")
        w.browser.set_mode(1)
        w.frames.show_page(0)
        settle(w)
    w.change_language("en")
    assert not cut, cut
    names = {p.name for p in shots_dir().glob("*.png")}
    for lang in ("en", "fr"):
        for n in ("inspect_stack", "inspect_stack_hist_readout", "inspect_denoise", "inspect_stars", "inspect_selection",
                  "inspect_color", "frames_sessions", "frames_browse", "grouping"):
            assert f"{n}_{lang}.png" in names, n


def test_the_modules_of_inspect_have_no_mockup_leftovers():
    root = ROOT / "astro_gpu" / "client"
    allowed = {"__future__", "os", "sys", "threading", "collections", "dataclasses", "pathlib", "typing", "weakref",
               "numpy", "PySide6", "PIL", "astropy", "datetime", "json", "html", "warnings", "traceback", "re"}
    for name in ("steps", "tune", "charts", "frames", "sources", "services/inspect"):
        src = (root / f"{name}.py").read_text(encoding="utf-8")
        for bad in ("alt2", "alt3", "data.json", "datatext", "Demo", "import project", "from project", "from steps",
                    "from model import", "DATA ", "mockup"):
            assert bad not in src, (name, bad)
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert node.module.split(".")[0] in allowed, (name, node.module)
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] in allowed for a in node.names), (name, [a.name for a in node.names])


def _shown_texts() -> dict[str, list[str]]:
    """Every English text the modules of P10 can show: the ``_("...")`` of the code, the captions of the pictures and the
    fixed tables."""
    root = ROOT / "astro_gpu" / "client"
    found: dict[str, list[str]] = {}
    for name in ("steps", "tune", "charts", "frames", "sources", "services/inspect"):
        tree = ast.parse((root / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_" and node.args:
                a = node.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    found.setdefault(a.value, []).append(f"{name}.py:{node.lineno}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "V" and len(node.args) > 1:
                if isinstance(node.args[1], ast.Constant):
                    found.setdefault(node.args[1].value, []).append(f"{name}.py:{node.lineno} (caption)")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in ("NEW_CHECKS", "KIND_COUNT"):
                vals = node.value.values if isinstance(node.value, ast.Dict) else [node.value]
                for v in vals:
                    for c in ast.walk(v):
                        if isinstance(c, ast.Constant) and isinstance(c.value, str):
                            found.setdefault(c.value, []).append(f"{name}.py:{node.targets[0].id}")
    for t in ("Grid", "Viewer", "List", "time", "level", "FWHM", "transparency", "low transparency", "large stars",
              "not aligned", "bias", "dark flats", "Lights", "Calibration", "Darks", "Flats", "Bias", "Dark flats",
              "File", "Time", "Date", "Kept", "Weight", "Level (ADU)", "Noise (ADU)", "FWHM (px)", "Transparency (%)",
              'Folder "{folder}" imported · {lights} lights'):
        found.setdefault(t, []).append("table")
    return found


def test_every_text_of_inspect_and_frames_has_its_french():
    cat: dict[str, str] = {}
    for f in sorted((ROOT / "astro_gpu" / "client" / "locale" / "fr").glob("*.json")):
        cat.update(json.loads(f.read_text(encoding="utf-8")))
    missing = {t: where[:2] for t, where in _shown_texts().items() if t and t not in cat}
    assert not missing, missing
    mine = json.loads((ROOT / "astro_gpu" / "client" / "locale" / "fr" / "inspect.json").read_text(encoding="utf-8"))
    assert len(mine) > 30 and all(isinstance(v, str) and v for v in mine.values())
    assert not any(texts.coverage("fr").values())


def test_the_gui_thread_keeps_beating_while_the_images_load(win):
    """A heartbeat in the GUI thread: from one block to the next, with images of 27 Mpx loading, stretching and drawing,
    no gap is longer than 400 ms, which is a busy machine's scheduling (alone: under 100 ms); the loads, the statistics and
    the thumbnails are the worker's."""
    from PySide6.QtCore import QTimer
    win.tune.service.cache.clear()                                    # freeing 2 GB of pictures is not what is measured
    settle(win)
    gaps, last = [], [time.perf_counter()]

    def beat():
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now
    tm = QTimer()
    tm.timeout.connect(beat)
    tm.start(10)
    try:
        for step in ("calibration", "empilement", "debruitage", "etoiles", "selection", "sources", "empilement"):
            win.s.select(index=0, step=step)
            win.set_mode(1)
            t_end = time.monotonic() + 60
            while time.monotonic() < t_end:                            # what a user sees: the event loop runs on
                QApplication.processEvents()
                time.sleep(0.005)
                if not win.tune.service.busy() and not win.worker.busy():
                    break
            QApplication.processEvents()
        win.frames.show_page(1)
        t_end = time.monotonic() + 3
        while time.monotonic() < t_end:
            QApplication.processEvents()
            time.sleep(0.005)
        win.frames.show_page(0)
    finally:
        tm.stop()
    worst = max(gaps)
    print(f"GUI thread: {len(gaps)} beats, the longest stop {worst * 1000:.0f} ms")
    assert len(gaps) > 100 and worst < 0.4, f"the GUI thread stopped for {worst * 1000:.0f} ms"


def test_the_test_project_was_not_written():
    """The views and the model read the test project: nothing in its definition, manifests or previews changed."""
    assert _snapshot(F_PROJECT) == PROJECT_BEFORE
