"""Inspect, Frames and Sources in the real main window, offscreen, on the test project (Full computed).

The model is read-only (nothing is saved into the project; the tests that edit work on a scratch copy of its
definition). ASTRO_GPU_SETTINGS points to a temporary file in every test; the user's settings file is checked untouched.
ASTRO_GPU_SHOTS names the folder of the screenshots (default: $TEMP/inspect_shots).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")                 # the GPUs are the runs' (DESIGN §6.1): these tests use the CPU
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")      # the offscreen platform has no fonts else

pytest.importorskip("PySide6")
from conftest import TEST_DATA, TEST_PROJECT, full_only, needs_test_project, scratch_project  # noqa: E402
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
from astro_gpu.client.charts import FramesChart  # noqa: E402
from astro_gpu.client.frames import FramesView, GroupingDialog, check_lines, night_text  # noqa: E402
from astro_gpu.client.imageview import ImageView, default_loader  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services import inspect as si  # noqa: E402
from astro_gpu.client.sources import ImportDialog, SourcesView, import_folder  # noqa: E402
from astro_gpu.client.steps import SkyOverlay, Steps  # noqa: E402
from astro_gpu.client.tune import (  # noqa: E402
    Table,  # noqa: E402
    TuneView,
    sky_text,
)
from astro_gpu.client.widgets import QSS  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
F_PROJECT = TEST_PROJECT
DATA = TEST_DATA
pytestmark = needs_test_project


def _user_settings() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "astro_gpu" / "settings.json"


def _fingerprint(p: Path):
    return (p.stat().st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest()) if p.exists() else None


def _snapshot(folder: Path):
    """The definition and the manifests of the test project (its store is big: names, sizes and dates of the small files)."""
    return sorted((p.relative_to(folder).as_posix(), p.stat().st_mtime_ns, p.stat().st_size)
                  for p in folder.rglob("*") if p.is_file() and (p.suffix in (".json", ".astroproj", ".jpg")))


def _test_facts():
    """What the test project holds, read from it (the bench may make it again with other frames): its numbers of
    lights, darks and flats, and the size of its stack."""
    if not (F_PROJECT / "project.astroproj").is_file():
        return 0, 0, 0, (0, 0)
    p = api.open_project(F_PROJECT)

    def count(kind):
        return sum(1 for s in p.sessions for d in getattr(s, kind) for f in Path(d).iterdir() if f.is_file())
    out = api.stage_outputs(p, p.workflows[0].id, "stack")
    return count("lights"), count("darks"), count("flats"), tuple(out.metrics["grid"]) if out is not None else (0, 0)


#: the test project's lights (all kept), darks, flats, its stack's width and height
N_LIGHTS, N_DARKS, N_FLATS, GRID = _test_facts()
USER_SETTINGS = _user_settings()
USER_BEFORE = _fingerprint(USER_SETTINGS)
PROJECT_BEFORE = _snapshot(F_PROJECT) if F_PROJECT.is_dir() else None


def shots_dir() -> Path:
    d = Path(os.environ.get("ASTRO_GPU_SHOTS") or Path(os.environ.get("TEMP", ".")) / "inspect_shots")
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
    default_loader().wait(max(end - time.monotonic(), 1))         # the viewers' images, read in their threads
    for v in win.findChildren(ImageView):
        if v.isVisible():
            v.wait(max(end - time.monotonic(), 1))
    _events()


class Harness(appmod.Main):
    """The main window with Inspect, Frames and Sources mounted (Graph and Post stay placeholders)."""

    def __init__(self, readonly=True, **kw):
        self.readonly = readonly
        super().__init__(**kw)

    def make_model(self):
        if len(self.project.definition.workflows) > 1:
            full_only(self.project.definition, fresh_ids=True)                 # Full, whatever the bench added
        super().make_model()

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

    def close_project(self, go_home=True, discard=False):
        return super().close_project(go_home, discard)


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
    w.tune.cmp_pid, w.tune.show_hist = None, True
    w.tune.tab_of.clear()
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
        elif isinstance(w, (QTableWidget, Table)):
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
    """Stacking: the stack in the common viewer (stretch with its histogram, the data never changed), its tabs, and the
    measurements from the manifest."""
    t = win.tune
    assert win.s.sel_step == "empilement" and t.pages.currentIndex() == 0
    v = t.view
    assert isinstance(v, ImageView) and v.prepared() is not None
    prep = v.prepared()
    assert prep.size() == GRID and prep.channels == 3 and prep.data.shape == (3, GRID[1], GRID[0])
    tb = t.findChildren(QTabBar)[0]
    assert [tb.tabText(i) for i in range(tb.count())] == ["Stack", "Rejection map", "Coverage", "Weights"]
    # the automatic stretch: the sky at about a quarter, the stars white; the toggle changes the display, never the data
    st = v.stretch()
    assert st.auto and not st.neutral                     # camera colours, as they are (Neutral sky: an option)
    before = prep.data[:, 100, 100].copy()
    v.auto_b.setChecked(False)
    assert v.stretch().mode == "linear"
    v.auto_b.setChecked(True)
    assert np.array_equal(prep.data[:, 100, 100], before)
    assert v.panel.isVisibleTo(v) and v.panel.hist.stats is not None   # the histogram of what is shown
    # compare with the previous steps: a calibrated frame and the half-stacks
    names = [v.cmp_cb.itemText(i) for i in range(v.cmp_cb.count())]
    assert "Half-stack A" in names and "Half-stack B" in names and any(n.endswith(", calibrated") for n in names)
    rows = dict(win.steps.facts(win.s.path, "empilement"))
    assert texts.metric_label("frames_used") in rows and rows[texts.metric_label("frames_used")] == str(N_LIGHTS)
    assert rows[texts.metric_label("grid")] == f"{GRID[0]} × {GRID[1]} px"
    panel_text = texts_of(t.panel)
    assert "Measurements" in panel_text and any(texts.metric_label("frames_used") in x for x in panel_text)
    shot(win, "inspect_stack")


def test_the_pixel_readout_gives_adu_and_sky(win):
    v = win.tune.view
    prep = v.prepared()
    txt = v.pixel_text(1500.4, 900.7)
    vals = prep.data[:, 900, 1500]
    assert txt.startswith("x  1500  y   900") and "ADU" in txt
    from astro_gpu.client.imageview import value_text
    assert all(f"{c} {value_text(float(x), prep.stats, False)}" in txt for c, x in zip("RGB", vals))
    assert prep.wcs is not None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra, dec = prep.wcs.pixel_to_world_values(1500.4 - 0.5, 900.7 - 0.5)
    assert sky_text(float(ra), float(dec)) in txt and "RA 02h" in txt and "Dec +6" in txt
    assert v.pixel_text(-5, 10) == ""
    appmod.apply_language("fr")
    assert "," in sky_text(40.50001, 61.0)
    appmod.apply_language("en")
    assert "." in sky_text(40.50001, 61.0)
    shot(win, "inspect_stack_hist_readout")


# ---------------------------------------------------------------------------------------------- the other blocks
def test_each_block_shows_its_real_picture(win):
    """Every tab of every block of Full shows its content: a picture in the common viewer (from the done results,
    read in the viewer's threads), a chart, a table or a sequence of frames."""
    want = {"calibration": ["masters", "darks", "flats", "calibrated"],
            "alignement": ["reference", "quality", "drift", "track", "distribution", "table"],
            "empilement": ["stack", "rejection", "coverage", "weights"], "astrometrie": ["solution"],
            "couleur": ["colors", "spectral", "lines", "fit"], "debruitage": ["denoised"],
            "etoiles": ["starless", "stars"],
            "fond": ["result", "removed"],
            "post": ["output"]}
    from astro_gpu.client.chartview import ChartView
    for step, tabs in want.items():
        assert [v.key for v in win.tune.visuals(win.s.path, step)] == tabs, step
        for key in tabs:
            if step == "post":
                continue                                               # the render worker's: its own test
            win.tune.tab_of[step] = key
            inspect(win, step)
            t = win.tune
            kind = t._cur[2].kind
            if kind in ("array", "preview", "color", "gradient", "removed", "calibrated", "masters", "calframes",
                        "callights", "regref"):
                assert t.view is not None and t.view.wait(120), (step, key)
                assert t.view.prepared() is not None, (step, key, t.view.canvas.message)
                assert min(t.view.prepared().size()) > 100
            elif kind in ("regchart", "chart", "colorfit", "weights"):
                assert t.findChildren(ChartView), (step, key)
            elif kind == "regtable":
                assert [x for x in t.findChildren(Table) if x.isVisible()][0].rowCount() == N_LIGHTS
            elif kind == "rejection":                                  # an older result: said plainly
                assert t.view is not None or any("Rejection maps" in x for x in texts_of(t))
            elif kind == "coverage":                                   # the drizzle's red, blue, green maps
                assert t.view is not None and t.view.wait(120), (step, key)
                assert {"Red", "Blue", "Green"} <= set(texts_of(t)), texts_of(t)
        win.tune.tab_of.pop(step, None)


def test_the_output_block_shows_the_final_image_the_render_worker_makes(win):
    """Output is what Post shows for the workflow's layer at its suggested settings, not an engine preview of a
    stage: the render worker's own half-resolution image, one lane for Inspect, in the common viewer."""
    from astro_gpu.client.postengine import new_workflow_layer
    from astro_gpu.client.services.render import HALF_LEVEL, Request
    inspect(win, "post")
    t = win.tune
    vis = t.visuals(win.s.path, "post")
    assert len(vis) == 1 and vis[0].kind == "render" and vis[0].caption == "Final image" and vis[0].name == ""
    for _i in range(120):
        if t.view is not None and t.view.prepared() is not None:
            break
        settle(win, 2)
    v = t.view
    vi = v.image()
    assert vi is not None and vi.stretched and v.prepared() is not None
    # one line on what it is: Gradient removed (of the denoised stack) with Post's stretch
    assert any(x.startswith("Final image = Gradient removed (of the denoised stack)") for x in texts_of(t))
    rw = win.render_worker()
    assert win.tune.service._rw is rw                                  # shared with Post: engines and images kept once
    got = []
    rw.rendered.connect(got.append)
    try:
        rw.request(Request(kind="half", layers=[new_workflow_layer(win.s.path.pid, "Full", True)], level=HALF_LEVEL))
        assert rw.settle(180)
    finally:
        rw.rendered.disconnect(got.append)
    ref = next(r for r in got if r.kind == "half")
    assert np.array_equal(vi.source, ref.rgb)
    # shown as it is (linear 0 to 255), its levels in the readout, the sky with the crop's offset
    assert v.stretch().mode == "linear" and float(v.stretch().white[0]) == 255.0
    txt = v.pixel_text(100.5, 120.5)
    r, g, b = (int(c) for c in ref.rgb[120, 100])
    assert txt.startswith(f"x   100  y   120    R {r}   G {g}   B {b}") and "RA " in txt
    names = [v.cmp_cb.itemText(i) for i in range(v.cmp_cb.count())]
    assert "Gradient removed" in names and "Colour-calibrated stack" in names
    # compared with Gradient removed: put on the final image's grid (its crop of the stack, at half size), said
    v.set_compare(names.index("Gradient removed") - 1)
    assert v.wait(120)
    al = v.compare_alignment()
    assert al is not None and al.how == "grid" and al.scale == pytest.approx(2.0, rel=0.01)
    assert v.align_bar.isVisible() and "geometry of the results" in v.alignment_note()
    shown = v.image()
    t.refresh()                                                         # nothing changed: the viewer stays (#24, 3)
    settle(win)
    assert t.view is v and v.image() is shown
    shot(win, "inspect_output")


def test_two_final_images_come_one_after_the_other_and_a_missing_result_fails_cleanly(win):
    sv = win.tune.service
    inspect(win, "post")
    spec = win.tune.visuals(win.s.path, "post")[0]
    pid = win.s.path.pid
    ready, failed, order = {}, [], []
    on_ready = lambda slot, token, im: (ready.__setitem__(slot, (token, im)), order.append(slot))     # noqa: E731
    on_failed = lambda slot, token, why: failed.append((slot, token, why))                            # noqa: E731
    sv.imageReady.connect(on_ready)
    sv.imageFailed.connect(on_failed)
    try:
        t0 = sv.request_image(0, pid, spec, "current")             # Compare with: two slots, one lane, in order
        t1 = sv.request_image(1, pid, spec, "current")
        assert sv.busy() and sv.wait(180) and not sv.busy()
        assert order == [0, 1] and ready[0][0] == t0 and ready[1][0] == t1 and not failed
        assert np.array_equal(ready[0][1].stretched.arr, ready[1][1].stretched.arr)
        assert ready[0][1].key == (spec.stage, win.steps.key_of(win.s.path, spec.stage), "", "current")
        assert sv.cache.get(ready[0][1].key) is ready[1][1]                  # the last one rendered is the one kept
        ready.clear()
        order.clear()
        t3 = sv.request_image(0, pid, spec, "current")
        t4 = sv.request_image(1, pid, spec, "current")
        t5 = sv.request_image(1, 999, spec, "current")             # a newer request of the slot: a workflow that is not there
        assert sv.wait(180)
        assert order == [0] and ready[0][0] == t3                  # t4 was replaced before it started
        assert [f[:2] for f in failed] == [(1, t5)] and "not computed" in failed[0][2] and t4 < t5
    finally:
        sv.imageReady.disconnect(on_ready)
        sv.imageFailed.disconnect(on_failed)


def test_the_astrometry_block_opens_on_the_users_image_with_its_solution(win):
    """The stack with the RA / Dec grid and the catalogue stars over it (an overlay of the common viewer); the named
    objects have their place (SkyData.objects)."""
    from PySide6.QtGui import QImage
    inspect(win, "astrometrie")
    settle(win)
    t = win.tune
    vis = t.visuals(win.s.path, "astrometrie")
    assert [v.key for v in vis] == ["solution"] and vis[0].kind == "array" and vis[0].sky
    v = t.view
    prep = v.prepared()
    assert prep.size() == GRID and prep.wcs is not None
    shown = texts_of(t)
    assert any(x.startswith("Your image with the sky grid") for x in shown)
    assert "Sky grid" in shown and "Catalogue stars" in shown
    sky = next(d for (name, _k), d in t._data.items() if name == "sky")
    assert isinstance(sky, si.SkyData) and {ln.kind for ln in sky.lines} == {"ra", "dec"}
    assert "Object names" in shown and (any(o.name == "IC 1805" for o in sky.objects) or sky.why)
    assert 4 <= len(sky.lines) <= 20 and sky.stars is not None and 100 < len(sky.stars) <= si.CATALOGUE_MAX
    assert sky.stars[:, 2].max() < si.CATALOGUE_G_MAX and (np.diff(sky.stars[:, 2]) >= 0).all()
    assert any(isinstance(o, SkyOverlay) for o in v.canvas.overlays)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ln in sky.lines:                                        # a line is where the solution puts its label
            pts = ln.pieces[0][::5]
            ra, dec = prep.wcs.all_pix2world(pts[:, 0].astype(float), pts[:, 1].astype(float), 0)
            want = ln.label.replace("\u2212", "-").replace("\u00b0", "")
            if ln.kind == "dec":
                assert np.allclose(dec, float(want), atol=2e-3), (ln.label, dec[:3])
            else:
                h, m = int(want[:2]), int(want[3:5])
                dra = (ra - (h + m / 60) * 15 + 180) % 360 - 180
                assert np.abs(dra).max() < 2e-3, (ln.label, ra[:3])

    def grab():
        QApplication.processEvents()
        return v.canvas.grab().toImage().convertToFormat(QImage.Format_RGB888)

    def diff(a, b):
        pa = np.frombuffer(a.constBits(), np.uint8).reshape(a.height(), a.bytesPerLine())[:, :a.width() * 3]
        pb = np.frombuffer(b.constBits(), np.uint8).reshape(b.height(), b.bytesPerLine())[:, :b.width() * 3]
        return int((pa != pb).sum())
    on = grab()
    t._sky("stars", False)
    no_stars = grab()
    t._sky("grid", False)
    t._sky("objects", False)                                        # the named objects found in the field
    plain = grab()
    assert diff(on, no_stars) > 200 and diff(no_stars, plain) > 200
    ov = v.canvas.overlays[:]
    v.canvas.overlays.clear()
    assert diff(plain, grab()) == 0                                 # all off: the picture as it is
    v.canvas.overlays[:] = ov
    t._sky("grid", True)
    t._sky("stars", True)
    t._sky("objects", True)
    found = list(sky.objects)
    sky.objects.append((3000.0, 2000.0, "IC 1805", "nebula"))       # all the annotation package has to fill
    try:
        with_obj = grab()
        sky.objects.clear()
        assert diff(with_obj, grab()) > 100
    finally:
        sky.objects[:] = found
    shot(win, "inspect_astrometry")
    t.tab_of.pop("astrometrie", None)


def test_the_labels_of_the_sky_grid_are_the_round_values_of_the_solution():
    assert si.ra_label(40.75, 120.0) == "02h43m" and si.ra_label(40.7541, 10.0) == "02h43m01s"
    assert si.ra_label(359.9999, 60.0) == "00h00m" and si.ra_label(0.0, 1.0) == "00h00m00s"
    assert si.dec_label(61.0, 60.0) == "+61\u00b0" and si.dec_label(-0.5, 5.0) == "\u221200\u00b030\u2032"
    assert si.dec_label(61.25, 10.0) == "+61\u00b015\u2032"
    # the steps: the smallest round one that keeps about six lines across the field
    assert si._step(5.0 * 60, si._DEC_STEPS) == 60 and si._step(0.2 * 60, si._DEC_STEPS) == 2
    assert si._step(30.0 * 240, si._RA_STEPS) == 1800 and si._step(1e9, si._RA_STEPS) == 3600


def test_the_colour_block_says_its_stars_and_its_white_balance(win):
    win.tune.tab_of["couleur"] = "fit"
    try:
        inspect(win, "couleur")
        rows = dict(win.steps.facts(win.s.path, "couleur"))
        # the stars of the result shown (its manifest's, not a number of one run); a colour stage of version 4 adds the
        # white balance (red, green, blue multipliers and the stars behind them), shown whenever the result has it
        out = win.steps.outputs(win.s.path, "color", win.steps.which_for(win.s.path, "color"))
        assert rows[texts.metric_label("stars")] == str(out.metrics["stars"])
        has_wb = "white_balance" in out.metrics
        assert (texts.metric_label("white_balance") in rows) == has_wb
        if has_wb:
            assert len(rows[texts.metric_label("white_balance")].replace(",", ".").split(" / ")) == 3
            assert rows[texts.metric_label("white_balance_stars")] == str(out.metrics["white_balance_stars"])
        shot(win, "inspect_color")
    finally:
        win.tune.tab_of.pop("couleur", None)


def test_the_selection_chart_updates_the_kept_count_through_selection_preview(win):
    """With the Thresholds selection rule (set in memory: the model is read-only), the chart's grips move the
    thresholds and the kept count is the engine's selection_preview."""
    stored = win.s.path.params.setdefault("stack", {})
    engine = win.s.p.workflow(1).params.setdefault("stack", {})         # what api.selection_preview reads below
    before, engine_before = stored.get("selection", "<unset>"), engine.get("selection", "<unset>")
    stored["selection"] = engine["selection"] = "thresholds"
    win.s._cache.clear()
    try:
        inspect(win, "selection")
        t = win.tune
        ch = t.chart
        assert isinstance(ch, FramesChart) and ch.n == N_LIGHTS and t.table.total == N_LIGHTS
        assert abs(ch.tmin - 60.0) < 1e-6 and abs(ch.fmax - win.s.thresholds_now()[1]) < 1e-6
        assert ch.kept() == N_LIGHTS
        # the frames below the chart: the common viewer, one frame per point, with the numbers of the frame table
        v = t.view
        assert isinstance(v, ImageView) and len(v.frames()) == N_LIGHTS
        assert v.frames()[0].name == t.table.rows[0].name and v.frames()[0].info["fwhm"] == t.table.rows[0].fwhm
        if win.s.state(win.s.path, "register") != "done":
            pytest.skip("Registration of the test project is Outdated: selection_preview needs it current")
        from PySide6.QtTest import QTest
        top, bot = ch._panels()
        (f0, f1), (t0, t1) = ch._ranges()
        start = ch._xy(bot, 0, ch.tmin, t0, t1)
        trs = sorted(100 * r.transparency for r in t.table.rows)
        goal = min(99.5, (trs[1] + trs[2]) / 2)                          # leaves out at least the least transparent
        dest = ch._xy(bot, 0, goal, t0, t1)
        QTest.mousePress(ch, Qt.LeftButton, Qt.NoModifier, QPoint(int(bot.left() + 30), int(start.y())))
        QTest.mouseMove(ch, QPoint(int(bot.left() + 30), int(dest.y())))
        QTest.mouseRelease(ch, Qt.LeftButton, Qt.NoModifier, QPoint(int(bot.left() + 30), int(dest.y())))
        settle(win)
        assert t.chart is ch                              # issue #24: the chart moved in place, not built again
        assert abs(ch.tmin - goal) < 2, (ch.tmin, goal)
        want = api.selection_preview(win.s.p, 1, ch.tmin / 100, ch.fmax / win.s.fwhm_median()).kept
        assert want < N_LIGHTS and ch.kept() == want
        assert t.kept_label.text().startswith(f"{want} of {N_LIGHTS} frames kept")
        assert win.s.pending_change is not None and win.s.pending_change["step"] == "selection"
        assert any(b.text().startswith("Try as variant") for b in t.panel.findChildren(QPushButton))
        win.s.pending_change = None
        shot(win, "inspect_selection")
    finally:
        for d, b in ((stored, before), (engine, engine_before)):
            if b == "<unset>":
                d.pop("selection", None)
            else:
                d["selection"] = b
        win.s._cache.clear()


def test_a_frame_is_picked_on_the_chart_and_excluded_by_hand_for_the_whole_project(win):
    inspect(win, "selection")
    t = win.tune
    s = win.s
    before = {st: s.stage_key(s.path, st) for st in ("inventory", "calibrate", "register", "stack", "color")}
    t._pick(3)
    settle(win)
    row = t.table.rows[3]
    v = t.view
    assert t.picked == 3 and v.index() == 3 and v.wait(60) and v.prepared() is not None    # the raw frame, read
    assert v.image().name == row.name and t.chart.current == 3
    v.next_frame()                                                    # the viewer moves: the chart follows
    assert t.chart.current == 4
    v.set_index(3)
    v.toggle_excluded()                                               # Exclude (X): by hand, for the whole project
    settle(win)
    assert (row.session, row.name) in s.frames_excluded
    after = {st: s.stage_key(s.path, st) for st in before}
    assert after["stack"] != before["stack"]                            # Exclude changes the Stacking key
    assert all(after[k] == before[k] for k in ("inventory", "calibrate", "register"))
    assert t.table.rows[3].excluded and 3 in t.chart.hand
    assert t.view.frames()[3].info["excluded"] and t.view.band.toggle.text() == "Include"
    s.set_excluded([(row.session, row.name)], False)                    # the shared window goes back as it was
    settle(win)
    assert s.stage_key(s.path, "stack") == before["stack"]


def test_excluding_a_calibrated_frame_with_x_keeps_the_viewer_and_its_frame(win):
    """Issue #24, case 2: the scan of the sources that follows Exclude (X) built the tab again, its viewer back on
    its first frame."""
    win.tune.tab_of["calibration"] = "calibrated"
    inspect(win, "calibration")
    t, s = win.tune, win.s
    v = t.view
    assert v is not None and len(v.frames()) > 3
    v.set_index(2)
    settle(win)
    it = t._seq_items[2]
    v.toggle_excluded()
    settle(win)
    try:
        assert (it.session, it.name) in s.frames_excluded
        assert t.view is v and v.index() == 2 and v.frames()[2].info["excluded"]
    finally:
        s.set_excluded([(it.session, it.name)], False)                  # the shared window goes back as it was
        settle(win)
        t.tab_of.pop("calibration", None)


def test_a_block_not_computed_says_so_and_offers_run_to_here(win):
    s = win.s
    q = s.add_path("Classic stack", {"debruitage": False, "etoiles": "garder", "fond": "simple"})
    s.set_param(q, "gradient", "degree", 3)                               # a Gradient the store does not have
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
        assert not any("Preview region" in x for x in texts_)       # ◇: Spec mode only
        shot(win, "inspect_pending")
        prefs.set("spec_mode", True)
        t.refresh()
        settle(win)
        assert any("Preview region" in x for x in texts_of(t))
        prefs.set("spec_mode", False)
    finally:
        s.remove_path(s.index_of(q.pid))
        s.select(index=0, step="empilement")
        settle(win)


def test_compare_shows_two_workflows_side_by_side_in_sync(win):
    """Compare with: the previous steps of the workflow (a calibrated frame, the half-stacks), side by side with zoom
    and pan linked, or split; the same stretch on both when asked."""
    inspect(win, "empilement")
    t = win.tune
    v = t.view
    names = [v.cmp_cb.itemText(i) for i in range(v.cmp_cb.count())]
    i = names.index("Half-stack A")
    v.set_compare_mode("side")
    v.cmp_cb.setCurrentIndex(i)
    assert v.wait(120) and v._cmp is not None and v.canvas2.isVisible()
    v.zoom_to(1.0, 3000, 2000)
    QApplication.processEvents()
    assert abs(v.canvas2.scale - 1.0) < 1e-9 and (round(v.canvas2.cx), round(v.canvas2.cy)) == (3000, 2000)
    v.set_same_stretch(True)
    assert v._cmp.stretch.same(v.stretch().adapted(v._cmp.prep.channels))
    shot(win, "inspect_compare")
    v.set_compare(None)
    v.set_compare_mode("split")


def test_no_disk_access_in_the_gui_thread(win):
    """A debug hook counts what the GUI thread opens, lists or writes under the project and its raws: the views ask the
    worker (and the viewer's threads) for every image, table, thumbnail and scan."""
    inspect(win, "calibration")
    sv = win.tune.service
    with si.GuiDiskWatch([F_PROJECT, DATA]) as w:
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
        for step, tabs in (("calibration", ("masters", "darks", "calibrated")), ("alignement", ("reference", "table")),
                           ("selection", ()), ("empilement", ("stack", "rejection", "weights")),
                           ("astrometrie", ("solution",)), ("couleur", ("colors", "fit")), ("debruitage", ()),
                           ("etoiles", ()), ("fond", ("result", "removed")), ("sources", ())):
            for key in tabs or (None,):
                if key:
                    win.tune.tab_of[step] = key
                inspect(win, step)
        win.frames.show_page(1)
        settle(win)
        win.browser.set_mode(0)
        win.browser.select_group("darks")
        win.browser.select_group("lights")
        win.browser.set_mode(1)
        settle(win)
        win.tune.tab_of.clear()
        win.tune.select(step="empilement")
        win.tune.refresh()
        settle(win)
        v = win.tune.view
        v.auto_b.setChecked(False)
        v.pixel_text(100.5, 100.5)
        v.auto_b.setChecked(True)
        win.frames.show_page(0)
        settle(win)
        where = list(w.where)
    mine = [(path, fr[-3:]) for path, fr in where
            if fr and fr[-1].split(":")[0] in ("tune.py", "frames.py", "sources.py", "steps.py", "charts.py", "inspect.py",
                                                "imageview.py")]
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
                   "FWHM", "Kept frames"]
    row = {h: t.item(0, i).text() for i, h in enumerate(hdr)}
    assert row["Lights"] == str(N_LIGHTS) and row["Flats"] == str(N_FLATS) and row["Darks"] == str(N_DARKS)
    assert row["Bias"] == "none" and row["Exposure"] == "15 s" and row["ISO / gain"] == "5000"
    assert re.fullmatch(r"\d+ → \d+ \w+ \d{4}", row["Night"]), row["Night"]
    assert re.fullmatch(r"\d\.\d px", row["FWHM"]) and row["Kept frames"] == f"{N_LIGHTS} / {N_LIGHTS}"
    assert row["Temp."] == "–"
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


def test_a_long_health_check_scrolls_and_never_makes_the_window_taller(win, monkeypatch):
    """GUI exploration 2: 16 warnings took a 768 window to 920 px (and it stayed so): the Sessions page scrolls."""
    from PySide6.QtWidgets import QScrollArea

    from astro_gpu.client import frames
    inspect(win, "sources")
    f = win.frames
    monkeypatch.setattr(frames, "check_lines", lambda _d: [("warn", f"Session 1: DSC{i:05d}.ARW cannot be read")
                                                          for i in range(60)])
    f._build()
    settle(win)
    assert isinstance(f.pages.widget(0), QScrollArea)
    assert win.minimumSizeHint().height() < 700 and win.minimumHeight() < 700
    monkeypatch.undo()
    f._build()
    settle(win)


def test_the_strip_shows_the_open_block(win):
    """GUI exploration 2: at 1366 x 768 the strip of a long chain showed only the first letter of the block opened."""
    size = win.size()
    try:
        from astro_gpu.client.blocks import chain
        win.resize(700, size.height())
        inspect(win, [k for k in chain(win.s.path) if k != "post"][-1])        # the last block: at the strip's end
        settle(win)
        b, sa = win.tune.strip.current, win.tune.strip_scroll
        bar = sa.horizontalScrollBar()
        # scrolled to it when the blocks overflow (the window may stay wider than 700 px: its minimum width depends
        # on what the module's earlier tests left open; under a parallel run it did)
        assert b is not None and b.isVisible() and (bar.maximum() == 0 or bar.value() > 0), (
            win.width(), sa.viewport().width(), win.tune.strip.width(), bar.maximum())
        left = b.mapTo(sa.viewport(), b.rect().topLeft()).x()
        assert left >= 0 and left + b.width() <= sa.viewport().width(), (left, b.width(), sa.viewport().width())
        # control round: only the blocks scroll, "◂ Graph" and the workflow list stay in view
        st = win.tune.strip
        back = next(x for x in st.findChildren(QPushButton) if x.text().startswith("◂"))
        assert not sa.isAncestorOf(back) and back.isVisible()
        assert 0 <= back.mapTo(win, back.rect().topLeft()).x() < win.width()
        # the graph's blocks with the graph's names: no Output (never a block), the Session first
        names = [x.text() for x in st.blocks.findChildren(QPushButton)]
        assert "Output" not in names and names[-1] == "Post" and names.count(names[0]) == 1, names
    finally:
        win.resize(size)
        settle(win)


def test_the_panel_says_the_state_the_graph_shows(win):
    """Control round: the last block (which holds the Output) said "Up to date" in its panel and in Inspect where the
    graph said "Outdated"; a session's Calibration said "Up to date" never computed. One state: the graph's."""
    from astro_gpu.client.blocks import STATE_WORD, Graph, chain
    keys = [x for x in chain(win.s.path) if x != "post"]
    k = keys[-1]                                            # the last block: it holds the Output
    inspect(win, k)
    g = Graph(win.s)
    want = i18n._(STATE_WORD[g.state(g.node_of(win.s.path, k))])
    chips = [lb.text() for lb in win.tune.panel.findChildren(QLabel)
             if lb.isVisible() and lb.text() in {i18n._(x) for x in STATE_WORD.values()}]
    assert chips[:1] == [want], (k, chips, want)


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
        assert "All together" in texts_of(f.pages.widget(0))          # compact names: the sentence in its tooltip
        assert "One stack per session" not in " ".join(texts_of(f.pages.widget(0)))        # ◇
        prefs.set("spec_mode", True)
        f._build()
        settle(win)
        b = [x for x in f.findChildren(QPushButton) if x.text().startswith("One per session")][0]   # compact name
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
@pytest.fixture()
def scratch(tmp_path):
    folder = scratch_project(tmp_path / "heart")
    p = api.open_project(folder)
    api.add_workflow(p, "Classic stack", "Classic stack", None)
    api.add_workflow(p, "Full", "Full", None)
    api.save(p)
    w = open_window(folder, readonly=False)
    yield w
    w.s.flush()
    w.close()


def test_a_setting_of_the_block_shown_keeps_its_picture_on_screen(scratch):
    """Issue #24: Away from emission ticked in the Gradient block: the block is Outdated, its picture (the previous
    result) is the same, so the viewer stays (built again, the image flickered); the header says it is Outdated."""
    w = scratch
    s = w.s
    full = next(p for p in s.paths if p.name == "Full")
    if s.state(full, "gradient") != "done":
        pytest.skip("Gradient of the test project is not up to date")
    inspect(w, "fond", index=s.index_of(full.pid))
    t = w.tune
    v0, img0 = t.view, t.view.image().identity()
    t.grad_editor._set_avoid(not t.grad_editor.avoid)
    settle(w)
    assert s.state(full, "gradient") == "stale"
    assert t.view is v0 and t.view.image().identity() == img0
    assert "previous result" in texts_of(t)


def test_an_outdated_block_shows_its_previous_result(scratch):
    w = scratch
    s = w.s
    full = s.paths[0]
    if s.state(full, "stack") != "done":
        pytest.skip("Stacking of the test project is not up to date")
    s.set_param(full, "stack", "kappa", 4.0)                              # Stacking and what follows are Outdated now
    settle(w)
    assert s.state(full, "stack") == "stale" and s.previous_key(full, "stack")
    inspect(w, "empilement", index=0)
    t = w.tune
    vi = t.view.image()
    assert vi is not None and ("stack", s.previous_key(full, "stack")) in vi.identity()[-1]
    assert t.view.wait(120) and t.view.prepared().size() == GRID
    texts_ = texts_of(t)
    assert "previous result" in texts_ and any(x.startswith("Run to here") for x in texts_)
    assert dict(w.steps.facts(full, "empilement"))[texts.metric_label("frames_used")] == str(N_LIGHTS)
    shot(w, "inspect_previous")
    run = [b for b in t.findChildren(QPushButton) if b.text().startswith("Run to here")][0]
    got = []
    t.runHere.connect(lambda: got.append(1))
    run.click()
    assert got == [1]
    inspect(w, "debruitage", index=0)
    assert w.tune.view.image() is not None and "previous result" in texts_of(w.tune)
    q = s.add_path("Classic stack", {"debruitage": False, "etoiles": "garder", "fond": "simple"})
    s.set_param(q, "gradient", "degree", 3)                               # never computed: no previous result
    inspect(w, "fond", index=s.index_of(q.pid))
    assert "Not computed yet" in texts_of(w.tune)
    s.remove_path(s.index_of(q.pid))
    s.set_param(full, "stack", "kappa", None)
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
    inspect(w, "debruitage")
    t = w.tune
    assert t.cmp_pid == full.pid and s.path is q
    v = t.view
    if v is None:                                                       # the variant's Denoise is not computed
        assert "Not computed yet" in texts_of(t)
        return
    names = [v.cmp_cb.itemText(i) for i in range(v.cmp_cb.count())]
    assert any(n.startswith(full.name + "  ·  ") for n in names) and v.compare_mode() == "side"


def test_the_frames_of_each_kind_are_chosen_by_their_files(env, monkeypatch, tmp_path):
    """The import: the darks chosen by three of their files, the flats taken away; read again, then into the project
    exactly so (the user's rule, 5 November: each kind chosen by selecting its files)."""
    root = tmp_path / "ws" / "Chosen"
    project2.create(root, "Chosen", Target(name="IC 1805"))
    w = open_window(root, readonly=False)
    try:
        darks = sorted(str(p) for p in (DATA / "darks").iterdir())[:3]

        def wait_scan(d):
            end = time.monotonic() + 120
            while (d.scan is None or not d.ok.isEnabled()) and time.monotonic() < end:
                QApplication.processEvents()
                time.sleep(0.01)
            assert d.scan is not None

        def fake_exec(self):
            wait_scan(self)
            assert "darks" in self.suggested and len(self.scan.frames["darks"]) == 15
            self.rows["darks"].changed.emit("darks", darks)                # Choose files…: three of them
            wait_scan(self)
            self.rows["flats"].changed.emit("flats", [])                   # ✕: no flats
            wait_scan(self)
            assert len(self.scan.frames["darks"]) == 3 and not self.scan.frames["flats"]
            assert "darks" not in self.suggested and "3 darks" in self.rows["darks"].detail.text()
            assert self.rows["flats"].what.text() == "None"
            self.accept()
            return 1
        monkeypatch.setattr(ImportDialog, "exec", fake_exec)
        assert import_folder(w.s, w, str(DATA / "lights"), service=w.tune.service)
        w.s.flush()
        d = project2.load(root)
        assert len(d.sessions) == 1 and sorted(d.sessions[0].darks) == darks and not d.sessions[0].flats
    finally:
        w.s.flush()
        w.close()


def test_change_frames_replaces_the_darks_of_a_session(env, monkeypatch, tmp_path):
    """Frames > Sessions: Change frames… opens the import on the session's frames; two darks chosen, the session holds
    them (and only them)."""
    from astro_gpu.client.sources import edit_session_frames
    root = tmp_path / "ws" / "Edit"
    p = project2.create(root, "Edit", Target(name="IC 1805"))
    api.add_session(p, lights=[DATA / "lights"], darks=[DATA / "darks"], flats=[DATA / "flats"], name="night")
    api.save(p)
    w = open_window(root, readonly=False)
    try:
        darks = sorted(str(x) for x in (DATA / "darks").iterdir())[:2]

        def wait_scan(d):
            end = time.monotonic() + 120
            while (d.scan is None or not d.ok.isEnabled()) and time.monotonic() < end:
                QApplication.processEvents()
                time.sleep(0.01)
            assert d.scan is not None

        def fake_exec(self):
            wait_scan(self)
            assert len(self.scan.frames["darks"]) == 15 and not self.suggested     # the session's own, no guess
            self.rows["darks"].changed.emit("darks", darks)
            wait_scan(self)
            self.accept()
            return 1
        monkeypatch.setattr(ImportDialog, "exec", fake_exec)
        assert edit_session_frames(w.s, w.s.p.sessions[0].id, w, service=w.tune.service)
        w.s.flush()
        d = project2.load(root)
        assert sorted(d.sessions[0].darks) == [Path(x).as_posix() for x in darks] and d.sessions[0].flats
    finally:
        w.s.flush()
        w.close()


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
        assert b.viewer.wait(60) and b.viewer.prepared() is not None and len(b.viewer.frames()) == 40
        b.set_mode(0)                                                   # the grid's thumbnails: the worker's
        settle(w)
        time.sleep(1.0)
        settle(w)
        assert len(w.tune.service.thumb_cache) >= 20
        have = [b.grid.item(i) for i in range(b.grid.count()) if i < 20]
        assert all(not it.icon().isNull() for it in have)
        shot(w, "frames_browse_import")
        # Add session…: everything goes into a session of its own
        assert import_folder(w.s, w, str(DATA / "flats"), new_session=True, service=w.tune.service)
        assert len(w.s.p.sessions) == 2
        # the same folder twice: said, not added
        from PySide6.QtWidgets import QToolTip
        assert not import_folder(w.s, w, str(DATA / "lights"), service=w.tune.service)
        assert len(w.s.p.sessions) == 2 and QToolTip.text() == "Folder already in project"   # said, not blocking (10.2)
    finally:
        w.s.flush()
        w.close()


def test_the_frame_chart_and_the_browser_keep_and_exclude_frames(win):
    inspect(win, "sources")
    win.frames.show_page(1)
    settle(win)
    b = win.browser
    s = win.s
    assert b.group == "lights" and len(b.items) == N_LIGHTS
    # the viewer: the frames as its sequence, with what the header and the engine say
    v = b.viewer
    assert isinstance(v, ImageView) and len(v.frames()) == N_LIGHTS and b.mode == 1
    info = v.frames()[0].info
    assert info["exposure"] == 15.0 and info["iso"] == 5000.0 and info["fwhm"] and info["transparency"]
    assert v.wait(60) and v.prepared() is not None and v.prepared().cfa == "RGGB"
    # sort by star size, the worst frame last: the viewer follows the order
    b.sort_cb.setCurrentIndex(b.sort_cb.findData("fwhm"))
    idx = b.order()
    assert b.items[idx[0]].fwhm <= b.items[idx[-1]].fwhm and v.frames()[0].name == b.items[idx[0]].name
    b.sort_cb.setCurrentIndex(b.sort_cb.findData("time"))
    # X in the viewer excludes the shown frame for the whole project, X again keeps it
    b._goto(5)
    row = b.items[5]
    assert v.image().name == row.name
    key0 = s.stage_key(s.path, "stack")
    v.toggle_excluded()
    settle(win)
    assert (row.session, row.name) in s.frames_excluded and b.items[5].excluded
    assert s.stage_key(s.path, "stack") != key0 and b.count.text().endswith("1 excluded")
    v.toggle_excluded()
    settle(win)
    assert not s.frames_excluded and s.stage_key(s.path, "stack") == key0
    # Blink plays the frames
    v.blink_b.setChecked(True)
    assert v._blink.isActive()
    v.blink_b.setChecked(False)
    assert not v._blink.isActive()
    # the calibration frames: darks, with the level and the noise measured in the info band
    b.select_group("darks")
    settle(win)
    time.sleep(1.5)
    settle(win)
    assert b.group == "darks" and len(b.items) == N_DARKS and len(v.frames()) == N_DARKS
    assert any(win.tune.service.stats.get(it.path) for it in b.items)
    assert v.band.mode == "on" and v.band.toggle.isEnabled()      # a calibration frame leaves its master
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


def test_missing_diagnostics_are_said_plainly_in_the_language():
    e = "rejection maps: missing: recompute Stacking to get it"
    assert si.missing_text(e) == "Rejection maps: older result · recompute Stacking"

    class Diag:
        missing = [e, "roundness: missing: recompute Registration to get it"]
    assert len(si.missing_of(Diag(), "rejection")) == 1 and len(si.missing_of(Diag())) == 2
    assert si.missing_of(None) == [] and si.missing_text("something else") == "Not in this result."
    appmod.apply_language("fr")
    try:
        fr = si.missing_of(Diag())
        assert fr[0].startswith("Les cartes de rejet : résultat ancien") and "recalculer Empilement" in fr[0]
        assert fr[1].startswith("La rondeur : ") and "Alignement" in fr[1]
        assert si.not_yet("stack") == "À recalculer : Empilement"
    finally:
        appmod.apply_language("en")


def test_the_colour_picture_has_a_grey_sky_and_untouched_colours():
    a = np.stack([np.full((50, 60), 100.0), np.full((50, 60), 80.0), np.full((50, 60), 40.0)]).astype(np.float32)
    a[:, 10, 10] += np.array([500.0, 300.0, 100.0], np.float32)
    b = si.grey_sky(a.copy())
    med = np.median(b.reshape(3, -1), 1)
    assert np.allclose(med, med.mean())                                     # the sky grey
    assert np.allclose(b[:, 10, 10] - med, [500.0, 300.0, 100.0])           # an offset only: the star's colour kept
    assert np.allclose(si.apply_matrix(2 * np.eye(3), a), 2 * a)


def test_a_calibration_frame_is_left_out_frame_by_frame_and_taken_back(tmp_path):
    """Exclude on a dark: the session lists its folder frame by frame without it (the engine's lists take files);
    the frame stays in the browser, marked; Include lists the folder again, as it was."""
    from contextlib import contextmanager
    darks = tmp_path / "darks"
    darks.mkdir()
    for i in range(3):
        (darks / f"D{i}.ARW").write_bytes(b"raw")
    p = api.create_project(tmp_path / "proj", "T")
    s = api.add_session(p, lights=[str(tmp_path / "lights")], darks=[str(darks)])

    class Model:                                   # the model's own method, on a model with no worker
        set_calibration_excluded = ProjectModel.set_calibration_excluded

        def __init__(self):
            self.p, self.edits, self.reloads = p, 0, 0

        @contextmanager
        def _edit(self):
            self.edits += 1
            yield

        def reload_sources(self):
            self.reloads += 1
    m = Model()
    items = si.browse_group(p, None, "darks")
    assert [i.name for i in items] == ["D0.ARW", "D1.ARW", "D2.ARW"] and not any(i.excluded for i in items)
    assert si.set_calibration_excluded(m, items[1], True, items)
    assert sorted(Path(x).name for x in s.darks) == ["D0.ARW", "D2.ARW"] and m.edits == m.reloads == 1
    items = si.browse_group(p, None, "darks")
    assert [(i.name, i.excluded) for i in items] == [("D0.ARW", False), ("D1.ARW", True), ("D2.ARW", False)]
    assert si.set_calibration_excluded(m, items[0], True, items)
    items = si.browse_group(p, None, "darks")
    assert not si.set_calibration_excluded(m, items[2], True, items)        # the last frame of a folder stays
    assert si.set_calibration_excluded(m, items[0], False, items)
    items = si.browse_group(p, None, "darks")
    assert si.set_calibration_excluded(m, items[1], False, items)
    assert s.darks == [str(darks)]                                           # the folder, as it was
    # the model's method alone lists the folder itself when it is not given the worker's listing
    assert m.set_calibration_excluded(s.id, "darks", str(darks / "D2.ARW"), True)
    assert sorted(Path(x).name for x in s.darks) == ["D0.ARW", "D1.ARW"]
    assert m.set_calibration_excluded(s.id, "darks", str(darks / "D2.ARW"), False)
    assert s.darks == [str(darks)]
    with pytest.raises(ValueError):
        m.set_calibration_excluded(s.id, "lights", str(darks / "D2.ARW"), True)


def test_the_drift_tab_shows_the_path_of_the_session_and_no_tab_hides(win):
    """The x / y path of the session (the "Lissajous"), coloured by time, is drawn under the drift in the Drift tab
    and has its own tab; on a narrow window the tabs of the block shorten their names, none goes behind scroll
    arrows."""
    from astro_gpu.client.charts import DriftChart, TrackChart
    win.tune.tab_of["alignement"] = "drift"
    inspect(win, "alignement")
    t = win.tune
    drift = [c for c in t.findChildren(DriftChart) if c.isVisibleTo(t)]
    track = [c for c in t.findChildren(TrackChart) if c.isVisibleTo(t)]
    assert len(drift) == 1 and len(track) == 1
    tr = track[0]
    assert np.isfinite(tr.d.dx).sum() > 10 and tr.colorbar is not None and tr.colorbar[0] == "Time"
    bar = next(b for b in t.findChildren(QTabBar) if b.isVisibleTo(t) and b.count() == 6)
    assert not bar.usesScrollButtons() and bar.tabToolTip(3) == "Drift path (x / y)"
    win.tune.tab_of["alignement"] = "track"
    inspect(win, "alignement")
    assert [c for c in t.findChildren(TrackChart) if c.isVisibleTo(t)]
    win.tune.tab_of.pop("alignement", None)


def test_the_offsets_are_made_when_the_registration_table_has_none(tmp_path, monkeypatch):
    """A registration whose table gives no offsets (the sensor size unknown to it) still draws its drift: the sensor
    size comes from the calibration's cache description, the offsets from the registration's polynomials."""
    from astro_gpu.client.charts import RegistrationData
    cal, reg = tmp_path / "calibrate", tmp_path / "register"
    cal.mkdir()
    reg.mkdir()
    (cal / "cache.json").write_text(json.dumps({"width": 200, "height": 100}), encoding="utf-8")
    coef = np.zeros((3, 2, 10))
    for i, (dx, dy) in enumerate(((0, 0), (5, -2), (9, 4))):    # the frame's centre lands at (100 - dx, 50 - dy)
        coef[i, 0, 0], coef[i, 1, 0] = 100 - dx, 50 - dy
        coef[i, 0, 1], coef[i, 1, 2] = 100, 50                   # no rotation, the same scale
    np.savez(reg / "reg.npz", coef=coef)
    dirs = {"calibrate": cal, "register": reg}
    monkeypatch.setattr(si.api, "stage_outputs", lambda p, wid, stage, which="current": (
        api.StageOutputs(stage, "k", which, str(dirs[stage])) if stage in dirs else None))
    nan = np.full(3, np.nan)
    d = RegistrationData(["a", "b", "c"], [""] * 3, nan, nan, nan, nan, nan, nan, np.ones(3, bool), nan.copy(),
                         nan.copy(), 0, None)
    assert si.sensor_size(None, 1) == (200.0, 100.0)
    si._offsets_anyway(None, 1, "current", d)
    assert d.dx.tolist() == pytest.approx([0, 5, 9]) and d.dy.tolist() == pytest.approx([0, -2, 4])


def test_the_pictures_are_placed_on_the_reference_frame_of_their_registration(tmp_path, monkeypatch):
    """Every picture on the stack grid sits on the registration's reference frame at the drizzle scale; an engine
    preview of it (a reduced copy) and the final image (a crop at half size) too: Compare puts them on each other."""
    from astro_gpu.client import imageview as iv
    st = tmp_path / "stack"
    st.mkdir()
    (st / "drz_meta.json").write_text(json.dumps({"scale": 1.5, "width": 600, "height": 450, "sensor": [400, 300]}),
                                      encoding="utf-8")
    monkeypatch.setattr(si.api, "stage_outputs", lambda p, wid, stage, which="current": (
        api.StageOutputs("stack", "s1", which, str(st), manifest={"inputs": {"register": "r1"}})
        if stage == "stack" else None))
    full = si.stack_place(None, 1)(600, 450)
    assert full.frame == ("reference frame", "r1") and full.m == pytest.approx((1 / 1.5, 0, 0, 0, 1 / 1.5, 0))
    preview = si.stack_place(None, 1)(150, 113)                     # a quarter of the grid (rounded up)
    assert preview.m[0] == pytest.approx(4 / 1.5) and preview.m[4] == pytest.approx(450 / 113 / 1.5)
    final = si.render_place(None, 1, "current", (480, 320), (64, 96), (240, 160))(240, 160)
    assert final.frame == full.frame
    # pixel (10, 20) of the final image is pixel (64 + 20, 96 + 40) of the stack: (84 / 1.5, 136 / 1.5) on the frame
    assert final.matrix() @ np.array([10.0, 20.0, 1.0]) == pytest.approx([84 / 1.5, 136 / 1.5, 1.0])
    ref = si.reference_frame_place(None, 1)(400, 300)
    assert ref.frame == full.frame and ref.m == iv.Placement(ref.frame).m


def test_the_output_says_what_the_final_image_is():
    """One line under Output says what the final image is and which steps it holds: the Gradient removed picture with
    Post's stretch; the Stars block shows its two images against the image with its stars."""
    from astro_gpu.client import steps as st
    base = {"stack", "color", "gradient", "render_simple"}
    assert st.output_note("render_simple", base) == st.OUTPUT_SIMPLE_NOTE
    assert st.output_note("render_simple", base | {"denoise"}) == st.OUTPUT_SIMPLE_DENOISED_NOTE
    assert st.output_note("render_simple", {"stack", "color", "render_simple"}) == st.OUTPUT_COLOR_NOTE
    assert st.output_note("render_simple", {"stack", "render_simple"}) == st.OUTPUT_RAW_NOTE
    assert st.output_note("anything else", base) == st.OUTPUT_NOTE
    assert "Final image = Gradient removed, then the steps of Post" in st.OUTPUT_SIMPLE_NOTE
    assert not hasattr(st, "OUTPUT_ENGINE_NOTE") and not hasattr(st, "OUTPUT_LEARNED_NOTE")


def test_the_tables_and_the_charts_of_the_blocks(qapp):
    from astro_gpu.client.tune import ColorFitChart, WeightsChart, colour_index, make_table, per_channel, weight_rows
    t = make_table(["Frame", "Weight"], [["b", 2.0], ["a", 10.0], ["c", None]], {1: 2}, [False, False, True])
    assert [t.item(r, 0).text() for r in range(3)] == ["b", "a", "c"]           # as given until a header is clicked
    assert t.item(2, 1).text() == "–" and t.item(0, 1).text() == "2.00"
    t.sortItems(1)
    assert [t.item(r, 0).text() for r in range(3)] == ["c", "b", "a"]           # by the number, not the text
    assert colour_index(0.5) == 0.5 and abs(colour_index((0.4, 0.4, 0.2)) + 2.5 * np.log10(0.5)) < 1e-9
    assert np.isnan(per_channel(None)).all() and list(per_channel(2.0)) == [2.0, 2.0, 2.0]
    rows = [api.FrameRow(i, f"p{i}", f"f{i}.ARW", 1, "2026-08-17T00:00:00", 3.0, 1.0, 0.4, 0.5 + i / 10,
                         "" if i else "low transparency", False) for i in range(3)]

    class Bundle:
        frames = api.FrameTable(rows, 2, 0.6, 2.0, "stack")
        diag = api.StackDiag(weights=[("f0.ARW", None, None, None), ("f1.ARW", 0.6, (1.0, 2.0, 3.0), (1.1, 1.0, 0.9)),
                                      ("f2.ARW", 0.7, (1.5, 2.5, 3.5), (1.2, 1.0, 0.8))],
                             excluded={"f0.ARW": "low transparency"})
    wr = weight_rows(Bundle())
    assert [r["status"] for r in wr] == ["low transparency", "", ""] and wr[1]["scale"] == (1.1, 1.0, 0.9)
    ch = WeightsChart(wr)
    assert {p.title for p in ch.panels} >= {"Weight", "Normalisation scale", "Normalisation offset (ADU)"}
    stars = [api.StarFit(10.0, 20.0, (0.4, 0.4, 0.2), (0.33, 0.33), (0.33, 0.33), 0.002, True, bp_rp=0.8),
             api.StarFit(30.0, 40.0, (0.5, 0.3, 0.2), (0.40, 0.35), None, None, False, "saturated", bp_rp=1.9)]
    cf = ColorFitChart(api.ColorFit(stars=stars))
    assert any(tr.name == "rejected" for tr in cf.traces) and cf.x_title.startswith("Catalogue colour")


def test_the_stacking_tabs_say_the_frames_left_out_the_rejection_rate_and_the_coverage(qapp):
    """Inspect > Stacking: each frame left out with the engine's reason (translated), the rejected pixels next to
    Siril's 0.1-0.5 %, the drizzle's coverage variation per colour against DrizzlePac's 20 % (none for Classic)."""
    from astro_gpu.client import i18n
    from astro_gpu.client.tune import coverage_text, rejection_rate_text, weight_rows
    rows = [api.FrameRow(i, f"p{i}", f"f{i}.ARW", 1, "2026-08-17T00:00:00", 3.0, 1.0, 0.4, 0.5, "", False)
            for i in range(2)]
    why = "stars {sigma:.1f} σ larger than the night's median ({value:.1f} px for {median:.1f} px): wind, focus or seeing?"

    class Bundle:
        frames = api.FrameTable(rows, 2, 0.6, 2.0, "stack")
        diag = api.StackDiag(weights=[("f0.ARW", 0.0, None, None), ("f1.ARW", 0.7, None, None)],
                             excluded={"f0.ARW": "stars 4.0 σ larger ..."},
                             exclusions={"f0.ARW": [{"code": "fwhm", "template": why,
                                                     "args": {"sigma": 3.99, "value": 3.04, "median": 1.87}}]},
                             rejection={"rejected_pct": 0.83, "reference_pct": [0.1, 0.5]},
                             coverage={"variation": {"R": 0.26, "G": 0.08, "B": 0.19}, "estimate": {"limit": 0.2}})
    wr = weight_rows(Bundle())
    assert wr[0]["why"] == "stars 4.0 σ larger than the night's median (3.0 px for 1.9 px): wind, focus or seeing?"
    assert wr[1]["why"] == ""
    t = rejection_rate_text(Bundle.diag)
    assert "Rejected pixels: 0.83 %" in t and "Usual range: 0.1 to 0.5 %" in t and "above it" in t
    assert "#e6a23c" in t.lower()                                       # amber: outside the usual range
    c = coverage_text(Bundle.diag, True)
    assert "red" in c and "26 %" in c and "blue" in c and "even under 20 %" in c
    assert coverage_text(Bundle.diag, False) == ""                      # Classic: the colours are filled alike
    ok = api.StackDiag(rejection={"rejected_pct": 0.3, "classic": "winsorized",
                                  "reason_template": {"template": "{n} frames: winsorized sigma clipping", "args": {"n": 9}}})
    t = rejection_rate_text(ok)
    assert "#5fbf7f" in t and "Rejection: Winsorized sigma clipping" in t
    appmod.apply_language("fr")
    try:
        assert weight_rows(Bundle())[0]["why"].startswith("étoiles plus grosses de 4,0 σ que la médiane")
        assert "Pixels rejetés : 0,83 %" in rejection_rate_text(Bundle.diag)
        assert "rouge" in coverage_text(Bundle.diag, True)
        left = {x for x in i18n.missing() if "Rejected" in x or "Coverage" in x or "range" in x}
        assert not left, left
    finally:
        appmod.apply_language("en")


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


NAMES = {"Full", "Classic stack", "Classic + denoise", "Raw stack", "Session 1", "heart_a7s",
         "heart_small", "IC 1805", "OSC", "ILCE-7S", "Heart", "FWHM"}


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
            if step == "empilement":                                   # the readout under the cursor
                v = w.tune.view
                v.readout.setText(v.pixel_text(1500.4, 900.7))
            seen += texts_of(w)
            for vis in w.tune.visuals(w.s.path, step)[1:] if w.s.path is not None and step != "sources" else []:
                w.tune.tab_of[step] = vis.key
                w.tune.refresh()
                settle(w)
                seen += texts_of(w)
            w.tune.tab_of.pop(step, None)
            default_loader().clear()                                   # the pictures seen: memory for the next
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
        assert len(set(re.findall(r"\d,\d px", joined))) >= 2 and re.search(r"\d+ min", joined)   # star sizes, time
        assert re.search(r"\d+ → \d+ \S+ \d{4}", joined) and "aucun" in joined
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
        w.tune.tab_of["empilement"] = "weights"
        inspect(w, "empilement", index=0)
        take("inspect_stack_weights")
        w.tune.tab_of.clear()
        inspect(w, "debruitage")
        take("inspect_denoise")
        inspect(w, "etoiles")
        take("inspect_stars")
        inspect(w, "couleur")
        take("inspect_color")
        inspect(w, "astrometrie")
        take("inspect_astrometry")
        w.tune.tab_of["astrometrie"] = "survey"
        inspect(w, "astrometrie")
        take("inspect_survey")
        w.tune.tab_of.clear()
        for k, key in (("calibration", "masters"), ("calibration", "darks"), ("alignement", "quality"),
                       ("alignement", "table"), ("couleur", "fit"), ("fond", "removed")):
            w.tune.tab_of[k] = key
            inspect(w, k)
            take(f"inspect_{k}_{key}")
        w.tune.tab_of.clear()
        inspect(w, "post")
        take("inspect_output")
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
        for n in ("inspect_stack", "inspect_stack_weights", "inspect_denoise", "inspect_stars", "inspect_selection",
                  "inspect_color", "inspect_astrometry", "inspect_survey", "inspect_output", "frames_sessions",
                  "frames_browse", "grouping"):
            assert f"{n}_{lang}.png" in names, n


def test_the_modules_of_inspect_import_only_the_client_libraries():
    """No leftover of the old prototype (its data files, its modules), and only the allowed imports."""
    root = ROOT / "astro_gpu" / "client"
    allowed = {"__future__", "os", "sys", "threading", "collections", "dataclasses", "pathlib", "typing", "weakref",
               "numpy", "PySide6", "PIL", "astropy", "datetime", "json", "html", "warnings", "traceback", "re", "time",
               "functools"}
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
    """Every English text the modules of Inspect can show: the ``_("...")`` of the code, the captions of the pictures and the
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
                    and node.targets[0].id.endswith("_NOTE") and isinstance(node.value, ast.Constant):
                found.setdefault(node.value.value, []).append(f"{name}.py:{node.targets[0].id}")     # a line under a picture
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


def test_edits_wait_for_save_and_closing_asks(qapp, tmp_path, monkeypatch):
    """Issue #1: an edit writes nothing; the title has a star until Save; closing with edits asks: Cancel keeps the
    project open, Don't save closes it without writing."""
    root = tmp_path / "ws" / "Saved"
    p = project2.create(root, "Saved", Target(name="IC 1805"))
    api.add_workflow(p, "Full", "Full", None)
    api.save(p)
    before = (root / project2.FILE).read_bytes()
    w = open_window(root, readonly=False)
    try:
        assert not w.windowTitle().endswith("*")
        w.s.rename(w.s.paths[0].pid, "Renamed")
        assert w.windowTitle().endswith(" *") and (root / project2.FILE).read_bytes() == before
        w.save()
        assert not w.windowTitle().endswith("*") and project2.load(root).workflows[0].name == "Renamed"
        w.s.rename(w.s.paths[0].pid, "Dropped")
        answers = iter(["cancel", "discard"])
        monkeypatch.setattr(type(w), "ask_save", lambda self: next(answers))
        assert w.close_project() is False and w.project is not None              # Cancel: still open
        assert w.close_project() is True and w.project is None                   # Don't save
        assert project2.load(root).workflows[0].name == "Renamed"
    finally:
        w.close()


def test_save_as_writes_the_edits_into_the_copy_only(qapp, tmp_path, monkeypatch):
    """GUI exploration 4: Save As cancelled writes nothing (the original keeps its file, the edits stay pending); Save
    As to a new folder puts the edits in the copy, opened at once without a question, the original as it was."""
    from astro_gpu.client import app as appmod_
    root = tmp_path / "ws" / "Orig"
    p = project2.create(root, "Orig", Target(name="IC 1805"))
    api.add_workflow(p, "Full", "Full", None)
    api.save(p)
    before = (root / project2.FILE).read_bytes()
    w = open_window(root, readonly=False)
    try:
        monkeypatch.setattr(type(w), "ask_save", lambda self: pytest.fail("no question on Save As"))
        w.s.rename(w.s.paths[0].pid, "Edited")
        monkeypatch.setattr(appmod_.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: ""))
        w.save_as()                                                              # cancelled
        assert (root / project2.FILE).read_bytes() == before and w.s.dirty
        copy = tmp_path / "ws" / "Copy"
        copy.mkdir()
        monkeypatch.setattr(appmod_.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(copy)))
        w.save_as()
        assert (root / project2.FILE).read_bytes() == before
        assert project2.load(copy).workflows[0].name == "Edited"
        assert Path(w.s.root) == copy and not w.s.dirty
    finally:
        w.close()


def test_save_takes_the_name_being_typed(qapp, tmp_path):
    """GUI exploration 4: a name typed in a field without Enter, then Ctrl+S: the name typed is saved (the field
    applies what it holds before the save, as in reference apps). And Ctrl+Shift+Z redoes, as Ctrl+Y."""
    from PySide6.QtGui import QKeySequence
    from PySide6.QtWidgets import QLineEdit
    root = tmp_path / "ws" / "Typed"
    p = project2.create(root, "Typed", Target(name="IC 1805"))
    api.add_workflow(p, "Full", "Full", None)
    api.save(p)
    w = open_window(root, readonly=False)
    try:
        ed = QLineEdit("Full", w)
        ed.editingFinished.connect(lambda: w.s.rename(w.s.paths[0].pid, ed.text()))
        ed.show()
        w.activateWindow()
        ed.setFocus()
        qapp.processEvents()
        assert QApplication.focusWidget() is ed
        ed.setText("Night of the 4th")                                       # typed, no Enter
        w.save()
        assert project2.load(root).workflows[0].name == "Night of the 4th"
        assert QKeySequence("Ctrl+Shift+Z") in w.a_redo.shortcuts()
    finally:
        w.close()


def test_run_to_here_is_greyed_while_a_run_is_in_progress(win):
    """Issue #33: during a run, Run to here would only open the run in progress: greyed, with its reason."""
    s = win.s
    t = win.tune
    p = s.paths[0]
    s.run_active = True
    try:
        rb = t._run_button(p, "fond")
        assert not rb.isEnabled() and "run is in progress" in rb.toolTip()
    finally:
        s.run_active = False
    assert "run is in progress" not in t._run_button(p, "fond").toolTip()
