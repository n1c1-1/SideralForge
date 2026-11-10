"""The common image viewer (client/imageview.py) and its stretch maths (client/stretch.py), offscreen, on the CPU.

Synthetic images for the behaviour (a sky with a gradient, noise, stars and a camera tint; a Bayer mosaic; a WCS);
the stack of the test project (tests/conftest.py, read only) for the timings, skipped when the project is not on the
machine. ASTRO_GPU_SETTINGS points to a temporary file and the
user's settings file is checked untouched; the viewer itself never reads or writes settings. Screenshots go to
$ASTRO_GPU_SHOTS, else $TEMP/imageview_shots; the timings to <shots>/timings.json.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")               # the viewer needs no GPU
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

pytest.importorskip("PySide6")
from conftest import NO_TEST_PROJECT, TEST_PROJECT  # noqa: E402
from PySide6.QtCore import QEvent, QLocale, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client import imageview as iv  # noqa: E402
from astro_gpu.client import stretch as sm  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
F_PROJECT = TEST_PROJECT
STACK = next(iter(sorted((F_PROJECT / "store").glob("stack/*/drz_full.npy"))), None) if F_PROJECT.is_dir() else None
real_data = pytest.mark.skipif(STACK is None, reason=NO_TEST_PROJECT)


def _user_settings() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "astro_gpu" / "settings.json"


def _fingerprint(p: Path):
    return (p.stat().st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest()) if p.exists() else None


USER_SETTINGS = _user_settings()
USER_BEFORE = _fingerprint(USER_SETTINGS)
USER_HOME = Path.home() / ".astro_gpu"


def shots_dir() -> Path:
    d = Path(os.environ.get("ASTRO_GPU_SHOTS") or Path(os.environ.get("TEMP", ".")) / "imageview_shots")
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_language(code: str):
    i18n.set_language(code)
    QLocale.setDefault(QLocale(QLocale.French, QLocale.France) if code == "fr"
                       else QLocale(QLocale.English, QLocale.UnitedStates))


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def qapp():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS)
    return a


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    set_language("en")
    home_before = sorted(p.name for p in USER_HOME.iterdir()) if USER_HOME.is_dir() else None
    yield {"tmp": tmp_path}
    while OPEN:
        w = OPEN.pop()
        w.close()
        w.deleteLater()
    QApplication.processEvents()
    set_language("en")
    assert _fingerprint(USER_SETTINGS) == USER_BEFORE, "the user's settings file was touched"
    assert not (tmp_path / "settings.json").exists(), "the viewer wrote settings"
    if home_before is not None:
        assert sorted(p.name for p in USER_HOME.iterdir()) == home_before, "~/.astro_gpu changed"


@pytest.fixture()
def loader(qapp):
    ld = iv.Loader()
    yield ld
    ld.stop()


def sky(h=300, w=400, channels=3, seed=1, tint=(1.0, 1.6, 0.7), stars=60, level=200.0) -> np.ndarray:
    """A linear sky (C, H, W) float32 in ADU: background with a gradient, noise, stars, the camera's tint."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = level + 30 * xx / w + 10 * yy / h
    img = np.empty((channels, h, w), np.float32)
    star = np.zeros((h, w), np.float32)
    for _i in range(stars):
        cx, cy, a = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(200, 8000)
        star += a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.6 ** 2))
    for c in range(channels):
        k = tint[c] if channels == 3 else 1.0
        img[c] = (base + star) * k + rng.normal(0, 6, (h, w))
    return img


OPEN: list = []


def make_view(qapp, loader, **kw) -> iv.ImageView:
    v = iv.ImageView(loader=loader, **kw)
    v.resize(1200, 800)
    v.show()
    QApplication.processEvents()
    OPEN.append(v)
    return v


def show(v: iv.ImageView, image, **kw):
    v.set_image(image, **kw)
    assert v.wait(30)
    QApplication.processEvents()


def wheel(widget, pos: QPointF, delta: int = 120):
    ev = QWheelEvent(pos, widget.mapToGlobal(pos), QPoint(0, 0), QPoint(0, delta), Qt.NoButton, Qt.NoModifier,
                     Qt.NoScrollPhase, False)
    QApplication.sendEvent(widget, ev)


def move(widget, pos: QPointF):
    """A mouse move with no button, as the cursor passes over a widget."""
    ev = QMouseEvent(QEvent.MouseMove, pos, widget.mapToGlobal(pos), Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(widget, ev)


def simple_wcs(w=400, h=300, ra=40.0, dec=61.0, scale=2.0 / 3600):
    from astropy.wcs import WCS
    x = WCS(naxis=2)
    x.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    x.wcs.crval = [ra, dec]
    x.wcs.crpix = [w / 2 + 0.5, h / 2 + 0.5]
    x.wcs.cdelt = [-scale, scale]
    return x


# ---------------------------------------------------------------------------------------------- the stretch maths
def test_quantize_round_trip_and_nan():
    x = np.array([-5, 0, 0.5, 1, 10, 100, 1000, 9999, 10000, 20000, np.nan], np.float32)
    q = sm.quantize(x, 0.0, 10000.0)
    vals = sm.level_values(0.0, 10000.0)
    assert q.dtype == np.uint16 and q[0] == 0 and q[-1] == 0 and q[-2] == sm.Q
    back = vals[q[1:9]]
    # the steps are finest in the shadows (square-root scale): 100 ADU within a few hundredths
    assert abs(back[4] - 100) < 0.2 and abs(back[2] - 1) < 0.05
    assert np.all(np.abs(back - x[1:9]) <= np.maximum(0.25, 2 * np.sqrt(x[1:9] * 10000) / sm.Q + 0.01))


def test_stats_read_the_sky():
    img = sky()
    st = sm.compute_stats(img)
    assert st.channels == 3
    for c in range(3):
        sample = img[c].ravel()
        assert abs(st.median[c] - np.median(sample)) < 3
        mad = 1.4826 * np.median(np.abs(sample - np.median(sample)))
        assert abs(st.sigma[c] - mad) / mad < 0.15
        assert st.vmin[c] == pytest.approx(float(sample.min()))
        assert st.vmax[c] == pytest.approx(float(sample.max()))
    assert st.lo <= float(img.min()) + 1 and st.hi == pytest.approx(float(img.max()))


@pytest.mark.parametrize("mode", ["mtf", "asinh"])
def test_auto_stretch_puts_the_sky_at_a_quarter(mode):
    img = sky()
    st = sm.compute_stats(img)
    a = sm.auto_stretch(st, mode, linked=False)
    cv = sm.curves(a, st)
    for c in range(3):
        q = sm.quantize(np.float32(st.median[c]), st.lo, st.hi)
        assert cv[c][q] == pytest.approx(sm.TARGET, abs=0.03)
        assert a.black[c] == pytest.approx(st.median[c] - sm.SHADOWS * st.sigma[c], rel=1e-6)
    linked = sm.auto_stretch(st, mode, linked=True)
    assert np.ptp(linked.black) == 0 and np.ptp(linked.white) == 0 and np.ptp(linked.mid) == 0
    unlinked_spread = np.ptp(a.black)
    assert unlinked_spread > 10                         # the tint: each channel its own black point


def test_neutral_sky_makes_the_sky_grey_and_keeps_the_data():
    img = sky(tint=(1.0, 1.8, 0.6))
    before = img.copy()
    st = sm.compute_stats(img)
    a = sm.auto_stretch(st, "mtf", linked=True, neutral=True)
    t = sm.transformed(st, a.gain, a.shift)
    assert np.allclose(t["median"], st.median[1])                 # every sky on the green one
    out = sm.render(sm.quantize(img, st.lo, st.hi), sm.luts(a, st))
    med = np.median(out.reshape(-1, 3), axis=0)
    assert np.ptp(med) <= 2                                        # a grey sky
    plain = sm.render(sm.quantize(img, st.lo, st.hi), sm.luts(sm.auto_stretch(st, "mtf", True, False), st))
    assert np.ptp(np.median(plain.reshape(-1, 3), axis=0)) > 20     # linked without it: the camera's green
    assert np.array_equal(img, before)
    # not said by the caller: guessed from the sky
    assert sm.looks_camera(st) and not sm.looks_camera(sm.compute_stats(sky(tint=(1.0, 1.02, 0.99))))
    assert iv.default_stretch(iv.prepare(iv.ViewImage(img))).neutral
    assert not iv.default_stretch(iv.prepare(iv.ViewImage(img, camera=False))).neutral


@pytest.mark.parametrize("mode", sm.MODES)
def test_every_mode_is_a_monotonic_table(mode):
    img = sky()
    st = sm.compute_stats(img)
    for linked in (True, False):
        a = sm.auto_stretch(st, mode, linked=linked, neutral=True)
        lut = sm.luts(a, st)
        assert lut.shape == (3, sm.Q + 1) and lut.dtype == np.uint8
        assert np.all(np.diff(lut.astype(int), axis=1) >= 0)
        vals = st.values
        for c in range(3):
            y = vals * a.gain[c] + a.shift[c]
            assert np.all(lut[c][y < a.black[c]] == 0)
            assert np.all(lut[c][y > a.white[c]] == 255)


def test_manual_points_keep_their_order():
    st = sm.compute_stats(sky())
    a = sm.auto_stretch(st, "mtf", linked=True)
    b = sm.manual(a, "black", float(a.white[0]) + 100, -1, st)
    assert not b.auto and np.all(b.black < b.mid) and np.all(b.mid < b.white)
    c = sm.manual(sm.auto_stretch(st, "mtf", linked=False), "white", 5000.0, 2, st)
    assert c.white[2] == 5000.0 and c.white[0] != 5000.0          # one channel only
    d = sm.manual(sm.auto_stretch(st, "mtf", linked=False), "black", 100.0, -1, st)
    assert np.allclose(np.diff(d.black), np.diff(sm.auto_stretch(st, "mtf", linked=False).black))   # all, shifted


def test_full_precision_stretch_matches_the_tables():
    img = sky()
    st = sm.compute_stats(img)
    for mode in sm.MODES:
        a = sm.auto_stretch(st, mode, linked=False, neutral=True)
        exact = sm.apply(img, a, st)
        shown = sm.render(sm.quantize(img, st.lo, st.hi), sm.luts(a, st)).transpose(2, 0, 1) / 255.0
        assert exact.dtype == np.float32 and exact.min() >= 0 and exact.max() <= 1
        assert np.percentile(np.abs(exact - shown), 99) < 0.012, mode


def test_pyramid_halves_by_averaging():
    img = sky(256, 640)
    lv = sm.pyramid(img, smallest=80)
    assert lv[0] is img
    assert [x.shape[1:] for x in lv] == [(256, 640), (128, 320), (64, 160), (32, 80)]
    assert np.allclose(lv[1][:, 10, 20], img[:, 20:22, 40:42].mean((1, 2)), atol=1e-3)


def test_superpixel_of_a_mosaic():
    m = np.zeros((4, 6), np.float32)
    m[0::2, 0::2], m[0::2, 1::2], m[1::2, 0::2], m[1::2, 1::2] = 10, 20, 30, 40      # RGGB
    s = sm.superpixel(m, "RGGB")
    assert s.shape == (3, 2, 3)
    assert np.all(s[0] == 10) and np.all(s[1] == 25) and np.all(s[2] == 40)
    assert sm.cfa_letter("RGGB", 0, 0) == "R" and sm.cfa_letter("RGGB", 1, 1) == "B"


# ---------------------------------------------------------------------------------------------- display
def test_display_at_one_to_one_is_the_data_through_the_table():
    img = sky(700, 900)
    prep = iv.prepare(iv.ViewImage(img))
    d = iv.Display(prep, iv.default_stretch(prep))
    buf, u0, v0 = d.view(1.0, 450.0, 350.0, 400, 300)
    assert (u0, v0) == (0, 0) and buf.shape == (300, 400, 3)
    direct = sm.render(sm.quantize(img[:, 200:500, 250:650], prep.stats.lo, prep.stats.hi), d.lut)
    assert np.array_equal(buf, direct)
    assert d.level_for(1.0) == 0 and d.level_for(0.5) == 1 and d.level_for(0.26) == 2 and d.level_for(0.4) == 1
    small, u0, v0 = d.view(0.25, 450.0, 350.0, 400, 300)         # fit: from the reduced level
    assert small.shape[1] == round(900 * 0.25) + 1 or abs(small.shape[1] - 225) <= 2


def test_mono_and_hwc_inputs():
    mono = sky(channels=1)[0]
    p = iv.prepare(iv.ViewImage(mono))
    assert p.channels == 1 and p.levels[0].shape == (1, 300, 400)
    d = iv.Display(p, iv.default_stretch(p))
    buf, _u, _v = d.view(1.0, 200, 150, 100, 100)
    assert buf.ndim == 2
    hwc = np.ascontiguousarray(sky().transpose(1, 2, 0))
    p2 = iv.prepare(iv.ViewImage(hwc))
    assert p2.channels == 3 and p2.size() == (400, 300)
    u16 = (sky() * 10).clip(0, 65535).astype(np.uint16)
    p3 = iv.prepare(iv.ViewImage(u16))
    assert p3.integer and p3.levels[0].dtype == np.float32


# ---------------------------------------------------------------------------------------------- the loader
def test_loader_works_in_worker_threads_and_caches(qapp, loader):
    img = sky()
    vi = iv.ViewImage(img, "A", key="a")
    got = []
    loader.ready.connect(lambda k, p: got.append((k, p, threading.get_ident())))
    loader.request(vi)
    assert loader.wait(30)
    assert len(got) == 1 and got[0][0] == "a" and got[0][2] == threading.get_ident()   # answered in the GUI thread
    assert loader.load_threads and threading.get_ident() not in loader.load_threads   # made in a worker
    loader.request(vi)
    assert loader.wait(5) and loader.loads == 1 and loader.cached("a") is got[0][1]
    bad = iv.ViewImage(lambda: (_ for _ in ()).throw(OSError("disk gone")), key="bad")
    fails = []
    loader.failed.connect(lambda k, why: fails.append((k, why)))
    loader.request(bad)
    assert loader.wait(10) and fails and "disk gone" in fails[0][1]


def test_loader_budget_drops_the_oldest(qapp):
    ld = iv.Loader(budget=8 * 2 ** 20)
    try:
        for i in range(4):
            ld.request(iv.ViewImage(sky(seed=i), key=i))
            assert ld.wait(30)
        assert ld.cached(3) is not None and ld.cached(0) is None
    finally:
        ld.stop()


# ---------------------------------------------------------------------------------------------- the viewer
def test_view_opens_fits_and_zooms(qapp, loader):
    v = make_view(qapp, loader)
    img = sky(600, 900)
    shown = []
    v.imageShown.connect(shown.append)
    show(v, iv.ViewImage(img, "Stack", key="stack"))
    c = v.canvas
    assert shown == ["stack"] and c.display is not None and c.fitted
    assert c.fit_scale() == pytest.approx(c.scale)
    assert v.zoom_b.text().endswith("%")
    v.one_b.click()
    assert c.zoom_percent() == pytest.approx(100)
    assert not c.whole_visible()
    v.fit_b.click()
    assert c.fitted and c.whole_visible()
    # the wheel zooms around the cursor: the image point under it stays there (once the image fills the view;
    # smaller than the view, it stays centred)
    wheel(c, QPointF(300, 200), 120)
    assert c.scale == pytest.approx(c.fit_scale() * iv.ZOOM_STEP)
    v.set_zoom_percent(200)
    pos = QPointF(300, 200)
    before = c.w_to_img(pos.x(), pos.y())
    wheel(c, pos, 120)
    after = c.w_to_img(pos.x(), pos.y())
    assert c.zoom_percent() == pytest.approx(200 * iv.ZOOM_STEP)
    assert np.allclose(before, after, atol=1e-6)
    wheel(c, pos, -120)
    assert c.zoom_percent() == pytest.approx(200)
    v.set_zoom_percent(400)
    assert c.zoom_percent() == pytest.approx(400)
    c.repaint()
    assert c.frame_ms


def test_pan_by_drag_and_the_minimap(qapp, loader):
    v = make_view(qapp, loader)
    show(v, iv.ViewImage(sky(600, 900), key="p"))
    c = v.canvas
    v.set_zoom_percent(200)
    cx, cy = c.cx, c.cy
    QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(400, 300))
    QTest.mouseMove(c, QPoint(300, 250))
    QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(300, 250))
    assert c.cx == pytest.approx(cx + 100 / c.scale) and c.cy == pytest.approx(cy + 50 / c.scale)
    r = c.minimap_rect()
    assert not c.whole_visible()
    QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, r.topLeft().toPoint() + QPoint(3, 3))
    assert c.cx == pytest.approx(c.width() / 2 / c.scale)    # the top-left corner, as far as the view goes
    assert c.cy == pytest.approx(c.height() / 2 / c.scale)


def test_readout_values_units_and_sky(qapp, loader):
    v = make_view(qapp, loader)
    img = sky()
    wcs = simple_wcs()
    show(v, iv.ViewImage(img, "S", key="s", wcs=wcs, units="adu", white=16383))
    t = v.pixel_text(10.5, 20.5)
    r, g, b = img[:, 20, 10]
    assert t.startswith("x    10  y    20")
    assert f"R {r:.1f}" in t and f"G {g:.1f}" in t and "ADU" in t
    ra, dec = wcs.pixel_to_world_values(10.0, 20.0)
    assert iv.sky_text(float(ra), float(dec)) in t
    v.units_cb.setCurrentIndex(1)
    t2 = v.pixel_text(10.5, 20.5)
    assert f"R {r / 16383:.5f}" in t2 and "ADU" not in t2
    assert v.pixel_text(-1, 5) == "" and v.pixel_text(400, 5) == ""
    # hovering the canvas fills the readout and says where
    got = []
    v.hovered.connect(lambda x, y: got.append((x, y)))
    move(v.canvas, QPointF(200, 200))
    assert got and v.readout.text().startswith("x")
    QApplication.sendEvent(v.canvas, QEvent(QEvent.Leave))
    assert v.readout.text() == ""


def test_sky_text_rounds_carry():
    assert iv.sky_text(40.0, 61.0) == "RA 02h40m00.0s   Dec +61°00′00″"
    assert iv.sky_text(15 * (1 + 59 / 60 + 59.99 / 3600), -0.5) == "RA 02h00m00.0s   Dec −00°30′00″"


def test_a_bayer_mosaic_is_shown_debayered_and_read_raw(qapp, loader):
    rng = np.random.default_rng(3)
    m = (rng.normal(600, 20, (200, 300)) + 512).astype(np.uint16)
    m[0::2, 0::2] += 100                                 # R brighter
    v = make_view(qapp, loader)
    show(v, iv.ViewImage(m, "raw", key="raw", cfa="RGGB"))
    p = v.prepared()
    assert p.factor == 2 and p.channels == 3 and p.size() == (300, 200) and p.integer
    assert v.stretch().neutral                           # a mosaic is in the camera's colours
    t = v.pixel_text(4.2, 6.7)
    assert t.startswith("x     4  y     6") and f"R {m[6, 4]}" in t
    assert f"G {m[6, 5]}" in v.pixel_text(5, 6)
    assert "Bayer RGGB" in v.size_l.text()


def test_stretch_panel_handles_fields_and_reset(qapp, loader):
    v = make_view(qapp, loader)
    img = sky(tint=(1.0, 1.7, 0.6))
    digest = hashlib.sha1(img.tobytes()).hexdigest()
    show(v, iv.ViewImage(img, key="h", camera=True))
    pn, h = v.panel, v.panel.hist
    st0 = v.stretch()
    assert st0.auto and st0.linked and st0.neutral and pn.auto_b.isChecked()
    changes = []
    v.stretchChanged.connect(changes.append)
    # drag the black handle to the right
    r = h.plot_rect()
    x0 = h.px_of(float(st0.black[0]))
    y = int(r.bottom() + 8)
    QTest.mousePress(h, Qt.LeftButton, Qt.NoModifier, QPoint(int(round(x0)), y))
    QTest.mouseMove(h, QPoint(int(round(x0)) + 25, y))
    QTest.mouseRelease(h, Qt.LeftButton, Qt.NoModifier, QPoint(int(round(x0)) + 25, y))
    st1 = v.stretch()
    assert changes and not st1.auto and st1.black[0] > st0.black[0]
    assert not pn.auto_b.isChecked()
    assert pn.fields["black"].value() == pytest.approx(float(st1.black[0]), abs=0.11)
    # the white field
    pn.fields["white"].setValue(5000.0)
    assert v.stretch().white[0] == pytest.approx(5000.0)
    # unlinked: one channel at a time
    pn.linked_c.setChecked(False)
    assert not v.stretch().linked and pn.chan_row.isVisibleTo(pn)
    pn.chan_btns[3].click()                              # B
    pn.fields["white"].setValue(3000.0)
    st2 = v.stretch()
    assert st2.white[2] == pytest.approx(3000.0) and st2.white[0] != pytest.approx(3000.0)
    # the modes
    for i in range(pn.mode_cb.count()):
        pn.mode_cb.setCurrentIndex(i)
        assert v.stretch().mode == pn.mode_cb.itemData(i)
        v.canvas.repaint()
    # Reset: the automatic stretch of this image
    pn.reset_b.click()
    st3 = v.stretch()
    assert st3.auto and st3.mode == "mtf" and st3.linked and st3.neutral
    assert np.allclose(st3.black, st0.black) and np.allclose(st3.white, st0.white)
    # neutral sky off
    pn.neutral_c.setChecked(False)
    assert not v.stretch().neutral and np.all(v.stretch().gain == 1)
    # the toolbar's Auto-stretch off: the data shown linear, min to max
    v.auto_b.setChecked(False)
    st4 = v.stretch()
    assert st4.mode == "linear" and st4.black[0] == pytest.approx(float(img.min()))
    v.auto_b.setChecked(True)
    assert v.stretch().auto
    # the clipped fractions and the statistics are written
    assert "%" in pn.clip_l.text() and pn.cells[(1, 1)].text()
    assert hashlib.sha1(img.tobytes()).hexdigest() == digest          # display only


def test_histogram_hides_mid_for_linear_and_axis_is_monotonic(qapp, loader):
    v = make_view(qapp, loader)
    show(v, iv.ViewImage(sky(), key="ax"))
    h = v.panel.hist
    ys = np.linspace(h.axis()[0], h.axis()[1], 50)
    us = h.u_of(ys)
    assert np.all(np.diff(us) > 0) and us[0] == pytest.approx(0) and us[-1] == pytest.approx(1)
    assert np.allclose(h.y_of(us), ys, rtol=1e-6, atol=1e-6)
    v.panel.mode_cb.setCurrentIndex(v.panel.mode_cb.findData("linear"))
    assert not v.panel.fields["mid"].isEnabled()
    assert iv.nice_ticks(100, 20000)[:3] == [100.0, 200.0, 500.0]


# ---------------------------------------------------------------------------------------------- compare
def test_compare_split_side_and_same_stretch(qapp, loader):
    v = make_view(qapp, loader)
    a, b = sky(seed=1), sky(seed=2, level=400)
    show(v, iv.ViewImage(a, "After", key="after"))
    assert not v.cmp_box.isVisible()
    picked = []
    v.compareChanged.connect(picked.append)
    v.set_compare_targets([iv.ViewImage(b, "Before", key="before"), iv.ViewImage(sky(seed=3), "Other", key="o")])
    assert v.cmp_box.isVisible() and not v.cmp_opts.isVisible() and v.cmp_cb.count() == 3
    v.cmp_cb.setCurrentIndex(1)
    assert v.wait(30)
    assert picked[-1] == "before" and v.cmp_opts.isVisible()
    c = v.canvas
    assert c.compare is not None and c.split_names == ("Before", "After")
    c.repaint()
    # the divider is dragged
    xs = int(c.width() * c.split)
    QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(xs, 100))
    QTest.mouseMove(c, QPoint(xs - 100, 100))
    QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(xs - 100, 100))
    assert c.split == pytest.approx((xs - 100) / c.width(), abs=0.01)
    # the readout on the left of the divider reads the image compared
    v._hover(c, 10, 10, 1)
    assert v.readout.text().startswith("Before")
    # each its own automatic stretch, or the same
    assert not np.allclose(v._cmp.stretch.black, v.stretch().black)
    v.same_c.setChecked(True)
    assert np.allclose(v._cmp.stretch.black, v.stretch().black)
    v.panel.fields["white"].setValue(4000.0)
    assert v._cmp.stretch.white[0] == pytest.approx(4000.0)
    # side by side: two canvases, zoom and pan linked
    v.side_b.click()
    QApplication.processEvents()
    assert v.canvas2.isVisible() and v.canvas.compare is None and v.canvas2.display is v._cmp
    v.set_zoom_percent(200)
    QApplication.processEvents()
    assert v.canvas2.scale == pytest.approx(v.canvas.scale) and v.canvas2.cx == pytest.approx(v.canvas.cx)
    v.canvas2.pan_by(40, 0)
    assert v.canvas.cx == pytest.approx(v.canvas2.cx)
    v.cmp_cb.setCurrentIndex(0)
    assert picked[-1] is None and not v.canvas2.isVisible()


def test_split_is_one_continuous_field(qapp, loader):
    """The divider shows the left of the field from the compared image and the right from this one: the same image
    on both sides, with the same stretch, draws exactly the picture without a compare (but the divider itself)."""
    v = make_view(qapp, loader)
    img = sky(700, 1000)
    show(v, iv.ViewImage(img, "A", key="cont-a"))
    v.set_zoom_percent(150)
    v.canvas.pan_by(-120, -60)
    v.canvas.minimap_on = False
    plain = v.canvas.grab().toImage()
    v.set_compare_targets([iv.ViewImage(img.copy(), "B", key="cont-b")], current=0)
    v.set_same_stretch(True)
    assert v.wait(30)
    v.canvas.split_names = ("", "")
    split = v.canvas.grab().toImage()
    xs = int(v.canvas.width() * v.canvas.split)

    def arr(q):
        q = q.convertToFormat(q.Format.Format_RGB888)
        a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine())[:, :q.width() * 3]
        return a.copy()                                       # the QImage goes when this returns
    a, b = arr(plain), arr(split)
    keep = np.ones(a.shape[1] // 3, bool)
    keep[max(0, xs - 14):xs + 14] = False                      # the divider and its handle
    assert np.array_equal(a.reshape(a.shape[0], -1, 3)[:, keep], b.reshape(b.shape[0], -1, 3)[:, keep])


def _grab(widget) -> np.ndarray:
    q = widget.grab().toImage()
    q = q.convertToFormat(q.Format.Format_RGB888)
    return np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine())[:, :q.width() * 3] \
        .reshape(q.height(), q.width(), 3).copy()


def _one_star(h, w, x, y, level=100.0, peak=20000.0, sigma=2.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = level + peak * np.exp(-((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2) / (2 * sigma ** 2))
    return np.repeat(img[None], 3, axis=0).astype(np.float32)


def _brightest(rgb: np.ndarray, x0: int, x1: int) -> tuple[float, float]:
    """(x, y) of the star between the columns x0 and x1 of a grab: the centroid of its pixels above half its peak (the
    brightest pixel alone is any pixel of a plateau)."""
    lum = rgb[:, x0:x1].astype(np.float64).sum(2)
    lum -= np.median(lum)
    lum[lum < 0.5 * lum.max()] = 0.0
    yy, xx = np.mgrid[0:lum.shape[0], 0:lum.shape[1]]
    return float((xx * lum).sum() / lum.sum()) + x0, float((yy * lum).sum() / lum.sum())


def test_compare_puts_both_images_on_one_grid(qapp, loader):
    """A compared image cut and reduced from another grid (a final image: a crop at half size of the stack) is drawn
    on the grid of the image shown, by their placements: the same star at the same place on the screen, in split and
    side by side, at any zoom; the line above the images says so; the readout reads the compared image's own pixel."""
    W, H = 800, 600
    sx, sy = 437.0, 263.0                                   # the star, on the big grid
    big = _one_star(H, W, sx, sy)
    ox, oy, f = 120, 80, 2                                  # the small one: the crop (120, 80) at half size
    small = _one_star(260, 300, (sx - ox) / f, (sy - oy) / f, sigma=1.0)
    frame = ("reference frame", "k")
    v = make_view(qapp, loader)
    v.canvas.minimap_on = v.canvas2.minimap_on = False
    show(v, iv.ViewImage(small, "Final", key="al-small", place=iv.Placement.scaled(frame, f, ox, oy)))
    v.set_compare_targets([iv.ViewImage(big, "Big", key="al-big", place=iv.Placement(frame))], current=0)
    assert v.wait(30)
    QApplication.processEvents()
    al = v.compare_alignment()
    assert al is not None and al.how == "grid" and al.scale == pytest.approx(2.0)
    assert al.apply(0.0, 0.0) == pytest.approx((ox, oy))
    note = v.alignment_note()
    assert v.align_bar.isVisible() and "geometry of the results" in note and "“Big”" in note and "× 0.50" in note
    # split: the divider over the star, half of the field from each image - the star's centre is in the same place
    c = v.canvas
    c.split_names = ("", "")
    for zoom in (None, 300.0):
        if zoom:
            v.set_zoom_percent(zoom)
            c.zoom_to(c.scale, (sx - ox) / f, (sy - oy) / f)
        star = c.img_to_w((sx - ox) / f, (sy - oy) / f)
        c.split = 1.0                                         # the whole canvas from the compared image
        QApplication.processEvents()
        left = _brightest(_grab(c), 0, c.width() - 30)
        c.split = 0.0                                         # the whole canvas from the image shown
        QApplication.processEvents()
        right = _brightest(_grab(c), 30, c.width())
        assert abs(left[0] - right[0]) <= 1 and abs(left[1] - right[1]) <= 1, (zoom, left, right)
        assert abs(left[0] - star.x()) <= 1 and abs(left[1] - star.y()) <= 1, (zoom, left, star)
    c.split = 0.5
    # the readout of the compared image reads its own pixel under the cursor
    v._hover(c, (sx - ox) / f, (sy - oy) / f, 1)
    assert v.readout.text().startswith("Big") and f"x {int(sx):>5}  y {int(sy):>5}" in v.readout.text()
    # side by side: the same point of the field in both canvases
    v.set_compare_mode("side")
    QApplication.processEvents()
    v.fit()
    QApplication.processEvents()
    assert v.canvas2.display is v._cmp and v.canvas2.display.size() == (300, 260)
    v.canvas.title = v.canvas2.title = ""                    # the name tags are bright too
    a, b = _grab(v.canvas), _grab(v.canvas2)
    pa, pb = _brightest(a, 0, a.shape[1]), _brightest(b, 0, b.shape[1])
    star = v.canvas.img_to_w((sx - ox) / f, (sy - oy) / f)
    assert abs(pa[0] - pb[0]) <= 1 and abs(pa[1] - pb[1]) <= 1 and abs(pa[0] - star.x()) <= 1, (pa, pb, star)
    v.canvas2.zoom_by(3.0)                                    # zoom and pan stay linked on the one grid
    QApplication.processEvents()
    assert v.canvas.scale == pytest.approx(v.canvas2.scale) and v.canvas.cx == pytest.approx(v.canvas2.cx)
    # no compare: no line
    v.set_compare(None)
    assert not v.align_bar.isVisible() and v.alignment_note() == ""


def test_compare_by_the_astrometric_solutions_and_by_width(qapp, loader):
    """Two images of the same sky with no common grid but each its solution (another program's result, turned and
    shifted) are put on each other by the solutions; with neither, the compared one is matched by its width, and the
    line says it is not aligned. The same grid needs nothing (no line)."""
    from astropy.wcs import WCS
    W, H = 500, 400
    a_wcs = simple_wcs(W, H)
    ra, dec = a_wcs.pixel_to_world_values(311.0 - 0.5, 147.0 - 0.5)
    b_wcs = WCS(naxis=2)                                      # the other one: turned 12°, shifted, 1.25x the pixels
    b_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    b_wcs.wcs.crval = [40.01, 61.005]
    b_wcs.wcs.crpix = [260.0, 190.0]
    s, t = 2.0 / 3600 / 1.25, np.radians(12.0)
    b_wcs.wcs.cd = [[-s * np.cos(t), s * np.sin(t)], [s * np.sin(t), s * np.cos(t)]]
    bx, by = b_wcs.world_to_pixel_values(ra, dec)
    A = _one_star(H, W, 311.0, 147.0)
    B = _one_star(H + 100, W + 120, float(bx) + 0.5, float(by) + 0.5, sigma=2.5)
    v = make_view(qapp, loader)
    v.canvas.minimap_on = False
    show(v, iv.ViewImage(A, "Ours", key="sky-a", wcs=a_wcs))
    v.set_compare_targets([iv.ViewImage(B, "Theirs", key="sky-b", wcs=b_wcs),
                           iv.ViewImage(A.copy(), "Same", key="sky-same", wcs=a_wcs),
                           iv.ViewImage(B.copy(), "Bare", key="sky-bare")], current=0)
    assert v.wait(30)
    QApplication.processEvents()
    al = v.compare_alignment()
    assert al is not None and al.how == "sky" and al.residual < 0.05
    assert abs(al.angle) == pytest.approx(12.0, abs=0.1) and al.scale == pytest.approx(1.25, rel=1e-3)
    assert "astrometric solutions" in v.alignment_note() and "turned" in v.alignment_note()
    c = v.canvas
    c.split_names = ("", "")
    c.split = 1.0
    QApplication.processEvents()
    left = _brightest(_grab(c), 0, c.width() - 30)
    star = c.img_to_w(311.0, 147.0)
    assert abs(left[0] - star.x()) <= 1 and abs(left[1] - star.y()) <= 1, (left, star)
    v.set_compare(1)                                          # the same grid: nothing to do, nothing said
    assert v.wait(30)
    assert v.compare_alignment() is None and not v.align_bar.isVisible()
    v.set_compare(2)                                          # no geometry: the width, said
    assert v.wait(30)
    al = v.compare_alignment()
    assert al.how == "width" and v.alignment_note().startswith("Not aligned")
    assert al.scale == pytest.approx((W + 120) / W)


class _Distorted:
    """A solution with a distortion of its own (as two programs' solutions of one field differ: a few pixels in the
    corners): its pixel = the plain one + k (r² - r0²) along the radius from the centre."""

    def __init__(self, wcs, cx, cy, k):
        self.wcs, self.cx, self.cy, self.k = wcs, cx, cy, k

    def world_to_pixel_values(self, ra, dec):
        x, y = self.wcs.world_to_pixel_values(ra, dec)
        dx, dy = np.asarray(x) - self.cx, np.asarray(y) - self.cy
        f = self.k * (dx * dx + dy * dy)
        return x + f * dx, y + f * dy

    def pixel_to_world_values(self, x, y):                     # (not needed by the compare: one way is enough)
        return self.wcs.pixel_to_world_values(x, y)


def test_compare_by_the_solutions_follows_their_distortions(qapp, loader):
    """When the two solutions differ by more than an affine map (their distortions), the compared image is drawn
    through them point by point: a star in a corner, 4 px off the affine map, lands where it is in the image shown."""
    W, H = 600, 400
    a_wcs = simple_wcs(W, H)
    b_plain = simple_wcs(W, H)
    b_wcs = _Distorted(b_plain, W / 2, H / 2, 6e-8)               # 4 px at the corners (r = 360 px)
    stars_a = [(60.0, 50.0), (300.0, 200.0), (540.0, 350.0)]
    A = sum(_one_star(H, W, x, y) for x, y in stars_a) / 3
    B = np.zeros_like(A)
    for x, y in stars_a:
        ra, dec = a_wcs.pixel_to_world_values(x - 0.5, y - 0.5)
        bx, by = b_wcs.world_to_pixel_values(ra, dec)
        B += _one_star(H, W, float(bx) + 0.5, float(by) + 0.5) / 3
    v = make_view(qapp, loader)
    v.canvas.minimap_on = False
    show(v, iv.ViewImage(A, "Ours", key="dist-a", wcs=a_wcs))
    v.set_compare_targets([iv.ViewImage(B, "Theirs", key="dist-b", wcs=b_wcs)], current=0)
    assert v.wait(30)
    al = v.compare_alignment()
    assert al.how == "sky" and al.field is not None and 1.0 < al.residual < 6.0
    assert "distortions followed" in v.alignment_note()
    for x, y in stars_a:                                       # the map, read back, is the solutions' own
        ra, dec = a_wcs.pixel_to_world_values(x - 0.5, y - 0.5)
        bx, by = b_wcs.world_to_pixel_values(ra, dec)
        assert al.apply(x, y) == pytest.approx((float(bx) + 0.5, float(by) + 0.5), abs=0.3)
    c = v.canvas
    c.split_names = ("", "")
    v.set_zoom_percent(400)
    for x, y in (stars_a[0], stars_a[2]):                      # the corners, at 4x
        c.zoom_to(c.scale, x, y)
        star = c.img_to_w(x, y)
        c.split = 1.0
        QApplication.processEvents()
        got = _brightest(_grab(c), 0, c.width() - 30)
        assert abs(got[0] - star.x()) <= 1.5 and abs(got[1] - star.y()) <= 1.5, ((x, y), got, star)
    c.split = 0.5


def test_placements_compose():
    stack = iv.Placement.scaled("f", 1 / 1.5)                  # a drizzle at 1.5 on its reference frame
    final = iv.Placement.scaled(None, 2.0, 96, 64).then(stack)  # its crop at (96, 64), half size
    assert final.frame == "f"
    assert final.matrix() @ np.array([10.0, 20.0, 1.0]) == pytest.approx([(96 + 20) / 1.5, (64 + 40) / 1.5, 1.0])


def test_sync_group_shares_view_and_optionally_the_stretch(qapp, loader):
    a = make_view(qapp, loader)
    b = make_view(qapp, loader)
    show(a, iv.ViewImage(sky(600, 800, seed=1), key="sa"))
    show(b, iv.ViewImage(sky(300, 400, seed=2), key="sb"))              # half the size: the same field
    g = iv.SyncGroup([a, b])
    a.set_zoom_percent(100)
    a.canvas.pan_by(50, 30)
    assert b.canvas.scale * 400 == pytest.approx(a.canvas.scale * 800)
    assert b.canvas.cx / 400 == pytest.approx(a.canvas.cx / 800)
    sb = b.stretch()
    a.panel.fields["white"].setValue(3333.0)
    assert np.allclose(b.stretch().white, sb.white)                       # not shared yet
    g.set_stretch_shared(True)
    a.panel.fields["white"].setValue(2222.0)
    assert b.stretch().white[0] == pytest.approx(2222.0)
    a.fit()
    assert b.canvas.fitted


# ---------------------------------------------------------------------------------------------- sequences
def frames_for_test(n=6):
    out = []
    for i in range(n):
        info = {"time": f"2026-08-17T00:{10 + i:02d}:00", "exposure": 15.0, "iso": 1600, "temperature": 12.5,
                "fwhm": 3.0 + i / 10, "roundness": 0.9, "transparency": 1.0 - i / 20, "excluded": i == 2,
                "extra": [("Stars", str(1000 + i))]}
        out.append(iv.ViewImage(lambda s=i: sky(seed=s, level=200 + 20 * s), f"L_{i:03d}.fits", key=("f", i),
                                info=info, camera=True))
    return out


def test_sequence_keys_box_strip_band_and_exclude(qapp, loader):
    v = make_view(qapp, loader)
    frames = frames_for_test()
    seen, excl = [], []
    v.frameChanged.connect(seen.append)
    v.excludeToggled.connect(lambda i, e: excl.append((i, e)))
    v.set_sequence(frames, 0)
    assert v.wait(30)
    assert v.seq.isVisible() and v.index() == 0 and v.total_l.text() == "/ 6"
    assert v.band.name_l.full_text() == "L_000.fits" and v.band.cells["exposure"][1].text() == "15 s"
    assert v.band.cells["fwhm"][1].text() == "3.00 px" and v.band.cells["transparency"][1].text() == "100 %"
    assert v.band.cells["temperature"][1].text() == "12.5 °C" and v.band.state.text() == "Kept"
    first = v.stretch()
    assert first.auto and not v.lock_c.isChecked()                       # by default each frame its own stretch
    loader.wait(30)
    assert loader.cached(("f", 1)) is not None                          # the next one, ahead of time
    v.activateWindow()
    v.canvas.setFocus()
    QTest.keyClick(v.canvas, Qt.Key_Right)
    assert v.wait(30) and v.index() == 1
    assert v.stretch().auto and not np.allclose(v.stretch().black, first.black)     # its own automatic stretch
    QTest.keyClick(v.canvas, Qt.Key_Left)
    assert v.wait(30) and v.index() == 0
    v.lock_c.setChecked(True)                                            # locked: the stretch of the frame shown
    QTest.keyClick(v.canvas, Qt.Key_Right)
    assert v.wait(30) and v.index() == 1
    assert np.allclose(v.stretch().black, first.black)                  # the same stretch for all frames
    QTest.keyClick(v.canvas, Qt.Key_End)
    assert v.index() == 5
    QTest.keyClick(v.canvas, Qt.Key_Home)
    assert v.index() == 0
    v.num.setValue(3)
    assert v.wait(30) and v.index() == 2
    assert v.band.state.text() == "Excluded" and v.band.toggle.text() == "Include" and 2 in v.strip.excluded
    QTest.keyClick(v.canvas, Qt.Key_X)
    assert excl == [(2, False)] and v.band.toggle.text() == "Exclude" and 2 not in v.strip.excluded
    v.band.toggle.click()
    assert excl[-1] == (2, True)
    QTest.mouseClick(v.strip, Qt.LeftButton, Qt.NoModifier,
                     QPoint(int(4 * (v.strip.CELL_W + v.strip.GAP) + 20 - v.strip.offset), 20))
    assert v.index() == 4
    assert seen[:2] == [0, 1] and seen[-1] == 4
    # unlocked: each frame its own automatic stretch
    v.lock_c.setChecked(False)
    v.set_index(5)
    assert v.wait(30)
    assert not np.allclose(v.stretch().black, first.black)
    loader.wait(30)
    assert v.strip.thumbs                                                # made from the frames loaded


def dark_frame(seed=1, level=512.0, h=300, w=400) -> np.ndarray:
    """A colour dark (superpixels of a mosaic): its level, read noise, an amp glow in one corner, hot pixels."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    glow = 60.0 * np.exp(-np.hypot(xx, yy) / 60.0)
    img = level + glow + rng.normal(0, 4, (3, h, w)).astype(np.float32)
    for _i in range(40):
        img[:, rng.integers(0, h), rng.integers(0, w)] = rng.uniform(2000, 16000)
    return img.astype(np.float32)


def flat_frame(h=300, w=400, peak=(18000.0, 30000.0, 22000.0)) -> np.ndarray:
    """A colour flat: vignetting to 70 % in the corners, a dust shadow, the light source's colour, a hot pixel."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    shape = 1.0 - 0.15 * r2
    shape *= 1.0 - 0.08 * np.exp(-((xx - 120) ** 2 + (yy - 90) ** 2) / (2 * 12.0 ** 2))
    rng = np.random.default_rng(3)
    img = np.stack([p * shape for p in peak]) + rng.normal(0, 40, (3, h, w))
    img[:, 200, 300] = 65535.0
    return img.astype(np.float32)


def model_surface(h=300, w=400) -> np.ndarray:
    """A sky model as the Seestar's M31 had it: smooth, nearly constant, blue above the other two (display RGB)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 6.0 * (xx / w) + 4.0 * (yy / h)
    return np.stack([260.0 + g, 268.0 + g, 408.0 + g]).astype(np.float32)


def test_each_kind_of_frame_has_its_automatic_stretch():
    # a dark, a bias: each channel centred on its own level (STF unlinked, the median at mid grey), no neutral sky
    for kind in ("darks", "bias", "flat_darks"):
        img = dark_frame()
        img[2] -= 2.0                                                            # a channel 2 ADU lower: no tint
        prep = iv.prepare(iv.ViewImage(img, "dark", camera=True, kind=kind))
        st = iv.default_stretch(prep)
        s = prep.stats
        assert st.auto and st.mode == "mtf" and not st.linked and not st.neutral
        assert np.allclose(st.black, s.median - sm.SHADOWS * s.sigma, rtol=1e-6)
        cv = sm.curves(st, s)
        for c in range(3):
            q = sm.quantize(np.float32(s.median[c]), s.lo, s.hi)
            assert cv[c][q] == pytest.approx(sm.LEVEL_TARGET, abs=0.03)          # each median at mid grey
        hot = sm.quantize(np.float32(15000), s.lo, s.hi)
        assert cv[1][hot] == 1.0                                                 # the hot pixels white
        glow = sm.quantize(np.float32(s.median[1] + 40), s.lo, s.hi)
        assert 0.75 < cv[1][glow] < 0.99                                         # the glow shaded, not clipped
    # a flat: linear, each channel from the lowest to the highest value of its smoothed field (a hot pixel ignored)
    img = flat_frame()
    prep = iv.prepare(iv.ViewImage(img, "flat", camera=True, kind="flats"))
    st = iv.default_stretch(prep)
    assert st.auto and st.mode == "linear" and not st.linked and not st.neutral
    for c, peak in enumerate((18000.0, 30000.0, 22000.0)):
        assert st.white[c] == pytest.approx(peak, rel=0.02)                     # the centre, not the hot pixel
        assert peak * 0.68 < st.black[c] < peak * 0.78                           # the corners
    out = sm.render(sm.quantize(img, prep.stats.lo, prep.stats.hi), sm.luts(st, prep.stats))
    assert out[150, 200].min() > 230 and out[5, 5].max() < 40                  # its shape fills the range
    # a model: linear, linked, from zero: its colour is its own, not blown up into a saturated one
    img = model_surface()
    prep = iv.prepare(iv.ViewImage(img, "Sky model", camera=False, kind="model"))
    st = iv.default_stretch(prep)
    assert st.auto and st.mode == "linear" and st.linked and np.all(st.black == 0)
    assert st.white[0] == pytest.approx(float(img.max()), rel=0.01)
    out = sm.render(sm.quantize(img, prep.stats.lo, prep.stats.hi), sm.luts(st, prep.stats)).reshape(-1, 3)
    shown = out.mean(0)
    assert shown[2] / shown[0] == pytest.approx(img[2].mean() / img[0].mean(), rel=0.03)     # ratios kept
    assert shown.min() > 140                                                     # a pale blue, not pure blue
    light = sm.render(sm.quantize(img, prep.stats.lo, prep.stats.hi),
                      sm.luts(sm.auto_stretch(prep.stats, "mtf", True), prep.stats)).reshape(-1, 3).mean(0)
    assert light[0] < 10 and light[1] < 10 and light[2] > 240                  # what the light rule made of it
    unlinked = sm.auto_for(prep.stats, "model", linked=False)
    assert np.all(unlinked.black > 250)                                          # unlinked: each channel's shape
    # a light (or no kind): the standard auto-stretch
    prep = iv.prepare(iv.ViewImage(sky(), "light", kind="lights"))
    assert iv.default_stretch(prep).black[0] == pytest.approx(
        iv.default_stretch(iv.prepare(iv.ViewImage(sky(), "light"))).black[0])
    assert sm.rule_of("lights") == sm.rule_of("") == sm.rule_of("anything") == "light"


def test_a_model_shows_from_zero_in_the_panel(qapp, loader):
    v = make_view(qapp, loader)
    show(v, iv.ViewImage(model_surface(), "Sky model", camera=False, kind="model"))
    assert v.panel.fields["black"].value() == 0.0 and v.auto_b.isChecked()
    assert v.panel.fields["white"].value() == pytest.approx(float(model_surface().max()), rel=0.01)
    assert "model's own" in v.panel.auto_b.toolTip()
    v.auto_b.setChecked(False)                                   # Auto-stretch off, then on: the model's own again
    v.auto_b.setChecked(True)
    assert v.stretch().mode == "linear" and v.stretch().linked and np.all(v.stretch().black == 0)


def test_the_lock_keeps_one_stretch_per_kind(qapp, loader):
    """"Same stretch for all frames" within one kind only: a dark's stretch never shows a flat; unlocked, each frame
    its own; the stretch panel's Auto says what the kind's automatic stretch does."""
    v = make_view(qapp, loader)
    frames = [iv.ViewImage(lambda s=i: dark_frame(seed=s, level=500 + 40 * s), f"D{i}", key=("d", i), camera=True,
                           kind="darks") for i in range(2)]
    frames += [iv.ViewImage(lambda s=i: flat_frame(peak=(18000.0 + 3000 * s, 30000.0, 22000.0)), f"F{i}",
                            key=("fl", i), camera=True, kind="flats") for i in range(2)]
    v.set_sequence(frames, 0)
    assert v.wait(30)
    d0 = v.stretch()
    assert d0.mode == "mtf" and "own level" in v.panel.auto_b.toolTip()
    v.set_index(1)
    assert v.wait(30)
    assert not np.allclose(v.stretch().black, d0.black)                      # unlocked: its own
    v.set_index(0)
    assert v.wait(30)
    v.set_lock_stretch(True)
    v.set_index(1)
    assert v.wait(30)
    assert np.allclose(v.stretch().black, d0.black)                          # locked: the darks' stretch
    v.set_index(2)
    assert v.wait(30)
    f0 = v.stretch()
    assert f0.mode == "linear" and f0.black[1] > 15000                       # a flat: never the dark's stretch
    assert "smoothed flat" in v.panel.auto_b.toolTip()
    v.set_index(3)
    assert v.wait(30)
    assert np.allclose(v.stretch().black, f0.black)                          # the flats' stretch
    v.set_index(1)
    assert v.wait(30)
    assert np.allclose(v.stretch().black, d0.black)                          # back to the darks': theirs
    v.set_lock_stretch(False)                                                # unlocked: the frame's own, at once
    assert not np.allclose(v.stretch().black, d0.black)
    v.set_index(3)
    assert v.wait(30)
    assert v.stretch().white[0] == pytest.approx(21000.0, rel=0.02)         # the second flat's own range


def test_blink_plays_the_loaded_frames(qapp, loader):
    v = make_view(qapp, loader)
    v.set_sequence(frames_for_test(3), 0)
    assert v.wait(30)
    loader.wait(30)
    v.blink_b.setChecked(True)
    end = time.monotonic() + 5
    while time.monotonic() < end and v.index() == 0:
        QApplication.processEvents()
        time.sleep(0.02)
    v.blink_b.setChecked(False)
    assert v.index() != 0


def test_thumbnails_from_the_caller(qapp, loader):
    v = make_view(qapp, loader)
    th = np.full((20, 30, 3), 128, np.uint8)
    frames = [iv.ViewImage(sky(seed=i), f"f{i}", key=("t", i), thumb=(th if i == 0 else (lambda: th)))
              for i in range(3)]
    v.set_sequence(frames, 0)
    assert v.wait(30)
    loader.wait(30)
    assert set(v.strip.thumbs) == {0, 1, 2}
    q = iv.quick_thumbnail(iv.ViewImage(sky(600, 900)), side=120)
    assert q.shape == (80, 120, 3) and q.dtype == np.uint8


# ---------------------------------------------------------------------------------------------- actions
def test_export_every_format(qapp, loader, tmp_path):
    from astropy.io import fits
    from PIL import Image
    v = make_view(qapp, loader)
    img = sky()
    wcs = simple_wcs()
    show(v, iv.ViewImage(img, "Stack one", key="ex", wcs=wcs, header={"HISTORY": ["made by the test"], "EXPTIME": 15}))
    p = v.export(tmp_path / "a.fits")
    with fits.open(p) as hl:
        h, d = hl[0].header, hl[0].data
    assert d.dtype.kind == "f" and d.shape == (3, 300, 400) and np.array_equal(d, img)
    assert h["CTYPE1"].startswith("RA---TAN") and h["OBJECT"] == "Stack one" and h["EXPTIME"] == 15
    assert any("made by the test" in str(x) for x in h["HISTORY"])
    # a crop: the solution moves with it
    p = v.export(tmp_path / "crop.fits", area=(100, 50, 300, 250))
    with fits.open(p) as hl:
        h, d = hl[0].header, hl[0].data
    assert d.shape == (3, 200, 200) and np.array_equal(d, img[:, 50:250, 100:300])
    assert h["CRPIX1"] == pytest.approx(wcs.wcs.crpix[0] - 100) and h["XORIGIN"] == 100
    from astropy.wcs import WCS
    assert np.allclose(WCS(h, naxis=2).pixel_to_world_values(5, 7), wcs.pixel_to_world_values(105, 57))
    # TIFF 16-bit: the view (stretched) and the data (ADU as they are)
    import tifffile
    p = v.export(tmp_path / "v.tif", content="view")
    t = tifffile.imread(p)
    exact = sm.apply(img, v.stretch(), v.prepared().stats).transpose(1, 2, 0)
    assert t.dtype == np.uint16 and t.shape == (300, 400, 3)
    assert np.abs(t / 65535.0 - exact).max() < 1e-4 and 0.1 < np.median(exact[..., 1]) < 0.9
    p = v.export(tmp_path / "d.tif", content="data")
    t = tifffile.imread(p)
    assert np.abs(t.astype(float) - np.clip(img.transpose(1, 2, 0), 0, 65535)).max() <= 0.51
    p = v.export(tmp_path / "v.png", fmt="png16", content="view")
    import cv2
    a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    assert a.dtype == np.uint16 and a.shape == (300, 400, 3)
    p = v.export(tmp_path / "v8.png", fmt="png8", content="view")
    assert np.asarray(Image.open(p)).dtype == np.uint8
    p = v.export(tmp_path / "v.jpg", content="view")
    with Image.open(p) as im:
        assert im.format == "JPEG" and im.size == (400, 300)
    with pytest.raises(ValueError):
        v.export(tmp_path / "x.fits", area=(10, 10, 10, 20))


def test_export_dialog_writes_in_a_worker(qapp, loader, tmp_path):
    v = make_view(qapp, loader)
    show(v, iv.ViewImage(sky(), "Dialog test", key="dlg", wcs=simple_wcs()))
    toasts = []
    v.toasted.connect(toasts.append)
    dlg = v.export_dialog()
    assert dlg.c_data.isChecked() and not dlg.c_view.isEnabled() and dlg.c_wcs.isChecked()   # FITS: the data
    dlg.fmt.setCurrentIndex(dlg.fmt.findData("tiff16"))
    assert dlg.c_view.isEnabled() and not dlg.c_wcs.isEnabled() and dlg.dest.text().endswith(".tif")
    dlg.c_view.setChecked(True)
    dlg.dest.setText((tmp_path / "out" / "x.tif").as_posix())
    dlg.go.click()
    assert dlg.wait(30)
    assert dlg.result_path is not None and dlg.result_path.is_file() and dlg.go.text() == "Close"
    assert toasts and "x.tif" in toasts[-1]
    dlg.go.click()
    assert not dlg.isVisible()
    # the same image (cached) shown under another name: the name of now is the one written
    img = v.prepared().data
    show(v, iv.ViewImage(img, "Renamed", key="dlg"))
    dlg = v.export_dialog()
    assert dlg.vi.name == "Renamed" and dlg.dest.text().endswith("Renamed.fits")
    dlg.reject()
    from astropy.io import fits
    p = v.export(tmp_path / "r.fits")
    assert fits.getheader(p)["OBJECT"] == "Renamed"
    # the night test pass: a French title (accents) made the FITS fail and its file name "Aper_u_gradient_retir";
    # the folder proposed was Pictures, not the one written to last
    show(v, iv.ViewImage(img, "Aperçu : gradient retiré", key="dlg"))
    dlg = v.export_dialog()
    assert dlg.dest.text() == (tmp_path / "out" / "Apercu_gradient_retire.fits").as_posix()
    dlg.go.click()
    assert dlg.wait(30) and not dlg.error
    assert fits.getheader(dlg.result_path)["OBJECT"] == "Apercu : gradient retire"
    dlg.go.click()
    v.export_dir = ""
    v.default_dir = lambda: (tmp_path / "project").as_posix()
    dlg = v.export_dialog()
    assert dlg.dest.text().startswith((tmp_path / "project").as_posix())
    dlg.reject()


def test_send_to_post_and_copy_view(qapp, loader):
    v = make_view(qapp, loader, post=True)
    assert not v.post_b.isEnabled()
    show(v, iv.ViewImage(sky(), "S", key=("stack", "k1")))
    sent = []
    v.sendToPost.connect(sent.append)
    v.post_b.trigger()
    assert sent == [("stack", "k1")]
    QGuiApplication.clipboard().clear()
    v.copy_b.trigger()
    im = QGuiApplication.clipboard().image()
    assert not im.isNull() and im.width() == v.area.width()


def test_overlays_of_the_old_viewer_still_draw(qapp, loader):
    v = make_view(qapp, loader)
    show(v, iv.ViewImage(sky(), key="ov"))
    seen = []

    def overlay(p, c):                                   # what steps.py's overlays use
        w, _h = c.source.size()
        seen.append((w, c.img_to_w(0, 0)))
        c._tag(p, QPointF(10, 10), "tag")
    v.add_overlay(overlay)
    v.canvas.repaint()
    assert seen and seen[-1][0] == 400


# ---------------------------------------------------------------------------------------------- what Inspect asks
def test_a_source_that_knows_why_it_is_missing_says_it_alone(qapp, loader):
    def gone():
        raise iv.Unavailable("To recompute: Stacking")

    v = make_view(qapp, loader)
    v.set_image(iv.ViewImage(gone, "Rejection", key="gone"))
    assert v.wait(30)
    assert v.canvas.message == "To recompute: Stacking"
    v.set_image(iv.ViewImage(lambda: 1 / 0, "Broken", key="broken"))      # any other failure: in plain words, the
    assert v.wait(30) and v.canvas.message == ("This image cannot be shown: an error of the program (the details "
                                               "are in the Log)")         # Python error is for the Log only (9.9)
    # a compared image that cannot be read is said too, and not waited for
    show(v, iv.ViewImage(sky(), "S", key="ok"))
    toasts = []
    v.toasted.connect(toasts.append)
    v.set_compare_targets([iv.ViewImage(gone, "Gone", key="gone2")], current=0)
    assert v.wait(30) and v.canvas2.message.startswith("To recompute") and toasts


def test_a_picture_for_the_eye_is_shown_as_it_is_and_read_as_levels(qapp, loader):
    rgb = (np.arange(300 * 400 * 3) % 256).astype(np.uint8).reshape(300, 400, 3)
    v = make_view(qapp, loader)
    show(v, iv.ViewImage(rgb, "Final", key="final", stretched=True, white=255.0))
    st = v.stretch()
    assert st.mode == "linear" and float(st.black[0]) == 0.0 and float(st.white[0]) == 255.0
    r, g, b = (int(c) for c in rgb[7, 9])
    assert v.pixel_text(9.5, 7.5) == f"x     9  y     7    R {r}   G {g}   B {b}"


def test_wcs_from_a_callable_and_a_readout_only_solution(qapp, loader):
    called = []

    def load_wcs():
        called.append(1)
        return simple_wcs()

    v = make_view(qapp, loader)
    show(v, iv.ViewImage(sky(), "S", key="cw", wcs=load_wcs))
    assert called and v.prepared().wcs is not None and "RA " in v.pixel_text(10.5, 10.5)

    class Readout:                                          # pixel_to_world_values only: the readout, not the export
        def pixel_to_world_values(self, x, y):
            return 40.0 + 0 * np.asarray(x), 61.0 + 0 * np.asarray(y)
    show(v, iv.ViewImage(sky(), "S2", key="ro", wcs=Readout()))
    assert "RA 02h40m00" in v.pixel_text(5.5, 5.5)
    dlg = v.export_dialog()
    assert not dlg.c_wcs.isChecked()
    dlg.reject()


def test_counts_and_numbers_without_a_unit(qapp, loader):
    counts = np.zeros((1, 300, 400), np.float32)
    counts[0, 10, 20] = 7
    v = make_view(qapp, loader)
    show(v, iv.ViewImage(counts, "Rejected", key="cnt", units="count"))
    assert v.pixel_text(20.5, 10.5).endswith("    7")
    show(v, iv.ViewImage(sky(channels=1), "Coverage", key="cov", units=""))
    t = v.pixel_text(20.5, 10.5)
    assert "ADU" not in t and v.panel.stats_l.text() == ""


def test_info_band_compact_flag_and_exclude_modes(qapp, loader):
    v = make_view(qapp, loader)
    fr = frames_for_test(2)
    fr[0].info = {"time": "2026-08-17T00:16:52", "exposure": 15.0, "iso": 5000, "compact": True,
                  "extra": [("Median level", "508 ADU")], "flag": "To check", "flag_reason": "an outlier"}
    v.set_sequence(fr, 0)
    assert v.wait(30)
    cells = v.band.cells
    assert cells["exposure"][1].isVisibleTo(v) and not cells["fwhm"][1].isVisibleTo(v)    # compact: no empty cells
    assert v.band.state.text() == "To check" and v.band.state.toolTip() == "an outlier"
    got = []
    v.excludeToggled.connect(lambda i, ex: got.append((i, ex)))
    v.set_exclude_mode("spec", "The project is open read-only.")
    assert not v.band.toggle.isEnabled() and "◇" in v.band.toggle.text()
    v.toggle_excluded()
    assert got == []                                        # ◇: nothing happens
    v.set_exclude_mode("off")
    assert not v.band.toggle.isVisibleTo(v) and not v.band.state.isVisibleTo(v)
    v.set_exclude_mode("on")
    v.toggle_excluded()
    assert got == [(0, True)] and v.band.toggle.isEnabled()


def test_send_to_post_can_be_withheld_with_its_reason(qapp, loader):
    v = make_view(qapp, loader, post=True)
    v.set_post_enabled(False, "Post does not take images yet.")
    show(v, iv.ViewImage(sky(), "S", key="np"))
    assert not v.post_b.isEnabled() and v.post_b.toolTip() == "Post does not take images yet."
    v.set_post_enabled(True)
    assert v.post_b.isEnabled()


# ---------------------------------------------------------------------------------------------- language
def _texts_of(path: Path) -> list[str]:
    out = []
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_" and n.args \
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            out.append(n.args[0].value)
    return out


def test_every_text_has_its_french():
    """Every text of the viewer has a French translation in the client's catalogues (all read together, as i18n does)."""
    cat: dict[str, str] = {}
    for f in sorted((ROOT / "astro_gpu/client/locale/fr").glob("*.json")):
        cat.update(json.loads(f.read_text(encoding="utf-8")))
    texts = _texts_of(ROOT / "astro_gpu/client/imageview.py") + [n for n, _s in iv.FORMATS.values()]
    missing = sorted({t for t in texts if t not in cat})
    assert not missing, missing


def test_french_view_has_no_missing_text_and_decimal_commas(qapp, loader, tmp_path):
    set_language("fr")
    try:
        v = make_view(qapp, loader, post=True)
        show(v, iv.ViewImage(sky(), "Empilement", key="fr", wcs=simple_wcs()))
        v.set_compare_targets([iv.ViewImage(sky(seed=5), "Avant", key="fr2")], current=0)
        assert v.wait(30)
        v.set_sequence(frames_for_test(2), 0)
        assert v.wait(30)
        assert "," in v.pixel_text(10.5, 10.5).split("R ")[1][:8]
        assert v.fit_b.toolTip().startswith("Adapter") and v.split_b.text() == "Rideau" and v.panel.neutral_c.text() == "Ciel neutre"  # noqa: E501
        assert v.band.cells["fwhm"][1].text().startswith("3,00") and v.band.cells["exposure"][1].text() == "15 s"
        assert "," in v.panel.fields["black"].text()
        dlg = v.export_dialog()
        dlg.fmt.setCurrentIndex(1)
        dlg.c_data.setChecked(True)
        dlg.reject()
        v.panel.mode_cb.setCurrentIndex(0)
        v.canvas.repaint()
        v.panel.hist.repaint()
        assert not i18n.missing(), sorted(i18n.missing())
    finally:
        set_language("en")


# ---------------------------------------------------------------------------------------------- the real stack
def _pct(xs, p):
    return float(np.percentile(np.asarray(xs), p))


@real_data
def test_real_stack_timings(qapp):
    """Open, zoom / pan frames and the stretch handle on the full-size stack, on the CPU (the report's numbers)."""
    ld = iv.Loader()
    try:
        v = iv.ImageView(loader=ld)
        v.resize(1680, 1000)
        v.show()
        QApplication.processEvents()
        arr = np.load(STACK)                                          # warm the file cache: the disk is not measured
        out: dict = {"canvas": [v.canvas.width(), v.canvas.height()]}
        t0 = time.perf_counter()
        v.set_image(iv.ViewImage(str(STACK), "Stack", key="real-file"))
        assert v.wait(60)
        out["open_file_s"] = time.perf_counter() - t0                 # np.load + pyramid + statistics + first frame
        t0 = time.perf_counter()
        v.set_image(iv.ViewImage(arr, "Stack", key="real-array"))
        assert v.wait(60)
        out["open_array_s"] = time.perf_counter() - t0
        c = v.canvas
        assert c.display.prep.size() == (arr.shape[-1], arr.shape[-2])
        QApplication.processEvents()
        # zoom: wheel steps from fit to 400 % and back, around a point; each frame painted at once
        frames = []
        pos = QPointF(c.width() * 0.45, c.height() * 0.4)
        for d in [120] * 12 + [-120] * 12:
            t = time.perf_counter()
            wheel(c, pos, d)
            c.repaint()
            frames.append(1000 * (time.perf_counter() - t))
        out["zoom_ms_median"], out["zoom_ms_p95"], out["zoom_ms_max"] = (_pct(frames, 50), _pct(frames, 95),
                                                                         max(frames))
        # pan at 100 %: 40 drags of 60 px (new tiles come in at the edges)
        v.set_zoom_percent(100)
        c.repaint()
        frames = []
        for i in range(40):
            t = time.perf_counter()
            c.pan_by(-60 if i < 20 else 0, -40 if i >= 20 else 0)
            c.repaint()
            frames.append(1000 * (time.perf_counter() - t))
        out["pan100_ms_median"], out["pan100_ms_p95"], out["pan100_ms_max"] = (_pct(frames, 50),
                                                                               _pct(frames, 95), max(frames))
        # pan at fit-ish (33 %)
        v.set_zoom_percent(33)
        c.repaint()
        frames = []
        for _i in range(20):
            t = time.perf_counter()
            c.pan_by(30, 20)
            c.repaint()
            frames.append(1000 * (time.perf_counter() - t))
        out["pan33_ms_median"] = _pct(frames, 50)
        # the stretch handle: a new black point, the canvas repainted (at fit and at 100 %)
        for name, zoom in (("fit", None), ("100", 100)):
            if zoom is None:
                v.fit()
            else:
                v.set_zoom_percent(zoom)
            c.repaint()
            st = v.stretch()
            frames = []
            for k in range(15):
                t = time.perf_counter()
                v.panel._edited("black", float(st.black[0]) + 2 * k)
                c.repaint()
                v.panel.hist.repaint()
                frames.append(1000 * (time.perf_counter() - t))
            out[f"stretch_{name}_ms_median"], out[f"stretch_{name}_ms_max"] = _pct(frames, 50), max(frames)
        out["cpu_threads"] = os.cpu_count()
        (shots_dir() / "timings.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1))
        assert out["open_array_s"] < 1.0 and out["open_file_s"] < 2.0
        assert out["zoom_ms_p95"] < 100 and out["pan100_ms_p95"] < 100
        assert out["stretch_fit_ms_median"] < 100 and out["stretch_100_ms_median"] < 100
        v.close()
    finally:
        ld.stop()
        ld.clear()
