"""Viewer and Inspect details on synthetic data: chart export, copy, camera colour, sky objects, calibration flags.

1. every chart has a visible Export CSV… button (not in its PNG); 2. Copy image says what it copies; 3. a camera-colour
picture is shown as it is, Neutral sky is a display option that Auto-stretch off / on keeps; 4. the compared picture
follows the main one's Neutral sky, or its own option; 6. the named objects: their outlines on the picture, burned into a
PNG and never into a FITS; 7. the engine's flags of a calibration frame: red marks, the reason written out, in French
too.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu import texts  # noqa: E402
from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client import imageview as iv  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS)
    return a


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    yield tmp_path
    i18n.set_language("en")


def camera_sky(h=240, w=320, seed=3) -> np.ndarray:
    """A linear sky in the camera's colours: a green cast, noise and a few stars."""
    rng = np.random.default_rng(seed)
    img = np.empty((3, h, w), np.float32)
    for c, k in enumerate((1.0, 1.7, 0.6)):
        img[c] = 200 * k + rng.normal(0, 4, (h, w))
    for _i in range(40):
        y, x = rng.integers(5, h - 5), rng.integers(5, w - 5)
        img[:, y - 1:y + 2, x - 1:x + 2] += 3000
    return img


def show(v, image, **kw):
    v.set_image(image, **kw)
    assert v.wait(30)
    QApplication.processEvents()


# ---------------------------------------------------------------------------------------------- 1. charts
def test_every_chart_has_a_visible_export_csv_button_that_the_png_leaves_out(qapp, tmp_path, monkeypatch):
    from astro_gpu.client.chartview import ChartView
    ch = ChartView("Weights")
    ch.add_line("w", np.arange(5.0), np.arange(5.0) ** 2)
    ch.fit()
    ch.resize(600, 300)
    ch.show()
    QApplication.processEvents()
    b = ch._csv_btn
    assert b.isVisible() and b.text() == "Export CSV…" and b.x() + b.width() <= ch.width()
    got = []
    monkeypatch.setattr(ch, "export_csv", lambda path=None: got.append(path))
    b.click()
    assert got == [None]                                          # the dialog of the export, as the menu's entry
    ch.render_image()
    assert b.isVisible()                                          # hidden only while the picture is made
    assert any(a.text() == "Export CSV…" for a in ch.build_menu().actions())


# ---------------------------------------------------------------------------------------------- 2-4. the viewer
def test_copy_image_says_what_it_copies(qapp):
    v = iv.ImageView()
    assert v.copy_b.text() == "Copy image" and "clipboard" in v.copy_b.toolTip()
    i18n.set_language("fr")
    v2 = iv.ImageView()
    assert v2.copy_b.text() == "Copier l'image"


def test_a_camera_colour_stack_shows_as_it_is_and_auto_stretch_off_on_keeps_the_choice(qapp):
    v = iv.ImageView()
    show(v, iv.ViewImage(camera_sky(), "Stack", key="stack", camera=True, neutral=False))
    assert v.stretch().auto and not v.stretch().neutral          # camera colours, from the first display
    v.auto_b.setChecked(False)
    v.auto_b.setChecked(True)                                     # the user's two clicks
    assert not v.stretch().neutral
    v.set_neutral(True)                                           # the explicit display option
    assert v.stretch().neutral and v.panel.neutral_c.isChecked()
    v.auto_b.setChecked(False)
    v.auto_b.setChecked(True)
    assert v.stretch().neutral                                    # was lost before: the cause of the colour switch
    # without ``neutral``, a camera-colour image keeps its default (a neutral sky): the raw frames
    w = iv.ImageView()
    show(w, iv.ViewImage(camera_sky(seed=4), "Frame", key="frame", camera=True))
    assert w.stretch().neutral


def test_the_before_follows_the_main_pictures_neutral_sky_or_its_own_option(qapp):
    v = iv.ImageView()
    after = iv.ViewImage(camera_sky(seed=5) * np.array([1.0, 0.6, 1.6], np.float32)[:, None, None], "After",
                         key="after", camera=False)
    before = iv.ViewImage(camera_sky(seed=6), "Before", key="before", camera=True, neutral=False)
    show(v, after)
    v.set_compare_targets([before], 0)
    assert v.wait(30)
    assert v._cmp is not None and not v._cmp.stretch.neutral     # "before" as it is: no sky correction
    v.set_compare_neutral(True)
    assert v._cmp.stretch.neutral and not v.stretch().neutral    # the before's own option: the after is not touched
    v.set_compare_neutral(None)
    assert not v._cmp.stretch.neutral


# ---------------------------------------------------------------------------------------------- 6. the named objects
def _wcs(w=400, h=300, scale_deg=0.01):
    from astropy.wcs import WCS
    c = WCS(naxis=2)
    c.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    c.wcs.crval = [40.0, 61.0]
    c.wcs.crpix = [w / 2 + 0.5, h / 2 + 0.5]
    c.wcs.cdelt = [-scale_deg, scale_deg]
    return c


def _objects():
    from astro_gpu.core.skyobjects import SkyObject
    return [SkyObject("IC 1805", "Heart Nebula", "cluster with nebula", 40.0, 61.0, 60.0, 60.0, 0.0),
            SkyObject("NGC 896", "", "emission nebula", 40.6, 61.3, 0.0, 0.0, 0.0),
            SkyObject("NGC 9999", "", "galaxy", 120.0, -20.0, 5.0, 2.0, 30.0)]


def test_named_objects_are_outlined_at_their_size_and_burned_into_a_picture_never_a_fits(qapp, tmp_path):
    from astro_gpu.client import annotate as an
    from astro_gpu.client.services.inspect import SkyData
    wcs = _wcs()
    objs = an.named_objects(wcs, 400, 300, objects=_objects())
    names = [o.name for o in objs]
    assert names[0] == "IC 1805" and "NGC 9999" not in names               # the common name first; far away: out
    heart = objs[0]
    assert heart.label == "IC 1805 · Heart Nebula" and heart.outline is not None
    assert heart.size_px == pytest.approx(100, rel=0.05)                   # 60' at 36"/px
    sky = SkyData(an.sky_grid_lines(wcs, 400, 300) if hasattr(an, "sky_grid_lines") else [], None, objs)
    img = np.full((300, 400, 3), 0.1, np.float32)
    out = an.burn(img, sky, {"grid": False, "objects": True})
    assert out.shape == img.shape and (np.abs(out - img) > 0.05).any()
    assert np.array_equal(an.burn(img, sky, {"grid": False, "objects": False}), img)
    # the viewer's export: the picture gets them, the FITS file keeps the data and the solution
    prep = iv.prepare(iv.ViewImage(camera_sky(300, 400), "Stack", key="s", wcs=wcs, camera=False))
    st = iv.default_stretch(prep)
    plain = iv.export_image(tmp_path / "plain.png", prep, st, fmt="png16", content="view")
    named = iv.export_image(tmp_path / "named.png", prep, st, fmt="png16", content="view",
                            annotations=(sky, {"objects": True}))
    import cv2
    a, b = cv2.imread(str(plain), cv2.IMREAD_UNCHANGED), cv2.imread(str(named), cv2.IMREAD_UNCHANGED)
    assert a.shape == b.shape and (a != b).any()
    f = iv.export_image(tmp_path / "data.fits", prep, st, fmt="fits", annotations=(sky, {"objects": True}))
    from astropy.io import fits
    with fits.open(f) as hl:
        assert np.allclose(hl[0].data, prep.data) and hl[0].header["CTYPE1"].startswith("RA---TAN")


def test_the_annotations_of_a_crop_move_with_it(qapp):
    from astro_gpu.client import annotate as an
    wcs = _wcs()
    full = an.named_objects(wcs, 400, 300, objects=_objects())
    crop = an.named_objects(an.crop_wcs(wcs, (50, 20)), 300, 250, objects=_objects())
    assert crop[0].x == pytest.approx(full[0].x - 50, abs=0.01) and crop[0].y == pytest.approx(full[0].y - 20, abs=0.01)
    moved = an.shifted(type("S", (), {"lines": [], "objects": full})(), 50, 20)
    assert moved.objects[0].x == pytest.approx(full[0].x - 50)


# ---------------------------------------------------------------------------------------------- 7. calibration flags
def test_the_flagged_frames_are_marked_red_and_their_reason_written_out(qapp):
    v = iv.ImageView()
    frames = [iv.ViewImage(camera_sky(seed=s), f"d{s}", key=f"d{s}", camera=True,
                           info={"flag": "Suspicious", "flag_level": lvl, "flag_reason": why} if lvl else {})
              for s, lvl, why in ((1, "bad", "Contains stars (410 found)"), (2, "warn", "Unlike the others"),
                                  (3, None, ""))]
    v.set_sequence(frames, 0)
    assert v.wait(30)
    assert v.strip.flagged == {0: "bad", 1: "warn"}
    assert v.band.why.isVisibleTo(v) and "Contains stars" in v.band.why.text() and "#" in v.band.why.styleSheet()
    assert iv.BAD.lower() in v.band.state.styleSheet().lower()                  # bad: red, more than warn's amber
    v.set_index(2)
    assert not v.band.why.isVisibleTo(v)


def test_every_check_of_the_engine_reads_as_the_engine_wrote_it_and_in_french():
    from types import SimpleNamespace

    from astro_gpu.client import tune
    from astro_gpu.core import diagnostics as dg
    base = {"median": 520.0, "noise": 8.0, "hot_pixels": 100, "gradient": 1.0, "temperature": 12.0, "exposure": 15.0,
            "saturated_pct": 0.0, "level_pct": 40.0, "stars": 2, "black": 512.0, "white": 16383.0, "camera_raw": True}
    odd = dict(base, stars=410, median=900.0, noise=30.0, hot_pixels=900, gradient=-12.0, temperature=25.0,
               exposure=30.0)
    for kind in ("darks", "bias", "flat_darks", "flats"):
        stats = [dict(base, level_pct=40.0 + i * 0.1) for i in range(5)] + [dict(odd, level_pct=90.0,
                                                                             saturated_pct=0.5)]
        flag, reason, checks = dg.flag_frames(kind, stats)[-1]
        assert flag == "bad" and checks
        cf = SimpleNamespace(flag=flag, reason=reason, checks=checks)
        en = tune.frame_reason(cf, kind, "en")
        assert en == reason, (kind, en, reason)                     # the templates say what the engine says
        fr = tune.frame_reason(cf, kind, "fr")
        assert fr != en and "étoiles" in fr and "found" not in fr and "," in fr.split("ADU")[0] + ",", (kind, fr)
        for c in checks:                                              # each sentence has its French
            t = tune.check_template(c, kind)
            assert t is not None and texts.has_translation(t[0], "fr"), (kind, c["key"])
