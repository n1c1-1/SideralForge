"""Issue #57: light frames already in colour, added in the GUI (dropped on a Session block): they are taken as frames,
the session says their limit plainly, the Stacking block says Bayer drizzle does not apply, and calibration frames of
another kind are refused before the run, with the reason. Offscreen, nothing computed."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402
from astro_gpu.client.frames import colour_note_text  # noqa: E402
from astro_gpu.client.graph import GraphPage  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.runs import RunDialog  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.client.sources import frame_filter, no_lights_text  # noqa: E402
from astro_gpu.core import project as project_mod  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

BLOCKS = [{"block": "calibration"}, {"block": "registration"}, {"block": "stacking"}]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    state = tmp_path / "state"
    monkeypatch.setattr(project_mod, "STATE", state)
    monkeypatch.setenv("ASTRO_GPU_HOME", str(state))
    appmod.apply_language("en")
    yield tmp_path
    appmod.apply_language("en")


def jpegs(folder: Path, n: int = 3) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        assert cv2.imwrite(str(folder / f"IMG_{i:03d}.jpg"), np.full((40, 60, 3), 30 + i, np.uint8))
    return folder


def bayer_fits(folder: Path, n: int = 2) -> Path:
    from astropy.io import fits
    folder.mkdir(parents=True, exist_ok=True)
    h = fits.Header()
    h["BAYERPAT"] = "RGGB"
    for i in range(n):
        fits.writeto(folder / f"d_{i}.fits", np.zeros((40, 60), np.uint16), h, overwrite=True)
    return folder


def opened(tmp_path, lights=None, darks=None):
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    p.sessions.append(project2.Session(1, "night 1", lights=[str(lights)] if lights else [],
                                       darks=[str(darks)] if darks else []))
    w = p.add_workflow("Workflow", dict(wfl.TEMPLATES["Full"]))
    p.set_blocks(w.id, list(BLOCKS))
    project2.save(p)
    worker = ApiWorker()
    m = ProjectModel(p, worker, readonly=True, gpu=False, device="cpu")
    assert worker.wait(120) and m.ready, m.error
    m._sync()
    page = GraphPage(m)
    page.resize(1500, 800)
    return worker, m, page


def settle(w, page):
    w.wait(60)
    QApplication.processEvents()
    page.refresh()


def close(w, m, page):
    page.close()
    m.close(save=False)
    w.stop()
    QApplication.processEvents()


def labels(page, name):
    return [x.text() for x in page.panel.findChildren(QLabel) if x.objectName() == name]


def node(page, step):
    return page.graph.graph.node_of(page.s.path, step)


def test_a_dropped_folder_of_jpegs_is_taken_and_the_session_says_the_limit(tmp_path):
    w, m, page = opened(tmp_path)
    try:
        lights = jpegs(tmp_path / "phone")
        page.add_dropped_files(1, [str(lights)])
        settle(w, page)
        assert m.session_files(1, "lights") and len(m.session_files(1, "lights")) == 3
        assert m.file_warnings(1) == [] and m.run_problems() == []
        assert m.colour_note(1) == {"type": "JPG", "stretched": True} and m.colour_frames(m.paths[0])
        page.select_node(Graph(m).root)
        settle(w, page)
        said = labels(page, "colour_note")
        assert len(said) == 1 and "not linear data" in said[0] and "approximate" in said[0]
        assert "no Bayer drizzle" in said[0]
        assert any(c.key == "colour_frames" for c in api.health_checks(m.p))
        first = m.session_files(1, "lights")[0]
        thumb = api.frame_thumb(m.p, first, 64)                                  # the frames browser: in colour
        assert thumb.ndim == 3 and thumb.shape[2] == 3 and thumb.dtype == np.uint8
        from astro_gpu.client.imageview import ViewImage, _read
        arr, cfa, _wcs = _read(ViewImage(first, "IMG", cfa="RGB", camera=True))   # the viewer: no mosaic to read
        assert cfa == "" and arr.shape == (40, 60, 3)
    finally:
        close(w, m, page)


def test_the_limit_is_said_before_the_run(tmp_path):
    w, m, page = opened(tmp_path, jpegs(tmp_path / "phone"))
    try:
        d = RunDialog.__new__(RunDialog)
        d.s, d.plan = m, m.plan_all()
        said = RunDialog._file_warnings(d)
        assert len(said) == 1 and "night 1" in said[0] and "not linear data" in said[0]
    finally:
        close(w, m, page)


def test_the_stacking_block_says_bayer_drizzle_does_not_apply(tmp_path):
    from astro_gpu.client.panel import StackMethodCard
    from astro_gpu.client.settings import stack_method
    w, m, page = opened(tmp_path, jpegs(tmp_path / "phone"))
    try:
        p = m.paths[0]
        assert stack_method(m, p) == "classic" and m.colour_frames(p)
        page.select_node(node(page, "empilement"))
        settle(w, page)
        page.panel.expert = True
        page.panel.refresh(page.panel.node)
        settle(w, page)
        said = labels(page, "colour_no_drizzle")
        assert len(said) == 1 and "Bayer drizzle does not apply" in said[0]
        assert not page.panel.findChildren(StackMethodCard)                      # no drizzle to choose
        rows = " ".join(x.text() for x in page.panel.findChildren(QLabel))
        assert "Debayering" not in rows                                          # nothing is demosaiced
    finally:
        page.panel.expert = False
        close(w, m, page)


def test_a_raw_mosaic_dark_for_jpeg_lights_is_refused_before_the_run(tmp_path):
    w, m, page = opened(tmp_path, jpegs(tmp_path / "phone"), bayer_fits(tmp_path / "darks"))
    try:
        why = m.run_problems()[0][1]
        assert "night 1: the darks (d_0.fits) are not of the same kind as the lights" in why
        assert "cannot calibrate them" in why and Graph(m).root.broken == why
        appmod.apply_language("fr")
        assert "ils ne peuvent pas les calibrer" in m.run_problems()[0][1]
    finally:
        appmod.apply_language("en")
        close(w, m, page)


def test_a_jpeg_dark_for_raw_lights_is_refused_too(tmp_path):
    w, m, page = opened(tmp_path, bayer_fits(tmp_path / "lights", 3), jpegs(tmp_path / "darks"))
    try:
        why = m.run_problems()[0][1]
        assert "the darks (IMG_000.jpg) are not of the same kind as the lights" in why
        warns = m.file_warnings(1)
        assert [x["key"] for x in warns] == ["wrong_kind"]
    finally:
        close(w, m, page)


def test_an_rgb_fits_session_is_said_too(tmp_path):
    from astropy.io import fits
    d = tmp_path / "rgb"
    d.mkdir()
    for i in range(2):
        fits.writeto(d / f"r_{i}.fits", np.zeros((3, 40, 60), np.uint16), overwrite=True)
    w, m, page = opened(tmp_path, d)
    try:
        assert m.colour_note(1) == {"type": "FITS", "stretched": False}
        assert "If they were stretched" in colour_note_text(m.colour_note(1))
    finally:
        close(w, m, page)


def test_the_file_dialog_shows_them_and_a_folder_of_other_files_says_why():
    assert "*.jpg" in frame_filter() and "*.tiff" in frame_filter() and "*.png" in frame_filter()


def test_no_lights_text_names_the_other_files(tmp_path):
    (tmp_path / "phone").mkdir()
    (tmp_path / "phone" / "a.MOV").write_bytes(b"\0")
    assert "only MOV files" in no_lights_text([str(tmp_path / "phone")])
    i18n.set_language("fr")
    try:
        assert "seulement des fichiers MOV" in no_lights_text([str(tmp_path / "phone")])
    finally:
        i18n.set_language("en")


def test_french_for_every_new_text():
    i18n.set_language("fr")
    try:
        note = colour_note_text({"type": "JPG", "stretched": True})
        assert "déjà en couleur" in note and "linéaires" in note and "approximative" in note
        assert "Si elles ont été étirées" in colour_note_text({"type": "FITS", "stretched": False})
        assert i18n._("Classic stack · colour frames") == "Empilement classique · images couleur"
    finally:
        i18n.set_language("en")
