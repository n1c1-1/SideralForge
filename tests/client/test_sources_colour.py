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
from astro_gpu.client.sources import frame_filter  # noqa: E402
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
        page.add_dropped_files(1, [str(lights)], "lights")
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


def computed(p, wid, stage, files, metrics):
    """A done result of a stage in the project's store, written as a run writes it."""
    import json

    from astro_gpu.core.store import Store
    k = api.stage_key(p, wid, stage)
    d = Store(p.root).dir(stage, k)
    d.mkdir(parents=True, exist_ok=True)
    for name, x in files.items():
        if isinstance(x, np.ndarray):
            np.save(d / name, x)
        else:
            (d / name).write_text(json.dumps(x), encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps({
        "stage": stage, "status": "done", "key": k, "run_id": k, "metrics": metrics, "params": {}, "origin": {},
        "auto_reasons": {}, "duration": 1.0, "finished": "2026-11-08T12:00:00"}), encoding="utf-8")


@pytest.mark.parametrize("on_disk", [True, False], ids=["from_the_cache", "from_the_masters"])
def test_inspect_shows_a_calibrated_colour_frame_in_colour(tmp_path, on_disk):
    """The Calibrated lights tab of the real Inspect view: the frame shown has its three colours (it was the grey
    luminance), from the calibrated planes on disk, or made again colour by colour when they are gone; the same
    through what Post reads."""
    from astro_gpu.client import app as appmod_
    from astro_gpu.client.frames import FramesView
    from astro_gpu.client.imageview import ImageView, default_loader
    from astro_gpu.client.services import inspect as si
    from astro_gpu.client.sources import SourcesView
    from astro_gpu.client.steps import Steps
    from astro_gpu.client.tune import TuneView
    H, W = 40, 60
    lights = jpegs(tmp_path / "phone", 2)
    for f in sorted(lights.iterdir()):
        assert cv2.imwrite(str(f), np.dstack([np.full((H, W), v, np.uint8) for v in (90, 60, 30)]))   # B, G, R
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    p.sessions.append(project2.Session(1, "night 1", lights=[str(lights)]))
    w = p.add_workflow("Workflow", {})
    p.set_blocks(w.id, list(BLOCKS))
    project2.save(p)
    paths = [str(f) for f in sorted(lights.iterdir())]
    computed(p, w.id, "inventory", {"frames.json": {
        "lights": [{"path": f, "cfa": "RGB", "white": 255.0, "black": 0.0, "time": "", "exposure": None, "iso": None,
                    "temperature": None} for f in paths], "darks": [], "flats": [], "bias": [], "flat_darks": []}},
        {"cfa": "RGB", "colour_frames": True, "n_lights": 2, "width": W, "height": H, "white_level": 255.0})
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    lum, rgb = scratch / "l.f16", scratch / "r.f16"
    planes = np.stack([np.full((2, H, W), v, np.float16) for v in (28.0, 58.0, 88.0)], 1)   # the calibrated cache
    if on_disk:
        planes.tofile(rgb)
    np.full((2, H, W), 50.0, np.float16).tofile(lum)
    cal = {"master_dark.npy": np.full((H, W), 2.0, np.float32), "master_flat.npy": np.ones((H, W), np.float32),
           "badmask.npy": np.zeros((H, W), bool), "frame_stats.npy": np.full((2, 4, 2), 5.0, np.float32),
           "master_dark_rgb.npy": np.full((3, H, W), 2.0, np.float32), "master_flat_rgb.npy": np.ones((3, H, W), np.float32),
           "cache.json": {"path": str(lum), "rgb_path": str(rgb), "n": 2, "height": H, "width": W, "scale": 1.0, "cfa": "GGGG",
                          "colour": True, "sat_mark": 65000.0, "valid_max": 60000.0, "white_level": 255.0,
                          "dark_level": 2.0}}
    computed(p, w.id, "calibrate", cal, {"cache_scale": 1.0})
    worker = ApiWorker()
    m = ProjectModel(p, worker, readonly=True, gpu=False, device="cpu")
    assert worker.wait(120) and m.ready, m.error
    m._sync()
    steps = Steps(m)
    t = TuneView(m, steps, FramesView(m, SourcesView(m)), appmod_.Placeholder("Post"))
    t.resize(1400, 800)
    t.show()

    def settle():
        for _i in range(12):
            worker.wait(30)
            t.service.wait(30)
            QApplication.processEvents()
        default_loader().wait(30)
        for v_ in t.findChildren(ImageView):
            v_.wait(30)
        QApplication.processEvents()
    try:
        t.tab_of["calibration"] = "calibrated"
        m.select(index=0, step="calibration")
        t.refresh()
        settle()
        v = t.view
        assert v is not None and len(v.frames()) == 2
        prep = v._prep
        assert prep is not None and prep.cfa == "" and prep.channels == 3
        assert "RGB" in v._size_text(prep) and "Bayer" not in v._size_text(prep)
        a = np.asarray(prep.data)
        a = a if a.shape[-1] == 3 else np.moveaxis(a, 0, -1)
        assert np.allclose(a.reshape(-1, 3).mean(0), (28.0, 58.0, 88.0), atol=1.5)        # red, green, blue above the dark
        out, cfa, how = si.calibrated_frame(m.p, w.id, paths[0])
        assert cfa == "" and out.shape == (H, W, 3) and how == ("cache" if on_disk else "masters")
        assert si.post_source(v.frames()[0])().shape == (H, W, 3)                         # Post reads the colours too
    finally:
        t.close()
        t.service.stop() if hasattr(t.service, "stop") else None
        m.close(save=False)
        worker.stop()
        QApplication.processEvents()


def test_the_file_dialog_shows_them():
    assert "*.jpg" in frame_filter() and "*.tiff" in frame_filter() and "*.png" in frame_filter()


def test_french_for_every_new_text():
    i18n.set_language("fr")
    try:
        note = colour_note_text({"type": "JPG", "stretched": True})
        assert "déjà en couleur" in note and "linéaires" in note and "approximative" in note
        assert "Si elles ont été étirées" in colour_note_text({"type": "FITS", "stretched": False})
        assert i18n._("Classic stack · colour frames") == "Empilement classique · images couleur"
    finally:
        i18n.set_language("en")
