"""Round 2 of the GUI exploration (engine_shapes): data that cannot be computed is refused BEFORE the run, with the
reason in plain words (PRINCIPLES: nothing runs without data) — a lights folder that is not there, .fits files that are
not FITS, a single light frame, darks of another size than the lights. The checks read headers only, in the worker;
the session's block is circled in red. Offscreen, tiny FITS frames, nothing computed."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402
from astro_gpu.client.model import ProjectModel, session_checks  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project as project_mod  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402


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


def fits_frames(folder: Path, n: int, h: int = 40, w: int = 60) -> Path:
    from astropy.io import fits
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        hd = fits.Header()
        hd["BAYERPAT"] = "RGGB"
        fits.writeto(folder / f"f_{i:03d}.fits", np.full((h, w), 1000, np.uint16), hd, overwrite=True)
    return folder


BLOCKS = [{"block": "calibration"}, {"block": "registration"}, {"block": "stacking"}]


def model(tmp: Path, lights: Path, **kinds):
    p = project2.create(tmp / "P", "P", Target(name="M 31"))
    p.sessions.append(project2.Session(1, "night 1", lights=[str(lights)], **{k: [str(v)] for k, v in kinds.items()}))
    w = p.add_workflow("Workflow", {})
    p.set_blocks(w.id, list(BLOCKS))
    project2.save(p)
    worker = ApiWorker()
    m = ProjectModel(p, worker, readonly=True, gpu=False, device="cpu")
    assert worker.wait(120) and m.ready, m.error
    return m, worker


def close(m, worker):
    m.close(save=False)
    worker.stop()
    QApplication.processEvents()


def test_good_frames_are_not_refused(env):
    m, wk = model(env, fits_frames(env / "l", 3), darks=fits_frames(env / "d", 2))
    try:
        assert m.run_problems() == []
        assert not Graph(m).root.broken
    finally:
        close(m, wk)


def test_a_lights_folder_not_there_is_refused_with_its_folder_and_the_session_block_is_red(env):
    """engine_shapes: a session's folder gone (disk unplugged): Run stayed active ("≈ 3 h 03"), then failed in the
    run. Now refused before, naming the session and the folder."""
    lights = fits_frames(env / "gone", 3)
    shutil.rmtree(lights)
    m, wk = model(env, lights)
    try:
        probs = m.run_problems()
        assert len(probs) == 1 and probs[0][0] == "Workflow"
        assert probs[0][1].startswith("night 1: the folder of the lights is not found:") and "gone" in probs[0][1]
        assert Graph(m).root.broken == probs[0][1]
        appmod.apply_language("fr")
        assert "dossier des brutes introuvable" in m.run_problems()[0][1]
    finally:
        close(m, wk)


def test_fits_files_that_are_not_fits_are_refused_naming_one(env):
    d = env / "bad"
    d.mkdir()
    for i in range(3):
        (d / f"light_{i:03d}.fits").write_text("not a fits file\n" * 50)
    m, wk = model(env, d)
    try:
        why = m.run_problems()[0][1]
        assert "light_000.fits is not a frame that can be read" in why
    finally:
        close(m, wk)


def test_a_single_light_frame_is_refused(env):
    """engine_shapes: one light: Stacking failed with "fewer than 2 usable frames after selection" (in English, and
    there was no Selection block)."""
    m, wk = model(env, fits_frames(env / "one", 1))
    try:
        assert m.run_problems() == [("Workflow", "only 1 light frame: stacking needs at least 2")]
        appmod.apply_language("fr")
        assert m.run_problems()[0][1] == "une seule brute : l'empilement en demande au moins 2"
    finally:
        close(m, wk)


def test_darks_of_another_size_are_refused(env):
    m, wk = model(env, fits_frames(env / "l", 3), darks=fits_frames(env / "d", 2, h=20, w=30))
    try:
        why = m.run_problems()[0][1]
        assert why == "night 1: the darks are 30×20 pixels, the lights 60×40: they cannot calibrate them"
    finally:
        close(m, wk)


def test_the_checks_read_headers_only_and_are_remembered(env, monkeypatch):
    """Cheap checks: a FITS frame's pixels are never read, and a frame already checked is not read again."""
    from astro_gpu.client import model as model_mod
    lights = fits_frames(env / "l", 4)
    p = project2.create(env / "P", "P", Target(name="M 31"))
    s = project2.Session(1, "night 1", lights=[str(lights)])
    files = {"lights": [x.as_posix() for x in sorted(lights.iterdir())]}
    assert session_checks(p, s, files) == []
    assert sum(1 for k in model_mod._FRAME_READS if k[0] in files["lights"] and k[3]) == 1   # one header read
    calls = []
    import astro_gpu.io.frames as frames_mod
    monkeypatch.setattr(frames_mod, "fits_header", lambda path: calls.append(path))
    assert session_checks(p, s, files) == []
    assert calls == []                                    # remembered: the headers are not read again
