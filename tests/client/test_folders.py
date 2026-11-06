"""Issues #31, #32: every file dialog opens in the last folder used (not a workspace), the import's kinds share it,
and New project chooses its location."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import prefs  # noqa: E402


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))       # the user's home folder
    yield tmp_path
    prefs.load()


def test_the_last_folder_used_is_where_the_next_dialog_opens(env):
    assert prefs.last_folder() == str(env)                                       # the first time: the home folder
    lights = env / "night" / "LIGHTS"
    lights.mkdir(parents=True)
    f = lights / "a.cr3"
    f.write_bytes(b"")
    prefs.remember_folder(f)                                                     # a file: its folder
    assert prefs.last_folder() == lights.as_posix()
    from astro_gpu.client.sources import KindRow
    QApplication.instance() or QApplication([])
    flats = KindRow("flats", "Flats")
    assert flats._start() == lights.as_posix()                                   # the flats start where the lights were


def test_new_project_chooses_its_location(env, monkeypatch):
    QApplication.instance() or QApplication([])
    from PySide6.QtWidgets import QFileDialog

    from astro_gpu.client.home import NewProjectDialog
    elsewhere = env / "disk2"
    elsewhere.mkdir()
    d = NewProjectDialog(prefs.last_folder(), "NGC 7000", "")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: elsewhere.as_posix())
    d._browse()
    assert d.folder() == elsewhere / "NGC_7000" or d.folder().parent == elsewhere
    assert prefs.last_folder() == elsewhere.as_posix()


def test_new_project_checks_its_target_and_takes_the_capture_data(env):
    """Issue #38: the target is looked up offline as it is entered (found: its coordinates go with the project), the
    capture data are optional; the astrometry never takes the project's name for a target."""
    QApplication.instance() or QApplication([])
    from astro_gpu.client.home import NewProjectDialog
    d = NewProjectDialog(prefs.last_folder(), "test", "NGC 7000")
    assert d.found.text().startswith("Found: RA 314.8")
    d.focal.setText("135")
    d.pixel.setText("5,36")
    det = d.details()
    assert abs(det["ra"] - 314.82) < 0.05 and det["focal_mm"] == 135.0 and det["pixel_um"] == 5.36
    d.target.setText("test")
    d._check_target()
    assert d.found.text().startswith("Not in the catalogues here") and "ra" not in d.details()
    import inspect as _inspect

    from astro_gpu.stages import astrometry
    src = _inspect.getsource(astrometry.Astrometry.auto) if hasattr(astrometry, "Astrometry") else ""
    assert "config.name" not in src
