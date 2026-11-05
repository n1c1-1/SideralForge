"""Issues #31, #32: every file dialog opens in the last folder used (not a workspace), the import's kinds share it,
and New project chooses its location."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import prefs  # noqa: E402


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    prefs.set("workspace", (tmp_path / "ws").as_posix())
    (tmp_path / "ws").mkdir()
    yield tmp_path
    prefs.load()


def test_the_last_folder_used_is_where_the_next_dialog_opens(env):
    assert prefs.last_folder() == (env / "ws").as_posix()                     # the first time: the old workspace
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
