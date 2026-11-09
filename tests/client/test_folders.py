"""Issues #31, #32: every file dialog opens in the last folder used (not a workspace), and New project chooses its
location."""
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


def test_new_project_chooses_its_location(env, monkeypatch):
    QApplication.instance() or QApplication([])
    from PySide6.QtWidgets import QFileDialog

    from astro_gpu.client.home import NewProjectDialog
    elsewhere = env / "disk2"
    elsewhere.mkdir()
    d = NewProjectDialog(prefs.last_folder(), "NGC 7000")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: elsewhere.as_posix())
    d._browse()
    assert d.folder() == elsewhere / "NGC_7000" or d.folder().parent == elsewhere
    assert prefs.last_folder() == elsewhere.as_posix()


def test_the_astrometry_never_takes_the_projects_name_for_a_target():
    """Issue #38: a project named "test" was looked up in the catalogues and on the network."""
    import inspect as _inspect

    from astro_gpu.stages import astrometry
    assert "config.name" not in _inspect.getsource(astrometry.Astrometry.auto)
