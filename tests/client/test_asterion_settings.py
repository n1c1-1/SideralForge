"""Asterion in the interface: its status in Settings › Catalogs & astrometry (installed or not, where; no install button
yet), the line of the health check when it is missing (information: the Astrometry block works without it), its
credit in Help › About. Offscreen; the program is a file standing for it, its version given."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402
from astro_gpu.client.settingsdlg import SettingsDialog  # noqa: E402
from astro_gpu.core import asterion  # noqa: E402

MACHINE = Machine([Gpu("cuda:0", "NVIDIA RTX 4060 Ti", 16.0)], 16, 64.0)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    for k in (asterion.ENV_PROGRAM, asterion.ENV_CATALOGS):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "userhome"))
    monkeypatch.setattr(asterion, "install_dir", lambda here=None: None)
    monkeypatch.setattr(asterion, "program_version", lambda exe: "0.4.2")
    prefs.load()
    i18n.set_language("en")
    yield tmp_path
    i18n.set_language("en")


def _install(program=True, index=True):
    if program:
        asterion.own_dir().mkdir(parents=True, exist_ok=True)
        (asterion.own_dir() / asterion.program_name()).write_bytes(b"")
    if index:
        asterion.own_catalogs().mkdir(parents=True, exist_ok=True)
        (asterion.own_catalogs() / "tycho2.astx").write_bytes(b"x")


def _section(qapp) -> list[str]:
    """The texts of the Asterion section of the Catalogs page (from its title to the next title)."""
    d = SettingsDialog(None, MACHINE)
    d.show()
    d.show_page("catalogs")
    qapp.processEvents()
    page = d.pages.currentWidget()
    texts = [w.text() for w in page.findChildren(QLabel) if w.isVisibleTo(page) and w.text()]
    buttons = [b.text() for b in page.findChildren(QPushButton) if b.isVisibleTo(page)]
    d.close()
    i = next(k for k, t in enumerate(texts) if t in ("Asterion plate solver", "Solveur astrométrique Asterion"))
    return texts[i:i + 6], buttons


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_settings_say_asterion_is_not_installed_without_a_button(env, qapp, lang):
    i18n.set_language(lang)
    texts, buttons = _section(qapp)
    assert texts.count("Not installed" if lang == "en" else "Non installé") == 2
    assert ("Program" if lang == "en" else "Programme") in texts
    assert not any("Asterion" in b or "index" in b.lower() for b in buttons)      # how it comes is not decided yet


def test_settings_say_where_asterion_and_its_index_are(env, qapp):
    _install()
    texts, _b = _section(qapp)
    prog = next(t for t in texts if t.startswith("Version 0.4.2 · "))
    assert prog.endswith(str(asterion.own_dir() / asterion.program_name()))
    assert f"tycho2 · {asterion.own_catalogs()}" in texts


def test_settings_say_when_the_program_is_there_without_its_index(env, qapp):
    _install(index=False)
    texts, _b = _section(qapp)
    assert any(t.startswith("Version 0.4.2 · ") for t in texts) and "Not installed" in texts


def test_the_health_check_says_it_as_information(env):
    from astro_gpu import api
    from astro_gpu.client.frames import check_text
    c = api.Check("info", "asterion", "x", {"missing": True, "why": "the Asterion plate solver is not installed"})
    assert check_text(c).startswith("Astrometry: Asterion is not installed, the field is found by FLOAT's own search")
    i18n.set_language("fr")
    assert check_text(c).startswith("Astrométrie : Asterion n'est pas installé")
    c = api.Check("info", "asterion", "x", {"missing": False, "why": "Asterion 0.3.9 is too old"})
    assert "Asterion ne peut pas tourner (Asterion 0.3.9 is too old)" in check_text(c)


def test_the_about_box_credits_asterion(env):
    from astro_gpu.client.credits import DATA_SOURCES, data_sources_html
    s = next(s for s in DATA_SOURCES if s["name"] == "Asterion")
    assert s["licence"] == "Apache License 2.0" and s["version"] == "0.4.2"
    i18n.set_language("fr")
    assert "Le solveur astrométrique qui donne à l&#x27;astrométrie sa première solution" in data_sources_html()


def test_the_health_check_says_asterion_is_missing_as_information(env, tmp_path, monkeypatch):
    """A workflow with Astrometry, Asterion not installed: one line of information (the block works without it, it is
    not greyed); installed: nothing; no Astrometry in the chain: nothing."""
    from astro_gpu import api
    from tests.test_api import write_fits
    for i in range(2):
        write_fits(tmp_path / "raw" / f"L{i}.fits", 200, f"2026-08-16T22:0{i}:00")
    p = api.create_project(tmp_path / "p", "p")
    api.add_frames(p, api.scan_folder(tmp_path / "raw"))
    api.add_workflow(p, "Classic", "Classic stack")
    keys = api._keys(p)
    assert any("astrometry" in keys.chain(w) for w in p.workflows)
    monkeypatch.setattr(asterion, "status", lambda *a, **k: asterion.Status(
        False, why="the Asterion plate solver is not installed", missing=True))
    got = [c for c in api.health_checks(p) if c.key == "asterion"]
    assert len(got) == 1 and got[0].level == "info" and got[0].args["missing"]
    assert got[0].text.startswith("Astrometry: Asterion is not installed")
    monkeypatch.setattr(asterion, "status", lambda *a, **k: asterion.Status(True))
    assert not [c for c in api.health_checks(p) if c.key == "asterion"]
    monkeypatch.setattr(api._keys(p), "chain", lambda w: ["inventory", "stack"])
    monkeypatch.setattr(asterion, "status", lambda *a, **k: asterion.Status(False, missing=True))
    assert not [c for c in api.health_checks(p) if c.key == "asterion"]
