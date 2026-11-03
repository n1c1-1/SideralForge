"""Home in sections (the workspace and the folders the user adds), and Settings › Folders, offscreen.

"My projects" (the workspace, projects up to two levels down), "Test copies" (closed until its header is clicked);
hidden folders; Settings › Folders edits the sections; every text has its French; no disk access in the GUI thread.

Everything is built in a scratch folder: the user's F:/astro_projets and F:/astro_essais are never read here.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import home, i18n, prefs  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402
from astro_gpu.client.settingsdlg import SettingsDialog  # noqa: E402
from astro_gpu.client.widgets import QSS, ElidedLabel  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "astro_gpu" / "client"
MACHINE = Machine([Gpu("cuda:0", "NVIDIA RTX 4060 Ti", 16.0)], 16, 64.0)


@pytest.fixture(scope="module")
def qapp():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


def _project(folder: Path, name: str, target: str = "", description: str = "") -> Path:
    p = project2.create(folder, name, Target(name=target))
    p.description = description
    project2.save(p)
    return folder


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    """A scratch workspace and test copies; the sections point at them; English; a fixed machine."""
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    prefs.load()
    ws, copies = tmp_path / "ws", tmp_path / "copies"
    for d in (ws, copies):
        d.mkdir()
    _project(ws / "Heart", "HeartSoul", "IC 1805")
    _project(ws / "Iris", "Iris", "NGC 7023")
    for hidden in ("_qc", "pytest-12", "tmp3", "temp", "Heart_tmp", ".git"):
        _project(ws / hidden, f"hidden {hidden}")
    _project(ws / "Old" / "Wizard", "Wizard", "NGC 7380")                 # a project one level down
    for n in ("heart_a7s", "heart_a7s_copy"):
        _project(copies / n, "heart_a7s", "IC 1805", "The reference Heart")
    prefs.set("workspace", ws.as_posix())
    prefs.set("cache_dir", (tmp_path / "scratch").as_posix())
    rows = home.section_rows()
    rows[1]["folder"] = copies.as_posix()
    home.save_rows(rows)
    appmod.apply_language("en")
    monkeypatch.setattr(appmod, "machine_info", lambda: MACHINE)
    windows = []
    yield {"ws": ws, "copies": copies, "tmp": tmp_path, "windows": windows}
    for w in windows:
        w.close()
        w.deleteLater()
    qapp.processEvents()
    appmod.apply_language("en")


def _main(env) -> appmod.Main:
    w = appmod.Main()
    w.resize(1600, 950)
    w.show()
    env["windows"].append(w)
    assert w.worker.wait()
    return w


def _section(w, title):
    return next(s for s in w.home.sections if s.title.text() == title)


def _card_texts(card) -> list[str]:
    return [lb.full_text() for lb in card.findChildren(ElidedLabel)]


# ---------------------------------------------------------------------------------------------- the rules

def test_hidden_folders():
    for name in ("_qc", "__pycache__", ".git", "pytest-12", "pytest-of-Nicolas", "tmp", "tmp3", "tmpabc", "temp",
                 "temp_2", "Temp-old", "heart_tmp", "M31_temp", "x.tmp"):
        assert home.hidden(name), name
    for name in ("Heart", "NGC7822", "Tempest", "Template", "heart_a7s_copy", "ours_express", "Rosette"):
        assert not home.hidden(name), name


def test_the_default_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    rows = home.section_rows()
    assert [(r["label"], r["folder"], r["open"], r["workspace"]) for r in rows] == [
        ("My projects", "F:/astro_projets", True, True), ("Test copies", "F:/astro_essais/v2", False, False)]
    assert all(r["note"] for r in rows)
    rows[0]["label"] = "changed"                                           # a copy: the defaults stay
    assert prefs.DEFAULTS["home_sections"][0]["label"] == "My projects"
    prefs.set("workspace", "D:/elsewhere")                                 # the old key still names "My projects"
    assert home.section_rows()[0]["folder"] == "D:/elsewhere"
    home.save_rows(home.section_rows())
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert "folder" not in saved["home_sections"][0] and saved["workspace"] == "D:/elsewhere"


def test_the_scan_finds_projects_two_levels_down_and_skips_hidden_ones(env):
    scan = home.scan_home(home.section_rows(), [])
    mine, copies = scan.sections
    assert sorted(i.name for i in mine.infos) == ["HeartSoul", "Iris", "Wizard"] and scan.n_ws == 3
    assert sorted(i.title for i in copies.infos) == ["heart_a7s", "heart_a7s_copy"]      # same name: the folders
    assert all(not i.title for i in mine.infos)
    missing = home.scan_home([{"label": "Gone", "folder": (env["tmp"] / "nowhere").as_posix()}], [])
    assert not missing.sections[0].found and not missing.sections[0].infos
    # the old one-folder scan keeps its rules
    infos, n = home.scan_workspace(env["ws"].as_posix(), [(env["ws"] / "Iris").as_posix()])
    assert n == 3 and infos[0].name == "Iris" and not [i for i in infos if i.name.startswith("hidden")]


# ---------------------------------------------------------------------------------------------- the view

def test_home_shows_the_sections_and_opens_test_copies_on_a_click(env, qapp):
    from astro_gpu.client.services.inspect import GuiDiskWatch
    prefs.add_recent((env["copies"] / "heart_a7s").as_posix(), "heart_a7s", "IC 1805")
    with GuiDiskWatch([env["ws"], env["copies"]]) as watch:
        next(f for f in (env["copies"] / "heart_a7s").iterdir() if f.is_file()).read_bytes()   # the watch sees it
        assert watch.count >= 1
        watch.hits.clear()
        watch.where.clear()
        w = _main(env)
        assert [s.title.text() for s in w.home.sections] == ["Recent projects", "My projects", "Test copies"]
        assert sorted(c.info.name for c in w.home.cards) == ["HeartSoul", "Iris", "Wizard"]
        assert w.home.count.text() == "3 projects in the workspace"
        recent = w.home.recent_cards[0]
        assert _card_texts(recent)[0] == "Test copies"                        # the section it comes from
        copies = _section(w, "Test copies")
        assert not copies.is_open and copies.body.isHidden() and copies.count.text() == "2 projects"
        copies.header.clicked.emit()
        w.worker.wait()
        assert copies.is_open and not copies.body.isHidden()
        assert sorted(c.title.full_text() for c in copies.cards) == ["heart_a7s", "heart_a7s_copy"]
    assert watch.count == 0, watch.where[:3]
    assert home.section_rows()[1]["open"]                                    # kept in the settings
    w.change_language("fr")                                                  # rebuilt: the opened section stays open
    assert w.worker.wait()
    assert _section(w, "Copies de test").is_open and _section(w, "Mes projets").count.text() == "3 projets"
    assert not i18n.missing(), sorted(i18n.missing())


def test_settings_edit_the_sections(env, qapp):
    changed = []
    d = SettingsDialog(None, MACHINE)
    d.homeChanged.connect(lambda: changed.append(1))
    d.show()
    d.show_page("folders")
    qapp.processEvents()
    assert len(d.sec_rows) == 2
    ws_row = d.sec_rows[0]
    assert not ws_row["folder"].isEnabled() and ws_row["folder"].text() == env["ws"].as_posix()
    removes = [b for b in d.pages.currentWidget().findChildren(QPushButton) if b.text() == "Remove"]
    assert len(removes) == 2 and not removes[0].isEnabled()
    add = next(b for b in d.pages.currentWidget().findChildren(QPushButton) if b.text() == "Add a section")
    add.click()
    qapp.processEvents()
    assert len(d.sec_rows) == 3 and len(home.section_rows()) == 3
    d.sec_rows[2]["name"].setText("More projects")
    d.sec_rows[2]["name"].editingFinished.emit()
    assert home.section_rows()[2]["label"] == "More projects" and changed
    d.sec_rows[2]["folder"].setText((env["tmp"] / "more").as_posix())
    d.sec_rows[2]["folder"].editingFinished.emit()
    assert home.section_rows()[2]["folder"] == (env["tmp"] / "more").as_posix()
    removes = [b for b in d.pages.currentWidget().findChildren(QPushButton) if b.text() == "Remove"]
    removes[2].click()
    qapp.processEvents()
    assert [r["label"] for r in home.section_rows()] == ["My projects", "Test copies"]
    # in French the default names show translated; left as they are, the English stays stored
    i18n.set_language("fr")
    d.rebuild()
    qapp.processEvents()
    assert d.sec_rows[0]["name"].text() == "Mes projets"
    d.sec_rows[1]["note"].editingFinished.emit()
    assert home.section_rows()[1]["label"] == "Test copies"
    assert all(isinstance(e, QLineEdit) for r in d.sec_rows for e in (r["name"], r["folder"], r["note"]))
    assert not i18n.missing(), sorted(i18n.missing())
    d.close()


def test_every_text_of_home_has_its_french():
    """Every text Home and Settings › Folders can show (their ``_()`` and ``plural()`` calls, the default sections) has
    a French translation, and no French text of Home is empty."""
    cat: dict[str, str] = {}
    for f in sorted((CLIENT / "locale" / "fr").glob("*.json")):
        cat.update(json.loads(f.read_text(encoding="utf-8")))
    checked: dict[str, int] = {}
    for name in ("home", "settingsdlg"):
        tree = ast.parse((CLIENT / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            args = node.args[:1] if node.func.id == "_" else node.args[1:3] if node.func.id == "plural" else []
            for a in args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    assert a.value in cat, (name, node.lineno, a.value)
                    checked[name] = checked.get(name, 0) + 1
    assert checked.get("home", 0) > 10 and checked.get("settingsdlg", 0) > 10, checked   # the scan found the texts
    shown = [r[k] for r in prefs.DEFAULTS["home_sections"] for k in ("label", "note")]
    assert not [t for t in shown if t not in cat]
    mine = json.loads((CLIENT / "locale" / "fr" / "home.json").read_text(encoding="utf-8"))
    assert mine and all(isinstance(v, str) and v for v in mine.values())
