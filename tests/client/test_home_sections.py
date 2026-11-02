"""Home in sections, offscreen: "My projects" (the workspace), "Bench against Siril" (projects two levels down, cards
that say the case, instrument, level and Siril), "Test copies" (closed until its header is clicked); hidden folders;
Settings › Folders edits the sections; every text has its French; no disk access in the GUI thread.

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


def _case(bench: Path, case: str, levels: dict, manifest: dict, siril=("express",), run_when="2026-11-01T10:17:16"):
    d = bench / case
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"case": case, "levels": levels, **manifest}), encoding="utf-8")
    for lv in levels:
        if levels[lv].get("same_as"):
            continue
        _project(d / f"ours_{lv}", f"bench {case} {lv}", manifest.get("target", ""))
        runs = [{"status": "done", "when": run_when,
                 "metrics": {"inventory": {"metrics": {"n_lights": levels[lv]["count"]}}}}]
        (d / f"ours_{lv}" / "bench_run.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    for lv in siril:
        (d / f"siril_{lv}").mkdir()
    (d / "data" / "lights_express").mkdir(parents=True)


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    """A scratch workspace, bench and test copies; the sections point at them; English; a fixed machine."""
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    prefs.load()
    ws, bench, copies = tmp_path / "ws", tmp_path / "bench", tmp_path / "copies"
    for d in (ws, bench, copies):
        d.mkdir()
    _project(ws / "Heart", "HeartSoul", "IC 1805")
    _project(ws / "Iris", "Iris", "NGC 7023")
    for hidden in ("_qc", "pytest-12", "tmp3", "temp", "Heart_tmp", ".git"):
        _project(ws / hidden, f"hidden {hidden}")
    _project(ws / "Old" / "Wizard", "Wizard", "NGC 7380")                 # a project one level down
    _case(bench, "heart", {"express": {"count": 30}, "base": {"count": 100}},
          {"target": "IC 1805", "camera_label": "Sony A7S (astro-modified), lossy ARW",
           "scope": "TS61EDPH refractor + 0.75x reducer (272 mm)"})
    _case(bench, "m8", {"express": {"count": 15}, "base": {"count": 15, "same_as": "express"}},
          {"target": "M 8"}, siril=("express",))                          # no camera in the manifest: the fallback
    _case(bench, "seestar", {"express": {"count": 30}, "base": {"count": 100}},
          {"target": "M 31", "camera_label": "ZWO Seestar S50 (FITS, IRCUT filter)",
           "scope": "Seestar S50 (250 mm, alt-azimuth)"}, siril=())
    (bench / "_logs").mkdir()
    for n in ("heart_a7s", "heart_a7s_p10"):
        _project(copies / n, "heart_a7s", "IC 1805", "The reference Heart")
    prefs.set("workspace", ws.as_posix())
    prefs.set("cache_dir", (tmp_path / "scratch").as_posix())
    rows = home.section_rows()
    rows[1]["folder"], rows[2]["folder"] = bench.as_posix(), copies.as_posix()
    home.save_rows(rows)
    appmod.apply_language("en")
    monkeypatch.setattr(appmod, "machine_info", lambda: MACHINE)
    windows = []
    yield {"ws": ws, "bench": bench, "copies": copies, "tmp": tmp_path, "windows": windows}
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
    for name in ("Heart", "NGC7822", "Tempest", "Template", "heart_a7s_p10", "ours_express", "Rosette"):
        assert not home.hidden(name), name


def test_the_three_default_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    rows = home.section_rows()
    assert [(r["label"], r["folder"], r["open"], r["workspace"]) for r in rows] == [
        ("My projects", "F:/astro_projets", True, True), ("Bench against Siril", "F:/astro_essais/bench", True, False),
        ("Test copies", "F:/astro_essais/v2", False, False)]
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
    mine, bench, copies = scan.sections
    assert sorted(i.name for i in mine.infos) == ["HeartSoul", "Iris", "Wizard"] and scan.n_ws == 3
    assert [(i.bench.case, i.bench.level) for i in bench.infos] == [
        ("heart", "express"), ("heart", "base"), ("m8", "express"), ("seestar", "express"), ("seestar", "base")]
    assert sorted(i.title for i in copies.infos) == ["heart_a7s", "heart_a7s_p10"]      # same name: the folders
    assert all(not i.title for i in mine.infos)
    missing = home.scan_home([{"label": "Gone", "folder": (env["tmp"] / "nowhere").as_posix()}], [])
    assert not missing.sections[0].found and not missing.sections[0].infos
    # the old one-folder scan keeps its rules
    infos, n = home.scan_workspace(env["ws"].as_posix(), [(env["ws"] / "Iris").as_posix()])
    assert n == 3 and infos[0].name == "Iris" and not [i for i in infos if i.name.startswith("hidden")]


def test_a_bench_card_says_case_instrument_level_and_siril(env):
    b = home.bench_info(env["bench"] / "heart" / "ours_express")
    assert (b.case, b.level, b.count, b.siril) == ("heart", "express", 30, True)
    assert home.bench_title(b) == "Heart · IC 1805"
    assert home.bench_instrument(b) == "Sony A7S + TS61EDPH refractor"
    assert home.bench_level(b) == "Express, 30 frames · 2026-11-01"
    base = home.bench_info(env["bench"] / "heart" / "ours_base")
    assert base.count == 100 and not base.siril and home.bench_level(base).startswith("Base, 100 frames")
    m8 = home.bench_info(env["bench"] / "m8" / "ours_express")              # from the fallback table
    assert home.bench_instrument(m8) == "ASI2600MC" and home.bench_title(m8) == "M8 · Siril tutorial frames"
    s = home.bench_info(env["bench"] / "seestar" / "ours_express")
    assert home.bench_instrument(s) == "Seestar S50" and home.bench_title(s) == "Seestar · M 31"
    assert home.bench_info(env["ws"] / "Heart") is None
    tip = home.card_tooltip(home._read(env["bench"] / "heart" / "ours_express"))
    assert "Sony A7S (astro-modified), lossy ARW" in tip and "Last bench run: 2026-11-01 10:17" in tip
    appmod.apply_language("fr")
    assert home.bench_level(b) == "Express, 30 images · 2026-11-01"
    assert home.bench_title(m8) == "M8 · images du tutoriel Siril"


# ---------------------------------------------------------------------------------------------- the view

def test_cards_alone_take_the_next_column_and_cases_start_on_their_columns(qapp):
    """A section with projects of no case (the bench's v2x_express folders) before the cases (Express, Base): the
    cards alone sit side by side, without a free column between them (2026-11-02: they took every other column); a
    case still starts on an even column."""
    from PySide6.QtWidgets import QWidget
    sv = home.SectionView("Bench")
    cards = [QWidget() for _ in range(7)]
    sv.set_cards(cards, groups=[[cards[0]], [cards[1]], [cards[2]], [cards[3], cards[4]], [cards[5], cards[6]]])
    sv.place(4, force=True)
    pos = [sv.grid.getItemPosition(sv.grid.indexOf(c))[:2] for c in cards]
    assert pos[:3] == [(0, 0), (0, 1), (0, 2)]
    assert pos[3:5] == [(1, 0), (1, 1)] and pos[5:] == [(1, 2), (1, 3)]
    sv.place(5, force=True)                                   # five columns: still a case on even columns
    pos = [sv.grid.getItemPosition(sv.grid.indexOf(c))[:2] for c in cards]
    assert pos[:3] == [(0, 0), (0, 1), (0, 2)] and pos[3:5] == [(1, 0), (1, 1)] and pos[5:] == [(1, 2), (1, 3)]


def test_home_shows_the_sections_and_opens_test_copies_on_a_click(env, qapp):
    from astro_gpu.client.services.inspect import GuiDiskWatch
    prefs.add_recent((env["bench"] / "heart" / "ours_express").as_posix(), "bench heart express", "IC 1805")
    with GuiDiskWatch([env["ws"], env["bench"], env["copies"]]) as watch:
        (env["bench"] / "heart" / "manifest.json").read_bytes()               # the watch sees the GUI thread
        assert watch.count == 1
        watch.hits.clear()
        watch.where.clear()
        w = _main(env)
        assert [s.title.text() for s in w.home.sections] == ["Recent projects", "My projects", "Bench against Siril",
                                                             "Test copies"]
        assert sorted(c.info.name for c in w.home.cards) == ["HeartSoul", "Iris", "Wizard"]
        assert w.home.count.text() == "3 projects in the workspace"
        recent = w.home.recent_cards[0]
        assert _card_texts(recent)[0] == "Bench against Siril"                # the section it comes from
        bench = _section(w, "Bench against Siril")
        assert bench.is_open and bench.count.text() == "5 projects" and bench.note.text().startswith("Test sets")
        texts = _card_texts(bench.cards[0])
        assert texts[:5] == ["Heart · IC 1805", "Sony A7S + TS61EDPH refractor", "Express, 30 frames · 2026-11-01",
                             "Compared with Siril", "No workflow yet"]
        assert "Not compared with Siril yet" in _card_texts(bench.cards[1])
        # the levels of a case side by side: Express in the even columns
        pos = {(c.info.bench.case, c.info.bench.level): bench.grid.getItemPosition(bench.grid.indexOf(c))[:2]
               for c in bench.cards}
        assert pos[("heart", "base")] == (pos[("heart", "express")][0], pos[("heart", "express")][1] + 1)
        assert all(col % 2 == 0 for (case, lv), (_r, col) in pos.items() if lv == "express")
        copies = _section(w, "Test copies")
        assert not copies.is_open and copies.body.isHidden() and copies.count.text() == "2 projects"
        copies.header.clicked.emit()
        w.worker.wait()
        assert copies.is_open and not copies.body.isHidden()
        assert sorted(c.title.full_text() for c in copies.cards) == ["heart_a7s", "heart_a7s_p10"]
    assert watch.count == 0, watch.where[:3]
    assert home.section_rows()[2]["open"]                                    # kept in the settings
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
    assert len(d.sec_rows) == 3
    ws_row = d.sec_rows[0]
    assert not ws_row["folder"].isEnabled() and ws_row["folder"].text() == env["ws"].as_posix()
    removes = [b for b in d.pages.currentWidget().findChildren(QPushButton) if b.text() == "Remove"]
    assert len(removes) == 3 and not removes[0].isEnabled()
    d.sec_rows[1]["name"].setText("Siril bench")
    d.sec_rows[1]["name"].editingFinished.emit()
    assert home.section_rows()[1]["label"] == "Siril bench" and changed
    add = next(b for b in d.pages.currentWidget().findChildren(QPushButton) if b.text() == "Add a section")
    add.click()
    qapp.processEvents()
    assert len(d.sec_rows) == 4 and len(home.section_rows()) == 4
    d.sec_rows[3]["folder"].setText((env["tmp"] / "more").as_posix())
    d.sec_rows[3]["folder"].editingFinished.emit()
    assert home.section_rows()[3]["folder"] == (env["tmp"] / "more").as_posix()
    removes = [b for b in d.pages.currentWidget().findChildren(QPushButton) if b.text() == "Remove"]
    removes[3].click()
    qapp.processEvents()
    assert [r["label"] for r in home.section_rows()] == ["My projects", "Siril bench", "Test copies"]
    # in French the default names show translated; left as they are, the English stays stored
    i18n.set_language("fr")
    d.rebuild()
    qapp.processEvents()
    assert d.sec_rows[0]["name"].text() == "Mes projets"
    d.sec_rows[2]["note"].editingFinished.emit()
    assert home.section_rows()[2]["label"] == "Test copies"
    assert all(isinstance(e, QLineEdit) for r in d.sec_rows for e in (r["name"], r["folder"], r["note"]))
    assert not i18n.missing(), sorted(i18n.missing())
    d.close()


def test_every_text_of_home_has_its_french():
    cat: dict[str, str] = {}
    for f in sorted((CLIENT / "locale" / "fr").glob("*.json")):
        cat.update(json.loads(f.read_text(encoding="utf-8")))
    for name in ("home", "settingsdlg"):
        tree = ast.parse((CLIENT / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            args = node.args[:1] if node.func.id == "_" else node.args[1:3] if node.func.id == "plural" else []
            for a in args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    assert a.value in cat, (name, node.lineno, a.value)
    shown = [r[k] for r in prefs.DEFAULTS["home_sections"] for k in ("label", "note")]
    shown += list(home.LEVELS.values()) + [v["about"] for v in home.BENCH_FALLBACK.values() if "about" in v]
    assert not [t for t in shown if t not in cat]
    mine = json.loads((CLIENT / "locale" / "fr" / "home.json").read_text(encoding="utf-8"))
    assert len(mine) > 20 and all(isinstance(v, str) and v for v in mine.values())
