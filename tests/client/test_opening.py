"""#115: what a project's opening does once, or not before it is shown (counted, never timed)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import graph as graphmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402
from astro_gpu.client.services import inspect as inspectmod  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.workflow import TEMPLATES  # noqa: E402

MACHINE = Machine([Gpu("cuda:0", "RTX", 16.0)], 16, 64.0)


@pytest.fixture(scope="session")
def qapp():
    from astro_gpu.client.widgets import QSS
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return app


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    prefs.load()
    prefs.set("home_sections", [])
    prefs.set("cache_dir", (tmp_path / "scratch").as_posix())
    appmod.apply_language("en")
    monkeypatch.setattr(appmod, "machine_info", lambda: MACHINE)
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, **k: None)
    windows = []
    yield {"tmp": tmp_path, "windows": windows, "mp": monkeypatch}
    for w in windows:
        w.close()
        w.deleteLater()
    QApplication.processEvents()


def _project(tmp: Path) -> Path:
    """One session with two light files (never read: their headers are what is counted), the Full template."""
    root = tmp / "ws" / "Opened"
    lights = tmp / "lights"
    lights.mkdir(parents=True)
    for n in ("a", "b"):
        (lights / f"{n}.fits").write_bytes(b"")
    p = project2.create(root, "Opened", Target(name="M 31"))
    p.sessions.append(project2.Session(1, "night", lights=[lights.as_posix()]))
    p.add_workflow("Full", TEMPLATES["Full"])
    project2.save(p)
    return root


def _settle(w, n=10):
    for _i in range(n):
        QApplication.processEvents()
        w.worker.wait(30)
    QApplication.processEvents()


def _open(env, root):
    w = appmod.Main()
    w.resize(1600, 900)
    w.show()
    env["windows"].append(w)
    w.worker.wait()
    w.open_project(root)
    w.worker.wait()
    _settle(w)
    return w


def test_the_palette_is_built_again_only_when_what_it_shows_changed(env):
    made = []

    class Card(graphmod.PaletteCard):
        def __init__(self, *a, **k):
            made.append(a[1])
            super().__init__(*a, **k)
    env["mp"].setattr(graphmod, "PaletteCard", Card)
    w = _open(env, _project(env["tmp"]))
    page = w.graph
    assert page.cards and len(made) <= 2 * len(page.cards)        # blank, then the project read: never once a refresh
    cards = dict(page.cards)
    made.clear()
    page.refresh()
    page.refresh()
    _settle(w)
    assert made == [] and page.cards == cards                    # the same cards, nothing built
    page.set_layout("docked", "folded")                           # what it shows changed: built again
    assert len(made) == len(page.cards) and page.cards["denoise"] is not cards["denoise"]


def test_the_frames_are_read_when_inspect_shows_them_not_as_the_project_opens(env):
    asked = []
    real = inspectmod.InspectService.request_group

    def request_group(self, *a, **k):
        asked.append(a)
        return real(self, *a, **k)
    env["mp"].setattr(inspectmod.InspectService, "request_group", request_group)
    w = _open(env, _project(env["tmp"]))
    assert asked == [] and w.frames.pages.currentIndex() == 1     # the Files page, waiting to be shown
    w.select(step="sources")
    w.set_mode(1)
    _settle(w)
    assert asked and w.browser.pages.currentIndex() == 1          # shown: read, its session's page


def test_the_start_loads_the_stage_graph_and_the_names_while_home_is_shown(monkeypatch):
    """The first opening needed every stage's module (the keys) and the catalogue's names (the Project block, in the GUI
    thread): the start's warm-up loads them first, then what the first render needs."""
    from astro_gpu.core import determinism, skyobjects, store
    done = []
    monkeypatch.setattr(store, "key_rules", lambda: done.append("stages"))
    monkeypatch.setattr(skyobjects, "names_index", lambda: done.append("names"))
    monkeypatch.setattr(determinism, "warm_up", lambda: done.append("render"))
    appmod._warm_up()
    assert done == ["stages", "names", "render"]


def test_home_stops_reading_the_other_projects_states_when_a_project_is_shown(env):
    """The states of Home's cards (the keys of every frame of every recent project) held the one worker before the
    project opened could be read: left for a project, Home reads no further project; shown again, it reads them anew."""
    from astro_gpu.client import home as homemod
    read = []
    env["mp"].setattr(homemod, "project_state", lambda r: (read.append(r), ("Never run", {}))[1])
    stop = homemod.threading.Event()
    assert homemod.project_states(["a", "b"], stop) == [("a", "Never run", {}), ("b", "Never run", {})]
    stop.set()
    read.clear()
    assert homemod.project_states(["a", "b"], stop) == [] and read == []
    w = appmod.Main()
    w.resize(1600, 900)
    w.show()
    env["windows"].append(w)
    w.worker.wait()
    _settle(w)
    asked = w.home._stop
    assert not asked.is_set()
    w.new_project()                                               # the project in Home's place
    assert asked.is_set()
    w.worker.wait()
    w.close_project()                                             # Home again: its own reads start again
    _settle(w)
    assert w.screens.currentWidget() is w.home
    from types import SimpleNamespace
    sent = []
    env["mp"].setattr(w.worker, "submit", lambda channel, fn, *a: sent.append((channel, a)))
    w.home.recent_cards = [SimpleNamespace(info=SimpleNamespace(root="r", state="workflows"))]
    w.home._ask_states()
    assert sent[0][0] == "home.states" and sent[0][1][0] == ["r"] and not sent[0][1][1].is_set()
