"""Frame blocks placed in parallel in the graph: a palette card that cannot go in the chain still drags for a new branch,
a Calibration beside the Session's with its own files. A scratch project whose frames are empty files: nothing computed."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QListWidget, QPushButton  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.blocks import Graph, node_title  # noqa: E402
from astro_gpu.client.graph import GraphPage  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.panel import Panel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

CHAIN = [wfl.node(b) for b in ("calibration", "registration", "selection", "stacking", "astrometry")]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def frames(folder: Path, names) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        (folder / n).write_bytes(b"\0" * (100 + i))
    return folder


def open_project(tmp_path, monkeypatch, nights):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    i18n.set_language("en")
    raw = tmp_path / "raw"
    darks, flats = frames(raw / "darks", ["d1.ARW", "d2.ARW"]), frames(raw / "flats", ["f1.ARW"])
    frames(raw / "other", ["o1.ARW", "o2.ARW"])
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    p.stack_mode = "per_session"
    for i in range(1, nights + 1):
        p.sessions.append(project2.Session(i, f"night {i}", lights=[str(frames(raw / f"n{i}", [f"a{i}.ARW", f"b{i}.ARW"]))],
                                           darks=[str(darks)], flats=[str(flats)]))
    w = p.add_workflow("Main", wfl.DEFAULT_OPTIONS)
    p.set_blocks(w.id, CHAIN)
    project2.save(p)
    worker = ApiWorker()
    m = ProjectModel(p, worker)
    assert worker.wait(120)
    m._sync()
    page = GraphPage(m)
    page.resize(1500, 800)
    return m, worker, page, raw


@pytest.fixture()
def one(qapp, tmp_path, monkeypatch):
    m, worker, page, raw = open_project(tmp_path, monkeypatch, 1)
    settle(m, worker, page)
    yield m, worker, page, raw
    page.close()
    m.close(save=False)
    worker.stop()


@pytest.fixture()
def two(qapp, tmp_path, monkeypatch):
    m, worker, page, raw = open_project(tmp_path, monkeypatch, 2)
    settle(m, worker, page)
    yield m, worker, page, raw
    page.close()
    m.close(save=False)
    worker.stop()


def settle(m, worker, page):
    worker.wait(60)
    QApplication.processEvents()
    page.refresh()
    page.build_tools()


def ghosts(page, block):
    v = page.graph
    v.drag_block = block
    v.rebuild()
    out = {g.branch_key: g for g in v.ghosts if getattr(g, "branch_key", None)}
    v.drag_block = None
    return out


# ---------------------------------------------------------------------------------------------- the palette
def test_a_card_that_can_only_be_set_in_parallel_is_available_by_a_drag(one):
    m, _w, page, _raw = one
    for b in ("calibration", "registration", "selection", "stacking"):
        c = page.cards[b]
        assert c.reason and c.branch and c.cursor().shape() == Qt.OpenHandCursor, b
        assert "new branch" in c.toolTip(), b
    assert not page.cards["astrometry"].branch        # the same result: nothing to set beside it


def test_a_greyed_card_says_why_in_its_tooltip(one):
    _m, _w, page, _raw = one
    c = page.cards["stars"]
    assert c.reason and not c.branch and c.cursor().shape() == Qt.ForbiddenCursor
    assert c.toolTip() == "Needs Denoise first"
    assert not hasattr(page, "hint")


def test_a_double_click_on_a_card_only_for_a_branch_places_nothing(one):
    m, _w, page, _raw = one
    n = len(m.paths)
    page.cards["stacking"].mouseDoubleClickEvent(None)
    assert len(m.paths) == n and len(m.graph(m.paths[0])) == len(CHAIN)


def test_the_places_are_shown_and_the_drop_makes_the_fork(one):
    m, _w, page, _raw = one
    assert list(ghosts(page, "calibration")) == ["sources"]
    assert set(ghosts(page, "stacking")) == {"calibration", "alignement", "selection"}
    assert set(ghosts(page, "registration")) == {"calibration"}
    page.branch_block("stacking", m.paths[0].pid, "alignement")
    g = Graph(m)
    a = [n for n in g.nodes if n.key == "empilement"]
    assert len(a) == 2 and {n.num for n in a} == {1, 2}
    assert {node_title(m, n) for n in a} == {"Stacking 1", "Stacking 2"}
    assert next(n for n in g.nodes if n.key == "alignement").shared


def test_a_block_in_parallel_with_the_same_settings_computes_once_and_shows_on_its_own(one):
    m, worker, page, _raw = one
    page.branch_block("selection", m.paths[0].pid, "alignement")
    worker.wait(60)
    QApplication.processEvents()
    b, trunk = m.paths[1], m.paths[0]
    assert m.block_key(b, "alignement") == m.block_key(trunk, "alignement")
    assert m.block_key(b, "selection") != m.block_key(trunk, "selection")
    assert len([n for n in Graph(m).nodes if n.key == "selection"]) == 2


def test_settings_of_a_block_in_parallel_stay_its_own(one):
    m, worker, page, _raw = one
    page.branch_block("stacking", m.paths[0].pid, "alignement")
    worker.wait(60)
    QApplication.processEvents()
    a, b = m.paths[0], m.paths[1]
    assert m.stage_key(a, "stack") == m.stage_key(b, "stack")       # the same computation, once
    m.p.workflow(b.pid).params["stack"] = {"kappa": 3.0}
    from astro_gpu.core.store import Keys
    k = Keys(m.p)
    assert k.key(a.pid, "stack") != k.key(b.pid, "stack")
    assert k.key(a.pid, "register") == k.key(b.pid, "register")


# ---------------------------------------------------------------------------------------------- Calibration, one night
def test_a_second_calibration_forks_from_the_session(one):
    m, _w, page, _raw = one
    page.branch_block("calibration", m.paths[0].pid, "sources")
    q = m.paths[1]
    assert m.p.workflow(q.pid).calibration == {"1": {"fork": q.pid}}
    assert [n["block"] for n in m.graph(q)] == ["calibration"]
    g = Graph(m)
    cal = [n for n in g.nodes if n.key == "calibration"]
    assert len(cal) == 2 and all(n.parent is g.root for n in cal)
    assert [node_title(m, n) for n in cal] == ["Calibration 1", "Calibration 2"]
    assert "as Session" in g.summary(cal[1]) and "2 darks" in g.summary(cal[1])
    assert m.sel_step == "calibration"


def test_its_files_are_changed_file_by_file_and_given_back(one):
    m, worker, page, raw = one
    page.branch_block("calibration", m.paths[0].pid, "sources")
    worker.wait(60)
    QApplication.processEvents()
    q, trunk = m.paths[1], m.paths[0]
    assert m.calibration_follows(q, 1, "darks") and len(m.calibration_files(q, 1, "darks")) == 2
    old = m.calibration_files(q, 1, "darks")
    assert m.remove_calibration_files([q], 1, "darks", [old[0]])
    assert not m.calibration_follows(q, 1, "darks")
    assert m.calibration_files(q, 1, "darks") == [old[1]]
    assert len(m.session_files(1, "darks")) == 2          # the Session keeps its own
    assert m.add_calibration_files([q], 1, "darks", [str(raw / "other" / "o1.ARW")]) == 1
    assert m.add_calibration_files([q], 1, "darks", [str(raw / "other" / "o1.ARW")]) == 0
    from astro_gpu.core.store import Keys
    assert Keys(m.p).key(q.pid, "calibrate") != Keys(m.p).key(trunk.pid, "calibrate")
    assert "own files" in Graph(m).summary(next(n for n in Graph(m).nodes if n.key == "calibration" and q in n.paths))
    m.undo()
    assert len(m.calibration_files(m.paths[1], 1, "darks")) == 1
    m.follow_session([m.paths[1]], 1, "darks")
    assert m.calibration_follows(m.paths[1], 1, "darks")


def test_its_files_are_listed_in_inspect_and_the_graph_panel_opens_it(one):
    m, worker, page, raw = one
    page.branch_block("calibration", m.paths[0].pid, "sources")
    worker.wait(60)
    QApplication.processEvents()
    q = m.paths[1]
    node = next(n for n in Graph(m).nodes if n.key == "calibration" and q in n.paths)
    m.current = m.paths.index(q)
    panel = Panel(m, None, tune=True)
    panel.refresh(node)
    buttons = [b.text() for b in panel.findChildren(QPushButton)]
    assert buttons.count("Add files…") == 4
    darks = next(lst for lst in panel.findChildren(QListWidget) if lst.count() == 2)
    assert {Path(darks.item(i).data(Qt.UserRole)).name for i in range(2)} == {"d1.ARW", "d2.ARW"}
    side = Panel(m, None, tune=False)
    side.refresh(node)
    assert "Own frames…" in [b.text() for b in side.findChildren(QPushButton)]
    panel.close()
    side.close()


def test_removed_with_nothing_else_its_branch_goes_and_the_trunk_stays(one):
    m, _w, page, _raw = one
    trunk = m.paths[0]
    page.branch_block("calibration", trunk.pid, "sources")
    g = Graph(m)
    node = next(n for n in g.nodes if n.key == "calibration" and m.paths[1] in n.paths)
    page.remove_tool(node, None)
    assert [q.pid for q in m.paths] == [trunk.pid]
    assert m.graph(trunk) and m.graph(trunk)[0]["block"] == "calibration"


def test_a_branch_after_it_keeps_the_same_calibration_block(one):
    m, worker, page, _raw = one
    page.branch_block("calibration", m.paths[0].pid, "sources")
    worker.wait(60)
    QApplication.processEvents()
    q = m.paths[1]
    m.set_graph(q, m.graph(q) + [wfl.node("registration"), wfl.node("stacking")])
    worker.wait(60)
    QApplication.processEvents()
    q = m.paths[1]
    b, _k = m.add_branch(q, "alignement", "stacking")
    assert m.p.workflow(b.pid).calibration == m.p.workflow(q.pid).calibration
    worker.wait(60)
    QApplication.processEvents()
    own = next(n for n in Graph(m).nodes if n.key == "calibration" and q in n.paths)
    assert {x.pid for x in own.paths} == {q.pid, b.pid}


# ---------------------------------------------------------------------------------------------- several nights
def test_on_several_nights_it_is_placed_under_one_night_the_others_stay_shared(two):
    m, _w, page, _raw = two
    assert set(k for k in ghosts(page, "calibration")) == {"session@1", "session@2"}
    trunk = m.paths[0]
    page.branch_block("calibration", trunk.pid, "session@2")
    q = m.paths[1]
    assert m.p.workflow(q.pid).calibration == {"2": {"fork": q.pid}}
    assert m.sel_step == "calibration@2"
    g = Graph(m)
    pair = {sn.sid: cn for sn, cn in g.sessions}
    assert q in pair[1].paths and q not in pair[2].paths         # night 1 shared, night 2 its own
    fork = g.calib_forks[2][0]
    assert q in fork.paths and fork.sid == 2
    assert g.node_of(q, "calibration@2") is fork and g.node_of(trunk, "calibration@2") is pair[2]
    assert [node_title(m, n) for n in (pair[2], fork)] == ["Calibration 1", "Calibration 2"]


def test_the_rows_of_the_nights_make_room(two):
    m, _w, page, _raw = two
    page.branch_block("calibration", m.paths[0].pid, "session@1")
    page.refresh()
    g = page.graph.graph
    fork = g.calib_forks[1][0]
    night2 = next(sn for sn, _c in g.sessions if sn.sid == 2)
    assert fork.y > next(sn for sn, _c in g.sessions if sn.sid == 1).y
    assert night2.y >= fork.y + 54                       # not on the next night's row


def test_its_files_on_a_night_and_its_removal(two):
    m, worker, page, raw = two
    trunk = m.paths[0]
    page.branch_block("calibration", trunk.pid, "session@2")
    worker.wait(60)
    QApplication.processEvents()
    q = m.paths[1]
    assert m.add_calibration_files([q], 2, "flats", [str(raw / "other" / "o2.ARW")]) == 1
    assert [Path(f).name for f in m.calibration_files(q, 2, "flats")][-1] == "o2.ARW"
    assert m.calibration_follows(q, 1, "flats")
    node = Graph(m).node_of(q, "calibration@2")
    page.remove_tool(node, None)
    assert [p.pid for p in m.paths] == [trunk.pid]
    assert m.p.workflow(trunk.pid).calibration is None


def test_on_several_nights_a_frame_block_in_parallel_starts_after_their_calibrations(two):
    m, _w, page, _raw = two
    assert set(ghosts(page, "registration")) == {"calibration"}
    page.branch_block("registration", m.paths[0].pid, "calibration")
    q = m.paths[1]
    assert [n["block"] for n in m.graph(q)] == ["calibration", "registration"]
    a = [n for n in Graph(m).nodes if n.key == "alignement"]
    assert len(a) == 2 and {n.num for n in a} == {1, 2}


def test_a_branch_of_a_branch_is_named_from_the_flow(one):
    m, _w, page, _raw = one
    page.branch_block("stacking", m.paths[0].pid, "alignement")
    page.branch_block("selection", m.paths[1].pid, "alignement")
    assert m.paths[2].name == "Main · Selection"


def test_inspect_lists_the_own_files_of_the_series_shown(one):
    from astro_gpu.client.services.inspect import browse_group
    m, worker, page, raw = one
    page.branch_block("calibration", m.paths[0].pid, "sources")
    worker.wait(60)
    QApplication.processEvents()
    q = m.paths[1]
    m.add_calibration_files([q], 1, "darks", [str(raw / "other" / "o1.ARW")])
    m.remove_calibration_files([q], 1, "darks", m.session_files(1, "darks"))
    assert [it.name for it in browse_group(m.p, q.pid, "darks", {}, 1)] == ["o1.ARW"]
    assert sorted(it.name for it in browse_group(m.p, m.paths[0].pid, "darks", {}, 1)) == ["d1.ARW", "d2.ARW"]


def test_a_session_removed_takes_its_own_calibration_with_it(two):
    m, _w, page, _raw = two
    trunk = m.paths[0]
    page.branch_block("calibration", trunk.pid, "session@2")
    m.remove_session(2)
    assert all(not w.calibration for w in m.p.workflows)


def test_nights_calibrated_together_say_what_to_do(qapp, tmp_path, monkeypatch):
    m, worker, page, _raw = open_project(tmp_path, monkeypatch, 2)
    try:
        m.set_stack_mode("together")
        settle(m, worker, page)
        assert m.palette_reason(m.paths[0], "calibration") == "calibrate per night first"
        assert not m.can_branch(m.paths[0], "sources", "calibration")
        assert not page.cards["calibration"].branch
    finally:
        page.close()
        m.close(save=False)
        worker.stop()
