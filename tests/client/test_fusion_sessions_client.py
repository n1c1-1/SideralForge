"""GUI exploration 2, the graph's structure: sessions joining a flow, a Fusion's channels and what it adds, names,
Image blocks without a file, a session's Calibration, panel fields going away; offscreen on a scratch project (nothing is
computed)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLineEdit, QPushButton, QWidget  # noqa: E402

from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.blocks import Graph, node_title  # noqa: E402
from astro_gpu.client.graph import GraphPage  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.client.widgets import clear  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

CHAIN = ["calibration", "registration", "stacking", "astrometry"]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def frames(folder: Path, names) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        (folder / n).write_bytes(b"\0" * (100 + i))
    return folder


def open_page(tmp_path, monkeypatch, filters=(), workflows=("Full",)):
    """One colour session and a template's workflow; or (``filters``) one session per filter, each its own flow."""
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    if filters:
        for i, f in enumerate(filters, 1):
            p.sessions.append(project2.Session(i, f, lights=[str(frames(tmp_path / "raw" / f, ["a1.fits"]))],
                                               filter=f))
        p.stack_mode = "per_session"
    else:
        p.sessions.append(project2.Session(1, "night 1", lights=[str(frames(tmp_path / "raw" / "n1", ["a1.ARW"]))]))
        for name in workflows:
            p.add_workflow(name, wfl.TEMPLATES[name])
    project2.save(p)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(120)
    m._sync()
    if filters:
        for i, f in enumerate(filters, 1):
            m.add_graph_workflow(f, [wfl.node(b) for b in CHAIN], sessions=[i])
        m._undo.clear()
    page = GraphPage(m)
    page.resize(1500, 800)
    page.toasts = []
    page.toast = page.toasts.append
    return w, m, page


def settle(w, page):
    w.wait(60)
    QApplication.processEvents()
    page.refresh()


def done(w, m, page):
    page.close()
    m.close(save=False)
    w.stop()


def blocks_of(m, name):
    return [n["block"] for n in m.graph(next(q for q in m.paths if q.name == name))]


def test_a_second_session_joins_the_chosen_flow_never_in_place_of_its_nights(qapp, tmp_path, monkeypatch):
    """Decision 1: double-click on the Session card (or a drop on "+ Session") adds the night to the chosen flow: the
    first night stays read, it says so; Ctrl+Z takes it back."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        full = m.paths[0]
        page.select(index=0, step="sources")
        page.palette_action("session")
        settle(w, page)
        assert [s.id for s in m.p.sessions_of(full.pid)] == [1, 2] and len(m.paths) == 1
        page.palette_action("session")                             # a third night: still all three
        settle(w, page)
        assert [s.id for s in m.p.sessions_of(full.pid)] == [1, 2, 3]
        m.undo()
        m.undo()
        assert [s.id for s in m.p.sessions] == [1]
    finally:
        done(w, m, page)


def test_a_session_joining_a_filter_flow_takes_its_filter_and_a_fusion_chosen_starts_a_flow(qapp, tmp_path,
                                                                                             monkeypatch):
    """Decision 1: Ha chosen, the new session joins Ha with the Ha filter (OIII untouched); a Fusion chosen: a flow of
    its own, said."""
    w, m, page = open_page(tmp_path, monkeypatch, filters=("Ha", "OIII"))
    try:
        ha, oiii = (next(q for q in m.paths if q.name == n) for n in ("Ha", "OIII"))
        page.select(index=m.paths.index(ha), step=f"session@{m.sessions_of(ha)[0]}")
        page.palette_action("session")
        settle(w, page)
        new = max(s.id for s in m.p.sessions)
        assert m.sessions_of(m.view(ha.pid)) == [1, new] and m.p.session(new).filter == "Ha"
        assert m.sessions_of(m.view(oiii.pid)) == [2]
        assert any("joins" in t and "Ha" in t for t in page.toasts)
        page.place_fusion()
        settle(w, page)
        fus = next(q for q in m.paths if (q.source or {}).get("fusion"))
        page.select(index=m.paths.index(fus), step="fusion")
        page.palette_action("session")
        settle(w, page)
        new2 = max(s.id for s in m.p.sessions)
        mine = [q for q in m.paths if q.source is None and m.sessions_of(q) == [new2]]
        assert len(mine) == 1 and "of its own" in page.toasts[-1]
        assert m.sessions_of(m.view(ha.pid)) == [1, new]                       # nothing else changed
    finally:
        done(w, m, page)


def test_a_fusion_is_greyed_until_its_channels_have_astrometry_and_says_what_it_adds(qapp, tmp_path, monkeypatch):
    """Decision 2: the card says what is missing in each channel; placed, the toast names the blocks it added and
    where; removed, the channels come back as they were; a second Fusion is a block of its own."""
    w, m, page = open_page(tmp_path, monkeypatch, filters=("Ha", "OIII"))
    try:
        for q in list(m.paths):
            m.remove_node(q, "astrometrie")
        why = page.palette_reason("fusion", m.path, None)
        assert "Astrometry" in why and "Ha" in why and "OIII" in why
        n = len(m.paths)
        page.place_fusion()                                       # refused, with the same reason
        assert len(m.paths) == n and "Astrometry" in page.toasts[-1]
        for q in list(m.paths):
            m.insert_block(m.view(q.pid), "astrometry")
        before = {n: blocks_of(m, n) for n in ("Ha", "OIII")}
        assert page.palette_reason("fusion", m.path, None) == ""
        page.place_fusion()
        settle(w, page)
        assert "Denoise" in page.toasts[-1] and "Ha, OIII" in page.toasts[-1] and "Ctrl+Z" in page.toasts[-1]
        assert all("stars" in blocks_of(m, n) for n in ("Ha", "OIII"))
        page.place_fusion()                                       # a second Fusion: its own block
        settle(w, page)
        fusions = [q for q in m.paths if (q.source or {}).get("fusion")]
        g = Graph(m)
        assert len(fusions) == 2 and len({id(g.node_of(q, "fusion")) for q in fusions}) == 2
        assert [node_title(m, g.node_of(q, "fusion")) for q in fusions] == ["Fusion 1", "Fusion 2"]
        page.remove_tool(g.node_of(fusions[0], "fusion"), None)   # the first goes: the second still needs the stars
        settle(w, page)
        assert all("stars" in blocks_of(m, n) for n in ("Ha", "OIII"))
        page.remove_tool(Graph(m).node_of(m.view(fusions[1].pid), "fusion"), None)
        settle(w, page)
        assert {n: blocks_of(m, n) for n in ("Ha", "OIII")} == before            # as before the first Fusion
        assert not [n for n in Graph(m).nodes if n.key == "composition"]
    finally:
        done(w, m, page)


def test_a_fusion_channel_keeps_what_the_fusion_needs(qapp, tmp_path, monkeypatch):
    """Decision 3: a channel's Astrometry cannot be deleted nor switched off, its workflow cannot be deleted (no
    Python error on Ctrl+Y any more); its session set to another filter: the Fusion red with the reason."""
    w, m, page = open_page(tmp_path, monkeypatch, filters=("Ha", "OIII"))
    try:
        page.place_fusion()
        settle(w, page)
        hoo = next(q for q in m.paths if (q.source or {}).get("fusion"))
        ha = next(q for q in m.paths if q.name == "Ha")
        assert hoo.name in m.removal_problem(ha, "astrometrie") and hoo.name in m.can_switch_off(ha, "astrometrie")
        node = Graph(m).node_of(ha, "astrometrie")
        page.remove_tool(node, None)
        assert "astrometry" in blocks_of(m, "Ha") and hoo.name in page.toasts[-1]
        n = len(m._undo)
        page.workflow_action("delete", ha.pid)
        settle(w, page)
        assert m.view(ha.pid) is not None and len(m._undo) == n and "kept" in page.toasts[-1]
        assert m.redo() is False                                  # nothing to redo, nothing broken
        m.p.validate()
        assert m.fusion_problem(m.view(hoo.pid)) == ""
        m.set_session_filter(1, "SII")
        assert "SII" in m.fusion_problem(m.view(hoo.pid)) and m.run_problems()
        assert Graph(m).node_of(m.view(hoo.pid), "fusion").broken
        m.undo()
        assert m.fusion_problem(m.view(hoo.pid)) == ""
        # round 2: every channel that lacks something is named at once (it said Ha only, then OIII once fixed); Run to
        # here on a channel is not refused for the fusion it does not compute
        m.set_session_filter(1, "SII")
        m.set_session_filter(2, "SII")
        why = m.fusion_problem(m.view(hoo.pid))
        assert why == "Ha, OIII: session filter SII", why
        assert m.run_problems({hoo.pid}) and not [x for x in m.run_problems({ha.pid}) if x[0] == hoo.name]
    finally:
        done(w, m, page)


def test_two_images_without_a_file_are_two_blocks(qapp, tmp_path, monkeypatch):
    """Decision 4: each Image placed is its own block, removed alone."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.palette_action("image")
        page.palette_action("image")
        settle(w, page)
        imgs = [q for q in m.paths if q.source is not None]
        g = Graph(m)
        nodes = [g.node_of(q, "image") for q in imgs]
        assert len(imgs) == 2 and nodes[0] is not nodes[1]
        page.remove_tool(nodes[0], None)
        settle(w, page)
        assert [q.pid for q in m.paths if q.source is not None] == [imgs[1].pid]
    finally:
        done(w, m, page)


def test_names_never_empty_nor_another_s(qapp, tmp_path, monkeypatch):
    """Decision 8: a session or a workflow cannot take an empty name or another's: refused with a short message."""
    w, m, page = open_page(tmp_path, monkeypatch, filters=("Ha", "OIII"))
    try:
        ha, oiii = (next(q for q in m.paths if q.name == n) for n in ("Ha", "OIII"))
        assert m.name_problem("OIII", sid=1) and m.name_problem("  ", sid=1) and not m.name_problem("Ha 2", sid=1)
        assert not m.rename_session(1, "OIII") and m.p.session(1).name == "Ha"
        page.rename(ha.pid, "OIII")
        page.rename(ha.pid, "")
        assert m.view(ha.pid).name == "Ha" and len(page.toasts) == 2 and "refused" in page.toasts[0]
        assert not m.rename(ha.pid, "OIII") and m.rename(ha.pid, "H-alpha") and m.view(ha.pid).name == "H-alpha"
        page.select(index=m.paths.index(m.view(oiii.pid)), step="session@2")
        page.refresh()
        ed = next(e for e in page.panel.findChildren(QLineEdit) if e.text() == "OIII")
        ed.setText("Ha")
        ed.editingFinished.emit()
        assert m.p.session(2).name == "OIII" and ed.text() == "OIII" and "refused" in page.toasts[-1]
    finally:
        done(w, m, page)


def test_removing_a_flow_or_its_session_says_what_goes(qapp, tmp_path, monkeypatch):
    """Two flows: the Session removed takes its flow, and says so; Delete workflow takes the Session only it read (no
    Session left unwired, its frames said used); Ctrl+Z brings both back."""
    w, m, page = open_page(tmp_path, monkeypatch, filters=("Ha", "OIII"))
    try:
        page.session_action("remove", 1)
        settle(w, page)
        assert [q.name for q in m.paths] == ["OIII"] and "workflow" in page.toasts[-1] and "Ha" in page.toasts[-1]
        m.undo()
        settle(w, page)
        oiii = next(q for q in m.paths if q.name == "OIII")
        page.workflow_action("delete", oiii.pid)
        settle(w, page)
        assert [s.id for s in m.p.sessions] == [1] and "OIII" in page.toasts[-1]
        m.undo()
        assert [s.id for s in m.p.sessions] == [1, 2] and len(m.paths) == 2
    finally:
        done(w, m, page)


def test_a_session_calibration_shows_its_scope_and_has_its_buttons(qapp, tmp_path, monkeypatch):
    """One Calibration for every session: switched off, every session's block looks so and the message says "all
    sessions"; its panel has Delete and Switch off; a pause that cannot be kept is never announced. The Session panel
    has its Delete (greyed with the reason when it cannot go)."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.select(index=0, step="sources")
        page.palette_action("session")
        settle(w, page)
        g = Graph(m)
        c1 = g.node_of(m.paths[0], "calibration@1")
        page.select_node(c1)
        page.refresh()
        texts = [b.text() for b in page.panel.findChildren(QPushButton)]
        assert "Delete" in texts and "Switch off" in texts
        page.toggle_cut(c1, None)
        settle(w, page)
        g = Graph(m)
        assert all(cn.cut for _s, cn in g.sessions) and "all sessions" in page.toasts[-1]
        page.toggle_pause(g.sessions[0][1])                      # announced only when it is kept, else said why
        kept = bool(m.p.workflow(m.paths[0].pid).breakpoints)
        assert ("Review pause" in page.toasts[-1]) == kept and (kept or "Registration" in page.toasts[-1])
        page.select_node(g.sessions[1][0])
        page.refresh()
        x = next(b for b in page.panel.findChildren(QPushButton) if b.text() == "Delete")
        assert x.isEnabled()
    finally:
        done(w, m, page)


def test_a_field_going_away_writes_nothing(qapp):
    """A panel's field hidden as its block goes (Delete, Ctrl+Z) does not say "editing finished" any more: it would
    write into a block, session or workflow that is gone."""
    host = QWidget()
    lay = QHBoxLayout(host)
    box = QWidget()
    inner = QHBoxLayout(box)
    ed, ed2 = QLineEdit("x"), QLineEdit("y")
    inner.addWidget(ed2)
    lay.addWidget(ed)
    lay.addWidget(box)
    host.show()
    ed.setFocus()
    calls = []
    ed.editingFinished.connect(lambda: calls.append(1))
    ed2.editingFinished.connect(lambda: calls.append(2))
    clear(lay)
    QApplication.processEvents()
    assert calls == []
    host.close()
