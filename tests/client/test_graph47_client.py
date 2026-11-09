"""Issue #47 in the client: workflows as free graphs built from a palette, guided placement, templates as shortcuts
(the blank one, the user's own: saved, exported, imported), offscreen on a scratch project (nothing is computed)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402
from astro_gpu.client.gallery import (  # noqa: E402
    TemplateGallery,
    export_template,
    import_template,
    make_template,
    save_user_template,
    template_blocks,
    user_templates,
)
from astro_gpu.client.graph import GraphPage, PaletteCard  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def frames(folder: Path, names) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        (folder / n).write_bytes(b"\0" * (100 + i))
    return folder


def open_page(tmp_path, monkeypatch, workflows=("Full",), lights=True):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    p.sessions.append(project2.Session(1, "night 1", lights=[str(frames(tmp_path / "raw" / "n1", ["a1.ARW"]))]
                                       if lights else []))
    for name in workflows:
        p.add_workflow(name, wfl.TEMPLATES[name])
    project2.save(p)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(120)
    m._sync()
    page = GraphPage(m)
    page.resize(1500, 800)
    return w, m, page


def settle(w, page):
    w.wait(60)
    QApplication.processEvents()
    page.refresh()


def test_the_palette_places_a_block_at_its_usual_place_and_twice(qapp, tmp_path, monkeypatch):
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        full = m.paths[0]
        cards = {c.block: c for c in page.findChildren(PaletteCard)}
        cards["denoise"].mouseDoubleClickEvent(None)              # a second Denoise, after the first
        settle(w, page)
        assert m.display_keys(m.view(full.pid))[-5:] == ["debruitage", "debruitage~2", "couleur", "etoiles", "post"]
        assert m.sel_step == "debruitage~2" and "denoise~2" in m.all_stages(m.view(full.pid))
        g = Graph(m)
        n2 = g.node_of(m.view(full.pid), "debruitage~2")
        assert n2 is not None and g.summary(n2) and "Denoise 2" in [it.title for it in page.graph.scene().items()
                                                                    if hasattr(it, "title")]
        assert page.panel is not None
        page.refresh()                                             # its panel: its own settings, its own stage
        cards = {c.block: c for c in page.findChildren(PaletteCard)}
        assert cards["stacking"].reason and cards["calibration"].reason       # once per workflow
        assert m.undo() and "debruitage~2" not in m.display_keys(m.view(full.pid))
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_blank_page_starts_from_a_session_and_builds_block_by_block(qapp, tmp_path, monkeypatch):
    """Issues #48, #49: a Session placed empty (its frames in its panel), a workflow built block by block, its Output
    made automatically once a block gives an image (hidden in the wire of its last block)."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        assert not m.paths
        cards = {c.block: c for c in page.findChildren(PaletteCard)}
        assert not cards["session"].reason and "output" not in cards
        page.palette_action("session")                             # placed empty, no dialog
        settle(w, page)
        assert len(m.p.sessions) == 2 and not m.p.sessions[-1].lights and len(m.paths) == 1
        assert m.p.workflows[0].blocks == [] and m.all_stages(m.paths[0]) == []     # nothing to run yet
        for b in ("calibration", "registration", "stacking"):
            page.place_block(b, m.paths[0].pid)
            settle(w, page)
        p = m.paths[0]
        assert [n["block"] for n in m.p.workflow(p.pid).blocks] == ["calibration", "registration", "stacking",
                                                                      "output"]      # the Output, automatic
        g = Graph(m)
        assert g.node_of(p, "post") is g.node_of(p, "empilement") and g.node_of(p, "empilement").handle
        page.place_block("gradient", p.pid)
        settle(w, page)
        assert m.all_stages(p)[-2:] == ["gradient", "render_simple"]
        toasts = []
        page.toasted.connect(toasts.append)
        page.place_block("color", p.pid)                           # Color without Astrometry: refused, said why
        assert toasts and "Astrometry" in toasts[-1]
        cards = {c.block: c for c in page.findChildren(PaletteCard)}
        assert "Astrometry" in cards["color"].reason
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_workflow_saved_as_a_template_comes_back_in_the_gallery_and_as_a_file(qapp, tmp_path, monkeypatch):
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        full = m.paths[0]
        m.insert_block(full, "denoise", None)
        g = m.graph(m.view(full.pid))
        t = save_user_template(make_template("Two denoise", "mine", "mine", "frames", g))
        assert [x["key"] for x in user_templates()] == [t["key"]]
        d = TemplateGallery(m)
        assert "mine" in d.tab_keys
        d.tabs.setCurrentIndex(d.tab_keys.index("mine"))
        mine = next(c.t for c in d.cards if c.t["key"] == t["key"])
        d.pick(mine)
        placed = lambda blocks: [n for n in blocks if n["block"] != "output"]      # noqa: E731 (#49: automatic)
        assert d.export_b.isVisibleTo(d) and placed(template_blocks(mine)) == g
        f = tmp_path / "two.sftemplate"
        export_template(mine, str(f))
        prefs.set("templates", [])
        assert not user_templates()
        back = import_template(str(f))
        assert placed(back["blocks"]) == g and [x["name"] for x in user_templates()] == ["Two denoise"]
        d.pick(next(c.t for c in d.cards if c.t["key"] == t["key"]))
        d.ok.click()                                               # Create: a workflow of that graph
        page.make_workflow(d)
        settle(w, page)
        assert m.path.name.startswith("Two denoise") and "debruitage~2" in m.display_keys(m.path)
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_filters_template_on_a_new_project_makes_its_empty_sessions(qapp, tmp_path, monkeypatch):
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        d = page.new_workflow()
        d.pick(next(c.t for c in d.cards if c.t["key"] == "filters_full"))
        d.palette.setCurrentText("HOO")
        assert d.ok.isEnabled()
        d.ok.click()
        settle(w, page)
        assert sorted(s.filter for s in m.p.sessions if s.filter) == ["Ha", "OIII"]
        assert [q.name for q in m.paths][-1] == "HOO" and all("couleur" not in m.display_keys(q)
                                                               for q in m.paths if m.is_mono(q))
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_what_the_play_test_found_stays_fixed(qapp, tmp_path, monkeypatch):
    """The play-test of issue #47: an empty workflow from sessions, a template that takes its place, no Color in
    monochrome, a refused delete keeps the selection, a fusion's channels are not Post layers by themselves."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        page.place_block("calibration")                            # sessions, no workflow: the block starts one
        settle(w, page)
        assert len(m.paths) == 1 and [n["block"] for n in m.p.workflows[0].blocks] == ["calibration"]
        d = page.new_workflow()
        assert d.choice["key"] == "colour_full"
        d.pick(next(c.t for c in d.cards if c.t["key"] == "colour_min"))
        page.make_workflow(d)                                     # an empty-ish workflow is kept: it has a block
        settle(w, page)
        assert len(m.paths) == 2
        # a refused delete keeps the block selected
        q = m.paths[-1]
        page.select(index=m.paths.index(q), step="empilement")
        toasts = []
        page.toasted.connect(toasts.append)
        page.remove_tool(Graph(m).node_of(q, "empilement"), None)
        assert toasts and m.sel_step == "empilement"
        # no Color on monochrome frames
        m.set_session_filter(1, "Ha")
        settle(w, page)
        ha = m.paths[0]
        assert m.palette_reason(ha, "color").startswith("monochrome") and m.place_indices(ha, "color") == []
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_template_takes_the_place_of_an_empty_workflow_and_channels_are_not_post_layers(qapp, tmp_path, monkeypatch):
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        assert m.ensure_frames_workflow() is not None and m.p.workflows[0].blocks == []
        d = page.new_workflow()
        d.pick(next(c.t for c in d.cards if c.t["key"] == "filters_full"))
        d.palette.setCurrentText("HOO")
        page.make_workflow(d)
        settle(w, page)
        assert not any(x.blocks == [] for x in m.p.workflows)     # the empty workflow made room
        hoo = m.paths[-1]
        g = Graph(m)
        assert [p.pid for p in m.paths if m.feeds_fusion(p)] and not any(
            pid for _n, pid in g.post_layers if pid is not None and m.feeds_fusion(m.view(pid)))
        assert any(pid == hoo.pid for _n, pid in g.post_layers)
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_branches_parallel_flows_removable_wires_and_the_palette_taking_a_block_back(qapp, tmp_path, monkeypatch):
    """Issue #47: a new branch from a block (what comes before shared); a Session dropped on the new-flow zone makes
    a parallel flow; #52: a workflow's wire to Post can be taken away (Del), the chain's cannot be selected; #54: every
    removable block has its cross; a placed block dragged onto the palette is removed."""
    from astro_gpu.client.graph import BlockItem, WireItem
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        full = m.paths[0]
        assert m.can_branch(full, "astrometrie", "gradient")
        q, key = m.add_branch(full, "astrometrie", "gradient")
        settle(w, page)
        assert [n["block"] for n in m.graph(q)][-2:] == ["astrometry", "gradient"] and key == "fond"
        g = Graph(m)
        assert g.node_of(q, "astrometrie") is g.node_of(m.view(full.pid), "astrometrie")   # shared, computed once
        page.palette_action("session:new")                         # a parallel flow, its own night
        settle(w, page)
        flow = m.paths[-1]
        assert m.p.workflow(flow.pid).blocks == [] and len(m.sessions_of(flow)) == 1
        assert all(len(m.sessions_of(x)) == 1 for x in m.paths[:2])          # the others keep their night
        # every removable block has its cross
        crosses = {it.node.key for it in page.graph.scene().items() if isinstance(it, BlockItem) and it.removable}
        assert {"fond", "couleur", "astrometrie", "debruitage"} <= crosses
        # the wires to Post can be taken away; the chain's are not items
        wires = [it for it in page.graph.scene().items() if isinstance(it, WireItem)]
        assert wires
        page.graph.select_wire(wires[0])
        page._delete_selected()
        settle(w, page)
        assert len([it for it in page.graph.scene().items() if isinstance(it, WireItem)]) < len(wires)
        # a placed block dragged onto the palette: removed
        n = Graph(m).node_of(m.view(full.pid), "couleur")
        page.graph._dragged = n
        page.remove_tool(n, None)
        settle(w, page)
        assert "couleur" not in m.display_keys(m.view(full.pid))
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_the_first_blocks_have_their_cross_and_the_last_session_can_go(qapp, tmp_path, monkeypatch):
    """Session and Calibration are removed as any block; the last Session takes its workflows from the frames with it
    (files kept), Ctrl+Z brings everything back."""
    from astro_gpu.client.graph import BlockItem
    w, m, page = open_page(tmp_path, monkeypatch)
    said = []
    page.toasted.connect(said.append)
    try:
        settle(w, page)
        crosses = {it.node.key for it in page.graph.scene().items() if isinstance(it, BlockItem) and it.removable}
        assert {"sources", "calibration"} <= crosses
        n = len(m.paths)
        assert len(m.p.sessions) == 1 and n >= 1
        m.select(step="sources")
        settle(w, page)
        page._delete_selected()                                    # Del, as the block's cross
        settle(w, page)
        assert not m.p.sessions and not [q for q in m.paths if q.source is None]
        assert "Ctrl+Z" in said[-1] and "files kept" in said[-1]
        assert m.undo()
        settle(w, page)
        assert len(m.p.sessions) == 1 and len(m.paths) == n
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_stage_phase_moves_the_bar_from_its_first_file():
    """Issue #50: progress within a phase fills that phase's share of the bar."""
    from types import SimpleNamespace

    from astro_gpu.core.stage import Context
    sent = []
    ctx = Context.__new__(Context)
    ctx.stage = SimpleNamespace(name="calibrate")
    ctx.journal = SimpleNamespace(emit=lambda kind, **kw: sent.append(kw["frac"]))
    ctx._t0, ctx._last_progress = 0.0, 0.0
    ctx.check_stop = lambda: None
    ctx.phase(0.0, 0.4)
    ctx.progress(0.5, "reading darks", force=True)
    ctx.phase(0.4, 1.0)
    ctx.progress(0.5, "lights", force=True)
    assert sent == [0.2, 0.7]


def test_the_second_play_test_stays_fixed(qapp, tmp_path, monkeypatch):
    """Dragging Combine does not break the page; a wire taken away stays away (Post gives no layer by itself any
    more); a branch beside the same block tries another method; a rebuilt graph keeps its selected wire chosen."""
    from astro_gpu.client.graph import WireItem
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.graph.drag_block = "composition"
        page.graph.rebuild()                                       # its places: after the Stars/background block, no error
        assert any(getattr(gh, "node", None) is not None for gh in page.graph.ghosts)
        page.graph.drag_block = None
        page.graph.rebuild()
        full = m.paths[0]
        q, key = m.add_branch(full, "astrometrie", "gradient")    # Full: Astrometry then a polynomial Gradient
        assert m.graph(q)[-1]["method"] != "polynomial"           # the branch tries another one
        settle(w, page)
        wires = [it for it in page.graph.scene().items() if isinstance(it, WireItem)]
        page.graph.select_wire(wires[0])
        page.graph.rebuild()
        sw = page.graph.sel_wire                                  # 2nd round: the same wire drawn again stays
        assert sw is not None and sw is not wires[0] and sw.scene() is page.graph.scene() and sw.selected   # chosen
        page.graph.clear_selection()
        page.graph.rebuild()
        assert page.graph.sel_wire is None                        # nothing chosen: none comes back
        n_before = len(Graph(m).post_layers)
        page.remove_post_link(q.pid)
        settle(w, page)
        assert len(Graph(m).post_layers) == n_before - 1 and q.pid not in Graph(m).to_post   # it stays away
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_fusion_channel_session_removed_keeps_the_project_valid_and_the_fusion_red(qapp, tmp_path, monkeypatch):
    """GUI exploration 2, decision 3: the only Session of a fusion's channel cannot be removed: refused with the reason
    (no empty session put in its place, no broken project, no undo step); the channel's workflow and Image likewise."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        assert m.ensure_frames_workflow() is not None
        d = page.new_workflow()
        d.pick(next(c.t for c in d.cards if c.t["key"] == "filters_full"))
        d.palette.setCurrentText("HOO")
        page.make_workflow(d)
        settle(w, page)
        hoo = next(q for q in m.paths if (q.source or {}).get("fusion"))
        ha = next(q for q in m.paths if m.feeds_fusion(q))
        sid = m.sessions_of(ha)[0]
        n_sessions, n_undo = len(m.p.sessions), len(m._undo)
        assert "Ha" in m.session_removal_problem(sid) and hoo.name in m.session_removal_problem(sid)
        assert not m.remove_session(sid)
        toasts = []
        page.toast = toasts.append
        page.session_action("remove", sid)                         # the cross, Del, the menu: said why
        page.workflow_action("delete", ha.pid)
        settle(w, page)
        m.p.validate()                                             # still a valid project
        assert len(m.p.sessions) == n_sessions and len(m._undo) == n_undo and m.view(ha.pid) is not None
        assert len(toasts) == 2 and all(hoo.name in t for t in toasts)
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_parallel_flow_on_a_template_without_night_and_a_calibration_in_the_second_flow(qapp, tmp_path,
                                                                                          monkeypatch):
    """GUI exploration 1.5: a template's flow of frames, no night yet: a Session dropped on the new-flow zone makes a
    valid project. 1.6: Calibration placed in the second flow is selected beside that flow's own session."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        m.p.sessions.clear()
        m.add_graph_workflow("Workflow", [])
        q = m.add_parallel_flow()
        m.p.validate()
        assert m.sessions_of(q) and m.sessions_of(m.paths[0]) and m.sessions_of(q) != m.sessions_of(m.paths[0])
        page.place_block("calibration", q.pid)
        settle(w, page)
        assert m.sel_step == f"calibration@{m.sessions_of(m.view(q.pid))[0]}"
        page.refresh()                                             # the panel shows it (no error)
        # round 2: its review pause is its own and names its session (it did nothing, "after Calibration")
        sid = m.sessions_of(m.view(q.pid))[0]
        ref = f"calibration@{sid}"
        page.toggle_pause(Graph(m).node_of(m.view(q.pid), ref))
        settle(w, page)
        # the flow stacks that one session: the engine computes its Calibration as the workflow's own
        assert m.has_breakpoint(m.view(q.pid), ref) and m.breakpoints(m.view(q.pid)) == ["calibration"]
        name = m.p.session(sid).name
        assert m.history[-1][1] == "Pause set after {block}" and m.history[-1][2] == {"block": f"Calibration · {name}"}
        assert m.pause_point().all_stops() == [(q.pid, "calibration")]
        assert m.review_state(m.view(q.pid), ref) is not None                    # the dot on the block drawn
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_block_placed_twice_has_its_own_settings_and_its_own_pause(qapp, tmp_path, monkeypatch):
    """GUI exploration 1.2: the second Gradient is a block of its own: its setting changes (the first keeps its
    own), it gets its pause for review, and Run to here aims at it."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        full = m.paths[0]
        page.place_block("gradient")
        settle(w, page)
        p = m.view(full.pid)
        assert "fond~2" in m.display_keys(p)
        m.set_param(p, "gradient~2", "degree", 4)
        assert m.p.workflow(full.pid).params.get("gradient~2", {}).get("degree") == 4
        assert m.p.workflow(full.pid).params.get("gradient", {}).get("degree") is None
        m.set_breakpoint([p], "fond~2", True)
        assert m.has_breakpoint(p, "fond~2") and not m.has_breakpoint(p, "fond")
        assert m.breakpoints(p) == ["fond~2"]
        m.p.validate()
        # round 2: Run planned up to it (an error hidden behind "Nothing to run" before), named as the graph names it
        pause = m.pause_point()
        assert pause is not None and pause.block == "fond~2" and pause.all_stops() == [(p.pid, "fond~2")]
        plan, pause = m.plan_run()
        assert plan.jobs and pause.block == "fond~2"
        assert m.pause_name(p.pid, "fond~2") == "Gradient 2" and m.pause_name(p.pid, "fond") == "Gradient 1"
        from astro_gpu.client.services.run import pause_text
        assert pause_text(m, pause) == "Gradient 2"
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_pause_stays_with_a_block_switched_off_and_works_on_a_sessions_calibration(qapp, tmp_path, monkeypatch):
    """Round 2: switching a block off kept no pause (switching it on did not bring it back); the pause on a session's
    Calibration did nothing and its name did not say which session."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        p = m.paths[0]
        m.set_breakpoint([p], "fond", True)
        g = Graph(m)
        node = g.node_of(m.view(p.pid), "fond")
        page.toggle_cut(node, [p.pid])                                 # D
        settle(w, page)
        assert "gradient" in m.cut_blocks(m.view(p.pid))
        assert m.has_breakpoint(m.view(p.pid), "fond")                # kept with the block switched off
        assert "fond" not in m.breakpoints(m.view(p.pid))              # but no stop while it is off
        m.undo()
        settle(w, page)
        assert m.breakpoints(m.view(p.pid)) == ["fond"]
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_the_answers_to_the_design_questions(qapp, tmp_path, monkeypatch):
    """The user's answers (GUI exploration, 7 November): a branch made only to send an image goes with its wire (1);
    a block placed starts from its defaults (9); Combine channels is greyed with what to do while there is nothing to combine
    (6); a workflow deleted at once, said (10.1); a block switched off is not placed twice (4.4); a removal refused in
    one workflow is refused in all (3.7)."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        full = m.paths[0]
        page.send_to_post(Graph(m).node_of(m.view(full.pid), "astrometrie"))
        settle(w, page)
        sent = m.paths[-1]
        assert m.is_sent_branch(sent)
        page.remove_post_link(sent.pid)
        settle(w, page)
        assert m.view(sent.pid) is None                            # question 1: the branch went with its wire
        # question 9: a Gradient placed again starts from its defaults
        m.set_param(m.view(full.pid), "gradient", "degree", 4)
        page.remove_tool(Graph(m).node_of(m.view(full.pid), "fond"), None)
        settle(w, page)
        page.place_block("gradient", full.pid)
        settle(w, page)
        assert m.p.workflow(full.pid).params.get("gradient", {}).get("degree") is None
        # question 6: nothing to combine yet in a project of one colour workflow without Stacking? it has one: placeable
        assert page.palette_reason("fusion", m.view(full.pid), None) == ""
        # 4.4: Registration switched off is not placed a second time
        page.toggle_cut(Graph(m).node_of(m.view(full.pid), "alignement"), None)
        settle(w, page)
        assert "off: switch on" in m.palette_reason(m.view(full.pid), "registration")
        # 10.1: Delete workflow at once, no question
        toasts = []
        page.toasted.connect(toasts.append)
        page.workflow_action("delete", full.pid)
        settle(w, page)
        assert m.view(full.pid) is None and toasts and "Ctrl+Z" in toasts[-1]
        assert m.undo() and m.view(full.pid) is not None
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_wire_taken_away_stays_away_and_what_is_sent_after_still_goes_to_post(qapp, tmp_path, monkeypatch):
    """GUI exploration 1.3: once a wire to Post was taken away, a block's image sent to Post, a new branch and the
    wire given back all reach Post; only the image taken out stays out (it is listed, not the whole composition)."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        full = m.paths[0]
        page.remove_post_link(full.pid)
        settle(w, page)
        assert full.pid not in Graph(m).to_post
        page.send_to_post(Graph(m).node_of(m.view(full.pid), "astrometrie"))      # a block's image, sent
        settle(w, page)
        sent = m.paths[-1]
        assert sent.pid in Graph(m).to_post and full.pid not in Graph(m).to_post
        q, _key = m.add_branch(full, "astrometrie", "gradient")                  # a new branch
        settle(w, page)
        assert q.pid in Graph(m).to_post and full.pid not in Graph(m).to_post
        m.add_post_layer(full.pid, None)                                          # the wire given back
        settle(w, page)
        assert full.pid in Graph(m).to_post and (full.pid, None) not in m.post_removed()
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_branch_names_are_numbered_as_names_not_by_their_blocks_number(qapp, tmp_path, monkeypatch):
    """GUI exploration 4: a branch whose new block is the chain's second Gradient was named "… · Gradient 2", and the
    next one "… · Gradient 2 2". Branches are named by their block's kind, numbered only to stay unique."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        full = m.paths[0]
        if "fond" not in m.display_keys(full):
            page.place_block("gradient", full.pid)
            settle(w, page)
        names = []
        for key in ("astrometrie", "fond", "fond"):
            q, _k = m.add_branch(m.view(full.pid), key, "gradient")
            names.append(q.name)
        assert names == [f"{full.name} · Gradient", f"{full.name} · Gradient 2", f"{full.name} · Gradient 3"]
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def _blank_page(tmp_path, monkeypatch):
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    for x in list(m.p.sessions):
        assert m.remove_session(x.id)
    settle(w, page)
    assert page.is_blank() and not m.paths
    return w, m, page


def test_a_blank_page_has_a_link_to_the_templates_in_place_of_the_graph(qapp, tmp_path, monkeypatch):
    """No window on a new project: the page stays blank, its link shows the six cards in place of the graph, a click
    makes the workflow, Ctrl+Z gives the blank page back."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from astro_gpu.client.graph import LinkItem
    w, m, page = _blank_page(tmp_path, monkeypatch)
    page.show()
    try:
        page.graph.rebuild()
        link = next(x for x in page.graph.scene().items() if isinstance(x, LinkItem))
        assert link.text == "Start from a template"
        assert not page.templates.isVisibleTo(page)
        QApplication.processEvents()                      # a real click, through the view (issue #59)
        QTest.mouseClick(page.graph.viewport(), Qt.LeftButton, pos=page.graph.mapFromScene(link.sceneBoundingRect().center()))
        assert page.templates.isVisibleTo(page) and not page.graph.isVisibleTo(page)
        assert [c.t["key"] for c in page.templates.cards] == ["colour_min", "colour_full", "image", "filters_min",
                                                              "filters_full", "images"]
        page.templates.back_b.click()
        assert page.graph.isVisibleTo(page) and not page.templates.isVisibleTo(page) and page.is_blank()
        assert page.new_workflow() is None and page.templates.isVisibleTo(page)   # the menu: no window on a blank page
        card = next(c for c in page.templates.cards if c.t["key"] == "colour_min")
        card.mousePressEvent(None)
        settle(w, page)
        assert page.graph.isVisibleTo(page) and not page.templates.isVisibleTo(page)
        assert len(m.paths) == 1 and len(m.p.sessions) == 1 and "etoiles" not in m.display_keys(m.paths[0])
        assert m.undo()
        settle(w, page)
        assert page.is_blank()
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_the_filters_card_makes_the_palette_chosen_on_it(qapp, tmp_path, monkeypatch):
    w, m, page = _blank_page(tmp_path, monkeypatch)
    try:
        page.show_templates()
        card = next(c for c in page.templates.cards if c.t["key"] == "filters_full")
        card.set_palette("LRGB")
        card.mousePressEvent(None)
        settle(w, page)
        assert sorted(x.filter for x in m.p.sessions) == ["B", "G", "L", "R"]
        assert m.paths[-1].name == "LRGB" and (m.paths[-1].source or {}).get("fusion")
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_several_images_make_one_layer_each_and_cancelling_makes_nothing(qapp, tmp_path, monkeypatch):
    """The files are asked first; a PNG is taken as stretched, a FITS as linear; each is a workflow going to Post."""
    import cv2
    import numpy as np
    from astropy.io import fits

    from astro_gpu.client import gallery
    png, fit = tmp_path / "a.png", tmp_path / "b.fits"
    cv2.imwrite(str(png), np.full((8, 8, 3), 128, np.uint8))
    fits.PrimaryHDU(np.full((3, 8, 8), 0.1, np.float32)).writeto(fit)
    w, m, page = _blank_page(tmp_path, monkeypatch)
    try:
        page.show_templates()
        card = next(c for c in page.templates.cards if c.t["key"] == "images")
        monkeypatch.setattr(gallery.QFileDialog, "getOpenFileNames", lambda *a, **k: ([], ""))
        card.mousePressEvent(None)
        settle(w, page)
        assert page.is_blank() and page.templates.isVisibleTo(page)
        monkeypatch.setattr(gallery.QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(png), str(fit)], ""))
        card.mousePressEvent(None)
        settle(w, page)
        assert [q.name for q in m.paths] == ["a", "b"] and [m.graph(q) for q in m.paths] == [[], []]
        assert [m.p.workflow(q.pid).source["stretched"] for q in m.paths] == [True, False]
        assert {q.pid for q in m.paths} <= set(Graph(m).to_post)
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_post_saving_its_layers_keeps_the_workflows_not_computed_yet(qapp, tmp_path, monkeypatch):
    """GUI exploration 4: Post holds layers only for the computed workflows; its saving a composition (a filter added)
    does not take the others' images out of Post (they used to be listed as removed, their wires gone for good)."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        full = m.paths[0]
        q, _key = m.add_branch(full, "astrometrie", "gradient")
        settle(w, page)
        before = set(Graph(m).to_post)
        assert q.pid in before
        post_page = type("PostView", (), {"is_post_page": True})()
        lay = {"id": 50, "kind": "filter", "filter": "curves", "name": "Curves", "visible": True, "opacity": 100.0,
               "mode": "Normal", "params": {}, "mask": None}
        m.set_post_layers([lay], post_page)                                      # nothing computed: no workflow layer
        settle(w, page)
        assert not m.post_removed() and set(Graph(m).to_post) == before
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_an_image_dropped_is_placed_empty_and_the_new_flow_zone_offers_no_template(qapp, tmp_path, monkeypatch):
    """The third use: an Image dropped is placed like a Session, its file chosen in its panel (no template window);
    without its file it is not run; once a graph is begun the new-flow zone takes drops only (no template)."""
    from astro_gpu import api
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.graph.rebuild()
        zone = page.graph.new_flow_ghost
        assert zone.on_click is None and "template" not in zone.text
        page.palette_action("image:new")
        settle(w, page)
        assert getattr(page, "_gallery", None) is None                    # no template window
        img = m.paths[-1]
        assert img.source == {"path": "", "stretched": False} and m.sel_step == "image"
        assert m.graph(img) == [] and not m.has_data(img) and m.has_data()  # the frames of the other workflow
        assert Graph(m).summary(Graph(m).node_of(img, "image")).endswith("file to choose")
        assert img.pid in api.plan_all(m.p).skipped                        # not run without its file
        m.set_image_file(img, str(tmp_path / "stack.fits"))
        settle(w, page)
        img = m.view(img.pid)
        assert m.has_data(img) and img.source["path"].endswith("stack.fits")
        assert img.pid not in api.plan_all(m.p).skipped
        m.set_image_file(img, stretched=True)                              # nothing follows it: either kind
        settle(w, page)
        assert m.view(img.pid).source["stretched"] and m.p.workflow(img.pid).blocks == []   # no Output: to Post as is
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_session_chosen_beside_an_image_gets_the_frame_blocks_and_sources_have_their_cross(qapp, tmp_path,
                                                                                            monkeypatch):
    """A Session chosen after an Image was placed: the palette places in its frames (Calibration offered, a
    workflow from the frames started when there is none); the Image, the sessions have their cross; the Image goes
    with the blocks after it, a session with its own flow; Ctrl+Z brings either back."""
    from astro_gpu.client.graph import NEW_FRAMES, BlockItem
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        page.palette_action("image")
        settle(w, page)
        drawn = {it.node.key: it for it in page.graph.scene().items() if isinstance(it, BlockItem)}
        assert "sources" in drawn and drawn["image"].removable                # the Session still drawn
        assert page.cards["calibration"].reason                               # the Image chosen
        page.select_node(Graph(m).root)
        settle(w, page)
        assert page.palette_path() is None and not page.cards["calibration"].reason
        page.graph.drag_block = "calibration"
        page.graph.rebuild()
        assert any(gh.pid == NEW_FRAMES for gh in page.graph.ghosts if hasattr(gh, "idx"))
        page.graph.drag_block = None
        page.palette_action("calibration")
        settle(w, page)
        frames = next(q for q in m.paths if q.source is None)
        assert [n["block"] for n in m.graph(frames)] == ["calibration"]
        img = next(q for q in m.paths if q.source is not None)
        page.remove_tool(Graph(m).node_of(img, "image"), None)
        settle(w, page)
        assert all(q.source is None for q in m.paths)
        assert m.undo() and any(q.source is not None for q in m.paths)
        page.palette_action("session:new")                                     # a second flow, its own session
        settle(w, page)
        flow = m.paths[-1]
        sid = m.sessions_of(flow)[0]
        page.session_action("remove", sid)
        settle(w, page)
        assert m.view(flow.pid) is None and len(m.p.sessions) == 1
        assert m.undo() and m.view(flow.pid) is not None and len(m.p.sessions) == 2
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_block_in_the_middle_sends_its_image_to_post(qapp, tmp_path, monkeypatch):
    """Issue #47: Astrometry's image to Post (menu or its port pulled onto Post): a branch up to it, shared, computed
    once; once sent, not offered again; the last block's image goes to Post already."""
    from astro_gpu.client.graph import BlockItem
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        full = m.paths[0]
        g = Graph(m)
        astro = g.node_of(full, "astrometrie")
        assert page.graph.can_exit(astro) and not page.graph.can_exit(g.node_of(full, "calibration"))
        page.graph.rebuild()
        item = next(it for it in page.graph.scene().items() if isinstance(it, BlockItem) and it.node.key == "astrometrie")
        assert item.ports                                                  # its port, to pull onto Post
        n_before = len(Graph(m).post_layers)
        page.send_to_post(astro)
        settle(w, page)
        q = m.paths[-1]
        assert [n["block"] for n in m.graph(q)][-1] == "astrometry"
        g2 = Graph(m)
        assert g2.node_of(q, "astrometrie") is g2.node_of(m.view(full.pid), "astrometrie")             # shared
        assert len(Graph(m).post_layers) == n_before + 1
        assert not page.graph.can_exit(Graph(m).node_of(m.view(full.pid), "astrometrie"))           # sent already
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_after_the_start_no_block_opens_the_template_window(qapp, tmp_path, monkeypatch):
    """The templates once, when the project is made: a Combine channels dropped is placed at once (here a colour workflow's
    channels, dual band), Duplicate copies at once, the new-flow zone is no button."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.graph.rebuild()
        assert page.graph.new_flow_ghost.on_click is None
        page.palette_action("fusion:new")
        settle(w, page)
        assert getattr(page, "_gallery", None) is None
        fu = m.paths[-1]
        f = fu.source["fusion"]
        assert f["palette"] == "HOO" and [i["channel"] for i in f["inputs"]] == ["R", "GB"] and m.sel_step == "fusion"
        n = len(m.paths)
        page.workflow_action("duplicate", m.paths[0].pid)
        settle(w, page)
        assert getattr(page, "_gallery", None) is None and len(m.paths) == n + 1
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_file_added_by_mistake_is_warned_and_taken_out_one_by_one(qapp, tmp_path, monkeypatch):
    """Issue #58: a JPEG added among the raw files is warned (Session block, panel, health check); its group opens on
    its files; one file taken out alone, a file of a folder chosen whole too (the folder becomes its other files);
    the last light can go."""
    from astro_gpu import api
    from astro_gpu.client.sources import FileList, SourcesView
    from astro_gpu.core.project import type_warnings
    assert [w["key"] for w in type_warnings({"lights": ["a.ARW", "b.ARW", "c.JPG"], "flats": ["f.fits"]})] == \
        ["unread", "other_type"]
    assert type_warnings({"lights": ["a.ARW", "b.CR2", "c.ARW"]})[0] == {
        "key": "mixed", "kind": "lights", "type": "CR2", "n": 1, "main": "ARW", "of": 2, "name": "b.CR2"}
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        raw = tmp_path / "raw" / "n1"
        (raw / "a2.ARW").write_bytes(b"\0" * 300)
        jpg = tmp_path / "phone" / "DSC05869.JPG"
        jpg.parent.mkdir()
        jpg.write_bytes(b"\0" * 50)
        assert m.add_session_entries(1, "lights", [str(jpg)]) == 1
        w.wait(60)
        settle(w, page)
        warns = m.file_warnings(1)
        assert [x["key"] for x in warns] == ["unread"] and warns[0]["name"] == "DSC05869.JPG"
        assert Graph(m).summary(Graph(m).root).startswith("⚠")
        assert any(c.key == "file_types" for c in api.health_checks(m.p))
        page.select_node(Graph(m).root)
        settle(w, page)
        view = SourcesView(m)
        view.refresh()
        lst = view.file_list
        assert isinstance(lst, FileList) and any("DSC05869.JPG" in (lst.item(i).data(256) or "") for i in range(lst.count()))
        assert any("DSC05869.JPG" in x for x in view.anomalies("lights"))           # said beside its kind
        view.close()
        assert m.remove_session_files(1, "lights", [jpg.as_posix()])
        w.wait(60)
        settle(w, page)
        assert not m.file_warnings(1) and [Path(e).name for e in m.session(1).lights] == ["n1"]
        assert m.remove_session_files(1, "lights", [(raw / "a1.ARW").as_posix()])     # a file of a whole folder
        w.wait(60)
        assert [Path(e).name for e in m.session(1).lights] == ["a2.ARW"]
        assert m.remove_session_files(1, "lights", [(raw / "a2.ARW").as_posix()])     # the last light too
        assert m.session(1).lights == []
    finally:
        page.close()
        m.close(save=False)
        w.stop()


# ------------------------------------------------------------ GUI exploration, control round (agent G)
def graph_page(tmp_path, monkeypatch, blocks):
    """A workflow built block by block from the palette, on the project's one session (nothing computed)."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    p = m.ensure_frames_workflow()
    for b in blocks:
        page.place_block(b, p.pid)
        settle(w, page)
    return w, m, page, m.view(p.pid)


def menu_texts(page, node, monkeypatch) -> list[str]:
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QMenu

    from astro_gpu.client import graph as graphmod
    got = []

    class Menu(QMenu):                                  # built, not run (exec would wait for a click)
        def exec(self, *a, **k):
            got.append(self)
    monkeypatch.setattr(graphmod, "QMenu", Menu)
    page.graph.block_menu(node, QPoint(0, 0))
    return [a.text() or "---" for a in got[-1].actions()] if got else []


def test_control_round_a_port_pulled_again_or_after_its_wire_went_says_so_or_gives_it_back(qapp, tmp_path,
                                                                                            monkeypatch):
    """5, 7, 10: every block giving an image has its port and "To Post", at one place in its menu; pulled again it
    says its image is a layer already; the last block after its wire was taken away gives the wire back; a shared
    block's image is named after the workflow that owns it."""
    w, m, page, p = graph_page(tmp_path, monkeypatch, ["calibration", "registration", "stacking", "astrometry",
                                                        "gradient", "color"])
    told = []
    page.toasted.connect(told.append)
    try:
        def nd(key, q=None):
            return page.graph.graph.node_of(q or m.view(p.pid), key)
        page.send_to_post(nd("fond"))
        settle(w, page)
        assert [q.name for q in m.paths] == ["Workflow", "Workflow · Gradient"]
        assert page.graph.has_port(nd("fond"))                      # its port stays
        page.send_to_post(nd("fond"))
        settle(w, page)
        assert len(m.paths) == 2 and "already a layer of Post" in told[-1]
        m.current = 1                                               # the branch chosen: the shared Stacking
        page.send_to_post(nd("empilement", m.paths[0]))
        settle(w, page)
        assert m.paths[-1].name == "Workflow · Raw stack"           # its owner's name, not the branch's
        page.send_to_post(nd("empilement", m.paths[0]))
        settle(w, page)
        assert "already a layer of Post" in told[-1]
        page.remove_post_link(p.pid)                                # the trunk's wire taken away...
        settle(w, page)
        assert p.pid not in page.graph.graph.to_post and page.graph.has_port(nd("couleur"))
        page.send_to_post(nd("couleur"))                            # ...its last block's port gives it back
        settle(w, page)
        assert p.pid in page.graph.graph.to_post and "to Post" in told[-1] and "Ctrl+Z" in told[-1]
        texts = menu_texts(page, nd("fond"), monkeypatch)
        i = texts.index("Inspect")
        assert texts[i + 1:i + 4] == ["---", "To Post", "---"]
        texts = menu_texts(page, nd("couleur"), monkeypatch)       # the last block too, not in its Workflow submenu
        i = texts.index("Inspect")
        assert texts[i + 1:i + 4] == ["---", "To Post", "---"]
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_control_round_a_branch_left_without_a_block_of_its_own_goes_with_it(qapp, tmp_path, monkeypatch):
    w, m, page, p = graph_page(tmp_path, monkeypatch, ["calibration", "registration", "stacking", "astrometry"])
    told = []
    page.toasted.connect(told.append)
    try:
        page.branch_block("color", p.pid, "astrometrie")
        settle(w, page)
        br = m.paths[-1]
        assert [q.name for q in m.paths] == ["Workflow", "Workflow · Color"]
        page.remove_tool(Graph(m).node_of(br, "couleur"), None)
        settle(w, page)
        assert [q.name for q in m.paths] == ["Workflow"]          # not a copy of the trunk sending its image
        assert "with its branch “Workflow · Color”" in told[-1] and "Ctrl+Z" in told[-1]
        assert m.undo() and [q.name for q in m.paths] == ["Workflow", "Workflow · Color"]      # one step: both back
        # a block's image sent to Post (a branch with no block of its own from the start) stays so
        page.send_to_post(Graph(m).node_of(m.view(p.pid), "empilement"))
        settle(w, page)
        sent = m.paths[-1]
        assert sent.name == "Workflow · Raw stack" and not m.is_emptied_branch(m.view(p.pid))
        assert m.is_emptied_branch(sent)
        # a copy (or any workflow not made as a branch) never goes with a block deleted in it
        page.workflow_action("duplicate", p.pid)
        settle(w, page)
        cp = m.paths[-1]
        page.remove_tool(Graph(m).node_of(cp, "astrometrie"), [cp.pid])
        settle(w, page)
        assert m.view(cp.pid) is not None and "astrometrie" not in m.display_keys(m.view(cp.pid))
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_control_round_a_fusion_channel_sent_to_post_keeps_the_channels_apart(qapp, tmp_path, monkeypatch):
    """1: the branch of a channel's block takes its channel's last lane: two channels never drawn on each other."""
    from PySide6.QtCore import QRectF

    from astro_gpu.client.graph import BlockItem, GraphView
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    for sid, f in ((1, "Ha"), (2, "OIII")):
        p.sessions.append(project2.Session(sid, f, lights=[str(frames(tmp_path / f, [f"{f}1.ARW"]))], filter=f))
        x = p.add_workflow(f, wfl.TEMPLATES["Classic stack"])
        x.sessions = [sid]
        p.set_blocks(x.id, [wfl.node(b) for b in ("calibration", "registration", "stacking", "astrometry")])
    project2.save(p)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(120)
    m._sync()
    page = GraphPage(m)
    page.resize(1500, 800)
    try:
        ha = m.paths[0]
        page.send_to_post(Graph(m).node_of(ha, "empilement"))
        settle(w, page)
        br = m.paths[-1]
        assert br.name == "Ha · Raw stack"
        order = GraphView.lane_order(Graph(m))
        assert order.index(br.pid) == order.index(ha.pid) + 1 < order.index(m.paths[1].pid)
        rects = [QRectF(it.pos(), it.boundingRect().size()) for it in page.graph.scene().items()
                 if isinstance(it, BlockItem) and it.node.key not in ("sources", "final")]
        assert not any(a.intersects(b) for i, a in enumerate(rects) for b in rects[i + 1:])
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_control_round_the_fusion_card_waits_for_the_channels_lights(qapp, tmp_path, monkeypatch):
    """2: a colour project without lights: the Combine channels card greyed, saying what to do."""
    w, m, page, p = graph_page(tmp_path, monkeypatch, ["calibration", "registration", "stacking"])
    try:
        assert page.palette_reason("fusion", m.view(p.pid), None) == ""
        assert m.remove_session_entry(1, "lights", m.session(1).lights[0])
        settle(w, page)
        assert page.palette_reason("fusion", m.view(p.pid), None) == "lights first in “Workflow”"
        n = len(m.paths)
        page.place_fusion()                                         # refused: nothing placed
        assert len(m.paths) == n
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_control_round_undo_selects_the_block_before_and_a_file_taken_out_is_one_undo_step(qapp, tmp_path,
                                                                                           monkeypatch):
    """12: Ctrl+Z of a placed block selects the block drawn before it; 13: a file taken out of a session is one
    undo step (Ctrl+Z puts it back, it never undoes an older step), said with Ctrl+Z."""
    w, m, page, p = graph_page(tmp_path, monkeypatch, ["calibration", "registration", "stacking"])
    told = []
    page.toasted.connect(told.append)
    try:
        assert m.sel_step == "empilement"
        assert m.undo()
        settle(w, page)
        assert m.sel_step == "alignement" and page.current_node() is not None
        assert m.redo()
        settle(w, page)
        lights = list(m.session(1).lights)
        blocks = [n["block"] for n in m.p.workflow(p.pid).blocks]
        from astro_gpu.client.sources import SourcesView
        view = SourcesView(m)
        view.toast.connect(told.append)
        view.refresh()
        view.files_out("lights", lights)
        view.close()
        settle(w, page)
        assert m.session(1).lights == [] and "Ctrl+Z" in told[-1]
        assert m.undo()
        settle(w, page)
        assert m.session(1).lights == lights and [n["block"] for n in m.p.workflow(p.pid).blocks] == blocks
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_control_round_a_variant_of_a_graph_keeps_its_blocks_and_each_choice_is_said(qapp, tmp_path, monkeypatch):
    """14: Try as variant on a graph: the variant has the original's blocks (no Selection added), Keep keeps them;
    Keep, Discard, Duplicate say what they did."""
    w, m, page, p = graph_page(tmp_path, monkeypatch, ["calibration", "registration", "stacking"])
    told = []
    page.toasted.connect(told.append)
    try:
        ch = dict(kind="param", stage="stack", name="kappa", value=3.5, suggested=None)
        before = [n["block"] for n in m.p.workflow(p.pid).blocks]
        q = m.add_variant(m.view(p.pid), ch, "Workflow · kappa")
        assert [n["block"] for n in m.p.workflow(q.pid).blocks] == before
        assert m.p.workflow(q.pid).sessions == m.p.workflow(p.pid).sessions
        page.select(index=m.paths.index(m.view(q.pid)), step="empilement")
        settle(w, page)
        from PySide6.QtWidgets import QPushButton
        keep = next(b for b in page.panel.findChildren(QPushButton) if b.text() == "Keep")
        keep.click()
        settle(w, page)
        assert [q.name for q in m.paths] == ["Workflow"] and "kept" in told[-1] and "Ctrl+Z" in told[-1]
        assert [n["block"] for n in m.p.workflow(m.paths[0].pid).blocks] == before
        page.workflow_action("duplicate", m.paths[0].pid)
        assert "duplicated" in told[-1] and "Ctrl+Z" in told[-1]
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def _shortcut(page, key):
    from PySide6.QtGui import QKeySequence, QShortcut
    return next(sc for sc in page.findChildren(QShortcut) if sc.key() == QKeySequence(key))


def test_f2_renames_and_ctrl_d_duplicates_what_is_selected(qapp, tmp_path, monkeypatch):
    """GUI exploration 4, Q5: F2 renames the workflow of the selected block (the Session, when it is selected), Ctrl+D
    duplicates the workflow, as in the reference apps; D alone still switches the block off."""
    from astro_gpu.client import graph as graphmod
    w, m, page = open_page(tmp_path, monkeypatch)
    names = iter(["Renamed", "Night A"])
    monkeypatch.setattr(graphmod, "ask_name", lambda parent, title, current: next(names))
    try:
        full = m.paths[0]
        page.select(index=0, step="empilement")
        settle(w, page)
        _shortcut(page, "F2").activated.emit()
        settle(w, page)
        assert m.view(full.pid).name == "Renamed"
        n0 = len(m.paths)
        _shortcut(page, "Ctrl+D").activated.emit()
        settle(w, page)
        assert len(m.paths) == n0 + 1 and m.paths[-1].name == "Renamed (copy)"
        assert not any(n.cut for n in Graph(m).nodes)
        page.select(index=0, step="sources")
        settle(w, page)
        _shortcut(page, "F2").activated.emit()
        _shortcut(page, "Ctrl+D").activated.emit()                   # a Session: nothing to duplicate
        settle(w, page)
        assert m.session(1).name == "Night A" and len(m.paths) == n0 + 1
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_show_all_is_kept_with_the_project(qapp, tmp_path, monkeypatch):
    """GUI exploration 4, Q9: "Show all" on, the project saved and opened again: it is still on (as the selection);
    it keys no result."""
    from astro_gpu import api
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        before = api._keys(m.p).project.to_dict()
        page.graph.b_all.setChecked(True)
        assert m.p.show_all
        assert {k: v for k, v in api._keys(m.p).project.to_dict().items() if k != "show_all"} == \
            {k: v for k, v in before.items() if k != "show_all"}
        project2.save(m.p)
        assert project2.load(m.root).show_all
        page2 = GraphPage(m)
        assert page2.graph.b_all.isChecked()
        page2.close()
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_recombine_is_shown_greyed_before_the_lights(qapp, tmp_path, monkeypatch):
    """GUI exploration 4, Q10: a workflow with Split stars but no lights yet has no layer in Post, so no Combine layers
    block; it is drawn greyed from the start ("after the lights"), with no wire, and a click says why instead of
    choosing it. Once the lights are there it is the real one."""
    w, m, page = open_page(tmp_path, monkeypatch, lights=False)
    told = []
    page.toasted.connect(told.append)
    try:
        full = m.paths[0]
        assert full.opts.get("etoiles") == "retirer"
        g = Graph(m)
        comp = [n for n in g.nodes if n.key == "composition"]
        assert len(comp) == 1 and comp[0].grey and comp[0].lid is None and not g.compositions
        assert not g.post_layers
        page.select_node(comp[0])
        assert m.sel_step != "composition" and "lights" in told[-1]
        from astro_gpu.client.graph import BlockItem
        assert any(isinstance(it, BlockItem) and it.node.key == "composition" for it in page.graph.scene().items())
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_chain_wire_says_it_cannot_be_removed(qapp, tmp_path, monkeypatch):
    """Issue #52: the chain's wires are not selectable, but say so: in their tooltip, and when clicked."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from astro_gpu.client.graph import ChainWireItem
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.show()
        settle(w, page)
        view = page.graph
        chain = [it for it in view.scene().items() if isinstance(it, ChainWireItem)]
        assert chain and all("cannot be removed" in it.toolTip() for it in chain)
        spots = [view.mapFromScene(it.path().pointAtPercent(0.5)) for it in chain]
        pos = next(p for p in spots if isinstance(view.itemAt(p), ChainWireItem))
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=pos)
        assert view.toast.text() == "Chain wire: not removable" and view.sel_wire is None
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def _carried(block: str):
    from PySide6.QtCore import QMimeData
    mime = QMimeData()
    mime.setText("block:" + block)
    return mime


def _drag_over(view, block: str, pos):
    """A palette card carried onto the graph, then held at ``pos`` (a function: the places appear once it is carried):
    the move event, whose answer says if it can drop there."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent
    mime = _carried(block)                                 # alive as long as its events (they hold a C++ pointer)
    view.dragEnterEvent(QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    move = QDragMoveEvent(pos(), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    view.dragMoveEvent(move)
    return move


def _drop(view, block: str, pos):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QDropEvent
    mime = _carried(block)
    drop = QDropEvent(QPointF(pos), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    view.dropEvent(drop)
    return drop


def test_fusion_is_dropped_in_the_composition_column_not_on_the_new_flow_band(qapp, tmp_path, monkeypatch):
    """Issue #53: Combine channels's drop zone is in the Composition column; the new-flow band refuses it (it starts the flow of
    a Session or an Image only)."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        view = page.graph
        view.rebuild()
        n = len(m.paths)

        def band():
            return view.mapFromScene(view.new_flow_ghost.rect.center())

        def place():
            return view.mapFromScene(view.ghosts[0].rect.center())
        move = _drag_over(view, "fusion", band)
        assert not move.isAccepted() and not view.new_flow_ghost.hot
        _drop(view, "fusion", band())
        settle(w, page)
        assert len(m.paths) == n                                    # refused on the band
        move = _drag_over(view, "fusion", place)
        ghost = view.ghosts[0]
        assert move.isAccepted() and ghost.text == "Combine channels" and ghost.hot
        assert view._cx["fusion"] <= ghost.rect.left() and ghost.rect.right() <= view._cx["fusion"] + view._cw["fusion"]
        assert _drop(view, "fusion", place()).isAccepted()
        settle(w, page)
        assert len(m.paths) == n + 1 and m.paths[-1].source.get("fusion") and m.sel_step == "fusion"
        assert view.col_of(Graph(m).node_of(m.paths[-1], "fusion")) == "fusion"
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def _a_fits(path: Path) -> str:
    import numpy as np
    from astropy.io import fits
    fits.PrimaryHDU(np.full((3, 32, 32), 100.0, np.float32)).writeto(path)
    return str(path)


def test_a_linear_image_is_chosen_as_a_fusion_channel_on_its_row(qapp, tmp_path, monkeypatch):
    """Issue #55: the two meet in a Combine channels: a channel's row chooses a linear Image's output (with its Astrometry the
    channels align); the starless modes work with it, it gets its Stars/background; an image already stretched is
    listed greyed, with its reason."""
    from PySide6.QtWidgets import QComboBox

    from astro_gpu.core.store import Keys
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.palette_action("image:new")
        settle(w, page)
        img = m.paths[-1]
        m.set_image_file(img, _a_fits(tmp_path / "stack.fits"))
        page.place_block("astrometry", img.pid)
        stretched = m.add_image_workflow("Done", _a_fits(tmp_path / "done.fits"), stretched=True)
        settle(w, page)
        page.place_fusion()
        settle(w, page)
        fu = m.paths[-1]
        assert fu.source["fusion"]["stars"]["mode"] == "workflow"     # as for a colour stack: its starless image
        offered = {q.pid: why for q, why in m.channel_choices(fu, "OIII")}
        assert offered[img.pid] == "" and "stretch" in offered[stretched.pid]
        assert not m.set_fusion_channel(fu, "OIII", stretched.pid)
        assert m.set_fusion_stars(m.view(fu.pid), None)
        page.select(index=m.paths.index(m.view(fu.pid)), step="fusion")
        settle(w, page)
        rows = [c for c in page.panel.findChildren(QComboBox) if c.findData(img.pid) >= 0]
        assert len(rows) == 2                                        # Ha, OIII
        grey = rows[1].model().item(rows[1].findData(stretched.pid))
        assert not grey.isEnabled() and "stretch" in grey.toolTip()
        i = rows[1].findData(img.pid)
        rows[1].setCurrentIndex(i)
        rows[1].activated.emit(i)
        settle(w, page)
        fu = m.view(fu.pid)
        assert [x["pid"] for x in fu.source["fusion"]["inputs"]] == [m.paths[0].pid, img.pid]
        assert m.fusion_problem(fu) == ""
        k = Keys(m.p)
        got = next(x for x in k.overrides(fu.pid, "fusion")["inputs"] if x["role"] == "OIII")
        assert "render_simple" in k.chain(img.pid) and got["dir"] and got["wcs"]
        g = Graph(m)
        assert g.leaves[img.pid] in g.fusion_inputs[id(g.node_of(fu, "fusion"))]
        page.select(index=m.paths.index(fu), step="fusion")
        settle(w, page)
        stars = next(c for c in page.panel.findChildren(QComboBox) if c.itemText(0) == "With channels (no split)")
        j = next(k for k in range(stars.count()) if stars.itemData(k) == {"mode": "palette"})
        assert stars.model().item(j).isEnabled()
        stars.activated.emit(j)
        settle(w, page)
        assert m.view(fu.pid).source["fusion"]["stars"] == {"mode": "palette"}
        assert "star_removal" in m.all_stages(m.view(img.pid))        # it gets its Stars/background, as a stack does
        assert m.fusion_problem(m.view(fu.pid)) == ""
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_fusion_can_start_from_a_linear_image_alone(qapp, tmp_path, monkeypatch):
    """Issue #55: no stack in the project: once its file is chosen, a linear Image gives the Combine channels its channels (dual
    band), with its stars (a star source is chosen in the Combine channels panel)."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        page.palette_action("image")
        settle(w, page)
        img = m.paths[-1]
        assert page.palette_reason("fusion", img, None) == "needs 2 filter workflows, a colour stack or image"
        m.set_image_file(img, _a_fits(tmp_path / "stack.fits"))
        settle(w, page)
        assert page.palette_reason("fusion", m.view(img.pid), None) == ""
        page.place_fusion()
        settle(w, page)
        f = m.paths[-1].source["fusion"]
        assert [(i["pid"], i["channel"]) for i in f["inputs"]] == [(img.pid, "R"), (img.pid, "GB")] and "stars" not in f
        assert m.fusion_problem(m.paths[-1]) == ""
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_linear_image_takes_stars_without_denoise(qapp, tmp_path, monkeypatch):
    """Issue #55: Stars on a linear Image, from the palette, needs no Denoise: its chain has the Output, then the
    network; the frames keep needing Denoise."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        page.palette_action("image")
        settle(w, page)
        img = m.paths[-1]
        m.set_image_file(img, _a_fits(tmp_path / "stack.fits"))
        settle(w, page)
        assert page.palette_reason("stars", m.view(img.pid), None) == ""
        page.place_block("stars", img.pid)
        settle(w, page)
        v = m.view(img.pid)
        assert "etoiles" in m.display_keys(v) and m.all_stages(v)[-2:] == ["render_simple", "star_removal"]
        assert "denoise" not in m.all_stages(v)
        with pytest.raises(wfl.BlockRefused, match="Denoise"):
            wfl.check_blocks([wfl.node("calibration"), wfl.node("registration"), wfl.node("selection"),
                              wfl.node("stacking"), wfl.node("output"), wfl.node("stars")], "frames")
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_linear_image_chosen_in_a_starless_fusion_gets_its_stars(qapp, tmp_path, monkeypatch):
    """Issue #55: a channel set to a linear Image while the Combine channels takes starless images: the Image gets its
    Stars/background (taken back with the Combine channels, decision 2)."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.palette_action("image:new")
        settle(w, page)
        img = m.paths[-1]
        m.set_image_file(img, _a_fits(tmp_path / "stack.fits"))
        page.place_block("astrometry", img.pid)
        settle(w, page)
        page.place_fusion()
        settle(w, page)
        fu = m.paths[-1]
        assert fu.source["fusion"]["stars"]["mode"] == "workflow"
        assert m.set_fusion_channel(fu, "OIII", img.pid)
        settle(w, page)
        assert "star_removal" in m.all_stages(m.view(img.pid)) and m.fusion_problem(m.view(fu.pid)) == ""
        m.remove_fusion(m.view(fu.pid))
        settle(w, page)
        assert "star_removal" not in m.all_stages(m.view(img.pid))
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_linear_image_is_a_fusion_star_source_once_it_has_its_astrometry(qapp, tmp_path, monkeypatch):
    """Issue #55: the Combine channels's Stars list offers a linear Image: choosing it gives it its Stars/background, and until
    it has its Astrometry the Combine channels says so (as for a channel) and Run is refused; then the engine reads its starless
    result and its plate solution. An image already stretched is listed greyed, with its reason, and refused."""
    from PySide6.QtWidgets import QComboBox

    from astro_gpu.core.store import Keys
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.palette_action("image:new")
        settle(w, page)
        img = m.paths[-1]
        m.set_image_file(img, _a_fits(tmp_path / "stack.fits"))
        stretched = m.add_image_workflow("Done", _a_fits(tmp_path / "done.fits"), stretched=True)
        settle(w, page)
        page.place_fusion()
        settle(w, page)
        fu = m.paths[-1]
        page.select(index=m.paths.index(fu), step="fusion")
        settle(w, page)
        stars = next(c for c in page.panel.findChildren(QComboBox) if c.itemText(0) == "With channels (no split)")
        at = {stars.itemData(k).get("pid"): k for k in range(stars.count()) if stars.itemData(k)}
        grey = stars.model().item(at[stretched.pid])
        assert not grey.isEnabled() and "stretch" in grey.toolTip()
        assert stars.model().item(at[img.pid]).isEnabled()
        assert not m.set_fusion_stars(m.view(fu.pid), {"mode": "workflow", "pid": stretched.pid})
        i18n.set_language("fr")
        try:
            assert "étirement" in next(why for q, why in m.star_choices(m.view(fu.pid)) if q.pid == stretched.pid)
        finally:
            i18n.set_language("en")
        stars.activated.emit(at[img.pid])
        settle(w, page)
        fu = m.view(fu.pid)
        assert fu.source["fusion"]["stars"] == {"mode": "workflow", "pid": img.pid}
        assert "star_removal" in m.all_stages(m.view(img.pid))        # the Stars block it needs, as for a stack
        assert m.fusion_problem(fu) == "Stars: needs Astrometry"
        assert (fu.name, "Stars: needs Astrometry") in m.run_problems()
        page.place_block("astrometry", img.pid)
        settle(w, page)
        fu = m.view(fu.pid)
        assert m.fusion_problem(fu) == ""
        k = Keys(m.p)
        got = k.overrides(fu.pid, "fusion")["stars"]
        assert k.key(img.pid, "star_removal") in got["dir"] and got["wcs"].endswith("wcs.fits")
        v = m.view(img.pid)                                            # a star source keeps what the Combine channels reads
        assert "needed by" in m.removal_problem(v, "astrometrie") and fu.name in m.channel_problem(v)
    finally:
        page.close()
        m.close(save=False)
        w.stop()
