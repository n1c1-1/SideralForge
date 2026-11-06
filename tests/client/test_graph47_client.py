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


def open_page(tmp_path, monkeypatch, workflows=("Full",)):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    p.sessions.append(project2.Session(1, "night 1", lights=[str(frames(tmp_path / "raw" / "n1", ["a1.ARW"]))]))
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
        d = page.new_workflow(tab="mono")
        d.pick(next(c.t for c in d.cards if c.t["key"] == "hoo"))
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
    monochrome, a refused delete keeps the selection, a fusion's channels are not Post layers by themselves, no
    duration for the blank page."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        page.place_block("calibration")                            # sessions, no workflow: the block starts one
        settle(w, page)
        assert len(m.paths) == 1 and [n["block"] for n in m.p.workflows[0].blocks] == ["calibration"]
        d = page.new_workflow()
        assert d.choice["key"] == "full"
        d.pick(next(c.t for c in d.cards if c.t["key"] == "blank"))
        assert d.duration.text() == ""                             # no duration for an empty page
        d.pick(next(c.t for c in d.cards if c.t["key"] == "classic"))
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
        assert m.palette_reason(ha, "color").startswith("a monochrome image") and m.place_indices(ha, "color") == []
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_template_takes_the_place_of_an_empty_workflow_and_channels_are_not_post_layers(qapp, tmp_path, monkeypatch):
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        assert m.ensure_frames_workflow() is not None and m.p.workflows[0].blocks == []
        d = page.new_workflow(tab="mono")
        d.pick(next(c.t for c in d.cards if c.t["key"] == "hoo"))
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
