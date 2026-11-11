"""THE OUTPUT IS NEVER A BLOCK OF THE GRAPH (issues #17, #49, GUI exploration 2, and 11 November: it came back a fourth
time, behind the Synthetic stars block). A workflow's Output lives in the wire of the block that holds it; whatever
block is added to the palette or placed after it, the graph draws no Output block.

If this file fails, a change put the Output block back in the graph: fix the change, never this test. A new block
drawn after the Output is taken care of by ``Graph._fold_outputs``; a new kind of workflow must be added HERE."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from astro_gpu.client.blocks import Graph, node_title  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from tests.client.test_fusion_sessions_client import done, open_page, qapp, settle  # noqa: E402, F401
from tests.client.test_image_source_fixes import fits_file, picture  # noqa: E402


def no_output_block(m, page=None, what=""):
    """No node of the graph is an Output, each workflow's Output is held by a block of another kind, and the page draws
    no block named Output."""
    g = Graph(m)
    assert [n for n in g.nodes if n.key == "post"] == [], f"an Output block is drawn ({what})"
    for q in m.paths:
        held = g.node_of(q, "post")
        assert held is None or held.key != "post", f"{q.name}: its Output is a block of its own ({what})"
    assert all(node_title(m, n) != "Output" for n in g.nodes), f"a block is named Output ({what})"
    if page is not None:
        page.refresh()
        drawn = [n for n in page.graph.graph.nodes if n.key == "post"]
        assert drawn == [], f"the page draws an Output block ({what})"


def test_no_template_and_no_block_placed_after_it_draws_an_output_block(qapp, tmp_path, monkeypatch):  # noqa: F811
    """Every template, then every block of the palette placed in each workflow, one after the other: no Output block
    at any step. The palette is read from the engine: a block added to it tomorrow is tried here by itself."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=tuple(wfl.TEMPLATES))
    try:
        settle(w, page)
        no_output_block(m, page, "the templates")
        for i in range(len(m.paths)):
            for block in wfl.PALETTE:
                q = m.paths[i]
                try:
                    m.insert_block(q, block)
                except ValueError:                           # it cannot go there: nothing placed
                    continue
                settle(w, page)
                no_output_block(m, page, f"{block} placed in {q.name}")
        placed = {n["block"] for q in m.paths for n in m.graph(q)}
        assert {"stars", "synth"} <= placed                  # the case of 11 November was reached
    finally:
        done(w, m, page)


def test_synthetic_stars_after_stars_and_background_draw_no_output_block(qapp, tmp_path, monkeypatch):  # noqa: F811
    """11 November: Stars/background, then Synthetic stars: the Output showed between them and Combine layers."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        q = m.paths[0]
        m.insert_block(q, "synth")
        settle(w, page)
        q = m.paths[0]
        g = Graph(m)
        stars, synth = g.node_of(q, "etoiles"), g.node_of(q, "synth")
        assert synth.parent is stars and g.node_of(q, "post") is stars       # held by Stars/background, as always (#17)
        no_output_block(m, page, "Synthetic stars after Stars/background")
    finally:
        done(w, m, page)


def test_images_and_their_blocks_draw_no_output_block(qapp, tmp_path, monkeypatch):  # noqa: F811
    """Workflows that start from an image: stretched (no block at all), linear, two of each, with blocks placed."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        for n in ("a", "b"):
            q = m.add_image_workflow(n, str(picture(tmp_path / f"{n}.jpg")), stretched=True)
            m.set_graph(q, [])
        for n in ("c", "d"):
            m.add_image_workflow(n, str(fits_file(tmp_path / f"{n}.fits")))
        settle(w, page)
        no_output_block(m, page, "images")
        for i in range(len(m.paths)):
            for block in wfl.PALETTE:
                try:
                    m.insert_block(m.paths[i], block)
                except ValueError:
                    continue
                settle(w, page)
                no_output_block(m, page, f"{block} placed in {m.paths[i].name}")
    finally:
        done(w, m, page)


def test_filter_workflows_and_their_fusion_draw_no_output_block(qapp, tmp_path, monkeypatch):  # noqa: F811
    w, m, page = open_page(tmp_path, monkeypatch, filters=("Ha", "OIII"))
    try:
        settle(w, page)
        no_output_block(m, page, "two filters")
        page.place_fusion()
        settle(w, page)
        no_output_block(m, page, "their fusion")
    finally:
        done(w, m, page)
