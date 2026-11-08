"""A workflow without lights: its wire to Post is drawn dashed, and never said to be a layer already. What a block does is
in its tooltip. A failure is a banner over the window, not a box."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGraphicsPathItem, QToolButton

from astro_gpu.client.app import Banner
from astro_gpu.client.blocks import Graph
from astro_gpu.client.graph import POST_WHAT, BlockItem

from test_graph47_client import open_page, qapp, settle  # noqa: F401  (a fixture)


def dashed(page):
    return [it for it in page.graph.scene().items()
            if isinstance(it, QGraphicsPathItem) and it.pen().style() == Qt.DashLine and "lights" in it.toolTip()]


def test_a_workflow_without_lights_has_a_dashed_wire_and_no_layer(qapp, tmp_path, monkeypatch):  # noqa: F811
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",), lights=False)
    try:
        settle(w, page)
        g = Graph(m)
        p = m.paths[0]
        assert not g.to_post and g.awaits_post == {p.pid} and not g.post_layers
        assert len(dashed(page)) == 1
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_to_post_on_the_last_block_says_lights_first_not_already_a_layer(qapp, tmp_path, monkeypatch):  # noqa: F811
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",), lights=False)
    said = []
    monkeypatch.setattr(page, "toast", said.append)
    try:
        settle(w, page)
        m.remove_node(m.paths[0], "etoiles")
        settle(w, page)
        g = Graph(m)
        p = m.paths[0]
        page.send_to_post(g.leaves[p.pid])
        assert said and "lights first" in said[-1] and "already" not in said[-1]
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_lights_turn_the_dashed_wire_into_a_layer(qapp, tmp_path, monkeypatch):  # noqa: F811
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",), lights=True)
    try:
        settle(w, page)
        g = Graph(m)
        assert not g.awaits_post and not dashed(page)
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_every_block_says_what_it_does(qapp, tmp_path, monkeypatch):  # noqa: F811
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        items = {it.node.key: it for it in page.graph.scene().items() if isinstance(it, BlockItem)}
        assert POST_WHAT.split(":")[0] in items["final"].toolTip()
        assert "half-stacks" in items["debruitage"].toolTip()
        assert "One night" in items["sources"].toolTip()
        assert all(len(line) <= 90 for it in items.values() for line in it.toolTip().split("\n")[:3])
    finally:
        page.close()
        m.close(save=False)
        w.stop()


def test_a_banner_stays_until_it_is_closed(qapp):  # noqa: F811
    b = Banner()
    assert b.isHidden()
    b.say("The project cannot be saved")
    assert not b.isHidden() and b.text.text() == "The project cannot be saved"
    QApplication.processEvents()
    assert not b.isHidden()
    b.findChild(QToolButton).click()
    assert b.isHidden()
