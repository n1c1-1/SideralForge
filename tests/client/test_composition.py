"""The Stars block's images in the composition: the layers Post gives a workflow, the model's edits of Post's layers,
the graph's Stars and background blocks, and the parts of postengine. Toy data, offscreen, no frame read.

The model works on an empty project (no frame) with two workflows added in memory: Full (with the Stars block) and
Classic stack (without). It is read-only: nothing is written.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import postengine  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402
from astro_gpu.client.model import POST_PART_NAMES, ProjectModel  # noqa: E402
from astro_gpu.client.postengine import (  # noqa: E402
    BLENDS,
    MODES,
    Composer,
    load_layers,
    new_workflow_layer,
    save_layers,
    source_id,
)
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def two(qapp, tmp_path_factory):
    """A model on an empty project with Full (Stars) and Classic stack (no Stars), read-only."""
    root = tmp_path_factory.mktemp("composition") / "P"
    p = project2.create(root, "P", Target(name="M 1"))
    api.add_workflow(p, "Full", "Full")
    api.add_workflow(p, "Classic stack", "Classic stack")
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert w.wait(120) and m.ready, m.error
    yield m
    w.stop()


@pytest.fixture()
def model(two):
    two.p.post = {}
    yield two
    two.p.post = {}


def _full(m):
    return next(v for v in m.paths if v.name == "Full")


def _classic(m):
    return next(v for v in m.paths if v.name == "Classic stack")


# ---------------------------------------------------------------------------------------------- the layers Post gives
def test_the_workflows_are_full_with_stars_and_classic_without(model):
    assert _full(model).opts.get("etoiles") == "retirer" and _classic(model).opts.get("etoiles") != "retirer"


def test_a_stars_workflow_comes_as_its_stars_and_background_block_and_the_others_as_their_output(model):
    """Before Post is opened, the graph counts the layer Post will give each workflow: with Stars, its Stars and
    background block (a block of its own, no layer id yet); without, its Output image."""
    g = Graph(model)
    full, classic = _full(model), _classic(model)
    assert [n.paths for n in g.compositions] == [[full]] and g.compositions[0].lid is None
    assert g.compositions[0].parent is g.node_of(full, "etoiles")
    assert g.to_post == {classic.pid} and g.direct == []
    assert sorted(g.post_layers) == sorted([("Full · Stars and background", full.pid),
                                            ("Classic stack", classic.pid)])
    assert g.summary(g.final) == "2 layers"
    assert g.summary(g.node_of(full, "post")) == "to Stars"                # its image goes to the Stars block
    assert g.summary(g.node_of(classic, "post")) == "to Post"
    assert g.summary(g.compositions[0]) == "stars in Screen"


def test_the_default_layer_of_a_stars_workflow_puts_the_stars_back_in_screen():
    lay = new_workflow_layer(1, "Full · " + POST_PART_NAMES["combined"], True, part="combined")
    assert lay["part"] == "combined" and lay["blend"] == "Screen" and lay["mode"] == "Normal"
    out = new_workflow_layer(2, "Classic stack", True)
    assert out["part"] is None and "blend" not in out and out["mode"] == "Normal"
    assert POST_PART_NAMES == {"starless": "Starless image", "stars": "Stars alone", "combined": "Stars and background"}


# ---------------------------------------------------------------------------------------------- the model's edits
def test_the_model_adds_changes_and_removes_a_post_layer_and_says_who_did(model):
    full = _full(model)
    said = []
    model.postChanged.connect(said.append)
    try:
        me = object()
        lid = model.add_post_layer(full.pid, "stars", me)
        assert said == [me]
        layers = model.p.post["layers"]
        assert [(x["pid"], x["part"], x["name"], x["mode"]) for x in layers] == [
            (full.pid, "stars", "Full · Stars alone", "Screen")]
        assert layers[0]["id"] == lid and layers[0]["visible"] and model.post_layer(lid)["part"] == "stars"
        top = model.add_post_layer(full.pid, "combined")                    # a new one goes on top
        assert said == [me, None] and [x["id"] for x in model.p.post["layers"]] == [top, lid]
        assert model.post_layer(top)["blend"] == "Screen"
        model.set_post_layer(top, me, blend="Linear dodge (add)", sliders={"star_strength": 3.0})
        assert said[-1] is me
        assert model.post_layer(top)["blend"] == "Linear dodge (add)"
        assert model.post_layer(top)["sliders"] == {"star_strength": 3.0}
        model.remove_post_layer(lid, me)
        assert said[-1] is me and [x["id"] for x in model.p.post["layers"]] == [top]
        assert model.post_layer(lid) is None
    finally:
        model.postChanged.disconnect(said.append)


def test_removing_a_layer_takes_its_own_filters_with_it(model):
    full = _full(model)
    lid = model.add_post_layer(full.pid, "starless")
    keep = model.add_post_layer(full.pid, None)
    flt = postengine.new_filter_layer("saturation", lid)
    model.set_post_layers([flt] + model.post_layers())
    model.remove_post_layer(lid)
    assert [x["id"] for x in model.p.post["layers"]] == [keep]


# ---------------------------------------------------------------------------------------------- the graph
def test_the_graph_has_a_block_per_stars_and_background_layer_and_sends_the_others_as_they_are(model):
    full, classic = _full(model), _classic(model)
    a = model.add_post_layer(full.pid, "combined")
    b = model.add_post_layer(full.pid, "combined")
    model.add_post_layer(full.pid, "starless")
    model.add_post_layer(full.pid, "stars")
    model.add_post_layer(full.pid, "stars")                                 # twice: one arrow
    model.set_post_layer(b, blend="Lighten")
    g = Graph(model)
    stars = g.node_of(full, "etoiles")
    assert [n.lid for n in g.compositions] == [b, a] and all(n.parent is stars for n in g.compositions)
    assert g.node_of(full, "composition", a).lid == a and g.node_of(full, "composition").lid == b
    assert g.direct == [(stars, "stars"), (stars, "starless")]               # Post's order, top first
    # Classic has no layer yet: Post will give it its Output image
    assert g.to_post == {classic.pid}
    assert g.summary(g.node_of(full, "post")) == "to Stars"                 # Full's Output image is in no layer
    assert g.summary(g.node_of(full, "composition", b)) == "stars in Lighten"
    assert g.summary(g.node_of(full, "composition", a)) == "stars in Screen"
    assert g.summary(g.final) == "6 layers" and len(g.post_layers) == 6
    assert g.state(g.compositions[0]) == g.state(stars)                  # computed in Post from the Stars block's images
    model.add_post_layer(full.pid, None)                                    # its Output image as a layer
    g = Graph(model)
    assert full.pid in g.to_post and g.summary(g.node_of(full, "post")) == "to Post"


def test_an_image_layer_counts_in_the_post_block_without_a_workflow(model):
    full = _full(model)
    model.add_post_layer(full.pid, "combined")
    model.set_post_layers([postengine.new_image_layer("stack/abc", "Stack")] + model.post_layers())
    g = Graph(model)
    assert ("Stack", None) in g.post_layers and g.summary(g.final).startswith("3 ")


# ---------------------------------------------------------------------------------------------- postengine
def test_a_part_layer_asks_the_stars_block_for_its_images():
    assert source_id(new_workflow_layer(4, "Full")) == 4
    for part in ("starless", "stars", "combined"):
        assert source_id(new_workflow_layer(4, "Full", part=part)) == ("stars", 4)
    assert source_id(postengine.new_image_layer("stack/abc", "Stack")) == ("image", "stack/abc")
    assert new_workflow_layer(4, "Full", part="stars")["mode"] == "Screen"   # black around the stars
    assert new_workflow_layer(4, "Full", part="starless")["mode"] == "Normal"
    with pytest.raises(ValueError):
        new_workflow_layer(4, "Full", part="halos")


def test_load_keeps_a_valid_blend_and_drops_an_unknown_part():
    good = new_workflow_layer(1, "Full · Stars and background", True, part="combined")
    good["blend"] = "Linear dodge (add)"
    odd = new_workflow_layer(1, "Full · Stars and background", True, part="combined")
    odd["blend"] = "Overlay"
    gone = new_workflow_layer(1, "Full · halos", True)
    gone["part"] = "halos"                                                  # a part this version does not know
    other = new_workflow_layer(2, "Classic", True)                          # its workflow is gone
    kept = load_layers(save_layers([good, odd, gone, other]), {1})
    assert [(x["id"], x.get("blend")) for x in kept] == [(good["id"], "Linear dodge (add)"), (odd["id"], BLENDS[0])]


def test_linear_dodge_adds_the_layers():
    assert "Linear dodge (add)" in MODES
    rng = np.random.default_rng(1)
    a = rng.random((8, 8, 3)).astype(np.float32)
    b = rng.random((8, 8, 3)).astype(np.float32) * 0.5
    got = Composer.blend("Linear dodge (add)", a, b)
    assert np.allclose(got, np.clip(a + b, 0, 1), atol=1e-6) and got.max() == 1.0
    assert np.allclose(Composer.blend("Screen", a, b), 1 - (1 - a) * (1 - b), atol=1e-6)
    assert BLENDS == ("Screen", "Linear dodge (add)", "Lighten") and set(BLENDS) <= set(MODES)
