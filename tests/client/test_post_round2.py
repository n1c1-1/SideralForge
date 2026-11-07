"""GUI exploration, round 2 (Post page side): the page lists the layers it will have before anything is computed; a
stretched Image's layer reaches Post; a layer deleted in Post takes its sent branch away, and Undo brings both back
with the layer's settings; the Output's channels setting undone shows its old value; the Split images' names
translated, numbered ones too. A real model on an empty project, read-only, offscreen; nothing is computed.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client import model as modelmod  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.post3 import MODEL_STEP, PostPage3, tr_name  # noqa: E402
from astro_gpu.client.services import render as rsv  # noqa: E402
from astro_gpu.client.services.render import RenderWorker  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def m(qapp, tmp_path):
    """A model with Classic stack (no Stars) and Raw stack (its Stacking's image sent to Post: a workflow of its own)."""
    root = tmp_path / "P"
    p = project2.create(root, "P", Target(name="M 1"))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a1.ARW").write_bytes(b"\0" * 100)
    p.sessions.append(project2.Session(1, "night 1", lights=[str(raw)]))
    api.add_workflow(p, "Classic stack", "Classic stack")
    w = ApiWorker()
    model = ProjectModel(p, w, readonly=True)
    assert w.wait(120) and model.ready, model.error
    yield model
    w.stop()


def _rendered(model, on=True):
    """The workflows as if computed (the page then shows them)."""
    orig = model._sync

    def fake(*a, **k):
        orig(*a, **k)
        for v in model.paths:
            v.rendered = on
    model._sync = fake
    model._sync()


def _page(model):
    return PostPage3(model, worker=RenderWorker(None, resolver=lambda sid: None, device="cpu"))


def test_nothing_computed_the_page_lists_the_layers_the_graph_counts(m):
    page = _page(m)
    try:
        assert page.pages.currentIndex() == 0
        texts = [lb.text() for lb in page.empty.findChildren(QLabel) if not lb.isHidden()]
        assert "Nothing computed yet" in texts and "Classic stack  ·  not computed yet" in texts
    finally:
        page.shutdown()
        page.deleteLater()


def test_a_layer_deleted_in_post_takes_its_sent_branch_and_undo_brings_both_back_with_its_settings(m):
    classic = next(v for v in m.paths if v.name == "Classic stack")
    m.add_exit(classic, "stacking", "Raw stack")
    _rendered(m)
    page = _page(m)
    v = page.view
    told = []
    v.toasted.connect(told.append)
    try:
        assert page.pages.currentIndex() == 1
        lay = next(x for x in v.layers if x["name"] == "Raw stack")
        assert lay["visible"]                             # a block's image sent: shown at once, as the Split images
        v._set(lay, "mode", "Screen")
        v.flush()
        v.sel = (lay["id"], "layer")
        v.delete()
        assert "Raw stack" not in [q.name for q in m.paths]               # its branch went with it
        assert v.undo[-1] is MODEL_STEP and told and "Ctrl+Z" in told[-1]
        v.do_undo()
        assert "Raw stack" in [q.name for q in m.paths]
        back = [(x["name"], x["mode"]) for x in v.layers if x["name"] == "Raw stack"]
        assert back == [("Raw stack", "Screen")]
        assert MODEL_STEP not in v.undo[-1:] or len(v.undo) == 1          # no step of Post's own from the Undo
        # the graph's side: the branch's wire removed (the branch goes), then Ctrl+Z: the layer with its settings
        raw = next(q for q in m.paths if q.name == "Raw stack")
        m.remove_path(m.paths.index(raw))
        assert "Raw stack" not in [x["name"] for x in v.layers]
        m.undo()
        assert [(x["name"], x["mode"]) for x in v.layers if x["name"] == "Raw stack"] == [("Raw stack", "Screen")]
    finally:
        page.shutdown()
        page.deleteLater()


def test_an_output_setting_undone_shows_its_old_value_at_once(m):
    classic = next(v for v in m.paths if v.name == "Classic stack")
    row = SimpleNamespace(name="linked", value=True, suggested=True, origin="default")
    m._pc[(classic.pid, "render_simple")] = {"key": "k", "epoch": m._epoch, "rows": [row], "asked": None, "tries": 0}
    m.set_param(classic, "render_simple", "linked", False)
    assert m.param_value(m.view(classic.pid), "render_simple", "linked") is False
    m.undo()
    # (the patched cached row showed the undone value until the worker answered: Post's box stayed unticked)
    assert m.param_value(m.view(classic.pid), "render_simple", "linked") is True
    m.redo()
    assert m.param_value(m.view(classic.pid), "render_simple", "linked") is False


def test_a_stretched_image_stands_for_the_output_its_layer_reaches_post(m, monkeypatch, tmp_path):
    """engine_shapes: a stretched Image + Split stars has no Output stage (the Image writes what it would); computed,
    its workflow is rendered (Post shows its layer) and the render worker finds its image in the Image's folder."""
    q = m.add_image_workflow("Image", str(tmp_path / "a.png"), stretched=True)
    chain = api.chain(m.p, q.pid)
    assert "render_simple" not in chain and chain[0] == "image"
    done = SimpleNamespace(state="done", previous_key="")
    real = api.states
    monkeypatch.setattr(modelmod.api, "states", lambda p, update=True: {**real(p, update=update),
                                                                         (q.pid, "image"): done})
    m._sync(prefetch=False)
    assert m.view(q.pid).rendered
    out = tmp_path / "image_out"
    out.mkdir()
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    (out / "prep_simple.json").write_text("{}", encoding="utf-8")
    got = {}

    def outputs(p, wid, stage, which="current"):
        got["stage"] = stage
        return SimpleNamespace(dir=str(out), which=which)
    monkeypatch.setattr(api, "stage_outputs", outputs)
    f = rsv.api_resolver(m.p)(q.pid)
    assert f is not None and got["stage"] == "image" and f.path == out
    (out / "prep_simple.json").unlink()                 # a linear image: no Output image of its own
    assert rsv.api_resolver(m.p)(q.pid) is None


def test_the_split_images_names_are_translated_numbered_ones_too():
    try:
        i18n.set_language("fr")
        assert tr_name("Workflow · Recombine") == "Workflow · Recombiner"
        assert tr_name("Workflow · Recombine 2") == "Workflow · Recombiner 2"
        assert tr_name("Workflow · Stars alone") == "Workflow · Étoiles seules"
        assert tr_name("Recombine 2 · Workflow 2") == "Recombiner 2 · Workflow 2"
    finally:
        i18n.set_language("en")


def test_control_round_the_post_page_rebuilt_never_answers_the_project_again(m, monkeypatch):
    """Control round: a Post page gone (a language switch builds it again) never answers the project's changes; its
    lambda did, and the next Ctrl+Z raised "QStackedWidget already deleted"."""
    import sys

    from PySide6.QtCore import QEvent
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: errors.append(a))
    page = _page(m)
    page.shutdown()
    page.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    m.postChanged.emit(None)
    m.pathsChanged.emit()
    QApplication.processEvents()
    assert errors == []


def test_control_round_a_layer_renamed_in_post_renames_its_workflow_when_it_had_its_name(m):
    _rendered(m)
    page = _page(m)
    v = page.view
    told = []
    v.toasted.connect(told.append)
    try:
        lay = next(x for x in v.layers if x["name"] == "Classic stack")
        pid = lay["pid"]
        v.rename(lay, "Background only")
        assert m.view(pid).name == "Background only"                       # the workflow (and its wire) follow
        assert [x["name"] for x in v.layers if x.get("pid") == pid] == ["Background only"]
        assert v.undo[-1] is MODEL_STEP and told and "Ctrl+Z" in told[-1]
        v.do_undo()                                                         # one step brings both back
        assert m.view(pid).name == "Classic stack"
        assert [x["name"] for x in v.layers if x.get("pid") == pid] == ["Classic stack"]
        lay = next(x for x in v.layers if x.get("pid") == pid)
        lay["name"] = "My own"                                              # a name of the user's: the layer alone
        v.rename(lay, "Other")
        assert m.view(pid).name == "Classic stack" and lay["name"] == "Other"
    finally:
        page.shutdown()
        page.deleteLater()


def test_control_round_delete_greyed_on_the_last_layer_says_why_and_the_star_notes_name_the_block(m):
    from astro_gpu.client.post3 import PART_NOTES
    _rendered(m)
    page = _page(m)
    v = page.view
    try:
        lay = next(x for x in v.layers if x["kind"] == "workflow")
        v.select(lay["id"], "layer")
        v.rebuild()
        assert not v.b_del.isEnabled() and "last image layer" in v.b_del.toolTip()
        assert all("Stars/background block" in PART_NOTES[k] for k in ("starless", "stars"))
    finally:
        page.shutdown()
        page.deleteLater()
