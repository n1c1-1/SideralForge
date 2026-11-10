"""Several versions of Post in a project (issue #82): what the model keeps true whatever is done."""
import json
import random

import pytest
from PySide6.QtWidgets import QApplication

from astro_gpu import api
from astro_gpu.client.model import ProjectModel
from astro_gpu.client.services.worker import ApiWorker
from astro_gpu.core import project2
from astro_gpu.core.project import Target
from astro_gpu.core.project2 import active_version, upgraded_post, versions_problem, workflow_images


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def two(qapp, tmp_path_factory):
    root = tmp_path_factory.mktemp("versions") / "P"
    p = project2.create(root, "P", Target(name="M 1"))
    raw = root.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "a1.ARW").write_bytes(b"\0" * 100)
    p.sessions.append(project2.Session(1, "night 1", lights=[str(raw)]))
    api.add_workflow(p, "Full", "Full")
    api.add_workflow(p, "Classic stack", "Classic stack")
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert w.wait(120) and m.ready, m.error
    yield m
    w.stop()


@pytest.fixture()
def model(two):
    two.p.post = project2.new_post()
    yield two
    two.p.post = project2.new_post()


def by_name(m, name):
    return next(v for v in m.paths if v.name == name)


def text(v):
    return json.dumps(v, sort_keys=True)


def test_a_project_saved_before_the_versions_opens_with_its_composition_as_the_first():
    old = {"layers": [{"id": 3, "kind": "workflow", "pid": 1, "part": None}], "crop": [1, 2, 3, 4], "removed": [[2, None]]}
    post = upgraded_post(old)
    assert "layers" not in post and "crop" not in post and post["removed"] == [[2, None]]
    v = active_version(post)
    assert v["name"] == "Version 1" and v["layers"] == old["layers"] and v["crop"] == [1, 2, 3, 4]
    assert upgraded_post(post) == post
    assert upgraded_post(None) == project2.new_post()


def test_every_change_says_which_versions_it_touches(model):
    with pytest.raises(TypeError):
        model.set_post_layers([])
    with pytest.raises(TypeError):
        model._post_edit(lambda layers: None)
    with pytest.raises(ValueError):
        model.set_post_layers([], scope="some")


def test_a_new_version_is_a_copy_with_ids_of_its_own_and_is_active(model):
    full = by_name(model, "Full")
    lid = model.add_post_layer(full.pid, "stars")
    first = model.post_version()["id"]
    vid = model.new_post_version()
    post = model.p.post
    assert post["active"] == vid and len(post["versions"]) == 2 and model.post_version()["name"] == "Version 2"
    assert versions_problem(post) is None
    copy = model.post_version()["layers"]
    source = next(v for v in post["versions"] if v["id"] == first)["layers"]
    assert [x["name"] for x in copy] == [x["name"] for x in source] and lid not in [x["id"] for x in copy]


def test_a_setting_touches_the_active_version_only(model):
    full = by_name(model, "Full")
    model.add_post_layer(full.pid, "stars")
    model.new_post_version()
    other = next(v for v in model.p.post["versions"] if v["id"] != model.p.post["active"])
    before = text(other)
    lid = model.post_version()["layers"][0]["id"]
    model.set_post_layer(lid, opacity=40.0)
    assert model.post_layer(lid)["opacity"] == 40.0 and text(other) == before


def test_a_workflow_image_added_or_taken_out_touches_every_version(model):
    full = by_name(model, "Full")
    model.add_post_layer(full.pid, "stars")
    model.new_post_version()
    model.add_post_layer(full.pid, "starless")
    post = model.p.post
    assert len({text(sorted(workflow_images(v).items(), key=str)) for v in post["versions"]}) == 1
    assert (full.pid, "starless") in workflow_images(post["versions"][0]) and versions_problem(post) is None
    model.remove_post_part(full.pid, "stars")
    assert all((full.pid, "stars") not in workflow_images(v) for v in model.p.post["versions"])
    assert versions_problem(model.p.post) is None


def test_the_last_version_stays_and_names_do_not_repeat(model):
    only = model.post_version()["id"]
    model.delete_post_version(only)
    assert [v["id"] for v in model.p.post["versions"]] == [only]
    vid = model.new_post_version()
    model.rename_post_version(vid, "Version 1")
    assert model.post_version()["name"] == "Version 2"
    model.rename_post_version(vid, "  Bold  ")
    assert model.post_version()["name"] == "Bold" and model.post_version_label() == "Bold"
    model.delete_post_version(vid)
    assert model.p.post["active"] == only and model.post_version_label() == ""


def test_the_save_says_when_the_versions_disagree_and_writes_all_the_same(model, monkeypatch):
    full = by_name(model, "Full")
    model.add_post_layer(full.pid, "stars")
    model.new_post_version()
    model.post_version()["layers"] = []
    saved, said = [], []
    monkeypatch.setattr(api, "save", lambda p: saved.append(True))
    monkeypatch.setattr(model, "log", lambda template, **a: said.append(a["problem"]))
    monkeypatch.setattr(model, "readonly", False)
    monkeypatch.setattr(model, "_dirty", True)
    model.flush()
    assert saved == [True] and "do not hold the same workflows" in said[0]


@pytest.mark.parametrize("seed", range(6))
def test_long_random_sequences_keep_the_versions_true(model, seed):
    """Graph actions, version actions, undo, save and reload, in any order."""
    r = random.Random(seed)
    m = model
    ids = [v.pid for v in m.paths]
    parts = ("stars", "starless", "combined")
    for step in range(120):
        act = r.choice(("add", "add", "remove", "new", "switch", "delete", "set", "set", "undo", "reload"))
        versions = m.p.post["versions"]
        before = {v["id"]: text(v) for v in versions if v["id"] != m.p.post["active"]}
        if act == "add":
            m.add_post_layer(r.choice(ids), r.choice(parts))
        elif act == "remove":
            m.remove_post_part(r.choice(ids), r.choice(parts))
        elif act == "new" and len(versions) < 5:
            m.new_post_version()
        elif act == "switch":
            m.set_post_version(r.choice(versions)["id"])
        elif act == "delete":
            m.delete_post_version(r.choice(versions)["id"])
        elif act == "set" and m.post_version()["layers"]:
            m.set_post_layer(r.choice(m.post_version()["layers"])["id"], opacity=float(r.randrange(100)))
            assert {v["id"]: text(v) for v in m.p.post["versions"] if v["id"] != m.p.post["active"]} == before, step
        elif act == "undo":
            snap = m._undo[-1][4] if m._undo else None
            if m.undo() and snap is not None:
                assert text(json.loads(snap)) == text(m.p.post), step
        elif act == "reload":
            stored = text(m.p.post)
            assert text(upgraded_post(json.loads(stored))) == stored, step
        assert versions_problem(m.p.post) is None, (step, act)
        active_version(m.p.post)
