"""Issue #47, workflows as free graphs, in the engine: the graph of a workflow made before has its chain and its keys;
the placement rules (what each block takes and gives, the blocks a block needs further up); repeated and reordered
image blocks get their own entries and keys."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from astro_gpu.core import project2
from astro_gpu.core import workflow as wfl
from astro_gpu.core.project import Target
from astro_gpu.core.project2 import Session
from astro_gpu.core.store import Keys
from astro_gpu.core.workflow import TEMPLATES, BlockRefused, node


def frames(folder: Path, names: dict[str, int]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for n, size in names.items():
        (folder / n).write_bytes(b"\0" * size)
    return folder


@pytest.fixture
def nights(tmp_path):
    raw = tmp_path / "raw"
    return {"n1": frames(raw / "n1", {"a1.ARW": 100, "a2.ARW": 101}), "n2": frames(raw / "n2", {"b1.ARW": 102}),
            "darks": frames(raw / "darks", {"d1.ARW": 103}), "flats": frames(raw / "flats", {"f1.ARW": 104})}


def project(root: Path, nights: dict, mode: str = "together") -> project2.Project2:
    p = project2.create(root, "P", Target(name="M 31"))
    p.stack_mode = mode
    for i, n in enumerate(("n1", "n2"), 1):
        p.sessions.append(Session(i, n, lights=[str(nights[n])], darks=[str(nights["darks"])],
                                  flats=[str(nights["flats"])]))
    return p


CASES = [
    ("Full", {}, None, "together"),
    ("Classic stack", {}, None, "together"),
    ("Classic + denoise", {}, None, "per_session"),
    ("Full", {"cut": ["calibration", "registration"]}, None, "together"),
    ("Full", {"cut": ["gradient"]}, None, "together"),
    ("Full", {"cut": ["color"]}, None, "per_session"),
    ("Full", {"cut": ["astrometry", "color"]}, None, "together"),
    ("Classic stack", {}, "stacking", "together"),
    ("Full", {"gradient": "rbf", "denoise": "adapted"}, None, "together"),
]


@pytest.mark.parametrize("template,extra,end,mode", CASES)
def test_the_graph_of_a_workflow_made_before_has_its_chain_and_its_keys(tmp_path, nights, template, extra, end, mode):
    p = project(tmp_path / "p", nights, mode)
    w = p.add_workflow("W", {**TEMPLATES[template], **extra}, end=end)
    before = Keys(p).keys(w.id)
    g = wfl.graph_of(w)
    p.set_blocks(w.id, g)
    assert w.blocks is not None and w.end is None
    after = Keys(p).keys(w.id)
    assert after == before                                      # the same stages, the same keys
    project2.save(p)
    q = project2.load(p.root)
    assert Keys(q).keys(w.id) == before and q.workflow(w.id).blocks == g


def test_an_image_source_and_a_fusion_keep_their_keys_as_graphs(tmp_path, nights):
    p = project(tmp_path / "p", nights)
    a = p.add_workflow("A", TEMPLATES["Full"])
    img = tmp_path / "m42.fits"
    img.write_bytes(b"\0" * 2880)
    i = p.add_workflow("I", TEMPLATES["Classic stack"])
    i.source = {"path": str(img), "stretched": False}
    f = p.add_workflow("F", {**TEMPLATES["Full"]})
    f.source = {"fusion": {"palette": "HOO", "stars": {"mode": "palette"},
                           "inputs": [{"pid": a.id, "role": "Ha", "channel": "R"},
                                      {"pid": a.id, "role": "OIII", "channel": "GB"}]}}
    for w in (i, f):
        before = Keys(p).keys(w.id)
        p.set_blocks(w.id, wfl.graph_of(w))
        assert Keys(p).keys(w.id) == before


def test_the_placement_rules():
    g = wfl.from_options(TEMPLATES["Full"])
    assert [n["block"] for n in g] == ["calibration", "registration", "selection", "stacking", "astrometry",
                                       "gradient", "denoise", "color", "output", "stars"]
    # image blocks: anywhere among the image blocks before the Output, more than once
    assert wfl.places(g, "frames", "denoise") == [4, 5, 6, 7, 8, 9]       # 8 and 9: the Output moves itself (#49)
    assert wfl.places(g, "frames", "gradient") == [4, 5, 6, 7, 8, 9]
    # Color needs Astrometry before it
    assert wfl.places(g, "frames", "color") == [5, 6, 7, 8, 9]
    e = wfl.place_problem(g, "frames", "color", 4)
    assert e is not None and e.template == wfl.WHY_NEEDS and "Astrometry" in str(e)
    # a block that is there once: nowhere, with the reason the palette shows
    e = wfl.palette_problem(g, "frames", "stacking")
    assert e is not None and e.template == wfl.WHY_ONCE
    # the frame blocks keep their order; a frame block after the stack takes the wrong data
    e = wfl.place_problem(g, "frames", "denoise", 2)
    assert e is not None and e.template == wfl.WHY_INPUT
    # Denoise needs the half-stacks of a stack: not on an image file
    img = [node("astrometry"), node("output")]
    e = wfl.palette_problem(img, "image", "denoise")
    assert e is not None and e.template == wfl.WHY_HALVES
    assert wfl.palette_problem(img, "image", "stars").template == wfl.WHY_NEEDS            # Stars needs Denoise
    assert wfl.places(img, "image", "gradient") == [0, 1, 2]
    # removing: Stacking makes the image the next blocks take; Astrometry is needed by Color; Output always there
    assert wfl.remove_problem(g, "frames", 3).template == wfl.WHY_REMOVE_BRIDGE
    assert wfl.remove_problem(g, "frames", 4).template == wfl.WHY_REMOVE_NEEDED
    assert wfl.remove_problem(g, "frames", 8).template == wfl.WHY_OUTPUT_AUTO             # made automatically (#49)
    assert wfl.remove_problem(g, "frames", 6).template == wfl.WHY_REMOVE_NEEDED            # Stars needs Denoise
    assert wfl.remove_problem(g, "frames", 1) is None                                      # no Registration: trails
    # switching off: only a block whose output is its input's kind
    assert wfl.off_problem(g, "frames", 3).template == wfl.WHY_OFF_KIND
    assert wfl.off_problem(g, "frames", 9).template == wfl.WHY_OFF_KIND
    assert wfl.off_problem(g, "frames", 5) is None
    # a double-click: the block's usual place, or the nearest valid one
    assert wfl.insert_place(g, "frames", "gradient") == 6
    blank: list = []
    assert wfl.palette_problem(blank, "frames", "gradient") is not None                     # no stack yet
    assert wfl.insert_place(blank, "frames", "calibration") == 0
    assert wfl.sequence_chain([node("calibration")]) == ["inventory", "calibrate"]       # a graph being built
    assert wfl.sequence_chain([node("stacking")]) == ["inventory", "calibrate", "register", "stack"]
    with pytest.raises(BlockRefused):
        wfl.check_blocks([node("stacking"), node("output"), node("gradient")], "frames")


def test_repeated_and_reordered_image_blocks_get_their_own_entries_and_keys(tmp_path, nights):
    p = project(tmp_path / "p", nights)
    w = p.add_workflow("W", TEMPLATES["Full"])
    ref = Keys(p).keys(w.id)
    g = wfl.graph_of(w)
    # Denoise before the Gradient, a second Denoise after it
    g2 = g[:5] + [node("denoise", name="denoise"), node("gradient"), node("denoise", name="denoise~2"),
                  node("color"), node("output"), node("stars")]
    p.set_blocks(w.id, g2)
    k = Keys(p)
    ch = k.chain(w.id)
    assert ch[-6:] == ["denoise", "gradient", "denoise~2", "color", "render_simple", "star_removal"]
    assert k.deps(w.id, "denoise") == ("stack",)                         # no Gradient before it
    assert set(k.deps(w.id, "gradient")) == {"stack", "surveys", "denoise"}   # fitted on the denoised image
    assert set(k.deps(w.id, "denoise~2")) == {"stack", "denoise", "gradient"}
    assert set(k.deps(w.id, "render_simple")) == {"stack", "denoise", "gradient", "denoise~2", "color"}
    assert k.key_params(w.id, "denoise").get("keep_halves") is True       # the later Denoise reads its halves
    assert "keep_halves" not in k.key_params(w.id, "denoise~2")
    assert k.key(w.id, "stack") == ref["stack"]                            # what comes before is shared
    assert k.key(w.id, "render_simple") != ref["render_simple"]
    # a switched-off block: as if absent; Calibration switched off runs as a pass-through
    g3 = [dict(n, off=n["name"] == "denoise~2") for n in g2]
    p.set_blocks(w.id, g3)
    assert "denoise~2" not in Keys(p).chain(w.id)
    assert "keep_halves" not in Keys(p).key_params(w.id, "denoise")
    g4 = [dict(n, off=n["block"] == "calibration") for n in g]
    p.set_blocks(w.id, g4)
    assert Keys(p).key_params(w.id, "calibrate").get("apply_masters") is False


def test_two_nights_with_frames_of_the_same_names_sizes_and_dates_have_their_own_keys(tmp_path):
    """Issue #56: capture programs number every night's frames alike ("Light_001.fits", the same size; a camera whose
    clock was reset gives them the same dates too). A frame of a session after the first carries its session's id:
    the two nights are not taken for the same frames. The first night keeps its key: a project of one session is
    unchanged, and adding a night does not make the first one Outdated."""
    raw = tmp_path / "raw"
    n1, n2 = (frames(raw / f / "lights", {"Light_001.fits": 2880, "Light_002.fits": 2880}) for f in ("n1", "n2"))
    for f in list(n1.iterdir()) + list(n2.iterdir()):
        os.utime(f, (1388534400, 1388534400))                    # 1 January 2014, as a dead clock writes it
    alone = project2.create(tmp_path / "a", "A", Target(name="M 31"))
    alone.sessions.append(Session(1, "night 1", lights=[str(n1)]))
    alone.add_workflow("W", TEMPLATES["Classic stack"])
    before = Keys(alone).key(1, "stack")
    p = project2.create(tmp_path / "p", "P", Target(name="M 31"))
    p.stack_mode = "per_session"
    p.sessions += [Session(1, "night 1", lights=[str(n1)]), Session(2, "night 2", lights=[str(n2)])]
    a = p.add_workflow("Night 1", TEMPLATES["Classic stack"])
    a.sessions = [1]
    b = p.add_workflow("Night 2", TEMPLATES["Classic stack"])
    b.sessions = [2]
    k = Keys(p)
    assert k.key(a.id, "stack") != k.key(b.id, "stack")          # two nights, two stacks
    assert k.key(a.id, "stack") == before                         # the first night keeps its key
    both = p.add_workflow("Both", TEMPLATES["Classic stack"])       # the two nights together, calibrated each alone
    assert Keys(p).key(both.id, "calibrate@1") == Keys(alone).key(1, "calibrate")
    assert Keys(p).key(both.id, "calibrate@2") != Keys(p).key(both.id, "calibrate@1")


@pytest.mark.parametrize("second_first", [False, True])
def test_a_block_placed_twice_runs_for_real_each_in_its_own_folder(tmp_path, second_first):
    """Two Gradient blocks one after the other, from an image (real stages, on the processor): each job's manifest
    in its own folder (the later block's used to go to the first one's, or nowhere: its run failed), in either order
    of placing; the first block's result stays as it was."""
    import json

    import numpy as np
    from astropy.io import fits

    from astro_gpu.core import runner2
    from astro_gpu.core.store import Store
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:120, 0:160].astype(np.float32)
    img = np.stack([0.1 + 0.0004 * xx + 0.0002 * yy + c * 0.01 for c in range(3)]) + rng.normal(0, 0.002, (3, 120, 160))
    f = tmp_path / "stack.fits"
    fits.writeto(f, img.astype(np.float32))
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    w = p.add_workflow("Image", TEMPLATES["Classic stack"])
    w.source = {"path": str(f), "stretched": False}
    p.set_blocks(w.id, [node("gradient"), node("gradient", "rbf", name="gradient~2")] if not second_first
                 else [node("gradient", "rbf", name="gradient~2"), node("gradient")])
    project2.save(p)
    plan = runner2.plan_all(p)
    assert [j.stage for j in plan.jobs if j.stage.startswith("gradient")] == \
        (["gradient", "gradient~2"] if not second_first else ["gradient~2", "gradient"])
    assert runner2.run_plan(p.root, runner2.prepare_run(p, plan, "cpu").plan_file, echo=False) == "done"
    k, store = Keys(project2.load(p.root)), Store(p.root)
    for stage in ("gradient", "gradient~2"):
        key = k.key(w.id, stage)
        m = json.loads((store.dir(stage, key) / "manifest.json").read_text(encoding="utf-8"))
        assert m["status"] == "done" and m["key"] == key
    assert not (p.root / "store" / "_absent").exists()
