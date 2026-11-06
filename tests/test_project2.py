"""Project format v2 (DESIGN §1.1, §1.2, §1.5): save / load, sessions, exclusions, ids, v1 folders, from_v1.

The real v1 projects are only read: each check snapshots the file list and mtimes before and after, and an audit
hook raises on any write under the project while from_v1 and load run.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from astro_gpu.core import project2 as p2
from astro_gpu.core import workflow as wfl
from astro_gpu.core.project import Target
from astro_gpu.core.project2 import LegacyProject, Project2, Session

CLASSIC, CLASSIC_DN, FULL = (wfl.TEMPLATES[k] for k in ("Classic stack", "Classic + denoise", "Full"))


def _project(root: Path) -> Project2:
    p = p2.create(root, "HeartSoul", Target(name="IC 1805", focal_mm=272.0))
    p.reference_image = "inputs/full.png"
    p.sessions = [
        Session(1, "2026-08-16", lights=["F:/photos/heart_nebula/raw/signal"], darks=["F:/photos/heart_nebula/raw/dark"],
                flats=["F:/photos/heart_nebula/raw/flat"]),
        Session(2, "2026-08-19", lights=["raw/night2"],
                darks=[f"G:/Astro/957MSDCF/DSC{n:05d}.ARW" for n in range(639, 739)],      # a list of files
                flats=["F:/photos/heart_nebula/raw/flat"]),
    ]
    p.excluded = [(1, "DSC01234.ARW"), (2, "DSC01234.ARW")]            # camera counters restart: same name twice
    p.add_workflow("Full", FULL, {"stack": {"scale": 1.5, "min_transparency": 0.6},
                                  "denoise": {"adaptation": "short"}})
    p.add_workflow("Classic stack", CLASSIC)
    p.render_bases = {"faithful": {"sky_level": 24.0}}
    return p


# ---------------------------------------------------------------------------------------------- save and load

def test_save_load_round_trip(tmp_path):
    root = tmp_path / "HeartSoul"
    p = _project(root)
    p2.save(p)
    q = p2.load(root)
    assert q == p and q.root == root
    assert q.to_dict() == p.to_dict()
    assert q.workflows[0].options == FULL and q.workflows[1].options == CLASSIC
    assert q.current == 1 and q.next_workflow_id == 3
    assert q.target.focal_mm == 272.0 and q.render_bases == {"faithful": {"sky_level": 24.0}}


def test_stored_value_equal_to_the_static_default_is_kept(tmp_path):
    """Typing the static default is how a user overrides an automatic value: saving never drops it."""
    stages = wfl.chains().stages
    assert stages["stack"].Params.model_fields["scale"].default == 1.5
    assert stages["stack"].Params.model_fields["min_transparency"].default == 0.6
    assert stages["denoise"].Params.model_fields["adaptation"].default == "short"
    root = tmp_path / "p"
    p2.save(_project(root))
    q = p2.load(root)
    assert q.workflow(1).params == {"stack": {"scale": 1.5, "min_transparency": 0.6}, "denoise": {"adaptation": "short"}}
    assert q.workflow(1).stored("stack") == {"scale": 1.5, "min_transparency": 0.6}
    raw = json.loads((root / p2.FILE).read_text(encoding="utf-8"))
    assert raw["workflows"][0]["params"]["stack"] == {"scale": 1.5, "min_transparency": 0.6}


def test_excluded_are_session_name_pairs(tmp_path):
    root = tmp_path / "p"
    p2.save(_project(root))
    raw = json.loads((root / p2.FILE).read_text(encoding="utf-8"))
    assert raw["format"] == 2
    assert raw["excluded"] == [[1, "DSC01234.ARW"], [2, "DSC01234.ARW"]]
    q = p2.load(root)
    assert q.excluded == [(1, "DSC01234.ARW"), (2, "DSC01234.ARW")]
    q.excluded.append((3, "DSC00001.ARW"))                       # no session 3
    with pytest.raises(ValueError, match="pair"):
        p2.save(q)


def test_sessions_take_file_lists(tmp_path):
    root = tmp_path / "p"
    p2.save(_project(root))
    q = p2.load(root)
    assert len(q.session(2).darks) == 100 and q.session(2).darks[0] == "G:/Astro/957MSDCF/DSC00639.ARW"
    inp = q.inputs()                                             # the union the engine gets, absolute
    assert inp.lights == ["F:/photos/heart_nebula/raw/signal", (root / "raw/night2").as_posix()]
    assert inp.darks[0] == "F:/photos/heart_nebula/raw/dark" and len(inp.darks) == 101
    assert inp.flats == ["F:/photos/heart_nebula/raw/flat"]     # the same folder once
    assert q.reference_path() == (root / "inputs" / "full.png").as_posix()
    assert q.abspath("F:\\photos\\x\\..\\y.png") == "F:/photos/y.png"
    assert q.scratch == root / "scratch"


def test_workflow_ids_are_never_reused(tmp_path):
    root = tmp_path / "p"
    p = _project(root)
    third = p.add_workflow("Full 2", FULL)
    assert third.id == 3
    p.remove_workflow(3)
    assert p.add_workflow("Full 2", FULL).id == 4
    p.remove_workflow(1)
    assert p.current == 2
    p2.save(p)
    q = p2.load(root)
    assert [w.id for w in q.workflows] == [2, 4] and q.next_workflow_id == 5
    assert q.add_workflow("Classic + denoise", CLASSIC_DN).id == 5


def test_workflow_chain_follows_the_project(tmp_path):
    p = _project(tmp_path / "p")
    assert p.chain(1) == wfl.chain(FULL, reference=True)
    assert p.chain(1)[-2:] == ["render_simple", "star_removal"]
    exit_ = p.add_workflow("Raw stack", FULL, end="stacking", variant_of=1)
    assert p.chain(exit_) == ["inventory", "calibrate", "register", "stack", "render_simple"]
    p.reference_image = ""
    assert "reference" not in p.chain(1)


@pytest.mark.parametrize("change", [
    lambda p: p.workflows[0].options.update(stars="neural", denoise=None),
    lambda p: setattr(p.workflows[0], "end", "astrometry"),
    lambda p: setattr(p.workflows[1], "id", 1),
    lambda p: setattr(p.workflows[1], "variant_of", 7),
    lambda p: setattr(p, "current", 9),
    lambda p: setattr(p, "next_workflow_id", 2),
    lambda p: setattr(p, "stack_mode", "per_night"),             # not a stack mode
    lambda p: setattr(p.sessions[1], "id", 1),
    lambda p: setattr(p, "name", " "),
])
def test_save_refuses_an_inconsistent_definition(tmp_path, change):
    p = _project(tmp_path / "p")
    change(p)
    before = (tmp_path / "p" / p2.FILE).read_bytes()
    with pytest.raises(ValueError):
        p2.save(p)
    assert (tmp_path / "p" / p2.FILE).read_bytes() == before


def test_load_checks_the_format(tmp_path):
    p2.save(_project(tmp_path / "p"))
    d = json.loads((tmp_path / "p" / p2.FILE).read_text(encoding="utf-8"))
    d["format"] = 1
    with pytest.raises(ValueError, match="format"):
        Project2.from_dict(d)
    d["format"], d["workflows"][0]["options"]["denoise"] = 2, "nlbayes"
    with pytest.raises(ValueError, match="denoise"):
        Project2.from_dict(d)
    with pytest.raises(FileNotFoundError):
        p2.load(tmp_path / "nothing")


def test_create_refuses_an_existing_project(tmp_path):
    p2.create(tmp_path / "p", "A")
    with pytest.raises(FileExistsError):
        p2.create(tmp_path / "p", "B")
    q = p2.load(tmp_path / "p")
    assert q.name == "A" and q.workflows == [] and q.current is None and q.created


def test_state_file(tmp_path):
    root = tmp_path / "p"
    p2.create(root, "A")
    assert p2.read_state(root) == {"last_keys": {}}
    p2.write_state(root, {"last_keys": {"1": {"stack": "0123456789abcdef"}}})
    assert p2.read_state(root)["last_keys"]["1"]["stack"] == "0123456789abcdef"


# ---------------------------------------------------------------------------------------------- breakpoints

def _keyed_project(root: Path) -> Project2:
    """A saved project of the real stages with two workflows and three raws (the keys read the frame list)."""
    raws = root.parent / "raw" / "lights"
    raws.mkdir(parents=True)
    for i in range(3):
        (raws / f"IMG_{i:03d}.ARW").write_bytes(b"\0" * (100 + 10 * i))
    p = p2.create(root, "Keyed", Target(name="IC 1805"))
    p.sessions = [Session(1, "night 1", lights=[str(raws)])]
    p.add_workflow("Full", FULL, {"stack": {"scale": 2.0}, "denoise": {"adaptation": "long"}})
    p.add_workflow("Classic stack", CLASSIC)
    return p


def test_breakpoints_are_saved_loaded_and_empty_by_default(tmp_path):
    p = _keyed_project(tmp_path / "k")
    assert all(w.breakpoints == [] for w in p.workflows)
    p.workflows[0].breakpoints = ["calibration", "stacking"]
    p2.save(p)
    d = json.loads((p.root / p2.FILE).read_text(encoding="utf-8"))
    assert [w["breakpoints"] for w in d["workflows"]] == [["calibration", "stacking"], []]
    q = p2.load(p.root)
    assert q.workflows[0].breakpoints == ["calibration", "stacking"] and q.workflows[1].breakpoints == []
    assert q == p and q.to_dict() == p.to_dict()


def test_a_project_saved_before_breakpoints_loads_with_none(tmp_path):
    p = _keyed_project(tmp_path / "k")
    p2.save(p)
    f = p.root / p2.FILE
    d = json.loads(f.read_text(encoding="utf-8"))
    for w in d["workflows"]:
        del w["breakpoints"]                                  # the file of an earlier version
    f.write_text(json.dumps(d), encoding="utf-8")
    q = p2.load(p.root)
    assert [w.breakpoints for w in q.workflows] == [[], []]
    p2.save(q)                                                # and it is saved with the field from now on
    assert [w["breakpoints"] for w in json.loads(f.read_text(encoding="utf-8"))["workflows"]] == [[], []]


def test_breakpoints_are_checked_and_kept_in_the_graph_order(tmp_path):
    p = _keyed_project(tmp_path / "k")
    p.workflows[0].breakpoints = ["stacking", "calibration", "stacking"]
    d = p.to_dict()
    assert Project2.from_dict(d, p.root).workflows[0].breakpoints == ["calibration", "stacking"]   # once, in order
    for bad, why in ((["nowhere"], "no such block"), ("calibration", "list"), ([3], "list"),
                     (["calibration", "calibration"], "twice")):
        p.workflows[0].breakpoints = bad
        with pytest.raises(ValueError, match=why):
            p.validate()
    d["workflows"][0]["breakpoints"] = ["nowhere"]
    with pytest.raises(ValueError, match="no such block"):
        Project2.from_dict(d, p.root)
    d["workflows"][0]["breakpoints"] = "calibration"
    with pytest.raises(ValueError, match="list"):
        Project2.from_dict(d, p.root)


def test_breakpoints_never_enter_a_key(tmp_path):
    """A breakpoint is how the user works, not how the image is computed: every stage key, every key part and every block
    key is what it was without them, and what the file of an earlier version (no field at all) gives."""
    from astro_gpu.core.store import Keys
    p = _keyed_project(tmp_path / "k")

    def everything(proj):
        k = Keys(proj)
        return {w.id: {"keys": k.keys(w.id), "parts": {s: k.key_parts(w.id, s) for s in k.chain(w.id)},
                       "blocks": {b: k.block_key(w.id, b) for b, _s in k.rules.chains.blocks_in(k.chain(w.id))}}
                for w in proj.workflows}
    without = everything(p)
    p2.save(p)
    old = json.loads((p.root / p2.FILE).read_text(encoding="utf-8"))
    for w in old["workflows"]:
        del w["breakpoints"]
    assert everything(Project2.from_dict(old, p.root)) == without                  # an earlier version's file
    for w in p.workflows:
        w.breakpoints = list(wfl.BLOCKS)                                           # a pause after every block
    assert everything(p) == without
    p.workflows[0].breakpoints = ["calibration"]
    assert everything(p) == without
    assert len({k for w in without.values() for k in w["keys"].values()}) > 14     # the keys do tell stages apart


def test_no_engine_module_that_makes_keys_or_plans_reads_breakpoints():
    """The rule written down: only the definition (project2) and the client know of breakpoints."""
    core = Path(p2.__file__).parent
    for name in ("store.py", "workflow.py", "runner2.py", "estimate.py", "params.py", "stage.py"):
        assert "breakpoint" not in (core / name).read_text(encoding="utf-8").lower(), name
    assert "breakpoint" not in (core.parent / "api.py").read_text(encoding="utf-8").lower()


# ---------------------------------------------------------------------------------------------- v1 folders

def _write_v1(root: Path, params: dict, variants: dict | None = None, active: str = "") -> None:
    root.mkdir(parents=True)
    d = {"name": "Toy", "inputs": {"lights": ["D:/raw/signal"], "darks": ["D:/raw/957MSDCF\\DSC00639.ARW"],
                                   "flats": [], "bias": [], "flat_darks": []},
         "target": {"name": "IC 405", "ra": None, "dec": None, "focal_mm": 893.0}, "scratch_dir": "C:/scratch/Toy",
         "reference_image": "D:/ref.png", "description": "toy", "params": params, "render": {}, "devices": {},
         "created": "2026-10-12T22:41:05", "format": 1}
    if variants is not None:
        d["variants"], d["active_variant"] = variants, active
    (root / "project.json").write_text(json.dumps(d), encoding="utf-8")


def _snapshot(root: Path) -> dict[str, tuple]:
    """Every file and folder under root (root included) with its kind, size and modification time."""
    out = {}
    for dp, dns, fns in os.walk(root):
        for n in ["", *dns, *fns]:
            f = Path(dp) / n
            st = f.stat()
            out[str(f.relative_to(root))] = (f.is_dir(), st.st_size if not f.is_dir() else 0, st.st_mtime_ns)
    return out


def test_v1_folder_is_never_written(tmp_path):
    root = tmp_path / "v1"
    _write_v1(root, {})
    before = _snapshot(root)
    with pytest.raises(LegacyProject) as e:
        p2.load(root)
    assert e.value.root == root
    p = p2.from_v1(root)
    for write in (lambda: p2.save(p, root), lambda: p2.create(root, "X"), lambda: p2.write_state(root, {})):
        with pytest.raises(LegacyProject):
            write()
    assert _snapshot(root) == before
    with pytest.raises(ValueError, match="no folder"):
        p2.save(p)                                               # built in memory: it needs a new folder
    p2.save(p, tmp_path / "v2")
    assert p2.load(tmp_path / "v2") == p


@pytest.mark.parametrize("params", [
    {}, {"star_net": {"enabled": True}}, {"star_net": {"enabled": False}},
    {"star_net": {"enabled": False}, "starless": {"learned_stars": "D:/l/star_layer_net.npy"}},
    {"surveys": {"survey": "none"}},
])
def test_v1_options(params):
    """Every v1 project had its stars taken off: whatever its star settings, a v2 "Full" workflow."""
    assert p2.v1_options(params) == (FULL, "Full")


def test_from_v1_with_variants(tmp_path):
    root = tmp_path / "v1"
    main = {"star_net": {"enabled": False}, "stack": {"scale": 1.5}, "render_prep": {}}
    _write_v1(root, main, variants={
        "neural": {"params": {"star_net": {"enabled": None}}, "slug": "neural"},              # back to automatic
        "old layer": {"params": {"starless": {"learned_stars": "D:/l/star_layer_net.npy"}}, "slug": "old_layer"},
        "poly": {"params": {"surveys": {"survey": "none"}, "stack": {"scale": 2.0}}, "slug": "poly"},
    }, active="old layer")
    before = _snapshot(root)
    p = p2.from_v1(root)
    assert _snapshot(root) == before
    assert p.root is None and p.name == "Toy" and p.description == "toy" and p.created == "2026-10-12T22:41:05"
    assert p.scratch_dir == "" and p.reference_image == "D:/ref.png" and p.target.focal_mm == 893.0
    assert p.devices == {"main": "", "aux": ""} and p.excluded == [] and p.stack_mode == "together"
    assert [(s.id, s.lights, s.darks) for s in p.sessions] == [(1, ["D:/raw/signal"], ["D:/raw/957MSDCF\\DSC00639.ARW"])]
    got = [(w.id, w.name, w.options, w.params, w.variant_of) for w in p.workflows]
    assert got == [
        (1, "Full", FULL, {"star_net": {"enabled": False}, "stack": {"scale": 1.5}}, None),
        (2, "neural", FULL, {"stack": {"scale": 1.5}}, 1),
        (3, "old layer", FULL, {"star_net": {"enabled": False}, "stack": {"scale": 1.5},
                                "starless": {"learned_stars": "D:/l/star_layer_net.npy"}}, 1),
        (4, "poly", FULL | {"gradient": "polynomial"},
         {"star_net": {"enabled": False}, "stack": {"scale": 2.0}, "surveys": {"survey": "none"}}, 1),
    ]
    assert p.current == 3 and p.next_workflow_id == 5
    assert p.chain(3)[-1] == "star_removal"                  # the old star settings stay stored, unread


def test_from_v1_without_variant_keys(tmp_path):
    """NGC7822 and M33 have no 'variants' / 'active_variant' keys."""
    root = tmp_path / "v1"
    _write_v1(root, {})
    p = p2.from_v1(root)
    assert [(w.id, w.name) for w in p.workflows] == [(1, "Full")] and p.current == 1


# ---------------------------------------------------------------------------------------------- the real projects

REAL = [   # (folder, project name, workflow name from main): DESIGN §1.5 on the project.json read 2026-10-31
    ("F:/photos/heart_nebula/astro_gpu", "HeartSoul", "Full"),
    ("F:/photos/NGC7822/astro_gpu", "NGC7822", "Full"),
    ("E:/astro_gpu/projects/FlamingStar", "Flaming Star", "Full"),
    ("E:/astro_gpu/projects/Horsehead", "Horsehead", "Full"),
    ("E:/astro_gpu/projects/Rosette", "Rosette", "Full"),
    ("E:/astro_gpu/projects/Wizard", "NGC 7380 - Wizard Nebula", "Full"),
    ("E:/astro_gpu/projects/M31", "M31", "Full"),
    ("C:/Users/Nicolas/astro_gpu_scratch/projects/Iris", "Iris", "Full"),
    ("C:/Users/Nicolas/astro_gpu_scratch/projects/M33", "M33", "Full"),
]

_GUARDED: list[str] = []              # folders no write may touch while the guard is on
_WRITE_EVENTS = {"os.mkdir", "os.rename", "os.remove", "os.rmdir", "os.utime", "os.chmod", "os.truncate",
                 "os.symlink", "os.link", "shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.copytree",
                 "shutil.move", "shutil.rmtree"}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def _under_guard(path) -> bool:
    if not isinstance(path, (str, os.PathLike)):
        return False
    f = os.path.normcase(os.path.abspath(os.fspath(path)))
    return any(f == r or f.startswith(r + os.sep) for r in _GUARDED)


def _audit(event: str, args: tuple) -> None:
    if not _GUARDED:
        return
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None))[:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                  (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
        if writing and _under_guard(path):
            raise PermissionError(f"write guard: open({path!r}, {mode!r}) under a real project")
    elif event in _WRITE_EVENTS and any(_under_guard(a) for a in args):
        raise PermissionError(f"write guard: {event}{args!r} under a real project")


sys.addaudithook(_audit)


@pytest.mark.parametrize("folder,name,wf_name", REAL, ids=[r[1] for r in REAL])
def test_from_v1_on_the_real_projects(folder, name, wf_name):
    root = Path(folder)
    if not (root / "project.json").exists():
        pytest.skip(f"not reachable: {root}")
    before = _snapshot(root)
    _GUARDED.append(os.path.normcase(os.path.abspath(root)))
    try:
        with pytest.raises(LegacyProject):
            p2.load(root)
        p = p2.from_v1(root)
        old = json.loads((root / "project.json").read_text(encoding="utf-8"))
    finally:
        _GUARDED.clear()
    assert _snapshot(root) == before                       # nothing written, created or touched
    assert p.name == name and p.root is None and p.scratch_dir == ""
    assert [(w.id, w.name, w.end, w.variant_of) for w in p.workflows] == [(1, wf_name, None, None)]
    w = p.workflows[0]
    assert w.options == {"denoise": "pretrained", "stars": "neural", "gradient": "polynomial", "deconvolution": None}
    assert w.params == {st: kv for st, kv in (old.get("params") or {}).items() if kv}
    assert [getattr(p.sessions[0], k) for k in p2.KINDS] == [old["inputs"].get(k, []) for k in p2.KINDS]
    c = p.chain(w)
    assert c[-2:] == ["render_simple", "star_removal"]
    assert ("reference" in c) == bool(old.get("reference_image"))


def test_the_write_guard_works(tmp_path):
    """The guard used above raises on a write under a guarded folder, and lets reads through."""
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    _GUARDED.append(os.path.normcase(os.path.abspath(tmp_path)))
    try:
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x"
        with pytest.raises(PermissionError):
            (tmp_path / "b.txt").write_text("y", encoding="utf-8")
        with pytest.raises(PermissionError):
            (tmp_path / "d").mkdir()
        with pytest.raises(PermissionError):
            os.replace(tmp_path / "a.txt", tmp_path / "c.txt")
    finally:
        _GUARDED.clear()
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.txt"]
