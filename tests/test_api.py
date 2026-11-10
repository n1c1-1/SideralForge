"""The engine API (astro_gpu/api.py) and cli2: projects, frames, workflows, settings, keys and states, plans and costs,
runs, outputs, describe, rendering and export.

Two kinds of project:

* toy stages (inventory → stack → b (the 'denoise' option) → c → e, a stack with an automatic scale), run for real by
  the runner; the detached worker builds the same toy graph from the spec 'test_api:toy_rules';
* the real stage graph, only planned and read: results are faked by writing done manifests in the store.

Everything lives in temporary folders; the timing history is the test's own.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from astro_gpu import api, cli2
from astro_gpu.core import project as project_mod
from astro_gpu.core import project2, runner2
from astro_gpu.core.params import Auto, P, StageParams
from astro_gpu.core.project import list_frames
from astro_gpu.core.stage import Stage, StageError
from astro_gpu.core.store import KeyRules, Store
from astro_gpu.core.workflow import Chains
from astro_gpu.io.frames import FrameInfo, read_info

TESTS = Path(__file__).parent
REPO = TESTS.parent
SPEC = "test_api:toy_rules"


# ---------------------------------------------------------------------------------------------- toy stages

def _auto(value, template: str, **args) -> Auto:
    try:
        return Auto(value, template, args)
    except TypeError:                                    # the engine's Auto without arguments (before P16)
        return Auto(value, template.format(**args))


class InvParams(StageParams):
    max_lights: int = P(0, label="max lights", help="", ge=0)


class Inv(Stage):
    name = "inventory"
    title = "Inventory"
    Params = InvParams

    def run(self, ctx, p):
        like = FrameInfo("x.ARW", "raw", 6, 4, "RGGB", 0.0, 1.0)
        out = {k: [read_info(f, like=like).to_dict() for f in list_frames(getattr(ctx.project.config.inputs, k))]
               for k in project2.KINDS}                                  # no EXIF in the toy raws: the file date
        ctx.save_json("frames.json", out)
        ctx.metric("n_lights", len(out["lights"]))
        ctx.metric("exposure_s", 15.0)


class ToyStackParams(StageParams):
    scale: float = P(1.5, label="Scale", help="Output size.", tip="1.5 for most", ge=1.0, le=3.0, unit="x")
    kappa: float = P(5.0, label="Kappa", help="", ge=2.0)
    workers: int = P(4, label="workers", help="", ge=1)
    exclude: list[str] = P([], label="exclude", help="", level="internal")


class ToyStack(Stage):
    name = "stack"
    title = "Stack"
    deps = ("inventory",)
    Params = ToyStackParams

    def auto(self, ctx):
        n = len(ctx.load_json("inventory", "frames.json")["lights"])
        return {"scale": _auto(2.0 if n >= 3 else 1.0, "{n} frames", n=n)}

    def run(self, ctx, p):
        lights = [f["path"] for f in ctx.load_json("inventory", "frames.json")["lights"]]
        ctx.save_json("stack.json", {"n": len([f for f in lights if f not in p.exclude]), "scale": p.scale})
        ctx.save("a.npy", np.arange(24, dtype=np.float32).reshape(4, 6))
        pv = ctx.out("previews")
        pv.mkdir(exist_ok=True)
        (pv / "stack.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        ctx.metric("frames_used", len(lights))


class BParams(StageParams):
    steps: int = P(10, label="steps", help="", ge=1)
    fail: bool = P(False, label="fail", help="")


class B(Stage):
    name = "b"
    title = "B"
    deps = ("stack",)
    Params = BParams

    def run(self, ctx, p):
        if p.fail:
            raise StageError("b failed on purpose")
        ctx.save_json("b.json", {"steps": p.steps})


class CParams(StageParams):
    k: float = P(1.0, label="k", help="")
    sleep: float = P(0.0, label="sleep", help="", ge=0)


class C(Stage):
    name = "c"
    title = "C"
    deps = ("stack", "b")                # b optional
    Params = CParams

    def run(self, ctx, p):
        t0 = time.time()
        while time.time() - t0 < p.sleep:
            time.sleep(0.1)
            ctx.progress(min(1.0, (time.time() - t0) / p.sleep), "sleeping")        # raises on STOP
        steps = ctx.load_json("b", "b.json")["steps"] if ctx.has("b") else None
        ctx.save_json("c.json", {"steps": steps})


class E(Stage):
    name = "e"
    title = "E"
    deps = ("c",)
    Params = CParams

    def run(self, ctx, p):
        ctx.save_json("e.json", ctx.load_json("c", "c.json"))


def toy_rule(options, end, reference):
    return ["inventory", "stack"] + (["b"] if options.get("denoise") else []) + ["c", "e"]


def toy_rules() -> KeyRules:
    """Also built by the detached worker (plan.rules = 'test_api:toy_rules')."""
    chains = Chains([Inv(), ToyStack(), B(), C(), E()], extra={"e": ("b",)}, optional={"c": ("b",), "e": ("b",)},
                    blocks={"calibration": ("inventory",), "stacking": ("stack",), "denoise": ("b",),
                            "color": ("c",), "output": ("e",)}, rule=toy_rule)
    return KeyRules(chains, auto_fields={"stack": ("scale",)}, non_key={"stack": ("workers",)},
                    implied=lambda options, stage: {})


RULES = toy_rules()


# ---------------------------------------------------------------------------------------------- fixtures

@pytest.fixture
def state(tmp_path, monkeypatch) -> Path:
    """A private ASTRO_GPU_HOME: the timing histories never reach the user's, in this process and in a worker."""
    s = tmp_path / "state"
    monkeypatch.setattr(project_mod, "STATE", s)
    monkeypatch.setenv("ASTRO_GPU_HOME", str(s))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")]))
    return s


def raw_folder(folder: Path, n: int = 3) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"IMG_{i:03d}.ARW").write_bytes(b"\0" * (100 + 10 * i))
    return folder


@pytest.fixture
def toy(tmp_path, state):
    """A saved project of the toy graph: two workflows (Classic: no b; Denoise: with b), three raws."""
    p = api.create_project(tmp_path / "toy", "toy")
    api.use_rules(p, SPEC)
    api.add_session(p, lights=[raw_folder(tmp_path / "raw" / "lights")])
    api.add_workflow(p, "Classic", "Classic stack")
    api.add_workflow(p, "Denoise", "Classic + denoise")
    p.devices["main"] = "cpu"
    api.save(p)
    return p


def run_inproc(p, plan) -> tuple[str, runner2.RunInfo]:
    """A run in this process, as the worker does it (toy rules)."""
    info = runner2.prepare_run(p, plan, "cpu")
    status = runner2.run_plan(p.root, info.plan_file, rules=RULES, echo=False)
    api.refresh(p)
    return status, info


@pytest.fixture
def real(tmp_path, state):
    """A project on the real stage graph, planned and read only: Classic stack, Classic + denoise, Full, and Full with
    the adapted denoising (ids 1-4), three fake raws."""
    p = api.create_project(tmp_path / "real", "real", "IC 1805")
    api.add_session(p, lights=[raw_folder(tmp_path / "rraw" / "lights")])
    for n in api.TEMPLATES:
        api.add_workflow(p, n, n)
    api.add_workflow(p, "Full · adapted", {**api.TEMPLATES["Full"], "denoise": "adapted"})
    p.devices["main"] = "cuda:0"
    api.save(p)
    return p


def fake_done(p, wf, stage, metrics=None, params=None, files=None) -> Path:
    """A done result of the real graph without running it: a done manifest and some files."""
    k = api.stage_key(p, wf, stage)
    d = Store(p.root).dir(stage, k)
    d.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {}).items():
        f = d / name
        f.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, np.ndarray):
            np.save(f, content)
        elif isinstance(content, (dict, list)):
            f.write_text(json.dumps(content), encoding="utf-8")
        else:
            f.write_bytes(content)
    (d / "manifest.json").write_text(json.dumps({
        "stage": stage, "status": "done", "key": k, "run_id": k, "metrics": metrics or {}, "params": params or {},
        "origin": {}, "auto_reasons": {}, "duration": 1.5, "finished": "2026-11-01T01:00:00"}), encoding="utf-8")
    api.refresh(p)
    return d


def cost_equals_plan(p) -> runner2.Plan:
    """cost() is the plan's totals: the same jobs, seconds and disk."""
    plan, c = api.plan_all(p), api.cost(p)
    assert c.keys == plan.keys and c.jobs == len(plan)
    assert c.seconds == plan.seconds and c.disk_gb == plan.disk_gb and c.measured == plan.measured
    assert api.cost(p, [w.id for w in p.workflows]).keys == plan.keys
    assert len(set(plan.keys)) == len(plan.keys)
    return plan


# ---------------------------------------------------------------------------------------------- import and projects

def test_importing_the_api_and_cli2_loads_no_heavy_module():
    code = ("import sys, astro_gpu.api, astro_gpu.cli2\n"
            "bad = [m for m in ('torch', 'cv2', 'PySide6', 'astropy', 'rawpy') if m in sys.modules]\n"
            "assert not bad, bad\n"
            "import astro_gpu.api as a\n"
            "assert a.TEMPLATES and a.methods('denoise')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=120)
    assert out.returncode == 0, out.stderr


def test_list_projects_skips_underscore_folders_and_folders_without_a_definition(tmp_path, state):
    ws = tmp_path / "ws"
    old = api.create_project(ws / "Alpha", "Alpha", {"name": "IC 1805", "focal_mm": 272})
    api.add_session(old, lights=["x"])
    api.add_workflow(old, "Full", "Full")
    api.save(old)
    new = api.create_project(ws / "Beta", "Beta")
    api.create_project(ws / "_scratch_test", "hidden")
    (ws / "notes").mkdir()
    (ws / "legacy").mkdir()
    (ws / "legacy" / "project.json").write_text("{}")
    (ws / "broken").mkdir()
    (ws / "broken" / "project.astroproj").write_text("{not json")
    os.utime(old.root / project2.FILE, (1_700_000_000, 1_700_000_000))
    os.utime(new.root / project2.FILE, (1_700_100_000, 1_700_100_000))
    os.utime(ws / "broken" / "project.astroproj", (1_600_000_000, 1_600_000_000))
    got = api.list_projects(ws)
    assert [s.name for s in got] == ["Beta", "Alpha", "broken"]                # most recent first
    a = got[1]
    assert (a.target, a.nights, a.workflows, a.state, a.integration_h, a.thumb) == ("IC 1805", 1, 1, "never_run",
                                                                                    None, "")
    assert got[0].state == "empty" and got[2].error and got[2].root == str(ws / "broken")
    assert api.list_projects(tmp_path / "nowhere") == []
    # results make a card richer: the integration time comes from a done inventory, no raw is read
    (old.root / "cache").mkdir()
    (old.root / "cache" / "thumb.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    k = "0123456789abcdef"
    inv = old.root / "store" / "inventory" / k
    inv.mkdir(parents=True)
    (inv / "manifest.json").write_text(json.dumps({"status": "done", "key": k,
                                                   "metrics": {"n_lights": 240, "exposure_s": 15.0}}))
    project2.write_state(old.root, {"last_keys": {"1": {"inventory": k, "stack": "feedfeedfeedfeed"}}})
    a = next(s for s in api.list_projects(ws) if s.name == "Alpha")
    assert (a.state, a.done, a.integration_h, a.thumb) == ("has_results", 1, 1.0, str(old.root / "cache/thumb.jpg"))
    deep = next(s for s in api.list_projects(ws, deep=True) if s.name == "Alpha")
    assert deep.state == "outdated" and deep.outdated == 1


def test_create_open_save_and_save_as(tmp_path, state):
    p = api.create_project(tmp_path / "p", "P", "M31")
    assert (tmp_path / "p" / "project.astroproj").is_file() and p.target.name == "M31"
    assert api.open_project(tmp_path / "p").name == "P"
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "project.json").write_text("{}")
    with pytest.raises(api.LegacyProject):
        api.open_project(tmp_path / "v1")
    with pytest.raises(FileNotFoundError):
        api.open_project(tmp_path / "nothing")
    # save_as copies the definition, relative paths become absolute, the original stays where it was
    api.add_session(p, lights=["raw/lights"], flats=["raw/flat"])
    p.reference_image = "inputs/full.png"
    api.add_workflow(p, "Full", "Full", {"stack": {"scale": 2.0}})
    api.save(p)
    before = (tmp_path / "p" / "project.astroproj").read_bytes()
    q = api.save_as(p, tmp_path / "q")
    assert q.root == tmp_path / "q" and p.root == tmp_path / "p"
    assert q.sessions[0].lights == [p.abspath("raw/lights")] and q.reference_image == p.abspath("inputs/full.png")
    assert q.workflows[0].params == {"stack": {"scale": 2.0}}
    assert api.open_project(tmp_path / "q").reference_image == q.reference_image
    assert (tmp_path / "p" / "project.astroproj").read_bytes() == before
    with pytest.raises(FileExistsError):
        api.save_as(p, tmp_path / "q")


def make_v1(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A toy v1 project run to the end by v1's Pipeline (the toy stages), and a path map to a copy of its raws."""
    from astro_gpu.core.journal import Journal
    from astro_gpu.core.project import Inputs, Project
    from astro_gpu.core.runner import Pipeline
    lights = raw_folder(tmp_path / "v1raw" / "signal", 3)
    root = tmp_path / "v1old"
    Project.create(root, "Old toy", Inputs(lights=[str(lights)]), scratch_dir=tmp_path / "v1scratch")
    stages = [Inv(), ToyStack(), B(), C(), E()]
    assert Pipeline(Project(root), stages, Journal(root / "logs", echo=False)).run() == "done"
    import shutil
    shutil.copytree(tmp_path / "v1raw", tmp_path / "v1raw_new")               # the raws moved (same names and dates)
    return root, {str(tmp_path / "v1raw"): (tmp_path / "v1raw_new").as_posix()}


def snapshot(*folders: Path) -> dict:
    return {str(f): sorted((str(x.relative_to(f)), x.stat().st_size, x.stat().st_mtime_ns)
                           for x in f.rglob("*")) for f in folders}


def test_import_v1_previews_then_imports_a_toy_project_and_never_writes_into_the_old_one(tmp_path, state):
    old, path_map = make_v1(tmp_path)
    watched = [old, tmp_path / "v1raw", tmp_path / "v1scratch"]
    before = snapshot(*watched)
    new = tmp_path / "ws" / "old_toy"
    pre = api.import_v1(old, new, path_map, dry_run=True, rules=SPEC)             # the preview
    assert not new.exists() and not (tmp_path / "ws").exists() and pre.dry_run and not pre.written
    assert pre.adopted(1) == ["inventory", "stack", "b", "c", "e"] and pre.recomputed(1) == []
    assert pre.copy_bytes > 0 and pre.enough_space and "adopted (5)" in pre.text()
    rep = api.import_v1(old, new, path_map, rules=RULES, progress=lambda m: None)
    assert rep.written and [r.key for r in rep.stages] == [r.key for r in pre.stages]
    assert snapshot(*watched) == before                                          # the old project is untouched
    assert (new / "project.astroproj").is_file() and not (new / "project.json").exists()
    p = api.open_project(new)
    api.use_rules(p, SPEC)
    assert [w.name for w in p.workflows] == ["Full"]
    # the results are the project's: nothing to run, every stage done, with the keys the report gave
    assert not api.plan_all(p, device="cpu") and api.cost(p).jobs == 0
    st = api.states(p, update=False)
    assert {x.state for x in st.values()} == {"done"}
    assert st[(1, "stack")].key == next(r.key for r in rep.stages if r.stage == "stack")
    # describe() on an adopted stack: value, suggested, origin and reason
    sc = {i.name: i for i in api.describe(p, 1, "stack")}["scale"]
    assert (sc.value, sc.suggested, sc.suggestion_known, sc.origin, sc.reason_en) == (2.0, 2.0, True, "auto", "3 frames")
    assert api.stage_outputs(p, 1, "stack").manifest["adopted_from"]
    # what the import refuses reaches the caller as it is
    with pytest.raises(ValueError):
        api.import_v1(new, tmp_path / "again", rules=SPEC)                         # a v2 folder is not a v1 project
    assert api.NotEnoughSpace is not None and api.WriteRefused is not None and api.ImportReport is not None


def test_import_v1_refuses_a_disk_that_is_too_small_before_copying(tmp_path, state, monkeypatch):
    from astro_gpu.core import import_v1 as iv
    old, path_map = make_v1(tmp_path)
    monkeypatch.setattr(iv, "disk_free", lambda path: 10)
    new = tmp_path / "ws" / "old_toy"
    pre = api.import_v1(old, new, path_map, dry_run=True, rules=RULES)
    assert pre.free_bytes == 10 and not pre.enough_space and "NOT ENOUGH" in pre.text()
    with pytest.raises(api.NotEnoughSpace):
        api.import_v1(old, new, path_map, rules=RULES)
    assert not (new / "store").exists()


def test_cli2_import_previews_and_imports(tmp_path, state, capsys):
    old, path_map = make_v1(tmp_path)
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps(path_map))
    new = tmp_path / "ws" / "old_toy"
    capsys.readouterr()
    assert cli2.main(["--rules", SPEC, "import", str(old), str(new), "--map", str(mp), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "adopted (5)" in out and not new.exists()
    assert cli2.main(["--rules", SPEC, "import", str(old), str(new), "--map", str(mp)]) == 0
    assert "written" in capsys.readouterr().out and (new / "project.astroproj").is_file()
    assert cli2.main(["--rules", SPEC, "plan", str(new)]) == 0
    assert "0 job(s)" in capsys.readouterr().out
    assert cli2.main(["--rules", SPEC, "import", str(tmp_path / "nothing"), str(tmp_path / "x")]) == 1      # clean error
    assert "not a v1 project" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------- frames and sessions

def write_fits(path: Path, level: float, when: str, exposure: float = 15.0, gain: float = 100.0) -> Path:
    from astropy.io import fits
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(level) + len(path.name))
    a = np.clip(rng.normal(level, 4, (12, 16)), 0, 65535).astype(np.uint16)
    h = fits.Header()
    for k, v in {"BAYERPAT": "RGGB", "EXPTIME": exposure, "GAIN": gain, "DATE-OBS": when,
                 "INSTRUME": "ZWO ASI2600MC Pro"}.items():
        h[k] = v
    fits.PrimaryHDU(a, h).writeto(path, overwrite=True)
    return path


@pytest.fixture
def card(tmp_path):
    """raw/lights: 3 frames on the night of 16 Aug (one after midnight) and 2 on 19 Aug; raw/dark; raw/flat on 19 Aug."""
    r = tmp_path / "raw"
    for i, (lvl, when) in enumerate([(200, "2026-08-16T22:00:00"), (260, "2026-08-16T23:30:00"),
                                     (210, "2026-08-17T02:30:00"), (220, "2026-08-19T23:00:00"),
                                     (230, "2026-08-19T23:10:00")]):
        write_fits(r / "lights" / f"L{i}.fits", lvl, when)
    for i in range(2):
        write_fits(r / "dark" / f"D{i}.fits", 50, "2026-08-16T18:00:00")
        write_fits(r / "flat" / f"F{i}.fits", 900, f"2026-08-19T20:0{i}:00")
    return r / "lights"


def test_scan_group_and_add_frames_make_one_session_per_night(tmp_path, card):
    seen = []
    scan = api.scan_folder(card, lambda f, m: seen.append((f, m)))
    assert [len(scan.frames[k]) for k in ("lights", "darks", "flats", "bias")] == [5, 2, 2, 0]
    assert scan.folders["darks"] == [str(card.parent / "dark")] and not scan.errors and seen[-1][0] == 1.0
    groups = api.group_sessions(scan.frames["lights"])
    assert [(g.name, g.count) for g in groups] == [("2026-08-16", 3), ("2026-08-19", 2)]     # noon to noon
    split = api.group_sessions(scan.frames["lights"] + [write_info(card, "X.fits", 8.0)],
                               api.GroupingCriteria(night=True, exposure=True, iso=True))
    assert any(g.exposure == 8.0 for g in split) and len(split) == 3
    assert len(api.group_sessions(scan.frames["lights"], api.GroupingCriteria(night=False, exposure=False,
                                                                              iso=False))) == 1
    p = api.create_project(tmp_path / "p", "p")
    made = api.add_frames(p, scan)
    assert [s.name for s in made] == ["2026-08-16", "2026-08-19"] and [s.id for s in p.sessions] == [1, 2]
    assert len(made[0].lights) == 3 and all(f.endswith(".fits") for f in made[0].lights)
    assert made[0].darks == [str(card.parent / "dark")] or made[1].darks == [str(card.parent / "dark")]
    assert made[1].flats == [str(card.parent / "flat")] and made[0].flats == []          # the flats of 19 Aug
    # one night: the folder itself
    q = api.create_project(tmp_path / "q", "q")
    api.add_frames(q, api.scan_folder(card.parent / "dark"))
    assert q.sessions[0].lights == [str(card.parent / "dark")]


def write_info(folder: Path, name: str, exposure: float):
    from astro_gpu.io.frames import read_info
    return read_info(write_fits(folder.parent / "extra" / name, 200, "2026-08-16T23:50:00", exposure))


def test_health_checks_flats_of_two_sessions_are_merged(tmp_path, card):
    p = api.create_project(tmp_path / "p", "p")
    assert [c.key for c in api.health_checks(p)] == ["no_lights"]
    api.add_frames(p, api.scan_folder(card))
    got = {c.key: c for c in api.health_checks(p)}
    assert "flats_merged" not in got and got["no_bias"].level == "info" and got["disk"].level == "ok"
    assert got["disk"].args["need_gb"] >= 0 and got["ok"].level == "ok"
    # a second session with its own flats: they merge into one master flat
    extra = card.parent / "second"
    write_fits(extra / "lights" / "S0.fits", 210, "2026-08-21T22:00:00")
    write_fits(extra / "flat" / "SF0.fits", 880, "2026-08-21T19:00:00")
    api.add_frames(p, api.scan_folder(extra / "lights"))
    got = {c.key: c for c in api.health_checks(p)}
    assert got["flats_merged"].level == "warn" and got["flats_merged"].args["n"] == 2 and "ok" not in got
    # a bigger need than the disk has is a warning
    big = api.add_session(p, lights=[card])
    p.scratch_dir = str(tmp_path / "scratch")
    api.remove_session(p, big.id)
    chk = api._disk_check(p)
    assert chk.level in ("ok", "warn") and chk.args["free_gb"] > 0


def test_exclusions_are_session_and_name_pairs_and_change_only_the_stack(tmp_path, card, state):
    p = api.create_project(tmp_path / "p", "p")
    api.add_frames(p, api.scan_folder(card))
    api.add_workflow(p, "Full", "Full")
    inv, reg, stack = (api.stage_key(p, 1, s) for s in ("inventory", "register", "stack"))
    api.set_excluded(p, [(1, "L0.fits"), (2, "L3.fits")])
    api.set_excluded(p, [(1, "L0.fits")])                                  # twice: once
    assert p.excluded == [(1, "L0.fits"), (2, "L3.fits")]
    assert api.stage_key(p, 1, "inventory") == inv and api.stage_key(p, 1, "register") == reg
    assert api.stage_key(p, 1, "stack") != stack                           # only Stacking and later change
    with pytest.raises(KeyError):
        api.set_excluded(p, [(9, "L0.fits")])
    api.set_excluded(p, [(1, "L0.fits")], excluded=False)
    assert p.excluded == [(2, "L3.fits")]
    api.remove_session(p, 2)
    assert p.excluded == [] and [s.id for s in p.sessions] == [1]
    api.save(p)
    assert api.open_project(p.root).excluded == []


def test_frame_thumb_shares_one_stretch_per_session(tmp_path, card):
    p = api.create_project(tmp_path / "p", "p")
    api.add_frames(p, api.scan_folder(card))
    s1 = p.sessions[0]
    dim, bright = s1.lights[0], s1.lights[1]                               # level 200 and 260 in one session
    a, b = api.frame_thumb(p, dim, 8), api.frame_thumb(p, bright, 8)
    assert a.dtype == np.uint8 and a.shape == (6, 8, 3)                    # the superpixel image, never enlarged
    assert b.mean() > a.mean() + 5                                         # the same stretch: differences stay
    assert api.frame_thumb(p, dim, 256).shape == (6, 8, 3)
    assert api.frame_thumb(p, dim, 4).shape == (3, 4, 3)                   # long side 4


# ---------------------------------------------------------------------------------------------- workflows

def test_workflow_editing_templates_copies_rename_delete_and_undo(tmp_path):
    p = api.create_project(tmp_path / "p", "p")
    a = api.add_workflow(p, "Classic", "Classic stack")
    b = api.add_workflow(p, "Full", "Full", {"stack": {"scale": 2.0}})
    c = api.add_workflow(p, "Custom", {"denoise": "pretrained", "stars": None, "gradient": "polynomial"})
    assert [w.id for w in p.workflows] == [1, 2, 3] and c.options["denoise"] == "pretrained" and a.options["stars"] is None
    d = api.add_workflow(p, "Full 2", None, {"stack": {"kappa": 6.0}}, copy_of=b.id)
    assert d.options == b.options and d.params == {"stack": {"scale": 2.0, "kappa": 6.0}}
    assert p.workflow(b.id).params == {"stack": {"scale": 2.0}}               # the source is untouched
    dup = api.duplicate_workflow(p, d.id)
    assert dup.name == "Full 2 copy" and dup.params == d.params and dup.id == 5
    with pytest.raises(ValueError):
        api.add_workflow(p, "bad", "No such template")
    with pytest.raises(ValueError):
        api.add_workflow(p, "bad", {"stars": "psf"})                          # Stars needs Denoise
    api.rename_workflow(p, 1, "  Classic v2 ")
    assert p.workflow(1).name == "Classic v2"
    with pytest.raises(ValueError):
        api.rename_workflow(p, 1, " ")
    snap = api.workflows(p)
    snap[0].name = "changed outside"                                          # a copy: the project is not touched
    assert p.workflow(1).name == "Classic v2"
    full = api.workflows(p)
    gone = api.delete_workflow(p, 3)
    assert gone.name == "Custom" and [w.id for w in p.workflows] == [1, 2, 4, 5]
    api.set_current(p, 4)
    api.delete_workflow(p, 4)
    assert p.current == 1                                                     # the current one moved
    api.set_workflows(p, full, current=3)                                     # undo
    assert [w.id for w in p.workflows] == [1, 2, 3, 4, 5] and p.current == 3 and p.workflow(3).name == "Custom"
    new = api.add_workflow(p, "After undo", "Classic stack")
    assert new.id == 6                                                        # ids are never reused
    with pytest.raises(ValueError):
        api.set_workflows(p, [w for w in api.workflows(p)] + [api.workflows(p)[0]])    # duplicate id: rolled back
    assert [w.id for w in p.workflows] == [1, 2, 3, 4, 5, 6]


def test_set_option_notes_refusals_and_all_or_nothing(tmp_path):
    p = api.create_project(tmp_path / "p", "p")
    api.add_workflow(p, "Classic", "Classic stack")
    api.add_workflow(p, "Full", "Full")
    notes = api.set_option(p, 1, "stars", "neural")
    assert notes and p.workflow(1).options["stars"] == "neural" and p.workflow(1).options["denoise"] == "pretrained"
    assert any("Denoise" in n for n in notes)
    with pytest.raises(api.OptionRefused):
        api.set_option(p, [2, 1], "denoise", None)                            # both have Stars
    assert p.workflow(1).options["denoise"] == "pretrained" and p.workflow(2).options["denoise"] == "pretrained"
    notes = api.set_option(p, [1, 2], "stars", None)                          # Stars off
    assert notes == [] and p.workflow(2).options["stars"] is None
    assert api.set_option(p, 1, "stars", None) == []                          # nothing to do
    assert api.set_option(p, [1, 2], "denoise", None) == []
    with pytest.raises(ValueError):
        api.set_option(p, 1, "stars", "psf_neural_v9")
    with pytest.raises(ValueError):
        api.set_option(p, 1, "sharpness", 3)
    with pytest.raises(KeyError):
        api.set_option(p, 99, "stars", None)


def test_variants_make_keep_discard_and_exits(real):
    p = real
    change = {"kind": "param", "stage": "gradient", "name": "degree", "value": 3}
    v = api.make_variant(p, 1, change, "Classic · degree 3")
    assert v.variant_of == 1 and v.params == {"gradient": {"degree": 3}} and p.workflow(1).params == {}
    assert api.stage_key(p, 1, "stack") == api.stage_key(p, v.id, "stack")           # shared up to the change
    assert api.stage_key(p, 1, "gradient") != api.stage_key(p, v.id, "gradient")
    k = api.keep_variant(p, v.id)
    assert k.name == "Classic stack" and k.variant_of is None and p.current == v.id
    assert [w.id for w in p.workflows] == [2, 3, 4, v.id]                                # the original is retired
    with pytest.raises(ValueError):
        api.keep_variant(p, 2)                                                          # not a variant
    w = api.make_variant(p, 3, {"kind": "opt", "key": "gradient", "value": "automatic"}, "Full · automatic")
    assert w.options["gradient"] == "automatic" and w.variant_of == 3
    child = api.make_variant(p, w.id, {"kind": "param", "stage": "stack", "name": "kappa", "value": 7}, "child")
    api.discard_variant(p, w.id)
    assert w.id not in [x.id for x in p.workflows] and p.current == 3
    assert p.workflow(child.id).variant_of == 3                                        # re-parented
    api.discard_variant(p, child.id)
    assert [x.id for x in p.workflows] == [2, 3, 4, v.id]
    with pytest.raises(ValueError):
        api.make_variant(p, 3, {"kind": "what"}, "x")
    # a Raw stack exit keeps only the settings of the stages it runs
    api.set_param(p, 3, "denoise", "adaptation", "long")
    api.set_param(p, 3, "stack", "kappa", 6)
    x = api.add_exit(p, 3, "stacking", "Raw stack")
    assert x.end == "stacking" and x.params == {"stack": {"kappa": 6.0}}
    assert api.chain(p, x.id)[-2:] == ["stack", "render_simple"]
    with pytest.raises(ValueError):
        api.add_exit(p, 3, "denoise", "no")
    with pytest.raises(api.OptionRefused):
        api.set_option(p, x.id, "denoise", "pretrained")                                       # nothing after an exit


def test_methods_table_marks_what_is_not_in_the_engine():
    dn = {m.key: m for m in api.methods("denoise")}
    assert list(dn) == ["pretrained", "adapted"]
    assert not dn["pretrained"].new and not dn["adapted"].new and dn["pretrained"].runs_on == "both"
    assert [(m.key, m.new) for m in api.methods("gradient")] == [("polynomial", False), ("rbf", False), ("automatic", False)]
    assert [(m.key, m.stage) for m in api.methods("stars")] == [("neural", "star_removal")]
    assert [(m.key, m.new) for m in api.methods("deconvolution")] == [("rl", False), ("neural", False)]
    from astro_gpu.core import workflow
    for block, opt in (("denoise", "denoise"), ("gradient", "gradient"), ("stars", "stars")):
        allowed = set(workflow.OPTION_VALUES[opt])
        assert all((m.key in allowed) != m.new for m in api.methods(block))          # new = not selectable
    with pytest.raises(ValueError):
        api.methods("colour")


# ---------------------------------------------------------------------------------------------- keys and blocks

def test_chain_deps_blocks_and_sharing_on_the_real_graph(real):
    p = real
    assert api.chain(p, 1) == ["inventory", "calibrate", "register", "stack", "astrometry", "catalogs", "surveys",
                               "gradient", "color", "render_simple"]
    assert api.chain(p, 3)[-2:] == ["render_simple", "star_removal"] and "star_removal" not in api.chain(p, 2)
    assert api.deps(p, 1, "render_simple") == ("stack", "color", "gradient")
    assert api.deps(p, 3, "star_removal") == ("render_simple", "stack", "gradient")
    b = {x.block: x for x in api.blocks(p, 3)}
    assert list(b) == ["calibration", "registration", "selection", "stacking", "astrometry", "gradient", "denoise",
                       "color", "stars", "output"]
    assert b["astrometry"].stages == ("astrometry", "catalogs", "surveys") and len(b["astrometry"].key) == 1
    assert b["selection"].stages == () and b["output"].key[-1] == "workflow 3"
    assert api.blocks(p, 1)[-1].key[-1] == "workflow 1" and api.blocks(p, 1)[-1].key != b["output"].key
    assert api.sharing(p, 1, "stack") == [1, 2, 3, 4] and api.sharing(p, 3, "denoise") == [2, 3]
    assert api.sharing(p, 3, "render_simple") == [2, 3] and api.sharing(p, 3, "star_removal") == [3]
    assert api.sharing(p, 4, "star_removal") == [4] and api.sharing(p, 1, "render_simple") == [1]
    # the Astrometry block's key does not depend on the survey, which follows the Gradient method
    k3, k4 = ({x.block: x for x in api.blocks(p, i)}["astrometry"].key for i in (3, 4))
    assert k3 == k4
    what_if = api.with_change(p, 3, {"kind": "param", "stage": "stack", "name": "kappa", "value": 8})
    assert what_if.id == 3 and p.workflow(3).params == {} and api.stage_key(p, what_if, "stack") != \
        api.stage_key(p, 3, "stack")
    assert api.sharing(p, what_if, "stack") == [3]                      # nobody else computes it this way


def test_the_toy_scenario_cost_equals_the_plan_after_each_change(real):
    p = real
    plan0 = cost_equals_plan(p)
    # 16: everything up to Color shared; Denoise and what follows by denoising
    assert len(plan0) == 16 and plan0.seconds > 300 and not plan0.measured
    # a setting on two workflows: the others still share the stack
    api.set_param(p, [3, 4], "stack", "kappa", 6)
    plan1 = cost_equals_plan(p)
    assert api.sharing(p, 3, "stack") == [3, 4] and api.sharing(p, 1, "stack") == [1, 2]
    assert len(plan1) == 24 and len(set(plan1.keys) & set(plan0.keys)) < len(plan0)
    # the what-if of a change equals the plan once the change is made
    change = {"kind": "param", "stage": "gradient", "name": "degree", "value": 3}
    ask = api.cost(p, [api.with_change(p, 1, change)])
    assert ask.jobs == 10 and ask.seconds < plan1.seconds                               # the whole chain, from nothing
    v = api.make_variant(p, 1, change, "Classic · degree 3")
    plan2 = cost_equals_plan(p)
    assert set(plan2.keys) - set(plan1.keys) == {api.stage_key(p, v.id, s) for s in ("gradient", "color",
                                                                                      "render_simple")}
    assert api.cost(p, [v.id]).keys == ask.keys                                       # the what-if was exact
    # keep: the variant replaces its original (whose Gradient and Color Classic + denoise still shares)
    gone = api.stage_key(p, 1, "render_simple")
    api.keep_variant(p, v.id)
    plan3 = cost_equals_plan(p)
    assert set(plan3.keys) - set(plan1.keys) == set(plan2.keys) - set(plan1.keys)
    assert set(plan1.keys) - set(plan3.keys) == {gone}
    # another variant, discarded: back to the plan before it
    w = api.make_variant(p, 3, {"kind": "param", "stage": "denoise", "name": "adaptation", "value": "long"},
                         "Full · short denoise")
    plan4 = cost_equals_plan(p)
    assert len(plan4) > len(plan3)
    api.discard_variant(p, w.id)
    assert cost_equals_plan(p).keys == plan3.keys
    api.rename_workflow(p, 3, "Full renamed")
    assert cost_equals_plan(p).keys == plan3.keys                                      # a name is not a key
    # delete one, then undo through set_workflows
    snap, cur = api.workflows(p), p.current
    api.delete_workflow(p, 4)
    plan5 = cost_equals_plan(p)
    assert len(plan5) < len(plan3)
    api.set_workflows(p, snap, cur)
    assert cost_equals_plan(p).keys == plan3.keys
    # results shrink the cost: the first stages done
    classic = v.id                                                                    # the kept variant is Classic now
    fake_done(p, classic, "inventory", files={"frames.json": {"lights": []}})
    fake_done(p, classic, "calibrate", files={"cache.json": {"path": str(list_frames(p.sessions[0].lights)[0])}})
    # a done result that lost its data is planned again when a later stage needs it: here the calibrated cache
    lost = fake_done(p, classic, "calibrate", files={"cache.json": {"path": str(p.root / "gone.f16")}})
    again = cost_equals_plan(p)
    assert len(again) == len(plan3) - 1 and any(j.stage == "calibrate" and j.rerun for j in again.jobs)
    (lost / "cache.json").write_text(json.dumps({"path": str(list_frames(p.sessions[0].lights)[0])}))
    api.refresh(p)
    for s in ("register", "stack"):
        fake_done(p, classic, s)
    done = cost_equals_plan(p)
    assert len(done) == len(plan3) - 4 and done.seconds < plan3.seconds
    one = api.plan_to_block(p, classic, "color")
    assert [j.stage for j in one.jobs] == ["astrometry", "catalogs", "surveys", "gradient", "color"]
    sel = api.plan_to_block(p, 2, "selection")                                 # Selection runs Registration
    assert not sel                                                             # it is done
    assert api.estimate(p, 1, "stack").seconds > 0 and not api.estimate(p, 1, "stack").measured


def test_gpu_decides_the_estimates_and_the_history_makes_them_measured(real, state):
    cpu, gpu = api.plan_all(real, device="cpu"), api.plan_all(real, device="cuda:0")
    assert cpu.seconds > 3 * gpu.seconds                                       # no GPU: denoise and stars take far longer
    assert api.plan_all(real, device="cpu", history={"stack": [[100.0, 1000.0, "cpu"]]}).jobs[0].measured is False
    assert api.cost(real, [1], device="cuda:0", history={}).seconds == api.plan_workflows(real, [1], "cuda:0").seconds
    m = api.machine()
    assert m.cores >= 1 and m.memory_gb > 0 and all(g.device.startswith("cuda:") for g in m.gpus)
    assert api.machine() is m and isinstance(m.cuda, bool)


def test_warm_fills_the_disk_caches_so_that_keys_states_and_plans_do_not_scan(real, monkeypatch):
    from astro_gpu.core import store as store_mod
    p = real
    assert not api.is_warm(p)
    seen = []
    api.warm(p, lambda f, m: seen.append(f))
    assert api.is_warm(p) and seen[0] == 0.0 and seen[-1] == 1.0

    def no_scan(*a, **k):
        raise AssertionError("a disk scan after warm()")
    monkeypatch.setattr(store_mod, "list_frames", no_scan)
    monkeypatch.setattr("astro_gpu.io.frames.read_info", no_scan)
    key = api.stage_key(p, 3, "stack")
    assert api.states(p, update=False)[(3, "stack")].key == key
    assert api.cost(p).jobs == len(api.plan_all(p)) == 16 and api.blocks(p, 3) and api.sharing(p, 1, "stack")
    api.set_param(p, 3, "stack", "kappa", 6)                       # an edit changes the keys, not the disk parts
    assert api.stage_key(p, 3, "stack") != key and api.cost(p).jobs > 20
    api.add_session(p, lights=["elsewhere"])                       # a new session: its files are not cached yet
    assert not api.is_warm(p)


# ---------------------------------------------------------------------------------------------- settings

def test_set_param_drops_a_value_only_at_a_known_suggestion(toy):
    p = toy
    stored = lambda wf: p.workflow(wf).params                                          # noqa: E731
    # the upstream result is not done: the suggestion of scale (automatic) is unknown, the value stays stored
    api.set_param(p, 1, "stack", "scale", 2.0)
    assert stored(1) == {"stack": {"scale": 2.0}}
    api.set_param(p, 1, "stack", "scale", 1.5)                                         # the static default: kept too
    assert stored(1) == {"stack": {"scale": 1.5}}
    api.set_param(p, 1, "stack", "kappa", 5.0)                                         # no automatic rule: default
    assert stored(1) == {"stack": {"scale": 1.5}}
    api.set_param(p, 1, "stack", "kappa", "6")                                         # coerced as Params reads it
    assert stored(1) == {"stack": {"scale": 1.5, "kappa": 6.0}}
    api.set_param(p, 1, "stack", "scale", None)                                        # back to the suggestion
    api.set_param(p, 1, "stack", "kappa", None)
    assert stored(1) == {}
    # the inventory is done: the suggestion becomes known (3 frames: 2.0)
    status, _ = run_inproc(p, api.plan_to_block(p, 1, "calibration"))
    assert status == "done"
    api.set_param(p, 1, "stack", "scale", 2.0)
    assert stored(1) == {}                                                             # equal to the suggestion: dropped
    api.set_param(p, 1, "stack", "scale", 1.5)
    assert stored(1) == {"stack": {"scale": 1.5}}                                      # not the suggestion: stored
    api.set_param(p, 1, "stack", "scale", 2.0)
    assert stored(1) == {}                                                             # the stored value is dropped
    # a scope of several workflows; a stage a workflow does not have is stored but inert
    api.set_param(p, [1, 2], "stack", "kappa", 7)
    api.set_param(p, [1, 2], "b", "steps", 12)
    assert stored(1)["stack"] == {"kappa": 7.0} == stored(2)["stack"] and stored(2)["b"] == {"steps": 12}
    assert api.stage_key(p, 1, "stack") == api.stage_key(p, 2, "stack")
    # invalid values name the setting and change nothing
    with pytest.raises(ValueError, match=r"stack\.scale"):
        api.set_param(p, 1, "stack", "scale", 9)
    with pytest.raises(ValueError, match="internal|set by the project"):
        api.set_param(p, 1, "stack", "exclude", ["x"])
    with pytest.raises(ValueError, match="nope"):
        api.set_param(p, 1, "stack", "nope", 1)
    with pytest.raises(ValueError, match="stage"):
        api.set_param(p, 1, "nostage", "x", 1)
    with pytest.raises(KeyError):
        api.set_param(p, [1, 9], "stack", "kappa", 3)
    assert stored(1)["stack"] == {"kappa": 7.0}


def test_describe_gives_value_suggestion_origin_reason_and_drift(toy):
    p = toy
    cold = {i.name: i for i in api.describe(p, 1, "stack")}
    assert (cold["scale"].value, cold["scale"].suggested, cold["scale"].suggestion_known, cold["scale"].origin) == \
        (1.5, 1.5, False, "default")
    assert cold["kappa"].suggestion_known and cold["kappa"].origin == "default"
    run_inproc(p, api.plan_to_block(p, 1, "calibration"))
    info = {i.name: i for i in api.describe(p, 1, "stack")}
    s = info["scale"]
    assert (s.value, s.suggested, s.suggestion_known, s.origin) == (2.0, 2.0, True, "auto")
    assert s.label == "Scale" and s.unit == "x" and (s.ge, s.le) == (1.0, 3.0) and s.default == 1.5
    assert s.reason_en == "3 frames" and s.type == "float"
    if hasattr(Auto, "template"):                                       # P16: the template and its arguments, unformatted
        assert s.reason == "{n} frames" and s.reason_args == {"n": 3}
    assert info["exclude"].level == "internal" and info["exclude"].locked and not info["kappa"].locked
    # the user's value wins; the suggestion is still shown
    api.set_param(p, 1, "stack", "scale", 1.0)
    u = {i.name: i for i in api.describe(p, 1, "stack")}["scale"]
    assert (u.value, u.suggested, u.origin, u.reason) == (1.0, 2.0, "user", "")
    api.set_param(p, 1, "stack", "scale", None)
    # a stage whose upstream is missing in another workflow's chain is a plain error
    with pytest.raises(ValueError):
        api.describe(p, 1, "b")
    # drift: the done result was made with other values than a fresh resolve gives
    run_inproc(p, api.plan_workflows(p, [1]))
    assert not any(i.drift for i in api.describe(p, 1, "stack"))
    man = Store(p.root).dir("stack", api.stage_key(p, 1, "stack")) / "manifest.json"
    m = json.loads(man.read_text())
    m["params"]["scale"], m["params"]["workers"] = 1.5, 99
    man.write_text(json.dumps(m))
    d = {i.name: i for i in api.describe(p, 1, "stack")}
    assert d["scale"].drift and d["scale"].result_value == 1.5 and d["scale"].value == 2.0
    assert not d["workers"].drift                                       # left out of the key: not a drift


def test_describe_a_real_stack_from_faked_upstream_results(real, tmp_path):
    """The real stack.auto on fake upstream results: value, suggested, origin and reason (the check after P6 does it
    on an adopted project)."""
    p = real
    raws = [f.name for f in list_frames([p.sessions[0].lights[0]])]
    n = len(raws)
    fake_done(p, 1, "inventory", files={"frames.json": {"lights": [{"path": str(f)} for f in
                                                                   list_frames(p.sessions[0].lights)]}})
    fake_done(p, 1, "calibrate", {"dark_frame_noise": 3.0, "cache_scale": 1.0},
              files={"frame_stats.npy": np.full((n, 4, 2), 2.0, np.float32),
                     "cache.json": {"width": 400, "height": 300, "cfa": "RGGB"}})
    assert not next(i for i in api.describe(p, 1, "stack") if i.name == "scale").suggestion_known     # register missing
    fake_done(p, 1, "register", files={"reg.npz": b""})
    np.savez(Store(p.root).dir("register", api.stage_key(p, 1, "register")) / "reg.npz", fwhm=np.full(n, 3.5),
             transp=np.ones(n), rms=np.full(n, 0.2), coef=np.zeros((n, 2, 10)), ref=0)
    info = {i.name: i for i in api.describe(p, 1, "stack")}
    sc = info["scale"]
    assert sc.suggestion_known and sc.origin == "auto" and sc.value == sc.suggested == 1.0
    assert sc.reason_en == "star size not measured: 1×"                 # the coverage rule; no Moffat fit faked
    # the method advice: given beside the choices, never applied (every frame at the same place: holes, Classic)
    m = info["method"]
    assert m.value == "drizzle" and m.origin == "default" and m.advice == "classic" and m.advice_reason
    assert info["half_block"].origin == "auto" and info["min_transparency"].reason
    assert info["exclude"].level == "internal"
    # an exclusion reaches the stack's effective settings, locked
    api.set_excluded(p, [(1, raws[0])])
    assert {i.name: i for i in api.describe(p, 1, "stack")}["exclude"].locked


# ---------------------------------------------------------------------------------------------- states

def test_states_follow_runs_changes_failures_and_the_previous_result(toy):
    p = toy
    st = api.states(p)
    assert {x.state for x in st.values()} == {"pending"} and st[(1, "stack")].reason == api.REASON_NEVER
    assert not (p.root / "state.json").exists()                                         # nothing done: nothing to record
    status, _ = run_inproc(p, api.plan_all(p))
    assert status == "done"
    st = api.states(p)
    assert {x.state for x in st.values()} == {"done"} and len(st) == 4 + 5
    last = project2.read_state(p.root)["last_keys"]
    assert last["2"]["b"] == api.stage_key(p, 2, "b") and set(last) == {"1", "2"}
    old_b = api.stage_key(p, 2, "b")
    api.set_param(p, 2, "b", "steps", 12)                                               # b, c, e of Denoise go Outdated
    st = api.states(p, update=False)
    assert st[(2, "b")] == api.StageState("stale", api.REASON_SETTINGS, api.stage_key(p, 2, "b"), old_b)
    assert st[(2, "c")].state == "stale" and st[(2, "c")].reason == api.REASON_UPSTREAM
    assert st[(2, "stack")].state == "done" and all(st[(1, s)].state == "done" for s in ("c", "e"))
    assert project2.read_state(p.root)["last_keys"]["2"]["b"] == old_b                  # update=False wrote nothing
    # the previous result is readable while the block is Outdated
    cur, prev = api.stage_outputs(p, 2, "b"), api.stage_outputs(p, 2, "b", "previous")
    assert cur is None and prev is not None and prev.key == old_b and prev.which == "previous"
    # a failing job: b fails, c and e are skipped, the other workflow is untouched
    api.set_param(p, 2, "b", "fail", True)
    status, _ = run_inproc(p, api.plan_all(p))
    assert status == "failed"
    st = api.states(p)
    assert st[(2, "b")].state == "failed" and "on purpose" in st[(2, "b")].reason
    assert st[(2, "c")].state == "skipped" and st[(2, "c")].reason == api.REASON_UPSTREAM_FAILED
    assert st[(2, "e")].state == "skipped" and st[(1, "e")].state == "done"
    # an interrupted job leaves its .tmp folder; a live run owning it shows it as running
    api.set_param(p, 2, "b", "fail", None)
    k = api.stage_key(p, 2, "b")
    tmp = Store(p.root).tmp_dir("b", k)
    tmp.mkdir(parents=True)
    (tmp / "manifest.json").write_text(json.dumps({"status": "running"}))
    assert api.states(p)[(2, "b")].state == "interrupted"                               # nobody runs it
    runs = p.root / "logs" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "r1.plan.json").write_text(json.dumps({"jobs": [{"key": k}], "offset": 0, "device": "cpu"}))
    (p.root / "logs" / "run.lock").write_text(json.dumps({"pid": os.getpid(), "run_id": "r1", "started": time.time()}))
    assert api.running(p).run_id == "r1"
    assert api.states(p)[(2, "b")].state == "running" and api.states(p)[(2, "c")].state == "stale"
    with pytest.raises(api.RunInProgress):
        api.start_run(p, api.plan_all(p))


# ---------------------------------------------------------------------------------------------- outputs

def test_stage_outputs_and_arrays_are_read_whole(toy):
    p = toy
    assert api.stage_outputs(p, 1, "stack") is None
    with pytest.raises(api.NotDone):
        api.stage_array(p, 1, "stack", "a")
    run_inproc(p, api.plan_all(p))
    o = api.stage_outputs(p, 1, "stack")
    assert o.key == api.stage_key(p, 1, "stack") and o.which == "current" and o.metrics["frames_used"] == 3
    assert list(o.previews) == ["stack"] and o.previews["stack"].endswith("stack.jpg")
    assert o.params["scale"] == 2.0 and o.origin["scale"] == "auto" and o.duration is not None and o.finished
    assert o.manifest["status"] == "done" and "scale" in o.auto_reasons
    a = api.stage_array(p, 1, "stack", "a")
    assert a.shape == (4, 6) and a.dtype == np.float32 and not isinstance(a, np.memmap)
    assert api.stage_array(p, 1, "stack", "a.npy").shape == (4, 6)
    half = api.stage_array(p, 1, "stack", "a", scale=0.5)
    assert half.shape == (2, 3) and half[0, 0] == pytest.approx(np.arange(24).reshape(4, 6)[:2, :2].mean())
    with pytest.raises(api.NotDone, match="no array"):
        api.stage_array(p, 1, "stack", "nothing")
    assert api.stage_outputs(p, 1, "inventory").metrics["n_lights"] == 3


def test_pixel_to_sky_comes_from_the_results(real):
    from astropy.io import fits
    from astropy.wcs import WCS
    p = real
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [322.0, 47.2]
    w.wcs.crpix = [100.0, 50.0]
    w.wcs.cdelt = [-0.0005, 0.0005]
    d = fake_done(p, 1, "astrometry")
    fits.PrimaryHDU(header=w.to_header()).writeto(d / "wcs.fits")
    ra, dec = api.pixel_to_sky(p, 1, 99.0, 49.0)                         # 0-based: the reference pixel
    assert (ra, dec) == pytest.approx((322.0, 47.2), abs=1e-6)
    assert api.pixel_to_sky(p, 3, 99.0, 49.0) == (ra, dec)                # Full shares the astrometry of Classic


# ---------------------------------------------------------------------------------------------- frame table

def fake_registration(p, wf: int, fwhm, transp, rms):
    lights = list_frames(p.sessions[0].lights)
    n = len(lights)
    fake_done(p, wf, "inventory", {"n_lights": n, "exposure_s": 15.0},
              files={"frames.json": {"lights": [{"path": str(f), "time": f"2026-08-16T22:0{i}:00"}
                                                for i, f in enumerate(lights)]}})
    fake_done(p, wf, "calibrate", {"dark_frame_noise": 3.0, "cache_scale": 1.0},
              files={"frame_stats.npy": np.full((n, 4, 2), 2.0, np.float32)})
    d = fake_done(p, wf, "register")
    np.savez(d / "reg.npz", fwhm=np.asarray(fwhm, float), transp=np.asarray(transp, float),
             rms=np.asarray(rms, float), coef=np.zeros((n, 2, 10)), ref=0)
    return lights


def test_frame_table_and_selection_preview_use_the_registration_and_the_thresholds(tmp_path, state):
    p = api.create_project(tmp_path / "p", "p")
    api.add_session(p, lights=[raw_folder(tmp_path / "raw" / "lights", 6)], name="night 1")
    api.add_workflow(p, "Classic", "Classic stack")
    p.devices["main"] = "cuda:0"
    with pytest.raises(api.NotDone):
        api.frame_table(p, 1)
    lights = fake_registration(p, 1, [3.5, 3.6, 3.4, 8.0, 3.5, 3.5], [1, 1, 1, 1, 0.3, 1],
                               [0.2, 0.2, 0.2, 0.2, 0.2, np.inf])
    t = api.frame_table(p, 1)
    assert t.source == "preview" and t.total == 6
    # the stack's default selection (WBPP's minimum weight): only the frame not aligned is left out; the large stars
    # and the hazy frame keep their lower weight
    assert [r.status for r in t.rows] == ["", "", "", "", "", "not aligned"] and t.kept == 5
    assert t.rows[3].weight < t.rows[0].weight and t.rows[4].weight < t.rows[0].weight
    assert [r.name for r in t.rows] == [f.name for f in lights] and {r.session for r in t.rows} == {1}
    assert t.rows[0].time == "2026-08-16T22:00:00" and t.rows[3].fwhm == 8.0
    # the thresholds act with the previous selection
    api.set_param(p, 1, "stack", "selection", "legacy")
    t = api.frame_table(p, 1)
    assert [r.status for r in t.rows] == ["", "", "", "large stars", "low transparency", "not aligned"]
    assert t.kept == 3 and (t.min_transparency, t.max_fwhm_ratio) == (0.6, 2.0)
    loose = api.selection_preview(p, 1, 0.2, 5.0)
    assert loose.source == "preview" and loose.kept == 5 and (loose.min_transparency, loose.max_fwhm_ratio) == (0.2, 5.0)
    api.set_excluded(p, [(1, lights[0].name)])
    ex = api.selection_preview(p, 1, 0.2, 5.0)
    assert ex.kept == 4 and ex.rows[0].status == "excluded by hand" and ex.rows[0].excluded
    assert ex.rows[1].weight > 0 and ex.rows[0].weight == 0
    # the thresholds are measured on all the frames: excluding one never changes the fate of another
    assert [r.status for r in ex.rows[1:]] == [r.status for r in loose.rows[1:]]
    got = [c for c in api.health_checks(p) if c.key == "fwhm"]
    assert len(got) == 1 and got[0].args["session"] == "night 1" and got[0].args["fwhm"] == pytest.approx(3.5)
    # once Stacking is done the table shows what the stack used
    stack = [{"path": str(f), "time": None, "weight": 0.0 if i in (0, 5) else 0.5, "transparency": 1.0, "fwhm": 3.5,
              "excluded": {0: "excluded by hand", 5: "not aligned"}.get(i, "")} for i, f in enumerate(lights)]
    fake_done(p, 1, "stack", files={"frames.json": stack})
    s = api.frame_table(p, 1)
    assert s.source == "stack" and s.kept == 4 and s.rows[5].status == "not aligned" and s.rows[1].weight == 0.5


# ---------------------------------------------------------------------------------------------- cache

def test_cache_usage_and_clear_unused_keep_what_a_workflow_can_use(toy, tmp_path):
    p = toy
    run_inproc(p, api.plan_all(p))
    api.states(p)
    store = Store(p.root)
    u = api.cache_usage(p)
    assert u.unused == [] and u.used_bytes > 0
    # old results: one with no workflow behind it, a trash folder, the .tmp of an old key and of a key in use
    ghost = store.dir("stack", "aaaaaaaaaaaaaaaa")
    ghost.mkdir(parents=True)
    (ghost / "big.bin").write_bytes(b"x" * 4000)
    (store.trash / "stack-bbbbbbbbbbbbbbbb-run1").mkdir(parents=True)
    (store.trash / "stack-bbbbbbbbbbbbbbbb-run1" / "f").write_bytes(b"y" * 1000)
    junk_tmp = store.tmp_dir("c", "cccccccccccccccc")
    junk_tmp.mkdir()
    (junk_tmp / "ck").write_bytes(b"z" * 300)
    api.set_param(p, 2, "c", "k", 2.0)                                      # Denoise's c and e need a new run
    live_tmp = store.tmp_dir("c", api.stage_key(p, 2, "c"))
    live_tmp.mkdir()
    (live_tmp / "ck").write_bytes(b"w" * 200)
    scratch = p.scratch / "dddddddddddddddd" / "calibrate"
    scratch.mkdir(parents=True)
    (scratch / "lights.f16").write_bytes(b"s" * 5000)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("mine")
    u = api.cache_usage(p)
    assert sorted(i.kind for i in u.unused) == ["result", "scratch", "tmp", "tmp", "trash"]
    by_path = {Path(i.path).name: i for i in u.unused}
    assert by_path["aaaaaaaaaaaaaaaa"].bytes == 4000 and by_path["aaaaaaaaaaaaaaaa"].kind == "result"
    assert by_path[junk_tmp.name].resumable is False and by_path[live_tmp.name].resumable is True
    assert by_path["dddddddddddddddd"].kind == "scratch" and u.unused_bytes >= 4000 + 1000 + 300 + 200 + 5000
    # the previous result of Outdated c and e (last keys) is still in use
    assert not any(i.kind == "result" and i.stage in ("c", "e") for i in u.unused)
    # a fake item outside the store is never deleted
    fake = api.CacheItem("result", "stack", "e" * 16, str(outside), 1)
    assert api.clear_unused(p, [fake]) == 0 and (outside / "keep.txt").exists()
    freed = api.clear_unused(p)
    assert freed == 4000 + 1000 + 300 + 5000 and not ghost.exists() and not junk_tmp.exists() and not scratch.exists()
    assert live_tmp.exists()                                               # a checkpoint the next run resumes from
    assert [i.kind for i in api.cache_usage(p).unused] == ["tmp"]
    assert api.clear_unused(p, include_resumable=True) == 200 and not live_tmp.exists()
    # never during a run
    (p.root / "logs").mkdir(exist_ok=True)
    (p.root / "logs" / "run.lock").write_text(json.dumps({"pid": os.getpid(), "run_id": "r", "started": time.time()}))
    with pytest.raises(api.RunInProgress):
        api.clear_unused(p)


# ---------------------------------------------------------------------------------------------- runs

def wait_for(pred, timeout=120.0, every=0.2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(every)
    raise AssertionError("timed out")


def test_a_detached_run_is_followed_through_events_and_cancelled(toy):
    p = toy
    api.set_param(p, 1, "c", "sleep", 30.0)                                 # long enough to be stopped
    plan = api.plan_workflows(p, [1])
    assert plan.rules == SPEC and [j.stage for j in plan.jobs] == ["inventory", "stack", "c", "e"]
    info = api.start_run(p, plan, "cpu")
    assert info.pid and Path(info.plan_file).is_file() and info.offset == 0
    wait_for(lambda: api.running(p))
    assert api.running(p).run_id == info.run_id
    with pytest.raises(api.RunInProgress):
        api.start_run(p, plan, "cpu")
    events, off = [], info.offset

    def seen(kind, stage=None):
        nonlocal off
        got, off = api.read_events(p, off, info.run_id)
        events.extend(got)
        return any(e["type"] == kind and (stage is None or e.get("stage") == stage) for e in events)
    wait_for(lambda: seen("progress", "c"))
    assert {"run_start", "plan", "stage_start"} <= {e["type"] for e in events}
    api.cancel(p)
    wait_for(lambda: seen("run_end"))
    assert events[-1]["status"] == "interrupted" and events[-1]["interrupted"]
    wait_for(lambda: api.running(p) is None, 30)
    assert all(e["run"] == info.run_id for e in events)
    # the next run goes on: the finished jobs are not planned again
    api.set_param(p, 1, "c", "sleep", None)
    again = api.plan_workflows(p, [1])
    assert [j.stage for j in again.jobs] == ["c", "e"]
    info2 = api.start_run(p, again, "cpu")
    evs, off = [], info2.offset
    wait_for(lambda: evs.extend(api.read_events(p, off, info2.run_id)[0]) or any(e["type"] == "run_end" for e in evs))
    assert [e for e in evs if e["type"] == "run_end"][0]["status"] == "done"
    wait_for(lambda: api.running(p) is None, 30)
    api.refresh(p)
    assert {s.state for (w, _), s in api.states(p).items() if w == 1} == {"done"}
    assert not api.plan_workflows(p, [1])


# ---------------------------------------------------------------------------------------------- cli2

def test_cli2_creates_plans_runs_follows_and_stops_a_toy_project(tmp_path, state, capsys):
    proj, raws = tmp_path / "cli", raw_folder(tmp_path / "craw", 3)

    def cli(*args) -> tuple[int, str]:
        capsys.readouterr()
        code = cli2.main(["--rules", SPEC, *map(str, args)])
        return code, capsys.readouterr().out

    code, out = cli("new", proj, "--name", "cli toy", "--lights", raws, "--template", "Classic stack")
    assert code == 0 and "1 session(s), 1 workflow(s)" in out and (proj / "project.astroproj").is_file()
    code, out = cli("add-workflow", proj, "Denoise", "--template", "Classic + denoise")
    assert code == 0 and "workflow 2 added" in out
    code, out = cli("add-workflow", proj, "Custom", "--option", "denoise=n2n", "--copy-of", 1)
    assert code == 0 and project2.load(proj).workflow(3).options["denoise"] == "pretrained"
    code, out = cli("set", proj, "--wf", 2, "b.steps=12", "stack.kappa=6", "--option", "denoise=pretrained")
    assert code == 0 and project2.load(proj).workflow(2).params == {"b": {"steps": 12}, "stack": {"kappa": 6.0}}
    code, out = cli("set", proj, "--wf", 2, "stack.kappa=auto")
    assert code == 0 and project2.load(proj).workflow(2).params == {"b": {"steps": 12}}
    code, out = cli("plan", proj)
    assert code == 0 and "inventory" in out and re.search(r"\d+ job\(s\)", out)
    code, out = cli("plan", proj, "--wf", 1, "--to", "stacking")
    assert code == 0 and "2 job(s)" in out
    code, out = cli("status", proj, "--read-only")
    assert "pending" in out and "to run:" in out
    # run in the foreground: starts the worker and follows its events
    code, out = cli("run", proj, "--timeout", 180)
    assert code == 0 and "started" in out and "run done" in out and "stack done" in out
    code, out = cli("status", proj)
    assert "pending" not in out and "everything is up to date" in out
    assert cli("run", proj)[1].startswith("nothing to run")
    assert (proj / "state.json").is_file()
    # a long job, detached, followed, stopped
    cli("set", proj, "--wf", 1, "c.sleep=30")
    code, out = cli("run", proj, "--detach")
    assert code == 0
    run_id = re.search(r"run (\S+) started", out).group(1)
    wait_for(lambda: api.running(project2.load(proj)))
    wait_for(lambda: '"stage": "c"' in (proj / "logs" / "events.jsonl").read_text() and
             '"type": "progress"' in (proj / "logs" / "events.jsonl").read_text())
    assert cli("stop", proj)[0] == 0
    code, out = cli("events", proj, "--run", run_id, "--follow", "--offset", 0, "--timeout", 120)
    assert code == 2 and "run interrupted" in out
    code, out = cli("events", proj, "--run", run_id)
    assert "run interrupted" in out and re.search(r"offset \d+", out)
    wait_for(lambda: api.running(project2.load(proj)) is None, 30)
    assert cli("run", proj, "--wf", 1, "--to", "color", "--detach")[0] == 0
    assert cli("status", proj, "--read-only")[0] == 0
    assert cli2.main(["--rules", SPEC, "set", str(proj), "--wf", "1", "stack.scale=99"]) == 1      # a clean error
    assert "stack.scale" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------- rendering and export

def simple_prep(d: Path, H: int = 120, W: int = 160, denoised: bool = True) -> Path:
    """A render_simple result: a nebula (a coloured blob) on a flat sky, noise, valid pixels, the preparation file."""
    from astro_gpu.render.simple import image_stats
    rng = np.random.default_rng(4)
    d.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:H, :W]
    blob = np.exp(-(((xx - W / 2) / (W / 5)) ** 2 + ((yy - H / 2) / (H / 5)) ** 2)).astype(np.float32)
    lin = (np.array([900, 500, 300], np.float32)[:, None, None] * blob + rng.normal(0, 3, (3, H, W))).astype(np.float32)
    valid = np.ones((H, W), bool)
    prep = {"M": np.eye(3).tolist(), "crop": [4, W - 4, 4, H - 4], "stats": image_stats(lin, valid), "linked": True,
            "stretch": "auto", "denoised": denoised, "color": True, "exit": None, "drizzle_scale": 1.0}
    np.save(d / "lin.npy", lin)
    if denoised:
        raw = (lin + rng.normal(0, 10, lin.shape)).astype(np.float32)
        np.save(d / "lin_raw.npy", raw)
        prep["stats_raw"] = image_stats(raw, valid)
    np.save(d / "valid.npy", valid)
    (d / "prep_simple.json").write_text(json.dumps(prep))
    return d


def test_sliders_render_layer_masks_and_export_of_a_simple_workflow(real, tmp_path):
    p = real
    sl = {s.key: s for s in api.sliders_of(p, 1)}
    assert len(sl) == 11 and sl["sky"].available and sl["noise"].reason == "noise smoothing needs the Denoise block"
    assert not sl["star_strength"].available and sl["star_strength"].reason
    stars = {"star_strength", "star_small", "star_size", "star_color"}
    assert {s.key for s in api.sliders_of(p, 3) if not s.available} == stars       # Full's Output image: no stars
    assert {s.key for s in api.sliders_of(p, 3, "combined") if not s.available} == {"noise"}
    assert {s.key for s in api.sliders_of(p, 3, "stars") if not s.available} == {"sky", "contrast", "saturation",
                                                                                     "colours", "halpha", "faint", "noise"}
    assert {s.key for s in api.sliders_of(p, 2) if not s.available} == stars
    x = api.add_exit(p, 3, "stacking", "Raw stack")
    assert not {s.key: s for s in api.sliders_of(p, x.id)}["noise"].available
    with pytest.raises(api.NotDone):
        api.render_layer(p, 2, None, None, 0.5)                                 # nothing computed yet
    d = fake_done(p, 2, "render_simple")
    simple_prep(d, denoised=True)
    p.render_bases["bright"] = {"sky": 0.3}
    lay = api.render_layer(p, 2, "bright", {"sky": 2.5, "contrast": 3}, 0.5)
    assert lay.rgb.shape == (3, 60, 80) and lay.rgb.dtype == np.float32 and 0 <= lay.rgb.min() <= lay.rgb.max() <= 1
    assert lay.crop == (4, 156, 4, 116) and lay.info["kind"] == "simple" and lay.info["scale"] == 0.5
    base = api.render_layer(p, 2, None, None, 0.5)
    assert not np.array_equal(base.rgb, lay.rgb)                                # the base and sliders change the picture
    region = api.render_layer(p, 2, None, None, 1.0, region=(20, 100, 10, 60))
    assert region.rgb.shape == (3, 50, 80)
    m = api.post_masks(p, 2, 0.5)
    assert set(m) == {"halpha"} and m["halpha"].shape == (60, 80) and 0 <= m["halpha"].min() <= m["halpha"].max() <= 1
    # export: files re-open with the right dtype and size; the history sidecar and the folder export/<name>/
    files = api.export_workflow(p, 2, None, {"sky": 2.5}, {"formats": ["tiff16", "png16", "jpeg", "fits32f"],
                                                            "name": "classic"})
    out = p.root / "export" / "classic"
    assert {f.name for f in files} == {"classic_16bit.tif", "classic.png", "classic.jpg", "classic.fits",
                                       "classic_history.json"} and all(f.parent == out for f in files)
    import cv2
    import tifffile
    from astropy.io import fits
    tif = tifffile.imread(out / "classic_16bit.tif")
    assert tif.dtype == np.uint16 and tif.shape == (112, 152, 3)                 # the crop of the layer
    assert cv2.imread(str(out / "classic.png"), cv2.IMREAD_UNCHANGED).dtype == np.uint16
    assert fits.getdata(out / "classic.fits").shape == (3, 112, 152)
    hist = json.loads((out / "classic_history.json").read_text())
    assert hist["workflow"]["id"] == 2 and hist["sliders"] == {"sky": 2.5} and hist["project"] == "real"
    # a composed image of Post: reduced to 64 px
    img = np.random.default_rng(1).random((3, 30, 40)).astype(np.float32)
    files = api.export_image(p, img, {"origin": (3, 4)}, {"name": "post", "formats": ["tiff32f", "jpeg"], "size": 20})
    assert tifffile.imread(p.root / "export" / "post" / "post_32bit_float.tif").shape == (15, 20, 3)
    assert len(files) == 2
    with pytest.raises(ValueError):
        api.export_image(p, img, None, {"name": "bad", "formats": ["bmp"]})


def test_metric_texts_come_from_the_metric_catalogue():
    from astro_gpu import texts
    m = texts.metric("n_lights")
    assert (texts.metric_label("n_lights"), texts.metric_unit("n_lights"), m.fmt) == ("Lights", "", "d")
    assert texts.metric_meaning("n_lights") == "Number of light frames found."
    assert texts.metric_label("n_lights", "fr") == "Brutes"
    assert texts.metric_unit("exposure_s") == "s"
    assert texts.metric("my_new_metric") is None and texts.metric_label("my_new_metric") == "My new metric"


def test_the_user_chooses_the_files_of_each_kind(tmp_path, card):
    darks = sorted((card.parent / "dark").iterdir())
    flats = sorted((card.parent / "flat").iterdir())
    scan = api.scan_files({"lights": [card], "darks": [darks[0]], "flats": [str(f) for f in flats]})
    assert [len(scan.frames[k]) for k in ("lights", "darks", "flats", "bias")] == [5, 1, 2, 0]
    assert scan.folders["darks"] == [str(darks[0])]                     # one dark of two: the file itself
    assert scan.folders["flats"] == [str(card.parent / "flat")]         # every flat of the folder: the folder
    p = project2.create(tmp_path / "chosen", "chosen")
    made = api.add_frames(p, scan)
    assert sum(len(s.darks) for s in made) == 1 and any(str(darks[0]) in s.darks for s in made)
    assert sum(len(s.lights) for s in made) >= 1
