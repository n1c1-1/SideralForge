"""Runner v2 (DESIGN §3): plans, the detached worker, the store folders, the lock, the estimates; and its engine
changes (stack.exclude, the cache-only calibrate, run_id, atomic timings).

Toy stages: inventory → stack → b (with the 'denoise' option, checkpointed) → c → e; c reads b when the chain has
it and e reads b through an EXTRA edge. Workflows: Classic (no b) and Denoise.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from astro_gpu.core import estimate as est
from astro_gpu.core import project as project_mod
from astro_gpu.core import project2, runner2
from astro_gpu.core.params import Auto, P, StageParams
from astro_gpu.core.project import list_frames
from astro_gpu.core.project2 import Session
from astro_gpu.core.stage import Stage, StageError
from astro_gpu.core.store import KeyRules, Keys, Store
from astro_gpu.core.workflow import TEMPLATES, Chains


def _gpu_hidden() -> bool:
    """CUDA_VISIBLE_DEVICES empty or -1 (only -1 hides the GPUs on Windows),
    or no GPU visible."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is not None and v.strip() in ("", "-1"):
        return True
    try:
        import torch
        return not torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return True


# tests that pin a worker to a GPU; skipped when the suite runs on the CPU (CUDA_VISIBLE_DEVICES=-1, during P8)
needs_gpu = pytest.mark.skipif(_gpu_hidden(), reason="needs a visible GPU (CUDA_VISIBLE_DEVICES is empty or -1)")

TESTS = Path(__file__).parent
RUNS: Counter = Counter()
SEEN: list[str] = []                     # the folder each job wrote into
GPU_FAULT = "gpu_fault"                  # file in the project: how many more times the Classic C stage faults
GPU_DEAD = "TOY_GPU_DEAD"                # the pid of the process whose (fake) CUDA context is dead


def fake_gpu(device):
    """The plans' gpu_check in the tests: the CUDA context is dead in the process where a toy stage faulted (an
    environment variable naming its pid: the worker it starts inherits it but is another process)."""
    return False if os.environ.get(GPU_DEAD) == str(os.getpid()) else None


def _ran(ctx) -> None:
    RUNS[ctx.stage.name] += 1
    SEEN.append(ctx.dir.name)
    time.sleep(0.03)                     # a duration the clock can measure


# ---------------------------------------------------------------------------------------------- toy stages

class InvParams(StageParams):
    max_lights: int = P(0, label="max lights", help="", ge=0)


class Inv(Stage):
    name = "inventory"
    title = "Inventory"
    Params = InvParams
    fail = False                         # the raws' drive is unplugged (the key does not change)

    def run(self, ctx, p):
        _ran(ctx)
        if Inv.fail:
            raise StageError("the raws cannot be read")
        files = list_frames(ctx.project.config.inputs.lights)
        ctx.save_json("frames.json", {"lights": [{"path": str(f)} for f in files]})


class ToyStackParams(StageParams):
    scale: float = P(1.5, label="scale", help="", ge=1.0, le=3.0)
    kappa: float = P(5.0, label="kappa", help="", ge=2.0)
    workers: int = P(4, label="workers", help="", ge=1)
    exclude: list[str] = P([], label="exclude", help="", level="internal")


class ToyStack(Stage):
    name = "stack"
    title = "Stack"
    deps = ("inventory",)
    Params = ToyStackParams

    def auto(self, ctx):
        n = len(ctx.load_json("inventory", "frames.json")["lights"])
        return {"scale": Auto(2.0 if n >= 3 else 1.0, f"{n} frames")}

    def run(self, ctx, p):
        _ran(ctx)
        lights = [f["path"] for f in ctx.load_json("inventory", "frames.json")["lights"]]
        ctx.save_json("stack.json", {"n": len([f for f in lights if f not in p.exclude]), "scale": p.scale})


class BParams(StageParams):
    steps: int = P(10, label="steps", help="", ge=1)
    fail: bool = P(False, label="fail", help="")


class B(Stage):
    name = "b"
    title = "B"
    deps = ("stack",)
    Params = BParams
    stop_at: int | None = None           # the user presses Stop at this step

    def run(self, ctx, p):
        _ran(ctx)
        if p.fail:
            raise StageError("b failed on purpose")
        ck = ctx.ckpt / "i.json"
        start = json.loads(ck.read_text())["i"] if ck.exists() else 0
        for i in range(start, p.steps):
            ck.write_text(json.dumps({"i": i + 1}))
            if B.stop_at == i:
                B.stop_at = None
                runner2.cancel(ctx.project.root)
            ctx.progress((i + 1) / p.steps, "step", force=True)          # raises Interrupted on STOP
        ctx.save_json("b.json", {"steps": p.steps, "resumed_from": start})


class CParams(StageParams):
    k: float = P(1.0, label="k", help="")
    sleep: float = P(0.0, label="sleep", help="", ge=0)


class C(Stage):
    name = "c"
    title = "C"
    deps = ("stack", "b")                # b optional
    Params = CParams

    def run(self, ctx, p):
        _ran(ctx)
        fault = Path(ctx.project.root) / GPU_FAULT
        if not ctx.has("b") and fault.exists() and int(fault.read_text() or 0) > 0:
            # a GPU fault in the Classic workflow's C: the CUDA context of this process is dead from now on
            fault.write_text(str(int(fault.read_text()) - 1))
            os.environ[GPU_DEAD] = str(os.getpid())
            raise RuntimeError("CUDA error: unspecified launch failure")
        t0 = time.time()
        while time.time() - t0 < p.sleep:
            time.sleep(0.1)
            ctx.progress(min(1.0, (time.time() - t0) / p.sleep), "sleeping")
        steps = ctx.load_json("b", "b.json")["steps"] if ctx.has("b") else None
        ctx.save_json("c.json", {"n": ctx.load_json("stack", "stack.json")["n"], "steps": steps,
                                 "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "cwd": os.getcwd(),
                                 "device": ctx.device_name, "pid": os.getpid()})


class E(Stage):
    name = "e"
    title = "E"
    deps = ("c",)                        # and b, through an EXTRA edge
    Params = CParams

    def run(self, ctx, p):
        _ran(ctx)
        ctx.save_json("e.json", ctx.load_json("c", "c.json"))


def toy_rule(options, end, reference):
    return ["inventory", "stack"] + (["b"] if options.get("denoise") else []) + ["c", "e"]


def toy_rules() -> KeyRules:
    """The rules of the toy stages (also built by the detached worker: plan.rules = 'test_runner2:toy_rules')."""
    chains = Chains([Inv(), ToyStack(), B(), C(), E()], extra={"e": ("b",)}, optional={"c": ("b",), "e": ("b",)},
                    blocks={"calibration": ("inventory",), "stacking": ("stack",), "denoise": ("b",),
                            "color": ("c",), "output": ("e",)}, rule=toy_rule)
    return KeyRules(chains, auto_fields={"stack": ("scale",)}, non_key={"stack": ("workers",)},
                    implied=lambda options, stage: {})


RULES = toy_rules()
CLASSIC, DENOISE = 1, 2


def K(p) -> Keys:
    return Keys(p, RULES)


def plan_all(p, **kw) -> runner2.Plan:
    return runner2.plan_all(p, keys=K(p), **kw)


def run(p, plan, device="cpu") -> tuple[str, runner2.RunInfo]:
    """A run in this process, as the worker does it."""
    info = runner2.prepare_run(p, plan, device)
    return runner2.run_plan(p.root, info.plan_file, rules=RULES, echo=False), info


def events(p, info) -> list[dict]:
    return runner2.read_events(p.root, info.offset, info.run_id)[0]


def write_frames(folder: Path, sizes: dict[str, int]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, n in sizes.items():
        (folder / name).write_bytes(b"\0" * n)
    return folder


@pytest.fixture
def state(tmp_path, monkeypatch) -> Path:
    """A private ASTRO_GPU_HOME: the timing histories of the tests never reach the user's."""
    s = tmp_path / "state"
    monkeypatch.setattr(project_mod, "STATE", s)
    return s


@pytest.fixture
def proj(tmp_path, state, monkeypatch):
    monkeypatch.setenv(GPU_DEAD, "")                    # (removed again after the test)
    RUNS.clear()
    SEEN.clear()
    Inv.fail = False
    B.stop_at = None
    raws = write_frames(tmp_path / "raw" / "s1", {"f1.ARW": 100, "f2.ARW": 200, "f3.ARW": 300})
    p = project2.create(tmp_path / "proj", "toy")
    p.sessions.append(Session(1, "night 1", lights=[str(raws)]))
    p.add_workflow("Classic", TEMPLATES["Classic stack"])
    p.add_workflow("Denoise", TEMPLATES["Classic + denoise"])
    project2.save(p)
    return p


# ---------------------------------------------------------------------------------------------- plans and runs

def test_shared_stages_run_once_events_carry_run_key_wf_and_a_second_plan_is_empty(proj):
    k = K(proj)
    plan = plan_all(proj)
    assert len(set(plan.keys)) == len(plan.keys) == 7          # inventory and stack once for both workflows
    assert [(j.stage, j.wf) for j in plan.jobs] == [
        ("inventory", [1, 2]), ("stack", [1, 2]), ("c", [1]), ("e", [1]),     # the cheaper workflow first
        ("b", [2]), ("c", [2]), ("e", [2])]
    assert plan.jobs[1].deps == {"inventory": k.key(CLASSIC, "inventory")} and plan.jobs[2].block == "color"
    assert all(j.est_s > 0 and not j.measured for j in plan.jobs) and not plan.measured
    status, info = run(proj, plan)
    assert status == "done"
    assert RUNS == {"inventory": 1, "stack": 1, "b": 1, "c": 2, "e": 2}
    evs = events(proj, info)
    assert evs and all(e["run"] == info.run_id for e in evs)
    assert [e["type"] for e in evs[:2]] == ["run_start", "plan"] and evs[-1]["type"] == "run_end"
    assert [j["key"] for j in evs[1]["jobs"]] == plan.keys
    staged = [e for e in evs if e.get("stage")]
    assert staged and all({"key", "wf", "block"} <= e.keys() for e in staged)
    ends = {(e["stage"], tuple(e["wf"])): e for e in evs if e["type"] == "stage_end"}
    assert ends[("stack", (1, 2))]["key"] == k.key(CLASSIC, "stack") and ends[("stack", (1, 2))]["status"] == "done"
    assert evs[-1]["status"] == "done" and evs[-1]["done"] == plan.keys
    store = Store(proj.root)
    for w in (CLASSIC, DENOISE):
        for s, key in k.keys(w).items():
            m = store.manifest(s, key)
            assert m["status"] == "done" and m["key"] == key and m["run_id"] == key and m["run"] == info.run_id
            assert m["key_parts"] == json.loads(json.dumps(k.key_parts(w, s)))
    assert not plan_all(proj)                                 # nothing left to do
    assert json.loads((store.dir("e", k.key(DENOISE, "e")) / "e.json").read_text())["steps"] == 10
    # the history learned the durations (on the CPU: the run's device): the next estimates are measured
    hist = est.load_history()
    assert {"inventory", "stack", "b", "c", "e"} <= set(hist) and hist["b"][0][2] == "cpu"
    proj.workflow(DENOISE).params["b"] = {"steps": 11}
    assert plan_all(proj, gpu=False).jobs[0].measured and not plan_all(proj, gpu=True).jobs[0].measured


def test_a_change_plans_only_its_downstream_and_reverting_plans_nothing(proj):
    assert run(proj, plan_all(proj))[0] == "done"
    k0 = K(proj)
    proj.workflow(DENOISE).params["b"] = {"steps": 20}
    plan = plan_all(proj)
    assert [(j.stage, j.wf) for j in plan.jobs] == [("b", [2]), ("c", [2]), ("e", [2])]
    proj.workflow(DENOISE).params.pop("b")
    assert not plan_all(proj)
    # run to a block: only that block and what it needs
    proj.workflow(CLASSIC).params["stack"] = {"kappa": 6.0}
    to_stack = runner2.plan_to_block(proj, CLASSIC, "stacking", keys=K(proj))
    assert [j.stage for j in to_stack.jobs] == ["stack"] and to_stack.targets == [(CLASSIC, "stack")]
    assert run(proj, to_stack)[0] == "done"
    assert [j.stage for j in runner2.plan_to_block(proj, CLASSIC, "output", keys=K(proj)).jobs] == ["c", "e"]
    assert K(proj).key(DENOISE, "stack") == k0.key(DENOISE, "stack")              # the other workflow kept its own
    assert not runner2.plan_to_block(proj, DENOISE, "output", keys=K(proj))


def test_every_job_writes_into_its_tmp_folder_and_renames_it_on_success(proj):
    assert run(proj, plan_all(proj))[0] == "done"
    assert len(SEEN) == 7 and all(d.endswith(".tmp") for d in SEEN)
    store = Store(proj.root)
    left = [d for d in store.path.rglob("*.tmp") if d.is_dir()]
    assert left == [] and not store.trash.exists() and not (store.path / "_absent").exists()


def test_a_failing_rerun_of_a_done_key_leaves_the_old_folder_intact(proj, tmp_path):
    assert run(proj, plan_all(proj))[0] == "done"
    k = K(proj)
    store = Store(proj.root)
    inv = store.dir("inventory", k.key(CLASSIC, "inventory"))
    old_frames = (inv / "frames.json").read_text()
    # the raws move (same names and sizes: same keys) and the Classic stack changes: the inventory must feed it
    # again, but its frames.json names files that are gone
    moved = tmp_path / "moved"
    shutil.move(str(tmp_path / "raw" / "s1"), str(moved))
    proj.sessions[0].lights = [str(moved)]
    proj.workflow(CLASSIC).params["stack"] = {"kappa": 6.0}
    plan = plan_all(proj)
    assert [(j.stage, j.rerun) for j in plan.jobs] == [("inventory", True), ("stack", False), ("c", False),
                                                       ("e", False)]
    Inv.fail = True
    status, info = run(proj, plan)
    assert status == "failed"
    assert (inv / "frames.json").read_text() == old_frames and store.manifest("inventory", inv.name)["status"] == "done"
    assert json.loads((store.tmp_dir("inventory", inv.name) / "manifest.json").read_text())["status"] == "failed"
    end = events(proj, info)[-1]
    assert end["failed"] == [inv.name] and len(end["skipped"]) == 3 and end["done"] == []
    # the drive is back: the re-run replaces the old folder, which goes to the trash only now
    Inv.fail = False
    status, info = run(proj, plan_all(proj))
    assert status == "done"
    paths = [Path(f["path"]) for f in json.loads((inv / "frames.json").read_text())["lights"]]
    assert [f.parent for f in paths] == [moved] * 3 and store.usable_as_input("inventory", inv.name)
    trashed = list(store.trash.iterdir())
    assert len(trashed) == 1 and (trashed[0] / "frames.json").read_text() == old_frames
    assert not store.tmp_dir("inventory", inv.name).exists()


def test_a_failed_job_skips_only_its_downstream_and_an_independent_workflow_completes(proj):
    proj.workflow(DENOISE).params["b"] = {"fail": True}
    plan = plan_all(proj)
    status, info = run(proj, plan)
    assert status == "failed"
    k = K(proj)
    evs = events(proj, info)
    ends = {e["key"]: e for e in evs if e["type"] == "stage_end"}
    assert ends[k.key(DENOISE, "b")]["status"] == "failed"
    for s in ("c", "e"):
        e = ends[k.key(DENOISE, s)]
        assert e["status"] == "skipped" and e["reason"] == "upstream failed" and e["wf"] == [DENOISE]
    store = Store(proj.root)
    assert all(store.done(s, key) for s, key in k.keys(CLASSIC).items())
    assert evs[-1]["status"] == "failed" and evs[-1]["skipped"] == [k.key(DENOISE, "c"), k.key(DENOISE, "e")]
    assert [j.stage for j in plan_all(proj).jobs] == ["b", "c", "e"]


def test_stop_interrupts_and_the_next_run_resumes_from_the_tmp_checkpoint(proj):
    B.stop_at = 4
    plan = plan_all(proj)
    status, info = run(proj, plan)
    assert status == "interrupted"
    k = K(proj)
    b = k.key(DENOISE, "b")
    end = events(proj, info)[-1]
    assert end["interrupted"] == [b] and end["not_started"] == [k.key(DENOISE, "c"), k.key(DENOISE, "e")]
    tmp = Store(proj.root).tmp_dir("b", b)
    assert json.loads((tmp / "_ckpt" / "i.json").read_text()) == {"i": 5}
    assert [j.stage for j in plan_all(proj).jobs] == ["b", "c", "e"]
    status, info = run(proj, plan_all(proj))                  # the launch removes STOP
    assert status == "done"
    out = json.loads((Store(proj.root).dir("b", b) / "b.json").read_text())
    assert out == {"steps": 10, "resumed_from": 5} and not tmp.exists()


@needs_gpu
def test_the_plan_is_frozen_with_the_whole_definition(proj):
    plan = plan_all(proj)
    info = runner2.prepare_run(proj, plan, "cuda:1")
    d = json.loads(Path(info.plan_file).read_text())
    assert d["definition"] == proj.to_dict() and d["run_id"] == info.run_id and d["device"] == "cuda:1"
    assert (d["visible"], d["inside"]) == ("1", "cuda:0") and [j["key"] for j in d["jobs"]] == plan.keys
    assert Path(info.plan_file).name == f"{info.run_id}.plan.json"
    # an edit after the launch changes nothing in this run
    proj.workflow(CLASSIC).params["stack"] = {"kappa": 9.0}
    project2.save(proj)
    assert runner2.run_plan(proj.root, info.plan_file, rules=RULES, echo=False) == "done"
    assert all(Store(proj.root).done(j.stage, j.key) for j in plan.jobs)
    assert [j.stage for j in plan_all(proj).jobs] == ["stack", "c", "e"]


def test_a_run_that_cannot_start_still_ends_for_its_follower(proj):
    plan = plan_all(proj)
    plan.rules = "no_such_module:rules"
    info = runner2.prepare_run(proj, plan)
    assert runner2.run_plan(proj.root, info.plan_file, echo=False) == "failed"
    end = events(proj, info)[-1]
    assert end["type"] == "run_end" and end["status"] == "failed" and end["not_started"] == plan.keys
    assert "no_such_module" in end["error"] and not (proj.root / "logs" / "run.lock").exists()


def test_launch_removes_stop_and_rotates_large_events(proj, monkeypatch):
    logs = proj.root / "logs"
    logs.mkdir()
    (logs / "STOP").write_text("stop")
    (logs / "events.jsonl").write_bytes(b'{"t": 0, "type": "log"}\n' * 100)
    monkeypatch.setattr(runner2, "EVENTS_LIMIT", 1000)
    info = runner2.prepare_run(proj, plan_all(proj))
    assert not (logs / "STOP").exists() and info.offset == 0 and not (logs / "events.jsonl").exists()
    rotated = list(logs.glob("events_*.jsonl"))
    assert len(rotated) == 1 and rotated[0].stat().st_size == 2400
    info2 = runner2.prepare_run(proj, plan_all(proj))           # small: kept
    assert info2.offset == 0 and len(list(logs.glob("events_*.jsonl"))) == 1


def test_real_stage_graph_plans_each_key_once_and_cheap_workflows_first(proj):
    """The three templates and a Raw stack exit on the v2 stages: one job per distinct key, in dependency order,
    the cheap workflows' outputs first (planning only)."""
    p = project2.create(proj.root.parent / "real", "real")
    p.sessions = [Session(1, "night", lights=proj.sessions[0].lights)]
    ids = {n: p.add_workflow(n, TEMPLATES[n]).id for n in TEMPLATES}
    raw = p.add_workflow("Raw stack", TEMPLATES["Classic stack"], end="stacking").id
    plan = runner2.plan_all(p, history={})
    k = Keys(p)
    distinct = {k.key(w.id, s) for w in p.workflows for s in k.chain(w.id)}
    assert sorted(plan.keys) == sorted(distinct) and len(plan) == 13
    pos = {(j.stage, tuple(j.wf)): i for i, j in enumerate(plan.jobs)}
    for i, j in enumerate(plan.jobs):
        assert all(plan.keys.index(d) < i for d in j.deps.values())
    by_stage = {}
    for j in plan.jobs:
        by_stage.setdefault(j.stage, []).append(j)
    assert [j.wf for j in by_stage["stack"]] == [sorted([*ids.values(), raw])]
    assert [j.wf for j in by_stage["denoise"]] == [sorted(ids[n] for n in ("Classic + denoise", "Full"))]
    assert [j.wf for j in by_stage["render_simple"] if ids["Full"] in j.wf] == [sorted(ids[n] for n in (
        "Classic + denoise", "Full"))]                                  # the Output's image: shared, the block is not
    assert plan.keys.index(k.key(raw, "render_simple")) == plan.keys.index(k.key(raw, "stack")) + 1
    classic_out = plan.keys.index(k.key(ids["Classic stack"], "render_simple"))
    assert classic_out < plan.keys.index(k.key(ids["Full"], "denoise")) < plan.keys.index(
        k.key(ids["Full"], "star_removal"))
    assert pos[("star_removal", (ids["Full"],))] > pos[("render_simple", tuple(sorted((ids["Full"],
                                                                                       ids["Classic + denoise"]))))]
    assert all(j.block == wf_block for j in plan.jobs for wf_block in [k.rules.chains.block_of(j.stage)])
    assert plan.seconds > 300 and not plan.measured


# ---------------------------------------------------------------------------------------------- the lock

def test_run_lock_is_created_with_o_excl_and_a_stale_lock_is_replaced(proj, monkeypatch):
    logs = proj.root / "logs"
    flags = []
    real_open = os.open
    monkeypatch.setattr(runner2.os, "open", lambda f, fl, *a: (flags.append(fl), real_open(f, fl, *a))[1])
    assert runner2.acquire_lock(logs, "r1")
    assert flags and flags[0] & os.O_EXCL and flags[0] & os.O_CREAT
    assert not runner2.acquire_lock(logs, "r2")                # held by this live process
    runner2.release_lock(logs, "r2")                           # not its lock: kept
    assert json.loads((logs / "run.lock").read_text())["run_id"] == "r1"
    runner2.release_lock(logs, "r1")
    assert not (logs / "run.lock").exists()
    monkeypatch.undo()
    # a live worker of another run: refused, its lock untouched
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        time.sleep(0.5)
        held = json.dumps({"pid": sleeper.pid, "run_id": "other", "started": time.time()})
        (logs / "run.lock").write_text(held)
        with pytest.raises(RuntimeError, match="already in progress"):
            runner2.prepare_run(proj, plan_all(proj))
        assert runner2.running(proj.root).pid == sleeper.pid
        plan_file = proj.root / "logs" / "runs" / "manual.plan.json"
        plan_file.parent.mkdir(parents=True)
        plan_file.write_text(json.dumps({"run_id": "manual", **plan_all(proj).to_dict(), "definition": proj.to_dict(),
                                         "device": "cpu", "visible": "-1", "inside": "cpu"}))
        assert runner2.run_plan(proj.root, plan_file, rules=RULES, echo=False) == "refused"
        assert (logs / "run.lock").read_text() == held and RUNS == {}
    finally:
        sleeper.kill()
        sleeper.wait()
    # its process is gone: the lock is stale and replaced
    assert runner2.running(proj.root) is None
    status, info = run(proj, plan_all(proj))
    assert status == "done" and not (logs / "run.lock").exists()
    # a pid reused by a process started after the lock is not the worker
    (logs / "run.lock").write_text(json.dumps({"pid": os.getpid(), "run_id": "old", "started": 1.0}))
    assert runner2.running(proj.root) is None and runner2.acquire_lock(logs, "new")
    assert json.loads((logs / "run.lock").read_text())["run_id"] == "new"


# ---------------------------------------------------------------------------------------------- the worker process

@needs_gpu
def test_worker_environment_and_creation_flags(proj, monkeypatch):
    got = []

    class FakePopen:
        pid = 4242

        def __init__(self, cmd, **kw):
            got.append((cmd, kw))
            if len(got) == 1 and kw.get("creationflags", 0) & runner2.CREATE_BREAKAWAY_FROM_JOB and refuse:
                raise PermissionError(5, "Access is denied")

    refuse = False
    monkeypatch.setattr(runner2.subprocess, "Popen", FakePopen)
    info = runner2.launch(proj, plan_all(proj), "cuda:1")
    cmd, kw = got[0]
    assert cmd[1:] == ["-m", "astro_gpu.core.runner2", info.root, info.plan_file] and info.pid == 4242
    assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "1" and kw["cwd"] == info.root
    assert json.loads(Path(info.plan_file).read_text())["inside"] == "cuda:0"
    if os.name == "nt":
        assert kw["creationflags"] & runner2.CREATE_BREAKAWAY_FROM_JOB
        assert kw["creationflags"] & runner2.CREATE_NO_WINDOW and kw["creationflags"] & runner2.CREATE_NEW_PROCESS_GROUP
        assert not kw["creationflags"] & runner2.DETACHED_PROCESS     # its console children would pop up windows
        # a job that forbids breaking away: launched anyway, with a note
        got.clear()
        refuse = True
        info = runner2.launch(proj, plan_all(proj), "cpu")
        assert len(got) == 2 and not got[1][1]["creationflags"] & runner2.CREATE_BREAKAWAY_FROM_JOB and info.notes
        assert got[1][1]["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert runner2.device_env("cuda:1") == ("1", "cuda:0") and runner2.device_env(None) == ("0", "cuda:0")
    assert runner2.device_env("cpu") == ("-1", "cpu") and runner2.device_env("cuda:0", "") == ("-1", "cpu")
    assert runner2.device_env("cuda:1", "1,0") == ("0", "cuda:0")
    with pytest.raises(ValueError):
        runner2.device_env("gpu2")


@needs_gpu
def test_a_detached_worker_survives_its_launcher_and_is_followed_through_events(proj, state):
    import psutil
    proj.workflow(CLASSIC).params["c"] = {"sleep": 3.0}
    project2.save(proj)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")]),
           "ASTRO_GPU_HOME": str(state)}
    code = ("import json, sys\n"
            "from astro_gpu.core import project2, runner2\n"
            "import test_runner2 as t\n"
            f"p = project2.load({str(proj.root)!r})\n"
            "plan = runner2.plan_workflows(p, [1], keys=t.K(p))\n"
            "plan.rules = 'test_runner2:toy_rules'\n"
            "info = runner2.launch(p, plan, 'cuda:1')\n"
            "print(json.dumps(info.__dict__))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env,
                         cwd=str(TESTS.parent))
    assert out.returncode == 0, out.stderr
    info = json.loads(out.stdout.strip().splitlines()[-1])           # the launcher has exited
    assert info["notes"] == []                                       # the worker left the launcher's job
    lock = proj.root / "logs" / "run.lock"
    t0 = time.time()
    while not lock.exists() and time.time() - t0 < 60:
        time.sleep(0.1)
    worker = json.loads(lock.read_text())["pid"]
    assert worker != os.getpid() and psutil.pid_exists(worker)
    assert runner2.running(proj.root).run_id == info["run_id"]
    evs, offset = [], info["offset"]
    while time.time() - t0 < 120 and not (evs and evs[-1]["type"] == "run_end"):
        got, offset = runner2.read_events(proj.root, offset, info["run_id"])
        evs += got
        time.sleep(0.2)
    assert evs and evs[-1]["type"] == "run_end" and evs[-1]["status"] == "done", \
        Path(info["log_file"]).read_text(errors="replace")
    assert any(e["type"] == "progress" and e["stage"] == "c" for e in evs)
    c = json.loads((Store(proj.root).dir("c", K(proj).key(CLASSIC, "c")) / "c.json").read_text())
    assert c["cuda_visible"] == "1" and c["device"] == "cuda:0"
    assert os.path.normcase(c["cwd"]) == os.path.normcase(info["root"])
    assert "run done" in Path(info["log_file"]).read_text(errors="replace")      # the worker's output
    t1 = time.time()
    while lock.exists() and time.time() - t1 < 30:
        time.sleep(0.1)
    assert not lock.exists() and runner2.running(proj.root) is None


# ---------------------------------------------------------------------------------------------- a GPU fault

LOST_C = "the graphics card stopped responding during C; the rest of the run continues in a fresh process"
LOST_C_AGAIN = ("the graphics card stopped responding again during C: it is not tried a third time; the rest of the "
                "run continues in a fresh process")


def _gpu_plan(proj, faults: int) -> runner2.Plan:
    """The plan of both workflows, the Classic C faulting ``faults`` times, with a CUDA context check that reports
    dead in the process where it faulted."""
    (proj.root / GPU_FAULT).write_text(str(faults))
    plan = plan_all(proj)
    plan.rules, plan.gpu_check = "test_runner2:toy_rules", "test_runner2:fake_gpu"
    return plan


def test_a_gpu_fault_hands_the_rest_of_the_run_and_the_lock_to_a_fresh_worker(proj, monkeypatch):
    started = []
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])       # the new worker
    monkeypatch.setattr(runner2, "_spawn_worker", lambda *a, **kw: (started.append(a), sleeper.pid)[1])
    try:
        plan = _gpu_plan(proj, 1)
        k = K(proj)
        c1 = k.key(CLASSIC, "c")
        assert plan.keys[2] == c1
        status, info = run(proj, plan)
        assert status == "handover" and RUNS["e"] == 0 and RUNS["b"] == 0      # nothing more in the dead process
        holder = json.loads((proj.root / "logs" / "run.lock").read_text())
        assert holder["pid"] == sleeper.pid and holder["run_id"] == info.run_id
        assert runner2.running(proj.root).pid == sleeper.pid                   # never free for another run
        with pytest.raises(RuntimeError, match="already in progress"):
            runner2.prepare_run(proj, plan_all(proj))
        root, part_file, visible, log_file = started[0][:4]
        assert Path(part_file).name == f"{info.run_id}.part2.plan.json" and visible == "-1"
        assert Path(log_file) == Path(info.log_file) and Path(root) == Path(info.root)
        part = json.loads(Path(part_file).read_text())
        assert [j["key"] for j in part["jobs"]] == plan.keys[2:]               # the failed job first, then the rest
        assert part["run_id"] == info.run_id and part["gpu_check"] == plan.gpu_check
        assert part["definition"] == proj.to_dict() and part["rules"] == plan.rules
        r = part["resume"]
        assert r["part"] == 2 and r["from"]["pid"] == os.getpid() and r["retried"] == [c1]
        assert r["done"] == plan.keys[:2] and r["failed"] == [] and r["skipped"] == [] and r["bad"] == []
        evs = events(proj, info)
        assert not [e for e in evs if e["type"] == "run_end"]                  # the run goes on in the new worker
        lost = [e for e in evs if e.get("gpu_lost")]
        assert [e["msg"] for e in lost] == [LOST_C] and lost[0]["level"] == "error"
        assert lost[0]["pid"] == sleeper.pid and lost[0]["retry"] is True and lost[0]["job"] == c1
    finally:
        sleeper.kill()
        sleeper.wait()


def test_without_a_fresh_worker_the_run_ends_failed_and_a_plain_failure_is_unchanged(proj, monkeypatch):
    def refuse(*a, **kw):
        raise OSError("no more processes")

    monkeypatch.setattr(runner2, "_spawn_worker", refuse)
    plan = _gpu_plan(proj, 1)
    c1 = K(proj).key(CLASSIC, "c")
    status, info = run(proj, plan)
    assert status == "failed" and not (proj.root / "logs" / "run.lock").exists()
    evs = events(proj, info)
    end = evs[-1]
    assert end["type"] == "run_end" and end["status"] == "failed"
    assert end["done"] == plan.keys[:2] and end["failed"] == [c1] and end["not_started"] == plan.keys[3:]
    lost = [e["msg"] for e in evs if e.get("gpu_lost")]
    assert len(lost) == 1 and lost[0].startswith("the graphics card stopped responding during C and the run cannot "
                                                 "go on: no fresh process could take over")
    # a failure that is not a GPU fault (the context check says it works): skips its downstream only, as before
    (proj.root / GPU_FAULT).unlink()
    monkeypatch.setenv(GPU_DEAD, "")                     # this process's (fake) CUDA context works again
    proj.workflow(DENOISE).params["b"] = {"fail": True}
    plan = plan_all(proj)
    plan.gpu_check = "test_runner2:fake_gpu"
    status, info = run(proj, plan)
    k = K(proj)
    end = events(proj, info)[-1]
    assert status == "failed" and end["failed"] == [k.key(DENOISE, "b")]
    assert end["skipped"] == [k.key(DENOISE, "c"), k.key(DENOISE, "e")] and sorted(end["done"]) == sorted(
        [k.key(CLASSIC, "c"), k.key(CLASSIC, "e")])
    assert not [e for e in events(proj, info) if e.get("gpu_lost")]


@pytest.mark.parametrize("faults", [1, 2])
def test_after_a_gpu_fault_a_fresh_worker_retries_the_job_once_and_the_lock_is_held_throughout(proj, state,
                                                                                                 monkeypatch, faults):
    """Real detached workers on the CPU: the Classic C faults (and its CUDA context is reported dead) once, or twice.
    Once: a second worker tries it again and the run completes. Twice: its second failure is a plain one (its
    downstream skipped), and a third worker finishes the rest. One run for the follower, the lock never free."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(state))
    plan = _gpu_plan(proj, faults)
    k = K(proj)
    c1, e1 = k.key(CLASSIC, "c"), k.key(CLASSIC, "e")
    info = runner2.launch(proj, plan, "cpu")
    lock = proj.root / "logs" / "run.lock"
    holders: list[int] = []
    free, evs, offset, t0 = 0, [], info.offset, time.time()
    while time.time() - t0 < 300 and not (evs and evs[-1]["type"] == "run_end"):
        h = runner2._lock_holder(lock)
        missing = not lock.exists()
        got, offset = runner2.read_events(proj.root, offset, info.run_id)
        evs += got
        if h is not None and (not holders or holders[-1] != h["pid"]):
            holders.append(int(h["pid"]))
        elif missing and holders and not any(e["type"] == "run_end" for e in evs):
            free += 1                                   # free while the run was not over (run_end comes first)
        time.sleep(0.01)
    log = Path(info.log_file).read_text(errors="replace")
    assert evs and evs[-1]["type"] == "run_end", log
    assert [e["type"] for e in evs].count("run_start") == 1 and [e["type"] for e in evs].count("run_end") == 1
    goes_on = [e for e in evs if e["type"] == "log" and e["msg"] == "a fresh process goes on with the run"]
    workers = [next(e["pid"] for e in evs if e["type"] == "run_start")] + [e["pid"] for e in goes_on]
    # each worker held the lock in turn (with, on Windows, the venv launcher of a new one for a moment in between)
    assert free == 0 and len(set(workers)) == faults + 1 and set(workers) <= set(holders), (holders, workers, free)
    assert [h for h in holders if h in workers] == workers
    assert [e["status"] for e in evs if e["type"] == "stage_end" and e["key"] == c1] == \
        (["failed", "done"] if faults == 1 else ["failed", "failed"])
    lost = [e["msg"] for e in evs if e.get("gpu_lost")]
    end = evs[-1]
    others = [x for x in plan.keys if x not in (c1, e1)]
    if faults == 1:
        assert lost == [LOST_C], log
        assert end["status"] == "done" and sorted(end["done"]) == sorted(plan.keys) and not end["failed"]
        c = json.loads((Store(proj.root).dir("c", c1) / "c.json").read_text())
        assert c["pid"] == workers[1]                   # computed again by the fresh worker
    else:
        assert lost == [LOST_C, LOST_C_AGAIN], log
        assert end["status"] == "failed" and end["failed"] == [c1] and end["skipped"] == [e1]
        assert sorted(end["done"]) == sorted(others) and not Store(proj.root).done("c", c1)
        part3 = json.loads((proj.root / "logs" / "runs" / f"{info.run_id}.part3.plan.json").read_text())
        assert [j["key"] for j in part3["jobs"]] == plan.keys[3:] and part3["resume"]["failed"] == [c1]
    assert (proj.root / GPU_FAULT).read_text() == "0"
    t1 = time.time()
    while lock.exists() and time.time() - t1 < 30:
        time.sleep(0.1)
    assert not lock.exists() and runner2.running(proj.root) is None


def test_the_new_worker_adopts_the_lock_and_waits_for_the_old_one(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    lock = logs / "run.lock"
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        time.sleep(0.5)
        old = {"pid": sleeper.pid, "started": time.time()}
        # another run's live worker holds it: refused
        lock.write_text(json.dumps({**old, "run_id": "other"}))
        assert not runner2.adopt_lock(logs, "r", old, timeout=1.0)
        assert json.loads(lock.read_text())["run_id"] == "other"
        # the old worker of this run handed it over: this process holds it
        lock.write_text(json.dumps({"pid": os.getpid(), "run_id": "r", "started": time.time()}))
        assert runner2.adopt_lock(logs, "r", old, timeout=1.0)
        # the old worker still holds it and never hands it: waits, then gives up
        lock.write_text(json.dumps({**old, "run_id": "r"}))
        t0 = time.time()
        assert not runner2.adopt_lock(logs, "r", old, timeout=0.5) and time.time() - t0 >= 0.5
        # an old worker that does not exit is ended
        assert "ended" in runner2.wait_gone(old, timeout=0.3)
        assert sleeper.wait(10) is not None
    finally:
        sleeper.kill()
        sleeper.wait()
    # it is gone without handing the lock over: the stale lock is taken
    assert runner2.wait_gone(old, timeout=0.3) == ""
    assert runner2.adopt_lock(logs, "r", old, timeout=1.0)
    assert json.loads(lock.read_text())["pid"] == os.getpid()


def test_the_cuda_check(monkeypatch):
    import types
    if _gpu_hidden():
        import torch  # noqa: F401
        assert runner2.cuda_alive("cuda:0") is None              # CUDA never used in this process: nothing to check
    else:
        import torch
        torch.ones(1, device="cuda:0")
        assert runner2.cuda_alive("cuda:0") is True

    def fake(op):
        cuda = types.SimpleNamespace(is_initialized=lambda: True, synchronize=lambda d=None: None)
        return types.SimpleNamespace(cuda=cuda, ones=op)

    def dead(*a, **kw):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    def hung(*a, **kw):
        time.sleep(5)

    monkeypatch.setitem(sys.modules, "torch", fake(dead))
    assert runner2.cuda_alive("cuda:0") is False
    monkeypatch.setitem(sys.modules, "torch", fake(hung))
    t0 = time.time()
    assert runner2.cuda_alive("cuda:0", timeout=0.3) is False and time.time() - t0 < 2


# ---------------------------------------------------------------------------------------------- estimates

def test_estimates_follow_units_history_and_the_cpu():
    f = est.Facts(frames=100, lights=80, mpx=24.0, scale=1.0, grid_mpx=24.0, iterations=500)
    heart = est.HEART_FACTS
    e = est.estimate("stack", f, history={})
    assert not e.measured and e.seconds == pytest.approx(2064.0 * 80 * 24.0 / (721 * heart.mpx * 2.25))
    assert est.estimate("calibrate", f, history={}).disk_gb == pytest.approx(0.11 * 80 * 24 / (721 * heart.mpx)
                                                                             + 80 * 24e6 * 2 / 1e9)
    # Denoise: its measured times per adaptation step and per Mpx applied, not the reference x the CPU factor
    assert est.estimate("denoise", f, gpu=False, history={}).seconds == pytest.approx(3.6 * 500 + 3.3 * 24.0)
    assert est.estimate("denoise", f, history={}).seconds == pytest.approx(0.088 * 500 + 0.75 * 24.0)
    assert est.units("denoise", f) == pytest.approx(500 + 8.5 * 24.0)
    hist = {"stack": [[100.0, 1000.0, "gpu"], [300.0, 1000.0, "gpu"], [9.0, 1.0, "cpu"]]}
    m = est.estimate("stack", f, history=hist)
    assert m.measured and m.seconds == pytest.approx(0.2 * 80 * 24.0)
    assert est.estimate("stack", f, gpu=False, history=hist).seconds == pytest.approx(9.0 * 80 * 24.0)
    assert not est.estimate("stack", f, gpu=False, history={"stack": hist["stack"][:2]}).measured
    assert est.denoise_steps("pretrained", None) == 0 and est.denoise_steps("adapted", None) == 500
    assert est.denoise_steps("adapted", "long") == 4000


def test_record_keeps_the_last_20_written_atomically(state):
    for i in range(25):
        est.record("stack", 10.0 + i, 5.0, True)
    h = est.load_history()
    assert len(h["stack"]) == 20 and h["stack"][-1] == [34.0, 5.0, "gpu"]
    assert sorted(p.name for p in state.iterdir()) == ["timings2.json"]     # no temporary file left


def test_workers_recording_at_once_all_keep_their_entries(tmp_path, monkeypatch):
    """The GPU queues' workers finish stages at the same time: each read-change-write of timings2.json holds a lock
    file, so no writer rewrites the file from a copy older than another one's entry. A lock left by a dead process
    (old) is removed; a live one is waited for LOCK_WAIT at most, then the entry is written anyway."""
    import threading
    p = tmp_path / "timings2.json"
    go = threading.Barrier(8)

    def worker(i):
        go.wait()
        for k in range(10):
            est.record(f"s{i}", 1.0 + k, 1.0, True, path=p)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    h = est.load_history(p)
    assert {s: [r[0] for r in rows] for s, rows in h.items()} == {f"s{i}": [1.0 + k for k in range(10)] for i in range(8)}
    assert sorted(x.name for x in tmp_path.iterdir()) == ["timings2.json"]                # the lock is gone
    lock = tmp_path / "timings2.json.lock"
    lock.write_text("")
    old = time.time() - est.LOCK_STALE - 5
    os.utime(lock, (old, old))
    est.record("stale", 1.0, 1.0, True, path=p)
    assert "stale" in est.load_history(p) and not lock.exists()
    lock.write_text("")                                                                  # a live writer
    monkeypatch.setattr(est, "LOCK_WAIT", 0.3)
    t0 = time.monotonic()
    est.record("waited", 1.0, 1.0, True, path=p)
    assert 0.3 <= time.monotonic() - t0 < 3 and "waited" in est.load_history(p) and lock.exists()


def test_a_manifest_a_reader_holds_open_is_replaced_once_it_lets_go(tmp_path, monkeypatch):
    """Windows refuses to replace a file that a reader has open (the client reading a job's manifest): the runners'
    JSON writes retry the replace for a few seconds instead of failing the job, and leave no temporary file."""
    import threading
    from types import SimpleNamespace

    from astro_gpu.core import files
    from astro_gpu.core import runner as runner_mod
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps({"status": "running"}))
    reader = open(mf, "rb")
    if os.name == "nt":                                   # the condition the retry is for
        probe = tmp_path / "probe.json"
        probe.write_text("{}")
        with pytest.raises(PermissionError):
            os.replace(probe, mf)
        probe.unlink()
    threading.Timer(0.4, reader.close).start()
    files.write_json(mf, {"status": "done"})                                         # runner2: the commit
    assert json.loads(mf.read_text()) == {"status": "done"}
    reader = open(mf, "rb")
    threading.Timer(0.4, reader.close).start()
    runner_mod._write_manifest(SimpleNamespace(stage_dir=lambda s: tmp_path), "x", {"status": "failed"})   # v1's
    assert json.loads(mf.read_text()) == {"status": "failed"}
    assert sorted(x.name for x in tmp_path.iterdir()) == ["manifest.json"]
    # a reader that never lets go: the error comes after the retries, and the temporary file is removed
    if os.name == "nt":
        with open(mf, "rb"):
            src = tmp_path / "new.json"
            src.write_text("{}")
            with pytest.raises(PermissionError):
                files.replace_file(src, mf, attempts=3)
            src.unlink()

    def refuse(src, dst):
        raise PermissionError("held")

    monkeypatch.setattr(files, "replace_file", refuse)
    with pytest.raises(PermissionError):
        files.write_json(mf, {"status": "x"})
    assert sorted(x.name for x in tmp_path.iterdir()) == ["manifest.json"]
    assert json.loads(mf.read_text()) == {"status": "failed"}


IRIS = Path("C:/Users/Nicolas/astro_gpu_scratch/projects/Iris")
OTHERS = [Path(p) for p in ("F:/photos/heart_nebula/astro_gpu", "F:/photos/NGC7822/astro_gpu",
                            "E:/astro_gpu/projects/FlamingStar", "E:/astro_gpu/projects/Horsehead",
                            "E:/astro_gpu/projects/Rosette", "E:/astro_gpu/projects/Wizard",
                            "E:/astro_gpu/projects/M31", "C:/Users/Nicolas/astro_gpu_scratch/projects/M33")]


def iris_check(history: dict) -> tuple[float, float, dict]:
    """(estimated, measured seconds, {stage: ratio}) of Iris's v2 workflow (read only)."""
    v2 = project2.from_v1(IRIS)
    w = v2.workflows[0]
    from astro_gpu.core import workflow as wfl
    chain = wfl.chain(w.options, w.end, reference=bool(v2.reference_image))
    measured = est.v1_durations(IRIS)
    f = est.v1_facts(IRIS)
    e = {s: est.estimate(s, f, True, history).seconds for s in chain}
    return sum(e.values()), sum(measured[s] for s in chain), {s: round(e[s] / max(measured[s], 1), 2) for s in chain}


@pytest.mark.skipif(not (IRIS / "work").is_dir() or not all((p / "work").is_dir() for p in OTHERS),
                    reason="the user's real projects are not here")
def test_estimates_are_within_50_percent_of_iris_measured_times(tmp_path, capsys):
    """Iris's whole run, estimated from the history the 8 other real projects give (read only), and from the
    Heart reference alone (≈)."""
    for r in [IRIS, *OTHERS]:
        assert not (r / "project.astroproj").exists()
    hist_file = tmp_path / "timings2.json"
    assert est.seed([e for r in OTHERS for e in est.learn_from_v1(r)], hist_file) > 100
    total, measured, ratios = iris_check(est.load_history(hist_file))
    ref_total, _, ref_ratios = iris_check({})
    with capsys.disabled():
        print(f"\nIris: measured {measured / 60:.1f} min; estimated {total / 60:.1f} min from the other projects "
              f"({100 * (total / measured - 1):+.0f} %), {ref_total / 60:.1f} min from Heart alone "
              f"({100 * (ref_total / measured - 1):+.0f} %)\n  per stage, measured history: {ratios}\n"
              f"  per stage, Heart alone: {ref_ratios}")
    assert 0.5 <= total / measured <= 1.5


# ---------------------------------------------------------------------------------------------- the engine

def test_select_frames_keeps_its_positional_signature_and_honours_exclude():
    from astro_gpu.stages.stack import EXCLUDED, excluded_mask, select_frames
    rng = np.random.default_rng(0)
    n = 12
    transp = rng.uniform(0.7, 1.0, n)
    transp[3] = 0.1
    fwhm = rng.uniform(2.0, 2.5, n)
    rms = np.full(n, 0.3)
    rms[5] = np.nan
    stats = rng.uniform(1, 2, (n, 4, 2))
    w0, r0 = select_frames(transp, fwhm, rms, stats, 0.6, 2.0)            # positional, as ComfyUI's block calls it
    ex = np.zeros(n, bool)
    ex[[1, 3, 5]] = True
    w1, r1 = select_frames(transp, fwhm, rms, stats, 0.6, 2.0, exclude=ex)
    assert list(r1[[1, 3, 5]]) == [EXCLUDED, EXCLUDED, "not aligned"] and w1[1] == 0
    keep = ~ex
    assert list(r0[keep]) == list(r1[keep]) and np.array_equal(w0[keep], w1[keep])   # the others are unchanged
    assert r0[3] == "low transparency"
    paths = ["F:/raw/A.ARW", "F:\\raw\\B.ARW", "F:/raw/C.ARW"]
    assert list(excluded_mask(paths, ["f:\\raw\\a.arw" if os.name == "nt" else "F:/raw/A.ARW", "F:/raw/B.ARW"])) == \
        [True, True, False]


def test_exclude_is_internal_and_old_stacks_stay_done():
    from astro_gpu.core.runner import _same_params
    from astro_gpu.stages.stack import Stack, StackParams
    d = {x["name"]: x for x in StackParams.describe()}
    assert d["exclude"]["level"] == "internal" and d["exclude"]["default"] == []
    stored = StackParams().model_dump(mode="json")
    stored.pop("exclude")                                   # a manifest written before the parameter existed
    assert _same_params(Stack(), StackParams(), stored)
    assert not _same_params(Stack(), StackParams(exclude=["F:/a.ARW"]), stored)


HEART_A7S_RAW = Path("F:/astro_essais/donnees/heart_a7s")


@pytest.mark.slow
@pytest.mark.skipif(not (HEART_A7S_RAW / "lights").is_dir(), reason="the heart_a7s raws are not here")
def test_cache_only_calibrate_gives_the_same_frames_as_a_full_run(tmp_path, state):
    """8 heart_a7s frames: inventory and calibrate run; the calibrated cache is lost; the re-run copies the masters
    and rebuilds the cache only, bit for bit the same."""
    import hashlib
    p = project2.create(tmp_path / "v2", "heart8")
    p.sessions.append(Session(1, "night", lights=[str(HEART_A7S_RAW / "lights")],
                              darks=[str(HEART_A7S_RAW / "darks")], flats=[str(HEART_A7S_RAW / "flats")]))
    p.add_workflow("Classic", TEMPLATES["Classic stack"], {"inventory": {"max_lights": 8},
                                                          "calibrate": {"workers": 4}})
    project2.save(p)
    plan = runner2.plan_to_block(p, 1, "calibration")
    assert [j.stage for j in plan.jobs] == ["inventory", "calibrate"]
    assert runner2.run_plan(p.root, runner2.prepare_run(p, plan, "cpu").plan_file, echo=False) == "done"
    k = Keys(p)
    key = k.key(1, "calibrate")
    store = Store(p.root)
    cal = store.dir("calibrate", key)
    cache = Path(json.loads((cal / "cache.json").read_text())["path"])
    assert cache.is_file() and str(cache).startswith(str(tmp_path))

    def digest(f: Path) -> str:
        return hashlib.sha1(f.read_bytes()).hexdigest()

    full = {"cache": digest(cache), "stats": digest(cal / "frame_stats.npy"), "dark": digest(cal / "master_dark.npy")}
    metrics = store.manifest("calibrate", key)["metrics"]
    cache.unlink()
    assert not Store(p.root).usable_as_input("calibrate", key)
    plan = runner2.plan_to_block(p, 1, "registration")
    assert [(j.stage, j.rerun) for j in plan.jobs] == [("calibrate", True), ("register", False)]
    plan.jobs = plan.jobs[:1]                                   # the calibration only
    info = runner2.prepare_run(p, plan, "cpu")
    assert runner2.run_plan(p.root, info.plan_file, echo=False) == "done"
    msgs = [e.get("msg") or "" for e in runner2.read_events(p.root, info.offset, info.run_id)[0]]
    assert any("rebuilt with the masters" in m for m in msgs)
    assert not any("reading darks" in m or "reading flats" in m for m in msgs)      # no master was made again
    assert {"cache": digest(cache), "stats": digest(cal / "frame_stats.npy"),
            "dark": digest(cal / "master_dark.npy")} == full
    assert store.manifest("calibrate", key)["metrics"] == metrics and not (cal / "masters.json").exists()
    assert len(list(store.trash.iterdir())) == 1 and Store(p.root).usable_as_input("calibrate", key)
