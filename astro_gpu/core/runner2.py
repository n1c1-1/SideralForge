"""Runner v2: plans the stage jobs of a v2 project's workflows and runs them in a detached worker (DESIGN §3).

A job computes one stage key (core/store.py): a result shared by several workflows is computed once. The plan::

    need(wf, s, as_input=False):
        k = key(wf, s)
        if done(k) and (not as_input or usable_as_input(s, k)) or k in jobs: return
        for d in deps_in_chain(wf, s): need(wf, d, as_input=True)
        jobs[k] = Job(k, s, block, wf=the workflows sharing k, deps={d: key(wf, d)}, est_s, disk_gb)

Jobs come in dependency order; among the jobs ready to run, those of the workflow with the least remaining time go
first, so cheap workflows finish first.

A run (``launch``, called by the API without Qt): STOP removed, events.jsonl rotated past 64 MB, the plan frozen in
``logs/runs/<run_id>.plan.json`` with a snapshot of the whole definition, then ``python -m astro_gpu.core.runner2
<project> <plan>`` started detached (it survives the process and the session that launched it), in the project's
folder, logging to ``logs/runs/<run_id>.log``, with ``CUDA_VISIBLE_DEVICES`` set to the run's GPU (cuda:0 inside,
so the stages that name cuda:0 stay on it).

The worker takes ``logs/run.lock`` (O_EXCL; a lock whose process is gone is replaced), then runs each job with v1's
``Pipeline._run_stage`` through the store's shim, into ``store/<stage>/<key>.tmp``, renamed ``<key>`` on success; a
done folder it replaces (a result that lost its external data) goes to ``store/_trash/`` only then, so a failed
re-run never destroys a result. A failed job skips only the jobs downstream of it. STOP ends the run; the ``.tmp``
folder keeps its checkpoints and the next run resumes there. Every event carries ``run``; stage events add ``key``,
``wf`` (workflow ids) and ``block``; a ``plan`` event lists the jobs; ``run_end`` lists the jobs done, failed,
skipped, interrupted and not started.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import estimate as est
from . import workflow as wfl
from .journal import Journal
from .journal import read_events as _read_events
from .params import params_hash
from .project2 import Project2, Workflow
from .runner import Pipeline, read_manifest, replace_file
from .stage import StageError
from .store import InputScan, KeyRules, Keys, Store, StoreProject, key_rules

LOCK = "run.lock"
STOP = "STOP"
EVENTS = "events.jsonl"
EVENTS_LIMIT = 64 * 1024 * 1024            # events.jsonl is rotated past this size when a run starts
RUNS = "runs"
#: an earlier learned star layer ``inputs/<layer>.npy`` may come with ``inputs/<layer>.json`` holding the key of the
#: stack it was made on under this name (written by the import); a workflow using it runs only on that stack
LAYER_STACK_KEY = "stack_key"
LAYER_OTHER_STACK = "the earlier layer belongs to another stack"

# Windows process creation flags of the detached worker
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


# ---------------------------------------------------------------------------------------------- the plan

@dataclass
class Job:
    """One stage key to compute."""

    key: str
    stage: str
    block: str
    wf: list[int]                    # the workflows that share this result
    run_wf: int                      # the workflow whose settings the job runs with (any of wf: the same key)
    deps: dict[str, str]             # {stage: key} of the results it reads
    est_s: float = 0.0
    measured: bool = False           # est_s from this machine's history (else ≈)
    disk_gb: float = 0.0
    rerun: bool = False              # a done result that lost its external data, recomputed to feed a later stage

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Job:
        return cls(key=d["key"], stage=d["stage"], block=d["block"], wf=[int(w) for w in d["wf"]],
                   run_wf=int(d["run_wf"]), deps=dict(d["deps"]), est_s=float(d.get("est_s", 0.0)),
                   measured=bool(d.get("measured")), disk_gb=float(d.get("disk_gb", 0.0)),
                   rerun=bool(d.get("rerun")))


@dataclass
class Plan:
    """The jobs of a run, in order, each key once. ``targets``: the (workflow, stage) pairs asked for;
    ``skipped``: workflows left out, with the reason; ``rules``: 'module:function' building the KeyRules of other
    stages than the v2 ones (tests and tools; None: the v2 stages)."""

    jobs: list[Job] = field(default_factory=list)
    targets: list[tuple[int, str]] = field(default_factory=list)
    skipped: dict[int, str] = field(default_factory=dict)
    rules: str | None = None

    def __len__(self) -> int:
        return len(self.jobs)

    def __bool__(self) -> bool:
        return bool(self.jobs)

    @property
    def keys(self) -> list[str]:
        return [j.key for j in self.jobs]

    @property
    def seconds(self) -> float:
        return sum(j.est_s for j in self.jobs)

    @property
    def measured(self) -> bool:
        """Whether every job's time comes from this machine's history (else the total is ≈)."""
        return all(j.measured for j in self.jobs)

    @property
    def disk_gb(self) -> float:
        return sum(j.disk_gb for j in self.jobs)

    def jobs_of(self, wf: int) -> list[Job]:
        return [j for j in self.jobs if wf in j.wf]

    def to_dict(self) -> dict[str, Any]:
        return {"jobs": [j.to_dict() for j in self.jobs], "targets": [list(t) for t in self.targets],
                "skipped": {str(k): v for k, v in self.skipped.items()}, "rules": self.rules}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Plan:
        return cls([Job.from_dict(j) for j in d.get("jobs") or []],
                   [(int(w), s) for w, s in d.get("targets") or []],
                   {int(k): v for k, v in (d.get("skipped") or {}).items()}, d.get("rules"))


def layer_record(layer: str | Path) -> Path | None:
    """The file recording the stack an earlier learned star layer belongs to (``<layer>.json`` beside it, or
    ``<layer file>.json``); None when there is none."""
    p = Path(layer)
    for f in (p.with_suffix(".json"), p.with_name(p.name + ".json")):
        if f.is_file():
            return f
    return None


class Planner:
    """Plans jobs over one state of a project definition (build a new one after an edit; pass the same InputScan
    to keep the disk parts). ``gpu``: estimates for a machine with a GPU; ``history``: the timing history to use
    (default: the machine's)."""

    def __init__(self, project: Project2, keys: Keys | None = None, store: Store | None = None, gpu: bool = True,
                 history: dict | None = None):
        self.project = project
        self.keys = keys or Keys(project)
        self.store = store if store is not None else (Store(project.root) if project.root is not None else None)
        self.gpu = gpu
        self.history = est.load_history() if history is None else history
        self.jobs: dict[str, Job] = {}
        self.skipped: dict[int, str] = {}
        self._facts: dict[int, est.Facts] = {}

    @property
    def chains(self) -> wfl.Chains:
        return self.keys.rules.chains

    # ------------------------------------------------------------------ state
    def done(self, stage: str, key: str) -> bool:
        return self.store is not None and self.store.done(stage, key)

    def usable(self, stage: str, key: str) -> bool:
        return self.store is not None and self.store.usable_as_input(stage, key)

    def blocked(self, wf: int | Workflow) -> str | None:
        """Why a workflow cannot run now (None when it can): its earlier star layer was made on another stack."""
        w = self.project.workflow(wf)
        v = (w.params.get("starless") or {}).get("learned_stars")
        if not wfl.names_file(v) or "stack" not in self.keys.chain(w):
            return None
        rec = layer_record(self.project.abspath(re.sub(r" \(run [0-9a-fA-F]*\)$", "", str(v).strip())))
        if rec is None:
            return None
        try:
            made_on = json.loads(rec.read_text(encoding="utf-8")).get(LAYER_STACK_KEY)
        except (OSError, ValueError, AttributeError):
            return None
        return LAYER_OTHER_STACK if made_on and made_on != self.keys.key(w, "stack") else None

    def _facts_of(self, w: Workflow) -> est.Facts:
        if id(w) not in self._facts:
            self._facts[id(w)] = est.facts(self.keys, w, self.store)
        return self._facts[id(w)]

    def _sharing(self, w: Workflow, stage: str, key: str) -> list[int]:
        out = []
        for o in self.project.workflows:
            try:
                if stage in self.keys.chain(o) and self.keys.key(o, stage) == key:
                    out.append(o.id)
            except ValueError:                   # another workflow with an invalid setting: it shares nothing
                continue
        if w.id not in out:
            out.append(w.id)
        return out

    # ------------------------------------------------------------------ planning
    def need(self, wf: int | Workflow, stage: str, as_input: bool = False) -> None:
        w = self.project.workflow(wf)
        k = self.keys.key(w, stage)
        if k in self.jobs:
            return
        done = self.done(stage, k)
        if done and (not as_input or self.usable(stage, k)):
            return
        deps = self.keys.deps(w, stage)
        for d in deps:
            self.need(w, d, as_input=True)
        e = est.estimate(stage, self._facts_of(w), self.gpu, self.history)
        self.jobs[k] = Job(k, stage, self.chains.block_of(stage), self._sharing(w, stage, k), w.id,
                           {d: self.keys.key(w, d) for d in deps}, round(e.seconds, 3), e.measured,
                           round(e.disk_gb, 3), rerun=done)

    def plan(self, targets: list[tuple[int | Workflow, str]]) -> Plan:
        """The plan computing the given (workflow, stage) results and what they need. Workflows that cannot run
        now are left out (``Plan.skipped``)."""
        asked: list[tuple[int, str]] = []
        for wf, stage in targets:
            w = self.project.workflow(wf)
            why = self.skipped.get(w.id) or self.blocked(w)
            if why:
                self.skipped[w.id] = why
                continue
            self.need(w, stage)
            asked.append((w.id, stage))
        jobs = order_jobs(self.jobs.values(), self.chains.order, {w for w, _ in asked})
        return Plan(jobs, asked, dict(self.skipped))


def order_jobs(jobs: Any, order: list[str], targets: set[int] | None = None) -> list[Job]:
    """Jobs in dependency order; among those ready, the one whose workflows have the least time left first (cheap
    workflows finish first), then the global stage order. ``targets``: the workflows the plan is for (another
    workflow that shares a job does not count)."""
    left = {j.key: j for j in jobs}

    def mine(j: Job) -> list[int]:
        return [w for w in j.wf if targets is None or w in targets] or [j.run_wf]

    remaining: dict[int, float] = {}
    for j in left.values():
        for w in mine(j):
            remaining[w] = remaining.get(w, 0.0) + j.est_s
    rank = {s: i for i, s in enumerate(order)}
    out: list[Job] = []
    while left:
        ready = [j for j in left.values() if not any(k in left for k in j.deps.values())]
        if not ready:
            raise RuntimeError(f"jobs that wait on each other: {sorted(left)}")
        j = min(ready, key=lambda j: (min(remaining[w] for w in mine(j)), rank.get(j.stage, len(rank)), j.key))
        out.append(j)
        del left[j.key]
        for w in mine(j):
            remaining[w] -= j.est_s
    return out


def plan_stages(project: Project2, targets: list[tuple[int | Workflow, str]], **kw: Any) -> Plan:
    """The plan for the given (workflow, stage) pairs (``Planner`` keywords: keys, store, gpu, history)."""
    return Planner(project, **kw).plan(targets)


def plan_workflows(project: Project2, workflows: list[int | Workflow], **kw: Any) -> Plan:
    """The plan computing every stage of the given workflows (ids, or Workflow objects for a what-if)."""
    p = Planner(project, **kw)
    return p.plan([(w, s) for w in workflows for s in p.keys.chain(w)])


def plan_all(project: Project2, **kw: Any) -> Plan:
    """Run all: every stage of every workflow of the project (the reference stage included when there is one)."""
    return plan_workflows(project, [w.id for w in project.workflows], **kw)


def plan_to_block(project: Project2, wf: int | Workflow, block: str, **kw: Any) -> Plan:
    """Run to here: the stages of a block in the workflow's chain (Selection: register), and what they need."""
    p = Planner(project, **kw)
    return p.plan([(wf, s) for s in p.chains.run_to(block, p.keys.chain(wf))])


# ---------------------------------------------------------------------------------------------- launching

@dataclass
class RunInfo:
    """A run: its id, its files, the byte offset of events.jsonl its events come after, the device it was given
    and the process started (``pid``; the worker writes its own pid in run.lock)."""

    run_id: str
    root: str
    plan_file: str
    log_file: str
    offset: int
    device: str
    pid: int | None = None
    notes: list[str] = field(default_factory=list)


def device_env(device: str | None, parent: str | None = None) -> tuple[str, str]:
    """(CUDA_VISIBLE_DEVICES of the worker, its device inside) for 'cuda:N', 'cuda', 'cpu' or None / 'auto'
    (cuda:0). ``parent``: the launching process's CUDA_VISIBLE_DEVICES, whose numbering 'cuda:N' follows."""
    d = (device or "cuda:0").strip().lower()
    if d == "auto":
        d = "cuda:0"
    if d == "cpu":
        return "-1", "cpu"
    m = re.fullmatch(r"cuda(?::(\d+))?", d)
    if not m:
        raise ValueError(f"unknown device {device!r}: 'cuda:N' or 'cpu'")
    n = int(m.group(1) or 0)
    if parent is None:
        return str(n), "cuda:0"
    seen = [x.strip() for x in parent.split(",") if x.strip()]
    if n < len(seen) and seen[n] != "-1":
        return seen[n], "cuda:0"
    return "-1", "cpu"                          # the launching process sees no such GPU


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def rotate_events(logs: Path, limit: int | None = None) -> Path | None:
    """Rename events.jsonl to events_<date>.jsonl when it is over ``limit`` bytes (default EVENTS_LIMIT; never
    deleted). Done when a run starts, before a follower takes its offset."""
    ev = Path(logs) / EVENTS
    try:
        if ev.stat().st_size <= (EVENTS_LIMIT if limit is None else limit):
            return None
    except FileNotFoundError:
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst, k = ev.with_name(f"events_{stamp}.jsonl"), 2
    while dst.exists():
        dst, k = ev.with_name(f"events_{stamp}_{k}.jsonl"), k + 1
    os.replace(ev, dst)
    return dst


def prepare_run(project: Project2, plan: Plan, device: str | None = None) -> RunInfo:
    """Everything a run needs before its worker starts: no run alive, STOP removed, events rotated, and the frozen
    plan file (run id, device, targets, the whole definition, the jobs). Raises RuntimeError when a run is alive."""
    if project.root is None:
        raise ValueError(f"project {project.name!r} has no folder yet")
    root = Path(os.path.abspath(project.root))
    logs = root / "logs"
    alive = running(root)
    if alive is not None:
        raise RuntimeError(f"a run is already in progress for this project (run {alive.run_id}, pid {alive.pid})")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / STOP).unlink(missing_ok=True)
    rotate_events(logs)
    ev = logs / EVENTS
    offset = ev.stat().st_size if ev.exists() else 0
    dev = device if device not in (None, "", "auto") else (project.devices.get("main") or "cuda:0")
    visible, inside = device_env(dev, os.environ.get("CUDA_VISIBLE_DEVICES"))
    run_id = new_run_id()
    runs = logs / RUNS
    runs.mkdir(parents=True, exist_ok=True)
    plan_file, log_file = runs / f"{run_id}.plan.json", runs / f"{run_id}.log"
    body = {"run_id": run_id, "created": datetime.now().isoformat(timespec="seconds"), "device": dev,
            "visible": visible, "inside": inside, "offset": offset, **plan.to_dict(),
            "definition": project.to_dict()}
    _write_json(plan_file, body)
    return RunInfo(run_id, str(root), str(plan_file), str(log_file), offset, dev)


def launch(project: Project2, plan: Plan, device: str | None = None, python: str | None = None) -> RunInfo:
    """Start a run in a detached worker process (``prepare_run``, then the process). Returns at once; the run is
    followed through ``read_events(root, info.offset, info.run_id)``."""
    info = prepare_run(project, plan, device)
    d = json.loads(Path(info.plan_file).read_text(encoding="utf-8"))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": d["visible"], "PYTHONUNBUFFERED": "1"}
    cmd = [python or sys.executable, "-m", "astro_gpu.core.runner2", info.root, info.plan_file]
    kw: dict[str, Any] = {"cwd": info.root, "env": env, "stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT,
                          "close_fds": True}
    with open(info.log_file, "ab") as log:
        kw["stdout"] = log
        if os.name == "nt":
            # a hidden console of its own, not DETACHED_PROCESS: without a console, every console child of the worker
            # (the multiprocessing pools of calibrate and register) would pop up a window of its own on the screen
            flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB
            try:
                proc = subprocess.Popen(cmd, creationflags=flags, **kw)
            except PermissionError:
                # the launching process is in a job that forbids breaking away: the worker stays in it, so it ends
                # with that job (a closed agent session, a killed terminal)
                proc = subprocess.Popen(cmd, creationflags=flags & ~CREATE_BREAKAWAY_FROM_JOB, **kw)
                info.notes.append("the worker could not leave the launching process's job: it stops with it")
        else:
            proc = subprocess.Popen(cmd, start_new_session=True, **kw)
    info.pid = proc.pid
    return info


def running(root: str | Path) -> RunInfo | None:
    """The run alive in a project (its run.lock names a live process), or None. Its plan file allows a client to
    reattach."""
    root = Path(os.path.abspath(root))
    holder = _lock_holder(root / "logs" / LOCK)
    if not holder or not _alive(holder):
        return None
    run_id = str(holder.get("run_id") or "")
    plan_file = root / "logs" / RUNS / f"{run_id}.plan.json"
    try:
        d = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    return RunInfo(run_id, str(root), str(plan_file), str(plan_file.with_name(f"{run_id}.log")),
                   int(d.get("offset") or 0), str(d.get("device") or ""), int(holder["pid"]))


def cancel(root: str | Path) -> None:
    """Ask the run to stop (the STOP file): the job in progress keeps its checkpoints for the next run."""
    logs = Path(root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / STOP).write_text("stop", encoding="utf-8")


def read_events(root: str | Path, offset: int = 0, run_id: str | None = None) -> tuple[list[dict], int]:
    """Events appended to the project's events.jsonl since byte ``offset`` (of one run when ``run_id`` is given),
    and the new offset."""
    evs, pos = _read_events(Path(root) / "logs" / EVENTS, offset)
    if run_id is not None:
        evs = [e for e in evs if e.get("run") == run_id]
    return evs, pos


def load_plan(plan_file: str | Path) -> tuple[Plan, dict[str, Any]]:
    """(the Plan, the whole plan file) of a run."""
    d = json.loads(Path(plan_file).read_text(encoding="utf-8"))
    return Plan.from_dict(d), d


# ---------------------------------------------------------------------------------------------- the lock

def _lock_holder(lock: Path) -> dict | None:
    try:
        d = json.loads(lock.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and "pid" in d else None
    except (OSError, ValueError):
        return None


def _alive(holder: dict) -> bool:
    """Whether the process named by a lock still runs (a pid reused by a process started after the lock is not)."""
    try:
        import psutil
        pid = int(holder["pid"])
        if not psutil.pid_exists(pid):
            return False
        started = float(holder.get("started") or 0)
        return not started or psutil.Process(pid).create_time() <= started + 1.0
    except Exception:  # noqa: BLE001 - psutil.NoSuchProcess, AccessDenied...: a process we cannot see is gone
        return False


def acquire_lock(logs: Path, run_id: str) -> bool:
    """Create logs/run.lock with O_EXCL ({pid, run_id, started}). A lock whose process is gone is replaced; a live
    one (or one being written this very moment) makes it refuse: returns False."""
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    lock = logs / LOCK
    body = json.dumps({"pid": os.getpid(), "run_id": run_id, "started": time.time()}).encode("utf-8")
    for _ in range(3):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = _lock_holder(lock)
            if holder is None:
                try:
                    if time.time() - lock.stat().st_mtime < 5.0:     # another worker is writing it
                        return False
                except FileNotFoundError:
                    continue
            elif _alive(holder):
                return False
            stale = lock.with_name(f"{LOCK}.stale-{os.getpid()}")
            try:
                os.replace(lock, stale)
                stale.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            except PermissionError:
                return False
            continue
        with os.fdopen(fd, "wb") as f:
            f.write(body)
        return True
    return False


def release_lock(logs: Path, run_id: str) -> None:
    lock = Path(logs) / LOCK
    holder = _lock_holder(lock)
    if holder is not None and holder.get("run_id") == run_id and int(holder.get("pid", -1)) == os.getpid():
        lock.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------- the worker

class RunJournal(Journal):
    """The journal of a run: every event carries the run id; the events of the job's stage add its key, the
    workflows sharing it and its block."""

    def __init__(self, log_dir: Path, run_id: str, echo: bool = True):
        super().__init__(log_dir, echo)
        self.run_id = run_id
        self.job: Job | None = None

    def emit(self, type_: str, **kw: Any) -> None:
        kw.setdefault("run", self.run_id)
        j = self.job
        if j is not None and kw.get("stage") == j.stage:
            kw.setdefault("key", j.key)
            kw.setdefault("wf", j.wf)
            kw.setdefault("block", j.block)
        super().emit(type_, **kw)


def _rules(spec: str | None) -> KeyRules:
    if not spec:
        return key_rules()
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)()


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _write_json(path: Path, obj: Any) -> None:
    """Write a JSON file (a manifest, the plan) atomically; the replace is retried while a reader holds the file
    (``runner.replace_file``)."""
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    try:
        replace_file(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _rename(src: Path, dst: Path, attempts: int = 20) -> None:
    """os.replace with retries: on Windows a folder cannot be renamed while a file in it is open (an antivirus,
    the indexer, a reader), usually for a moment only."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            gc.collect()
            time.sleep(0.1 * (i + 1))


def _empty(folder: Path) -> None:
    for p in folder.iterdir():
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()


class _Worker:
    def __init__(self, root: Path, d: dict[str, Any], rules: KeyRules | None, echo: bool):
        self.root = root
        self.d = d
        self.run_id = d["run_id"]
        self.project = Project2.from_dict(d["definition"], root)          # frozen: edits during a run change nothing
        self.rules = rules or _rules(d.get("rules"))
        self.keys = Keys(self.project, self.rules, InputScan())
        self.store = Store(root)
        self.journal = RunJournal(root / "logs", self.run_id, echo)
        self.jobs = Plan.from_dict(d).jobs
        same_gpu = os.environ.get("CUDA_VISIBLE_DEVICES") == d.get("visible")
        self.device = d.get("inside") if same_gpu else d.get("device")   # in-process: the device as named
        self.device = self.device or "cuda:0"
        self._gpu: bool | None = None

    @property
    def gpu(self) -> bool:
        if self._gpu is None:
            self._gpu = self.device != "cpu" and _cuda()
        return self._gpu

    def stop_requested(self) -> bool:
        return (self.root / "logs" / STOP).exists()

    # ------------------------------------------------------------------ the run
    def run(self) -> str:
        j = self.journal
        j.emit("run_start", stages=[x.stage for x in self.jobs], jobs=len(self.jobs), device=self.d.get("device"),
               targets=self.d.get("targets"), pid=os.getpid())
        j.emit("plan", jobs=[{k: v for k, v in x.to_dict().items() if k not in ("run_wf", "deps")} for x in self.jobs],
               seconds=round(sum(x.est_s for x in self.jobs), 1), skipped=self.d.get("skipped") or {})
        out: dict[str, list[str]] = {"done": [], "failed": [], "skipped": [], "interrupted": [], "not_started": []}
        bad: set[str] = set()
        status = "done"
        try:
            for i, job in enumerate(self.jobs):
                if self.stop_requested():
                    out["not_started"] = [x.key for x in self.jobs[i:]]
                    status = "interrupted"
                    break
                up = [s for s, k in job.deps.items() if k in bad]
                if up:
                    out["skipped"].append(job.key)
                    bad.add(job.key)
                    self.journal.job = job
                    j.emit("stage_end", stage=job.stage, status="skipped", reason="upstream failed", upstream=up)
                    continue
                res = self.run_job(job)
                out[res].append(job.key)
                if res == "failed":
                    bad.add(job.key)
                elif res == "interrupted":
                    out["not_started"] = [x.key for x in self.jobs[i + 1:]]
                    status = "interrupted"
                    break
        finally:
            self.journal.job = None
            if status == "done" and (out["failed"] or out["skipped"]):
                status = "failed"
            j.emit("run_end", status=status, **out)
        return status

    def run_job(self, job: Job) -> str:
        """Run one job: 'done', 'failed' or 'interrupted'."""
        self.journal.job = job
        self.store.forget()
        if self.store.done(job.stage, job.key) and self.store.usable_as_input(job.stage, job.key):
            self.journal.emit("stage_end", stage=job.stage, status="done", reused=True, duration=0.0)
            return "done"
        try:
            w = self.project.workflow(job.run_wf)
            sp = StoreProject(self.keys, w, job.stage, devices={"main": self.device})
            if sp.key != job.key:
                raise StageError(f"the inputs of {job.stage} changed since the run was planned (key {sp.key}, "
                                 f"planned {job.key}): plan it again")
            st = self.rules.chains.stages[job.stage]
            pipe = Pipeline(sp, [self.rules.chains.stages[s] for s in sp.chain], self.journal)
            if job.stage == "calibrate" and self.store.done(job.stage, job.key):
                self._give_masters(pipe, st, job)
            res = pipe._run_stage(st, run_id=job.key)
            if res != "done":
                return res
            self._commit(job, sp, w)
        except Exception as e:  # noqa: BLE001 - a job that cannot run or be stored fails alone
            msg = str(e) if isinstance(e, StageError) else f"{type(e).__name__}: {e}"
            self._mark_failed(job, msg, traceback.format_exc())
            self.journal.emit("log", stage=job.stage, level="error", msg=msg)
            self.journal.emit("stage_end", stage=job.stage, status="failed", error=msg)
            return "failed"
        return "done"

    def _mark_failed(self, job: Job, msg: str, tb: str) -> None:
        mf = self.store.tmp_dir(job.stage, job.key) / "manifest.json"
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        m.update(status="failed", error=msg, traceback=tb)
        try:
            _write_json(mf, m)
        except OSError:
            pass

    def _commit(self, job: Job, sp: StoreProject, w: Workflow) -> None:
        """``<key>.tmp`` → ``<key>`` with the key fields in its manifest; a done folder it replaces goes to
        ``_trash`` only now, and comes back if the rename fails."""
        tmp, final = sp.stage_dir(job.stage), self.store.dir(job.stage, job.key)
        mf = tmp / "manifest.json"
        m = json.loads(mf.read_text(encoding="utf-8"))
        m.update(key=job.key, key_parts=self.keys.key_parts(w, job.stage), adopted_from=None, run=self.run_id)
        _write_json(mf, m)
        gc.collect()                                  # no array of the job keeps a file of its folder open
        old = None
        if final.exists():
            self.store.trash.mkdir(parents=True, exist_ok=True)
            old = self.store.trash / f"{job.stage}-{job.key}-{self.run_id}"
            _rename(final, old)
        try:
            _rename(tmp, final)
        except OSError:
            if old is not None:
                _rename(old, final)
            raise
        self.store.forget()
        try:
            f = est.facts(self.keys, w, self.store)
            est.record(job.stage, float(m.get("duration") or 0.0), est.units(job.stage, f), self.gpu)
        except Exception:  # noqa: BLE001 - the history is a hint
            pass

    def _give_masters(self, pipe: Pipeline, st: Any, job: Job) -> None:
        """A done calibrate whose frame cache is lost: its masters go into the job's folder, with their settings
        and metrics and the checkpoint tag the job will have, so calibrate rebuilds the cache only (the same files,
        so the same cache). An interrupted job of this key already there is resumed as it is."""
        from ..stages.calibrate import MASTER_FILES, MASTERS, result_params
        old = self.store.dir(job.stage, job.key)
        tmp = pipe.project.stage_dir(job.stage)
        p, _, _ = pipe.effective_params(st)                     # (creates the job's folder)
        inputs = {d: read_manifest(pipe.project, d).get("run_id") for d in st.deps}
        tag = {"params_hash": params_hash(p), "inputs": inputs, "version": st.version}   # as _run_stage builds it
        ck = tmp / "_ckpt" / "meta.json"
        try:
            if json.loads(ck.read_text()) == tag:
                return
        except (OSError, ValueError):
            pass
        om = self.store.manifest(job.stage, job.key)
        given = dict(om.get("params") or {})
        given.pop("workers", None)
        if given != result_params(p) or not all((old / n).is_file() for n in MASTER_FILES):
            return                                              # a full calibration
        _empty(tmp)
        for n in MASTER_FILES:
            shutil.copy2(old / n, tmp / n)
        _write_json(tmp / MASTERS, {"params": result_params(p), "metrics": om.get("metrics") or {},
                                    "from": str(old)})
        ck.parent.mkdir(parents=True, exist_ok=True)
        ck.write_text(json.dumps(tag))
        self.journal.emit("log", stage=job.stage, level="info",
                          msg="the calibrated frames were lost: they are rebuilt with the masters of the result")


def run_plan(root: str | Path, plan_file: str | Path, rules: KeyRules | None = None, echo: bool = True) -> str:
    """Run a plan file in this process (the worker's body): 'done', 'failed', 'interrupted', or 'refused' when
    another run holds the project's lock. ``rules``: the stages' KeyRules (default: the plan's, else v2's)."""
    root = Path(os.path.abspath(root))
    d = json.loads(Path(plan_file).read_text(encoding="utf-8"))
    logs = root / "logs"
    if not acquire_lock(logs, d["run_id"]):
        print(f"run {d['run_id']} refused: another run holds {logs / LOCK}", flush=True)
        return "refused"
    try:
        try:
            worker = _Worker(root, d, rules, echo)
        except Exception as e:  # noqa: BLE001 - a plan that cannot even start still ends its run for the follower
            msg = f"{type(e).__name__}: {e}"
            print(traceback.format_exc(), flush=True)
            j = RunJournal(logs, d["run_id"], echo)
            j.emit("log", level="error", msg=f"the run cannot start: {msg}")
            j.emit("run_end", status="failed", error=msg, done=[], failed=[], skipped=[], interrupted=[],
                   not_started=[x["key"] for x in d.get("jobs") or []])
            return "failed"
        return worker.run()
    finally:
        release_lock(logs, d["run_id"])


EXIT = {"done": 0, "failed": 1, "interrupted": 2, "refused": 3}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="astro_gpu.core.runner2", description="run the frozen plan of a v2 project")
    ap.add_argument("project")
    ap.add_argument("plan")
    a = ap.parse_args(argv)
    os.chdir(a.project)                               # relative paths of stage code resolve in the project
    return EXIT[run_plan(a.project, a.plan)]


if __name__ == "__main__":
    sys.exit(main())
