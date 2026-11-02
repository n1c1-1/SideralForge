"""Pipeline runner: decides what is up to date, runs what is not, and makes every run resumable.

A stage is *up to date* when its manifest says ``done`` and the stage version, the effective parameters and the
run ids of its dependencies are unchanged. Anything else (never run, failed, interrupted, parameters changed,
an upstream stage re-ran) makes it - and everything downstream - *stale*.

Long stages keep checkpoints in ``work/<stage>/_ckpt`` tagged with the parameter hash; an interrupted stage
restarted with the same parameters continues where it stopped.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .determinism import deterministic
from .files import write_json
from .journal import Journal
from .params import params_hash, resolve
from .project import Project
from .stage import Context, Interrupted, Stage, StageError


@dataclass
class StageStatus:
    name: str
    title: str
    state: str          # done | ready | stale | pending | failed | interrupted | running
    reason: str = ""
    duration: float | None = None
    finished: str = ""


def _manifest_path(project: Project, stage: str) -> Path:
    return project.stage_dir(stage) / "manifest.json"


def read_manifest(project: Project, stage: str) -> dict:
    p = _manifest_path(project, stage)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_manifest(project: Project, stage: str, m: dict) -> None:
    p = _manifest_path(project, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(p, m)


class Pipeline:
    def __init__(self, project: Project, stages: list[Stage], journal: Journal | None = None):
        self.project = project
        self.stages = stages
        self.by_name = {s.name: s for s in stages}
        self.journal = journal or Journal(project.logs)

    # ------------------------------------------------------------------ parameters
    def effective_params(self, stage: Stage):
        """(params, origin, auto) for a stage. Dependencies must be done for auto values to be available."""
        ctx = Context(self.project, stage, self.journal, self._device())
        try:
            auto = stage.auto(ctx)
        except (StageError, FileNotFoundError, KeyError):
            auto = {}
        p, origin = resolve(stage.Params, auto, self.project.overrides(stage.name))
        return p, origin, auto

    def _device(self) -> str:
        return self.project.config.devices.get("main", "cuda:0")

    # ------------------------------------------------------------------ status
    def status(self) -> list[StageStatus]:
        out: dict[str, StageStatus] = {}
        for st in self.stages:
            m = read_manifest(self.project, st.name)
            deps_bad = [d for d in st.deps if out[d].state != "done"]
            if m.get("status") == "running" and _run_alive(self.project):
                out[st.name] = StageStatus(st.name, st.title, "running")
                continue
            if not m:
                state, reason = ("pending" if deps_bad else "ready"), "never run"
            elif m.get("status") in ("failed", "interrupted", "running"):
                state = "interrupted" if m.get("status") != "failed" else "failed"
                reason = m.get("error", "")[:300] or "interrupted"
            elif deps_bad:
                state, reason = "stale", f"upstream not up to date: {', '.join(deps_bad)}"
            else:
                state, reason = "done", ""
                if m.get("version") != st.version:
                    state, reason = "stale", f"algorithm updated ({m.get('version')} -> {st.version})"
                else:
                    stored_fp = m.get("inputs", {}).get("_project_inputs")
                    for d in st.deps:
                        if m.get("inputs", {}).get(d) != read_manifest(self.project, d).get("run_id"):
                            state, reason = "stale", f"'{d}' was recomputed"
                            break
                    else:
                        p, _, _ = self.effective_params(st)
                        if not st.deps and stored_fp is not None and stored_fp != _inputs_fingerprint(self.project):
                            state, reason = "stale", "the project's input files changed"
                        if params_hash(p) != m.get("params_hash") and not _same_params(st, p, m.get("params", {})):
                            state, reason = "stale", "parameters changed"
            out[st.name] = StageStatus(st.name, st.title, state, reason, m.get("duration"), m.get("finished", ""))
        return [out[s.name] for s in self.stages]

    # ------------------------------------------------------------------ run
    def plan(self, until: str | None = None, force: tuple[str, ...] = ()) -> list[str]:
        names = [s.name for s in self.stages]
        last = names.index(until) if until else len(names) - 1
        needed = _ancestors(self.by_name, names[last]) | {names[last]} if until else set(names)
        todo, dirty = [], set(force)
        for st in self.status():
            if st.name not in needed:
                continue
            s = self.by_name[st.name]
            if st.state != "done" or st.name in dirty or any(d in dirty for d in s.deps):
                todo.append(st.name)
                dirty.add(st.name)
        return todo

    def run(self, until: str | None = None, force: tuple[str, ...] = ()) -> str:
        lock = self.project.logs / "run.lock"
        if _run_alive(self.project):
            raise RuntimeError("a run is already in progress for this project")
        self.project.logs.mkdir(parents=True, exist_ok=True)
        (self.project.logs / "STOP").unlink(missing_ok=True)
        lock.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
        todo = self.plan(until, force)
        self.journal.emit("run_start", stages=todo, variant=self.project.config.active_variant)
        status = "done"
        order = [s.name for s in self.stages]
        ran: set[str] = set()
        try:
            while todo:
                name = todo[0]
                res = self._run_stage(self.by_name[name])
                ran.add(name)
                if res != "done":
                    status = res
                    break
                # a stage's data-driven parameters may read an optional output of an earlier stage (the star removal
                # takes the layer of the learned star removal network when there is one): what is left is planned
                # again after each stage. A stage never runs twice in one run.
                later = self.plan(until, tuple(f for f in force if f not in ran))
                todo = sorted((set(todo[1:]) | set(later)) - ran, key=order.index)
        finally:
            lock.unlink(missing_ok=True)
            self.journal.emit("run_end", status=status)
        return status

    def _run_stage(self, st: Stage, run_id: str | None = None) -> str:
        """Run one stage in its folder. ``run_id``: the id its manifest gets when done (a fresh random one by
        default; a v2 job passes its key, so that results reading it stay the same when it is recomputed)."""
        p, origin, auto = self.effective_params(st)
        ph = params_hash(p)
        ctx = Context(self.project, st, self.journal, self._device())
        # checkpoints are only valid for the same parameters and inputs
        inputs = {d: read_manifest(self.project, d).get("run_id") for d in st.deps}
        if not st.deps:                      # the first stage reads the project's files: their lists are its input
            inputs["_project_inputs"] = _inputs_fingerprint(self.project)
        ck_meta = ctx.dir / "_ckpt" / "meta.json"
        tag = {"params_hash": ph, "inputs": inputs, "version": st.version}
        if not (ck_meta.exists() and json.loads(ck_meta.read_text()) == tag):
            ctx.clear_ckpt()
            _clear_outputs(ctx.dir)
        ctx.ckpt  # noqa: B018  (creates the folder)
        ck_meta.write_text(json.dumps(tag))
        m = {"stage": st.name, "title": st.title, "version": st.version, "status": "running",
             "params": p.model_dump(mode="json"), "origin": origin,
             "auto_reasons": {k: a.reason for k, a in auto.items()}, "params_hash": ph, "inputs": inputs,
             "started": datetime.now().isoformat(timespec="seconds")}
        _write_manifest(self.project, st.name, m)
        self.journal.emit("stage_start", stage=st.name, title=st.title)
        t0 = time.time()
        try:
            with deterministic():                # same inputs, same outputs (core/determinism.py)
                st.run(ctx, p)
        except Interrupted:
            m.update(status="interrupted", error="stopped by user", duration=time.time() - t0)
            _write_manifest(self.project, st.name, m)
            self.journal.emit("stage_end", stage=st.name, status="interrupted", duration=time.time() - t0)
            return "interrupted"
        except Exception as e:  # noqa: BLE001 - report every failure to the user
            tb = traceback.format_exc()
            msg = str(e) if isinstance(e, StageError) else f"{type(e).__name__}: {e}"
            m.update(status="failed", error=msg, traceback=tb, duration=time.time() - t0)
            _write_manifest(self.project, st.name, m)
            self.journal.emit("log", stage=st.name, level="error", msg=msg)
            self.journal.emit("stage_end", stage=st.name, status="failed", duration=time.time() - t0)
            print(tb, flush=True)
            return "failed"
        dur = time.time() - t0
        m.update(status="done", run_id=run_id or uuid.uuid4().hex, duration=dur, metrics=ctx.metrics,
                 finished=datetime.now().isoformat(timespec="seconds"))
        _write_manifest(self.project, st.name, m)
        ctx.clear_ckpt()
        self.journal.emit("stage_end", stage=st.name, status="done", duration=dur)
        return "done"


def _clear_outputs(d: Path) -> None:
    """Remove what an earlier run left in the stage folder before a fresh start. An earlier version may have written
    files the current one does not: a later stage that looks for an optional file must never find a stale one."""
    for p in d.iterdir():
        if p.name in ("_ckpt", "manifest.json"):
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def _inputs_fingerprint(project: Project) -> str:
    """Hash of the project's input file lists (lights, darks, flats...): adding flats to an existing project must
    make the inventory - and so everything after it - run again (it did not: the flats were silently ignored)."""
    import hashlib
    from dataclasses import asdict
    return hashlib.sha1(json.dumps(asdict(project.config.inputs), sort_keys=True).encode()).hexdigest()[:16]


def _same_params(stage: Stage, p, stored: dict) -> bool:
    """Equal parameters, tolerating parameters added since the stage ran when they are at their default value
    (a new option that keeps the previous behaviour must not invalidate past results) and removed parameters."""
    cur = p.model_dump(mode="json")
    for k, v in cur.items():
        if k in stored:
            if stored[k] != v:
                return False
        else:
            default = stage.Params.model_fields[k].default
            if json.loads(json.dumps(default, default=str)) != v:
                return False
    return True


def request_stop(project: Project) -> None:
    project.logs.mkdir(parents=True, exist_ok=True)
    (project.logs / "STOP").write_text("stop")


def _run_alive(project: Project) -> bool:
    lock = project.logs / "run.lock"
    if not lock.exists():
        return False
    try:
        pid = json.loads(lock.read_text())["pid"]
        import psutil
        return psutil.pid_exists(pid) and pid != os.getpid()
    except Exception:  # noqa: BLE001
        return False


def _ancestors(by_name: dict[str, Stage], name: str) -> set[str]:
    out, todo = set(), list(by_name[name].deps)
    while todo:
        d = todo.pop()
        if d not in out:
            out.add(d)
            todo += list(by_name[d].deps)
    return out
