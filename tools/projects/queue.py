"""P8 step 4: a queue of projects run one after another on one GPU, detached (it survives the session).

    python tools/projects/queue.py start NAME --device cuda:N PROJECT [PROJECT ...] [--all-at-once]
    python tools/projects/queue.py stop NAME        the job in progress stops at its next checkpoint, then the queue
    python tools/projects/queue.py resume NAME      starts the queue again: done results are kept, an interrupted job
                                                    resumes from its checkpoints, failed jobs get one more try
    python tools/projects/queue.py serve NAME       the queue itself (started detached by start and resume)

Rules (DESIGN §3, review point 9):
- each run is one ``api.start_run(project, plan, "cuda:N")``: a detached worker with ``CUDA_VISIBLE_DEVICES=N``
  (cuda:0 inside), so a stage that names cuda:0 stays on the queue's GPU. The queue itself has no
  CUDA_VISIBLE_DEVICES, so 'cuda:N' is the physical GPU N (nvidia-smi's numbering, checked with torch);
- cheap workflows first: pass 1 runs "Classic stack" and "Classic + denoise" of every project of the queue, pass 2
  everything else (``--all-at-once``: one pass). Within a run the planner puts the cheapest workflow's jobs first;
- a failed job skips only its downstream jobs: the worker does it within a run, and the queue drops the failed keys
  and what depends on them from the project's later plans; the queue then moves on to the next project;
- one project at a time; a project that already has a live run (another queue, the client) is followed, not
  started twice.

Files, all under F:/astro_projets/_p8/queues/: NAME.json (the definition), NAME.state.json (the state status.py
reads), NAME.log (what happened), NAME.out (the queue process's own output), NAME.lock, NAME.STOP.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p8  # noqa: E402  (p8 takes this folder off sys.path again: this file would hide the stdlib's queue)

p8.use_home()

PASSES_DEFAULT: list[list[str] | None] = [["Classic stack", "Classic + denoise"], None]
POLL = 15.0                     # seconds between two reads of a run's events
PROGRESS_EVERY = 600.0          # a progress line per stage at most every 10 minutes in the log
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def files(name: str) -> dict[str, Path]:
    q = p8.QUEUES
    return {k: q / f"{name}{ext}" for k, ext in (("def", ".json"), ("state", ".state.json"), ("log", ".log"),
                                                 ("out", ".out"), ("lock", ".lock"), ("stop", ".STOP"))}


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _alive(pid: int | None, started: float | None = None) -> bool:
    if not pid:
        return False
    import psutil
    try:
        if not psutil.pid_exists(int(pid)):
            return False
        return not started or psutil.Process(int(pid)).create_time() <= float(started) + 1.0
    except (psutil.Error, ValueError):
        return False


def queue_alive(name: str) -> dict | None:
    """The live queue process of this name ({pid, started}), or None."""
    d = _read_json(files(name)["lock"])
    return d if d and _alive(d.get("pid"), d.get("started")) else None


# ---------------------------------------------------------------------------------------------- start / stop

def spawn(name: str) -> int:
    """Start ``serve NAME`` detached (it leaves this session's job), without CUDA_VISIBLE_DEVICES."""
    f = files(name)
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env.update(ASTRO_GPU_HOME=str(p8.HOME), PYTHONUNBUFFERED="1")
    cmd = [str(p8.PYTHON), str(Path(__file__).resolve()), "serve", name]
    kw = dict(cwd=str(p8.QUEUES), env=env, stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT, close_fds=True)
    with open(f["out"], "ab") as out:
        flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB
        try:
            proc = subprocess.Popen(cmd, creationflags=flags, stdout=out, **kw)
        except PermissionError:
            proc = subprocess.Popen(cmd, creationflags=flags & ~CREATE_BREAKAWAY_FROM_JOB, stdout=out, **kw)
            print("warning: the queue could not leave this session's job: it stops with the session")
    return proc.pid


def cmd_start(a: argparse.Namespace) -> int:
    f = files(a.name)
    p8.QUEUES.mkdir(parents=True, exist_ok=True)
    if queue_alive(a.name):
        print(f"queue {a.name} is already running ({queue_alive(a.name)})")
        return 1
    for n in a.projects:
        if not (p8.WS / n / "project.astroproj").is_file():
            raise SystemExit(f"not a v2 project: {p8.WS / n}")
    d = {"name": a.name, "device": a.device, "projects": a.projects,
         "passes": [None] if a.all_at_once else PASSES_DEFAULT, "created": _now()}
    _write_json(f["def"], d)
    f["stop"].unlink(missing_ok=True)
    pid = spawn(a.name)
    print(f"queue {a.name} started on {a.device}: pid {pid}; log {f['log']}")
    return 0


def cmd_resume(a: argparse.Namespace) -> int:
    f = files(a.name)
    if not f["def"].is_file():
        raise SystemExit(f"no queue {a.name}: {f['def']}")
    if queue_alive(a.name):
        print(f"queue {a.name} is already running")
        return 1
    f["stop"].unlink(missing_ok=True)
    pid = spawn(a.name)
    print(f"queue {a.name} resumed: pid {pid}; log {f['log']}")
    return 0


def cmd_stop(a: argparse.Namespace) -> int:
    from astro_gpu import api
    f = files(a.name)
    f["stop"].write_text(_now(), encoding="utf-8")
    cur = _read_json(f["state"]).get("current") or {}
    if cur.get("project"):
        p = api.open_project(p8.WS / cur["project"])
        if api.running(p):
            api.cancel(p)
            print(f"stop asked to the run of {cur['project']}: the job in progress stops at its next checkpoint")
    print(f"queue {a.name} stops after that (resume: queue.py resume {a.name})")
    return 0


# ---------------------------------------------------------------------------------------------- the queue

class Queue:
    def __init__(self, name: str):
        from astro_gpu import api
        self.api = api
        self.name = name
        self.f = files(name)
        self.d = _read_json(self.f["def"])
        self.device = self.d["device"]
        self.state = _read_json(self.f["state"]) or {"name": name, "projects": {}}
        self.state.update(name=name, device=self.device, pid=os.getpid(), status="running", serve_started=_now(),
                          current=None, order=self.d["projects"], passes=self.d["passes"])
        self.state.pop("ended", None)
        self.failed: dict[str, set[str]] = {}          # this session's failed keys per project
        self.cancelled = False

    # ------------------------------------------------------------------ log and state
    def log(self, msg: str) -> None:
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        with open(self.f["log"], "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

    def save(self) -> None:
        self.state["seen"] = _now()
        _write_json(self.f["state"], self.state)

    def proj_state(self, proj: str) -> dict:
        return self.state["projects"].setdefault(proj, {"runs": []})

    def stop_asked(self) -> bool:
        return self.f["stop"].exists()

    # ------------------------------------------------------------------ the loop
    def serve(self) -> str:
        self.log(f"queue {self.name} on {self.device}: {', '.join(self.d['projects'])}; passes {self.d['passes']}")
        self.save()
        status = "done"
        try:
            for i, names in enumerate(self.d["passes"], 1):
                for proj in self.d["projects"]:
                    if self.stop_asked():
                        status = "stopped"
                        break
                    try:
                        self.run_project(proj, names, i)
                    except Exception as e:  # noqa: BLE001 - one project's trouble never stops the queue
                        self.log(f"{proj}: ERROR {type(e).__name__}: {e}")
                        self.log(traceback.format_exc().rstrip())
                        self.proj_state(proj)["error"] = f"{type(e).__name__}: {e}"
                        self.wait_idle(proj)
                    self.state["current"] = None
                    self.save()
                if status == "stopped":
                    break
        finally:
            self.state.update(status=status, ended=_now(), current=None)
            self.save()
            self.log(f"queue {self.name} {status}")
        return status

    def wait_idle(self, proj: str) -> None:
        """Never two workers on the GPU: after an error, wait until the project has no live run."""
        errors = 0
        while True:
            try:
                if self.api.running(self.api.open_project(p8.WS / proj)) is None:
                    return
            except Exception as e:  # noqa: BLE001
                errors += 1
                self.log(f"{proj}: cannot tell whether a run is alive ({e})")
                if errors >= 3:
                    return
            if self.stop_asked() and not self.cancelled:
                try:
                    self.api.cancel(self.api.open_project(p8.WS / proj))
                    self.cancelled = True
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(60)

    def prune(self, proj: str, plan) -> list[str]:
        """Drop from the plan the keys that failed earlier in this queue session and every job that reads them."""
        bad = set(self.failed.get(proj) or ())
        if not bad:
            return []
        keep, dropped = [], []
        for j in plan.jobs:
            if j.key in bad or any(k in bad for k in j.deps.values()):
                bad.add(j.key)
                dropped.append(f"{j.stage} {j.key[:8]}")
            else:
                keep.append(j)
        plan.jobs = keep
        return dropped

    def run_project(self, proj: str, names: list[str] | None, pass_no: int) -> None:
        api = self.api
        root = p8.WS / proj
        p = api.open_project(root)
        live = api.running(p)
        if live is not None:
            self.log(f"{proj}: a run is already alive ({live.run_id}, pid {live.pid}): following it")
            self.follow(proj, p, live, pass_no, reattached=True)
            p = api.open_project(root)
        ids = [w.id for w in p.workflows if names is None or w.name in names]
        if not ids:
            return
        plan = api.plan_workflows(p, ids, self.device)
        dropped = self.prune(proj, plan)
        if dropped:
            self.log(f"{proj}: left out, upstream failed earlier: {', '.join(dropped)}")
        for wid, why in plan.skipped.items():
            self.log(f"{proj}: workflow {wid} left out: {why}")
        what = "all workflows" if names is None else " / ".join(names)
        if not plan:
            self.log(f"{proj}: pass {pass_no} ({what}): nothing to run")
            return
        info = api.start_run(p, plan, self.device)
        self.log(f"{proj}: pass {pass_no} ({what}): run {info.run_id} started, {len(plan)} jobs, about "
                 f"{plan.seconds / 3600:.2f} h; worker pid {info.pid}; log {info.log_file}")
        for n in info.notes:
            self.log(f"{proj}: {n}")
        self.follow(proj, p, info, pass_no, plan=plan)

    def follow(self, proj: str, p, info, pass_no: int, plan=None, reattached: bool = False) -> None:
        api = self.api
        ps = self.proj_state(proj)
        run = {"run_id": info.run_id, "pass": pass_no, "started": _now(), "t0": time.time(), "pid": info.pid,
               "jobs": len(plan) if plan is not None else None,
               "est_s": round(plan.seconds, 1) if plan is not None else None, "reattached": reattached,
               "status": "running"}
        ps["runs"].append(run)
        self.state["current"] = {"project": proj, "run_id": info.run_id, "pass": pass_no, "stage": None,
                                 "key": None, "stage_started": None, "frac": None, "msg": None}
        self.save()
        offset, last_progress, cur_key, gone, errors = info.offset, {}, None, 0, 0
        end = None
        while end is None:
            try:
                evs, offset = api.read_events(p, offset, info.run_id)
            except Exception as e:  # noqa: BLE001 - a busy file: read again at the next poll
                errors += 1
                if errors in (1, 10, 100):
                    self.log(f"{proj}: events not read ({type(e).__name__}: {e}); trying again")
                evs = []
            for e in evs:
                t = e.get("type")
                st = e.get("stage", "")
                cur = self.state["current"]
                if t == "stage_start":
                    cur_key = e.get("key")
                    cur.update(stage=st, key=cur_key, stage_started=_now(), frac=0.0, msg=None)
                    self.log(f"{proj}:   {st} started ({(cur_key or '')[:8]}, wf {e.get('wf')})")
                elif t == "stage_end":
                    s = e.get("status")
                    dur = e.get("duration")
                    extra = f" in {dur / 60:.1f} min" if isinstance(dur, (int, float)) and dur else ""
                    why = f": {e.get('error') or e.get('reason')}" if (e.get("error") or e.get("reason")) else ""
                    if not e.get("reused"):
                        self.log(f"{proj}:   {st} {s}{extra}{why}")
                    cur.update(stage=None, key=None, frac=None, msg=None)
                elif t == "progress":
                    cur.update(frac=e.get("frac"), msg=e.get("msg"))
                    now = time.time()
                    if now - last_progress.get(st, 0) >= PROGRESS_EVERY:
                        last_progress[st] = now
                        self.log(f"{proj}:     {st} {100 * (e.get('frac') or 0):5.1f} %  {e.get('msg') or ''}")
                elif t == "log" and e.get("level") == "error":
                    self.log(f"{proj}:     [{st}] error: {e.get('msg')}")
                elif t == "run_end":
                    end = e
            if end is not None:
                break
            if self.stop_asked() and not self.cancelled:
                api.cancel(p)
                self.cancelled = True
                self.log(f"{proj}: stop asked: the job in progress stops at its next checkpoint")
            alive = _alive(info.pid) or api.running(p) is not None
            gone = 0 if alive else gone + 1
            if gone >= 3:                              # gone, and its last events were read
                end = {"status": "lost", "failed": [cur_key] if cur_key else []}
                self.log(f"{proj}: the worker of run {info.run_id} is gone without ending its run")
                break
            self.save()
            time.sleep(POLL)
        status = str(end.get("status") or "done")
        bad = set(end.get("failed") or []) | set(end.get("skipped") or [])
        if bad:
            self.failed.setdefault(proj, set()).update(bad)
        n = {k: len(end.get(k) or []) for k in ("done", "failed", "skipped", "interrupted", "not_started")}
        run.update(status=status, ended=_now(), seconds=round(time.time() - run["t0"], 1), counts=n)
        self.log(f"{proj}: run {info.run_id} {status} in {(time.time() - run['t0']) / 3600:.2f} h: "
                 + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in n.items() if v))
        try:
            api.refresh(p)
            api.states(p)                              # records the done keys (the "previous result")
        except Exception as e:  # noqa: BLE001
            self.log(f"{proj}: states not recorded: {e}")
        self.save()


def cmd_serve(a: argparse.Namespace) -> int:
    f = files(a.name)
    p8.QUEUES.mkdir(parents=True, exist_ok=True)
    import psutil
    me = {"pid": os.getpid(), "started": psutil.Process().create_time(), "since": _now()}
    try:
        fd = os.open(f["lock"], os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if queue_alive(a.name):
            print(f"queue {a.name} is already running: {_read_json(f['lock'])}", flush=True)
            return 3
        f["lock"].unlink(missing_ok=True)              # its process is gone
        fd = os.open(f["lock"], os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(me, fh)
    try:
        status = Queue(a.name).serve()
    finally:
        if _read_json(f["lock"]).get("pid") == me["pid"]:
            f["lock"].unlink(missing_ok=True)
    return 0 if status == "done" else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("name")
    s.add_argument("--device", required=True, help="cuda:N, the physical GPU (nvidia-smi's index)")
    s.add_argument("projects", nargs="+", help="project folders under F:/astro_projets, in order")
    s.add_argument("--all-at-once", action="store_true", help="one pass: every workflow of a project in one run")
    s.set_defaults(fn=cmd_start)
    for n, fn in (("stop", cmd_stop), ("resume", cmd_resume), ("serve", cmd_serve)):
        x = sub.add_parser(n)
        x.add_argument("name")
        x.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
