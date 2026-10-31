"""Command line of v2 projects (project.astroproj), on the engine API (astro_gpu/api.py).

    python -m astro_gpu.cli2 new PROJECT --lights DIR [--darks DIR ...] [--template NAME ...]
    python -m astro_gpu.cli2 import OLD_PROJECT NEW_PROJECT [--map FILE.json] [--dry-run]
    python -m astro_gpu.cli2 add-workflow PROJECT NAME [--template NAME | --option KEY=VALUE ... | --copy-of ID]
    python -m astro_gpu.cli2 set PROJECT [--wf ID ...] [--option KEY=VALUE ...] [STAGE.NAME=VALUE ...]
    python -m astro_gpu.cli2 status PROJECT
    python -m astro_gpu.cli2 plan PROJECT [--wf ID ...] [--to BLOCK]
    python -m astro_gpu.cli2 run PROJECT [--wf ID ...] [--to BLOCK] [--device cuda:1] [--detach]
    python -m astro_gpu.cli2 events PROJECT [--run RUN_ID] [--offset N] [--follow]
    python -m astro_gpu.cli2 stop PROJECT
    python -m astro_gpu.cli2 export PROJECT --wf ID [--base NAME] [--slider KEY=POS ...] [--format tiff16 ...]

``run`` starts the detached worker and, without ``--detach``, follows its events until the run ends (Ctrl-C leaves
the worker running; ``stop`` asks it to stop). Exit codes: 0 done, 1 failed, 2 interrupted, 3 lost or timed out.
Nothing is written into a v1 project folder. ``--rules module:function`` (before the command) replaces the stage
graph with toy stages, for tests and tools.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

EXIT = {"done": 0, "failed": 1, "interrupted": 2, "lost": 3, "timeout": 3}


def _api() -> Any:
    from . import api
    return api


def _open(a: argparse.Namespace) -> Any:
    api = _api()
    p = api.open_project(a.project)
    if a.rules:
        api.use_rules(p, a.rules)
    return p


def _value(v: str) -> Any:
    """A command-line value: JSON when it is JSON ('3', 'true', '"x"'), 'auto' and 'none' as None, else text."""
    if v.lower() in ("auto", "none", "null"):
        return None
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _pairs(items: list[str] | None) -> dict[str, Any]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = _value(v)
    return out


def _hm(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def _wfs(p: Any, a: argparse.Namespace) -> list[int]:
    ids = list(getattr(a, "wf", None) or [])
    return ids or ([p.current] if p.current is not None else [])


# ---------------------------------------------------------------------------------------------- commands

def cmd_new(a: argparse.Namespace) -> int:
    from .core.project import Target
    api = _api()
    t = Target(name=a.target or "", ra=a.ra, dec=a.dec, focal_mm=a.focal)
    p = api.create_project(a.project, a.name or Path(a.project).name, t)
    if a.rules:
        api.use_rules(p, a.rules)
    if any([a.lights, a.darks, a.flats, a.bias, a.flat_darks]):
        api.add_session(p, a.lights or [], a.darks or [], a.flats or [], a.bias or [], a.flat_darks or [],
                        a.session_name or "")
    for name in a.template or []:
        api.add_workflow(p, name, name)
    api.save(p)
    print(f"project created: {p.root} ({len(p.sessions)} session(s), {len(p.workflows)} workflow(s))")
    return 0


def cmd_import(a: argparse.Namespace) -> int:
    api = _api()
    path_map = json.loads(Path(a.map).read_text(encoding="utf-8")) if a.map else None
    report = api.import_v1(a.old, a.new, path_map, dry_run=a.dry_run, rules=a.rules,
                           progress=lambda msg: print(f"  {msg}", flush=True))
    print(report.text())
    if a.dry_run and not report.enough_space:
        print("not enough free space for the import", file=sys.stderr)
        return 1
    return 0


def cmd_add_workflow(a: argparse.Namespace) -> int:
    api = _api()
    p = _open(a)
    if a.end:
        w = api.add_exit(p, a.copy_of if a.copy_of is not None else _wfs(p, a)[0], a.end, a.name)
    else:
        opts: dict[str, Any] | None = None
        if a.template or a.option:
            opts = {**(api.TEMPLATES[a.template] if a.template else {}), **_pairs(a.option)}
        w = api.add_workflow(p, a.name, opts, None, copy_of=a.copy_of)
    api.save(p)
    print(f"workflow {w.id} added: {w.name}")
    return 0


def cmd_set(a: argparse.Namespace) -> int:
    api = _api()
    p = _open(a)
    ids = _wfs(p, a)
    for key, value in _pairs(a.option).items():
        for note in api.set_option(p, ids, key, value):
            print(f"  {note}")
    for item in a.assign or []:
        key, eq, val = item.partition("=")
        stage, dot, name = key.partition(".")
        if not (eq and dot and name):
            raise SystemExit(f"expected STAGE.NAME=VALUE, got {item!r}")
        api.set_param(p, ids, stage, name, _value(val))
    api.save(p)
    print("ok")
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    api = _api()
    p = _open(a)
    st = api.states(p, update=not a.read_only)
    for w in p.workflows:
        print(f"workflow {w.id}: {w.name}" + (f"  (variant of {w.variant_of})" if w.variant_of else "")
              + ("  *" if w.id == p.current else ""))
        for s in api.chain(p, w.id):
            x = st[(w.id, s)]
            print(f"  {s:20s} {x.state:12s} {x.key[:8]:8s} {x.reason}")
    plan = api.plan_all(p)
    print(f"to run: {len(plan)} job(s), about {_hm(plan.seconds)}" if plan else "everything is up to date")
    run = api.running(p)
    if run:
        print(f"run {run.run_id} is in progress (pid {run.pid})")
    return 0


def _plan(api: Any, p: Any, a: argparse.Namespace) -> Any:
    if a.to:
        return api.plan_to_block(p, _wfs(p, a)[0], a.to, a.device)
    if a.wf:
        return api.plan_workflows(p, a.wf, a.device)
    return api.plan_all(p, a.device)


def cmd_plan(a: argparse.Namespace) -> int:
    api = _api()
    p = _open(a)
    plan = _plan(api, p, a)
    for i, j in enumerate(plan.jobs, 1):
        print(f"{i:3d}. {j.stage:20s} wf {','.join(map(str, j.wf)):8s} {'' if j.measured else '≈'}{_hm(j.est_s):>8s}  "
              f"{j.disk_gb:6.2f} GB  {j.key}")
    for wid, why in plan.skipped.items():
        print(f"skipped workflow {wid}: {why}")
    print(f"{len(plan)} job(s), {'' if plan.measured else '≈'}{_hm(plan.seconds)}, {plan.disk_gb:.1f} GB")
    return 0


def _line(e: dict, last: dict[str, float]) -> str:
    t, st = e.get("type"), e.get("stage", "")
    if t == "run_start":
        return f"run {e.get('run', '')} started: {e.get('jobs', 0)} job(s)"
    if t == "stage_start":
        return f"  {st} started"
    if t == "stage_end":
        d = e.get("duration")
        extra = f" in {_hm(d)}" if d else (f" ({e['reason']})" if e.get("reason") else "")
        return f"  {st} {e.get('status')}{extra}" + (f": {e['error']}" if e.get("error") else "")
    if t == "progress":
        now = time.time()
        if now - last.get(st, 0) < 5 and e.get("frac", 0) < 1:
            return ""
        last[st] = now
        return f"    {st} {100 * e.get('frac', 0):5.1f} %  {e.get('msg', '')}"
    if t == "log" and e.get("level") in ("error", "warning"):
        return f"    [{st}] {e.get('level')}: {e.get('msg')}"
    if t == "run_end":
        n = {k: len(e.get(k) or []) for k in ("done", "failed", "skipped", "interrupted", "not_started")}
        return f"run {e.get('status')}: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in n.items() if v)
    return ""


def _alive(pid: int | None) -> bool:
    if not pid:
        return True
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001
        return True


def follow(p: Any, offset: int, run_id: str | None, pid: int | None = None, poll: float = 0.5,
           timeout: float | None = None, echo=print) -> str:
    """Print the events of a run until its ``run_end``; returns the run's status ('lost' when the worker is gone
    without ending it, 'timeout' after ``timeout`` seconds)."""
    api = _api()
    t0, last = time.time(), {}
    gone = 0
    while True:
        evs, offset = api.read_events(p, offset, run_id)
        for e in evs:
            line = _line(e, last)
            if line:
                echo(line)
            if e.get("type") == "run_end":
                return str(e.get("status") or "done")
        if not _alive(pid):
            gone += 1
            if gone > 2:                       # the worker is gone and its last events were read
                return "lost"
        if timeout is not None and time.time() - t0 > timeout:
            return "timeout"
        time.sleep(poll)


def cmd_run(a: argparse.Namespace) -> int:
    api = _api()
    p = _open(a)
    plan = _plan(api, p, a)
    for wid, why in plan.skipped.items():
        print(f"skipped workflow {wid}: {why}")
    if not plan:
        print("nothing to run: everything is up to date")
        return 0
    info = api.start_run(p, plan, a.device)
    print(f"run {info.run_id} started: {len(plan)} job(s), about {_hm(plan.seconds)}; pid {info.pid}; "
          f"offset {info.offset}; log {info.log_file}")
    for note in info.notes:
        print(f"  {note}")
    if a.detach:
        return 0
    try:
        status = follow(p, info.offset, info.run_id, info.pid, timeout=a.timeout)
    except KeyboardInterrupt:
        print("\nleft running in the background: `stop` asks it to stop")
        return 0
    api.refresh(p)
    api.states(p)                                    # records the keys of the results for the "previous result"
    return EXIT.get(status, 1)


def cmd_events(a: argparse.Namespace) -> int:
    api = _api()
    p = _open(a)
    if not a.follow:
        evs, off = api.read_events(p, a.offset or 0, a.run)
        for e in evs:
            line = _line(e, {})
            if line:
                print(line)
        print(f"offset {off}")
        return 0
    off = a.offset
    if off is None:
        run = api.running(p)
        off = run.offset if run else 0
    status = follow(p, off, a.run, timeout=a.timeout)
    return EXIT.get(status, 1)


def cmd_stop(a: argparse.Namespace) -> int:
    api = _api()
    p = _open(a)
    api.cancel(p)
    print("stop requested: the job in progress stops at its next checkpoint")
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    api = _api()
    p = _open(a)
    opts: dict[str, Any] = {}
    if a.format:
        opts["formats"] = a.format
    if a.size is not None:
        opts["size"] = a.size
    if a.name:
        opts["name"] = a.name
    files = api.export_workflow(p, a.wf, a.base, {k: float(v) for k, v in _pairs(a.slider).items()}, opts,
                                lambda f, m: print(f"  {100 * f:5.1f} %  {m}", flush=True), device=a.render_device)
    for f in files:
        print(f)
    return 0


# ---------------------------------------------------------------------------------------------- parser

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="astro-gpu-v2", description="processing projects of format 2")
    ap.add_argument("--rules", help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn, help_: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_)
        s.set_defaults(fn=fn)
        return s

    n = add("new", cmd_new, "create a project")
    n.add_argument("project")
    n.add_argument("--name")
    n.add_argument("--target")
    n.add_argument("--ra", type=float)
    n.add_argument("--dec", type=float)
    n.add_argument("--focal", type=float, help="effective focal length in mm")
    n.add_argument("--session-name")
    for k in ("lights", "darks", "flats", "bias", "flat-darks"):
        n.add_argument(f"--{k}", nargs="+", dest=k.replace("-", "_"))
    n.add_argument("--template", action="append", help="add a workflow from a template (repeat)")

    i = add("import", cmd_import, "import a v1 project and its results (copy only)")
    i.add_argument("old")
    i.add_argument("new")
    i.add_argument("--map", help="JSON file: where the raw frames are now {old name: new path}")
    i.add_argument("--dry-run", action="store_true")

    w = add("add-workflow", cmd_add_workflow, "add a workflow")
    w.add_argument("project")
    w.add_argument("name")
    w.add_argument("--template")
    w.add_argument("--option", action="append", help="KEY=VALUE (denoise, stars, gradient)")
    w.add_argument("--copy-of", type=int)
    w.add_argument("--end", choices=["stacking"], help="an early exit after this block (a copy of --copy-of)")

    s = add("set", cmd_set, "set options and settings")
    s.add_argument("project")
    s.add_argument("assign", nargs="*", help="STAGE.NAME=VALUE ('auto' goes back to the suggestion)")
    s.add_argument("--wf", type=int, action="append", help="a workflow id, repeatable (default: the current one)")
    s.add_argument("--option", action="append", help="KEY=VALUE (denoise, stars, gradient)")

    st = add("status", cmd_status, "the state of every stage of every workflow")
    st.add_argument("project")
    st.add_argument("--read-only", action="store_true", help="do not record the results in state.json")

    for name, fn, h in (("plan", cmd_plan, "what a run would do"), ("run", cmd_run, "run the plan")):
        r = add(name, fn, h)
        r.add_argument("project")
        r.add_argument("--wf", type=int, action="append", help="a workflow id, repeatable")
        r.add_argument("--to", help="run to this block (of the first --wf, else the current workflow)")
        r.add_argument("--device", help="cuda:N or cpu (default: the project's)")
        if name == "run":
            r.add_argument("--detach", action="store_true", help="start the worker and return")
            r.add_argument("--timeout", type=float, help="stop following after this many seconds")

    ev = add("events", cmd_events, "the events of a run")
    ev.add_argument("project")
    ev.add_argument("--run")
    ev.add_argument("--offset", type=int)
    ev.add_argument("--follow", action="store_true")
    ev.add_argument("--timeout", type=float)

    sp = add("stop", cmd_stop, "ask the running job to stop at its next checkpoint")
    sp.add_argument("project")

    ex = add("export", cmd_export, "export a workflow's layer headless")
    ex.add_argument("project")
    ex.add_argument("--wf", type=int, required=True)
    ex.add_argument("--base", help="a named rendering of the project, or a preset")
    ex.add_argument("--slider", action="append", help="KEY=POSITION (0-4, 2 = the base)")
    ex.add_argument("--format", nargs="+", choices=["tiff16", "tiff32f", "png16", "jpeg", "fits32f"])
    ex.add_argument("--size", type=int, choices=[0, 2048, 4096])
    ex.add_argument("--name")
    ex.add_argument("--render-device", default="cpu")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    try:
        return int(a.fn(a))
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
