"""Where the project queues (tools/projects/queue.py) are, in one call (read only: nothing is written).

    python tools/projects/status.py [--json]

Per queue: alive or not, the project and stage in progress, the time elapsed and the estimated time left. Per
project and workflow: the jobs (stages of its chain) done, running, failed (failed, or skipped because an upstream
job failed) and pending (never run, outdated or interrupted), the time the queue spent on the project, and the
estimated time left (the planner's estimates of the jobs not done yet, less the time already spent on the job in
progress). A project's time left counts each shared job once, so it is less than the sum of its workflows'.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p8  # noqa: E402

p8.use_home()

from astro_gpu import api  # noqa: E402

CATS = {"done": "done", "running": "running", "failed": "failed", "skipped": "failed", "pending": "pending",
        "stale": "pending", "interrupted": "pending"}


def _hm(s: float | None) -> str:
    if s is None:
        return "-"
    s = max(0.0, s)
    return f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


def _t(iso: str | None) -> float | None:
    try:
        return datetime.fromisoformat(iso).timestamp() if iso else None
    except ValueError:
        return None


def _queues() -> list[dict]:
    out = []
    for f in sorted(p8.QUEUES.glob("*.json")):
        if f.name.endswith(".state.json"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        st = {}
        sf = f.with_name(f"{d['name']}.state.json")
        if sf.is_file():
            try:
                st = json.loads(sf.read_text(encoding="utf-8"))
            except ValueError:
                st = {}
        lock = {}
        lf = f.with_name(f"{d['name']}.lock")
        if lf.is_file():
            try:
                lock = json.loads(lf.read_text(encoding="utf-8"))
            except ValueError:
                lock = {}
        out.append({"def": d, "state": st, "lock": lock})
    return out


def _alive(lock: dict) -> bool:
    import psutil
    try:
        pid = int(lock.get("pid") or 0)
        return bool(pid) and psutil.pid_exists(pid) and \
            psutil.Process(pid).create_time() <= float(lock.get("started") or 0) + 1.0
    except (psutil.Error, ValueError, TypeError):
        return False


def project_status(name: str, device: str, qstate: dict) -> dict:
    now = time.time()
    p = api.open_project(p8.WS / name)
    states = api.states(p, update=False)
    plan = api.plan_all(p, device)
    run = api.running(p)
    cur = qstate.get("current") or {}
    running_key = cur.get("key") if cur.get("project") == name else None
    spent = (now - _t(cur.get("stage_started"))) if running_key and cur.get("stage_started") else 0.0
    ps = (qstate.get("projects") or {}).get(name) or {}
    elapsed = 0.0
    for r in ps.get("runs") or []:
        t0 = _t(r.get("started"))
        t1 = _t(r.get("ended")) or (now if r.get("status") == "running" else None)
        if t0 and t1:
            elapsed += t1 - t0
    rows = []
    for w in p.workflows:
        n = {"done": 0, "running": 0, "failed": 0, "pending": 0}
        errors = []
        for s in api.chain(p, w.id):
            x = states[(w.id, s)]
            n[CATS.get(x.state, "pending")] += 1
            if x.state == "failed":
                errors.append(f"{s}: {x.reason}"[:160])
        left = 0.0
        for j in plan.jobs:
            if w.id in j.wf:
                left += max(0.0, j.est_s - spent) if j.key == running_key else j.est_s
        rows.append({"id": w.id, "workflow": w.name, **n, "left_s": round(left), "errors": errors})
    left_p = sum(max(0.0, j.est_s - spent) if j.key == running_key else j.est_s for j in plan.jobs)
    return {"project": name, "workflows": rows, "elapsed_s": round(elapsed), "left_s": round(left_p),
            "jobs_left": len(plan), "measured": plan.measured, "live_run": run.run_id if run else None,
            "error": ps.get("error")}


def collect() -> list[dict]:
    out = []
    now = time.time()
    for q in _queues():
        d, st = q["def"], q["state"]
        alive = _alive(q["lock"])
        projs = [project_status(n, d["device"], st) for n in d["projects"]]
        starts = [_t(r.get("started")) for ps in (st.get("projects") or {}).values() for r in ps.get("runs") or []]
        starts = [s for s in starts if s]
        cur = st.get("current") or {}
        spent = (now - _t(cur["stage_started"])) if cur.get("stage_started") else None
        out.append({"queue": d["name"], "device": d["device"], "alive": alive, "status": st.get("status"),
                    "pid": (q["lock"] or {}).get("pid"), "seen": st.get("seen"),
                    "elapsed_s": round(((now if alive else _t(st.get("ended")) or now)) - min(starts)) if starts else 0,
                    "left_s": sum(x["left_s"] for x in projs), "current": cur, "stage_elapsed_s": spent,
                    "projects": projs})
    return out


def show(qs: list[dict]) -> None:
    print(f"P8 status at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    for q in qs:
        state = "running" if q["alive"] else f"NOT RUNNING (last state: {q['status']}, seen {q['seen']})"
        print(f"\nqueue {q['queue']} on {q['device']}: {state}; {_hm(q['elapsed_s'])} elapsed, "
              f"about {_hm(q['left_s'])} left")
        c = q["current"] or {}
        if c.get("project"):
            what = f"{c.get('stage')} {100 * (c.get('frac') or 0):.0f} % for {_hm(q['stage_elapsed_s'])}" \
                if c.get("stage") else "between jobs"
            print(f"  now: {c['project']} (pass {c.get('pass')}, run {c.get('run_id')}): {what}"
                  + (f" ({c['msg']})" if c.get("msg") else ""))
        print(f"  {'project':12s} {'workflow':28s} {'done':>4s} {'run':>4s} {'fail':>4s} {'pend':>4s} "
              f"{'elapsed':>8s} {'left':>8s}")
        for pr in q["projects"]:
            first = True
            for r in pr["workflows"]:
                print(f"  {pr['project'] if first else '':12s} {r['workflow'][:28]:28s} {r['done']:4d} "
                      f"{r['running']:4d} {r['failed']:4d} {r['pending']:4d} "
                      f"{_hm(pr['elapsed_s']) if first else '':>8s} {_hm(r['left_s']):>8s}")
                first = False
            print(f"  {'':12s} {'(project, shared jobs once)':28s} {'':19s} {'':8s} {_hm(pr['left_s']):>8s}"
                  + ("" if pr["measured"] else " ≈"))
            for r in pr["workflows"]:
                for e in r["errors"]:
                    print(f"  {'':12s}   failed in {r['workflow']}: {e}")
            if pr.get("error"):
                print(f"  {'':12s}   queue error: {pr['error']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    qs = collect()
    if a.json:
        print(json.dumps(qs, indent=1, default=str))
    else:
        show(qs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
