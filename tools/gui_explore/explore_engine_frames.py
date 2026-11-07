"""Engine exploration, frames: tiny synthetic Bayer FITS frames (RGGB, 200 x 300) through the frame blocks, on the CPU.

    python tools/gui_explore/explore_engine_frames.py --all
    python tools/gui_explore/explore_engine_frames.py <scenario>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_engine_common import Findings, bayer_frames, drive, fresh, main_one, run_and_check  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.core import project2, runner2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.project2 import Session  # noqa: E402
from astro_gpu.core.store import Keys  # noqa: E402

FAST = {"calibrate": {"workers": 2}, "register": {"workers": 2}}


def project(name: str, nights: int = 1, n: int = 8, calib: bool = False, mode: str = "per_session"):
    d = fresh(name)
    p = project2.create(d / "P", "P", Target(name="M 31"))
    p.stack_mode = mode
    for i in range(1, nights + 1):
        lights = bayer_frames(d / f"n{i}" / "lights", n=n, seed=10 + i)
        kw = {}
        if calib:
            kw = {"darks": [str(bayer_frames(d / f"n{i}" / "darks", n=3, kind="dark", seed=20 + i))],
                  "flats": [str(bayer_frames(d / f"n{i}" / "flats", n=3, kind="flat", seed=30 + i))],
                  "bias": [str(bayer_frames(d / f"n{i}" / "bias", n=3, kind="bias", seed=40 + i))]}
        p.sessions.append(Session(i, f"night {i}", lights=[str(lights)], **kw))
    return d, p


def frames_workflow(p, blocks, name="W", sessions=None):
    w = p.add_workflow(name, wfl.TEMPLATES["Classic stack"], params=FAST)
    w.sessions = sessions
    p.set_blocks(w.id, blocks)
    return w


N = wfl.node


def stack_only(f: Findings):
    _d, p = project("stack_only")
    frames_workflow(p, [N("stacking")])
    return run_and_check(p, f, "stack only") | {"chain": Keys(p).chain(1)}


def calib_only(f: Findings):
    _d, p = project("calib_only", calib=True)
    w = frames_workflow(p, [N("calibration")])
    return run_and_check(p, f, "calibration only") | {"chain": Keys(p).chain(w.id)}


def calib_stack(f: Findings):
    _d, p = project("calib_stack", calib=True)
    frames_workflow(p, [N("calibration"), N("stacking")])
    return run_and_check(p, f, "calibration + stacking") | {"chain": Keys(p).chain(1)}


def full_frames(f: Findings):
    _d, p = project("full", calib=True)
    frames_workflow(p, [N("calibration"), N("registration"), N("selection"), N("stacking")])
    return run_and_check(p, f, "calibration + registration + selection + stacking")


def registration_off(f: Findings):
    _d, p = project("reg_off")
    frames_workflow(p, [N("calibration"), N("registration", off=True), N("stacking")])
    k = Keys(p)
    r = run_and_check(p, f, "registration switched off")
    r["register_params"] = k.key_params(1, "register")
    return r


def registration_only(f: Findings):
    _d, p = project("reg_only")
    w = frames_workflow(p, [N("registration")])
    return run_and_check(p, f, "registration only") | {"chain": Keys(p).chain(w.id)}


def selection_only(f: Findings):
    _d, p = project("sel_only")
    w = frames_workflow(p, [N("selection")])
    return run_and_check(p, f, "selection only") | {"chain": Keys(p).chain(w.id)}


def gradient_after_stack(f: Findings):
    _d, p = project("grad_after")
    frames_workflow(p, [N("calibration"), N("registration"), N("stacking"), N("gradient"),
                        N("gradient", "rbf", name="gradient~2")])
    return run_and_check(p, f, "gradients after stacking")


def two_sessions(f: Findings):
    _d, p = project("two_sessions", nights=2, n=6)
    w = frames_workflow(p, [N("calibration"), N("registration"), N("stacking")])
    return run_and_check(p, f, "two sessions per session") | {"chain": Keys(p).chain(w.id)}


def two_sessions_together(f: Findings):
    _d, p = project("two_together", nights=2, n=6, mode="together")
    w = frames_workflow(p, [N("calibration"), N("registration"), N("stacking")])
    return run_and_check(p, f, "two sessions together") | {"chain": Keys(p).chain(w.id)}


def parallel_flows(f: Findings):
    """Two flows, each its own session (a Session dropped on the new-flow zone), and a third over both."""
    _d, p = project("parallel", nights=2, n=6)
    a = frames_workflow(p, [N("calibration"), N("registration"), N("stacking")], name="A", sessions=[1])
    b = frames_workflow(p, [N("calibration"), N("registration"), N("stacking")], name="B", sessions=[2])
    c = frames_workflow(p, [N("calibration"), N("registration"), N("stacking")], name="Both", sessions=None)
    k = Keys(p)
    out = {"chains": {w.name: k.chain(w.id) for w in (a, b, c)}}
    # Both calibrates each night as A and B do: those results are shared
    if k.key(c.id, "calibrate@1") != k.key(a.id, "calibrate"):
        f.warn("parallel flows: the calibration of night 1 in 'Both' is not A's (computed twice)",
               json.dumps(out["chains"]))
    plan = runner2.plan_all(p)
    out["jobs"] = [(j.stage, j.wf) for j in plan.jobs]
    return out | run_and_check(p, f, "parallel flows", plan=plan)


def branch_after_stack(f: Findings):
    """A branch at Stacking (two gradients methods): the frames' stages computed once for both."""
    _d, p = project("branch_frames")
    a = frames_workflow(p, [N("calibration"), N("registration"), N("stacking"), N("gradient")], name="A")
    b = frames_workflow(p, [N("calibration"), N("registration"), N("stacking"), N("gradient", "rbf")], name="B")
    c = frames_workflow(p, [N("calibration"), N("registration"), N("stacking")], name="C (stack to Post)")
    plan = runner2.plan_all(p)
    stages = [j.stage for j in plan.jobs]
    for s in ("inventory", "calibrate", "register", "stack"):
        if stages.count(s) != 1:
            f.fail(f"branch at Stacking: {s} planned {stages.count(s)} times", str(stages))
    return {"jobs": stages} | run_and_check(p, f, "branch at stacking", plan=plan)


def denoise_cpu(f: Findings):
    """Denoise (pretrained) after the stack, on the CPU: does it fit in 2 minutes on a 200 x 300 stack?"""
    _d, p = project("denoise")
    frames_workflow(p, [N("calibration"), N("registration"), N("stacking"), N("gradient"), N("denoise")])
    return run_and_check(p, f, "denoise on cpu")


def denoise_twice_cpu(f: Findings):
    _d, p = project("denoise2")
    w = frames_workflow(p, [N("calibration"), N("registration"), N("stacking"), N("denoise"), N("gradient"),
                            N("denoise", "pretrained", name="denoise~2")])
    out = {"denoise_params": Keys(p).key_params(w.id, "denoise")}
    return out | run_and_check(p, f, "denoise twice on cpu")


def stars_cpu(f: Findings):
    _d, p = project("stars")
    frames_workflow(p, [N("calibration"), N("registration"), N("stacking"), N("gradient"), N("denoise"), N("stars")])
    return run_and_check(p, f, "denoise + stars on cpu")


def _big(name):
    d = fresh(name)
    p = project2.create(d / "P", "P", Target(name="M 31"))
    p.stack_mode = "per_session"
    p.sessions.append(Session(1, "night 1", lights=[str(bayer_frames(d / "n1", n=4, h=860, w=860, nstars=400))]))
    return d, p


def denoise_big(f: Findings):
    """Denoise on a stack of 840 x 840 (frames 560 x 560, drizzle 1.5): past the preview's 800 px."""
    _d, p = _big("denoise_big")
    frames_workflow(p, [N("calibration"), N("registration"), N("stacking"), N("gradient"), N("denoise")])
    return run_and_check(p, f, "denoise 840 px")


def denoise_twice_big(f: Findings):
    _d, p = _big("denoise2_big")
    frames_workflow(p, [N("calibration"), N("registration"), N("stacking"), N("denoise"), N("gradient"),
                        N("denoise", "pretrained", name="denoise~2")])
    return run_and_check(p, f, "denoise twice 840 px")


def stars_big(f: Findings):
    _d, p = _big("stars_big")
    frames_workflow(p, [N("calibration"), N("registration"), N("stacking"), N("gradient"), N("denoise"), N("stars")])
    return run_and_check(p, f, "denoise + stars 840 px")


def no_frames(f: Findings):
    """A Session placed empty, its flow with blocks: what Run says (plan, then a run)."""
    d = fresh("noframes")
    p = project2.create(d / "P", "P", Target(name="M 31"))
    p.sessions.append(Session(1, "empty"))
    frames_workflow(p, [N("calibration"), N("registration"), N("stacking")])
    project2.save(p)
    plan = runner2.plan_all(p)
    out = {"jobs": [j.stage for j in plan.jobs], "skipped": plan.skipped}
    hc = api.health_checks(p)
    out["health"] = [(c.level, c.key, c.text) for c in hc]
    if plan.jobs and not plan.skipped:
        info = runner2.prepare_run(p, plan, "cpu")
        st = runner2.run_plan(p.root, info.plan_file, echo=False)
        evs, _ = runner2.read_events(p.root, info.offset, info.run_id)
        out["status"] = st
        out["errors"] = [e.get("error") for e in evs if e.get("type") == "stage_end" and e.get("status") == "failed"]
        f.info("no frames: the plan is not refused, the run fails", json.dumps(out, ensure_ascii=False))
    return out


def missing_folder(f: Findings):
    """A session whose folder is gone (drive unplugged): plan and run messages."""
    d = fresh("missing")
    p = project2.create(d / "P", "P", Target(name="M 31"))
    p.sessions.append(Session(1, "gone", lights=[str(d / "nowhere")]))
    frames_workflow(p, [N("stacking")])
    project2.save(p)
    plan = runner2.plan_all(p)
    out = {"jobs": [j.stage for j in plan.jobs], "skipped": plan.skipped}
    info = runner2.prepare_run(p, plan, "cpu")
    out["status"] = runner2.run_plan(p.root, info.plan_file, echo=False)
    evs, _ = runner2.read_events(p.root, info.offset, info.run_id)
    out["errors"] = [e.get("error") for e in evs if e.get("type") == "stage_end" and e.get("status") == "failed"]
    return out


def keys_frames(f: Findings):
    """Frame workflows: switching Registration off changes register's key and what follows; Calibration absent and
    switched off are the same key; keys stable across save/load; an unrelated flow's edit changes nothing."""
    _d, p = project("keys_frames", nights=2, n=4)
    a = frames_workflow(p, [N("calibration"), N("registration"), N("stacking")], name="A", sessions=[1])
    b = frames_workflow(p, [N("calibration", off=True), N("registration"), N("stacking")], name="B", sessions=[1])
    c = frames_workflow(p, [N("registration"), N("stacking")], name="C", sessions=[1])
    e = frames_workflow(p, [N("calibration"), N("registration"), N("stacking")], name="E", sessions=[2])
    k = Keys(p)
    out = {"A": k.keys(a.id), "B": k.keys(b.id), "C": k.keys(c.id)}
    if k.key(b.id, "stack") != k.key(c.id, "stack"):
        f.warn("Calibration switched off and Calibration absent give other stack keys", json.dumps(out))
    if k.key(a.id, "stack") == k.key(b.id, "stack"):
        f.fail("Calibration switched off gives the same key as on", json.dumps(out))
    k0 = {w.id: k.keys(w.id) for w in p.workflows}
    project2.save(p)
    q = project2.load(p.root)
    k1 = {w.id: Keys(q).keys(w.id) for w in q.workflows}
    if k0 != k1:
        f.fail("frame keys change across save/load", json.dumps([k0, k1]))
    q.set_blocks(e.id, [N("calibration"), N("stacking")])
    k2 = {w.id: Keys(q).keys(w.id) for w in q.workflows}
    for w in (a, b, c):
        if k2[w.id] != k0[w.id]:
            f.fail("editing flow E changed another flow's keys", w.name)
    return out


SCENARIOS = ["stack_only", "calib_only", "calib_stack", "full_frames", "registration_off", "registration_only",
             "selection_only", "gradient_after_stack", "two_sessions", "two_sessions_together", "parallel_flows",
             "branch_after_stack", "no_frames", "missing_folder", "keys_frames", "denoise_cpu", "denoise_twice_cpu",
             "stars_cpu", "denoise_big", "denoise_twice_big", "stars_big"]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        main_one(globals())
    else:
        drive(__file__, sys.argv[2:] if len(sys.argv) > 2 else SCENARIOS)
