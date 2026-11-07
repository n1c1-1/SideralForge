"""Engine exploration, fusions: a dual-band fusion of one image workflow computed on the CPU; fusions of other shapes
planned, or run to read the message the user gets.

    python tools/gui_explore/explore_engine_fusion.py --all
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_engine_common import (Findings, drive, fresh, linear_fits, main_one, plan_problems,  # noqa: E402
                                   run_and_check)

from astro_gpu.core import project2, runner2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.store import Keys  # noqa: E402

N = wfl.node


def _p(name):
    d = fresh(name)
    return d, project2.create(d / "P", "P", Target(name="M 31"))


def _img(p, path, blocks, name):
    w = p.add_workflow(name, wfl.TEMPLATES["Classic stack"])
    w.source = {"path": str(path), "stretched": False}
    p.set_blocks(w.id, blocks)
    return w


def _fusion(p, inputs, palette="HOO", name="Fusion", stars=None, blocks=None):
    w = p.add_workflow(name, wfl.TEMPLATES["Classic stack"])
    f = {"palette": palette, "inputs": inputs}
    if stars:
        f["stars"] = stars
    w.source = {"fusion": f}
    p.set_blocks(w.id, blocks if blocks is not None else [])
    return w


def dualband_run(f: Findings):
    d, p = _p("fus_dual")
    img = linear_fits(d)
    a = _img(p, img, [N("gradient")], "Dual")
    fu = _fusion(p, [{"pid": a.id, "role": "Ha", "channel": "R"}, {"pid": a.id, "role": "OIII", "channel": "GB"}])
    k = Keys(p)
    out = {"chain": k.chain(fu.id), "jobs": [(j.stage, j.wf) for j in runner2.plan_all(p).jobs]}
    out |= run_and_check(p, f, "dual-band fusion")
    # the input's Gradient changed: the fusion is Outdated (its key follows), the input's image is not
    q = project2.load(p.root)
    before = Keys(q).keys(fu.id)
    q.set_blocks(a.id, [N("gradient", "rbf")])
    after = Keys(q).keys(fu.id)
    out["fusion_key_follows_input"] = before != after
    if before == after:
        f.fail("a fusion's key does not follow its input's change")
    # an unrelated workflow added: nothing moves
    _img(q, img, [N("gradient", "automatic")], "Other")
    if Keys(q).keys(fu.id) != after:
        f.fail("an unrelated workflow changed a fusion's keys")
    return out


def dualband_then_gradient(f: Findings):
    """A fusion with a block after it (Output only is allowed? a Gradient?): the rules, and a run of what is allowed."""
    d, p = _p("fus_more")
    img = linear_fits(d)
    a = _img(p, img, [N("gradient")], "Dual")
    out = {"fusion_places": {b: wfl.places([], "fusion", b) for b in wfl.PALETTE if b != "output"},
           "reasons": {b: str(wfl.palette_problem([], "fusion", b) or "") for b in wfl.PALETTE}}
    _fusion(p, [{"pid": a.id, "role": "Ha", "channel": "R"}, {"pid": a.id, "role": "OIII", "channel": "GB"}])
    return out | run_and_check(p, f, "fusion + output")


def two_images_no_astrometry(f: Findings):
    """Two image workflows without Astrometry combined: the rules let it be made? the run says what?"""
    d, p = _p("fus_noastro")
    a = _img(p, linear_fits(d, "ha.fits", seed=1), [N("gradient")], "Ha")
    b = _img(p, linear_fits(d, "oiii.fits", seed=2), [N("gradient")], "OIII")
    fu = _fusion(p, [{"pid": a.id, "role": "Ha", "channel": "L"}, {"pid": b.id, "role": "OIII", "channel": "L"}])
    project2.save(p)
    plan = runner2.plan_all(p)
    out = {"skipped": plan.skipped, "jobs": [j.stage for j in plan.jobs]}
    r = run_and_check(p, f, "fusion without astrometry", expect="failed")
    out |= r
    f.info("fusion of two images without astrometry: allowed, the run fails", json.dumps(r["errors"]))
    return out


def input_without_file(f: Findings):
    """An Image block placed without its file yet, a channel of a fusion: plan (skipped with its reason?) and run."""
    d, p = _p("fus_nofile")
    img = linear_fits(d)
    a = _img(p, img, [N("gradient")], "Ha")
    b = p.add_workflow("OIII", wfl.TEMPLATES["Classic stack"])
    b.source = {"path": "", "stretched": False}
    p.set_blocks(b.id, [])
    fu = _fusion(p, [{"pid": a.id, "role": "Ha", "channel": "L"}, {"pid": b.id, "role": "OIII", "channel": "L"}])
    project2.save(p)
    plan = runner2.plan_all(p)
    out = {"skipped": plan.skipped, "jobs": [(j.stage, j.wf) for j in plan.jobs]}
    if fu.id not in plan.skipped and any(b.id in j.wf for j in plan.jobs):
        f.warn("a fusion whose channel has no image file yet is planned (its channel's image job too) instead of "
               "being left out with the reason", json.dumps(out))
    if plan.jobs:
        info = runner2.prepare_run(p, plan, "cpu")
        out["status"] = runner2.run_plan(p.root, info.plan_file, echo=False)
        evs, _ = runner2.read_events(p.root, info.offset, info.run_id)
        out["errors"] = [(e.get("stage"), e.get("error")) for e in evs
                         if e.get("type") == "stage_end" and e.get("status") == "failed"]
    return out


def fusion_with_stars_plan(f: Findings):
    """Option A (stars): the inputs need Split stars; from images that cannot have it: plan only."""
    d, p = _p("fus_stars")
    img = linear_fits(d)
    a = _img(p, img, [N("gradient")], "Dual")
    try:
        fu = _fusion(p, [{"pid": a.id, "role": "Ha", "channel": "R"}, {"pid": a.id, "role": "OIII", "channel": "GB"}],
                     stars={"mode": "palette"}, blocks=[N("stars")])
    except Exception as e:  # noqa: BLE001
        return {"set": f"{type(e).__name__}: {e}"}
    project2.save(p)
    k = Keys(p)
    out = {"chain": k.chain(fu.id), "inputs": k.fusion_inputs(fu.id)}
    plan = runner2.plan_all(p)
    out["jobs"] = [j.stage for j in plan.jobs]
    for e in plan_problems(p, plan):
        f.fail("fusion stars plan", e)
    if "star_removal" not in k.chain(a.id):
        r = run_and_check(p, f, "fusion stars without split inputs", expect="failed")
        out["run"] = r
        f.info("fusion with stars whose inputs have no Split stars: allowed, the run fails", json.dumps(r["errors"]))
    return out


SCENARIOS = ["dualband_run", "dualband_then_gradient", "two_images_no_astrometry", "input_without_file",
             "fusion_with_stars_plan"]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        main_one(globals())
    else:
        drive(__file__, sys.argv[2:] if len(sys.argv) > 2 else SCENARIOS)
