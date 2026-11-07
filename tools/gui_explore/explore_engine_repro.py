"""Engine exploration: reproductions of the findings, each a scenario (run twice before noting).

    python tools/gui_explore/explore_engine_repro.py --all
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_engine_common import (Findings, bayer_frames, drive, fresh, linear_fits, main_one,  # noqa: E402
                                   run_and_check, stretched_png)

from astro_gpu.core import project2, runner2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.project2 import Session  # noqa: E402
from astro_gpu.core.store import Keys  # noqa: E402

N = wfl.node


def _model(p):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication.instance() or QCoreApplication([])
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services.worker import ApiWorker
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True, gpu=False, device="cpu")
    w.wait(60)
    return app, w, m


def second_block_after_switch_off(f: Findings):
    """Registration switched off, then Registration dropped from the palette (the model's insert_block): the palette
    offers places; the drop fails with what message? Same for Astrometry, Selection, Calibration."""
    from astro_gpu.client.model import node_key
    d = fresh("dup_off")
    p = project2.create(d / "P", "P", Target(name="M 31"))
    p.sessions.append(Session(1, "n1", lights=[str(bayer_frames(d / "n1", n=3, h=20, w=30))]))
    w = p.add_workflow("W", wfl.TEMPLATES["Classic stack"])
    p.set_blocks(w.id, [N("calibration"), N("registration"), N("selection"), N("stacking"), N("astrometry"),
                        N("gradient")])
    project2.save(p)
    app, worker, m = _model(p)
    out = {}
    try:
        v = m.view(w.id)
        for blk in ("registration", "astrometry", "selection", "calibration"):
            key = node_key(next(n for n in m.graph(v) if n["block"] == blk))
            m.set_cut([v], key, True)
            idx = m.place_indices(v, blk)
            reason = m.palette_reason(v, blk)
            r = {"places_offered": idx, "palette_reason": reason}
            if idx:
                try:
                    m.insert_block(v, blk, idx[0])
                    r["insert"] = "ok"
                    r["graph"] = [n["name"] for n in m.graph(v)]
                except Exception as e:  # noqa: BLE001
                    r["insert"] = f"{type(e).__name__}: {e}"
                    f.fail(f"{blk} switched off, a second {blk} from the palette: places offered, drop fails",
                           r["insert"])
            m.set_cut([v], key, False)
            out[blk] = r
    finally:
        worker.stop()
    return out


def stars_reason_on_image(f: Findings):
    """The palette reason of Stars on a linear image (Denoise cannot be placed there either)."""
    return {"stars": str(wfl.palette_problem([N("gradient")], "image", "stars")),
            "denoise": str(wfl.palette_problem([N("gradient")], "image", "denoise"))}


def settings_of_repeated_blocks(f: Findings):
    """A setting of Gradient 2 (stage "gradient~2"): api.set_param stores it? its key follows? Removing Gradient 2 and
    placing a Gradient again: does the new block come back with the removed one's settings? A branch's new Gradient 2:
    does it get the trunk's Gradient 2 settings (another block)?"""
    from astro_gpu import api
    from astro_gpu.client.model import node_key
    d = fresh("rep_settings")
    img = linear_fits(d)
    p = project2.create(d / "P", "P", Target(name="M 31"))
    w = p.add_workflow("W", wfl.TEMPLATES["Classic stack"])
    w.source = {"path": str(img), "stretched": False}
    p.set_blocks(w.id, [N("gradient"), N("astrometry"), N("gradient", "rbf", name="gradient~2")])
    project2.save(p)
    out = {}
    k0 = Keys(p).key(w.id, "gradient~2")
    try:
        api.set_param(p, [w.id], "gradient~2", "degree", 4)
        out["set_param"] = p.workflow(w.id).params.get("gradient~2")
    except Exception as e:  # noqa: BLE001
        out["set_param"] = f"{type(e).__name__}: {e}"
        f.fail("api.set_param on a repeated block (gradient~2) raises", out["set_param"])
        p.workflow(w.id).params["gradient~2"] = {"degree": 4}
    k1 = Keys(p).key(w.id, "gradient~2")
    out["key_changed"] = k0 != k1
    try:
        api.set_param(p, [w.id], "gradient~2", "samples", 24)
    except Exception as e:  # noqa: BLE001
        out["set_param_samples"] = f"{type(e).__name__}: {e}"
    project2.save(p)
    app, worker, m = _model(p)
    try:
        v = m.view(w.id)
        m.remove_node(v, node_key(next(n for n in m.graph(v) if n["name"] == "gradient~2")))
        out["params_after_remove"] = dict(p.workflow(w.id).params)
        key = m.insert_block(v, "gradient", len(m.graph(v)), "rbf")
        out["new_node"] = key
        kp = Keys(p).key_params(w.id, "gradient~2")
        out["new_node_key_params"] = kp
        if kp.get("degree") == 4:
            f.warn("a Gradient placed after Gradient 2 was removed comes back with the removed block's settings",
                   json.dumps(out, default=str))
        # a branch: trunk [gradient, astrometry, gradient~2 (degree 4)], a branch after the first Gradient adds one
        m.remove_node(v, key)
        m.insert_block(v, "gradient", len(m.graph(v)), "rbf")
        p.workflow(w.id).params["gradient~2"] = {"degree": 5}
        q, bkey = m.add_branch(v, "fond", "gradient")
        out["branch"] = {"graph": m.graph(q), "params": p.workflow(q.pid).params,
                         "key_params": Keys(p).key_params(q.pid, "gradient~2")}
        if Keys(p).key_params(q.pid, "gradient~2").get("degree") == 5:
            f.warn("a branch's new Gradient takes the settings of the trunk's Gradient 2, another block",
                   json.dumps(out["branch"], default=str))
    finally:
        worker.stop()
    return out


def run_window_of_repeated_blocks(f: Findings):
    """A real run of [Gradient, Gradient 2] from an image; its events fed to the client's RunTrack (what the run
    window shows): each job's final state. And a failing Gradient 2 (the image file removed after the first run is
    not enough: a bad setting instead): whose state says failed?"""
    from astro_gpu.client.services.run import RunTrack
    d = fresh("runwin")
    img = linear_fits(d)
    p = project2.create(d / "P", "P", Target(name="M 31"))
    w = p.add_workflow("W", wfl.TEMPLATES["Classic stack"])
    w.source = {"path": str(img), "stretched": False}
    p.set_blocks(w.id, [N("gradient"), N("gradient", "rbf", name="gradient~2")])
    project2.save(p)
    plan = runner2.plan_all(p)
    info = runner2.prepare_run(p, plan, "cpu")
    st = runner2.run_plan(p.root, info.plan_file, echo=False)
    evs, _ = runner2.read_events(p.root, info.offset, info.run_id)
    tr = RunTrack.of_plan(plan, info.run_id)
    tr.apply(evs)
    states = {j.stage: j.state for j in tr.jobs}
    keyless = [e for e in evs if e.get("type") in ("stage_start", "stage_end", "progress") and not e.get("key")]
    out = {"status": st, "states": states, "events_without_key": len(keyless),
           "stages_of_keyless": sorted({e.get("stage") for e in keyless})}
    if any(v != "done" for v in states.values()):
        f.fail("the run window's states after a done run of Gradient + Gradient 2", json.dumps(out))
    if keyless:
        f.fail("the events of a repeated block's job carry no key, workflow or block", json.dumps(out))
    return out


def panel_setting_of_gradient2(f: Findings):
    """What the panel does for a setting of Gradient 2 (its rows' stage is "gradient~2", settings._stages_of_step):
    model.throws_away then model.apply_change, as Panel._propose calls them; and api.describe of it."""
    from astro_gpu import api
    import traceback
    d = fresh("panel_g2")
    img = linear_fits(d)
    p = project2.create(d / "P", "P", Target(name="M 31"))
    w = p.add_workflow("W", wfl.TEMPLATES["Classic stack"])
    w.source = {"path": str(img), "stretched": False}
    p.set_blocks(w.id, [N("gradient"), N("gradient", "rbf", name="gradient~2")])
    project2.save(p)
    out = {}
    try:
        out["describe"] = len(api.describe(p, w.id, "gradient~2"))
    except Exception as e:  # noqa: BLE001
        out["describe"] = f"{type(e).__name__}: {e}"
    app, worker, m = _model(p)
    try:
        v = m.view(w.id)
        for name, value in (("degree", 4), ("rbf_smooth", 0.3), ("samples", 20)):
            ch = {"kind": "param", "stage": "gradient~2", "name": name, "value": value, "suggested": None}
            try:
                m.throws_away([v], ch)
                m.apply_change([v], ch)
                out[name] = ("ok", p.workflow(w.id).params.get("gradient~2"))
            except Exception as e:  # noqa: BLE001
                out[name] = (f"{type(e).__name__}: {e}", traceback.format_exc()[-900:])
                f.fail(f"changing {name} of Gradient 2 as the panel does raises {type(e).__name__}", str(e))
        # the first Gradient: the same works?
        ch = {"kind": "param", "stage": "gradient", "name": "degree", "value": 4, "suggested": None}
        try:
            m.throws_away([v], ch)
            m.apply_change([v], ch)
            out["gradient1"] = ("ok", p.workflow(w.id).params.get("gradient"))
        except Exception as e:  # noqa: BLE001
            out["gradient1"] = f"{type(e).__name__}: {e}"
    finally:
        worker.stop()
    return out


SCENARIOS = ["second_block_after_switch_off", "stars_reason_on_image", "settings_of_repeated_blocks",
             "run_window_of_repeated_blocks", "panel_setting_of_gradient2"]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        main_one(globals())
    else:
        drive(__file__, sys.argv[2:] if len(sys.argv) > 2 else SCENARIOS)
