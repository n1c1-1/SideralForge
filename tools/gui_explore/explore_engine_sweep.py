"""Engine exploration, plan only: many graphs from random valid edits (the placement rules, and the client model's own
edit functions with Undo / Redo), each state checked without computing anything.

Checks of a state: the definition validates and survives save/load with the same keys; every workflow's chain, keys
and blocks; every chain entry belongs to an active node of its graph (or to the source); plan_all and plan_to_block of
every block raise nothing, plan each key once, in dependency order, with finite durations, only stages of the
workflows' chains, and never past the block (Run to here); api.cost, api.health_checks and api.states raise nothing;
an edit of one workflow leaves the other workflows' keys alone; a refused edit says why without a raw placeholder.

    python tools/gui_explore/explore_engine_sweep.py --all
    python tools/gui_explore/explore_engine_sweep.py <scenario>
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_engine_common import (Findings, bayer_frames, drive, fresh, linear_fits, main_one,  # noqa: E402
                                   plan_problems, stretched_png)

from astro_gpu import api  # noqa: E402
from astro_gpu.core import project2, runner2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.project2 import Session  # noqa: E402
from astro_gpu.core.store import Keys  # noqa: E402

PLACEABLE = [b for b in wfl.PALETTE if b != "output"]
SEEN: dict[str, int] = {}


def once(f: Findings, kind: str, what: str, detail: str):
    """A finding once per distinct title (the sweep meets the same defect many times)."""
    SEEN[what] = SEEN.get(what, 0) + 1
    if SEEN[what] <= 2:
        getattr(f, kind)(what, detail)


def stage_owner_ok(w, entry: str) -> bool:
    """Whether a chain entry belongs to the source or to an active node of the graph."""
    act = [n for n in (w.blocks or []) if not n.get("off")]
    b = wfl.base(entry)
    if b in ("image", "fusion", "inventory", "calibrate", "sessions", "register", "stack"):
        return True
    if b in wfl.IMAGE_OPS:
        return any(n["name"] == entry for n in act)
    blk = {"astrometry": "astrometry", "reference": "astrometry", "catalogs": "astrometry", "surveys": "astrometry",
           "render_simple": "output", "star_removal": "stars"}.get(b)
    if blk == "output" and wfl.source_of(w) in ("stretched",):
        return True
    return blk is not None and any(n["block"] == blk for n in act)


def same_upstream(k, w1, w2, s) -> bool:
    """Whether a stage and everything it reads come from the same nodes in two workflows (a branch and its trunk):
    then their keys must be the same. Frame stages and catalogs (their settings follow the whole graph) left out."""
    for a in k.ancestors(w1.id, s) + [s]:
        b = wfl.base(a)
        if b in ("inventory", "calibrate", "sessions", "register", "stack", "catalogs", "reference"):
            return False
        if b in ("image", "fusion"):
            continue
        n1, n2 = wfl.node_of(w1.blocks or [], a), wfl.node_of(w2.blocks or [], a)
        if n1 is None or n1 != n2:
            return False
    return k.ancestors(w1.id, s) == k.ancestors(w2.id, s)


def check_state(p, f: Findings, ctx: str, deep: bool = True) -> dict:
    """Every check of one state of the project (nothing computed)."""
    out = {}
    try:
        p.validate()
    except Exception as e:  # noqa: BLE001
        once(f, "fail", "a state the edits allow does not validate", f"{ctx}: {type(e).__name__}: {e}")
        return out
    k = Keys(p)
    keys = {}
    for w in p.workflows:
        try:
            ch = k.chain(w.id)
            keys[w.id] = k.keys(w.id)
            api.blocks(p, w.id)
        except Exception as e:  # noqa: BLE001
            once(f, "fail", "a workflow's chain or keys raise", f"{ctx}: {w.name} {w.blocks} {w.source}: "
                 f"{type(e).__name__}: {e}")
            continue
        if w.blocks is not None:
            bad = [s for s in ch if not stage_owner_ok(w, s)]
            if bad:
                once(f, "fail", "a chain holds stages of no active block", f"{ctx}: {w.name} {bad} graph {w.blocks}")
            for n in w.blocks:
                if n["off"] or n["block"] in wfl.FRAME_BLOCKS:
                    continue
                want = [n["name"]] if n["block"] in wfl.IMAGE_OPS else [
                    s for s in wfl.BLOCKS[n["block"]] if s not in ("reference", "catalogs")]
                miss = [s for s in want if s not in ch]
                if miss:
                    once(f, "fail", "an active block's stage is missing from the chain",
                         f"{ctx}: {w.name} node {n} missing {miss} chain {ch}")
    out["keys"] = keys
    try:
        plan = runner2.plan_all(p, keys=k, gpu=False)
        for e in plan_problems(p, plan, k):
            once(f, "fail", "plan_all: " + e.split(":")[0], f"{ctx}: {e}")
        if not math.isfinite(plan.seconds):
            once(f, "fail", "plan_all: total duration not finite", ctx)
        out["jobs"] = len(plan)
    except Exception as e:  # noqa: BLE001
        once(f, "fail", "plan_all raises", f"{ctx}: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
    if deep:
        for w in p.workflows:
            if w.id not in keys:
                continue
            ch = k.chain(w.id)
            for blk, _ss in k.rules.chains.blocks_in(ch):
                try:
                    pl = runner2.plan_to_block(p, w.id, blk, keys=k, gpu=False)
                except Exception as e:  # noqa: BLE001
                    once(f, "fail", "plan_to_block raises", f"{ctx}: {w.name} {blk}: {type(e).__name__}: {e}")
                    continue
                for e in plan_problems(p, pl, k):
                    once(f, "fail", "plan_to_block: " + e.split(":")[0], f"{ctx}: {w.name} {blk}: {e}")
                want = set(k.rules.chains.run_to(blk, ch))
                anc = set()
                for s in want:
                    anc |= set(k.ancestors(w.id, s))
                got = {j.stage for j in pl.jobs if w.id in j.wf}
                if not want and not pl.skipped:
                    once(f, "warn", "Run to here on a block plans nothing (no stage)", f"{ctx}: {w.name} {blk}")
                if got - want - anc:
                    once(f, "fail", "Run to here plans stages after the block", f"{ctx}: {w.name} {blk}: "
                         f"{got - want - anc}")
        for name, fn in (("cost", lambda: api.cost(p, device="cpu")), ("health_checks", lambda: api.health_checks(p)),
                         ("states", lambda: api.states(p, update=False))):
            try:
                r = fn()
                if name == "cost" and not (math.isfinite(r.seconds) and r.seconds >= 0 and math.isfinite(r.disk_gb)):
                    once(f, "fail", "api.cost not finite", f"{ctx}: {r}")
            except Exception as e:  # noqa: BLE001
                once(f, "fail", f"api.{name} raises", f"{ctx}: {type(e).__name__}: {e}\n"
                     f"{traceback.format_exc()[-1200:]}")
        try:
            project2.save(p)
            q = project2.load(p.root)
            kq = Keys(q)
            for w in q.workflows:
                if w.id in keys and kq.keys(w.id) != keys[w.id]:
                    once(f, "fail", "keys change across save/load", f"{ctx}: {w.name}")
        except Exception as e:  # noqa: BLE001
            once(f, "fail", "save/load of a valid state fails", f"{ctx}: {type(e).__name__}: {e}")
    return out


def check_refusal(f: Findings, e, ctx: str):
    if e is None:
        return
    s = str(e)
    if not s or "{" in s or "}" in s or "Traceback" in s or "None" in s.split():
        once(f, "fail", "a refusal message is malformed", f"{ctx}: {s!r}")


# ---------------------------------------------------------------------------------------------- rules sweep
def base_project(name: str, rng: random.Random):
    d = fresh(name)
    p = project2.create(d / "P", "P", Target(name="M 31"))
    nights = rng.choice([1, 1, 2, 3])
    p.stack_mode = rng.choice(["per_session", "together"])
    for i in range(1, nights + 1):
        lights = bayer_frames(d / f"n{i}", n=3, h=20, w=30, seed=i)
        kw = {"darks": [str(bayer_frames(d / f"d{i}", n=2, h=20, w=30, kind="dark", seed=i))]} if rng.random() < .5 \
            else {}
        p.sessions.append(Session(i, f"night {i}", lights=[str(lights)], **kw))
    img = linear_fits(d / "img", h=40, w=50)
    png = stretched_png(d / "img", h=40, w=50)
    return d, p, img, png


def random_edit_graph(g: list, src: str, rng: random.Random, f: Findings, ctx: str, steps: int) -> list:
    g = [n for n in g if n["block"] != "output"]
    for _ in range(steps):
        prev = [dict(n) for n in g]
        r = rng.random()
        if r < 0.6:
            b = rng.choice(PLACEABLE)
            m = rng.choice(wfl.METHODS[b]) if b in wfl.METHODS else None
            idx = wfl.places(g, src, b)
            if not idx:
                check_refusal(f, wfl.palette_problem(g, src, b), f"{ctx} palette {b}")
                continue
            i = rng.choice(idx)
            g.insert(i, wfl.node(b, m, name=wfl.new_name(g, b)))
        elif r < 0.8 and g:
            i = rng.randrange(len(g))
            e = wfl.remove_problem(g, src, i)
            check_refusal(f, e, f"{ctx} remove {g[i]['name']}")
            if e is None:
                del g[i]
        elif g:
            i = rng.randrange(len(g))
            if g[i]["off"]:
                trial = [dict(n, off=False) if j == i else n for j, n in enumerate(g)]
                if wfl.graph_problem(wfl.with_output(trial, src), src) is None:
                    g = trial
            else:
                e = wfl.off_problem(g, src, i)
                check_refusal(f, e, f"{ctx} off {g[i]['name']}")
                if e is None:
                    g[i] = dict(g[i], off=True)
        # every step must stay valid
        try:
            wfl.check_blocks(wfl.with_output(g, src), src)
        except Exception as e:  # noqa: BLE001
            once(f, "fail", "a sequence of allowed edits gives a graph check_blocks refuses",
                 f"{ctx}: {g}: {type(e).__name__}: {e}")
            g = prev
    return g


def rules_sweep(f: Findings, seed: int, n_projects: int = 8):
    rng = random.Random(seed)
    stats = {"states": 0, "workflows": 0, "jobs": 0}
    for pi in range(n_projects):
        d, p, img, png = base_project(f"sweep{seed}_{pi}", rng)
        n_flows = rng.randint(1, 4)
        for fi in range(n_flows):
            src = rng.choice(["frames", "frames", "image", "stretched"])
            w = p.add_workflow(f"F{fi}", wfl.TEMPLATES["Classic stack"])
            if src == "image":
                w.source = {"path": str(img), "stretched": False}
            elif src == "stretched":
                w.source = {"path": str(png), "stretched": True}
            elif p.sessions and rng.random() < 0.4:
                w.sessions = [rng.choice(p.sessions).id]
            ctx = f"seed {seed} project {pi} flow {fi} ({src})"
            g = random_edit_graph([], src, rng, f, ctx, rng.randint(1, 10))
            before = {x.id: Keys(p).keys(x.id) for x in p.workflows if x.id != w.id and x.blocks is not None}
            try:
                p.set_blocks(w.id, g)
            except Exception as e:  # noqa: BLE001
                once(f, "fail", "set_blocks refuses a graph built by allowed edits", f"{ctx}: {g}: {e}")
                p.set_blocks(w.id, [])
            after = {x.id: Keys(p).keys(x.id) for x in p.workflows if x.id in before}
            if before != after:
                once(f, "fail", "editing one workflow changed another's keys", ctx)
            stats["states"] += 1
            check_state(p, f, ctx, deep=(fi == n_flows - 1))
        # branches: a random block of a random graph workflow, then a block after it
        for bi in range(rng.randint(0, 3)):
            ws = [x for x in p.workflows if x.blocks]
            if not ws:
                break
            w = rng.choice(ws)
            src = wfl.source_of(w)
            g = [n for n in w.blocks if n["block"] != "output"]
            if not g:
                continue
            i = rng.randrange(len(g))
            if g[i]["off"]:
                continue
            head = [dict(n) for n in g[:i + 1]]
            b = rng.choice(PLACEABLE)
            if wfl.place_problem(head, src, b, len(head)) is not None:
                continue
            m = rng.choice(wfl.METHODS[b]) if b in wfl.METHODS else None
            head.append(wfl.node(b, m, name=wfl.new_name(head, b)))
            x = p.add_workflow(f"{w.name} branch {bi}", w.options, params=copy.deepcopy(w.params))
            x.source, x.sessions = copy.deepcopy(w.source), copy.deepcopy(w.sessions)
            p.set_blocks(x.id, head)
            k = Keys(p)
            shared = [s for s in k.chain(x.id) if s in k.chain(w.id) and same_upstream(k, x, w, s)]

            def sig(wf, s):
                return [(a, wfl.node_of(wf.blocks, a)) for a in k.ancestors(wf.id, s) + [s]]
            for s in k.chain(x.id):
                if s in k.chain(w.id) and wfl.base(s) not in ("render_simple", "star_removal")                         and sig(x, s) == sig(w, s) and k.key(x.id, s) != k.key(w.id, s):
                    px, pw = k.key_parts(x.id, s), k.key_parts(w.id, s)
                    diff = {f: (px[f], pw[f]) for f in px if px[f] != pw[f]}
                    once(f, "warn", "loose: same upstream names and nodes, other key",
                         f"{w.name} -> {x.name}: {s}: {json.dumps(diff, default=str)[:1500]} trunk {w.blocks} "
                         f"branch {x.blocks}")
            for s in shared:
                if wfl.base(s) in ("render_simple", "star_removal"):
                    continue
                if k.key(x.id, s) != k.key(w.id, s):
                    once(f, "fail", "a branch does not share a stage it has the same as its trunk",
                         f"{w.name} -> {x.name}: {s}")
            plan = runner2.plan_all(p, keys=k, gpu=False)
            cnt = {}
            for j in plan.jobs:
                cnt[j.key] = cnt.get(j.key, 0) + 1
            stats["states"] += 1
            check_state(p, f, f"seed {seed} project {pi} branch {bi}", deep=False)
        # a fusion of the frames workflows that give an image (RGB) when there are three
        out_ws = [x for x in p.workflows if x.source is None and x.blocks and
                  any(n["block"] == "output" for n in x.blocks)]
        if len(out_ws) >= 1:
            fu = p.add_workflow("Fusion", wfl.TEMPLATES["Classic stack"])
            fu.source = {"fusion": {"palette": "HOO", "inputs": [
                {"pid": out_ws[0].id, "role": "Ha", "channel": "R"},
                {"pid": out_ws[-1].id, "role": "OIII", "channel": "GB"}]}}
            try:
                p.set_blocks(fu.id, [])
                check_state(p, f, f"seed {seed} project {pi} fusion")
            except Exception as e:  # noqa: BLE001
                once(f, "fail", "a fusion of two graph workflows cannot be set", f"{type(e).__name__}: {e}")
        stats["workflows"] += len(p.workflows)
    return stats


def sweep_a(f):
    return rules_sweep(f, 1)


def sweep_b(f):
    return rules_sweep(f, 2)


def sweep_c(f):
    return rules_sweep(f, 3)


def sweep_d(f):
    return rules_sweep(f, 4)


def sweep_e(f):
    return rules_sweep(f, 5)


# ---------------------------------------------------------------------------------------------- the model's edits
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


def model_sweep(f: Findings, seed: int, actions: int = 35):
    from astro_gpu.client.model import node_key
    rng = random.Random(seed)
    d, p, img, png = base_project(f"model{seed}", rng)
    project2.save(p)
    app, worker, m = _model(p)
    log = []
    try:
        if p.sessions:
            m.ensure_frames_workflow()
        for step in range(actions):
            app.processEvents()
            snap_keys = {w.id: Keys(p).keys(w.id) for w in p.workflows}
            snap_defs = json.dumps(p.to_dict()["workflows"], sort_keys=True, default=str)
            r = rng.random()
            views = list(m.paths)
            act = ""
            try:
                if r < 0.35 and views:
                    v = rng.choice(views)
                    b = rng.choice(PLACEABLE)
                    idx = m.place_indices(v, b)
                    if idx:
                        meth = rng.choice(wfl.METHODS[b]) if b in wfl.METHODS else None
                        i = rng.choice(idx)
                        act = f"insert {b}:{meth} at {i} in {v.name}"
                        m.insert_block(v, b, i, meth)
                    else:
                        reason = m.palette_reason(v, b)
                        act = f"refused {b} in {v.name}: {reason}"
                        if not reason:
                            once(f, "fail", "a block with no place has no palette reason", f"{v.name} {b}")
                        check_refusal(f, reason, act)
                elif r < 0.5 and views:
                    v = rng.choice(views)
                    g = m.graph(v)
                    if g:
                        n = rng.choice(g)
                        act = f"remove {n['name']} of {v.name}"
                        try:
                            m.remove_node(v, node_key(n))
                        except ValueError as e:
                            act += f" -> refused: {e}"
                            check_refusal(f, e, act)
                elif r < 0.62 and views:
                    v = rng.choice(views)
                    g = m.graph(v)
                    if g:
                        n = rng.choice(g)
                        act = f"switch {'on' if n['off'] else 'off'} {n['name']} of {v.name}"
                        try:
                            m.set_cut([v], node_key(n), not n["off"])
                        except ValueError as e:
                            act += f" -> refused: {e}"
                            check_refusal(f, e, act)
                elif r < 0.75 and views:
                    v = rng.choice(views)
                    g = m.graph(v)
                    if g:
                        n = rng.choice(g)
                        b = rng.choice(PLACEABLE)
                        if m.can_branch(v, node_key(n), b):
                            act = f"branch {b} after {n['name']} of {v.name}"
                            q, key = m.add_branch(v, node_key(n), b)
                            k = Keys(p)
                            # the trunk's stages up to the fork: one key for both
                            wq, wv = p.workflow(q.pid), p.workflow(v.pid)
                            for s in k.chain(q.pid):
                                if s in k.chain(v.pid) and wfl.base(s) not in ("render_simple", "star_removal") \
                                        and same_upstream(k, wq, wv, s) and k.key(q.pid, s) != k.key(v.pid, s):
                                    once(f, "fail", "a branch made by the model does not share its trunk's stage",
                                         f"{act}: {s}")
                elif r < 0.82 and views:
                    v = rng.choice(views)
                    g = m.graph(v)
                    if g:
                        n = rng.choice(g)
                        if m.can_send_to_post(v, node_key(n)):
                            act = f"send {n['name']} of {v.name} to Post"
                            m.send_to_post(v, node_key(n), f"{v.name} {n['name']}")
                elif r < 0.86:
                    act = "parallel flow"
                    m.add_parallel_flow()
                elif r < 0.9:
                    stretched = rng.random() < 0.4
                    act = f"image flow (stretched {stretched})"
                    with m.batch():
                        q = m.add_image_workflow("Image", str(png if stretched else img), stretched=stretched)
                        m.set_graph(q, [])
                elif r < 0.95 and m.can_undo():
                    act = "undo"
                    m.undo()
                elif m.can_redo():
                    act = "redo"
                    m.redo()
            except Exception as e:  # noqa: BLE001
                once(f, "fail", "a model edit raises", f"seed {seed} step {step} {act}: {type(e).__name__}: {e}\n"
                     f"{traceback.format_exc()[-1500:]}")
            log.append(act)
            if act:
                check_state(p, f, f"model seed {seed} step {step} {act}", deep=(step % 5 == 0))
            # undo once in a while: the keys come back exactly
            if act and act not in ("undo", "redo") and not act.startswith("refused") and "refused" not in act \
                    and rng.random() < 0.25 and m.can_undo():
                m.undo()
                back = {w.id: Keys(p).keys(w.id) for w in p.workflows}
                if back != snap_keys:
                    once(f, "fail", "Undo does not bring the keys back", f"seed {seed} step {step} {act}")
                if json.dumps(p.to_dict()["workflows"], sort_keys=True, default=str) != snap_defs:
                    once(f, "warn", "Undo does not bring the workflows back exactly",
                         f"seed {seed} step {step} {act}")
                m.redo()
                log.append("(undo+redo)")
    finally:
        worker.stop()
    return {"actions": log, "workflows": len(p.workflows)}


def model_a(f):
    return model_sweep(f, 11)


def model_b(f):
    return model_sweep(f, 12)


def model_c(f):
    return model_sweep(f, 13)


def model_d(f):
    return model_sweep(f, 14)


for _i in range(6, 21):
    globals()[f"sweep_{_i}"] = (lambda i: lambda f: rules_sweep(f, i))(_i)
for _i in range(15, 27):
    globals()[f"model_{_i}"] = (lambda i: lambda f: model_sweep(f, i))(_i)

SCENARIOS = (["sweep_a", "sweep_b", "sweep_c", "sweep_d", "sweep_e"] + [f"sweep_{i}" for i in range(6, 21)]
             + ["model_a", "model_b", "model_c", "model_d"] + [f"model_{i}" for i in range(15, 27)])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        main_one(globals())
    else:
        drive(__file__, sys.argv[2:] if len(sys.argv) > 2 else SCENARIOS, timeout=120)
