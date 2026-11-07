"""Engine exploration, image sources: every chain of image blocks the graph rules allow, computed for real on the CPU
(tiny synthetic images), or planned only when a block needs the network (Astrometry, then Color).

    python tools/gui_explore/explore_engine_image.py            # the driver: every scenario, each in its own process
    python tools/gui_explore/explore_engine_image.py <scenario> # one scenario
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_engine_common import (Findings, drive, fresh, keys_of, linear_fits, main_one,  # noqa: E402
                                   plan_problems, run_and_check, stretched_png)

from astro_gpu import api  # noqa: E402
from astro_gpu.core import project2, runner2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.store import Keys, Store  # noqa: E402

GRAD = [("gradient", m) for m in wfl.METHODS["gradient"]]
ALPHABET = GRAD + [("astrometry", None), ("color", None), ("denoise", "pretrained"), ("stars", "neural")]


def build(seq) -> list[dict]:
    """Nodes of a sequence of (block, method), named as the palette names them (new_name)."""
    g: list[dict] = []
    for b, m in seq:
        g.append(wfl.node(b, m, name=wfl.new_name(g, b)))
    return g


def valid_sequences(source: str, max_len: int = 4, alphabet=ALPHABET) -> list[list[dict]]:
    """Every sequence the placement rules accept when built block by block at its end (place_problem)."""
    out = []
    frontier: list[list] = [[]]
    for _ in range(max_len):
        nxt = []
        for seq in frontier:
            g = build(seq)
            for b, m in alphabet:
                if wfl.place_problem(g, source, b, len(g)) is None:
                    nxt.append(seq + [(b, m)])
        out += nxt
        frontier = nxt
    return out


def label(seq) -> str:
    return "+".join(f"{b}:{m}" if m else b for b, m in seq) or "(empty)"


def new_project(name: str):
    d = fresh(name)
    p = project2.create(d / "P", "P", Target(name="M 31"))
    return d, p


def image_workflow(p, path: Path, blocks, name="Image", stretched=False):
    w = p.add_workflow(name, wfl.TEMPLATES["Classic stack"])
    w.source = {"path": str(path), "stretched": stretched}
    p.set_blocks(w.id, blocks)
    return w


# ---------------------------------------------------------------------------------------------- scenarios
def enumerate_rules(f: Findings):
    """The valid sequences per source (rules only), and what is refused for an image with the reasons."""
    out = {}
    for src in ("image", "stretched"):
        seqs = valid_sequences(src, 4)
        out[src] = [label(s) for s in seqs]
    # what an image source refuses, with its reason (palette)
    reasons = {b: (str(wfl.palette_problem([], "image", b)) if wfl.palette_problem([], "image", b) else "")
               for b in wfl.PALETTE}
    out["image_palette_reasons"] = reasons
    sreasons = {b: (str(wfl.palette_problem([], "stretched", b)) if wfl.palette_problem([], "stretched", b) else "")
                for b in wfl.PALETTE}
    out["stretched_palette_reasons"] = sreasons
    # every valid sequence is accepted by check_blocks and by set_blocks (with_output), and its chain is consistent
    _d, p = new_project("rules")
    img = linear_fits(_d)
    for s in valid_sequences("image", 4):
        g = build(s)
        try:
            w = image_workflow(p, img, g, name=label(s)[:60])
            ch = Keys(p).chain(w.id)
            names = [n["name"] for n in g if n["block"] in wfl.IMAGE_OPS]
            if [x for x in ch if wfl.base(x) in wfl.IMAGE_OPS] != names:
                f.fail("chain order differs from the graph", f"{label(s)}: chain {ch}")
            Keys(p).keys(w.id)
        except Exception as e:  # noqa: BLE001
            f.fail("a valid sequence cannot be set or keyed", f"{label(s)}: {type(e).__name__}: {e}")
    return {k: (len(v) if isinstance(v, list) else v) for k, v in out.items()} | {"stretched": out["stretched"]}


def _run_seqs(f: Findings, seqs, tag: str, one_project: bool):
    results = {}
    img_dir = fresh(f"img_{tag}")
    img = linear_fits(img_dir)
    if one_project:
        _d, p = new_project(tag)
        for s in seqs:
            image_workflow(p, img, build(s), name=label(s)[:60])
        plan = runner2.plan_all(p)
        njobs = len(plan)
        distinct = {(st, k) for w in p.workflows for st, k in Keys(p).keys(w.id).items()}
        if njobs != len(distinct):
            f.fail(f"{tag}: the plan has {njobs} jobs for {len(distinct)} distinct results")
        results["all"] = run_and_check(p, f, tag, plan=plan)
        results["all"]["distinct"] = len(distinct)
        return results
    for s in seqs:
        _d, p = new_project(tag)
        image_workflow(p, img, build(s))
        results[label(s)] = run_and_check(p, f, f"{tag} {label(s)}")
    return results


def gradients_len1(f: Findings):
    return _run_seqs(f, [[g] for g in GRAD], "len1", False)


def gradients_len2(f: Findings):
    return _run_seqs(f, [list(x) for x in itertools.product(GRAD, repeat=2)], "len2", False)


def _len3(f: Findings, first: int):
    seqs = [[GRAD[first]] + list(x) for n in (0, 1, 2) for x in itertools.product(GRAD, repeat=n)]
    return _run_seqs(f, seqs, f"len3_{first}", True)


def gradients_len3_poly(f: Findings):
    return _len3(f, 0)


def gradients_len3_rbf(f: Findings):
    return _len3(f, 1)


def gradients_len3_auto(f: Findings):
    return _len3(f, 2)


def gradients_len4_sample(f: Findings):
    seqs = [[GRAD[0]] * 4, [GRAD[1]] * 4, [GRAD[2]] * 4, [GRAD[0], GRAD[1], GRAD[2], GRAD[0]],
            [GRAD[2], GRAD[1], GRAD[0], GRAD[2]]]
    return _run_seqs(f, seqs, "len4", True)


def image_only(f: Findings):
    """An image source with no block: Output only."""
    return _run_seqs(f, [[]], "empty", False)


def switched_off(f: Findings):
    """Switched-off nodes: as if absent (their key and the next's); computed for real."""
    out = {}
    d = fresh("off")
    img = linear_fits(d)
    cases = {
        "g1off": [wfl.node("gradient", "polynomial", off=True), wfl.node("gradient", "rbf", name="gradient~2")],
        "g2off": [wfl.node("gradient", "polynomial"), wfl.node("gradient", "rbf", name="gradient~2", off=True)],
        "alloff": [wfl.node("gradient", "polynomial", off=True), wfl.node("gradient", "rbf", name="gradient~2",
                                                                           off=True)],
        "middleoff": [wfl.node("gradient", "polynomial"), wfl.node("gradient", "rbf", name="gradient~2", off=True),
                      wfl.node("gradient", "automatic", name="gradient~3")],
    }
    ref = {
        "g1off": [wfl.node("gradient", "rbf")],
        "g2off": [wfl.node("gradient", "polynomial")],
        "alloff": [],
        "middleoff": [wfl.node("gradient", "polynomial"), wfl.node("gradient", "automatic", name="gradient~2")],
    }
    _d, p = new_project("off")
    ws = {}
    for name, g in cases.items():
        try:
            ws[name] = image_workflow(p, img, g, name=name)
            ws["ref_" + name] = image_workflow(p, img, ref[name], name="ref_" + name)
        except Exception as e:  # noqa: BLE001
            f.fail(f"switched off: {name} cannot be set", f"{type(e).__name__}: {e}")
    k = Keys(p)
    for name in cases:
        if name not in ws:
            continue
        a, b = k.keys(ws[name].id), k.keys(ws["ref_" + name].id)
        out[name] = {"chain": list(a), "ref_chain": list(b)}
        if a.get("render_simple") != b.get("render_simple"):
            f.warn(f"switched off: {name}: the Output's key differs from the same graph without the switched-off "
                   "node (the result is the same image, computed twice)", json.dumps({"off": a, "without": b}))
    out["run"] = run_and_check(p, f, "off")
    return out


def stretched_png_alone(f: Findings):
    d = fresh("str")
    png = stretched_png(d)
    _d, p = new_project("stretched")
    image_workflow(p, png, [], stretched=True)
    return run_and_check(p, f, "stretched alone")


def stretched_png_stars(f: Findings):
    d = fresh("strs")
    png = stretched_png(d)
    _d, p = new_project("stretched_stars")
    image_workflow(p, png, [wfl.node("stars")], stretched=True)
    return run_and_check(p, f, "stretched + stars")


def png_as_linear(f: Findings):
    """A PNG (8-bit) given as a linear image (the user did not tick 'stretched'): accepted? runs?"""
    d = fresh("pnglin")
    png = stretched_png(d)
    _d, p = new_project("pnglin")
    image_workflow(p, png, [wfl.node("gradient")])
    return run_and_check(p, f, "png as linear")


def mono_fits(f: Findings):
    """A 2-D (monochrome) FITS image, values above 1.5 (no x65535)."""
    from astropy.io import fits
    import numpy as np
    d = fresh("mono")
    rng = np.random.default_rng(5)
    a = (1000 + rng.normal(0, 10, (120, 160)) + np.linspace(0, 50, 160)[None]).astype(np.float32)
    fits.writeto(d / "mono.fits", a)
    _d, p = new_project("mono")
    image_workflow(p, d / "mono.fits", [wfl.node("gradient"), wfl.node("gradient", "automatic", name="gradient~2")])
    return run_and_check(p, f, "mono fits")


def branches_shared_prefix(f: Findings):
    """Two workflows sharing a prefix (a branch): the shared Gradient computed once; then a third flow."""
    d = fresh("branch")
    img = linear_fits(d)
    _d, p = new_project("branch")
    a = image_workflow(p, img, [wfl.node("gradient", "polynomial"), wfl.node("gradient", "rbf", name="gradient~2")],
                       name="A")
    b = image_workflow(p, img, [wfl.node("gradient", "polynomial"),
                                wfl.node("gradient", "automatic", name="gradient~2")], name="B")
    c = image_workflow(p, img, [wfl.node("gradient", "polynomial")], name="C (sent to Post)")
    k = Keys(p)
    if k.key(a.id, "gradient") != k.key(b.id, "gradient") or k.key(a.id, "gradient") != k.key(c.id, "gradient"):
        f.fail("branch: the shared prefix has different keys")
    plan = runner2.plan_all(p)
    g = [j for j in plan.jobs if j.stage == "gradient"]
    if len(g) != 1 or sorted(g[0].wf) != sorted([a.id, b.id, c.id]):
        f.fail("branch: the shared Gradient is not one job of the three workflows",
               str([(j.stage, j.wf) for j in plan.jobs]))
    img_jobs = [j for j in plan.jobs if j.stage == "image"]
    if len(img_jobs) != 1:
        f.fail("branch: the image stage is planned more than once", str([(j.stage, j.wf) for j in img_jobs]))
    return run_and_check(p, f, "branch", plan=plan)


def two_sources_same_file(f: Findings):
    """Two parallel flows on the same image file: one image job, shared."""
    d = fresh("same")
    img = linear_fits(d)
    _d, p = new_project("same")
    a = image_workflow(p, img, [wfl.node("gradient")], name="A")
    b = image_workflow(p, img, [wfl.node("gradient", "rbf")], name="B")
    plan = runner2.plan_all(p)
    if sum(j.stage == "image" for j in plan.jobs) != 1:
        f.fail("same file: two image jobs", str([(j.stage, j.wf) for j in plan.jobs]))
    return run_and_check(p, f, "same file", plan=plan)


def two_files_same_name(f: Findings):
    """Two different images of the same file name, size and date in two folders: one key? (the path is not keyed)"""
    import os
    d = fresh("samename")
    a1 = linear_fits(d / "n1", "stack.fits", seed=1)
    a2 = linear_fits(d / "n2", "stack.fits", seed=99)
    st = os.stat(a1)
    os.utime(a2, ns=(st.st_atime_ns, st.st_mtime_ns))
    _d, p = new_project("samename")
    wa = image_workflow(p, a1, [wfl.node("gradient")], name="A")
    wb = image_workflow(p, a2, [wfl.node("gradient")], name="B")
    k = Keys(p)
    same = k.key(wa.id, "image") == k.key(wb.id, "image")
    out = {"same_image_key": same}
    if same:
        f.warn("two different image files (same name, size and date, other folders) share one key: the second "
               "workflow shows the first image's results", f"{a1} / {a2}")
    out["run"] = run_and_check(p, f, "same name")
    if same:
        import numpy as np
        s = Store(p.root)
        arr = np.load(s.dir("image", k.key(wb.id, "image")) / "drz_full.npy")
        from astro_gpu.stages.image import read_image
        out["B_reads_A"] = bool(np.allclose(arr / arr.max(), read_image(a1) / read_image(a1).max()))
    return out


def keys_properties(f: Findings):
    """Keys stable across save/load; a workflow's keys unchanged when an unrelated one is edited; reordered gradients
    give other keys; removing a block makes the right stages Outdated (states after a real run)."""
    d = fresh("keys")
    img = linear_fits(d)
    _d, p = new_project("keys")
    a = image_workflow(p, img, [wfl.node("gradient", "polynomial"), wfl.node("gradient", "rbf", name="gradient~2")],
                       name="A")
    b = image_workflow(p, img, [wfl.node("gradient", "automatic")], name="B")
    k0 = keys_of(p)
    project2.save(p)
    q = project2.load(p.root)
    if keys_of(q) != k0:
        f.fail("keys change across save/load", json.dumps([k0, keys_of(q)]))
    # an unrelated edit
    q.set_blocks(b.id, [wfl.node("gradient", "rbf"), wfl.node("gradient", "polynomial", name="gradient~2")])
    if keys_of(q)[a.id] != k0[a.id]:
        f.fail("editing workflow B changed workflow A's keys")
    # reordered
    r = image_workflow(q, img, [wfl.node("gradient", "rbf"), wfl.node("gradient", "polynomial", name="gradient~2")],
                       name="R")
    kr = Keys(q).keys(r.id)
    ka = k0[a.id]
    if kr["render_simple"] == ka["render_simple"] or kr["gradient~2"] == ka["gradient~2"]:
        f.fail("reordered gradients give the same keys", json.dumps([ka, kr]))
    # the same graph with the node names swapped (gradient~2 placed first, as a palette drop at index 0 gives)
    s = image_workflow(q, img, [wfl.node("gradient", "rbf", name="gradient~2"), wfl.node("gradient", "polynomial")],
                       name="S")
    ks = Keys(q).keys(s.id)
    out = {"R": kr, "S": ks, "A": ka}
    if ks["render_simple"] != kr["render_simple"]:
        f.warn("the same graph (rbf then polynomial) has other keys when its nodes are named the other way round "
               "(the block placed second put first): computed twice", json.dumps({"R": kr, "S": ks}))
    # a real run of A, then remove its first Gradient: what is Outdated?
    p2 = project2.load(p.root)
    run_and_check(p2, f, "keys run")
    api.states(p2)                                                     # writes last_keys
    p2.set_blocks(a.id, [wfl.node("gradient", "rbf", name="gradient~2")])
    project2.save(p2)
    st = api.states(project2.load(p.root))
    out["after_remove"] = {f"{w}:{s}": (v.state, v.reason) for (w, s), v in st.items() if w == a.id}
    exp_done = {"image"}
    for (w, s), v in st.items():
        if w != a.id:
            if v.state != "done":
                f.fail("removing a block of A changed B's state", f"{s}: {v.state}")
            continue
        if s in exp_done and v.state != "done":
            f.fail("after removing the first Gradient, the image is not done", f"{s}: {v.state}")
    return out


def plan_to_each_block(f: Findings):
    """Run to here on every block of image graphs (plan only), then run one of them for real."""
    d = fresh("runto")
    img = linear_fits(d)
    _d, p = new_project("runto")
    seqs = [s for s in valid_sequences("image", 2)]
    ws = []
    for s in seqs:
        ws.append(image_workflow(p, img, build(s), name=label(s)[:60]))
    project2.save(p)
    k = Keys(p)
    out = {"n": len(ws)}
    for w in ws:
        for blk, stages in k.rules.chains.blocks_in(k.chain(w.id)):
            try:
                plan = runner2.plan_to_block(p, w.id, blk)
            except Exception as e:  # noqa: BLE001
                f.fail("plan_to_block raises", f"{w.name} {blk}: {type(e).__name__}: {e}")
                continue
            for e in plan_problems(p, plan, k):
                f.fail("plan_to_block shape", f"{w.name} {blk}: {e}")
            want = set(k.rules.chains.run_to(blk, k.chain(w.id)))
            got = {j.stage for j in plan.jobs if w.id in j.wf}
            if not want <= got:
                f.fail("plan_to_block misses the block's own stages", f"{w.name} {blk}: want {want} got {got}")
            anc = set()
            for s in want:
                anc |= set(k.ancestors(w.id, s))
            extra = got - want - anc
            if extra:
                f.fail("plan_to_block plans stages after the block", f"{w.name} {blk}: {extra}")
    # one for real: run to the second Gradient of a 3-gradient graph
    w = image_workflow(p, img, build([GRAD[0], GRAD[1], GRAD[2]]), name="three")
    project2.save(p)
    plan = runner2.plan_to_block(p, w.id, "gradient~2")
    out["runto_jobs"] = [j.stage for j in plan.jobs]
    if any(j.stage in ("gradient~3", "render_simple") for j in plan.jobs):
        f.fail("run to Gradient 2 plans what comes after it", str(out["runto_jobs"]))
    info = runner2.prepare_run(p, plan, "cpu")
    st = runner2.run_plan(p.root, info.plan_file, echo=False)
    out["runto_status"] = st
    if st != "done":
        f.fail("run to Gradient 2 fails", st)
    after = runner2.plan_to_block(project2.load(p.root), w.id, "gradient~2")
    if len(after):
        f.fail("run to Gradient 2 again plans jobs", str([j.stage for j in after.jobs]))
    return out


def astrometry_color_plan(f: Findings):
    """Image chains with Astrometry and Color: planned only (the solve needs the network), shape and keys checked."""
    d = fresh("astro")
    img = linear_fits(d)
    _d, p = new_project("astro")
    seqs = [s for s in valid_sequences("image", 4) if any(b in ("astrometry", "color") for b, _ in s)]
    ws = [image_workflow(p, img, build(s), name=label(s)[:60]) for s in seqs]
    project2.save(p)
    plan = runner2.plan_all(p)
    for e in plan_problems(p, plan):
        f.fail("astrometry plan shape", e)
    k = Keys(p)
    out = {"n": len(ws), "jobs": len(plan)}
    # one astrometry job for all (same image, same target)
    astro = [j for j in plan.jobs if j.stage == "astrometry"]
    if len(astro) != 1:
        f.warn("astrometry planned more than once for one image", str([(j.key, j.wf) for j in astro]))
    # a Color twice: the second reads the first's corrected image
    for w in ws:
        ch = k.chain(w.id)
        colors = [x for x in ch if wfl.base(x) == "color"]
        if len(colors) >= 2:
            dd = k.deps(w.id, colors[1])
            if colors[0] not in dd:
                f.fail("the second Color does not read the first", f"{w.name}: {dd}")
        if "catalogs" in ch and not colors:
            f.fail("catalogs in a chain without Color", w.name)
        if colors and "catalogs" not in ch:
            f.fail("Color without catalogs in its chain", f"{w.name}: {ch}")
        for s in ch:
            try:
                k.deps(w.id, s)
            except Exception as e:  # noqa: BLE001
                f.fail("deps raise", f"{w.name} {s}: {e}")
    # api.cost and health checks
    c = api.cost(p, device="cpu")
    out["cost"] = {"seconds": c.seconds, "jobs": c.jobs if hasattr(c, "jobs") else None}
    hc = api.health_checks(p)
    out["health"] = [(x.level, x.key, x.text) for x in hc] if hc and hasattr(hc[0], "key") else [str(x) for x in hc]
    return out


SCENARIOS = ["enumerate_rules", "image_only", "gradients_len1", "gradients_len2", "gradients_len3_poly", "gradients_len3_rbf", "gradients_len3_auto",
             "gradients_len4_sample", "switched_off", "stretched_png_alone", "stretched_png_stars", "png_as_linear",
             "mono_fits", "branches_shared_prefix", "two_sources_same_file", "two_files_same_name",
             "keys_properties", "plan_to_each_block", "astrometry_color_plan"]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        main_one(globals())
    else:
        only = sys.argv[2:] if len(sys.argv) > 2 else SCENARIOS
        drive(__file__, only)
