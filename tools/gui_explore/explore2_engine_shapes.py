"""Engine shapes, round 2: more graphs built in the window and computed on the CPU (tiny data): switching off after a
run, Run to here on a repeated block, a sweep of frame graphs in one project, Split stars and Combine, a dual-band
Fusion, two parallel flows with a branch.

    python explore2_engine_shapes.py <scenario> [--en]
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_engine_window as W  # noqa: E402
from explore2_engine_window import (P, blk, blocks_desc, click_run, cpu, dump_dialog, ex, frames_project,  # noqa: E402
                                    go_post, image_project, log_rows, place, show_state, stage_runs, store_dirs,
                                    toast, wait_run)

from PySide6.QtCore import Qt  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402


def sc_switch_off_after_run():
    """Image [Gradient 1, Gradient 2] computed; Gradient 1 switched off (D): what is Outdated, what the run computes."""
    root = image_project("offrun")
    ex.open(root)
    cpu()
    place("gradient", "gradient")
    click_run()
    wait_run()
    ex.close_dialogs()
    n0 = len(stage_runs(root))
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    show_state("G1 off")
    for it in ex.blocks():
        P(f"    {it.node.key}: state={getattr(it, 'state', None)!r} title={it.title!r}")
    ex.shot("offrun_g1_off")
    click_run()
    ex.settle(3)
    dump_dialog("rerun start")
    wait_run()
    dump_dialog("rerun end")
    P("  stage_end rerun:", stage_runs(root))
    ex.close_dialogs()
    # back on: everything is there already?
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    show_state("G1 on again")
    for it in ex.blocks():
        P(f"    {it.node.key}: state={getattr(it, 'state', None)!r}")


def sc_run_to_here():
    """Image [Gradient 1, 2, 3]: Run to here (Shift+F5) on Gradient 2: only what it needs; then Gradient 3 stays to do."""
    root = image_project("runto")
    ex.open(root)
    cpu()
    place("gradient", "gradient", "gradient")
    g2 = blk("fond~2")
    ex.click_block(g2)
    m = ex.right_click_block(blk("fond~2"))
    ex.click_block(blk("fond~2"))
    P("  menu G2:", [(a.get("text"), a.get("enabled")) for a in m])
    ex.key(Qt.Key_F5, ex.w, Qt.ShiftModifier)
    ex.settle(5)
    dump_dialog("run to here G2")
    d = W.run_dialog()
    if d is not None and d.b_run.isVisible() and d.b_run.isEnabled():
        ex.click(d.b_run)
        wait_run()
        dump_dialog("end")
        ex.shot("runto_end", W.run_dialog())
    P("  stage_end:", stage_runs(root))
    ex.close_dialogs()
    show_state("after run to G2")
    for it in ex.blocks():
        P(f"    {it.node.key}: state={getattr(it, 'state', None)!r}")


def clear_graph():
    """Every block of the flow deleted, the last first (click, Del)."""
    for _ in range(12):
        xs = [it for it in ex.blocks() if it.node.key not in ("final", "sources", "image")]
        if not xs:
            return
        it = max(xs, key=lambda b: b.node.col)
        ex.click_block(it)
        ex.key(Qt.Key_Delete)
        ex.settle(4)


FRAME_PREFIXES = [["stacking"], ["calibration", "stacking"], ["registration", "stacking"],
                  ["selection", "stacking"], ["calibration", "registration", "stacking"],
                  ["registration", "selection", "stacking"], ["calibration", "registration", "selection", "stacking"],
                  ["calibration"], ["registration"], ["selection"], ["calibration", "registration"]]
TAILS = [[], ["gradient"], ["denoise"], ["gradient", "denoise"], ["gradient", "gradient"], ["denoise", "denoise"],
         ["gradient", "denoise", "stars"], ["denoise", "stars"], ["stars"]]


def _sweep(pairs, root, budget_s):
    t0 = time.time()
    res = []
    for pre, tail in pairs:
        if time.time() - t0 > budget_s:
            P("  budget reached")
            break
        ex.close_dialogs()
        clear_graph()
        want = pre + tail
        refused = []
        for b in want:
            before = [n["name"] for n in (ex.s.p.workflows[0].blocks or [])]
            c = ex.card(b)
            why = getattr(c, "reason", None) if c is not None else "no card"
            place(b)
            after = [n["name"] for n in (ex.s.p.workflows[0].blocks or [])]
            if after == before:
                refused.append((b, why or toast()))
        got = [n["name"] for n in (ex.s.p.workflows[0].blocks or []) if n["block"] != "output"]
        b = ex.w.bt_run
        line = {"want": "+".join(want), "got": "+".join(got), "refused": refused,
                "run_enabled": b.isEnabled(), "run_tip": b.toolTip()}
        if b.isEnabled():
            ex.click(b)
            st = wait_run(180)
            d = W.run_dialog()
            line["status"] = st
            try:
                line["rows"] = [(r.name.text(), r.mark.state, r.toolTip()[:200]) for r in d.rows] if d else None
            except RuntimeError:
                line["rows"] = "gone"
            if st != "done":
                ex.shot(f"sweep_{'_'.join(got)[:60]}_{st}", d)
                line["stage_end"] = [x for x in stage_runs(root) if x[2] != "done"]
        P("  SWEEP", line)
        res.append(line)
    return res


def sc_sweep_frames():
    root = frames_project("sweep", n=6, calib=True)
    ex.open(root)
    cpu()
    pairs = [(p, ["gradient"]) for p in FRAME_PREFIXES] + \
            [(["calibration", "registration", "stacking"], t) for t in TAILS if t != ["gradient"]]
    return _sweep(pairs, root, float(os.environ.get("SWEEP_BUDGET", "1100")))


def sc_sweep_frames_b():
    """The other prefixes with Denoise / Stars tails."""
    root = frames_project("sweepb", n=6)
    ex.open(root)
    cpu()
    pairs = [(p, t) for p in (["stacking"], ["registration", "stacking"], ["selection", "stacking"])
             for t in (["denoise"], ["denoise", "stars"], ["gradient", "gradient", "denoise"])]
    return _sweep(pairs, root, float(os.environ.get("SWEEP_BUDGET", "1100")))


def sc_split_combine():
    """Frames: Registration, Stacking, Gradient, Denoise, Split stars (Stars/background); its two images sent to Post,
    Combine; run; the Post page."""
    root = frames_project("split")
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient", "denoise", "stars")
    show_state("placed")
    st = blk("etoiles")
    m = ex.right_click_block(st) if st is not None else []
    P("  stars menu:", [(a.get("text"), a.get("enabled"), [s.get("text") for s in a.get("sub") or []]) for a in m])
    ex.shot("split_placed")
    click_run()
    ex.settle(3)
    dump_dialog("start")
    s = wait_run(300)
    dump_dialog("end")
    ex.shot("split_run_end", W.run_dialog())
    P("  stage_end:", stage_runs(root))
    ex.close_dialogs()
    show_state("after run")
    ex.shot("split_graph_after")
    page = go_post()
    ex.shot("split_post")
    P("  saved layers:", (getattr(ex.s.p, "post", None) or {}).get("layers"))
    return s


def sc_fusion_dual():
    """A dual-band Fusion (HOO) of one session's stack (the same flow for Ha and OIII): built in the definition, run
    from the window; the Post page."""
    root = frames_project("fdual")
    p = project2.load(root)
    w = p.workflows[0]
    p.set_blocks(w.id, [wfl.node("registration"), wfl.node("stacking"), wfl.node("gradient")])
    fu = p.add_workflow("Fusion", wfl.TEMPLATES["Classic stack"])
    fu.source = {"fusion": {"palette": "HOO", "inputs": [{"pid": w.id, "role": "Ha", "channel": "R"},
                                                          {"pid": w.id, "role": "OIII", "channel": "GB"}]}}
    p.set_blocks(fu.id, [])
    project2.save(p)
    ex.open(root)
    cpu()
    show_state("opened")
    ex.shot("fdual_graph")
    click_run()
    ex.settle(3)
    dump_dialog("start")
    s = wait_run(240)
    dump_dialog("end")
    ex.shot("fdual_run_end", W.run_dialog())
    P("  stage_end:", stage_runs(root))
    ex.close_dialogs()
    show_state("after run")
    # a block after the fusion? the palette's cards with the Fusion selected
    ex.click_block(blk("fusion"))
    for b in ("gradient", "denoise", "stars", "color", "astrometry"):
        c = ex.card(b)
        P(f"  card {b} after Fusion: why={getattr(c, 'reason', None)!r}")
    place("gradient")
    show_state("gradient after fusion?")
    if ex.w.bt_run.isEnabled():
        click_run()
        s2 = wait_run(120)
        dump_dialog("after gradient on fusion")
        P("  stage_end:", stage_runs(root))
    ex.close_dialogs()
    go_post()
    ex.shot("fdual_post")
    return s


def sc_two_flows_branch():
    """Two sessions: a flow each (the second Session dropped on the new-flow zone), the first with a branch at Stacking;
    run; shared parts computed once."""
    root = frames_project("twofl", nights=2, n=5)
    p = project2.load(root)
    p.workflows[0].sessions = [1]
    w2 = p.add_workflow("Flux 2", wfl.TEMPLATES["Classic stack"])
    w2.sessions = [2]
    p.set_blocks(w2.id, [])
    project2.save(p)
    ex.open(root)
    cpu()
    ex.click_block(blk("session", 0))
    place("registration", "stacking", "gradient")
    show_state("flow 1")
    s2 = [it for it in ex.blocks() if it.node.key == "session" and 2 in [q.pid for q in it.node.paths]]
    if s2:
        ex.click_block(s2[0])
        ex.double_click_card("registration")
        ex.settle(8)
        ex.double_click_card("stacking")
        ex.settle(8)
    show_state("flow 2")
    P("  workflows:", [(w.name, w.sessions, w.source, [n["name"] for n in (w.blocks or [])]) for w in ex.s.p.workflows])
    # the branch: a Gradient dropped under the Stacking of the first flow
    stk = next(it for it in ex.blocks() if it.node.key == "empilement" and 1 in [q.pid for q in it.node.paths])
    sx = ex.point_of(stk).x()
    ex.drag_palette_hover("gradient", ex.point_of(stk))
    ghosts = [(ex.ghost_center(g), g.toolTip()) for g in ex.ghosts()]
    P("  ghosts:", [(c.x(), c.y(), t[:60]) for c, t in ghosts])
    ex.drag_leave()
    br = [c for c, t in ghosts if "branch" in t.lower()]
    if br:
        target = min(br, key=lambda c: abs(c.x() - sx))
        rec = ex.drag_palette("gradient", target)
        P("  branch drop:", rec, "toast", toast())
    show_state("branched")
    P("  workflows:", [(w.name, w.sessions, [n["name"] for n in (w.blocks or [])]) for w in ex.s.p.workflows])
    ex.shot("twofl_graph")
    if ex.w.bt_run.isEnabled():
        click_run()
        ex.settle(3)
        dump_dialog("start")
        s = wait_run(300)
        dump_dialog("end")
        ex.shot("twofl_run_end", W.run_dialog())
        ends = stage_runs(root)
        P("  stage_end:", ends)
        from collections import Counter
        P("  computed twice:", [k for k, n in Counter(k for _s, k, st, re, _e in ends if st == "done" and not re).items()
                                if n > 1])
    ex.close_dialogs()
    go_post()
    ex.shot("twofl_post")


def sc_pause_repeated():
    """Image [Gradient 1, Gradient 2, Gradient 3]: review pause on Gradient 2; Run stops after it; Continue."""
    root = image_project("pauserep")
    ex.open(root)
    cpu()
    place("gradient", "gradient", "gradient")
    m = ex.right_click_block(blk("fond~2"), "Pause vérif" if W.LANG == "fr" else "Review pause")
    show_state("pause on G2")
    P("  breakpoints:", [(w.name, w.breakpoints) for w in ex.s.p.workflows])
    click_run()
    ex.settle(3)
    dump_dialog("start")
    wait_run()
    dump_dialog("paused")
    ex.shot("pauserep_paused", W.run_dialog())
    P("  stage_end:", stage_runs(root))
    d = W.run_dialog()
    if d is not None and d.b_continue.isVisible():
        ex.click(d.b_continue)
        ex.settle(4)
        wait_run()
        dump_dialog("after continue")
        P("  stage_end:", stage_runs(root))
    show_state("end")
    P("  log:", log_rows()[:6])


def sc_pause_repeated2():
    """Second look: frames [Registration, Stacking, Gradient 1, Gradient 2], pause on Gradient 2, Run (button and F5);
    then the pause moved to Gradient 1, Run."""
    root = frames_project("pauserep2")
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient", "gradient")
    ex.right_click_block(blk("fond~2"), "Pause vérif")
    P("  breakpoints:", [(w.name, w.breakpoints) for w in ex.s.p.workflows])
    click_run()
    ex.settle(4)
    dump_dialog("button, pause on G2")
    ex.shot("pauserep2_g2", W.run_dialog())
    ex.close_dialogs()
    ex.key(Qt.Key_F5, ex.w)
    ex.settle(4)
    dump_dialog("F5, pause on G2")
    ex.close_dialogs()
    ex.right_click_block(blk("fond~2"), "Pause vérif")
    ex.right_click_block(blk("fond"), "Pause vérif")
    P("  breakpoints:", [(w.name, w.breakpoints) for w in ex.s.p.workflows])
    click_run()
    ex.settle(4)
    dump_dialog("button, pause on G1")
    if ex.w.run.phase != "idle":
        wait_run()
        dump_dialog("G1 paused")


def sc_reattach():
    """A run in progress, the project closed (Home) and opened again: the run is followed again? its end shown?"""
    root = frames_project("reatt", n=10)
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient", "gradient")
    click_run()
    t0 = time.time()
    while ex.w.run.phase != "running" and time.time() - t0 < 30:
        ex.settle(3)
    ex.close_dialogs()
    ex.w.show_home()
    ex.settle(10)
    P("  home; popups:", ex.popups[-2:])
    ex.w.open_project(root)
    ex.settle(15)
    P("  reopened: phase", ex.w.run.phase, "| Run", repr(ex.w.bt_run.text()), "| status", repr(ex.w.status.text()))
    ex.shot("reatt_reopened")
    st = wait_run(200)
    dump_dialog("end after reopen")
    show_state("after")
    P("  stage_end:", stage_runs(root))
    P("  log:", log_rows()[:6])


def sc_undo_after_run():
    """Image [Gradient] computed; a second Gradient placed, then Ctrl+Z, Ctrl+Y: states and Run."""
    root = image_project("undorun")
    ex.open(root)
    cpu()
    place("gradient")
    click_run()
    wait_run()
    ex.close_dialogs()
    place("gradient")
    show_state("G2 placed")
    ex.undo()
    show_state("after Ctrl+Z")
    ex.redo()
    show_state("after Ctrl+Y")
    ex.click_block(blk("fond~2"))
    ex.key(Qt.Key_Delete)
    show_state("G2 deleted")
    ex.undo()
    show_state("delete undone")


def sc_fusion_stars_real():
    """Frames [Registration, Stacking, Gradient, Denoise, Stars]; a dual-band Fusion with its stars (palette mode):
    accepted? computed? Post."""
    root = frames_project("fstars")
    p = project2.load(root)
    w = p.workflows[0]
    p.set_blocks(w.id, [wfl.node(b) for b in ("registration", "stacking", "gradient", "denoise", "stars")])
    fu = p.add_workflow("Fusion", wfl.TEMPLATES["Classic stack"])
    fu.source = {"fusion": {"palette": "HOO", "stars": {"mode": "palette"},
                            "inputs": [{"pid": w.id, "role": "Ha", "channel": "R"},
                                       {"pid": w.id, "role": "OIII", "channel": "GB"}]}}
    p.set_blocks(fu.id, [wfl.node("stars")])
    project2.save(p)
    ex.open(root)
    cpu()
    show_state("opened")
    ex.shot("fstars_graph")
    if click_run():
        ex.settle(3)
        dump_dialog("start")
        wait_run(300)
        dump_dialog("end")
        ex.shot("fstars_run_end", W.run_dialog())
        P("  stage_end:", stage_runs(root))
    ex.close_dialogs()
    show_state("after")
    go_post()
    ex.shot("fstars_post")


def sc_pause_in_branch():
    """Image [Gradient] with a branch [Gradient, Gradient 2]; pause on the shared Gradient; Run, Continue."""
    root = image_project("pbr")
    ex.open(root)
    cpu()
    place("gradient")
    g = blk("fond")
    gx = ex.point_of(g).x()
    ex.drag_palette_hover("gradient", ex.point_of(g))
    ghosts = [(ex.ghost_center(x), x.toolTip()) for x in ex.ghosts()]
    ex.drag_leave()
    br = [c for c, t in ghosts if "branch" in t.lower()]
    if br:
        ex.drag_palette("gradient", min(br, key=lambda c: abs(c.x() - gx)))
    ex.click_block(blk("fond"))
    m = ex.right_click_block(blk("fond"), "Pause vérif")
    P("  menu:", [(a.get("text"), a.get("enabled"), [s.get("text") for s in a.get("sub") or []]) for a in m])
    P("  breakpoints:", [(w.name, w.breakpoints) for w in ex.s.p.workflows])
    show_state("pause on shared")
    click_run()
    ex.settle(3)
    dump_dialog("start")
    wait_run()
    dump_dialog("paused")
    ex.shot("pbr_paused", W.run_dialog())
    d = W.run_dialog()
    if d is not None and d.b_continue.isVisible():
        ex.click(d.b_continue)
        ex.settle(4)
        dump_dialog("continued")
        wait_run()
        dump_dialog("after continue")
    P("  stage_end:", stage_runs(root))
    show_state("end")


def sc_denoise2_panel():
    """Frames [Registration, Stacking, Denoise, Gradient, Denoise 2]: the panels of Denoise 1 and Denoise 2."""
    from PySide6.QtWidgets import QComboBox
    from explore2_engine_window import panel_texts
    root = frames_project("dn2")
    ex.open(root)
    place("registration", "stacking", "denoise", "gradient", "denoise")
    P("  blocks:", [it.node.key for it in ex.blocks()])
    for key in ("debruitage", "debruitage~2"):
        it = blk(key)
        if it is None:
            P("  no", key)
            continue
        ex.click_block(it)
        P(f"  {key} panel:", panel_texts()[:20])
        P(f"  {key} combos:", [(c.currentText(), c.isEnabled(), c.toolTip()) for c in
                               ex.page.panel.findChildren(QComboBox) if c.isVisibleTo(ex.page.panel)])
        ex.shot(f"dn2_{key.replace('~', '')}")


def sc_session_and_image():
    """A session flow and an image flow in one project; Run all; both layers in Post."""
    root = frames_project("sesimg")
    p = project2.load(root)
    w = p.add_workflow("Image", wfl.TEMPLATES["Classic stack"])
    w.source = {"path": str(W.linear_fits(W.TMP / "data" / "sesimg")), "stretched": False}
    p.set_blocks(w.id, [wfl.node("gradient")])
    p.set_blocks(p.workflows[0].id, [wfl.node("registration"), wfl.node("stacking"), wfl.node("gradient")])
    project2.save(p)
    ex.open(root)
    cpu()
    show_state("opened")
    click_run()
    ex.settle(3)
    dump_dialog("start")
    wait_run(200)
    dump_dialog("end")
    P("  stage_end:", stage_runs(root))
    ex.close_dialogs()
    go_post()
    ex.shot("sesimg_post")


def sc_astrometry_try():
    """Image (synthetic stars, no real sky) → Astrometry, Gradient, Color: accepted? what the run says (it cannot solve
    a synthetic field)."""
    root = image_project("astro")
    ex.open(root)
    cpu()
    place("astrometry", "gradient", "color")
    show_state("placed")
    click_run()
    ex.settle(3)
    dump_dialog("start")
    st = wait_run(180)
    dump_dialog("end")
    ex.shot("astro_run_end", W.run_dialog())
    P("  stage_end:", stage_runs(root))
    if ex.w.run.phase != "idle":
        ex.w.run.stop()
        wait_run(60)


def sc_edit_during_run():
    """Frames run in progress: a Gradient placed and the Stacking's setting changed meanwhile; then what the graph
    says when the run ends, and what the next Run computes."""
    root = frames_project("editrun", n=8)
    ex.open(root)
    cpu()
    place("registration", "stacking")
    click_run()
    t0 = time.time()
    while ex.w.run.phase != "running" and time.time() - t0 < 30:
        ex.settle(3)
    ex.close_dialogs()
    P("  running; Run button:", ex.w.bt_run.text())
    place("gradient")
    show_state("gradient placed during run")
    ex.click_block(blk("empilement"))
    from explore2_engine_window import panel_rows, set_setting
    r = next((r for r in panel_rows() if r.r.get("name") == "kappa"), None)
    if r is not None:
        set_setting("kappa", 4.0)
        from PySide6.QtWidgets import QPushButton
        rep = next((b for b in ex.page.panel.findChildren(QPushButton) if b.isVisible() and
                    b.text().startswith(("Replace", "Remplacer"))), None)
        if rep is not None:
            ex.click(rep)
            ex.settle(5)
    show_state("setting changed during run")
    ex.shot("editrun_during")
    wait_run(200)
    dump_dialog("end")
    ex.close_dialogs()
    show_state("after run")
    for it in ex.blocks():
        P(f"    {it.node.key}: state={getattr(it, 'state', None)!r}")
    ex.shot("editrun_after")
    click_run()
    ex.settle(3)
    dump_dialog("next run")
    wait_run(200)
    dump_dialog("next run end")
    P("  stage_end:", stage_runs(root))


SC = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("sc_")}

if __name__ == "__main__":
    name = W.ARGS[0]
    P(f"=== {name}")
    try:
        r = SC[name]()
        P("=== result", r)
    except Exception as e:  # noqa: BLE001
        P("!! SCENARIO EXCEPTION", type(e).__name__, e, traceback.format_exc()[-2500:])
    P("=== slot errors:", len(W.SLOT_ERRORS))
    for x in W.SLOT_ERRORS:
        P(x[-1500:])
    os._exit(0)
