"""Engine shapes, round 2: the round-1 findings replayed in the real window (gestures; CPU runs on tiny data).

    python explore2_engine_recheck.py <scenario> [--en]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_engine_window as W  # noqa: E402  (the Explorer, CPU runs allowed)
from explore2_engine_window import (P, blk, blocks_desc, click_run, cpu, dump_dialog, ex, frames_project,  # noqa: E402
                                    image_project, panel_rows, panel_texts, place, set_setting, show_state,
                                    stage_runs, toast, wait_run)

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402


def card_state(b):
    c = ex.card(b)
    if c is None:
        return None
    return {"enabled": c.isEnabled(), "tip": c.toolTip(), "greyed": getattr(c, "greyed", None),
            "why": getattr(c, "why", None) or getattr(c, "reason", None)}


def sc_switch_off_redrop():
    root = frames_project("swoff")
    ex.open(root)
    place("calibration", "registration", "selection", "stacking")
    for key, b in (("alignement", "registration"), ("calibration", "calibration"), ("selection", "selection")):
        it = blk(key)
        ex.click_block(it)
        ex.key(W.Qt.Key_D)
        P(f"  {key} off? {[n.get('off') for n in ex.s.p.workflows[0].blocks if n['block'] == b]}")
        P(f"  card {b}:", card_state(b))
        rec = ex.drag_palette(b, ex.point_of(blk("empilement")))
        P("  drop:", rec, "toast:", repr(toast()))
        ex.double_click_card(b)
        ex.settle(6)
        P("  dbl-click toast:", repr(toast()), "blocks:", [n["name"] for n in ex.s.p.workflows[0].blocks])
        ex.shot(f"recheck_swoff_{b}")
        ex.click_block(blk(key))
        ex.key(W.Qt.Key_D)
    show_state("end")


def sc_stars_reason_image():
    root = image_project("starsimg")
    ex.open(root)
    place("gradient")
    for b in ("stars", "denoise"):
        P(f"  card {b}:", card_state(b))
    c = ex.card("stars")
    ex.hover(c, c.rect().center())
    ex.shot("recheck_stars_reason_image")


def sc_settings_after_removal():
    root = image_project("setrm")
    ex.open(root)
    place("gradient", "gradient")
    g2 = blk("fond~2") or blk("fond", 1)
    ex.click_block(g2)
    rows = [(r.r.get("name"), r.r.get("type"), r.r.get("value")) for r in panel_rows()]
    P("  g2 rows:", rows)
    num = next((r for r in panel_rows() if r.r.get("type") == "int" and r.editor.isEnabled()), None) or \
        next((r for r in panel_rows() if r.r.get("type") == "float" and r.editor.isEnabled()), None)
    name, v0 = num.r["name"], num.r["value"]
    nv = int(v0) + 2 if num.r["type"] == "int" else round(float(v0) * 2 + 0.1, 3)
    set_setting(name, nv)
    P("  params:", ex.s.p.workflows[0].params)
    ex.click_block(blk("fond~2") or blk("fond", 1))
    ex.key(W.Qt.Key_Delete)
    P("  after delete:", [n["name"] for n in ex.s.p.workflows[0].blocks], "params:", ex.s.p.workflows[0].params)
    place("gradient")
    g2 = blk("fond~2") or blk("fond", 1)
    ex.click_block(g2)
    after = {r.r.get("name"): r.r.get("value") for r in panel_rows()}
    P(f"  new Gradient 2 {name} = {after.get(name)} (was set to {nv}, default {v0})")
    ex.shot("recheck_settings_after_removal")


def sc_fusion_nofile():
    """A Fusion whose channel is an Image block without its file: Run refused before the run, with the reason?"""
    root = ex.ws / "fusnf"
    p = project2.create(root, "fusnf", W.Target(name="M 31"))
    a = p.add_workflow("Ha", wfl.TEMPLATES["Classic stack"])
    a.source = {"path": str(W.linear_fits(W.TMP / "data" / "fusnf")), "stretched": False}
    p.set_blocks(a.id, [wfl.node("gradient")])
    b = p.add_workflow("OIII", wfl.TEMPLATES["Classic stack"])
    b.source = {"path": "", "stretched": False}
    p.set_blocks(b.id, [])
    fu = p.add_workflow("Fusion", wfl.TEMPLATES["Classic stack"])
    fu.source = {"fusion": {"palette": "HOO", "inputs": [{"pid": a.id, "role": "Ha", "channel": "L"},
                                                          {"pid": b.id, "role": "OIII", "channel": "L"}]}}
    p.set_blocks(fu.id, [])
    project2.save(p)
    ex.open(root)
    cpu()
    show_state("opened")
    ok = click_run()
    ex.settle(4)
    dump_dialog("run")
    ex.shot("recheck_fusion_nofile")
    if ok:
        wait_run()
        dump_dialog("end")
        P("  stage_end:", stage_runs(root))
    f5_and_run(root)


def f5_and_run(root):
    """F5 (Run all from the keyboard): the plan dialog; its Run button clicked."""
    ex.key(W.Qt.Key_F5, ex.w)
    ex.settle(4)
    dump_dialog("F5")
    d = W.run_dialog()
    if d is None:
        return
    P(f"  F5 dialog Run: visible {d.b_run.isVisible()} enabled {d.b_run.isEnabled()} tip {d.b_run.toolTip()!r}")
    ex.shot(f"{W.ARGS[0]}_f5_dialog", d)
    if d.b_run.isVisible() and d.b_run.isEnabled():
        ex.click(d.b_run)
        ex.settle(10)
        P("  after click: phase", ex.w.run.phase, "| status", repr(ex.w.status.text()), "| toast", repr(toast()))
        dump_dialog("after F5 Run click")
        if ex.w.run.phase != "idle":
            wait_run()
            dump_dialog("F5 end")
            P("  stage_end:", stage_runs(root))


def sc_fusion_noastro():
    """Two image flows without Astrometry combined in a Fusion; and a fusion with stars whose channel has no Split."""
    root = ex.ws / "fusna"
    p = project2.create(root, "fusna", W.Target(name="M 31"))
    d = W.TMP / "data" / "fusna"
    a = p.add_workflow("Ha", wfl.TEMPLATES["Classic stack"])
    a.source = {"path": str(W.linear_fits(d, "ha.fits", seed=1)), "stretched": False}
    p.set_blocks(a.id, [wfl.node("gradient")])
    b = p.add_workflow("OIII", wfl.TEMPLATES["Classic stack"])
    b.source = {"path": str(W.linear_fits(d, "oiii.fits", seed=2)), "stretched": False}
    p.set_blocks(b.id, [wfl.node("gradient")])
    fu = p.add_workflow("Fusion", wfl.TEMPLATES["Classic stack"])
    fu.source = {"fusion": {"palette": "HOO", "inputs": [{"pid": a.id, "role": "Ha", "channel": "L"},
                                                          {"pid": b.id, "role": "OIII", "channel": "L"}]}}
    p.set_blocks(fu.id, [])
    project2.save(p)
    ex.open(root)
    cpu()
    show_state("opened")
    ok = click_run()
    ex.settle(4)
    dump_dialog("run")
    ex.shot("recheck_fusion_noastro")
    if ok:
        wait_run()
        dump_dialog("end")
    f5_and_run(root)


def sc_fusion_stars():
    root = ex.ws / "fusst"
    p = project2.create(root, "fusst", W.Target(name="M 31"))
    a = p.add_workflow("Dual", wfl.TEMPLATES["Classic stack"])
    a.source = {"path": str(W.linear_fits(W.TMP / "data" / "fusst")), "stretched": False}
    p.set_blocks(a.id, [wfl.node("gradient")])
    fu = p.add_workflow("Fusion", wfl.TEMPLATES["Classic stack"])
    fu.source = {"fusion": {"palette": "HOO", "stars": {"mode": "palette"},
                            "inputs": [{"pid": a.id, "role": "Ha", "channel": "R"},
                                       {"pid": a.id, "role": "OIII", "channel": "GB"}]}}
    p.set_blocks(fu.id, [wfl.node("stars")])
    project2.save(p)
    ex.open(root)
    cpu()
    show_state("opened")
    ok = click_run()
    ex.settle(4)
    dump_dialog("run")
    ex.shot("recheck_fusion_stars")
    if ok:
        wait_run()
        dump_dialog("end")
        P("  stage_end:", stage_runs(root))


def sc_missing_folder():
    root = ex.ws / "missing"
    p = project2.create(root, "missing", W.Target(name="M 31"))
    p.sessions.append(W.Session(1, "Session 1", lights=[str(W.TMP / "data" / "nowhere")]))
    w = p.add_workflow("Workflow", wfl.TEMPLATES["Classic stack"])
    p.set_blocks(w.id, [wfl.node("stacking")])
    project2.save(p)
    ex.open(root)
    cpu()
    show_state("opened")
    ex.click_block(blk("session") or blk("sources"))
    P("  session panel:", panel_texts()[:30])
    ok = click_run()
    ex.settle(4)
    dump_dialog("run")
    if ok:
        wait_run()
        dump_dialog("end")
        ex.shot("recheck_missing_folder", W.run_dialog())
        P("  stage_end:", stage_runs(root))
        P("  log:", W.log_rows()[:6])


def sc_denoise_small():
    root = frames_project("dnsmall")
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient", "denoise")
    show_state("placed")
    click_run()
    wait_run(240)
    dump_dialog("end")
    ex.shot("recheck_denoise_small", W.run_dialog())
    P("  stage_end:", stage_runs(root))


def sc_method_probe():
    """The Method of a Gradient (and of Gradient 2) in the panel: can the user change it?"""
    root = image_project("method")
    ex.open(root)
    place("gradient", "gradient")
    for key in ("fond", "fond~2"):
        ex.click_block(blk(key))
        for r in panel_rows():
            if r.r.get("name") == "method":
                e = r.editor
                P(f"  {key} method: enabled {e.isEnabled()} tip {e.toolTip()!r} locked {r.r.get('locked')} "
                  f"kind {r.r.get('kind')} choices {r.r.get('choices')}")
        m = ex.right_click_block(blk(key))
        P(f"  {key} menu:", [(a.get("text"), a.get("enabled"), [s.get("text") for s in a.get("sub") or []]) for a in m])
    ex.shot("method_probe")


def sc_method_g1():
    """Gradient alone, then Gradient 1 when a second is placed: its panel's Method (cards or a list), changeable?"""
    from PySide6.QtWidgets import QComboBox
    root = image_project("methodg1")
    ex.open(root)
    place("gradient")
    ex.click_block(blk("fond"))
    P("  G alone panel:", panel_texts()[:25])
    P("  G alone combos:", [(c.currentText(), c.isEnabled(), c.toolTip()) for c in ex.page.panel.findChildren(QComboBox)
                             if c.isVisibleTo(ex.page.panel)])
    ex.shot("method_g_alone")
    place("gradient")
    for key in ("fond", "fond~2"):
        ex.click_block(blk(key))
        P(f"  {key} panel:", panel_texts()[:25])
        P(f"  {key} combos:", [(c.currentText(), c.isEnabled(), c.toolTip()) for c in
                               ex.page.panel.findChildren(QComboBox) if c.isVisibleTo(ex.page.panel)])
        ex.shot(f"method_{key.replace('~', '')}")


def sc_selection_row():
    """Calibration, Registration, Selection, Stacking on a fresh project: is Selection a row of the run window?"""
    root = frames_project("selrow")
    ex.open(root)
    cpu()
    place("calibration", "registration", "selection", "stacking")
    click_run()
    ex.settle(3)
    dump_dialog("start")
    wait_run()
    dump_dialog("end")
    ex.shot("selrow_run_end", W.run_dialog())


def sc_branch_image():
    """Image [Gradient]: a branch under the Gradient (a Gradient dropped on the branch place): what is drawn, the run
    window's rows."""
    root = image_project("brimg")
    ex.open(root)
    cpu()
    place("gradient")
    g = blk("fond")
    gx = ex.point_of(g).x()
    ex.drag_palette_hover("gradient", ex.point_of(g))
    ghosts = [(ex.ghost_center(x), x.toolTip()) for x in ex.ghosts()]
    P("  ghosts:", [(c.x(), c.y(), t[:60]) for c, t in ghosts])
    ex.drag_leave()
    br = [c for c, t in ghosts if "branch" in t.lower()]
    if br:
        rec = ex.drag_palette("gradient", min(br, key=lambda c: abs(c.x() - gx)))
        P("  branch drop:", rec, "toast", toast())
    show_state("branched")
    P("  workflows:", [(w.name, [n["name"] for n in (w.blocks or [])]) for w in ex.s.p.workflows])
    ex.shot("brimg_graph")
    s = blk("post")
    if s is not None:
        m = ex.right_click_block(s)
        P("  Sortie menu:", [(a.get("text"), a.get("enabled")) for a in m])
    click_run()
    ex.settle(3)
    dump_dialog("start")
    wait_run()
    dump_dialog("end")
    ex.shot("brimg_run_end", W.run_dialog())


SC = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("sc_")}

if __name__ == "__main__":
    import os
    import traceback
    name = W.ARGS[0]
    P(f"=== {name}")
    try:
        SC[name]()
    except Exception as e:  # noqa: BLE001
        P("!! SCENARIO EXCEPTION", type(e).__name__, e, traceback.format_exc()[-2500:])
    P("=== slot errors:", len(W.SLOT_ERRORS))
    for x in W.SLOT_ERRORS:
        P(x[-1500:])
    os._exit(0)
