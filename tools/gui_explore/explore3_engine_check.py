"""Engine shapes, round 3 (control): the round-2 points replayed in the real window (gestures; CPU runs on tiny data).

    python explore3_engine_check.py <scenario> [--en]
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore3_engine_window as W  # noqa: E402  (the Explorer, CPU runs allowed)
from explore3_engine_window import (P, TMP, blk, blocks_desc, click_run, cpu, dump_dialog, ex, frames_project,  # noqa: E402
                                    go_post, image_project, log_rows, panel_rows, panel_texts, place, show_state,
                                    stage_runs, toast, wait_run)
from explore_engine_common import bayer_frames  # noqa: E402

from PySide6.QtCore import Qt  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.project2 import Session  # noqa: E402

PAUSE = "Pause vérif" if W.LANG == "fr" else "Review pause"


def _proj(name, lights, **kw):
    root = ex.ws / name
    p = project2.create(root, name, Target(name="M 31"))
    p.sessions.append(Session(1, "Session 1", lights=[str(lights)], **kw))
    w = p.add_workflow("Workflow", wfl.TEMPLATES["Classic stack"])
    p.set_blocks(w.id, [])
    project2.save(p)
    return root


def _run_and_tell(root, tag, timeout=200):
    show_state(tag)
    ok = click_run()
    if not ok:
        ex.shot(f"c_{W.ARGS[0]}_refused")
        return "refused"
    ex.settle(3)
    dump_dialog("start")
    st = wait_run(timeout)
    dump_dialog("end")
    ex.shot(f"c_{W.ARGS[0]}_{st}", W.run_dialog())
    P("  stage_end:", stage_runs(root))
    P("  log:", log_rows()[:4])
    return st


def _f5_dialog(tag):
    ex.key(Qt.Key_F5, ex.w)
    ex.settle(5)
    dump_dialog(tag)
    d = W.run_dialog()
    if d is not None:
        P(f"  [{tag}] dialog Run: visible {d.b_run.isVisible()} enabled {d.b_run.isEnabled()} tip {d.b_run.toolTip()!r}")
    return d


# ------------------------------------------------------------------------------- N0 + N10: pause on a repeated block
def sc_pause_rep():
    """Frames [Registration, Stacking, Gradient 1, Gradient 2]: pause on Gradient 2, Run (button), stops after it?,
    Continue; then a pause on Gradient 1 (new project) and the run window's words."""
    root = frames_project("pauserep")
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient", "gradient")
    ex.right_click_block(blk("fond~2"), PAUSE)
    P("  breakpoints:", [(w.name, w.breakpoints) for w in ex.s.p.workflows], "toast", repr(toast()))
    click_run()
    ex.settle(4)
    dump_dialog("start, pause on G2")
    wait_run(240)
    dump_dialog("paused?")
    ex.shot("c_pauserep_g2_paused", W.run_dialog())
    P("  status:", repr(ex.w.status.text()))
    P("  stage_end:", stage_runs(root))
    d = W.run_dialog()
    if d is not None and d.b_continue.isVisible():
        P("  continue:", d.b_continue.text(), d.b_continue.isEnabled())
        ex.click(d.b_continue)
        ex.settle(5)
        wait_run(240)
        dump_dialog("after continue")
        ex.shot("c_pauserep_g2_continued", W.run_dialog())
        P("  stage_end after continue:", stage_runs(root))
    else:
        P("  NO CONTINUE")
    show_state("end")
    P("  log:", log_rows()[:8])


def sc_pause_g1():
    root = frames_project("pauseg1")
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient", "gradient")
    ex.right_click_block(blk("fond"), PAUSE)
    P("  breakpoints:", [(w.name, w.breakpoints) for w in ex.s.p.workflows], "toast", repr(toast()))
    click_run()
    ex.settle(4)
    dump_dialog("start, pause on G1")
    wait_run(240)
    dump_dialog("paused")
    P("  status:", repr(ex.w.status.text()))
    ex.shot("c_pauseg1_paused", W.run_dialog())
    P("  log:", log_rows()[:8])


# ------------------------------------------------------------------------------- N1: stretched + Stars/background -> Post
def sc_stretched_stars():
    root = image_project("strstars", stretched=True)
    ex.open(root)
    cpu()
    P("  cards at opening:", {b: getattr(ex.card(b), "reason", None) for b in ("gradient", "stars", "denoise")})
    ex.shot("c_cards_at_opening")
    place("stars")
    show_state("placed")
    st = _run_and_tell(root, "stretched stars")
    ex.close_dialogs()
    show_state("after run")
    W.go_post()
    ex.shot("c_stretched_stars_post")
    ex.click(ex.w.mode_btns[0])
    ex.settle(6)
    m = ex.right_click_block(blk("etoiles"), "Sans étoiles vers Post" if W.LANG == "fr" else "Starless to Post")
    P("  menu:", [(a.get("text"), a.get("enabled")) for a in m])
    ex.settle(6)
    show_state("starless sent")
    W.go_post()
    ex.shot("c_stretched_stars_post_after_send")
    return st


# ------------------------------------------------------------------------------- N2: no Registration
def sc_no_reg():
    root = frames_project("noreg", n=6, calib=True)
    ex.open(root)
    cpu()
    res = []
    for want in (["stacking", "gradient"], ["selection", "stacking", "gradient"]):
        ex.close_dialogs()
        for _ in range(8):
            xs = [it for it in ex.blocks() if it.node.key not in ("final", "sources", "session", "image", "post")]
            if not xs:
                break
            ex.click_block(max(xs, key=lambda b: b.node.col))
            ex.key(Qt.Key_Delete)
            ex.settle(4)
        reasons = {}
        for b in want:
            reasons[b] = getattr(ex.card(b), "reason", None)
            place(b)
        got = [n["name"] for n in (ex.s.p.workflows[0].blocks or [])]
        b = ex.w.bt_run
        line = {"want": want, "got": got, "reasons": reasons, "run": (b.text(), b.isEnabled(), b.toolTip())}
        if b.isEnabled():
            ex.click(b)
            line["status"] = wait_run(200)
            d = W.run_dialog()
            try:
                line["rows"] = [(r.name.text(), r.mark.state, r.toolTip()[:200]) for r in d.rows] if d else None
                line["texts"] = W.vis_texts(d)[:12] if d else None
            except RuntimeError:
                pass
            ex.shot(f"c_noreg_{'_'.join(want)}", d)
            line["ends"] = [x for x in stage_runs(root) if x[2] != "done"]
        else:
            ex.shot(f"c_noreg_{'_'.join(want)}_refused")
        P("  LINE", line)
        res.append(line)


# ------------------------------------------------------------------------------- N3: impossible data
def sc_one_frame():
    d = bayer_frames(TMP / "data" / "one" / "lights", n=1, seed=4)
    root = _proj("one", d)
    ex.open(root)
    cpu()
    place("registration", "stacking")
    return _run_and_tell(root, "1 frame")


def sc_unreadable():
    d = TMP / "data" / "bad" / "lights"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        (d / f"light_{i:03d}.fits").write_text("not a fits file\n" * 50)
    root = _proj("bad", d)
    ex.open(root)
    cpu()
    ex.click_block(blk("sources") or blk("session"))
    P("  session panel:", panel_texts()[:20])
    place("registration", "stacking")
    return _run_and_tell(root, "unreadable")


def sc_dark_size():
    lights = bayer_frames(TMP / "data" / "dsz" / "lights", n=5, seed=5)
    darks = bayer_frames(TMP / "data" / "dsz" / "darks", n=3, kind="dark", h=100, w=150, seed=6)
    root = _proj("dsz", lights, darks=[str(darks)])
    ex.open(root)
    cpu()
    place("calibration", "registration", "stacking")
    return _run_and_tell(root, "darks of another size")


def sc_astrometry():
    root = image_project("astro")
    ex.open(root)
    cpu()
    place("astrometry", "gradient")
    st = _run_and_tell(root, "astrometry synthetic", timeout=150)
    if ex.w.run.phase != "idle":
        ex.w.run.stop()
        wait_run(60)
    return st


# ------------------------------------------------------------------------------- N4: missing folder
def sc_missing_folder():
    root = ex.ws / "missing"
    p = project2.create(root, "missing", Target(name="M 31"))
    p.sessions.append(Session(1, "Session 1", lights=[str(TMP / "data" / "nowhere")]))
    w = p.add_workflow("Workflow", wfl.TEMPLATES["Classic stack"])
    p.set_blocks(w.id, [wfl.node("stacking")])
    project2.save(p)
    ex.open(root)
    cpu()
    show_state("opened")
    ex.click_block(blk("session") or blk("sources"))
    P("  session panel:", panel_texts()[:30])
    ex.shot("c_missing_panel")
    return _run_and_tell(root, "missing")


# ------------------------------------------------------------------------------- N5: method of Gradient 2 / Denoise 2
def sc_method2():
    from PySide6.QtWidgets import QComboBox, QPushButton
    root = image_project("method")
    ex.open(root)
    place("gradient", "gradient")
    for key in ("fond", "fond~2"):
        ex.click_block(blk(key))
        P(f"  {key} panel:", panel_texts()[:30])
        P(f"  {key} combos:", [(c.currentText(), c.isEnabled(), c.toolTip()) for c in
                               ex.page.panel.findChildren(QComboBox) if c.isVisibleTo(ex.page.panel)])
        P(f"  {key} buttons:", [b.text() for b in ex.page.panel.findChildren(QPushButton) if b.isVisibleTo(ex.page.panel)])
        ex.shot(f"c_method_{key.replace('~', '')}")
    root2 = frames_project("dn2")
    ex.open(root2)
    place("registration", "stacking", "denoise", "gradient", "denoise")
    for key in ("debruitage", "debruitage~2"):
        it = blk(key)
        if it is None:
            P("  no", key, [b.node.key for b in ex.blocks()])
            continue
        ex.click_block(it)
        P(f"  {key} panel:", panel_texts()[:20])
        P(f"  {key} combos:", [(c.currentText(), c.isEnabled(), c.toolTip()) for c in
                               ex.page.panel.findChildren(QComboBox) if c.isVisibleTo(ex.page.panel)])
        ex.shot(f"c_method_{key.replace('~', '')}")


def sc_method2_run():
    """Gradient 2's method changed (cards or list) to another one, then a real run: Gradient 2 uses it?"""
    from PySide6.QtWidgets import QAbstractButton, QComboBox
    root = image_project("method2run")
    ex.open(root)
    cpu()
    place("gradient", "gradient")
    ex.click_block(blk("fond~2"))
    pn = ex.page.panel
    combos = [c for c in pn.findChildren(QComboBox) if c.isVisibleTo(pn)]
    rows = {r.r.get("name"): (r.r.get("value"), r.r.get("choices"), r.editor.isEnabled()) for r in panel_rows()}
    P("  g2 rows:", rows)
    btns = [(b.text(), b.isCheckable(), b.isChecked()) for b in pn.findChildren(QAbstractButton) if b.isVisibleTo(pn)]
    P("  g2 buttons:", btns)
    # a method card other than the checked one
    cards = [b for b in pn.findChildren(QAbstractButton) if b.isVisibleTo(pn) and b.isCheckable() and not b.isChecked()
             and b.text() and any(k in b.text() for k in ("RBF", "Auto", "Polyn"))]
    if cards:
        P("  clicking card:", cards[0].text())
        ex.click(cards[0])
        ex.settle(8)
    P("  params:", [(w.name, w.params) for w in ex.s.p.workflows], "toast", repr(toast()))
    ex.shot("c_method2_set")
    click_run()
    wait_run(200)
    dump_dialog("end")
    P("  stage_end:", stage_runs(root))


# ------------------------------------------------------------------------------- N6 + N13: branch, last block menu
def sc_branch_image():
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
    ex.shot("c_brimg_graph")
    for key in [it.node.key for it in ex.blocks()]:
        it = blk(key)
        m = ex.right_click_block(it)
        P(f"  menu {key}:", [(a.get("text"), a.get("enabled"), [s.get("text") for s in a.get("sub") or []]) for a in m])
        ex.settle(3)
    click_run()
    ex.settle(3)
    dump_dialog("start")
    wait_run()
    dump_dialog("end")
    ex.shot("c_brimg_run_end", W.run_dialog())


# ------------------------------------------------------------------------------- N7: drizzle before/after
def sc_drizzle():
    root = frames_project("driz")
    ex.open(root)
    cpu()
    place("registration", "stacking")
    stk = blk("empilement")
    P("  before:", stk.title, "|", stk.summary)
    ex.click_block(stk)
    P("  panel before:", [(r.r.get("name"), r.r.get("value")) for r in panel_rows()])
    click_run()
    wait_run(200)
    ex.close_dialogs()
    stk = blk("empilement")
    P("  after:", stk.title, "|", stk.summary, "| tip", stk.toolTip()[:200])
    ex.click_block(stk)
    P("  panel after:", [(r.r.get("name"), r.r.get("value")) for r in panel_rows()])
    ex.shot("c_drizzle_after")
    show_state("after run")


# ------------------------------------------------------------------------------- N8: off / on words
def sc_offon():
    root = image_project("offon")
    ex.open(root)
    place("gradient", "gradient")
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    ex.settle(4)
    P("  toast off:", repr(toast()))
    g = blk("fond")
    P("  block:", g.title, "|", g.summary, "| tip", g.toolTip()[:200])
    c = ex.card("gradient")
    P("  card gradient:", getattr(c, "reason", None))
    ex.shot("c_offon_off")
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    ex.settle(4)
    P("  toast on:", repr(toast()))
    root2 = frames_project("offon2")
    ex.open(root2)
    place("registration", "stacking")
    ex.click_block(blk("alignement"))
    ex.key(Qt.Key_D)
    ex.settle(4)
    P("  toast reg off:", repr(toast()))
    P("  card registration:", getattr(ex.card("registration"), "reason", None))
    P("  log:", log_rows()[:4])


# ------------------------------------------------------------------------------- N9: fusion, a channel without file
def sc_fusion_nofile():
    root = ex.ws / "fusnf"
    p = project2.create(root, "fusnf", Target(name="M 31"))
    a = p.add_workflow("Ha", wfl.TEMPLATES["Classic stack"])
    a.source = {"path": str(W.linear_fits(TMP / "data" / "fusnf")), "stretched": False}
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
    d = _f5_dialog("F5")
    ex.shot("c_fusion_nofile_f5", d)


# ------------------------------------------------------------------------------- N11: Selection after Registration
def sc_selection_row():
    root = frames_project("selrow")
    ex.open(root)
    cpu()
    place("calibration", "registration", "selection", "stacking")
    click_run()
    ex.settle(3)
    dump_dialog("start")
    wait_run()
    dump_dialog("end")
    ex.shot("c_selrow_run_end", W.run_dialog())


# ------------------------------------------------------------------------------- N12: palette at opening
def sc_cards_opening():
    root = image_project("cardsopen")
    p = project2.load(root)
    p.set_blocks(p.workflows[0].id, [])
    project2.save(p)
    ex.open(root)
    P("  selection:", [it.node.key for it in ex.blocks() if it.isSelected()])
    P("  cards:", {b: getattr(ex.card(b), "reason", None) for b in ("gradient", "denoise", "stars", "astrometry",
                                                                       "color")})
    ex.shot("c_cards_opening")
    ex.double_click_card("gradient")
    ex.settle(8)
    P("  after dbl-click gradient:", [it.node.key for it in ex.blocks()], repr(toast()))


# ------------------------------------------------------------------------------- R4: cache key (decision 10)
def sc_offrun():
    root = image_project("offrun")
    ex.open(root)
    cpu()
    place("gradient", "gradient")
    click_run()
    wait_run()
    ex.close_dialogs()
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    show_state("G1 off")
    ex.shot("c_offrun_g1_off")
    click_run()
    ex.settle(3)
    wait_run()
    dump_dialog("rerun end")
    P("  stage_end rerun:", stage_runs(root))


# ------------------------------------------------------------------------------- run window following an Undo
def sc_undo_window():
    """Frames [Registration, Stacking, Gradient]; Shift+F5 on Gradient; Ctrl+Z with the window open; its Run clicked:
    what is really computed?"""
    root = frames_project("undowin")
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient")
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_F5, ex.w, Qt.ShiftModifier)
    ex.settle(5)
    dump_dialog("before undo")
    ex.shot("c_undowin_before", W.run_dialog())
    ex.undo()
    ex.settle(6)
    P("  blocks after undo:", [it.node.key for it in ex.blocks()])
    dump_dialog("after undo")
    ex.shot("c_undowin_after", W.run_dialog())
    d = W.run_dialog()
    if d is not None and d.isVisible() and d.b_run.isVisible() and d.b_run.isEnabled():
        ex.click(d.b_run)
        ex.settle(5)
        wait_run(200)
        dump_dialog("ran")
        ex.shot("c_undowin_ran", W.run_dialog())
    P("  stage_end:", stage_runs(root))
    show_state("end")


def sc_undo_window_running():
    """A run started from the window, then Ctrl+Z (a block removed) while it runs: the window and the graph after."""
    root = frames_project("undorun", n=8)
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient")
    click_run()
    t0 = time.time()
    while ex.w.run.phase != "running" and time.time() - t0 < 30:
        ex.settle(3)
    dump_dialog("running")
    ex.undo()
    ex.settle(5)
    P("  blocks after undo (running):", [it.node.key for it in ex.blocks()], repr(toast()))
    dump_dialog("after undo running")
    ex.shot("c_undorun_during", W.run_dialog())
    wait_run(200)
    dump_dialog("end")
    ex.shot("c_undorun_end", W.run_dialog())
    P("  stage_end:", stage_runs(root))
    ex.close_dialogs()
    show_state("after")


def sc_probe_misc():
    """No run: the Stacking panel before any run (drizzle); a block switched off then on (toast, Journal)."""
    root = frames_project("probemisc")
    ex.open(root)
    place("registration", "stacking", "gradient")
    ex.click_block(blk("empilement"))
    P("  stacking panel:", panel_texts()[:40])
    ex.shot("c_drizzle_panel_before")
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    ex.settle(4)
    P("  toast off:", repr(toast()))
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    ex.settle(1)
    P("  toast on (at once):", repr(toast()))
    ex.settle(6)
    P("  toast on (later):", repr(toast()), "| status", repr(ex.w.status.text()))
    ex.shot("c_reenable_toast")
    P("  log:", log_rows()[:5])
    m = ex.right_click_block(blk("fond"))
    P("  menu last block:", [(a.get("text"), a.get("enabled"), [s.get("text") for s in a.get("sub") or []]) for a in m])
    ex.settle(3)
    m = ex.right_click_block(blk("empilement"))
    P("  menu stacking:", [(a.get("text"), a.get("enabled"), [s.get("text") for s in a.get("sub") or []]) for a in m])


def sc_last_state():
    """Frames [Registration, Stacking, Gradient] computed; the Gradient deleted: what the Stacking (now last) says, what
    Run plans and computes."""
    root = frames_project("laststate")
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient")
    click_run()
    wait_run(200)
    ex.close_dialogs()
    show_state("computed")
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_Delete)
    ex.settle(6)
    show_state("gradient deleted")
    ex.click_block(blk("empilement"))
    P("  stacking panel head:", panel_texts()[:4])
    ex.shot("c_laststate_after_delete")
    d = _f5_dialog("F5 after delete")
    ex.shot("c_laststate_f5", d)
    if d is not None and d.b_run.isVisible() and d.b_run.isEnabled():
        ex.click(d.b_run)
        ex.settle(4)
        wait_run(200)
        dump_dialog("end")
        ex.shot("c_laststate_ran", W.run_dialog())
    P("  stage_end:", stage_runs(root))
    ex.close_dialogs()
    show_state("after")


def sc_last_state2():
    """Second look: Image [Gradient 1, Gradient 2] computed; Gradient 2 deleted: Gradient (now last) in the graph and in
    the panel; an Astrometry's summary before any run."""
    root = image_project("laststate2")
    ex.open(root)
    cpu()
    place("gradient", "gradient")
    click_run()
    wait_run(200)
    ex.close_dialogs()
    ex.click_block(blk("fond~2"))
    ex.key(Qt.Key_Delete)
    ex.settle(6)
    show_state("G2 deleted")
    ex.click_block(blk("fond"))
    P("  gradient panel head:", panel_texts()[:4])
    ex.shot("c_laststate2_after_delete")
    ex.key(Qt.Key_I, ex.w, Qt.NoModifier)
    ex.click(ex.w.mode_btns[1])
    ex.settle(8)
    P("  inspection texts:", W.vis_texts(ex.w)[:30])
    ex.shot("c_laststate2_inspection")
    ex.click(ex.w.mode_btns[0])
    ex.settle(4)
    root2 = frames_project("astrosum")
    ex.open(root2)
    place("registration", "stacking", "astrometry")
    a = blk("astrometrie")
    P("  astrometry before any run:", a.title, "|", a.summary, "| tip", a.toolTip()[:200])
    ex.click_block(a)
    P("  astrometry panel head:", panel_texts()[:8])
    ex.shot("c_astrometry_summary")


SC = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("sc_")}

if __name__ == "__main__":
    name = W.ARGS[0]
    P(f"=== {name}")
    try:
        P("=== result", SC[name]())
    except Exception as e:  # noqa: BLE001
        P("!! SCENARIO EXCEPTION", type(e).__name__, e, traceback.format_exc()[-2500:])
    P("=== slot errors:", len(W.SLOT_ERRORS))
    for x in W.SLOT_ERRORS:
        P(x[-1500:])
    os._exit(0)
