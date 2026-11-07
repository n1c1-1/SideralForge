"""run_status round 2: targeted probes (prints and screenshots only). python explore2_run_probe.py p11 p12 ..."""
from __future__ import annotations

from explore2_run_recheck import (  # noqa: F401
    EXC, P, QPushButton, Qt, back, blk, dlg, dump, ex, f5, fresh, inspect, obs, place, refs, run_button, run_checks,
    shift_f5, shot, texts)


def inspect_any(ref, pid=None):
    """Inspect a block; the Run-to-here buttons of the page."""
    ex.close_dialogs()
    ex.double_click_block(blk(ref, pid))
    ex.settle(15)
    page = ex.w.tune.pages.currentWidget()
    btns = [b for b in page.findChildren(QPushButton) if b.isVisible() and b.text().startswith("Jusqu")]
    return page, btns


def p11_session_calibration_state():
    fresh("two_sessions")
    place("calibration", "registration", "stacking")
    for r in ("calibration@1", "calibration@2", "alignement", "empilement"):
        it = blk(r)
        obs("p11", **{r: it.toolTip() if it else None})
    page, btns = inspect_any("calibration@1")
    obs("p11", page=texts(page)[:8], strip=[x for x in texts(ex.w.tune.strip) if x != "›"],
        shot=shot("p11_inspect_cal1"))
    back()
    page, btns = inspect_any("alignement")
    obs("p11", page_align=texts(page)[:5])
    back()


def p12_pause_session_calibration():
    fresh("two_sessions")
    place("calibration", "registration", "stacking")
    it = blk("calibration@1")
    ex.right_click_block(it, "Pause vérif")
    ex.settle(8)
    it = blk("calibration@1")
    obs("p12", bp=[w.breakpoints for w in ex.s.p.workflows], mark=getattr(it, "bp", "?"), tip=it.toolTip(),
        menu_again=[a.get("text") for a in ex.right_click_block(it)], status=ex.w.status.text(),
        has=ex.s.has_breakpoint(it.node.paths[0], it.node.key) if hasattr(ex.s, "has_breakpoint") else "?")
    shot("p12_after_pause_cal1")
    d = shift_f5("empilement")
    obs("p12", here=dump(d), pause_note=d.pause_note.text() if d else None, shot=shot("p12_here_stack", d or ex.w))
    ex.close_dialogs()
    # the same on Alignement (a block of the workflow) for comparison
    ex.right_click_block(blk("alignement"), "Pause vérif")
    ex.settle(8)
    d = shift_f5("empilement")
    obs("p12", bp_align=[w.breakpoints for w in ex.s.p.workflows], here_align=dump(d),
        pause_note=d.pause_note.text() if d else None)
    ex.close_dialogs()
    ex.undo()
    obs("p12", after_undo=[w.breakpoints for w in ex.s.p.workflows], status=ex.w.status.text())
    ex.redo()
    obs("p12", after_redo=[w.breakpoints for w in ex.s.p.workflows], status=ex.w.status.text())


def p17_image_without_file():
    fresh("session")
    place("stacking")
    z = ex.new_flow_zone()
    ex.drag_palette("image", ex.ghost_center(z))
    ex.settle(15)
    w = ex.w
    obs("p17", status=w.status.text(), run=(w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip()),
        paths=[(q.name, q.source) for q in ex.s.paths], refs=refs(),
        image_tip=(blk("image").toolTip() if blk("image") else None))
    d = shift_f5("empilement")
    obs("p17", here=dump(d))
    ex.close_dialogs()
    d = run_button()
    obs("p17", run_all=dump(d), shot=shot("p17_run_all", d or ex.w))


def p25_jpg_plan():
    fresh("jpg_mistake")
    place("stacking")
    d = shift_f5("empilement")
    obs("p25", here=dump(d), shot=shot("p25_jpg_here", d or ex.w))
    ex.close_dialogs()
    w = ex.w
    obs("p25", run=(w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip()), status=w.status.text())
    d = run_button()
    obs("p25", run_all=dump(d))


def p40_fusion_entry_points():
    fresh("mono")
    place("fusion")
    w = ex.w
    fu = blk("fusion")
    obs("p40", red=getattr(fu, "problem", None) or getattr(fu, "error", None), fusion_tip=fu.toolTip(),
        a_run=(w.a_run.isEnabled(), w.a_run.toolTip()), run=(w.bt_run.text(), w.bt_run.isEnabled(),
                                                             w.bt_run.toolTip()), status=w.status.text())
    shot("p40_graph_fusion")
    # Shift+F5 on the Fusion
    d = shift_f5("fusion")
    obs("p40", here_fusion=dump(d), shot=shot("p40_here_fusion", d or ex.w))
    if d is not None and d.b_run.isEnabled():
        ex.click(d.b_run)
        ex.settle(20)
        obs("p40", clicked_here_fusion=dump(dlg()))
    ex.close_dialogs()
    # Shift+F5 on Ha's Stacking (the plan does not hold the Fusion)
    ha = ex.s.paths[0].pid
    d = shift_f5("empilement", ha)
    obs("p40", here_ha_stack=dump(d), shot=shot("p40_here_ha_stack", d or ex.w))
    ex.close_dialogs()
    # Inspect Fusion: its Run to here
    page, btns = inspect_any("fusion")
    obs("p40", inspect_fusion=[(b.text(), b.toolTip(), b.isEnabled()) for b in btns], page=texts(page)[:10],
        shot=shot("p40_inspect_fusion"))
    if btns and btns[0].isEnabled():
        ex.click(btns[0])
        ex.settle(15)
        d = dlg()
        obs("p40", inspect_fusion_click=dump(d))
        if d is not None and d.b_run.isEnabled():
            ex.click(d.b_run)
            ex.settle(20)
            obs("p40", inspect_fusion_click_run=dump(dlg()))
        ex.close_dialogs()
    back()
    # the menu Run > Run all (the menu item)
    w.a_run.trigger()
    ex.settle(20)
    obs("p40", menu_run_all=dump(dlg()))
    ex.close_dialogs()


def p41_fusion_fixed_then_undo():
    fresh("mono")
    place("fusion")
    w = ex.w
    ha, oiii = ex.s.paths[0].pid, ex.s.paths[1].pid
    for pid in (ha, oiii):
        ex.click_block(blk("empilement", pid))
        place("astrometry")
        obs("p41", **{f"after_astro_{pid}": (w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip(),
                                              w.status.text(), blk("fusion").toolTip())})
    ex.undo()
    obs("p41", after_undo=(w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip(), blk("fusion").toolTip()),
        refs=refs())
    ex.redo()
    obs("p41", after_redo=(w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip(), blk("fusion").toolTip()),
        refs=refs())
    shot("p41_fixed")


if __name__ == "__main__":
    run_checks(dict(globals()), prefix="p", dump_name="probe_dump.json")
