"""run_status round 3 (control), part 2: Fusion points (the Fusion is now greyed until Astrometry; broken by a
session's filter change), and suspicions met in part 1.

    python explore3_run_check2.py [f01 ...]
"""
from __future__ import annotations

from explore2_run_recheck import (  # noqa: F401
    EXC, P, QPushButton, Qt, back, blk, dlg, dump, ex, f5, fresh, inspect, obs, place, refs, run_button, run_checks,
    shift_f5, texts)
from explore2_run_explore import run_menu
from PySide6.QtWidgets import QComboBox


def shot(name, widget=None):
    return ex.shot(f"k{name}", widget)


def topbtn():
    w = ex.w
    return (w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip(), w.status.text())


def card(b):
    c = ex.card(b)
    return None if c is None else (c.isEnabled(), c.toolTip(), getattr(c, "why", None) or getattr(c, "reason", None))


def mono_fusion():
    """Mono Ha/OIII, Astrometry in both, Fusion placed (decision 2)."""
    fresh("mono")
    obs("f", card_before=card("fusion"))
    ha, oiii = ex.s.paths[0].pid, ex.s.paths[1].pid
    for pid in (ha, oiii):
        ex.click_block(blk("empilement", pid))
        place("astrometry")
    ex.click_block(blk("astrometrie", ha))
    place("fusion")
    obs("f", refs=refs(), top=topbtn())
    return ha, oiii


def set_filter(sref, value):
    ex.click_block(blk(sref))
    ex.settle(8)
    cbs = [c for c in ex.page.panel.findChildren(QComboBox) if c.isVisibleTo(ex.page.panel) and c.findData(value) >= 0
           and c.findData("Ha") >= 0]
    if not cbs:
        P("  !! no filter combo")
        return False
    cb = cbs[0]
    cb.setCurrentIndex(cb.findData(value))
    ex.settle(12)
    return True


def f01_fusion_broken_by_filter():          # N1, N2, N12
    ha, oiii = mono_fusion()
    fu = blk("fusion")
    obs("f01", fusion_tip=fu.toolTip() if fu else None)
    ok = set_filter("session@2", "SII")
    fu = blk("fusion")
    obs("f01", filter_set=ok, top=topbtn(), fusion_tip=fu.toolTip() if fu else None, refs=refs(),
        shot=shot("f01_fusion_red"))
    d = shift_f5("empilement", ha)
    obs("f01", here_ha=dump(d), shot=shot("f01_here_ha", d or ex.w))
    ex.close_dialogs()
    ex.w.run.dialog = None
    d = f5()
    obs("f01", f5=dump(d), shot=shot("f01_f5", d or ex.w))
    ex.close_dialogs()
    if blk("fusion") is not None:
        page, btns = inspect("fusion")
        obs("f01", inspect=[(b.text(), b.isEnabled(), b.toolTip()) for b in btns], page=texts(page)[:12],
            shot=shot("f01_inspect_fusion"))
        back()
    obs("f01", run_menu=run_menu())


def f02_fusion_card_reason():               # N2 with decision 2
    fresh("mono")
    c = ex.card("fusion")
    ex.hover(c, c.rect().center())
    ex.settle(5)
    obs("f02", card=card("fusion"), hint=ex.page.hint.text() if hasattr(ex.page, "hint") else None,
        shot=shot("f02_card"))
    ex.double_click_card("fusion")
    ex.settle(8)
    obs("f02", refs=refs(), popups=ex.popups[-2:], status=ex.w.status.text(),
        hint=ex.page.hint.text() if hasattr(ex.page, "hint") else None)
    ex.click_block(blk("empilement", ex.s.paths[0].pid))
    place("astrometry")
    obs("f02", card_after_ha=card("fusion"))


def f03_delete_astro_plan_open():           # N20
    ha, oiii = mono_fusion()
    d = shift_f5("fusion")
    obs("f03", before=dump(d))
    n0 = len(ex.popups)
    ex.click_block(blk("astrometrie", ha))
    ex.key(Qt.Key_Delete)
    ex.settle(10)
    d2 = ex.w.run.dialog
    obs("f03", still=blk("astrometrie", ha) is not None, popups=ex.popups[n0:], status=ex.w.status.text(),
        dialog=dump(d2) if d2 else None, visible=d2.isVisible() if d2 else None, shot=shot("f03_after_del"))
    ex.close_dialogs()
    # the filter change with the window open
    ex.w.run.dialog = None
    d = shift_f5("fusion")
    set_filter("session@2", "SII")
    d2 = ex.w.run.dialog
    info = dump(d2) if d2 else None
    obs("f03", after_filter=info, visible=d2.isVisible() if d2 else None, top=topbtn(),
        shot=shot("f03_plan_after_filter", d2 or ex.w))
    if d2 is not None and d2.isVisible() and d2.b_run.isEnabled():
        ex.click(d2.b_run)
        ex.settle(20)
        obs("f03", clicked=dump(ex.w.run.dialog), phase=ex.w.run.phase)
    ex.close_dialogs()


def f04_status_image_without_file():        # suspicion: status bar "Rien à calculer" with a flow to compute
    for i in range(2):
        fresh("session")
        place("stacking")
        obs(f"f04_{i}", before=topbtn())
        z = ex.new_flow_zone()
        ex.drag_palette("image", ex.ghost_center(z))
        ex.settle(15)
        obs(f"f04_{i}", after=topbtn(), refs=refs(), shot=shot(f"f04_status_{i}"))


def f05_here_without_lights():              # suspicion: Shift+F5 / Jusqu'ici enabled, nothing happens
    for i in range(2):
        ex.new_project(f"SansBrutesH{i}")
        ex.settle(10)
        place("session", "stacking")
        ex.click_block(blk("empilement"))
        ex.settle(4)
        w = ex.w
        before = (w.a_here.isEnabled(), w.a_here.toolTip(), w.status.text())
        n0 = len(ex.popups)
        ex.key(Qt.Key_F5, mod=Qt.ShiftModifier)
        ex.settle(10)
        d = dlg()
        obs(f"f05_{i}", before=before, dialog=dump(d), popups=ex.popups[n0:], status=w.status.text(),
            shot=shot(f"f05_here_{i}", d or w))
        ex.close_dialogs()
        page, btns = inspect("empilement")
        obs(f"f05_{i}", inspect=[(b.text(), b.isEnabled(), b.toolTip()) for b in btns], page=texts(page)[:6])
        back()



def toast():
    t = getattr(ex.view, "toast", None)
    try:
        return t.text() if t is not None and t.isVisible() else ""
    except Exception:  # noqa: BLE001
        return ""


def f06_delete_astro_refused():             # N20 / decision 3: the refusal says why
    ha, oiii = mono_fusion()
    ex.click_block(blk("astrometrie", ha))
    ex.key(Qt.Key_Delete)
    ex.settle(6)
    obs("f06", still=blk("astrometrie", ha) is not None, toast=toast(), status=ex.w.status.text(),
        popups=ex.popups[-2:], shot=shot("f06_delete_astro"))


def f07_stale_status():                     # suspicion: the status bar of the previous project stays
    for i in range(2):
        ex.new_project(f"SansBrutesS{i}")
        ex.settle(10)
        place("session", "stacking")
        ex.click_block(blk("empilement"))
        ex.key(Qt.Key_F5, mod=Qt.ShiftModifier)
        ex.settle(8)
        s1 = ex.w.status.text()
        fresh("session")
        place("stacking")
        s2 = (ex.w.status.text(), ex.w.bt_run.text(), ex.w.bt_run.isEnabled())
        z = ex.new_flow_zone()
        ex.drag_palette("image", ex.ghost_center(z))
        ex.settle(15)
        obs(f"f07_{i}", no_lights=s1, other_project=s2, after_image=(ex.w.status.text(), ex.w.bt_run.text()),
            shot=shot(f"f07_status_{i}"))


def f08_pause_on_off_block_only():          # regression near N8: the pause kept on a switched-off block
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "gradient", "denoise")
    ex.right_click_block(blk("fond"), "Pause vérif")
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    ex.settle(8)
    d = f5()
    obs("f08", rows=[r.name.text() for r in d.rows] if d else None,
        pause=(d.pause_note.text(), d.pause_more.text()) if d else None, status=ex.w.status.text(),
        shot=shot("f08_f5", d or ex.w))
    ex.close_dialogs()


def f09_undo_redo_plan_open():              # regression near N11: Ctrl+Y with the window still open
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "gradient")
    d = shift_f5("fond")
    ex.undo()
    ex.settle(8)
    a = dump(ex.w.run.dialog)
    ex.redo()
    ex.settle(8)
    b = dump(ex.w.run.dialog)
    obs("f09", after_undo=(a.get("summary"), a.get("rows"), a.get("run")), after_redo=(b.get("summary"), b.get("rows"),
                                                                                      b.get("run")),
        shot=shot("f09_after_redo", ex.w.run.dialog or ex.w))
    ex.close_dialogs()
    # Delete of another block (not the target) with the window open
    ex.w.run.dialog = None
    d = shift_f5("fond")
    ex.click_block(blk("calibration"))
    ex.key(Qt.Key_Delete)
    ex.settle(8)
    c = dump(ex.w.run.dialog)
    obs("f09", after_del_cal=(c.get("summary"), c.get("rows"), c.get("run")), refs=refs())
    ex.close_dialogs()


def f10_enter_refused():                    # regression near N6: Enter in a refused window
    ha, oiii = mono_fusion()
    set_filter("session@2", "SII")
    d = shift_f5("fusion")
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication
    fw = d.focusWidget() or d
    QApplication.sendEvent(fw, QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
    QApplication.sendEvent(fw, QKeyEvent(QEvent.KeyRelease, Qt.Key_Return, Qt.NoModifier))
    ex.settle(10)
    obs("f10", phase=ex.w.run.phase, after=dump(ex.w.run.dialog), visible=ex.w.run.dialog.isVisible()
        if ex.w.run.dialog else None)
    ex.close_dialogs()



def f11_stale_status_time():                # the stale status: does it go after a while, or after a click?
    import time as _t
    ex.new_project("SansBrutesT")
    ex.settle(10)
    place("session", "stacking")
    ex.click_block(blk("empilement"))
    ex.key(Qt.Key_F5, mod=Qt.ShiftModifier)
    ex.settle(8)
    fresh("session")
    place("stacking")
    a = ex.w.status.text()
    t0 = _t.time()
    while _t.time() - t0 < 12:
        ex.settle(5)
    b = ex.w.status.text()
    ex.click_block(blk("sources"))
    ex.settle(5)
    obs("f11", right_after=a, after_12s=b, after_click=ex.w.status.text(), run=ex.w.bt_run.text(),
        shot=shot("f11_status_12s"))


def f12_pause_on_off_block_here():          # second time: the pause of a switched-off block, Shift+F5 further
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "gradient", "denoise")
    ex.right_click_block(blk("fond"), "Pause vérif")
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    ex.settle(8)
    tip = blk("fond").toolTip()
    d = shift_f5("debruitage")
    obs("f12", tip=tip, bp=[x.breakpoints for x in ex.s.p.workflows], here=dump(d),
        pause=(d.pause_note.text(), d.pause_more.text()) if d else None, shot=shot("f12_here", d or ex.w))
    ex.close_dialogs()
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)                                   # on again: the pause works again?
    ex.settle(8)
    d = shift_f5("debruitage")
    obs("f12", on_again=dump(d).get("rows"), pause_on=(d.pause_note.text(),) if d else None)
    ex.close_dialogs()


if __name__ == "__main__":
    run_checks(dict(globals()), prefix="f", dump_name="check2_dump.json")
