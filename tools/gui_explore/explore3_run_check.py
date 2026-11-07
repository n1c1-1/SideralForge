"""run_status round 3 (control): each open round-2 point replayed with real gestures; prints what the window shows and
takes screenshots (k*.png in exploration3/run_status). Verdicts written to recheck.jsonl by hand.

    python explore3_run_check.py [k01 ...]
"""
from __future__ import annotations

import time as _t

from explore2_run_explore import _fail_with, menu_action, run_menu, truncated  # noqa: F401
from explore2_run_recheck import (  # noqa: F401
    EXC, P, QPushButton, Qt, back, blk, dlg, dump, ex, f5, fresh, inspect, obs, place, refs, run_button, run_checks,
    shift_f5, texts)
from PySide6.QtCore import QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QFrame


def shot(name, widget=None):
    return ex.shot(f"k{name}", widget)


def topbtn():
    w = ex.w
    return (w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip(), w.status.text())


def bps():
    return [x.breakpoints for x in ex.s.p.workflows]


def k10_session_calibration_inspect():
    fresh("two_sessions")
    place("calibration", "registration", "stacking")
    it = blk("calibration@1")
    obs("k10", tip=it.toolTip() if it else None, refs=refs())
    page, btns = inspect("calibration@1")
    obs("k10", page=texts(page)[:10], btns=[(b.text(), b.isEnabled(), b.toolTip()) for b in btns],
        strip=[x for x in texts(ex.w.tune.strip) if x != "›"], shot=shot("10_inspect_cal1"))
    back()


def k11_pause_session_calibration():
    fresh("two_sessions")
    place("calibration", "registration", "stacking")
    m = ex.right_click_block(blk("calibration@1"), "Pause vérif")
    ex.settle(8)
    it = blk("calibration@1")
    obs("k11", menu=[(a.get("text"), a.get("enabled"), a.get("tip")) for a in m], bp=bps(), tip=it.toolTip(),
        status=ex.w.status.text(), log=[x for _t2, x in ex.w.logp.rows()][:3], popups=ex.popups[-2:],
        shot=shot("11_after_pause_cal1"))
    d = f5()
    obs("k11", f5=dump(d), pause=(d.pause_note.text(), d.pause_more.text()) if d else None)
    ex.close_dialogs()


def k15_no_lights():
    ex.new_project("SansBrutes3")
    ex.settle(10)
    place("session", "stacking")
    d = shift_f5("empilement")
    obs("k15", here=dump(d), shot=shot("15_no_lights_here", d or ex.w))
    ex.close_dialogs()
    d = f5()
    obs("k15", f5=dump(d), top=topbtn())
    ex.close_dialogs()


def k16_image_without_file():
    fresh("session")
    place("stacking")
    z = ex.new_flow_zone()
    ex.drag_palette("image", ex.ghost_center(z))
    ex.settle(15)
    obs("k16", top=topbtn(), refs=refs(), image_tip=blk("image").toolTip() if blk("image") else None)
    d = f5()
    obs("k16", f5=dump(d), shot=shot("16_f5", d or ex.w))
    ex.close_dialogs()
    d = run_button()
    obs("k16", run_all=dump(d), shot=shot("16_run_button", d or ex.w))
    ex.close_dialogs()


def k24_jpg():
    fresh("jpg_mistake")
    place("stacking")
    obs("k24", top=topbtn())
    d = shift_f5("empilement")
    obs("k24", here=dump(d), shot=shot("24_jpg_here", d or ex.w))
    ex.close_dialogs()
    ex.w.run.dialog = None
    d = f5()
    obs("k24", f5=dump(d), shot=shot("24_jpg_f5", d or ex.w))
    ex.close_dialogs()
    ex.w.run.dialog = None
    d = run_button()
    obs("k24", run_all=dump(d), shot=shot("24_jpg_run_button", d or ex.w))
    ex.close_dialogs()


def k27_strip():
    fresh("image")
    ex.double_click_block(blk("image"))
    ex.settle(12)
    st = [x for x in texts(ex.w.tune.strip) if x != "›"]
    obs("k27", strip=st, graph=refs(), shot=shot("27_strip"))
    back()
    ex.click_block(blk("image"))
    place("gradient")
    ex.double_click_block(blk("fond"))
    ex.settle(12)
    obs("k27", strip_g=[x for x in texts(ex.w.tune.strip) if x != "›"], graph_g=refs(), shot=shot("27_strip_g"))
    back()


def k31_fusion_scope_and_reason():          # N1, N2, N12
    fresh("mono")
    place("fusion")
    fu = blk("fusion")
    obs("k31", top=topbtn(), fusion_tip=fu.toolTip() if fu else None, toast=ex.popups[-2:], refs=refs())
    ha = ex.s.paths[0].pid
    d = shift_f5("empilement", ha)
    obs("k31", here_ha=dump(d), shot=shot("31_here_ha_stack", d or ex.w))
    ex.close_dialogs()
    ex.w.run.dialog = None
    d = f5()
    obs("k31", f5=dump(d), shot=shot("31_f5", d or ex.w))
    ex.close_dialogs()
    page, btns = inspect("fusion")
    obs("k31", inspect=[(b.text(), b.isEnabled(), b.toolTip()) for b in btns], page=texts(page)[:12],
        shot=shot("31_inspect_fusion"))
    back()
    m = run_menu()
    obs("k31", run_menu=m)


def k33_mono_rows():                        # N3
    fresh("mono")
    d = f5()
    obs("k33", mono=dump(d), shot=shot("33_mono_f5", d or ex.w))
    ex.close_dialogs()
    fresh("two_sessions")
    place("calibration", "registration", "stacking")
    d = f5()
    obs("k33", two_sessions=dump(d))
    ex.close_dialogs()


def k34_off_block():                        # N4, N8
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "gradient")
    ex.right_click_block(blk("fond"), "Pause vérif")
    ex.settle(5)
    bp_on = bps()
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    ex.settle(8)
    bp_off = bps()
    m_off = ex.right_click_block(blk("fond"))
    toast = ex.w.toast.text() if hasattr(ex.w, "toast") and hasattr(ex.w.toast, "text") else None
    obs("k34", bp_on=bp_on, bp_off=bp_off, tip_off=blk("fond").toolTip(), toast=toast,
        log=[x for _t2, x in ex.w.logp.rows()][:3],
        menu_off=[(a.get("text"), a.get("enabled"), a.get("tip")) for a in m_off], shot=shot("34_off"))
    ex.right_click_block(blk("empilement"), "Pause vérif")
    ex.settle(5)
    d = f5()
    obs("k34", f5=dump(d), shot=shot("34_off_f5", d or ex.w))
    ex.close_dialogs()
    ex.undo()
    ex.undo()
    obs("k34", after_2undo=bps(), tip=blk("fond").toolTip())
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    ex.settle(8)
    obs("k34", after_D_again=bps(), tip=blk("fond").toolTip())


def k35_run_menu():                         # N5
    out = {}
    fresh("session")
    out["no_wf"] = run_menu()
    ex.new_project("MenuSansBrutes3")
    ex.settle(8)
    place("session", "stacking")
    out["no_lights"] = run_menu()
    fresh("two_flows")
    out["empty_flows"] = run_menu()
    obs("k35", **out)


def k36_enter():                            # N6
    fresh("session")
    place("stacking")
    d = shift_f5("empilement")
    default = [(b.text(), b.isVisible()) for b in d.findChildren(QPushButton) if b.isDefault()]
    fw = d.focusWidget() or d
    QApplication.sendEvent(fw, QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
    QApplication.sendEvent(fw, QKeyEvent(QEvent.KeyRelease, Qt.Key_Return, Qt.NoModifier))
    ex.settle(15)
    d2 = ex.w.run.dialog
    obs("k36", default=default, focus=fw.__class__.__name__, phase=ex.w.run.phase, after=dump(d2),
        shot=shot("36_after_enter", d2 or ex.w))
    ex.close_dialogs()


def k37_long_title():                       # N7
    fresh("two_sessions")
    w = ex.w
    w.resize(1366, 768)
    ex.settle(5)
    s = ex.s
    s.p.sessions[0].name = "Nuit du 12 au 13 octobre, Bortle 4, lune 30 %"
    s.p.workflows[0].name = "Nébuleuse du Cœur, champ large avec la caméra principale"
    s.flush()
    ex.w.open_project(s.p.root)
    ex.settle(15)
    place("calibration", "registration", "stacking")
    d = shift_f5("calibration@1")
    obs("k37", title=d.windowTitle() if d else None, trunc=truncated(d) if d else None,
        width=d.width() if d else None, shot=shot("37_long_title", d or ex.w))
    ex.close_dialogs()
    w.resize(*ex.size)


def k38_branch_output():                    # N9
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "astrometry")
    ex.drag_palette_hover("gradient", ex.view.viewport().rect().center())
    gs = [(g.branch_key, ex.ghost_center(g)) for g in ex.ghosts() if getattr(g, "branch_key", None)]
    ex.drag_leave()
    tgt = next((c for k, c in gs if k == "astrometrie"), None)
    obs("k38", ghosts=[k for k, _c in gs])
    if tgt is None:
        return
    ex.drag_palette("gradient", tgt)
    ex.settle(15)
    obs("k38", refs=refs(), titles=[x.title for x in ex.blocks()], shot=shot("38_branch_graph"))
    d = f5()
    obs("k38", f5=dump(d))
    ex.close_dialogs()


def k39_log_undo_names():                   # N10
    fresh("mono")
    ex.right_click_block(blk("calibration@2"), "Pause vérif")
    ex.settle(5)
    ex.undo()
    ex.redo()
    rows = [x for _t2, x in ex.w.logp.rows()][:5]
    ex.w.b_log.setChecked(True)
    ex.settle(5)
    obs("k39", log=rows, shot=shot("39_log"))
    ex.w.b_log.setChecked(False)


def k40_undo_plan_open():                   # N11
    from astro_gpu.client import prefs
    fresh("session_empty_wf")
    w = ex.w
    place("calibration", "registration", "stacking", "gradient")
    d = shift_f5("fond")
    before = dump(d)
    ex.undo()
    ex.settle(8)
    d2 = ex.w.run.dialog
    after = dump(d2) if d2 is not None else None
    obs("k40", before=before["rows"], refs=refs(), after=after, visible=d2.isVisible() if d2 else None,
        shot=shot("40_plan_after_undo", d2 or w))
    ex.close_dialogs()
    ex.w.run.dialog = None
    prefs.set("device", "auto")
    w.set_device("auto")
    place("stacking") if blk("empilement") is None else None
    d = shift_f5("empilement")
    b = dump(d)
    i = w.device_cb.findData("cpu")
    w.device_cb.setCurrentIndex(i)
    w.device_cb.activated.emit(i)
    ex.settle(10)
    a = dump(ex.w.run.dialog)
    obs("k40", dev_before=(b.get("summary"), b.get("dur")), dev_after=(a.get("summary"), a.get("dur")),
        top=(w.bt_run.text(), w.device_cb.currentText()), shot=shot("40_plan_after_cpu", ex.w.run.dialog or w))
    ex.close_dialogs()
    w.set_device("auto")
    prefs.set("device", "auto")


def k43_settings():                         # N13, N14
    fresh("session")
    seen = {}

    def look(d):
        from PySide6.QtWidgets import QAbstractButton, QListWidget
        d.show_page("folders")
        ex.settle(20)
        seen["folders"] = texts(d)
        seen["shot"] = ex.shot("k43_settings_folders", d)
        out = []
        nav = d.findChild(QListWidget)
        for i in range(nav.count()):
            nav.setCurrentRow(i)
            ex.settle(3)
            for b in d.pages.currentWidget().findChildren(QAbstractButton):
                if b.isVisible() and b.text():
                    out.append((b.text(), b.toolTip()))
        seen["buttons"] = out
        d.reject()
    ex.dialog_handler = look
    menu_action("Édition", "Param").trigger()
    ex.settle(25)
    obs("k43", folders=[x for x in seen.get("folders", []) if "Go" in x or "Cache" in x or "cache" in x],
        buttons=seen.get("buttons"), shot=seen.get("shot"))


def k45_failure_detail():                   # N15
    fresh("session")
    w = ex.w
    place("stacking")
    d = _fail_with(RuntimeError("worker crashed: CUDA error 999 (unknown error)"))
    txt = texts(d)
    b = next((x for x in d.findChildren(QPushButton) if x.isVisible() and "journal" in x.text().lower()), None)
    if b is not None:
        ex.click(b)
        ex.settle(8)
    obs("k45", dialog=txt, console=w.logp.console.toPlainText(), log=[x for _t2, x in w.logp.rows()][:3],
        shot=shot("45_log_after_failure"))
    ex.close_dialogs()


def k46_stop_while_starting():              # N16, N17
    from astro_gpu.core import runner2
    fresh("session")
    w = ex.w
    place("stacking")

    def slow(*a, **k):
        _t.sleep(4)
        raise OSError(28, "No space left on device")
    old = runner2._spawn_worker
    runner2._spawn_worker = slow
    try:
        ex.click(w.bt_run)
        ex.settle(5)
        rs = next((x for x in w.findChildren(QFrame) if x.__class__.__name__ == "RunStatus" and x.isVisible()), None)
        d = ex.w.run.dialog
        info = {"bar": (rs.b_stop.isEnabled(), rs.b_stop.toolTip()) if rs else None,
                "window": (d.b_stop.isVisible(), d.b_stop.isEnabled(), d.b_stop.toolTip()) if d else None,
                "menu": (w.a_stop.isEnabled(), w.a_stop.toolTip()), "phase": ex.w.run.phase,
                "texts": texts(d)}
        info["shot"] = shot("46_starting", d or w)
        info["shot_w"] = shot("46_starting_window")
        w.a_stop.trigger()
        ex.settle(5)
        info["after_menu_stop"] = (ex.w.run.phase, w.status.text(), dump(ex.w.run.dialog).get("summary"))
        t0 = _t.time()
        while ex.w.run.running and _t.time() - t0 < 20:
            ex.settle(5)
        info["end"] = (ex.w.run.phase, dump(ex.w.run.dialog).get("summary"), w.status.text())
    finally:
        runner2._spawn_worker = old
    obs("k46", **info)


def k48_pauses_two_flows():                 # N18
    fresh("two_flows")
    for sid in (1, 2):
        ex.click_block(next(x for x in ex.blocks() if x.node.key == "session" and x.node.sid == sid))
        place("calibration", "registration", "stacking", "gradient")
    p1, p2 = ex.s.paths[0].pid, ex.s.paths[1].pid
    ex.right_click_block(blk("empilement", p1), "Pause vérif")
    ex.right_click_block(blk("alignement", p2), "Pause vérif")
    ex.settle(5)
    ex.close_dialogs()
    ex.w.run.dialog = None
    d = f5()
    obs("k48", names=[q.name for q in ex.s.paths], bp=bps(),
        rows=[(r.name.text(), r.who.text()) for r in d.rows] if d else None,
        pause=(d.pause_note.text(), d.pause_more.text()) if d else None, f5=dump(d), shot=shot("48_f5", d or ex.w))
    ex.close_dialogs()


def k49_log_gender():                       # N19
    fresh("session_empty_wf")
    place("calibration", "registration", "selection", "stacking", "astrometry", "color")
    rows = [x for _t2, x in ex.w.logp.rows()]
    obs("k49", log=rows[:7])
    ex.w.b_log.setChecked(True)
    ex.settle(5)
    shot("49_log")
    ex.w.b_log.setChecked(False)


def k50_fusion_broken_plan_open():          # N20
    fresh("mono")
    w = ex.w
    place("fusion")
    ha, oiii = ex.s.paths[0].pid, ex.s.paths[1].pid
    for pid in (ha, oiii):
        ex.click_block(blk("empilement", pid))
        place("astrometry")
    obs("k50", top=topbtn(), refs=refs())
    d = shift_f5("fusion")
    obs("k50", before=dump(d))
    n0 = len(ex.popups)
    ex.click_block(blk("astrometrie", ha))
    ex.key(Qt.Key_Delete)
    ex.settle(10)
    d2 = ex.w.run.dialog
    obs("k50", after_del=dump(d2), visible=d2.isVisible() if d2 else None, still=blk("astrometrie", ha) is not None,
        popups=ex.popups[n0:], status=w.status.text(), top=topbtn(), shot=shot("50_after_delete", w))
    ex.close_dialogs()
    # another way to break it: the filter of the Ha session changed (decision 3: Fusion red, Run refused)
    ex.w.run.dialog = None
    d = shift_f5("fusion")
    obs("k50", reopened=dump(d))
    try:
        ex.s.p.sessions[0].filter = "OIII"
    except Exception as e:  # noqa: BLE001
        P("filter change failed", e)
    ex.close_dialogs()


if __name__ == "__main__":
    run_checks(dict(globals()), prefix="k", dump_name="check_dump.json")
