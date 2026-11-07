"""run_status round 2, step 2: exploration dumps (prints and screenshots; notes are written by explore2_run_confirm.py).

    python explore2_run_explore.py e01 e02 ...  [--en] [--small]
"""
from __future__ import annotations

import sys

from explore2_run_recheck import (  # noqa: F401
    EXC, P, QPushButton, Qt, back, blk, dlg, dump, ex, f5, fresh, obs, place, refs, run_button, run_checks, shift_f5,
    shot, texts)
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QListWidget

SMALL = "--small" in sys.argv
if "--en" in sys.argv:
    ex.w = None
    from astro_gpu.client import app as appmod
    appmod.apply_language("en")
    ex.lang = "en"


def win():
    w = ex.open()
    if SMALL:
        w.resize(1366, 768)
        ex.settle(5)
    return w


def menu_action(*path):
    """A menu item of the menu bar, by the visible texts (a part of each)."""
    mb = ex.w.menuBar()
    acts = mb.actions()
    a = None
    for part in path:
        a = next((x for x in acts if part.lower() in x.text().replace("&", "").lower()), None)
        if a is None:
            return None
        acts = a.menu().actions() if a.menu() is not None else []
    return a


def truncated(root):
    """Labels whose text does not fit their width (without a tooltip)."""
    out = []
    for lb in root.findChildren(QLabel):
        try:
            if not lb.isVisible() or not lb.text() or lb.wordWrap():
                continue
            fm = lb.fontMetrics()
            need = fm.horizontalAdvance(lb.text().split("\n")[0])
            if need > lb.width() + 2 and "<" not in lb.text():
                out.append((lb.text()[:60], need, lb.width(), bool(lb.toolTip())))
        except RuntimeError:
            pass
    return out


def e01_settings():
    fresh("session")
    w = win()
    seen = {}

    def look(d):
        d.resize(d.size())
        ex.settle(3)
        pages = []
        nav = d.findChild(QListWidget)
        for i in range(nav.count()):
            nav.setCurrentRow(i)
            ex.settle(4)
            pg = d.pages.currentWidget()
            pages.append({"nav": nav.item(i).text(), "texts": texts(pg)[:40],
                          "combos": [[c.itemText(k) for k in range(c.count())] for c in pg.findChildren(QComboBox)
                                     if c.isVisible()],
                          "trunc": truncated(pg),
                          "tips": [x.toolTip() for x in pg.findChildren(QLabel) if x.isVisible() and x.toolTip()][:5]})
            ex.shot(f"e01_settings_{ex.lang}_{'small' if SMALL else 'big'}_{i}", d)
        seen["pages"] = pages
        seen["size"] = (d.width(), d.height(), d.title if hasattr(d, "title") else d.windowTitle())
        d.reject()
    ex.dialog_handler = look
    a = menu_action("Édition" if ex.lang == "fr" else "Edit", "Param" if ex.lang == "fr" else "Settings")
    P("  action", a.text() if a else None, a.shortcut().toString() if a else None)
    if a is not None:
        a.trigger()
    ex.settle(15)
    obs("e01", **seen)


def e02_language_with_windows():
    fresh("session")
    w = win()
    place("stacking")
    d = shift_f5("empilement")
    obs("e02", plan_fr=dump(d))
    w.change_language("en")
    ex.settle(10)
    d2 = ex.w.run.dialog
    obs("e02", plan_after_en=dump(d2), visible=d2.isVisible() if d2 else None, menus=[a.text() for a in w.menuBar().actions()],
        status=w.status.text(), run=(w.bt_run.text(), w.bt_run.toolTip()), shot=shot("e02_plan_after_en", d2 or w))
    ex.close_dialogs()
    run_button()
    con_en = w.logp.console.toPlainText()
    w.change_language("fr")
    ex.settle(10)
    obs("e02", console_en=con_en, console_fr=w.logp.console.toPlainText(), log=[x for _t, x in w.logp.rows()][:4],
        recap=dump(ex.w.run.dialog))
    ex.close_dialogs()


def e03_log_panel():
    fresh("session")
    w = win()
    place("stacking", "astrometry")
    d = run_button()
    if d is not None:
        lb = next((b for b in d.findChildren(QPushButton) if b.isVisible() and b.text() in ("Voir journal", "Show log")),
                  None)
        obs("e03", show_log_btn=lb.text() if lb else None)
        if lb is not None:
            ex.click(lb)
            ex.settle(8)
            obs("e03", log_visible_after_btn=w.logp.isVisible(), dialog_visible=d.isVisible() if d else None)
    ex.close_dialogs()
    if not w.logp.isVisible():
        ex.key(Qt.Key_L, w, Qt.ControlModifier)
    ex.settle(6)
    obs("e03", log_visible=w.logp.isVisible(), rows=[x for _t, x in w.logp.rows()][:6],
        console=w.logp.console.toPlainText(), graph_h=ex.view.height(), shot=shot("e03_log_open"))
    # export the log
    f = str(ex.ws / "exported_log.txt")
    ex.file_queue.append(f)
    ex.click(w.logp.b_export)
    ex.settle(5)
    from pathlib import Path
    obs("e03", asked=ex.file_asks[-1:], written=Path(f).read_text(encoding="utf-8")[:600] if Path(f).exists() else None)
    # undo, redo: the log lines
    ex.undo()
    ex.redo()
    obs("e03", rows_after_undo_redo=[x for _t, x in w.logp.rows()][:4], status=w.status.text())
    ex.key(Qt.Key_L, w, Qt.ControlModifier)


def e04_pause_flow():
    fresh("session_empty_wf")
    w = win()
    place("calibration", "registration", "stacking", "gradient")
    it = blk("empilement")
    m1 = ex.right_click_block(it, "Pause vérif")
    ex.settle(6)
    it = blk("empilement")
    obs("e04", menu=[(a.get("text"), a.get("checked"), a.get("enabled")) for a in m1], status=w.status.text(),
        tip=it.toolTip(), bp=[x.breakpoints for x in ex.s.p.workflows], shot=shot("e04_pause_set"))
    m2 = ex.right_click_block(blk("empilement"))
    obs("e04", menu_after=[(a.get("text"), a.get("checked"), a.get("tip")) for a in m2])
    d = shift_f5("fond")
    obs("e04", here_gradient=dump(d), pause_note=(d.pause_note.text(), d.pause_more.text()) if d else None,
        shot=shot("e04_here_gradient_paused", d or w))
    ex.close_dialogs()
    d = f5()
    obs("e04", f5=dump(d))
    ex.close_dialogs()
    ex.undo()
    obs("e04", undo_bp=[x.breakpoints for x in ex.s.p.workflows], refs=refs(), status=w.status.text())
    ex.redo()
    obs("e04", redo_bp=[x.breakpoints for x in ex.s.p.workflows], refs=refs())
    # pause then switch off the block
    ex.click_block(blk("fond"))
    ex.right_click_block(blk("fond"), "Pause vérif")
    ex.settle(4)
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    ex.settle(6)
    m3 = ex.right_click_block(blk("fond"))
    obs("e04", off_menu=[(a.get("text"), a.get("enabled"), a.get("tip")) for a in m3],
        bp=[x.breakpoints for x in ex.s.p.workflows], tip=blk("fond").toolTip())
    d = f5()
    obs("e04", f5_paused_off=dump(d), pause_note=(d.pause_note.text(), d.pause_more.text()) if d else None,
        shot=shot("e04_f5_paused_off", d or w))
    ex.close_dialogs()
    d = shift_f5("fond")
    obs("e04", here_off=dump(d), pause_note=(d.pause_note.text(), d.pause_more.text()) if d else None)
    ex.close_dialogs()
    # delete the paused block, undo: the pause comes back?
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_Delete)
    ex.settle(6)
    obs("e04", after_delete=[x.breakpoints for x in ex.s.p.workflows], refs=refs(), a_here=(w.a_here.isEnabled(),
                                                                                           w.a_here.toolTip()),
        sel=ex.s.sel_step)
    d = shift_f5()
    obs("e04", here_after_delete=dump(d))
    ex.close_dialogs()
    ex.undo()
    obs("e04", undo_delete=[x.breakpoints for x in ex.s.p.workflows], refs=refs())


def e05_run_window_small():
    fresh("mono")
    w = win()
    w.resize(1366, 768)
    ex.settle(6)
    place("fusion")
    d = f5()
    if d is not None:
        obs("e05", size=(d.width(), d.height(), d.x(), d.y()), trunc=truncated(d), summary=d.summary.text(),
            shot=shot("e05_fusion_small", d))
    ex.shot("e05_window_small")
    obs("e05", run=(w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip()), status=w.status.text(),
        top_trunc=truncated(w.bt_run.parentWidget()))
    ex.close_dialogs()
    w.resize(*ex.size)


def e06_fusion_en():
    fresh("mono")
    w = win()
    place("fusion")
    w.change_language("en")
    ex.settle(10)
    obs("e06", run=(w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip()), status=w.status.text(),
        tip=blk("fusion").toolTip())
    d = f5()
    obs("e06", f5=dump(d), shot=shot("e06_fusion_en", d or w))
    ex.close_dialogs()
    w.change_language("fr")
    ex.settle(10)


def e07_delete_selected_then_here():
    fresh("session_empty_wf")
    w = win()
    place("calibration", "registration", "stacking", "gradient", "denoise")
    ex.click_block(blk("debruitage"))
    ex.key(Qt.Key_Delete)
    ex.settle(8)
    obs("e07", sel=ex.s.sel_step, refs=refs(), a_here=(w.a_here.isEnabled(), w.a_here.toolTip()))
    d = shift_f5()
    obs("e07", here=dump(d))
    ex.close_dialogs()
    ex.undo()
    obs("e07", undo_sel=ex.s.sel_step, refs=refs())
    d = shift_f5()
    obs("e07", here_after_undo=dump(d))
    ex.close_dialogs()


def e08_undo_after_failed_start():
    fresh("session_empty_wf")
    w = win()
    place("calibration", "registration", "stacking")
    run_button()
    ex.close_dialogs()
    before = refs()
    ex.undo()
    after = refs()
    obs("e08", before=before, after=after, status=w.status.text(), log=[x for _t, x in w.logp.rows()][:4])
    ex.redo()
    obs("e08", redo=refs())


def e09_status_bar_states():
    fresh("session_empty_wf")
    w = win()
    seq = []
    seq.append(("empty", w.status.text(), w.bt_run.text(), w.bt_run.toolTip()))
    place("stacking")
    seq.append(("stacking", w.status.text(), w.bt_run.text(), w.bt_run.toolTip()))
    ex.settle(30)
    import time
    time.sleep(4.5)
    ex.settle(5)
    seq.append(("after flash", w.status.text(), w.bt_run.text(), w.bt_run.toolTip()))
    ex.undo()
    seq.append(("undo", w.status.text(), w.bt_run.text(), w.bt_run.toolTip()))
    time.sleep(4.5)
    ex.settle(5)
    seq.append(("undo later", w.status.text(), w.bt_run.text(), w.bt_run.toolTip()))
    obs("e09", seq=seq)


def _pause_on(ref, pid=None):
    n = len(EXC)
    it = blk(ref, pid)
    if it is None:
        return {"ref": ref, "missing": True}
    m = ex.right_click_block(it, "Pause vérif")
    ex.settle(6)
    item = next((a for a in m if (a.get("text") or "").startswith("Pause")), None)
    return {"ref": ref, "item": item, "exc": [e[-600:] for e in EXC[n:]], "status": ex.w.status.text(),
            "bp": [x.breakpoints for x in ex.s.p.workflows], "log": [x for _t, x in ex.w.logp.rows()][:1]}


def e10_pause_everywhere():
    fresh("mono")
    win()
    place("fusion")
    for ref in ("fusion", "composition", "etoiles", "session@1", "final", "calibration@2"):
        obs("e10", **{ref: _pause_on(ref)})
    fresh("image")
    for ref in ("image",):
        obs("e10", **{"img_" + ref: _pause_on(ref)})
    ex.click_block(blk("image"))
    place("astrometry")
    obs("e10", img_astro=_pause_on("astrometrie"))
    d = f5()
    obs("e10", f5_img=dump(d), pause_note=(d.pause_note.text(), d.pause_more.text()) if d else None)
    ex.close_dialogs()


def e11_pause_shared_branch():
    fresh("session_empty_wf")
    win()
    place("calibration", "registration", "stacking", "astrometry")
    ex.drag_palette_hover("gradient", ex.view.viewport().rect().center())
    gs = [(g.branch_key, ex.ghost_center(g)) for g in ex.ghosts() if getattr(g, "branch_key", None)]
    ex.drag_leave()
    tgt = next((c for k, c in gs if k == "astrometrie"), None)
    if tgt is not None:
        ex.drag_palette("gradient", tgt)
        ex.settle(15)
    obs("e11", paths=[q.name for q in ex.s.paths], refs=refs())
    r = _pause_on("astrometrie")
    obs("e11", pause=r, toast=[x for x in texts(ex.w) if "Pause" in x][:4], shot=shot("e11_pause_shared"))
    d = f5()
    obs("e11", f5=dump(d), pause_note=(d.pause_note.text(), d.pause_more.text()) if d else None,
        shot=shot("e11_f5_shared", d or ex.w))
    ex.close_dialogs()
    d = shift_f5("fond")
    obs("e11", here_branch=dump(d), pause_note=(d.pause_note.text(), d.pause_more.text()) if d else None)
    ex.close_dialogs()


def e12_settings_language_live():
    from astro_gpu.client import prefs
    prefs.set("language", ex.lang)        # the harness sets the language without the setting (the app reads it)
    fresh("session")
    w = win()
    place("stacking")
    d = shift_f5("empilement")
    seen = {}

    def go(dlg_):
        nav = dlg_.findChild(QListWidget)
        nav.setCurrentRow(0)
        ex.settle(3)
        c = next(c for c in dlg_.pages.currentWidget().findChildren(QComboBox) if c.isVisible())
        c.setCurrentIndex(c.findData("en"))
        ex.settle(10)
        seen["after"] = {"title": dlg_.windowTitle(), "nav": [nav.item(i).text() for i in range(nav.count())] if
                         isValid(nav) else None}
        nav2 = dlg_.findChild(QListWidget)
        seen["after"]["nav2"] = [nav2.item(i).text() for i in range(nav2.count())]
        seen["after"]["texts"] = texts(dlg_)[:12]
        seen["main"] = {"menus": [a.text() for a in ex.w.menuBar().actions()], "status": ex.w.status.text(),
                        "run": ex.w.bt_run.text()}
        rd = ex.w.run.dialog
        seen["run_dialog"] = dump(rd) if rd is not None else None
        ex.shot("e12_settings_after_en", dlg_)
        c2 = next(c for c in dlg_.pages.currentWidget().findChildren(QComboBox) if c.isVisible())
        c2.setCurrentIndex(c2.findData("fr"))
        ex.settle(10)
        seen["back"] = texts(dlg_)[:6]
        dlg_.reject()
    from shiboken6 import isValid
    ex.dialog_handler = go
    menu_action("Édition", "Param").trigger()
    ex.settle(30)
    obs("e12", **seen)
    ex.close_dialogs()


def e13_settings_device_live():
    from astro_gpu.client import prefs
    fresh("session")
    w = win()
    place("stacking")
    prefs.set("device", "auto")
    w.set_device("auto")
    ex.settle(5)
    seen = {"before": (w.bt_run.text(), w.device_cb.currentText())}

    def go(dlg_):
        nav = dlg_.findChild(QListWidget)
        nav.setCurrentRow(2)
        ex.settle(3)
        c = next(c for c in dlg_.pages.currentWidget().findChildren(QComboBox) if c.isVisible())
        c.setCurrentIndex(c.findData("cpu"))
        ex.settle(8)
        seen["during"] = (ex.w.bt_run.text(), ex.w.device_cb.currentText(), ex.w.device_cb.styleSheet())
        dlg_.reject()
    ex.dialog_handler = go
    menu_action("Édition", "Param").trigger()
    ex.settle(20)
    seen["after"] = (w.bt_run.text(), w.device_cb.currentText(), [w.device_cb.itemData(i, Qt.ToolTipRole)
                                                                   for i in range(w.device_cb.count())])
    ex.shot("e13_cpu_top")
    obs("e13", **seen)
    w.set_device("auto")
    prefs.set("device", "auto")


def e14_reopen_selection():
    root = ex.project("session_empty_wf")
    ex.open(root)
    ex.settle(15)
    w = win()
    place("calibration", "registration", "stacking", "gradient")
    ex.click_block(blk("alignement"))
    ex.key(Qt.Key_S, w, Qt.ControlModifier)
    ex.settle(10)
    sel0 = ex.s.sel_step
    ex.w.open_project(root)
    ex.settle(15)
    obs("e14", before=sel0, after=ex.s.sel_step, a_here=(w.a_here.isEnabled(), w.a_here.toolTip()),
        selected=[x.node.ref for x in ex.blocks() if x.isSelected()])
    d = shift_f5()
    obs("e14", here=dump(d))
    ex.close_dialogs()


def run_menu():
    a = menu_action("Exécuter" if ex.lang == "fr" else "Run")
    return [(x.text(), x.isEnabled(), x.toolTip(), x.shortcut().toString()) for x in a.menu().actions()
            if not x.isSeparator()]


def e16_run_menu_states():
    out = {}
    fresh("session")
    out["no_wf"] = run_menu()
    place("stacking")
    out["normal"] = run_menu()
    ex.new_project("MenuSansBrutes")
    ex.settle(8)
    place("session", "stacking")
    out["no_lights"] = run_menu()
    fresh("mono")
    place("fusion")
    out["fusion_refused"] = run_menu()
    fresh("two_flows")
    out["empty_flows"] = run_menu()
    obs("e16", **out)


def e17_enter_in_plan():
    fresh("session")
    w = win()
    place("stacking")
    d = shift_f5("empilement")
    btns = [(b.text(), b.isDefault(), b.autoDefault(), b.isVisible()) for b in d.findChildren(QPushButton)]
    ex.key(Qt.Key_Return, d)
    ex.settle(15)
    d2 = ex.w.run.dialog
    obs("e17", buttons=btns, after_enter=dump(d2) if d2 is not None else None,
        visible=d2.isVisible() if d2 is not None else None, track=ex.w.run.track is not None)
    ex.close_dialogs()
    d = shift_f5("empilement")
    ex.key(Qt.Key_Escape, d)
    ex.settle(5)
    obs("e17", after_esc_visible=d.isVisible() if d is not None else None)


def e19_device_change_with_plan_open():
    from astro_gpu.client import prefs
    fresh("session")
    w = win()
    place("stacking")
    prefs.set("device", "auto")
    w.set_device("auto")
    d = shift_f5("empilement")
    before = dump(d)
    i = w.device_cb.findData("cpu")
    w.device_cb.setCurrentIndex(i)
    w.device_cb.activated.emit(i)
    ex.settle(10)
    after = dump(ex.w.run.dialog)
    obs("e19", before=(before["summary"], before["dur"]), after=(after.get("summary"), after.get("dur")),
        top=(w.bt_run.text(), w.device_cb.currentText()), shot=shot("e19_plan_after_cpu", ex.w.run.dialog or w))
    ex.close_dialogs()
    w.set_device("auto")
    prefs.set("device", "auto")


def e20_undo_with_plan_open():
    fresh("session_empty_wf")
    w = win()
    place("calibration", "registration", "stacking", "gradient")
    d = shift_f5("fond")
    before = dump(d)
    ex.undo()                                     # the Gradient goes
    ex.settle(8)
    d2 = ex.w.run.dialog
    after = dump(d2) if d2 is not None else None
    obs("e20", before=before["rows"], refs=refs(), after=after, visible=d2.isVisible() if d2 else None,
        shot=shot("e20_plan_after_undo", d2 or w))
    if d2 is not None and d2.isVisible() and d2.b_run.isEnabled():
        ex.click(d2.b_run)
        ex.settle(20)
        obs("e20", after_run=dump(ex.w.run.dialog), log=[x for _t, x in w.logp.rows()][:3])
    ex.close_dialogs()


def e21_long_names_small():
    fresh("two_sessions")
    w = win()
    w.resize(1366, 768)
    ex.settle(5)
    s = ex.s
    try:
        for sess, name in zip(s.p.sessions, ("Nuit du 12 au 13 octobre, Bortle 4, lune 30 %",
                                             "Nuit du 14 au 15 octobre, jardin, filtre L-eXtreme")):
            sess.name = name
        s.p.workflows[0].name = "Nébuleuse du Cœur, champ large avec la caméra principale"
        s.flush()
    except Exception as e:  # noqa: BLE001
        P("rename failed", e)
    ex.w.open_project(s.p.root)
    ex.settle(15)
    place("calibration", "registration", "stacking")
    d = f5()
    if d is not None:
        obs("e21", size=(d.width(), d.height()), trunc=truncated(d), rows=[r.name.text() for r in d.rows],
            who=[r.who.text() for r in d.rows], title=d.windowTitle(), shot=shot("e21_long_names_small", d))
    ex.close_dialogs()
    d = shift_f5("calibration@1")
    if d is not None:
        obs("e21", here_title=d.windowTitle(), here_size=(d.width(), d.height()), trunc2=truncated(d),
            shot=shot("e21_here_long_small", d))
    ex.close_dialogs()
    w.resize(*ex.size)


def e22_fusion_session_removed():
    fresh("mono")
    w = win()
    place("fusion")
    ha, oiii = ex.s.paths[0].pid, ex.s.paths[1].pid
    for pid in (ha, oiii):
        ex.click_block(blk("empilement", pid))
        place("astrometry")
    st0 = (w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip(), w.status.text())
    ex.click_block(blk("session@2"))
    ex.key(Qt.Key_Delete)
    ex.settle(10)
    fu = blk("fusion")
    st1 = (w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip(), w.status.text(), fu.toolTip() if fu else None,
           refs(), ex.popups[-2:])
    shot("e22_session_removed")
    d = f5()
    obs("e22", before=st0, after_delete=st1, f5=dump(d), shot=shot("e22_f5", d or w))
    ex.close_dialogs()
    ex.undo()
    fu = blk("fusion")
    obs("e22", undo=(w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip(), fu.toolTip() if fu else None, refs()))
    ex.redo()
    obs("e22", redo=(w.bt_run.isEnabled(), w.bt_run.toolTip(), refs()))
    # a channel's Stars/background removed
    ex.undo()
    it = blk("etoiles", ha)
    ex.click_block(it)
    ex.key(Qt.Key_Delete)
    ex.settle(10)
    obs("e22", stars_removed=(w.bt_run.isEnabled(), w.bt_run.toolTip(), refs(), w.status.text(),
                              ex.popups[-2:]))


def _fail_with(exc):
    from astro_gpu.core import runner2

    def boom(*a, **k):
        raise exc
    old = runner2._spawn_worker
    runner2._spawn_worker = boom
    try:
        return run_button()
    finally:
        runner2._spawn_worker = old


def e23_start_failures():
    fresh("session")
    w = win()
    place("stacking")
    out = {}
    for name, exc in (("disk", OSError(28, "No space left on device")),
                      ("perm", PermissionError(13, "Access is denied", "F:/x")),
                      ("nofile", FileNotFoundError(2, "No such file or directory", "G:/lights")),
                      ("other", RuntimeError("worker crashed: CUDA error 999"))):
        ex.close_dialogs()
        d = _fail_with(exc)
        out[name] = {"texts": texts(d) if d else None, "status": w.status.text(),
                     "console": w.logp.console.toPlainText().splitlines()[-1:],
                     "log": [x for _t, x in w.logp.rows()][:1]}
        if name == "disk":
            shot("e23_disk_full", d or w)
    obs("e23", **out)


def e24_failure_detail_in_log():
    fresh("session")
    w = win()
    place("stacking")
    d = _fail_with(RuntimeError("worker crashed: CUDA error 999 (unknown error)"))
    b = next((x for x in d.findChildren(QPushButton) if x.isVisible() and x.text() == "Voir journal"), None)
    if b is not None:
        ex.click(b)
        ex.settle(8)
    obs("e24", console=w.logp.console.toPlainText(), log=[x for _t, x in w.logp.rows()][:3],
        log_visible=w.logp.isVisible(), shot=shot("e24_log_after_failure"))
    f = str(ex.ws / "log_after_failure.txt")
    ex.file_queue.append(f)
    ex.click(w.logp.b_export)
    ex.settle(5)
    from pathlib import Path
    obs("e24", exported=Path(f).read_text(encoding="utf-8") if Path(f).exists() else None)


def _state(w):
    from PySide6.QtWidgets import QFrame
    rs = [x for x in w.findChildren(QFrame) if x.__class__.__name__ == "RunStatus" and x.isVisible()]
    return {"run_btn": (w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip()), "status": w.status.text(),
            "combo": (w.device_cb.isEnabled(), w.device_cb.toolTip()), "a_here": (w.a_here.isEnabled(),
                                                                                  w.a_here.toolTip()),
            "a_stop": (w.a_stop.isEnabled(),), "a_run": (w.a_run.isEnabled(),),
            "runstatus": [texts(x) for x in rs]}


def e25_slow_start():
    import time as _t
    from astro_gpu.core import runner2
    fresh("session_empty_wf")
    w = win()
    place("calibration", "registration", "stacking", "gradient")

    def slow(*a, **k):
        _t.sleep(6)
        raise OSError(28, "No space left on device")
    old = runner2._spawn_worker
    runner2._spawn_worker = slow
    try:
        ex.click(w.bt_run)
        ex.settle(5)
        d = ex.w.run.dialog
        obs("e25", during=_state(w), dialog=dump(d), shot=shot("e25_starting", d or w))
        ex.shot("e25_starting_window")
        # Inspect a block meanwhile
        ex.close_dialogs()
        ex.double_click_block(blk("fond"))
        ex.settle(5)
        page = ex.w.tune.pages.currentWidget()
        btns = [(b.text(), b.isEnabled(), b.toolTip()) for b in page.findChildren(QPushButton)
                if b.isVisible() and b.text().startswith("Jusqu")]
        obs("e25", inspect_btn=btns)
        back()
        # an edit meanwhile: Ctrl+Z
        before = refs()
        ex.undo()
        obs("e25", undo_during=(before, refs(), w.status.text()))
        ex.redo()
        ex.key(Qt.Key_F5)
        ex.settle(3)
        obs("e25", f5_during=dump(ex.w.run.dialog), vis=ex.w.run.dialog.isVisible() if ex.w.run.dialog else None)
        for _ in range(60):
            ex.settle(3)
            if not ex.w.run.running:
                break
        obs("e25", after=_state(w), dialog_after=dump(ex.w.run.dialog))
    finally:
        runner2._spawn_worker = old


def e26_stop_while_starting():
    import time as _t
    from astro_gpu.core import runner2
    fresh("session")
    w = win()
    place("stacking")

    def slow(*a, **k):
        _t.sleep(4)
        raise OSError(28, "No space left on device")
    old = runner2._spawn_worker
    runner2._spawn_worker = slow
    try:
        ex.click(w.bt_run)
        ex.settle(5)
        from PySide6.QtWidgets import QFrame
        rs = next((x for x in w.findChildren(QFrame) if x.__class__.__name__ == "RunStatus" and x.isVisible()), None)
        stop_btn = (rs.b_stop.isEnabled(), rs.b_stop.toolTip()) if rs else None
        d = ex.w.run.dialog
        dstop = (d.b_stop.isVisible(), d.b_stop.isEnabled(), d.b_stop.toolTip()) if d else None
        w.a_stop.trigger()
        ex.settle(5)
        mid = (ex.w.run.phase, _state(w)["runstatus"], dump(ex.w.run.dialog)["summary"])
        t0 = _t.time()
        while ex.w.run.running and _t.time() - t0 < 20:
            ex.settle(5)
        obs("e26", stop_btn=stop_btn, dialog_stop=dstop, after_menu_stop=mid, waited=round(_t.time() - t0, 1),
            end=_state(w), dialog=dump(ex.w.run.dialog), log=[x for _t2, x in w.logp.rows()][:3],
            shot=shot("e26_after_stop", ex.w.run.dialog or w))
    finally:
        runner2._spawn_worker = old


def e27_keys_in_other_views():
    fresh("session_empty_wf")
    w = win()
    place("calibration", "registration", "stacking", "gradient")
    ex.click_block(blk("fond"))
    out = {}
    for mode, name in ((1, "inspect"), (2, "post")):
        w.set_mode(mode)
        ex.settle(10)
        ex.close_dialogs()
        ex.key(Qt.Key_F5, mod=Qt.ShiftModifier, widget=w)
        ex.settle(10)
        d = ex.w.run.dialog
        out[name + "_shift_f5"] = (d.windowTitle(), d.isVisible()) if d is not None else None
        ex.close_dialogs()
        ex.w.run.dialog = None
        ex.key(Qt.Key_F5, widget=w)
        ex.settle(20)
        d = ex.w.run.dialog
        out[name + "_f5"] = (d.windowTitle(), d.summary.text(), d.isVisible()) if d is not None else None
        ex.close_dialogs()
        ex.w.run.dialog = None
    w.set_mode(0)
    obs("e27", **out)


def e28_pause_two_flows():
    fresh("two_flows")
    w = win()
    for sid in (1, 2):
        ex.click_block(next(x for x in ex.blocks() if x.node.key == "session" and x.node.sid == sid))
        place("calibration", "registration", "stacking", "gradient")
    P("  refs", refs())
    p1, p2 = ex.s.paths[0].pid, ex.s.paths[1].pid
    ex.right_click_block(blk("empilement", p1), "Pause vérif")
    ex.right_click_block(blk("fond", p1), "Pause vérif")
    ex.right_click_block(blk("alignement", p2), "Pause vérif")
    ex.settle(5)
    ex.close_dialogs()
    ex.w.run.dialog = None
    d = f5()
    obs("e28", bp=[x.breakpoints for x in ex.s.p.workflows], f5=dump(d),
        notes=(d.pause_note.text(), d.pause_more.text()) if d else None, shot=shot("e28_two_flows_pauses", d or w))
    ex.close_dialogs()
    w.change_language("en")
    ex.settle(10)
    d = shift_f5("fond", p1)
    obs("e28", here_en=dump(d), notes_en=(d.pause_note.text(), d.pause_more.text()) if d else None)
    ex.close_dialogs()
    w.change_language("fr")
    ex.settle(10)


def e29_selection_visible():
    root = ex.project("session_empty_wf")
    ex.open(root)
    ex.settle(15)
    w = win()
    place("calibration", "registration", "stacking", "gradient")
    ex.click_block(blk("alignement"))
    shot("e29_selected_before")
    ex.key(Qt.Key_S, w, Qt.ControlModifier)
    ex.settle(10)
    ex.w.open_project(root)
    ex.settle(15)
    obs("e29", sel=ex.s.sel_step, panel=texts(ex.page.panel)[:3], shot=shot("e29_after_reopen"))
    # a click on the empty canvas
    vp = ex.view.viewport()
    from PySide6.QtCore import QPoint
    ex.click(vp, QPoint(vp.width() // 2, 30))
    ex.settle(5)
    obs("e29", sel_after_empty_click=ex.s.sel_step, a_here=(w.a_here.isEnabled(), w.a_here.toolTip()),
        panel=texts(ex.page.panel)[:3])
    d = shift_f5()
    obs("e29", here=(d.windowTitle() if d else None), shot=shot("e29_here_after_empty_click", d or w))
    ex.close_dialogs()


def e30_fusion_broken_with_plan_open():
    fresh("mono")
    w = win()
    place("fusion")
    ha, oiii = ex.s.paths[0].pid, ex.s.paths[1].pid
    for pid in (ha, oiii):
        ex.click_block(blk("empilement", pid))
        place("astrometry")
    d = shift_f5("fusion")
    before = (d.summary.text(), d.b_run.isEnabled()) if d else None
    ex.click_block(blk("astrometrie", ha))
    ex.key(Qt.Key_Delete)
    ex.settle(10)
    d2 = ex.w.run.dialog
    mid = (d2.isVisible(), d2.summary.text(), d2.b_run.isEnabled(), d2.b_run.toolTip()) if d2 else None
    if d2 is not None and d2.isVisible() and d2.b_run.isEnabled():
        ex.click(d2.b_run)
        ex.settle(25)
    d3 = ex.w.run.dialog
    obs("e30", before=before, after_delete=mid, after_click=dump(d3) if d3 else None,
        run_btn=(w.bt_run.isEnabled(), w.bt_run.toolTip()), tracking=ex.w.run.track is not None,
        shot=shot("e30_plan_fusion_broken", d3 or w))
    ex.close_dialogs()


def e31_here_on_sources_and_post():
    fresh("image")
    w = win()
    ex.click_block(blk("image"))
    place("astrometry")
    out = {}
    for ref in ("image", "final", "astrometrie"):
        ex.click_block(blk(ref))
        ex.settle(4)
        out[ref] = {"sel": ex.s.sel_step, "a_here": (w.a_here.isEnabled(), w.a_here.toolTip())}
        d = shift_f5()
        out[ref]["dialog"] = (d.windowTitle(), [r.name.text() for r in d.rows], d.summary.text()) if d else None
        ex.close_dialogs()
        ex.w.run.dialog = None
    obs("e31", **out)


if __name__ == "__main__":
    run_checks(dict(globals()), prefix="e", dump_name="explore_dump.json")
