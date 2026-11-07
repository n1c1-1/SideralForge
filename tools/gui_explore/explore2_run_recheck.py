"""run_status, round 2, step 1: replay each round-1 anomaly with real gestures and dump what the window shows.

No notes: every check prints what it saw and takes a screenshot (rNN_*.png); the verdicts go to recheck.jsonl by hand.

    python explore2_run_recheck.py [check ...]
"""
from __future__ import annotations

import faulthandler
import json
import sys
import traceback

from harness import TMP, Explorer, stray_qt_errors
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QComboBox, QLabel, QPushButton

faulthandler.dump_traceback_later(900, exit=True)
stray_qt_errors()
ex = Explorer("run_status", lang="fr", size=(1600, 950))
EXC: list[str] = []
OBS: dict[str, dict] = {}


def hook(t, v, tb):
    EXC.append("".join(traceback.format_exception(t, v, tb)))
    print("!! UNCAUGHT", EXC[-1][-1500:], flush=True)


sys.excepthook = hook


def P(*a):
    print(*a, flush=True)


def texts(w):
    out = []
    if w is None:
        return out
    for x in w.findChildren(QLabel) + w.findChildren(QAbstractButton):
        try:
            if x.isVisible() and x.text():
                out.append(x.text())
        except RuntimeError:
            pass
    return out


def blk(ref, pid=None):
    return next((x for x in ex.blocks() if x.node.ref == ref and
                 (pid is None or (x.node.paths and x.node.paths[0].pid == pid))), None)


def refs():
    return [(x.node.ref, x.title) for x in ex.blocks()]


def place(*blocks):
    for b in blocks:
        ex.double_click_card(b)
        ex.settle(12)


def fresh(kind):
    root = ex.project(kind)
    ex.open(root)
    ex.settle(15)
    ex.close_dialogs()


def dlg():
    d = ex.w.run.dialog
    return d if d is not None and d.isVisible() else None


def shift_f5(ref=None, pid=None):
    """Click the block (when given), then Shift+F5 as the user does."""
    ex.close_dialogs()
    ex.w.run.dialog = None if ex.w.run.dialog is not None and not ex.w.run.dialog.isVisible() else ex.w.run.dialog
    if ref is not None:
        it = blk(ref, pid)
        if it is None:
            P("  !! no block", ref, refs())
            return None
        ex.click_block(it)
    ex.key(Qt.Key_F5, mod=Qt.ShiftModifier)
    ex.settle(10)
    return dlg()


def f5():
    ex.close_dialogs()
    ex.key(Qt.Key_F5)
    ex.settle(30)
    return dlg()


def run_button():
    ex.close_dialogs()
    ex.click(ex.w.bt_run)
    ex.settle(30)
    return dlg()


def dump(d):
    if d is None:
        return {"dialog": None}
    return {"title": d.windowTitle(), "rows": [r.name.text() for r in d.rows],
            "who": [r.who.text() for r in d.rows], "dur": [r.dur.text() for r in d.rows],
            "tips": [r.toolTip() for r in d.rows], "summary": d.summary.text(), "detail": d.detail.text(),
            "run": (d.b_run.isVisible(), d.b_run.isEnabled(), d.b_run.toolTip()), "texts": texts(d),
            "jobs": [j.key for j in d.plan.jobs] if d.plan is not None else None}


def inspect(ref, pid=None):
    ex.close_dialogs()
    ex.double_click_block(blk(ref, pid))
    ex.settle(15)
    page = ex.w.tune.pages.currentWidget()
    btns = [b for b in page.findChildren(QPushButton) if b.isVisible() and b.text().startswith("Jusqu")]
    return page, btns


def back():
    ex.key(Qt.Key_Escape, ex.w)
    ex.settle(4)
    ex.w.set_mode(0)
    ex.settle(6)


def obs(name, **kw):
    OBS.setdefault(name, {}).update(kw)
    P(f"  [{name}]", json.dumps(kw, ensure_ascii=False, default=str)[:3000])


def shot(name, widget=None):
    return ex.shot(f"r{name}", widget)


# ---------------------------------------------------------------------------------------------- checks
def c01_stack_only():
    fresh("session")
    place("stacking")
    d = f5()
    obs("01", f5=dump(d), shot=shot("01_f5_stacking_only", d or ex.w))
    ex.close_dialogs()
    d = shift_f5("empilement")
    obs("01", here=dump(d))
    ex.close_dialogs()
    page, btns = inspect("empilement")
    obs("01", inspect_btn=[(b.text(), b.toolTip(), b.isEnabled()) for b in btns])
    back()


def c02_failed_start():
    fresh("session")
    place("stacking")
    d = run_button()
    hist = [x for _t, x in ex.w.logp.rows()]
    obs("02", after_run=dump(d), status=ex.w.status.text(), hist=hist[:5],
        console=ex.w.logp.console.toPlainText()[:800], popups=ex.popups[-3:],
        shot=shot("02_failed_start", d or ex.w))
    ex.shot("r02_failed_start_window")


def c04_repeated():
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "gradient", "denoise", "gradient", "denoise")
    P("  refs", refs())
    d = shift_f5("debruitage~2")
    obs("04", here_d2=dump(d), tip=blk("debruitage~2").toolTip(), shot=shot("04_here_denoise2", d or ex.w))
    ex.close_dialogs()
    d = shift_f5("fond~2")
    obs("04", here_g2=dump(d))
    ex.close_dialogs()
    page, btns = inspect("debruitage~2")
    info = [(b.text(), b.toolTip(), b.isEnabled()) for b in btns]
    obs("05", inspect_d2=info, head=(texts(page) or [""])[0])
    if btns and btns[0].isEnabled():
        ex.click(btns[0])
        ex.settle(10)
        d = dlg()
        obs("05", click=dump(d), shot=shot("05_inspect_d2_runhere", d or ex.w))
        ex.close_dialogs()
    back()
    page, btns = inspect("debruitage")
    obs("06", inspect_d1=[(b.text(), b.toolTip(), b.isEnabled()) for b in btns])
    back()
    page, _b = inspect("fond")
    head = (texts(page) or [""])[0]
    strip = [x for x in texts(ex.w.tune.strip) if x.startswith("Gradient") or x.startswith("Débruit")]
    obs("07", head_g1=head, strip=strip, shot=shot("07_inspect_g1"))
    back()
    # pause on Gradient 1: words of the menu, the message, the log, the run window
    n = len(EXC)
    m = ex.right_click_block(blk("fond"), "Pause")
    ex.settle(8)
    obs("07", menu_g1=[a.get("text") for a in m], status=ex.w.status.text(),
        log=[x for _t, x in ex.w.logp.rows()][:3])
    d = f5()
    obs("07", f5_pause=dump(d))
    ex.close_dialogs()
    # pause on Gradient 2 (round 1: a Python error)
    m = ex.right_click_block(blk("fond~2"), "Pause")
    ex.settle(8)
    obs("08", menu_g2=[a.get("text") for a in m], exc=EXC[n:], status=ex.w.status.text(),
        log=[x for _t, x in ex.w.logp.rows()][:3], bp=[w.breakpoints for w in ex.s.p.workflows],
        shot=shot("08_pause_g2"))
    m = ex.right_click_block(blk("debruitage~2"), "Pause")
    ex.settle(8)
    obs("08", d2_exc=EXC[n:], bp2=[w.breakpoints for w in ex.s.p.workflows], status2=ex.w.status.text())


def c09_two_sessions():
    fresh("two_sessions")
    place("calibration", "registration", "stacking")
    P("  refs", refs())
    d = f5()
    obs("09", f5=dump(d), shot=shot("09_two_sessions_f5", d or ex.w))
    ex.close_dialogs()
    d = shift_f5("calibration@1")
    obs("11", here_cal1=dump(d), shot=shot("11_here_cal1", d or ex.w))
    ex.close_dialogs()
    page, btns = inspect("calibration@1")
    t = texts(page)
    obs("11", inspect=t[:12], btns=[(b.text(), b.toolTip(), b.isEnabled()) for b in btns])
    back()
    it = blk("calibration@1")
    m = ex.right_click_block(it, "Pause")
    ex.settle(8)
    obs("12", menu=[a.get("text") for a in m], bp=[w.breakpoints for w in ex.s.p.workflows],
        status=ex.w.status.text(), log=[x for _t, x in ex.w.logp.rows()][:3], shot=shot("12_pause_cal1"))
    d = f5()
    obs("12", f5_after_pause=dump(d))
    ex.close_dialogs()
    d = shift_f5("session@1") if blk("session@1") else None
    obs("29", here_session=dump(d), a_here=(ex.w.a_here.isEnabled(), ex.w.a_here.toolTip(), ex.w.a_here.text()),
        sel=ex.s.sel_step, refs=refs())
    ex.close_dialogs()


def c13_empty_wfs():
    fresh("two_flows")
    w = ex.w
    obs("13", status=w.status.text(), run=(w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip()),
        shot=shot("13_two_empty_flows"))
    obs("14", a_here=(w.a_here.isEnabled(), w.a_here.toolTip()), sel=ex.s.sel_step, refs=refs())
    d = shift_f5()
    obs("14", here=dump(d), status=w.status.text())
    ex.close_dialogs()
    d = f5()
    obs("13", f5=dump(d), status_after_f5=w.status.text())


def c15_session_no_wf():
    fresh("session")
    w = ex.w
    obs("15", run=(w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip()), status=w.status.text(),
        shot=shot("15_session_no_wf"))


def c16_no_lights():
    ex.new_project("SansBrutes2")
    ex.settle(10)
    place("session", "stacking")
    d = shift_f5("empilement")
    obs("16", here=dump(d), shot=shot("16_no_lights_here", d or ex.w))
    ex.close_dialogs()
    d = f5()
    obs("16", f5=dump(d), status=ex.w.status.text(), run=(ex.w.bt_run.text(), ex.w.bt_run.isEnabled(),
                                                           ex.w.bt_run.toolTip()))
    ex.close_dialogs()
    page, btns = inspect("empilement")
    obs("16", inspect=[(b.text(), b.toolTip(), b.isEnabled()) for b in btns], page=texts(page)[:10])
    obs("36", status=ex.w.status.text(), inspect_page=[x for x in texts(page) if "encore" in x or "brut" in x])
    back()


def c17_image_without_file():
    fresh("session")
    place("stacking")
    z = ex.new_flow_zone()
    ex.drag_palette("image", ex.ghost_center(z))
    ex.settle(15)
    d = f5()
    obs("17", f5=dump(d), shot=shot("17_image_without_file", d or ex.w))
    ex.close_dialogs()
    ex.new_project("ImageSansFichier2")
    ex.settle(10)
    place("image", "astrometry")
    page, btns = inspect("astrometrie")
    obs("18", btns=[(b.text(), b.toolTip(), b.isEnabled()) for b in btns], page=texts(page)[:12],
        shot=shot("18_inspect_astro_no_file"))
    back()


def c19_split():
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "astrometry", "gradient", "color", "denoise", "stars")
    P("  refs", refs())
    d = shift_f5("etoiles")
    obs("19", here=dump(d), shot=shot("19_here_split", d or ex.w))
    ex.close_dialogs()
    obs("20", title=getattr(blk("etoiles"), "title", None), cards=[c for c in ex.page.cards])
    d = shift_f5("composition") if blk("composition") else None
    obs("21", here=dump(d), shot=shot("21_here_composition", d or ex.w))
    ex.close_dialogs()


def c22_branch_log():
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "astrometry")
    ex.drag_palette_hover("gradient", ex.view.viewport().rect().center())
    gs = [(g.branch_key, ex.ghost_center(g)) for g in ex.ghosts() if getattr(g, "branch_key", None)]
    ex.drag_leave()
    tgt = next((c for k, c in gs if k == "astrometrie"), None)
    P("  branch ghosts", [k for k, _c in gs])
    if tgt is not None:
        ex.drag_palette("gradient", tgt)
        ex.settle(15)
    rows = [x for _t, x in ex.w.logp.rows()]
    ex.w.b_log.setChecked(True)
    ex.settle(5)
    obs("22", log=rows[:5], paths=[q.name for q in ex.s.paths], shot=shot("22_branch_log"))
    ex.w.b_log.setChecked(False)


def c23_device():
    from astro_gpu.client import prefs
    fresh("session")
    place("stacking")
    w = ex.w
    prefs.set("device", "auto")
    w.set_device("auto")
    ex.settle(6)
    items = [(w.device_cb.itemText(k), w.device_cb.itemData(k)) for k in range(w.device_cb.count())]
    auto, shown = w.bt_run.text(), w.device_cb.currentText()
    res = {"items": items, "auto": auto, "shown": shown}
    for data in ("cpu", "cuda:0"):
        i = w.device_cb.findData(data)
        if i >= 0:
            w.device_cb.setCurrentIndex(i)
            w.device_cb.activated.emit(i)
            ex.settle(8)
            res[data] = (w.bt_run.text(), w.device_cb.currentText())
    obs("23", **res, shot=shot("23_device"))
    seen = {}

    def look(d):
        seen["c"] = [[c.itemText(k) for k in range(c.count())] for c in d.findChildren(QComboBox) if c.isVisible()]
        seen["t"] = texts(d)
        d.reject()
    ex.dialog_handler = look
    w.open_settings("compute")
    ex.settle(8)
    obs("24", top=items, settings=seen.get("c"), settings_texts=seen.get("t"))
    i = w.device_cb.findData("auto")
    if i >= 0:
        w.device_cb.setCurrentIndex(i)
        w.device_cb.activated.emit(i)
    w.set_device("auto")


def c25_health():
    import numpy as np
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    from PIL import Image
    name = "fits_jpg_note2"
    root = ex.ws / name
    p = project2.create(root, name, Target(name="M 31"))
    p.stack_mode = "per_session"
    dd = ex.fits_frames(f"{name}/lights", 8)
    j = TMP / "data" / name / "phone" / "IMG_0001.JPG"
    j.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(j)
    p.sessions.append(project2.Session(1, "Session 1", lights=[str(dd), str(j)]))
    project2.save(p)
    ex.open(root)
    ex.settle(20)
    place("stacking")
    d = f5()
    obs("25", f5=dump(d), shot=shot("25_jpg_f5", d or ex.w))
    ex.close_dialogs()
    ex.double_click_block(blk("sources") or blk("session@1") or blk("session"))
    for _ in range(30):
        ex.settle(5)
    t = texts(ex.w.tune.pages.currentWidget())
    obs("26", inspect=t[:25], shot=shot("26_health"))
    back()
    ex.click_block(blk("sources") or blk("session@1") or blk("session"))
    ex.settle(8)
    obs("27", panel=[x for x in texts(ex.page.panel) if "/" in x or "utilis" in x or "Brut" in x][:6],
        shot=shot("27_panel"))


def c28_strip():
    fresh("image")
    ex.double_click_block(blk("image"))
    ex.settle(12)
    st = [x for x in texts(ex.w.tune.strip) if x != "›"]
    obs("28", strip=st, graph=refs(), shot=shot("28_strip"))
    back()


def c30_tooltips():
    fresh("image")
    ex.click_block(blk("image"))
    place("astrometry")
    d = f5()
    obs("30", f5=dump(d))
    ex.close_dialogs()


def c31_log_language():
    fresh("session")
    place("stacking", "astrometry")
    ex.w.change_language("en")
    ex.settle(10)
    rows_en = [x for _t, x in ex.w.logp.rows()]
    ex.w.b_log.setChecked(True)
    ex.settle(5)
    obs("31", log_en=rows_en[:4], shot=shot("31_log_en"))
    ex.w.b_log.setChecked(False)
    ex.w.change_language("fr")
    ex.settle(10)


def c32_fusion():
    fresh("mono")
    place("fusion")
    P("  refs", refs())
    fu = blk("fusion") or next((x for x in ex.blocks() if "fusion" in x.node.ref), None)
    obs("32", fusion_block=(fu.title, fu.toolTip()) if fu else None,
        run=(ex.w.bt_run.text(), ex.w.bt_run.isEnabled(), ex.w.bt_run.toolTip()), status=ex.w.status.text())
    d = f5()
    obs("32", f5=dump(d), shot=shot("32_fusion_f5", d or ex.w))
    ex.close_dialogs()
    page, btns = inspect("calibration@1")
    obs("33", head=(texts(page) or [""])[0])
    back()


def c34_recap_two_flows():
    fresh("two_flows")
    for sid in (1, 2):
        ex.click_block(next(x for x in ex.blocks() if x.node.key == "session" and x.node.sid == sid))
        place("stacking")
    d = run_button()
    obs("34", after=dump(d), shot=shot("34_recap_two_flows", d or ex.w))
    ex.close_dialogs()


def c35_recap_language():
    fresh("session")
    place("stacking")
    run_button()
    ex.w.change_language("en")
    ex.settle(10)
    d = ex.w.run.dialog
    obs("35", after=dump(d), visible=d.isVisible() if d else None, shot=shot("35_recap_en", d or ex.w))
    if d is not None and not d.isVisible():
        ex.w.run.show_dialog()
        ex.settle(5)
        obs("35", shown=dump(ex.w.run.dialog))
    ex.close_dialogs()
    ex.w.change_language("fr")
    ex.settle(10)


def run_checks(g: dict, prefix: str = "c", dump_name: str = "recheck_dump.json"):
    checks = {k: v for k, v in g.items() if callable(v) and k.startswith(prefix) and len(k) > 3 and k[1:3].isdigit()}
    for name in [a for a in sys.argv[1:] if not a.startswith("--")] or list(checks):
        fn = checks.get(name) or next((v for k, v in checks.items() if k.startswith(name)), None)
        P(f"\n######## {name}")
        n0 = len(EXC)
        try:
            fn()
        except Exception:  # noqa: BLE001
            P("!! CHECK EXC")
            traceback.print_exc(file=sys.stdout)
        P("  popups", ex.popups[-4:], "modals", ex.modals[-3:], "uncaught", len(EXC) - n0)
        if len(EXC) > n0:
            P("  EXC:", EXC[-1][-1500:])
        ex.popups.clear()
        ex.close_dialogs()
    out = ex.out / dump_name
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    old.update(OBS)
    out.write_text(json.dumps(old, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ex.close()
    P("END")


if __name__ == "__main__":
    run_checks(dict(globals()))
