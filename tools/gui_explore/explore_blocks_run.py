"""block_actions: one scenario (a project state and a graph built through the GUI), every block, every action.

    python explore_blocks_run.py <scenario> [lang]

Writes <out>/raw_<scenario>.json (every action: menus, panel, what changed, Ctrl+Z / Ctrl+Y, popups, toasts,
exceptions, overlaps). The anomalies are judged afterwards (explore_blocks_confirm.py notes them with screenshots)."""
from __future__ import annotations

import faulthandler
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(int(sys.argv[3]) if len(sys.argv) > 3 else 2400, exit=True)

from explore_blocks_lib import (EXC, TOASTS, Lab, bid, click_panel_button, diff, drawn, find,  # noqa: E402
                                overlaps, panel_buttons, panel_info, state)
from harness import Explorer  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QLineEdit  # noqa: E402

SC = sys.argv[1]
LANG = sys.argv[2] if len(sys.argv) > 2 else "fr"
ex = Explorer("block_actions", lang=LANG)
OUT = ex.out / f"raw_{SC}{'' if LANG == 'fr' else '_' + LANG}{'_ports' if os.environ.get('BA_ONLY_PORTS') else ''}.json"


# ------------------------------------------------------------------ gestures the harness lacks
def place(block):
    ex.double_click_card(block)
    ex.settle(6)


def select(key, pid=None, sid=None):
    ex.w.activateWindow()
    it = ex.block(key, pid=pid, sid=sid)
    if it is not None:
        ex.click_block(it)
    return it


def pid_of(name):
    return next(q.pid for q in ex.s.paths if q.name == name)


def branch(block, under_key, pid=None):
    """A palette block dropped on the "new branch" place under a block (as a user drags it)."""
    ex.drag_palette_hover(block, QPoint(5, 5))
    allg = [(getattr(g, "branch_key", None), g.pid) for g in ex.ghosts()]
    ghosts = [g for g in ex.ghosts() if getattr(g, "branch_key", None) == under_key and (pid is None or g.pid == pid)]
    pt = ex.ghost_center(ghosts[0]) if ghosts else None
    ex.drag_leave()
    if pt is None:
        print("no branch ghost for", block, under_key, allg, flush=True)
        return False
    rec = ex.drag_palette(block, pt)
    ex.settle(6)
    print("branch", block, under_key, rec, flush=True)
    return bool(rec and rec.get("accepted"))


def cross_point(it):
    from astro_gpu.client.graph import BH  # noqa: F401
    return ex.view.mapFromScene(it.mapToScene(QPointF(it.w - 12, 12)))


def click_cross(it):
    vp = ex.view.viewport()
    p = cross_point(it)
    ex.hover(vp, p)
    ex.click(vp, p)


def pull_port(it, y):
    vp = ex.view.viewport()
    a = ex.view.mapFromScene(it.mapToScene(QPointF(it.w, y)))
    post = ex.view.final_item
    b = ex.view.mapFromScene(post.sceneBoundingRect().center())
    ex.hover(vp, a)
    ex._mouse(vp, QEvent.MouseButtonPress, a)
    for t in (0.3, 0.6, 1.0):
        ex._mouse(vp, QEvent.MouseMove, QPoint(int(a.x() + (b.x() - a.x()) * t), int(a.y() + (b.y() - a.y()) * t)),
                  Qt.NoButton, Qt.LeftButton)
    ex._mouse(vp, QEvent.MouseButtonRelease, b, Qt.LeftButton, Qt.NoButton)
    ex.settle(6)


def accept_with_name(d):
    for e in d.findChildren(QLineEdit):
        if e.isVisible():
            e.setText("Nom essai")
            break
    d.accept()


# ------------------------------------------------------------------ the scenarios
def build_chain(keys):
    for b in keys:
        place(b)


FULL = ["calibration", "registration", "selection", "stacking", "astrometry", "gradient", "denoise", "color", "stars"]


def sc_chain():
    ex.open(ex.project("session", "chain"))
    build_chain(FULL)


def sc_chainb():                                       # the same chain, for the blocks a first run did not reach
    sc_chain()


def sc_sessions2():
    ex.open(ex.project("two_sessions", "sessions2"))
    select("session", sid=1) or select("sources")
    build_chain(["calibration", "registration", "selection", "stacking", "astrometry", "gradient", "color"])


def sc_repeat():
    ex.open(ex.project("session", "repeat"))
    build_chain(["calibration", "registration", "stacking", "astrometry", "gradient", "denoise", "gradient", "color"])
    # one more Gradient placed first among the image blocks (dropped in front of Astrometry)
    p = ex.s.paths[0]
    ex.page.place_block("gradient", p.pid, 3)
    ex.settle(6)
    place("color")


def sc_branch():
    ex.open(ex.project("session", "branch"))
    build_chain(["calibration", "registration", "stacking", "astrometry", "gradient"])
    branch("denoise", "empilement")
    select("astrometrie", pid=pid_of("Workflow"))         # the palette places in the chosen block's workflow
    branch("color", "astrometrie", pid=pid_of("Workflow"))


def sc_two_flows():
    ex.open(ex.project("two_flows", "twoflows"))
    for sid in (1, 2):
        it = ex.block("session", sid=sid)
        if it is not None:
            ex.click_block(it)
        build_chain(["calibration", "registration", "stacking"])


def sc_image():
    ex.open(ex.project("image", "image"))
    select("image")
    build_chain(["astrometry", "gradient", "color", "stars"])


def sc_image_empty():
    ex.open(ex.project("blank", "imageempty"))
    place("image")


def sc_blank_session():
    ex.open(ex.project("blank", "blanksession"))
    place("session")
    place("image")


def sc_mono():
    ex.open(ex.project("mono", "mono"))
    for name in ("Ha", "OIII"):
        it = ex.block("empilement", pid=pid_of(name))
        if it is not None:
            ex.click_block(it)
        place("astrometry")
    place("fusion")


def sc_dualband():
    ex.open(ex.project("session", "dualband"))
    build_chain(["calibration", "registration", "stacking", "astrometry"])
    place("fusion")


def sc_legacy():
    ex.open(ex.project("legacy", "legacy"))


def sc_jpg():
    ex.open(ex.project("jpg_mistake", "jpg"))
    build_chain(["calibration", "registration", "stacking"])


def sc_cut():
    ex.open(ex.project("session", "cut"))
    build_chain(["calibration", "registration", "stacking", "astrometry", "gradient", "color"])
    select("fond")
    ex.key(Qt.Key_D)
    select("astrometrie")
    ex.key(Qt.Key_D)


def sc_exits():
    ex.open(ex.project("session", "exits"))
    build_chain(["calibration", "registration", "stacking", "astrometry", "gradient", "color", "stars"])
    it = ex.block("fond")
    ex.right_click_block(it, pick="Post")
    ex.settle(6)
    it = ex.block("etoiles", pid=pid_of("Workflow"))
    if it is not None:
        ex.right_click_block(it, pick=None)
        items = ex.menus[-1] if ex.menus else []
        first = next((x["text"] for x in items if "Post" in x.get("text", "")), None)
        if first:
            ex.right_click_block(ex.block("etoiles", pid=pid_of("Workflow")), pick=first)
            ex.settle(6)


def sc_session_image():
    ex.open(ex.project("session+image", "sessimage"))
    select("sources") or select("session", sid=1)
    build_chain(["calibration", "registration", "stacking"])
    select("image")
    place("astrometry")


def sc_paused():
    ex.open(ex.project("session", "paused"))
    build_chain(["calibration", "registration", "stacking", "gradient"])
    it = ex.block("empilement")
    ex.right_click_block(it, pick="pause")       # "Pause here for review" (fr: ... pour examen)
    ex.settle(4)


def sc_session_empty_wf():
    ex.open(ex.project("session_empty_wf", "sessemptywf"))


SCENARIOS = {k[3:]: v for k, v in globals().items() if k.startswith("sc_")}


# ------------------------------------------------------------------ the actions on one block
def run_action(lab, b, nth, name, fn, dialog=None, files=None):
    """A fresh copy, the block found again, the action, then Ctrl+Z and Ctrl+Y: everything recorded."""
    lab.fresh()
    it = find(ex, b, nth)
    if it is None:
        lab.records.append({"block": list(b), "nth": nth, "action": name, "error": "block not found in the copy"})
        return None
    s0 = state(ex)
    m = lab.mark()
    if dialog is not None:
        ex.dialog_handler = dialog
    if files:
        ex.file_queue[:] = list(files)
    t = time.time()
    try:
        extra = fn(it)
    except Exception as e:  # noqa: BLE001
        import traceback
        EXC.append(f"script: {e}\n{traceback.format_exc()[-1500:]}")
        extra = None
    ex.dialog_handler = None
    ex.file_queue[:] = []
    ex.settle(6)
    dt = time.time() - t
    mode_after = ex.w.mode
    sel_after = (ex.s.sel_step, ex.s.path.name if ex.s.path is not None else None,
                 [i.title for i in ex.blocks() if i.selected])
    s1 = state(ex)
    pan = panel_info(ex)
    ov = overlaps(ex)
    dr = drawn(ex)
    if ex.w.mode != 0:
        ex.w.set_mode(0)
        ex.settle(4)
    ex.close_dialogs()
    ex.w.activateWindow()                     # offscreen, a closed menu leaves no active window (not in the real app)
    ex.settle(2)
    can_undo = ex.s.can_undo()
    ex.undo()
    ex.settle(4)
    s2 = state(ex)
    ov2 = overlaps(ex)
    ex.redo()
    ex.settle(4)
    s3 = state(ex)
    rec = {"block": list(b), "nth": nth, "action": name, "seconds": round(dt, 2), "extra": extra,
           "changed": diff(s0, s1), "undo_ok": s2 == s0, "undo_diff": diff(s0, s2), "redo_ok": s3 == s1,
           "redo_diff": diff(s1, s3), "can_undo": can_undo, "mode_after": mode_after, "sel_after": sel_after,
           "overlaps": ov, "overlaps_after_undo": ov2, "panel_after": pan, "drawn_after": dr, **lab.since(m)}
    lab.records.append(rec)
    lab.counts["actions"] += 1
    flag = []
    if rec["exc"]:
        flag.append("EXC")
    if rec["changed"] and not rec["undo_ok"]:
        flag.append("UNDO")
    if rec["changed"] and not rec["redo_ok"]:
        flag.append("REDO")
    if rec["popups"]:
        flag.append("POPUP")
    if ov:
        flag.append("OVERLAP")
    print(f"  {name:38s} {dt:5.1f}s changed={len(rec['changed'])} {' '.join(flag)} toasts={rec['toasts'][-1:]}",
          flush=True)
    return rec


def actions_for(lab, b, nth):
    """Every action on one block: the observations (click, hover, double click, right click), then each change on a
    fresh copy."""
    lab.fresh()
    it = find(ex, b, nth)
    if it is None:
        return
    lab.counts["blocks"] += 1
    obs = {"block": list(b), "nth": nth, "action": "observe", "title": it.title, "summary": it.summary,
           "tooltip": it.toolTip(), "removable": it.removable, "ports": [w for w, _y in it.ports],
           "port_ys": [(w, y) for w, y in it.ports]}
    m = lab.mark()
    ex.click_block(it)
    ex.settle(4)
    if find(ex, b, nth) is None:
        print("  block gone after a click", b, flush=True)
        lab.records.append({"block": list(b), "nth": nth, "action": "observe-lost", "error": "gone after click"})
        return
    obs["click_sel"] = (ex.s.sel_step, ex.s.path.name if ex.s.path is not None else None, list(ex.s.scope or []))
    obs["panel"] = panel_info(ex)
    obs["palette_head"] = [c.text() for c in ex.page.findChildren(type(ex.page.hint)) if c.isVisible()
                           and c.parent() is not None and "for" in c.text().lower()][:2]
    it = find(ex, b, nth)
    ex.hover(ex.view.viewport(), ex.point_of(it))
    it = find(ex, b, nth)
    obs["hover_tooltip"] = it.toolTip() if it is not None else None
    obs["menu"] = ex.right_click_block(it)
    it = find(ex, b, nth)
    if it is None:
        lab.records.append({"block": list(b), "nth": nth, "action": "observe-lost", "error": "gone after right click"})
        return
    ex.double_click_block(it)
    ex.settle(6)
    obs["dclick"] = {"mode": ex.w.mode, "sel": ex.s.sel_step, "path": ex.s.path.name if ex.s.path else None}
    try:
        tv = ex.w.tune
        obs["dclick"]["inspect_texts"] = [x.text() for x in tv.findChildren(type(ex.page.hint))
                                          if x.isVisible() and x.text()][:12]
    except Exception as e:  # noqa: BLE001
        obs["dclick"]["inspect_texts"] = f"error {e}"
    obs["dclick"]["shot"] = ex.shot(f"{SC}_{b[0]}_{nth}_inspect")
    ex.w.set_mode(0)
    ex.settle(4)
    obs.update(lab.since(m))
    lab.records.append(obs)
    print(f"[{SC}] {it.title if it else b} menu={[x.get('text') for x in obs['menu']]}", flush=True)

    ports = [(w, y) for w, y in obs["port_ys"]]
    if os.environ.get("BA_ONLY_PORTS"):              # a second pass: the ports only (each on a fresh copy)
        for what, y in ports:
            run_action(lab, b, nth, f"pull port {what} to Post", lambda it, y=y: pull_port(it, y))
        return
    menu = obs["menu"]
    for x in menu:
        t = x.get("text", "")
        if t == "---" or not x.get("enabled"):
            continue
        dlg = accept_with_name if any(w in t.lower() for w in ("renommer", "rename", "modèle", "template")) else None

        def pick(it, t=t):
            ex.right_click_block(it, pick=t)
            ex.settle(6)
        run_action(lab, b, nth, "menu: " + t, pick, dialog=dlg)
        if dlg is not None:                                    # and once cancelled
            run_action(lab, b, nth, "menu (annulé): " + t, pick)
    if obs["removable"]:
        run_action(lab, b, nth, "cross", lambda it: click_cross(it))
    run_action(lab, b, nth, "Del key", lambda it: (ex.click_block(it), ex.key(Qt.Key_Delete)))
    run_action(lab, b, nth, "D key", lambda it: (ex.click_block(it), ex.key(Qt.Key_D)))
    run_action(lab, b, nth, "D key twice", lambda it: (ex.click_block(it), ex.key(Qt.Key_D), ex.key(Qt.Key_D)))
    run_action(lab, b, nth, "drag to palette", lambda it: ex.drag_block_to_palette(it))
    for what, y in ports:
        run_action(lab, b, nth, f"pull port {what} to Post", lambda it, y=y: pull_port(it, y))
    # the panel's own buttons
    lab.fresh()
    it = find(ex, b, nth)
    if it is None:
        return
    ex.click_block(it)
    ex.settle(4)
    seen_t: dict = {}
    for t in panel_buttons(ex):
        k_t = seen_t.get(t, 0)
        seen_t[t] = k_t + 1
        files = None
        if "image" in t.lower() and ("choisir" in t.lower() or "changer" in t.lower() or "choose" in t.lower()
                                     or "change" in t.lower()):
            files = [str(ex.image_file("autre.fits"))]

        def press(it, t=t, k_t=k_t):
            ex.click_block(it)
            ex.settle(4)
            return click_panel_button(ex, t, k_t)
        dlg = accept_with_name if any(w in t.lower() for w in ("modèle", "template")) else None
        run_action(lab, b, nth, "panel: " + t + (f" #{k_t + 1}" if k_t else ""), press, dialog=dlg, files=files)
        if files:
            run_action(lab, b, nth, "panel (annulé): " + t, press)


def main():
    t0 = time.time()
    lab = Lab(ex, SC)
    SCENARIOS[SC]()
    ex.settle(10)
    ex.close_dialogs()
    built = {"scenario": SC, "state": state(ex), "drawn": drawn(ex), "overlaps": overlaps(ex),
             "popups_build": list(ex.popups), "modals_build": list(ex.modals), "exc_build": list(EXC),
             "toasts_build": list(TOASTS)}
    ex.shot(f"{SC}_built")
    lab.save_master()
    print("built", SC, [d["title"] for d in built["drawn"]], "overlaps", built["overlaps"], flush=True)
    seen: dict = {}
    targets = []
    for it in sorted(ex.blocks(), key=lambda i: (i.pos().x(), i.pos().y())):
        b = bid(it)
        targets.append((b, seen.get(b, 0)))
        seen[b] = seen.get(b, 0) + 1
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    for b, nth in targets:
        if only and b[0] not in only:
            continue
        actions_for(lab, b, nth)
        OUT.write_text(json.dumps({"built": built, "records": lab.records, "counts": lab.counts}, ensure_ascii=False,
                                  indent=1, default=str), encoding="utf-8")
    OUT.write_text(json.dumps({"built": built, "records": lab.records, "counts": lab.counts,
                               "seconds": time.time() - t0}, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    print("DONE", SC, lab.counts, round(time.time() - t0), "s", flush=True)
    ex.close()


main()
