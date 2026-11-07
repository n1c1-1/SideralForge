"""Round 2, area "fuzz", step 1: every round-1 anomaly of the area replayed (twice, fresh projects) with explicit
gestures; one line per anomaly in exploration2/fuzz/recheck.jsonl.

    python explore2_fuzz_recheck.py [n ...]      # all, or only the numbers given (1..17)
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

sys.argv = [sys.argv[0], "run"] + sys.argv[1:]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import explore_fuzz as F  # noqa: E402

OUT = Path("F:/astro_essais/A_REGARDER/exploration2/fuzz")
REC = OUT / "recheck.jsonl"
fz = F.Fuzz("fr")
ex = fz.ex


# ------------------------------------------------------------------ helpers
def start(st, tag):
    fz.errs.clear()
    fz.start(st, tag)
    fz.errs.clear()


def step(name, fn):
    """One gesture, then every check of the fuzz: the failures of this gesture."""
    s = ex.w.s
    before = fz.state() if s is not None else None
    last = s._undo[-1] if s is not None and s._undo else None
    p0, m0 = len(ex.popups), len(ex.modals)
    fz._menus0, fz._popups0 = len(ex.menus), len(ex.popups)
    fz.step_fails = []
    try:
        fn()
    except Exception:  # noqa: BLE001
        fz.step_fails.append({"kind": "harness", "sig": "harness:" + name, "detail": traceback.format_exc()[-1500:]})
    fz.wait_ready()
    global AFTER
    AFTER = ([s.name for s in ex.s.p.sessions], [w.name for w in ex.s.p.workflows], titles(), said())
    try:
        fz.check(name, before, last, p0, m0)
    except Exception:  # noqa: BLE001
        fz.step_fails.append({"kind": "harness", "sig": "harness:check", "detail": traceback.format_exc()[-1500:]})
    return list(fz.step_fails)


def act(a):
    return step(a[0], lambda: fz.act(a))


def dbl(key):
    return step("dbl_card", lambda: ex.double_click_card(key))


def block(title_part, key=None):
    for it in fz.items():
        if key is not None and it.node.key != key:
            continue
        if title_part is None or title_part == fz.title(it) or title_part in fz.title(it):
            return it
    return None


def titles():
    return [fz.title(it) for it in fz.items()]


def wires():
    from astro_gpu.client.graph import WireItem
    return [x for x in ex.view.scene().items() if isinstance(x, WireItem)]


def wire_click_del(w):
    from PySide6.QtCore import Qt
    vp = ex.view.viewport()
    p = ex.view.mapFromScene(w.mapToScene(w.path().pointAtPercent(0.5)))
    ex.click(vp, p)
    ex.key(Qt.Key_Delete)


def wire_menu_remove(w):
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtWidgets import QApplication
    from astro_gpu.client.i18n import _ as tr
    vp = ex.view.viewport()
    p = ex.view.mapFromScene(w.mapToScene(w.path().pointAtPercent(0.5)))
    ex.menu_pick = tr("Remove")
    QApplication.sendEvent(vp, QContextMenuEvent(QContextMenuEvent.Mouse, p, vp.mapToGlobal(p)))
    ex.settle(6)


def drag_card_to(key, where):
    """A palette card dragged onto: "bottom" (new flow zone), "empty" (an empty spot), or a ghost whose text holds
    ``where``."""
    from PySide6.QtCore import QPoint
    vp = ex.view.viewport()
    if where == "bottom":
        z = ex.new_flow_zone()
        pt = ex.ghost_center(z) if z is not None else None
    elif where == "empty":
        pt = QPoint(int(vp.width() * 0.85), int(vp.height() * 0.15))
    else:
        ex.drag_palette_hover(key, QPoint(3, 3))
        gs = [g for g in ex.ghosts() if where in (g.toolTip() or getattr(g, "text", "") or "")]
        pt = ex.ghost_center(gs[0]) if gs else None
        ex.drag_leave()
    if pt is None:
        raise RuntimeError(f"no place {where!r} for {key}")
    return ex.drag_palette(key, pt)


def said():
    out = []
    try:
        t = ex.view.toast
        if t.isVisible() and t.text():
            out.append("toast: " + t.text())
    except Exception:  # noqa: BLE001
        pass
    try:
        if ex.w.status.text():
            out.append("status: " + ex.w.status.text())
    except Exception:  # noqa: BLE001
        pass
    return out


def sigs(fails):
    return sorted({f["sig"] + ("  " + f["detail"][-400:] if f["kind"] in ("harness", "exc", "validate") else "")
                   for f in fails}) + said() + \
        [f"wf={len(ex.s.p.workflows)} ses={len(ex.s.p.sessions)} drops={ex.drops[-1:] if ex.drops else None}"]


def has(fails, *kinds):
    return [f for f in fails if f["kind"] in kinds or any(f["sig"].startswith(k) for k in kinds)]


RESULTS = []


def record(n, title, status, note, shot=True, extra=None):
    f = ex.shot(f"recheck_{n:02d}") if shot else None
    rec = {"title": title, "status": status, "note": note, "shot": f}
    if extra:
        rec["_debug"] = extra
    with REC.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({k: v for k, v in rec.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
    print("RECHECK", n, status, title, "|", note, "|", extra, flush=True)


# ------------------------------------------------------------------ the 17 round-1 anomalies
def r1(k):
    start({"template": "sho", "lang": "fr"}, f"r1_{k}")
    t0 = titles()
    out = []
    for how in ("palette", "del", "menu"):
        if how != "palette":
            start({"template": "sho", "lang": "fr"}, f"r1_{k}{how}")
        it = block("Ha", "session") or block("Ha")
        if it is None:
            return ("?", f"pas de bloc Ha: {t0}")
        if how == "palette":
            fs = step("to_palette", lambda: ex.drag_block_to_palette(it))
            out.append(("état juste après (sans contrôle)", AFTER))
        elif how == "del":
            fs = step("del", lambda: (ex.click_block(it), ex.key(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Key_Delete)))
        else:
            fs = step("menu", lambda: ex.right_click_block(it, pick="Supprimer"))
        out.append((how, sigs(fs), titles(), ex.popups[-1:] if ex.popups else []))
    return out


def r2(k):
    start({"template": "raw", "lang": "fr"}, f"r2_{k}")
    t0 = titles()
    fs = step("drag_card", lambda: drag_card_to("session", "bottom"))
    return [t0, sigs(fs), titles(), [s.name for s in ex.s.p.sessions]]


def r3(k):
    start({"kind": "image", "lang": "fr"}, f"r3_{k}")
    ws = wires()
    tips = [(w.toolTip() or "").split("\n")[0] for w in ws]
    if not ws:
        return ["no wire", titles()]
    fs = step("wire", lambda: wire_menu_remove(ws[0]))
    return [tips, sigs(fs), len(wires()), ex.s.post_layers()]


def r4(k):
    out = []
    start({"template": "sho", "lang": "fr"}, f"r4_{k}")
    ws = wires()
    out.append(("fils", [(w.toolTip() or "").split("\n")[0] for w in ws], len(ex.s.post_layers())))
    if ws:
        fs = step("wire", lambda: wire_click_del(ws[0]))
        out.append(("fil retiré (clic+Suppr)", sigs(fs), len(wires()), len(ex.s.post_layers())))
        fz.undo()
        fz.wait_ready()
        out.append(("après Ctrl+Z", len(wires()), len(ex.s.post_layers())))
    start({"kind": "legacy", "lang": "fr"}, f"r4b_{k}")
    it = block(None, "astrometrie") or block("Astrom")
    if it is not None:
        fs = step("menu", lambda: ex.right_click_block(it, pick="Vers Post"))
        out.append(("Vers Post (menu)", sigs(fs), ex.menus[-1] if ex.menus else None))
    start({"kind": "legacy", "lang": "fr"}, f"r4c_{k}")
    it = block(None, "empilement") or block("Empilement")
    if it is not None:
        n0 = (len(wires()), len(ex.s.post_layers()))
        fs = step("pull_port", lambda: ex.pull_port_to_post(it))
        out.append(("port tiré sur Post", sigs(fs), n0, (len(wires()), len(ex.s.post_layers()))))
    return out


def r5(k):
    start({"template": "sho", "lang": "fr"}, f"r5_{k}")
    fs = step("drag_card", lambda: drag_card_to("session", "empty"))
    out = [sigs(fs), AFTER]
    start({"template": "sho", "lang": "fr"}, f"r5b_{k}")
    fs = step("drag_card", lambda: drag_card_to("session", "bottom"))
    out += [sigs(fs), AFTER]
    start({"template": "rgb", "lang": "fr"}, f"r5c_{k}")
    fs = step("dbl_card", lambda: ex.double_click_card("session"))
    out += [sigs(fs), AFTER]
    return out


def r6(k):
    out = []
    start({"kind": "two_flows", "lang": "fr"}, f"r6_{k}")
    out.append(sigs(dbl("fusion")))
    out.append(sigs(dbl("session")))
    out.append(titles())
    it = block("Session 2", "session")
    if it is not None:
        out.append(sigs(step("to_palette", lambda: ex.drag_block_to_palette(it))))
    out.append(titles())
    out.append(len(ex.s.p.sessions))
    # with a Fusion (it needs two filter flows): Ha and OIII, a Fusion, a session added, Ha taken out
    start({"kind": "mono", "lang": "fr"}, f"r6b_{k}")
    out.append(("mono fusion", sigs(dbl("fusion"))))
    out.append(("mono session", sigs(dbl("session"))))
    out.append(titles())
    it = block("Ha", "session")
    if it is not None:
        out.append(("Ha → palette", sigs(step("to_palette", lambda: ex.drag_block_to_palette(it))), AFTER))
    it = next((x for x in fz.items() if x.node.key == "session" and fz.title(x) not in ("Ha", "OIII")), None)
    if it is not None:
        out.append(("ajoutée → palette", sigs(step("to_palette", lambda: ex.drag_block_to_palette(it))), AFTER))
    return out


def r7(k):
    start({"kind": "session", "lang": "en"}, f"r7_{k}")
    a = sigs(step("drag_card", lambda: drag_card_to("session", "bottom")))
    start({"kind": "session", "lang": "en"}, f"r7b_{k}")
    b = sigs(dbl("image"))
    return [a, b]


def r8(k):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit
    start({"kind": "session", "lang": "en"}, f"r8_{k}")
    it = block(None, "session") or block(None, "sources")
    ex.click_block(it)
    eds = [e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisible() and e.text() == "Session 1"]
    if not eds:
        return ["no name field", [e.text() for e in ex.page.panel.findChildren(QLineEdit) if e.isVisible()]]
    w = eds[0]

    def typ():
        w.setFocus()
        w.selectAll()
        QTest.keyClick(w, Qt.Key_Delete)
        QTest.keyClicks(w, "Nuit 2 bis")
        QTest.keyClick(w, Qt.Key_Return)
        w.editingFinished.emit()
    fs = step("panel", typ)
    names = [s.name for s in ex.s.p.sessions]
    fz.undo()
    fz.wait_ready()
    after_undo = [s.name for s in ex.s.p.sessions]
    start({"kind": "session", "lang": "en"}, f"r8b_{k}")
    it = block(None, "session") or block(None, "sources")
    fs2 = step("menu", lambda: ex.right_click_block(it, pick="Rename session"))
    names2 = [s.name for s in ex.s.p.sessions]
    return [sigs(fs), names, after_undo, sigs(fs2), names2]


def r9(k):
    start({"kind": "two_sessions", "lang": "fr"}, f"r9_{k}")
    for b in ("calibration", "registration", "stacking", "gradient"):
        dbl(b)
    dbl("gradient")
    t = titles()
    # Inspect on the second Gradient, then the Calibration of session 2
    out = [t]
    for part in ("Gradient 2", "Calibration"):
        fz.ensure_graph([])
        it = block(part)
        if it is None:
            out.append(f"no {part}")
            continue
        ex.click_block(it)
        fz._menus0, fz._popups0 = len(ex.menus), len(ex.popups)
        fz.step_fails = []
        ex.click(ex.w.mode_btns[1])
        fz.wait_ready()
        fz.check_texts()
        from PySide6.QtWidgets import QAbstractButton
        tips = [(b.text(), b.toolTip()) for b in ex.w.findChildren(QAbstractButton) if b.isVisible() and
                ("ici" in b.text().lower() or "here" in b.text().lower())]
        out.append((part, sigs(fz.step_fails), tips))
    return out


def r10(k):
    from astro_gpu.client.i18n import _ as tr
    start({"template": "classic_dn", "lang": "en"}, f"r10_{k}")
    t0 = titles()
    from PySide6.QtCore import QPoint
    ex.drag_palette_hover("color", QPoint(3, 3))
    gtips = [(g.toolTip() or getattr(g, "text", "")) for g in ex.ghosts()]
    ex.drag_leave()
    same = next((t for t in gtips if "branch" not in t.lower() and "Color" in t and " in " in t), None)
    fs = step("drag_card", lambda: drag_card_to("color", same or "Color"))
    t1 = titles()
    out = [t0, gtips, same, sigs(fs), t1]
    for name in ("Color 2", "Color 1"):
        it = block(name)
        if it is None:
            out.append(f"no {name}")
            continue
        fs = step("menu", lambda: ex.right_click_block(it, pick=tr("Review pause")))
        out.append((name, "pause", sigs(fs), fz.ex.w.s.breakpoints if hasattr(ex.w.s, "breakpoints") else None))
    # a setting of Color 2 changed in its panel
    it = block("Color 2")
    if it is not None:
        ex.click_block(it)
        out.append(("panel Color 2", sigs(act(["panel", 0.5, 0.3, 0.7]))))
        out.append(("panel Color 2", sigs(act(["panel", 0.5, 0.6, 0.2]))))
    return out


def r11(k):
    start({"kind": "two_flows", "lang": "fr"}, f"r11_{k}")
    out = [sigs(dbl("registration"))]
    out.append(sigs(step("drag_card", lambda: drag_card_to("stacking", "Flow"))))
    out.append(sigs(dbl("calibration")))
    out.append(titles())
    return out


def r12(k):
    out = []
    start({"kind": "legacy", "lang": "fr"}, f"r12_{k}")
    out.append(sigs(dbl("image")))
    it = block("Image")
    if it is not None:
        ex.click_block(it)
    out.append(("undo", sigs(step("undo", fz.undo))))
    start({"kind": "legacy", "lang": "fr"}, f"r12b_{k}")
    out.append(sigs(dbl("session")))
    it = block("Session 2")
    if it is not None:
        ex.click_block(it)
    out.append(("undo", sigs(step("undo", fz.undo))))
    out.append(("redo", sigs(step("redo", fz.redo))))
    it = block("Session 2")
    if it is not None:
        ex.click_block(it)
        out.append(("suppr", sigs(step("del", lambda: ex.key(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Key_Delete)))))
    # the round-1 sequence itself
    seq = {"seed": None, "start": {"kind": "legacy", "lang": "fr"}, "actions": json.loads(
        Path("F:/astro_essais/A_REGARDER/exploration/fuzz/cases/exc_KeyError_project2_py_workflow.json").read_text(
            encoding="utf-8"))["actions"]}
    res = fz.run(seq, f"r12c_{k}", watchdog=60)
    out.append(("seq r1", sorted({f["sig"] for f in res["failures"]})))
    return out


def r13(k):
    start({"kind": "mono", "lang": "fr"}, f"r13_{k}")
    out = []
    for i in range(len(fz.items())):
        it = fz.items()[i]
        items = ex.right_click_block(it)
        it = fz.items()[i]
        lv = F._leaves(items)
        if any("workflow" in x.lower() for x in lv):
            out.append((fz.title(it), lv))
    if out:
        name, lv = out[0]
        it = block(name)
        dl = next((x for x in lv if "Supprimer" in x and "workflow" in x), None)
        if dl:
            n0 = len(ex.popups)
            fs = step("menu", lambda: ex.right_click_block(it, pick=dl))
            out.append(("picked", dl, sigs(fs), ex.popups[n0:], titles()))
    return out


def r14(k):
    start({"template": "image_stretched", "lang": "en"}, f"r14_{k}")
    out = [sigs(dbl("session"))]
    out.append(sigs(step("drag_card", lambda: drag_card_to("session", "empty"))))
    out.append(titles())
    start({"kind": "jpg_mistake", "lang": "fr"}, f"r14b_{k}")
    out.append(sigs(dbl("image")))
    out.append(sigs(dbl("image")))
    out.append(sigs(dbl("selection")))
    out.append(sigs(dbl("selection")))
    return out


def r15(k):
    from astro_gpu.client.i18n import _ as tr
    start({"template": "rgb", "lang": "fr"}, f"r15_{k}")
    out = [titles()]
    it = block("G", "session")
    fz.fileans = 0.4                                     # a FITS folder whose frames say FILTER = ...
    if it is not None:
        fs = step("add_session", lambda: ex.right_click_block(it, pick=tr("Add session…")))
        out.append((sigs(fs), [(s.name, s.filter) for s in ex.s.p.sessions], titles()))
    start({"template": "blank", "lang": "fr"}, f"r15b_{k}")
    from astro_gpu.client.graph import GhostItem
    gs = [g for g in ex.view.scene().items() if isinstance(g, GhostItem) and g.on_click is not None]
    out.append(("ghosts", [g.text for g in gs]))
    for g in gs[:1]:
        fz.fileans = 0.0
        fs = step("add_session", lambda: ex.click(ex.view.viewport(), ex.ghost_center(g)))
        out.append((g.text, sigs(fs), [(s.name, s.filter) for s in ex.s.p.sessions], titles()))
    return out


def r16(k):
    start({"template": "hoo", "lang": "fr"}, f"r16_{k}")
    out = [titles()]
    out.append(sigs(step("drag_card", lambda: drag_card_to("image", "bottom"))))
    it = block("Fusion", "fusion")
    if it is not None:
        out.append(sigs(step("menu", lambda: ex.right_click_block(it, pick="Supprimer"))))
    out.append(titles())
    it = next((x for x in fz.items() if x.node.key == "composition"), None)
    if it is not None:
        out.append(sigs(step("click_block", lambda: ex.click_block(it))))
    return out


def r17(k):
    start({"template": "image_stretched", "lang": "fr"}, f"r17_{k}")
    s = ex.s
    return [(q.name, (q.source or {}).get("stretched"), s.display_keys(q)) for q in s.paths]



if __name__ == "__main__":
    import faulthandler
    faulthandler.dump_traceback_later(900, exit=True)
    want = [int(x) for x in sys.argv[2:]] or list(range(1, 18))
    for n in want:
        fn = globals()[f"r{n}"]
        for k in range(2):
            try:
                res = fn(k)
            except Exception:  # noqa: BLE001
                res = "HARNESS " + traceback.format_exc()[-1500:]
            print(f"R{n}.{k}", json.dumps(res, ensure_ascii=False, default=str)[:3000], flush=True)
            ex.shot(f"recheck_{n:02d}_{k}")
    os._exit(0)
