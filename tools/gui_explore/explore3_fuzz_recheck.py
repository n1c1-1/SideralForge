"""Round 3 (control) fuzz: the round-2 fuzz points replayed with real gestures, twice; prints what is seen.

    python explore3_fuzz_recheck.py [point names...]
"""
import json
import os
import sys
import traceback
from pathlib import Path

ONLY = sys.argv[1:]
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_blocks_lib import TOASTS  # noqa: E402
import explore3_fuzz as G  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QSlider, QStyle, QStyleOptionSlider  # noqa: E402

fz = G.Fuzz2("fr")
fz.panels = False
ex = fz.ex
P = {}


def point(fn):
    P[fn.__name__] = fn
    return fn


def errs():
    e = [x.strip().splitlines()[-1] for x in fz.errs]
    fz.errs.clear()
    return e


def blk(pred):
    return next((x for x in fz.items() if pred(x)), None)


def by_title(t):
    return blk(lambda x: fz.title(x) == t)


def menu_undo():
    a = next((a2 for a in ex.w.menuBar().actions() if a.menu() is not None for a2 in a.menu().actions()
              if a2.text().replace("&", "").split("\t")[0] in ("Annuler", "Undo")), None)
    a.trigger()


def sessions():
    return [(s.id, s.name, s.filter, len(s.lights)) for s in ex.s.p.sessions]


def wfs():
    return [w.name for w in ex.s.p.workflows]


@point
def stars_card_reason():                                # A1
    for k in range(2):
        fz.start({"kind": "two_flows", "lang": "fr"}, f"a1_{k}")
        c = ex.card("stars")
        txt = [x.text() for x in c.findChildren(QLabel) if x.text()] if c is not None else None
        print(k, "card stars:", c is not None and c.isEnabled(), "| tip:", c.toolTip() if c else None, "| texts:", txt)
        if k == 1:
            ex.shot("r3_stars_card", ex.w)


@point
def fusion_session_to_palette():                        # A2 / R1
    for k in range(2):
        fz.start({"template": "sho", "lang": "fr"}, f"a2_{k}")
        fz.activate()
        s0, n = sessions(), len(TOASTS)
        ex.drag_block_to_palette(by_title("Ha"))
        fz.wait_ready()
        print(k, "drag Ha to palette:", TOASTS[n:], "| sessions", s0, "->", sessions(), "| errs", errs())
        n = len(TOASTS)
        ex.click_block(by_title("Ha"))
        fz.activate()
        ex.key(Qt.Key_Delete)
        fz.wait_ready()
        print("   Suppr on Ha:", TOASTS[n:], "| sessions", sessions(), "| drawn", [fz.title(x) for x in fz.items()][:6])
        if k == 1:
            ex.shot("r3_sho_ha_removed")


@point
def run_to_here_no_frames():                            # A3
    for k in range(2):
        fz.start({"template": "hoo", "lang": "fr"}, f"a3_{k}")
        ex.key(Qt.Key_F5, ex.w, Qt.NoModifier)
        ex.settle(8)
        f5 = ([d.__class__.__name__ for d in ex.dialogs()], ex.w.status.text())
        ex.close_dialogs()
        n = len(TOASTS)
        ex.click_block(blk(lambda x: x.node.key == "composition"))
        ex.key(Qt.Key_F5, ex.w, Qt.ShiftModifier)
        ex.settle(8)
        ds = [d.__class__.__name__ for d in ex.dialogs()]
        print(k, "F5:", f5, "| Shift+F5 dialogs:", ds, "| status:", ex.w.status.text(), "| toasts:", TOASTS[n:])
        if k == 1:
            ex.shot("r3_run_to_here")
        ex.close_dialogs()


@point
def panel_field_gone_block():                           # A4
    for k in range(2):
        fz.start({"template": "hoo", "lang": "fr"}, f"a4a_{k}")
        fz.activate()
        errs()
        ex.click_block(blk(lambda x: x.node.key == "fusion"))
        b = next((b for b in ex.page.panel.findChildren(QPushButton) if b.isVisible() and b.text() == "Supprimer"), None)
        n = len(TOASTS)
        if b is not None:
            ex.click(b)
        fz.wait_ready()
        print(k, "(1) HOO Fusion panel Supprimer:", b is not None, TOASTS[n:], "| errs", errs())
        fz.start({"kind": "two_flows", "lang": "fr"}, f"a4b_{k}")
        fz.activate()
        ex.double_click_card("session")
        fz.wait_ready()
        it = blk(lambda x: fz.title(x) == "Session 3") or blk(lambda x: x.node.key == "session" and "3" in fz.title(x))
        print("   sessions after dbl Session card:", sessions(), "| toasts", TOASTS[-1:])
        if it is not None:
            ex.click_block(it)
            e = next((w for w in ex.page.panel.findChildren(QLineEdit) if w.isVisible() and w.text() == fz.title(it)), None)
            if e is not None:
                errs()
                ex.click(e)
                e.setFocus()
                e.selectAll()
                QTest.keyClicks(e, "Nuit du 12")
                ex.settle()
                menu_undo()
                fz.wait_ready()
                ex.settle(6)
                print("   (2) new session name typed + Edition>Annuler: errs", errs(), "| sessions", sessions())
        fz.start({"template": "hoo", "lang": "fr"}, f"a4c_{k}")
        fz.activate()
        ha = by_title("Ha")
        ex.click_block(ha)
        e = next((w for w in ex.page.panel.findChildren(QLineEdit) if w.isVisible() and w.text() == "Ha"), None)
        if e is not None:
            errs()
            ex.click(e)
            e.setFocus()
            e.selectAll()
            QTest.keyClicks(e, "Nuit Ha")
            ex.settle()
            menu_undo()
            fz.wait_ready()
            ex.settle(6)
            print("   (3) template session name typed + Edition>Annuler: errs", errs(), "| sessions", sessions(),
                  "| workflows", wfs())


@point
def template_first_undo():                              # A5
    for k in range(2):
        for tpl in ("classic_dn", "sho"):
            fz.start({"template": tpl, "lang": "fr"}, f"a5_{tpl}_{k}")
            w0 = wfs()
            n = len(TOASTS)
            fz.undo()
            fz.wait_ready()
            print(k, tpl, "before", w0, "| after first Ctrl+Z", wfs(), sessions(), "| undo enabled",
                  ex.w.a_undo.isEnabled(), "| toasts", TOASTS[n:], "| status", ex.w.statusBar().currentMessage())


@point
def offview():                                          # A6
    ex.open()
    for size in ((1366, 768), (1600, 950)):
        ex.w.resize(*size)
        ex.settle(4)
        for k in range(2):
            for tpl in ("sho", "hoo", "classic_dn"):
                fz.start({"template": tpl, "lang": "fr"}, f"a6{size[0]}_{k}_{tpl}")
                ex.settle(6)
                vis = ex.view.mapToScene(ex.view.viewport().rect()).boundingRect()
                out = [fz.title(it) for it in fz.items() if not vis.contains(it.sceneBoundingRect())]
                print(size, k, tpl, "hors vue:", out, "| zoom", round(ex.view.transform().m11(), 2), flush=True)
                if k == 1 and tpl == "sho":
                    ex.shot(f"r3_offview_{tpl}_{size[0]}")
    ex.w.resize(1600, 950)


@point
def drop_files():                                       # A7
    for k in range(2):
        for title, start, acts in (("session", {"kind": "session", "lang": "fr"}, [["drop_urls", 0.0, 0.0, 0.0]]),
                                   ("raw", {"template": "raw", "lang": "fr"}, [["drop_urls", 0.0, 0.0, 0.3]])):
            fz.start(start, f"a7_{title}_{k}")
            s0 = sessions()
            res = fz.run({"seed": None, "start": start, "actions": acts}, f"a7r_{title}_{k}", watchdog=60)
            print(k, title, "| steps", res["steps"], "| fails", [(f["sig"], f["detail"][:300]) for f in res["failures"]],
                  "| sessions", s0, "->", sessions(), "| toasts", TOASTS[-2:])
            if k == 1 and title == "raw":
                ex.shot("r3_drop_files")


@point
def stacking_menu():                                    # A8
    for k in range(2):
        for tpl in ("full", "classic", "nights", "calibrated", "aligned"):
            fz.start({"template": tpl, "lang": "fr"}, f"a8_{tpl}_{k}")
            fz.activate()
            it = blk(lambda x: x.node.key == "stacking")
            m = ex.right_click_block(it) if it is not None else None
            ex.settle(4)
            print(k, tpl, "Empilement menu:", [x.get("text") for x in m] if m else m)
            if k == 1 and tpl == "full":
                ex.shot("r3_stacking_menu")


@point
def delete_workflow_fusion():                           # A9
    for k in range(2):
        for tpl in ("lrgb", "hoo", "sho", "rgb"):
            fz.start({"template": tpl, "lang": "fr"}, f"a9_{tpl}_{k}")
            fz.activate()
            first = ex.s.p.workflows[0]
            cands = [x for x in fz.items() if first.id in [q.pid for q in x.node.paths] and x.node.key != "session"]
            hit = None
            for it in reversed(cands):
                m = ex.right_click_block(it)
                ex.settle(3)
                if any("Supprimer workflow" in (x.get("text") or "") for x in m):
                    hit = (it, [x for x in m if "Supprimer workflow" in (x.get("text") or "")][0])
                    break
            if hit is None:
                print(k, tpl, "no 'Supprimer workflow' in the menus of", first.name)
                continue
            n, w0 = len(TOASTS), wfs()
            errs()
            ex.right_click_block(hit[0], pick="Supprimer workflow")
            fz.wait_ready()
            w1 = wfs()
            fz.undo()
            fz.wait_ready()
            fz.redo()
            fz.wait_ready()
            try:
                ex.s.p.validate()
                ok = "valide"
            except Exception as e:  # noqa: BLE001
                ok = f"INVALIDE {e}"
            print(k, tpl, f"on {fz.title(hit[0])} ({first.name}): item {hit[1]}; toasts {TOASTS[n:]}; workflows {w0} -> "
                  f"{w1}; after Z/Y {wfs()} {ok}; errs {errs()}")
            if k == 1 and tpl == "lrgb":
                ex.shot("r3_delete_wf_fusion")


@point
def sortie_blocks():                                    # A10
    for k in range(2):
        fz.start({"template": "sho", "lang": "fr"}, f"a10_{k}")
        t = [fz.title(x) for x in fz.items()]
        print(k, "drawn SHO:", t, "| Sortie drawn:", [x for x in t if "Sortie" in x or "Output" in x])


@point
def two_images():                                       # A11
    for k in range(2):
        fz.start({"kind": "blank", "lang": "fr"}, f"a11_{k}")
        fz.activate()
        ex.double_click_card("image")
        fz.wait_ready()
        ex.double_click_card("image")
        fz.wait_ready()
        imgs = [x for x in fz.items() if x.node.key == "image"]
        print(k, "workflows", wfs(), "| Image blocks", [(fz.title(x), [q.name for q in x.node.paths]) for x in imgs],
              "| toasts", TOASTS[-2:])
        if len(imgs) >= 2:
            n = len(TOASTS)
            ex.click_block(imgs[0])
            fz.activate()
            ex.key(Qt.Key_Delete)
            fz.wait_ready()
            print("   Suppr on the first Image:", TOASTS[n:], "| workflows", wfs())
        if k == 1:
            ex.shot("r3_two_images")


@point
def recombine_slider():                                 # A13
    def sliders():
        return json.dumps([lay.get("sliders") for lay in ex.s.post_layers()])
    for k in range(2):
        fz.start({"template": "sho", "lang": "fr"}, f"a13_{k}")
        fz.activate()
        ex.click_block(blk(lambda x: x.node.key == "composition"))
        sl = [w for w in ex.page.panel.findChildren(QSlider) if w.isVisible()][0]
        n0, s0 = len(ex.s._undo), sliders()
        opt = QStyleOptionSlider()
        sl.initStyleOption(opt)
        a = sl.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, sl).center()
        QTest.mousePress(sl, Qt.LeftButton, Qt.NoModifier, a)
        for dx in (10, 25, 40):
            QTest.mouseMove(sl, QPoint(a.x() + dx, a.y()))
            ex.settle(1)
        QTest.mouseRelease(sl, Qt.LeftButton, Qt.NoModifier, QPoint(a.x() + 40, a.y()))
        fz.wait_ready()
        n1, s1 = len(ex.s._undo), sliders()
        fz.undo()
        fz.wait_ready()
        s2 = sliders()
        print(k, "drag: +", n1 - n0, "entries | back to before after one Ctrl+Z:", s2 == s0, "|", s0, "|", s1, "|", s2)


@point
def fusion_delete_hoo():                                # R16
    for k in range(2):
        fz.start({"template": "hoo", "lang": "fr"}, f"r16_{k}")
        fz.activate()
        t0 = [fz.title(x) for x in fz.items()]
        n = len(TOASTS)
        ex.click_block(blk(lambda x: x.node.key == "fusion"))
        fz.activate()
        ex.key(Qt.Key_Delete)
        fz.wait_ready()
        t1 = [fz.title(x) for x in fz.items()]
        comp = blk(lambda x: x.node.key == "stars" or "toiles" in fz.title(x))
        fz.undo()
        fz.wait_ready()
        print(k, "Suppr Fusion:", TOASTS[n:], "| drawn before", t0, "| after", t1, "| after Ctrl+Z same:",
              [fz.title(x) for x in fz.items()] == t0, "| errs", errs())


for name, fn in P.items():
    if ONLY and name not in ONLY:
        continue
    print("=====", name, flush=True)
    try:
        fn()
    except Exception:  # noqa: BLE001
        print(traceback.format_exc()[-1500:], flush=True)
    sys.stdout.flush()
os._exit(0)
