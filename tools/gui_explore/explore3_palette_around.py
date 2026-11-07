"""palette_matrix, round 3, step 3: a short look around the fixes (Fusion pose/removal after other edits, the palette
target after a click in the void, undo of the new gestures).   python explore3_palette_around.py [case...]"""
from __future__ import annotations

import json
import sys

from explore3_palette_lib import Matrix

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

m = Matrix("around3_" + "_".join(sys.argv[1:])[:40])
m.watchdog(600)
ex = m.ex
ONLY = sys.argv[1:]
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def out(name, **kw):
    print("RESULT", name, json.dumps(kw, ensure_ascii=False, default=str)[:2500], flush=True)


def shot(name):
    return ex.shot("r3_" + name)


def fresh(kind):
    if kind.startswith("new_"):
        m._k = getattr(m, "_k", 0) + 1
        m.new_project(f"A{m._k}_{kind}", kind[4:])
    else:
        ex.open(ex.project(kind))
    ex.settle(8)
    m.hook_page()
    ex.popups.clear()
    return ex.s


def mono_astro():
    from astro_gpu.core import project2, workflow as wfl
    root = ex.project("mono")
    p = project2.load(root)
    for w in p.workflows:
        p.set_blocks(w.id, [wfl.node(b) for b in ("calibration", "registration", "stacking", "astrometry")])
    project2.save(p)
    ex.open(root)
    ex.settle(8)
    m.hook_page()


def flows():
    s = ex.s
    return [(q.name, list(s.sessions_of(q)), [n["block"] for n in s.graph(q)]) for q in s.paths]


def items(prefix, flow=None):
    return sorted([it for it in ex.blocks() if it.node.key.split("~")[0] == prefix and
                   (flow is None or flow in [q.name for q in it.node.paths])],
                  key=lambda i: (i.sceneBoundingRect().x(), i.sceneBoundingRect().y()))


def panel():
    return [x.text() for x in ex.page.panel.findChildren(QLabel) if x.isVisible() and x.text()][:3]


@case
def fusion_then_edit_then_remove():
    for a in (1, 2):
        mono_astro()
        f0 = flows()
        s0 = m.state()
        ex.double_click_card("fusion")
        sp = m.state()
        ex.undo()
        undo_exact = m.state() == s0
        ex.redo()
        redo_exact = m.state() == sp
        # an edit in a channel after the Fusion: a Gradient after Astrometry in Ha
        ex.click_block(items("astrometrie", "Ha")[0])
        ex.double_click_card("gradient")
        f1 = flows()
        t = len(m.toasts)
        ex.right_click_block(items("fusion")[0], "Supprimer")
        out("fusion_edit_remove", a=a, undo_exact=undo_exact, redo_exact=redo_exact, after_edit=f1, after_remove=flows(),
            before=f0, toasts=m.toasts[t:], comps=[it.title for it in items("composition")],
            shot=shot("fusion_edit_retrait") if a == 2 else None)
        ex.undo()
        out("fusion_edit_remove_undo", a=a, back=flows() == f1)


@case
def image_void_palette():
    for a in (1, 2):
        fresh("image")
        vp = ex.view.viewport()
        ex.click(vp, m.empty_point())
        t = len(m.toasts)
        ex.double_click_card("gradient")
        f1 = flows()
        ex.undo()
        out("image_void_dbl", a=a, flows=f1, toasts=m.toasts[t:], sel=ex.s.sel_step, panel=panel())
    for a in (1, 2):
        fresh("two_flows")
        vp = ex.view.viewport()
        ex.click(vp, m.empty_point())
        h = m.heading()
        r = {k: v for k, v in m.reasons().items() if k in ("calibration", "gradient")}
        t = len(m.toasts)
        ex.double_click_card("calibration")
        out("two_flows_void_dbl", a=a, heading=h, reasons=r, flows=flows(), toasts=m.toasts[t:])


@case
def nights_undo_chain():
    for a in (1, 2):
        fresh("new_nights")
        s0 = m.state()
        ex.click_block(ex.block("session") or ex.block("sources"))
        ex.double_click_card("session")
        ex.double_click_card("session")
        f2 = flows()
        ex.undo()
        one = flows()
        ex.undo()
        out("nights_two_sessions", a=a, after2=f2, undo1=one, undo2_exact=m.state() == s0,
            names=[x.name for x in ex.s.p.sessions])


@case
def pause_copy_undo():
    for a in (1, 2):
        fresh("image")
        ex.click_block(ex.block("image"))
        for b in ("astrometry", "gradient", "gradient"):
            ex.double_click_card(b)
        it = items("fond")[1]
        s0 = m.state()
        t = len(m.toasts)
        ex.right_click_block(it, "Pause")
        tt = m.toasts[t:]
        ex.undo()
        u = m.state() == s0
        # remove Gradient 1 while Gradient 2 has the pause
        ex.right_click_block(items("fond")[1], "Pause")
        t = len(m.toasts)
        ex.drag_block_to_palette(items("fond")[0])
        g = [(i.title, i.toolTip()[:120]) for i in items("fond")]
        try:
            pl, pause = ex.s.plan_run()
            plan = (len(pl.jobs), str(pause)[:120])
        except Exception as e:  # noqa: BLE001
            plan = f"raises {type(e).__name__}: {e}"
        out("pause_copy_undo", a=a, toast=tt, undo_exact=u, after_remove_first=g, toasts=m.toasts[t:], plan=plan)


for fn in CASES:
    if not ONLY or fn.__name__ in ONLY:
        m.case = fn.__name__
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            print("CASE FAILED", fn.__name__, type(e).__name__, e, traceback.format_exc()[-1500:], flush=True)
print("EXC", [x.splitlines()[0] for x in m.exc])
ex.close()
