"""palette_matrix, round 3, step 2: the points the first pass left unclear, and a look around the fixes.
    PM_LANG=fr|en PM_SIZE=WxH python explore3_palette_more.py [case...]"""
from __future__ import annotations

import json
import sys

from explore3_palette_lib import LANG, SIZE, Matrix

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtWidgets import QLabel

m = Matrix("more3_" + LANG + "_" + "_".join(sys.argv[1:])[:40])
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
    return ex.shot(f"r3_{LANG}_" + name)


def fresh(kind):
    if kind.startswith("new_"):
        m._k = getattr(m, "_k", 0) + 1
        m.new_project(f"M{m._k}_{kind}", kind[4:])
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
    return ex.s


def flows():
    s = ex.s
    return [(q.name, list(s.sessions_of(q)), json.dumps(q.source, default=str)[:80] if q.source else None,
             [n["block"] for n in s.graph(q)]) for q in s.paths]


def items(prefix):
    return sorted([it for it in ex.blocks() if it.node.key.split("~")[0] == prefix],
                  key=lambda i: (i.sceneBoundingRect().x(), i.sceneBoundingRect().y()))


def panel():
    return [x.text() for x in ex.page.panel.findChildren(QLabel) if x.isVisible() and x.text()][:3]


@case
def fusion2_reads():                   # what the second Fusion reads; a Session with a Fusion chosen
    for a in (1, 2):
        mono_astro()
        ex.double_click_card("fusion")
        ex.double_click_card("fusion")
        f = items("fusion")
        out("fusion2", a=a, flows=flows(), fus=[(it.title, it.summary, it.toolTip()[:200]) for it in f])
        ex.click_block(f[-1])
        out("fusion2_panel", a=a, panel=panel(), heading=m.heading(), shot=shot("fusion2_panneau") if a == 2 else None)
        # a Session with the Fusion chosen
        ex.click_block(items("fusion")[0])
        h = m.heading()
        t = len(m.toasts)
        ex.double_click_card("session")
        new = ex.s.p.sessions[-1]
        out("session_on_fusion", a=a, heading=h, new=(new.id, new.name, new.filter), flows=flows(),
            toasts=m.toasts[t:], shot=shot("session_sur_fusion") if a == 2 else None)


@case
def sel_after_undo():                  # N11
    for a in (1, 2):
        fresh("session_empty_wf")
        for b in ("calibration", "registration", "stacking"):
            ex.double_click_card(b)
        ex.undo()
        sel = ex.s.sel_step
        f = shot("sel_apres_ctrl_z") if a == 2 else None
        s0 = m.state()
        res = []
        for k in (Qt.Key_Delete, Qt.Key_D):
            t = len(m.toasts)
            ex.key(k)
            res.append((k, m.state() != s0, m.toasts[t:]))
            if m.state() != s0:
                ex.undo()
        out("sel_after_undo", a=a, sel=sel, panel=panel(), keys=res, shot=f)


@case
def small_drag():                      # N8 (run with PM_SIZE=1366x768)
    for a in (1, 2):
        fresh("legacy")
        vp = ex.view.viewport()
        ex.drag_palette_hover("gradient", QPoint(vp.width() // 2, vp.height() // 2))
        r = vp.rect()
        gh = [(g.text.strip()[:25], getattr(g, "branch_key", None), r.contains(ex.ghost_center(g))) for g in ex.ghosts()]
        post = ex.view.final_item
        post_in = r.contains(ex.view.mapFromScene(post.sceneBoundingRect().center())) if post else None
        f = shot(f"drag_{SIZE[0]}x{SIZE[1]}") if a == 2 else None
        # the drag near the right edge: does the view scroll?
        h0 = ex.view.horizontalScrollBar().value()
        ex.drag_palette_hover("gradient", QPoint(vp.width() - 5, vp.height() // 2))
        ex.settle(20)
        h1 = ex.view.horizontalScrollBar().value()
        ex.drag_leave()
        out("small_drag", a=a, size=SIZE, vp=(vp.width(), vp.height()), places=gh, visible=sum(1 for g in gh if g[2]),
            total=len(gh), post_visible=post_in, scroll=(h0, h1), shot=f)


@case
def words_en():                        # the new messages, in English (run with PM_LANG=en)
    for a in (1, 2):
        mono_astro()
        t = len(m.toasts)
        ex.double_click_card("fusion")
        ex.drag_block_to_palette(items("fusion")[0])
        fresh("new_nights")
        ex.click_block(ex.block("session") or ex.block("sources"))
        ex.double_click_card("session")
        fresh("legacy")
        ex.drag_block_to_palette(items("astrometrie")[0])
        ex.drag_block_to_palette(items("empilement")[0])
        ex.click_block(items("fond")[0])
        ex.key(Qt.Key_D)
        out("words", a=a, toasts=m.toasts[t:], wording={x: m_w for x in m.toasts[t:] if (m_w := __import__(
            "explore3_palette_lib").wording(x))}, reasons={k: v for k, v in m.reasons().items() if v})


@case
def recombiner_names():                # copies of Recombiner: numbering after a removal, and Ctrl+Z
    for a in (1, 2):
        fresh("legacy")
        ex.double_click_card("composition")
        ex.double_click_card("composition")
        n1 = [it.title for it in items("composition")]
        t = len(m.toasts)
        ex.drag_block_to_palette(items("composition")[1])
        n2 = [it.title for it in items("composition")]
        tt = m.toasts[t:]
        ex.undo()
        n3 = [it.title for it in items("composition")]
        out("recombiner_names", a=a, three=n1, after_remove=n2, toast=tt, undo=n3,
            layers=[x.get("name") for x in ex.s.post_layers()], shot=shot("recombiner_noms") if a == 2 else None)


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
