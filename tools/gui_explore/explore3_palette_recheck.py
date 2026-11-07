"""palette_matrix, round 3: each open round-2 point (rechecks not fixed + new anomalies) replayed twice with real
gestures; what is seen is printed (RESULT lines) and the screenshots go to exploration3/palette_matrix.
    python explore3_palette_recheck.py [case...]"""
from __future__ import annotations

import json
import sys
import time

from explore3_palette_lib import Matrix

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QAbstractButton, QLabel

m = Matrix("recheck3_" + "_".join(sys.argv[1:])[:40])
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
        m.new_project(f"R{m._k}_{kind}", kind[4:])
    else:
        ex.open(ex.project(kind))
    ex.settle(8)
    m.hook_page()
    ex.popups.clear()
    return ex.s


def mono_astro():
    """Ha and OIII, each Calibration → Alignement → Empilement → Astrométrie (what a Fusion needs)."""
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
    return [(q.name, list(s.sessions_of(q)), [n["block"] for n in s.graph(q)]) for q in s.paths]


def items(prefix):
    return sorted([it for it in ex.blocks() if it.node.key.split("~")[0] == prefix],
                  key=lambda i: (i.sceneBoundingRect().x(), i.sceneBoundingRect().y()))


def panel():
    return [x.text() for x in ex.page.panel.findChildren(QLabel) if x.isVisible() and x.text()][:3]


def toast_since(n):
    return m.toasts[n:]


@case
def composition():                     # R3, R5, R6
    for a in (1, 2):
        fresh("legacy")
        c0 = [(it.title) for it in items("composition")]
        t = len(m.toasts)
        s0 = m.state()
        ex.double_click_card("composition")
        c1 = [it.title for it in items("composition")]
        ch1 = m.state() != s0
        f1 = shot("combiner_1") if a == 2 else None
        ex.double_click_card("composition")
        c2 = [it.title for it in items("composition")]
        f2 = shot("combiner_2") if a == 2 else None
        names = [x.get("name") for x in ex.s.post_layers()]
        out("composition", a=a, at_open=c0, after1=c1, changed1=ch1, after2=c2, toasts=toast_since(t),
            reason=ex.card("composition").reason, layers=names, f1=f1, f2=f2)
        # Post's layer list as shown
        ex.w.set_mode(2)
        ex.settle(6)
        txt = [x for x in ex.texts() if "ecombin" in x or "Full" in x][:10]
        out("composition_post", a=a, texts=txt, shot=shot("combiner_post") if a == 2 else None)
        ex.w.set_mode(0)
        ex.settle(3)


@case
def image_open():                      # R16, N1
    for a in (1, 2):
        fresh("image")
        out("image_open", a=a, heading=m.heading(), sel=ex.s.sel_step, panel=panel(),
            reasons={k: v for k, v in m.reasons().items() if k in ("astrometry", "gradient", "calibration")},
            shot=shot("image_ouverture") if a == 2 else None)
        vp = ex.view.viewport()
        p = m.empty_point()
        ex.click(vp, p)
        out("image_open_click_empty", a=a, heading=m.heading(), sel=ex.s.sel_step, panel=panel(),
            reasons={k: v for k, v in m.reasons().items() if k in ("astrometry", "gradient")},
            hint=m.hint_of("gradient")[0], shot=shot("image_clic_vide") if a == 2 else None)
    # English too
    from astro_gpu.client import app as appmod


@case
def pause_copy():                      # N0
    def open_run(tag):
        b = ex.w.bt_run
        ex.click(b)
        for _ in range(10):
            ex.settle(5)
            time.sleep(0.05)
        d = ex.w.run.dialog if ex.w.run else None
        rows = [r.name.text() for r in d.rows] if d else None
        txt = [x.text() for x in d.findChildren(QLabel) + d.findChildren(QAbstractButton) if x.isVisible() and x.text()] \
            if d else None
        f = ex.shot(f"r3_pause_{tag}", d) if d else None
        if d:
            d.close()
            ex.settle(4)
        return rows, txt, f, b.text()
    for a in (1, 2):
        fresh("image")
        ex.click_block(ex.block("image"))
        for b in ("astrometry", "gradient", "gradient", "color", "color"):
            ex.double_click_card(b)
            ex.settle(4)
        for prefix, i in (("fond", 1), ("couleur", 1)):
            it = items(prefix)[i]
            ttl = it.title
            ex.right_click_block(it, "Pause")
            rows, txt, f, bt = open_run(f"a{a}_{prefix}{i + 1}")
            out("pause_copy", a=a, on=ttl, rows=rows, texts=(txt or [])[:8], run_button=bt, shot=f)
            ex.right_click_block(items(prefix)[i], "Pause")
    # complete workflow from raws: pause on Débruitage 3? legacy: pause on a copy of denoise
    for a in (1, 2):
        fresh("legacy")
        ex.click_block(items("debruitage")[0])
        ex.double_click_card("denoise")
        ex.settle(4)
        its = items("debruitage")
        it = its[-1]
        ttl = it.title
        ex.right_click_block(it, "Pause")
        rows, txt, f, bt = open_run(f"legacy_a{a}")
        out("pause_copy_legacy", a=a, on=ttl, all=[x.title for x in its], rows=rows, texts=(txt or [])[:8],
            run_button=bt, shot=f)


@case
def fusion_second():                   # N2, N3, N4, N12
    for a in (1, 2):
        fresh("mono")
        r = ex.card("fusion").reason
        h = m.hint_of("fusion")[0]
        s0 = m.state()
        t = len(m.toasts)
        ex.double_click_card("fusion")
        out("fusion_no_astrometry", a=a, reason=r, hint=h, changed=m.state() != s0, toasts=toast_since(t),
            popups=ex.popups[-2:], fus=[it.title for it in items("fusion")], run=(ex.w.bt_run.isEnabled(),
            ex.w.bt_run.toolTip()[:120]), shot=shot(f"fusion_sans_astro_{a}") if a == 2 else None)
    for a in (1, 2):
        mono_astro()
        f0 = flows()
        s0 = m.state()
        r = ex.card("fusion").reason
        t = len(m.toasts)
        ex.double_click_card("fusion")
        f1 = flows()
        tt1 = toast_since(t)
        fus1 = [(it.title, it.summary) for it in items("fusion")]
        f_one = shot(f"fusion_pose") if a == 2 else None
        t = len(m.toasts)
        ex.double_click_card("fusion")
        fus2 = [(it.title, it.summary) for it in items("fusion")]
        out("fusion_two", a=a, reason=r, before=f0, after1=f1, toast1=tt1, fus1=fus1, fus2=fus2, toast2=toast_since(t),
            flows2=[q.name for q in ex.s.paths], heading=m.heading(), comps=[it.title for it in items("composition")],
            f_one=f_one, shot=shot("fusion_deux") if a == 2 else None)
        # remove the second (by its menu), then the first by dragging to the palette
        its = items("fusion")
        if len(its) >= 2:
            t = len(m.toasts)
            ex.right_click_block(its[-1], "Supprimer" if ex.lang == "fr" else "Delete")
            out("fusion_two_del_second", a=a, fus=[it.title for it in items("fusion")], toasts=toast_since(t),
                flows=[q.name for q in ex.s.paths])
        its = items("fusion")
        if its:
            t = len(m.toasts)
            ex.drag_block_to_palette(its[0])
            out("fusion_removed", a=a, flows=flows(), same_as_before=flows() == f0, exact=m.state() == s0,
                toasts=toast_since(t), comps=[it.title for it in items("composition")],
                shot=shot("fusion_retiree") if a == 2 else None)


@case
def reasons_words():                   # N5, N6, N13, N17
    for a in (1, 2):
        fresh("session_empty_wf")
        r1 = {k: m.reasons()[k] for k in ("stars", "astrometry") if k in m.reasons()}
        fresh("legacy")
        r2 = {k: v for k, v in m.reasons().items() if v and ("seul" in v or "only" in v.lower())}
        fresh("image")
        ex.click_block(ex.block("image"))
        r3 = {k: m.reasons()[k] for k in ("registration", "stacking", "selection") if k in m.reasons()}
        fresh("new_image_stretched")
        ex.click_block(ex.block("image"))
        r4 = {k: m.reasons()[k] for k in ("registration", "astrometry", "stacking") if k in m.reasons()}
        out("reasons", a=a, empty_wf=r1, legacy=r2, image=r3, stretched=r4,
            shot=shot("raisons_etiree") if a == 2 else None)
        # removal messages
        fresh("legacy")
        msgs = []
        for key in ("astrometrie", "calibration", "empilement"):
            t = len(m.toasts)
            p0 = len(ex.popups)
            s0 = m.state()
            ex.drag_block_to_palette(items(key)[0])
            msgs.append((key, toast_since(t), ex.popups[p0:], m.state() != s0))
            if m.state() != s0:
                ex.undo()
        out("removal_msgs", a=a, msgs=msgs, shot=shot("retraits") if a == 2 else None)
        # off
        fresh("legacy")
        t = len(m.toasts)
        ex.click_block(items("fond")[0])
        ex.key(Qt.Key_D)
        tg = toast_since(t)
        ex.click_block(items("alignement")[0])
        ex.key(Qt.Key_D)
        out("off_words", a=a, toast_grad=tg, reason_reg=ex.card("registration").reason,
            titles=[(it.title, it.summary) for it in items("fond") + items("alignement")],
            shot=shot("off") if a == 2 else None)


@case
def branch_gradient():                 # N7, N16b
    for a in (1, 2):
        fresh("legacy")
        vp = ex.view.viewport()
        ex.drag_palette_hover("gradient", QPoint(vp.width() // 2, 4))
        gh = [(getattr(g, "branch_key", None), ex.ghost_center(g)) for g in ex.ghosts()]
        ex.drag_leave()
        tgt = next((g for g in gh if g[0] == "fond"), None)
        t = len(m.toasts)
        ex.drag_palette("gradient", tgt[1])
        out("branch_after_gradient", a=a, flows=[(q.name, [n.get("name") for n in ex.s.graph(q)]) for q in ex.s.paths],
            titles=[(it.title, [q.name for q in it.node.paths]) for it in items("fond")], toasts=toast_since(t),
            shot=shot("branche_gradient") if a == 2 else None)
        fresh("legacy")
        ex.drag_palette_hover("gradient", QPoint(vp.width() // 2, 4))
        gh = [(getattr(g, "branch_key", None), ex.ghost_center(g)) for g in ex.ghosts()]
        ex.drag_leave()
        tgt = next((g for g in gh if g[0] == "empilement"), None)
        t = len(m.toasts)
        if tgt:
            ex.drag_palette("gradient", tgt[1])
        out("branch_under_stacking", a=a, keys=[g[0] for g in gh], toasts=toast_since(t),
            flows=[q.name for q in ex.s.paths])
        # stretched image + stars: the time message
        fresh("new_image_stretched")
        ex.click_block(ex.block("image"))
        t = len(m.toasts)
        ex.double_click_card("stars")
        out("stretched_stars_toast", a=a, toasts=toast_since(t))


@case
def nights():                          # N9, N15
    for a in (1, 2):
        fresh("new_nights")
        f0 = flows()
        ex.click_block(ex.block("session") or ex.block("sources"))
        t = len(m.toasts)
        ex.double_click_card("session")
        out("nights_dbl", a=a, before=f0, after=flows(), toasts=toast_since(t),
            shot=shot("nuits_dbl") if a == 2 else None)
    for kind in ("mono", "new_sho"):
        for a in (1, 2):
            fresh(kind)
            h = m.heading()
            t = len(m.toasts)
            ex.double_click_card("session")
            new = ex.s.p.sessions[-1]
            out("session_filter", kind=kind, a=a, heading=h, new=(new.id, new.name, new.filter), flows=flows(),
                toasts=toast_since(t), shot=shot(f"session_filtre_{kind}") if a == 2 else None)


@case
def hint_sel():                        # N10, N11
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QEnterEvent
    from PySide6.QtWidgets import QApplication
    for a in (1, 2):
        fresh("session_empty_wf")
        for b in ("calibration", "registration", "stacking"):
            ex.double_click_card(b)
        ex.undo()
        sel_z = (ex.s.sel_step, panel(), [it.title for it in ex.blocks() if it.isSelected()])
        c = ex.card("astrometry")
        p = QPointF(c.width() / 2, c.height() / 2)
        QApplication.sendEvent(c, QEnterEvent(p, c.mapToGlobal(p), c.mapToGlobal(p)))
        ex.settle(2)
        shown = (ex.page.hint.isVisible(), ex.page.hint.text())
        ex.redo()
        ex.settle(6)
        out("hint_after_redo", a=a, shown_before=shown, reason_after=ex.card("astrometry").reason,
            hint_after=(ex.page.hint.isVisible(), ex.page.hint.text()), sel_after_undo=sel_z,
            sel_after_redo=(ex.s.sel_step, panel(), [it.title for it in ex.blocks() if it.isSelected()]),
            shot=shot("hint_ctrl_y") if a == 2 else None)


@case
def sho_template():                    # N14
    for a in (1, 2):
        fresh("new_sho")
        bl = [(it.node.key, it.title) for it in sorted(ex.blocks(), key=lambda i: (i.sceneBoundingRect().x(),
                                                                                    i.sceneBoundingRect().y()))]
        out("sho_open", a=a, blocks=bl, n_align=len(items("alignement")), sortie=[b for b in bl if "ortie" in b[1]
                                                                                 or "utput" in b[1]],
            shot=shot("sho_ouverture") if a == 2 else None)


@case
def small_screen():                    # N8
    pass


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
