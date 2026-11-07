"""palette_matrix, round 2, step 1: each round-1 anomaly replayed twice with real gestures, from a fresh project. Raw
results (what was seen, attempt 2) go to <out>/recheck_raw.jsonl with their screenshot; the verdict lines (recheck.jsonl)
are written after reading them.   python explore2_palette_recheck.py [case...]"""
from __future__ import annotations

import json
import sys

from explore_palette_lib import Matrix

from PySide6.QtCore import QPoint

m = Matrix("recheck2")
ex = m.ex
RAW = ex.out / "recheck_raw.jsonl"
ONLY = sys.argv[1:]
DONE = []


def case(fn):
    if not ONLY or fn.__name__ in ONLY:
        DONE.append(fn)
    return fn


def out(name, **kw):
    rec = {"case": name, **kw}
    with RAW.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print("RESULT", json.dumps(rec, ensure_ascii=False, default=str)[:1500], flush=True)


def shot(name):
    return ex.shot("rc_" + name)


def fresh(kind):
    if kind.startswith("new_"):
        tpl = {"new_blank": None, "new_full": "full", "new_nights": "nights", "new_sho": "sho",
               "new_stretched": "image_stretched", "new_image": "image", "new_raw": "raw"}[kind]
        m._k = getattr(m, "_k", 0) + 1
        root, _got = m.new_project(f"R{m._k}", tpl)
    else:
        root = ex.project(kind)
        ex.open(root)
    ex.settle(8)
    m.hook_page()
    ex.popups.clear()
    ex.modals.clear()
    return root


def paths():
    s = ex.s
    return [(q.name, [n["block"] for n in s.graph(q)]) for q in s.paths]


def item(key, nth=0):
    its = [it for it in ex.blocks() if it.node.key == key]
    its.sort(key=lambda it: (it.sceneBoundingRect().y(), it.sceneBoundingRect().x()))
    return its[nth] if len(its) > nth else None


@case
def image_undo_two_steps():
    for attempt in (1, 2):
        fresh("session+image")
        before = paths()
        b0 = m.state()
        ex.double_click_card("image")
        placed = paths()
        ex.undo()
        one = paths()
        exact = m.state() == b0
        ex.redo()
        redo = paths()
        f = shot("image_undo") if attempt == 2 else None
        out("image_undo", attempt=attempt, before=before, placed=placed, after_undo=one, exact=exact, redo=redo, shot=f)
    # by drag on the bottom zone
    fresh("session+image")
    b0 = m.state()
    z = ex.new_flow_zone()
    rec = ex.drag_palette("image", ex.view.mapFromScene(z.sceneBoundingRect().center())) if z else None
    placed = paths()
    ex.undo()
    out("image_undo_drag", rec=rec, placed=placed, after_undo=paths(), exact=m.state() == b0)


@case
def zone_not_lit():
    for attempt in (1, 2):
        fresh("session+image")
        res = {}
        for blk in ("session", "image", "fusion"):
            z = ex.new_flow_zone()
            if z is None:
                res[blk] = "no zone"
                continue
            pt = ex.view.mapFromScene(z.sceneBoundingRect().center())
            ex.drag_palette_hover(blk, pt)
            z = ex.new_flow_zone()
            res[blk] = (getattr(z, "hot", None), [(g.text, g.hot) for g in ex.ghosts()])
            if attempt == 2 and blk == "session":
                f = shot("zone_lit_session")
            ex.drag_leave()
        out("zone_not_lit", attempt=attempt, res=res, shot=f if attempt == 2 else None)


@case
def session_state_no_place():
    for attempt in (1, 2):
        fresh("session")
        reason = ex.card("calibration").reason
        vp = ex.view.viewport()
        ex.drag_palette_hover("calibration", QPoint(vp.width() // 2, vp.height() // 2))
        gh = [(g.text, ex.ghost_center(g)) for g in ex.ghosts()]
        f = shot("session_sans_wf_glisser") if attempt == 2 else None
        ex.drag_leave()
        tgt = gh[0][1] if gh else QPoint(vp.width() // 2, vp.height() // 2)
        rec = ex.drag_palette("calibration", tgt)
        after = paths()
        out("session_state_no_place", attempt=attempt, reason=reason, ghosts=[g[0] for g in gh], rec=rec, after=after,
            shot=f)


def comp_items():
    return [(it.node.lid, it.title) for it in ex.blocks() if it.node.key == "composition"]


@case
def composition_undo():
    for attempt in (1, 2):
        fresh("legacy")
        c0 = comp_items()
        r0 = ex.card("composition").reason
        ex.double_click_card("composition")
        c1 = comp_items()
        f1 = shot("combiner_1") if attempt == 2 else None
        ex.double_click_card("composition")
        c2 = comp_items()
        f2 = shot("combiner_2") if attempt == 2 else None
        undo_on = ex.w.a_undo.isEnabled()
        ex.undo()
        c3 = comp_items()
        ex.redo()
        c4 = comp_items()
        names = [x.get("name") for x in ex.s.post_layers()]
        out("composition", attempt=attempt, reason=r0, c0=c0, c1=c1, c2=c2, after_undo=c3, after_redo=c4,
            undo_on=undo_on, layer_names=names, shot1=f1, shot2=f2)


@case
def fusion_on_empty_colour():
    for attempt in (1, 2):
        fresh("session+image")
        reason = ex.card("fusion").reason
        hint = m.hint_of("fusion")[0]
        ex.double_click_card("fusion")
        fus = [q.name for q in ex.s.paths if q.source and "fusion" in q.source]
        f = shot("fusion_vide") if attempt == 2 else None
        out("fusion_empty", attempt=attempt, reason=reason, hint=hint, fusion=fus, shot=f)


@case
def session_dropped_anywhere():
    for attempt in (1, 2):
        res = {}
        for where in ("post", "block", "void"):
            fresh("session+image")
            n0 = len(ex.s.p.sessions)
            if where == "post":
                pt = ex.view.mapFromScene(ex.view.final_item.sceneBoundingRect().center())
            elif where == "block":
                pt = ex.point_of(item("image"))
            else:
                vp = ex.view.viewport()
                ex.drag_palette_hover("session", QPoint(vp.width() // 2, 4))
                pt = m.empty_point()
                ex.drag_leave()
            rec = ex.drag_palette("session", pt)
            res[where] = (rec and rec["accepted"], n0, len(ex.s.p.sessions), [t for t in m.toasts[-1:]])
        f = shot("session_sur_post") if attempt == 2 else None
        out("session_anywhere", attempt=attempt, res=res, shot=f)


@case
def multi_step_undo():
    for attempt in (1, 2):
        fresh("blank")
        b0 = m.state()
        ex.double_click_card("session")
        ex.undo()
        s_ok = m.state() == b0
        s_diff = m.diff(b0, m.state())
        fresh("session")
        b0 = m.state()
        ex.double_click_card("calibration")
        a1 = paths()
        ex.undo()
        c_ok = m.state() == b0
        c_diff = m.diff(b0, m.state())
        f = shot("calibration_ctrlz") if attempt == 2 else None
        out("multi_undo", attempt=attempt, session_ok=s_ok, session_diff=s_diff, cal_ok=c_ok, cal_diff=c_diff,
            placed=a1, shot=f)


def new_window():
    if ex.w is not None:
        ex.w.hide()
    ex.w = None
    ex.open()


@case
def template_session_zone():
    for attempt in (1, 2):
        new_window()
        fresh("new_full")
        e0 = len(m.exc)
        w0 = [(q.name, ex.s.sessions_of(q)) for q in ex.s.paths]
        z = ex.new_flow_zone()
        rec = ex.drag_palette("session", ex.view.mapFromScene(z.sceneBoundingRect().center())) if z else "no zone"
        ex.settle(6)
        w1 = [(q.name, ex.s.sessions_of(q)) for q in ex.s.paths]
        exc = [x.splitlines()[0] for x in m.exc[e0:]]
        sess = [x.name for x in ex.s.p.sessions]
        f = shot("modele_complet_zone") if attempt == 2 else None
        save_err = None
        try:
            ex.s.flush()
        except Exception as e:  # noqa: BLE001
            save_err = f"{type(e).__name__}: {e}"
        ex.undo()
        w2 = [(q.name, ex.s.sessions_of(q)) for q in ex.s.paths]
        out("template_zone", attempt=attempt, rec=rec, w0=w0, w1=w1, exc=exc, sess=sess, save=save_err,
            after_undo=w2, sess_after_undo=[x.name for x in ex.s.p.sessions], popups=ex.popups[-3:], shot=f)
    new_window()


@case
def calibration_second_flow():
    for attempt in (1, 2):
        fresh("two_flows")
        e0 = len(m.exc)
        vp = ex.view.viewport()
        ex.drag_palette_hover("calibration", QPoint(vp.width() // 2, 4))
        gh = [(g.pid, ex.ghost_center(g), g.toolTip(), g.text) for g in ex.ghosts()]
        ex.drag_leave()
        tgt = next((g for g in gh if g[0] == 2), None)
        if tgt is None:
            out("cal_flow2", attempt=attempt, ghosts=[(g[0], g[3]) for g in gh], error="no ghost for flow 2")
            continue
        rec = ex.drag_palette("calibration", tgt[1])
        ex.settle(6)
        exc = [x.splitlines()[0] for x in m.exc[e0:]]
        f = shot("deux_flux_cal_flux2") if attempt == 2 else None
        out("cal_flow2", attempt=attempt, rec=rec, exc=exc, paths=paths(), heading=m.heading(), sel=ex.s.sel_step,
            cur=ex.s.path.name if ex.s.path else None, ghosts=[(g[0], g[3], g[2]) for g in gh], shot=f)


@case
def mono_branch_ghost_overlap():
    for attempt in (1, 2):
        fresh("mono")
        vp = ex.view.viewport()
        ex.drag_palette_hover("astrometry", QPoint(vp.width() // 2, 4))
        gh = [(f"ghost:{g.text}[{g.pid}]", g.sceneBoundingRect()) for g in ex.ghosts()]
        over = [o for o in m.overlaps(gh) if "ghost" in o]
        f = shot("mono_branche") if attempt == 2 else None
        ex.drag_leave()
        out("mono_overlap", attempt=attempt, overlaps=over, ghosts=[g[0] for g in gh], shot=f)


@case
def zone_over_blocks():
    for attempt in (1, 2):
        found = {}
        for kind in ("session+image", "two_sessions"):
            fresh(kind)
            found[kind] = m.overlaps()
            if attempt == 2:
                found[kind + "_shot"] = shot(f"zone_{kind.replace('+', '_')}")
        fresh("session_empty_wf")
        ex.double_click_card("image")
        found["session_empty_wf+image"] = m.overlaps()
        if attempt == 2:
            found["img_shot"] = shot("zone_image_posee")
        out("zone_over_blocks", attempt=attempt, found=found)


@case
def zone_over_calibration2():
    for attempt in (1, 2):
        fresh("legacy")
        ex.double_click_card("session")
        o = m.overlaps()
        f = shot("zone_cal2") if attempt == 2 else None
        out("zone_cal2", attempt=attempt, overlaps=o, shot=f)


@case
def stretched_template():
    for attempt in (1, 2):
        fresh("new_stretched")
        s = ex.s
        p = s.paths[0] if s.paths else None
        f = shot("modele_etiree") if attempt == 2 else None
        out("stretched", attempt=attempt, wname=p.name if p else None, source=p.source if p else None,
            blocks=[n["block"] for n in s.graph(p)] if p else None,
            avail=[b for b, c in ex.page.cards.items() if not c.reason],
            reasons={b: c.reason for b, c in ex.page.cards.items() if c.reason}, shot=f)


@case
def empty_panel_at_open():
    from PySide6.QtWidgets import QLabel
    for attempt in (1, 2):
        fresh("image")
        pan = [x.text() for x in ex.page.panel.findChildren(QLabel) if x.isVisible() and x.text()]
        f = shot("panneau_ouverture") if attempt == 2 else None
        out("empty_panel", attempt=attempt, sel=ex.s.sel_step, node=str(ex.page.current_node()), panel=pan[:12],
            heading=m.heading(), shot=f)


@case
def session_orphan():
    for attempt in (1, 2):
        for kind in ("two_flows", "mono"):
            fresh(kind)
            ex.double_click_card("session")
            s = ex.s
            new = max(x.id for x in s.p.sessions)
            drawn = [it for it in ex.blocks() if it.node.sid == new and it.node.key in ("session", "sources")]
            f = shot(f"session_orpheline_{kind}") if attempt == 2 else None
            out("orphan", attempt=attempt, kind=kind, new=new, flows=[(q.name, s.sessions_of(q)) for q in s.paths],
                drawn=len(drawn), heading=m.heading(), shot=f)


@case
def composition_stale_redo():
    for attempt in (1, 2):
        fresh("legacy")
        n_g = lambda: len([n for n in ex.s.graph(ex.s.paths[0]) if n["block"] == "gradient"])  # noqa: E731
        ex.double_click_card("gradient")
        a = n_g()
        ex.undo()
        b = n_g()
        ex.double_click_card("composition")
        ex.double_click_card("composition")
        c = len(comp_items())
        redo_on = ex.w.a_redo.isEnabled() if hasattr(ex.w, "a_redo") else None
        ex.redo()
        d = n_g()
        e = len(comp_items())
        f = shot("combiner_ctrly") if attempt == 2 else None
        out("stale_redo", attempt=attempt, grad_after_place=a, grad_after_undo=b, comp=c, redo_enabled=redo_on,
            grad_after_redo=d, comp_after_redo=e, shot=f)


@case
def reasons():
    for attempt in (1, 2):
        got = {}
        fresh("new_image")
        got["stars"] = (ex.card("stars").reason, ex.card("denoise").reason, m.hint_of("stars")[0], paths())
        if attempt == 2:
            got["stars_shot"] = shot("raison_split")
        fresh("session_empty_wf")
        got["astro"] = (ex.card("astrometry").reason, m.hint_of("astrometry")[0])
        if attempt == 2:
            got["astro_shot"] = shot("raison_astrometrie")
        fresh("image")
        got["fusion"] = (ex.card("fusion").reason, m.hint_of("fusion")[0])
        if attempt == 2:
            got["fusion_shot"] = shot("raison_fusion")
        out("reasons", attempt=attempt, got=got)


m.watchdog(900)
for fn in DONE:
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        import traceback
        out(fn.__name__, crash=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
print("EXC:", [x.splitlines()[0] for x in m.exc])
ex.close()
