"""palette_matrix round 2: repeated blocks (Gradient, Denoise, Color placed twice): numbered by position, the same name
everywhere (graph, panel, menus, pause, run window, Inspect), independent, each gesture undone/redone exactly.
    PM_LANG=fr|en python explore2_palette_repeat.py [block...]"""
from __future__ import annotations

import json
import sys

from explore2_palette_lib import LANG, Matrix

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QAbstractButton, QLabel

m = Matrix(f"repeat_{LANG}")
ex = m.ex
m.watchdog(600)
BLOCKS = sys.argv[1:] or ["gradient", "denoise", "color"]
ENGINE = {"gradient": "fond", "denoise": "debruitage", "color": "couleur"}


def P(*a):
    print(*a, flush=True)


def items(eng):
    its = [it for it in ex.blocks() if it.node.key.split("~")[0] == eng]
    return sorted(its, key=lambda it: it.sceneBoundingRect().x())


def titles(eng):
    return [(it.node.key, it.title) for it in items(eng)]


def panel_title():
    labs = [x.text() for x in ex.page.panel.findChildren(QLabel) if x.isVisible() and x.text()]
    return labs[0] if labs else None


def vis(w):
    out = []
    for x in w.findChildren(QLabel) + w.findChildren(QAbstractButton):
        try:
            if x.isVisible() and x.text():
                out.append(x.text().replace("\n", " / "))
        except RuntimeError:
            pass
    return out


def everywhere(eng, tag):
    """The names of the copies in the graph, the panel (each chosen), the menu, the pause badge."""
    res = {"graph": titles(eng)}
    res["panel"] = []
    res["menu"] = []
    res["tips"] = [it.toolTip()[:160] for it in items(eng)]
    for i in range(len(items(eng))):
        ex.click_block(items(eng)[i])
        res["panel"].append((ex.s.sel_step, panel_title()))
    for i in range(len(items(eng))):
        res["menu"].append([a.get("text") for a in ex.right_click_block(items(eng)[i]) if a.get("text") != "---"])
    # Inspect (double-click): the name it shows
    ex.double_click_block(items(eng)[-1])
    ex.settle(8)
    res["inspect"] = [t for t in vis(ex.w) if "radient" in t or "ébruit" in t or "enoise" in t or "ouleur" in t
                      or "olor" in t][:8]
    ex.shot(f"rep_{eng}_{LANG}_inspect")
    ex.w.set_mode(0)
    ex.settle(6)
    P(tag, json.dumps(res, ensure_ascii=False))
    return res


for blk in BLOCKS:
    eng = ENGINE[blk]
    P("=====", blk)
    for attempt in (1, 2):
        ex.open(ex.project("legacy"))
        ex.settle(8)
        m.hook_page()
        s0 = m.state()
        t0 = titles(eng)
        reason = ex.card(blk).reason
        P("start", t0, "card reason", repr(reason))
        # 1. double-click: a second copy
        ex.double_click_card(blk)
        s1 = m.state()
        t1 = titles(eng)
        P("after dbl", t1, "sel", ex.s.sel_step, "toasts", m.toasts[-1:])
        if attempt == 2:
            ex.shot(f"rep_{blk}_{LANG}_dbl")
            ev = everywhere(eng, "everywhere")
        ex.undo()
        P("undo exact", m.state() == s0, titles(eng))
        ex.redo()
        P("redo exact", m.state() == s1, titles(eng))
        # 2. a copy dropped on the first place (before everything else of the image phase): renumbered by position?
        vp = ex.view.viewport()
        ex.drag_palette_hover(blk, QPoint(vp.width() // 2, 4))
        gh = [(g.text, g.toolTip(), getattr(g, "idx", None), getattr(g, "branch_key", None), ex.ghost_center(g))
              for g in ex.ghosts()]
        if attempt == 2:
            ex.shot(f"rep_{blk}_{LANG}_ghosts")
        ex.drag_leave()
        P("ghosts", [(g[0], g[1], g[2], g[3]) for g in gh])
        ins = [g for g in gh if g[3] is None]
        if ins:
            first = min(ins, key=lambda g: g[4].x())
            s2a = m.state()
            rec = ex.drag_palette(blk, first[4])
            P("drop on first place", first[:3], rec, titles(eng), "sel", ex.s.sel_step, "panel", panel_title())
            if attempt == 2:
                ex.shot(f"rep_{blk}_{LANG}_three")
            s2 = m.state()
            ex.undo()
            P("undo exact", m.state() == s2a)
            ex.redo()
            P("redo exact", m.state() == s2)
        # 3. pause on the last copy: the badge / menu name
        its = items(eng)
        if its:
            menu = ex.right_click_block(its[-1], "Pause")
            P("pause menu", [a.get("text") for a in menu if a.get("text") != "---"])
            P("after pause", [(it.title, it.toolTip()[:120]) for it in items(eng)], "toasts", m.toasts[-2:])
        # 4. Run (blocked): the run window's rows
        b = ex.w.bt_run
        P("run button", b.text(), b.isEnabled())
        if b.isEnabled():
            ex.click(b)
            ex.settle(10)
            d = ex.w.run.dialog
            if d is not None:
                try:
                    rows = [r.name.text() for r in d.rows]
                except Exception as e:  # noqa: BLE001
                    rows = repr(e)
                P("run rows", rows)
                P("run texts", vis(d)[:30])
                if attempt == 2:
                    ex.shot(f"rep_{blk}_{LANG}_run", d)
                d.close()
                ex.settle(4)
        # 5. delete the first copy: the others renumbered?
        its = items(eng)
        if len(its) >= 2:
            s5a = m.state()
            ex.click_block(its[0])
            ex.key(Qt.Key_Delete)
            P("after delete first", titles(eng), "toasts", m.toasts[-1:], "popups", ex.popups[-1:])
            s5 = m.state()
            ex.undo()
            P("undo exact", m.state() == s5a, titles(eng))
            ex.redo()
            P("redo exact", m.state() == s5)
        # 6. independence: the params of each copy
        q = ex.s.paths[0]
        P("params", {k: v for k, v in ex.s.p.workflow(q.pid).params.items() if k.split("~")[0] == eng})
        P("graph", [(n["block"], n.get("name")) for n in ex.s.graph(q)])
        P("exceptions", [x.splitlines()[0] for x in m.exc])
    if blk != "denoise":                               # the run window of an image workflow with the block twice
        ex.open(ex.project("image"))
        ex.settle(8)
        ex.click_block(ex.block("image"))
        for b in ("astrometry", blk, blk):
            ex.double_click_card(b)
            ex.settle(6)
        P("image graph", titles(eng))
        its = items(eng)
        if its:
            ex.right_click_block(its[-1], "Pause")
        b = ex.w.bt_run
        P("image run button", b.text(), b.isEnabled(), b.toolTip()[:200])
        if b.isEnabled():
            ex.click(b)
            ex.settle(10)
            d = ex.w.run.dialog
            if d is not None:
                try:
                    rows = [(r.name.text(), r.toolTip()[:100]) for r in d.rows]
                except Exception as e:  # noqa: BLE001
                    rows = repr(e)
                P("image run rows", rows)
                P("image run texts", vis(d)[:30])
                ex.shot(f"rep_{blk}_{LANG}_run_image", d)
                d.close()
                ex.settle(4)
ex.close()
