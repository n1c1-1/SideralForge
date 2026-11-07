"""Round 4, area clavier_fenetre, axis 1a: the shortcuts the app declares and those a user expects, focus traps, Tab.
Prints; candidates written when EXPLORE_NOTE=1.

    python explore5_clavier_keys.py [case ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from explore_post_common import P, start  # noqa: E402
from explore2_blocks_common import G, Explorer, drawn  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QWidget, QAbstractButton  # noqa: E402

start(1200)
ex = Explorer("clavier_fenetre")
g = G(ex)
pp = P(ex)
CASES = {}
ONLY = sys.argv[1:]
CHAIN = ["calibration", "registration", "stacking", "astrometry", "gradient", "denoise", "color"]
ST = ["Nouveau projet, une session de 15 brutes", "Double-clic dans la palette : Calibration, Alignement, Empilement, "
      "Astrométrie, Gradient, Débruitage, Couleur"]


def case(fn):
    CASES[fn.__name__] = fn
    return fn


def dump():
    return json.dumps(ex.s.p.to_dict(), sort_keys=True, default=str)


def fw():
    w = QApplication.focusWidget()
    if w is None:
        return "None"
    t = w.text() if hasattr(w, "text") and callable(w.text) else ""
    return f"{type(w).__name__}:{w.objectName()}:{str(t)[:25]}"


def sel():
    return ex.s.sel_step, ex.s.current


def press(k, mod=Qt.NoModifier, widget=None):
    pp.activate()
    w = widget or QApplication.focusWidget() or ex.w
    QTest.keyClick(w, k, mod)
    ex.settle(6)


@case
def keys_on_block():
    g.new("session_empty_wf", CHAIN)
    ex.click_block(g.blk("fond"))
    print("selected", sel(), "focus", fw())
    # Ctrl+Shift+Z, Ctrl+C/V/D, Backspace, arrows on a selected block
    g.place  # noqa
    ex.right_click_block(g.blk("fond"), None)
    pp.del_key()
    print("Del on Gradient:", [d["title"] for d in drawn(ex)][:9], "toast", pp.toast())
    pp.undo()
    print("Ctrl+Z:", "fond" in [it.node.key for it in ex.blocks()], "sel", sel(), "toast", pp.toast())
    pp.redo()
    print("Ctrl+Y:", "fond" in [it.node.key for it in ex.blocks()], "toast", pp.toast())
    pp.undo()
    pp.key(Qt.Key_Z, ex.w, Qt.ControlModifier | Qt.ShiftModifier)
    print("Ctrl+Shift+Z (redo?):", "fond" in [it.node.key for it in ex.blocks()], "toast", pp.toast())
    pp.redo() if "fond" in [it.node.key for it in ex.blocks()] else None
    ex.click_block(g.blk("debruitage"))
    for name, k, mod in (("Backspace", Qt.Key_Backspace, Qt.NoModifier), ("Ctrl+C", Qt.Key_C, Qt.ControlModifier),
                         ("Ctrl+V", Qt.Key_V, Qt.ControlModifier), ("Ctrl+D", Qt.Key_D, Qt.ControlModifier),
                         ("Ctrl+X", Qt.Key_X, Qt.ControlModifier), ("Ctrl+A", Qt.Key_A, Qt.ControlModifier),
                         ("Left", Qt.Key_Left, Qt.NoModifier), ("Right", Qt.Key_Right, Qt.NoModifier),
                         ("Up", Qt.Key_Up, Qt.NoModifier), ("Down", Qt.Key_Down, Qt.NoModifier),
                         ("Home", Qt.Key_Home, Qt.NoModifier), ("F2", Qt.Key_F2, Qt.NoModifier),
                         ("Space", Qt.Key_Space, Qt.NoModifier)):
        ex.click_block(g.blk("debruitage"))
        d0, s0 = dump(), sel()
        hs = ex.view.horizontalScrollBar().value(), ex.view.verticalScrollBar().value()
        pp.key(k, None, mod)
        d1 = dump()
        print(f"{name}: model changed={d0 != d1} sel {s0}->{sel()} scroll {hs}->"
              f"{(ex.view.horizontalScrollBar().value(), ex.view.verticalScrollBar().value())} toast={pp.toast()!r} "
              f"modals={ex.modals[-1:]} mode={ex.w.mode}", flush=True)
        if d0 != d1:
            pp.undo()


@case
def enter_escape():
    g.new("session_empty_wf", CHAIN)
    ex.click_block(g.blk("fond"))
    pp.key(Qt.Key_Return)
    print("Enter on Gradient: mode", ex.w.mode, "focus", fw())
    ex.shot("enter_on_block")
    # Escape in Inspect: back?
    pp.key(Qt.Key_Escape, QApplication.focusWidget() or ex.w)
    print("Escape in Inspect: mode", ex.w.mode, "focus", fw())
    ex.w.set_mode(0)
    ex.settle(6)
    ex.click_block(g.blk("fond"))
    pp.key(Qt.Key_Escape)
    print("Escape on graph: sel", sel(), "selected items", [it.title for it in ex.blocks() if it.selected])
    # Enter with nothing selected
    pp.key(Qt.Key_Return)
    print("Enter nothing selected: mode", ex.w.mode)
    ex.w.set_mode(0)
    ex.settle(4)

    # Escape in a rename dialog: the dialog must close, the name stays
    res = {}

    def esc_dialog(d):
        e = d.findChildren(QLineEdit)
        if e:
            e[0].setFocus()
            e[0].setText("ZZZ")
        QTest.keyClick(e[0] if e else d, Qt.Key_Escape)
        QApplication.processEvents()
        res["visible_after_esc"] = d.isVisible()
        res["title"] = d.windowTitle()
        if d.isVisible():
            d.reject()
    ex.dialog_handler = esc_dialog
    g.menu(g.blk("couleur"), "Renommer")
    ex.settle(8)
    print("Escape in rename dialog:", res, "paths", [q.name for q in ex.s.paths], "modals", ex.modals[-2:])

    res2 = {}

    def enter_dialog(d):
        e = d.findChildren(QLineEdit)
        if e:
            e[0].setFocus()
            e[0].selectAll()
            QTest.keyClicks(e[0], "Flux A")
        QTest.keyClick(e[0] if e else d, Qt.Key_Return)
        QApplication.processEvents()
        res2["visible_after_enter"] = d.isVisible()
        if d.isVisible():
            d.reject()
    ex.dialog_handler = enter_dialog
    g.menu(g.blk("couleur"), "Renommer")
    ex.settle(8)
    print("Enter in rename dialog:", res2, "paths", [q.name for q in ex.s.paths], "toast", pp.toast())
    pp.undo()
    print("Ctrl+Z after rename:", [q.name for q in ex.s.paths], "toast", pp.toast())

    # Escape in Settings (Ctrl+,)
    res3 = {}

    def esc_any(d):
        res3["class"] = type(d).__name__
        res3["title"] = d.windowTitle()
        res3["size"] = (d.width(), d.height())
        QTest.keyClick(QApplication.focusWidget() or d, Qt.Key_Escape)
        QApplication.processEvents()
        res3["visible_after_esc"] = d.isVisible()
        if d.isVisible():
            d.reject()
    ex.dialog_handler = esc_any
    pp.key(Qt.Key_Comma, ex.w, Qt.ControlModifier)
    ex.settle(8)
    print("Ctrl+, then Escape:", res3, "modals", ex.modals[-1:])
    res3.clear()
    ex.dialog_handler = esc_any
    pp.key(Qt.Key_F1, ex.w)
    ex.settle(8)
    print("F1 then Escape:", res3, "popups", ex.popups[-1:])


@case
def escape_port_pull():
    """A port pulled toward Post, Escape pressed before release: the pull must be cancelled."""
    g.new("session_empty_wf", CHAIN)
    it = g.blk("fond")
    vp = ex.view.viewport()
    y = next((py for w, py in it.ports if w is None), 30)
    from astro_gpu.client.graph import BH  # noqa
    a = ex.view.mapFromScene(it.mapToScene(QPointF(it.w - 1, BH / 2)))
    b = ex.view.mapFromScene(ex.view.final_item.sceneBoundingRect().center())
    n0 = len(pp.graph_layers())
    ex._mouse(vp, QEvent.MouseButtonPress, a)
    ex.settle(1)
    for t in (0.3, 0.6, 0.9):
        ex._mouse(vp, QEvent.MouseMove, QPoint(int(a.x() + (b.x() - a.x()) * t), int(a.y() + (b.y() - a.y()) * t)),
                  Qt.NoButton, Qt.LeftButton)
    ex.settle(2)
    ex.shot("port_pull_before_escape")
    QTest.keyClick(vp, Qt.Key_Escape)
    ex.settle(3)
    ex.shot("port_pull_after_escape")
    ex._mouse(vp, QEvent.MouseMove, b, Qt.NoButton, Qt.LeftButton)
    ex._mouse(vp, QEvent.MouseButtonRelease, b, Qt.LeftButton, Qt.NoButton)
    ex.settle(8)
    print("port pull + Escape + release on Post: layers", n0, "->", len(pp.graph_layers()), "toast", pp.toast())
    # a block moved (if blocks can be dragged in the graph) then Escape
    it = g.blk("couleur")
    a = ex.point_of(it)
    ex.drop_at = None
    ex._mouse(vp, QEvent.MouseButtonPress, a)
    for t in (0.3, 0.6):
        ex._mouse(vp, QEvent.MouseMove, QPoint(a.x() + int(150 * t), a.y() + int(80 * t)), Qt.NoButton, Qt.LeftButton)
    QTest.keyClick(vp, Qt.Key_Escape)
    ex._mouse(vp, QEvent.MouseButtonRelease, QPoint(a.x() + 90, a.y() + 48), Qt.LeftButton, Qt.NoButton)
    ex.settle(6)
    print("block drag + Escape: drops", ex.drops[-1:], "blocks", [d["title"] for d in drawn(ex)][:10])


@case
def focus_traps():
    g.new("session_empty_wf", CHAIN)
    # 1. a block selected, focus moved into the panel by a click on a button / a spin box, then Del / D
    for what in ("method", "spin", "advanced"):
        ex.click_block(g.blk("fond"))
        pn = ex.page.panel
        if what == "method":
            w = next((c for c in pn.findChildren(QWidget) if type(c).__name__ == "MethodCard" and c.isVisibleTo(pn)), None)
        elif what == "spin":
            from PySide6.QtWidgets import QSpinBox
            w = next((c for c in pn.findChildren(QSpinBox) if c.isVisibleTo(pn)), None)
        else:
            w = next((c for c in pn.findChildren(QAbstractButton) if c.isVisibleTo(pn) and "avancés" in c.text()), None)
        if w is None:
            print(what, "not found")
            continue
        ex.click(w)
        ex.settle(4)
        f = fw()
        for name, k in (("Del", Qt.Key_Delete), ("D", Qt.Key_D)):
            d0 = dump()
            n0 = len([it for it in ex.blocks() if it.node.key == "fond"])
            pp.activate()
            QTest.keyClick(QApplication.focusWidget() or ex.w, k)
            ex.settle(6)
            print(f"panel {what} (focus {f}) then {name}: model changed={dump() != d0} "
                  f"fond blocks {n0}->{len([it for it in ex.blocks() if it.node.key == 'fond'])} sel {sel()} "
                  f"toast={pp.toast()!r}", flush=True)
            if dump() != d0:
                pp.undo()
    # 2. focus on the palette (a card clicked), a block selected, then Del
    ex.click_block(g.blk("fond"))
    c = ex.card("selection")
    if c is not None:
        ex.click(c)
        d0 = dump()
        pp.activate()
        QTest.keyClick(QApplication.focusWidget() or ex.w, Qt.Key_Delete)
        ex.settle(6)
        print("palette card clicked then Del: sel", sel(), "changed", dump() != d0, "focus", fw(), "toast", pp.toast())
    # 3. a text field of the panel: Del / D / Ctrl+Z / Ctrl+A stay in the field
    ex.click_block(g.blk("sources"))
    ed = next((e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisibleTo(ex.page.panel)), None)
    if ed is not None:
        ex.click(ed)
        ed.end(False)
        for name, k, mod in (("D", Qt.Key_D, Qt.NoModifier), ("Del", Qt.Key_Delete, Qt.NoModifier),
                             ("Home", Qt.Key_Home, Qt.NoModifier), ("Del", Qt.Key_Delete, Qt.NoModifier),
                             ("Ctrl+Z", Qt.Key_Z, Qt.ControlModifier), ("Ctrl+A", Qt.Key_A, Qt.ControlModifier),
                             ("F5", Qt.Key_F5, Qt.NoModifier), ("Escape", Qt.Key_Escape, Qt.NoModifier)):
            d0 = dump()
            pp.activate()
            ed.setFocus()
            QTest.keyClick(ed, k, mod)
            ex.settle(5)
            print(f"name field {name}: text={ed.text()!r} model changed={dump() != d0} sel={sel()} "
                  f"blocks={len(ex.blocks())} focus={fw()} status={ex.w.status.text()[:60]!r}", flush=True)


@case
def tab_order():
    g.new("session_empty_wf", CHAIN)
    ex.click_block(g.blk("fond"))
    seen = []
    for i in range(40):
        pp.activate()
        w = QApplication.focusWidget() or ex.w
        QTest.keyClick(w, Qt.Key_Tab)
        ex.settle(2)
        f = QApplication.focusWidget()
        vis = f.isVisible() if f else None
        r = f.mapTo(ex.w, QPoint(0, 0)) if f else None
        seen.append(f"{fw()} vis={vis} at={(r.x(), r.y()) if r else None}")
    for i, x in enumerate(seen):
        print(f"TAB {i}: {x}")


@case
def view_keys():
    g.new("session_empty_wf", CHAIN)
    for name, k, mod in (("Ctrl+2", Qt.Key_2, Qt.ControlModifier), ("Ctrl+3", Qt.Key_3, Qt.ControlModifier),
                         ("Ctrl+1", Qt.Key_1, Qt.ControlModifier), ("Ctrl+L", Qt.Key_L, Qt.ControlModifier),
                         ("Ctrl+L", Qt.Key_L, Qt.ControlModifier), ("Ctrl+H", Qt.Key_H, Qt.ControlModifier)):
        pp.key(k, ex.w, mod)
        ex.settle(8)
        print(f"{name}: mode={ex.w.mode} log={ex.w.b_log.isChecked()} screen={type(ex.w.screens.currentWidget()).__name__}"
              f" modals={ex.modals[-1:]}", flush=True)
    ex.open(ex.project("session_empty_wf"))
    g.act()
    g.place(*CHAIN)
    # Inspect: the image viewer has its own Ctrl+1 (100 %) against the window's Ctrl+1 (Graph)
    ex.click_block(g.blk("fond"))
    pp.key(Qt.Key_Return)
    ex.settle(8)
    print("in Inspect, mode", ex.w.mode, "focus", fw())
    from astro_gpu.client import imageview
    views = [v for v in ex.w.findChildren(QWidget) if type(v).__name__ in ("ImageView", "Viewer", "ImageViewer")
             and v.isVisible()]
    print("viewers", [type(v).__name__ for v in views])
    if views:
        views[0].setFocus()
        ex.settle(2)
        QTest.keyClick(views[0], Qt.Key_1, Qt.ControlModifier)
        ex.settle(6)
        print("Ctrl+1 with the viewer focused: mode", ex.w.mode)
        ex.w.set_mode(1)
        ex.settle(6)
    pp.key(Qt.Key_Escape, QApplication.focusWidget() or ex.w)
    print("Escape in Inspect: mode", ex.w.mode)


def followups():
    # F5 while a name is typed but not validated
    for rep in range(2):
        g.new("session_empty_wf", CHAIN)
        ex.click_block(g.blk("sources"))
        ed = next((e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisibleTo(ex.page.panel)), None)
        ex.click(ed)
        ed.selectAll()
        QTest.keyClicks(ed, "Nuit du 3")
        ex.settle(3)
        pp.activate()
        ed.setFocus()
        QTest.keyClick(ed, Qt.Key_F5)
        ex.settle(10)
        print(f"[{rep}] typed 'Nuit du 3' then F5: session name={ex.s.p.sessions[0].name!r} drawn="
              f"{[d['title'] for d in drawn(ex)][:2]} status={ex.w.status.text()[:80]!r} undo={len(ex.s._undo)}")
        ex.shot(f"f5_while_typing_{rep}")
        # same, but Ctrl+S
        ex.click_block(g.blk("sources"))
        ed = next((e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisibleTo(ex.page.panel)), None)
        ex.click(ed)
        ed.selectAll()
        QTest.keyClicks(ed, "Nuit du 4")
        pp.activate()
        ed.setFocus()
        QTest.keyClick(ed, Qt.Key_S, Qt.ControlModifier)
        ex.settle(10)
        root = Path(ex.s.p.root) if hasattr(ex.s.p, "root") else None
        disk = None
        for f in (root.glob("*.astroproj") if root else []):
            disk = json.loads(f.read_text(encoding="utf-8"))
        print(f"[{rep}] typed 'Nuit du 4' then Ctrl+S: model={ex.s.p.sessions[0].name!r} field={ed.text() if ed else None!r}"
              f" disk={[x['name'] for x in disk['sessions']] if disk else None} root={root}")
        # Escape during a port pull
        it = g.blk("couleur")
        from astro_gpu.client.graph import BH
        vp = ex.view.viewport()
        a = ex.view.mapFromScene(it.mapToScene(QPointF(it.w - 1, BH / 2)))
        b = ex.view.mapFromScene(ex.view.final_item.sceneBoundingRect().center())
        n0 = len(pp.graph_layers())
        ex._mouse(vp, QEvent.MouseButtonPress, a)
        for t in (0.3, 0.6, 0.9):
            ex._mouse(vp, QEvent.MouseMove, QPoint(int(a.x() + (b.x() - a.x()) * t), int(a.y() + (b.y() - a.y()) * t)),
                      Qt.NoButton, Qt.LeftButton)
        ex.settle(2)
        pp.activate()
        QTest.keyClick(vp, Qt.Key_Escape)
        ex.settle(3)
        sh = ex.shot(f"port_pull_escape_{rep}")
        ex._mouse(vp, QEvent.MouseButtonRelease, b, Qt.LeftButton, Qt.NoButton)
        ex.settle(8)
        print(f"[{rep}] port pull Couleur, Escape, release on Post: layers {n0} -> {len(pp.graph_layers())}, toast "
              f"{pp.toast()!r}")
    # Inspect: Ctrl+1 with the image viewer focused
    ex.click_block(g.blk("fond"))
    pp.key(Qt.Key_Return)
    ex.settle(8)
    views = [v for v in ex.w.findChildren(QWidget) if type(v).__name__ == "ImageView"]
    print("ImageViews:", [(v.isVisible(), v.width()) for v in views], "mode", ex.w.mode)
    v = next((v for v in views if v.isVisible()), None)
    if v is not None:
        pp.activate()
        v.setFocus()
        ex.settle(2)
        print("focus", fw())
        QTest.keyClick(QApplication.focusWidget() or v, Qt.Key_1, Qt.ControlModifier)
        ex.settle(6)
        print("Ctrl+1 in Inspect with the viewer focused: mode", ex.w.mode, "(0 = Graph)")
        ex.shot("inspect_ctrl1")


CASES["followups"] = followups


for name, fn in CASES.items():
    if ONLY and name not in ONLY:
        continue
    print("=====", name, flush=True)
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
    pp.check_slot_errors(name, ST)
ex.close()

