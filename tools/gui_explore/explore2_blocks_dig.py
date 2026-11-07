"""block_actions round 2: digging one case at a time (python explore2_blocks_dig.py <case>)."""
from __future__ import annotations

import faulthandler
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(600, exit=True)

from explore2_blocks_common import CHAIN, EXC, SHORT, TOASTS, G, Explorer, drawn, panel_info, state  # noqa: E402,F401
from PySide6.QtCore import Qt  # noqa: E402

CASE = sys.argv[1]
LANG = sys.argv[2] if len(sys.argv) > 2 else "fr"
ex = Explorer("block_actions", lang=LANG)
g = G(ex)


def show(label):
    st = state(ex)
    print("--", label)
    print("   sessions:", [(x["id"], x["name"], x["lights"]) for x in st["sessions"]])
    print("   workflows:", [(w["name"], w["sessions"], [b[0] for b in (w["blocks"] or [])]) for w in st["workflows"]])
    print("   drawn:", [(d["title"], d["summary"][:30]) for d in drawn(ex)])
    print("   post:", st["post"])
    print("   toasts:", TOASTS[-3:], "undo:", len(ex.s._undo), "redo:", len(ex.s._redo), "exc:", len(EXC), flush=True)


if CASE == "fusion_session":
    for how in ("cross", "del", "menu"):
        g.mono_fusion()
        show("built " + how)
        it = g.blk("session", sid=1)
        if how == "cross":
            g.cross(it)
        elif how == "del":
            ex.click_block(it)
            g.act()
            ex.key(Qt.Key_Delete)
        else:
            g.menu(it, pick="Supprimer")
        show("after " + how)
        ex.shot("dig_fusion_session_" + how)
        g.undo()
        show("undo")
        g.redo()
        show("redo")
        ex.key(Qt.Key_S, ex.w, Qt.ControlModifier)
        ex.settle(4)
        print("   exc after save:", len(EXC), EXC[-1][-300:] if EXC else "")

if CASE == "send_post":
    g.new("session", SHORT)
    show("built")
    g.menu(g.blk("fond"), pick="Vers Post")
    show("after Vers Post on Gradient")
    ex.shot("dig_send_post")
    g.undo()
    show("undo")
    g.redo()
    show("redo")
    g.new("session", SHORT)
    ex.pull_port_to_post(g.blk("fond"))
    show("after pulling Gradient port to Post")
    ex.shot("dig_pull_port")
    it = g.blk(title="Sortie")
    if it is not None:
        ex.click_block(it)
        print("   panel of Sortie:", panel_info(ex)["titles"], [b["text"] for b in panel_info(ex)["buttons"]])
        g.menu(g.blk(title="Sortie"), pick="Supprimer")
        show("Sortie: Supprimer from its menu")
        it = g.blk(title="Sortie")
        if it is not None:
            g.cross(it)
            show("Sortie: its cross")
            it = g.blk(title="Sortie")
        if it is not None:
            ex.click_block(it)
            g.act()
            ex.key(Qt.Key_Delete)
            show("Sortie: Del")
        ex.shot("dig_sortie_delete")

if CASE == "repeat_send":
    g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
    g.drop_idx("gradient", ex.s.paths[0].pid, 3)
    show("two gradients")
    g.menu(g.blk(title="Gradient 1"), pick="Vers Post")
    show("Gradient 1 to Post")
    ex.shot("dig_repeat_send")

if CASE == "session_calib":
    for rep in range(2):
        g.new("two_sessions", ["calibration", "registration", "stacking", "astrometry", "gradient"])
        c1 = g.blk("calibration", sid=1)
        print("   bp attr before:", [getattr(it, "bp", None) for it in ex.blocks() if it.node.key == "calibration"])
        g.menu(c1, pick="Pause vérif")
        print("   bp attr after:", [(it.node.sid, getattr(it, "bp", None)) for it in ex.blocks()
                                    if it.node.key == "calibration"], "bps:", state(ex)["workflows"][0]["bp"],
              "undo:", len(ex.s._undo))
        ex.shot(f"dig_session_calib_pause_{rep}")
        g.new("two_sessions", ["calibration", "registration", "stacking", "astrometry", "gradient"])
        ex.click_block(g.blk("calibration", sid=1))
        g.act()
        ex.key(Qt.Key_D)
        print("   after D on Calibration of session 1:", [(it.node.sid, it.title, it.summary, getattr(it.node, "off", None))
                                                        for it in ex.blocks() if it.node.key == "calibration"],
              g.graph_of(), TOASTS[-1:])
        ex.shot(f"dig_session_calib_off_{rep}")
        g.undo()
        ex.click_block(g.blk("calibration", sid=2))
        g.act()
        ex.key(Qt.Key_Delete)
        print("   after Del on Calibration of session 2:", [(it.node.sid, it.title) for it in ex.blocks()
                                                          if it.node.key == "calibration"], g.graph_of(), TOASTS[-1:])
        ex.shot(f"dig_session_calib_del_{rep}")

if CASE == "plus_session":
    for kind in ("session", "two_sessions", "blank"):
        g.new(kind, ["calibration", "registration", "stacking"] if kind != "blank" else [])
        gh = getattr(ex.view, "plus_session_ghost", None)
        if gh is None:
            print(kind, "no + Session")
            continue
        pt = ex.ghost_center(gh)
        sp = ex.view.mapToScene(pt)
        print(kind, "items at point:", [(type(x).__name__, getattr(x, "text", ""), x.zValue()) for x in ex.view.scene().items(sp)])
        fa, md, n = len(ex.file_asks), len(ex.modals), len(TOASTS)
        ex.hover(ex.view.viewport(), pt)
        ex.click(ex.view.viewport(), pt)
        ex.settle(8)
        print("   click: files", ex.file_asks[fa:], "modals", ex.modals[md:], "toasts", TOASTS[n:])
        fa = len(ex.file_asks)
        gh = getattr(ex.view, "plus_session_ghost", None)
        if gh is not None:
            gh.on_click()
            ex.settle(8)
        print("   direct on_click: files", ex.file_asks[fa:])

if CASE == "fusion_wf_delete":
    from PySide6.QtWidgets import QPushButton
    for rep in range(2):
        g.mono_fusion()
        e0 = len(EXC)
        ha = [it for it in ex.blocks() if it.node.key == "etoiles" and g.pid_of("Ha") in [q.pid for q in it.node.paths]][0]
        g.menu(ha, pick="Supprimer workflow")
        show(f"[{rep}] Ha's Étoiles/fond → Supprimer workflow")
        fus = g.blk("fusion")
        print("   fusion:", fus.summary if fus else None, repr(fus.toolTip()[:200]) if fus else None)
        ex.shot(f"dig_fusion_wf_delete_{rep}")
        g.undo()
        show("undo")
        g.redo()
        show("redo")
        print("   EXC:", [x.strip().splitlines()[-1] for x in EXC[e0:]])
        ex.shot(f"dig_fusion_wf_redo_{rep}")
        e1 = len(EXC)
        ex.key(Qt.Key_S, ex.w, Qt.ControlModifier)
        print("   Ctrl+S EXC:", [x.strip().splitlines()[-1] for x in EXC[e1:]], "title:", ex.w.windowTitle())
    for rep in range(2):
        g.mono_fusion()
        e0 = len(EXC)
        ex.click_block(g.blk("fusion"))
        b = next((b for b in ex.page.panel.findChildren(QPushButton) if b.isVisible() and b.text() == "Supprimer"), None)
        ex.click(b)
        ex.settle(8)
        show(f"[{rep}] Fusion panel → Supprimer")
        print("   EXC:", [x.strip().splitlines()[-1] for x in EXC[e0:]])
        if len(EXC) > e0:
            print(EXC[-1][-1500:])
        ex.shot(f"dig_fusion_panel_delete_{rep}")
        e0 = len(EXC)
        g.mono_fusion()
        ex.click_block(g.blk("fusion"))
        g.act()
        ex.key(Qt.Key_Delete)
        show(f"[{rep}] Fusion selected → Suppr")
        print("   EXC:", [x.strip().splitlines()[-1] for x in EXC[e0:]])

if CASE == "image_delete":
    for rep in range(2):
        g.new("image")
        ex.click_block(g.blk("image"))
        g.place("astrometry", "gradient")
        ex.click_block(g.blk("image"))
        g.act()
        ex.key(Qt.Key_Delete)
        show(f"[{rep}] Image deleted")
        ex.shot(f"dig_image_delete_{rep}")
        g.undo()
        show("undo")
        g.redo()
        show("redo")
        ex.double_click_card("image")
        ex.settle(6)
        show("Image placed again by double-click")
        ex.shot(f"dig_image_delete_replace_{rep}")

if CASE == "open_in_post":
    for rep in range(2):
        g.new("session", CHAIN)
        it = next((b for b in ex.blocks() if b.node.key == "composition"), None)
        s0 = getattr(ex.s.p, "post", None)
        print("mode before", ex.w.mode, "post", s0)
        ex.click_block(it)
        from PySide6.QtWidgets import QPushButton
        b = next((b for b in ex.page.panel.findChildren(QPushButton) if b.isVisible() and b.text() == "Ouvrir dans Post"), None)
        ex.click(b)
        ex.settle(10)
        print("mode after", ex.w.mode, "post", json.dumps(getattr(ex.s.p, "post", None), ensure_ascii=False)[:300],
              "undo", len(ex.s._undo))
        ex.shot(f"dig_open_in_post_{rep}")
        ex.w.set_mode(0)
        ex.settle(6)
        g.undo()
        print("after back to graph + Ctrl+Z: post", json.dumps(getattr(ex.s.p, "post", None), ensure_ascii=False)[:200],
              "undo", len(ex.s._undo), "toasts", TOASTS[-2:], "drawn", [d["title"] for d in drawn(ex)][-3:])

if CASE == "dup_names":
    from PySide6.QtWidgets import QLineEdit
    for rep in range(2):
        g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
        g.drag_branch("color", "astrometrie", ex.s.paths[0].pid)

        def h(d):
            for e in d.findChildren(QLineEdit):
                e.setText("Workflow")
            d.accept()
        ex.dialog_handler = h
        g.menu(g.blk("couleur"), pick="Renommer")
        ex.dialog_handler = None
        m = g.texts(g.menu(g.blk("calibration")))
        ex.click_block(g.blk("calibration"))
        print(f"[{rep}] workflows {[q.name for q in ex.s.paths]} ; Calibration menu {m} ; panel "
              f"{[b['text'] for b in panel_info(ex)['buttons']]}", flush=True)
        ex.shot(f"dig_dup_names_{rep}")

if CASE == "two_flows_wf_delete":
    for rep in range(2):
        g.new("two_flows", clear=False)
        for q in list(ex.s.paths):
            it = next((b for b in ex.blocks() if b.node.key in ("session", "sources")
                       and q.pid in [p.pid for p in b.node.paths]), None)
            ex.click_block(it)
            g.place("calibration", "registration", "stacking")
        w = g.pid_of("Workflow")
        g.menu(g.blk("empilement", pid=w), pick="Supprimer workflow")
        show(f"[{rep}] Supprimer workflow « Workflow »")
        s1 = g.blk("session", sid=1)
        print("   Session 1 block:", s1.title if s1 else None, s1.summary if s1 else None, "paths",
              [q.name for q in s1.node.paths] if s1 else None, "menu", g.texts(g.menu(s1)) if s1 else None)
        ex.shot(f"dig_two_flows_wf_delete_{rep}")

if CASE == "opens":
    for rep in range(2):
        for how in ("menu Inspection on Gradient", "double-click Gradient", "double-click Post", "Post tab",
                    "double-click Recombiner"):
            g.new("session", CHAIN)
            post0 = json.dumps(getattr(ex.s.p, "post", None), sort_keys=True)
            nu = len(ex.s._undo)
            if how.startswith("menu"):
                g.menu(g.blk("fond"), pick="Inspection")
            elif how == "double-click Gradient":
                ex.double_click_block(g.blk("fond"))
            elif how == "double-click Post":
                ex.double_click_block(g.blk("final"))
            elif how == "Post tab":
                ex.w.set_mode(2)
            else:
                ex.double_click_block(next(b for b in ex.blocks() if b.node.key == "composition"))
            ex.settle(10)
            mode = ex.w.mode
            ex.w.set_mode(0)
            ex.settle(6)
            post1 = json.dumps(getattr(ex.s.p, "post", None), sort_keys=True)
            print(f"[{rep}] {how}: mode {mode}, undo entries +{len(ex.s._undo) - nu}, composition changed "
                  f"{post1 != post0}: {post1[:160]}", flush=True)

if CASE == "layer_names":
    for rep in range(2):
        g.new("session", CHAIN)
        g.menu(g.blk("etoiles"), pick="Étoiles vers Post")
        g.menu(g.blk("etoiles"), pick="Sans étoiles vers Post")
        n = len(TOASTS)
        g.menu(g.blk("etoiles"), pick="+ Recombiner")
        print(f"[{rep}] + Recombiner msgs {TOASTS[n:]}")
        print(f"[{rep}] layers: {[x.get('name') for x in state(ex)['post']]}")
        ex.w.set_mode(2)
        ex.settle(12)
        from PySide6.QtWidgets import QLabel, QListWidget, QTreeWidget
        names = [x.text() for x in ex.w.findChildren(QLabel) if x.isVisible() and ("Stars" in x.text() or "Recombine" in x.text())]
        for lw in ex.w.findChildren(QListWidget) + ex.w.findChildren(QTreeWidget):
            if lw.isVisible():
                names += [lw.item(i).text() for i in range(lw.count())] if isinstance(lw, QListWidget) else []
        print("   Post page texts with English:", names[:8])
        ex.shot(f"dig_layer_names_{rep}")
        ex.w.set_mode(0)

if CASE == "main_wire":
    from astro_gpu.client.graph import WireItem
    for rep in range(2):
        g.new("session", SHORT)
        ws = [x for x in ex.view.scene().items() if isinstance(x, WireItem) and x.isVisible()]
        print(f"[{rep}] wires: {[w.toolTip().splitlines()[0] for w in ws]}")
        w = ws[0]
        pt = ex.view.mapFromScene(w.path().pointAtPercent(0.5))
        ex.hover(ex.view.viewport(), pt)
        ex.click(ex.view.viewport(), pt)
        g.act()
        ex.key(Qt.Key_Delete)
        post = g.blk("final")
        print(f"   after Suppr on the wire: msgs {TOASTS[-1:]} Post says « {post.summary} » removed="
              f"{ex.s.post_removed()} ; Couleur menu {g.texts(g.menu(g.blk('couleur')))[:5]}")
        ex.shot(f"dig_main_wire_{rep}")
        m = g.texts(g.menu(g.blk("couleur")))
        if "Vers Post" in m:
            n = len(TOASTS)
            g.menu(g.blk("couleur"), pick="Vers Post")
            print(f"   Couleur → Vers Post: {TOASTS[n:]} wires {len([x for x in ex.view.scene().items() if isinstance(x, WireItem)])} "
                  f"Post « {g.blk('final').summary} » workflows {[q.name for q in ex.s.paths]}")
        else:
            ex.pull_port_to_post(g.blk("couleur"))
            print(f"   port of Couleur pulled to Post: {TOASTS[-1:]} Post « {g.blk('final').summary} » workflows "
                  f"{[q.name for q in ex.s.paths]}")

if CASE == "fusion_place":
    for rep in range(2):
        g.new("mono", clear=False)
        for name in ("Ha", "OIII"):
            ex.click_block(g.blk("empilement", pid=g.pid_of(name)))
            g.place("astrometry")
        before = {q.name: g.graph_of(q.name) for q in ex.s.paths}
        n, nu = len(TOASTS), len(ex.s._undo)
        g.place("fusion")
        after = {q.name: g.graph_of(q.name) for q in ex.s.paths}
        print(f"[{rep}] before Fusion {before}\n   after {after}\n   msgs {TOASTS[n:]} undo +{len(ex.s._undo) - nu}")
        g.undo()
        print("   one Ctrl+Z:", {q.name: g.graph_of(q.name) for q in ex.s.paths} == before)
        ex.shot(f"dig_fusion_place_{rep}")

if CASE == "two_flows_session":
    for rep in range(2):
        g.new("two_flows", clear=False)
        for q in list(ex.s.paths):
            it = next((b for b in ex.blocks() if b.node.key in ("session", "sources")
                       and q.pid in [p.pid for p in b.node.paths]), None)
            ex.click_block(it)
            g.place("calibration", "registration", "stacking")
        show(f"[{rep}] built")
        g.cross(g.blk("session", sid=1))
        show(f"[{rep}] cross on Session 1")
        ex.shot(f"dig_two_flows_session_{rep}")

if CASE == "comp_warning":
    for step in ("cross", "undo", "click", "del", "dblclick"):
        g.new("session", CHAIN)
        comp = lambda: next((it for it in ex.blocks() if it.node.key == "composition"), None)  # noqa: E731
        print("== step", step, flush=True)
        if step in ("cross", "undo"):
            g.cross(comp())
            if step == "undo":
                g.undo()
        elif step == "click":
            ex.click_block(comp())
        elif step == "del":
            ex.click_block(comp())
            g.act()
            ex.key(Qt.Key_Delete)
        else:
            ex.double_click_block(comp())
            ex.settle(8)
            print("   mode", ex.w.mode)
        print("   done", step, TOASTS[-1:], flush=True)

if CASE == "mono_more":
    for rep in range(2):
        g.mono_fusion()
        e0 = len(EXC)
        g.cross(g.blk("session", sid=1))
        ph = next((it for it in ex.blocks() if it.node.key == "session" and it.title == "Ha"), None)
        print(f"[{rep}] placeholder Ha: removable={ph.removable if ph else None}")
        if ph is not None:
            g.cross(ph)
            show(f"[{rep}] cross on the empty Ha")
            ph = next((it for it in ex.blocks() if it.node.key == "session" and it.title == "Ha"), None)
            if ph is not None:
                g.menu(ph, pick="Supprimer")
                show(f"[{rep}] menu Supprimer on the empty Ha")
        ex.shot(f"dig_mono_placeholder_{rep}")
        g.mono_fusion()
        g.menu(g.blk("fusion"), pick="Vers Post")
        show(f"[{rep}] Fusion → Vers Post")
        g.undo()
        show("undo")
        g.redo()
        show("redo")
        # a pause, then the block switched off
        g.mono_fusion()
        ha = g.pid_of("Ha")
        g.menu(g.blk("astrometrie", pid=ha), pick="Pause vérif")
        ex.click_block(g.blk("astrometrie", pid=ha))
        g.act()
        ex.key(Qt.Key_D)
        st = state(ex)
        print(f"[{rep}] pause then D on Ha's Astrométrie: {[(w['name'], w['bp'], [b[0] + ('/off' if b[3] else '') for b in w['blocks'] or []]) for w in st['workflows']][:1]} msgs {TOASTS[-2:]}")
        print("   EXC", [x.strip().splitlines()[-1] for x in EXC[e0:]])

if CASE == "pause_off":
    for rep in range(2):
        g.new("session", SHORT)
        g.menu(g.blk("fond"), pick="Pause vérif")
        b0 = state(ex)["workflows"][0]["bp"]
        ex.click_block(g.blk("fond"))
        g.act()
        ex.key(Qt.Key_D)
        b1 = state(ex)["workflows"][0]["bp"]
        t1 = TOASTS[-1:]
        ex.shot(f"dig_pause_off_{rep}")
        ex.click_block(g.blk("fond"))
        g.act()
        ex.key(Qt.Key_D)
        b2 = state(ex)["workflows"][0]["bp"]
        g.undo()
        g.undo()
        b3 = state(ex)["workflows"][0]["bp"]
        mm = g.menu(g.blk("fond"))
        m = g.texts(mm)
        print("   menu item when the pause is set:", [(x.get("text"), x.get("checked")) for x in mm if "ause" in (x.get("text") or "")])
        print(f"[{rep}] pause {b0}; D -> {b1} ({t1}); D again -> {b2}; Z Z -> {b3}; menu {[x for x in m if x and 'ause' in x]}")

if CASE == "join":
    for rep in range(2):
        for how in ("dbl", "plus"):
            g.new("session", ["calibration", "registration", "stacking"])
            w = ex.s.p.workflows[0]
            print(f"[{rep}] {how} before: sessions of the workflow {w.sessions} -> read: {ex.s.p.sessions_of(w.id)}")
            if how == "dbl":
                ex.double_click_card("session")
            else:
                ex.drag_palette("session", ex.ghost_center(ex.view.plus_session_ghost))
            ex.settle(8)
            w = ex.s.p.workflows[0]
            print(f"   after: sessions of the workflow {w.sessions} -> read: {ex.s.p.sessions_of(w.id)} ; "
                  f"project sessions {[(x.id, x.name, len(x.lights)) for x in ex.s.p.sessions]} ; toasts {TOASTS[-1:]}")
            ex.click_block(g.blk("empilement"))
            print("   Empilement panel:", [t for t in panel_info(ex)["texts"] if "ession" in t][:6])
            ex.shot(f"dig_join_{how}_{rep}")

if CASE == "filter_fusion":
    from PySide6.QtWidgets import QComboBox
    for rep in range(2):
        for f in ("OIII", "SII", ""):
            g.mono_fusion()
            e0 = len(EXC)
            ex.click_block(g.blk("session", sid=1))
            cb = next((c for c in ex.page.panel.findChildren(QComboBox) if c.isVisible() and c.findData("Ha") >= 0), None)
            ex.click(cb)
            cb.setCurrentIndex(cb.findData(f))
            ex.settle(10)
            show(f"[{rep}] filter of Ha's session -> {f!r}")
            fus = g.blk("fusion")
            print("   fusion:", fus.summary if fus else None, repr(fus.toolTip()[-160:]) if fus else None,
                  "EXC", [x.strip().splitlines()[-1] for x in EXC[e0:]])
            if rep == 0:
                ex.shot(f"dig_filter_fusion_{f or 'none'}")
            g.undo()
            print("   undo -> sessions", [(x["id"], x["name"], x["filter"]) for x in state(ex)["sessions"]],
                  "EXC", [x.strip().splitlines()[-1] for x in EXC[e0:]])

if CASE == "edit_menu":
    for rep in range(2):
        g.new("session", SHORT)
        a_u, a_r = ex.w.a_undo, getattr(ex.w, "a_redo", None)
        print(f"[{rep}] at open: undo «{a_u.text()}» {a_u.isEnabled()} redo «{a_r.text() if a_r else None}» "
              f"{a_r.isEnabled() if a_r else None} status «{ex.w.statusBar().currentMessage() if ex.w.statusBar() else ''}»")
        n = len(TOASTS)
        g.undo()
        print(f"   Ctrl+Z with nothing to undo: msgs {TOASTS[n:]} status «{ex.w.statusBar().currentMessage()}»")
        ex.click_block(g.blk("fond"))
        g.act()
        ex.key(Qt.Key_Delete)
        print(f"   after Suppr Gradient: undo «{a_u.text()}» {a_u.isEnabled()} redo {a_r.isEnabled() if a_r else None}")
        g.undo()
        print(f"   after Ctrl+Z: undo «{a_u.text()}» {a_u.isEnabled()} redo «{a_r.text()}» {a_r.isEnabled()} "
              f"status «{ex.w.statusBar().currentMessage()}» msgs {TOASTS[-1:]}")
        # nothing selected: D and Del
        ex.click(ex.view.viewport(), ex.view.viewport().rect().topLeft() + __import__("PySide6.QtCore", fromlist=["QPoint"]).QPoint(30, 30))
        print("   selected after a click on the empty graph:", ex.s.sel_step)
        ex.shot(f"dig_empty_click_{rep}")
        n = len(TOASTS)
        s0 = state(ex)
        g.act()
        ex.key(Qt.Key_Delete)
        ex.key(Qt.Key_D)
        print(f"   Suppr and D with the empty graph clicked: changed {state(ex) != s0} msgs {TOASTS[n:]} "
              f"graph {g.graph_of()}")

for e in EXC:
    print("EXC", e[-800:])
ex.close()
