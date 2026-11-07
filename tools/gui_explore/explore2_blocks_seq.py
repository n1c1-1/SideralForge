"""block_actions round 2: sessions, wires, branches, Post composition and sequences of gestures with Ctrl+Z / Ctrl+Y.

    python explore2_blocks_seq.py [case ...] [--lang en]
"""
from __future__ import annotations

import faulthandler
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(900, exit=True)

from explore2_blocks_common import CHAIN, EXC, SHORT, TOASTS, G, Explorer, drawn, panel_info, state  # noqa: E402,F401
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QPushButton  # noqa: E402

args = sys.argv[1:]
LANG = "fr"
if "--lang" in args:
    i = args.index("--lang")
    LANG = args[i + 1]
    args = args[:i] + args[i + 2:]
ONLY = args
ex = Explorer("block_actions", lang=LANG)
g = G(ex)
CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


def snap():
    st = state(ex)
    return json.dumps({"w": [(w["name"], w["blocks"], w["sessions"], w["bp"]) for w in st["workflows"]],
                       "s": [(x["id"], x["name"], x["filter"], x["lights"]) for x in st["sessions"]],
                       "post": getattr(ex.s.p, "post", None)}, sort_keys=True, default=str)


def short(s):
    d = json.loads(s)
    return {"w": [(w[0], [b[0] + ("/off" if b[3] else "") for b in (w[1] or [])], w[2], w[3]) for w in d["w"]],
            "s": d["s"], "post": d["post"]}


def report(label, s0, s1, s2, s3, t, extra=""):
    ok_u, ok_r = s2 == s0, s3 == s1
    flag = "" if (s1 != s0 and ok_u and ok_r) else "SUSPECT "
    print(f"{flag}{label}: changed={s1 != s0} undo_ok={ok_u} redo_ok={ok_r} undo_n={len(ex.s._undo)} msgs={t} {extra}",
          flush=True)
    if flag:
        print("    before:", short(s0))
        print("    after: ", short(s1))
        print("    undo:  ", short(s2))
        print("    redo:  ", short(s3), flush=True)


def ur(label, fn, extra=lambda: ""):
    """One user step, then one Ctrl+Z and one Ctrl+Y."""
    s0 = snap()
    n = len(TOASTS)
    nu = len(ex.s._undo)
    fn()
    s1 = snap()
    t = TOASTS[n:]
    e = extra()
    steps = len(ex.s._undo) - nu
    g.undo()
    s2 = snap()
    g.redo()
    s3 = snap()
    report(label + f" (undo entries +{steps})", s0, s1, s2, s3, t, e)
    return s0, s1


def type_into(ed: QLineEdit, text: str):
    ex.click(ed)
    ed.selectAll()
    from PySide6.QtTest import QTest
    QTest.keyClicks(ed, text)
    QTest.keyClick(ed, Qt.Key_Return)
    ex.settle(6)


@case
def session_rename_panel():
    for rep in range(2):
        g.new("two_sessions", ["calibration", "registration", "stacking"])
        ex.click_block(g.blk("session", sid=1))
        ed = next((e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisible() and e.text() == "Session 1"), None)
        if ed is None:
            print("no name field")
            return
        ur(f"[{rep}] rename session in its panel", lambda: type_into(ed, "Nuit A"),
           lambda: str([d["title"] for d in drawn(ex)][:3]))


@case
def session_filter():
    for rep in range(2):
        g.new("two_sessions", ["calibration", "registration", "stacking"])
        ex.click_block(g.blk("session", sid=1))
        cb = next((c for c in ex.page.panel.findChildren(QComboBox) if c.isVisible() and c.findData("Ha") >= 0), None)
        if cb is None:
            print("no filter combo")
            return

        def pick():
            ex.click(cb)
            cb.setCurrentIndex(cb.findData("Ha"))
            ex.settle(8)
        ur(f"[{rep}] session filter -> Ha", pick, lambda: str([(d["title"], d["summary"][:25]) for d in drawn(ex)][:4]))
        ex.shot(f"seq_filter_{rep}")


@case
def add_session_menu():
    for rep in range(2):
        g.new("session", ["calibration", "registration", "stacking"])
        d = ex.raws(f"extra{rep}", 5, start=7000 + rep * 10)
        ex.file_queue = [str(d), str(d), [str(p) for p in sorted(d.iterdir())]]
        md = len(ex.modals)

        def acc(dlg):
            ok = next((b for b in dlg.findChildren(QPushButton) if b.isVisible() and b.isDefault()), None)
            print("    dialog:", dlg.__class__.__name__, dlg.windowTitle(),
                  [b.text() for b in dlg.findChildren(QPushButton) if b.isVisible()])
            ok.click() if ok is not None else dlg.accept()
        ex.dialog_handler = acc
        ur(f"[{rep}] Ajouter session… (menu of the Session)", lambda: g.menu(g.blk("sources"), pick="Ajouter session"),
           lambda: f"modals={ex.modals[md:]} files_asked={ex.file_asks[-3:]} drawn={[x['title'] for x in drawn(ex)][:4]}")
        ex.dialog_handler = None
        ex.file_queue = []
        ex.shot(f"seq_add_session_{rep}")


@case
def plus_session_ghost():
    for rep in range(2):
        g.new("session", ["calibration", "registration", "stacking"])
        gh = next((x for x in ex.view.scene().items() if getattr(x, "text", "") and "Session" in str(x.text)
                   and getattr(x, "on_click", None)), None)
        if gh is None:
            print("no + Session place")
            return
        md = len(ex.modals)
        ur(f"[{rep}] click « + Session »", lambda: ex.click(ex.view.viewport(), ex.ghost_center(gh)),
           lambda: f"modals={ex.modals[md:]} files={ex.file_asks[-2:]} drawn={[x['title'] for x in drawn(ex)][:4]}")
        ex.shot(f"seq_plus_session_{rep}")


def wires():
    from astro_gpu.client.graph import WireItem
    return [x for x in ex.view.scene().items() if isinstance(x, WireItem) and x.isVisible()]


@case
def wire_remove():
    for rep in range(2):
        g.new("session", SHORT)
        g.menu(g.blk("fond"), pick="Vers Post")
        ex.s._undo.clear()
        ex.s._redo.clear()
        ws = wires()
        print(f"  [{rep}] wires to Post: {len(ws)} tips={[w.toolTip().splitlines()[0] for w in ws]}")
        if not ws:
            return
        w = ws[0]
        tip = w.toolTip().splitlines()[0]
        pt = ex.view.mapFromScene(w.path().pointAtPercent(0.5))

        def click_del():
            ex.hover(ex.view.viewport(), pt)
            ex.click(ex.view.viewport(), pt)
            g.act()
            ex.key(Qt.Key_Delete)
        ur(f"[{rep}] click wire « {tip} » + Suppr", click_del,
           lambda: f"wires left={len(wires())} drawn={[x['title'] for x in drawn(ex)][-4:]}")
        ex.shot(f"seq_wire_del_{rep}")
        g.new("session", SHORT)
        g.menu(g.blk("fond"), pick="Vers Post")
        ex.s._undo.clear()
        ex.s._redo.clear()
        w = wires()[0]
        pt = ex.view.mapFromScene(w.path().pointAtPercent(0.5))

        def right():
            vp = ex.view.viewport()
            ex.menu_pick = "Retirer" if LANG == "fr" else "Remove"
            QApplication.sendEvent(vp, QContextMenuEvent(QContextMenuEvent.Mouse, pt, vp.mapToGlobal(pt)))
            ex.settle(8)
        e0 = len(EXC)
        ur(f"[{rep}] right-click wire → Retirer", right,
           lambda: f"menu={[x.get('text') for x in ex.menus[-1]]} wires left={len(wires())} exc={EXC[e0:][-1:] }")


@case
def sequence_two_steps():
    for rep in range(2):
        g.new("session", SHORT)
        s0 = snap()
        ex.click_block(g.blk("fond"))
        g.act()
        ex.key(Qt.Key_Delete)
        s1 = snap()
        ex.click_block(g.blk("couleur"))
        g.act()
        ex.key(Qt.Key_D)
        s2 = snap()
        g.undo()
        a = snap()
        g.undo()
        b = snap()
        g.redo()
        c = snap()
        g.redo()
        d = snap()
        ok = (a == s1, b == s0, c == s1, d == s2)
        print(("" if all(ok) else "SUSPECT ") + f"[{rep}] Del Gradient, D Couleur, Z, Z, Y, Y: {ok}", flush=True)
        # a new step after Ctrl+Z: Ctrl+Y must do nothing
        g.undo()
        ex.click_block(g.blk("calibration"))
        g.act()
        ex.key(Qt.Key_D)
        e = snap()
        g.redo()
        f = snap()
        print(("" if e == f else "SUSPECT ") + f"[{rep}] Ctrl+Y after a new step does nothing: {e == f}", flush=True)


@case
def branch_delete():
    for rep in range(2):
        g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
        pid = ex.s.paths[0].pid
        g.drag_branch("color", "astrometrie", pid)
        ex.s._undo.clear()
        ex.s._redo.clear()
        names = [q.name for q in ex.s.paths]
        cols = [it for it in ex.blocks() if it.node.key == "couleur"]
        if not cols:
            print("no branch", names)
            return
        ur(f"[{rep}] Suppr on the branch's Couleur (workflows {names})",
           lambda: (ex.click_block(cols[0]), g.act(), ex.key(Qt.Key_Delete)),
           lambda: f"workflows={[q.name for q in ex.s.paths]} drawn={[x['title'] for x in drawn(ex)]}")
        ex.shot(f"seq_branch_del_{rep}")
        g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
        g.drag_branch("color", "astrometrie", ex.s.paths[0].pid)
        ex.s._undo.clear()
        ex.s._redo.clear()
        ur(f"[{rep}] Suppr on Gradient (main line, branch Couleur kept?)",
           lambda: (ex.click_block(g.blk("fond")), g.act(), ex.key(Qt.Key_Delete)),
           lambda: f"workflows={[q.name for q in ex.s.paths]} drawn={[x['title'] for x in drawn(ex)]}")
        ex.shot(f"seq_branch_main_del_{rep}")


@case
def pause_toggle():
    for rep in range(2):
        g.new("session", SHORT)
        m1 = g.texts(g.menu(g.blk("fond"), pick="Pause vérif"))
        m2 = g.texts(g.menu(g.blk("fond")))
        bp1 = state(ex)["workflows"][0]["bp"]
        n = len(TOASTS)
        pick = next((x for x in m2 if x and ("pause" in x.lower() or "Pause" in x)), None)
        g.menu(g.blk("fond"), pick=pick.split("\t")[0] if pick else "Pause")
        bp2 = state(ex)["workflows"][0]["bp"]
        print(f"[{rep}] pause: menu after {[x for x in m2 if x and 'ause' in x]} bp {bp1} -> {bp2} msgs {TOASTS[n:]}",
              flush=True)
        ex.shot(f"seq_pause_{rep}")


@case
def stars_parts():
    for rep in range(2):
        g.new("session", CHAIN)
        ur(f"[{rep}] Sans étoiles vers Post", lambda: g.menu(g.blk("etoiles"), pick="Sans étoiles vers Post"),
           lambda: str([x for x in g.texts(g.menu(g.blk("etoiles"))) if x and "Post" in x]))
        opp = next((x for x in g.texts(g.menu(g.blk("etoiles"))) if x and "Sans étoiles" in x), "Sans étoiles")
        ur(f"[{rep}] then its opposite entry « {opp} »", lambda: g.menu(g.blk("etoiles"), pick=opp),
            lambda: str([x for x in g.texts(g.menu(g.blk("etoiles"))) if x and "Post" in x]))
        n = len(TOASTS)
        ur(f"[{rep}] Étoiles vers Post", lambda: g.menu(g.blk("etoiles"), pick="Étoiles vers Post"),
           lambda: f"{TOASTS[n:]} layers={[x.get('name') for x in state(ex)['post']]}")
        ur(f"[{rep}] Recombiner block deleted then", lambda: g.cross(next(b for b in ex.blocks()
                                                                          if b.node.key == "composition")),
           lambda: str([x["title"] for x in drawn(ex)][-3:]))
        ur(f"[{rep}] + Recombiner (menu)", lambda: g.menu(g.blk("etoiles"), pick="+ Recombiner"),
           lambda: str([x["title"] for x in drawn(ex)][-4:]))
        ex.shot(f"seq_stars_parts_{rep}")


@case
def workflow_rename_panel():
    for rep in range(2):
        g.new("session", SHORT)
        ex.click_block(g.blk("couleur"))
        g.menu(g.blk("couleur"), pick=None)
        # the workflow's name: its handle block's panel shows it in an edit field
        ed = next((e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisible() and e.text() == "Workflow"), None)
        if ed is None:
            print(f"[{rep}] no workflow name field in the Couleur panel:",
                  [e.text() for e in ex.page.panel.findChildren(QLineEdit) if e.isVisible()])
            return
        ur(f"[{rep}] rename workflow in panel", lambda: type_into(ed, "Mon flux"))


@case
def mono_session_removed():
    for rep in range(2):
        g.mono_fusion()
        g.cross(g.blk("session", sid=1))
        ph = next((it for it in ex.blocks() if it.node.key == "session" and it.title == "Ha"), None)
        fus = g.blk("fusion")
        print(f"[{rep}] after cross on Ha: placeholder={ph.summary if ph else None} fusion tip="
              f"{fus.toolTip()[:200] if fus else None!r}", flush=True)
        if ph is not None:
            m = g.texts(g.menu(ph))
            print("    placeholder menu:", m)
        ex.click_block(g.blk("fusion"))
        print("    fusion panel:", panel_info(ex)["texts"][:8])
        ex.hover(ex.view.viewport(), ex.point_of(g.blk("fusion")))
        ex.shot(f"seq_mono_removed_{rep}")
        ex.click(ex.w.findChild(QPushButton, None) if False else ex.view.viewport(), QPoint(5, 5))
        n = len(TOASTS)
        run_btn = next((b for b in ex.w.findChildren(QPushButton) if b.isVisible() and b.text().startswith(
            "Exécuter" if LANG == "fr" else "Run")), None)
        print("    run button:", run_btn.text() if run_btn else None, run_btn.isEnabled() if run_btn else None,
              repr(run_btn.toolTip()[:200]) if run_btn else None)


@case
def rename_mistakes():
    for rep in range(2):
        for txt in ("", "   ", "Session 2"):
            g.new("two_sessions", ["calibration", "registration", "stacking"])
            ex.click_block(g.blk("session", sid=1))
            ed = next((e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisible() and e.text() == "Session 1"), None)
            n = len(TOASTS)
            ed.selectAll()
            ex.click(ed)
            ed.selectAll()
            from PySide6.QtTest import QTest
            QTest.keyClick(ed, Qt.Key_Delete)
            if txt:
                QTest.keyClicks(ed, txt)
            QTest.keyClick(ed, Qt.Key_Return)
            ex.settle(6)
            print(f"[{rep}] session renamed to {txt!r} in its panel: sessions="
                  f"{[(x['id'], x['name']) for x in state(ex)['sessions']]} drawn={[d['title'] for d in drawn(ex)][:2]} "
                  f"msgs={TOASTS[n:]}", flush=True)
            if rep == 0:
                ex.shot(f"seq_rename_{len(txt)}")
        for txt in ("", "Workflow"):
            g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
            g.drag_branch("color", "astrometrie", ex.s.paths[0].pid)
            n = len(TOASTS)

            def h(d, txt=txt):
                for e in d.findChildren(QLineEdit):
                    e.setText(txt)
                d.accept()
            ex.dialog_handler = h
            g.menu(g.blk("couleur"), pick="Renommer")      # the branch's last block: its workflow's menu
            ex.dialog_handler = None
            print(f"[{rep}] workflow « Workflow · Couleur » renamed to {txt!r}: workflows={[q.name for q in ex.s.paths]} "
                  f"msgs={TOASTS[n:]} popups={ex.popups[-1:]}", flush=True)
            if rep == 0:
                ex.shot(f"seq_rename_wf_{len(txt)}")


@case
def lone_session_del():
    for rep in range(2):
        g.new("session", ["calibration", "registration", "stacking"])
        n = len(TOASTS)
        ex.click_block(g.blk("sources"))
        g.act()
        ex.key(Qt.Key_Delete)
        t1 = TOASTS[n:]
        ex.drag_block_to_palette(g.blk("sources"))
        t2 = TOASTS[n:]
        print(f"[{rep}] lone Session: Suppr -> {t1} ; dragged to palette -> {t2} ; still {[d['title'] for d in drawn(ex)][:1]}",
              flush=True)
        ex.shot(f"seq_lone_session_del_{rep}")


@case
def combos():
    for rep in range(2):
        # a block switched off, then deleted; two Ctrl+Z
        g.new("session", SHORT)
        s0 = snap()
        ex.click_block(g.blk("fond"))
        g.act()
        ex.key(Qt.Key_D)
        s1 = snap()
        ex.click_block(g.blk("fond"))
        g.act()
        ex.key(Qt.Key_Delete)
        g.undo()
        a = snap()
        g.undo()
        b = snap()
        print(("" if (a == s1 and b == s0) else "SUSPECT ") + f"[{rep}] D then Suppr on Gradient, Z, Z: {a == s1}, {b == s0}")
        # the menu of a switched-off block: Vers Post?
        g.new("session", SHORT)
        ex.click_block(g.blk("fond"))
        g.act()
        ex.key(Qt.Key_D)
        m = g.texts(g.menu(g.blk("fond")))
        print(f"[{rep}] menu of a switched-off Gradient: {m}")
        n = len(TOASTS)
        if "Vers Post" in m:
            g.menu(g.blk("fond"), pick="Vers Post")
            print(f"    Vers Post on it: {TOASTS[n:]} workflows {[(q.name, g.graph_of(q.name)) for q in ex.s.paths]}")
            ex.shot(f"seq_off_vers_post_{rep}")
        # a block with a pause deleted, then Ctrl+Z: the pause comes back?
        g.new("session", SHORT)
        g.menu(g.blk("fond"), pick="Pause vérif")
        bp0 = state(ex)["workflows"][0]["bp"]
        ex.click_block(g.blk("fond"))
        g.act()
        ex.key(Qt.Key_Delete)
        bp1 = state(ex)["workflows"][0]["bp"]
        g.undo()
        bp2 = state(ex)["workflows"][0]["bp"]
        print(("" if bp2 == bp0 else "SUSPECT ") + f"[{rep}] pause on Gradient {bp0}, Gradient deleted {bp1}, Ctrl+Z {bp2}")
        # Ctrl+Z while the session name field has the focus (typed, not validated)
        g.new("two_sessions", ["calibration", "registration", "stacking"])
        ex.click_block(g.blk("alignement"))
        g.act()
        ex.key(Qt.Key_Delete)                     # a graph edit to undo
        ex.click_block(g.blk("session", sid=1))
        ed = next((e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisible() and e.text() == "Session 1"), None)
        from PySide6.QtTest import QTest
        ex.click(ed)
        QTest.keyClicks(ed, "XY")
        s_typed = snap()
        QTest.keyClick(ed, Qt.Key_Z, Qt.ControlModifier)
        ex.settle(6)
        txt = ed.text() if ed is not None and __import__("shiboken6").isValid(ed) else None
        print(f"[{rep}] Ctrl+Z in the session name field: field now {txt!r}; graph edit undone: "
              f"{'registration' in g.graph_of()} ; sessions {[(x['id'], x['name']) for x in state(ex)['sessions']]}")
        # another project opened: Ctrl+Z does not reach the first one
        g.new("session", SHORT)
        ex.click_block(g.blk("fond"))
        g.act()
        ex.key(Qt.Key_Delete)
        g.new("session", ["calibration"], clear=False)
        before = snap()
        nu = len(ex.s._undo)
        g.undo()
        print(f"[{rep}] new project opened, undo stack {nu}, Ctrl+Z changed it: {snap() != before} -> "
              f"{g.graph_of()}")


@case
def session_join():
    for rep in range(2):
        for how in ("double-click Session card", "Session on the Flux zone", "Image on the Flux zone",
                    "Session on « + Session »"):
            g.new("session", ["calibration", "registration", "stacking"])
            d = ex.raws(f"join{rep}{len(how)}", 4, start=8000 + rep * 10)
            ex.file_queue = [str(d), str(d)]
            md = len(ex.modals)

            def acc(dlg):
                ok = next((b for b in dlg.findChildren(QPushButton) if b.isVisible() and b.isDefault()), None)
                ok.click() if ok is not None else dlg.accept()
            ex.dialog_handler = acc

            def act_():
                if how.startswith("double"):
                    ex.double_click_card("session")
                elif how.startswith("Session on the"):
                    ex.drag_palette("session", ex.ghost_center(ex.new_flow_zone()))
                elif how.startswith("Image"):
                    ex.drag_palette("image", ex.ghost_center(ex.new_flow_zone()))
                else:
                    ex.drag_palette("session", ex.ghost_center(ex.view.plus_session_ghost))
                ex.settle(8)
            ur(f"[{rep}] {how}", act_, lambda: f"modals={ex.modals[md:]} files={ex.file_asks[-1:]} "
               f"workflows={[(q.name, ex.s.p.workflow(q.pid).sessions) for q in ex.s.paths]} "
               f"drawn={[x['title'] for x in drawn(ex)][:4]}")
            ex.dialog_handler = None
            ex.file_queue = []


for name, fn in CASES.items():
    if ONLY and name not in ONLY:
        continue
    print("#####", name, flush=True)
    e0 = len(EXC)
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        import traceback
        print("  script error", e, traceback.format_exc()[-1200:], flush=True)
    for x in EXC[e0:]:
        print("  EXC", x[-700:], flush=True)
ex.close()
