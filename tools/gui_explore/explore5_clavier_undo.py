"""Round 4, area clavier_fenetre, axis 1b: a long sequence of varied gestures, then Ctrl+Z one by one back to the
start (each state compared with a full dump of the model), then Ctrl+Y all; undo past a Save; undo then a new gesture
(redo history dropped). Prints; writes candidates when EXPLORE_NOTE=1.

    python explore5_clavier_undo.py [long|save|mixed] [--lang en]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from explore_post_common import P, start  # noqa: E402  (harness first)
from explore2_blocks_common import G, Explorer, drawn  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtWidgets import QWidget, QComboBox, QLineEdit, QCheckBox, QSlider, QDoubleSpinBox, QSpinBox  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

args = sys.argv[1:]
LANG = "fr"
if "--lang" in args:
    i = args.index("--lang")
    LANG = args[i + 1]
    args = args[:i] + args[i + 2:]
WHAT = args or ["long", "save", "mixed"]
start(1500)
ex = Explorer("clavier_fenetre", lang=LANG)
g = G(ex)
pp = P(ex)
VOLATILE = ("current", "current_block", "next_workflow_id")


def dump() -> dict:
    d = json.loads(json.dumps(ex.s.p.to_dict(), default=str))
    d["_drawn"] = sorted((x["title"]) for x in drawn(ex))
    d["_paths"] = [q.name for q in ex.s.paths]
    return d


def diff(a: dict, b: dict, pre="") -> list[str]:
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                out += diff(a.get(k), b.get(k), f"{pre}.{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                out += diff(x, y, f"{pre}[{i}]")
    else:
        out.append(f"{pre}: {json.dumps(a, default=str)[:160]} -> {json.dumps(b, default=str)[:160]}")
    return out


def visible_diff(a, b):
    return [x for x in diff(a, b) if not any(x.startswith("." + v + ":") or x.startswith("." + v + ".") for v in VOLATILE)]


def type_into(ed: QLineEdit, text: str):
    ex.click(ed)
    ed.selectAll()
    QTest.keyClicks(ed, text)
    QTest.keyClick(ed, Qt.Key_Return)
    ex.settle(6)


def rename_to(name):
    def h(d):
        e = d.findChildren(QLineEdit)
        if e:
            e[0].setText(name)
        d.accept()
    return h


def blk(key, **k):
    return g.blk(key, **k)


def click(key, **k):
    it = blk(key, **k)
    if it is None:
        raise RuntimeError(f"no block {key}")
    ex.click_block(it)
    g.act()


def key_on(key, k, mod=Qt.NoModifier, **kw):
    click(key, **kw)
    pp.key(k, None, mod)
    ex.settle(4)


def menu(key, pick, **kw):
    it = blk(key, **kw)
    if it is None:
        raise RuntimeError(f"no block {key}")
    m = g.menu(it, pick)
    return m


def vis(cls):
    pn = ex.page.panel
    return [c for c in pn.findChildren(cls) if c.isVisibleTo(pn) and c.isEnabled()]


def panel_method(key, **kw):
    click(key, **kw)
    cards = [c for c in ex.page.panel.findChildren(QWidget) if type(c).__name__ in ("MethodCard", "StackMethodCard")
             and c.isVisibleTo(ex.page.panel)]
    if len(cards) < 2:
        raise RuntimeError(f"no 2 method cards in {key} panel")
    ex.click(cards[1])
    ex.settle(8)


def panel_spin(key, **kw):
    click(key, **kw)
    ss = vis(QDoubleSpinBox) + vis(QSpinBox)
    if not ss:
        raise RuntimeError(f"no spin box in {key} panel")
    ex.click(ss[0])
    QTest.keyClick(ss[0], Qt.Key_Up)
    QTest.keyClick(ss[0], Qt.Key_Return)
    ex.settle(8)


def panel_combo(key, idx_shift=1, **kw):
    click(key, **kw)
    cbs = [c for c in vis(QComboBox) if c.count() > 1]
    if not cbs:
        raise RuntimeError(f"no combo in {key} panel")
    c = cbs[0]
    c.setCurrentIndex((c.currentIndex() + idx_shift) % c.count())
    ex.settle(8)
    return c.currentText()


def panel_check(key, **kw):
    click(key, **kw)
    cbs = vis(QCheckBox)
    if not cbs:
        raise RuntimeError(f"no check box in {key} panel")
    ex.click(cbs[0])
    ex.settle(8)
    return cbs[0].text()


def panel_slider(key, **kw):
    click(key, **kw)
    ss = vis(QSlider)
    if not ss:
        raise RuntimeError(f"no slider in {key} panel")
    s = ss[0]
    s.setFocus()
    for _ in range(3):
        QTest.keyClick(s, Qt.Key_Right)
    ex.settle(8)
    return s.value()


def branch(block, under):
    pid = ex.s.paths[0].pid
    if not g.drag_branch(block, under, pid):
        raise RuntimeError(f"no branch ghost under {under}")


def rename_session(name):
    click("sources")
    ed = next((e for e in vis(QLineEdit) if e.text().startswith("Session") or e.text().startswith("Nuit")), None)
    if ed is None:
        ed = next(iter(vis(QLineEdit)), None)
    if ed is None:
        raise RuntimeError("no session name field")
    type_into(ed, name)


def rename_wf(key, name):
    ex.dialog_handler = rename_to(name)
    menu(key, "Renommer" if LANG == "fr" else "Rename")
    ex.settle(6)


def cross(key, **kw):
    g.cross(blk(key, **kw))


GESTURES = [
    ("pose Calibration", lambda: g.place("calibration")),
    ("pose Alignement", lambda: g.place("registration")),
    ("pose Empilement", lambda: g.place("stacking")),
    ("pose Astrométrie", lambda: g.place("astrometry")),
    ("pose Gradient", lambda: g.place("gradient")),
    ("pose Débruitage", lambda: g.place("denoise")),
    ("pose Couleur", lambda: g.place("color")),
    ("pose Étoiles", lambda: g.place("stars")),
    ("D sur Gradient", lambda: key_on("fond", Qt.Key_D)),
    ("D sur Gradient (rallume)", lambda: key_on("fond", Qt.Key_D)),
    ("Vers Post sur Empilement", lambda: menu("empilement", "Vers Post" if LANG == "fr" else "To Post")),
    ("réglage Gradient (méthode)", lambda: panel_method("fond")),
    ("réglage Débruitage (méthode)", lambda: panel_method("debruitage")),
    ("réglage Couleur (nombre)", lambda: panel_spin("couleur")),
    ("renomme la session (panneau)", lambda: rename_session("Nuit A")),
    ("renomme le workflow (menu)", lambda: rename_wf("couleur", "Mon flux")),
    ("Suppr sur Débruitage", lambda: key_on("debruitage", Qt.Key_Delete)),
    ("pose Gradient 2 (double-clic)", lambda: g.place("gradient")),
    ("croix sur Couleur", lambda: cross("couleur")),
    ("branche Débruitage sous Astrométrie", lambda: branch("denoise", "astrometrie")),
    ("Vers Post sur Astrométrie", lambda: menu("astrometrie", "Vers Post" if LANG == "fr" else "To Post")),
    ("D sur Empilement", lambda: key_on("empilement", Qt.Key_D)),
    ("pause vérif sur Alignement", lambda: menu("alignement", "Pause vérif" if LANG == "fr" else "Pause")),
    ("réglage Empilement (case)", lambda: panel_check("empilement")),
    ("réglage Alignement (nombre)", lambda: panel_spin("alignement")),
    ("pose Couleur à nouveau", lambda: g.place("color")),
    ("D sur Calibration", lambda: key_on("calibration", Qt.Key_D)),
    ("Suppr sur Étoiles", lambda: key_on("etoiles", Qt.Key_Delete)),
    ("Image sur la zone du bas", lambda: ex.drag_palette("image", ex.ghost_center(ex.new_flow_zone()))),
    ("réglage Astrométrie (liste)", lambda: panel_combo("astrometrie")),
    ("renomme la session encore", lambda: rename_session("Nuit B")),
    ("Suppr sur Gradient", lambda: key_on("fond", Qt.Key_Delete)),
    ("réglage Calibration (nombre)", lambda: panel_spin("calibration")),
    ("pose Sélection", lambda: g.place("selection")),
    ("Vers Post sur Gradient 2", lambda: menu("fond~2", "Vers Post" if LANG == "fr" else "To Post")),
    ("croix sur Astrométrie (refus attendu?)", lambda: cross("astrometrie")),
    ("pose Étoiles à nouveau", lambda: g.place("stars")),
    ("réglage Étoiles (méthode)", lambda: panel_method("etoiles")),
]


def long_seq(gestures=None, kind="session_empty_wf"):
    g.new(kind)
    ex.s._undo.clear()
    ex.s._redo.clear()
    states = {len(ex.s._undo): dump()}
    log = []
    for name, fn in (gestures or GESTURES):
        before = dump()
        nu = len(ex.s._undo)
        ok = True
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"SKIP {name}: {type(e).__name__}: {e}", flush=True)
        ex.settle(6)
        pp.check_slot_errors(name)
        after = dump()
        d = visible_diff(before, after)
        steps = len(ex.s._undo) - nu
        log.append((name, ok, steps, len(d)))
        print(f"G {name}: ok={ok} steps={steps} changed={len(d)} undo_len={len(ex.s._undo)} toast={pp.toast()!r}",
              flush=True)
        if d and steps == 0:
            print("   NOT UNDOABLE:", d[:6], flush=True)
        if steps > 1:
            print("   SEVERAL STEPS:", steps, flush=True)
        if steps == 0 and not d:
            print("   no change", flush=True)
        if steps >= 1:
            states[len(ex.s._undo)] = after
        elif d:
            # a change outside undo: the previous state is not reachable; record what is now at this depth
            states[len(ex.s._undo)] = after
    ex.shot("undo_long_end")
    end = dump()
    total = len(ex.s._undo)
    print("TOTAL undo entries:", total, flush=True)
    bad_u = []
    for k in range(total, 0, -1):
        pp.undo()
        ex.settle(6)
        pp.check_slot_errors(f"Ctrl+Z n°{total - k + 1}")
        now = dump()
        n = len(ex.s._undo)
        if n != k - 1:
            print(f"U depth {k}: stack {n} (expected {k - 1})", flush=True)
        exp = states.get(n)
        if exp is None:
            continue
        d = visible_diff(exp, now)
        if d:
            bad_u.append((k, d))
            print(f"U MISMATCH at depth {n}: {d[:8]}", flush=True)
            ex.shot(f"undo_mismatch_{n:02d}")
    print("after all undo: can_undo", ex.s.can_undo(), "diff to start:", visible_diff(states[0], dump())[:10], flush=True)
    ex.shot("undo_long_start_again")
    bad_r = []
    for k in range(1, total + 1):
        pp.redo()
        ex.settle(6)
        pp.check_slot_errors(f"Ctrl+Y n°{k}")
        now = dump()
        n = len(ex.s._undo)
        exp = states.get(n)
        if exp is None:
            continue
        d = visible_diff(exp, now)
        if d:
            bad_r.append((k, d))
            print(f"R MISMATCH at depth {n}: {d[:8]}", flush=True)
            ex.shot(f"redo_mismatch_{n:02d}")
    print("after all redo: diff to end:", visible_diff(end, dump())[:10], flush=True)
    ex.shot("redo_long_end")
    print("SUMMARY", json.dumps(log, ensure_ascii=False), flush=True)


def pid_named(name):
    return next(q.pid for q in ex.s.paths if q.name == name)


def click_pid(key, name):
    it = ex.block(key, pid=pid_named(name))
    if it is None:
        raise RuntimeError(f"no {key} in {name}")
    ex.click_block(it)
    g.act()


def del_wire(i=0):
    ws = [w for w in pp.wires() if pp.free_point_on(w) is not None]
    if len(ws) <= i:
        raise RuntimeError("no removable wire")
    pp.click_wire(ws[i])
    pp.del_key()


def session_filter(sid, data):
    it = ex.block("session", sid=sid) or ex.block("sources")
    ex.click_block(it)
    cb = next((c for c in vis(QComboBox) if c.findData(data) >= 0), None)
    if cb is None:
        raise RuntimeError("no filter combo")
    cb.setCurrentIndex(cb.findData(data))
    ex.settle(8)


def rename_sid(sid, name):
    it = ex.block("session", sid=sid)
    if it is None:
        raise RuntimeError(f"no session {sid}")
    ex.click_block(it)
    ed = next(iter(vis(QLineEdit)), None)
    type_into(ed, name)


def image_kind():
    it = ex.block("image")
    ex.click_block(it)
    cbs = [c for c in vis(QComboBox) if c.count() > 1]
    if not cbs:
        raise RuntimeError("no kind combo on Image")
    cbs[0].setCurrentIndex((cbs[0].currentIndex() + 1) % cbs[0].count())
    ex.settle(8)


def cross_sid(sid):
    g.cross(ex.block("session", sid=sid))


TP = "To Post"
GESTURES2 = [
    ("place chain on Workflow (calibration)", lambda: (click_pid("session", "Workflow") if False else None,
                                                       ex.click_block(ex.block("session", sid=1)), g.place("calibration"))),
    ("place registration", lambda: g.place("registration")),
    ("place stacking", lambda: g.place("stacking")),
    ("place astrometry", lambda: g.place("astrometry")),
    ("select Session 2, place calibration", lambda: (ex.click_block(ex.block("session", sid=2)), g.act(),
                                                    g.place("calibration"))),
    ("place registration (flow 2)", lambda: g.place("registration")),
    ("place stacking (flow 2)", lambda: g.place("stacking")),
    ("rename Session 2 (panel)", lambda: rename_sid(2, "Night B")),
    ("Session 1 filter Ha", lambda: session_filter(1, "Ha")),
    ("gradient on flow 1", lambda: (click_pid("astrometrie", "Workflow"), g.place("gradient"))),
    ("Del on Workflow wire to Post", lambda: del_wire(0)),
    ("To Post on flow 1 Gradient", lambda: menu("fond", TP)),
    ("Image on bottom zone", lambda: ex.drag_palette("image", ex.ghost_center(ex.new_flow_zone()))),
    ("gradient after Image", lambda: (ex.click_block(ex.block("image")), g.act(), g.place("gradient"))),
    ("Image kind changed", lambda: image_kind()),
    ("D on flow 2 Registration", lambda: (click_pid("alignement", "Flow"), pp.key(Qt.Key_D))),
    ("cross on Session 2", lambda: cross_sid(2)),
    ("Del on flow 1 Gradient", lambda: (click_pid("fond", "Workflow"), pp.key(Qt.Key_Delete))),
    ("denoise on flow 1", lambda: (click_pid("astrometrie", "Workflow"), g.place("denoise"))),
    ("Session 1 filter back", lambda: session_filter(1, "")),
    ("Del on a wire", lambda: del_wire(0)),
    ("color on flow 1", lambda: (click_pid("debruitage", "Workflow"), g.place("color"))),
    ("stars on flow 1", lambda: g.place("stars")),
    ("Recombine", lambda: g.place("composition")),
    ("Del on Recombine", lambda: (ex.click_block(ex.block("composition")), pp.key(Qt.Key_Delete))),
    ("rename Session 1", lambda: rename_sid(1, "Night A")),
    ("pause on flow 1 Stacking", lambda: menu("empilement", "Pause", pid=pid_named("Workflow"))),
    ("D on Stars", lambda: (ex.click_block(ex.block("etoiles")), pp.key(Qt.Key_D))),
    ("Session card double-click (new session?)", lambda: g.place("session")),
    ("Image card double-click", lambda: g.place("image")),
]


def save_seq():
    g.new("session_empty_wf")
    ex.s._undo.clear()
    ex.s._redo.clear()
    s0 = dump()
    for b in ("calibration", "registration", "stacking"):
        g.place(b)
    s1 = dump()
    title1 = ex.w.windowTitle()
    pp.key(Qt.Key_S, ex.w, Qt.ControlModifier)
    ex.settle(10)
    disk = json.loads((Path(ex.s.p.root) / "project.astroproj").read_text(encoding="utf-8")) \
        if hasattr(ex.s.p, "root") and (Path(ex.s.p.root) / "project.astroproj").exists() else None
    print("after save: title", ex.w.windowTitle(), "(before", title1, ") status", ex.w.status.text(),
          "undo", len(ex.s._undo), "dirty", getattr(ex.s, "_dirty", None), flush=True)
    for b in ("astrometry", "gradient"):
        g.place(b)
    print("after 2 more: title", ex.w.windowTitle(), "undo", len(ex.s._undo), flush=True)
    for _ in range(2):
        pp.undo()
    print("back to save point: diff", visible_diff(s1, dump())[:5], "title", ex.w.windowTitle(),
          "dirty", getattr(ex.s, "_dirty", None), flush=True)
    ex.shot("save_point")
    for _ in range(3):
        pp.undo()
    print("past save: diff to start", visible_diff(s0, dump())[:5], "title", ex.w.windowTitle(),
          "dirty", getattr(ex.s, "_dirty", None), "can_undo", ex.s.can_undo(), "toast", pp.toast(), flush=True)
    ex.shot("past_save")
    for _ in range(5):
        pp.redo()
    print("redo all: blocks", g.titles(), flush=True)
    if disk is not None:
        print("disk workflows", [[b["block"] for b in (w.get("blocks") or [])] for w in disk["workflows"]], flush=True)


def mixed_seq():
    g.new("session_empty_wf")
    ex.s._undo.clear()
    ex.s._redo.clear()
    for b in ("calibration", "registration", "stacking", "astrometry", "gradient", "color"):
        g.place(b)
    for _ in range(3):
        pp.undo()
    mid = dump()
    print("after 3 undo:", g.titles(), "redo", len(ex.s._redo), flush=True)
    g.place("denoise") if g.blk("debruitage") is None else None
    print("new gesture:", g.titles(), "can_redo", ex.s.can_redo(), "redo", len(ex.s._redo), flush=True)
    s_new = dump()
    pp.redo()
    print("Ctrl+Y after new gesture: changed", visible_diff(s_new, dump())[:4], "toast", pp.toast(), flush=True)
    pp.undo()
    print("Ctrl+Z: back to mid", visible_diff(mid, dump())[:4], flush=True)
    ex.shot("mixed_end")
    # undo in a text field: Ctrl+Z while the session name field has the focus
    click("sources")
    ed = next((e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisible() and e.isEnabled()), None)
    if ed is not None:
        before = dump()
        ex.click(ed)
        ed.end(False)
        QTest.keyClicks(ed, "xyz")
        ex.settle(3)
        pp.activate()
        ed.setFocus()
        QTest.keyClick(ed, Qt.Key_Z, Qt.ControlModifier)
        ex.settle(6)
        print("Ctrl+Z in name field: field", repr(ed.text()), "model diff", visible_diff(before, dump())[:4],
              "focus", type(ex.app.focusWidget()).__name__, flush=True)
        for k in (Qt.Key_Delete, Qt.Key_D, Qt.Key_Backspace):
            b2 = dump()
            QTest.keyClick(ed, k)
            ex.settle(4)
            print("key", k, "in field:", repr(ed.text()), "model diff", visible_diff(b2, dump())[:4], flush=True)


for w in WHAT:
    print("=====", w, flush=True)
    {"long": long_seq, "save": save_seq, "mixed": mixed_seq,
     "long2": lambda: long_seq(GESTURES2, "two_flows")}[w]()
ex.close()
