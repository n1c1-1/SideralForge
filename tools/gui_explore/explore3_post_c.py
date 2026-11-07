"""post_wires round 3 (control), part C: last rechecks (Canaux liés undo, Hors Post message) and a short look for
regressions around the fixes (wire menu, Post-page delete undo, English, 1366). Each scenario twice. Prints only.
Args: [lang] scenarios..."""
import sys
import time

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QAbstractButton, QCheckBox, QLineEdit, QPushButton

args = sys.argv[1:]
lang = args.pop(0) if args and args[0] in ("fr", "en") else "fr"
size = (1366, 768) if "1366" in args else (1600, 950)
args = [a for a in args if a != "1366"]
start(1500)
ex = Explorer("post_wires", lang=lang, size=size)
pp = P(ex)
RUN = [1]
TO_POST = "Vers Post" if lang == "fr" else "To Post"


def wait(sec):
    t = time.time()
    while time.time() - t < sec:
        ex.settle(5)


def simulate():
    s = ex.s
    orig = s._sync

    def fake(*a, **k):
        orig(*a, **k)
        for v in s.paths:
            v.rendered = True
            v.computed = False
    s._sync = fake
    s._sync()


def shot(name):
    return ex.shot(f"{name}_{lang}{size[0]}_run{RUN[0]}")


def state():
    return {"paths": [q.name for q in ex.s.paths], "wires": [w.toolTip().splitlines()[0] for w in pp.wires()],
            "final": pp.final_summary(), "layers": pp.graph_layers(), "toast": pp.toast()}


def post_page():
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    return ex.w.post_holder.page.view


def graph_page():
    ex.click(ex.w.mode_btns[0])
    ex.settle(8)


def c_linked_undo():                         # R16
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    v = post_page()
    lay = next(lay for lay in v.layers if lay["kind"] == "workflow")
    v.select(lay["id"], "layer")
    ex.settle(4)
    cb = next((x for x in v.right.findChildren(QAbstractButton) if x.isVisible() and x.text() in ("Canaux liés", "Linked channels")), None)
    if cb is None:
        print(" no linked box", [x.text() for x in v.right.findChildren(QAbstractButton) if x.isVisible()])
        return
    before = cb.isChecked()
    ex.click(cb)
    ex.settle(6)
    mid = cb.isChecked()
    pp.undo()
    ex.settle(8)
    cb2 = next((x for x in v.right.findChildren(QAbstractButton) if x.isVisible() and x.text() == cb.text()), None)
    after = cb2.isChecked() if cb2 else None
    pp.redo()
    ex.settle(8)
    cb3 = next((x for x in v.right.findChildren(QAbstractButton) if x.isVisible() and x.text() == cb.text()), None)
    print(" linked", before, "-> click", mid, "-> Ctrl+Z", after, "-> Ctrl+Y", cb3.isChecked() if cb3 else None,
          "tip", cb.toolTip()[:200])
    shot("c_linked_undo")
    graph_page()


def c_split_out():                           # R38
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    ex.click_block(pp.block("etoiles"))
    b = [x for x in ex.page.panel.findChildren(QPushButton) if x.isVisible() and x.text() == TO_POST]
    ex.click(b[0])
    ex.settle(6)
    ex.click_block(pp.block("etoiles"))
    print(" split panel", [(x.text(), x.toolTip()[:80]) for x in ex.page.panel.findChildren(QPushButton) if x.isVisible()])
    shot("c_split_panel")
    out = [x for x in ex.page.panel.findChildren(QPushButton) if x.isVisible() and x.text() in ("Hors Post", "Out of Post")]
    if out:
        wait(6)
        ex.click(out[0])
        ex.settle(6)
        print(" after Hors Post", state())
        shot("c_split_out")
        pp.undo()
        print(" undo", state())


def c_menu_order():                          # check: where "Vers Post" sits in block menus
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    for k in ("empilement", "fond", "debruitage", "couleur"):
        print(" menu", k, [d.get("text") for d in ex.right_click_block(pp.block(k))])


def c_wire_menu():                           # check around the wire menu (Duplicate, Delete workflow, undo)
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), TO_POST)
    ex.settle(6)
    w = next(w for w in pp.wires() if "Gradient" in w.toolTip())
    m = pp.right_click_wire(w, "Dupliquer" if lang == "fr" else "Duplicate")
    ex.settle(6)
    print(" wire menu", [(d.get("text"), d.get("enabled")) for d in m or []])
    print(" after duplicate", state())
    shot("c_wire_dup")
    w = next(w for w in pp.wires() if "Gradient" in w.toolTip())
    pp.right_click_wire(w, "Supprimer workflow" if lang == "fr" else "Delete workflow")
    ex.settle(6)
    print(" after delete workflow", state(), ex.popups[-1:], ex.modals[-1:])
    pp.undo()
    print(" undo", state())
    pp.redo()
    print(" redo", state())
    # main workflow wire: its menu
    w = next(w for w in pp.wires() if w.toolTip().startswith("« Workflow »") or "Workflow »" in w.toolTip())
    print(" main wire menu", [(d.get("text"), d.get("enabled"), d.get("tip")) for d in pp.right_click_wire(w) or []])
    shot("c_wire_main_menu")


def c_post_delete_undo():                    # check around n1: the layer deleted in Post, then Ctrl+Z
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    ex.right_click_block(pp.block("fond"), TO_POST)
    ex.settle(6)
    v = post_page()
    tgt = next(lay for lay in v.layers if "Gradient" in lay["name"])
    v.select(tgt["id"], "layer")
    ex.settle(3)
    ex.click(v.b_del)
    ex.settle(10)
    print(" post rows after delete", [r.n.text() for r in v.rows], "post undo enabled", v.b_undo.isEnabled())
    pp.undo()
    ex.settle(8)
    print(" post rows after Ctrl+Z (in Post)", [r.n.text() for r in v.rows])
    graph_page()
    print(" graph after", state())
    pp.undo()
    print(" graph after Ctrl+Z in graph", state())
    shot("c_post_delete_undo")


def c_texts():                               # words on the wires / Post at this language and size
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    ex.right_click_block(pp.block("fond"), TO_POST)
    ex.settle(6)
    print(" toast", pp.toast())
    for w in pp.wires():
        print(" wire tip", w.toolTip().replace("\n", " | "))
        print(" wire menu", [d.get("text") for d in pp.right_click_wire(w) or []])
    for a in pp.post_arrows():
        print(" arrow tip", a.toolTip())
    ex.click_block(pp.final())
    print(" post panel", pp.panel_texts())
    vr = ex.view.mapToScene(ex.view.viewport().rect()).boundingRect()
    print(" post visible", vr.contains(pp.final().sceneBoundingRect()), "scale", round(ex.view.transform().m11(), 2))
    shot("c_texts")


SC = {k: v for k, v in globals().items() if k.startswith("c_")}
for k in args or list(SC):
    for r in (1, 2):
        RUN[0] = r
        print(f"=== {k} {lang} {size} run {r}", flush=True)
        try:
            SC[k]()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("FAILED", k, e)
    print("slot errors", [t.strip().splitlines()[-1] for t in pp.take_slot_errors()], flush=True)
ex.close()
