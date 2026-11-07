"""post_wires round 2: the Post page's own undo (Channels, blend mode), a layer deleted in Post, a wire removed then
the project saved and reopened."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox

start(900)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["undo", "reload"]


def R(n, *a):
    print(f"R{n}:", *a, flush=True)


def paths():
    return [(q.name, [n['block'] for n in (ex.s.p.workflow(q.pid).blocks or [])]) for q in ex.s.paths]


def st(tag=""):
    return (f"[{tag}] post={pp.final_summary()!r} layers={pp.graph_layers()} "
            f"wires={[w.toolTip().splitlines()[0] for w in pp.wires()]}")


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


def combos(v):
    return [c for c in v.right.findChildren(QComboBox) if c.isVisible()]


if "undo" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    v = ex.w.post_holder.page.view
    R(0, "rows", [r.n.text() for r in v.rows])
    lay = next(lay for lay in v.layers if lay["kind"] == "workflow")
    v.select(lay["id"], "layer")
    ex.settle(4)
    cs = combos(v)
    R(0, "combos", [[c.itemText(i) for i in range(c.count())] for c in cs])
    ch = next(c for c in cs if c.count() == 2)
    bl = next(c for c in cs if c.count() > 2)
    for name, c in (("canaux", ch), ("fusion", bl)):
        before = c.currentIndex()
        nu = len(v.undo)
        # as the user: a click on the combo, then a pick in its list (activated)
        c.setFocus()
        ex.settle(1)
        QT = __import__("PySide6.QtTest", fromlist=["QTest"]).QTest
        QT.keyClick(c, Qt.Key_Down)
        ex.settle(8)
        R(17, name, "changed", before, "->", c.currentIndex(), "undo stack", nu, "->", len(v.undo),
          "b_undo enabled", v.b_undo.isEnabled())
        pp.undo()
        ex.settle(8)
        c2 = [x for x in combos(v) if x.count() == c.count()]
        R(17, name, "after Ctrl+Z", [x.currentText() for x in c2], "undo stack", len(v.undo))
        pp.redo()
        ex.settle(8)
        c2 = [x for x in combos(v) if x.count() == c.count()]
        R(17, name, "after Ctrl+Y", [x.currentText() for x in c2])
        pp.undo()
        ex.settle(8)
    ex.shot("post_undo_channels")
    # delete the "Sortie Gradient" layer in Post
    tgt = next((lay for lay in v.layers if "Gradient" in lay["name"]), None)
    if tgt is not None:
        v.select(tgt["id"], "layer")
        ex.settle(3)
        ex.click(v.b_del)
        ex.settle(10)
        R(50, "post layers after delete", [lay["name"] for lay in v.layers], "rows", [r.n.text() for r in v.rows])
        ex.click(ex.w.mode_btns[0])
        ex.settle(10)
        R(50, st("graph after layer deleted in Post"), paths())
        ex.shot("post_layer_deleted_graph")
        ex.click(ex.w.mode_btns[2])
        ex.settle(10)
        R(50, "post layers after round trip", [lay["name"] for lay in v.layers])
        pp.undo()
        ex.settle(8)
        R(50, "post layers after Ctrl+Z in Post", [lay["name"] for lay in v.layers])
        ex.click(ex.w.mode_btns[0])
        ex.settle(10)
        R(50, st("graph after Ctrl+Z in Post"), paths())

if "reload" in which:
    root = session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    pp.pull_port(pp.block("empilement", pid=ex.s.paths[0].pid))
    R(60, st("2 sends"), paths())
    # remove the main wire
    w = next(w for w in pp.wires() if "« Workflow »" in w.toolTip())
    pp.click_wire(w)
    pp.del_key()
    R(60, st("main wire removed"), pp.toast())
    pp.key(Qt.Key_S, ex.w, Qt.ControlModifier)
    ex.settle(8)
    R(60, "saved; popups", ex.popups[-2:], "dirty", getattr(ex.s, "dirty", None))
    ex.w.open_project(root)
    ex.settle(12)
    R(60, st("reopened"), paths())
    ex.shot("post_reload_wire_removed")
    # undo after reload
    R(60, "can undo after reopen", ex.s.can_undo())
    ex.right_click_block(pp.block("couleur"), "Vers Post")
    ex.settle(6)
    R(60, st("main sent back after reload"), paths())
print("popups", ex.popups, "modals", ex.modals, "slot", pp.take_slot_errors())
ex.close()
