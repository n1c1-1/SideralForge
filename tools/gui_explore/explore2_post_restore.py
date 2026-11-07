"""post_wires round 2: a layer's Post settings after its wire is removed and Ctrl+Z (exactly what was there?), and after
remove + "To Post" again; Split's wire and a deleted Recombine after Save and reopen."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox

start(600)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["settings", "reload"]


def st(tag=""):
    return (f"[{tag}] post={pp.final_summary()!r} layers={pp.graph_layers()} comps={len(pp.blocks_of('composition'))} "
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


def post_layers():
    v = ex.w.post_holder.page.view
    return [(lay["name"], lay.get("mode"), lay.get("opacity")) for lay in v.layers]


if "settings" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    v = ex.w.post_holder.page.view
    for name in ("Sortie Gradient", "Workflow"):
        lay = next(lay for lay in v.layers if lay["name"] == name)
        v.select(lay["id"], "layer")
        ex.settle(4)
        c = next(c for c in v.right.findChildren(QComboBox) if c.isVisible() and c.count() > 2)
        c.setCurrentIndex(2)                       # Écran
        ex.settle(6)
    print("post layers set:", post_layers())
    ex.click(ex.w.mode_btns[0])
    ex.settle(8)
    # remove the main wire, Ctrl+Z
    w = next(w for w in pp.wires() if "« Workflow »" in w.toolTip())
    pp.click_wire(w)
    pp.del_key()
    print(st("main wire removed"))
    pp.undo()
    print(st("undo"))
    ex.click(ex.w.mode_btns[2])
    ex.settle(10)
    print("post layers after remove + Ctrl+Z:", post_layers())
    ex.click(ex.w.mode_btns[0])
    ex.settle(8)
    # remove then send again from the menu
    w = next(w for w in pp.wires() if "« Workflow »" in w.toolTip())
    pp.click_wire(w)
    pp.del_key()
    ex.right_click_block(pp.block("couleur"), "Vers Post")
    ex.settle(6)
    print(st("main removed then sent back"))
    ex.click(ex.w.mode_btns[2])
    ex.settle(10)
    print("post layers after remove + Vers Post:", post_layers())
    ex.shot("restore_layers_after_resend")
    ex.click(ex.w.mode_btns[0])
    ex.settle(8)
    # branch wire removed (branch deleted) then Ctrl+Z: its layer settings?
    w = next(w for w in pp.wires() if "Sortie Gradient" in w.toolTip())
    pp.click_wire(w)
    pp.del_key()
    print(st("branch wire removed"))
    pp.undo()
    print(st("undo"))
    ex.click(ex.w.mode_btns[2])
    ex.settle(10)
    print("post layers after branch removed + Ctrl+Z:", post_layers())

if "reload" in which:
    root = session_project(ex)
    build_chain(ex, CHAINS["stars"])
    pp.pull_port(pp.block("etoiles"), "starless")
    pp.pull_port(pp.block("etoiles"), "stars")
    print(st("both split images sent"))
    w = next(w for w in pp.wires() if "toiles seules" in w.toolTip())
    pp.click_wire(w)
    pp.del_key()
    ex.click_block(pp.blocks_of("composition")[0])
    pp.del_key()
    print(st("stars wire and Recombine removed"))
    pp.key(Qt.Key_S, ex.w, Qt.ControlModifier)
    ex.settle(8)
    ex.w.open_project(root)
    ex.settle(12)
    print(st("reopened"))
    ex.shot("restore_split_reopened")
print("popups", ex.popups, "slot", pp.take_slot_errors())
ex.close()
