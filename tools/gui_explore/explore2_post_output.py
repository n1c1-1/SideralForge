"""post_wires round 2: the Output block of a branch and the Post block under the user's gestures: its cross, dragged to the
palette, Inspection, Pause vérif, Del on Post; a stretched Image source with Split."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton

start(600)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["output", "post", "stretched"]


def st(tag=""):
    return (f"[{tag}] post={pp.final_summary()!r} paths={[q.name for q in ex.s.paths]} wires={len(pp.wires())} "
            f"outs={[(b.title, b.summary) for b in pp.blocks_of('post')]} toast={pp.toast()!r}")


def fresh():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)


if "output" in which:
    fresh()
    o = pp.blocks_of("post")[0]
    ex.hover(ex.view.viewport(), ex.point_of(o))
    print("output cross?", getattr(o, "cross", None), "hover tip", o.toolTip())
    ex.shot("output_hover")
    ex.click(ex.view.viewport(), ex.point_of(o, "cross"))
    print(st("cross clicked"))
    if len(ex.s.paths) == 1:
        pp.undo()
    fresh()
    ex.drag_block_to_palette(pp.blocks_of("post")[0])
    print(st("output dragged to palette"))
    ex.shot("output_to_palette")
    fresh()
    ex.right_click_block(pp.blocks_of("post")[0], "Inspection")
    ex.settle(8)
    print("Inspection on output: mode", ex.w.mode, "sel", ex.s.sel_step, [t for t in ex.texts() if t][:12])
    ex.shot("output_inspection")
    ex.w.set_mode(0)
    ex.settle(4)
    ex.right_click_block(pp.blocks_of("post")[0], "Pause vérif")
    ex.settle(6)
    print(st("pause on output"), [(b.title, b.summary) for b in pp.blocks_of("post")])
    m = ex.right_click_block(pp.blocks_of("post")[0])
    print("menu after pause", [(d.get("text"), d.get("checked")) for d in m])
    ex.shot("output_pause")

if "post" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.click_block(pp.final())
    pp.del_key()
    print(st("Del on Post"))
    ex.hover(ex.view.viewport(), ex.point_of(pp.final()))
    ex.click(ex.view.viewport(), ex.point_of(pp.final(), "cross"))
    print(st("cross point on Post"))
    ex.drag_block_to_palette(pp.final())
    print(st("Post dragged to palette"))
    pp.key(Qt.Key_D)
    print(st("D on Post"))

if "stretched" in which:
    root = ex.project("image")
    ex.open(root)
    ex.w.activateWindow()
    ex.click_block(pp.block("image"))
    b = [x for x in ex.page.panel.findChildren(QAbstractButton) if x.isVisible() and x.text() in ("Étirée", "Stretched")]
    if b:
        ex.click(b[0])
        ex.settle(6)
    print(st("stretched"), pp.graph_layers())
    print("cards", {k: c.isEnabled() for k, c in ex.page.cards.items()})
    ex.double_click_card("stars")
    ex.settle(6)
    print(st("stars on stretched"), pp.graph_layers(), [(b.title, b.summary) for b in pp.blocks_of("composition")])
    sp = pp.block("etoiles")
    if sp is not None:
        m = ex.right_click_block(sp)
        print("split menu", [d.get("text") for d in m])
        ex.right_click_block(pp.block("etoiles"), "Vers Post")
        ex.settle(6)
        print(st("split's own Vers Post"), pp.graph_layers())
        ex.shot("stretched_split_send")
        pp.undo()
        print(st("undo"), pp.graph_layers())
    m = ex.right_click_block(pp.block("image"))
    print("image menu", [d.get("text") for d in m])
ex.close()
