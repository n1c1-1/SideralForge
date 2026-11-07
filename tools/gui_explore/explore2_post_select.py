"""post_wires round 2: wire selection then another click (empty canvas, a block, Escape) then Del: what goes? A branch
renamed to the name of the main workflow."""
from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QLineEdit

start(600)
ex = Explorer("post_wires")
pp = P(ex)


def st(tag=""):
    return (f"[{tag}] post={pp.final_summary()!r} paths={[(q.name, len(ex.s.p.workflow(q.pid).blocks)) for q in ex.s.paths]} "
            f"wires={len(pp.wires())} toast={pp.toast()!r}")


for how in ("empty", "escape", "block"):
    session_project(ex)
    build_chain(ex, CHAINS["gradient"])
    w = pp.wires()[0]
    pp.click_wire(w)
    print(how, "wire selected", w.selected)
    vp = ex.view.viewport()
    if how == "empty":
        ex.click(vp, QPoint(vp.width() // 2, 40))
    elif how == "escape":
        pp.key(Qt.Key_Escape)
    else:
        ex.click_block(pp.block("alignement"))
    print(how, "after: wire selected", pp.wires()[0].selected if pp.wires() else None, "sel_step", ex.s.sel_step)
    pp.del_key()
    print(how, st("Del"))
    ex.shot(f"select_{how}")

# rename a branch to the main workflow's name
session_project(ex)
build_chain(ex, CHAINS["color"])
ex.right_click_block(pp.block("fond"), "Vers Post")
ex.settle(6)


def h(d):
    e = d.findChildren(QLineEdit)
    e[0].setText("Workflow")
    d.accept()


ex.dialog_handler = h
ex.right_click_block(pp.blocks_of("post")[0], "Renommer")
ex.settle(8)
print(st("branch renamed Workflow"), pp.graph_layers(), ex.popups[-1:])
ex.click_block(pp.final())
print("post panel", pp.panel_texts()[2:3])
ex.shot("rename_same_name")
ex.close()
