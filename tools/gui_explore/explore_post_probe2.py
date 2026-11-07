"""Probe 2: the wire to Post of a chain ending at Stacking: hover, click, Del, menu, undo, send back."""
from explore_post_common import P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt

start(300)
ex = Explorer("post_wires_probe")
pp = P(ex)
session_project(ex)
build_chain(ex, ["calibration", "registration", "stacking"])
pp.dump("chain to stacking")
ws = pp.wires()
w = ws[0]
print("wire tip:", w.toolTip(), "pt:", pp.free_point_on(w))
pen0 = w.pen().color().name(), w.pen().widthF()
ex.hover(ex.view.viewport(), pp.free_point_on(w))
print("hover pen:", w.pen().color().name(), w.pen().widthF(), "before", pen0)
pp.click_wire(w)
print("selected:", w.selected, "sel_wire:", ex.view.sel_wire is w, "sel_step", ex.s.sel_step)
ex.shot("probe2_selected")
pp.del_key()
pp.dump("after Del")
print("toast:", pp.toast())
ex.shot("probe2_after_del")
ex.undo()
pp.dump("after undo")
print("toast:", pp.toast())
ex.redo()
pp.dump("after redo")
# menu of the last block
it = pp.block("empilement")
m = ex.right_click_block(it)
print("menu empilement:", [d.get("text") for d in m])
ex.right_click_block(it, "Vers Post")
pp.dump("after send back")
# the wire's own menu
w = pp.wires()[0] if pp.wires() else None
if w:
    m = pp.right_click_wire(w)
    print("wire menu:", m)
    m = pp.right_click_wire(w, "Retirer")
    pp.dump("after wire menu remove")
# panel of Post block
ex.click_block(pp.final())
print("panel:", pp.panel_texts())
print("buttons:", pp.panel_buttons())
ex.close()
