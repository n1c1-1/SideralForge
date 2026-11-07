"""Probe 3: does Del reach the graph's shortcut in the harness? (block Del, then wire Del), then undo of a wire removal."""
from explore_post_common import P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

start(300)
ex = Explorer("post_wires_probe")
pp = P(ex)
session_project(ex)
build_chain(ex, ["calibration", "registration", "selection", "stacking"])
ex.w.activateWindow()
ex.settle()
print("active:", QApplication.activeWindow(), "focus:", QApplication.focusWidget())
it = pp.block("selection")
ex.click_block(it)
print("sel_step", ex.s.sel_step)
ex.key(Qt.Key_Delete)
pp.dump("after Del on Selection block")
w = pp.wires()[0]
pp.click_wire(w)
print("sel_wire", ex.view.sel_wire is w, "focus:", QApplication.focusWidget())
ex.key(Qt.Key_Delete)
pp.dump("after Del on wire")
print("toast:", pp.toast())
# variant: key to the view itself
w = pp.wires()[0] if pp.wires() else None
if w:
    pp.click_wire(w)
    ex.view.setFocus()
    QTest.keyClick(ex.view, Qt.Key_Delete)
    ex.settle()
    pp.dump("after Del to view")
# undo
ex.undo()
pp.dump("after ctrl+z")
print("toast:", pp.toast())
ex.undo()
pp.dump("after ctrl+z 2")
ex.close()
