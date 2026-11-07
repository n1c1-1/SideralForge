"""Probe 4: why Del did not reach the wire in explore_post_wires (focus, active window)."""
from explore_post_common import P, Explorer, build_chain, session_project, start, CHAINS
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

start(200)
ex = Explorer("post_wires_probe")
pp = P(ex)
session_project(ex)
build_chain(ex, CHAINS["stacking"])
print("active:", QApplication.activeWindow(), "focus:", QApplication.focusWidget())
w = pp.wires()[0]
pp.click_wire(w)
print("after click: active:", QApplication.activeWindow(), "focus:", QApplication.focusWidget(), "sel", ex.view.sel_wire is w)
ex.key(Qt.Key_Delete)
print("after Del wires:", len(pp.wires()), "focus", QApplication.focusWidget())
ex.w.activateWindow(); ex.settle()
print("active now:", QApplication.activeWindow())
w = pp.wires()[0]
pp.click_wire(w)
ex.key(Qt.Key_Delete)
print("after Del2 wires:", len(pp.wires()))
ex.close()
