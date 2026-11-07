"""Probe: does Ctrl+Y reach the window when no widget has the focus (harness check, block_actions area)."""
import faulthandler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(200, exit=True)
from harness import Explorer  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

ex = Explorer("_probe_blocks")
g = lambda: [n["block"] + ("/off" if n["off"] else "") for n in ex.s.graph(ex.s.paths[0])]  # noqa: E731
ex.open(ex.project("session"))
for b in ["calibration", "registration", "stacking", "gradient"]:
    ex.double_click_card(b)
ex.w.activateWindow()
ex.settle(4)
ex.click_block(ex.block("alignement"))
QTest.keyClick(ex.view.viewport(), Qt.Key_D)
ex.settle(4)
print("after D", g())
ex.undo()
print("after undo", g(), "focus", QApplication.focusWidget())
f = QApplication.focusWidget()
if f is not None:
    f.clearFocus()
ex.settle(2)
print("focus now", QApplication.focusWidget())
QTest.keyClick(ex.w, Qt.Key_Y, Qt.ControlModifier)
ex.settle(4)
print("after Ctrl+Y with no focus", g())
ex.close()
