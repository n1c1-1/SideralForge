"""Probe: where the keyboard focus goes when Empilement is clicked (block_actions area)."""
import faulthandler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(200, exit=True)
from explore_blocks_lib import TOASTS  # noqa: E402
from harness import Explorer  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

ex = Explorer("_probe_blocks")
g = lambda: [n["block"] + ("/off" if n["off"] else "") for n in ex.s.graph(ex.s.paths[0])]  # noqa: E731
ex.open(ex.project("session"))
ex.w.activateWindow()
for b in ["calibration", "registration", "stacking", "gradient"]:
    ex.double_click_card(b)


def fw():
    f = QApplication.focusWidget()
    return f"{type(f).__name__}:{f.objectName() if f else ''}"


for key in ("alignement", "empilement", "fond", "alignement", "empilement"):
    ex.w.activateWindow()
    ex.settle(4)
    print("active", type(QApplication.activeWindow()).__name__, fw())
    ex.click_block(ex.block(key))
    ex.settle(6)
    print(key, "focus after click:", fw(), "active:", QApplication.activeWindow(), "popup:", QApplication.activePopupWidget(),
          "modal:", QApplication.activeModalWidget(), "tops:", [type(w).__name__ for w in QApplication.topLevelWidgets() if w.isVisible()])
    n = len(TOASTS)
    vp = ex.view.viewport()
    QTest.keyClick(vp, Qt.Key_D)          # no setFocus: as the user's next key
    ex.settle(4)
    print("   D (no setFocus):", TOASTS[n:], g(), "focus", fw())
    if g()[-1].endswith("/off") or any(x.endswith("/off") for x in g()):
        ex.undo()
ex.close()
