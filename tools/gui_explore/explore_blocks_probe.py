"""Probe: when is Ctrl+Y lost after Ctrl+Z (block_actions area)."""
import faulthandler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(250, exit=True)
from explore_blocks_lib import TOASTS  # noqa: E402
from harness import Explorer  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

ex = Explorer("_probe_blocks")
g = lambda: [n["block"] + ("/off" if n["off"] else "") for n in ex.s.graph(ex.s.paths[0])]  # noqa: E731


def setup(chain):
    ex.open(ex.project("session"))
    ex.w.activateWindow()
    for b in chain:
        ex.double_click_card(b)
    ex.s._undo.clear()
    ex.s._redo.clear()


for chain in (["calibration", "registration", "stacking", "gradient"],) or (["calibration", "registration", "stacking", "gradient"],
              ["calibration", "registration", "stacking", "astrometry", "gradient", "denoise", "color", "stars"]):
    for pre in ("empilement", "calibration", "fond"):
        setup(chain)
        ex.click_block(ex.block(pre))
        z = g()
        m = ex.right_click_block(ex.block("fond"), pick="Désactiver")
        ex.settle(6)
        ex.w.activateWindow()
        ex.settle(4)
        from PySide6.QtWidgets import QApplication
        print("active", type(QApplication.activeWindow()).__name__, "focus", type(QApplication.focusWidget()).__name__,
              "undo stack", len(ex.s._undo), "a_undo enabled", ex.w.a_undo.isEnabled())
        a = g()
        ex.undo()
        b = g()
        r = len(ex.s._redo)
        from PySide6.QtWidgets import QApplication as QA
        print("  before redo: focus", type(QA.focusWidget()).__name__, QA.focusWidget().objectName() if QA.focusWidget() else "", "redo", len(ex.s._redo), "undo", len(ex.s._undo), g())
        ex.redo()
        print("  after redo: redo", len(ex.s._redo), "undo", len(ex.s._undo), g())
        if pre == "empilement":
            ex.w.a_redo.trigger(); ex.settle(4)
            print("  after a_redo.trigger:", g())
        c = g()
        print([x.get("text") for x in m], z == a)
        print(len(chain), "pre", pre, "| after", a == b, "| redo stack", r, "| redo ok", c == a, "| sel", ex.s.sel_step)
    # with the D key while Empilement is selected
    setup(chain)
    ex.click_block(ex.block("empilement"))
    n = len(TOASTS)
    ex.key(Qt.Key_D)
    print("D on Empilement:", TOASTS[n:], g())
    ex.key(Qt.Key_Delete)
    print("Del on Empilement:", TOASTS[n:], g())
ex.close()
