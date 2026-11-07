"""Round 4 (agent I, item 15): with the Log open, opening Post (double-click on Recombine) must not make a 1366×768
window taller. Prints the window's height and the tallest widgets.

    python explore4_height.py fr
"""
from __future__ import annotations

import faulthandler
import sys

faulthandler.dump_traceback_later(300, exit=True)
from harness import Explorer, stray_qt_errors  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
stray_qt_errors()
ex = Explorer("words_layout", lang=LANG, size=(1366, 768))


def tallest(root, n=8):
    rows = []
    for x in root.findChildren(QWidget):
        if x.isVisible():
            rows.append((max(x.minimumSizeHint().height(), x.minimumHeight()), type(x).__name__, x.objectName(),
                         (x.text()[:30] if hasattr(x, "text") and callable(x.text) and isinstance(x.text(), str) else "")))
    rows.sort(key=lambda r: -r[0])
    return rows[:n]


def report(tag):
    ex.settle(6)
    print(f"{tag}: window {ex.w.width()}×{ex.w.height()}, minimum {ex.w.minimumSizeHint().width()}×"
          f"{ex.w.minimumSizeHint().height()}", flush=True)
    return ex.w.height()


ex.open(ex.project("two_sessions", f"Haut {LANG}"))
ex.settle(6)
ex.w.resize(1366, 768)
ex.click_block(ex.block("session", sid=1))
for k in ("calibration", "registration", "selection", "stacking", "astrometry", "gradient", "color", "denoise",
          "stars", "composition"):
    ex.double_click_card(k)
    ex.settle(3)
report("chain built")
ex.w.b_log.setChecked(True)
report("log open")
it = ex.block("composition")
ex.double_click_block(it)
h = report("Post opened from Recombine, log open")
if h > 768:
    print("   tallest:", tallest(ex.w), flush=True)
ex.shot(f"explore4_height_{LANG}_post_log")
ex.w.set_mode(0)
h = report("back on Graph")
ex.close()
