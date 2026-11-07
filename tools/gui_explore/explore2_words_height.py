"""Layout, round 2: does the window grow taller than a 1366×768 screen? The window's height after each step (a Fusion
placed, the Log opened, Inspect), and the widgets whose minimum height is the largest.

    python explore2_words_height.py fr
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


def tallest(root, n=6):
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


ex.open(ex.project("mono"))
report("mono opened")
ex.click_block(ex.block("session", sid=1) or ex.blocks()[0])
ex.double_click_card("fusion")
h1 = report("fusion placed")
it = ex.block("fusion")
if it is not None:
    ex.click_block(it)
h2 = report("fusion selected")
print("   panel tallest:", tallest(ex.page.panel), flush=True)
ex.w.b_log.setChecked(True)
h3 = report("log open")
print("   tallest:", tallest(ex.w)[:4], flush=True)
ex.shot(f"height_{LANG}_fusion_log")
ex.w.b_log.setChecked(False)
report("log closed")
ex.w.resize(1366, 768)
report("resized to 1366×768")
# the same with Stacking selected and the Log open
ex.click_block(ex.block("empilement") or ex.blocks()[0])
ex.w.b_log.setChecked(True)
report("stacking + log")
ex.shot(f"height_{LANG}_stacking_log")
ex.w.b_log.setChecked(False)
# a full chain, a second flow, a Session selected, then the Log
ex.open(ex.project("two_sessions", f"Haut {LANG}"))
ex.settle(6)
ex.w.resize(1366, 768)
ex.click_block(ex.block("session", sid=1))
for k in ("calibration", "registration", "selection", "stacking", "astrometry", "gradient", "color", "denoise",
          "stars", "composition"):
    ex.double_click_card(k)
    ex.settle(3)
report("chain built")
for key in ("session", "calibration", "composition", "etoiles", "empilement"):
    it = ex.block(key, sid=2) if key == "session" else ex.block(key)
    if it is None:
        continue
    ex.click_block(it)
    ex.w.b_log.setChecked(True)
    h = report(f"{key} selected + log")
    if h > 768:
        print("   tallest:", tallest(ex.w)[:5], flush=True)
        print("   panel tallest:", tallest(ex.page.panel)[:4], flush=True)
        ex.shot(f"height_{LANG}_{key}_log")
    ex.w.b_log.setChecked(False)
    ex.settle(3)
# Inspect a block first (its page stays in the window), back to the Graph, then the Log
for key in ("session", "post", "fond", "composition"):
    ex.w.set_mode(0)
    ex.w.resize(1366, 768)
    ex.settle(3)
    it = ex.block(key)
    if it is None:
        continue
    ex.double_click_block(it)
    ex.settle(6)
    report(f"Inspect {key}")
    ex.w.b_log.setChecked(True)
    h = report(f"Inspect {key} + log")
    if h > 768:
        print("   tallest:", tallest(ex.w)[:6], flush=True)
        ex.shot(f"height_{LANG}_inspect_{key}_log")
    ex.w.set_mode(0)
    h = report(f"back on Graph, log open (after {key})")
    if h > 768:
        ex.shot(f"height_{LANG}_graph_after_{key}_log")
    ex.w.b_log.setChecked(False)
    report("log closed")
ex.close()
