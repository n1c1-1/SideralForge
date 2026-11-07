"""Layout, reproduction: the window at 1366×768 grows wider than the screen in Inspect. Which screen, which widget.

    python explore_words_width.py fr
"""
from __future__ import annotations

import faulthandler
import sys

faulthandler.dump_traceback_later(600, exit=True)
from harness import Explorer  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
ex = Explorer("words_layout", lang=LANG, size=(1366, 768))


def widest(root: QWidget, n=12):
    """The widgets whose minimum width is the largest, deepest first (the one that pushes)."""
    rows = []
    for x in root.findChildren(QWidget):
        if x.isVisible():
            m = max(x.minimumSizeHint().width(), x.minimumWidth())
            rows.append((m, type(x).__name__, x.objectName(), (getattr(x, "text", lambda: "")() if callable(
                getattr(x, "text", None)) else "")[:50]))
    rows.sort(reverse=True)
    return rows[:n]


for kind, blocks in (("legacy", ["etoiles", "composition", "fond", "post"]),):
    root = ex.project(kind)
    ex.open(root)
    ex.settle(8)
    print("graph:", ex.w.width(), ex.w.minimumSizeHint().width())
    for key in blocks:
        ex.w.resize(1366, 768)
        ex.settle(4)
        it = ex.block(key)
        if it is None:
            print("no block", key)
            continue
        ex.double_click_block(it)
        ex.settle(8)
        print(f"== Inspect {key}: window {ex.w.width()}×{ex.w.height()}, minimum {ex.w.minimumSizeHint().width()}")
        for r in widest(ex.w.tune):
            print("   ", r)
        ex.shot(f"width_{LANG}_inspect_{key}")
        ex.w.set_mode(0)
        ex.settle(4)
        print(f"   back on the graph: window {ex.w.width()}, minimum {ex.w.minimumSizeHint().width()}")
        ex.w.resize(1366, 768)
        ex.settle(4)
        print(f"   after resize to 1366: {ex.w.width()}")
ex.close()
