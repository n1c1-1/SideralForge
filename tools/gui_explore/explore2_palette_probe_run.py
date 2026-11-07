"""palette_matrix round 2: the run window of a workflow with blocks placed twice (rows: their names), with and without
a review pause on a copy.   PM_LANG=fr|en python explore2_palette_probe_run.py"""
from __future__ import annotations

import time

from explore2_palette_lib import LANG, Matrix

from PySide6.QtWidgets import QAbstractButton, QLabel

m = Matrix(f"probe_run_{LANG}")
ex = m.ex


def blk(prefix, i):
    xs = sorted([it for it in ex.blocks() if it.node.key.split("~")[0] == prefix], key=lambda i: i.sceneBoundingRect().x())
    return xs[i] if len(xs) > abs(i) else None


def open_run(tag):
    b = ex.w.bt_run
    ex.click(b)
    for _ in range(10):
        ex.settle(5)
        time.sleep(0.05)
    d = ex.w.run.dialog
    rows = [(r.name.text(), r.toolTip()[:150].replace("\n", " / ")) for r in d.rows] if d else None
    txt = [x.text() for x in d.findChildren(QLabel) + d.findChildren(QAbstractButton) if x.isVisible() and x.text()] \
        if d else None
    print(tag, "| run button", b.text(), "| rows", rows, "| texts", txt, flush=True)
    f = ex.shot(f"probe_run_{LANG}_{tag}", d) if d else None
    if d:
        d.close()
        ex.settle(4)
    return rows, txt, f


for attempt in (1, 2):
    ex.open(ex.project("image"))
    ex.settle(8)
    ex.click_block(ex.block("image"))
    for b in ("astrometry", "gradient", "gradient", "color", "color"):
        ex.double_click_card(b)
        ex.settle(6)
    print("graph", [(it.node.key, it.title) for it in sorted(ex.blocks(), key=lambda i: i.sceneBoundingRect().x())])
    open_run(f"a{attempt}_sans_pause")
    for prefix, i in (("fond", 0), ("fond", 1), ("couleur", 1)):
        it = blk(prefix, i)
        ex.right_click_block(it, "Pause")
        print("pause on", blk(prefix, i).title, "toast", m.toasts[-1:])
        open_run(f"a{attempt}_pause_{prefix}{i + 1}")
        try:
            pl, pause = ex.s.plan_run()
            print("   plan", len(pl.jobs), "pause", pause)
        except Exception as e:  # noqa: BLE001
            print("   PLAN RAISES", type(e).__name__, e)
        ex.right_click_block(blk(prefix, i), "Pause")      # off again
ex.close()
