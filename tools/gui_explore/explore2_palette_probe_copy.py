"""palette_matrix round 2: a block placed twice: Run to here on each copy, switch off a copy (D), drag a copy back onto
the palette, change a setting of one copy (the other keeps its own), each undone and redone.
    PM_LANG=fr|en python explore2_palette_probe_copy.py"""
from __future__ import annotations

import time

from explore2_palette_lib import LANG, Matrix

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QComboBox, QLabel

m = Matrix(f"probe_copy_{LANG}")
ex = m.ex


def g(i):
    xs = sorted([it for it in ex.blocks() if it.node.key.split("~")[0] == "fond"], key=lambda i: i.sceneBoundingRect().x())
    return xs[i] if len(xs) > i else None


def names():
    return [(it.node.key, it.title, getattr(it.node, "cut", None)) for it in
            sorted([it for it in ex.blocks() if it.node.key.split("~")[0] == "fond"], key=lambda i: i.sceneBoundingRect().x())]


for attempt in (1, 2):
    ex.open(ex.project("image"))
    ex.settle(8)
    ex.click_block(ex.block("image"))
    for b in ("astrometry", "gradient", "gradient"):
        ex.double_click_card(b)
        ex.settle(6)
    print("graph", names())
    # Run to here on each copy (Shift+F5)
    for i in (0, 1):
        ex.click_block(g(i))
        ex.key(Qt.Key_F5, ex.w, Qt.ShiftModifier)
        for _ in range(6):
            ex.settle(5)
            time.sleep(0.05)
        d = ex.w.run.dialog
        rows = [r.name.text() for r in d.rows] if d else None
        print(f"run to here on {g(i).title}:", d.windowTitle() if d else None, rows,
              [x.text() for x in d.findChildren(QLabel) if x.isVisible() and x.text()][:6] if d else None)
        if d is not None:
            if attempt == 2:
                ex.shot(f"copy_{LANG}_runhere_{i + 1}", d)
            d.close()
            ex.settle(4)
    # switch off the second copy (D)
    s0 = m.state()
    ex.click_block(g(1))
    ex.key(Qt.Key_D)
    print("after D on copy 2", names(), "toast", m.toasts[-1:])
    s1 = m.state()
    ex.undo()
    print("undo exact", m.state() == s0, names())
    ex.redo()
    print("redo exact", m.state() == s1, names())
    ex.undo()
    # a setting of copy 2 changed in its panel: copy 1 keeps its own
    ex.click_block(g(1))
    combos = [c for c in ex.page.panel.findChildren(QComboBox) if c.isVisible() and c.count() > 1]
    p0 = dict(ex.s.p.workflow(ex.s.paths[0].pid).params)
    if combos:
        c = combos[0]
        c.setCurrentIndex((c.currentIndex() + 1) % c.count())
        c.activated.emit(c.currentIndex())
        ex.settle(8)
    p1 = dict(ex.s.p.workflow(ex.s.paths[0].pid).params)
    print("params before", p0, "after a change on copy 2", p1, "summaries", [(it.title, it.summary) for it in
                                                                             [g(0), g(1)] if it])
    if attempt == 2:
        ex.shot(f"copy_{LANG}_setting")
    ex.undo()
    print("undo setting exact", dict(ex.s.p.workflow(ex.s.paths[0].pid).params) == p0)
    # the first copy dragged back onto the palette: removed, the other renamed "Gradient"
    s0 = m.state()
    ex.drag_block_to_palette(g(0))
    ex.settle(6)
    print("after drag copy 1 to palette", names(), "toast", m.toasts[-1:])
    ex.undo()
    print("undo exact", m.state() == s0, names())
    print("exceptions", [x.splitlines()[0] for x in m.exc])
ex.close()
