"""Round 2 fuzz: a slider of the Recombine panel dragged with the mouse from its handle: how many Undo entries, and
where one Ctrl+Z goes (seed 4036), twice."""
import json
import os
import sys
import traceback
from pathlib import Path

sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider  # noqa: E402

fz = G.Fuzz2("fr")
fz.panels = False
ex = fz.ex
sys.excepthook = lambda *a: traceback.print_exception(*a)


def sliders():
    return json.dumps([lay.get("sliders") for lay in ex.s.post_layers()])


for k in range(2):
    fz.start({"template": "sho", "lang": "fr"}, f"sl{k}")
    fz.activate()
    ex.click_block(next(x for x in fz.items() if x.node.key == "composition"))
    sl = [w for w in ex.page.panel.findChildren(QSlider) if w.isVisible()][0]
    n0, s0 = len(ex.s._undo), sliders()
    opt = QStyleOptionSlider()
    sl.initStyleOption(opt)
    a = sl.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, sl).center()
    QTest.mousePress(sl, Qt.LeftButton, Qt.NoModifier, a)
    for dx in (10, 25, 40):
        QTest.mouseMove(sl, QPoint(a.x() + dx, a.y()))
        ex.settle(1)
    QTest.mouseRelease(sl, Qt.LeftButton, Qt.NoModifier, QPoint(a.x() + 40, a.y()))
    fz.wait_ready()
    n1, s1 = len(ex.s._undo), sliders()
    fz.undo()
    fz.wait_ready()
    s2 = sliders()
    print(k, "drag: +", n1 - n0, "entries | before", s0, "| after", s1, "| after one Ctrl+Z", s2, flush=True)
os._exit(0)
