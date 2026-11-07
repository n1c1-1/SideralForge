"""post_wires round 3 (control), part D: Canaux liés and Ctrl+Z in Post; the message after Image placed in a blank
project. Twice each. Prints only."""
import time

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton

start(900)
ex = Explorer("post_wires")
pp = P(ex)


def simulate():
    s = ex.s
    orig = s._sync

    def fake(*a, **k):
        orig(*a, **k)
        for v in s.paths:
            v.rendered = True
            v.computed = False
    s._sync = fake
    s._sync()


def linked(v):
    return next((x for x in v.right.findChildren(QAbstractButton) if x.isVisible() and x.text() == "Canaux liés"), None)


for run in (1, 2):
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    v = ex.w.post_holder.page.view
    lay = next(lay for lay in v.layers if lay["kind"] == "workflow")
    v.select(lay["id"], "layer")
    ex.settle(4)
    cb = linked(v)
    print(run, "box", type(cb).__name__, cb.isEnabled(), cb.isChecked(), "undo", len(v.undo))
    QTest.mouseClick(cb, Qt.LeftButton, Qt.NoModifier, cb.rect().adjusted(2, 2, -2, -2).topLeft() + cb.rect().center() * 0)
    ex.settle(10)
    cb = linked(v)
    print(run, "after click on its box", cb.isChecked(), "undo", len(v.undo), "layer", {k: lay.get(k) for k in lay if "link" in k or "chan" in k})
    ex.shot(f"d_linked_clicked_run{run}")
    pp.undo()
    ex.settle(10)
    cb = linked(v)
    print(run, "after Ctrl+Z", cb.isChecked(), "undo", len(v.undo))
    ex.shot(f"d_linked_undo_run{run}")
    ex.click(ex.w.mode_btns[0])
    ex.settle(6)
    # Image twice in a blank project: the message after each
    ex.new_project(f"Img{run}{time.time():.0f}")
    ex.w.activateWindow()
    print(run, "toast before", repr(pp.toast()))
    ex.double_click_card("image")
    ex.settle(8)
    print(run, "toast after 1st Image", repr(pp.toast()), [q.name for q in ex.s.paths])
    ex.double_click_card("image")
    ex.settle(8)
    print(run, "toast after 2nd Image", repr(pp.toast()), [q.name for q in ex.s.paths])
    ex.shot(f"d_two_images_run{run}")
ex.close()
