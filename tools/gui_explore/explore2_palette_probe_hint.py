"""palette_matrix round 2: the "why greyed" message of a card when the palette changes under the mouse (Ctrl+Y makes
the card available while the mouse is still on it): does a stale message stay?"""
from __future__ import annotations

from explore2_palette_lib import Matrix

from PySide6.QtCore import QPointF
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QApplication

m = Matrix("probe_hint")
ex = m.ex
for attempt in (1, 2):
    ex.open(ex.project("session_empty_wf"))
    ex.settle(8)
    for b in ("calibration", "registration", "stacking"):
        ex.double_click_card(b)
    ex.undo()
    c = ex.card("astrometry")
    print("astrometry reason", repr(c.reason))
    p = QPointF(c.width() / 2, c.height() / 2)
    QApplication.sendEvent(c, QEnterEvent(p, c.mapToGlobal(p), c.mapToGlobal(p)))
    ex.settle(2)
    print("hint shown", ex.page.hint.isVisible(), repr(ex.page.hint.text()))
    ex.redo()
    ex.settle(6)
    print("after Ctrl+Y: astrometry reason", repr(ex.card("astrometry").reason), "hint visible", ex.page.hint.isVisible(),
          repr(ex.page.hint.text()))
    if attempt == 2:
        ex.shot("hint_apres_ctrl_y")
    # and the same with a double-click placing the block that makes it available, while the message is shown
    ex.undo()
    c = ex.card("astrometry")
    QApplication.sendEvent(c, QEnterEvent(p, c.mapToGlobal(p), c.mapToGlobal(p)))
    ex.settle(2)
    ex.w.set_mode(2)
    ex.settle(4)
    print("after going to Post: hint visible", ex.page.hint.isVisible(), "page visible", ex.page.isVisible())
    ex.w.set_mode(0)
    ex.settle(4)
    print("back to Graph: hint visible", ex.page.hint.isVisible(), repr(ex.page.hint.text()))
    if attempt == 2:
        ex.shot("hint_retour_graphe")
ex.close()
