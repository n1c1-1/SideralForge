"""Words, round 2: the screenshots of the new notes, taken at the moment it shows (palette reasons hovered, the Inspect
band, the settings pages).

    python explore2_words_shots.py fr
"""
from __future__ import annotations

import faulthandler
import sys

faulthandler.dump_traceback_later(300, exit=True)
from harness import Explorer, stray_qt_errors  # noqa: E402
from PySide6.QtCore import QPoint, QPointF  # noqa: E402
from PySide6.QtGui import QEnterEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
stray_qt_errors()
ex = Explorer("words_layout", lang=LANG, size=(1366, 768))
D = ex.out / "notes2"
D.mkdir(exist_ok=True)


def hover(block, name):
    card = ex.page.cards.get(block)
    QApplication.sendEvent(card, QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(card.mapToGlobal(QPoint(5, 5)))))
    ex.settle(3)
    print(block, "->", ex.page.hint.text() if ex.page.hint.isVisible() else "(no hint)", flush=True)
    ex.w.grab().save(str(D / f"{LANG}_{name}.png"))
    QApplication.sendEvent(card, QEnterEvent(QPointF(-5, -5), QPointF(-5, -5), QPointF(-5, -5)))
    ex.page.hide_hint()


ex.open(ex.project("session"))
ex.settle(6)
ex.click_block(ex.block("session") or ex.blocks()[0])
ex.double_click_card("calibration")
ex.settle(4)
hover("stars", "raison_etoiles_sortie")
ex.open(ex.project("image"))
ex.settle(6)
hover("registration", "raison_pas_de_alignement")
hover("fusion", "raison_fusion_flux")
ex.close()
