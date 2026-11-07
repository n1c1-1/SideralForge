"""palette_matrix round 2: the Log (Journal) lines written by palette gestures with repeated blocks: names, internal
words, language (written in French, then read in English)."""
from __future__ import annotations

from explore2_palette_lib import Matrix, frenchness, englishness

from PySide6.QtCore import QPoint, Qt

from astro_gpu.client.logview import event_text

m = Matrix("log")
ex = m.ex


def g(prefix, i):
    xs = sorted([it for it in ex.blocks() if it.node.key.split("~")[0] == prefix], key=lambda i: i.sceneBoundingRect().x())
    return xs[i] if len(xs) > i else None


for attempt in (1, 2):
    ex.open(ex.project("legacy"))
    ex.settle(8)
    n0 = len(ex.s.history or [])
    ex.double_click_card("gradient")
    ex.double_click_card("color")
    ex.right_click_block(g("fond", 1), "Pause")
    ex.click_block(g("couleur", 1))
    ex.key(Qt.Key_D)
    ex.click_block(g("fond", 0))
    ex.key(Qt.Key_Delete)
    vp = ex.view.viewport()
    ex.drag_palette_hover("denoise", QPoint(vp.width() // 2, 4))
    gh = [(getattr(x, "branch_key", None), ex.ghost_center(x)) for x in ex.ghosts()]
    ex.drag_leave()
    b = next((x for x in gh if x[0]), None)
    if b:
        ex.drag_palette("denoise", b[1])
    ex.double_click_card("composition")
    rows = (ex.s.history or [])[n0:]
    fr = [event_text(r)[1] for r in rows]
    print(attempt, "FR log:", fr)
    for t in fr:
        for p in frenchness(t):
            print("   FR problem:", t, "|", p)
    from astro_gpu.client import app as appmod
    appmod.apply_language("en")
    en = [event_text(r)[1] for r in rows]
    print(attempt, "EN log:", en)
    for t in en:
        for p in englishness(t):
            print("   EN problem:", t, "|", p)
    appmod.apply_language("fr")
    print(attempt, "raw rows", [r[1:] for r in rows][:12])
ex.close()
