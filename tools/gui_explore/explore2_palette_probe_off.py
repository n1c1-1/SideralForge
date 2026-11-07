"""palette_matrix round 2: a block switched off (D), then the same block placed from the palette (double-click and drop
on each place): accepted? names? messages? Undo/redo."""
from __future__ import annotations

from explore2_palette_lib import Matrix

from PySide6.QtCore import QPoint, Qt

m = Matrix("probe_off")
ex = m.ex


def item(key):
    its = [it for it in ex.blocks() if it.node.key == key]
    return its[0] if its else None


for eng, card in (("fond", "gradient"), ("alignement", "registration"), ("debruitage", "denoise")):
    for attempt in (1, 2):
        ex.open(ex.project("legacy"))
        ex.settle(8)
        m.hook_page()
        ex.click_block(item(eng))
        ex.key(Qt.Key_D)
        print(card, attempt, "off:", [(it.node.key, it.title, it.node.cut) for it in ex.blocks() if it.node.key.startswith(eng)],
              "card reason", repr(ex.card(card).reason), "toast", m.toasts[-1:])
        s0 = m.state()
        e0 = len(m.exc)
        ex.double_click_card(card)
        print("   dbl:", "changed", m.state() != s0, [(it.node.key, it.title, it.node.cut) for it in ex.blocks()
                                                      if it.node.key.startswith(eng)], "toast", m.toasts[-1:],
              "popups", ex.popups[-1:], "exc", [x.splitlines()[0] for x in m.exc[e0:]])
        if m.state() != s0:
            ex.undo()
            print("   undo exact", m.state() == s0)
        vp = ex.view.viewport()
        ex.drag_palette_hover(card, QPoint(vp.width() // 2, 4))
        gh = [(g.text, getattr(g, "branch_key", None), ex.ghost_center(g)) for g in ex.ghosts()]
        ex.drag_leave()
        print("   places", len(gh), [g[:2] for g in gh][:4])
        for g in [x for x in gh if x[1] is None][:2]:
            s1 = m.state()
            rec = ex.drag_palette(card, g[2])
            print("   drop", rec, "changed", m.state() != s1, "toast", m.toasts[-1:], "exc",
                  [x.splitlines()[0] for x in m.exc[e0:]])
            if m.state() != s1:
                ex.undo()
        if attempt == 2:
            ex.shot(f"off_{card}")
ex.close()
