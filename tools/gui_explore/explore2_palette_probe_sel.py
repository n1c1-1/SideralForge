"""palette_matrix round 2: which block is chosen, and what the right panel shows, after a placement, its Ctrl+Z and its
Ctrl+Y."""
from __future__ import annotations

from explore2_palette_lib import Matrix

from PySide6.QtWidgets import QLabel

m = Matrix("probe_sel")
ex = m.ex
_hooked = []


def panel():
    return [x.text() for x in ex.page.panel.findChildren(QLabel) if x.isVisible() and x.text()][:3]


for kind, cards in (("session_empty_wf", ("calibration", "registration", "stacking")), ("image", ("astrometry", "gradient"))):
    for attempt in (1, 2):
        ex.open(ex.project(kind))
        ex.settle(8)
        if kind == "image":
            ex.click_block(ex.block("image"))
        for b in cards:
            ex.double_click_card(b)
        print(kind, attempt, "placed: sel", ex.s.sel_step, "panel", panel())
        ex.undo()
        print(kind, attempt, "Ctrl+Z: sel", ex.s.sel_step, "panel", panel())
        ex.redo()
        print(kind, attempt, "Ctrl+Y: sel", ex.s.sel_step, "panel", panel(), "node", ex.page.current_node() is not None)
        if attempt == 2:
            ex.shot(f"sel_apres_ctrl_y_{kind}")
        # the keys of the chosen block, with that selection
        from PySide6.QtCore import Qt
        s0 = m.state()
        for k in (Qt.Key_Delete, Qt.Key_D, Qt.Key_Return):
            ex.key(k)
            ex.settle(4)
            print("   key", k, "changed", m.state() != s0, "toast", m.toasts[-1:], "mode", ex.w.screens.currentIndex()
                  if hasattr(ex.w, "screens") else None, "exc", [x.splitlines()[0] for x in m.exc])
            ex.w.set_mode(0)
            ex.settle(3)
ex.close()
