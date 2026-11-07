"""palette_matrix round 2: image workflow, Astrometry + Gradient placed, Ctrl+Z, Ctrl+Y: the panel shows Gradient;
what do Del and D act on then?"""
from __future__ import annotations

from explore2_palette_lib import Matrix

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

m = Matrix("probe_del")
ex = m.ex


def panel():
    return [x.text() for x in ex.page.panel.findChildren(QLabel) if x.isVisible() and x.text()][:3]


def g():
    q = ex.s.paths[0]
    return [(n["block"], n.get("name"), n.get("off")) for n in ex.s.graph(q)], ex.s.gives_post(q), \
        [(it.node.key, it.title) for it in ex.blocks()], [x.get("kind") for x in ex.s.post_layers()]


for attempt in (1, 2):
    ex.open(ex.project("image"))
    ex.settle(8)
    m.hook_page()
    ex.click_block(ex.block("image"))
    for b in ("astrometry", "gradient"):
        ex.double_click_card(b)
    print(attempt, "placed", g(), "sel", ex.s.sel_step)
    ex.undo()
    ex.redo()
    print(attempt, "after Ctrl+Z Ctrl+Y: sel", ex.s.sel_step, "panel", panel(), "highlighted",
          [it.title for it in ex.blocks() if it.isSelected()])
    ex.key(Qt.Key_Delete)
    print(attempt, "after Del", g(), "toast", m.toasts[-1:], "sel", ex.s.sel_step, "panel", panel())
    if attempt == 2:
        ex.shot("del_apres_ctrl_y_image")
    ex.undo()
    print(attempt, "undo", g())
ex.close()
