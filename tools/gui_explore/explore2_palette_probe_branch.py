"""palette_matrix round 2: a repeated block as a branch: Gradient dropped on the "new branch" place after Gradient
(workflow complete); names of the shared Gradient and of the branch's one, in the graph, the panel, the run window."""
from __future__ import annotations

import time

from explore2_palette_lib import Matrix

from PySide6.QtCore import QPoint

m = Matrix("probe_branch")
ex = m.ex


def grads():
    return [(it.node.key, it.title, [q.name for q in it.node.paths]) for it in
            sorted(ex.blocks(), key=lambda i: (i.sceneBoundingRect().x(), i.sceneBoundingRect().y()))
            if it.node.key.split("~")[0] == "fond"]


for attempt in (1, 2):
    ex.open(ex.project("legacy"))
    ex.settle(8)
    vp = ex.view.viewport()
    ex.drag_palette_hover("gradient", QPoint(vp.width() // 2, 4))
    gh = [(getattr(g, "branch_key", None), ex.ghost_center(g)) for g in ex.ghosts()]
    ex.drag_leave()
    tgt = next(g for g in gh if g[0] == "fond")
    s0 = m.state()
    print("drop", ex.drag_palette("gradient", tgt[1]))
    print("flows", [(q.name, [n.get("name") for n in ex.s.graph(q)]) for q in ex.s.paths])
    print("grads", grads(), "sel", ex.s.sel_step, "heading", m.heading(), "toast", m.toasts[-1:])
    if attempt == 2:
        ex.shot("branche_gradient_apres_gradient")
    # each Gradient chosen: its panel title
    from PySide6.QtWidgets import QLabel
    for i in range(len(grads())):
        its = sorted([it for it in ex.blocks() if it.node.key.split("~")[0] == "fond"],
                     key=lambda i: (i.sceneBoundingRect().x(), i.sceneBoundingRect().y()))
        ex.click_block(its[i])
        labs = [x.text() for x in ex.page.panel.findChildren(QLabel) if x.isVisible() and x.text()]
        print("  chosen", its[i].title if i < len(its) else None, "panel", labs[:2], "heading", m.heading())
    # run to here on the branch's Gradient (Shift+F5)
    from PySide6.QtCore import Qt
    its = sorted([it for it in ex.blocks() if it.node.key.split("~")[0] == "fond"],
                 key=lambda i: (i.sceneBoundingRect().x(), i.sceneBoundingRect().y()))
    ex.click_block(its[-1])
    ex.key(Qt.Key_F5, ex.w, Qt.ShiftModifier)
    for _ in range(6):
        ex.settle(5)
        time.sleep(0.05)
    d = ex.w.run.dialog
    print("  run to here", d.windowTitle() if d else None, [r.name.text() for r in d.rows] if d else None)
    if d:
        if attempt == 2:
            ex.shot("branche_gradient_jusquici", d)
        d.close()
    ex.undo()
    print("undo exact", m.state() == s0)
ex.close()
