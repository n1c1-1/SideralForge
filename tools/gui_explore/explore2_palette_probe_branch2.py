"""palette_matrix round 2: Gradient dropped on the "new branch" place under Stacking: the toast and the branch's name."""
from __future__ import annotations

from explore2_palette_lib import Matrix

from PySide6.QtCore import QPoint

m = Matrix("probe_branch2")
ex = m.ex
for attempt in (1, 2):
    ex.open(ex.project("legacy"))
    ex.settle(8)
    m.hook_page()
    vp = ex.view.viewport()
    ex.drag_palette_hover("gradient", QPoint(vp.width() // 2, 4))
    gh = [(getattr(g, "branch_key", None), ex.ghost_center(g)) for g in ex.ghosts()]
    ex.drag_leave()
    tgt = next(g for g in gh if g[0] == "empilement")
    ex.drag_palette("gradient", tgt[1])
    print(attempt, "toast", m.toasts[-1:], "flows", [(q.name, [n["block"] for n in ex.s.graph(q)][3:]) for q in ex.s.paths])
    if attempt == 2:
        ex.shot("branche_sous_empilement")
ex.close()
