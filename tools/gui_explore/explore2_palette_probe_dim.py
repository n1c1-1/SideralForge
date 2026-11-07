"""palette_matrix round 2: blocks drawn faint in a mono project after a Fusion: why (their state and tooltip)?"""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("probe_dim")
ex = m.ex
for attempt in (1, 2):
    ex.open(ex.project("mono"))
    ex.settle(8)
    st = lambda: [(it.node.key, [q.name for q in it.node.paths], it.node.cut, it.node.grey, it.node.broken,  # noqa: E731
                   it.toolTip().replace("\n", " / ")[:160]) for it in sorted(ex.blocks(), key=lambda i: (i.node.col, i.node.y))
                  if it.node.key in ("alignement", "empilement", "debruitage")]
    print("before", st())
    ex.double_click_card("fusion")
    ex.settle(6)
    print("after", st())
    ex.click_block(ex.block("alignement"))
    print("panel after click on Alignement", ex.s.sel_step, [m.heading()])
    ex.shot(f"dim_{attempt}")
ex.close()
