"""palette_matrix round 2: a second Fusion (double-click, and dropped on the bottom zone): is it drawn?"""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("probe_fusion2")
ex = m.ex


def fus():
    return [(it.node.key, it.title, it.summary, [q.name for q in it.node.paths], it.sceneBoundingRect().getRect(),
             it.toolTip()[:200]) for it in ex.blocks() if it.node.key == "fusion"]


for how in ("dbl", "zone"):
    for attempt in (1, 2):
        ex.open(ex.project("mono"))
        ex.settle(8)
        ex.double_click_card("fusion")
        print(how, attempt, "after 1", fus(), [q.name for q in ex.s.paths])
        if how == "dbl":
            ex.double_click_card("fusion")
        else:
            z = ex.new_flow_zone()
            print("  drop", ex.drag_palette("fusion", ex.view.mapFromScene(z.sceneBoundingRect().center())))
        print(how, attempt, "after 2", fus(), [q.name for q in ex.s.paths], "heading", m.heading(),
              "sel", ex.s.sel_step, ex.s.path.name if ex.s.path else None)
        print("  run", ex.w.bt_run.text(), ex.w.bt_run.isEnabled(), repr(ex.w.bt_run.toolTip()[:200]))
        if attempt == 2:
            ex.shot(f"fusion2_{how}")
ex.close()
