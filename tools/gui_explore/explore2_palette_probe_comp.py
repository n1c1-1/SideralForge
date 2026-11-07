"""palette_matrix round 2: the Recombine card on a full workflow: before / after the first double-click, the names in
the graph, the panel, Post's layer list."""
from __future__ import annotations

from explore2_palette_lib import Matrix

from PySide6.QtWidgets import QLabel, QListWidget, QTreeWidget, QWidget

m = Matrix("probe_comp")
ex = m.ex


def comp():
    return [(it.node.lid, it.title, getattr(it, "summary", "")) for it in ex.blocks() if it.node.key == "composition"]


def post_sum():
    it = ex.view.final_item
    return getattr(it, "summary", "")


for attempt in (1, 2):
    ex.open(ex.project("legacy"))
    ex.settle(8)
    print("open", comp(), post_sum(), [x.get("name") for x in ex.s.post_layers()])
    if attempt == 2:
        ex.shot("probe_comp_0_ouverture")
    ex.double_click_card("composition")
    print("dbl1", comp(), post_sum(), [x.get("name") for x in ex.s.post_layers()], m.toasts[-2:])
    ex.double_click_card("composition")
    print("dbl2", comp(), post_sum(), [x.get("name") for x in ex.s.post_layers()])
    # the names in Post
    ex.w.set_mode(2)
    ex.settle(8)
    txt = [t for t in ex.texts() if "ecomb" in t or "Combin" in t or "Full" in t]
    print("post texts", txt[:20])
    if attempt == 2:
        ex.shot("probe_comp_post")
    ex.w.set_mode(0)
    ex.settle(4)
ex.close()

