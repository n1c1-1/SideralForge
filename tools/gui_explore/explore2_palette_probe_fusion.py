"""palette_matrix round 2: Fusion placed twice in a mono project (names, reasons, undo/redo), then the language changed
at runtime with copies placed (names in the graph, the palette, its heading, the hints)."""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("probe_fusion")
ex = m.ex


def srcs():
    return [(it.node.key, it.title, getattr(it, "summary", "")) for it in
            sorted(ex.blocks(), key=lambda i: (i.sceneBoundingRect().y(), i.sceneBoundingRect().x()))
            if it.node.key in ("fusion", "session", "image")]


for attempt in (1, 2):
    ex.open(ex.project("mono"))
    ex.settle(8)
    print("start", srcs(), "fusion reason", repr(ex.card("fusion").reason))
    s0 = m.state()
    ex.double_click_card("fusion")
    s1 = m.state()
    print("fusion 1", srcs(), "flows", [q.name for q in ex.s.paths], "heading", m.heading(), "sel", ex.s.sel_step,
          "reason now", repr(ex.card("fusion").reason), "toasts", m.toasts[-1:], "modals", ex.modals[-1:])
    ex.double_click_card("fusion")
    s2 = m.state()
    print("fusion 2", srcs(), "flows", [q.name for q in ex.s.paths], "heading", m.heading())
    if attempt == 2:
        ex.shot("fusion_deux_fois")
    ex.undo()
    print("undo exact", m.state() == s1)
    ex.undo()
    print("undo2 exact", m.state() == s0)
    ex.redo()
    print("redo exact", m.state() == s1)
    ex.redo()
    print("redo2 exact", m.state() == s2)
    # the language at runtime
    ex.double_click_card("gradient") if not ex.card("gradient").reason else None
    ex.double_click_card("gradient") if not ex.card("gradient").reason else None
    before = [(it.node.key, it.title) for it in ex.blocks()]
    ex.w.change_language("en")
    ex.settle(10)
    after = [(it.node.key, it.title) for it in ex.blocks()]
    print("names fr", sorted(before))
    print("names en", sorted(after))
    print("palette en", m.heading(), {b: (c.title, c.reason) for b, c in ex.page.cards.items()})
    if attempt == 2:
        ex.shot("langue_en_apres_copies")
    ex.w.change_language("fr")
    ex.settle(10)
    print("palette fr again", m.heading(), {b: c.title for b, c in ex.page.cards.items()})
    print("exceptions", [x.splitlines()[0] for x in m.exc])
ex.close()
