"""palette_matrix round 2: an already stretched image: Stars/background then Recombine from the palette, names, Post's
summary, undo/redo of each."""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("probe_stretched")
ex = m.ex


def view():
    return [(it.node.key, it.title, it.summary[:30] if isinstance(it.summary, str) else "") for it in
            sorted(ex.blocks(), key=lambda i: (i.node.col, i.node.y))], \
        {b: c.reason for b, c in ex.page.cards.items() if b in ("stars", "composition", "gradient")}


for attempt in (1, 2):
    m._k = getattr(m, "_k", 0) + 1
    m.new_project(f"ST{m._k}", "image_stretched")
    ex.settle(8)
    m.hook_page()
    ex.click_block(ex.block("image"))
    print(attempt, "start", view())
    s0 = m.state()
    ex.double_click_card("stars")
    s1 = m.state()
    print(attempt, "stars", view(), m.toasts[-1:])
    ex.double_click_card("composition")
    s2 = m.state()
    print(attempt, "recombine", view(), m.toasts[-1:])
    if attempt == 2:
        ex.shot("etiree_etoiles_recombiner")
    ex.undo()
    print(attempt, "undo1", m.state() == s1)
    ex.undo()
    print(attempt, "undo2", m.state() == s0)
    ex.redo()
    ex.redo()
    print(attempt, "redo2", m.state() == s2, [x.splitlines()[0] for x in m.exc])
ex.close()
