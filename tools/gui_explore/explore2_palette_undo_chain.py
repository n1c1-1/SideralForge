"""palette_matrix round 2: a chain of palette gestures (double-clicks and drops), then Ctrl+Z back to the start one step
at a time, then Ctrl+Y forward again: each state exactly the one of that step.   PM_LANG=fr python ..."""
from __future__ import annotations

from explore2_palette_lib import Matrix

from PySide6.QtCore import QPoint

m = Matrix("undo_chain")
ex = m.ex


def drop_first_ghost(card, want=None):
    vp = ex.view.viewport()
    ex.drag_palette_hover(card, QPoint(vp.width() // 2, 4))
    gh = [(g.text, getattr(g, "branch_key", None), ex.ghost_center(g)) for g in ex.ghosts()]
    ex.drag_leave()
    g = next((x for x in gh if (want is None and x[1] is None) or (want and x[1] == want)), None)
    if g is None:
        return f"no place for {card}"
    return ex.drag_palette(card, g[2])


def zone(card):
    z = ex.new_flow_zone()
    return ex.drag_palette(card, ex.view.mapFromScene(z.sceneBoundingRect().center()))


for attempt in (1, 2):
    ex.open(ex.project("session+image"))
    ex.settle(8)
    m.hook_page()
    ex.click_block(ex.block("session") or ex.block("sources"))
    steps = [("dbl calibration", lambda: ex.double_click_card("calibration")),
             ("drop registration", lambda: drop_first_ghost("registration")),
             ("dbl stacking", lambda: ex.double_click_card("stacking")),
             ("dbl astrometry", lambda: ex.double_click_card("astrometry")),
             ("dbl gradient", lambda: ex.double_click_card("gradient")),
             ("dbl gradient (2)", lambda: ex.double_click_card("gradient")),
             ("branch gradient after stacking", lambda: drop_first_ghost("gradient", "empilement")),
             ("image on zone", lambda: zone("image")),
             ("session on zone", lambda: zone("session")),
             ("dbl color", lambda: ex.double_click_card("color"))]
    states = [m.state()]
    for name, fn in steps:
        r = fn()
        ex.settle(4)
        st = m.state()
        changed = st != states[-1]
        print(attempt, "do", name, "changed" if changed else "NOTHING", r if r not in (None,) else "", "| heading",
              m.heading(), flush=True)
        if changed:
            states.append(st)
    n = len(states) - 1
    bad = []
    for i in range(n, 0, -1):
        ex.undo()
        if m.state() != states[i - 1]:
            bad.append(("undo", i, m.diff(states[i - 1], m.state())[:600]))
    extra = m.state()
    ex.undo()                                              # one more: nothing left to undo
    if m.state() != extra:
        bad.append(("undo past start", m.diff(extra, m.state())[:600]))
    ex.redo() if m.state() != extra else None
    for i in range(1, n + 1):
        ex.redo()
        if m.state() != states[i]:
            bad.append(("redo", i, m.diff(states[i], m.state())[:600]))
    print(attempt, "steps", n, "problems", bad, "exceptions", [x.splitlines()[0] for x in m.exc], flush=True)
    if attempt == 2:
        ex.shot("undo_chain_fin")
ex.close()
