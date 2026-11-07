"""palette_matrix round 2: a workflow of two sessions; the Session card double-clicked, dropped on "+ Session", or a
click on "+ Session": which sessions does the workflow keep?"""
from __future__ import annotations

import sys

from explore2_palette_lib import Matrix

from PySide6.QtCore import QPoint

m = Matrix("probe_nights")
ex = m.ex
SHAPE = sys.argv[1] if len(sys.argv) > 1 else "new_nights"


def wf():
    s = ex.s
    return [(q.name, list(s.sessions_of(q)), [n["block"] for n in s.graph(q)][:3]) for q in s.paths], \
        [(x.id, x.name, len(x.lights)) for x in s.p.sessions]


def fresh():
    if SHAPE == "new_nights":
        m._k = getattr(m, "_k", 0) + 1
        m.new_project(f"Nights{m._k}", "nights")
        # the template's sessions have no frames: give each a folder of raws (as the user would add them)
        for x in ex.s.p.sessions:
            x.lights = [str(ex.raws(f"nights{m._k}/s{x.id}", 8, start=5000 + 100 * x.id))]
        ex.s.flush() if hasattr(ex.s, "flush") else None
        root = ex.s.p.root
        ex.w.close_project(go_home=True)
        ex.settle(4)
        from astro_gpu.core import project2
        p = project2.load(root)
        for x in p.sessions:
            x.lights = [str(ex.raws(f"nights{m._k}/s{x.id}", 8, start=5000 + 100 * x.id))]
        project2.save(p)
        ex.open(root)
    else:
        ex.open(ex.project(SHAPE))
    ex.settle(8)
    m.hook_page()


for how in ("dbl", "drop_plus", "click_plus"):
    for attempt in (1, 2):
        fresh()
        ex.click_block(ex.block("session") or ex.block("sources"))
        print(how, attempt, "start", wf(), "run", ex.w.bt_run.text(), ex.w.bt_run.isEnabled())
        if attempt == 2:
            ex.shot(f"nuits_{how}_avant")
        s0 = m.state()
        if how == "dbl":
            ex.double_click_card("session")
        else:
            from astro_gpu.client.graph import GhostItem
            plus = [g for g in ex.view.scene().items() if isinstance(g, GhostItem) and g.text.strip().endswith("Session")]
            pt = ex.view.mapFromScene(plus[0].sceneBoundingRect().center()) if plus else None
            print("   + Session place", [g.text for g in plus])
            if pt is not None and how == "drop_plus":
                print("   drop", ex.drag_palette("session", pt))
            elif pt is not None:
                ex.click(ex.view.viewport(), pt)
        print(how, attempt, "after", wf(), "toast", m.toasts[-1:], "run", ex.w.bt_run.text(), ex.w.bt_run.isEnabled(),
              "modals", ex.modals[-1:])
        if attempt == 2:
            ex.shot(f"nuits_{how}_apres")
        ex.undo()
        print(how, attempt, "undo exact", m.state() == s0)
ex.close()
