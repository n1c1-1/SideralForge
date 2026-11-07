"""palette_matrix round 2: a Session double-clicked in a project of filter workflows (mono Ha/OIII; the SHO template with
its Fusion chosen): which workflow takes it, with which filter?"""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("probe_filter")
ex = m.ex
for kind in ("mono", "sho"):
    for attempt in (1, 2):
        if kind == "sho":
            m._k = getattr(m, "_k", 0) + 1
            m.new_project(f"F{m._k}", "sho")
        else:
            ex.open(ex.project("mono"))
        ex.settle(8)
        m.hook_page()
        h = m.heading()
        ex.double_click_card("session")
        s = ex.s
        new = s.p.sessions[-1]
        print(kind, attempt, "heading before", h, "| new", (new.id, new.name, new.filter), "| flows",
              [(q.name, list(s.sessions_of(q))) for q in s.paths], "| toast", m.toasts[-1:])
        if attempt == 2:
            ex.shot(f"session_filtre_{kind}")
ex.close()
