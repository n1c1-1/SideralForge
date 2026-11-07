"""Probe: what each project state shows (blocks, palette heading, card reasons, ghosts per dragged card)."""
import faulthandler
import sys

from harness import Explorer

faulthandler.dump_traceback_later(600, exit=True)
ex = Explorer("palette_matrix")
kinds = sys.argv[1:] or ["blank", "session", "session_empty_wf", "image", "session+image", "two_sessions",
                         "two_flows", "mono", "legacy", "jpg_mistake"]
for kind in kinds:
    root = ex.project(kind)
    ex.open(root)
    s = ex.s
    print("=====", kind, "ready", s.ready)
    print(" paths:", [(q.pid, q.name, (q.source or {}).get("path", "-") if q.source else None, s.graph(q)) for q in s.paths])
    print(" sessions:", [(x.id, x.name, x.filter) for x in s.p.sessions])
    print(" blocks:", [(it.node.key, it.node.sid, [q.pid for q in it.node.paths], it.title if hasattr(it, 'title') else '') for it in ex.blocks()])
    print(" sel:", s.sel_step, "current", s.current)
    from PySide6.QtWidgets import QLabel
    heads = [x.text() for x in ex.page.palette_bar.widget().findChildren(QLabel)][:3]
    print(" head:", heads)
    for b, c in ex.page.cards.items():
        print("   card", b, repr(c.title), "reason:", repr(c.reason))
    for b in list(ex.page.cards):
        ex.drag_palette_hover(b, ex.view.viewport().rect().center())
        gs = ex.ghosts()
        nf = ex.new_flow_zone()
        print("   drag", b, "ghosts:", [(type(g).__name__, getattr(g, "text", ""), g.pid if hasattr(g, "pid") else None,
                                         getattr(g, "idx", None), getattr(g, "branch_key", None)) for g in gs],
              "nf hot:", getattr(nf, "hot", None))
        ex.drag_leave()
ex.close()
