"""palette_matrix round 2: a quick probe (what a project shape shows at open, after a click on its source)."""
from __future__ import annotations

import sys

from explore_palette_lib import Matrix

m = Matrix("probe2")
ex = m.ex
kind = sys.argv[1] if len(sys.argv) > 1 else "image"
ex.open(ex.project(kind))
ex.settle(8)
print("heading", m.heading(), "sel", ex.s.sel_step, "path", ex.s.path.name if ex.s.path else None)
print("reasons", {b: r for b, r in m.reasons().items()})
src = ex.block("image") or ex.block("session")
if src is not None:
    ex.click_block(src)
    print("after click: heading", m.heading(), "sel", ex.s.sel_step)
    print("reasons", {b: r for b, r in m.reasons().items()})
    ex.shot("probe_" + kind + "_click")
ex.double_click_card("astrometry")
print("after dbl astrometry", [(q.name, ex.s.graph(q)) for q in ex.s.paths])
ex.close()
