"""usages K: the Full template's graph before and after the first lights are added (a Recombine appears?), what is
said, the undo steps; the same for Several nights."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore5_usages_common import EXC, U, log, start  # noqa: E402

start(600)
rep = sys.argv[1] if len(sys.argv) > 1 else "1"
u = U("fr")
ex = u.ex
for key in ("full", "nights", "classic"):
    lights = ex.raws(f"K{key}{rep}/l", 5)
    root, _c = u.new_from_home(f"K_{key}_{rep}", key)
    b0 = sorted(it.title for it in ex.blocks())
    w0 = [(w.name, [b["block"] for b in (w.blocks or [])]) for w in ex.s.p.workflows]
    ex.shot(f"K_{key}{rep}_created")
    n_t = len(u.toasts)
    u0 = len(ex.s._undo)
    u.add_frames(1, lights, 0)
    b1 = sorted(it.title for it in ex.blocks())
    w1 = [(w.name, [b["block"] for b in (w.blocks or [])]) for w in ex.s.p.workflows]
    ex.shot(f"K_{key}{rep}_with_lights")
    log("K", key, "blocks added by the lights:", sorted(set(b1) - set(b0)) or [x for x in b1 if b1.count(x) > b0.count(x)],
        "removed:", sorted(set(b0) - set(b1)), "\n  def before", w0, "\n  def after", w1, "\n  toasts", u.toasts[n_t:],
        "undo steps", len(ex.s._undo) - u0)
    ex.w.close_project()
log("EXC", len(EXC))
ex.w.close()
