"""usages B: monochrome uses from Home (SHO, LRGB, HOO of two nights per filter): the gallery's template, each filter's
frames added, a second night per filter, a setting, Save, the window closed, the project opened in a new window."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore5_usages_common import EXC, U, _, log, start, wait  # noqa: E402

start(900)
which = sys.argv[1] if len(sys.argv) > 1 else "sho"
rep = sys.argv[2] if len(sys.argv) > 2 else "1"
u = U("fr")
ex = u.ex
FILTERS = {"sho": ("SII", "Ha", "OIII"), "lrgb": ("L", "R", "G", "B"), "hoo": ("Ha", "OIII")}[which]
name = f"Mono{which.upper()}{rep}"
root, cap = u.new_from_home(name, which)
s = ex.s


def sessions():
    return [(x.id, x.name, x.filter, len(x.lights)) for x in ex.s.p.sessions]


def flows():
    return [(q.name, ex.s.p.workflow(q.pid).sessions) for q in ex.s.paths]


log("created", sessions(), flows(), [it.title for it in ex.blocks()])
ex.shot(f"B_{which}{rep}_created")
for x in list(ex.s.p.sessions):
    d = ex.fits_frames(f"{name}/{x.filter or x.name}_n1", 4, x.filter or x.name)
    u.add_frames(x.id, d, 0)
log("frames n1", sessions(), flows())
# a second night for each filter: the filter's Session chosen, Session double-clicked in the palette
for f in FILTERS[:2] if which != "hoo" else FILTERS:
    sid = next(x.id for x in ex.s.p.sessions if x.filter == f)
    ex.click_block(u.blk("session", sid=sid))
    wait(ex, 0.3)
    n0 = len(ex.s.p.sessions)
    u.place("session")
    new = [x for x in ex.s.p.sessions][n0:]
    log("second night for", f, [(x.id, x.name, x.filter) for x in new], flows(), u.toasts[-1:])
    if new:
        d = ex.fits_frames(f"{name}/{f}_n2", 3, f)
        u.add_frames(new[0].id, d, 0)
log("frames n2", sessions(), flows())
ex.shot(f"B_{which}{rep}_two_nights")
# a setting in a channel's Stacking, a block selected
stk = next((it for it in ex.blocks() if it.node.key in ("empilement", "stacking")), None)
ch = u.change_setting(stk) if stk else None
fu = u.blk("fusion")
if fu is not None:
    ex.click_block(fu)
    wait(ex, 0.3)
before, saved, after, diffs = u.roundtrip(root, which, [f"Accueil > Nouveau projet, modèle {which}",
                                                        "brutes de chaque filtre", "deuxième nuit par filtre"])
ex.shot(f"B_{which}{rep}_reopened")
log("after", sessions(), flows())
(ex.out / f"B_{which}{rep}_snap.json").write_text(json.dumps({"before": before, "saved": saved, "after": after,
                                                               "diffs": diffs}, ensure_ascii=False, indent=1,
                                                              default=str), encoding="utf-8")
log("EXC", len(EXC), EXC[-1][-1500:] if EXC else "")
ex.w.close()
