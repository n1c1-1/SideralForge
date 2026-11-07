"""usages J: the Several nights template from Home, both nights filled; a branch, Post, settings, Save, reopen (from A)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore5_usages_common import U, _, log, start, wait, EXC  # noqa: E402

start(900)
rep = sys.argv[1] if len(sys.argv) > 1 else "1"
u = U("fr")
ex = u.ex
name = f"NuitsJ{rep}"
lights = ex.raws(f"{name}/lights", 8)
darks = ex.raws(f"{name}/darks", 4, start=8000)
flats = ex.raws(f"{name}/flats", 4, start=8100)
root, cap = u.new_from_home(name, "nights")
log("created", root, ex.w.project.root if ex.w.project else None, [it.title for it in ex.blocks()])
l2 = ex.raws(f"{name}/n2", 5, start=7000)
u.add_frames(1, lights, 0)
u.add_frames(1, darks, 1)
u.add_frames(1, flats, 2)
u.add_frames(2, l2, 0)
log("nights", [(x.id, x.name, len(x.lights)) for x in ex.s.p.sessions], [it.title for it in ex.blocks()])
pass
log("chain", [it.title for it in ex.blocks()], [q.name for q in ex.s.paths])
ok = u.branch("gradient", "empilement")
log("branch", ok, [q.name for q in ex.s.paths], [it.title for it in ex.blocks()])
ex.shot(f"J{rep}_built")
# images to Post: the Stacking (raw) by its menu
st = u.blk("stacking") or u.blk("empilement")
log("stacking block", st.title if st else None)
if st is not None:
    u.to_post(st)
log("post layers", [(l.get("name"), l.get("pid")) for l in ex.s.p.post.get("layers", [])], [q.name for q in ex.s.paths])
# settings
g = u.title_blk("Gradient") or u.blk("gradient") or u.blk("fond")
ch1 = u.change_setting(g) if g else None
stk = u.blk("stacking") or u.blk("empilement")
ch2 = u.change_setting(stk) if stk else None
log("settings changed", ch1, ch2, "toasts", u.toasts[-4:])
# rename the workflow by its panel? select a block for the reopen
ex.click_block(u.blk("color") or u.blk("couleur") or stk)
wait(ex, 0.3)
ex.shot(f"J{rep}_before_save")
before, saved, after, diffs = u.roundtrip(root, "Couleur", ["Accueil > Nouveau projet, Vierge", "Session + brutes, darks, flats",
                                                            "Calibration … Débruitage", "branche Gradient sous Empilement",
                                                            "Empilement > Vers Post", "deux réglages changés"])
ex.shot(f"J{rep}_reopened")
(ex.out / f"J{rep}_snap.json").write_text(json.dumps({"before": before, "saved": saved, "after": after, "diffs": diffs},
                                                      ensure_ascii=False, indent=1, default=str), encoding="utf-8")
log("EXC", len(EXC), EXC[-1][-800:] if EXC else "")
ex.w.close()
