"""usages I: the less common settings across a reopen: a block switched off (D), a review pause, the Fusion's palette,
a channel weight and its stars, darks shared from another session, the mode (Graph / Inspect / Post) left open."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore5_usages_common import (  # noqa: E402
    EXC, U, _, buttons, compare, log, menu_button, panel, ptexts, start, wait,
)
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QPushButton  # noqa: E402

start(900)
rep = sys.argv[1] if len(sys.argv) > 1 else "1"
u = U("fr")
ex = u.ex
root, _c = u.new_from_home(f"ReglagesI_{rep}", "sho")
for x in list(ex.s.p.sessions):
    u.add_frames(x.id, ex.fits_frames(f"ReglagesI_{rep}/{x.filter}", 4, x.filter), 0)
# darks on SII's session, shared with Ha's
dk = ex.raws(f"ReglagesI_{rep}/darks", 3, start=8000)
u.add_frames(1, dk, 1)
ex.click_block(u.blk("session", sid=2))
wait(ex, 0.3)
bs = buttons(panel(ex), _("Add") + " ▾", QPushButton)
items = menu_button(ex, bs[1], "SII")
wait(ex, 0.6)
log("I share menu", [i.get("text") for i in items], ex.s.p.session(2).darks)
# Ha's Gradient switched off (D), a review pause after OIII's Stacking
pids = {q.name: q.pid for q in ex.s.paths}
g = ex.block("fond", pid=pids.get("Ha")) or ex.block("gradient", pid=pids.get("Ha"))
ex.click_block(g)
ex.key(Qt.Key_D)
wait(ex, 0.5)
st = ex.block("empilement", pid=pids.get("OIII"))
ex.right_click_block(st, _("Review pause"))
wait(ex, 0.5)
log("I off/pause", u.toasts[-2:])
# the Fusion: palette HSO, OIII's weight, the stars
fu = u.blk("fusion")
ex.click_block(fu)
wait(ex, 0.4)
b = next((x for x in panel(ex).findChildren(QPushButton) if x.isVisible() and x.text() == "HSO"), None)
if b is not None:
    ex.click(b)
    wait(ex, 0.5)
ex.click_block(u.blk("fusion"))
wait(ex, 0.4)
sps = [x for x in panel(ex).findChildren(QDoubleSpinBox) if x.isVisible()]
if sps:
    sps[-1].setValue(0.7)
    sps[-1].editingFinished.emit()
    wait(ex, 0.5)
ex.click_block(u.blk("fusion"))
wait(ex, 0.4)
cb = next((c for c in panel(ex).findChildren(QComboBox) if c.isVisible() and c.count() > 1), None)
if cb is not None:
    i = (cb.currentIndex() + 1) % cb.count()
    cb.setCurrentIndex(i)
    cb.activated.emit(i)
    wait(ex, 0.5)
ex.click_block(u.blk("fusion"))
wait(ex, 0.4)
fpanel = ptexts(ex)[:30]
log("I fusion panel", fpanel)
fdef = [w.to_dict() if hasattr(w, "to_dict") else None for w in ex.s.p.workflows][-1]
# Inspect mode left open
ex.w.set_mode(1)
wait(ex, 1.0)
ex.shot(f"I{rep}_before_save")
before, saved, after, diffs = u.roundtrip(root, "réglages rares", ["SHO", "darks partagés", "D", "pause", "Fusion"])
ex.shot(f"I{rep}_reopened")
ex.click_block(u.blk("fusion"))
wait(ex, 0.4)
fpanel2 = ptexts(ex)[:30]
log("I fusion panel after", fpanel2)
log("I mode after", after["mode"], "before", saved["mode"])
(ex.out / f"I_{rep}.json").write_text(json.dumps({"saved": saved, "after": after, "diffs": diffs, "fpanel": fpanel,
                                                   "fpanel2": fpanel2}, ensure_ascii=False, indent=1, default=str),
                                      encoding="utf-8")
d = saved["definition"]
log("I saved def bits", [(w["name"], w.get("breakpoints"), [b.get("off") for b in (w.get("blocks") or [])],
                          w.get("params", {}).get("fusion")) for w in d["workflows"]], d["sessions"][1].get("darks"))
log("EXC", len(EXC), "\n".join(e[-1200:] for e in EXC))
ex.w.close()
