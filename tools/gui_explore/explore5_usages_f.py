"""usages F: the Full template used to the end of the graph: frames, a second night, the Recombine's blend and a
slider (its Post layer), the starless image sent to Post, the workflow renamed, a wire to Post removed; Save, the window
closed, reopened in a new window: Post's layers and wires compared."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore5_usages_common import EXC, U, _, log, panel, start, wait  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QComboBox, QLineEdit  # noqa: E402

start(900)
rep = sys.argv[1] if len(sys.argv) > 1 else "1"
u = U("fr")
ex = u.ex


def name_dialog(newname):
    def h(d):
        eds = d.findChildren(QLineEdit)
        if eds:
            eds[0].setText(newname)
        d.accept()
    return h


def layers():
    return [(l.get("name"), l.get("pid"), l.get("blend"), l.get("sliders"), l.get("visible", True), l.get("part"))
            for l in ex.s.p.post.get("layers", [])]


l1 = ex.raws(f"Complet_{rep}/n1", 6)
l2 = ex.raws(f"Complet_{rep}/n2", 5, start=7000)
root, _c = u.new_from_home(f"CompletF_{rep}", "full")
u.add_frames(1, l1, 0)
ex.click_block(u.blk("session") or u.blk("sources"))
u.place("session")
sids = [x.id for x in ex.s.p.sessions]
if len(sids) > 1:
    u.add_frames(sids[-1], l2, 0)
log("F sessions", [(x.id, x.name, len(x.lights)) for x in ex.s.p.sessions],
    [(q.name, ex.s.p.workflow(q.pid).sessions) for q in ex.s.paths])
# the Recombine: its blend and a slider
rc = u.blk("composition")
ex.click_block(rc)
wait(ex, 0.4)
cb = next((c for c in panel(ex).findChildren(QComboBox) if c.isVisible() and c.count() == 3), None)
if cb is not None:
    cb.setCurrentIndex(1)
    wait(ex, 0.5)
from astro_gpu.client.post3 import ContSlider  # noqa: E402
sl = next((x for x in panel(ex).findChildren(ContSlider) if x.isVisible()), None)
if sl is not None:
    sl.done.emit(3.0)
    wait(ex, 0.5)
log("F layers after recombine edits", layers(), u.toasts[-2:])
# the starless image to Post, the workflow renamed
et = u.blk("etoiles")
items = ex.right_click_block(et, _("Starless to Post"))
wait(ex, 0.6)
log("F etoiles menu", [i.get("text") for i in items], u.toasts[-1:], layers())
ex.dialog_handler = name_dialog("M31 final")
ex.right_click_block(u.blk("etoiles"), _("Rename"))
wait(ex, 0.6)
log("F renamed", [q.name for q in ex.s.paths], layers())
# a wire to Post removed (the starless one if any, else the last)
from astro_gpu.client.graph import WireItem  # noqa: E402
ws = [x for x in ex.view.scene().items() if isinstance(x, WireItem)]
log("F wires", [(x.pid, x.part, x.toolTip().split(chr(10))[0]) for x in ws])
if len(ws) > 1:
    wv = next((x for x in ws if x.part is not None), ws[0])
    p = ex.view.mapFromScene(wv.path().pointAtPercent(0.5))
    ex.click(ex.view.viewport(), p)
    ex.key(Qt.Key_Delete)
    wait(ex, 0.6)
    log("F wire removed", u.toasts[-1:], layers())
ex.click_block(u.blk("composition"))
wait(ex, 0.3)
ex.shot(f"F_{rep}_built")
before, saved, after, diffs = u.roundtrip(root, "Complet jusqu'au bout", ["Complet", "Recombiner réglé", "renommé"])
ex.shot(f"F_{rep}_reopened")
log("F layers after reopen", layers(), after["wires"])
ex.click_block(u.blk("composition"))
wait(ex, 0.4)
cb = next((c for c in panel(ex).findChildren(QComboBox) if c.isVisible() and c.count() == 3), None)
sl = next((x for x in panel(ex).findChildren(ContSlider) if x.isVisible()), None)
log("F recombine panel after reopen", cb.currentText() if cb else None, sl.value() if sl and hasattr(sl, "value")
    else None)
ex.shot(f"F_{rep}_recombine_panel_reopened")
from explore5_usages_common import ptexts  # noqa: E402
ex.click_block(u.blk("final"))
wait(ex, 0.5)
log("F post panel", ptexts(ex)[:25])
ex.shot(f"F_{rep}_post_panel_reopened")
ex.w.set_mode(2)
wait(ex, 2.0)
from explore3_sources_common import texts as wtexts  # noqa: E402
pt = [x for x in wtexts(ex.w.post_holder) if "·" in x][:20] if ex.w.post_holder is not None else None
log("F post page", pt)
ex.shot(f"F_{rep}_post_page_reopened")
(ex.out / f"F_{rep}.json").write_text(json.dumps({"saved": saved, "after": after, "diffs": diffs},
                                                  ensure_ascii=False, indent=1, default=str), encoding="utf-8")
log("EXC", len(EXC), "\n".join(e[-1200:] for e in EXC))
ex.w.close()
