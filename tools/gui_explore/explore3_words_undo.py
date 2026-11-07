"""Words, round 2: the short messages each gesture leaves, then Ctrl+Z and Ctrl+Y after it (what they say, and that the
graph comes back exactly): switch off, delete, place twice, branch, send to Post, remove a session, rename. Prints
and writes <out>/undo_<lang>.json.

    python explore2_words_undo.py fr
"""
from __future__ import annotations

import faulthandler
import json
import sys

faulthandler.dump_traceback_later(600, exit=True)
from harness import Explorer, stray_qt_errors  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import graph as graphmod  # noqa: E402
from astro_gpu.client import i18n  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1366
H = int(sys.argv[3]) if len(sys.argv) > 3 else 768
T = i18n._
stray_qt_errors()
ex = Explorer("words_layout", lang=LANG, size=(W, H))
SAID: list[str] = []
_st = graphmod.GraphView.show_toast
graphmod.GraphView.show_toast = lambda self, t: (SAID.append(str(t)), _st(self, t))[1]
_fl = appmod.Main.flash
appmod.Main.flash = lambda self, t: (SAID.append("[barre] " + str(t)), _fl(self, t))[1]
OUT = {}


def state():
    return sorted((it.title, it.node.key, bool(it.node.cut)) for it in ex.blocks())


def blk(key, nth=0, sid=None):
    its = sorted([x for x in ex.blocks() if x.node.key.split("~")[0] == key and (sid is None or x.node.sid == sid)],
                 key=lambda x: x.pos().x())
    return its[nth] if len(its) > nth else None


def gesture(tag, fn):
    """The gesture, then Ctrl+Z, then Ctrl+Y: what each says, and whether the graph is back to before / after."""
    before = state()
    n = len(SAID)
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        SAID.append(f"EXCEPTION {type(e).__name__}: {e}")
    ex.settle(6)
    after = state()
    said_do = SAID[n:]
    n = len(SAID)
    ex.undo()
    ex.settle(6)
    undone = state()
    said_undo = SAID[n:]
    n = len(SAID)
    ex.redo()
    ex.settle(6)
    redone = state()
    said_redo = SAID[n:]
    rec = {"do": said_do, "undo": said_undo, "redo": said_redo, "changed": before != after,
           "undo_ok": undone == before, "redo_ok": redone == after, "popups": [list(p) for p in ex.popups],
           "status": ex.w.statusBar().currentMessage() if ex.w.statusBar() else ""}
    if not rec["undo_ok"]:
        rec["diff_undo"] = [list(x) for x in set(undone) ^ set(before)]
    if not rec["redo_ok"]:
        rec["diff_redo"] = [list(x) for x in set(redone) ^ set(after)]
    ex.popups.clear()
    OUT[tag] = rec
    print(tag, json.dumps(rec, ensure_ascii=False), flush=True)


root = ex.project("two_sessions", f"Undo {LANG}")
ex.open(root)
ex.settle(8)
ex.click_block(blk("session", sid=1))
for k in ("calibration", "registration", "stacking", "astrometry", "gradient", "color", "denoise", "stars"):
    ex.double_click_card(k)
    ex.settle(4)
gesture("place gradient 2", lambda: (ex.click_block(blk("couleur")), ex.double_click_card("gradient")))
gesture("switch off gradient 2", lambda: ex.right_click_block(blk("fond", 1), T("Switch off")))
gesture("D on gradient 1", lambda: (ex.click_block(blk("fond", 0)), ex.key(Qt.Key_D)))
gesture("delete denoise (Del)", lambda: (ex.click_block(blk("debruitage")), ex.key(Qt.Key_Delete)))
gesture("delete astrometry (refused)", lambda: ex.right_click_block(blk("astrometrie"), T("Delete")))
gesture("stacking to Post", lambda: ex.right_click_block(blk("empilement"), T("To Post")))
gesture("starless to Post", lambda: ex.right_click_block(blk("etoiles"), T("Starless to Post")))
gesture("recombine", lambda: ex.right_click_block(blk("etoiles"), T("+ Recombine")))
gesture("pause", lambda: ex.right_click_block(blk("couleur"), T("Review pause")))
gesture("delete session 2", lambda: ex.right_click_block(blk("session", sid=2), T("Delete")))


def branch():
    ex.drag_palette_hover("denoise", ex.view.viewport().rect().center())
    br = [g for g in ex.ghosts() if getattr(g, "branch_key", None)]
    pt = ex.ghost_center(br[0]) if br else None
    ex.drag_leave()
    if pt is not None:
        ex.drag_palette("denoise", pt)


gesture("branch denoise", branch)
gesture("drag gradient 1 onto the palette", lambda: ex.drag_block_to_palette(blk("fond", 0)))
ex.w.b_log.setChecked(True)
ex.settle(3)
t = ex.w.logp.table
OUT["_log"] = [t.item(r, 1).text() for r in range(t.rowCount()) if t.item(r, 1)]
ex.shot(f"undo_{LANG}_log")
(ex.out / f"undo_{LANG}.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
ex.close()
print("DONE")
