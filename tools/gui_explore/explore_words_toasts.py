"""Words, probe: the short messages of refused or done actions (delete a needed block, the last session, switch off
where it cannot, a Fusion without filters...), and the Log lines they leave, in one language.

    python explore_words_toasts.py fr
"""
from __future__ import annotations

import faulthandler
import json
import sys

faulthandler.dump_traceback_later(600, exit=True)
from harness import Explorer  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from astro_gpu.client import graph as graphmod  # noqa: E402
from astro_gpu.client import i18n  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
ex = Explorer("words_layout", lang=LANG, size=(1920, 1080))
said = []
_orig = graphmod.GraphView.show_toast
graphmod.GraphView.show_toast = lambda self, t: (said.append(str(t)), _orig(self, t))[1]
out = {}


def act(tag, fn):
    n = len(said)
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        said.append(f"EXCEPTION {type(e).__name__}: {e}")
    ex.settle(6)
    out[tag] = {"toasts": said[n:], "popups": [list(p) for p in ex.popups]}
    ex.popups.clear()
    print(tag, out[tag], flush=True)


def select_and_del(key):
    it = next((x for x in ex.blocks() if x.node.key == key), None)
    if it is None:
        raise LookupError(key)
    ex.click_block(it)
    ex.page.remove_tool(ex.page.current_node(), None)


root = ex.project("legacy")
ex.open(root)
ex.settle(8)
act("delete astrometry (Color needs it)", lambda: select_and_del("astrometrie"))
act("delete stacking", lambda: select_and_del("empilement"))
act("D on stacking (cannot switch off)", lambda: (ex.click_block(ex.block("empilement")), ex.page.toggle_cut(ex.page.current_node(), None)))
act("D on gradient", lambda: (ex.click_block(ex.block("fond")), ex.page.toggle_cut(ex.page.current_node(), None)))
act("D on gradient again", lambda: (ex.click_block(ex.block("fond")), ex.page.toggle_cut(ex.page.current_node(), None)))
act("delete the session (last one)", lambda: select_and_del("session") if ex.block("session") else select_and_del("sources"))
act("fusion without filters", lambda: ex.page.place_fusion())
act("duplicate", lambda: ex.right_click_block(ex.block("etoiles"), i18n._("Duplicate")))
act("save as template of the fusion", lambda: ex.page.save_as_template(ex.s.paths[-1]))
act("undo", ex.undo)
act("template of a fusion", lambda: None)
ex.w.b_log.setChecked(True)
ex.settle(3)
out["log"] = [ex.w.logp.table.item(r, 1).text() for r in range(ex.w.logp.table.rowCount())]
print("LOG", out["log"])
ex.shot(f"toasts_{LANG}_log")
root = ex.project("two_sessions")
ex.open(root)
ex.settle(8)
act("remove session 2", lambda: ex.right_click_block(ex.block("session", sid=2), i18n._("Delete")))
act("undo", ex.undo)
act("place stacking on blank flow", lambda: ex.double_click_card("stacking"))
root = ex.project("blank")
ex.open(root)
ex.settle(8)
act("calibration on blank", lambda: ex.page.place_block("calibration"))
(ex.out / f"toasts_{LANG}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
ex.close()
print("DONE")
