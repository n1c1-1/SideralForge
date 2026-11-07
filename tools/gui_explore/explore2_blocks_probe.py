"""block_actions round 2: dump the context menus and panel buttons of every drawn block in typical states."""
from __future__ import annotations

import faulthandler
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(600, exit=True)

from explore_blocks_lib import EXC, TOASTS, drawn, panel_info, state  # noqa: E402
from harness import Explorer  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
ex = Explorer("block_actions", lang=LANG)
CHAIN = ["calibration", "registration", "selection", "stacking", "astrometry", "gradient", "denoise", "color", "stars"]


def act():
    ex.w.activateWindow()
    ex.settle(3)


def new(kind, chain=()):
    if ex.w is not None and ex.w.s is not None:
        ex.w.s._dirty = False
    ex.open(ex.project(kind))
    act()
    for b in chain:
        ex.double_click_card(b)
        ex.settle(4)
    act()


def dump(label):
    print("=====", label, flush=True)
    def items():
        return sorted(ex.blocks(), key=lambda i: (i.pos().x(), i.pos().y()))
    for k in range(len(items())):
        it = items()[k]
        m = ex.right_click_block(it)
        ex.settle(3)
        act()
        it = items()[k]
        ex.click_block(it)
        it = items()[k]
        pb = [(b["text"], b["enabled"], b["tip"][:60]) for b in panel_info(ex)["buttons"]]
        print(f"[{it.node.key} sid={it.node.sid} '{it.title}' rem={it.removable}]")
        print("   menu:", json.dumps([(x.get("text"), x.get("enabled")) for x in m], ensure_ascii=False))
        print("   panel:", json.dumps(pb, ensure_ascii=False))
    print("   toasts:", TOASTS[-5:], "exc:", len(EXC), flush=True)


new("session", CHAIN)
dump("session chain")
new("two_sessions", ["calibration", "registration", "stacking"])
dump("two sessions")
new("mono")
for name in ("Ha", "OIII"):
    pid = next(q.pid for q in ex.s.paths if q.name == name)
    ex.click_block(ex.block("empilement", pid=pid))
    ex.double_click_card("astrometry")
    ex.settle(4)
ex.double_click_card("fusion")
ex.settle(4)
dump("mono fusion")
new("image", ["astrometry", "gradient"])
dump("image")
for e in EXC:
    print("EXC", e[-600:])
ex.close()
