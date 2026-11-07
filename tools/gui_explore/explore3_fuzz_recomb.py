"""Round 3 (control) fuzz: the round-2 point « two Recombine blocks drawn on top of each other » replayed by hand."""
import os
import sys
import traceback
from pathlib import Path
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_blocks_lib import TOASTS, overlaps  # noqa: E402
import explore3_fuzz as G  # noqa: E402
fz = G.Fuzz2("en")
fz.panels = False
ex = fz.ex
sys.excepthook = sys.__excepthook__
def blk(pred):
    return next((x for x in fz.items() if pred(x)), None)
try:
    for k in range(2):
        fz.start({"template": "classic_dn", "lang": "en"}, f"ovc_{k}")
        fz.activate()
        ex.double_click_card("fusion"); fz.wait_ready()
        fu = blk(lambda x: x.node.key == "fusion"); print(k, "after Fusion:", fu.summary if fu else None, TOASTS[-1:], [fz.title(x) for x in fz.items()][-5:], flush=True)
        ex.drag_palette_hover("composition", ex.view.viewport().rect().center())
        gh = [(type(g).__name__, getattr(g, "text", "")) for g in ex.ghosts()]
        cand = [ex.ghost_center(g) for g in ex.ghosts()]
        ex.drag_leave()
        print("   ghosts for Recombine:", gh, flush=True)
        if cand:
            ex.drag_palette("composition", cand[0]); fz.wait_ready()
        print("   after drop:", TOASTS[-1:], [fz.title(x) for x in fz.items() if x.node.key == "composition"], flush=True)
        m = ex.right_click_block(blk(lambda x: x.node.key == "etoiles"), pick="+ Recombine"); fz.wait_ready()
        print("   menu:", [x.get("text") for x in m], "| comps", [fz.title(x) for x in fz.items() if x.node.key == "composition"],
              "| overlaps", overlaps(ex), "| toasts", TOASTS[-1:], flush=True)
        if k == 1:
            ex.shot("r3_two_recombine")
except Exception:
    traceback.print_exc()
os._exit(0)
