"""Round 3 (control) fuzz: blocks drawn on top of each other after Vers Post / a port pulled to Post in a Fusion
project; and the round-2 Recombine overlap (classic_dn, en), twice each."""
import os
import sys
from pathlib import Path
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_blocks_lib import TOASTS, overlaps  # noqa: E402
import explore3_fuzz as G  # noqa: E402
fz = G.Fuzz2("fr")
fz.panels = False
ex = fz.ex
def blk(pred):
    return next((x for x in fz.items() if pred(x)), None)
for k in range(2):
    for tpl, steps in (("hoo", ("menu_astro", "pull_grad")), ("hoo", ("pull_grad",)), ("hoo", ("menu_astro",)),
                       ("sho", ("menu_grad",)), ("classic_dn", ("pull_grad",))):
        fz.start({"template": tpl, "lang": "fr"}, f"ov_{tpl}_{k}_{len(steps)}")
        fz.activate()
        for st in steps:
            if st == "menu_astro":
                ex.right_click_block(blk(lambda x: x.node.key == "astrometrie"), pick="Vers Post")
            elif st == "menu_grad":
                ex.right_click_block(blk(lambda x: x.node.key == "fond"), pick="Vers Post")
            else:
                ex.pull_port_to_post(blk(lambda x: x.node.key == "fond"))
            fz.wait_ready()
        print(k, tpl, steps, "| toasts", TOASTS[-len(steps):], "| workflows", [w.name for w in ex.s.p.workflows],
              "| overlaps", overlaps(ex)[:4], flush=True)
        if k == 1 and steps == ("menu_astro", "pull_grad"):
            ex.shot("r3_overlap_hoo")
    # round-2 Recombine overlap
    fz.start({"template": "classic_dn", "lang": "en"}, f"ovc_{k}")
    fz.activate()
    ex.double_click_card("fusion"); fz.wait_ready()
    ex.drag_palette_hover("composition", ex.view.viewport().rect().center())
    gh = [g for g in ex.ghosts() if getattr(g, "block", None) == "composition" or "ecomb" in str(getattr(g, "text", ""))]
    ex.drag_leave()
    if gh:
        ex.drag_palette("composition", ex.ghost_center(gh[0])); fz.wait_ready()
    ex.right_click_block(blk(lambda x: x.node.key == "etoiles"), pick="+ Recombine"); fz.wait_ready()
    print(k, "classic_dn en Fusion+Recombine+«+ Recombine»: ghost found", bool(gh), "| comps",
          [fz.title(x) for x in fz.items() if x.node.key == "composition"], "| overlaps", overlaps(ex), flush=True)
os._exit(0)
