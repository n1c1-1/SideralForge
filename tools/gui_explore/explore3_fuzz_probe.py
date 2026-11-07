"""Round 3 (control) fuzz: hand replays of two new fuzz signatures (seeds 7017, 7025), twice."""
import os
import sys
import traceback
from pathlib import Path
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_blocks_lib import TOASTS, overlaps  # noqa: E402
import explore3_fuzz as G  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from astro_gpu.client.graph import WireItem  # noqa: E402
fz = G.Fuzz2("fr")
fz.panels = False
ex = fz.ex
def blk(pred):
    return next((x for x in fz.items() if pred(x)), None)
for k in range(2):
    try:
        fz.start({"template": "image_plain", "lang": "fr"}, f"p17_{k}")
        fz.activate()
        w = [x for x in ex.view.scene().items() if isinstance(x, WireItem) and x.isVisible()][0]
        ex.right_click_block  # noqa
        pt = ex.view.mapFromScene(w.path().pointAtPercent(0.5))
        ex.hover(ex.view.viewport(), pt); ex.click(ex.view.viewport(), pt); fz.activate(); ex.key(Qt.Key_Delete); fz.wait_ready()
        n = len(TOASTS)
        ex.pull_port_to_post(blk(lambda x: x.node.key == "image")); fz.wait_ready()
        print(k, "7017 image_plain: wire removed then Image port pulled:", TOASTS[n:], "| workflows",
              [w.name for w in ex.s.p.workflows], "| images", [fz.title(x) for x in fz.items() if x.node.key == "image"],
              "| overlaps", overlaps(ex), flush=True)
        if k == 1:
            ex.shot("r3_image_port_overlap")
    except Exception:
        print(traceback.format_exc()[-800:], flush=True)
    for tpl, lang in (("rgb", "en"), ("hoo", "fr")):
        try:
            fz.start({"template": tpl, "lang": lang}, f"p25_{tpl}_{k}")
            fz.activate()
            fz.errs.clear()
            ex.double_click_card("fusion"); fz.wait_ready()
            fz.undo(); fz.wait_ready()
            comp = blk(lambda x: x.node.key == "composition")
            vp = ex.view.viewport()
            pt = ex.view.mapFromScene(comp.mapToScene(comp.w - 12, 12))
            ex.hover(vp, pt); ex.click(vp, pt); fz.wait_ready()
            print(k, tpl, "Fusion, Ctrl+Z, cross on Recombiner:", TOASTS[-1:], "| errs",
                  [x.strip().splitlines()[-1] for x in fz.errs], flush=True)
        except Exception:
            print(traceback.format_exc()[-800:], flush=True)
os._exit(0)
