"""Round 3 (control) fuzz: the Fusion card on a colour project (no mono channel), twice, fr and en."""
import os
import sys
import traceback
from pathlib import Path
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_blocks_lib import TOASTS  # noqa: E402
import explore3_fuzz as G  # noqa: E402
fz = G.Fuzz2("fr")
fz.panels = False
ex = fz.ex
sys.excepthook = sys.__excepthook__
try:
    for k in range(2):
        for tpl in ("classic_dn", "full"):
            fz.start({"template": tpl, "lang": "fr"}, f"fc_{tpl}_{k}")
            c = ex.card("fusion")
            print(k, tpl, "Fusion card reason:", repr(getattr(c, "reason", None)), "| sessions",
                  [(s.name, s.filter) for s in ex.s.p.sessions], flush=True)
            ex.double_click_card("fusion"); fz.wait_ready()
            fu = next((x for x in fz.items() if x.node.key == "fusion"), None)
            w = ex.s.p.workflows[-1]
            print("   placed:", TOASTS[-1:], "| block", fu.summary if fu else None, "| tip",
                  repr(fu.toolTip()[-200:]) if fu else None, "| status", ex.w.status.text(), flush=True)
            print("   fusion def:", {kk: v for kk, v in vars(w).items() if kk in ("name", "fusion", "palette", "channels", "reads")}, flush=True)
            if k == 1 and tpl == "classic_dn":
                ex.shot("r3_fusion_on_colour")
except Exception:
    traceback.print_exc()
os._exit(0)
