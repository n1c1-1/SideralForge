"""Round 2 fuzz: is every drawn block inside the visible graph after the window opens a project (no scrolling)?"""
import os
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402

fz = G.Fuzz2("fr")
ex = fz.ex
ex.open()
for size in ((1600, 950), (1366, 768), (1920, 1080)):
    ex.w.resize(*size)
    ex.settle(4)
    for k in range(2):
        for tpl in ("sho", "hoo", "classic_dn"):
            fz.start({"template": tpl, "lang": "fr"}, f"vis{size[0]}_{k}_{tpl}")
            ex.settle(6)
            vis = ex.view.mapToScene(ex.view.viewport().rect()).boundingRect()
            out = [fz.title(it) for it in fz.items() if not vis.contains(it.sceneBoundingRect())]
            print(size, k, tpl, "hors vue:", out, flush=True)
            if out and k == 1:
                ex.shot(f"offview_{tpl}_{size[0]}")
os._exit(0)
