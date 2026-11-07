"""Round 2 fuzz: every enabled item of every block's right-click menu, each on a fresh project, with all the fuzz's
checks (Python errors, valid project, undo/redo, pop-ups, drawn graph, Post layers, panels). An item already played
for the same block kind and text in an earlier template is skipped.

    python explore2_fuzz_menus.py [templates...]
"""
import json
import os
import re
import sys
from pathlib import Path

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402

OUT = Path("F:/astro_essais/A_REGARDER/exploration2/fuzz/menu_sweep.jsonl")
fz = G.Fuzz2("fr")
fz.panels = False
ex = fz.ex
STARTS = [{"template": t, "lang": "fr"} for t in (ARGS or ["lrgb", "classic_dn", "image_colours"])] + \
    ([] if ARGS else [{"kind": "two_flows", "lang": "fr"}, {"kind": "mono", "lang": "fr"},
                      {"kind": "session+image", "lang": "fr"}])
done = set()
n = 0
for st in STARTS:
    tag0 = (st.get("template") or st.get("kind")).replace("+", "_")
    fz.start(st, f"ms_{tag0}_0")
    plan = []
    for i in range(len(fz.items())):
        it = fz.items()[i]
        title, key = fz.title(it), it.node.key
        leaves = G.F._leaves(ex.right_click_block(it))
        for txt in leaves:
            k = (re.sub(r"\d+", "N", key), re.sub(r"\d+", "N", txt))
            if k in done:
                continue
            done.add(k)
            plan.append((i, title, key, txt))
    print(st, len(plan), "items", flush=True)
    for i, title, key, txt in plan:
        n += 1
        res = fz.run({"seed": None, "start": st, "actions": [["menu", title, txt, 0.5]]}, f"ms_{tag0}_{n}",
                     watchdog=60)
        fails = [(f["sig"], f["detail"][:500]) for f in res["failures"]]
        rec = {"start": st, "block": title, "key": key, "item": txt, "steps": res["steps"], "fails": fails,
               "popups": ex.popups[-2:], "dialogs": list(fz.dialogs)}
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        if fails:
            print("FAIL", title, "→", txt, [f[0] for f in fails], flush=True)
print("played", n, flush=True)
os._exit(0)
