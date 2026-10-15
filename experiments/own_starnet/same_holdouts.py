"""Give a new prep the hold-out boxes of an earlier one (so that two runs are measured on the same regions): the
earlier boxes become the hold-out boxes of the new meta.json, and the weight map excludes both its own boxes and
the earlier ones from the training targets.
python same_holdouts.py <earlier prep dir> <new prep dir>"""
import json
import sys
from pathlib import Path

import numpy as np

old, new = Path(sys.argv[1]), Path(sys.argv[2])
mo, mn = json.load(open(old / "meta.json")), json.load(open(new / "meta.json"))
if "struct_box" not in mo:                  # the first experiment (Iris) named its structured box after the nebula
    mo["struct_box"] = mo["iris_box"]
wm = np.load(new / "weight_map.npy")
for key in ("struct_box", "faint_box"):
    for b in (mo[key], mn[key]):
        x0, y0, x1, y1 = b
        wm[y0:y1, x0:x1] = 0
print("new boxes", mn["struct_box"], mn["faint_box"], "-> earlier", mo["struct_box"], mo["faint_box"])
mn["struct_box"], mn["faint_box"] = mo["struct_box"], mo["faint_box"]
mn["eval_stars"], mn["n_bright_eval"] = mo["eval_stars"], mo.get("n_bright_eval", 4)
np.save(new / "weight_map.npy", wm)
json.dump(mn, open(new / "meta.json", "w"), indent=1)
