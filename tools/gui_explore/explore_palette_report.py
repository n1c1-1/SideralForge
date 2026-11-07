"""palette_matrix: the raw findings of all the states, grouped by signature (to read before the confirm pass)."""
import glob
import json
import os
import sys
from collections import defaultdict

from explore_palette_lib import SCRATCH

skip = sys.argv[1:]
by = defaultdict(list)
for f in glob.glob(str(SCRATCH / "*.jsonl")):
    if os.path.basename(f).startswith(("_", "confirm", "layout")) and "layout" not in skip:
        pass
    for line in open(f, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            by[r["sig"]].append(r)
for sig, rs in sorted(by.items(), key=lambda x: -len(x[1])):
    if any(sig.startswith(s) for s in skip):
        continue
    tags = sorted({r["tag"] for r in rs})
    print(f"## {sig}  x{len(rs)}  states={tags}")
    print("   e.g.", rs[0]["case"], "| shot:", os.path.basename(rs[0].get("shot", "")))
    print("   ", rs[0]["detail"][:700].replace("\n", "\n    "))
