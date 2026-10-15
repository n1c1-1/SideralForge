"""Markdown table of the profiles: per star, the lower quarter of the data minus the model (pixel noise sigmas) over
80-250 px and 250-600 px, old and new halo, per channel; the lowest factor of the bound.
python summary.py <dir>[,<dir>...] > table.md"""
import json
import sys
from pathlib import Path

import numpy as np

D = 4                      # the profiles are in 4 x 4 block sigmas


def band(rings, lo, hi, key="p25"):
    v = [[(s[key][c] - s["h"][c]) for c in range(3)] for s in rings if s is not None and lo <= s["r"] < hi]
    return np.array(v) if v else np.zeros((0, 3))


def main():
    print("| project | star | halo | data 25 % - model, 80-250 px (R G B, mean / lowest ring) | 250-600 px (mean) |"
          " median - model 80-250 px (G) |")
    print("|---|---|---|---|---|---|")
    for d in sys.argv[1].split(","):
        d = Path(d)
        js = sorted(d.glob("profiles_old_new.json"))
        if not js:
            continue
        r = json.load(open(js[0]))
        # block noise of each project: the profiles store levels in ADU; recompute the scale from the stored excess
        for name, v in r.items():
            out = []
            for t in ("old", "new"):
                rings = v[t]["rings"]
                ex = v[t]["excess_p25_sigma"]       # (h - p25) / block sigma per ring
                # scale ADU -> pixel sigma from one ring with a non-zero difference
                sc = None
                for s, (_, e) in zip([s for s in rings if s is not None], ex):
                    dd = s["h"][1] - s["p25"][1]
                    if abs(dd) > 1e-6 and abs(e[1]) > 1e-6:
                        sc = e[1] / dd / D
                        break
                if sc is None:
                    out.append(None)
                    continue
                a = band(rings, 80, 250) * sc
                b = band(rings, 250, 600) * sc
                m = band(rings, 80, 250, "p50") * sc
                out.append((a, b, m))
            if out[0] is None or out[1] is None or not len(out[0][0]):
                continue
            own = v["old"]["own"]

            def fmt(x):
                return " ".join(f"{x[c]:+.1f}" for c in range(3))
            for t, (a, b, m) in zip(("old", "new"), out):
                print(f"| {d.name} | {name.split('_')[0]} ({name.split('_')[1]},{name.split('_')[2]}), {own} | {t} | "
                      f"{fmt(a.mean(0))} / {fmt(a.min(0))} | {fmt(b.mean(0)) if len(b) else '-'} | "
                      f"{m[:, 1].mean():+.1f} |")


if __name__ == "__main__":
    main()
