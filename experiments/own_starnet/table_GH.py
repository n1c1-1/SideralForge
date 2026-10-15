"""Markdown tables of two (or more) runs evaluated in different output folders (eval metrics.json and
lost_light.py's JSON): the evaluation stars, the hold-out regions, the light lost at rendering.
python table_GH.py --runs name=<run dir>,name=<run dir> --lost name=<lost json>,... [--stars 3898,406;...]"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--lost", default="")
    ap.add_argument("--stars", default="", help="saturated stars to list (x,y;...), besides the 8 that lose most")
    a = ap.parse_args()
    runs = {}
    for spec in a.runs.split(","):
        n, d = spec.split("=", 1)
        runs[n] = json.load(open(Path(d) / "metrics.json"))
    names = list(runs)
    st0 = runs[names[0]]["stars"]
    print("Evaluation stars (ring 20-150 px around the star, smoothed; residue / holes: % of the ring 3 sigma above /"
          " below a local background fit; core: mean within 20 px, sigma)\n")
    print("| star | " + " | ".join(f"{n}: residue / holes %, core" for n in names) + " | model starless (" +
          names[-1] + " input) residue / holes % |")
    print("|---|" + "---|" * (len(names) + 1))
    for k, s0 in enumerate(st0):
        cells = []
        for n in names:
            s = runs[n]["stars"][k]["net"]
            cells.append(f"{s['residue_pct']} / {s['hole_pct']}, {s['core_mean_dev_sigma']:+.2f}")
        sl = runs[names[-1]]["stars"][k]["sl_final"]
        print(f"| G{s0['g']} ({s0['x']},{s0['y']}) | " + " | ".join(cells) + f" | {sl['residue_pct']} / "
              f"{sl['hole_pct']} |")
    print("\nHold-out regions (never training targets), star-free pixels: network minus model starless image, rms /"
          " mean (sigma, R G B)\n")
    print("| region | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for reg in ("structured_holdout", "faint_nebula_holdout"):
        cells = []
        for n in names:
            r = runs[n]["regions"][reg]
            cells.append(" ".join(f"{v:.2f}" for v in r["star_free_rms_net_minus_sl_sigma"]) + " / " +
                         " ".join(f"{v:+.2f}" for v in r["star_free_mean_net_minus_sl_sigma"]))
        print(f"| {reg.replace('_', ' ')} | " + " | ".join(cells) + " |")
    print("| reliable area, rms | " + " | ".join(" ".join(f"{v:.2f}" for v in runs[n]
                                                         ["reliable_area_rms_net_minus_sl_sigma"]) for n in names) + " |")
    if a.lost:
        lost = {}
        for spec in a.lost.split(","):
            n, f = spec.split("=", 1)
            lj = json.load(open(f))
            lost[n] = (lj["stars"], lj["totals"])
        print("\nLight lost at rendering around the saturated stars (the kept layer minus the star light shown in the "
              "footprint, faded below 1.5 sigma; ring medians: the stars around do not count)\n")
        tot = {n: v[1].get(n, list(v[1].values())[0]) for n, v in lost.items()}
        print("| | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
        print("| all saturated stars: lost / shown | " + " | ".join(f"{tot[n]['lost_pct']} % ({tot[n]['n_saturated']} stars)"
                                                         for n in names if n in tot) + " |")
        first = lost[names[0]][0]
        order = sorted(first, key=lambda k: -first[k].get(names[0], list(first[k].values())[0])["lost"])[:8]
        want = [s_.split(",") for s_ in a.stars.split(";")] if a.stars else []
        for k in first:
            if any(f"({w[0]},{w[1]})" in k for w in want) and k not in order:
                order.append(k)
        print("\n| star | " + " | ".join(f"{n}: lost % (share beyond the footprint), peak sigma, lost above "
                                         f"0.5 sigma between (px), footprint (px)" for n in names) + " |")
        print("|---|" + "---|" * len(names))
        for k in order:
            cells = []
            for n in names:
                d = lost[n][0].get(k)
                if d is None:
                    cells.append("-")
                    continue
                d = d.get(n, list(d.values())[0])
                rr = d["lost_above_half_sigma_px"]
                out_ = d["lost_beyond_footprint"] / max(d["lost"], 1e-9)
                cells.append(f"{d['lost_pct']} ({100 * out_:.0f} %), {d['peak_sigma']}, "
                             + (f"{rr[0]}-{rr[1]}" if rr[0] is not None else "-") + f", {d['footprint_px']}")
            print(f"| {k} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
