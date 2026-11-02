"""Redo the target step of a measure (the SNR on the target) from the saved box maps, without reading the images.

    python tools/bench/retarget.py CASE today|drizzle [LEVEL]

``today``: F:/astro_essais/bench/<case>/today_express.json (two_methods.py measure) from target_maps_express.npz;
``drizzle``: <case>/drizzle_siril_express.json (siril/measure_drizzle.py) from target_maps_drizzle_express.npz.
Only the "target" entries of the JSON are rewritten (regions.py: the regions chosen on an image independent of the
stacks compared, every stack measured on the same regions).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402
import regions as RG  # noqa: E402


def retarget(case: str, kind: str, level: str = "express") -> dict:
    maps_f = C.BENCH / case / (f"target_maps_{level}.npz" if kind == "today" else f"target_maps_drizzle_{level}.npz")
    json_f = C.BENCH / case / ("today_express.json" if kind == "today" else f"drizzle_siril_{level}.json")
    maps, sel0, src = RG.load_maps(maps_f)
    ref = "Siril"
    reg, sel, src = RG.choose(maps, ref, sel0)
    res = json.loads(json_f.read_text(encoding="utf-8"))
    info = {"source": src}
    if reg is not None:
        info.update({"target_fraction": reg["fraction"], "level_sigma": reg["level_sigma"]})
    tm = {n: RG.on_regions(reg, sel, m) if reg is not None else {"target_boxes": 0} for n, m in maps.items()}
    for n, e in res["by_image"].items():
        t = tm[n]
        e["target"] = {"snr": t.get("snr"), "signal_sb": t.get("signal_sb"), "noise_sb": t.get("noise_sb"),
                       "boxes": t.get("target_boxes"), "sky_boxes": t.get("sky_boxes"),
                       "ref_snr": tm[ref].get("snr"), "regions": info}
    json_f.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return {n: tm[n].get("snr") for n in tm}


if __name__ == "__main__":
    for c in sys.argv[1].split(","):
        print(c, retarget(c, sys.argv[2], *(sys.argv[3:4])))
