"""Why does the network dig a crater at the real brightest stars when it removes their pasted stamps cleanly?

1. tiling: the region around the star inferred with several tile sizes (one tile holding the whole star included);
2. real against synthetic: the real input, and its own training stamp (scale 1) pasted on the real surroundings with
   the star's area interpolated (membrane fill of the input), go through the same network.
Ring medians of the output (G channel, noise sigma, relative to the global sky) are printed.
python crater_check.py --project <snapshot> --out <dir> --run runF --star x,y [--half 704]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import own_starnet as o  # noqa: E402


def rings(img, cx, cy, bk, sg, edges=((40, 80), (80, 150), (150, 250), (250, 350), (350, 500))):
    yy, xx = np.mgrid[0:img.shape[1], 0:img.shape[2]]
    r = np.hypot(yy - cy, xx - cx)
    return [round(float((np.median(img[1][(r >= a) & (r < b)]) - bk[1]) / sg[1]), 1) for a, b in edges]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project")
    ap.add_argument("--out")
    ap.add_argument("--run", default="runF")
    ap.add_argument("--star")
    ap.add_argument("--half", type=int, default=704)
    args = ap.parse_args()
    o.set_paths(args.project, args.out)
    meta = json.load(open(o.PREP / "meta.json"))
    o.set_radius(meta)
    dev = torch.device("cuda")
    net, ck = o.load_net(o.OUT / args.run, dev)
    bk_t, sg_t = torch.tensor(meta["bk"], device=dev), torch.tensor(meta["sg"], device=dev)
    bk, sg = np.array(meta["bk"]), np.array(meta["sg"])
    sx, sy = (float(v) for v in args.star.split(","))
    h = args.half
    x0, y0 = int(round(sx)) - h, int(round(sy)) - h
    L = np.load(o.PROJ / "optics/lat_full.npy", mmap_mode="r")
    Hh = np.load(o.PROJ / "stars/stars_model_halo.npy", mmap_mode="r")
    H, W = L.shape[1:]
    ys, xs = slice(max(0, y0), min(H, y0 + 2 * h)), slice(max(0, x0), min(W, x0 + 2 * h))
    inp = np.asarray(L[:, ys, xs], np.float32) - np.asarray(Hh[:, ys, xs], np.float32)
    cx, cy = sx - xs.start, sy - ys.start
    print("input ring medians (40-80, 80-150, 150-250, 250-350, 350-500 px):", rings(inp, cx, cy, bk, sg))
    full = np.load(o.OUT / args.run / "starless_net.npy", mmap_mode="r")
    print("full-frame output (512 px tiles)            :", rings(np.asarray(full[:, ys, xs], np.float32), cx, cy, bk, sg))
    for tile, ov in ((512, 128), (768, 192), (1024, 256), (2 * h // 32 * 32, 0)):
        out = o.apply_tiled(net, inp, bk_t, sg_t, dev, tile=tile, ov=max(ov, 32), batch=1)
        print(f"tiles {tile:4d} px, overlap {ov:3d}                   :", rings(out, cx, cy, bk, sg))
    # its own training stamp pasted on the real surroundings, the star's area interpolated
    from astro_gpu.core.imgutil import membrane_fill
    lib = torch.load(o.PREP / o.lib_name(True, ck.get("stamps", "")), weights_only=False)["bright"]
    j = int(np.argmin([math.hypot(b["x"] - sx, b["y"] - sy) for b in lib]))
    b = lib[j]
    R = b["R"]
    yy, xx = np.mgrid[0:inp.shape[1], 0:inp.shape[2]]
    hole = np.hypot(yy - cy, xx - cx) < R
    bgd = np.stack([membrane_fill(inp[c], hole) for c in range(3)])
    syn = torch.from_numpy(bgd.copy()).to(dev)
    clip = np.where(b["sat"], b["plateau"], np.array(meta["sat99"]) * 1.15)
    synth = o.Synth(meta, {"bright": [b], "faint": []}, dev)
    torch.manual_seed(0)
    synth.apply(syn, [("b", 0, int(round(cy)), int(round(cx)), 1.0, np.ones(3), clip)])
    syn = syn.cpu().numpy()
    out_s = o.apply_tiled(net, syn, bk_t, sg_t, dev, batch=1)
    print("synthetic input (stamp on interpolated surroundings):", rings(syn, cx, cy, bk, sg))
    print("synthetic: interpolated background (the truth)      :", rings(bgd, cx, cy, bk, sg))
    print("synthetic: network output                           :", rings(out_s, cx, cy, bk, sg))


if __name__ == "__main__":
    main()
