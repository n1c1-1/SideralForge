"""Side-by-side sheet of one or several runs at the same places (100 %, same stretch per row).

python experiments/own_starnet/compare_runs.py runA,runE [--project <folder>] [--out <dir>] [--at x,y[,half[,label]]]
-> <out>/compare_<runs>.jpg : rows = the bright evaluation stars, the structured hold-out box, the first medium
   evaluation star; columns = input | sl_final | each run.
   With --at (repeatable): rows = those places instead (nebula stretch) -> <out>/compare_<runs>_at.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import own_starnet as o  # noqa: E402


def places(meta):
    nbe = meta.get("n_bright_eval", 4)
    out = [(f"G{s['g']:.1f} star ({s['x']:.0f},{s['y']:.0f})", s["x"], s["y"], 200, False)
           for s in meta["eval_stars"][:nbe]]
    x0, y0, x1, y1 = o.boxes(meta)[0]
    out.append((f"structured hold-out box centre ({(x0 + x1) // 2},{(y0 + y1) // 2})", (x0 + x1) / 2, (y0 + y1) / 2,
                200, True))
    if len(meta["eval_stars"]) > nbe:
        s = meta["eval_stars"][nbe]
        out.append((f"G{s['g']:.1f} star ({s['x']:.0f},{s['y']:.0f})", s["x"], s["y"], 128, False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--project", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--at", action="append", default=[],
                    help="extra place x,y[,half[,label]] (nebula stretch); the sheet is then <out>/compare_<runs>_at.jpg")
    args = ap.parse_args()
    o.set_paths(args.project, args.out)
    runs = args.runs.split(",")
    meta = json.load(open(o.PREP / "meta.json"))
    L = np.load(o.PROJ / "optics/lat_full.npy", mmap_mode="r")
    S = np.load(o.PROJ / "starless/sl_final.npy", mmap_mode="r")
    nets = [np.load(o.OUT / r / "starless_net.npy", mmap_mode="r") for r in runs]
    sg = np.array(meta["sg"])
    H, W = L.shape[1:]
    pl = places(meta)
    if args.at:
        pl = []
        for a in args.at:
            v = a.split(",")
            pl.append((v[3] if len(v) > 3 else f"({v[0]},{v[1]})", float(v[0]), float(v[1]),
                       int(v[2]) if len(v) > 2 else 200, True))
    rows = []
    width = None
    for lab, x, y, h, neb in pl:
        x, y = int(round(min(max(x, h), W - h))), int(round(min(max(y, h), H - h)))
        sl = (slice(None), slice(y - h, y + h), slice(x - h, x + h))
        s_ = np.asarray(S[sl], np.float32)
        yy, xx = np.mgrid[-h:h, -h:h]
        rr = np.hypot(yy, xx)
        m = np.array([np.median(s_[c][rr > 0.8 * h]) for c in range(3)])
        top, base = (300.0, 0.4) if neb else (3000.0, 0.25)
        panels = [o.to_bgr(o.stretch(np.asarray(a[sl], np.float32), m, sg, top=top, base=base))
                  for a in [L, S] + nets]
        zoom = h < 200 and not args.at
        if zoom:
            panels = [cv2.resize(p, (400, 400), interpolation=cv2.INTER_NEAREST) for p in panels]
        sep = np.full((panels[0].shape[0], 4, 3), 255, np.uint8)
        img = panels[0]
        for p in panels[1:]:
            img = np.concatenate([img, sep, p], 1)
        rows.append(o.label(img, f"{lab}{' (x1.56)' if zoom else ''}: input | sl_final | " + " | ".join(runs)))
        width = max(width or 0, img.shape[1])
    out = o.OUT / f"compare_{'_'.join(runs)}{'_at' if args.at else ''}.jpg"
    cv2.imwrite(str(out), np.concatenate([np.pad(r, ((0, 4), (0, width - r.shape[1]), (0, 0))) for r in rows], 0),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out)


if __name__ == "__main__":
    main()
