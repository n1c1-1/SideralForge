"""Overview of what prep chose for a project: hold-out boxes (structured: cyan, faint: yellow), evaluation stars
(bright: red, medium: magenta), bright-star zones where the starless image is not trusted (dark red), on the starless
image at 1/8 scale.

python experiments/own_starnet/prep_overview.py [--project <folder>] [--out <dir>]  -> <out>/prep/overview.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import own_starnet as o  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    o.set_paths(args.project, args.out)
    meta = json.load(open(o.PREP / "meta.json"))
    f = 8
    S = np.load(o.PROJ / "starless/sl_final.npy", mmap_mode="r")
    L = np.load(o.PROJ / "optics/lat_full.npy", mmap_mode="r")
    WM = np.load(o.PREP / "weight_map.npy", mmap_mode="r")
    bk, sg = np.array(meta["bk"]), np.array(meta["sg"])
    imgs = []
    for a in (L, S):
        v = np.asarray(a[:, ::f, ::f], np.float32)
        v = np.arcsinh((v - bk[:, None, None]) / sg[:, None, None]) / np.arcsinh(3000)
        imgs.append(np.ascontiguousarray((np.clip(0.2 + 0.8 * v, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255)
                                         .astype(np.uint8)))
    wm = np.asarray(WM[::f, ::f]) == 0
    for im in imgs:
        im[wm] = (0.6 * im[wm] + 0.4 * np.array([0, 0, 120])).astype(np.uint8)
        for (x0, y0, x1, y1), c in zip(o.boxes(meta), ((255, 255, 0), (0, 255, 255))):
            cv2.rectangle(im, (x0 // f, y0 // f), (x1 // f, y1 // f), c, 2)
        nbe = meta.get("n_bright_eval", 4)
        for k, s in enumerate(meta["eval_stars"]):
            c = (0, 0, 255) if k < nbe else (255, 0, 255)
            cv2.circle(im, (int(s["x"] / f), int(s["y"] / f)), 12, c, 2)
            cv2.putText(im, f"G{s['g']:.1f}", (int(s["x"] / f) + 14, int(s["y"] / f)), 0, 0.45, c, 1)
    out = np.concatenate([imgs[0], np.full((imgs[0].shape[0], 6, 3), 255, np.uint8), imgs[1]], 1)
    cv2.imwrite(str(o.PREP / "overview.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(o.PREP / "overview.jpg")


if __name__ == "__main__":
    main()
