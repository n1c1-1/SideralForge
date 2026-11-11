"""Sharp images of the deconvolution network from real stacks: each pair of half-stacks denoised, then reduced.

    python tools/bench/deconv_truth_stacks.py [--out E:/float_train/truth_stacks]

Reduced by 2 and by 3, a star of 3 px becomes a star of 1 to 1.5 px, the width the network aims at; the noise goes down
with it. Each colour is one image. The instruments and targets of the validation are left out (``HELD_OUT``), and
the shallower versions of a stack (LEVELS) too: they are the same sky.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astro_gpu.stages import denoise as dn  # noqa: E402

PAIRS = Path("F:/astro_essais/denoise_dataset")
LEVELS = ("_half", "_quarter", "_eighth")
#: never seen by the training: the targets the benches judge on, and whole instruments
HELD_OUT = ("bench_", "set_02_", "set_03_", "user_Heart", "user_Iris", "user_Rosette", "set_18_", "set_33_", "set_37_", "set_38_")
REDUCTIONS = (2, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("E:/float_train/truth_stacks"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    net = dn.pretrained(dev)
    net.eval()
    names = [d for d in sorted(PAIRS.iterdir()) if (d / "pair.json").exists() and not d.name.endswith(LEVELS)
             and not d.name.startswith(HELD_OUT)]
    print(len(names), "stacks", flush=True)
    bk, sg = np.zeros(3, np.float32), np.ones(3, np.float32)
    for d in names:
        if (a.out / f"{d.name}_2_b{REDUCTIONS[-1]}.npy").exists():
            continue
        t0 = time.time()
        x = 0.5 * (np.load(d / "A.npy").astype(np.float32) + np.load(d / "B.npy").astype(np.float32))
        nm = np.load(d / "nm.npy").astype(np.float32)
        den = dn.apply_net(net, x, (nm / np.sqrt(2)).astype(np.float32), bk, sg, dev)
        valid = (nm > 0).astype(np.uint8)
        H, W = valid.shape
        m = cv2.erode(valid, np.ones((65, 65))) > 0
        rows = np.nonzero(m.any(1))[0]
        cols = np.nonzero(m.any(0))[0]
        y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        for c in range(3):
            plane = den[c, y0:y1, x0:x1]
            for r in REDUCTIONS:
                np.save(a.out / f"{d.name}_{c}_b{r}.npy", cv2.resize(plane, None, fx=1 / r, fy=1 / r, interpolation=cv2.INTER_AREA).astype(np.float32))
        print(d.name, H, W, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
