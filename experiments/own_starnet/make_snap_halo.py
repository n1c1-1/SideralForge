"""A read-only project copy for the network runs with a corrected far halo (run H).

python make_snap_halo.py <halo2 project dir> <new snapshot dir> [--tag new]
<halo2 project dir> (D:/astro_gpu_diag/halo2/<project>) holds snapshot/work (copies of the project files) and
halo_<tag>.npy (the far halo + rays computed offline by experiments/halo2/halo_lab.py). The new snapshot gets hard
links to the unchanged files, the new halo as stars/stars_model_halo.npy, and the model starless image with that
halo (sl_final + old halo - new halo: the starless stage subtracts the halo from the half-stacks, everything else
equal) as starless/sl_final.npy. Nothing is written outside the new snapshot.
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np

LINKS = ["optics/lat_full.npy", "optics/lat_A.npy", "optics/lat_B.npy", "optics/lateral.json", "optics/manifest.json",
         "stars/star_fluxes.npz", "stars/manifest.json", "starless/sl_mask.npy", "starless/manifest.json",
         "denoise/n2n_full.npy", "denoise/manifest.json", "background/flat.npy", "color/M.npy",
         "render_prep_learned/prep.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tag", default="new")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst) / "work"
    old = src / "snapshot/work"
    for f in LINKS:
        if not (old / f).exists():
            continue
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        if (dst / f).exists():
            (dst / f).unlink()
        try:
            os.link(old / f, dst / f)
        except OSError:
            shutil.copy2(old / f, dst / f)
    hn = dst / "stars/stars_model_halo.npy"
    if hn.exists():
        hn.unlink()
    os.link(src / f"halo_{a.tag}.npy", hn)
    sl = np.load(old / "starless/sl_final.npy", mmap_mode="r")
    h0 = np.load(old / "stars/stars_model_halo.npy", mmap_mode="r")
    h1 = np.load(src / f"halo_{a.tag}.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(dst / "starless/sl_final.npy", mode="w+", dtype=np.float32, shape=sl.shape)
    for c in range(sl.shape[0]):
        out[c] = np.asarray(sl[c], np.float32) + np.asarray(h0[c], np.float32) - np.asarray(h1[c], np.float32)
    out.flush()
    del out
    print(f"{dst}: halo {a.tag}, model starless image adjusted")


if __name__ == "__main__":
    main()
