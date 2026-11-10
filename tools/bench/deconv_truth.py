"""The sharp images of the deconvolution network's training: the Euclid Q1 VIS mosaics, brought to 1 arcsec per pixel.

    python tools/bench/deconv_truth.py [--out E:/float_train/truth] [--max N]

Each mosaic (19 200 px at 0.1 arcsec, 1.4 GB) is downloaded alone, averaged by 10 pixels, saved as ``<tile>.npy`` (float32,
about 15 MB) and its download removed: the disk never holds more than one mosaic. A run that stops goes on from where
it was. A coarser scale is made from these when the training pairs are drawn.
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/MER"
#: the same files on the archive's cloud copy: 24 MB/s against 0.5 (measured on 10 November 2026)
DATA = "https://nasa-irsa-euclid-q1.s3.amazonaws.com/q1/MER"
BIN = 10


def listing(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def tiles() -> list[str]:
    return sorted(set(re.findall(r'href="(\d+)/"', listing(BASE + "/"))))


def mosaic_name(tile: str) -> str | None:
    names = re.findall(r'href="(EUC_MER_BGSUB-MOSAIC-VIS_[^"]+\.fits)"', listing(f"{BASE}/{tile}/VIS/"))
    return names[0] if names else None


def download(url: str, to: Path) -> None:
    part = to.with_name(to.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, part.open("wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    part.replace(to)


def binned(fits_file: Path) -> np.ndarray:
    """The mosaic averaged over BIN x BIN pixels, its pixels without data left out of each mean (0 where none has data)."""
    from astropy.io import fits
    with fits.open(fits_file, memmap=True) as h:
        hdu = next(x for x in h if x.data is not None and x.data.ndim == 2)
        a = np.asarray(hdu.data, np.float32)
    H, W = (a.shape[0] // BIN) * BIN, (a.shape[1] // BIN) * BIN
    a = a[:H, :W]
    good = np.isfinite(a) & (a != 0)
    v = np.where(good, a, 0).reshape(H // BIN, BIN, W // BIN, BIN).sum((1, 3))
    n = good.reshape(H // BIN, BIN, W // BIN, BIN).sum((1, 3))
    return (v / np.maximum(n, 1)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("E:/float_train/truth"))
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = a.out / "current.fits"
    done = 0
    all_tiles = tiles()
    print(len(all_tiles), "tiles", flush=True)
    for t in all_tiles:
        out = a.out / f"{t}.npy"
        if out.exists():
            continue
        name = mosaic_name(t)
        if name is None:
            print(t, "no VIS mosaic", flush=True)
            continue
        t0 = time.time()
        for attempt in range(3):
            try:
                download(f"{DATA}/{t}/VIS/{name}", tmp)
                break
            except OSError as e:
                print(t, "download failed:", e, flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            continue
        t1 = time.time()
        np.save(out, binned(tmp))
        tmp.unlink()
        done += 1
        print(f"{t}: downloaded in {t1 - t0:.0f} s, reduced in {time.time() - t1:.0f} s", flush=True)
        if a.max and done >= a.max:
            break


if __name__ == "__main__":
    main()
