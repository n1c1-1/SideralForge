"""The zones where two renders of the same project differ most, as fractions of the render, for blind.py --zones.

    python tools/qc/changed_zones.py BEFORE.tif AFTER.tif [N]

Reads two full-resolution renders (16-bit TIFF of the same size), smooths the largest channel difference over
~60 px, and prints the N (default 6) strongest zones at least 600 px apart as "x:y" fractions, then their
strength in 8-bit levels on stderr.
"""
import sys

import cv2
import numpy as np
import tifffile


def main():
    a = tifffile.imread(sys.argv[1]).astype(np.float32) / 257.0
    b = tifffile.imread(sys.argv[2]).astype(np.float32) / 257.0
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    H, W = a.shape[:2]
    k = 8                                               # work at 1/8: zones, not pixels
    d = cv2.resize(np.abs(b - a).max(-1), (W // k, H // k), interpolation=cv2.INTER_AREA)
    d = cv2.GaussianBlur(d, (0, 0), 60 / k)
    zones, strength = [], []
    for _ in range(n):
        y, x = np.unravel_index(np.argmax(d), d.shape)
        if d[y, x] <= 0:
            break
        zones.append(f"{(x + 0.5) * k / W:.3f}:{(y + 0.5) * k / H:.3f}")
        strength.append(round(float(d[y, x]), 2))
        cv2.circle(d, (int(x), int(y)), 600 // k, 0.0, -1)
    print(" ".join(zones))
    print("levels:", strength, file=sys.stderr)


if __name__ == "__main__":
    main()
