"""The star tools of Post on the stars and background layer (render/stars.py: Small stars, Star size), at full
resolution on a crowded part of each field, side by side: today's defaults, then the small stars removed by half and
whole, then whole with the stars one step smaller.

    python tools/bench/stars_check.py [CASE ...]      (rosette, m31, northamerica; all by default)

Out: F:/astro_essais/etoiles/<case>.jpg. Reads the bench projects only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ours  # noqa: E402, I001  (the bench's environment before the engine)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from astro_gpu import api  # noqa: E402

OUT = Path("F:/astro_essais/etoiles")
#: case -> (project, workflow, the crop's centre as a share of the field)
CASES = {"rosette": ("rosette/ours_full", "Full", (0.78, 0.30)),
         "m31": ("m31/ours_base", "Full", (0.25, 0.75)),
         "northamerica": ("northamerica/ours_full", "Full", (0.5, 0.5))}
SETS = (("défauts", {}),
        ("petites étoiles 1", {"star_small": 1}),
        ("petites étoiles 0", {"star_small": 0}),
        ("petites étoiles 0, taille 1", {"star_small": 0, "star_size": 1}))
CROP = (900, 600)


def label(a: np.ndarray, text: str) -> np.ndarray:
    a = a.copy()
    cv2.putText(a, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5)
    cv2.putText(a, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return a


def sheet(case: str) -> Path:
    rel, name, (fx, fy) = CASES[case]
    p = api.open_project(ours.C.BENCH / rel)
    w = next(w for w in p.workflows if w.name == name)
    full = api.render_layer(p, w.id, None, {}, scale=0.125, part="combined")
    x0, x1, y0, y1 = full.crop
    W, H = x1 - x0, y1 - y0                                  # the useful frame, full-resolution pixels
    cx, cy = int(x0 + fx * W), int(y0 + fy * H)
    region = (cx - CROP[0] // 2, cx + CROP[0] // 2, cy - CROP[1] // 2, cy + CROP[1] // 2)
    tiles = []
    for text, pos in SETS:
        im = api.render_layer(p, w.id, None, pos, scale=1.0, region=region, part="combined")
        a = (np.clip(np.moveaxis(np.asarray(im.rgb), 0, -1), 0, 1) * 255).astype(np.uint8)[..., ::-1]
        tiles.append(label(np.ascontiguousarray(a), f"{case} - {text}"))
    out = OUT / f"{case}.jpg"
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


if __name__ == "__main__":
    for c in sys.argv[1:] or list(CASES):
        print(sheet(c), flush=True)
