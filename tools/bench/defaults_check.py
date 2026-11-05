"""The check of the default settings on the user's targets, before any default changes (the user's rule, 5 November):
Heart, Iris and M31 (his own captures, base sessions) and M8 (express: its only session), each in its bench project.

    python tools/bench/defaults_check.py run      the workflow "Full" at the engine's defaults, computed in each
    python tools/bench/defaults_check.py sheet    the Stars and background layer rendered at today's slider defaults
                                                  and at the proposed ones, side by side, with the user's own render
                                                  when there is one (M31: traitement/m31_flat.jpg)

Out: F:/astro_essais/defauts/<case>.jpg and planche.jpg. Nothing is written in the user's projects; the bench projects
only gain the workflow "Full".
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ours  # noqa: E402, I001  (sets the bench's environment before the engine is imported)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from astro_gpu import api  # noqa: E402

OUT = Path("F:/astro_essais/defauts")
CASES = (("heart", "base"), ("iris", "base"), ("m8", "express"), ("m31", "base"))
#: the slider positions compared: today's defaults, and the proposed ones (star sliders 1/1, Faint tones raised)
SETS = {"défauts actuels": {},
        "proposés : étoiles 1/1": {"star_strength": 1, "star_size": 1},
        "proposés : étoiles 1/1, tons faibles 3": {"star_strength": 1, "star_size": 1, "faint": 3}}
NAME = "Full"


def full_of(case: str, level: str):
    p, _w, man = ours.project(case, level)
    w = next((w for w in p.workflows if w.name == NAME), None)
    if w is None:
        w = api.add_workflow(p, NAME, NAME)
        api.save(p)
    return p, w, man


def run() -> None:
    for case, level in CASES:
        p, w, _man = full_of(case, level)
        t = time.time()
        r = ours.run(p, w, "cuda:0", lambda m: None)
        print(f"{case} {level}: {r['status']} in {(time.time() - t) / 60:.1f} min", flush=True)


def tile(rgb: np.ndarray, label: str, h: int = 520) -> np.ndarray:
    a = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if a.shape[0] == 3:
        a = np.moveaxis(a, 0, -1)
    a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h), interpolation=cv2.INTER_AREA)
    cv2.putText(a, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(a, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return a


def sheet() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, level in CASES:
        p, w, man = full_of(case, level)
        tiles = []
        for label, pos in SETS.items():
            im = api.render_layer(p, w.id, None, pos, scale=0.25, part="combined")
            x0, x1, y0, y1 = im.crop
            tiles.append(tile(np.asarray(im.rgb)[:, y0:y1, x0:x1], f"{case} - {label}"))
        ref = ours.C.CASES[case].get("reference")
        if ref and Path(ref).is_file():
            tiles.append(tile(cv2.cvtColor(cv2.imread(ref), cv2.COLOR_BGR2RGB) / 255.0, f"{case} - son image"))
        row = np.hstack(tiles)
        cv2.imwrite(str(OUT / f"{case}.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append(row)
        print(case, "ok", flush=True)
    W = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((r.shape[0], W - r.shape[1], 3), np.uint8)]) for r in rows]
    cv2.imwrite(str(OUT / "planche.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    {"run": run, "sheet": sheet}[sys.argv[1]]()
