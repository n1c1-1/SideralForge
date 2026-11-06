"""Bench of issue #43, option A: a fusion of filters made on their starless images, its stars put back from a source.

    python tools/bench/fusion_stars.py BENCH_PROJECT MODE [ROLE_OR_WORKFLOW] [--out IMAGE]

On a bench project holding channel workflows and a fusion (tools/bench/mono_path.py --fusion, or dualband.py): the
channel workflows get Split stars & background (Denoise with it), the fusion its star source (MODE palette, channel
with a role, or workflow with the name of the workflow whose stars are taken), then everything is run and the fusion's
starless image, its star layer and the two recombined (the Combine block's default settings) are rendered side by side.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "bench"))

from mono_path import run  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.store import Keys, Store  # noqa: E402
from astro_gpu.render.layers import LayerService  # noqa: E402


def prepare(root: Path, mode: str, which: str) -> project2.Project2:
    p = project2.load(root)
    fusion = next(w for w in p.workflows if (w.source or {}).get("fusion"))
    inputs = {i["pid"] for i in fusion.source["fusion"]["inputs"]}
    for w in p.workflows:
        if w.id in inputs or w.name == which:
            w.options = wfl.check_options({**w.options, "denoise": w.options.get("denoise") or "pretrained",
                                           "stars": "neural"})
    stars = {"mode": mode}
    if mode == "channel":
        stars["role"] = which
    elif mode == "workflow":
        stars["pid"] = next(w.id for w in p.workflows if w.name == which)
    fusion.source = {"fusion": {**fusion.source["fusion"], "stars": stars}}
    fusion.options = wfl.check_options({**fusion.options, "denoise": "pretrained", "stars": "neural"})
    project2.save(p)
    return project2.load(root)


def sheet(p: project2.Project2, out: Path) -> None:
    fusion = next(w for w in p.workflows if (w.source or {}).get("fusion"))
    k = Keys(p)
    d = Store(p.root).dir("star_removal", k.key(fusion.id, "star_removal"))
    svc = LayerService("cuda:0")
    tiles = []
    for lab, part in (("starless", "starless"), ("stars", "stars"), ("combined", "combined")):
        li = svc.render(d, None, {}, 0.5, part=part)
        a = (np.clip(np.moveaxis(li.rgb, 0, -1), 0, 1)[..., ::-1] * 255).astype(np.uint8)
        a = cv2.resize(a, (600, int(600 * a.shape[0] / a.shape[1]))).copy()
        cv2.putText(a, lab, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(a)
    cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("mode", choices=("palette", "channel", "workflow"))
    ap.add_argument("which", nargs="?", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    p = prepare(Path(a.project), a.mode, a.which)
    print(run(p).get("status"), flush=True)
    if a.out:
        sheet(project2.load(a.project), Path(a.out))
