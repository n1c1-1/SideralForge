"""Bench of issue #43: a workflow that starts from an image file.

    python tools/bench/image_source.py SOURCE_PROJECT BENCH_FOLDER

The source project's stack (its first workflow's drz_full, read only) is written as a FITS file in BENCH_FOLDER; a new
project there has one workflow starting from that image (the "Process a stacked image" template: astrometry, sky
gradient, colour, output). The whole workflow is run; the log says what each block measured.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from astro_gpu import api  # noqa: E402
from astro_gpu.core import project2, runner2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.store import Keys, Store  # noqa: E402


def main(source: Path, bench: Path, stretched: str = "") -> dict:
    src = project2.load(source)
    w0 = src.workflows[0]
    d = Store(src.root).dir("stack", Keys(src).key(w0.id, "stack"))
    img = bench / "stack.fits"
    bench.mkdir(parents=True, exist_ok=True)
    if not img.is_file():
        from astropy.io import fits
        fits.PrimaryHDU(np.load(d / "drz_full.npy").astype(np.float32)).writeto(img)
    root = bench / "project"
    name = src.name + (" (stretched image)" if stretched else " (image)")
    p = project2.load(root) if (root / project2.FILE).exists() else project2.create(root, name, src.target)
    if not p.workflows and stretched:                # an image already stretched, its stars taken off
        w = p.add_workflow("Stretched", wfl.TEMPLATES["Full"])
        w.source = {"path": str(Path(stretched)), "stretched": True}
    elif not p.workflows:
        w = p.add_workflow("From the image", wfl.TEMPLATES["Classic stack"])
        w.source = {"path": img.as_posix(), "stretched": False}
    project2.save(p)
    p = project2.load(root)
    wid = p.workflows[0].id
    print("chain:", Keys(p).chain(wid), flush=True)
    plan = runner2.plan_workflows(p, [wid])
    print(f"{len(plan.jobs)} jobs: {[j.stage for j in plan.jobs]}", flush=True)
    info = api.start_run(p, plan)
    t0, off, end = time.time(), 0, None
    while end is None:
        time.sleep(5)
        events, off = runner2.read_events(p.root, off, info.run_id)
        end = next((e for e in events if e.get("type") == "run_end"), None)
    out = {"status": end.get("status"), "seconds": round(time.time() - t0), "failed": end.get("failed")}
    keys = Keys(project2.load(root))
    for st in keys.chain(wid):
        m = Store(root).manifest(st, keys.key(wid, st))
        out[st] = {k: v for k, v in (m.get("metrics") or {}).items()
                   if k in ("scale_arcsec", "stars_matched", "n_stars", "white_balance", "sky_variation_pct",
                            "width", "height", "factor", "stars_used", "fit_stars")} or m.get("status")
    print(json.dumps(out, indent=1, default=str), flush=True)
    (bench / "result.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
