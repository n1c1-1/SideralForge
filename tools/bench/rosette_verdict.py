"""The Rosette verdict (docs: F:/astro_essais/rosette_verdict/JOURNAL.md): the bench project of the whole night, driven
through the engine API only (what the GUI does), so that every setting can be opened and seen in the GUI.

    python tools/bench/rosette_verdict.py setup      copy the 358 lights, create bench/rosette/ours_full
    python tools/bench/rosette_verdict.py plan       the jobs and the engine's estimate, nothing run
    python tools/bench/rosette_verdict.py run        compute what is not (the store keeps the rest)

The inputs are the old render2's (E:/astro_gpu/projects/Rosette): the 358 lights of 28 December 2024; the Iris night's
48 darks (the bench's variant B: the night's own "darks" are sky frames, controles/darks_controle.png) and its 62 flats,
used for the vignetting only (Calibration > flat_scale_px = 40, as render2: their dust is not this night's). One
workflow, "Full" (gradient, colour, denoise, stars), every other setting the engine's suggestion. The environment is
the bench's (tools/bench/ours.py: its home, the local Gaia extracts).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402
import ours  # noqa: E402  (sets the bench environment before the engine is imported)

from astro_gpu import api  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

SOURCE = Path("F:/astro_projets/_raw/Rosette/lights")          # read only
LIGHTS = C.BENCH / "rosette" / "data" / "lights_full"
ROOT = C.BENCH / "rosette" / "ours_full"
WF = "Full"
PARAMS = {"calibrate": {"flat_scale_px": 40.0}}


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def setup():
    LIGHTS.mkdir(parents=True, exist_ok=True)
    src = sorted(f for f in SOURCE.iterdir() if f.is_file() and f.suffix.upper() == ".ARW")
    n = 0
    for f in src:
        d = LIGHTS / f.name
        if not d.exists() or d.stat().st_size != f.stat().st_size:
            shutil.copy2(f, d)
            n += 1
    log(f"{len(src)} lights in {LIGHTS} ({n} copied)")
    if (ROOT / "project.astroproj").exists():
        log(f"{ROOT}: already there")
        return api.open_project(ROOT)
    man = json.loads((C.BENCH / "rosette" / "manifest.json").read_text(encoding="utf-8"))
    cal = C.calibration(man, "B")
    p = api.create_project(ROOT, "bench rosette full",
                           Target(name=man["target"], ra=man["ra"], dec=man["dec"], focal_mm=man["focal_mm"]))
    api.add_session(p, lights=[str(LIGHTS)], darks=[cal["darks"]["folder"]], flats=[cal["flats"]["folder"]],
                    name="rosette full")
    api.add_workflow(p, WF, WF, PARAMS)
    p.description = ("The Rosette verdict: the 358 lights of 28 Dec 2024, the Iris darks and flats (vignetting only, "
                     "flat_scale_px 40), as render2")
    api.save(p)
    log(f"{ROOT}: created, workflow {WF} with {PARAMS}")
    return p


def plan(p):
    w = next(w for w in p.workflows if w.name == WF)
    pl = api.plan_workflows(p, [w.id], "cuda:0")
    total = 0.0
    for j in pl.jobs:
        s = api.estimate(p, w.id, j.stage, "cuda:0").seconds
        total += s
        log(f"  {j.stage:16s} ≈ {s / 60:5.1f} min")
    log(f"{len(pl.jobs)} jobs, ≈ {total / 60:.0f} min")
    return w


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "plan"
    p = setup() if what == "setup" else api.open_project(ROOT)
    w = plan(p)
    if what == "run":
        t0 = time.time()
        r = ours.run(p, w, "cuda:0", log)
        log(f"{r['status']} in {(time.time() - t0) / 60:.1f} min")
        (ROOT / "verdict_run.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
        return 0 if r["status"] in ("done", "up to date") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
