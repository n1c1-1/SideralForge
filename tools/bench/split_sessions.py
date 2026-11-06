"""Bench of issue #43 (the user's test D): one session split in two, each half calibrated on its own with the same
calibration frames, then joined and stacked; compared with the stack of the whole session.

    python tools/bench/split_sessions.py SOURCE_PROJECT BENCH_FOLDER

A new project is made in BENCH_FOLDER (never in the source project): the source's definition, its session's lights
split in two sessions by file (sorted names, first half and second half), the same darks, flats and bias for both,
"per_session". Its first workflow is run to Stacking, then its stack is compared with the source's stack of the same
workflow (the source's results are only read).
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
from astro_gpu.core.project import list_frames  # noqa: E402
from astro_gpu.core.store import Keys, Store  # noqa: E402


def make(source: Path, bench: Path) -> project2.Project2:
    src = project2.load(source)
    if (bench / project2.FILE).exists():
        return project2.load(bench)
    d = src.to_dict()
    s = src.sessions[0]
    lights = [str(f) for f in list_frames([src.abspath(x) for x in s.lights])]
    half = len(lights) // 2
    calib = {k: [src.abspath(x) for x in getattr(s, k)] for k in ("darks", "flats", "bias", "flat_darks")}
    d["sessions"] = [{"id": 1, "name": "first half", "lights": lights[:half], **calib},
                     {"id": 2, "name": "second half", "lights": lights[half:], **calib}]
    d["stack_mode"] = "per_session"
    d["excluded"] = []
    d["post"] = {"layers": []}
    d["render_bases"] = {}
    d["scratch_dir"] = ""
    d["name"] = src.name + " (2 sessions)"
    p = project2.Project2.from_dict(d, bench)
    bench.mkdir(parents=True, exist_ok=True)
    project2.save(p, bench)
    return project2.load(bench)


def run_to_stack(p: project2.Project2) -> None:
    wid = p.workflows[0].id
    plan = runner2.plan_to_block(p, wid, "stacking")
    print(f"{len(plan.jobs)} jobs: {[j.stage for j in plan.jobs]}", flush=True)
    info = api.start_run(p, plan)
    t0 = time.time()
    off, done = 0, False
    while not done:                                  # until the run's end event (the lock comes a moment later)
        time.sleep(10)
        events, off = runner2.read_events(p.root, off, info.run_id)
        done = any(e.get("type") == "run_end" for e in events)
        print(f"  ... {time.time() - t0:.0f} s", flush=True)
    print(f"run over in {time.time() - t0:.0f} s ({info.run_id})", flush=True)


def stack_dir(p: project2.Project2) -> Path:
    w = p.workflows[0]
    d = Store(p.root).dir("stack", Keys(p).key(w.id, "stack"))
    if not (d / "drz_full.npy").is_file():
        raise SystemExit(f"no stack in {d}")
    return d


def compare(src: project2.Project2, p: project2.Project2) -> dict:
    a = np.load(stack_dir(src) / "drz_full.npy").astype(np.float64)
    b = np.load(stack_dir(p) / "drz_full.npy").astype(np.float64)
    if a.shape != b.shape:
        return {"shape": [a.shape, b.shape]}
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    diff = (b - a)[ok]
    scale = float(np.median(np.abs(a[ok] - np.median(a[ok])))) * 1.4826
    fa = json.loads((stack_dir(src) / "frames.json").read_text(encoding="utf-8"))
    fb = json.loads((stack_dir(p) / "frames.json").read_text(encoding="utf-8"))
    rows = [f if isinstance(f, list) else (f.get("frames") or []) for f in (fa, fb)]
    used = [sum(1 for r in rs if (r.get("weight") or 0) > 0) for rs in rows]
    return {"identical": bool(np.array_equal(a, b)), "max_abs_diff": float(np.abs(diff).max()),
            "median_abs_diff": float(np.median(np.abs(diff))), "sky_noise": scale,
            "max_diff_in_noise": float(np.abs(diff).max() / scale) if scale else None, "frames_used": used}


if __name__ == "__main__":
    source, bench = Path(sys.argv[1]), Path(sys.argv[2])
    src = project2.load(source)
    p = make(source, bench)
    print("chain:", Keys(p).chain(p.workflows[0].id)[:8], flush=True)
    run_to_stack(p)
    res = compare(src, project2.load(bench))
    print(json.dumps(res, indent=1), flush=True)
    (bench / "comparison.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
