"""Our engine on the bench: a v2 project per case and level, workflow "Classic stack", run on a GPU, then exported.

    python tools/bench/ours.py CASE LEVEL [--variant V] [--device cuda:0] [--no-export]

The project is F:/astro_essais/bench/<case>/ours_<level>/ (ours_<level>_<V> for a case with variants; created once
through the engine API; afterwards it is only re-run: the store recomputes what changed, e.g. after engine fixes).
Its single session reads the same frozen copies as Siril (data/lights_<level>, data/darks, flats, bias; a variant's
own darks from data/darks_<V>); the target carries the same approximate centre and focal length Siril is given.
Everything stays at the engine's suggested settings.

Environment of the run: USERPROFILE = bench/_home (the engine's catalogue cache, ~/.astro_gpu/cache, lands in
bench/_home/.astro_gpu/cache instead of the user's), ASTRO_GPU_HOME = bench/_home/.astro_gpu (camera profiles copied
from the user's), ASTRO_GPU_SIRIL_CATALOGUES = bench/_siril/catalogues (local Gaia extracts first),
CUDA_VISIBLE_DEVICES chosen per run by the engine's runner from --device.

Writes ours_<level>/bench_run.json: the run's status, wall time, the duration of every stage (computed or reused),
the stage metrics the bench reads, and the exported final (export/bench_final/: TIFF 16 bits and PNG 16 bits).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

C.engine_home()                                  # before astro_gpu is imported: Path.home() -> the bench home
os.environ["PYTHONUNBUFFERED"] = "1"
C.offline_env()
sys.path.insert(0, str(C.REPO))
C.utf8_console()

from astro_gpu import api  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

WF_NAME = "Classic stack"


def project(case: str, level: str, variant: str = ""):
    man = json.loads((C.BENCH / case / "manifest.json").read_text(encoding="utf-8"))
    root = C.ours_dir(case, level, variant)
    if (root / "project.astroproj").exists():
        p = api.open_project(root)
    else:
        lv = man["levels"][level]
        cal = C.calibration(man, variant)
        tag = f" {variant}" if variant else ""
        p = api.create_project(root, f"bench {case} {level}{tag}",
                               Target(name=man["target"], ra=man["ra"], dec=man["dec"], focal_mm=man["focal_mm"]))
        api.add_session(p, lights=[lv["folder"]], darks=[cal["darks"]["folder"]] if "darks" in cal else [],
                        flats=[cal["flats"]["folder"]] if "flats" in cal else [],
                        bias=[cal["bias"]["folder"]] if "bias" in cal else [], name=f"{case} {level}{tag}")
        api.add_workflow(p, WF_NAME, WF_NAME)
        p.description = f"Quality bench (tools/bench): {man['camera_label']}, {man['scope']}; {lv['count']} lights"
        if variant:
            p.description += f"; variant {variant}: {man['variants'][variant]['label']}"
        api.save(p)
    w = next(w for w in p.workflows if w.name == WF_NAME)
    return p, w, man


def run(p, w, device: str, log, stop_before: str | None = None) -> dict:
    """Start the plan (or attach to the project's live run) and follow it to its end (cli2.follow: the worker's
    events, its pid). The durations come from the run's events: wall time from run_start to run_end."""
    from astro_gpu.cli2 import follow
    info = api.running(p)
    out: dict = {"stages": {}}
    if info is None:
        plan = api.plan_workflows(p, [w.id], device)
        if stop_before:                  # only the stages before it (the catalogue pre-fetch goes in between)
            ch = api.chain(p, w.id)
            cut = ch.index(stop_before)
            plan.jobs = [j for j in plan.jobs if ch.index(j.stage) < cut]
            plan.targets = [t for t in plan.targets if t[1] in ch[:cut]]
        out.update({"jobs": len(plan), "planned": [j.stage for j in plan.jobs]})
        if not plan:
            out["status"] = "up to date"
            return out
        info = api.start_run(p, plan, device)
        log(f"run {info.run_id} started: {len(plan)} job(s) {out['planned']}; pid {info.pid}")
    else:
        log(f"attached to the live run {info.run_id} (pid {info.pid})")
    out["run_id"] = info.run_id
    status = follow(p, info.offset, info.run_id, info.pid, poll=2.0, echo=lambda s: log(s.strip()))
    evs, _ = api.read_events(p, info.offset, info.run_id)
    t_start = next((e["t"] for e in evs if e.get("type") == "run_start"), None)
    t_end = next((e["t"] for e in evs if e.get("type") == "run_end"), None)
    for e in evs:
        if e.get("type") == "stage_end":
            out["stages"][e.get("stage")] = {k: e.get(k) for k in ("status", "duration", "reused", "error", "reason")}
    out["status"] = status
    if t_start and t_end:
        out["wall_s"] = round(t_end - t_start, 1)
    api.refresh(p)
    api.states(p)
    return out


def metrics(p, w) -> dict:
    """The stage metrics the bench reads (from each stage's manifest in the store)."""
    out = {}
    for s in api.chain(p, w.id):
        try:
            o = api.stage_outputs(p, w.id, s)
        except Exception as e:  # noqa: BLE001
            out[s] = {"error": str(e)}
            continue
        if o is None:
            continue
        out[s] = {"dir": str(o.dir) if hasattr(o, "dir") else None,
                  "metrics": getattr(o, "metrics", None), "duration": getattr(o, "duration", None)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=list(C.CASES))
    ap.add_argument("level", choices=list(C.LEVELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--stop-before", help="run only the stages before this one (e.g. catalogs)")
    ap.add_argument("--variant", help="the variant of a case that has some (rosette: A, B, C)")
    a = ap.parse_args()
    variant = C.norm_variant(a.case, a.variant)
    if variant and a.level not in C.variant_levels(a.case, variant):
        print(f"{a.case} {variant}: no {a.level} level")
        return 1
    p, w, man = project(a.case, a.level, variant)
    if man["levels"][a.level].get("same_as"):
        print(f"{a.case} {a.level}: same lights as {man['levels'][a.level]['same_as']}: nothing to run")
        return 0
    rec_f = Path(p.root) / "bench_run.json"
    hist = json.loads(rec_f.read_text(encoding="utf-8")) if rec_f.exists() else {"runs": []}

    def log(msg: str) -> None:
        print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)

    t0 = time.time()
    r = run(p, w, a.device, log, a.stop_before)
    if a.stop_before:
        r["partial"] = f"stages before {a.stop_before}"
    r.update({"when": datetime.now().isoformat(timespec="seconds"), "device": a.device,
              "CUDA_VISIBLE_DEVICES_parent": os.environ.get("CUDA_VISIBLE_DEVICES")})
    if r["status"] in ("done", "up to date") and not a.no_export and not a.stop_before:
        te = time.time()
        files = api.export_workflow(p, w.id, None, None, {"name": "bench_final", "formats": ["tiff16", "png16"]},
                                    device="cpu")
        r["export"] = [str(f) for f in files]
        r["export_s"] = round(time.time() - te, 1)
        log(f"exported {[Path(f).name for f in files]} in {r['export_s']} s")
    r["metrics"] = metrics(p, w)
    r["total_s"] = round(time.time() - t0, 1)
    hist["runs"].append(r)
    hist["last"] = r
    rec_f.write_text(json.dumps(hist, indent=1, default=str), encoding="utf-8")
    log(f"{a.case} {a.level}{C.suffix(variant)}: {r['status']} in {r.get('wall_s', 0) / 60:.1f} min")
    return 0 if r["status"] in ("done", "up to date") else 1


if __name__ == "__main__":
    sys.exit(main())
