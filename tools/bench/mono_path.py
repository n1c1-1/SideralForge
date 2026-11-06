"""Bench of issue #43: monochrome workflows, one per filter, on a data-bank set (read only).

    python tools/bench/mono_path.py BENCH_FOLDER SET_FOLDER FILTER [FILTER ...] [--template Full]

A project is made in BENCH_FOLDER: one session per filter (its lights, the set's darks and bias, the filter's flats
when there are some), one workflow per filter stacking that filter's session. Every workflow is run whole; the log
says what each block measured.

The set's folders: lights/<filter>, flats/<filter>, darks, bias (F:/dataset_astro/data layout).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from astro_gpu import api  # noqa: E402
from astro_gpu.core import project2, runner2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.project2 import Session  # noqa: E402
from astro_gpu.core.store import Keys, Store  # noqa: E402

#: the data bank's filter folders → the session's filter (project2.FILTERS)
FILTER = {"H-Alpha": "Ha", "Ha": "Ha", "OIII": "OIII", "SII": "SII", "L": "L", "Luminance": "L", "Red": "R", "R": "R",
          "Green": "G", "G": "G", "Blue": "B", "B": "B", "rp": "R", "V": "G"}


def make(bench: Path, data: Path, filters: list[str], template: str, target: str,
         fusion: str = "") -> project2.Project2:
    if (bench / project2.FILE).exists():
        return project2.load(bench)
    p = project2.create(bench, bench.name, Target(name=target))
    p.stack_mode = "per_session"
    for i, f in enumerate(filters, 1):
        flats = data / "flats" / f
        p.sessions.append(Session(i, f, lights=[str(data / "lights" / f)],
                                  darks=[str(data / "darks")] if (data / "darks").is_dir() else [],
                                  bias=[str(data / "bias")] if (data / "bias").is_dir() else [],
                                  flats=[str(flats)] if flats.is_dir() else [], filter=FILTER.get(f, "")))
        w = p.add_workflow(FILTER.get(f, f), wfl.TEMPLATES[template])
        w.sessions = [i]
    if fusion:                                        # the filters' outputs combined with a palette
        f = p.add_workflow(fusion, wfl.TEMPLATES["Classic stack"])
        f.source = {"fusion": {"palette": fusion, "inputs": [
            {"pid": w.id, "role": w.name, "channel": "L"} for w in p.workflows if w is not f]}}
    project2.save(p)
    return project2.load(bench)


def run(p: project2.Project2) -> dict:
    plan = runner2.plan_all(p)
    print(f"{len(plan.jobs)} jobs: {[j.stage for j in plan.jobs]}", flush=True)
    info = api.start_run(p, plan)
    t0, off, end = time.time(), 0, None
    while end is None:
        time.sleep(10)
        events, off = runner2.read_events(p.root, off, info.run_id)
        for e in events:
            if e.get("type") == "stage_end":
                print(f"  {e.get('stage')}: {e.get('status')} {e.get('error') or ''}", flush=True)
        end = next((e for e in events if e.get("type") == "run_end"), None)
    out = {"status": end.get("status"), "minutes": round((time.time() - t0) / 60, 1)}
    q = project2.load(p.root)
    k = Keys(q)
    for w in q.workflows:
        res = {}
        for st in k.chain(w.id):
            m = Store(q.root).manifest(st, k.key(w.id, st))
            met = m.get("metrics") or {}
            pick = {x: met.get(x) for x in ("n_lights", "mono", "dark_scale", "aligned", "fwhm_px", "frames_used",
                                            "scale_arcsec", "stars_used", "palette", "channels")
                    if met.get(x) is not None}
            res[st] = pick or m.get("status")
        out[w.name] = res
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("data")
    ap.add_argument("filters", nargs="+")
    ap.add_argument("--template", default="Full")
    ap.add_argument("--target", default="")
    ap.add_argument("--fusion", default="", help="a palette: the filters' outputs combined (SHO, HOO, RGB...)")
    a = ap.parse_args()
    p = make(Path(a.bench), Path(a.data), a.filters, a.template, a.target, a.fusion)
    res = run(p)
    print(json.dumps(res, indent=1, default=str), flush=True)
    (Path(a.bench) / "result.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
