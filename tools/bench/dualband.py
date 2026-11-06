"""Bench of issue #43: a colour camera behind a dual-band filter (H-alpha and O III), on a data-bank set (read only).

    python tools/bench/dualband.py BENCH_FOLDER SET_FOLDER [--target NAME]

A project is made in BENCH_FOLDER: one colour session (the set's lights, darks, flats, bias), one workflow stacking it
(Classic + denoise), and a fusion HOO of that workflow's channels: its red as H-alpha, its green and blue as O III.
Everything is run; the log says what each block measured.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "bench"))

from mono_path import run  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.project2 import Session  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("data")
    ap.add_argument("--target", default="")
    a = ap.parse_args()
    bench, data = Path(a.bench), Path(a.data)
    if not (bench / project2.FILE).exists():
        p = project2.create(bench, bench.name, Target(name=a.target))
        p.stack_mode = "per_session"
        p.sessions.append(Session(1, "dual band", **{k: [str(data / k)] for k in ("lights", "darks", "flats", "bias")
                                                     if (data / k).is_dir()}))
        w = p.add_workflow("Dual band", wfl.TEMPLATES["Classic + denoise"])
        f = p.add_workflow("HOO", wfl.TEMPLATES["Classic stack"])
        f.source = {"fusion": {"palette": "HOO", "inputs": [{"pid": w.id, "role": "Ha", "channel": "R"},
                                                            {"pid": w.id, "role": "OIII", "channel": "GB"}]}}
        project2.save(p)
    res = run(project2.load(bench))
    print(json.dumps(res, indent=1, default=str), flush=True)
    (bench / "result.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
