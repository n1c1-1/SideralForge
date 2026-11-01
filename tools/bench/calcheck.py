"""The engine's own check of the calibration frames (api.calibration_frames: core/diagnostics.py judges each dark,
flat, bias and flat dark alone; "bad" = not a frame of its kind, e.g. a dark with stars) on every bench project.

    python tools/bench/calcheck.py [CASE ...]          (every case by default)

For each project <case>/ours_<level>[_<variant>] (Express and Base read the same calibration copies): every frame's
flag (ok, warn, bad), its reason and the number of stars the engine counted. Writes <case>/calcheck.json and
bench/calcheck.json (the summary). Nothing is excluded or changed.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

os.environ["USERPROFILE"] = str(C.HOME)
os.environ["ASTRO_GPU_HOME"] = str(C.AG_HOME)
C.offline_env()
sys.path.insert(0, str(C.REPO))
C.utf8_console()

from astro_gpu import api  # noqa: E402


def projects(case: str) -> list[Path]:
    root = C.BENCH / case
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("ours_")
                  and (d / "project.astroproj").exists())


def check(d: Path) -> dict:
    p = api.open_project(d)
    out: dict = {"project": d.name, "sessions": []}
    for s in p.sessions:
        rep = api.calibration_frames(p, s.id, require_done=False)
        kinds: dict = {}
        for f in rep.frames:
            if f.kind == "lights":
                continue
            k = kinds.setdefault(f.kind, {"count": 0, "ok": 0, "warn": 0, "bad": 0, "stars_max": 0, "flagged": []})
            k["count"] += 1
            k[f.flag] = k.get(f.flag, 0) + 1
            stars = f.stats.get("stars")
            if isinstance(stars, (int, float)):
                k["stars_max"] = max(k["stars_max"], int(stars))
            if f.flag != "ok":
                k["flagged"].append({"name": f.name, "flag": f.flag, "reason": f.reason, "stars": stars,
                                     "median": f.stats.get("median")})
        out["sessions"].append({"session": s.id, "kinds": kinds, "notes": rep.notes})
    return out


def main() -> int:
    names = sys.argv[1:] or list(C.CASES)
    t0 = time.time()
    summary_f = C.BENCH / "calcheck.json"
    summary = json.loads(summary_f.read_text(encoding="utf-8")) if summary_f.exists() else {}
    summary.setdefault("cases", {})
    for case in names:
        res = {"case": case, "when": datetime.now().isoformat(timespec="seconds"), "projects": []}
        for d in projects(case):
            r = check(d)
            res["projects"].append(r)
            line = "; ".join(f"{k} {v['count']}: {v['ok']} ok, {v['warn']} warn, {v['bad']} bad (stars max "
                             f"{v['stars_max']})" for s in r["sessions"] for k, v in s["kinds"].items())
            print(f"{case} {d.name}: {line or 'no calibration frames'} ({time.time() - t0:.0f} s)", flush=True)
            for s in r["sessions"]:
                for k, v in s["kinds"].items():
                    for f in v["flagged"]:
                        print(f"   {k} {f['name']}: {f['flag']} - {f['reason']} (stars {f['stars']})", flush=True)
        (C.BENCH / case / "calcheck.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        summary["cases"][case] = {
            r["project"]: {k: {x: v[x] for x in ("count", "ok", "warn", "bad", "stars_max")}
                           for s in r["sessions"] for k, v in s["kinds"].items()} for r in res["projects"]}
    summary["when"] = datetime.now().isoformat(timespec="seconds")
    summary_f.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"written {summary_f} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
