"""P8 step 1: the pilot ASTRO_GPU_HOME (F:/astro_projets/_home).

    python tools/projects/setup_home.py

- copies the camera profiles of the user's ASTRO_GPU_HOME (~/.astro_gpu/camera_profiles; copied, never moved; a
  profile already in the pilot home is left as it is);
- seeds the timing history (_home/timings2.json) with estimate.learn_from_v1 on the 9 projects, read only, under the
  import's write guard;
- writes _p8/home.json.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p8  # noqa: E402

p8.use_home()

from astro_gpu.core import estimate as est  # noqa: E402
from astro_gpu.core import import_v1 as iv  # noqa: E402


def main() -> int:
    p8.HOME.mkdir(parents=True, exist_ok=True)
    p8.P8.mkdir(parents=True, exist_ok=True)
    src = p8.USER_HOME / "camera_profiles"
    dst = p8.HOME / "camera_profiles"
    dst.mkdir(parents=True, exist_ok=True)
    copied, kept = [], []
    with iv.WriteGuard([p8.USER_HOME]):
        for f in sorted(src.iterdir()):
            if not f.is_file():
                continue
            t = dst / f.name
            if t.exists():
                kept.append(f.name)
                continue
            shutil.copy2(f, t)
            copied.append(f.name)
    print(f"camera profiles: copied {copied}, already there {kept}")
    seeded: dict[str, int] = {}
    for proj in p8.PROJECTS:
        with iv.WriteGuard(iv.folders_of(proj.old)):
            entries = est.learn_from_v1(proj.old)
        seeded[proj.name] = est.seed(entries)
        print(f"{proj.name}: {len(entries)} stage durations read, {seeded[proj.name]} added to the history")
    hist = est.load_history()
    report = {"home": str(p8.HOME), "profiles_from": str(src), "copied": copied, "kept": kept, "seeded": seeded,
              "history_stages": {s: len(v) for s, v in hist.items()},
              "created": datetime.now().isoformat(timespec="seconds")}
    (p8.P8 / "home.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"history: {est.history_path()}: " + ", ".join(f"{s} {n}" for s, n in report["history_stages"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
