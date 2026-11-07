"""Timing against Siril on this machine: ours on the CPU (or a GPU) and Siril, each in a project of its own made for the
measure (nothing reused from an earlier run), on the same frozen frames of a bench case and level.

    python tools/bench/timing.py ours CASE LEVEL TAG [--device cpu]
    python tools/bench/timing.py siril CASE LEVEL TAG [--threads N]

The projects are F:/astro_essais/bench/<case>/ours_<level>_<TAG>/ and siril_<level>_<TAG>/ (a new TAG: everything
computed again). ours.py writes bench_run.json (wall time, every stage's duration); run_siril.py its siril.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "siril"))


def main() -> int:
    who, case, level, tag, *rest = sys.argv[1:]
    import cases as C
    if who == "ours":
        C.ours_dir = lambda c, lv, v="": C.BENCH / c / f"ours_{lv}_{tag}"
        import ours
        sys.argv = ["ours.py", case, level, "--no-export", *(rest or ["--device", "cpu"])]
        return ours.main()
    C.siril_dir = lambda c, lv, v="": C.BENCH / c / f"siril_{lv}_{tag}"
    import run_siril
    sys.argv = ["run_siril.py", case, level, *rest]
    return run_siril.main()


if __name__ == "__main__":
    sys.exit(main())
