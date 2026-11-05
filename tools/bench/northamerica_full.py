"""North America, every frame (the user, 5 November: "on est sur du rendu final, je veux voir le vrai résultat"): a
bench project that reads the 109 lights and 14 darks of the AstroBackyard practice files where they are (read only),
the workflow "Full" at the engine's defaults, computed.

    python tools/bench/northamerica_full.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ours  # noqa: E402  (the bench's environment before the engine)

from astro_gpu import api  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

SRC = Path("D:/astro_donnees_test/canon_ra_rokinon135_northamerica")
ROOT = ours.C.BENCH / "northamerica" / "ours_full"


def main() -> int:
    c = ours.C.CASES["northamerica"]
    if (ROOT / "project.astroproj").exists():
        p = api.open_project(ROOT)
    else:
        p = api.create_project(ROOT, "bench north america full",
                               Target(name=c["target"], ra=c["ra"], dec=c["dec"], focal_mm=c["focal"]))
        api.add_session(p, lights=[str(SRC / "LIGHTS")], darks=[str(SRC / "DARKS")], name="north america")
        api.add_workflow(p, "Full", "Full")
        p.description = f"Quality bench: {c['camera']}, {c['scope']}; every frame ({c['note']})"
        api.save(p)
    w = next(w for w in p.workflows if w.name == "Full")
    t = time.time()
    r = ours.run(p, w, "cuda:0", lambda m: print(m, flush=True))
    print(f"Full: {r['status']} in {(time.time() - t) / 60:.1f} min", flush=True)
    return 0 if r["status"] in ("done", "up to date") else 1


if __name__ == "__main__":
    raise SystemExit(main())
