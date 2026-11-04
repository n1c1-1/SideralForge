"""The bench projects ready to browse the Denoise block: in each of them (``ours_base`` and ``ours_express`` of every
case), two copies of the Classic stack workflow with Denoise, the pretrained network and the network adapted to the
image; then what is not computed is computed (the stack is shared and kept).

    python tools/bench/denoise_projects.py [--cases heart,iris,...] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

BENCH = Path("F:/astro_essais/bench")
CASES = ("heart", "iris", "m51", "m8", "rosette", "seestar")
SESSIONS = ("ours_base", "ours_express")
SOURCE = "Classic stack"
#: the workflows added: name -> the Denoise method
ADDED = {"Classic stack · denoise": "pretrained", "Classic stack · denoise adapted": "adapted"}


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def prepare(api, root: Path):
    p = api.open_project(root)
    src = next((w for w in p.workflows if w.name == SOURCE), None)
    if src is None:
        log(f"{root}: no '{SOURCE}' workflow, left as it is")
        return p
    for name, method in ADDED.items():
        if not any(w.name == name for w in p.workflows):
            w = api.duplicate_workflow(p, src.id, name)
            api.set_option(p, w.id, "denoise", method)
            log(f"{root}: workflow '{name}' added (id {w.id})")
    api.save(p)
    return p


def main(argv=None) -> int:
    import make_test_project as mtp

    from astro_gpu import api
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    t0 = time.time()
    bad = []
    for case in a.cases.split(","):
        for s in SESSIONS:
            root = BENCH / case / s
            if not root.is_dir():
                continue
            p = prepare(api, root)
            plan = api.plan_all(p, a.device)
            log(f"{case} {s}: {len(plan.jobs)} jobs: {', '.join(j.stage for j in plan.jobs)}")
            if not mtp.run(api, p, plan, a.device):
                bad.append(f"{case} {s}")
    log(f"done in {(time.time() - t0) / 60:.1f} min; not complete: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
