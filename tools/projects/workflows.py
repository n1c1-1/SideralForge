"""P8 step 3: the workflows of each imported project (DESIGN §6.3 P8, step 4).

    python tools/projects/workflows.py [--dry-run] [PROJECT ...]

The import made one workflow from the v1 project's main settings: "Full + neural stars" (star_net on), or "Full +
earlier neural layer" (the project used a learned star layer: Iris, Rosette, Horsehead, Flaming Star). This adds
"Classic stack", "Classic + denoise" and "Full" from the templates, each with the imported workflow's stored
settings (so they share its results: inventory → optics stay adopted), less the earlier star layer and the star_net
settings, which only the neural workflow uses. A workflow is kept only where the data allow it: its stages can be
keyed and the planner does not leave it out (an earlier layer made on another stack). Re-running adds only what is
missing. The order is cheap first; the imported workflow stays the current one. The table goes to
_p8/workflows.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p8  # noqa: E402

p8.use_home()

from astro_gpu import api  # noqa: E402

ADDED = ["Classic stack", "Classic + denoise", "Full"]
NEURAL = ("Full + neural stars", "Full + earlier neural layer")


def base_params(params: dict) -> dict:
    """The imported workflow's settings without what only its neural star removal uses."""
    out = copy.deepcopy(params)
    out.pop("star_net", None)
    st = out.get("starless")
    if isinstance(st, dict):
        st.pop("learned_stars", None)
        if not st:
            out.pop("starless")
    return out


def setup(name: str, dry: bool) -> dict:
    root = p8.WS / name
    p = api.open_project(root)
    names = [w.name for w in p.workflows]
    main = next((w for w in p.workflows if w.name in NEURAL or w.name == "Full"), p.workflows[0])
    added, refused = [], {}
    for t in ADDED:
        if t in names:
            continue
        w = api.add_workflow(p, t, t, base_params(main.params))
        try:
            api.chain(p, w.id)
            for s in api.chain(p, w.id):
                api.stage_key(p, w.id, s)
        except ValueError as e:
            api.delete_workflow(p, w.id)
            refused[t] = str(e)
            continue
        added.append(t)
    plan = api.plan_all(p, "cuda:0")
    for wid, why in plan.skipped.items():
        w = p.workflow(wid)
        refused[w.name] = why
        if w.name in ADDED:
            api.delete_workflow(p, wid)
    rank = {n: i for i, n in enumerate(ADDED + list(NEURAL))}
    api.set_workflows(p, sorted(api.workflows(p), key=lambda w: (rank.get(w.name, 99), w.id)), main.id)
    if not dry:
        api.save(p)
    keys = {w.name: {s: api.stage_key(p, w.id, s)[:8] for s in api.chain(p, w.id)} for w in p.workflows}
    shared = {w.name: [s for s in api.chain(p, w.id) if len(api.sharing(p, w.id, s)) > 1] for w in p.workflows}
    plan = api.plan_all(p, "cuda:0")
    per_wf = {w.name: round(sum(j.est_s for j in plan.jobs if w.id in j.wf) / 3600, 2) for w in p.workflows}
    return {"workflows": [{"id": w.id, "name": w.name, "options": w.options, "chain": api.chain(p, w.id)}
                          for w in p.workflows],
            "current": p.current, "added": added, "refused": refused, "keys": keys, "shared": shared,
            "jobs": len(plan), "est_h": round(plan.seconds / 3600, 2), "est_h_per_workflow": per_wf,
            "measured": plan.measured, "disk_gb": round(plan.disk_gb, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projects", nargs="*")
    ap.add_argument("--dry-run", action="store_true", help="show what would be added, save nothing")
    a = ap.parse_args()
    names = a.projects or [p.name for p in p8.PROJECTS]
    out_file = p8.P8 / "workflows.json"
    table = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() and not a.dry_run else {}
    for n in names:
        r = setup(n, a.dry_run)
        table[n] = r
        wf = ", ".join(f"{w['id']} {w['name']}" for w in r["workflows"])
        print(f"{n:12s} {wf}")
        print(f"{'':12s} {r['jobs']} jobs, about {r['est_h']} h ({'measured' if r['measured'] else '≈'}), "
              f"{r['disk_gb']} GB; per workflow {r['est_h_per_workflow']}")
        for t, why in r["refused"].items():
            print(f"{'':12s} not added: {t}: {why}")
    if not a.dry_run:
        table["_written"] = datetime.now().isoformat(timespec="seconds")
        out_file.write_text(json.dumps(table, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
