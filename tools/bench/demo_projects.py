"""Clean projects for the user to open and play with (5 November): a copy of a bench project with its workflow "Full"
alone, its results copied (no computation), Post empty as a new user finds it. The bench projects are not touched.

    python tools/bench/demo_projects.py

Out: F:/astro_essais/projets_de_demo/<Name>/ (Rosette, North America, M31). Only the results of the stages of "Full"
are copied (the calibrated frames of the scratch folder and the other workflows' results stay in the bench).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ours  # noqa: E402  (the bench's environment before the engine)

from astro_gpu import api  # noqa: E402

OUT = Path("F:/astro_essais/projets_de_demo")
#: the demo's name -> (the bench project, its description)
DEMOS = {
    "Rosette": ("rosette/ours_full", "Rosette, 28.12.24: every light (358), the Iris night's darks and flats"),
    "North America": ("northamerica/ours_full", "North America: AstroBackyard's practice files, every light (109)"),
    "M31": ("m31/ours_base", "M31, 24.08.23: 100 of the 508 lights, chosen evenly over the night"),
}


def demo(name: str, rel: str, text: str) -> None:
    src_root = ours.C.BENCH / rel
    dst_root = OUT / name
    if dst_root.exists():
        raise SystemExit(f"{dst_root} exists already: nothing done")
    p = api.open_project(src_root)
    full = next(w for w in p.workflows if w.name == "Full")
    keys = {st: api.stage_key(p, full, st) for st in api.chain(p, full)}
    missing = [st for st, k in keys.items() if not (src_root / "store" / st / k).is_dir()]
    if missing:
        raise SystemExit(f"{name}: not computed: {missing}")
    for st, k in keys.items():                           # the results of Full only
        shutil.copytree(src_root / "store" / st / k, dst_root / "store" / st / k)
    shutil.copy2(src_root / "project.astroproj", dst_root / "project.astroproj")
    d = api.open_project(dst_root)
    for w in [w for w in d.workflows if w.name != "Full"]:
        api.delete_workflow(d, w.id)
    d.name, d.description = name, text
    d.post = {"layers": []}                              # Post as a new user finds it
    d.render_bases = {}
    api.set_current(d, next(w.id for w in d.workflows))
    api.save(d)
    d = api.open_project(dst_root)
    api.refresh(d)
    bad = {st: s.state for (_w, st), s in api.states(d).items() if s.state != "done"}
    print(f"{name}: {len(keys)} stages copied, {'all done' if not bad else f'NOT done: {bad}'}", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (rel, text) in DEMOS.items():
        demo(name, rel, text)
