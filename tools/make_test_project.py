"""Build the test project of the client tests, from scratch, with the current engine.

    python tools/make_test_project.py [--out FOLDER] [--data FOLDER] [--device cuda:0] [--force] [--dry-run]

What it makes (default ``F:/astro_essais/tests/heart_small``, the folder ``tests/conftest.py`` names in
``TEST_PROJECT``): a v2 project on the heart_a7s raw frames (``F:/astro_essais/donnees/heart_a7s``: 40 lights, 15
darks, 15 flats, Sony A7S at 272 mm, one night), created, scanned, planned and run through ``astro_gpu.api`` as the
client does:

- one workflow in the definition, ``Full`` (id 1), every stage computed; Denoise at 6000 training iterations, the
  setting the tests give the workflows they add in memory, so that the plans share it. The workflows the tests add
  (Classic stack, Classic + denoise, Full + neural stars) are not computed beyond what they share with Full;
- the results of ``Classic`` too: a Classic stack whose background setting differs from the default
  (``CLASSIC_PARAMS``, the same in tests/conftest.py ``TEST_PROJECT_CLASSIC``), run, then taken out of the definition
  (the next workflow id is 2 again): the Post tests add it in memory and find its final image computed.

Run it again after an engine change that makes the results outdated (the client tests then show stale blocks):
``--force`` deletes the old folder first (only inside ``F:/astro_essais/tests``). One GPU (``--device``). With the
Gaia tiles already in ``<out>/../_state/cache`` it takes about 50 minutes (the Stars stage alone about 30); the first
build also downloads about 160 Gaia tiles (an hour when VizieR does not answer). The engine's state (timings, recent
projects, download cache) goes to ``<out>/../_state``, never the user's ``~/.astro_gpu`` (its camera profiles are
copied, its download cache and its offline Gaia extracts are read: with the extracts installed, nothing is
downloaded). The tests only read this project; a test that changes it works on a copy.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("F:/astro_essais/tests/heart_small")
DATA = Path("F:/astro_essais/donnees/heart_a7s")
ALLOWED = Path("F:/astro_essais/tests")
DENOISE_ITERATIONS = 6000
#: the Heart (IC 1805) and the instrument's focal length: the astrometry starts at the right scale instead of a blind
#: search over every scale (which downloads hundreds of Gaia tiles and takes most of an hour)
TARGET = {"name": "IC 1805", "ra": 38.173, "dec": 61.4569, "focal_mm": 272.0}
#: the computed Classic workflow (tests/conftest.py TEST_PROJECT_CLASSIC): not the default gradient setting, so that
#: a Classic stack the tests add with the defaults is still to compute
CLASSIC_NAME = "Classic"
CLASSIC_PARAMS = {"gradient": {"samples": 18}}
INFO = "test_project.json"


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def follow(api, p, info, poll: float = 3.0, grace: float = 240.0) -> str:
    """Print the stages of a detached run until its end; returns its status ('done', 'failed'... or 'lost')."""
    offset, last = info.offset, time.time()
    while True:
        evs, offset = api.read_events(p, offset, info.run_id)
        for e in evs:
            last = time.time()
            t = e.get("type")
            if t == "stage_start":
                log(f"  start {e.get('stage')} (workflows {e.get('wf')})")
            elif t == "stage_end":
                log(f"  end   {e.get('stage')} {e.get('status')} in {float(e.get('duration') or 0):.1f} s"
                    + (f" -- {e.get('error')}" if e.get("error") else ""))
            elif t == "log" and e.get("level") in ("warning", "error"):
                log(f"  {e.get('level')}: [{e.get('stage')}] {str(e.get('msg'))[:200]}")
            elif t == "run_end":
                return str(e.get("status"))
        if api.running(p) is None and time.time() - last > grace:
            return "lost"
        time.sleep(poll)


def run(api, p, plan, device: str) -> bool:
    """Run a plan to its end; True when every stage of every workflow is done."""
    if not plan.jobs:
        return True
    log(f"plan: {len(plan.jobs)} jobs: {', '.join(j.stage for j in plan.jobs)}")
    info = api.start_run(p, plan, device)
    log(f"run {info.run_id} on {info.device}, pid {info.pid}")
    status = follow(api, p, info)
    api.refresh(p)
    bad = {f"{wf}/{st}": s.state for (wf, st), s in api.states(p).items() if s.state != "done"}
    log(f"run {status}; not done: {bad or 'none'}")
    return status == "done" and not bad


def add_classic(api, p):
    return api.add_workflow(p, CLASSIC_NAME, "Classic stack", CLASSIC_PARAMS)


def finish(api, p, full_id: int, classic_id: int) -> None:
    """The definition the tests read: Full alone, the next workflow id 2 (the tests' added workflows are 2, 3, 4)."""
    from astro_gpu.core import project2
    api.delete_workflow(p, classic_id)                 # its results stay in the store
    p.next_workflow_id = full_id + 1
    api.set_current(p, full_id)
    api.save(p)
    # the deleted workflow's "previous results" would be given to the first workflow a test adds (same id)
    st = project2.read_state(p.root)
    st["last_keys"] = {k: v for k, v in st["last_keys"].items() if k == str(full_id)}
    project2.write_state(p.root, st)


def write_info(out: Path, data: Path, counts: dict, device: str, minutes: float) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = ""
    (out / INFO).write_text(json.dumps({
        "made_by": "tools/make_test_project.py", "date": datetime.now().isoformat(timespec="seconds"),
        "commit": commit, "data": str(data), "frames": counts, "device": device, "workflows": ["Full"],
        "also_in_store": {CLASSIC_NAME: {"template": "Classic stack", "params": CLASSIC_PARAMS}},
        "minutes": round(minutes, 1),
    }, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help=f"project folder (default {OUT})")
    ap.add_argument("--data", type=Path, default=DATA, help=f"raw frames: lights/, darks/, flats/ (default {DATA})")
    ap.add_argument("--device", default="cuda:0", help="device of the run (default cuda:0)")
    ap.add_argument("--force", action="store_true", help="delete an existing project folder first")
    ap.add_argument("--dry-run", action="store_true", help="create and plan, print the plan, run nothing")
    a = ap.parse_args(argv)
    out = a.out.resolve()
    if not out.is_relative_to(ALLOWED.resolve()):
        print(f"refused: the test project lives under {ALLOWED}")
        return 2
    if not (a.data / "lights").is_dir():
        print(f"no raw frames in {a.data}")
        return 2
    if (out / "project.astroproj").exists():
        if not a.force:
            print(f"{out} exists: --force rebuilds it")
            return 2
        log(f"deleting {out}")
        shutil.rmtree(out)

    # the engine's per-user state, before astro_gpu is imported (the detached worker inherits it)
    state = out.parent / "_state"
    state.mkdir(parents=True, exist_ok=True)
    os.environ["ASTRO_GPU_HOME"] = str(state)
    profiles = Path.home() / ".astro_gpu" / "camera_profiles"
    if profiles.is_dir():
        shutil.copytree(profiles, state / "camera_profiles", dirs_exist_ok=True)
    # the offline Gaia extracts the user installed (Settings › Catalogs & astrometry), read in place: no download
    extracts = Path.home() / ".astro_gpu" / "cache" / "gaia_extracts"
    if extracts.is_dir():
        os.environ.setdefault("ASTRO_GPU_GAIA_EXTRACTS", str(extracts))
    sys.path.insert(0, str(ROOT))
    from astro_gpu import api

    t0 = time.time()
    p = api.create_project(out, "heart_small", TARGET)
    scan = api.scan_folder(a.data / "lights")
    counts = {k: len(v) for k, v in scan.frames.items()}
    log(f"scan: {counts}, errors {len(scan.errors)}")
    api.add_frames(p, scan)
    p.description = "Test project of the client tests (tools/make_test_project.py)"
    p.devices["main"] = a.device
    full = api.add_workflow(p, "Full", "Full")
    api.set_param(p, [full.id], "denoise", "iterations", DENOISE_ITERATIONS)
    classic = add_classic(api, p)
    api.save(p)
    plan = api.plan_all(p, a.device)
    if a.dry_run:
        log(f"plan: {len(plan.jobs)} jobs: {', '.join(j.stage for j in plan.jobs)}")
        return 0
    if not run(api, p, plan, a.device):
        return 1
    finish(api, p, full.id, classic.id)
    write_info(out, a.data, counts, a.device, (time.time() - t0) / 60)
    log(f"done in {(time.time() - t0) / 60:.1f} min: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
