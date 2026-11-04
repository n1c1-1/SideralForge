"""The samples of the star removal work item: StarNet against our method, on the bench cases
(docs/chantiers/etoiles_etat_de_lart.md).

    python tools/bench/stars_samples.py projects [--cases ...] [--device cuda:1]   our method on the bench
    python tools/bench/stars_samples.py export   [--cases ...]                    the input image of each case
    python tools/bench/stars_samples.py starnet  [--cases ...]                    StarNet on it

The input of every method is the same linear image: the Classic stack · denoise workflow of ``ours_express`` (the
stack minus its gradient, denoised by the pretrained network, times the Color stage's white balance; M8, without
Gaia today, keeps the camera's colours), divided by one factor so that its brightest pixel is 1 (StarNet clips above
1). ``projects`` adds to each bench project the workflows of our method (Full: the star model measured with Gaia and
the starless image; Full + neural stars: and our per-image neural star removal) and computes them.

StarNet 2 (command line, 2.6.2): ``--linear`` (it stretches with a midtones transfer, removes the stars, then undoes the
stretch), the starless image in the input's units and the star layer as its "unscreen" fraction U (the original is
starless + U x (1 - starless)), as its README says.

Files: F:/astro_essais/stars_compare/<case>/{input.fits, meta.json, starnet_starless.fits, starnet_unscreen.fits}.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

BENCH = Path("F:/astro_essais/bench")
OUT = Path("F:/astro_essais/stars_compare")
CASES = ("heart", "iris", "m51", "m8", "rosette", "seestar")
SOURCE = "Classic stack · denoise"
#: our method's workflows: name -> the Stars option
OURS = {"Full · denoise": "psf", "Full + neural stars · denoise": "psf_neural"}
STARNET = Path.home() / "astro_gpu_scratch/tools/starnet2_2.6.2/starnet2_win_2.6.2-0242_ORT_x64_cli/starnet2.exe"


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def _api():
    import os
    os.environ.setdefault("ASTRO_GPU_HOME", str(BENCH / "_state_probe"))
    from astro_gpu import api
    return api


def projects(cases, device: str) -> int:
    import make_test_project as mtp
    api = _api()
    bad = []
    for case in cases:
        root = BENCH / case / "ours_express"
        p = api.open_project(root)
        src = next((w for w in p.workflows if w.name == SOURCE), None)
        if src is None:
            log(f"{case}: no '{SOURCE}' workflow (tools/bench/denoise_projects.py first)")
            continue
        for name, stars in OURS.items():
            if not any(w.name == name for w in p.workflows):
                w = api.duplicate_workflow(p, src.id, name)
                api.set_option(p, w.id, "stars", stars)
                log(f"{case}: workflow '{name}' added (id {w.id})")
        api.save(p)
        plan = api.plan_all(p, device)
        log(f"{case}: {len(plan.jobs)} jobs: {', '.join(j.stage for j in plan.jobs)}")
        if not mtp.run(api, p, plan, device):
            bad.append(case)
    log(f"not complete: {bad or 'none'}")
    return 1 if bad else 0


def export(cases) -> int:
    from astropy.io import fits
    api = _api()
    for case in cases:
        p = api.open_project(BENCH / case / "ours_express")
        w = next(x for x in p.workflows if x.name == SOURCE)
        dn = api.stage_outputs(p, w.id, "denoise")
        col = api.stage_outputs(p, w.id, "color")
        img = np.load(Path(dn.dir) / "n2n_full.npy").astype(np.float32)
        M = np.load(Path(col.dir) / "M.npy") if col is not None else np.eye(3)
        img = np.einsum("ij,jhw->ihw", M, img)
        img = np.clip(img, 0, None)
        scale = float(img.max())
        d = OUT / case
        d.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU((img / scale).astype(np.float32)).writeto(d / "input.fits", overwrite=True)
        (d / "meta.json").write_text(json.dumps({"case": case, "scale": scale, "colour": col is not None,
                                                 "source": str(dn.dir), "shape": list(img.shape)}, indent=1))
        log(f"{case}: input.fits {img.shape[2]}x{img.shape[1]} (scale {scale:.1f}"
            f"{'' if col is not None else ', camera colours: no Color result'})")
    return 0


def starnet(cases) -> int:
    for case in cases:
        d = OUT / case
        t = time.time()
        r = subprocess.run([str(STARNET), "--input", str(d / "input.fits"), "--output", str(d / "starnet_starless.fits"),
                            "--unscreen", str(d / "starnet_unscreen.fits"), "--linear", "--quiet"],
                           capture_output=True, text=True, cwd=str(STARNET.parent))
        ok = (d / "starnet_starless.fits").exists()
        log(f"{case}: StarNet {'done' if ok else 'FAILED'} in {time.time() - t:.0f} s")
        if not ok or r.returncode:
            log((r.stdout + r.stderr)[-1500:])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("projects", "export", "starnet"))
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args(argv)
    cases = a.cases.split(",")
    if a.cmd == "projects":
        return projects(cases, a.device)
    if a.cmd == "export":
        return export(cases)
    return starnet(cases)


if __name__ == "__main__":
    sys.exit(main())
