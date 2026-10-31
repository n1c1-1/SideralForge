"""Test sets: small, frozen copies of raw sessions (other cameras and instruments, and the user's own), processed as
projects that stay apart from the user's (their own list of projects and camera profiles, ASTRO_GPU_HOME).

    python tools/qc/jeux.py build [NAME ...]        copy the frames chosen for each set and create its project
    python tools/qc/jeux.py run [NAME ...] [--until STAGE] [--force STAGE ...] [--label LABEL]
                                                     process the sets, then render them with the 'Natural' preset
    python tools/qc/jeux.py list                     the sets, what they test and their last run

Everything lives under ROOT (the T9 since 2026-10-30): donnees/<set>/{lights,darks,flats,bias}, projets/<set>,
scratch/<set>, etat/ (ASTRO_GPU_HOME), journaux/<set>_<label>.log, runs.json. The sources are only read.
A set's frames are chosen evenly over its session (drift, transparency and sky changes stay representative). The
test projects have their own settings (TEST_PARAMS): learned star removal off, shorter denoiser training.
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

ROOT = Path(os.environ.get("ASTRO_TESTS", r"F:\astro_essais"))
os.environ["ASTRO_GPU_HOME"] = str(ROOT / "etat")            # before astro_gpu is imported
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from astro_gpu.core.project import Inputs, Project, Target, frame_kind  # noqa: E402

D = r"D:\astro_donnees_test"
#: name -> source folder, (subfolder, how many) per kind, target, what the set tests
SETS: dict[str, dict] = {
    "canon6d_m51": dict(
        src=rf"{D}\canon6d_m51_n200\WhirlpoolGalaxy", lights=("lights", 30), darks=("darks", 15),
        flats=("flats", 12), bias=("biases", 10), target="M 51",
        what="Canon EOS 6D (CR2, full frame), Newton 200/1000, galaxy, complete calibration"),
    "canon40d_rosette": dict(
        src=rf"{D}\canon40da_rosette_72ed\RosetteNebula", lights=("lights", 30), darks=("darks", 15),
        flats=("flats", 17), bias=("biases", 15), target="NGC 2244",
        what="Canon EOS 40D (CR2, old 10 Mpx sensor), 72ED refractor without flattener; the Rosette, as the user's"),
    "asi2600mc_m8": dict(
        src=rf"{D}\siril_asi2600mc_m8", lights=("Brutes_180s", 15), darks=("Darks_180s", 15),
        flats=("Flats_3s", 15), bias=("Offsets_3s", 15), target="M 8",
        what="ZWO ASI2600MC Pro (cooled colour camera, FITS), 388 mm, complete calibration; 15 frames only"),
    "seestar_m31": dict(
        src=rf"{D}\seestar_s50_s30_m31_m42\VergleicheS50vsS30\S50\M 31", lights=("M 31_sub1", 40), target="M 31",
        what="ZWO Seestar S50 smart telescope (alt-azimuth: the field rotates), 250 mm, portrait 1080x1920, "
             "no calibration; M31 as the user's"),
    "canonra_northamerica": dict(
        src=rf"{D}\canon_ra_rokinon135_northamerica", lights=("LIGHTS", 30), darks=("DARKS", 14),
        target="NGC 7000",
        what="Canon EOS Ra (CR3, H-alpha modified), Rokinon 135 mm f/2 lens, wide field, darks only"),
    "stellina_ic1318": dict(
        src=rf"{D}\stellina_ic1318", lights=("IC1318-20221026", 40), target="IC 1318",
        what="Vaonis Stellina (alt-azimuth with field rotator), IMX178, 400 mm, exposures in ms, no calibration"),
    "stellina_ngc891": dict(
        src=rf"{D}\stellina_ngc891", lights=("NGC891-20221002", 40), target="NGC 891",
        what="Vaonis Stellina, BGGR pattern (the other Stellina session is RGGB), edge-on galaxy"),
    "iris_a7s": dict(
        src=r"D:\photos\2.3.25_Nebuleuse_iris\raw", lights=("signal", 60), darks=("dark", 15), flats=("flat", 15),
        target="NGC 7023",
        what="The user's Iris (Sony A7S, Newton 130/900), 60 of 1242 frames: the reference for 'not worse'"),
    "heart_a7s": dict(
        src=r"F:\photos\heart_nebula\raw", lights=("signal", 40), darks=("dark", 15), flats=("flat", 15),
        target="IC 1805",
        what="The user's Heart (Sony A7S, refractor), 40 frames: the reference for 'not worse'"),
}


#: settings of every test project: the learned star removal off (35 min, judged on its own), the denoiser trained
#: 6 000 steps instead of 24 000 (8 min instead of 33; a little more noise left, the same for every run compared)
TEST_PARAMS = {"star_net": {"enabled": False}, "denoise": {"iterations": 6000}}


def _pick(folder: Path, n: int) -> list[Path]:
    files = sorted(f for f in folder.iterdir() if f.is_file() and frame_kind(f))
    if n >= len(files):
        return files
    return [files[round(i * (len(files) - 1) / (n - 1))] for i in range(n)] if n > 1 else files[:1]


def build(names: list[str]) -> None:
    prof_src, prof_dst = Path.home() / ".astro_gpu" / "camera_profiles", ROOT / "etat" / "camera_profiles"
    if prof_src.is_dir() and not prof_dst.exists():
        shutil.copytree(prof_src, prof_dst)          # the user's camera profiles, as the real projects see them
    for name in names:
        s = SETS[name]
        data, proj = ROOT / "donnees" / name, ROOT / "projets" / name
        if (proj / "project.json").exists():
            print(f"{name}: already built")
            continue
        inputs = {}
        for kind in ("lights", "darks", "flats", "bias"):
            if kind not in s:
                continue
            sub, n = s[kind]
            chosen = _pick(Path(s["src"]) / sub, n)
            dst = data / kind
            dst.mkdir(parents=True, exist_ok=True)
            for f in chosen:
                if not (dst / f.name).exists() or (dst / f.name).stat().st_size != f.stat().st_size:
                    shutil.copy2(f, dst / f.name)
            inputs[kind] = [str(dst)]
            print(f"{name}: {len(chosen)} {kind}")
        p = Project.create(proj, name, Inputs(**inputs), Target(name=s["target"]), ROOT / "scratch" / name)
        p.config.description = s["what"]
        p.config.params = TEST_PARAMS
        p.save()


def _runs() -> dict:
    f = ROOT / "runs.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _save_runs(d: dict) -> None:
    f = ROOT / "runs.json"
    f.write_text(json.dumps(d, indent=1), encoding="utf-8")


def run(names: list[str], until: str | None, force: list[str], label: str) -> None:
    py = sys.executable
    env = dict(os.environ, ASTRO_GPU_HOME=str(ROOT / "etat"), PYTHONUNBUFFERED="1")
    (ROOT / "journaux").mkdir(parents=True, exist_ok=True)
    for name in names:
        proj = ROOT / "projets" / name
        log = ROOT / "journaux" / f"{name}_{label}.log"
        cmd = [py, "-m", "astro_gpu.cli", "run", str(proj)] + (["--until", until] if until else []) + \
              (["--force", *force] if force else [])
        t0 = time.time()
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M} {' '.join(cmd)}\n")
            lf.flush()
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
            rr = None
            if rc == 0 and not until:
                rr = subprocess.call([py, "-m", "astro_gpu.cli", "render", str(proj), "--preset", "Natural",
                                      "--name", f"natural_{label}"], cwd=REPO, env=env, stdout=lf,
                                     stderr=subprocess.STDOUT)
        st = _status(proj)
        d = _runs()
        d.setdefault(name, {})[label] = {"when": datetime.now().isoformat(timespec="seconds"), "exit": rc,
                                         "render_exit": rr, "minutes": round((time.time() - t0) / 60, 1),
                                         "stages": st}
        _save_runs(d)
        bad = {k: v for k, v in st.items() if v["status"] != "done"}
        print(f"{name}: exit {rc}, {d[name][label]['minutes']} min" + (f", not done: {list(bad)}" if bad else ""),
              flush=True)


def _status(proj: Path) -> dict:
    out = {}
    for m in sorted((proj / "work").glob("*/manifest.json")):
        try:
            j = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out[m.parent.name] = {"status": j.get("status"), "minutes": round((j.get("duration") or 0) / 60, 1),
                              "error": (j.get("error") or "")[:400]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "list"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--until")
    ap.add_argument("--force", nargs="+", default=[])
    ap.add_argument("--label", default=datetime.now().strftime("%Y%m%d_%H%M"))
    a = ap.parse_args()
    names = a.names or list(SETS)
    unknown = [n for n in names if n not in SETS]
    if unknown:
        raise SystemExit(f"unknown set(s): {unknown}; known: {list(SETS)}")
    if a.cmd == "build":
        build(names)
    elif a.cmd == "run":
        run(names, a.until, a.force, a.label)
    else:
        runs = _runs()
        for n in names:
            last = sorted(runs.get(n, {}).items(), key=lambda kv: kv[1]["when"])[-1:] or [("-", {})]
            print(f"{n:22s} {SETS[n]['what'][:70]:70s} last: {last[0][0]} exit {last[0][1].get('exit', '-')}")


if __name__ == "__main__":
    main()
