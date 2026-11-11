"""The workers of the Calibration and Registration pools start with one maths thread each (issue #125): OpenBLAS no
longer reserves a buffer per processor thread in each of them (about 1 GB a worker on 16 threads). Nor do they load
PyTorch with the runner's module, which they import again as they start and whose maths they never use. Their results
are the same to the bit: a real run of both stages on the CPU over small FITS frames with stars, with real worker
processes, compared file by file with the run whose workers keep every thread, and with the run through a runner's
worker whose pools' workers load PyTorch, as before."""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pytest

import astro_gpu
from astro_gpu import api
from astro_gpu.core import devices, project2, runner2
from astro_gpu.core import workflow as wfl
from astro_gpu.core.project2 import Session
from astro_gpu.core.store import Keys, Store

H, W, N = 360, 480, 6
#: what a manifest holds that tells which run, when and how long, not what was computed
TIMES = {"run", "started", "finished", "duration"}
#: the folder astro_gpu is imported from, for the runner's worker processes the tests start
SOURCE = Path(astro_gpu.__file__).resolve().parent.parent


def _fits(path: Path, a: np.ndarray, kind: str, i: int) -> None:
    from astropy.io import fits
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.Header()
    for k, v in {"BAYERPAT": "RGGB", "EXPTIME": 30.0 if kind != "flat" else 1.0, "GAIN": 100.0, "INSTRUME": "TESTCAM",
                 "IMAGETYP": kind.upper(), "DATE-OBS": f"2026-08-16T22:{i:02d}:00"}.items():
        h[k] = v
    fits.PrimaryHDU(np.clip(a, 0, 65535).astype(np.uint16), h).writeto(path, overwrite=True)


def session(raw: Path) -> dict[str, Path]:
    """N lights of a field of 300 stars dithered by a few pixels on an RGGB sensor, 3 darks, 3 flats."""
    rng = np.random.default_rng(7)
    sx, sy = rng.uniform(12, W - 12, 300), rng.uniform(12, H - 12, 300)
    flux = np.exp(rng.uniform(np.log(800), np.log(30000), 300))
    yy, xx = np.mgrid[0:H, 0:W]
    vignet = 1.0 - 0.2 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    for i in range(N):
        dx, dy = rng.uniform(-4, 4, 2)
        img = np.full((H, W), 600.0)
        for x0, y0, f in zip(sx + dx, sy + dy, flux):
            ys, xs = slice(max(int(y0) - 6, 0), int(y0) + 7), slice(max(int(x0) - 6, 0), int(x0) + 7)
            img[ys, xs] += f / (2 * np.pi * 1.3 ** 2) * np.exp(-0.5 * ((xx[ys, xs] - x0) ** 2 + (yy[ys, xs] - y0) ** 2)
                                                             / 1.3 ** 2)
        img = img * vignet + 100 + rng.normal(0, 1, img.shape) * np.sqrt(img + 36)
        _fits(raw / "L" / f"L{i}.fits", img, "light", i)
    for i in range(3):
        _fits(raw / "D" / f"D{i}.fits", 100 + rng.normal(0, 6, (H, W)), "dark", i)
        _fits(raw / "F" / f"F{i}.fits", 100 + 20000 * vignet + rng.normal(0, 140, (H, W)), "flat", i)
    return {"lights": raw / "L", "darks": raw / "D", "flats": raw / "F"}


def run(root: Path, folders: dict[str, Path], command: list[str] | None = None) -> dict[str, Path]:
    """Calibration and Registration of the session on the CPU, two workers each: {stage: its result's folder}. In this
    process, or, given ``command``, in the process it starts with the project and the plan (the runner's worker)."""
    p = project2.create(root, "p")
    p.devices = {"main": "cpu", "aux": ""}
    p.sessions.append(Session(1, "n", lights=[str(folders["lights"])], darks=[str(folders["darks"])],
                              flats=[str(folders["flats"])]))
    p.add_workflow("Main", wfl.DEFAULT_OPTIONS)
    p.set_blocks(1, [wfl.node("calibration"), wfl.node("registration")])
    p.workflows[0].params = {"calibrate": {"workers": 2}, "register": {"workers": 2}}
    project2.save(p)
    plan = api.plan_workflows(p, [1], device="cpu")
    plan_file = runner2.prepare_run(p, plan, "cpu").plan_file
    if command is None:
        assert runner2.run_plan(p.root, plan_file, echo=False) == "done"
    else:
        done = subprocess.run([*command, str(p.root), str(plan_file)], cwd=SOURCE, capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(SOURCE), "CUDA_VISIBLE_DEVICES": "-1"})
        assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    k, s = Keys(project2.load(p.root)), Store(p.root)
    return {st: s.dir(st, k.key(1, st)) for st in ("calibrate", "register")}


def _content(f: Path, root: Path):
    """What a result file holds, the times of its run left out and its project's folder named alike."""
    if f.suffix == ".npy":
        return np.load(f)
    if f.suffix == ".npz":
        with np.load(f) as z:
            return {k: z[k] for k in z.files}
    if f.suffix == ".json":
        def strip(o):
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if k not in TIMES}
            if isinstance(o, str):
                return o.replace(str(root), "<project>")
            return [strip(v) for v in o] if isinstance(o, list) else o
        return strip(json.loads(f.read_text(encoding="utf-8")))
    return f.read_bytes()


def _same(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    return a == b


def _same_results(one: dict[str, Path], other: dict[str, Path], root_one: Path, root_other: Path) -> None:
    """Two runs of ``run``: the same files to the bit, the calibrated frames' cache included."""
    for stage in ("calibrate", "register"):
        a, b = one[stage], other[stage]
        files = sorted(f.relative_to(a) for f in a.rglob("*") if f.is_file())
        assert files == sorted(f.relative_to(b) for f in b.rglob("*") if f.is_file())
        assert {"master_dark.npy", "master_flat.npy", "frame_stats.npy"} <= {f.name for f in files} or \
            "reg.npz" in {f.name for f in files}, files
        for f in files:
            assert _same(_content(a / f, root_one), _content(b / f, root_other)), f"{stage}/{f}"
    with np.load(one["register"] / "reg.npz") as z:      # a real registration: every frame aligned, its stars fitted
        assert np.isfinite(z["rms"]).all() and np.isfinite(z["fwhm_moffat"]).all() and len(z["rms"]) == N
    # the calibrated frames themselves (the cache the pool's workers write, in the scratch folder)
    caches = [json.loads((d["calibrate"] / "cache.json").read_text(encoding="utf-8"))["path"] for d in (one, other)]
    assert Path(caches[0]).read_bytes() == Path(caches[1]).read_bytes() and Path(caches[0]).stat().st_size > 0


def test_one_maths_thread_in_the_workers_gives_the_same_files(tmp_path, monkeypatch):
    folders = session(tmp_path / "raw")
    one = run(tmp_path / "one", folders)
    # the workers as before: every thread (no variable of this test session limits them either)
    monkeypatch.setattr(devices, "one_maths_thread", contextlib.nullcontext)
    for k in devices.MATHS_THREADS:
        monkeypatch.delenv(k, raising=False)
    every = run(tmp_path / "every", folders)
    _same_results(one, every, tmp_path / "one", tmp_path / "every")


#: the runner's worker as it was: PyTorch loaded with the runner's module, so that the pools' workers, which import
#: the main module of their parent again (here this file), loaded it too
WITH_TORCH = """import sys

import torch  # noqa: F401

from astro_gpu.core.runner2 import main

if __name__ == "__main__":
    sys.exit(main())
"""


def test_the_runners_workers_without_pytorch_give_the_same_files(tmp_path):
    """Both stages run by the runner's worker process as the program starts it (``python -m astro_gpu.core.runner2``),
    whose pools' workers no longer load PyTorch, and by a worker that loads it with its main module, as before: the
    same files to the bit."""
    folders = session(tmp_path / "raw")
    new = run(tmp_path / "new", folders, [sys.executable, "-m", "astro_gpu.core.runner2"])
    old_main = tmp_path / "runner_with_torch.py"
    old_main.write_text(WITH_TORCH, encoding="utf-8")
    old = run(tmp_path / "old", folders, [sys.executable, str(old_main)])
    _same_results(new, old, tmp_path / "new", tmp_path / "old")


def test_what_the_pools_workers_import_holds_no_pytorch():
    """A spawned worker of the runner imports the runner's module (its parent's main module, under another name) and
    the modules of the stages whose functions it runs: none of them loads PyTorch."""
    code = ("import sys, astro_gpu.core.runner2, astro_gpu.stages.calibrate, astro_gpu.stages.register; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=SOURCE, capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": str(SOURCE), "CUDA_VISIBLE_DEVICES": "-1"})
    assert out.stdout.strip() == "False"


def _worker_threads(_):
    import numpy  # noqa: F401 - what the pool's workers load
    import psutil
    m = psutil.Process().memory_info()
    return {k: os.environ.get(k) for k in devices.MATHS_THREADS}, int(getattr(m, "private", m.vms))


def test_the_workers_get_one_thread_and_this_process_keeps_its_variables(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with devices.one_maths_thread():
        pool = get_context("spawn").Pool(1)
    with pool:
        env, _ = pool.map(_worker_threads, [0])[0]
    assert env == {k: "1" for k in devices.MATHS_THREADS}
    assert os.environ["OMP_NUM_THREADS"] == "3" and "OPENBLAS_NUM_THREADS" not in os.environ


@pytest.mark.skipif(os.name != "nt" or (os.cpu_count() or 1) < 8, reason="the reservation is measured on Windows, "
                    "where it is counted as private memory, with many processor threads")
def test_a_worker_reserves_far_less(monkeypatch):
    for k in devices.MATHS_THREADS:
        monkeypatch.delenv(k, raising=False)
    with get_context("spawn").Pool(1) as pool:
        every = pool.map(_worker_threads, [0])[0][1]
    with devices.one_maths_thread():
        pool = get_context("spawn").Pool(1)
    with pool:
        one = pool.map(_worker_threads, [0])[0][1]
    assert one < every - 200 * 2 ** 20
