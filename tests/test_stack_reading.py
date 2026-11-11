"""The Stack stage reads each calibrated frame by one explicit read into a buffer, no longer through a memory map of
the cache (issue #125): every file it writes is the same to the bit as when it read through the map, with the drizzle
(its own rejection, the integration's) and the Classic method, on mosaic frames and on frames already in colour; and
the log says the time of each pass."""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pytest
from test_stacking import copy_project, run_stage, toy_night

from astro_gpu.stages import calibrate as cal
from astro_gpu.stages import stack as st

#: what a file holds that tells how long, not what was computed
TIMES = {"seconds", "started", "finished", "duration"}


@pytest.fixture(scope="module")
def night(tmp_path_factory):
    root = tmp_path_factory.mktemp("reading")
    proj, _ = toy_night(root / "p", n=8, n_stars=140, noisy=2, hazy=4, trail=3, size=(192, 256))
    return root, proj


def _content(f: Path):
    if f.suffix == ".npy":
        return np.load(f)
    if f.suffix == ".npz":
        with np.load(f) as z:
            return {k: z[k] for k in z.files}
    if f.suffix == ".json":
        def strip(o):
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if k not in TIMES}
            return [strip(v) for v in o] if isinstance(o, list) else o
        return strip(json.loads(f.read_text(encoding="utf-8")))
    return f.read_bytes()


def _same(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    return a == b


def _same_files(a: Path, b: Path) -> list[str]:
    files = sorted(f.name for f in a.iterdir() if f.is_file())
    assert files == sorted(f.name for f in b.iterdir() if f.is_file())
    assert {"drz_full.npy", "drz_A.npy", "drz_B.npy", "drz_weight.npy", st.SATURATION, st.REJECTION, st.DIAG} <= set(files)
    for f in files:
        assert _same(_content(a / f), _content(b / f)), f
    return files


def _mapped(monkeypatch) -> None:
    """The Stack stage reading through the memory map, as before."""
    monkeypatch.setattr(st, "CachedFrames", functools.partial(cal.CachedFrames, mapped=True))


@pytest.mark.parametrize("settings", [{}, {"drizzle_rejection": "integration"}, {"method": "classic"}])
def test_read_or_mapped_the_stack_writes_the_same_files(night, monkeypatch, settings):
    root, proj = night
    name = "_".join(settings.values()) or "drizzle"
    params = st.StackParams(scale=1.5, pixfrac=0.7, gain=1.0, read_noise=6.0, half_block=2, **settings)
    read = copy_project(proj.root, root / f"read_{name}")
    run_stage(read, st.Stack(), params)
    _mapped(monkeypatch)
    mapped = copy_project(proj.root, root / f"mapped_{name}")
    run_stage(mapped, st.Stack(), params)
    _same_files(read.stage_dir("stack"), mapped.stage_dir("stack"))
    log = (read.logs / "journal.md").read_text(encoding="utf-8")
    passes = {"": ["pass 1", "pass 2", "pass 3"],
              "integration": ["pass 1", "classic: debayering and resampling", "pass 3"],
              "classic": ["normalisation", "classic: debayering and resampling"]}[settings.get("drizzle_rejection") or
                                                                                    settings.get("method", "")]
    for what in passes:
        assert f"{what} done in " in log and " s reading the frames" in log, what
    if settings:
        assert "classic: rejection and means done in " in log


def test_frames_already_in_colour_read_or_mapped_the_same(tmp_path, monkeypatch):
    """The Classic stack of JPEG frames (three planes per frame in their own cache), through the whole chain."""
    from test_colour_frames import jpeg_frames

    from astro_gpu.core.project import Inputs, Project
    from astro_gpu.core.runner import Pipeline
    from astro_gpu.stages.inventory import Inventory
    from astro_gpu.stages.register import Register
    lights = tmp_path / "L"
    jpeg_frames(lights, n=5)
    pr = Project.create(tmp_path / "p", "p", Inputs(lights=[str(lights)]))
    pr.config.devices = {"main": "cpu"}
    pr.config.params = {"calibrate": {"workers": 1}, "register": {"workers": 1}}
    pr.save()
    Pipeline(pr, [Inventory(), cal.Calibrate(), Register()]).run()
    assert json.loads((pr.stage_dir("calibrate") / "cache.json").read_text())["rgb_path"]
    params = st.StackParams(method="classic")
    read = copy_project(pr.root, tmp_path / "read")
    run_stage(read, st.Stack(), params)
    _mapped(monkeypatch)
    mapped = copy_project(pr.root, tmp_path / "mapped")
    run_stage(mapped, st.Stack(), params)
    _same_files(read.stage_dir("stack"), mapped.stage_dir("stack"))


def test_a_frame_is_one_read_of_its_bytes(tmp_path):
    """Frame i is bytes i x H x W x 2 to (i + 1) x H x W x 2 of the cache, float16, as float32; nothing stays open."""
    n, H, W = 3, 5, 7
    a = (np.arange(n * H * W, dtype=np.float32).reshape(n, H, W) * 1.7).astype(np.float16)
    a[1, 2, 3] = np.nan
    path = tmp_path / "c.f16"
    a.tofile(path)
    frames = cal.CachedFrames({"path": str(path), "n": n, "height": H, "width": W})
    for i in (2, 0, 1):
        got = frames[np.int64(i)]
        assert got.dtype == np.float32 and np.array_equal(got, a[i].astype(np.float32), equal_nan=True)
    assert frames[0] is not frames[0] and len(frames) == n and frames.seconds > 0
    with pytest.raises(IndexError):
        frames[n]
    path.unlink()                                       # no handle left open on the cache (Windows refuses otherwise)
