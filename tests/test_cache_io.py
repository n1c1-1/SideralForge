"""The calibrated cache without memory maps when the machine's memory is short (issue #125, ``memory_short``; through
a map, every frame written or read stayed in the workers' working sets): Calibration's workers write each frame by one
plain write, Registration's read each frame by one plain read. The same cache and the same results to the bit as
through the maps, on a real run of both stages on the CPU with real worker processes (tests/test_maths_threads.py's
session), on mosaic frames and on frames already in colour."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_maths_threads import _same, _same_results, run, session

from astro_gpu.core import devices
from astro_gpu.stages import calibrate as cal
from astro_gpu.stages import register as rg


def _short(monkeypatch, short: bool) -> None:
    """The memory short (3 GB available) or not, whatever the machine's: plain writes and reads, or the maps."""
    for m in (cal, rg):
        monkeypatch.setattr(m, "memory_short", lambda *a: 3.0 if short else None)


def test_plain_writes_give_the_same_cache_and_results(tmp_path, monkeypatch):
    folders = session(tmp_path / "raw")
    _short(monkeypatch, True)
    plain = run(tmp_path / "plain", folders)
    _short(monkeypatch, False)
    mapped = run(tmp_path / "mapped", folders)
    _same_results(plain, mapped, tmp_path / "plain", tmp_path / "mapped")
    events = next((tmp_path / "plain").rglob("events.jsonl")).read_text(encoding="utf-8")
    assert "the calibrated frames (0.0 GB) are written straight to the disk" in events


def test_frames_already_in_colour_written_plain_the_same(tmp_path, monkeypatch):
    """The luminance cache and the three planes' cache of JPEG frames, both ways."""
    from test_colour_frames import jpeg_frames

    from astro_gpu.core.project import Inputs, Project
    from astro_gpu.core.runner import Pipeline
    from astro_gpu.stages.inventory import Inventory
    lights = tmp_path / "L"
    jpeg_frames(lights, n=4)
    files = {}
    for plain in (True, False):
        _short(monkeypatch, plain)
        pr = Project.create(tmp_path / f"p{plain}", "p", Inputs(lights=[str(lights)]))
        pr.config.devices = {"main": "cpu"}
        pr.config.params = {"calibrate": {"workers": 2}}
        pr.save()
        Pipeline(pr, [Inventory(), cal.Calibrate()]).run()
        info = json.loads((pr.stage_dir("calibrate") / "cache.json").read_text())
        files[plain] = [Path(info["path"]).read_bytes(), Path(info["rgb_path"]).read_bytes(),
                        np.load(pr.stage_dir("calibrate") / "frame_stats.npy")]
    assert files[True][0] == files[False][0] and files[True][1] == files[False][1] and len(files[True][1]) > 0
    assert _same(files[True][2], files[False][2])


def test_a_frame_written_is_its_astype_float16(tmp_path):
    """Every float32 value, those float16 rounds, overflows to infinity, takes to zero, NaN and the marks of the cache,
    written plain is the bytes ``astype(np.float16)`` gives, at the place of its frame."""
    n, shape = 3, (5, 7)
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((n, *shape)) * np.array([1e-6, 3e3, 1e5])[:, None, None]).astype(np.float32)
    x[1, 0, :4] = [np.nan, np.inf, -np.inf, cal.SAT_MARK]
    path = tmp_path / "c.f16"
    np.memmap(path, dtype=np.float16, mode="w+", shape=(n, *shape)).flush()
    w = cal.FrameWriter(path, n, shape, plain=True)
    with np.errstate(over="ignore"):                     # the overflows to infinity are meant
        for i in (2, 0, 1):
            w[i] = x[i]
        assert path.read_bytes() == x.astype(np.float16).tobytes()


@pytest.mark.parametrize("room,short", [(64, False), (20, True), (4, True)])
def test_without_maps_only_when_the_memory_is_short(monkeypatch, room, short):
    """A night of 796 frames of 12 Mpx (an 18 GB cache) and a pool of 12 workers (7.1 GB): through the maps with 64 GB
    to fill, as before; plainly with 20 GB or 4 GB, said in one line."""
    monkeypatch.setattr(devices, "memory_budget", lambda: (room * 2 ** 30, (room + 5) * 2 ** 30))
    assert (cal.memory_short(796 * 2848 * 4256 * 2, 12, 2848 * 4256) is not None) is short
    lines: list[str] = []
    ctx = SimpleNamespace(log=lambda m, level="info": lines.append(m))
    assert cal.plain_writes(ctx, 796 * 2848 * 4256 * 2, 12, 2848 * 4256) is short
    assert lines == ([f"memory: {room + 5:.1f} GB available, the calibrated frames (18.0 GB) are written straight to "
                      f"the disk"] if short else [])
    monkeypatch.setattr(devices, "memory_budget", lambda: None)
    assert cal.memory_short(10 ** 12, 12, 10 ** 8) is None
