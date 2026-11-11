"""The Classic stack within the machine's memory (issue #125): when the memory available is short, its resampled
frames go to a temporary file and its rejection runs on pieces of its bands; the result stays the same to the bit, the
file is gone after, and one plain line says so. Small sessions on the CPU, the memory available made short."""
from __future__ import annotations

import numpy as np
import pytest
from test_rejection_classic import NF, H, W, make_frames
from test_stacking import copy_project, run_stage, toy_night

from astro_gpu.core import stack_classic as sc

#: a machine whose memory leaves room for nothing: every frame on disk, pieces of PIECE_ALIGN pixels
SHORT = (0, 3 * 2 ** 30)


@pytest.fixture(scope="module")
def frames():
    return make_frames()


def _stack(frames, rejection="auto", band=1 << 20, used=range(NF), **kw):
    coef, fr, norm = frames
    return sc.stack_classic(lambda i: fr[i], coef, None, norm, np.ones(NF),
                            sc.ClassicParams(half_block=4, rejection=rejection, band_pixels=band), "cpu",
                            used=used, ch_of_k=[0, 1, 1, 2], H=H, W=W, masks=True, **kw)


def _same(a: sc.ClassicResult, b: sc.ClassicResult) -> None:
    for name in ("full", "A", "B", "weight", "saturation", "rejection_low", "rejection_high", "masks"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True), name
    assert a.frames == b.frames and a.noise == b.noise
    assert (a.rejected_pct, a.rejected_pct_rgb, a.tested) == (b.rejected_pct, b.rejected_pct_rgb, b.tested)


# band: the default (one band), and bands of 11 rows whose 968 pixels end off the blocks the CPU sums by
@pytest.mark.parametrize("rejection,band,used", [("auto", 1 << 20, range(NF)), ("auto", 1000, range(NF)),
                                                 ("winsorized", 1000, range(NF)), ("percentile", 1 << 20, range(5)),
                                                 ("gesdt", 1000, range(NF))])
def test_on_disk_and_in_pieces_the_result_is_the_same_to_the_bit(frames, tmp_path, rejection, band, used):
    ref = _stack(frames, rejection, band, used)
    lines: list[str] = []
    got = _stack(frames, rejection, band, used, memory=SHORT, spill=tmp_path, log=lines.append)
    assert isinstance(got.masks, np.memmap)                  # on disk indeed
    _same(ref, got)
    rows = max(1, band // W)
    n = len(sc._parts(H, W, rows, sc.PIECE_ALIGN))
    assert n > -(-H // rows)                                 # in pieces indeed
    assert lines == [f"memory: 3.0 GB available, the {len(used)} registered frames (0.0 GB) are kept on disk and "
                     f"the rejection is computed in {n} parts"]
    del got
    assert list(tmp_path.iterdir()) == []                    # the temporary files are gone with the arrays


def test_with_room_nothing_changes_and_nothing_is_said(frames, tmp_path):
    lines: list[str] = []
    res = _stack(frames, memory=(2 ** 34, 2 ** 35), spill=tmp_path, log=lines.append)
    assert not isinstance(res.masks, np.memmap) and lines == [] and list(tmp_path.iterdir()) == []


def test_the_plan_of_a_long_session():
    """754 frames of 4256 x 2848 (12 Mpx) on a machine with 9 GB to fill: 110.7 GB of resampled frames go to disk, the
    bands of 246 rows (1 Mpx) are cut into pieces a multiple of PIECE_ALIGN that hold at most the 9 GB."""
    Hs, Ws, nu, budget, disk = 2848, 4256, 754, 9 * 2 ** 30, 500 * 2 ** 30
    plan = sc.MemoryPlan.of((budget, 12 * 2 ** 30), nu, Hs, Ws, 4, False, (1 << 20) // Ws, "cpu", disk)
    assert plan.on_disk and plan.piece % sc.PIECE_ALIGN == 0
    assert nu * plan.piece * sc.REJECTION_BYTES + sc.OUTPUT_BYTES * Hs * Ws <= budget
    parts = sc._parts(Hs, Ws, (1 << 20) // Ws, plan.piece)
    assert parts[0][0] == 0 and parts[-1][1] == Hs * Ws and all(a[1] == b[0] for a, b in zip(parts[:-1], parts[1:]))
    assert plan.note == (f"memory: 12.0 GB available, the 754 registered frames (110.7 GB) are kept on disk and the "
                         f"rejection is computed in {len(parts)} parts")
    # without a disk with the room for them, the frames stay in memory (as before); the card's rejection is never cut
    assert not sc.MemoryPlan.of((budget, budget), nu, Hs, Ws, 4, False, 246, "cpu", 0).on_disk
    assert not sc.MemoryPlan.of((budget, budget), nu, Hs, Ws, 4, False, 246, "cpu", 100 * 2 ** 30).on_disk
    gpu = sc.MemoryPlan.of((budget, budget), 30, Hs, Ws, 4, False, 246, "cuda:0", disk)
    assert gpu.piece == 246 * Ws
    assert sc.MemoryPlan.of(None, nu, Hs, Ws, 4, False, 246, "cpu", disk) == sc.MemoryPlan(False, 246 * Ws)


@pytest.fixture(scope="module")
def small_night(tmp_path_factory):
    root = tmp_path_factory.mktemp("small")
    proj, _ = toy_night(root / "p", n=8, n_stars=140, noisy=2, hazy=4, trail=3, size=(192, 256))
    return root, proj


@pytest.mark.parametrize("settings", [{"method": "classic"}, {"drizzle_rejection": "integration"}])
def test_the_stack_stage_short_of_memory_gives_the_same_files(small_night, monkeypatch, settings):
    """The Stack stage, Classic or drizzle with the integration's rejection, on a machine with plenty of memory and on
    one with none to spare: the same files to the bit, the line in the journal, the scratch folder left empty."""
    from astro_gpu.core import devices
    from astro_gpu.stages.stack import REJECTION, Stack, StackParams
    root, proj = small_night
    name = "_".join(settings.values())
    params = StackParams(scale=1.0, pixfrac=1.0, gain=1.0, read_noise=6.0, half_block=2, **settings)
    roomy = copy_project(proj.root, root / f"roomy_{name}")
    run_stage(roomy, Stack(), params)
    monkeypatch.setattr(devices, "memory_budget", lambda: SHORT)
    short = copy_project(proj.root, root / f"short_{name}")
    ctx = run_stage(short, Stack(), params)
    a, b = roomy.stage_dir("stack"), short.stage_dir("stack")
    for f in ("drz_full.npy", "drz_A.npy", "drz_B.npy", "drz_weight.npy", "saturation.npy"):
        assert np.array_equal(np.load(a / f), np.load(b / f), equal_nan=True), f
    with np.load(a / REJECTION) as x, np.load(b / REJECTION) as y:
        assert all(np.array_equal(x[k], y[k]) for k in x.files)
    log = (short.logs / "journal.md").read_text(encoding="utf-8")
    assert "memory: 3.0 GB available, the 8 registered frames (0.0 GB) are kept on disk and the rejection is " \
           "computed in" in log
    assert list(ctx.scratch.iterdir()) == []
