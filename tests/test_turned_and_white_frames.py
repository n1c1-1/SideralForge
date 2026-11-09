"""Frames turned by about half a turn from the reference (issue #63), white frames (issue #64), a block that fails
before it starts (issue #65)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from astro_gpu.core import frame_weights as fwm
from astro_gpu.core import runner2
from astro_gpu.stages import register
from astro_gpu.stages.calibrate import _frame_stat
from astro_gpu.stages.register import fit_frame, fit_half_turn
from astro_gpu.stages.stack import quality_weights, saturated_frames, select_frames

W, H = 1920, 1280


def field(n=400, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform([0, 0], [W, H], (n, 2))
    flux = np.sort(rng.pareto(1.5, n) + 1)[::-1] * 1000
    return np.c_[xy, flux, np.full(n, 2.0)]


def turned(ref, ang, shift=(60.0, -40.0)):
    t = np.deg2rad(ang)
    c = np.array([W / 2, H / 2])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    xy = (ref[:, :2] - c) @ R.T + c + np.array(shift)
    inside = (xy[:, 0] > 0) & (xy[:, 0] < W) & (xy[:, 1] > 0) & (xy[:, 1] < H)
    cat = np.c_[xy, ref[:, 2:]][inside]
    cat[:, :2] += np.random.default_rng(1).normal(0, 0.15, (len(cat), 2))
    return cat


@pytest.mark.parametrize("ang", [178.0, 181.5])
def test_a_frame_turned_by_about_half_a_turn_is_aligned(ang):
    ref = field()
    cat = turned(ref, ang)
    tree = cKDTree(ref[:, :2])
    r = fit_frame(cat, ref, tree, W, H)
    assert r is None or r[1] < 0.3 * len(cat)                # what a search within 45 degrees gives
    r, at = fit_half_turn(cat, ref, tree, W, H)
    assert r is not None and r[1] > 0.8 * len(cat) and r[2] < 0.6
    assert abs(abs(at - 180.0) - abs(ang - 180.0)) < 1.0


def test_the_next_frames_of_a_series_are_aligned_without_a_search(monkeypatch):
    ref = field()
    tree = cKDTree(ref[:, :2])
    _r, at = fit_half_turn(turned(ref, 178.0), ref, tree, W, H)

    def no_search(*_a, **_k):
        raise AssertionError("searched")
    monkeypatch.setattr(register, "initial_rotation", no_search)
    r, again = fit_half_turn(turned(ref, 178.05, (70.0, -35.0)), ref, tree, W, H, at)
    assert r is not None and again == at


def test_a_frame_of_another_field_is_not_taken_for_a_half_turn():
    ref = field()
    r, at = fit_half_turn(field(seed=7), ref, cKDTree(ref[:, :2]), W, H)
    assert r is None and at is None


def test_a_white_frame_has_no_statistics_but_a_normal_one_has():
    white = np.full(4000, 65000.0)
    almost = white.copy()
    almost[:20] = 500.0 + np.arange(20)
    assert np.isnan(_frame_stat(white)).all() and np.isnan(_frame_stat(almost)).all()
    sky = 500.0 + np.random.default_rng(0).normal(0, 10, 4000)
    assert np.isfinite(_frame_stat(sky)).all()


def night(n=6):
    stats = np.ones((n, 4, 2))
    stats[0] = np.nan
    rms = np.full(n, 0.2)
    rms[0] = np.inf
    return stats, rms, np.ones(n), np.full(n, 3.0)


def test_a_white_frame_is_left_out_with_its_reason():
    stats, rms, transp, fwhm = night()
    assert saturated_frames(stats).tolist() == [True] + [False] * 5
    fs = fwm.select(np.array([0.0, 1, 1, 1, 1, 1]), aligned=np.isfinite(rms), saturated=saturated_frames(stats))
    assert fs.codes[0] == "saturated" and fs.reasons[0] == "saturated frame" and fs.kept.sum() == 5
    _w, reasons = select_frames(transp, fwhm, rms, stats, 0.6, 2.0)
    assert reasons[0] == "saturated frame" and not any(reasons[1:])


def test_a_white_frame_does_not_change_the_measure_of_the_others():
    stats, rms, transp, fwhm = night()
    noise = np.full(6, 2.0)
    noise[0] = np.nan
    weight, sig = quality_weights(transp, fwhm, rms, stats, noise)
    assert (sig[1:] == 2.0).all() and weight[0] == 0.0 and (weight[1:] == 1.0).all()


def test_a_block_that_fails_before_it_starts_is_recorded_as_failed(tmp_path):
    job = SimpleNamespace(stage="stack", key="k1")
    me = SimpleNamespace(store=SimpleNamespace(tmp_dir=lambda s, k: tmp_path / f"{s}-{k}.tmp"))
    runner2._Worker._mark_failed(me, job, "ValidationError: gain", "trace")
    m = json.loads((tmp_path / "stack-k1.tmp" / "manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "failed" and m["error"] == "ValidationError: gain" and m["stage"] == "stack"
