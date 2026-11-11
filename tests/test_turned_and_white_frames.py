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
from astro_gpu.stages.register import apply_poly, fit_any_turn, fit_frame, fit_half_turn
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


# ------------------------------------------------------------------- a turn by any angle (issue #63, second case)

def seen(ref, ang, shift=(60.0, -40.0), scale=1.0002, seed=1, size=(W, H)):
    """The reference's stars seen by a frame of a camera turned by ``ang`` degrees: shifted, 0.02 % larger, a tenth of
    the stars missed, a tenth of spurious detections added, fluxes and positions noisy, the brightest first as the
    detection gives them. Returns the catalogue and, for its true stars, their reference positions."""
    w, h = size
    rng = np.random.default_rng(seed)
    t = np.deg2rad(ang)
    c = np.array([w / 2, h / 2])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    xy = (ref[:, :2] - c) @ R.T * scale + c + np.array(shift)
    keep = (xy[:, 0] > 0) & (xy[:, 0] < w) & (xy[:, 1] > 0) & (xy[:, 1] < h) & (rng.random(len(xy)) > 0.1)
    cat = np.c_[xy, ref[:, 2:]][keep]
    cat[:, :2] += rng.normal(0, 0.15, (len(cat), 2))
    cat[:, 2] *= rng.lognormal(0, 0.1, len(cat))
    k = len(cat) // 10
    fake = np.c_[rng.uniform([0, 0], [w, h], (k, 2)), rng.uniform(1000, 3000, k), np.full((k, ref.shape[1] - 3), 2.0)]
    truth = np.r_[ref[keep, :2], np.full((k, 2), np.nan)]
    order = np.argsort(-np.r_[cat[:, 2], fake[:, 2]], kind="stable")
    return np.r_[cat, fake][order], truth[order]


def off_by(coef, cat, truth, w=W, h=H):
    """Median distance (px) between the true stars' reference positions and where the fit puts them."""
    x, y = apply_poly(coef, cat[:, 0], cat[:, 1], w, h)
    ok = np.isfinite(truth[:, 0])
    return float(np.median(np.hypot(x[ok] - truth[ok, 0], y[ok] - truth[ok, 1])))


@pytest.mark.parametrize("ang", [82.2, 60.0, 100.0, 137.0, 263.0, 300.0])
def test_a_frame_turned_by_any_angle_is_aligned(ang):
    """The camera turned between two nights: neither the shift, nor the turn within 45 degrees, nor (but at 137
    degrees, within 45 of the half turn) the half turn finds it; the triangles do, and the series' next frame is fitted
    at its angle without a search."""
    ref = field()
    tree = cKDTree(ref[:, :2])
    cat, truth = seen(ref, ang)
    r = fit_frame(cat, ref, tree, W, H)
    assert r is None or r[1] < 0.3 * len(cat)
    if abs(ang - 180.0) > 45.0:
        assert fit_half_turn(cat, ref, tree, W, H)[0] is None
    r, at = fit_any_turn(cat, ref, tree, W, H)
    assert r is not None and r[1] > 0.75 * len(cat) and r[2] < 0.6 and off_by(r[0], cat, truth) < 0.3
    assert abs((at + ang + 180.0) % 360.0 - 180.0) < 0.05 and at == pytest.approx(register.turn_of(r[0]))
    nxt, truth2 = seen(ref, ang + 0.02, (70.0, -35.0), seed=2)
    r2, again = fit_half_turn(nxt, ref, tree, W, H, at)
    assert r2 is not None and again == at and off_by(r2[0], nxt, truth2) < 0.3


def test_a_turn_within_45_degrees_is_left_to_its_own_search():
    """A frame the shift or the turn within 45 degrees aligns (or leaves out) today is not taken from them."""
    ref = field()
    r, at = fit_any_turn(seen(ref, 12.0)[0], ref, cKDTree(ref[:, :2]), W, H)
    assert r is None and at is None


def test_a_frame_of_another_field_is_not_taken_for_a_turned_one():
    ref = field()
    tree = cKDTree(ref[:, :2])
    for seed in (7, 8, 9):
        assert fit_any_turn(field(seed=seed), ref, tree, W, H) == (None, None)


# ---------------------------------------------------------------- the stage: two nights, the camera turned between

def night_pair(ang, n_stars, seed=0):
    """Four frames of a first night turned by ``ang`` degrees, five of a second night (the reference's), and a frame of
    another field last: star catalogues as the stage's checkpoint holds them, and the true positions."""
    from test_register_sparse import H as Hs
    from test_register_sparse import W as Ws
    rng = np.random.default_rng(seed)
    sky = np.c_[rng.uniform([0, 0], [Ws, Hs], (n_stars, 2)), np.sort(rng.pareto(1.5, n_stars) + 1)[::-1] * 1000,
                np.full(n_stars, 2.0), np.full((n_stars, len(register.APERTURES)), np.nan)]
    shots = [seen(sky, ang + 0.01 * i, (8.0 * i, -5.0 * i), seed=10 + i, size=(Ws, Hs)) for i in range(4)]
    shots += [seen(sky, 0.0, (6.0 * i, 4.0 * i), seed=20 + i, size=(Ws, Hs)) for i in range(5)]
    other = np.c_[rng.uniform([0, 0], [Ws, Hs], (n_stars, 2)), np.sort(rng.pareto(1.5, n_stars) + 1)[::-1] * 1000,
                  np.full(n_stars, 2.0), np.full((n_stars, len(register.APERTURES)), np.nan)]
    shots.append((other, None))
    return [c for c, _ in shots], [t for _, t in shots], (Ws, Hs)


@pytest.mark.parametrize("ang,n_stars", [(82.2, 400), (263.0, 1800)])
def test_two_nights_with_the_camera_turned_between_are_aligned(tmp_path, monkeypatch, ang, n_stars):
    """Every frame of both nights aligned, the turned night's first frame searched once and the others fitted at its
    angle, the frame of another field refused, and the turn said in one line. 1 800 stars: the dense field whose turned
    frames the shift's vote matched on chance pairs (48 stars at 2.3 px, kept under the residual limit of 2.5 px)."""
    from test_register_sparse import _run
    cats, truth, (Ws, Hs) = night_pair(ang, n_stars)
    searched = []
    real = register.fit_any_turn
    monkeypatch.setattr(register, "fit_any_turn", lambda *a, **k: searched.append(1) or real(*a, **k))
    z, logs, metrics = _run(tmp_path, cats, monkeypatch)
    ok = np.isfinite(z["rms"])
    assert ok[:9].all() and not ok[9] and metrics["aligned"] == 9 and metrics["not_aligned"] == 1
    assert int(z["ref"]) >= 4
    for i in range(9):
        assert z["nmatch"][i] > 0.75 * len(cats[i]) and off_by(z["coef"][i], cats[i], truth[i], Ws, Hs) < 0.3
    assert len(searched) == 2                       # the first turned frame and the frame of another field
    turn = 360.0 - ang if ang > 180.0 else ang
    assert ("info", f"4 frame(s) aligned after a turn of the camera of {turn:.1f} degrees") in logs
    assert ("warning", "1 frame(s) could not be aligned and will be ignored, e.g. f9.fits") in logs


def test_a_session_that_aligns_today_never_reaches_the_new_search(tmp_path, monkeypatch):
    """The usual drift and a field turning within 45 degrees: the same path as before, and no turn said."""
    from test_register_sparse import _run
    cats, _truth, _size = night_pair(0.0, 400)
    cats[:4] = night_pair(14.0, 400)[0][:4]

    def never(*_a, **_k):
        raise AssertionError("searched")
    monkeypatch.setattr(register, "fit_any_turn", never)
    z, logs, metrics = _run(tmp_path, cats[:9], monkeypatch)
    assert metrics["aligned"] == 9 and not any("turn of the camera" in m for _, m in logs)


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


# ------------------------------------------------------------------------ the stack of frames at two orientations

def two_orientations(root, n=10, turned=5, ang=82.2, n_stars=160, seed=0, sigma=1.1, size=(192, 256)):
    """A v1 project with done inventory, calibrate (a float16 cache of RGGB frames) and register results: the first
    ``turned`` frames of a camera turned by ``ang`` degrees, the others not, each dithered; the polynomials from the
    true turns and dithers. Returns the project and the truth (the stars on the reference, their colours)."""
    from test_stacking import CH

    from astro_gpu.core.project import Inputs, Project
    Hs, Ws = size
    rng = np.random.default_rng(seed)
    proj = Project.create(root, "toy", Inputs())
    sxy = rng.uniform([12, 12], [Ws - 12, Hs - 12], (n_stars, 2))
    flux = np.exp(rng.uniform(np.log(1500), np.log(30000), n_stars))
    col = np.c_[rng.uniform(0.6, 1.4, n_stars), np.ones(n_stars), rng.uniform(0.6, 1.4, n_stars)]
    dith = rng.uniform(-3, 3, (n, 2))
    yy, xx = np.mgrid[0:Hs, 0:Ws]
    chan = CH[2 * (yy % 2) + (xx % 2)]
    cache = root / "lights.f16"
    mm = np.memmap(cache, dtype=np.float16, mode="w+", shape=(n, Hs, Ws))
    stats = np.zeros((n, 4, 2), np.float32)
    coef = np.zeros((n, 2, 10))
    c0 = np.array([Ws / 2, Hs / 2])
    for i in range(n):
        t = np.deg2rad(ang if i < turned else 0.0)
        ct, st = np.cos(t), np.sin(t)
        fxy = (sxy - c0) @ np.array([[ct, -st], [st, ct]]).T + c0 + dith[i]      # the sky on this frame's sensor
        img = np.full((Hs, Ws), 300.0)
        for (x0, y0), f, c in zip(fxy, flux, col):
            ix, iy = int(round(x0)), int(round(y0))
            ys, xs = slice(max(iy - 7, 0), max(min(iy + 8, Hs), 0)), slice(max(ix - 7, 0), max(min(ix + 8, Ws), 0))
            g = f / (2 * np.pi * sigma ** 2) * np.exp(-0.5 * ((xx[ys, xs] - x0) ** 2 + (yy[ys, xs] - y0) ** 2) / sigma ** 2)
            img[ys, xs] += g * c[chan[ys, xs]]                                   # each sample its colour of the star
        img = img + rng.normal(0, 1, img.shape) * np.sqrt(img + 36.0)
        mm[i] = img
        for k in range(4):
            pl = img[k // 2::2, k % 2::2]
            stats[i, k] = [np.median(pl), 1.4826 * np.median(np.abs(pl - np.median(pl)))]
        dx, dy = dith[i]                                  # frame -> reference: X = c + R(-t) (x - c - dither)
        coef[i, 0, :3] = [Ws / 2 - ct * dx - st * dy, ct * Ws / 2, st * Hs / 2]
        coef[i, 1, :3] = [Hs / 2 + st * dx - ct * dy, -st * Ws / 2, ct * Hs / 2]
    mm.flush()
    del mm

    def stage(name, files):
        d = proj.stage_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        for k, v in files.items():
            if k.endswith(".json"):
                (d / k).write_text(json.dumps(v))
            elif k.endswith(".npz"):
                np.savez(d / k, **v)
            else:
                np.save(d / k, v)
        (d / "manifest.json").write_text(json.dumps({"status": "done", "metrics": {}}))
    stage("inventory", {"frames.json": {"lights": [{"path": f"L{i:02d}.fits", "time": f"2026-01-01T21:{i:02d}:00"}
                                                   for i in range(n)]}})
    stage("calibrate", {"cache.json": {"path": str(cache), "n": n, "height": Hs, "width": Ws, "scale": 1.0,
                                       "cfa": "RGGB", "white_level": 60000.0, "dark_level": 0.0},
                        "frame_stats.npy": stats, "master_dark.npy": np.zeros((Hs, Ws), np.float32),
                        "master_flat.npy": np.ones((Hs, Ws), np.float32)})
    stage("register", {"reg.npz": {"coef": coef, "transp": np.ones(n), "fwhm": np.full(n, 2.6), "rms": np.full(n, 0.1),
                                   "ref": n - 2, "nmatch": np.full(n, 100), "nstars": np.full(n, 150),
                                   "noise": np.full(n, 36.0), "roundness": np.full(n, 0.95)}})
    return proj, SimpleNamespace(sxy=sxy, flux=flux, col=col, ang=ang, size=size)


def seen_turned(X, Y, truth, margin):
    """Whether the reference position (X, Y) is on the turned frames' sensor, ``margin`` px inside its edges."""
    Hs, Ws = truth.size
    t = np.deg2rad(truth.ang)
    x = Ws / 2 + np.cos(t) * (X - Ws / 2) - np.sin(t) * (Y - Hs / 2)
    y = Hs / 2 + np.sin(t) * (X - Ws / 2) + np.cos(t) * (Y - Hs / 2)
    return (x > margin) & (x < Ws - margin) & (y > margin) & (y < Hs - margin)


@pytest.mark.parametrize("method", ["drizzle", "classic"])
def test_frames_at_two_orientations_stack_into_one_image(tmp_path, method):
    """Five frames turned by 82 degrees and five not, stacked by both methods: on the part of the sky both see and on
    the part only the straight ones see, the stars round, of their size, of their whole flux (a frame put a turn wrong
    would leave half of it elsewhere) and of their own colours through the mosaic; no step of the sky at the edge of
    the turned frames; and the weight map twice as high where both orientations cover."""
    from test_stacking import copy_project, run_stage

    from astro_gpu.stages.stack import Stack, StackParams
    proj, truth = two_orientations(tmp_path / "p")
    Hs, Ws = truth.size
    p = copy_project(proj.root, tmp_path / method)
    params = {"scale": 1.0, "pixfrac": 1.0, "gain": 1.0, "read_noise": 6.0, "half_block": 2}
    run_stage(p, Stack(), StackParams(**params, **({"method": "classic"} if method == "classic" else {})))
    img, wt = np.load(p.stage_dir("stack") / "drz_full.npy"), np.load(p.stage_dir("stack") / "drz_weight.npy")
    yy, xx = np.mgrid[-5:6, -5:6]
    rows = []
    for (x0, y0), f, c in zip(truth.sxy, truth.flux, truth.col):
        ix, iy = int(round(x0)), int(round(y0))
        if not (10 < ix < Ws - 10 and 10 < iy < Hs - 10) or f < 4000:
            continue
        sky = np.median(img[:, iy - 9:iy + 10, ix - 9:ix + 10].reshape(3, -1), axis=1)
        star = img[:, iy - 5:iy + 6, ix - 5:ix + 6] - sky[:, None, None]
        ap = (xx - (x0 - ix)) ** 2 + (yy - (y0 - iy)) ** 2 <= 16
        fl = star[:, ap].sum(1)
        g = np.clip(star[1], 0, None) * ap                   # the green image's second moments: size and roundness
        mx, my = (g * xx).sum() / g.sum(), (g * yy).sum() / g.sum()
        cov = np.cov(np.c_[xx.ravel() - mx, yy.ravel() - my].T, aweights=g.ravel(), bias=True)
        ev = np.linalg.eigvalsh(cov)
        rows.append((bool(seen_turned(x0, y0, truth, 6.0)), fl[1] / f, fl[0] / fl[1] / c[0], fl[2] / fl[1] / c[2],
                     np.sqrt(ev[0] / ev[1]), 2.3548 * np.sqrt(ev.mean())))
    m = np.array(rows)
    for part in (m[:, 0] == 1, m[:, 0] == 0):
        assert part.sum() >= 20
        flux, red, blue, round_, fwhm = np.median(m[part, 1:], axis=0)
        assert abs(flux - 1) < 0.04 and abs(red - 1) < 0.05 and abs(blue - 1) < 0.05
        assert round_ > 0.92 and fwhm < 1.15 * 2.3548 * 1.1
    Y, X = np.mgrid[0:Hs, 0:Ws]
    inner = (X > 12) & (X < Ws - 12) & (Y > 12) & (Y < Hs - 12)
    edge_in = seen_turned(X, Y, truth, 0.0) & ~seen_turned(X, Y, truth, 10.0) & inner
    edge_out = ~seen_turned(X, Y, truth, 0.0) & seen_turned(X, Y, truth, -10.0) & inner
    both, one = seen_turned(X, Y, truth, 6.0) & inner, ~seen_turned(X, Y, truth, -6.0) & inner
    for c in range(3):
        pl = img[c]
        s = 1.4826 * np.median(np.abs(pl[inner] - np.median(pl[inner])))
        sky = pl < np.median(pl[inner]) + 3 * s
        assert abs(np.median(pl[edge_in & sky]) - np.median(pl[edge_out & sky])) < 0.25 * s
        assert wt[c][both].mean() / wt[c][one].mean() == pytest.approx(2.0, rel=0.1)
