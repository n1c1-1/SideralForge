"""The diagnostics of the basic stages for the Inspect mode: the measures and flags of the calibration frames, the
geometry and star roundness of the registration, the rejection maps, normalisation and noise of the stack, the per-star
colour fit, and the api functions that serve them (calibration_frames, registration_table, stack_diagnostics,
color_fit), on synthetic data (CPU)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from scipy.spatial import cKDTree

from astro_gpu import api
from astro_gpu.core import diagnostics as dg
from astro_gpu.core.journal import Journal
from astro_gpu.core.project import Inputs, Project
from astro_gpu.core.stage import Context
from astro_gpu.core.store import Store

RECOMPUTE = "missing: recompute {} to get it"


# ---------------------------------------------------------------------------------------------- helpers

def write_fits(path: Path, data: np.ndarray, **header) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.Header()
    h["BAYERPAT"] = "RGGB"
    for k, v in header.items():
        h[k.replace("_", "-")] = v
    fits.PrimaryHDU(np.clip(data, 0, 65535).astype(np.uint16), h).writeto(path, overwrite=True)
    return path


def dark(rng, h=128, w=192, level=500.0, noise=8.0):
    return level + rng.normal(0, noise, (h, w))


def glow(h, w, amp, sigma):
    yy, xx = np.mgrid[0:h, 0:w]
    return amp * np.exp(-0.5 * ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / sigma ** 2)


def vignetted(rng, level, h=128, w=192, noise=30.0):
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2, yy - h / 2) / np.hypot(w / 2, h / 2)
    return 300 + level * (1 - 0.4 * r ** 2) + rng.normal(0, noise, (h, w))


def fake_done(p, wf, stage, metrics=None, params=None, files=None) -> Path:
    """A done result of the real graph without running it (as tests/test_api.py makes them)."""
    k = api.stage_key(p, wf, stage)
    d = Store(p.root).dir(stage, k)
    d.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {}).items():
        f = d / name
        if isinstance(content, np.ndarray):
            np.save(f, content)
        else:
            f.write_text(json.dumps(content), encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps({
        "stage": stage, "status": "done", "key": k, "run_id": k, "metrics": metrics or {}, "params": params or {},
        "origin": {}, "auto_reasons": {}, "duration": 1.0, "finished": "2026-11-01T01:00:00"}), encoding="utf-8")
    api.refresh(p)
    return d


# ---------------------------------------------------------------------------------------------- calibration frames

def test_the_odd_darks_are_flagged_with_their_reason_in_plain_words(tmp_path):
    rng = np.random.default_rng(1)
    h, w = 128, 192
    frames = {f"ok{i}": dark(rng) for i in range(6)}
    frames["glow"] = dark(rng) + glow(h, w, 60, 25)          # the "growing central light"
    hot = dark(rng)
    hot.flat[rng.choice(h * w, 200, replace=False)] += 600
    frames["hot"] = hot
    frames["level"] = dark(rng, level=540)
    stats = []
    for name, a in frames.items():
        f = write_fits(tmp_path / f"{name}.fits", a, EXPTIME=60.0, CCD_TEMP=-10.0 if name != "level" else 2.0)
        st = dg.raw_stats(f, 65535.0)
        st["temperature"] = -10.0 if name != "level" else 2.0
        st["exposure"] = 60.0
        stats.append(st)
    got = dict(zip(frames, dg.flag_frames("darks", stats)))
    assert all(got[f"ok{i}"][0] == "ok" for i in range(6))
    assert got["glow"][0] == "warn" and "gradient" in [c["key"] for c in got["glow"][2]]
    assert "brighter than the corners" in got["glow"][1]
    assert got["hot"][0] == "warn" and [c["key"] for c in got["hot"][2]] == ["hot_pixels"]
    assert got["hot"][2][0]["value"] >= 190 and "hot pixels" in got["hot"][1]
    keys = [c["key"] for c in got["level"][2]]
    assert "median" in keys and "temperature" in keys and "Taken at 2 °C, the other darks at -10 °C" in got["level"][1]
    ok = stats[0]
    assert ok["median"] == pytest.approx(500, abs=1) and ok["noise"] == pytest.approx(8, rel=0.15)
    assert abs(ok["gradient"]) < 2 and ok["hot_pixels"] < 5 and len(ok["rings"]) == dg.RINGS
    assert stats[6]["gradient"] > 30                                # the glow frame


def test_flats_give_their_level_vignetting_and_exposure_problems(tmp_path):
    rng = np.random.default_rng(2)
    white = 65535.0
    levels = {"a": 26000, "b": 26500, "c": 25800, "over": 64000, "under": 3000}
    stats = []
    for name, lv in levels.items():
        st = dg.raw_stats(write_fits(tmp_path / f"{name}.fits", vignetted(rng, lv)), white)
        st.update(dg.flat_measures(st, 300.0, white))
        stats.append(st)
    a = stats[0]
    assert a["level_pct"] == pytest.approx(100 * (26000 * (1 - 0.4 * 0.33) / (white - 300)), rel=0.1)
    assert a["vignetting"][0] == pytest.approx(1.0, abs=0.02) and a["vignetting"][-1] < 0.7
    assert np.all(np.diff(a["vignetting"]) < 0.01) and a["corner_pct"] < 70
    got = dict(zip(levels, dg.flag_frames("flats", stats)))
    assert [got[k][0] for k in "abc"] == ["ok"] * 3
    over = [c["key"] for c in got["over"][2]]
    assert "saturated" in over and "level" in over and got["over"][0] == "warn"
    assert "underexposed" in [c["key"] for c in got["under"][2]] and "only" in got["under"][1]


def starry(rng, n, h=512, w=768, level=520.0, noise=8.0, amp=400.0, sigma=2.0):
    """A frame of the sky: ``n`` stars on a jittered grid inside the part of the frame the star count looks at (the
    border of two tiles of 2x2-binned pixels is left out), on a sky ``level`` (raw pixels)."""
    a = level + rng.normal(0, noise, (h, w))
    m = 4 * dg.STAR_TILE + 8
    cols = int(np.ceil(np.sqrt(n * (w - 2 * m) / (h - 2 * m))))
    rows = int(np.ceil(n / cols))
    yy, xx = np.mgrid[-10:11, -10:11]
    for k in range(n):
        i, j = divmod(k, cols)
        y = m + (i + 0.5) * (h - 2 * m) / rows + rng.uniform(-3, 3)
        x = m + (j + 0.5) * (w - 2 * m) / cols + rng.uniform(-3, 3)
        yi, xi = int(y), int(x)
        a[yi - 10:yi + 11, xi - 10:xi + 11] += amp * np.exp(-((xx + xi - x) ** 2 + (yy + yi - y) ** 2)
                                                             / (2 * sigma ** 2))
    return a


def binned(a):
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean((1, 3))


def test_the_star_count_sees_stars_but_not_hot_pixels_glow_or_dust():
    rng = np.random.default_rng(11)
    h, w = 512, 768
    assert dg.count_stars(binned(starry(rng, 40))) == 40
    assert dg.count_stars(binned(starry(rng, 40, amp=60.0))) >= 36            # faint ones: 60 ADU on a noise of 8
    d = dark(rng, h, w, 508.0)
    d.flat[rng.choice(h * w, 3000, replace=False)] += 800                       # single hot pixels
    pairs = rng.choice(h * (w - 1), 300, replace=False)
    d.flat[pairs] += 600
    d.flat[pairs + 1] += 600                                                     # hot pairs
    d += glow(h, w, 40, 120)                                                     # the growing central light
    yy, xx = np.mgrid[0:h, 0:w]
    d += 80 * np.exp(-np.hypot(xx, yy) / 40)                                     # amp glow in a corner
    assert dg.count_stars(binned(d)) == 0
    f = vignetted(rng, 20000, h, w, noise=100.0)                                 # a flat with dust shadows
    for _ in range(30):
        cy, cx = rng.uniform(140, h - 140), rng.uniform(140, w - 140)
        r = np.hypot(xx - cx, yy - cy)
        f -= 600 * (r < 14) - 250 * ((r >= 14) & (r < 18))                      # dark core, bright diffraction rim
    assert dg.count_stars(binned(f)) < dg.STARS_NOT_CALIBRATION
    assert dg.count_stars(binned(dark(rng, 128, 192))) == 0                      # too small to say: none


def test_a_whole_set_of_sky_frames_given_as_darks_is_flagged_though_they_agree(tmp_path):
    """The 28.12.24 night: 100 "darks" that were frames of the sky (the Rosette, telescope open): alike, so the
    comparison with the others of their kind saw nothing (docs/pilot/DARKS_28_12_24.md)."""
    rng = np.random.default_rng(12)
    sky = []
    for i in range(5):
        f = write_fits(tmp_path / "sky" / f"S{i}.fits", starry(rng, 30 + 10 * i), EXPTIME=15.0)
        sky.append(dict(dg.raw_stats(f, 65535.0), exposure=15.0))
    real = [dict(dg.raw_stats(write_fits(tmp_path / "real" / f"D{i}.fits", dark(rng, 512, 768, 508.0)), 65535.0),
                 exposure=15.0) for i in range(5)]
    assert [s["stars"] for s in sky] == [30, 40, 50, 60, 70] and all(s["stars"] == 0 for s in real)
    assert not sky[0]["camera_raw"]                                              # FITS: its black level is not the camera's
    for kind, one, cap in (("darks", "dark", True), ("bias", "bias frame", True), ("flat_darks", "flat dark", True),
                           ("flats", "flat", False)):
        got = dg.flag_frames(kind, sky)
        assert [g[0] for g in got] == ["bad"] * 5, kind
        flag, reason, checks = got[0]
        assert checks[0]["key"] == "stars" and checks[0]["value"] == 30
        assert reason.startswith("Contains stars (30 found): a frame of the sky, not a " + one)
        assert ("cap" in reason) == cap
        assert [g[0] for g in dg.flag_frames(kind, real)] == ["ok"] * 5
    assert [g[0] for g in dg.flag_frames("lights", sky)] == ["ok"] * 5           # lights hold stars
    mixed = dg.flag_frames("darks", real + sky[:1])                              # and the relative flags still work
    assert [g[0] for g in mixed] == ["ok"] * 5 + ["bad"]


def test_a_camera_raw_dark_far_above_the_black_level_is_flagged():
    base = {"median": 508.0, "noise": 8.4, "black": 512.0, "white": 16383.0, "camera_raw": True, "stars": 0,
            "exposure": 15.0}
    lit = dict(base, median=1612.0, noise=40.0)                                  # a flat put among the darks
    got = dg.flag_frames("darks", [base, dict(base), lit])
    assert [g[0] for g in got] == ["ok", "ok", "bad"]
    assert got[2][2][0]["key"] == "sky_level" and got[2][2][0]["value"] == pytest.approx(1100.0)
    assert "1100 ADU above the camera's black level (512 ADU), 28x its noise" in got[2][1] and "not a dark" in got[2][1]
    assert dg.flag_frames("bias", [lit])[0][0] == "bad"
    assert dg.flag_frames("darks", [dict(lit, camera_raw=False)])[0][0] == "ok"   # a FITS offset is no black level
    assert dg.flag_frames("darks", [dict(base, median=520.0)])[0][0] == "ok"      # the 28.12.24 sky: 8 ADU, 1x the noise
    assert dg.flag_frames("flats", [dict(lit, level_pct=7.0)])[0][0] != "bad"     # a flat is meant to be lit


def test_the_measures_of_a_frame_are_kept_in_memory(tmp_path, monkeypatch):
    f = write_fits(tmp_path / "d.fits", dark(np.random.default_rng(3)))
    a = dg.raw_stats(f, 65535.0)
    calls = []
    monkeypatch.setattr(dg, "_measure", lambda *x: calls.append(1) or {})
    assert dg.raw_stats(f, 65535.0) == a and not calls


def test_calibrate_keeps_the_master_bias_and_flat_dark_and_a_rebuild_copies_them(tmp_path, monkeypatch):
    from astro_gpu.stages import calibrate as cal
    rng = np.random.default_rng(4)
    raws = tmp_path / "raw"
    kinds = {"lights": 2, "darks": 3, "flats": 3, "bias": 3, "flat_darks": 3}
    paths = {k: [str(write_fits(raws / k / f"{k}{i}.fits",
                                vignetted(rng, 20000, 64, 96) if k == "flats" else dark(rng, 64, 96)))
                 for i in range(n)] for k, n in kinds.items()}
    proj = Project.create(tmp_path / "p", "p", Inputs())
    inv = proj.stage_dir("inventory")
    inv.mkdir(parents=True)
    (inv / "frames.json").write_text(json.dumps({k: [{"path": f} for f in v] for k, v in paths.items()}))
    (inv / "manifest.json").write_text(json.dumps({"status": "done", "metrics": {
        "height": 64, "width": 96, "black_level": 0.0, "cfa": "RGGB", "white_level": 65535.0}}))
    seen = {}
    monkeypatch.setattr(cal, "calibrate_lights", lambda ctx, *a: seen.setdefault("full", True))
    monkeypatch.setattr(cal, "rebuild_cache", lambda ctx, p, l, given, *a: seen.setdefault("given", given))
    ctx = Context(proj, cal.Calibrate(), Journal(proj.logs, echo=False), "cpu")
    cal.Calibrate().run(ctx, cal.CalibrateParams(workers=1))
    bias = np.load(ctx.out("master_bias.npy"))
    assert seen == {"full": True} and bias.shape == (64, 96) and abs(float(np.median(bias)) - 500) < 3
    assert np.load(ctx.out("master_flat_dark.npy")).shape == (64, 96)
    # the cache-only rebuild: the runner copies MASTER_FILES with MASTERS ("from": the earlier result)
    old = tmp_path / "old"
    old.mkdir()
    for n in cal.EXTRA_MASTERS:
        np.save(old / n, np.full((64, 96), 7.0, np.float32))
    new = tmp_path / "new"
    proj2 = Project.create(new, "q", Inputs())
    d = proj2.stage_dir("calibrate")
    d.mkdir(parents=True)
    for n in cal.MASTER_FILES:
        np.save(d / n, np.zeros((64, 96), bool if n == "badmask.npy" else np.float32))
    monkeypatch.undo()
    monkeypatch.setattr(cal, "_calibrated_cache", lambda *a: None)
    ctx2 = Context(proj2, cal.Calibrate(), Journal(proj2.logs, echo=False), "cpu")
    cal.rebuild_cache(ctx2, cal.CalibrateParams(), [], {"metrics": {}, "from": str(old)}, "RGGB", 65535.0)
    assert all(float(np.load(d / n)[0, 0]) == 7.0 for n in cal.EXTRA_MASTERS)


# ---------------------------------------------------------------------------------------------- registration

def _turned(ang):
    t = np.deg2rad(ang)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def test_the_geometry_of_a_frame_comes_from_its_polynomial():
    from astro_gpu.stages.register import fit_frame
    W, H = 1200, 800
    c = np.array([W / 2, H / 2])
    # the polynomial the registration fits: a field turned by 1.5 degrees and shifted by (14, -9) px
    rng = np.random.default_rng(5)
    ref = np.c_[rng.uniform([0, 0], [W, H], (400, 2)), np.full(400, 1000.0), np.full(400, 2.0)]
    R, shift = _turned(1.5), np.array([14.0, -9.0])
    xy = (ref[:, :2] - c) @ R.T + c + shift                       # the frame: the sky turned and shifted
    inside = (xy[:, 0] > 0) & (xy[:, 0] < W) & (xy[:, 1] > 0) & (xy[:, 1] < H)
    coef, nmatch, rms, _ = fit_frame(np.c_[xy, ref[:, 2:]][inside], ref, cKDTree(ref[:, :2]), W, H)
    # an exact one with a scale: frame = s R (reference - c) + c + shift, so reference = c + R^-1 (frame - c - shift) / s
    s, R2, shift2 = 1.002, _turned(-0.7), np.array([-3.0, 25.0])
    Ri = np.linalg.inv(R2) / s
    exact = np.zeros((2, 10))
    for k in range(2):
        exact[k, :3] = [c[k] - (Ri @ shift2)[k], Ri[k, 0] * W / 2, Ri[k, 1] * H / 2]
    dx, dy, rot, sc = dg.frame_geometry(np.stack([coef, exact, np.zeros_like(coef)]), W, H,
                                        np.array([True, True, False]))
    # dx, dy: the frame's star at the centre of the frame minus the same star on the reference
    want = np.linalg.inv(R) @ shift
    assert (dx[0], dy[0]) == (pytest.approx(want[0], abs=0.02), pytest.approx(want[1], abs=0.02))
    assert rot[0] == pytest.approx(1.5, abs=0.01) and sc[0] == pytest.approx(1.0, abs=1e-4)
    want = Ri @ shift2
    assert (dx[1], dy[1], rot[1], sc[1]) == (pytest.approx(want[0]), pytest.approx(want[1]), pytest.approx(-0.7),
                                             pytest.approx(s))
    assert np.isnan([dx[2], dy[2], rot[2], sc[2]]).all()


@pytest.mark.parametrize("sx,sy,lo,hi", [(3.0, 3.0, 0.85, 1.01), (4.0, 2.0, 0.3, 0.7)])
def test_registration_measures_the_roundness_of_the_stars(tmp_path, sx, sy, lo, hi):
    from astro_gpu.stages import register as reg
    rng = np.random.default_rng(6)
    H = W = 512
    yy, xx = np.mgrid[0:H, 0:W]
    img = 1000 + rng.normal(0, 10, (H, W))
    for x, y in rng.uniform(30, W - 30, (40, 2)):
        img += 6000 * np.exp(-0.5 * (((xx - x) / sx) ** 2 + ((yy - y) / sy) ** 2))
    f = tmp_path / "lights.f16"
    mm = np.memmap(f, dtype=np.float16, mode="w+", shape=(1, H, W))
    mm[0] = img
    mm.flush()
    del mm
    reg._init(str(f), 1, H, W, 8.0, 2500, "RGGB")
    idx, cat, fwhm, rms, rnd, _ = reg._catalog(0)
    reg._G.clear()
    assert idx == 0 and len(cat) >= 30 and lo < rnd < hi



def test_a_star_whose_background_ring_is_all_bad_pixels_is_left_out(tmp_path):
    """A star in a small island of good pixels, a wide ring of bad (NaN) pixels around it: its background annulus has
    no valid pixel; the frame is still catalogued, that star without its aperture fluxes."""
    from astro_gpu.stages import register as reg
    rng = np.random.default_rng(7)
    H = W = 512
    yy, xx = np.mgrid[0:H, 0:W]
    img = 1000 + rng.normal(0, 10, (H, W))
    for x, y in rng.uniform(30, W - 30, (30, 2)):
        img += 6000 * np.exp(-0.5 * (((xx - x) / 3.0) ** 2 + ((yy - y) / 3.0) ** 2))
    img += 20000 * np.exp(-0.5 * (((xx - 256) / 3.0) ** 2 + ((yy - 256) / 3.0) ** 2))
    d = np.hypot(xx - 256, yy - 256)
    img[(d > 14) & (d < 90)] = np.nan
    f = tmp_path / "lights.f16"
    mm = np.memmap(f, dtype=np.float16, mode="w+", shape=(1, H, W))
    mm[0] = img
    mm.flush()
    del mm
    reg._init(str(f), 1, H, W, 8.0, 2500, "RGGB")
    idx, cat, fwhm, rms, rnd, _ = reg._catalog(0)
    reg._G.clear()
    assert idx == 0 and len(cat) >= 15 and np.isfinite(fwhm)
    lone = np.hypot(cat[:, 0] - 256, cat[:, 1] - 256) < 4
    assert lone.any() and np.isfinite(cat[lone, 4]).all() and np.isnan(cat[lone, 6:]).all()   # ring: only the small


# ---------------------------------------------------------------------------------------------- stacking

def stack_project(root: Path, n: int = 8, H: int = 192, W: int = 256):
    """A v1 project with done inventory, calibrate (a float16 cache) and register results: a flat sky of 100 ADU
    with stars, frame 2 crossed by a satellite at sensor row 60, frame 4 with a dark hole, frame 3 with a sky 50 ADU
    higher; the frames shifted by a pixel or two, so that each half-stack (blocks of one frame) has every colour at
    every output pixel."""
    rng = np.random.default_rng(7)
    proj = Project.create(root, "s", Inputs())
    shifts = [(0, 0), (1, 1), (1, 0), (0, 1), (0, 1), (1, 0), (1, 1), (2, 2)][:n]
    yy, xx = np.mgrid[0:H, 0:W]
    stars = [(40, 20), (200, 25), (130, 100), (60, 140), (220, 160), (150, 175)]   # away from the trail and hole
    cache = root / "lights.f16"
    mm = np.memmap(cache, dtype=np.float16, mode="w+", shape=(n, H, W))
    coef = np.zeros((n, 2, 10))
    for i, (dx, dy) in enumerate(shifts):
        a = 100 + (50 if i == 3 else 0) + rng.normal(0, 10, (H, W))
        for x, y in stars:
            a += 3000 * np.exp(-0.5 * ((xx - x - dx) ** 2 + (yy - y - dy) ** 2) / 1.5 ** 2)
        if i == 2:
            a[60] += 3000
        if i == 4:
            a[30:36, 90:96] -= 500
        mm[i] = a
        coef[i, 0, :3] = [W / 2 - dx, W / 2, 0]
        coef[i, 1, :3] = [H / 2 - dy, 0, H / 2]
    mm.flush()
    del mm
    lights = [{"path": f"L{i}.fits", "time": f"2026-01-01T00:0{i}:00"} for i in range(n)]

    def stage(name, files, metrics=None):
        d = proj.stage_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        for k, v in files.items():
            if k.endswith(".json"):
                (d / k).write_text(json.dumps(v))
            elif k.endswith(".npz"):
                np.savez(d / k, **v)
            else:
                np.save(d / k, v)
        (d / "manifest.json").write_text(json.dumps({"status": "done", "metrics": metrics or {}}))
    stage("inventory", {"frames.json": {"lights": lights}}, {"exposure_s": 60.0})
    stage("calibrate", {"cache.json": {"path": str(cache), "n": n, "height": H, "width": W, "scale": 1.0,
                                       "cfa": "RGGB", "white_level": 16383.0, "dark_level": 0.0},
                        "frame_stats.npy": np.full((n, 4, 2), [100.0, 10.0], np.float32),
                        "master_dark.npy": np.zeros((H, W), np.float32),
                        "master_flat.npy": np.ones((H, W), np.float32)}, {"cache_scale": 1.0})
    stage("register", {"reg.npz": {"coef": coef, "transp": np.ones(n), "fwhm": np.full(n, 3.0),
                                   "rms": np.full(n, 0.1), "ref": 0, "nmatch": np.full(n, 12),
                                   "nstars": np.full(n, 12)}})
    return proj, shifts


@pytest.fixture(scope="module")
def stacked(tmp_path_factory):
    from astro_gpu.stages.stack import Stack, StackParams
    proj, shifts = stack_project(tmp_path_factory.mktemp("stack") / "p")
    ctx = Context(proj, Stack(), Journal(proj.logs, echo=False), "cpu")
    Stack().run(ctx, StackParams(scale=1.0, half_block=1, min_transparency=0.0, gain=1.0, read_noise=10.0))
    return proj, ctx.dir, shifts


def test_the_stack_keeps_its_rejection_maps_low_and_high(stacked):
    from astro_gpu.stages.stack import DIAG, REJECTION
    _, d, shifts = stacked
    with np.load(d / REJECTION) as z:
        low, high, f = z["low"], z["high"], int(z["factor"])
    assert f == 1 and low.shape == high.shape == (192, 256)
    frames = json.loads((d / DIAG).read_text())["frames"]           # the counts of the frames are the maps'
    assert int(high.sum()) == sum(x["rejected_high"] for x in frames)
    assert int(low.sum()) == sum(x["rejected_low"] for x in frames)
    # a sample with nothing to be compared with (the margins the shifts uncover) is never rejected
    assert sum(x["rejected_outside"] for x in frames) == 0
    row = 60 - shifts[2][1]                                         # the satellite, on the reference grid
    along = high[row - 1:row + 2].sum(0)
    assert (along[5:-5] > 0).mean() > 0.9                           # the whole trail, with its edges
    hy, hx = 30 - shifts[4][1], 90 - shifts[4][0]                   # the hole of frame 4
    assert low[hy:hy + 6, hx:hx + 6].sum() >= 20 and low.sum() - low[hy:hy + 6, hx:hx + 6].sum() < 20
    # no high rejection elsewhere, not on the outer rows and columns of the grid (the prediction there was half zero
    # padding before version 4); only around the hole, where with these whole-pixel shifts a colour of a pixel holds
    # the samples of two frames only, the hole's and one other (with real dithering the cleaned prediction holds there)
    inner = high.astype(int)
    inner[row - 1:row + 2] = 0
    inner[hy - 6:hy + 12, hx - 6:hx + 12] = 0
    assert inner.sum() == 0


def test_the_stack_keeps_the_normalisation_of_each_frame_and_the_noise_of_the_result(stacked):
    from astro_gpu.stages.stack import DIAG
    _, d, _ = stacked
    diag = json.loads((d / DIAG).read_text())
    fr = {f["index"]: f for f in diag["frames"]}
    assert sorted(fr) == list(range(8)) and [fr[i]["half"] for i in range(8)] == list("ABABABAB")
    assert fr[3]["offset"] == [pytest.approx(50, abs=6)] * 3        # frame 3's sky, 50 ADU above the reference's
    assert all(abs(v) < 6 for i in (1, 2, 5) for v in fr[i]["offset"]) and fr[0]["scale"] == [1.0] * 3
    assert fr[2]["rejected_high"] > 300 and fr[4]["rejected_low"] >= 20
    drz, wt = np.load(d / "drz_full.npy"), np.load(d / "drz_weight.npy")
    for c, name in enumerate("RGB"):
        v = drz[c][wt[c] > 0]
        direct = 1.4826 * np.median(np.abs(v - np.median(v)))      # sky + a few stars: robust
        assert diag["noise"][name] == pytest.approx(direct, rel=0.35)


# ---------------------------------------------------------------------------------------------- photometric colour

def test_the_colour_star_table_is_what_colour_stars_keeps():
    from astro_gpu.stages.color import STAR_REASONS, colour_star_table, colour_stars
    rng = np.random.default_rng(8)
    H, W = 300, 400
    img = 50 + rng.normal(0, 1, (3, H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    gx, gy = np.meshgrid([80, 130, 180, 230], [80, 140, 200, 240])
    xy = np.r_[np.c_[gx.ravel(), gy.ravel()], [[20, 150], [330, 220]]]     # 16 stars, one at the edge, one bright
    amp = np.r_[rng.uniform(200, 900, 16), 500, 20000]
    for (x, y), a in zip(xy, amp):
        img += a * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / 1.5 ** 2)[None]
    img[2, 100:110, 300:310] = -500                                 # a star on a hole in blue
    xy = np.r_[xy, [[305, 105]]].astype(float)
    idx, flux = colour_stars(img, xy[:, 0], xy[:, 1], 4.0)
    table, why = colour_star_table(img, xy[:, 0], xy[:, 1], 4.0)
    assert np.array_equal(idx, np.where(why == "")[0]) and np.array_equal(flux, table[idx]) and len(idx) == 16
    assert why[16] == "edge" and np.isnan(table[16]).all() and why[17] == "saturated" and why[18] == "not_positive"
    assert set(why) <= set(STAR_REASONS) | {""}


# ---------------------------------------------------------------------------------------------- the api

@pytest.fixture
def project(tmp_path):
    """A v2 project (Classic stack) with one session of FITS frames: 3 lights, 5 darks (one warmer), 4 flats."""
    rng = np.random.default_rng(9)
    raw = tmp_path / "raw"
    for i in range(3):
        write_fits(raw / "lights" / f"L{i}.fits", dark(rng, 64, 96, 1500), EXPTIME=60.0)
    for i in range(5):
        write_fits(raw / "darks" / f"D{i}.fits", dark(rng, 64, 96, 520 if i == 4 else 500), EXPTIME=60.0,
                   CCD_TEMP=5.0 if i == 4 else -10.0)
    for i in range(4):
        write_fits(raw / "flats" / f"F{i}.fits", vignetted(rng, 30000, 64, 96), EXPTIME=1.0)
    p = api.create_project(tmp_path / "p", "p")
    api.add_session(p, lights=[str(raw / "lights")], darks=[str(raw / "darks")], flats=[str(raw / "flats")])
    api.add_workflow(p, "Classic", "Classic stack")
    api.save(p)
    return p


def test_the_diagnostics_are_not_done_before_their_stage(project):
    for f in (api.registration_table, api.stack_diagnostics, api.color_fit):
        with pytest.raises(api.NotDone):
            f(project, 1)
    with pytest.raises(api.NotDone):
        api.calibration_frames(project, 1)
    rep = api.calibration_frames(project, 1, require_done=False)     # the frames can be measured before any run
    assert not rep.masters and {f.kind for f in rep.frames} == {"darks", "flats"}


def test_calibration_frames_gives_the_masters_the_frame_measures_and_the_calibrated_lights(project, tmp_path):
    lights = sorted(str(f).replace("\\", "/") for f in (tmp_path / "raw" / "lights").iterdir())
    fake_done(project, 1, "inventory", {"width": 96, "height": 64}, files={"frames.json": {
        "lights": [{"path": f, "time": f"2026-01-01T00:0{i}:00", "exposure": 60.0} for i, f in enumerate(lights)],
        "darks": [{"path": "x"}], "flats": [{"path": "y"}], "bias": [], "flat_darks": []}})
    cache = tmp_path / "lights.f16"
    mm = np.memmap(cache, dtype=np.float16, mode="w+", shape=(3, 64, 96))
    mm[:] = np.arange(3, dtype=np.float16)[:, None, None]
    mm.flush()
    del mm
    d = fake_done(project, 1, "calibrate", {"dark_level": 500.0}, files={
        "master_dark.npy": np.full((64, 96), 500, np.float32), "master_flat.npy": np.ones((64, 96), np.float32),
        "badmask.npy": np.zeros((64, 96), bool),
        "frame_stats.npy": np.array([[[1000.0 + 100 * i, 10.0]] * 4 for i in range(3)], np.float32),
        "cache.json": {"path": str(cache), "n": 3, "height": 64, "width": 96, "scale": 1.0}})
    rep = api.calibration_frames(project, 1)
    assert rep.stage_dir == str(d) and not rep.missing
    assert set(rep.masters) == {"dark", "flat", "bad_pixels"} and Path(rep.masters["dark"]).name == "master_dark.npy"
    darks = [f for f in rep.frames if f.kind == "darks"]
    assert [f.name for f in darks] == [f"D{i}.fits" for i in range(5)]
    assert [f.flag for f in darks] == ["ok"] * 4 + ["warn"] and "5 °C" in darks[4].reason
    assert darks[0].stats["temperature"] == -10.0 and darks[0].stats["exposure"] == 60.0
    flats = [f for f in rep.frames if f.kind == "flats"]
    assert len(flats) == 4 and all(f.flag == "ok" for f in flats)
    assert flats[0].stats["zero"] == 500.0 and 30 < flats[0].stats["level_pct"] < 50
    ls = [f for f in rep.frames if f.kind == "lights"]
    assert [f.index for f in ls] == [0, 1, 2] and ls[1].stats["median"] == 1100.0 and ls[1].calibrated == str(cache)
    assert float(api.calibrated_light(ls[2]).mean()) == 2.0
    assert not [n for n in rep.notes if "darks are" in n or "No " in n]      # darks and lights of 60 s
    cache.unlink()
    rep = api.calibration_frames(project, 1)
    assert not [f for f in rep.frames if f.kind == "lights" and f.calibrated]
    assert any("no longer on the scratch disk" in n for n in rep.notes)
    with pytest.raises(api.NotDone):
        api.calibrated_light([f for f in rep.frames if f.kind == "lights"][0])


def test_calibration_frames_flags_darks_that_are_frames_of_the_sky(tmp_path):
    rng = np.random.default_rng(13)
    raw = tmp_path / "raw"
    for i in range(3):
        write_fits(raw / "lights" / f"L{i}.fits", starry(rng, 60, level=1500.0), EXPTIME=15.0)
        write_fits(raw / "darks" / f"D{i}.fits", starry(rng, 25, level=520.0), EXPTIME=15.0)
    p = api.create_project(tmp_path / "p", "p")
    api.add_session(p, lights=[str(raw / "lights")], darks=[str(raw / "darks")])
    api.save(p)
    rep = api.calibration_frames(p, 1, require_done=False)
    darks = [f for f in rep.frames if f.kind == "darks"]
    assert [f.flag for f in darks] == ["bad"] * 3 and all(f.stats["stars"] == 25 for f in darks)
    assert darks[0].reason.startswith("Contains stars (25 found): a frame of the sky, not a dark")
    assert darks[0].checks[0]["key"] == "stars"


def test_an_older_calibration_without_the_master_bias_says_to_recompute(project, tmp_path):
    fake_done(project, 1, "inventory", files={"frames.json": {"lights": [], "darks": [], "flats": [],
                                                             "bias": [{"path": "b"}], "flat_darks": []}})
    fake_done(project, 1, "calibrate", files={"master_dark.npy": np.zeros((4, 4), np.float32)})
    rep = api.calibration_frames(project, 1)
    assert rep.missing == ["master bias: " + RECOMPUTE.format("Calibration")]


def test_registration_table_rows_and_the_roundness_of_an_older_result(project, tmp_path):
    n = 3
    fake_done(project, 1, "inventory", {"width": 96, "height": 64}, files={"frames.json": {"lights": [
        {"path": f"C:/x/L{i}.fits", "time": f"2026-01-01T00:0{i}:00"} for i in range(n)]}})
    fake_done(project, 1, "calibrate")
    d = fake_done(project, 1, "register")
    coef = np.zeros((n, 2, 10))
    for i, (dx, dy) in enumerate([(0, 0), (3, -2), (0, 0)]):
        coef[i, 0, :3] = [48 - dx, 48, 0]
        coef[i, 1, :3] = [32 - dy, 0, 32]
    arrays = {"coef": coef, "fwhm": np.array([2.5, 2.7, 9.0]), "transp": np.array([1.0, 0.9, 0.0]),
              "nmatch": np.array([300, 280, 0]), "rms": np.array([0.0, 0.2, np.inf]), "ref": 0,
              "nstars": np.array([350, 330, 12])}
    np.savez(d / "reg.npz", **arrays)
    t = api.registration_table(project, 1)
    assert isinstance(t, list) and len(t) == 3 and t.missing == ["roundness: " + RECOMPUTE.format("Registration")]
    r0, r1, r2 = t
    assert r0.reference and (r0.dx, r0.dy, r0.rotation, r0.scale) == (0, 0, 0, 1)
    assert (r1.dx, r1.dy) == (pytest.approx(3), pytest.approx(-2)) and r1.nmatch == 280 and r1.nstars == 330
    assert r1.name == "L1.fits" and r1.time == "2026-01-01T00:01:00" and r1.used and r1.roundness is None
    assert not r2.used and r2.reason == "not aligned" and np.isnan(r2.dx) and r2.nmatch == 0
    np.savez(d / "reg.npz", roundness=np.array([0.95, 0.6, np.nan]), **arrays)
    t = api.registration_table(project, 1)
    assert not t.missing and [r.roundness for r in t] == [0.95, 0.6, None]


def test_stack_diagnostics_of_an_older_stack_and_of_one_that_kept_them(project, tmp_path):
    from astro_gpu.stages.stack import DIAG, REJECTION
    rng = np.random.default_rng(10)
    fake_done(project, 1, "inventory")
    fake_done(project, 1, "calibrate")
    fake_done(project, 1, "register")
    frames = [{"path": f"C:/x/L{i}.fits", "weight": w, "excluded": r}
              for i, (w, r) in enumerate([(1.0, ""), (0.0, "large stars"), (1.0, ""), (1.0, "")])]
    sky = np.full((3, 60, 80), 100.0)
    A, B = sky + rng.normal(0, 2, sky.shape), sky + rng.normal(0, 2, sky.shape)
    d = fake_done(project, 1, "stack", {"rejected_pct": 0.5}, {"half_block": 1}, files={
        "frames.json": frames, "drz_A.npy": A.astype(np.float32), "drz_B.npy": B.astype(np.float32),
        "drz_meta.json": {"width": 80, "height": 60}})
    s = api.stack_diagnostics(project, 1)
    assert s.used == ["L0.fits", "L2.fits", "L3.fits"] and s.excluded == {"L1.fits": "large stars"}
    assert s.rejection_low is None and s.grid == (80, 60) and s.rejected_pct == 0.5
    assert s.missing == [f"{w}: " + RECOMPUTE.format("Stacking") for w in ("normalisation of each frame",
                                                                           "rejection maps")]
    # halves A = frames 0 and 2 (weight 2), B = frame 3 (weight 1): sigma(A - B) x sqrt(2) / 3
    expect = 2 * np.sqrt(2) * np.sqrt(2) / 3
    assert all(v == pytest.approx(expect, rel=0.05) for v in s.noise.values())
    assert s.weights[1] == ("L1.fits", 0.0, None, None)
    rec = [{"index": i, "offset": [1.0, 2.0, 3.0], "scale": [1.25] * 3} for i in (0, 2, 3)]
    (d / DIAG).write_text(json.dumps({"frames": rec, "noise": {"R": 1.0, "G": 0.5, "B": None}}))
    np.savez_compressed(d / REJECTION, low=np.zeros((60, 80), np.uint16), high=np.ones((60, 80), np.uint16),
                        factor=1, grid=np.array([80, 60]))
    s = api.stack_diagnostics(project, 1)
    assert not s.missing and s.noise["G"] == 0.5 and np.isnan(s.noise["B"])
    assert s.rejection_high.sum() == 4800 and s.rejection_factor == 1
    assert s.weights[0] == ("L0.fits", 1.0, (1.0, 2.0, 3.0), (1.25, 1.25, 1.25)) and s.weights[1][2] is None


def colour_scene(rng):
    """A stack (3, 400, 600) holding the stars of a catalogue of black bodies (unreddened to reddened) seen through
    the generic colour sensor; three stars at the edge, two very bright ones (a hot and a cool one)."""
    from astro_gpu.stages.color import generic_response, predict, spectra_xyz, synthetic_spectra
    lam = np.linspace(336, 1020, 41)
    spec = synthetic_spectra(lam)
    n = len(spec)
    b = 10 ** rng.uniform(-1.0, -0.3, n)
    gx, gy = np.meshgrid(np.linspace(70, 530, 12), np.linspace(70, 330, 10))
    x, y = gx.ravel()[:n], gy.ravel()[:n]
    x, y = np.r_[x, 20, 580, 300, 132.7, 467.3], np.r_[y, 200, 200, 15, 84.4, 315.6]   # the bright: between stars
    b = np.r_[b, 0.5, 0.5, 0.5, 8.0, 8.0]
    spec = np.r_[spec, spec[[60, 61, 62, n - 5, 0]]] * b[:, None]
    cam = predict(spec, lam, generic_response(lam))
    cam *= 3e5 / cam.max()
    H, W = 400, 600
    img = 50 + rng.normal(0, 0.5, (3, H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(len(x)):
        g = np.exp(-0.5 * ((xx - x[i]) ** 2 + (yy - y[i]) ** 2) / 1.5 ** 2) / (2 * np.pi * 1.5 ** 2)
        img += cam[i][:, None, None] * g[None]
    src = np.arange(1000, 1000 + len(x))
    xp = {"source": src, "x": x, "y": y, "g": np.full(len(x), 12.0), "lam": lam, "flux": spec,
          "xyz": spectra_xyz(lam, spec)}
    deep = {"source": src[::-1].copy(), "bp": np.linspace(13, 14, len(x)), "rp": np.full(len(x), 12.0)}
    return img.astype(np.float32), xp, deep




def test_a_crowded_object_that_sep_cannot_deblend_is_extracted_with_a_higher_threshold(monkeypatch):
    """sep's "deblending overflow" (a bright star in nebulosity, as near Alnitak): the threshold is doubled until sep
    extracts the frame; any other error is raised; a frame extracted at once is unchanged."""
    import sep

    from astro_gpu.core.imgutil import sep_extract
    calls = []

    def fake(data, thresh, **kw):
        calls.append((thresh, kw.get("deblend_nthresh")))
        if thresh < 40 and kw.get("deblend_nthresh") != 1:
            raise Exception("object deblending overflow: limit of sub-objects reached while deblending.")
        return "objects"

    monkeypatch.setattr(sep, "extract", fake)
    assert sep_extract(np.zeros((4, 4), np.float32), 10.0, minarea=4) == "objects"
    assert [c[0] for c in calls] == [10.0, 20.0, 40.0]
    calls.clear()
    assert sep_extract(np.zeros((4, 4), np.float32), 0.1) == "objects"           # never enough: not deblended at last
    assert calls[-1] == (1.6, 1)

    def broken(data, thresh, **kw):
        raise Exception("internal pixel buffer full")

    monkeypatch.setattr(sep, "extract", broken)
    with pytest.raises(Exception, match="pixel buffer"):
        sep_extract(np.zeros((4, 4), np.float32), 10.0)


def test_an_unlit_flat_edge_does_not_make_a_still_sky_look_moved(monkeypatch):
    """A sensor's overscan (a band the flat's light never reached, ~0): the band is a bad pixel, and the sky's drift is
    measured without it; a still sky stays still, so its stars are not taken for hot pixels (a set of 20 s frames on
    one field read as moved by 1000 px had 70 000 star pixels flagged)."""
    from scipy.ndimage import median_filter

    from astro_gpu.stages import calibrate as cal
    rng = np.random.default_rng(3)
    H, W = 256, 320
    yy, xx = np.mgrid[0:H, 0:W]
    sky = np.zeros((H, W), np.float32)
    stars = rng.uniform(20, [H - 20, W - 60], (40, 2))
    for y, x in stars:
        sky += 3000 * np.exp(-0.5 * (((xx - x) / 1.5) ** 2 + ((yy - y) / 1.5) ** 2))
    frames = {f"f{i}": 100 + sky + rng.normal(0, 5, (H, W)).astype(np.float32) for i in range(16)}
    monkeypatch.setattr(cal, "read_cfa", lambda p: frames[p].copy())
    flat = np.ones((H, W), np.float32)
    flat[:, -30:] = 1e-5                                                     # the unlit overscan band
    dark = np.full((H, W), 100, np.float32)
    bad = flat < cal.FLAT_UNLIT * float(np.median(flat))                     # as the calibration flags it
    msg, moved = cal._hot_from_lights(list(frames), dark, flat, bad, median_filter, cfa="GGGG")
    assert not moved and "lone pixels" in msg
    cores = [(int(round(y)), int(round(x))) for y, x in stars]
    assert not any(bad[y, x] for y, x in cores)                              # no star taken for a hot pixel


def test_the_master_frames_are_read_in_their_place_whichever_is_read_first(monkeypatch):
    """The files of the masters are read on several threads: each lands at its own rank, and every one is counted."""
    import time

    from astro_gpu.stages import calibrate as cal
    paths = [str(i) for i in range(12)]

    def slow(q):
        time.sleep(0.01 * (12 - int(q)))                     # the first ones come back last
        return np.full((2, 2), float(q), np.float32)

    monkeypatch.setattr(cal, "read_cfa", slow)
    got = np.full((12, 2, 2), -1.0, np.float32)
    ticks = []

    def put(i, a):
        got[i] = a

    cal._read_frames(paths, put, 1, ticks.append)
    assert got[0, 0, 0] == -1.0 and [float(v) for v in got[1:, 0, 0]] == [float(i) for i in range(1, 12)]
    assert ticks == list(range(2, 13))
