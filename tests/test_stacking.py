"""Registration's star measures and the stack's settings on a toy night (CPU): the Moffat FWHM, the frame selection
(automatic, k x sigma, thresholds) with its reasons, the coverage rule and the method advice, the Classic method, the
drizzle with the integration's rejection, what the API serves; and the store keys of projects that still store
removed settings.

The toy night is a v1 project with done inventory, calibrate (a float16 cache of CFA frames) and register results:
stars with a colour of their own and red and blue images offset from the green one, dithered by a few pixels, a
satellite trail, a noisier frame and a hazy one."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astro_gpu.core.determinism import deterministic
from astro_gpu.core.journal import Journal
from astro_gpu.core.project import Inputs, Project
from astro_gpu.core.stage import Context, StageError

H, W = 320, 448
CH = np.array([0, 1, 1, 2])                    # RGGB: the colour of each 2 x 2 position k = 2 row + column



# ---------------------------------------------------------------------------------------------- the toy night

def toy_night(root: Path, n: int = 12, n_stars: int = 360, seed: int = 0, red=(0.45, 0.0), blue=(0.0, -0.35),
              red_drift: float = 0.0, noisy: int = 5, hazy: int = 7, trail: int = 3, sigma: float = 1.0,
              size: tuple[int, int] = (H, W)):
    """A v1 project with done inventory, calibrate and register results (coef from the true dithers), and the truth.

    Frame ``noisy`` has twice the read noise; frame ``hazy`` loses 35 % of its blue and 15 % of its red and green;
    frame ``trail`` holds a satellite trail along a row; the red image of every star is ``red`` px off the green one
    (plus ``red_drift`` x the frame's place in the night), the blue one ``blue``."""
    Hs, Ws = size
    rng = np.random.default_rng(seed)
    proj = Project.create(root, "toy", Inputs())
    sx, sy = rng.uniform(16, Ws - 16, n_stars), rng.uniform(16, Hs - 16, n_stars)
    flux = np.exp(rng.uniform(np.log(600), np.log(40000), n_stars))
    col = np.c_[rng.uniform(0.6, 1.4, n_stars), np.ones(n_stars), rng.uniform(0.6, 1.4, n_stars)]
    dith = rng.uniform(-3, 3, (n, 2))
    t = 1 - 0.2 * rng.random(n)
    t_rgb = np.repeat(t[:, None], 3, 1)
    t_rgb[hazy] *= [0.85, 0.85, 0.65]
    rn = np.full(n, 6.0)
    rn[noisy] = 12.0
    yy, xx = np.mgrid[0:Hs, 0:Ws]
    chan = CH[2 * (yy % 2) + (xx % 2)]
    cache = root / "lights.f16"
    mm = np.memmap(cache, dtype=np.float16, mode="w+", shape=(n, Hs, Ws))
    stats = np.zeros((n, 4, 2), np.float32)
    coef = np.zeros((n, 2, 10))
    R = 7
    for i in range(n):
        u, v = xx / (Ws / 2) - 1, yy / (Hs / 2) - 1
        img = 300 + 20 * i / n + (8 + i) * u - 6 * v + 4 * u * v                  # each frame its own sky gradient
        img = img * np.where(chan == 2, t_rgb[i, 2] / t[i], 1.0)
        for c in range(3):
            off = np.zeros(2)
            if c == 0:
                off = np.array(red) + [red_drift * i / max(n - 1, 1), 0.0]
            elif c == 2:
                off = np.array(blue)
            px, py = sx + dith[i, 0] + off[0], sy + dith[i, 1] + off[1]
            amp = flux * col[:, c] * t_rgb[i, c] / (2 * np.pi * sigma ** 2)
            for x0, y0, a in zip(px, py, amp):
                ix, iy = int(round(x0)), int(round(y0))
                ys, xs = slice(max(iy - R, 0), min(iy + R + 1, Hs)), slice(max(ix - R, 0), min(ix + R + 1, Ws))
                g = a * np.exp(-0.5 * ((xx[ys, xs] - x0) ** 2 + (yy[ys, xs] - y0) ** 2) / sigma ** 2)
                img[ys, xs] += np.where(chan[ys, xs] == c, g, 0.0)
        img = img + rng.normal(0, 1, img.shape) * np.sqrt(np.maximum(img, 0) / 1.0 + rn[i] ** 2)
        if i == trail:
            img[Hs // 2 + 9: Hs // 2 + 11] += 4000.0
        mm[i] = img
        for k in range(4):
            pl = img[k // 2::2, k % 2::2]
            stats[i, k] = [np.median(pl), 1.4826 * np.median(np.abs(pl - np.median(pl)))]
        coef[i, 0, :3] = [Ws / 2 - dith[i, 0], Ws / 2, 0]
        coef[i, 1, :3] = [Hs / 2 - dith[i, 1], 0, Hs / 2]
    mm.flush()
    del mm
    lights = [{"path": f"L{i:02d}.fits", "time": f"2026-01-01T{21 + i // 6:02d}:{(i % 6) * 10:02d}:00"}
              for i in range(n)]

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
    stage("calibrate", {"cache.json": {"path": str(cache), "n": n, "height": Hs, "width": Ws, "scale": 1.0,
                                       "cfa": "RGGB", "white_level": 60000.0, "dark_level": 0.0},
                        "frame_stats.npy": stats, "master_dark.npy": np.zeros((Hs, Ws), np.float32),
                        "master_flat.npy": np.ones((Hs, Ws), np.float32)}, {"cache_scale": 1.0})
    noise = 2 * np.array([np.sqrt(300 + r ** 2) for r in rn])
    stage("register", {"reg.npz": {"coef": coef, "transp": t, "fwhm": 4.7 + 0.1 * rng.normal(size=n),
                                   "rms": np.full(n, 0.1), "ref": n // 2, "nmatch": np.full(n, 100),
                                   "nstars": np.full(n, 150), "noise": noise, "roundness": np.full(n, 0.95)}})
    truth = SimpleNamespace(n=n, t=t, t_rgb=t_rgb, rn=rn, dith=dith, red=red, blue=blue, red_drift=red_drift,
                            noisy=noisy, hazy=hazy, trail=trail, fwhm=2.3548 * sigma, sx=sx, sy=sy, flux=flux)
    return proj, truth



class InProcessPool:
    """multiprocessing's Pool, in this process: the registration's workers without spawning."""

    def __init__(self, processes=None, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, f, items, chunksize=1):
        return (f(i) for i in items)


def in_process(module):
    module.get_context = lambda kind=None: SimpleNamespace(Pool=InProcessPool)
    return module


def run_stage(proj, stage, params, device="cpu"):
    ctx = Context(proj, stage, Journal(proj.logs, echo=False), device)
    with deterministic():
        stage.run(ctx, params)
    return ctx


def copy_project(src: Path, dst: Path, stages=("inventory", "calibrate", "register")) -> Project:
    """Another v1 project reading the same cache, with copies of the given stage folders."""
    import shutil
    proj = Project.create(dst, "toy", Inputs())
    for s in stages:
        shutil.copytree(src / "work" / s, proj.stage_dir(s), dirs_exist_ok=True)
    return proj


@pytest.fixture(scope="module")
def night(tmp_path_factory):
    root = tmp_path_factory.mktemp("night")
    proj, truth = toy_night(root / "p")
    return root, proj, truth


@pytest.fixture(scope="module")
def small_night(tmp_path_factory):
    """A smaller night for the Classic method's runs."""
    root = tmp_path_factory.mktemp("small")
    proj, truth = toy_night(root / "p", n=8, n_stars=140, noisy=2, hazy=4, trail=3, size=(192, 256))
    return root, proj, truth


def stack_with(night, name: str, **settings):
    """The stack of the toy night with these settings (scale 1, its sensor model), in a project of its own."""
    from astro_gpu.stages.stack import Stack, StackParams
    root, proj, truth = night
    p = copy_project(proj.root, root / name)
    params = {"scale": 1.0, "pixfrac": 1.0, "gain": 1.0, "read_noise": 6.0, "half_block": 2, **settings}
    ctx = run_stage(p, Stack(), StackParams(**params))
    return p, ctx


# ---------------------------------------------------------------------------------------------- the store keys

def _project2(tmp_path):
    """A v2 project (nothing computed) with two workflows, one with stored settings (pixfrac at its default among
    them) and a frame excluded by hand."""
    from astro_gpu.core import project2 as p2
    from astro_gpu.core import workflow as wfl
    p = p2.create(tmp_path / "proj", "keys")
    (tmp_path / "raw").mkdir()
    for i in range(3):
        (tmp_path / "raw" / f"L{i}.fits").write_bytes(b"x" * (100 + i))
    p.sessions.append(p2.Session(1, "night", lights=[str(tmp_path / "raw")]))
    p.add_workflow("Classic stack", wfl.TEMPLATES["Classic stack"])
    w2 = p.add_workflow("Full", wfl.TEMPLATES["Full"])
    w2.params = {"stack": {"pixfrac": 0.7, "scale": 1.5, "kappa": 4.0}, "register": {"detect_sigma": 6.0}}
    p.excluded.append((1, "L1.fits"))
    return p


def _keys_of(p):
    from astro_gpu.core.store import InputScan, Keys
    k = Keys(p, scan=InputScan())
    return {w.id: k.keys(w.id) for w in p.workflows}


def test_removed_settings_stored_in_a_project_are_ignored(tmp_path):
    """The old star size measure and the old drizzle scale rule are gone: a project that still stores them gets the
    keys of the current behaviour, whose last values stay in every key (stages/v2.KEY_CONSTANTS) so that the results
    computed with them keep their keys; the settings removed earlier are ignored the same way."""
    from astro_gpu.core.store import Keys
    from astro_gpu.stages.v2 import KEY_CONSTANTS, RETIRED
    assert {"fwhm_method"} <= RETIRED["register"] and {"scale_rule"} <= RETIRED["stack"]
    p = _project2(tmp_path)
    before = _keys_of(p)
    w = p.workflows[1]
    parts = Keys(p).key_parts(w.id, "stack")["p"]
    assert parts["scale_rule"] == KEY_CONSTANTS["stack"]["scale_rule"] == "coverage"
    assert Keys(p).key_parts(w.id, "register")["p"]["fwhm_method"] == "moffat"
    for x in p.workflows:
        x.params.setdefault("register", {}).update(fwhm_method="legacy", channel_align="per_block")
        x.params.setdefault("stack", {}).update(scale_rule="legacy", rejection="hubble", weighting="noise",
                                                normalisation="local")
    assert _keys_of(p) == before


def test_the_thresholds_selection_keeps_its_keys_under_its_old_name(tmp_path):
    """``selection = thresholds`` was stored as "legacy": both read the same, with the key such results always had
    (the previous default, out of the key); the default (minimum weight) is another result."""
    from astro_gpu.stages.stack import StackParams
    assert StackParams(selection="legacy").selection == "thresholds"
    p = _project2(tmp_path)
    w = p.workflows[1]
    default = _keys_of(p)[w.id]
    w.params["stack"]["selection"] = "legacy"
    old = _keys_of(p)[w.id]
    w.params["stack"]["selection"] = "thresholds"
    assert _keys_of(p)[w.id] == old and old["stack"] != default["stack"]
    assert old["register"] == default["register"]


def test_the_colour_correction_option_keeps_the_keys_of_its_default(tmp_path):
    """The colour stage's ``correction`` at its default ("full"), stored or not, keeps every key; "per_colour"
    changes the colour key and what follows it, not what comes before."""
    from astro_gpu.stages.color import ColorParams
    assert ColorParams().correction == "full"
    p = _project2(tmp_path)
    before = _keys_of(p)
    for w in p.workflows:
        w.params.setdefault("color", {})["correction"] = "full"
    assert _keys_of(p) == before
    w = p.workflows[0]
    w.params["color"]["correction"] = "per_colour"
    after = _keys_of(p)[w.id]
    assert after["stack"] == before[w.id]["stack"] and after["color"] != before[w.id]["color"]
    with pytest.raises(ValueError):
        ColorParams(correction="matrix")


def test_the_classic_part_has_a_revision_of_its_own_in_the_keys(tmp_path):
    """The Classic part's revision (stack.CLASSIC_VERSION) is in the key of the stacks that use it, the Classic method
    and the drizzle with the integration's rejection, and of no other."""
    from astro_gpu.core.store import Keys
    from astro_gpu.stages.stack import CLASSIC_VERSION, Stack
    p = _project2(tmp_path)
    w = p.workflows[1]
    drizzle = Keys(p).key_parts(w.id, "stack")
    assert drizzle["v"] == Stack.version
    for kv in ({"method": "classic"}, {"drizzle_rejection": "integration"}):
        w.params["stack"].update(kv)
        parts = Keys(p).key_parts(w.id, "stack")
        assert parts["v"] == f"{Stack.version}+classic{CLASSIC_VERSION}" and parts["p"] == {**drizzle["p"], **kv}
        assert Keys(p).key(w.id, "stack") != Keys(p).key(p.workflows[0].id, "stack")
        for k in kv:
            del w.params["stack"][k]
    w.params["stack"]["drizzle_rejection"] = "own"           # the default stored: the drizzle's key
    assert Keys(p).key_parts(w.id, "stack") == drizzle


def test_pixfrac_is_automatic(tmp_path):
    """stack.pixfrac is set by the coverage rule: stored at its default, it overrides the automatic drop size and
    stays in the key."""
    from astro_gpu.core.store import Keys, key_rules
    assert "pixfrac" in key_rules().auto_fields("stack")
    p = _project2(tmp_path)
    assert Keys(p).key_parts(p.workflows[1].id, "stack")["p"]["pixfrac"] == 0.7


# ---------------------------------------------------------------------------------------------- register

@pytest.fixture(scope="module")
def registered(night):
    """Register run on the toy night (in this process)."""
    from astro_gpu.stages import register as reg
    in_process(reg)
    root, proj, truth = night
    p = copy_project(proj.root, root / "reg", ("inventory", "calibrate"))
    ctx = run_stage(p, reg.Register(), reg.RegisterParams(workers=1))
    return p, ctx, truth


def test_moffat_fwhm_is_measured_beside_the_2x2_one(registered):
    from astro_gpu.stages import register as reg
    p, ctx, truth = registered
    assert "fwhm_method" not in reg.RegisterParams.model_fields
    z = np.load(p.stage_dir("register") / "reg.npz")
    assert {"fwhm_moffat", "roundness_moffat", "beta_moffat", "stars_moffat"} <= set(z.files)
    assert np.nanmedian(z["fwhm_moffat"]) == pytest.approx(truth.fwhm, rel=0.08)
    assert np.nanmedian(z["fwhm"]) > 1.3 * truth.fwhm                      # the 2 x 2 measure
    assert np.nanmedian(z["roundness_moffat"]) > 0.85
    assert ctx.metrics["fwhm_moffat_px"] == pytest.approx(truth.fwhm, rel=0.08)


# ---------------------------------------------------------------------------------------------- the stack

def test_the_removed_settings_are_gone():
    from astro_gpu.stages import stack as st
    for name in ("rejection", "weighting", "normalisation", "scale_rule"):
        assert name not in st.StackParams.model_fields and name not in st.SWITCHES
    for mod in ("rejection_hubble", "channel_align"):
        with pytest.raises(ImportError):
            __import__(f"astro_gpu.core.{mod}")


def test_the_defaults():
    from astro_gpu.stages.stack import SELECTIONS, StackParams
    p = StackParams()
    assert (p.method, p.selection, p.drizzle_rejection) == ("drizzle", "min_weight", "own")
    assert SELECTIONS == ("min_weight", "sota", "thresholds")


def test_the_colour_alignment_for_the_whole_night_is_on(night):
    """The whole night's channel alignment, measured on the first mean, is applied by default."""
    from astro_gpu.stages.stack import DIAG
    p, ctx = stack_with(night, "chan_night")
    info = ctx.metrics["channel_offsets"]
    assert info["applied"] and info["centre_px"]["R"][0] == pytest.approx(night[2].red[0], abs=0.1)
    diag = json.loads((p.stage_dir("stack") / DIAG).read_text())
    assert diag["channel_alignment"]["mode"] == "night" and diag["channel_alignment"]["applied"]


def test_the_default_selection_keeps_every_frame_above_the_minimum_weight(night, tmp_path):
    """By default only a frame weighing less than 5 % of the best one is left out, with its reason; the hazy and the
    noisy frames are kept with their lower weight."""
    from astro_gpu.core import frame_weights as fwm
    from astro_gpu.stages.stack import DIAG, Stack, StackParams, quality_weights
    root, proj, truth = night
    p = copy_project(proj.root, tmp_path / "minw")
    z = dict(np.load(p.stage_dir("register") / "reg.npz"))
    z["transp"] = z["transp"].copy()
    z["transp"][9] = 0.15                                              # a cloud on frame 9: weight about 2 %
    np.savez(p.stage_dir("register") / "reg.npz", **z)
    ctx = run_stage(p, Stack(), StackParams(scale=1.0, pixfrac=1.0, gain=1.0, read_noise=6.0, half_block=2))
    rows = json.loads((p.stage_dir("stack") / "frames.json").read_text())
    assert [i for i, r in enumerate(rows) if r["weight"] == 0] == [9]
    assert rows[9]["exclusion_codes"] == "weight" and rows[9]["exclusions"][0]["template"] == fwm.LOW_WEIGHT
    assert rows[9]["excluded"].endswith("under the minimum of 5 %")
    assert rows[truth.hazy]["weight"] > 0 and rows[truth.noisy]["weight"] > 0
    stats = np.load(p.stage_dir("calibrate") / "frame_stats.npy")
    w, _ = quality_weights(z["transp"], z["fwhm"], z["rms"], stats, z["noise"])
    kept = np.arange(truth.n) != 9
    np.testing.assert_allclose([r["weight"] for r in rows if r["weight"] > 0], w[kept])
    assert ctx.metrics["frames_used"] == truth.n - 1
    diag = json.loads((p.stage_dir("stack") / DIAG).read_text())
    assert diag["selection"]["selection"] == "min_weight" and diag["selection"]["excluded"]["9"][0]["code"] == "weight"


def test_selection_at_k_sigma_with_reasons(night, tmp_path):
    from astro_gpu.core import frame_weights as fwm
    from astro_gpu.stages.stack import Stack, StackParams
    root, proj, truth = night
    p = copy_project(proj.root, tmp_path / "sel")
    z = dict(np.load(p.stage_dir("register") / "reg.npz"))
    fm = np.full(truth.n, truth.fwhm) * (1 + 0.02 * np.random.default_rng(1).normal(size=truth.n))
    fm[2] = 2.0 * truth.fwhm                                           # wind on frame 2
    z["fwhm_moffat"], z["roundness_moffat"] = fm, np.full(truth.n, 0.95)
    np.savez(p.stage_dir("register") / "reg.npz", **z)
    params = dict(scale=1.0, pixfrac=1.0, gain=1.0, read_noise=6.0, half_block=2, selection="sota")
    ctx = run_stage(p, Stack(), StackParams(**params))
    rows = json.loads((p.stage_dir("stack") / "frames.json").read_text())
    assert rows[2]["weight"] == 0 and "fwhm" in rows[2]["exclusion_codes"]
    assert rows[2]["exclusions"][0]["template"] in fwm.REASON_TEMPLATES
    assert "wind, focus or seeing" in rows[2]["excluded"]
    kept = sum(r["weight"] > 0 for r in rows)
    assert ctx.metrics["frames_used"] == kept
    # the transparency filter off keeps more (never fewer) frames
    run_stage(p, Stack(), StackParams(**params, selection_transparency="off"))
    rows2 = json.loads((p.stage_dir("stack") / "frames.json").read_text())
    assert sum(r["weight"] > 0 for r in rows2) >= kept
    assert not any("transparency" in r["exclusion_codes"] for r in rows2)
    # the default (no filter) keeps the windy frame, with its weight
    run_stage(p, Stack(), StackParams(**{k: v for k, v in params.items() if k != "selection"}))
    rows3 = json.loads((p.stage_dir("stack") / "frames.json").read_text())
    assert rows3[2]["weight"] > 0


def test_selection_by_thresholds_with_reasons(night, tmp_path):
    """The user's thresholds: a frame under the minimum transparency is left out with its reason (and its code), as
    is a frame excluded by hand; the others are kept."""
    from astro_gpu.stages.stack import DIAG, Stack, StackParams
    root, proj, truth = night
    p = copy_project(proj.root, tmp_path / "thr")
    z = dict(np.load(p.stage_dir("register") / "reg.npz"))
    z["transp"] = z["transp"].copy()
    z["transp"][5] = 0.3                                               # haze on frame 5
    np.savez(p.stage_dir("register") / "reg.npz", **z)
    ctx = run_stage(p, Stack(), StackParams(scale=1.0, pixfrac=1.0, gain=1.0, read_noise=6.0, half_block=2,
                                            selection="thresholds", min_transparency=0.6, exclude=["L01.fits"]))
    rows = json.loads((p.stage_dir("stack") / "frames.json").read_text())
    out = {i: r["exclusion_codes"] for i, r in enumerate(rows) if r["weight"] == 0}
    assert out == {1: "by_hand", 5: "transparency"}
    assert ctx.metrics["frames_used"] == truth.n - 2
    diag = json.loads((p.stage_dir("stack") / DIAG).read_text())
    assert diag["selection"]["selection"] == "thresholds"


def test_coverage_rule_and_method_advice_on_the_frames_kept(night, tmp_path):
    """The scale, the drop and the method advice judge the frames the selection keeps (not every aligned frame): the
    advice counts them; it is an advice (never applied)."""
    from astro_gpu.core import coverage as cv
    from astro_gpu.core.params import resolve
    from astro_gpu.stages.stack import Stack, StackParams
    root, proj, truth = night
    p = copy_project(proj.root, tmp_path / "cov")
    z = dict(np.load(p.stage_dir("register") / "reg.npz"))
    z["fwhm_moffat"], z["roundness_moffat"] = np.full(truth.n, 1.6), np.full(truth.n, 0.95)
    np.savez(p.stage_dir("register") / "reg.npz", **z)
    ctx = Context(p, Stack(), Journal(p.logs, echo=False), "cpu")
    a = Stack().auto(ctx)
    assert a["scale"].template in cv.REASON_TEMPLATES and a["pixfrac"].template in cv.REASON_TEMPLATES
    assert a["scale"].value in (1.0, 1.5, 2.0) and 0.5 <= a["pixfrac"].value <= 1.0
    m = a["method"]
    assert m.advice and m.value == "classic" and m.template == cv.M_FEW and m.args == {"n": truth.n}
    params, origin = resolve(StackParams, a, {})
    assert params.method == "drizzle" and origin["method"] == "default"      # never applied
    p.set_override("stack", "exclude", ["L00.fits", "L01.fits"])          # two frames out by hand: 10 judged
    assert Stack().auto(ctx)["method"].args == {"n": truth.n - 2}


def test_coverage_is_kept_per_colour(night):
    from astro_gpu.core import coverage as cv
    from astro_gpu.stages.stack import COVERAGE, DIAG
    p, ctx = stack_with(night, "cov_on")
    with np.load(p.stage_dir("stack") / COVERAGE) as z:
        assert z["measured"].shape == (3, H, W) and z["estimate_map"].shape[0] == 3 and len(z["variation"]) == 3
        var = z["variation"]
    assert var[1] < var[0] and var[1] < var[2]               # green has twice the samples of red and blue
    assert ctx.metrics["coverage_variation_pct"] == [round(100 * v, 1) for v in var]
    diag = json.loads((p.stage_dir("stack") / DIAG).read_text())
    rule = diag["coverage"]["rule"]
    assert rule["scale"]["template"] and diag["switches"]["method"] == "drizzle"
    assert rule["method"]["value"] == "classic" and rule["method"]["template"] == cv.M_FEW



def test_classic_method_says_when_it_is_not_available(night, monkeypatch):
    from astro_gpu.stages import stack as st
    monkeypatch.setattr(st, "classic_missing", lambda: ["debayer"])
    with pytest.raises(StageError, match="not available yet"):
        stack_with(night, "classic_missing", method="classic")


def test_classic_method_writes_the_drizzle_files(small_night):
    """method = classic (core/stack_classic.py through its hook): debayered and resampled at the sensor's resolution
    whatever the drizzle scale, rejected by the frame count (8 frames: winsorized), the trail gone; the drizzle's files
    and records, so that the stages after it read them unchanged; its own settings reach it."""
    from astro_gpu.stages import stack as st
    if st.classic_missing():
        pytest.skip(f"the Classic modules are not there: {st.classic_missing()}")
    root, proj, truth = small_night
    h, w = 192, 256
    p, ctx = stack_with(small_night, "classic", method="classic", scale=1.5)
    d = p.stage_dir("stack")
    for f in ("drz_full.npy", "drz_A.npy", "drz_B.npy", "drz_weight.npy", st.SATURATION, st.REJECTION, st.DIAG,
              "drz_meta.json"):
        assert (d / f).is_file(), f
    full = np.load(d / "drz_full.npy")
    assert full.shape == (3, h, w) and ctx.metrics["grid"] == [w, h] and np.isfinite(full).all()
    diag = json.loads((d / st.DIAG).read_text())
    assert diag["rejection"]["method"] == "classic" and diag["rejection"]["classic"] == "winsorized"
    assert diag["switches"]["method"] == "classic" and diag["switches"]["classic_debayer"] == "rcd"
    assert {"offset", "scale", "rejected_high", "half"} <= set(diag["frames"][0])
    with np.load(d / st.REJECTION) as z:
        high, f = z["high"], int(z["factor"])
    row = (h // 2 + 10 - truth.dith[truth.trail, 1]) / f
    assert (high[int(row) - 2:int(row) + 3].sum(0)[8:-8] > 0).mean() > 0.9      # the satellite trail
    meta = json.loads((d / "drz_meta.json").read_text())
    assert meta["method"] == "classic" and meta["scale"] == 1.0 and meta["cfa_balance"] is True
    # its own settings
    p2, _ = stack_with(small_night, "classic_pct", method="classic", classic_rejection="percentile",
                       classic_interpolation="bilinear", classic_cfa_balance="off")
    diag2 = json.loads((p2.stage_dir("stack") / st.DIAG).read_text())
    assert diag2["rejection"]["classic"] == "percentile" and diag2["switches"]["classic_interpolation"] == "bilinear"
    assert json.loads((p2.stage_dir("stack") / "drz_meta.json").read_text())["cfa_balance"] is False
    assert diag2["switches"]["classic_cfa_balance"] == "off" and diag2["frames"][0]["cfa_balance"] is None
    assert not np.array_equal(np.load(p2.stage_dir("stack") / "drz_full.npy"), full)


def test_drizzle_with_the_integrations_rejection(small_night):
    """drizzle_rejection = integration (the CFA drizzle of an image integration): the Classic method's rejection on the debayered,
    registered frames (8 frames: winsorized), applied to the drops; the trail is gone from the drizzle, the rejection's
    rate and map are in the diagnostics as the drizzle's own are, and the drizzle's grid and files stay."""
    from astro_gpu.stages import stack as st
    root, proj, truth = small_night
    h, w = 192, 256
    p, ctx = stack_with(small_night, "integration", drizzle_rejection="integration")
    d = p.stage_dir("stack")
    full = np.load(d / "drz_full.npy")
    assert full.shape == (3, h, w) and np.isfinite(full).all()
    diag = json.loads((d / st.DIAG).read_text())
    rj = diag["rejection"]
    assert rj["method"] == "integration" and rj["classic"] == "winsorized" and rj["reason_template"]["template"]
    assert 0 < rj["rate_pct"] < 5 and len(rj["rate_rgb_pct"]) == 3 and 0 < rj["drizzle_excluded_pct"] < 5
    assert ctx.metrics["rejected_pct"] == round(rj["rate_pct"], 3) and "drizzle_excluded_pct" in ctx.metrics
    assert diag["switches"]["drizzle_rejection"] == "integration" and diag["switches"]["classic_cfa_balance"] == "on"
    fr = {f["index"]: f for f in diag["frames"]}
    clean = [f["rejected_high"] for i, f in fr.items() if i != truth.trail]
    assert fr[truth.trail]["rejected_high"] > max(clean) and "tested" in fr[truth.trail]
    with np.load(d / st.REJECTION) as z:
        high, f = z["high"], int(z["factor"])
    row = h // 2 + 10 - truth.dith[truth.trail, 1]
    assert (high[int(row / f) - 2:int(row / f) + 3].sum(0)[8:-8] > 0).mean() > 0.9      # the satellite trail
    # the trail gone from the image: its rows no brighter than the rows around (green, a robust level per row)
    g = np.median(full[1, :, 16:-16], axis=1)
    r0 = int(round(row))
    around = np.r_[g[r0 - 8:r0 - 3], g[r0 + 4:r0 + 9]]
    assert g[r0 - 1:r0 + 2].max() - np.median(around) < 5 * (around.std() + 1.0)
    # the drizzle with its own rejection is another result
    p2, _ = stack_with(small_night, "own")
    assert not np.array_equal(np.load(p2.stage_dir("stack") / "drz_full.npy"), full)



def test_choice_values_are_checked():
    from astro_gpu.stages.stack import StackParams
    for name in ("selection", "classic_rejection", "drizzle_rejection", "classic_cfa_balance"):
        with pytest.raises(ValueError, match=name):
            StackParams(**{name: "wrong"})


# ---------------------------------------------------------------------------------------------- the API

def _served_by(monkeypatch, proj):
    """api's result folders pointed at a v1 toy project's stage folders."""
    from astro_gpu import api
    monkeypatch.setattr(api, "_need", lambda p, wf, stage, which="current": proj.stage_dir(stage))
    monkeypatch.setattr(api, "_upstream", lambda p, wf, d, up, which: proj.stage_dir(up))
    return api


@pytest.mark.parametrize("selection", ["min_weight", "sota", "thresholds"])
def test_api_serves_the_stack_diagnostics(night, monkeypatch, selection):
    root, proj, truth = night
    p, ctx = stack_with(night, f"api_{selection}", selection=selection)
    (p.stage_dir("stack") / "manifest.json").write_text(json.dumps({"status": "done", "metrics": ctx.metrics}))
    api = _served_by(monkeypatch, p)
    s = api.stack_diagnostics(None, 1)
    assert s.rejection["method"] == "own" and s.coverage["maps"].shape == (3, H, W) and "estimate" in s.coverage
    assert set(s.coverage["variation"]) == set("RGB") and len(s.weights_rgb) == truth.n
    assert s.channel_alignment["mode"] == "night" and s.used
    assert all(r and r[0]["template"] for r in s.exclusions.values())
    assert s.selection["selection"] == selection and s.switches["selection"] == selection


def test_api_registration_table_with_the_moffat_fit(registered, monkeypatch):
    p, _, truth = registered
    api = _served_by(monkeypatch, p)
    t = api.registration_table(None, 1)
    r = t[truth.n // 2]
    assert r.fwhm_moffat == pytest.approx(truth.fwhm, rel=0.1)
