"""Stacking v2 behind its switches (docs/spec/STACKING.md): every switch at its default is the previous version bit for
bit with the same store keys; each switch on shows its module's effect on a toy night (CPU).

The previous version is the engine of BASE (git), loaded from git; the toy night is a v1 project with done inventory,
calibrate (a float16 cache of CFA frames) and register results: stars with a colour of their own and red and blue
images offset from the green one, dithered by a few pixels, a satellite trail, a noisier frame and a hazy one."""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astro_gpu.core.determinism import deterministic
from astro_gpu.core.journal import Journal
from astro_gpu.core.project import Inputs, Project
from astro_gpu.core.stage import Context, StageError

REPO = Path(__file__).resolve().parents[1]
#: the engine before stacking v2 was wired in (stack.py and register.py as the modules were committed)
BASE = "bdbbc77"
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
    """multiprocessing's Pool, in this process: the registration's workers without spawning (and the engine of BASE,
    loaded from git, has no file a spawned worker could import)."""

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


def base_module(path: str, name: str):
    """A module of the engine at BASE (from git), in its package (relative imports work)."""
    try:
        src = subprocess.run(["git", "show", f"{BASE}:{path}"], cwd=REPO, capture_output=True, text=True,
                             check=True, encoding="utf-8").stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {BASE} not available")
    pkg = path.rsplit("/", 1)[0].replace("/", ".")
    mod = types.ModuleType(f"{pkg}.{name}")
    mod.__package__ = pkg
    sys.modules[mod.__name__] = mod
    exec(compile(src, f"{BASE}:{path}", "exec"), mod.__dict__)
    return mod


def git_text(path: str) -> str:
    try:
        return subprocess.run(["git", "show", f"{BASE}:{path}"], cwd=REPO, capture_output=True, text=True,
                              check=True, encoding="utf-8").stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {BASE} not available")


def _function_source(src: str, name: str) -> str:
    import ast
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, node) + "\n"


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


def same_outputs(a: Path, b: Path) -> list[str]:
    """The files that differ between two stage folders (arrays bit for bit, JSON equal, other files byte for byte);
    the manifests and checkpoints left out."""
    skip = {"manifest.json"}
    fa = {p.relative_to(a) for p in a.rglob("*") if p.is_file() and p.name not in skip and "_ckpt" not in p.parts}
    fb = {p.relative_to(b) for p in b.rglob("*") if p.is_file() and p.name not in skip and "_ckpt" not in p.parts}
    diff = sorted(str(x) for x in fa ^ fb)
    for rel in sorted(fa & fb):
        x, y = a / rel, b / rel
        if x.suffix == ".npy":
            u, v = np.load(x), np.load(y)
            ok = u.dtype == v.dtype and u.shape == v.shape and u.tobytes() == v.tobytes()
        elif x.suffix == ".npz":
            with np.load(x) as zu, np.load(y) as zv:
                ok = sorted(zu.files) == sorted(zv.files) and all(
                    zu[k].dtype == zv[k].dtype and zu[k].tobytes() == zv[k].tobytes() for k in zu.files)
        elif x.suffix == ".json":
            ok = json.loads(x.read_text()) == json.loads(y.read_text())
        else:
            ok = x.read_bytes() == y.read_bytes()
        if not ok:
            diff.append(str(rel))
    return diff


@pytest.fixture(scope="module")
def night(tmp_path_factory):
    root = tmp_path_factory.mktemp("night")
    proj, truth = toy_night(root / "p")
    return root, proj, truth


@pytest.fixture(scope="module")
def small_night(tmp_path_factory):
    """A smaller night for the bit-for-bit comparisons (no star count to reach)."""
    root = tmp_path_factory.mktemp("small")
    proj, truth = toy_night(root / "p", n=8, n_stars=140, noisy=2, hazy=4, trail=3, size=(192, 256))
    return root, proj, truth


def stack_with(night, name: str, **switches):
    """The stack of the toy night with these settings (scale 1, its sensor model), in a project of its own."""
    from astro_gpu.stages.stack import Stack, StackParams
    root, proj, truth = night
    p = copy_project(proj.root, root / name)
    params = {"scale": 1.0, "pixfrac": 1.0, "gain": 1.0, "read_noise": 6.0, "half_block": 2, **switches}
    ctx = run_stage(p, Stack(), StackParams(**params))
    return p, ctx


# ---------------------------------------------------------------------------------------------- the previous version

STACK_CASES = {
    "defaults": {},
    "drizzle 1.5 / 0.7, fixed pattern, a frame by hand": {"scale": 1.5, "pixfrac": 0.7, "pattern_removal": True,
                                                         "pattern_frames": 8, "exclude": ["L01.fits"]},
}


@pytest.mark.parametrize("case, explicit", [("defaults", True), ("defaults", False),
                                            ("drizzle 1.5 / 0.7, fixed pattern, a frame by hand", True)],
                         ids=["defaults set", "defaults left out", "drizzle 1.5, pattern, by hand"])
def test_every_switch_at_its_default_is_the_previous_stack_bit_for_bit(small_night, case, explicit):
    """The stack of BASE and today's, with every switch left out or set to its default value: the same files, bit
    for bit (the manifest aside)."""
    from astro_gpu.stages.stack import SWITCHES, Stack, StackParams
    old = base_module("astro_gpu/stages/stack.py", "_stack_base")
    root, proj, _ = small_night
    params = {"gain": 1.0, "read_noise": 6.0, "half_block": 2, "min_transparency": 0.6, **STACK_CASES[case]}
    tag = f"{case[:8]}_{explicit}".replace(" ", "_")
    pa = copy_project(proj.root, root / f"old_{tag}")
    pb = copy_project(proj.root, root / f"new_{tag}")
    ca = run_stage(pa, old.Stack(), old.StackParams(**params))
    new = {**params, **({k: v[0] for k, v in SWITCHES.items()} if explicit else {})}
    cb = run_stage(pb, Stack(), StackParams(**new))
    assert same_outputs(pa.stage_dir("stack"), pb.stage_dir("stack")) == []
    assert ca.metrics == cb.metrics
    assert sorted(p.name for p in pb.stage_dir("stack").iterdir()) == sorted(p.name for p in
                                                                             pa.stage_dir("stack").iterdir())


def test_every_switch_at_its_default_is_the_previous_registration_bit_for_bit(small_night):
    """The registration of BASE and today's on the same star catalogues: the same files, bit for bit. (sep's
    extraction of a frame depends slightly on what the same process extracted before, in BASE as today: the
    catalogues come from one run, through the checkpoint both read; the worker that makes them is BASE's, unchanged.)"""
    import inspect
    import shutil

    from astro_gpu.stages import register as new
    old = in_process(base_module("astro_gpu/stages/register.py", "_register_base"))
    in_process(new)
    assert inspect.getsource(new._catalog) == _function_source(git_text("astro_gpu/stages/register.py"), "_catalog")
    root, proj, _ = small_night
    p0 = copy_project(proj.root, root / "reg_ck", ("inventory", "calibrate"))
    run_stage(p0, new.Register(), new.RegisterParams(workers=1))
    pa = copy_project(proj.root, root / "reg_old", ("inventory", "calibrate"))
    pb = copy_project(proj.root, root / "reg_new", ("inventory", "calibrate"))
    for q in (pa, pb):
        shutil.copytree(p0.stage_dir("register") / "_ckpt", q.stage_dir("register") / "_ckpt")
    ca = run_stage(pa, old.Register(), old.RegisterParams(workers=1))
    cb = run_stage(pb, new.Register(), new.RegisterParams(workers=1, fwhm_method="legacy", channel_align="legacy"))
    assert same_outputs(pa.stage_dir("register"), pb.stage_dir("register")) == []
    assert same_outputs(p0.stage_dir("register"), pb.stage_dir("register")) == []
    assert ca.metrics == cb.metrics and not (pb.stage_dir("register") / new.CHAN).exists()


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


def _keys_of(p, rules):
    from astro_gpu.core.store import InputScan, Keys
    k = Keys(p, rules=rules, scan=InputScan())
    return {w.id: k.keys(w.id) for w in p.workflows}


def base_rules():
    """The key rules of BASE: its store.py, with its stack and register (their settings before stacking v2)."""
    from astro_gpu.core import workflow as wfl
    from astro_gpu.stages import v2
    store = base_module("astro_gpu/core/store.py", "_store_base")
    stages = []
    for s in v2.V2_STAGES():
        if s.name == "stack":
            s = base_module("astro_gpu/stages/stack.py", "_stack_base").Stack()
        elif s.name == "register":
            s = base_module("astro_gpu/stages/register.py", "_register_base").Register()
        stages.append(s)
    chains = wfl.Chains(stages, order=wfl.ORDER, extra=wfl.EXTRA, optional=wfl.OPTIONAL, blocks=wfl.BLOCKS,
                        rule=wfl.chain_stages)
    return store, store.KeyRules(chains, v2.AUTO_FIELDS, v2.PATH_FIELDS)


def test_new_settings_at_their_default_leave_every_key_unchanged(tmp_path):
    """The rule: a setting added later, at its default and without an automatic rule there, is not in the key,
    stored or not. Today's keys are BASE's; storing every new switch at its default changes none; a switch set
    changes its stage and what follows, not what comes before."""
    from astro_gpu.core.store import key_rules
    from astro_gpu.stages.register import RegisterParams
    from astro_gpu.stages.stack import StackParams
    store, rules0 = base_rules()
    p = _project2(tmp_path)
    before = {w.id: store.Keys(p, rules=rules0).keys(w.id) for w in p.workflows}
    assert _keys_of(p, key_rules()) == before
    new = {"stack": set(StackParams.model_fields) - set(rules0.params("stack").model_fields),
           "register": set(RegisterParams.model_fields) - set(rules0.params("register").model_fields)}
    assert new["stack"] == {"method", "scale_rule", "rejection", "weighting", "selection", "selection_transparency",
                            "normalisation", "classic_debayer", "classic_interpolation", "classic_clamping",
                            "classic_rejection"} and new["register"] == {"fwhm_method", "channel_align"}
    for w in p.workflows:                                   # every new setting stored at its default
        for st, names in new.items():
            w.params.setdefault(st, {}).update({k: key_rules().defaults(st)[k] for k in names})
    assert _keys_of(p, key_rules()) == before
    w = p.workflows[1]
    w.params["stack"]["rejection"] = "hubble"
    after = _keys_of(p, key_rules())[w.id]
    assert after["register"] == before[w.id]["register"] and after["calibrate"] == before[w.id]["calibrate"]
    assert after["stack"] != before[w.id]["stack"] and after["color"] != before[w.id]["color"]
    w.params["stack"]["rejection"] = "legacy"
    w.params["register"]["fwhm_method"] = "moffat"
    after = _keys_of(p, key_rules())[w.id]
    assert after["register"] != before[w.id]["register"] and after["calibrate"] == before[w.id]["calibrate"]


def test_pixfrac_is_automatic_only_under_the_coverage_rule(tmp_path):
    """stack.pixfrac stored at its default: dropped from the key under the previous scale rule (as before), kept under
    the coverage rule, whose automatic drop size it overrides (AUTO_WHEN)."""
    from astro_gpu.core.store import Keys, key_rules
    r = key_rules()
    assert "pixfrac" not in r.auto_fields("stack", {"scale_rule": "legacy"})
    assert "pixfrac" in r.auto_fields("stack", {"scale_rule": "coverage"}) and "pixfrac" in r.auto_fields("stack")
    p = _project2(tmp_path)
    w = p.workflows[1]
    assert "pixfrac" not in Keys(p).key_parts(w.id, "stack")["p"]
    w.params["stack"]["scale_rule"] = "coverage"
    kp = Keys(p).key_parts(w.id, "stack")["p"]
    assert kp["pixfrac"] == 0.7 and kp["scale_rule"] == "coverage"
    with pytest.raises(ValueError, match="scale_rule"):
        w.params["stack"]["scale_rule"] = "wrong"
        Keys(p).key(w.id, "stack")


REAL = [Path("F:/astro_projets/Horsehead"), Path("F:/astro_essais/bench/heart/ours_express")]


@pytest.mark.parametrize("root", REAL, ids=[r.name for r in REAL])
def test_real_projects_keep_their_keys(root):
    """The keys of real projects, read-only: today's equal BASE's, for every workflow and stage."""
    if not (root / "project.astroproj").is_file():
        pytest.skip(f"{root} not available")
    from astro_gpu.core import project2 as p2
    from astro_gpu.core.store import key_rules
    before = {f: f.stat().st_mtime_ns for f in root.glob("*") if f.is_file()}
    p = p2.load(root)
    store, rules0 = base_rules()
    old = {w.id: store.Keys(p, rules=rules0).keys(w.id) for w in p.workflows}
    assert old and _keys_of(p, key_rules()) == old
    assert {f: f.stat().st_mtime_ns for f in root.glob("*") if f.is_file()} == before


# ---------------------------------------------------------------------------------------------- register's switches

@pytest.fixture(scope="module")
def registered(night):
    """Register run on the toy night with both of its switches on (in this process)."""
    from astro_gpu.stages import register as reg
    in_process(reg)
    root, proj, truth = night
    p = copy_project(proj.root, root / "reg_v2", ("inventory", "calibrate"))
    ctx = run_stage(p, reg.Register(), reg.RegisterParams(workers=1, fwhm_method="moffat", channel_align="per_block"))
    return p, ctx, truth


def test_moffat_fwhm_is_measured_beside_the_old_one(registered, night):
    from astro_gpu.stages import register as reg
    p, ctx, truth = registered
    z = np.load(p.stage_dir("register") / "reg.npz")
    assert {"fwhm_moffat", "roundness_moffat", "beta_moffat", "stars_moffat"} <= set(z.files)
    assert np.nanmedian(z["fwhm_moffat"]) == pytest.approx(truth.fwhm, rel=0.08)
    assert np.nanmedian(z["fwhm"]) > 1.3 * truth.fwhm                      # the 2 x 2 measure, kept as it was
    assert np.nanmedian(z["roundness_moffat"]) > 0.85
    assert ctx.metrics["fwhm_moffat_px"] == pytest.approx(truth.fwhm, rel=0.08)
    # the old arrays are still there, the same measures (to sep's run-to-run differences: see the bit-for-bit test)
    root = night[0]
    q = copy_project(night[1].root, root / "reg_v2_off", ("inventory", "calibrate"))
    in_process(reg)
    run_stage(q, reg.Register(), reg.RegisterParams(workers=1))
    old = np.load(q.stage_dir("register") / "reg.npz")
    assert set(old.files) < set(z.files) and int(old["ref"]) == int(z["ref"])
    for k in ("fwhm", "transp", "coef", "noise"):
        assert np.allclose(old[k], z[k], rtol=2e-3, atol=2e-3), k


def test_colour_alignment_per_block_is_measured_and_saved(registered):
    from astro_gpu.core.channel_align import ChannelAlignment
    from astro_gpu.stages.register import CHAN
    p, ctx, truth = registered
    with np.load(p.stage_dir("register") / CHAN) as z:
        al = ChannelAlignment.from_npz(z)
    assert al is not None and al.applied and ctx.metrics["channel_alignment"]["applied"]
    c = al.at(truth.n // 2, [W / 2], [H / 2])                             # (3, 1, 2): offsets from green at the centre
    assert c[0, 0] == pytest.approx(truth.red, abs=0.08) and c[2, 0] == pytest.approx(truth.blue, abs=0.08)


# ---------------------------------------------------------------------------------------------- stack's switches

def test_colour_alignment_per_block_is_applied_in_the_drizzle(night, registered):
    """With register's chan.npz the stack places the red and blue samples by it (no measure on its first mean): the
    red and blue stars of the stack sit on the green ones."""
    from astro_gpu.core.psf_moffat import find_stars
    from astro_gpu.stages.register import CHAN
    from astro_gpu.stages.stack import DIAG
    root, proj, truth = night
    p, _ = stack_with(night, "chan_off")
    q, ctx = stack_with(night, "chan_on")
    import shutil
    shutil.copy(registered[0].stage_dir("register") / CHAN, q.stage_dir("register") / CHAN)
    ctx = run_stage(q, ctx.stage, ctx.stage.Params(scale=1.0, pixfrac=1.0, gain=1.0, read_noise=6.0, half_block=2))
    diag = json.loads((q.stage_dir("stack") / DIAG).read_text())
    assert diag["channel_alignment"]["mode"] == "per_block" and "channel_offsets" not in ctx.metrics
    drz = np.load(q.stage_dir("stack") / "drz_full.npy")
    xy = find_stars(drz[1], k=20)[:40]

    def centroid(img, x, y, r=3):
        ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
        a = img[int(round(y)) + ys, int(round(x)) + xs]
        a = np.clip(a - np.median(img), 0, None)
        return (a * xs).sum() / a.sum() + round(x), (a * ys).sum() / a.sum() + round(y)

    ok = (xy[:, 0] > 10) & (xy[:, 0] < W - 10) & (xy[:, 1] > 10) & (xy[:, 1] < H - 10)
    d = [np.subtract(centroid(drz[0], x, y), centroid(drz[1], x, y)) for x, y in xy[ok]]
    assert np.abs(np.median(d, axis=0)).max() < 0.1


def test_coverage_rule_sets_scale_and_drop_with_its_reason(night, tmp_path):
    from astro_gpu.core.coverage import FRENCH
    from astro_gpu.stages.stack import Stack
    root, proj, truth = night
    p = copy_project(proj.root, tmp_path / "cov")
    z = dict(np.load(p.stage_dir("register") / "reg.npz"))
    z["fwhm_moffat"], z["roundness_moffat"] = np.full(truth.n, 1.6), np.full(truth.n, 0.95)
    np.savez(p.stage_dir("register") / "reg.npz", **z)
    ctx = Context(p, Stack(), Journal(p.logs, echo=False), "cpu")
    legacy = Stack().auto(ctx)
    assert "pixfrac" not in legacy and legacy["scale"].template not in FRENCH
    p.set_override("stack", "scale_rule", "coverage")
    a = Stack().auto(ctx)
    assert a["scale"].template in FRENCH and a["pixfrac"].template in FRENCH
    assert a["scale"].value in (1.0, 1.5, 2.0) and 0.5 <= a["pixfrac"].value <= 1.0
    assert "1.6 px" in a["scale"].reason or a["scale"].value > 1.0


def test_coverage_is_kept_per_colour_with_the_switches(night):
    from astro_gpu.stages.stack import COVERAGE, DIAG
    p, ctx = stack_with(night, "cov_on", scale_rule="coverage")
    with np.load(p.stage_dir("stack") / COVERAGE) as z:
        assert z["measured"].shape == (3, H, W) and z["estimate_map"].shape[0] == 3 and len(z["variation"]) == 3
        var = z["variation"]
    assert var[1] < var[0] and var[1] < var[2]               # green has twice the samples of red and blue
    assert ctx.metrics["coverage_variation_pct"] == [round(100 * v, 1) for v in var]
    diag = json.loads((p.stage_dir("stack") / DIAG).read_text())
    assert diag["switches"]["scale_rule"] == "coverage" and diag["coverage"]["rule"]["scale"]["template"]


def test_hubble_rejection_finds_the_trail_and_keeps_its_maps(night):
    from astro_gpu.stages.stack import DIAG, REJECTION
    root, proj, truth = night
    p, ctx = stack_with(night, "hubble", rejection="hubble")
    with np.load(p.stage_dir("stack") / REJECTION) as z:
        assert {"low", "high", "untested", "factor", "grid"} <= set(z.files)
        high = z["high"]
    row = H // 2 + 10 - truth.dith[truth.trail, 1]
    band = high[int(row) - 2:int(row) + 3].sum(0)
    assert (band[8:-8] > 0).mean() > 0.9                                 # the whole trail
    assert ctx.metrics["rejected_pct"] < 2.0 and "untested_pct" in ctx.metrics
    diag = json.loads((p.stage_dir("stack") / DIAG).read_text())
    assert diag["rejection"]["method"] == "hubble" and diag["rejection"]["summary"]["reference_pct"] == [0.1, 0.5]
    fr = {f["index"]: f for f in diag["frames"]}
    assert fr[truth.trail]["rejected_high"] > 300 and "z_spread" in fr[0]


@pytest.mark.parametrize("weighting", ["noise", "sharp", "none"])
def test_weighting_per_colour(night, weighting):
    from astro_gpu.stages.stack import DIAG
    root, proj, truth = night
    p, ctx = stack_with(night, f"w_{weighting}", weighting=weighting)
    rows = json.loads((p.stage_dir("stack") / "frames.json").read_text())
    w = np.array([r["weights_rgb"] for r in rows])
    if weighting == "none":
        assert np.allclose(w[w > 0], 1.0)
    else:                                                # inverse variance: the noisy frame weighs about 1/4 of a twin
        ratio = w[truth.noisy, 1] / np.median(w[:, 1])
        want = (truth.t[truth.noisy] / np.median(truth.t)) ** 2 * (300 + 36) / (300 + 144)
        assert ratio == pytest.approx(want, rel=0.25)
    diag = json.loads((p.stage_dir("stack") / DIAG).read_text())
    assert diag["selection"]["weighting"] == weighting and "weights_rgb" in diag["frames"][0]
    assert isinstance(diag["half_weights"][0], list)


def test_selection_at_k_sigma_with_reasons(night, tmp_path):
    from astro_gpu.core import frame_weights as fwm
    root, proj, truth = night
    p = copy_project(proj.root, tmp_path / "sel")
    z = dict(np.load(p.stage_dir("register") / "reg.npz"))
    fm = np.full(truth.n, truth.fwhm) * (1 + 0.02 * np.random.default_rng(1).normal(size=truth.n))
    fm[2] = 2.0 * truth.fwhm                                           # wind on frame 2
    z["fwhm_moffat"], z["roundness_moffat"] = fm, np.full(truth.n, 0.95)
    np.savez(p.stage_dir("register") / "reg.npz", **z)
    from astro_gpu.stages.stack import Stack, StackParams
    params = dict(scale=1.0, pixfrac=1.0, gain=1.0, read_noise=6.0, half_block=2, selection="sota", weighting="noise")
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


@pytest.mark.parametrize("mode", ["local", "global"])
def test_normalisation_per_colour(night, mode):
    from astro_gpu.stages.stack import DIAG
    root, proj, truth = night
    p, ctx = stack_with(night, f"norm_{mode}", normalisation=mode)
    diag = json.loads((p.stage_dir("stack") / DIAG).read_text())
    nd = diag["normalisation"]
    assert nd["mode"] == mode and nd["reason"]["template"]
    fr = {f["index"]: f for f in diag["frames"]}
    h = fr[truth.hazy]
    if mode == "local":                                  # the hazy frame's blue is divided by its own loss
        ref = np.median([truth.t_rgb[i] for i in nd["frames"]], axis=0)
        want = truth.t_rgb[truth.hazy] / ref
        assert h["divisor"] == pytest.approx(list(want), rel=0.06)
        assert h["divisor"][2] < 0.9 * h["divisor"][1]
    assert h["scale"] == pytest.approx([1 / s for s in h["divisor"]])
    assert "normalisation_spread_sigma" in ctx.metrics or nd["diagnostic"]["max_sigma"] is None


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
    assert meta["method"] == "classic" and meta["scale"] == 1.0
    # its own settings
    p2, _ = stack_with(small_night, "classic_pct", method="classic", classic_rejection="percentile",
                       classic_interpolation="bilinear")
    diag2 = json.loads((p2.stage_dir("stack") / st.DIAG).read_text())
    assert diag2["rejection"]["classic"] == "percentile" and diag2["switches"]["classic_interpolation"] == "bilinear"
    assert not np.array_equal(np.load(p2.stage_dir("stack") / "drz_full.npy"), full)


def test_switch_values_are_checked():
    from astro_gpu.stages.register import RegisterParams
    from astro_gpu.stages.stack import StackParams
    for model, name in ((StackParams, "weighting"), (StackParams, "normalisation"), (StackParams, "classic_rejection"),
                        (RegisterParams, "fwhm_method")):
        with pytest.raises(ValueError, match=name):
            model(**{name: "wrong"})
    assert not StackParams().v2 and StackParams(selection_transparency="off").v2 is False
    assert StackParams(rejection="hubble").v2


# ---------------------------------------------------------------------------------------------- the API

def _served_by(monkeypatch, proj):
    """api's result folders pointed at a v1 toy project's stage folders."""
    from astro_gpu import api
    monkeypatch.setattr(api, "_need", lambda p, wf, stage, which="current": proj.stage_dir(stage))
    monkeypatch.setattr(api, "_upstream", lambda p, wf, d, up, which: proj.stage_dir(up))
    return api


def test_api_serves_what_the_switches_add(night, monkeypatch):
    """stack_diagnostics reads what the switches add, and gives a stack of the previous version the same fields from
    what it has (its coverage from its weight map, its reasons from frames.json)."""
    root, proj, truth = night
    for name, sw, method in (("api_old", {}, "legacy"),
                             ("api_new", {"rejection": "hubble", "weighting": "noise", "normalisation": "local",
                                          "selection": "sota"}, "hubble")):
        p, ctx = stack_with(night, name, **sw)
        (p.stage_dir("stack") / "manifest.json").write_text(json.dumps({"status": "done", "metrics": ctx.metrics}))
        api = _served_by(monkeypatch, p)
        s = api.stack_diagnostics(None, 1)
        assert s.rejection["method"] == method and s.coverage["maps"].shape == (3, H, W)
        assert set(s.coverage["variation"]) == set("RGB") and len(s.weights_rgb) == truth.n
        assert s.channel_alignment["mode"] == "legacy" and s.used
        assert all(r and r[0]["template"] for r in s.exclusions.values())
        if method == "hubble":
            assert s.normalisation["mode"] == "local" and s.selection["weighting"] == "noise"
            assert "estimate" in s.coverage and s.rejection_untested is not None
            assert s.switches["rejection"] == "hubble" and s.rejection["untested_pct"] is not None
            assert s.weights[0][3] is not None and len(s.weights[0][3]) == 3
        else:
            assert s.switches["rejection"] == "legacy" and s.selection is None and s.rejection_untested is None
            assert s.exclusions == {}


def test_api_registration_table_with_the_switches(registered, monkeypatch):
    p, _, truth = registered
    api = _served_by(monkeypatch, p)
    t = api.registration_table(None, 1)
    assert t.fwhm_method == "moffat" and t.channel_alignment["applied"]
    r = t[truth.n // 2]
    assert r.fwhm_moffat == pytest.approx(truth.fwhm, rel=0.1) and r.colour_block is not None
    assert r.colour_shift[0] == pytest.approx(truth.red[0], abs=0.08)
