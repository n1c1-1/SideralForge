"""Stacking v2 (docs/spec/STACKING.md; the decisions of 2 November, docs/pilot/PILOT.md §3c): every setting at its
previous version's value is the previous version bit for bit with the same store keys; the new defaults (Moffat FWHM,
coverage rule, minimum-weight selection) and the method advice on a toy night (CPU); the removed changes are gone.

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


@pytest.mark.parametrize("case", list(STACK_CASES), ids=["previous values", "drizzle 1.5, pattern, by hand"])
def test_every_setting_at_its_previous_value_is_the_previous_stack_bit_for_bit(small_night, case):
    """The stack of BASE and today's with every stacking-v2 setting at its previous version's value: the same files,
    bit for bit (the manifest aside)."""
    from astro_gpu.stages.stack import PREVIOUS, Stack, StackParams
    old = base_module("astro_gpu/stages/stack.py", "_stack_base")
    root, proj, _ = small_night
    params = {"gain": 1.0, "read_noise": 6.0, "half_block": 2, "min_transparency": 0.6, **STACK_CASES[case]}
    tag = case[:8].replace(" ", "_")
    pa = copy_project(proj.root, root / f"old_{tag}")
    pb = copy_project(proj.root, root / f"new_{tag}")
    ca = run_stage(pa, old.Stack(), old.StackParams(**params))
    cb = run_stage(pb, Stack(), StackParams(**params, **PREVIOUS))
    assert same_outputs(pa.stage_dir("stack"), pb.stage_dir("stack")) == []
    assert ca.metrics == cb.metrics
    assert sorted(p.name for p in pb.stage_dir("stack").iterdir()) == sorted(p.name for p in
                                                                             pa.stage_dir("stack").iterdir())


def test_the_previous_registration_is_bit_for_bit(small_night):
    """The registration of BASE and today's with ``fwhm_method = legacy`` on the same star catalogues: the same files,
    bit for bit. (sep's extraction of a frame depends slightly on what the same process extracted before, in BASE as
    today: the catalogues come from one run, through the checkpoint both read; the worker that makes them is BASE's,
    unchanged.)"""
    import inspect
    import shutil

    from astro_gpu.stages import register as new
    old = in_process(base_module("astro_gpu/stages/register.py", "_register_base"))
    in_process(new)
    assert inspect.getsource(new._catalog) == _function_source(git_text("astro_gpu/stages/register.py"), "_catalog")
    root, proj, _ = small_night
    p0 = copy_project(proj.root, root / "reg_ck", ("inventory", "calibrate"))
    run_stage(p0, new.Register(), new.RegisterParams(workers=1, fwhm_method="legacy"))
    pa = copy_project(proj.root, root / "reg_old", ("inventory", "calibrate"))
    pb = copy_project(proj.root, root / "reg_new", ("inventory", "calibrate"))
    for q in (pa, pb):
        shutil.copytree(p0.stage_dir("register") / "_ckpt", q.stage_dir("register") / "_ckpt")
    ca = run_stage(pa, old.Register(), old.RegisterParams(workers=1))
    cb = run_stage(pb, new.Register(), new.RegisterParams(workers=1, fwhm_method="legacy"))
    assert same_outputs(pa.stage_dir("register"), pb.stage_dir("register")) == []
    assert same_outputs(p0.stage_dir("register"), pb.stage_dir("register")) == []
    assert ca.metrics == cb.metrics and not (pb.stage_dir("register") / "chan.npz").exists()


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


def store_previous(w):
    """The previous version's values of the settings whose default changed, stored in a workflow (where it stores
    none of its own)."""
    from astro_gpu.stages.v2 import KEY_BASELINE
    for st, fields in KEY_BASELINE.items():
        d = w.params.setdefault(st, {})
        for k, v in fields.items():
            d.setdefault(k, v)


def test_keys_with_the_previous_values_stored_are_the_previous_keys(tmp_path):
    """The previous values of the changed defaults stored, every key is BASE's, also with every other new setting
    stored at its default and the removed settings still stored; left unset (the new defaults), the registration and
    the stack get new keys, and what follows them, not what comes before."""
    from astro_gpu.core.store import key_rules
    from astro_gpu.stages.register import RegisterParams
    from astro_gpu.stages.stack import StackParams
    from astro_gpu.stages.v2 import KEY_BASELINE, RETIRED
    store, rules0 = base_rules()
    p = _project2(tmp_path)
    before = {w.id: store.Keys(p, rules=rules0).keys(w.id) for w in p.workflows}
    new = {"stack": set(StackParams.model_fields) - set(rules0.params("stack").model_fields),
           "register": set(RegisterParams.model_fields) - set(rules0.params("register").model_fields)}
    assert new["stack"] == {"method", "scale_rule", "selection", "selection_transparency", "classic_debayer",
                            "classic_interpolation", "classic_clamping", "classic_rejection", "classic_cfa_balance",
                            "drizzle_rejection"}
    assert new["register"] == {"fwhm_method"}
    # the new defaults: the registration (Moffat), the stack and what follows change; calibration does not
    now = _keys_of(p, key_rules())
    for w in p.workflows:
        assert now[w.id]["calibrate"] == before[w.id]["calibrate"]
        assert now[w.id]["register"] != before[w.id]["register"] and now[w.id]["stack"] != before[w.id]["stack"]
    for w in p.workflows:
        store_previous(w)
    assert _keys_of(p, key_rules()) == before
    for w in p.workflows:                                   # every other new setting stored at its default
        for st, names in new.items():
            w.params.setdefault(st, {}).update({k: key_rules().defaults(st)[k] for k in names
                                                if k not in KEY_BASELINE.get(st, {})})
    assert _keys_of(p, key_rules()) == before
    for w in p.workflows:                                   # the removed settings, as the bench's projects store them
        w.params["stack"].update(rejection="hubble", weighting="noise", normalisation="local")
        w.params["register"]["channel_align"] = "per_block"
    assert RETIRED["stack"] == {"rejection", "weighting", "normalisation"} and RETIRED["register"] == {"channel_align"}
    assert _keys_of(p, key_rules()) == before
    w = p.workflows[1]
    w.params["stack"]["selection"] = "min_weight"
    after = _keys_of(p, key_rules())[w.id]
    assert after["register"] == before[w.id]["register"] and after["calibrate"] == before[w.id]["calibrate"]
    assert after["stack"] != before[w.id]["stack"] and after["color"] != before[w.id]["color"]
    # the new default stored or left unset: the same key
    del w.params["stack"]["selection"]
    assert _keys_of(p, key_rules())[w.id]["stack"] == after["stack"]


def test_the_colour_correction_option_keeps_the_keys_of_its_default(tmp_path):
    """The colour stage's ``correction`` (2 November): at its default ("full", the behaviour before it), stored or
    not, every key stays what it was without the setting; "per_colour" changes the colour key and what follows it,
    not what comes before."""
    from astro_gpu.core.store import key_rules
    from astro_gpu.stages.color import ColorParams
    assert ColorParams().correction == "full"
    p = _project2(tmp_path)
    before = _keys_of(p, key_rules())
    for w in p.workflows:
        w.params.setdefault("color", {})["correction"] = "full"
    assert _keys_of(p, key_rules()) == before
    w = p.workflows[0]
    w.params["color"]["correction"] = "per_colour"
    after = _keys_of(p, key_rules())[w.id]
    assert after["stack"] == before[w.id]["stack"] and after["color"] != before[w.id]["color"]
    with pytest.raises(ValueError):
        ColorParams(correction="matrix")


def test_the_classic_part_has_a_revision_of_its_own_in_the_keys(tmp_path):
    """The Classic part's revision (stack.CLASSIC_VERSION: the white balance before RCD) is in the key of the stacks
    that use it, the Classic method and the drizzle with the integration's rejection, and of no other: the stacks
    made before it are made again, the drizzle stacks keep their keys."""
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


def test_pixfrac_is_automatic_only_under_the_coverage_rule(tmp_path):
    """stack.pixfrac stored at its default: dropped from the key under the previous scale rule (as before), kept under
    the coverage rule (now the default), whose automatic drop size it overrides (AUTO_WHEN)."""
    from astro_gpu.core.store import Keys, key_rules
    r = key_rules()
    assert "pixfrac" not in r.auto_fields("stack", {"scale_rule": "legacy"})
    assert "pixfrac" in r.auto_fields("stack", {"scale_rule": "coverage"}) and "pixfrac" in r.auto_fields("stack")
    p = _project2(tmp_path)
    w = p.workflows[1]
    w.params["stack"]["scale_rule"] = "legacy"
    assert "pixfrac" not in Keys(p).key_parts(w.id, "stack")["p"]
    del w.params["stack"]["scale_rule"]                     # the default: the coverage rule
    kp = Keys(p).key_parts(w.id, "stack")["p"]
    assert kp["pixfrac"] == 0.7 and kp["scale_rule"] == "coverage"
    with pytest.raises(ValueError, match="scale_rule"):
        w.params["stack"]["scale_rule"] = "wrong"
        Keys(p).key(w.id, "stack")


REAL = [Path("F:/astro_projets/Horsehead"), Path("F:/astro_essais/bench/heart/ours_express")]


@pytest.mark.parametrize("root", REAL, ids=[r.name for r in REAL])
def test_real_projects_keep_their_keys_with_the_previous_values(root):
    """The keys of real projects, read-only: with the previous values of the changed defaults stored (in memory),
    today's equal BASE's, for every workflow and stage."""
    if not (root / "project.astroproj").is_file():
        pytest.skip(f"{root} not available")
    from astro_gpu.core import project2 as p2
    from astro_gpu.core.store import key_rules
    before = {f: f.stat().st_mtime_ns for f in root.glob("*") if f.is_file()}
    p = p2.load(root)
    store, rules0 = base_rules()
    old = {w.id: store.Keys(p, rules=rules0).keys(w.id) for w in p.workflows}
    for w in p.workflows:
        store_previous(w)
    assert old and _keys_of(p, key_rules()) == old
    assert {f: f.stat().st_mtime_ns for f in root.glob("*") if f.is_file()} == before


# ---------------------------------------------------------------------------------------------- register

@pytest.fixture(scope="module")
def registered(night):
    """Register run on the toy night with its defaults (the Moffat fit; in this process)."""
    from astro_gpu.stages import register as reg
    in_process(reg)
    root, proj, truth = night
    p = copy_project(proj.root, root / "reg_v2", ("inventory", "calibrate"))
    ctx = run_stage(p, reg.Register(), reg.RegisterParams(workers=1))
    return p, ctx, truth


def test_moffat_fwhm_is_measured_by_default_beside_the_old_one(registered, night):
    from astro_gpu.stages import register as reg
    p, ctx, truth = registered
    assert reg.RegisterParams().fwhm_method == "moffat" and "channel_align" not in reg.RegisterParams.model_fields
    z = np.load(p.stage_dir("register") / "reg.npz")
    assert {"fwhm_moffat", "roundness_moffat", "beta_moffat", "stars_moffat"} <= set(z.files)
    assert np.nanmedian(z["fwhm_moffat"]) == pytest.approx(truth.fwhm, rel=0.08)
    assert np.nanmedian(z["fwhm"]) > 1.3 * truth.fwhm                      # the 2 x 2 measure, kept as it was
    assert np.nanmedian(z["roundness_moffat"]) > 0.85
    assert ctx.metrics["fwhm_moffat_px"] == pytest.approx(truth.fwhm, rel=0.08)
    assert not (p.stage_dir("register") / "chan.npz").exists() and "channel_alignment" not in ctx.metrics
    # the old arrays are still there, the same measures (to sep's run-to-run differences: see the bit-for-bit test)
    root = night[0]
    q = copy_project(night[1].root, root / "reg_v2_off", ("inventory", "calibrate"))
    in_process(reg)
    run_stage(q, reg.Register(), reg.RegisterParams(workers=1, fwhm_method="legacy"))
    old = np.load(q.stage_dir("register") / "reg.npz")
    assert set(old.files) < set(z.files) and int(old["ref"]) == int(z["ref"])
    for k in ("fwhm", "transp", "coef", "noise"):
        assert np.allclose(old[k], z[k], rtol=2e-3, atol=2e-3), k


# ---------------------------------------------------------------------------------------------- the stack

def test_the_removed_changes_are_gone():
    from astro_gpu.stages import stack as st
    for name in ("rejection", "weighting", "normalisation"):
        assert name not in st.StackParams.model_fields and name not in st.SWITCHES
        with pytest.raises(ValueError):
            st.StackParams(**{name: "legacy"})
    for mod in ("rejection_hubble", "channel_align"):
        with pytest.raises(ImportError):
            __import__(f"astro_gpu.core.{mod}")
    assert not hasattr(st, "v2_frames") and not hasattr(st, "FramesV2")


def test_the_new_defaults():
    from astro_gpu.stages.stack import PREVIOUS, StackParams
    p = StackParams()
    assert (p.method, p.scale_rule, p.selection) == ("drizzle", "coverage", "min_weight") and p.v2
    assert not StackParams(**PREVIOUS).v2 and StackParams(**{**PREVIOUS, "selection_transparency": "off"}).v2 is False


def test_the_colour_alignment_for_the_whole_night_stays_on(night):
    """No per-block mode any more; the whole night's alignment, measured on the first mean, is applied by default."""
    from astro_gpu.stages.stack import DIAG
    p, ctx = stack_with(night, "chan_night")
    info = ctx.metrics["channel_offsets"]
    assert info["applied"] and info["centre_px"]["R"][0] == pytest.approx(night[2].red[0], abs=0.1)
    diag = json.loads((p.stage_dir("stack") / DIAG).read_text())
    assert diag["channel_alignment"]["mode"] == "legacy" and diag["channel_alignment"]["applied"]


def test_the_default_selection_keeps_every_frame_above_the_minimum_weight(night, tmp_path):
    """Siril and PixInsight keep almost every frame: by default only a frame weighing less than 5 % of the best one
    (WBPP) is left out, with its reason; the hazy and the noisy frames are kept with their lower weight."""
    from astro_gpu.core import frame_weights as fwm
    from astro_gpu.stages.stack import DIAG, Stack, StackParams, legacy_weights
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
    w, _ = legacy_weights(z["transp"], z["fwhm"], z["rms"], stats, z["noise"])
    kept = np.arange(truth.n) != 9
    np.testing.assert_allclose([r["weight"] for r in rows if r["weight"] > 0], w[kept])     # the weights unchanged
    assert ctx.metrics["frames_used"] == truth.n - 1
    diag = json.loads((p.stage_dir("stack") / DIAG).read_text())
    assert diag["selection"]["selection"] == "min_weight" and diag["selection"]["excluded"]["9"][0]["code"] == "weight"


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


def test_coverage_rule_and_method_advice_on_the_frames_kept(night, tmp_path):
    """The scale, the drop and the method advice judge the frames the selection keeps (not every aligned frame): the
    advice counts them; it is an advice (never applied); the previous scale rule keeps its own reasons."""
    from astro_gpu.core import coverage as cv
    from astro_gpu.core.params import resolve
    from astro_gpu.stages.stack import Stack, StackParams
    root, proj, truth = night
    p = copy_project(proj.root, tmp_path / "cov")
    z = dict(np.load(p.stage_dir("register") / "reg.npz"))
    z["fwhm_moffat"], z["roundness_moffat"] = np.full(truth.n, 1.6), np.full(truth.n, 0.95)
    np.savez(p.stage_dir("register") / "reg.npz", **z)
    ctx = Context(p, Stack(), Journal(p.logs, echo=False), "cpu")
    a = Stack().auto(ctx)                                            # the defaults: the coverage rule
    assert a["scale"].template in cv.FRENCH and a["pixfrac"].template in cv.FRENCH
    assert a["scale"].value in (1.0, 1.5, 2.0) and 0.5 <= a["pixfrac"].value <= 1.0
    m = a["method"]
    assert m.advice and m.value == "classic" and m.template == cv.M_FEW and m.args == {"n": truth.n}
    params, origin = resolve(StackParams, a, {})
    assert params.method == "drizzle" and origin["method"] == "default"      # never applied
    p.set_override("stack", "exclude", ["L00.fits", "L01.fits"])          # two frames out by hand: 10 judged
    assert Stack().auto(ctx)["method"].args == {"n": truth.n - 2}
    p.set_override("stack", "scale_rule", "legacy")
    legacy = Stack().auto(ctx)
    assert "pixfrac" not in legacy and legacy["scale"].template not in cv.FRENCH
    assert legacy["method"].args == {"n": truth.n - 2}                   # the advice is given under both rules


def test_coverage_is_kept_per_colour_with_the_switches(night):
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
    assert diag["switches"]["scale_rule"] == "coverage" and rule["scale"]["template"] and rule["applied"]
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
    """drizzle_rejection = integration (PixInsight's CFA drizzle): the Classic method's rejection on the debayered,
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


def test_switch_values_are_checked():
    from astro_gpu.stages.register import RegisterParams
    from astro_gpu.stages.stack import StackParams
    for model, name in ((StackParams, "selection"), (StackParams, "scale_rule"), (StackParams, "classic_rejection"),
                        (StackParams, "drizzle_rejection"), (StackParams, "classic_cfa_balance"),
                        (RegisterParams, "fwhm_method")):
        with pytest.raises(ValueError, match=name):
            model(**{name: "wrong"})


# ---------------------------------------------------------------------------------------------- the API

def _served_by(monkeypatch, proj):
    """api's result folders pointed at a v1 toy project's stage folders."""
    from astro_gpu import api
    monkeypatch.setattr(api, "_need", lambda p, wf, stage, which="current": proj.stage_dir(stage))
    monkeypatch.setattr(api, "_upstream", lambda p, wf, d, up, which: proj.stage_dir(up))
    return api


def test_api_serves_what_the_settings_add(night, monkeypatch):
    """stack_diagnostics reads what the v2 settings add, and gives a stack of the previous version the same fields
    from what it has (its coverage from its weight map, its reasons from frames.json)."""
    from astro_gpu.stages.stack import PREVIOUS
    root, proj, truth = night
    for name, sw in (("api_old", dict(PREVIOUS)), ("api_new", {"selection": "sota"})):
        p, ctx = stack_with(night, name, **sw)
        (p.stage_dir("stack") / "manifest.json").write_text(json.dumps({"status": "done", "metrics": ctx.metrics}))
        api = _served_by(monkeypatch, p)
        s = api.stack_diagnostics(None, 1)
        assert s.rejection["method"] == "legacy" and s.coverage["maps"].shape == (3, H, W)
        assert set(s.coverage["variation"]) == set("RGB") and len(s.weights_rgb) == truth.n
        assert s.channel_alignment["mode"] == "legacy" and s.used
        assert all(r and r[0]["template"] for r in s.exclusions.values())
        if name == "api_new":
            assert s.selection["selection"] == "sota" and "estimate" in s.coverage
            assert s.switches["selection"] == "sota" and s.switches["scale_rule"] == "coverage"
        else:
            assert s.switches == PREVIOUS and s.selection is None and s.rejection_untested is None


def test_api_registration_table_with_the_moffat_fit(registered, monkeypatch):
    p, _, truth = registered
    api = _served_by(monkeypatch, p)
    t = api.registration_table(None, 1)
    assert t.fwhm_method == "moffat" and t.channel_alignment is None
    r = t[truth.n // 2]
    assert r.fwhm_moffat == pytest.approx(truth.fwhm, rel=0.1) and r.colour_block is None
