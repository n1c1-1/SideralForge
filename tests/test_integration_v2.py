"""End to end: the v2 path (project, store, runner, API) against v1, bit for bit (P7).

* Fast tests, on the CPU, on toy stages that use the real new stages (background_poly, render_simple): the same
  chains are run by v1's Pipeline and, through ``astro_gpu.api`` in a detached worker, by the v2 runner; the stage
  folders must be identical; the sharing, the empty second plan and ``render_layer`` against the rendering engine are
  checked, and the comparison tools are shown to catch a difference.
* The slow test runs ``tools/v2_smoke.py all`` on the heart_a7s test set (about two hours on a GPU the first time,
  minutes once ``F:/astro_essais/p7`` is complete: every phase resumes) and asserts on its report:
  stack and denoise bit for bit against the v1 test project in the full chain; render_prep bit for bit when the v2
  runner recomputes it from v1's own inputs, and the full chain's final image within the measured tolerance of v1's;
  ``render_layer(faithful, 2)`` equal to ``RenderEngine.render(faithful)``; Full and Classic stack sharing their
  common stages; a second plan empty; the new exits giving sane images.

The toy graph is also built by the detached worker, from the spec ``test_integration_v2:toy_rules``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from astro_gpu import api
from astro_gpu.core import project as project_mod
from astro_gpu.core.journal import Journal
from astro_gpu.core.params import P, StageParams
from astro_gpu.core.project import Inputs, Project, list_frames
from astro_gpu.core.runner import Pipeline
from astro_gpu.core.stage import Stage
from astro_gpu.core.store import KeyRules
from astro_gpu.core.workflow import Chains
from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings
from astro_gpu.stages.background_poly import BackgroundPoly
from astro_gpu.stages.render_simple import RenderSimple

TESTS = Path(__file__).parent
REPO = TESTS.parent
SMOKE = REPO / "tools" / "v2_smoke.py"
SPEC = "test_integration_v2:toy_rules"
H, W = 192, 256
M = np.array([[1.6, -0.4, -0.2], [-0.3, 1.5, -0.2], [0.0, -0.5, 1.5]])


def _load_smoke():
    spec = importlib.util.spec_from_file_location("v2_smoke", SMOKE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v2_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


smoke = _load_smoke()


# ---------------------------------------------------------------------------------------------- toy stages

def scene(n: int):
    """A raw stack (3, H, W) in camera colours: a degree-2 sky per channel, a nebula, stars, noise; seeded by ``n``."""
    rng = np.random.default_rng(100 + n)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    u, v = xx / (W / 2) - 1, yy / (H / 2) - 1
    sky = np.stack([lvl * (1 + a * u + b * v + c * u * u + d * u * v + e * v * v) for lvl, a, b, c, d, e in
                    [(900, .08, -.05, .06, .03, -.04), (1000, .06, -.07, .05, -.02, .05), (700, .1, .02, -.05, .04, .03)]])
    neb = np.zeros((H, W))
    for _ in range(4):
        cx, cy, s, a = rng.uniform(.2 * W, .8 * W), rng.uniform(.2 * H, .8 * H), rng.uniform(12, 30), rng.uniform(100, 400)
        neb += a * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / s ** 2)
    st = np.zeros((H, W))
    for _ in range(80):
        st[rng.integers(3, H - 3), rng.integers(3, W - 3)] += rng.uniform(200, 20000)
    st = cv2.GaussianBlur(st, (0, 0), 1.3)
    cam = sky + np.stack([neb + .9 * st, .5 * neb + st, .3 * neb + .8 * st])
    return (cam + rng.normal(0, 30, cam.shape)).astype(np.float32)


class InvParams(StageParams):
    max_lights: int = P(0, label="max lights", help="", ge=0)


class Inv(Stage):
    name = "inventory"
    title = "Inventory"
    Params = InvParams

    def run(self, ctx, p):
        files = list_frames(ctx.project.config.inputs.lights)
        ctx.save_json("frames.json", {"lights": [{"path": str(f), "size": Path(f).stat().st_size} for f in files]})
        ctx.metric("n_lights", len(files))


class ToyStackParams(StageParams):
    scale: float = P(1.5, label="Scale", help="Output size.", ge=1.0, le=3.0)
    kappa: float = P(5.0, label="Kappa", help="", ge=2.0)
    workers: int = P(4, label="workers", help="", ge=1)
    exclude: list[str] = P([], label="exclude", help="", level="internal")


class ToyStack(Stage):
    name = "stack"
    title = "Stack"
    deps = ("inventory",)
    Params = ToyStackParams

    def run(self, ctx, p):
        lights = [f["path"] for f in ctx.load_json("inventory", "frames.json")["lights"]]
        n = len([f for f in lights if f not in p.exclude])
        raw = scene(n)
        wt = np.full((3, H, W), 20, np.float32)
        wt[:, :6] = 0
        wt[:, :, -5:] = 0
        raw[wt == 0] = 0
        ctx.save("drz_full.npy", raw)
        ctx.save("drz_weight.npy", wt)
        ctx.save_json("drz_meta.json", {"scale": p.scale})
        ctx.metric("frames_used", n)


class ToyColorParams(StageParams):
    aperture: float = P(4.1, label="aperture", help="", ge=1.0)


class ToyColor(Stage):
    name = "color"
    title = "Color"
    deps = ("stack",)
    Params = ToyColorParams

    def run(self, ctx, p):
        ctx.save("M.npy", M.astype(np.float32))
        ctx.metric("aperture", p.aperture)


class ToyDenoiseParams(StageParams):
    iterations: int = P(10, label="iterations", help="", ge=1)


class ToyDenoise(Stage):
    name = "denoise"
    title = "Denoise"
    deps = ("stack",)
    Params = ToyDenoiseParams

    def run(self, ctx, p):
        src = ctx.load("stack", "drz_full.npy")
        out = np.stack([cv2.GaussianBlur(c, (0, 0), 0.8 + 0.01 * p.iterations) for c in src]).astype(np.float32)
        ctx.save("n2n_full.npy", out)
        ctx.metric("noise_reduction", float(src.std() / out.std()))


def toy_rule(options, end, reference, learned_layer):
    out = ["inventory", "stack"]
    if end == "stacking":
        return out + ["render_simple"]
    return out + ["color"] + (["denoise"] if options.get("denoise") else []) + ["background_poly", "render_simple"]


def toy_rules() -> KeyRules:
    """Also built by the detached worker (plan.rules = SPEC)."""
    chains = Chains([Inv(), ToyStack(), ToyColor(), ToyDenoise(), BackgroundPoly(), RenderSimple()],
                    optional={"background_poly": ("denoise",), "render_simple": ("color", "denoise", "background_poly")},
                    blocks={"calibration": ("inventory",), "stacking": ("stack",), "color": ("color",),
                            "denoise": ("denoise",), "gradient": ("background_poly",), "output": ("render_simple",)},
                    rule=toy_rule)
    return KeyRules(chains, auto_fields={}, path_fields={}, non_key={"stack": ("workers",)},
                    implied=lambda options, stage: {})


def v1_chain(name: str) -> list[Stage]:
    return {"classic": [Inv(), ToyStack(), ToyColor(), BackgroundPoly(), RenderSimple()],
            "classic_dn": [Inv(), ToyStack(), ToyColor(), ToyDenoise(), BackgroundPoly(), RenderSimple()],
            "raw": [Inv(), ToyStack(), RenderSimple()]}[name]


# ---------------------------------------------------------------------------------------------- fixtures

@pytest.fixture
def state(tmp_path, monkeypatch) -> Path:
    """A private ASTRO_GPU_HOME (timing histories, camera profiles): in this process and in the detached worker."""
    s = tmp_path / "state"
    monkeypatch.setattr(project_mod, "STATE", s)
    monkeypatch.setenv("ASTRO_GPU_HOME", str(s))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")]))
    return s


def raws(folder: Path, n: int = 4) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"IMG_{i:03d}.ARW").write_bytes(b"\0" * (100 + 10 * i))
    return folder


# ---------------------------------------------------------------------------------------------- the tools

def test_array_diff_measures_a_difference():
    a = np.linspace(0, 1, 1000, dtype=np.float32).reshape(10, 100)
    assert smoke.array_diff(a, a.copy())["equal"]
    b = a.copy()
    b[3, 7] = np.nextafter(b[3, 7], np.float32(2))                       # one unit in the last place
    d = smoke.array_diff(a, b)
    assert not d["equal"] and d["differing"] == 1 and d["max_ulp"] == 1 and d["at"] == [3, 7]
    assert d["elements_over_4_ulp"] == 0 and 0 < d["max_abs"] < 1e-6
    c = a.copy()
    c[0, 0] = 5.0
    d = smoke.array_diff(a, c)
    assert d["max_abs"] == pytest.approx(5.0) and d["differing_pct"] == pytest.approx(0.1)
    n1, n2 = a.copy(), a.copy()
    n1[1, 1] = n2[1, 1] = np.nan                                         # the same NaN is not a difference
    assert smoke.array_diff(n1, n2)["equal"]
    n2[1, 1] = 0.5
    assert smoke.array_diff(n1, n2)["nan_mismatch"] == 1
    assert not smoke.array_diff(a, a[:5])["equal"]                       # another shape
    assert not smoke.array_diff(a, a.astype(np.float64))["equal"]        # another type


def test_compare_dirs_files_json_npz_and_the_skipped_parts(tmp_path):
    def make(d: Path, tweak: bool = False):
        d.mkdir()
        (d / "previews").mkdir()
        (d / "previews" / "x.jpg").write_bytes(b"jpg" + (b"1" if tweak else b"0"))
        (d / "manifest.json").write_text(json.dumps({"run_id": "a" if tweak else "b"}))
        np.save(d / "a.npy", np.arange(12, dtype=np.float32).reshape(3, 4))
        np.savez(d / "b.npz", x=np.ones(5), y=np.arange(3) + (1 if tweak else 0))
        (d / "c.json").write_text(json.dumps({"k": [1, 2, 3], "z": {"w": 2 if tweak else 1}}))
        (d / "d.bin").write_bytes(b"abc")
    make(tmp_path / "v1")
    make(tmp_path / "v2")
    cmp = smoke.compare_dirs(tmp_path / "v1", tmp_path / "v2")
    assert sorted(cmp) == ["a.npy", "b.npz", "c.json", "d.bin"]          # manifests and previews are not results
    assert all(c["status"] == "identical" for c in cmp.values())
    make(tmp_path / "v3", tweak=True)
    (tmp_path / "v3" / "extra.npy").write_bytes(b"x")
    cmp = smoke.compare_dirs(tmp_path / "v1", tmp_path / "v3")
    s = smoke.summarize(cmp)
    assert s["different"] == ["b.npz", "c.json", "extra.npy"] and s["identical"] == 2
    assert cmp["extra.npy"]["status"] == "missing_v1"
    assert "y" in cmp["b.npz"]["detail"] and "/z/w" in cmp["c.json"]["detail"][0]
    skip = smoke.compare_dirs(tmp_path / "v1", tmp_path / "v3", skip_content={"b.npz", "c.json"})
    assert skip["b.npz"]["note"] == "presence only"


def test_compare_manifests_ignores_workers_device_and_an_empty_exclude_only():
    a = {"version": "3", "metrics": {"n": 1}, "params": {"scale": 1.5, "workers": 12}}
    assert smoke.compare_manifests(a, {"version": 3, "metrics": {"n": 1}, "params": {"scale": 1.5, "workers": 4,
                                                                                       "exclude": []}}) == []
    assert smoke.compare_manifests(a, {"version": 3, "metrics": {"n": 1}, "params": {"scale": 1.5, "exclude": ["x"]}})
    d = smoke.compare_manifests(a, {"version": "3", "metrics": {"n": 2}, "params": {"scale": 2.0}})
    assert len(d) == 2 and any("metrics/n" in x for x in d) and any("params/scale" in x for x in d)


def test_sanity_and_the_image_helpers(tmp_path):
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal(0.25, 0.1, (3, 120, 160)), 0, 1).astype(np.float32)
    s = smoke.sanity(img)
    assert s["finite"] and s["in_range"] and s["lum_std"] > 0.03 and 0.1 < s["lum_p50"] < 0.4
    bad = img.copy()
    bad[0, 0, 0] = np.nan
    assert not smoke.sanity(bad)["finite"]
    assert smoke.rank_corr(img, img) == pytest.approx(1.0)
    assert smoke.rank_corr(img, rng.random(img.shape).astype(np.float32)) < 0.5
    smoke.save_png(tmp_path / "x.png", img)
    smoke.sheet(tmp_path / "sheet.png", {"a": img, "b": img})
    assert cv2.imread(str(tmp_path / "x.png")).shape == (120, 160, 3) and (tmp_path / "sheet.png").is_file()


def test_star_peaks_tell_the_same_field_from_a_shifted_one():
    rng = np.random.default_rng(3)
    img = np.zeros((3, 240, 320), np.float32)
    ys, xs = rng.integers(20, 220, 150), rng.integers(20, 300, 150)
    img[:, ys, xs] = rng.uniform(0.2, 1.0, (3, 150))
    img = np.stack([cv2.GaussianBlur(c, (0, 0), 1.5) for c in img]) * 20 + 0.02 * rng.random(img.shape, dtype=np.float32)
    assert smoke.peak_match(img, img) == pytest.approx(1.0)
    assert smoke.peak_match(img, img * 0.5 + 0.1) >= 0.9                          # another stretch, the same stars
    assert smoke.peak_match(np.roll(img, (3, 5), axis=(1, 2)), img) <= 0.15        # a few pixels off (chance coincidences only)


def test_tensor_comparison_reads_arrays_and_tensors_in_a_saved_dict(tmp_path):
    """psf_model.pt holds numpy arrays next to tensors: the first real comparison crashed on ``a == b`` of arrays."""
    import torch
    a = {"E": torch.arange(6, dtype=torch.float32), "sel": np.arange(5), "wing_prof": np.ones((2, 3)),
         "frac": [0.25, 0.5], "width": 640}
    b = {"E": a["E"].clone(), "sel": np.arange(5), "wing_prof": np.ones((2, 3)), "frac": [0.25, 0.5], "width": 640}
    torch.save(a, tmp_path / "a.pt")
    torch.save(b, tmp_path / "b.pt")
    assert smoke.compare_file(tmp_path / "a.pt", tmp_path / "b.pt")["status"] == "identical"
    b["E"][2] += 1e-6
    b["wing_prof"] = b["wing_prof"] + 1e-9
    b["sel"] = np.arange(5) + 1
    torch.save(b, tmp_path / "c.pt")
    c = smoke.compare_file(tmp_path / "a.pt", tmp_path / "c.pt")
    assert c["status"] == "different" and len(c["detail"]) == 3
    assert any(x.startswith("/E: 1/6") for x in c["detail"]) and any(x.startswith("/sel") for x in c["detail"])


def test_the_image_tolerance_is_looser_than_what_was_measured_and_tight_enough_to_catch_a_broken_render():
    measured = {"psnr_db": 34.4, "p50": 3.2, "p99": 12.5}                 # heart_a7s, faithful at 1/4, F:/astro_essais/p7
    t = smoke.IMAGE_TOLERANCE
    assert measured["psnr_db"] > t["psnr_db_min"] and measured["p50"] < t["p50_max"] and measured["p99"] < t["p99_max"]
    rng = np.random.default_rng(0)
    a = np.clip(rng.normal(0.25, 0.1, (3, 100, 100)), 0, 1)
    wrong = np.clip(a + 0.05, 0, 1)                                         # a sky 13 levels off everywhere
    assert 10 * np.log10(1 / np.mean((a - wrong) ** 2)) < t["psnr_db_min"]


def test_snapshot_shows_a_write(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f.txt").write_text("1")
    before = smoke.snapshot(tmp_path)
    assert smoke.snapshot(tmp_path) == before
    (tmp_path / "a" / "f.txt").write_text("22")
    assert smoke.snapshot(tmp_path) != before


# ---------------------------------------------------------------------------------------------- the flow, on toy stages

def test_toy_v2_run_equals_v1_bit_for_bit(tmp_path, state):
    lights = raws(tmp_path / "raw" / "lights")
    # v1: Pipeline.run on the three chains, in projects of their own
    v1 = {}
    for name in ("classic", "classic_dn", "raw"):
        root = tmp_path / "v1" / name
        Project.create(root, name, Inputs(lights=[str(lights)]), scratch_dir=tmp_path / "v1scratch" / name)
        assert Pipeline(Project(root), v1_chain(name), Journal(root / "logs", echo=False)).run() == "done"
        v1[name] = root / "work"
    # v2: the same chains as three workflows of one project, planned and run through the API
    p = api.create_project(tmp_path / "v2", "toy")
    api.use_rules(p, SPEC)
    api.add_session(p, lights=[lights])
    classic = api.add_workflow(p, "Classic stack", "Classic stack").id
    dn = api.add_workflow(p, "Classic + denoise", "Classic + denoise").id
    raw = api.add_exit(p, classic, "stacking", "Raw stack").id
    p.devices["main"] = "cpu"
    api.save(p)
    plan = api.plan_all(p, "cpu")

    # the sharing: one job per distinct key, the common stages shared by every workflow that has them
    distinct = {(s, api.stage_key(p, w, s)) for w in (classic, dn, raw) for s in api.chain(p, w)}
    assert {(j.stage, j.key) for j in plan.jobs} == distinct and len(plan.jobs) == len(distinct) == 9
    shared = {j.stage: j.wf for j in plan.jobs if len(j.wf) > 1}
    assert shared == {"inventory": [classic, dn, raw], "stack": [classic, dn, raw], "color": [classic, dn]}
    assert api.sharing(p, dn, "stack") == [classic, dn, raw] and api.sharing(p, classic, "render_simple") == [classic]
    assert api.stage_key(p, dn, "stack") == api.stage_key(p, raw, "stack")

    info = api.start_run(p, plan, "cpu")
    res = smoke.follow(api, p, info, lambda m: None, poll=0.2, grace=120, max_s=300)
    assert res["status"] == "done", [e for e in res["events"] if e.get("level") == "error"]
    tm = smoke.timings_of(res["events"])
    assert set(tm) == {j.key for j in plan.jobs} and all(t["status"] == "done" and not t["reused"] for t in tm.values())
    starts = [e["key"] for e in res["events"] if e["type"] == "stage_start"]
    assert sorted(starts) == sorted(j.key for j in plan.jobs)               # every key computed once
    assert all(e["run"] == info.run_id for e in res["events"])
    api.refresh(p)
    assert len(api.plan_all(p, "cpu")) == 0 and api.cost(p).jobs == 0        # a second plan is empty
    assert {s.state for s in api.states(p).values()} == {"done"}

    # a change plans only its downstream
    api.set_param(p, [classic], "background_poly", "degree", 3)
    again = api.plan_all(p, "cpu")
    assert [(j.stage, j.wf) for j in again.jobs] == [("background_poly", [classic]), ("render_simple", [classic])]
    api.set_param(p, [classic], "background_poly", "degree", 2)              # reverting restores the old keys
    assert len(api.plan_all(p, "cpu")) == 0

    # v2 equals v1, file by file, for every stage both have
    for name, wf in (("classic", classic), ("classic_dn", dn), ("raw", raw)):
        for stage in api.chain(p, wf):
            out = api.stage_outputs(p, wf, stage)
            cmp = smoke.compare_dirs(v1[name] / stage, Path(out.dir))
            m1 = json.loads((v1[name] / stage / "manifest.json").read_text())
            assert cmp and all(c["status"] == "identical" for c in cmp.values()), (name, stage, cmp)
            assert smoke.compare_manifests(m1, out.manifest) == [], (name, stage)

    # render_layer at slider position 2 equals the rendering engine, on v2's and on v1's folder
    twos = {k: 2.0 for k in ("sky", "contrast", "saturation", "noise")}
    for name, wf in (("classic", classic), ("classic_dn", dn), ("raw", raw)):
        d2 = Path(api.stage_outputs(p, wf, "render_simple").dir)
        lay = api.render_layer(p, wf, None, twos, 0.5, device="cpu")
        for d in (d2, v1[name] / "render_simple"):
            ref = SimpleRenderEngine(d, device="cpu").render(SimpleSettings(), 0.5).final.float().cpu().numpy()
            assert smoke.array_diff(ref, lay.rgb)["equal"], (name, d)
        s = smoke.sanity(lay.rgb)
        assert s["finite"] and s["in_range"] and s["lum_std"] > 0.02
    # a slider moved away from 2 does change the layer
    moved = api.render_layer(p, classic, None, {"sky": 4.0}, 0.5, device="cpu").rgb
    assert moved.mean() > api.render_layer(p, classic, None, None, 0.5, device="cpu").rgb.mean()
    # the Raw stack exit has no colour cast: its sky is grey in every corner (the median pixel of this scene is nebula).
    # Before the sky was made grey the corners were 39 to 100 % off. The sky samples of this 192 x 256 toy are 4 px
    # blocks: taking the darkest of them biases the green sky a few ADU low (up to 8 here, against a noise of 15), hence
    # 15 %; real frames use blocks of 30-70 px (heart_a7s: corners within 1.4 levels of 255)
    lay = api.render_layer(p, raw, None, None, 0.5, device="cpu")
    x0, x1, y0, y1 = (v // 2 for v in lay.crop)
    img = lay.rgb[:, y0:y1, x0:x1]
    h, w = img.shape[1:]
    for c in (img[:, :h // 5, :w // 5], img[:, :h // 5, -w // 5:], img[:, -h // 5:, :w // 5], img[:, -h // 5:, -w // 5:]):
        sk = np.median(c.reshape(3, -1), axis=1)
        assert sk.max() / max(sk.min(), 1e-9) < 1.15, sk

    # the comparison catches a v2 that differs: one value of the stack changed in a copy
    out = Path(api.stage_outputs(p, classic, "stack").dir)
    bad = tmp_path / "tampered"
    shutil.copytree(out, bad)
    a = np.load(bad / "drz_full.npy")
    a[1, 50, 60] = np.nextafter(a[1, 50, 60], np.float32(1e9))
    np.save(bad / "drz_full.npy", a)
    cmp = smoke.compare_dirs(v1["classic"] / "stack", bad)
    s = smoke.summarize(cmp)
    assert s["different"] == ["drz_full.npy"] and cmp["drz_full.npy"]["detail"]["max_ulp"] == 1
    assert cmp["drz_full.npy"]["detail"]["at"] == [1, 50, 60]


def test_the_toy_run_changes_nothing_in_the_v1_projects(tmp_path, state):
    """A v1 project that v2 reads (the reference) is listed before and after the whole flow: no byte changes."""
    lights = raws(tmp_path / "raw" / "lights")
    root = tmp_path / "v1ref"
    Project.create(root, "ref", Inputs(lights=[str(lights)]), scratch_dir=tmp_path / "v1scratch")
    assert Pipeline(Project(root), v1_chain("classic"), Journal(root / "logs", echo=False)).run() == "done"
    before = smoke.snapshot(root)
    for stage in ("stack", "render_simple"):
        smoke.compare_dirs(root / "work" / stage, root / "work" / stage)
    assert smoke.snapshot(root) == before


# ---------------------------------------------------------------------------------------------- heart_a7s, the real thing

def _check(rep: dict, start: str) -> dict:
    """The one recorded check whose name starts with ``start``."""
    hits = [c for n, c in rep["checks"].items() if n.startswith(start)]
    assert len(hits) == 1, (start, [c["name"] for c in hits])
    return hits[0]


@pytest.mark.slow
@pytest.mark.gpu
def test_heart_a7s_v2_equals_v1():
    """Full and Classic stack through the API on the heart_a7s test set, on the GPU, against the v1 test project
    (``F:/astro_essais/projets/heart_a7s``, read only). Everything is in ``F:/astro_essais/p7``, resumed when it
    exists (the first time takes about two hours)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    for need in (smoke.DATA / "lights", smoke.V1_PROJECT / "work" / "render_prep" / "prep.json"):
        if not need.exists():
            pytest.skip(f"missing {need}")
    root = Path(os.environ.get("ASTRO_P7_ROOT") or smoke.DEFAULT_ROOT)
    r = subprocess.run([sys.executable, str(SMOKE), "--root", str(root), "all"], cwd=str(REPO), capture_output=True,
                       text=True, timeout=6 * 3600)
    tail = (r.stdout or "")[-3000:] + (r.stderr or "")[-2000:]
    rep = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert not rep.get("failed_required"), f"{rep.get('failed_required')}\n{tail}"
    assert r.returncode == 0, tail

    # stack and denoise: bit for bit in the full chain (and so is everything they read)
    for stage in ("inventory", "calibrate", "register", *smoke.BIT_FOR_BIT):
        c = rep["compare"][stage]
        assert not c["different"] and not c["manifest_differences"], (stage, c)
    # render_prep: bit for bit from v1's own inputs; the key of every adopted v1 result is the key v2 computed
    assert _check(rep, "import: the keys of v1's adopted results")["ok"]
    for stage in smoke.ON_V1_INPUTS:
        c = rep["reproduce"][stage]
        assert not c["different"] and not c["manifest_differences"] and c["files"] > 0, (stage, c)
    # in the full chain the stages after the astrometry fit differ (the WCS has other last digits; optics' lateral
    # grid and the stars photometry follow, stages that use sep.extract, which does not return the same objects twice):
    # the final picture stays within the measured tolerance
    t = smoke.IMAGE_TOLERANCE
    for base in ("faithful", "artistic"):
        im = rep["render"][f"{base}_0.25"]["against_v1_render_prep"]["image"]
        lv = im["abs_diff_levels"]
        assert im["psnr_db"] >= t["psnr_db_min"] and lv["p50"] <= t["p50_max"] and lv["p99"] <= t["p99_max"], (base, im)
    # render_layer(faithful, 2) equals RenderEngine.render(faithful), at every size, and the artistic base too
    for k in ("faithful_0.25", "faithful_0.5", "faithful_1.0", "artistic_0.25"):
        assert rep["render"][k]["equal"], (k, rep["render"][k])
    # Full and Classic stack: one job and one key for each common stage; the first plan; a second plan empty
    assert _check(rep, "Full and Classic stack share inventory to color")["ok"]
    assert _check(rep, "the first plan: Full's 14 stages + 2 of Classic")["ok"]
    assert _check(rep, "first: a second plan is empty")["ok"]
    assert rep["run_first"]["status"] == "done" and len(rep["run_first"]["jobs"]) == 16
    assert _check(rep, "extras: only the new jobs are planned")["ok"]
    # the new exits give sane images
    for name in ("full", "classic_stack", "classic_denoise", "raw_stack"):
        assert _check(rep, f"image {name}:")["ok"], name
    assert _check(rep, "Raw stack exit: no colour cast")["ok"]
    assert _check(rep, "Classic stack and Full show the same field on the same pixels")["ok"]
    assert _check(rep, "polynomial gradient")["ok"]
