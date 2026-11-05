"""The new stages of v2 (gradient, render_simple), the simple rendering and the v2 stage registry, on synthetic
stacks (CPU)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from astro_gpu.core.journal import Journal
from astro_gpu.core.project import Inputs, Project
from astro_gpu.core.stage import Context
from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings, image_stats
from astro_gpu.stages.gradient import Gradient, GradientParams, sky_variation
from astro_gpu.stages.render_simple import RenderSimple, RenderSimpleParams

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
M = np.array([[1.6, -0.4, -0.2], [-0.3, 1.5, -0.2], [0.0, -0.5, 1.5]])
H, W = 480, 640
HEART_A7S = Path("F:/astro_essais/projets/heart_a7s")


def scene(h: int, w: int, seed: int = 1):
    """(sky, raw stack, denoised stack) in camera RGB: a degree-2 sky per channel, red nebulae, stars, noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = xx / (w / 2) - 1, yy / (h / 2) - 1
    sky = np.stack([lvl * (1 + a * u + b * v + c * u * u + d * u * v + e * v * v) for lvl, a, b, c, d, e in
                    [(900, .08, -.05, .06, .03, -.04), (1000, .06, -.07, .05, -.02, .05), (700, .1, .02, -.05, .04, .03)]])
    neb = np.zeros((h, w))
    for _ in range(6):
        cx, cy, s, a = rng.uniform(.2 * w, .8 * w), rng.uniform(.2 * h, .8 * h), rng.uniform(20, 60), rng.uniform(100, 400)
        neb += a * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / s ** 2)
    st = np.zeros((h, w))
    for _ in range(300):
        st[rng.integers(3, h - 3), rng.integers(3, w - 3)] += rng.uniform(200, 20000)
    st = cv2.GaussianBlur(st, (0, 0), 1.3)
    cam = sky + np.stack([neb + .9 * st, .5 * neb + st, .3 * neb + .8 * st])
    raw = (cam + rng.normal(0, 30, cam.shape)).astype(np.float32)
    dn = (cam + rng.normal(0, 6, cam.shape)).astype(np.float32)
    return sky, raw, dn


def make_project(root: Path, denoise: bool = True, color: bool = True, gradient: bool = True):
    """A v1 project folder holding done stack (and gradient, color, denoise) results; the denoised stack is made from
    the stack with its gradient removed, as the Denoise block reads it. Returns (project, true sky, the gradient's
    context or None)."""
    sky, raw, dn = scene(H, W)
    proj = Project.create(root, "t", Inputs(lights=[]))
    wt = np.full((3, H, W), 20, np.float32)
    wt[:, :12] = 0
    wt[:, :, -9:] = 0
    raw[wt == 0] = 0
    dn[wt == 0] = 0

    def stage(name, files):
        d = proj.stage_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        for k, val in files.items():
            if k.endswith(".json"):
                (d / k).write_text(json.dumps(val))
            else:
                np.save(d / k, val)
        (d / "manifest.json").write_text(json.dumps({"status": "done"}))
    stage("stack", {"drz_full.npy": raw, "drz_weight.npy": wt, "drz_meta.json": {"scale": 1.5}})
    g = run_stage(proj, Gradient(), GradientParams()) if gradient else None
    if color:
        stage("color", {"M.npy": M})
    if denoise:
        stage("denoise", {"n2n_full.npy": dn - sky_variation(g, H, W) if g else dn})
    return proj, sky, g


def run_stage(proj, st, p):
    ctx = Context(proj, st, Journal(proj.logs, echo=False), "cpu")
    st.run(ctx, p)
    return ctx


def is_level(a: np.ndarray, valid: np.ndarray, tol: float) -> bool:
    """Whether ``a`` (3, H, W) is a constant per channel over ``valid``, to ``tol``."""
    return all(float(np.ptp(a[c][valid])) < tol for c in range(a.shape[0]))


@pytest.fixture(scope="module")
def classic(tmp_path_factory):
    """Classic + denoise: gradient, then render_simple."""
    proj, sky, g = make_project(tmp_path_factory.mktemp("classic") / "p")
    out = run_stage(proj, RenderSimple(), RenderSimpleParams())
    return proj, sky, g, out


def test_gradient_recovers_a_degree_2_gradient(classic):
    proj, sky, g, _ = classic
    info = json.loads((g.dir / "gradient.json").read_text())
    assert info["degree"] == 2 and (info["width"], info["height"]) == (W, H) and info["method"] == "polynomial"
    var = sky_variation(g, H, W)
    true = sky[:, 12:, :-9]                              # the covered field, in camera colours
    level = np.array(info["level"])
    assert np.allclose(level, np.median(true, axis=(1, 2)), rtol=0.01)    # the sky's level is kept...
    err = np.abs(var[:, 12:, :-9] + level[:, None, None] - true) / true
    assert float(err.max()) < 0.01                       # ...its variation within 1 % everywhere, objects left out
    s = np.load(g.dir / "samples.npz")
    assert s["used"].shape == (len(s["x"]),) and s["used"].any() and not bool(s["manual"])
    assert 0 < g.metrics["samples_used"] < len(s["x"])
    assert abs(g.metrics["gradient_pct"] - 100 * _range_ratio(true)) < 2.0
    assert (g.dir / "previews" / "gradient.jpg").exists()


def _range_ratio(rgb):
    L = np.tensordot([0.2126, 0.7152, 0.0722], rgb[:, ::8, ::8], 1)
    return float((np.percentile(L, 99) - np.percentile(L, 1)) / np.median(L))


def test_gradient_params():
    p = GradientParams()
    assert (p.method, p.degree, p.samples, p.object_clip) == ("polynomial", 2, 16, 2.5)
    for bad in ({"degree": 6}, {"samples": 2}, {"method": "spline"}):
        with pytest.raises(ValueError):
            GradientParams(**bad)


def test_render_simple_writes_its_files(classic):
    proj, sky, g, out = classic
    names = {p.name for p in out.dir.iterdir()}
    assert {"lin.npy", "lin_raw.npy", "valid.npy", "prep_simple.json"} <= names
    assert (out.dir / "previews" / "output.jpg").exists()
    prep = json.loads((out.dir / "prep_simple.json").read_text())
    for k in ("M", "crop", "median", "mad", "p99995", "stats", "stats_raw", "linked", "stretch", "drizzle_scale"):
        assert k in prep
    assert prep["denoised"] and prep["color"] and prep["exit"] is None and prep["sky"] == "sky level of the Gradient block"
    assert np.allclose(prep["M"], M)
    # lin = M x denoised stack (its gradient removed before) - a level; lin_raw the same from the raw stack
    lin, raw = np.load(out.dir / "lin.npy"), np.load(out.dir / "lin_raw.npy")
    dn = np.load(proj.stage_dir("denoise") / "n2n_full.npy")
    valid = np.load(out.dir / "valid.npy")
    assert valid.dtype == bool and not valid[:12].any() and valid[100:200, 100:200].all()
    assert is_level(lin - np.tensordot(M, dn, 1), valid, 1e-2)
    left = np.tensordot(M, sky - sky_variation(g, H, W), 1)                 # the true sky as it reads in lin
    assert is_level(left, valid, 0.02 * float(left.mean()))                 # flat to 2 % of the sky
    assert abs(float(np.median((raw - lin)[:, valid]))) < 1                # the same sky removed from both
    noise = lambda a: float(np.median(np.abs(np.diff(a[1, 50:400, 50:600], axis=1))))   # noqa: E731
    assert noise(raw) > 3 * noise(lin)                                      # the raw stack is the noisier one
    assert lin.dtype == np.float32 and lin.shape == (3, H, W)


def test_render_simple_without_denoise(tmp_path):
    proj, _, _ = make_project(tmp_path / "p", denoise=False)
    out = run_stage(proj, RenderSimple(), RenderSimpleParams())
    prep = json.loads((out.dir / "prep_simple.json").read_text())
    assert not (out.dir / "lin_raw.npy").exists() and not prep["denoised"] and prep["source"] == "stack/drz_full"


# the colour stage's white balance (one factor per colour) and the camera colour of a line emission
M_WB = np.diag([1.13, 0.75, 1.29])
EMISSION = np.array([1.0, 0.35, 0.20])


def emission_project(root: Path, denoise: bool):
    """A stack with stars of many colours (on a plane of camera colours), a red line-emission nebula off that plane,
    a sky gradient and noise, its Gradient block, its colour stage (a white balance) and its Gaia XP stars."""
    rng = np.random.default_rng(5)
    h, w = 1024, 1536
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    E = np.zeros((h, w), np.float32)
    for _ in range(8):
        cx, cy, s, a = rng.uniform(.2 * w, .8 * w), rng.uniform(.2 * h, .8 * h), rng.uniform(60, 160), rng.uniform(30, 90)
        E += a * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / s ** 2)
    n = 900
    x, y = rng.uniform(60, w - 60, n), rng.uniform(60, h - 60, n)
    g = rng.uniform(0.7, 1.4, n)                                    # star colours (1, g, 0.6 g - 0.22): a plane
    col = np.stack([np.ones(n), g, 0.6 * g - 0.22])
    flux = 10 ** rng.uniform(3.0, 4.6, n)
    st = np.zeros((3, h, w), np.float32)
    for c in range(3):
        np.add.at(st[c], (np.round(y).astype(int), np.round(x).astype(int)), (col[c] * flux).astype(np.float32))
        st[c] = cv2.GaussianBlur(st[c], (0, 0), 1.3)
    sky = np.stack([lv * (1 + 0.05 * (xx / w) - 0.03 * (yy / h)) for lv in (400, 520, 260)])
    cam = sky + EMISSION[:, None, None] * E + st
    raw = (cam + rng.normal(0, 4, cam.shape)).astype(np.float32)
    proj = Project.create(root, "t", Inputs(lights=[]))

    def stage(name, files):
        d = proj.stage_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        for k, val in files.items():
            np.save(d / k, val)
        (d / "manifest.json").write_text(json.dumps({"status": "done", "params": {"aperture": 4.0}}))
    stage("stack", {"drz_full.npy": raw, "drz_weight.npy": np.full((3, h, w), 20, np.float32)})
    (proj.stage_dir("stack") / "drz_meta.json").write_text(json.dumps({"scale": 1.0}))
    g = run_stage(proj, Gradient(), GradientParams())
    stage("color", {"M.npy": M_WB})
    stage("catalogs", {})
    if denoise:
        stage("denoise", {"n2n_full.npy": raw - sky_variation(g, h, w)})
    np.savez(proj.stage_dir("catalogs") / "gaia_xp.npz", x=x, y=y)
    return proj, col


@pytest.mark.parametrize("denoise", [False, True])
def test_render_simple_keeps_the_white_balance_and_the_gradient_sky_level(tmp_path, denoise):
    """The simple output applies the colour stage's white balance as it is (no matrix mixes the channels: the
    emission keeps its camera colour, balanced) and takes off the sky level the Gradient block measured between the
    objects (a median over the whole field would take the nebula for sky)."""
    proj, col = emission_project(tmp_path / "p", denoise)
    out = run_stage(proj, RenderSimple(), RenderSimpleParams())
    prep = json.loads((out.dir / "prep_simple.json").read_text())
    Ma = np.array(prep["M"])
    assert np.allclose(Ma, M_WB) and prep["sky"] == "sky level of the Gradient block"
    after = Ma @ EMISSION
    assert after[0] > after[2] > 0 and after[0] > after[1] > 0             # red stays red: no blue mixed in
    level = np.array(json.loads((proj.stage_dir("gradient") / "gradient.json").read_text())["level"])
    # lin is the stack (its gradient removed before) in the balanced colours minus the Gradient block's level
    lin = np.load(out.dir / "lin.npy")
    if denoise:
        src = np.load(proj.stage_dir("denoise") / "n2n_full.npy")
    else:
        src = np.load(proj.stage_dir("stack") / "drz_full.npy")
        src -= sky_variation(Context(proj, Gradient(), Journal(proj.logs, echo=False), "cpu"), *src.shape[1:])
    assert is_level(lin - np.tensordot(Ma, src, 1), np.ones(src.shape[1:], bool), 0.05)
    assert np.allclose((np.tensordot(Ma, src, 1) - lin).reshape(3, -1)[:, 0], Ma @ level, atol=0.05)


def test_raw_stack_exit_has_no_colour_cast(tmp_path):
    proj, _, _ = make_project(tmp_path / "p", denoise=False, color=False, gradient=False)
    out = run_stage(proj, RenderSimple(), RenderSimpleParams())
    prep = json.loads((out.dir / "prep_simple.json").read_text())
    assert prep["exit"] == "stacking" and prep["sky"] == "camera-colour polynomial, made grey" and not prep["color"]
    x0, x1, y0, y1 = prep["crop"]
    img = SimpleRenderEngine(out.dir, DEV).render(SimpleSettings(), 1.0).final.cpu().numpy()[:, y0:y1, x0:x1]
    # the sky gradient (another colour in each channel) stays visible but grey: every corner is neutral (the median
    # pixel of this scene is not sky: its nebulae are red)
    h, w = img.shape[1:]
    corners = [img[:, :h // 5, :w // 5], img[:, :h // 5, -w // 5:], img[:, -h // 5:, :w // 5], img[:, -h // 5:, -w // 5:]]
    cm_ = np.array([np.median(c.reshape(3, -1), axis=1) for c in corners])
    assert float((cm_.max(1) - cm_.min(1)).max()) < 0.01                  # 2.5 levels of 255
    assert float(cm_.mean(1).max() - cm_.mean(1).min()) > 0.01             # its brightness is kept
    # without the normalisation, camera RGB under a linked stretch is tinted: the sky of the scene is green-dominated
    cam = np.load(proj.stage_dir("stack") / "drz_full.npy")[:, y0:y1, x0:x1]
    cm = np.median(cam, axis=(1, 2))
    assert float(cm.max() - cm.min()) > 0.2 * float(cm.mean())


def test_stretch_choice_is_auto_only():
    assert RenderSimpleParams().stretch == "auto" and RenderSimpleParams().linked is True
    with pytest.raises(ValueError, match="not in the engine yet"):
        RenderSimpleParams(stretch="ghs")


# ---------------------------------------------------------------------- simple rendering
def simple_folder(d: Path, h: int, w: int, raw: bool = True, seed: int = 2) -> Path:
    """A render_simple folder written directly: a nebula and stars on a zero sky, noise."""
    rng = np.random.default_rng(seed)
    neb = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 30)
    neb = (neb - neb.min()) / (neb.max() - neb.min()) * 200
    lin = np.stack([neb, .5 * neb, .35 * neb]) + rng.normal(0, 3, (3, h, w)).astype(np.float32)
    d.mkdir(parents=True, exist_ok=True)
    valid = np.ones((h, w), bool)
    valid[:, :8] = False
    lin[:, ~valid] = 0
    np.save(d / "lin.npy", lin.astype(np.float32))
    shown = valid.copy()
    prep = {"M": np.eye(3).tolist(), "crop": [8, w, 0, h], "stats": image_stats(lin, shown), "linked": True,
            "stretch": "auto", "denoised": raw, "color": True, "exit": None, "drizzle_scale": 1.5}
    if raw:
        lr = lin + rng.normal(0, 12, lin.shape).astype(np.float32)
        lr[:, ~valid] = 0
        np.save(d / "lin_raw.npy", lr)
        prep["stats_raw"] = image_stats(lr, shown)
    np.save(d / "valid.npy", valid)
    (d / "prep_simple.json").write_text(json.dumps(prep))
    return d


def test_simple_render_2000x1500_under_2_s(tmp_path):
    d = simple_folder(tmp_path / "big", 1500, 2000)
    for dev in sorted({DEV, "cpu"}):
        eng = SimpleRenderEngine(d, dev)
        eng.render(SimpleSettings(), 0.25)                       # first use of the device
        s = SimpleSettings(noise_mix=0.7, smooth_px=1.2, contrast=0.2)
        t = time.perf_counter()
        lay = eng.render(s, 1.0)
        img = lay.final.cpu()
        dt = time.perf_counter() - t
        assert img.shape == (3, 1500, 2000) and lay.scale == 1.0
        assert dt < 2.0, f"{dev}: {dt:.2f} s"


def test_simple_render_settings_and_regions(tmp_path):
    d = simple_folder(tmp_path / "s", 300, 400)
    eng = SimpleRenderEngine(d, DEV)
    full = eng.render(SimpleSettings(smooth_px=1.6, noise_mix=0.8), 1.0).final.cpu().numpy()
    assert full.min() >= 0 and full.max() <= 1 and float(np.abs(full[:, :, :8]).max()) == 0
    # a region is the crop of the full rendering (the statistics are the folder's)
    reg = eng.render(SimpleSettings(smooth_px=1.6, noise_mix=0.8), 1.0, region=(100, 260, 50, 210))
    assert reg.info["region"] == [100, 260, 50, 210]
    assert float(np.abs(reg.final.cpu().numpy() - full[:, 50:210, 100:260]).max()) < 1e-5
    # the reduced previews look like the full image
    q = eng.render(SimpleSettings(smooth_px=1.6, noise_mix=0.8), 0.25).final.cpu().numpy()
    assert q.shape == (3, 75, 100)
    red = full[:, :300, :400].reshape(3, 75, 4, 100, 4).mean(axis=(2, 4))
    assert float(np.abs(q - red)[:, :, 3:].mean()) < 0.03
    # each setting moves the image the expected way
    base = eng.render(SimpleSettings(), 0.5).final.cpu().numpy()
    brighter = eng.render(SimpleSettings(sky=0.3), 0.5).final.cpu().numpy()
    assert brighter.mean() > base.mean()
    vivid = eng.render(SimpleSettings(saturation=2.0), 0.5).final.cpu().numpy()
    assert (vivid.max(0) - vivid.min(0)).mean() > (base.max(0) - base.min(0)).mean()
    noisy = eng.render(SimpleSettings(noise_mix=0.4), 1.0).final.cpu().numpy()
    plain = eng.render(SimpleSettings(), 1.0).final.cpu().numpy()
    hp = lambda a: np.abs(a[1] - cv2.GaussianBlur(a[1], (0, 0), 2))[:, 10:].mean()   # noqa: E731
    assert hp(noisy) > 1.5 * hp(plain) and hp(full) < hp(eng.render(SimpleSettings(noise_mix=0.8), 1.0)
                                                         .final.cpu().numpy())


# ---------------------------------------------------------------------- the v2 registry
def test_v2_stages_and_chains():
    from astro_gpu.core import workflow
    from astro_gpu.stages import all_stages
    from astro_gpu.stages.v2 import AUTO_FIELDS, NEW_STAGES, V2_STAGES
    names = [s.name for s in V2_STAGES()]
    assert names == [s.name for s in all_stages()] + list(NEW_STAGES)
    assert "gradient" not in [s.name for s in all_stages()]                      # v1's Pipeline never runs them
    workflow._CHAINS.clear()
    ch = workflow.chains()
    assert type(ch.stages["render_simple"]).__name__ == "RenderSimple"           # the real stages, no stand-ins
    classic = ch.chain(workflow.TEMPLATES["Classic + denoise"])
    assert classic[-4:] == ["gradient", "color", "denoise", "render_simple"]
    assert ch.deps_in_chain(classic, "render_simple") == ("stack", "color", "denoise", "gradient")
    raw_exit = ch.chain(workflow.TEMPLATES["Classic stack"], "stacking")
    assert ch.deps_in_chain(raw_exit, "render_simple") == ("stack",)
    assert ch.deps_in_chain(ch.chain(workflow.TEMPLATES["Classic stack"]), "gradient") == ("stack",)
    full = ch.chain(workflow.TEMPLATES["Full"])
    assert full[-2:] == ["render_simple", "star_removal"]
    assert ch.deps_in_chain(full, "star_removal") == ("render_simple", "stack", "gradient")
    stages = {s.name: s for s in V2_STAGES()}
    for st, fields in AUTO_FIELDS.items():
        assert fields <= set(stages[st].Params.model_fields), st


class _ReadOnlyProject:
    """A v1 project seen read-only: its done stage folders; any other stage maps to a temporary folder."""

    def __init__(self, root: Path, tmp: Path):
        self.root, self._tmp = root, tmp
        self.config = Project(root).config                      # reads project.json only
        self.logs = tmp / "logs"
        self.scratch = tmp / "scratch"
        self.results = tmp / "results"

    def stage_dir(self, stage: str) -> Path:
        d = self.root / "work" / stage
        return d if d.is_dir() else self._tmp / "absent" / stage

    def overrides(self, stage: str) -> dict:
        return dict(self.config.params.get(stage, {}))


@pytest.mark.skipif(not HEART_A7S.exists(), reason="test project F:/astro_essais/projets/heart_a7s not available")
def test_every_auto_key_of_heart_a7s_is_in_auto_fields(tmp_path):
    from astro_gpu.stages.v2 import AUTO_FIELDS, V2_STAGES
    before = {p: p.stat().st_mtime_ns for p in (HEART_A7S / "work").rglob("manifest.json")}
    proj = _ReadOnlyProject(HEART_A7S, tmp_path)
    journal = Journal(tmp_path / "logs", echo=False)
    seen = {}
    for st in V2_STAGES():
        ctx = Context(proj, st, journal, "cpu")
        autos = st.auto(ctx)
        keys = {k for k, a in autos.items() if not a.advice}      # an advice (the stack's method) is never applied
        seen[st.name] = keys
        allowed = AUTO_FIELDS.get(st.name, frozenset())
        assert keys <= allowed, (st.name, keys)
    assert seen["stack"] and seen["denoise"] == {"adaptation"} and seen["calibrate"] == {"workers"}
    assert {p: p.stat().st_mtime_ns for p in (HEART_A7S / "work").rglob("manifest.json")} == before
