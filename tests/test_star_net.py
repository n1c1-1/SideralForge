"""Learned star removal network: registration, off by default, its layer handed to the star removal, the training
loop and the tiled inference on tiny synthetic data (CPU, seconds)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from astro_gpu.core.journal import Journal
from astro_gpu.core.params import Auto, P, StageParams, resolve
from astro_gpu.core.project import Inputs, Project
from astro_gpu.core.runner import Pipeline
from astro_gpu.core.stage import Context, Interrupted, Stage
from astro_gpu.stages import all_stages
from astro_gpu.stages import star_net_lib as lib
from astro_gpu.stages.star_net import StarNet, StarNetParams, layer_file, stage_layer
from astro_gpu.stages.starless import Starless, StarlessParams


@pytest.fixture
def project(tmp_path):
    return Project.create(tmp_path / "proj", "test", Inputs(lights=[]))


def _ctx(project, stage):
    return Context(project, stage, Journal(project.logs, echo=False), "cpu")


def _fake_star_net(project, status="done", run_id="0123456789abcdef"):
    d = project.stage_dir("star_net")
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "star_layer.npy", np.zeros((3, 4, 4), np.float32))
    (d / "manifest.json").write_text(json.dumps({"status": status, "run_id": run_id}))
    return d / "star_layer.npy"


# ------------------------------------------------------------------------------------------------ the stage

def test_registered_between_stars_and_starless():
    stages = all_stages()
    names = [s.name for s in stages]
    assert names.index("stars") < names.index("star_net") < names.index("starless")
    net = stages[names.index("star_net")]
    assert all(names.index(d) < names.index("star_net") for d in net.deps)
    assert "star_net" not in Starless.deps             # an optional input: no dependency that could make it stale
    assert StarNetParams().enabled is True and StarNetParams().steps == 11000      # on by default (2026-10-16)
    assert {d["name"] for d in StarNetParams.describe()} >= {"enabled", "steps", "device"}


def test_disabled_writes_nothing(project):
    ctx = _ctx(project, StarNet())
    StarNet().run(ctx, StarNetParams(enabled=False))
    assert list(ctx.dir.iterdir()) == []


# ------------------------------------------------------------------------- the layer for the star removal

def test_starless_takes_the_stage_layer(project):
    ctx = _ctx(project, Starless())
    assert Starless().auto(ctx) == {}                   # no layer: the star removal's parameters do not change
    f = _fake_star_net(project)
    auto = Starless().auto(ctx)
    p, origin = resolve(StarlessParams, auto, {})
    assert origin["learned_stars"] == "auto" and layer_file(p.learned_stars) == f
    assert "01234567" in p.learned_stars               # a new layer (new run of the stage) is a new value
    p, origin = resolve(StarlessParams, auto, {"learned_stars": "D:/elsewhere/layer.npy"})
    assert origin["learned_stars"] == "user" and layer_file(p.learned_stars) == Path("D:/elsewhere/layer.npy")
    assert layer_file("") is None and layer_file("none") is None


def test_unfinished_stage_layer_is_ignored(project):
    _fake_star_net(project, status="running")
    assert stage_layer(project) is None
    _fake_star_net(project, status="done")
    (project.stage_dir("star_net") / "star_layer.npy").unlink()
    assert stage_layer(project) is None


class _UpParams(StageParams):
    k: int = P(1, label="k", help="k")


class _Up(Stage):
    name = "up"
    title = "up"
    Params = _UpParams

    def run(self, ctx, p):
        ctx.save_json("out.json", {"k": p.k})


class _NetParams(StageParams):
    enabled: bool = P(False, label="on", help="on")


class _Net(Stage):
    name = "net"
    title = "net"
    deps = ("up",)
    Params = _NetParams

    def run(self, ctx, p):
        if p.enabled:
            ctx.save_json("layer.json", {"ok": True})


class _DownParams(StageParams):
    layer: str = P("", label="layer", help="layer")


class _Down(Stage):
    name = "down"
    title = "down"
    deps = ("up",)
    Params = _DownParams
    runs = 0

    def auto(self, ctx):
        m = ctx.manifest("net")
        f = ctx.project.stage_dir("net") / "layer.json"
        if m.get("status") == "done" and f.exists():
            return {"layer": Auto(f"{f} (run {m['run_id'][:8]})", "the net's layer")}
        return {}

    def run(self, ctx, p):
        _Down.runs += 1
        ctx.save_json("out.json", {"layer": p.layer})


def test_optional_layer_in_the_pipeline(project):
    _Down.runs = 0
    j = Journal(project.logs, echo=False)
    assert Pipeline(project, [_Up(), _Down()], j).run() == "done" and _Down.runs == 1
    pl = Pipeline(project, [_Up(), _Net(), _Down()], j)           # the new stage, off: nothing downstream re-runs
    assert {s.name: s.state for s in pl.status()}["down"] == "done"
    assert pl.run() == "done" and _Down.runs == 1
    project.set_override("net", "enabled", True)                  # on: its layer is taken in the same run
    assert pl.run() == "done" and _Down.runs == 2
    assert "layer.json" in json.loads((project.stage_dir("down") / "out.json").read_text())["layer"]
    assert pl.plan() == []


# ------------------------------------------------------------------------------- the library, tiny data

def _tiny(H=192, W=192, seed=0):
    rng = np.random.default_rng(seed)
    bk, sg = np.full(3, 10.0, np.float32), np.ones(3, np.float32)
    S = (bk[:, None, None] + rng.normal(0, 0.1, (3, H, W))).astype(np.float32)
    n = rng.normal(0, 1, (3, H, W)).astype(np.float32)
    yy, xx = np.mgrid[-15:16, -15:16]
    star = (500 * np.exp(-(xx ** 2 + yy ** 2) / 8.0) + 20 * np.exp(-np.abs(yy)) * (np.abs(xx) < 15)).astype(np.float32)
    stamp = np.stack([star] * 3)
    library = {"bright": [dict(x=0.0, y=0.0, g=8.0, R=15, sat=np.zeros(3, bool), plateau=np.full(3, 1e4, np.float32),
                               t=stamp)],
               "faint": [dict(x=0.0, y=0.0, g=13.0, R=4, sat=np.zeros(3, bool), plateau=np.full(3, 1e4, np.float32),
                              t=stamp[:, 11:20, 11:20].astype(np.float16))]}
    dev = torch.device("cpu")
    synth = lib.Synth(library, np.full(3, 1e4), np.full(3, 0.03), sg, dev)
    WM, VAL = np.ones((H, W), np.uint8), np.ones((H, W), bool)
    cys, cxs = lib.patch_centres(WM, np.zeros(0), np.zeros(0), np.zeros(0), H, W, 64)
    data = dict(L=S + n, T=S + n, S_den=S, WM=WM, VAL=VAL, cys=cys, cxs=cxs, bk=bk, sg=sg,
                neb_col=np.ones(3, np.float32), items=lib.val_items([(32, 32, 160, 160)], synth, P=64))
    return data, synth, dev


def test_training_resumes_and_inference_runs(tmp_path):
    data, synth, dev = _tiny()
    ck = tmp_path / "train.pt"

    def stop(frac, msg):
        raise Interrupted("stop")
    with pytest.raises(Interrupted):
        lib.train_net(data, synth, dev, steps=5, batch=2, patch=64, progress=stop, ckpt=ck, report_every=1,
                      log=lambda m: None)
    assert ck.exists() and torch.load(ck, weights_only=False)["it"] == 1
    logs = []
    net, hist, steps = lib.train_net(data, synth, dev, steps=5, batch=2, patch=64, ckpt=ck, max_steps=2,
                                     log=logs.append)
    assert steps == 2 and any("resumed at step 1" in m for m in logs)
    assert np.isfinite(hist[-1]["loss"]) and np.isfinite(hist[-1]["val"])
    img = data["L"][:, :128, :128]
    out = lib.apply_tiled(net, img, data["bk"], data["sg"], dev, tile=64, ov=16, batch=2)
    assert out.shape == img.shape and np.isfinite(out).all()


def test_star_extent_law_and_hold_outs():
    g = np.linspace(5, 10.5, 20)
    true_a, true_b = 6.0, -0.2
    R = np.exp(true_a + true_b * g) / 1.15
    ext = {i: (int(r), int(r * 0.8)) for i, r in enumerate(R)}
    a, b = lib.fit_extent(ext, g)
    assert abs(b - true_b) < 0.02 and lib.radius_of(5.0, (a, b)) > lib.radius_of(10.0, (a, b))
    assert lib.fit_extent({0: (50, 60)}, g) is None and lib.radius_of(11.0, None) == pytest.approx(40.0)
    # hold-out boxes: the structured one where the luminance varies most, outside the bright-star zones
    H, W = 2048, 3072
    lum4 = np.zeros((H // 4, W // 4), np.float32)
    yy, xx = np.mgrid[0:H // 4, 0:W // 4]
    lum4 += 5 * np.sin(xx / 6.0) * ((xx > 500) & (xx < 640) & (yy > 300) & (yy < 440))
    zone4 = np.zeros_like(lum4, np.uint8)
    b1, b2 = lib.hold_out_boxes(lum4, zone4, H, W)
    assert 1800 <= (b1[0] + b1[2]) / 2 <= 2800 and 1000 <= (b1[1] + b1[3]) / 2 <= 1900
    assert b2[2] < b1[0] or b2[0] > b1[2] or b2[3] < b1[1] or b2[1] > b1[3]


# ------------------------------------------------------------------- the stage end to end, synthetic project

def _synthetic_project(root, H=416, W=544, seed=0):
    """Upstream outputs the stage reads, made up: a nebula, stars with a halo and horizontal spikes, the star model,
    two half-stacks, the denoised stack, catalogue fluxes."""
    rng = np.random.default_rng(seed)
    proj = Project.create(root, "synthetic", Inputs(lights=[]))
    w = proj.work

    def save(stage, name, arr):
        (w / stage).mkdir(parents=True, exist_ok=True)
        np.save(w / stage / name, arr)
    yy, xx = np.mgrid[0:H, 0:W]
    sky = (20 + 5 * np.exp(-((xx - 0.6 * W) ** 2 + (yy - 0.5 * H) ** 2) / 1.5e4))[None] * \
        np.array([1.0, 1.2, 0.8])[:, None, None]
    n = 100
    x, y, g = rng.uniform(0, W, n), rng.uniform(0, H, n), rng.uniform(10.5, 16, n)
    g[:3] = [7.0, 8.0, 9.0]
    flux = (10 ** (-0.4 * (g - 20)))[None] * np.ones((3, 1))
    model = np.zeros((3, H, W), np.float32)
    for i in range(n):
        r2 = (xx - x[i]) ** 2 + (yy - y[i]) ** 2
        prof = np.exp(-r2 / 8.0) / (8 * np.pi) + 0.02 * np.exp(-np.sqrt(r2) / 10) / 628 + \
            0.002 * np.exp(-np.abs(yy - y[i]) / 0.8) * np.exp(-np.abs(xx - x[i]) / 60) / 60
        model += (flux[:, i:i + 1, None] * prof[None]).astype(np.float32)
    truth = sky + model
    sat = np.percentile(truth, 99.99)
    A = np.minimum(truth + rng.normal(0, 1.4, truth.shape), sat).astype(np.float32)
    B = np.minimum(truth + rng.normal(0, 1.4, truth.shape), sat).astype(np.float32)
    save("optics", "lat_A.npy", A)
    save("optics", "lat_B.npy", B)
    save("optics", "lat_full.npy", (0.5 * (A + B)).astype(np.float32))
    (w / "optics" / "lateral.json").write_text(json.dumps(np.zeros((2, 2, 4)).tolist()))
    save("denoise", "n2n_full.npy", np.minimum(truth, sat).astype(np.float32))
    save("stars", "stars_model.npy", model)
    save("stars", "stars_model_faint.npy", np.zeros_like(model))
    save("stars", "stars_model_halo.npy", (0.02 * model).astype(np.float32))
    np.savez(w / "stars" / "star_fluxes.npz", x=x, y=y, g=g, flux=flux.astype(np.float32), pred=flux)
    save("color", "M.npy", np.eye(3, dtype=np.float32))
    for st in ("optics", "denoise", "stars", "color"):
        (w / st / "manifest.json").write_text(json.dumps({"status": "done", "run_id": st, "metrics": {}}))
    return proj


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_stage_end_to_end_on_a_synthetic_project(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "PATCH", 128)              # small frame: smaller training patches
    proj = _synthetic_project(tmp_path / "synthetic")
    st = StarNet()
    ctx = _ctx(proj, st)
    p = StarNetParams.model_construct(enabled=True, steps=3, device="auto", batch=1, seed=0)
    st.run(ctx, p)
    layer = np.load(ctx.dir / "star_layer.npy")
    assert layer.shape == (3, 416, 544) and layer.dtype == np.float32 and np.isfinite(layer).all()
    assert (ctx.dir / "star_net_model.pt").exists() and (ctx.dir / "previews" / "star_removal.jpg").exists()
    assert ctx.metrics["training_steps"] >= 1 and set(ctx.metrics["star_extent_px"]) == {"G5", "G7", "G9", "G11"}
    (ctx.dir / "manifest.json").write_text(json.dumps({"status": "done", "run_id": "feedbeef00"}))
    auto = Starless().auto(_ctx(proj, Starless()))
    assert layer_file(auto["learned_stars"].value) == ctx.dir / "star_layer.npy"


def _stamp_scene(H=301, c=150):
    yy, xx = np.mgrid[0:H, 0:H]
    r = np.hypot(yy - c, xx - c)
    neb = np.repeat((20 + 0.02 * xx)[None], 3, 0).astype(np.float32)                 # smooth nebula
    star = np.repeat((1000 * np.exp(-r ** 2 / 18) + 30 * np.exp(-r / 15))[None], 3, 0).astype(np.float32)
    return yy, xx, r, neb, star


def _stamps(NL, S, filled, H=301, c=150, fill=80.0):
    x, y, g = np.array([float(c)]), np.array([float(c)]), np.array([6.0])
    args = (NL, S, NL, x, y, g, np.array([100]), np.full(3, 1e5, np.float32), H, H)
    old = lib.stamp_library(*args)["bright"][0]
    new = lib.stamp_library(*args, fill=np.array([fill]), filled=filled)["bright"][0]
    return old, new


def test_bright_stamp_does_not_take_the_holes_of_the_starless_image():
    yy, xx, r, neb, star = _stamp_scene()
    S = neb.copy()
    S[:, r < 40] -= 5.0                                   # the starless image's hole under the bright star
    old, new = _stamps(neb + star, S, np.zeros(r.shape, bool))
    assert old["R"] == 100 and new["R"] == 100           # 80 px + 25 %
    band = (r[50:251, 50:251] >= 10) & (r[50:251, 50:251] <= 30)
    truth = star[:, 50:251, 50:251]
    assert abs(float((old["t"] - truth)[:, band].mean()) - 5.0) < 0.5      # the hole became "star light"
    assert abs(float((new["t"] - truth)[:, band].mean())) < 0.5            # not any more
    assert lib.fill_radii({0: (60, 100)}, np.array([6.0, 9.0]), None)[0] == pytest.approx(115.0)


def test_bright_stamp_keeps_the_nebula_out_and_symmetric_dark_rings_in():
    yy, xx, r, neb, star = _stamp_scene()
    glow = np.repeat((8 * np.exp(-(r / 25) ** 2))[None], 3, 0).astype(np.float32)    # a reflection nebula around it
    ring = np.repeat((-4.0 * ((r > 45) & (r < 60)))[None], 3, 0).astype(np.float32)  # point-symmetric dark ring
    lane = np.repeat((-6.0 * ((np.abs(yy - 150) < 4) & (xx > 160) & (xx < 220)))[None], 3, 0).astype(np.float32)
    NL = neb + glow + star + ring + lane
    S = neb + glow + lane                                # the model keeps the nebula and the lane
    filled = r < 8                                       # S's filled core
    _, new = _stamps(NL, S, filled)
    t = new["t"]
    win = (slice(None), slice(50, 251), slice(50, 251))
    rr, Y, X = r[win[1:]], yy[win[1:]], xx[win[1:]]
    glow_band = (rr >= 15) & (rr <= 35)
    assert abs(float((t - star[win])[:, glow_band].mean())) < 0.5           # the glow stays out of the stamp
    ring_band = (rr > 47) & (rr < 58) & ~((np.abs(Y - 150) < 6) & (X > 150))
    assert float(t[:, ring_band].mean()) < -3.0                             # the dark ring is in it
    lane_px = (np.abs(Y - 150) < 3) & (X > 165) & (X < 215) & ((rr < 45) | (rr > 60))
    d = (t - star[win])[:, lane_px]
    assert d.max() < 0.5 and d.mean() > -6.5             # the one-sided lane is not star light: never more than
    mirror = (np.abs(Y - 150) < 3) & (X > 80) & (X < 135) & ((rr < 45) | (rr > 60))   # the star, at most its depth less
    assert abs(float((t - star[win])[:, mirror].mean())) < 0.5             # and the other side of the star is exact


def test_stamps_are_saturated_as_the_rendering_sees_them():
    H, c = 301, 150
    yy, xx = np.mgrid[0:H, 0:H]
    r = np.hypot(yy - c, xx - c)
    neb = np.full((3, H, H), 20.0, np.float32)
    star = np.repeat(np.minimum(5000 * np.exp(-r ** 2 / 50), 1000.0)[None], 3, 0).astype(np.float32)
    x, y, g = np.array([float(c)]), np.array([float(c)]), np.array([5.0])
    b = lib.stamp_library(neb + star, neb, neb + star, x, y, g, np.array([100]), np.full(3, 1e5, np.float32), H, H,
                          fill=np.array([80.0]), filled=np.zeros((H, H), bool))["bright"][0]
    assert bool(np.asarray(b["sat"]).all())            # its plateau is the frame's brightest: saturated


def test_synth_glow_on_bright_stars_goes_into_the_target_too():
    t = np.zeros((3, 21, 21), np.float32)
    t[:, 10, 10] = 50.0
    lib_ = {"bright": [dict(x=0.0, y=0.0, g=6.0, R=10, sat=np.zeros(3, bool), plateau=np.full(3, 1e5, np.float32),
                            t=t)], "faint": []}
    sy = lib.Synth(lib_, np.full(3, 1e5), np.full(3, 0.1), np.ones(3), torch.device("cpu"), glow=True,
                   neb_col=[1.2, 1.0, 0.6])
    rng = np.random.default_rng(3)
    plans = [sy.plan(rng, 64) for _ in range(20)]
    assert any(p_[0][0] == "g" for p_ in plans) and any(p_[0][0] == "b" for p_ in plans)   # about half of them
    plan = next(p_ for p_ in plans if p_[0][0] == "g")
    X, T = torch.zeros(3, 64, 64), torch.zeros(3, 64, 64)
    light = sy.apply(X, plan, T)
    assert float(T.sum()) > 0                                          # the glow is in the target
    assert float((X - T).sum()) == pytest.approx(float(light.sum()), rel=0.05)   # the input holds it and the star
