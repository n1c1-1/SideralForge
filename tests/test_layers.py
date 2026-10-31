"""Workflow layers (render/layers.py) and region rendering (render/engine.py), on synthetic preparations."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from astro_gpu.render.engine import RenderEngine, settings_key
from astro_gpu.render.layers import NOTE_CPU, ExportOptions, LayerService, export_rgb, layer_kind, oriented_wcs
from astro_gpu.render.settings import PRESETS, RenderSettings
from astro_gpu.render.simple import image_stats

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
REPO = Path(__file__).resolve().parents[1]
V1_COMMIT = "f6de621"           # the engine before region rendering (P4)
SETTINGS = [RenderSettings(), RenderSettings(**PRESETS["Artistic"]),
            RenderSettings(oiii_boost=1.5, sky_neutral=0.7, bright_halo_desat=0.5, faint_smooth=12, colour_detail=0.8)]


def engine_prep(d: Path, H: int, W: int, layered: bool = True, seed: int = 3) -> Path:
    """A render_prep folder: a nebula, stars, the optional layers of a layered preparation, an uncovered border."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    neb = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), max(H, W) / 30)
    neb = (neb - neb.min()) / (neb.max() - neb.min()) * 120
    astro = np.stack([neb + 5, neb * .45 + 5, neb * .25 + 5]) + rng.normal(0, .8, (3, H, W)).astype(np.float32)
    valid = np.ones((H, W), bool)
    valid[:, :20] = False
    np.save(d / "valid.npy", valid)
    lum = np.array([0.2126, 0.7152, 0.0722])
    pj = {"crop": [24, W - 4, 4, H - 4], "alpha": 0.87, "vh": [1.0, 0.45, 0.25], "M": np.eye(3).tolist(),
          "drizzle_scale": 1.5}
    if layered:
        np.save(d / "N.npy", astro.astype(np.float32))
        np.save(d / "Eg0.npy", cv2.GaussianBlur(np.tensordot(lum, astro, 1), (0, 0), 8).astype(np.float32))
        np.save(d / "dY.npy", rng.normal(0, .8, (H, W)).astype(np.float16))
        np.save(d / "R.npy", (0.2 * astro - 0.2 * astro.mean((1, 2), keepdims=True)).astype(np.float32))
        O = np.zeros((H, W), np.float32)
        O[H // 4:H // 2, W // 4:W // 2] = 5
        np.save(d / "O.npy", O + rng.normal(0, .05, (H, W)).astype(np.float32))
        np.save(d / "chroma_fine.npy", rng.normal(0, .5, (3, H, W)).astype(np.float16))
        pj.update({"w_cam": [1, 1, 1], "o": [0, 1, .75], "o_display": [0, 1, .75], "oiii_sigma": 1.0,
                   "oiii_scale": 4.0})
    else:
        np.save(d / "astro_prep.npy", astro.astype(np.float32))
    n = int(H * W / 300)
    g = rng.uniform(6, 19, n)
    f = 10 ** (-0.4 * (g - 10)) * 2e4
    flux = np.stack([f * .9, f, f * .8]).astype(np.float32)
    np.savez(d / "stars.npz", x=rng.uniform(-30, W + 30, n), y=rng.uniform(-30, H + 30, n), g=g, flux=flux,
             pred=flux, glow_flux=flux)
    r = np.arange(0, 61, 0.25)
    np.savez(d / "profile.npz", r=r, prof=np.exp(-0.5 * (r / 1.1) ** 2) + 1e-4 * np.exp(-r / 8))
    (d / "prep.json").write_text(json.dumps(pj))
    return d


def simple_prep(d: Path, H: int, W: int, exit_: str | None = None, denoised: bool = True) -> Path:
    rng = np.random.default_rng(4)
    d.mkdir(parents=True, exist_ok=True)
    neb = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 20) * 400
    lin = np.stack([neb, .5 * neb, .3 * neb]) - 150 + rng.normal(0, 3, (3, H, W)).astype(np.float32)
    valid = np.ones((H, W), bool)
    prep = {"M": np.eye(3).tolist(), "crop": [4, W - 4, 4, H - 4], "stats": image_stats(lin, valid), "linked": True,
            "stretch": "auto", "denoised": denoised, "color": exit_ is None, "exit": exit_, "drizzle_scale": 1.0}
    np.save(d / "lin.npy", lin.astype(np.float32))
    if denoised:
        raw = (lin + rng.normal(0, 10, lin.shape)).astype(np.float32)
        np.save(d / "lin_raw.npy", raw)
        prep["stats_raw"] = image_stats(raw, valid)
    np.save(d / "valid.npy", valid)
    (d / "prep_simple.json").write_text(json.dumps(prep))
    return d


@pytest.fixture(scope="module")
def preps(tmp_path_factory):
    t = tmp_path_factory.mktemp("layers")
    return {"small": engine_prep(t / "small", 240, 360), "small_v1": engine_prep(t / "small_v1", 240, 360, False),
            "big": engine_prep(t / "big", 720, 1080), "simple": simple_prep(t / "simple", 300, 400),
            "stacking": simple_prep(t / "stacking", 300, 400, "stacking", False)}


# ---------------------------------------------------------------------- v1 rendering unchanged
def _v1_engine_module():
    try:
        src = subprocess.run(["git", "show", f"{V1_COMMIT}:astro_gpu/render/engine.py"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {V1_COMMIT} not available")
    name = "astro_gpu.render._engine_v1"
    mod = types.ModuleType(name)
    mod.__package__ = "astro_gpu.render"
    sys.modules[name] = mod
    exec(compile(src, f"{V1_COMMIT}:engine.py", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("folder", ["small", "small_v1"])
def test_v1_render_is_unchanged_bit_for_bit(preps, folder):
    """RenderEngine.render(s, scale) gives exactly the images of the engine before region rendering (same device)."""
    old = _v1_engine_module().RenderEngine(preps[folder], device=DEV)
    new = RenderEngine(preps[folder], device=DEV)
    for s in SETTINGS + [RenderSettings(stars=False), RenderSettings(**PRESETS["Prominent stars"])]:
        for scale in (1.0, 0.5, 0.25):
            a, b = old.render(s, scale), new.render(s, scale)
            for x, y in ((a.final, b.final), (a.starless, b.starless), (a.stars, b.stars)):
                assert torch.equal(x, y), (folder, scale)
            assert a.info == b.info and a.scale == b.scale


# ---------------------------------------------------------------------- regions
def _levels(a: np.ndarray, b: np.ndarray, margin: int = 64) -> float:
    """p99.9 of the difference in 8-bit levels, away from a margin."""
    d = np.abs(a - b)[:, margin:-margin, margin:-margin] * 255
    return float(np.percentile(d, 99.9))


@pytest.mark.parametrize("s", SETTINGS, ids=["default", "artistic", "oiii"])
def test_region_with_the_quarter_fit_equals_the_full_render(preps, s):
    eng = RenderEngine(preps["big"], device=DEV)
    full = eng.render(s, 1.0).final.cpu().numpy()
    fit = eng.fit_of(s)
    assert fit["f"] == 4 and fit["key"] == settings_key(s) and "tone_p" in fit["stats"]
    for region in [(300, 800, 200, 560), (0, 420, 0, 330), (700, 1080, 380, 720)]:     # inside, at the corners
        x0, x1, y0, y1 = region
        lay = eng.render(s, 1.0, region=region, fit=fit)
        r = lay.final.cpu().numpy()
        assert r.shape == (3, y1 - y0, x1 - x0) and lay.info["region"] == list(region) and lay.scale == 1.0
        assert _levels(r, full[:, y0:y1, x0:x1]) <= 2.0, region
        # with the same statistics, the region is the crop of the whole rendering
        same = eng.render(s, 1.0, fit=fit).final.cpu().numpy()
        assert _levels(r, same[:, y0:y1, x0:x1]) <= 0.5, region


def test_region_fit_reuse_scales_and_errors(preps):
    eng = RenderEngine(preps["big"], device=DEV)
    s = RenderSettings()
    region = (300, 800, 200, 560)
    a = eng.render(s, 1.0, region=region)                       # no fit given: the 1/4 rendering's, rendered once
    assert (settings_key(s), 4) in eng._fit_memo
    b = eng.render(s, 1.0, region=region, fit={"key": "other settings", "f": 4, "stats": {}})   # stale: refitted
    assert torch.equal(a.final, b.final)
    # a star slider keeps the fit (its statistics are the nebula's) and the region's cached nebula
    view, n_fits = eng._view_cache[1], len(eng._fit_memo)
    bigger = RenderSettings(d10=6.0, d7=12.0)
    assert settings_key(bigger) == settings_key(s)
    c = eng.render(bigger, 1.0, region=region)
    assert eng._view_cache[1] is view and len(eng._fit_memo) == n_fits
    assert torch.equal(c.starless, a.starless) and float(c.stars.sum()) > float(a.stars.sum())
    half = eng.render(s, 0.5, region=region)
    full_half = eng.render(s, 0.5).final
    assert half.final.shape == (3, 180, 250) and half.scale == 0.5
    assert _levels(half.final.cpu().numpy(), full_half[:, 100:280, 150:400].cpu().numpy(), 32) <= 2.0
    with pytest.raises(ValueError, match="empty region"):
        eng.render(s, 1.0, region=(500, 500, 10, 20))


# ---------------------------------------------------------------------- the layer service
def test_layer_service_renders_engine_and_simple_layers(preps):
    svc = LayerService(device=DEV)
    assert layer_kind(preps["big"]) == "engine" and layer_kind(preps["simple"]) == "simple"
    assert layer_kind(preps["stacking"]) == "stacking"
    assert svc.sliders_of(preps["stacking"])["noise"] and not svc.sliders_of(preps["simple"])["noise"]
    li = svc.render(preps["big"], "Artistic", {"sky": 3}, scale=0.25)
    assert li.rgb.shape == (3, 180, 270) and li.rgb.dtype == np.float32 and li.crop == (24, 1076, 4, 716)
    assert li.info["kind"] == "engine" and li.info["settings"]["sky_level"] == pytest.approx(40.0)
    assert li.info["grid"] == [1080, 720] and li.info["drizzle_scale"] == 1.5 and li.info["fit"]["f"] == 4
    # the sliders at 2 give the base rendering
    eng = RenderEngine(preps["big"], device=DEV)
    base = eng.render(RenderSettings(**PRESETS["Artistic"]), 0.25).final.cpu().numpy()
    assert np.array_equal(svc.render(preps["big"], "Artistic", {k: 2 for k in ("sky", "star_size")}).rgb, base)
    # parts: the starless image, and the stars as the layer minus it
    st = svc.render(preps["big"], None, None, 0.25, part="stars").rgb
    sl = svc.render(preps["big"], None, None, 0.25, part="starless").rgb
    fin = svc.render(preps["big"], None, None, 0.25).rgb
    assert np.allclose(st, np.clip(fin - sl, 0, 1)) and st.max() > 0.2
    with pytest.raises(ValueError):
        svc.render(preps["simple"], part="stars")
    # a region at full resolution
    reg = svc.render(preps["big"], None, {"sky": 2.5}, 1.0, region=(300, 800, 200, 560))
    assert reg.rgb.shape == (3, 360, 500) and reg.info["region"] == [300, 800, 200, 560]
    # simple layers: noise slider through the raw stack
    s0 = svc.render(preps["simple"], None, {"noise": 0}, 1.0)
    s2 = svc.render(preps["simple"], None, None, 1.0)
    assert s0.rgb.shape == (3, 300, 400) and s0.info["kind"] == "simple" and s0.crop == (4, 396, 4, 296)
    assert s0.info["settings"]["noise_mix"] == pytest.approx(0.4) and not np.array_equal(s0.rgb, s2.rgb)
    assert svc.render(preps["stacking"], None, {"noise": 0}, 0.5).info["settings"]["noise_mix"] == 1.0


def test_layer_service_evicts_the_least_recently_used_engine(preps):
    from astro_gpu.render.layers import folder_bytes
    sizes = {k: folder_bytes(preps[k]) for k in ("small", "simple", "stacking")}
    svc = LayerService(device="cpu", budget_bytes=sizes["small"] + sizes["simple"] + sizes["stacking"] // 2)
    svc.engine(preps["small"])
    svc.engine(preps["simple"])
    svc.engine(preps["small"])                                  # used again: "simple" is now the least recent
    svc.engine(preps["stacking"])
    loaded = [Path(p).name for p in svc.loaded()]
    assert loaded == ["small", "stacking"]
    svc.evict("all")
    assert svc.loaded() == []


def test_gpu_out_of_memory_falls_back_to_the_cpu(preps):
    class FullGPU:
        dev = torch.device("cuda", 0)

        def render(self, *a, **k):
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
    svc = LayerService(device="cpu")
    k = svc._key(preps["small"])
    svc._eng[k], svc._size[k] = FullGPU(), 1
    li = svc.render(preps["small"], None, None, 0.5)
    assert li.info["note"] == NOTE_CPU and li.info["device"] == "cpu" and li.rgb.shape == (3, 120, 180)
    assert isinstance(svc.engine(preps["small"]), RenderEngine)


def test_masks(preps):
    svc = LayerService(device=DEV)
    m = svc.masks(preps["big"], 0.25)
    assert m["stars"].shape == m["halpha"].shape == (180, 270)
    for v in m.values():
        assert v.dtype == np.float32 and v.min() >= 0 and v.max() <= 1 and v.max() > 0.5
    full = svc.masks(preps["big"], 1.0)
    reg = svc.masks(preps["big"], 1.0, region=(300, 800, 200, 560))
    assert reg["halpha"].shape == (360, 500)
    assert float(np.abs(reg["halpha"] - full["halpha"][200:560, 300:800]).max()) < 1e-4
    assert float(np.abs(reg["stars"] - full["stars"][200:560, 300:800])[16:-16, 16:-16].max()) < 0.02
    s = svc.masks(preps["simple"], 0.5)
    assert s["stars"] is None and s["halpha"].shape == (150, 200)


# ---------------------------------------------------------------------- export
def test_exports_reopen_with_the_right_dtype_and_size(tmp_path):
    import tifffile
    from astropy.io import fits
    from PIL import Image
    rng = np.random.default_rng(0)
    rgb = rng.random((3, 300, 500)).astype(np.float32)
    files = export_rgb(rgb, tmp_path, {"name": "t", "formats": ["tiff16", "tiff32f", "png16", "jpeg", "fits32f"]},
                       history={"layers": ["Full"]})
    names = sorted(p.name for p in files)
    assert names == ["t.fits", "t.jpg", "t.png", "t_16bit.tif", "t_32bit_float.tif", "t_history.json"]
    a = tifffile.imread(tmp_path / "t_16bit.tif")
    assert a.dtype == np.uint16 and a.shape == (300, 500, 3)
    assert np.abs(a.astype(np.float64) / 65535 - rgb.transpose(1, 2, 0)).max() < 1e-4
    b = tifffile.imread(tmp_path / "t_32bit_float.tif")
    assert b.dtype == np.float32 and b.shape == (300, 500, 3) and np.array_equal(b, rgb.transpose(1, 2, 0))
    c = cv2.imread(str(tmp_path / "t.png"), cv2.IMREAD_UNCHANGED)
    assert c.dtype == np.uint16 and c.shape == (300, 500, 3)
    assert np.abs(c[:, :, ::-1].astype(np.float64) / 65535 - rgb.transpose(1, 2, 0)).max() < 1e-4
    with Image.open(tmp_path / "t.jpg") as im:
        assert im.size == (500, 300) and im.mode == "RGB" and im.info.get("icc_profile")
    with fits.open(tmp_path / "t.fits") as h:
        assert h[0].data.dtype.kind == "f" and h[0].data.dtype.itemsize == 4 and h[0].data.shape == (3, 300, 500)
    # reduced (long side) and turned
    small = export_rgb(rgb, tmp_path / "s", {"name": "s", "formats": ["tiff16"], "size": 2048})
    assert tifffile.imread(small[0]).shape == (300, 500, 3)                       # never enlarged
    big = rng.random((3, 1500, 4600)).astype(np.float32)
    f = export_rgb(big, tmp_path / "b", {"name": "b", "formats": ["tiff16", "jpeg"], "size": 2048, "rot90": 1})
    assert tifffile.imread(f[0]).shape == (2048, 668, 3)
    with Image.open(f[1]) as im:
        assert im.size == (668, 2048)
    with pytest.raises(ValueError):
        export_rgb(rgb, tmp_path, {"formats": ["xisf"]})


@pytest.mark.parametrize("k,flip,zoom", [(0, False, 1.0), (1, False, 1.0), (2, True, 1.0), (3, False, 0.5),
                                         (1, True, 0.25)])
def test_the_fits_wcs_follows_crop_orientation_and_size(k, flip, zoom):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [40.0, 61.0]
    w.wcs.crpix = [540.5, 360.5]
    w.wcs.cd = [[-2.0e-4, 0.5e-4], [0.4e-4, 2.0e-4]]
    H, W, origin = 600, 900, (37, 21)                   # the exported image: a 900 x 600 crop at (37, 21) of the grid
    hdr = oriented_wcs(w.to_header(), H, W, origin, k, flip, zoom)
    w2 = WCS(hdr)
    rng = np.random.default_rng(1)
    for _ in range(5):
        # a pixel of the grid inside the crop, and where it lands in the exported image
        gx, gy = origin[0] + rng.uniform(0, W - 1), origin[1] + rng.uniform(0, H - 1)
        x, y = gx - origin[0], gy - origin[1]
        if flip:
            y = H - 1 - y
        h_, w_ = H, W
        for _ in range(k):                              # np.rot90: (x, y) -> (y, w - 1 - x)
            x, y = y, w_ - 1 - x
            h_, w_ = w_, h_
        x, y = (x + 0.5) * zoom - 0.5, (y + 0.5) * zoom - 0.5
        a = w.pixel_to_world_values(gx, gy)
        b = w2.pixel_to_world_values(x, y)
        assert np.allclose(a, b, atol=1e-9), (k, flip, zoom)


def test_export_writes_the_wcs_of_the_grid(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [40.0, 61.0]
    w.wcs.crpix = [100.5, 80.5]
    w.wcs.cd = [[-2.0e-4, 0.0], [0.0, 2.0e-4]]
    rgb = np.random.default_rng(0).random((3, 120, 200)).astype(np.float32)
    f = export_rgb(rgb, tmp_path, ExportOptions(name="w", formats=["fits32f"]), wcs=w, origin=(10, 5))
    w2 = WCS(fits.getheader(f[0])).celestial
    assert np.allclose(w.pixel_to_world_values(60, 45), w2.pixel_to_world_values(50, 40))


def test_budgets_are_reported(preps):
    """The service reports its render time (the §5.3 budgets are measured on Heart, outside the fast suite)."""
    svc = LayerService(device=DEV)
    svc.render(preps["big"], None, None, 0.25)
    t = time.perf_counter()
    li = svc.render(preps["big"], None, {"sky": 3}, 0.25)
    assert li.info["render_ms"] <= 1000 * (time.perf_counter() - t) + 1
