"""Workflow layers (render/layers.py): the Output block's and the Stars block's folders, on synthetic preparations;
the export."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from astro_gpu.render.layers import NOTE_CPU, ExportOptions, LayerService, export_rgb, layer_kind, oriented_wcs
from astro_gpu.render.simple import SimpleRenderEngine, image_stats
from astro_gpu.render.sliders import WHY_STARS_ONLY
from astro_gpu.render.stars import StarsRenderEngine

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


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


def stars_prep(d: Path, H: int, W: int) -> Path:
    """A star_removal folder: a nebula without its stars, the stars alone, an uncovered border."""
    rng = np.random.default_rng(5)
    d.mkdir(parents=True, exist_ok=True)
    neb = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 20) * 400
    starless = np.stack([neb, .5 * neb, .3 * neb]) + rng.normal(0, 3, (3, H, W)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    star = np.zeros((H, W), np.float32)
    for x0, y0, a in zip(rng.uniform(0, W, 80), rng.uniform(0, H, 80), 10 ** rng.uniform(2, 3.5, 80)):
        star += (a * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 1.2 ** 2)).astype(np.float32)
    stars = np.stack([.9 * star, star, .8 * star])
    valid = np.ones((H, W), bool)
    valid[:, :8] = False
    np.save(d / "starless.npy", starless.astype(np.float32))
    np.save(d / "stars.npy", stars.astype(np.float32))
    np.save(d / "valid.npy", valid)
    prep = {"crop": [8, W - 4, 4, H - 4], "M": np.eye(3).tolist(), "drizzle_scale": 1.5, "linked": True,
            "stats_full": image_stats(starless + stars, valid)}
    (d / "prep_stars.json").write_text(json.dumps(prep))
    return d


@pytest.fixture(scope="module")
def preps(tmp_path_factory):
    t = tmp_path_factory.mktemp("layers")
    return {"stars": stars_prep(t / "stars", 360, 480), "simple": simple_prep(t / "simple", 300, 400),
            "stacking": simple_prep(t / "stacking", 300, 400, "stacking", False)}


# ---------------------------------------------------------------------- the layer service
def test_layer_service_renders_the_output_and_the_stars_layers(preps, tmp_path):
    svc = LayerService(device=DEV)
    assert layer_kind(preps["simple"]) == "simple" and layer_kind(preps["stacking"]) == "stacking"
    assert layer_kind(preps["stars"], "combined") == "combined"
    with pytest.raises(ValueError):
        layer_kind(preps["stars"], "other")
    with pytest.raises(FileNotFoundError):
        layer_kind(tmp_path)
    assert svc.sliders_of(preps["stacking"])["noise"] and not svc.sliders_of(preps["simple"])["noise"]
    assert svc.sliders_of(preps["stars"], "stars")["sky"] == WHY_STARS_ONLY
    li = svc.render(preps["stars"], None, {"sky": 3}, scale=0.25, part="combined")
    assert li.rgb.shape == (3, 90, 120) and li.rgb.dtype == np.float32 and li.crop == (8, 476, 4, 356)
    assert li.info["kind"] == "combined" and li.info["settings"]["sky"] == pytest.approx(0.26)
    assert li.info["grid"] == [480, 360] and li.info["drizzle_scale"] == 1.5
    # the sliders at 2 give the base rendering
    base = svc.render(preps["stars"], None, None, 0.25, part="combined").rgb
    every = {k: 2 for k in ("sky", "contrast", "saturation", "star_strength", "star_size", "star_color")}
    assert np.array_equal(svc.render(preps["stars"], None, every, 0.25, part="combined").rgb, base)
    # parts: the starless image, the stars alone, the two put back together in Screen (or the blend asked)
    sl = svc.render(preps["stars"], None, None, 0.25, part="starless").rgb
    st = svc.render(preps["stars"], None, None, 0.25, part="stars").rgb
    assert np.allclose(base, 1 - (1 - sl) * (1 - st), atol=1e-6) and st.max() > 0.2
    li = svc.render(preps["stars"], None, None, 0.25, part="combined", blend="Lighten")
    assert li.info["settings"]["blend"] == "Lighten" and np.allclose(li.rgb, np.maximum(sl, st), atol=1e-6)
    with pytest.raises(ValueError):
        svc.render(preps["simple"], part="stars")
    with pytest.raises((ValueError, FileNotFoundError)):            # the Stars block's folder needs a part
        svc.render(preps["stars"])
    # a region at full resolution
    reg = svc.render(preps["stars"], None, {"sky": 2.5}, 1.0, region=(100, 400, 80, 280), part="combined")
    assert reg.rgb.shape == (3, 200, 300) and reg.info["region"] == [100, 400, 80, 280]
    # simple layers: noise slider through the raw stack
    s0 = svc.render(preps["simple"], None, {"noise": 0}, 1.0)
    s2 = svc.render(preps["simple"], None, None, 1.0)
    assert s0.rgb.shape == (3, 300, 400) and s0.info["kind"] == "simple" and s0.crop == (4, 396, 4, 296)
    assert s0.info["settings"]["noise_mix"] == pytest.approx(0.4) and not np.array_equal(s0.rgb, s2.rgb)
    assert svc.render(preps["stacking"], None, {"noise": 0}, 0.5).info["settings"]["noise_mix"] == 1.0


def test_layer_service_evicts_the_least_recently_used_engine(preps):
    from astro_gpu.render.layers import folder_bytes
    sizes = {k: folder_bytes(preps[k]) for k in ("stars", "simple", "stacking")}
    svc = LayerService(device="cpu", budget_bytes=sizes["stars"] + sizes["simple"] + sizes["stacking"] // 2)
    assert isinstance(svc.engine(preps["stars"]), StarsRenderEngine)
    svc.engine(preps["simple"])
    svc.engine(preps["stars"])                                  # used again: "simple" is now the least recent
    svc.engine(preps["stacking"])
    loaded = [Path(p).name for p in svc.loaded()]
    assert loaded == ["stars", "stacking"]
    svc.evict("all")
    assert svc.loaded() == []


def test_gpu_out_of_memory_falls_back_to_the_cpu(preps):
    class FullGPU:
        dev = torch.device("cuda", 0)

        def render(self, *a, **k):
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
    svc = LayerService(device="cpu")
    k = svc._key(preps["simple"])
    svc._eng[k], svc._size[k] = FullGPU(), 1
    li = svc.render(preps["simple"], None, None, 0.5)
    assert li.info["note"] == NOTE_CPU and li.info["device"] == "cpu" and li.rgb.shape == (3, 150, 200)
    assert isinstance(svc.engine(preps["simple"]), SimpleRenderEngine)


def test_masks(preps):
    svc = LayerService(device=DEV)
    m = svc.masks(preps["stars"], 0.25)
    assert set(m) == {"halpha"} and m["halpha"].shape == (90, 120)
    v = m["halpha"]
    assert v.dtype == np.float32 and v.min() >= 0 and v.max() <= 1 and v.max() > 0.5
    full = svc.masks(preps["stars"], 1.0)
    reg = svc.masks(preps["stars"], 1.0, region=(100, 400, 80, 280))
    assert reg["halpha"].shape == (200, 300)
    assert float(np.abs(reg["halpha"] - full["halpha"][80:280, 100:400]).max()) < 1e-4
    assert svc.masks(preps["simple"], 0.5)["halpha"].shape == (150, 200)


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
        # the rows bottom up, as FITS (and Siril) have them; the history in the header
        assert np.array_equal(h[0].data[:, ::-1], rgb) and h[0].header["ROWORDER"] == "BOTTOM-UP"
        assert any("layer" in str(x) or "astro_gpu" in str(x) for x in h[0].header["HISTORY"])
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
    # the image's pixel (50, 40) from the top is the FITS row 120 - 1 - 40 from the bottom
    assert np.allclose(w.pixel_to_world_values(60, 45), w2.pixel_to_world_values(50, 79), atol=1e-12)


def test_reversing_the_rows_keeps_every_sky_position_with_the_distortion():
    """flip_rows_wcs: the solution of the image with its rows reversed, SIP terms included, gives each pixel the sky
    position it had (the FITS export writes its rows bottom up)."""
    from astropy.wcs import WCS

    from astro_gpu.render.layers import flip_rows_wcs
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    w.wcs.crval, w.wcs.crpix = [40.0, 61.0], [300.5, 210.5]
    w.wcs.cd = [[-2.0e-4, 0.3e-4], [0.4e-4, 2.0e-4]]
    hdr = w.to_header(relax=True)
    for name, terms in (("A", {(2, 0): 3e-6, (1, 1): -2e-6, (0, 2): 4e-6, (0, 3): 2e-9, (2, 1): -1e-9}),
                        ("B", {(2, 0): -1e-6, (1, 1): 5e-6, (0, 2): 2e-6, (3, 0): 1e-9, (1, 2): 3e-9})):
        hdr[f"{name}_ORDER"] = 3
        for (p_, q_), v in terms.items():
            hdr[f"{name}_{p_}_{q_}"] = v
    H = 420
    w1, w2 = WCS(hdr), WCS(flip_rows_wcs(hdr, H))
    assert w1.sip is not None and w2.sip is not None
    rng = np.random.default_rng(2)
    for x, y in rng.uniform(0, [600, H - 1], (20, 2)):
        assert np.allclose(w1.all_pix2world([[x, y]], 0), w2.all_pix2world([[x, H - 1 - y]], 0), atol=1e-11, rtol=0)


def test_budgets_are_reported(preps):
    """The service reports its render time (the §5.3 budgets are measured on Heart, outside the fast suite)."""
    svc = LayerService(device=DEV)
    svc.render(preps["stars"], None, None, 0.25, part="combined")
    t = time.perf_counter()
    li = svc.render(preps["stars"], None, {"sky": 3}, 0.25, part="combined")
    assert li.info["render_ms"] <= 1000 * (time.perf_counter() - t) + 1
