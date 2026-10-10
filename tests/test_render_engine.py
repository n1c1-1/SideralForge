"""Rendering engine on a small synthetic 'render_prep' folder (runs on CPU in CI)."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

from astro_gpu.render.engine import RenderEngine
from astro_gpu.render.settings import PRESETS, RenderSettings

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    d = tmp_path_factory.mktemp("render_prep")
    rng = np.random.default_rng(3)
    H, W = 240, 360
    neb = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 12) * 60
    astro = np.stack([neb * 1.0 + 5, neb * 0.45 + 5, neb * 0.25 + 5]) + rng.normal(0, 0.5, (3, H, W)).astype(np.float32)
    np.save(d / "astro_prep.npy", astro.astype(np.float32))
    np.save(d / "valid.npy", np.ones((H, W), bool))
    n = 400
    g = rng.uniform(6, 19, n)
    x, y = rng.uniform(0, W, n), rng.uniform(0, H, n)
    f = 10 ** (-0.4 * (g - 10)) * 2e4
    flux = np.stack([f * 0.9, f, f * 0.8]).astype(np.float32)
    np.savez(d / "stars.npz", x=x, y=y, g=g, flux=flux, pred=flux, glow_flux=flux, n_measured=n)
    r = np.arange(0, 61, 0.25)
    np.savez(d / "profile.npz", r=r, prof=np.exp(-0.5 * (r / 1.1) ** 2) + 1e-4 * np.exp(-r / 8))
    (d / "prep.json").write_text(json.dumps({"crop": [2, W - 2, 2, H - 2], "alpha": 0.87, "vh": [1.0, 0.45, 0.25],
                                             "M": np.eye(3).tolist(), "drizzle_scale": 1.5, "width": W, "height": H,
                                             "mag_offset": 0.0, "e": [1.0, 0.45, 0.25]}))
    return d


def test_render_full_and_preview(prep):
    eng = RenderEngine(prep, device=DEV)
    s = RenderSettings()
    lay = eng.render(s, 1.0)
    for t in (lay.final, lay.starless, lay.stars):
        assert t.shape == (3, 240, 360)
        assert float(t.min()) >= 0 and float(t.max()) <= 1
    blend = 1 - (1 - lay.starless) * (1 - lay.stars)
    assert float((blend - lay.final).abs().mean()) < 1e-5
    small = eng.render(s, 0.5)
    assert small.final.shape == (3, 120, 180)
    # the preview looks like the reduced full-resolution image
    red = torch.nn.functional.avg_pool2d(lay.final[None], 2)[0]
    assert float((red - small.final).abs().mean()) < 0.03


def test_settings_change_and_cache(prep):
    eng = RenderEngine(prep, device=DEV)
    a = eng.render(RenderSettings(), 0.5).final.clone()
    b = eng.render(RenderSettings(sky_level=40), 0.5).final
    assert float(b.mean()) > float(a.mean())
    st0 = eng.render(RenderSettings(), 0.5).stars.clone()
    st1 = eng.render(RenderSettings(d10=7.0, d7=14.0), 0.5).stars
    assert float(st1.sum()) > float(st0.sum())
    off = eng.render(RenderSettings(stars=False), 0.5)
    assert float(off.stars.abs().max()) == 0


def test_presets_are_valid():
    for values in PRESETS.values():
        RenderSettings(**values)


def test_export_files_and_checks(prep, tmp_path):
    import tifffile

    from astro_gpu.core.project import Inputs, Project
    from astro_gpu.render.export import ExportSettings, export
    proj = Project.create(tmp_path / "proj", "t", Inputs(lights=[]))
    eng = RenderEngine(prep, device=DEV)
    meta = export(proj, eng, RenderSettings(), ExportSettings(name="test"))
    names = sorted(p.name for p in (proj.results / "test").iterdir())
    assert "test_full_resolution_16bit.tif" in names and "test.png" in names and "test.jpg" in names
    assert "test_starless.png" in names and "test_stars.png" in names and "test_settings.json" in names
    with tifffile.TiffFile(proj.results / "test" / "test_full_resolution_16bit.tif") as t:
        page = t.pages[0]
        assert page.dtype == np.uint16 and page.shape == (236, 356, 3)
        assert 34675 in page.tags                     # embedded ICC profile
    qa = meta["qa"]
    assert qa["layer_blend_error_255"] < 0.5 and "warnings" in qa
