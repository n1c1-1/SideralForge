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


@pytest.fixture(scope="module")
def prep_layers(prep, tmp_path_factory):
    """The same scene as a layered preparation (render_prep version 7): full-resolution base N + optional layers."""
    import shutil
    d = tmp_path_factory.mktemp("render_prep_layers")
    for n in ("valid.npy", "stars.npz", "profile.npz"):
        shutil.copy(prep / n, d / n)
    astro = np.load(prep / "astro_prep.npy")
    rng = np.random.default_rng(5)
    H, W = astro.shape[1:]
    np.save(d / "N.npy", astro.astype(np.float32))
    np.save(d / "R.npy", (0.2 * astro - 0.2 * astro.mean((1, 2), keepdims=True)).astype(np.float32))
    O = np.zeros((H, W), np.float32)
    O[20:60, 30:90] = 5.0
    np.save(d / "O.npy", O + rng.normal(0, 0.05, (H, W)).astype(np.float32))
    lum = np.array([0.2126, 0.7152, 0.0722])                 # M = identity in the fixture
    np.save(d / "Eg0.npy", cv2.GaussianBlur(np.tensordot(lum, astro, 1), (0, 0), 8).astype(np.float32))
    np.save(d / "dY.npy", rng.normal(0, 0.8, (H, W)).astype(np.float16))
    np.save(d / "chroma_fine.npy", rng.normal(0, 0.5, (3, H, W)).astype(np.float16))
    pj = json.loads((prep / "prep.json").read_text())
    pj.update({"layers": 3, "w_cam": [1.0, 1.0, 1.0], "o": [0.0, 1.0, 0.75], "o_display": [0.0, 1.0, 0.75],
               "oiii_sigma": 1.0, "oiii_scale": 4.0})
    (d / "prep.json").write_text(json.dumps(pj))
    return d


def test_layers_mix_live(prep, prep_layers):
    plain = RenderSettings(fine_k=0, faint_smooth=0)
    old = RenderEngine(prep, device=DEV).render(plain, 1.0).final
    eng = RenderEngine(prep_layers, device=DEV)
    new = eng.render(plain, 1.0).final
    assert float((old - new).abs().max()) < 2e-3            # the base layer alone = the calibrated image
    base = eng._astro(plain, 1)
    # each slider changes the image through its own layer only
    d = eng._astro(RenderSettings(fine_k=1.5, fine_lo_pct=0, fine_hi_pct=1, faint_smooth=0), 1) - base
    e = torch.from_numpy(np.load(prep_layers / "dY.npy").astype(np.float32)).to(eng.dev)[None].expand(3, -1, -1)
    assert float((d - 1.5 * e).abs().mean()) < 0.03 * float((1.5 * e).abs().mean())
    c = eng._astro(RenderSettings(fine_k=0, faint_smooth=0, colour_detail=1.0), 1) - base
    assert float(c.abs().mean()) > 0.3
    r = eng._astro(RenderSettings(fine_k=0, faint_smooth=0, resid_boost=1.0), 1) - base
    R = torch.from_numpy(np.load(prep_layers / "R.npy")).to(eng.dev)
    assert float((r - R).abs().max()) < 1e-3
    # the base layer itself is never modified by the mixing (cached or host tensors)
    assert float((eng._astro(plain, 1) - base).abs().max()) == 0
    # OIII: added in its own colour above the threshold only
    ob = eng._astro(RenderSettings(fine_k=0, faint_smooth=0, oiii_boost=2.0, oiii_smooth=4, oiii_threshold=3.0), 1) - base
    assert abs(float(ob[1, 40, 60]) - 2.0 * 5.0) < 0.5 and abs(float(ob[2, 40, 60]) / float(ob[1, 40, 60]) - 0.75) < 1e-3
    assert float(ob[0].abs().max()) < 1e-4 and float((ob[1, 120:] > 0).float().mean()) < 0.01   # noise below threshold
    # faint-area smoothing removes small scales in faint areas only (decided on the whole signal)
    sm = eng._astro(RenderSettings(fine_k=0, faint_smooth=32, faint_lo_pct=55, faint_hi_pct=85), 1)
    hp = lambda a: (a[0] - torch.from_numpy(cv2.GaussianBlur(a.cpu().numpy()[0], (0, 0), 2)).to(eng.dev)).abs()  # noqa: E731
    Eg = np.load(prep_layers / "Eg0.npy")
    faint = torch.from_numpy(Eg < np.percentile(Eg, 30)).to(eng.dev)
    bright = torch.from_numpy(Eg > np.percentile(Eg, 95)).to(eng.dev)
    assert float(hp(sm)[faint].mean()) < 0.5 * float(hp(base)[faint].mean())
    assert float(hp(sm)[bright].mean()) > 0.8 * float(hp(base)[bright].mean())
    # previews mix the reduced layers
    small = eng.render(RenderSettings(colour_detail=0.8, resid_boost=0.5), 0.5).final
    assert small.shape == (3, 120, 180)


def test_saved_settings_of_other_versions():
    s = RenderSettings.from_saved({"cont_detail": 1.0, "chroma": 0.9, "unknown": 3})
    assert s.chroma == 0.9


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


def test_the_white_point_can_leave_out_a_small_very_bright_object(prep, tmp_path):
    import shutil
    for n in ("valid.npy", "stars.npz", "profile.npz", "prep.json"):
        shutil.copy(prep / n, tmp_path / n)
    astro = np.load(prep / "astro_prep.npy")
    astro[:, 100:106, 150:156] += 3000.0                     # 36 px, 0.04 % of the image, far above the nebula
    np.save(tmp_path / "astro_prep.npy", astro)
    eng = RenderEngine(tmp_path, device=DEV)
    s = RenderSettings(stars=False)
    w_default = eng.render(s, 1.0).info["tone_curve"]["white"]
    w_ignored = eng.render(RenderSettings(stars=False, white_ignore_pct=0.3), 1.0).info["tone_curve"]["white"]
    w_plain = RenderEngine(prep, device=DEV).render(s, 1.0).info["tone_curve"]["white"]
    assert w_default > 5 * w_plain                           # by default the object sets the white point
    assert abs(w_ignored - w_plain) < 0.2 * w_plain          # left out, the nebula sets it again
    assert RenderSettings().white_ignore_pct == 0.005 and 100.0 - 0.005 == 99.995   # default = the former rule
