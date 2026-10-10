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
    """The same scene stored as separate layers (render_prep version 2)."""
    import shutil
    d = tmp_path_factory.mktemp("render_prep_layers")
    for n in ("valid.npy", "stars.npz", "profile.npz"):
        shutil.copy(prep / n, d / n)
    astro = np.load(prep / "astro_prep.npy")
    rng = np.random.default_rng(5)
    Am = np.array([[1.0, 1.2], [0.45, 1.0], [0.25, 0.8]])
    EC = np.tensordot(np.linalg.pinv(Am), astro, 1)
    R_raw = astro - np.tensordot(Am, EC, 1)
    C = cv2.GaussianBlur(EC[1], (0, 0), 24)                 # as render_prep: smooth continuum and rest
    R = np.stack([cv2.GaussianBlur(c, (0, 0), 6) for c in R_raw])
    H, W = astro.shape[1:]
    np.save(d / "E.npy", EC[0].astype(np.float32))
    np.save(d / "C.npy", C.astype(np.float32))
    np.save(d / "R.npy", R.astype(np.float32))
    np.save(d / "R_raw.npy", R_raw.astype(np.float16))
    np.save(d / "C_raw.npy", EC[1].astype(np.float32))
    O = np.zeros((H, W), np.float32)
    O[20:60, 30:90] = 5.0
    np.save(d / "O.npy", O + rng.normal(0, 0.05, (H, W)).astype(np.float32))
    foot = np.zeros((H, W), np.float32)
    foot[100:140, 150:200] = 1
    np.save(d / "foot_w.npy", foot.astype(np.float16))
    np.save(d / "Eg0.npy", cv2.GaussianBlur(astro[0] - 0.87 * astro[1], (0, 0), 8).astype(np.float32))
    np.save(d / "dE.npy", rng.normal(0, 0.8, (H, W)).astype(np.float16))
    np.save(d / "chroma_fine.npy", rng.normal(0, 0.5, (3, H, W)).astype(np.float16))
    pj = json.loads((prep / "prep.json").read_text())
    pj.update({"Am": Am.tolist(), "layers": 2, "split": True, "o": [0.0, 1.0, 0.75], "o_display": [0.0, 1.0, 0.75],
               "oiii_sigma": 1.0, "oiii_scale": 4.0})
    (d / "prep.json").write_text(json.dumps(pj))
    # the equivalent pre-mixed preparation (version 1)
    d1 = tmp_path_factory.mktemp("render_prep_mixed")
    for n in ("valid.npy", "stars.npz", "profile.npz", "prep.json"):
        shutil.copy(prep / n, d1 / n)
    mix = Am[:, :1, None] * EC[0][None] + Am[:, 1:, None] * C[None] + R
    np.save(d1 / "astro_prep.npy", mix.astype(np.float32))
    return d, d1


def test_layers_mix_live(prep_layers):
    prep_layers, mixed = prep_layers
    plain = RenderSettings(fine_k=0, faint_smooth=0)
    old = RenderEngine(mixed, device=DEV).render(plain, 1.0).final
    eng = RenderEngine(prep_layers, device=DEV)
    new = eng.render(plain, 1.0).final
    assert float((old - new).abs().max()) < 2e-3            # layers without extras = the pre-mixed image
    base = eng._astro(plain, 1)
    # each layer slider changes the image, and only through its own layer
    d = eng._astro(RenderSettings(fine_k=1.5, fine_lo_pct=0, fine_hi_pct=1, faint_smooth=0), 1) - base
    e = eng.Am[:, 0, None, None] * torch.from_numpy(np.load(prep_layers / "dE.npy").astype(np.float32)).to(eng.dev)
    assert float((d - 1.5 * e).abs().mean()) < 0.03 * float((1.5 * e).abs().mean())
    c = eng._astro(RenderSettings(fine_k=0, faint_smooth=0, colour_detail=1.0), 1) - base
    assert float(c.abs().mean()) > 0.3
    r = eng._astro(RenderSettings(fine_k=0, faint_smooth=0, resid_boost=1.0), 1) - base
    R = torch.from_numpy(np.load(prep_layers / "R.npy")).to(eng.dev)
    assert float((r - R).abs().max()) < 1e-3
    # continuum detail: the unsmoothed continuum comes back, except on star footprints
    cd = eng._astro(RenderSettings(fine_k=0, faint_smooth=0, cont_detail=1.0), 1)
    full = np.tensordot(eng.Am.cpu().numpy(), np.stack([np.load(prep_layers / "E.npy"), np.load(prep_layers / "C_raw.npy")]), 1)
    full = torch.from_numpy(full + np.load(prep_layers / "R.npy")).float().to(eng.dev)
    assert float((cd - full)[:, :90].abs().max()) < 1e-3
    assert float((cd - base)[:, 105:135, 155:195].abs().max()) < 1e-3
    # OIII: added in its own colour above the threshold only
    ob = eng._astro(RenderSettings(fine_k=0, faint_smooth=0, oiii_boost=2.0, oiii_smooth=4, oiii_threshold=3.0), 1) - base
    assert abs(float(ob[1, 40, 60]) - 2.0 * 5.0) < 0.5 and abs(float(ob[2, 40, 60]) / float(ob[1, 40, 60]) - 0.75) < 1e-3
    assert float(ob[0].abs().max()) < 1e-4 and float((ob[1, 120:] > 0).float().mean()) < 0.01   # noise below threshold
    # faint-area smoothing removes small scales in faint areas only
    sm = eng._astro(RenderSettings(fine_k=0, faint_smooth=32, faint_lo_pct=55, faint_hi_pct=85), 1)
    hp = lambda a: (a[0] - torch.from_numpy(cv2.GaussianBlur(a.cpu().numpy()[0], (0, 0), 2)).to(eng.dev)).abs()  # noqa: E731
    Eg = np.load(prep_layers / "Eg0.npy")
    faint = torch.from_numpy(Eg < np.percentile(Eg, 30)).to(eng.dev)
    assert float(hp(sm)[faint].mean()) < 0.5 * float(hp(base)[faint].mean())
    # previews mix the reduced layers
    small = eng.render(RenderSettings(colour_detail=0.8, resid_boost=0.5), 0.5).final
    assert small.shape == (3, 120, 180)


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
