"""The Gradient block's RBF, placed samples and excluded areas (stages/gradient.py), on a synthetic sky."""
from __future__ import annotations

import numpy as np

from astro_gpu.stages.gradient import GradientParams, exclusion_mask, read_grid, sky_model_of

H, W = 300, 450


def scene(nebula: float = 0.0):
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W], np.float32)[:, None, None]
    sky = 100 + 20 * xx + 10 * yy ** 2 + 15 * np.exp(-np.hypot(xx - 1, yy) / 0.3)       # a corner glow
    neb = nebula * np.exp(-((xx - 0.45) ** 2 + (yy - 0.5) ** 2) / 0.03)
    img = np.stack([sky + neb] * 3) + rng.normal(0, 0.5, (3, H, W))
    return img.astype(np.float32), sky


def model(img, **kw):
    m = sky_model_of(img, np.ones((H, W), bool), GradientParams(**kw))
    return read_grid(m["grid"], m["info"], H, W)[0], m


def test_rbf_follows_a_corner_glow_better_than_the_polynomial():
    img, sky = scene()
    err = {}
    for meth in ("polynomial", "rbf"):
        g, _m = model(img, method=meth, sample_points=[], rbf_smooth=0.0) if meth == "rbf" else model(img, method=meth)
        d = (g - np.median(g)) - (sky - np.median(sky))
        err[meth] = float(np.sqrt((d ** 2).mean()))
    assert err["rbf"] < 0.85 * err["polynomial"], err


def test_placed_samples_are_the_samples():
    img, _sky = scene()
    pts = [[40.0, 40.0], [400.0, 40.0], [40.0, 260.0], [400.0, 260.0], [220.0, 150.0], [300.0, 100.0], [120.0, 200.0]]
    _g, m = model(img, method="polynomial", degree=1, sample_points=pts)
    assert m["samples"]["manual"] and len(m["samples"]["x"]) == len(pts) and m["samples"]["used"].all()


def test_an_excluded_ellipse_keeps_a_nebula_out_of_the_automatic_model():
    img, sky = scene(nebula=2.0)                          # a faint, wide nebula
    ell = [[0.45 * W, 0.5 * H, 0.25 * W, 0.3 * H]]
    c = (slice(140, 160), slice(190, 215))
    lift = []
    for kw in ({}, {"exclude_ellipses": ell}):
        g, _ = model(img, method="automatic", **kw)
        d = g - sky
        lift.append(float(d[c].mean() - np.median(d)))  # how far the model rises into the nebula
    assert lift[0] > 1.5 and lift[1] < 0.5 * lift[0], lift


def test_exclusion_mask_paints_and_erases_in_order():
    p = GradientParams(exclude_ellipses=[[100, 100, 50, 30]], exclude_brush=[[100, 100, 10, 0], [300, 50, 20, 1]])
    m = exclusion_mask(p, 200, 400)
    assert m[100, 70] and not m[100, 100] and m[50, 300] and not m[180, 380]
    half = exclusion_mask(p, 100, 200, 0.5)
    assert half.shape == (100, 200) and half[25, 150]
    assert exclusion_mask(GradientParams(), 10, 10) is None


def test_a_turned_ellipse_is_excluded_along_its_axes():
    from astro_gpu.client.gradientedit import _inside, _local, _world
    e = [200.0, 100.0, 80.0, 20.0, 90.0]                    # turned a quarter: its long axis vertical
    m = exclusion_mask(GradientParams(exclude_ellipses=[e]), 200, 400)
    assert m[160, 200] and not m[100, 270]
    assert _inside(e, 200, 160) and not _inside(e, 270, 100)
    u, v = _local(e, *_world(e, 30.0, -7.0))
    assert abs(u - 30) < 1e-9 and abs(v + 7) < 1e-9


def test_the_grid_leaves_the_survey_emission_out():
    from astro_gpu.stages.gradient import EMISSION_FREE, away_from_emission, grid_points
    from astro_gpu.stages.surveys import emission_level
    img, _sky = scene()
    yy, xx = np.mgrid[0:H, 0:W]
    survey = 20 + 900 * np.exp(-((xx - 200) ** 2 + (yy - 150) ** 2) / (2 * 35.0 ** 2))   # a nebula seen by the survey
    level = emission_level(survey.astype(np.float32))
    assert abs(float(np.percentile(level, 1))) < 1e-6 and abs(float(np.percentile(level, 99.5)) - 1) < 1e-6
    pts, kept, box = grid_points(img, np.ones((H, W), bool), 8, 2.5)
    out, used = away_from_emission(pts, kept, level, box)
    assert used and 0 < out.sum() < kept.sum()
    for (x, y), k in zip(pts, out):
        if k:                                               # every sample kept sits on the survey's sky
            assert level[int(y), int(x)] <= EMISSION_FREE * 1.5
    assert not out[np.argmin(np.hypot(pts[:, 0] - 200, pts[:, 1] - 150))]     # none on the nebula


def test_a_survey_without_emission_leaves_no_sample_out():
    from astro_gpu.stages.gradient import away_from_emission, grid_points
    from astro_gpu.stages.surveys import emission_level
    img, _sky = scene()
    noise = np.random.default_rng(1).normal(100, 1, (H, W)).astype(np.float32)   # a galaxy field: noise only
    pts, kept, box = grid_points(img, np.ones((H, W), bool), 8, 2.5)
    out, used = away_from_emission(pts, kept, emission_level(noise), box)
    assert not used and (out == kept).all()


def test_survey_on_grid_falls_back_and_says_when_none_covers(monkeypatch):
    import pytest

    from astro_gpu.core.stage import StageError
    from astro_gpu.stages import surveys
    seen = []

    def fake(name, wd, w, h, scale, log=print, **k):
        seen.append(name)
        a = np.ones((h, w), np.float32)
        if name == "nsns":
            a[:, : w // 2] = np.nan                         # NSNS covers half the field
        return a
    monkeypatch.setattr(surveys, "fetch_survey", fake)
    img, used = surveys.survey_on_grid(None, 40, 20, 10.0, log=lambda *a: None)
    assert used == "finkbeiner" and seen == ["nsns", "finkbeiner"] and np.isfinite(img).all()
    monkeypatch.setattr(surveys, "fetch_survey", lambda *a, **k: np.full((20, 40), np.nan, np.float32))
    with pytest.raises(StageError):
        surveys.survey_on_grid(None, 40, 20, 10.0, log=lambda *a: None)


def test_every_text_of_the_editor_has_its_french():
    import ast
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "astro_gpu/client"
    cat: dict[str, str] = {}
    for f in sorted((root / "locale/fr").glob("*.json")):
        cat.update(json.loads(f.read_text(encoding="utf-8")))
    texts = [n.args[0].value for n in ast.walk(ast.parse((root / "gradientedit.py").read_text(encoding="utf-8")))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_" and n.args
             and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)]
    missing = sorted({t for t in texts if t not in cat})
    assert not missing, missing


def test_the_block_samples_avoid_the_survey_emission_and_keep_a_faint_nebula():
    from astro_gpu.stages.surveys import emission_level
    img, sky = scene(nebula=4.0)                            # a faint wide nebula the sky rejection does not catch
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W], np.float32)[:, None, None]
    survey = 20 + 900 * np.exp(-((xx - 0.45) ** 2 + (yy - 0.5) ** 2) / 0.03)              # the survey sees it
    level = emission_level(survey.astype(np.float32))
    centre = (slice(130, 170), slice(180, 230))
    err = {}
    for avoid in (False, True):
        m = sky_model_of(img, np.ones((H, W), bool), GradientParams(method="polynomial", avoid_emission=avoid),
                         emission=level)
        g = read_grid(m["grid"], m["info"], H, W)[0]
        err[avoid] = float(np.median(g[centre] - sky[centre]))     # what the model takes of the nebula as sky
        assert bool(m["info"].get("away_from_emission")) == avoid
    assert err[True] < 0.5 * err[False], err
    placed = GradientParams(method="polynomial", degree=1, sample_points=[[40.0, 40.0], [400.0, 40.0], [40.0, 260.0],
                                                                          [400.0, 260.0], [220.0, 150.0]])
    assert not sky_model_of(img, np.ones((H, W), bool), placed, emission=level)["info"].get("away_from_emission")


def test_without_a_survey_the_run_goes_on(monkeypatch, tmp_path):
    from astro_gpu.core.stage import StageError
    from astro_gpu.stages import surveys
    saved, logs = {}, []

    class Ctx:                                               # what the Surveys stage reads and writes
        def input(self, stage, name):
            return tmp_path / "wcs.fits"

        def load_json(self, stage, name):
            return {"width": 40, "height": 20}

        def metric_of(self, stage, key):
            return 2.0

        def save_json(self, name, d):
            saved[name] = d

        def metric(self, k, v):
            saved[k] = v

        def log(self, msg, level="info"):
            logs.append((level, msg))

        def progress(self, *a):
            pass
    monkeypatch.setattr("astropy.io.fits.getheader", lambda p: None)
    monkeypatch.setattr("astropy.wcs.WCS", lambda h: None)

    def none_covers(*a, **k):
        raise StageError("no H-alpha survey covers this field")
    monkeypatch.setattr(surveys, "survey_on_grid", none_covers)
    surveys.Surveys().run(Ctx(), surveys.SurveysParams())
    assert saved["survey.json"]["survey"] == "none" and "covers" in saved["survey.json"]["why"]
    assert any(level == "warning" for level, _m in logs)
