"""Post's simplified sliders (render/sliders.py): tables, position 2 = the base, monotonic, greyed with reasons."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

from astro_gpu.render import sliders as S
from astro_gpu.render.settings import PRESETS, RenderSettings
from astro_gpu.render.simple import SimpleSettings

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
ENGINE_BASES = [None, "Artistic", "Prominent stars", {"sky_level": 30.0, "star_glow": 0.0, "faint_smooth": 12.0},
                RenderSettings(chroma=1.2, d10=6.0)]
SIMPLE_BASES = [None, {"sky": 0.3, "saturation": 0.8, "noise_mix": 0.9}, SimpleSettings(contrast=0.2)]
POSITIONS = [i / 4 for i in range(17)]


def test_the_nine_sliders_of_post():
    assert S.KEYS == ("sky", "contrast", "saturation", "detail", "noise", "stars_number", "star_size", "star_color",
                      "star_halo")
    assert set(S.MOCKUP_KEYS) == set(S.KEYS)
    # position 2 of every table is the suggested rendering (the defaults)
    r, s = RenderSettings(), SimpleSettings()
    for sl in S.SLIDERS:
        assert len(sl.ends) == 2 and sl.group in (S.NEBULA, S.STARS)
        for name, t in sl.engine.items():
            assert len(t) == 5 and abs(t[2] - getattr(r, name)) < 1e-9, (sl.key, name)
        for name, t in sl.simple.items():
            assert len(t) == 5 and abs(t[2] - getattr(s, name)) < 1e-9, (sl.key, name)
    assert S.BY_KEY["sky"].engine["sky_level"] == (12, 18, 24, 30, 36)
    assert S.BY_KEY["noise"].simple["noise_mix"][:3] == (0.4, 0.7, 1.0)


@pytest.mark.parametrize("kind,bases", [("engine", ENGINE_BASES), ("simple", SIMPLE_BASES)])
def test_position_2_is_the_base(kind, bases):
    for base in bases:
        b = S.base_settings(kind, base)
        at2 = S.settings_at(kind, base, {k: 2.0 for k in S.KEYS})
        assert at2 == b
        assert S.settings_at(kind, base, {}) == b


def test_default_rendering_follows_the_tables():
    for pos in range(5):
        for sl in S.SLIDERS:
            got = S.settings_at("engine", None, {sl.key: pos})
            for name, t in sl.engine.items():
                assert getattr(got, name) == pytest.approx(t[pos]), (sl.key, name, pos)
            if sl.simple:
                got = S.settings_at("simple", None, {sl.key: pos})
                for name, t in sl.simple.items():
                    assert getattr(got, name) == pytest.approx(t[pos]), (sl.key, name, pos)


@pytest.mark.parametrize("kind,bases", [("engine", ENGINE_BASES), ("simple", SIMPLE_BASES)])
def test_each_slider_is_monotonic(kind, bases):
    for base in bases:
        for sl in S.SLIDERS:
            table = sl.engine if kind == "engine" else sl.simple
            if not table:
                continue
            vals = [S.settings_at(kind, base, {sl.key: p}) for p in POSITIONS]
            moved = False
            for name in table:
                v = [getattr(x, name) for x in vals]
                assert all(b >= a - 1e-12 for a, b in zip(v, v[1:])), (kind, base, sl.key, name, v)
                moved |= v[-1] > v[0]
            assert moved, (kind, base, sl.key)              # never frozen, even on a base of 0 (star_glow)


def test_steps_and_ratios():
    b = S.settings_at("engine", {"sky_level": 30.0, "chroma": 1.1}, {"sky": 4, "saturation": 0})
    assert b.sky_level == pytest.approx(42.0)                           # a step: +12
    assert b.chroma == pytest.approx(1.1 * 0.45 / 0.88)                 # a ratio
    half = S.settings_at("engine", None, {"sky": 1.5, "contrast": 3.5})
    assert half.sky_level == pytest.approx(21.0) and half.lc_medium == pytest.approx((1.55 + 1.85) / 2)
    top = S.settings_at("engine", {"sky_level": 78.0}, {"sky": 4})      # inside the setting's bounds (0-80)
    assert top.sky_level == 80.0
    assert S.settings_at("engine", None, {"sky": 9}).sky_level == 36.0  # positions beyond 0-4 are clamped
    assert S.settings_at("engine", None, {"unknown": 3}) == RenderSettings()
    with pytest.raises(ValueError):
        S.settings_at("engine", "No such preset", {})
    with pytest.raises(ValueError):
        S.availability("other", True)
    assert S.base_settings("engine", "Artistic").sky_level == PRESETS["Artistic"]["sky_level"]


def test_the_sky_slider_never_falls_in_the_dead_zone_of_the_tone_curve():
    """The tone curve keeps the dark level first: when the sky lies close above the darkest areas, sky levels below a
    floor give one curve (heart_a7s: 34.6 /255 for anything asked below, the faithful default 24 included). Every
    position of the slider must still move the sky, and position 2 must keep the base's curve exactly."""
    from astro_gpu.render.engine import fit_tone_curve, tone_curve
    # the statistics of heart_a7s (P7, Full, faithful and Artistic at 1/4): dark and sky percentiles, white
    for base, (p_dark, p_sky, white) in ((None, (1.6264, 2.9055, 14.816)), ("Artistic", (1.4968, 2.8013, 18.741))):
        sky = []
        for pos in POSITIONS:
            s = S.settings_at("engine", base, {"sky": pos})
            lo, b, w = fit_tone_curve(p_dark, p_sky, white, s.dark_level / 255, s.sky_level / 255, s.toe)
            sky.append(float(tone_curve(p_sky, lo, b, w, s.toe)))
            if pos == 2:
                b0 = S.base_settings("engine", base)
                assert (lo, b, w) == fit_tone_curve(p_dark, p_sky, white, b0.dark_level / 255, b0.sky_level / 255,
                                                    b0.toe)
        assert all(b > a + 1e-4 for a, b in zip(sky, sky[1:])), (base, [round(255 * v, 1) for v in sky])
    # the levels the curve cannot reach: with the sky level alone the slider was dead from 0 to 2 here
    lo, b, w = fit_tone_curve(1.6264, 2.9055, 14.816, 10 / 255, 12 / 255, 0.015)
    assert (lo, b, w) == fit_tone_curve(1.6264, 2.9055, 14.816, 10 / 255, 24 / 255, 0.015) and b == 1.0


def test_greyed_sliders_give_their_reasons():
    eng = S.availability("engine", True)
    assert all(v == "" for v in eng.values())
    simple = S.availability("simple", False)
    assert simple["sky"] == simple["contrast"] == simple["saturation"] == ""
    assert simple["detail"] == S.WHY_DETAIL and simple["noise"] == S.WHY_DENOISE
    assert all(simple[k] == S.WHY_STARS for k in ("stars_number", "star_size", "star_color", "star_halo"))
    assert S.availability("simple", True)["noise"] == ""
    exit_ = S.availability("stacking", False)
    assert exit_["noise"] == S.WHY_EXIT_DENOISE and exit_["detail"] == exit_["star_size"] == S.WHY_EXIT_STARS
    # a greyed slider's position is ignored
    assert S.settings_at("simple", None, {"noise": 0}, has_denoise=False) == SimpleSettings()
    assert S.settings_at("simple", None, {"noise": 0}, has_denoise=True).noise_mix == pytest.approx(0.4)
    assert S.settings_at("stacking", None, {"star_size": 4, "sky": 3}) == SimpleSettings(sky=0.26)


def test_position_text():
    assert S.position_text(2) == "suggested" and S.position_text(2.7) == "+0.7" and S.position_text(0.8) == "−1.2"


@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    """A small render_prep folder (v1 layout)."""
    d = tmp_path_factory.mktemp("prep")
    rng = np.random.default_rng(3)
    H, W = 160, 240
    neb = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 12) * 60
    astro = np.stack([neb + 5, neb * .45 + 5, neb * .25 + 5]) + rng.normal(0, .5, (3, H, W)).astype(np.float32)
    np.save(d / "astro_prep.npy", astro.astype(np.float32))
    np.save(d / "valid.npy", np.ones((H, W), bool))
    n = 200
    g = rng.uniform(6, 19, n)
    f = 10 ** (-0.4 * (g - 10)) * 2e4
    flux = np.stack([f * .9, f, f * .8]).astype(np.float32)
    np.savez(d / "stars.npz", x=rng.uniform(0, W, n), y=rng.uniform(0, H, n), g=g, flux=flux, pred=flux,
             glow_flux=flux)
    r = np.arange(0, 61, 0.25)
    np.savez(d / "profile.npz", r=r, prof=np.exp(-0.5 * (r / 1.1) ** 2) + 1e-4 * np.exp(-r / 8))
    (d / "prep.json").write_text(json.dumps({"crop": [2, W - 2, 2, H - 2], "alpha": 0.87, "vh": [1.0, 0.45, 0.25],
                                             "M": np.eye(3).tolist(), "drizzle_scale": 1.5}))
    return d


def test_rendered_sliders_move_the_image(prep):
    from astro_gpu.render.engine import RenderEngine
    eng = RenderEngine(prep, device=DEV)
    base = {"white_mult": 2.0}                  # headroom above the sky of this small scene
    sky = [float(eng.render(S.settings_at("engine", base, {"sky": p}), 0.5).final.mean()) for p in (0, 1, 2, 3, 4)]
    assert all(b > a for a, b in zip(sky, sky[1:]))
    size = [float(eng.render(S.settings_at("engine", None, {"star_size": p}), 0.5).stars.sum()) for p in (0, 2, 4)]
    assert size[0] < size[1] < size[2]
    base = eng.render(RenderSettings(), 0.5).final
    assert torch.equal(eng.render(S.settings_at("engine", None, {k: 2 for k in S.KEYS}), 0.5).final, base)
