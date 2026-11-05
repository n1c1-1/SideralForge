"""Post's simplified sliders (render/sliders.py): tables, position 2 = the base, monotonic, greyed with reasons."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from astro_gpu.render import sliders as S
from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings
from astro_gpu.render.stars import StarSettings

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
SIMPLE_BASES = [None, {"sky": 0.3, "saturation": 0.8, "noise_mix": 0.9}, SimpleSettings(contrast=0.2),
                {"saturation": 0.0}]
STAR_BASES = [None, {"sky": 0.3, "star_strength": 1.4, "blend": "Lighten"}, StarSettings(star_size=1.0),
              {"star_saturation": 0.0}]
POSITIONS = [i / 4 for i in range(17)]


def test_the_eleven_sliders_of_post():
    assert S.KEYS == ("sky", "contrast", "faint", "saturation", "colours", "halpha", "noise", "star_strength",
                      "star_small", "star_size", "star_color")
    # position 2 of every table is the suggested rendering (the defaults)
    s = StarSettings()
    for sl in S.SLIDERS:
        assert len(sl.ends) == 2 and sl.group in (S.NEBULA, S.STARS)
        assert (sl.group == S.STARS) == (sl.needs == "stars")
        for name, t in sl.table.items():
            assert len(t) == 5 and abs(t[2] - getattr(s, name)) < 1e-9, (sl.key, name)
    assert S.BY_KEY["sky"].table["sky"] == (0.10, 0.15, 0.20, 0.26, 0.32)
    assert S.BY_KEY["noise"].table["noise_mix"][:3] == (0.4, 0.7, 1.0)


@pytest.mark.parametrize("kind,bases", [("simple", SIMPLE_BASES), ("combined", STAR_BASES)])
def test_position_2_is_the_base(kind, bases):
    for base in bases:
        b = S.base_settings(kind, base)
        at2 = S.settings_at(kind, base, {k: 2.0 for k in S.KEYS})
        assert at2 == b
        assert S.settings_at(kind, base, {}) == b


def test_default_rendering_follows_the_tables():
    for pos in range(5):
        for sl in S.SLIDERS:
            kind = "simple" if sl.key == "noise" else "combined"
            got = S.settings_at(kind, None, {sl.key: pos})
            for name, t in sl.table.items():
                assert getattr(got, name) == pytest.approx(t[pos]), (sl.key, name, pos)


@pytest.mark.parametrize("kind,bases", [("simple", SIMPLE_BASES), ("combined", STAR_BASES)])
def test_each_slider_is_monotonic(kind, bases):
    offered = [sl for sl in S.SLIDERS if not S.availability(kind, True)[sl.key]]
    assert offered
    for base in bases:
        for sl in offered:
            vals = [S.settings_at(kind, base, {sl.key: p}) for p in POSITIONS]
            moved = False
            for name in sl.table:
                v = [getattr(x, name) for x in vals]
                up = all(b >= a - 1e-12 for a, b in zip(v, v[1:]))
                down = all(b <= a + 1e-12 for a, b in zip(v, v[1:]))      # Small stars: left = fewer stars (#7)
                assert up or down, (kind, base, sl.key, name, v)
                moved |= v[-1] != v[0]
            assert moved, (kind, base, sl.key)              # never frozen, even on a base of 0 (saturation)


def test_steps_and_ratios():
    b = S.settings_at("simple", {"sky": 0.3, "saturation": 1.0}, {"sky": 4, "saturation": 0})
    assert b.sky == pytest.approx(0.42)                                 # a step: +0.12
    assert b.saturation == pytest.approx(1.0 * 0.6 / 1.2)               # a ratio
    half = S.settings_at("simple", None, {"sky": 1.5, "contrast": 3.5})
    assert half.sky == pytest.approx(0.175) and half.contrast == pytest.approx(0.375)
    assert S.settings_at("simple", {"sky": 0.55}, {"sky": 4}).sky == 0.6   # inside the setting's bounds (0.02-0.6)
    assert S.settings_at("simple", None, {"sky": 9}).sky == pytest.approx(0.32)   # positions beyond 0-4 are clamped
    assert S.settings_at("simple", None, {"unknown": 3}) == SimpleSettings()
    with pytest.raises(ValueError):
        S.settings_at("simple", "No such preset", {})
    with pytest.raises(ValueError):
        S.availability("other", True)
    # the Stars block's images have the star settings; the Output block's drop them
    assert isinstance(S.base_settings("combined"), StarSettings)
    assert S.base_settings("simple", StarSettings(sky=0.3, star_size=1.0)) == SimpleSettings(sky=0.3)
    assert S.settings_at("combined", None, {"star_size": 3}).star_size == pytest.approx(0.0)


def test_greyed_sliders_give_their_reasons():
    stars = ("star_strength", "star_small", "star_size", "star_color")
    simple = S.availability("simple", False)
    assert simple["sky"] == simple["contrast"] == simple["saturation"] == ""
    assert simple["noise"] == S.WHY_DENOISE and all(simple[k] == S.WHY_STARS for k in stars)
    assert S.availability("simple", True)["noise"] == ""
    exit_ = S.availability("stacking", True)
    assert exit_["noise"] == S.WHY_EXIT_DENOISE and all(exit_[k] == S.WHY_EXIT_STARS for k in stars)
    starless = S.availability("starless", True)
    assert starless["sky"] == "" and starless["noise"] == S.WHY_WHOLE and all(starless[k] == S.WHY_STARS for k in stars)
    alone = S.availability("stars", True)
    assert all(alone[k] == S.WHY_STARS_ONLY for k in ("sky", "contrast", "saturation", "noise"))
    assert all(alone[k] == "" for k in stars)
    combined = S.availability("combined", True)
    assert {k for k, v in combined.items() if v} == {"noise"} and combined["noise"] == S.WHY_WHOLE
    # a greyed slider's position is ignored
    assert S.settings_at("simple", None, {"noise": 0}, has_denoise=False) == SimpleSettings()
    assert S.settings_at("simple", None, {"noise": 0}, has_denoise=True).noise_mix == pytest.approx(0.4)
    assert S.settings_at("stacking", None, {"star_size": 4, "sky": 3}) == SimpleSettings(sky=0.26)
    assert S.settings_at("stars", None, {"sky": 0, "star_size": 4}) == StarSettings(star_size=1.0)


def test_rendered_sliders_move_the_image():
    rng = np.random.default_rng(3)
    H, W = 160, 240
    neb = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 12)
    neb = (neb - neb.min()) / (neb.max() - neb.min()) * 60
    lin = np.stack([neb, neb * .8, neb * .7]) + 20 + rng.normal(0, 2, (3, H, W)).astype(np.float32)
    eng = SimpleRenderEngine.from_array(lin, device=DEV)
    sky = [float(eng.render(S.settings_at("simple", None, {"sky": p}), 0.5).final.mean()) for p in (0, 1, 2, 3, 4)]
    assert all(b > a for a, b in zip(sky, sky[1:]))
    sat = [eng.render(S.settings_at("simple", None, {"saturation": p}), 0.5).final for p in (0, 4)]
    assert float(sat[0].std(0).mean()) < float(sat[1].std(0).mean())          # paler to more vivid
    base = eng.render(SimpleSettings(), 0.5).final
    assert torch.equal(eng.render(S.settings_at("simple", None, {k: 2 for k in S.KEYS}), 0.5).final, base)
