"""The Stars block (stages/star_removal.py) and the rendering of its two images (render/stars.py): the network takes a
star off and leaves the sky, the two images add up to the image, each part renders, the star sliders move what they
say, and the workflows put the block after the Output."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from astro_gpu.core import workflow as wfl
from astro_gpu.render import sliders as S
from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings, image_stats
from astro_gpu.render.stars import BLENDS, StarSettings, StarsRenderEngine, blend, resize_stars
from astro_gpu.stages import star_removal as sr

DEV = torch.device("cpu")


def moffat(h, w, x0, y0, amp, fwhm=3.0, beta=3.0):
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    a = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    return amp * (1 + ((xx - x0) ** 2 + (yy - y0) ** 2) / a ** 2) ** -beta


@pytest.fixture(scope="module")
def net():
    return sr.network(DEV)


def test_the_network_takes_a_star_off_and_leaves_the_sky(net):
    rng = np.random.default_rng(1)
    h = w = 256
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    sky = 4.0 + 3.0 * np.exp(-((xx - 90) ** 2 + (yy - 160) ** 2) / (2 * 50.0 ** 2))       # a smooth nebula, noise units
    img = np.repeat(sky[None], 3, 0) + 0.1 * rng.standard_normal((3, h, w)).astype(np.float32)   # denoised
    star = moffat(h, w, 150.3, 100.6, 300.0)
    out = sr.apply(net, img + star[None], DEV)
    assert out.shape == img.shape and out.dtype == np.float32
    box = np.s_[:, 90:112, 140:162]
    left = float((out[box] - img[box]).sum()) / float((3 * star[90:112, 140:162]).sum())
    assert left < 0.1, f"{left:.0%} of the star left"
    far = np.s_[:, 180:240, 20:80]                    # the nebula far from the star: unchanged
    assert abs(float(np.median(out[far] - img[far]))) < 0.3


def test_a_tiny_image_is_padded_to_one_tile(net):
    out = sr.apply(net, np.ones((3, 40, 70), np.float32), DEV)
    assert out.shape == (3, 40, 70) and np.isfinite(out).all()


def _folder(tmp_path, h=64, w=96):
    """A star_removal folder: a sky, a nebula and two stars, the starless image and the stars."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    starless = np.stack([np.full((h, w), 2.0, np.float32) + 30 * np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / 300)] * 3)
    starless += 0.5 * rng.standard_normal(starless.shape).astype(np.float32)
    stars = np.stack([moffat(h, w, 70, 20, 400.0), moffat(h, w, 50, 50, 80.0)]).sum(0)
    stars = np.repeat(stars[None], 3, 0).astype(np.float32)
    whole = starless + stars
    valid = np.ones((h, w), bool)
    np.save(tmp_path / "starless.npy", starless)
    np.save(tmp_path / "stars.npy", stars)
    np.save(tmp_path / "valid.npy", valid)
    prep = {"crop": [0, w, 0, h], "M": np.eye(3).tolist(), "drizzle_scale": 1.0, "unit": 1.0,
            "background": [0, 0, 0], "stats_full": image_stats(whole, valid)}
    (tmp_path / "prep_stars.json").write_text(json.dumps(prep))
    return whole, starless, stars


def test_each_part_renders_and_the_recombined_image_is_close_to_the_whole(tmp_path):
    whole, *_ = _folder(tmp_path)
    eng = StarsRenderEngine(tmp_path, device="cpu")
    s = StarSettings()
    parts = {p: eng.render(s, 1.0, part=p).final.numpy() for p in ("starless", "stars", "combined")}
    for a in parts.values():
        assert a.shape == (3, 64, 96) and 0 <= a.min() and a.max() <= 1
    # the stars on black; the starless image without them; recombined, the whole image's look (the same curve)
    assert parts["stars"][:, 0:5, 0:5].max() < 0.02
    assert parts["stars"][:, 20, 70].mean() > 0.9 and parts["starless"][:, 20, 70].mean() < 0.4
    ref = SimpleRenderEngine.from_array(whole)
    ref.prep["stats"] = image_stats(whole, np.ones(whole.shape[1:], bool))
    full = ref.render(SimpleSettings(), 1.0).final.numpy()
    assert np.abs(parts["combined"] - full).mean() < 0.03


def test_the_star_sliders_move_what_they_say(tmp_path):
    _folder(tmp_path)
    eng = StarsRenderEngine(tmp_path, device="cpu")

    def stars(**kw):
        return eng.render(StarSettings(**kw), 1.0, part="stars").final.numpy()
    base = stars()
    assert stars(star_strength=0.5).sum() < base.sum() < stars(star_strength=2.0).sum()
    assert stars(star_size=-1.0).sum() < base.sum() < stars(star_size=1.0).sum()
    grey = stars(star_saturation=0.0)
    assert np.abs(grey - grey.mean(0, keepdims=True)).max() < 1e-5          # no colour left


def test_blends_and_the_size_change():
    a = torch.full((3, 4, 4), 0.5)
    b = torch.zeros((3, 4, 4))
    b[:, 1, 1] = 0.5
    assert torch.allclose(blend(a, b, "Screen")[:, 1, 1], torch.tensor(0.75))
    assert torch.allclose(blend(a, b, BLENDS[1])[:, 1, 1], torch.tensor(1.0))
    assert torch.allclose(blend(a, b, "Lighten")[:, 1, 1], torch.tensor(0.5))
    assert torch.equal(blend(a, b, "Screen")[:, 0, 0], a[:, 0, 0])            # black stars change nothing
    y = torch.zeros((3, 9, 9))
    y[:, 4, 4] = 1.0
    assert resize_stars(y, 1.0).sum() == 27 and resize_stars(y, -1.0).sum() == 0
    assert 3 < resize_stars(y, 0.5).sum() < 27                                 # half a pixel: half way


def test_the_sliders_of_each_kind():
    ok = S.availability("combined", True)
    assert not any(ok[k] for k in ("sky", "contrast", "saturation", "star_strength", "star_size", "star_color"))
    assert ok["noise"] == S.WHY_WHOLE
    stars = S.availability("stars", True)
    assert stars["sky"] == S.WHY_STARS_ONLY and not stars["star_size"]
    simple = S.availability("simple", True)
    assert simple["star_strength"] == S.WHY_STARS and not simple["noise"]
    assert S.availability("stacking", False)["star_color"] == S.WHY_EXIT_STARS
    st = S.settings_at("combined", None, {"star_strength": 0, "star_size": 4})
    assert isinstance(st, StarSettings) and st.star_strength == pytest.approx(0.35) and st.star_size == 2.0
    assert type(S.settings_at("simple", None, {"star_size": 0})) is SimpleSettings      # greyed: ignored
    with pytest.raises(ValueError):
        S.availability("engine", True)


def test_the_stars_block_comes_after_the_output_and_reads_the_old_options():
    full = {"denoise": "pretrained", "stars": "neural"}
    assert wfl.chain(full)[-2:] == ["render_simple", "star_removal"]
    for old in ("psf", "psf_neural"):
        assert wfl.check_options({**full, "stars": old})["stars"] == "neural"
    assert "star_removal" not in wfl.chain({"denoise": "pretrained"})
    with pytest.raises(ValueError):
        wfl.check_options({"stars": "neural"})                                # Stars needs Denoise
    o, notes = wfl.set_option({"denoise": None}, "stars", "neural")
    assert o["denoise"] == "pretrained" and notes == [wfl.NOTE_DENOISE_ADDED]
    c = wfl.chains()
    assert [b for b, _ in c.blocks_in(wfl.chain(full))][-2:] == ["stars", "output"]   # drawn before the Output
