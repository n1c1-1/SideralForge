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
from astro_gpu.render.stars import BLENDS, StarSettings, StarsRenderEngine, _morph, blend, resize_stars
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


def test_the_channel_attention_averages_over_a_window_only_when_applied():
    """Issue #45: applied to more than the training size, the attention's average is local (Test-time Local Converter);
    within the window, and in training, it is the average over the whole tile, as the network was trained."""
    pool = sr._LocalPool(16)
    x = torch.zeros(1, 2, 64, 64)
    x[:, :, :, 32:] = 1.0
    pool.eval()
    local = pool(x)
    assert local.shape == x.shape
    assert local[0, 0, 32, 2] == pytest.approx(0.0) and local[0, 0, 32, 61] == pytest.approx(1.0)
    assert local[0, 0, 32, 32] == pytest.approx(0.5, abs=0.07)                      # an even window: half a pixel off
    assert pool(x[:, :, 24:40, 24:40]).shape == (1, 2, 1, 1)                        # within the window: the whole tile
    pool.train()
    assert pool(x).shape == (1, 2, 1, 1) and pool(x)[0, 0, 0, 0] == pytest.approx(0.5)


def test_the_local_average_is_exact_in_half_precision():
    """Sums over a big tile in bfloat16 would lose the small values; the average of a constant stays the constant."""
    pool = sr._LocalPool(32).eval()
    x = torch.full((1, 1, 512, 512), 3.0, dtype=torch.bfloat16)
    assert torch.allclose(pool(x).float(), torch.full((1, 1, 512, 512), 3.0), rtol=1e-3)


def test_the_window_follows_each_level_of_the_network():
    """The window is the training size in the pixels of each level, and the bundled network's weights still load."""
    net = sr.NAFNet()
    pools = [net.encs[0][0].sca[0], net.encs[3][0].sca[0], net.mid[0].sca[0], net.decs[0][0].sca[0], net.decs[3][0].sca[0]]
    assert [p.window for p in pools] == [sr.WINDOW, sr.WINDOW // 8, sr.WINDOW // 16, sr.WINDOW // 8, sr.WINDOW]
    assert not any("sca.0" in k for k in net.state_dict())                          # no weight added: the checkpoint loads


class _Tiles(torch.nn.Module):
    """Stands for the network: the input unchanged, the tiles it was given noted."""

    def __init__(self):
        super().__init__()
        self.sizes = []

    def forward(self, x):
        self.sizes.append(tuple(x.shape))
        return x


def test_a_large_image_goes_in_tiles_larger_than_the_largest_stars():
    """Issue #45: the tiles hold the halo of the largest stars with their core; the same memory at a time; the result
    is the image when the network changes nothing (the tiles add up)."""
    fake = _Tiles()
    img = np.random.default_rng(3).normal(0.0, 1.0, (3, 1300, 1500)).astype(np.float32)
    out = sr.apply(fake, img, DEV)
    assert {s[2:] for s in fake.sizes} == {(sr.TILE, sr.TILE)} and {s[0] for s in fake.sizes} == {1}
    assert sr.TILE > 280                                                            # the largest halos measured (M45)
    assert np.allclose(out, img, atol=1e-3)


def test_a_small_image_keeps_the_tiles_of_the_training_size():
    fake = _Tiles()
    sr.apply(fake, np.ones((3, 600, 1500), np.float32), DEV)
    assert {s[2:] for s in fake.sizes} == {(sr.NET_TILE, sr.NET_TILE)}


def test_a_small_card_takes_smaller_tiles_on_a_large_image(monkeypatch):
    monkeypatch.setattr(sr.devices, "free_memory", lambda d: 600 * 2 ** 20)
    fake = _Tiles()
    out = sr.apply(fake, np.ones((3, 1300, 1500), np.float32), DEV)
    assert all(s[2] < sr.TILE for s in fake.sizes) and np.allclose(out, 1.0, atol=1e-3)


def test_without_half_stacks_the_noise_unit_is_the_spread_of_the_sky():
    """Issue #55: a linear Image has no half-stack: the unit is read on its own sky, the stars and the nebula above the
    sky do not widen it, the uncovered pixels do not count; an image without noise keeps a defined unit."""
    rng = np.random.default_rng(0)
    lin = 3.0 * rng.standard_normal((3, 200, 200)).astype(np.float32)
    lin[:, 60:90, 60:90] += 500.0
    shown = np.ones((200, 200), bool)
    shown[:20] = False
    lin[:, :20] = 1000.0 * rng.standard_normal((3, 20, 200)).astype(np.float32)
    assert sr.sky_unit(lin, shown) == pytest.approx(3.0, rel=0.1)
    assert sr.sky_unit(np.zeros((3, 50, 50), np.float32), np.ones((50, 50), bool)) == 1.0


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
    assert stars(star_size=-2.0).sum() < base.sum() < stars(star_size=0.0).sum()       # -1 by default
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
    assert resize_stars(y, 1.0).sum() == 15 and resize_stars(y, -1.0).sum() < 0.01  # a cross: round, not a square
    assert resize_stars(y, 2.0)[0].bool().sum() == 21                                    # then a square: an octagon
    assert 3 < resize_stars(y, 0.5).sum() < 15                                 # half a pixel: half way


def test_smaller_stars_leave_the_grain_of_a_dense_field_as_it_is():
    """Issue #18: the stars shrink through a star mask (Photoshop's Minimum through a selection of the stars,
    PixInsight's MorphologicalTransformation through a star mask): the faint grain between the stars of a dense
    field, above STAR_LEVEL but within its noise, is not eroded into a web of filaments."""
    g = torch.Generator().manual_seed(3)
    y = (0.04 + 0.03 * torch.randn((1, 96, 96), generator=g)).clamp(0, 1).repeat(3, 1, 1)
    yy, xx = torch.meshgrid(torch.arange(96.0), torch.arange(96.0), indexing="ij")
    star = 0.9 * torch.exp(-((yy - 48) ** 2 + (xx - 48) ** 2) / (2 * 2.0 ** 2))
    y = (y + star).clamp(0, 1)
    out = resize_stars(y, -1.0)
    far = ((yy - 48) ** 2 + (xx - 48) ** 2) > 12 ** 2
    moved = (out - y).abs()[:, far].mean() / (_morph(y, 1, False) - y).abs()[:, far].mean()
    assert moved < 0.1, moved                                                  # the grain as it was: under a tenth of
    #                                                                            what the erosion alone takes (#30: 6 %)
    assert out[:, 46:51, 46:51].sum() < 0.85 * y[:, 46:51, 46:51].sum()        # the star smaller


def test_a_wider_star_shrinks_whole_without_a_ring():
    """Issue #30: the sky of the star mask is measured with the stars clipped out, so a wider star's wings stand above
    it as its core does: the star shrinks whole, its radial profile still falls outward (no ring)."""
    yy, xx = torch.meshgrid(torch.arange(80.0), torch.arange(80.0), indexing="ij")
    r2 = (yy - 40) ** 2 + (xx - 40) ** 2
    y = (0.03 + 0.95 * torch.exp(-r2 / (2 * 3.0 ** 2))).clamp(0, 1).repeat(3, 1, 1)       # FWHM 7 px
    out = resize_stars(y, -2.0)[0]
    rr = r2.sqrt().round().long()
    prof = torch.stack([out[rr == k].mean() for k in range(0, 20)])
    assert torch.all(prof[1:] <= prof[:-1] + 1e-3), prof                       # falls outward: no ring
    assert (out - 0.03).sum() < 0.85 * (y[0] - 0.03).sum()                      # and it is smaller


def test_the_sliders_of_each_kind():
    ok = S.availability("combined", True)
    assert not any(ok[k] for k in ("sky", "contrast", "saturation", "star_strength", "star_small", "star_size",
                                   "star_color"))
    assert ok["noise"] == S.WHY_WHOLE
    stars = S.availability("stars", True)
    assert stars["sky"] == S.WHY_STARS_ONLY and not stars["star_size"]
    simple = S.availability("simple", True)
    assert simple["star_strength"] == S.WHY_STARS and not simple["noise"]
    assert S.availability("stacking", False)["star_color"] == S.WHY_EXIT_STARS
    st = S.settings_at("combined", None, {"star_strength": 0, "star_size": 4})
    assert isinstance(st, StarSettings) and st.star_strength == pytest.approx(0.25) and st.star_size == 1.0
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


def test_the_small_stars_go_and_the_large_ones_stay_exactly():
    from astro_gpu.render.stars import fade_small_stars, measure_fwhm
    y = torch.zeros(3, 60, 60)
    y[:, 10, 10] = 0.9                                                   # a small star (one pixel)
    yy, xx = torch.meshgrid(torch.arange(60.0), torch.arange(60.0), indexing="ij")
    y += 0.95 * torch.exp(-((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * 3.0 ** 2))[None]      # a large one
    z = fade_small_stars(y, 1.0, 2.0)
    assert float(z[0, 10, 10]) < 0.01                                    # the small star gone
    assert float((z - y)[:, 34:47, 34:47].abs().max()) < 0.01           # the large one as it was
    half = fade_small_stars(y, 0.5, 2.0)
    assert abs(float(half[0, 10, 10]) - 0.45) < 0.01                    # half way
    assert torch.equal(fade_small_stars(y, 0.0, 2.0), y)
    rng = np.random.default_rng(2)
    img = np.zeros((300, 300), np.float32)
    gy, gx = np.mgrid[0:300, 0:300]
    for _ in range(200):
        cy, cx, a = rng.uniform(10, 290), rng.uniform(10, 290), rng.uniform(0.1, 1)
        img += a * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * 2.0 ** 2))
    assert abs(measure_fwhm(torch.from_numpy(np.stack([img] * 3))) - 2.355 * 2.0) < 0.35   # the width at half height
