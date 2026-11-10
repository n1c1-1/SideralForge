"""The Synthetic stars block: the instrument's shape, the detection, the two methods, and its row in Post."""
import numpy as np

from astro_gpu.client import postengine as pe
from astro_gpu.core import workflow as wfl
from astro_gpu.stages import synth_stars_lib as lib


def small_instrument(**kw):
    return lib.Instrument(**{**dict(diameter_mm=150, focal_mm=750, pixel_um=3.72, seeing_px=2.5), **kw})


def test_a_spider_makes_spikes_and_a_refractor_makes_none():
    R = 90
    newton = lib.theoretical_shape(small_instrument(obstruction=0.33, spider_arms=4, spider_mm=1.5), R)
    lens = lib.theoretical_shape(small_instrument(), R)
    assert np.allclose(newton.sum((1, 2)), 1, atol=1e-4) and newton.shape == (3, 2 * R + 1, 2 * R + 1)
    g = newton[1]
    assert g[R, R + 50] > 5 * g[R + 35, R + 35]                        # along the arms, against between them
    assert lens[1][R, R + 50] < 1.5 * lens[1][R + 35, R + 35]
    turned = lib.theoretical_shape(small_instrument(obstruction=0.33, spider_arms=4, spider_mm=1.5, angle_deg=45), R)
    assert turned[1][R + 35, R + 35] > 5 * turned[1][R, R + 50]


def stars_alone(ring=False, n=49, size=640):
    """A stars-alone image (3, H, W) with stars on a grid; ``ring``: defocused, a hollow disc."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[-16:17, -16:17]
    rr = np.hypot(yy, xx)
    stamp = (np.exp(-((rr - 6) ** 2) / 2) if ring else np.exp(-(rr ** 2) / (2 * 1.3 ** 2))).astype(np.float32)
    stamp /= stamp.sum()
    s = rng.normal(0, 0.05, (3, size, size)).astype(np.float32)
    pts = [(y, x) for y in np.linspace(50, size - 51, int(n ** 0.5)).astype(int)
           for x in np.linspace(50, size - 51, int(n ** 0.5)).astype(int)]
    fl = rng.uniform(800, 9000, len(pts))
    fl[0] = 30000.0                                                    # the brightest: the others are not saturated
    for (y, x), f in zip(pts, fl):
        s[:, y - 16:y + 17, x - 16:x + 17] += f * stamp * np.array([1.2, 1.0, 0.8], np.float32)[:, None, None]
    return s, np.ones((size, size), bool), np.array(pts), fl


def test_a_defocused_ring_is_one_star_and_is_drawn_narrow_with_its_light():
    s, valid, pts, fl = stars_alone(ring=True)
    L = s.mean(0)
    ys, xs, _p, _T, size = lib.detect(L, valid)
    assert size > 3 and len(ys) == len(pts)
    drawn, info = lib.from_the_instrument(s, valid, small_instrument(), None)
    assert info["stars_drawn"] == len(pts)
    y, x = pts[24]
    a = drawn[1, y - 16:y + 17, x - 16:x + 17]
    assert a.argmax() in range(16 * 33 + 13, 16 * 33 + 20)              # the light at the centre: not a ring any more
    assert abs(float(a.sum()) / fl[24] - 1) < 0.15


def test_the_image_shape_keeps_each_star_in_its_place_and_colour():
    s, valid, pts, fl = stars_alone()
    drawn, info = lib.from_the_image(s, valid)
    assert info["stars_drawn"] + info["stars_kept"] == len(pts)
    y, x = pts[6]
    for c, k in ((0, 1.2), (2, 0.8)):
        assert abs(float(drawn[c, y - 12:y + 13, x - 12:x + 13].sum()) / float(drawn[1, y - 12:y + 13, x - 12:x + 13].sum()) - k) < 0.05


def test_the_saturated_stars_take_their_light_from_the_catalogue():
    fluxes = np.array([[120.0, 100.0, 80.0]] * 12 + [[50.0, 50.0, 50.0]]).T
    ys = np.arange(13) * 40.0
    xs = np.arange(13) * 40.0
    gaia = {"x": xs + 1, "y": ys, "g": np.array([15.0] * 12 + [10.0]), "bp": np.full(13, 11.0), "rp": np.full(13, 10.0)}
    sat = np.array([False] * 12 + [True])
    out, n = lib.saturated_fluxes(ys, xs, sat, fluxes, gaia)
    assert n == 1 and abs(out[1, 12] / 10000 - 1) < 0.01 and abs(out[0, 12] / out[1, 12] - 1.2) < 0.01
    no_colour = {**gaia, "bp": np.full(13, np.nan)}                     # the offline extract
    out, n = lib.saturated_fluxes(ys, xs, sat, fluxes, no_colour)
    assert n == 1 and abs(out[1, 12] / 10000 - 1) < 0.01 and abs(out[0, 12] / out[1, 12] - 1.2) < 0.01


def test_the_block_comes_after_the_stars_and_says_its_method():
    full = {"denoise": "pretrained", "stars": "neural"}
    blocks = wfl.from_options(full)
    assert wfl.palette_problem(blocks, "frames", "synth") is None
    assert wfl.palette_problem(wfl.from_options({"denoise": "pretrained"}), "frames", "synth") is not None
    with_it = blocks + [wfl.node("synth", "instrument")]
    assert "synth_stars" in wfl.sequence_chain(with_it, "frames")
    assert wfl.node_implied(with_it, "synth_stars") == {"method": "instrument"}
    assert "synth_stars" not in wfl.sequence_chain(blocks, "frames")


def test_the_row_of_post_opens_closed_with_the_image_shape_and_open_with_the_instrument():
    group = pe.new_workflow_layer(1, "Full", part="combined")
    layers = pe.with_synth_rows([group], {1: "measured"})
    row = next(x for x in layers if x.get("filter") == "synth")
    assert row["target"] == group["id"] and row["sub"] == "stars" and row["mask"]["parts"][0]["strokes"] == []
    assert pe.with_synth_rows(layers, {1: "measured"}) == layers          # once
    open_ = next(x for x in pe.with_synth_rows([pe.new_workflow_layer(2, "F", part="combined")], {2: "instrument"})
                 if x.get("filter") == "synth")
    assert open_["mask"] is None
    assert [x for x in pe.with_synth_rows(layers, {}) if x.get("filter") == "synth"] == []
    assert pe.source_id(dict(group, part="synth")) == ("synth", 1) and pe.part_sliders({"star_size": 1, "noise": 2}, "synth") == {"star_size": 1}
