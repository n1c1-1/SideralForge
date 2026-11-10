"""The Deconvolution block: the star shape measured on the stack, its two methods, and its place in the chain."""
import numpy as np
import pytest
import torch

from astro_gpu.core import workflow as wfl
from astro_gpu.stages import deconv_lib as lib
from astro_gpu.stages.deconv import Deconv, DeconvParams

SKY, NOISE = 100.0, 0.5


@pytest.fixture(autouse=True)
def _few_threads():
    """Small transforms: more threads only wait for one another (70 s a test at 8 threads, a few at 2)."""
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


def field(seed=1, size=300, fwhm=4.0, stars=36):
    """A flat sky with blurred stars (3, H, W), two half-stacks with independent noise, and where the stars are."""
    rng = np.random.default_rng(seed)
    s = fwhm / 2.3548
    img = np.zeros((size, size), np.float32)
    grid = np.linspace(40, size - 40, int(stars ** 0.5))
    pts = [(y + rng.uniform(-3, 3), x + rng.uniform(-3, 3)) for y in grid for x in grid]
    yy, xx = np.mgrid[:size, :size]
    fluxes = []
    for y, x in pts:
        flux = rng.uniform(3000, 9000)
        fluxes.append(flux)
        img += (flux / (2 * np.pi * s * s) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s * s))).astype(np.float32)
    a = img + SKY + rng.normal(0, NOISE * 2 ** 0.5, img.shape).astype(np.float32)
    b = img + SKY + rng.normal(0, NOISE * 2 ** 0.5, img.shape).astype(np.float32)
    three = lambda m: np.repeat(m[None], 3, axis=0)
    return three(0.5 * (a + b)), three(a), three(b), np.array(pts), np.array(fluxes)


def width_and_flux(img, pts, r=14):
    """Median width at half height of the stars of plane 1, and each star's flux above the sky in an aperture."""
    widths, fluxes = [], []
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    for y, x in pts:
        y, x = int(round(y)), int(round(x))
        s = img[1, y - r:y + r + 1, x - r:x + r + 1] - SKY
        fluxes.append(float(s[yy ** 2 + xx ** 2 <= r * r].sum()))
        w = np.where(s > 0.2 * s.max(), s, 0)                            # the core: a second moment, not a pixel count
        widths.append(2.3548 * float((w * (yy ** 2 + xx ** 2)).sum() / w.sum() / 2) ** 0.5)
    return float(np.median(widths)), np.array(fluxes)


def test_the_stars_become_narrower_and_keep_their_light():
    full, a, b, pts, _ = field()
    valid, sat = np.ones(full.shape[1:], bool), np.zeros(full.shape[1:], bool)
    st = lib.StackDeconvSettings(tiles_long=1, log=lambda *x, **k: None, iterations=15)
    of, oa, ob, info = lib.run(full, a, b, sat, valid, st, "cpu")
    assert info["applied"] and info["stars"] >= 10
    w0, f0 = width_and_flux(full, pts)
    w1, f1 = width_and_flux(of, pts)
    assert w1 < 0.9 * w0, (w0, w1)
    assert abs(float(np.median(f1 / f0)) - 1) < 0.15, np.median(f1 / f0)     # the deringing adds light (issue #92: +13 %)
    assert oa.shape == full.shape and ob.shape == full.shape and not np.array_equal(oa, a)
    d = (of - full)[1]
    ring = np.ones(d.shape, bool)
    for y, x in pts:
        ring[max(0, int(y) - 12):int(y) + 13, max(0, int(x) - 12):int(x) + 13] = False
    assert abs(float(np.median(d[ring]))) < NOISE                         # the sky stays where it is


def test_a_saturated_core_and_the_pixels_without_data_stay_as_they_were():
    full, a, b, pts, _ = field()
    valid = np.ones(full.shape[1:], bool)
    valid[:20] = False
    sat = np.zeros(full.shape[1:], bool)
    y, x = int(pts[10][0]), int(pts[10][1])
    sat[y - 2:y + 3, x - 2:x + 3] = True
    st = lib.StackDeconvSettings(tiles_long=1, log=lambda *x, **k: None, iterations=6)
    of, *_ = lib.run(full, a, b, sat, valid, st, "cpu")
    assert np.array_equal(of[:, :20], full[:, :20])
    assert np.abs(of[:, y, x] - full[:, y, x]).max() < 1e-3


def test_a_saturated_star_on_a_bright_sky_keeps_no_moat_and_no_pattern():
    full, a, b, pts, _ = field()
    y, x = int(pts[12][0]), int(pts[12][1])
    yy, xx = np.mgrid[:full.shape[1], :full.shape[2]]
    glow = 40 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 60.0 ** 2)).astype(np.float32)
    core = (((yy - y) ** 2 + (xx - x) ** 2) < 9).astype(np.float32) * 5000
    for im in (full, a, b):
        im += glow + core
    sat = core > 0
    st = lib.StackDeconvSettings(tiles_long=1, log=lambda *x, **k: None, iterations=12)
    of, *_ = lib.run(full, a, b, sat, np.ones(full.shape[1:], bool), st, "cpu")
    r = np.hypot(yy - y, xx - x)
    ring = (r > 4) & (r < 16)
    assert float((of[1] - full[1])[ring].min()) > -10 * NOISE            # around a star of 5000: neither dug nor rippled


def test_richardson_lucy_keeps_the_colour_of_a_star():
    full, a, b, pts, _ = field()
    for im in (full, a, b):
        im[0] = SKY + (im[0] - SKY) * 1.4
        im[2] = SKY + (im[2] - SKY) * 0.7
    st = lib.StackDeconvSettings(tiles_long=1, log=lambda *x, **k: None, iterations=12)
    of, *_ = lib.run(full, a, b, np.zeros(full.shape[1:], bool), np.ones(full.shape[1:], bool), st, "cpu")
    y, x = int(round(pts[12][0])), int(round(pts[12][1]))
    core = of[:, y - 1:y + 2, x - 1:x + 2].mean((1, 2)) - SKY
    assert abs(core[0] / core[1] - 1.4) < 0.08 and abs(core[2] / core[1] - 0.7) < 0.06


def test_too_few_stars_leave_the_image_as_it_is():
    rng = np.random.default_rng(0)
    a = (SKY + rng.normal(0, 1, (3, 200, 200))).astype(np.float32)
    b = (SKY + rng.normal(0, 1, (3, 200, 200))).astype(np.float32)
    full = 0.5 * (a + b)
    of, oa, ob, info = lib.run(full, a, b, None, np.ones((200, 200), bool),
                               lib.StackDeconvSettings(log=lambda *x, **k: None), "cpu")
    assert not info["applied"] and of is full and oa is a and ob is b


def test_richardson_lucy_restores_a_blurred_point_and_stays_positive():
    d = torch.full((1, 64, 64), 50.0)
    k = 13
    yy, xx = torch.meshgrid(torch.arange(k) - 6.0, torch.arange(k) - 6.0, indexing="ij")
    K = torch.exp(-(yy ** 2 + xx ** 2) / (2 * 1.7 ** 2))
    K /= K.sum()
    p = torch.zeros(64, 64)
    p[32, 32] = 4000.0
    pad = torch.zeros(64, 64)
    pad[:k, :k] = K
    blurred = torch.fft.irfft2(torch.fft.rfft2(p) * torch.fft.rfft2(torch.roll(pad, (-6, -6), (0, 1))), s=(64, 64))
    u = lib.restore_saturated(d + blurred[None], K, 30, 0.0, 0.5, 1e6)
    assert float(u.min()) > 0 and float(u[0, 32, 32]) > 2.5 * float((d + blurred[None])[0, 32, 32])
    assert abs(float(u.sum() - (d + blurred[None]).sum())) < 0.01 * float((d + blurred[None]).sum())


def test_the_block_comes_before_denoise_and_only_its_method_sets_its_stage():
    full = {"denoise": "pretrained", "stars": "neural", "deconvolution": "rl"}
    chain = wfl.chain(full)
    assert chain.index("gradient") < chain.index("deconv") < chain.index("denoise")
    assert "deconv" not in wfl.chain({"denoise": "pretrained", "stars": "neural"})
    assert wfl.implied(full, "deconv") == {"method": "rl"}
    with pytest.raises(ValueError):
        wfl.check_options({"deconvolution": "wiener"})
    assert wfl.greyed({"deconvolution": "rl"}, image=True)["deconv"] == wfl.NEEDS_HALF_STACKS
    assert Deconv.name == "deconv" and DeconvParams().method == "rl"
    blocks = wfl.from_options(full)
    names = [n["block"] for n in blocks]
    assert names.index("gradient") < names.index("deconv") < names.index("denoise")
    assert wfl.options_of(blocks)["deconvolution"] == "rl"
    assert wfl.node_implied(blocks, "deconv") == {"method": "rl"}
    without = wfl.sync_blocks(blocks, "frames", {**full, "deconvolution": None})
    assert "deconv" not in [n["block"] for n in without]
