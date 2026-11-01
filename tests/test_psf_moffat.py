"""Star FWHM by Moffat fit (astro_gpu/core/psf_moffat.py, stacking v2 point 3): synthetic stars of known FWHM on a
plane, on the samples of one colour of a CFA frame and on a stack; elongated stars; the stars left out (saturated,
close pairs); against the super-pixel measure of register.py; and, on the real heart_a7s frames (slow), the width the
fit reads."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

from astro_gpu.core.psf_moffat import MIN_STARS, find_stars, measure_psf, measure_psf_rgb


def moffat_field(fwhm, beta, H=240, W=240, step=24, seed=0, noise=5.0, bkg=500.0, roundness=1.0, angle=0.0):
    """A plane of point-sampled Moffat stars (value at the pixel centre) of known FWHM and beta on a grid, sub-pixel
    centres, gaussian noise. Returns (image, true positions brightest first, amplitudes)."""
    rng = np.random.default_rng(seed)
    gy, gx = np.mgrid[step:H - step + 1:step, step:W - step + 1:step]
    xy = np.c_[gx.ravel(), gy.ravel()] + rng.uniform(-3, 3, (gx.size, 2))
    amp = rng.uniform(2000, 20000, len(xy))
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    a_maj, a_min = alpha / np.sqrt(roundness), alpha * np.sqrt(roundness)       # same area as the round star
    t = np.radians(angle)
    img = np.full((H, W), bkg)
    r = 15
    for (x0, y0), A in zip(xy, amp):
        xi, yi = int(round(x0)), int(round(y0))
        yy, xx = np.mgrid[yi - r:yi + r + 1, xi - r:xi + r + 1].astype(float)
        X, Y = xx - x0, yy - y0
        u = X * np.cos(t) + Y * np.sin(t)
        v = -X * np.sin(t) + Y * np.cos(t)
        img[yi - r:yi + r + 1, xi - r:xi + r + 1] += A * (1 + (u / a_maj) ** 2 + (v / a_min) ** 2) ** (-beta)
    img += rng.normal(0, noise, img.shape)
    order = np.argsort(-amp)
    return img, xy[order], amp[order]


def _jitter(xy, sd=0.5, seed=5):
    return xy + np.random.default_rng(seed).normal(0, sd, xy.shape)


@pytest.mark.parametrize("fwhm,beta", [(1.3, 3.0), (1.8, 2.5), (1.8, 4.5), (3.0, 3.0), (5.0, 2.5)])
def test_known_fwhm_is_recovered_on_a_plane(fwhm, beta):
    img, xy, _ = moffat_field(fwhm, beta, step=max(24, int(6 * fwhm)), H=300, W=300)   # isolated: 2 windows apart
    r = measure_psf(img, _jitter(xy))
    assert r.ok and r.n >= 0.8 * len(xy)
    assert abs(r.fwhm - fwhm) < 0.02 * fwhm, r
    assert abs(r.beta - beta) < 0.1 * beta, r
    assert r.roundness > 0.97 and r.eccentricity < 0.25
    assert r.channel == ""


@pytest.mark.parametrize("cfa", ["RGGB", "GRBG"])
@pytest.mark.parametrize("fwhm", [1.8, 3.0])
def test_one_colour_of_a_cfa_frame_at_native_resolution(cfa, fwhm):
    """The red and blue samples are one pixel in four, the green one in two: each colour, fitted on its own samples at
    their positions, reads the true width."""
    img, xy, _ = moffat_field(fwhm, 3.0, seed=1)
    for c in "RGB":
        r = measure_psf(img, _jitter(xy), cfa=cfa, channel=c)
        assert r.ok and r.channel == c
        assert abs(r.fwhm - fwhm) < 0.03 * fwhm, (c, r)


def test_each_plane_of_a_stack():
    want = {"R": 2.0, "G": 1.8, "B": 2.4}
    planes = [moffat_field(want[c], 3.0, seed=2)[0] for c in "RGB"]
    _, xy, _ = moffat_field(1.8, 3.0, seed=2)
    res = measure_psf_rgb(np.stack(planes), _jitter(xy))
    for c, r in res.items():
        assert abs(r.fwhm - want[c]) < 0.02 * want[c], (c, r)
    d = res["G"].as_dict()
    assert set(d) >= {"fwhm", "roundness", "beta", "n"} and "stars" not in d
    json.dumps(d)


def test_elongated_stars_give_roundness_and_angle():
    img, xy, _ = moffat_field(3.0, 3.0, roundness=0.7, angle=30.0, seed=3)
    r = measure_psf(img, xy)
    assert abs(r.fwhm - 3.0) < 0.06
    assert abs(r.roundness - 0.7) < 0.02                       # Siril's r: FWHM_minor / FWHM_major
    assert abs(r.eccentricity - np.sqrt(1 - 0.49)) < 0.02      # SubframeSelector's eccentricity
    assert abs(r.fwhm_major / r.fwhm_minor - 1 / 0.7) < 0.05
    assert abs(np.median(r.stars["angle"]) - 30.0) < 2.0


def test_saturated_stars_and_close_pairs_are_left_out():
    img, xy, amp = moffat_field(2.2, 3.0, seed=4)
    sat_level = 9000.0
    img = np.minimum(img, sat_level)                           # the brightest stars flat-topped
    xi, yi = np.rint(xy).astype(int).T
    saturated = np.array([img[y - 2:y + 3, x - 2:x + 3].max() >= sat_level for x, y in zip(xi, yi)])
    assert saturated.sum() > 10
    # a pair: a second star 4 px from each of five faint stars, given in the list too
    faint = np.where(~saturated)[0][-5:]
    pair = xy[faint] + [4.0, 0.0]
    alpha = 2.2 / (2 * np.sqrt(2 ** (1 / 3.0) - 1))
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    for x0, y0 in pair:
        img += 3000 * (1 + ((xx - x0) ** 2 + (yy - y0) ** 2) / alpha ** 2) ** -3.0
    allxy = np.r_[xy, pair]
    r = measure_psf(img, allxy, saturation=sat_level)
    assert abs(r.fwhm - 2.2) < 0.05, r
    fitted = np.c_[r.stars["x"], r.stars["y"]]
    for bad in np.r_[xy[saturated], xy[faint], pair]:
        assert np.hypot(*(fitted - bad).T).min() > 2.0
    # the same with a mask of the saturated samples (a stack's saturation map)
    r2 = measure_psf(img, allxy, saturation=img >= sat_level)
    assert r2.n == r.n and abs(r2.fwhm - r.fwhm) < 1e-9
    # without the saturation level, the flat tops widen the stars
    r3 = measure_psf(img, allxy, max_stars=20)
    assert r3.fwhm > r.fwhm + 0.2


def test_the_super_pixel_measure_reads_about_twice_the_width():
    """What register.py does today (half-flux radius on 2x2 super-pixels x 4) against the fit, on stars of 1.8 px."""
    import sep
    img, xy, _ = moffat_field(1.8, 3.0, seed=6, H=480, W=480)
    H, W = img.shape
    s = img.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3)).astype(np.float32) * 4.0
    x = s - sep.Background(s, bw=64, bh=64).back()
    sp = (xy - 0.5) / 2.0
    hfr, _ = sep.flux_radius(x, sp[:, 0], sp[:, 1], np.full(len(sp), 5.0), 0.5, subpix=5)
    old = 2.0 * 2.0 * np.median(hfr)
    new = measure_psf(img, xy, cfa="RGGB", channel="G").fwhm
    assert abs(new - 1.8) < 0.05
    assert old > 1.6 * new                                     # heart_a7s: 4.0 px read for stars of 1.7 to 1.9 px


def test_find_stars_on_a_plane_and_on_a_cfa_frame():
    img, xy, _ = moffat_field(2.5, 3.0, seed=7)
    for cfa in (None, "RGGB"):
        found = find_stars(img, cfa=cfa)
        d = np.hypot(*(found[:, None, :] - xy[None]).transpose(2, 0, 1)).min(1)
        assert len(found) >= 0.9 * len(xy) and np.median(d) < 0.6, cfa
    r = measure_psf(img, find_stars(img, cfa="RGGB"), cfa="RGGB", channel="G")
    assert abs(r.fwhm - 2.5) < 0.05


def test_too_few_stars_is_said():
    img, xy, _ = moffat_field(2.0, 3.0)
    r = measure_psf(img, xy[:MIN_STARS - 2])
    assert not r.ok and r.n == MIN_STARS - 2
    r = measure_psf(np.full((100, 100), 10.0), np.zeros((0, 2)))
    assert not r.ok and r.n == 0 and np.isnan(r.fwhm)
    with pytest.raises(ValueError):
        measure_psf(np.zeros((3, 50, 50)), xy, cfa="RGGB")


# ------------------------------------------------------------------------------------------------ real data
HEART = "F:/astro_essais/bench/heart/ours_express/store"


@pytest.mark.slow
def test_heart_a7s_frames_read_the_true_width():
    """The calibrated frames of heart_a7s (read-only): about 1.7 px in green at native resolution, where register.py's
    super-pixels read 3.97 px."""
    caches = [json.load(open(f)) for f in glob.glob(f"{HEART}/calibrate/*/cache.json")]
    caches = [c for c in caches if os.path.exists(c["path"])]
    if not caches:
        pytest.skip("the heart_a7s bench cache is not here")
    c = caches[0]
    mm = np.memmap(c["path"], dtype=np.float16, mode="r", shape=(c["n"], c["height"], c["width"]))
    fw = []
    for i in (0, c["n"] // 2, c["n"] - 1):
        fr = np.asarray(mm[i], np.float32)
        r = measure_psf(fr, find_stars(fr, cfa=c["cfa"], max_stars=2500), cfa=c["cfa"], channel="G",
                        saturation=60000.0, max_stars=150)
        assert r.ok
        fw.append(r.fwhm)
    assert 1.3 < np.median(fw) < 2.2
