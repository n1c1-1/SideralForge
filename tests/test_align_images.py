"""Align images: an image brought onto another by their stars or their plate solutions, of any size, rotation and scale."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from astro_gpu.core.stage import StageError
from astro_gpu.stages import align_images as ai

H, W = 400, 600


def field(n=60, seed=1):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(30, W + 60, n), rng.uniform(30, H + 60, n), rng.uniform(0.2, 1.0, n)]


def render(stars, shape, angle=0.0, scale=1.0, shift=(0.0, 0.0), sigma=1.6, seed=2):
    """An image of the stars seen through a similarity about the origin."""
    h, w = shape
    c, s = np.cos(np.radians(angle)) * scale, np.sin(np.radians(angle)) * scale
    x = c * stars[:, 0] - s * stars[:, 1] + shift[0]
    y = s * stars[:, 0] + c * stars[:, 1] + shift[1]
    rng = np.random.default_rng(seed)
    g = rng.normal(0.02, 0.0008, (h, w)).astype(np.float32)
    r = int(sigma * 5)
    for x0, y0, f in zip(x, y, stars[:, 2]):
        ix, iy = int(round(x0)), int(round(y0))
        if r <= ix < w - r and r <= iy < h - r:
            yy, xx = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
            g[iy - r:iy + r + 1, ix - r:ix + r + 1] += f * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return np.repeat(g[None], 3, axis=0), np.c_[x, y]


def write(path, a):
    fits.PrimaryHDU(a).writeto(path)
    return str(path)


def positions(a):
    c = ai.stars_of(a)
    return c[:, :2]


@pytest.mark.parametrize("angle,scale,shift,shape", [(0, 1, (11.3, -7.8), (H, W)),
                                                     (32, 1, (150.0, -60.0), (H + 100, W + 100)),
                                                     (-8, 0.7, (40.0, 30.0), (360, 520))])
def test_an_image_lands_on_the_reference_through_its_stars(tmp_path, angle, scale, shift, shape):
    stars = field()
    ref, _ = render(stars, (H, W))
    img, _ = render(stars, shape, angle, scale, shift, seed=3)
    a = write(tmp_path / "a.fits", img)
    r = write(tmp_path / "r.fits", ref)
    out, valid = ai.align_onto(img, a, ref, r)
    assert out.shape == ref.shape and valid.shape == (H, W) and valid.any()
    want, got = positions(ref), positions(out)
    d, _i = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(want).query(got)
    assert (d < 1.2).sum() >= 0.7 * len(got) and len(got) >= 15
    if scale != 1 or angle:
        assert not valid.all()                                    # outside the image: marked empty
    assert (out[:, ~valid] == 0).all()


def test_two_plate_solutions_are_used_when_both_files_have_one(tmp_path):
    from astropy.wcs import WCS

    def solved(crpix):
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [83.8, -5.4]
        w.wcs.crpix = crpix
        w.wcs.cdelt = [-0.001, 0.001]
        return w
    g = np.zeros((3, H, W), np.float32)
    g[:, 200, 300] = 1.0
    ref_f, img_f = tmp_path / "r.fits", tmp_path / "a.fits"
    fits.PrimaryHDU(g, solved([300, 200]).to_header()).writeto(ref_f)
    fits.PrimaryHDU(np.roll(g, (10, 20), (1, 2)), solved([320, 210]).to_header()).writeto(img_f)
    said = []
    out, _valid = ai.align_onto(np.roll(g, (10, 20), (1, 2)), img_f, g, ref_f, said.append)
    assert "plate solutions" in said[0]
    y, x = np.unravel_index(out[0].argmax(), out[0].shape)
    assert abs(x - 300) <= 1 and abs(y - 200) <= 1


def test_an_image_without_its_stars_is_said_plainly(tmp_path):
    ref, _ = render(field(), (H, W))
    blank = np.full_like(ref, 0.1)
    with pytest.raises(StageError, match="too few stars"):
        ai.align_onto(blank, tmp_path / "s.fits", ref, tmp_path / "r.fits")


def test_two_different_fields_are_refused(tmp_path):
    ref, _ = render(field(seed=1), (H, W))
    img, _ = render(field(seed=9), (H, W), seed=5)
    with pytest.raises(StageError, match="not the same field"):
        ai.align_onto(img, tmp_path / "a.fits", ref, tmp_path / "r.fits")
