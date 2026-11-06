"""X-Trans sensors (issue #35): the 6 x 6 pattern read, walked and stacked as the Bayer one."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from astro_gpu.core import cfa
from astro_gpu.core import debayer as db

#: the pattern of a Fujifilm X-T2's visible image (read from its RAF files)
XT = "RBGBRG" "GGRGGB" "GGBGGR" "BRGRBG" "GGBGGR" "GGRGGB"
RAF = Path("F:/astro_essais/donnees/xtrans_2023-08-09/DSCF4520.RAF")


def _mosaic(pattern, H, W, levels=(100.0, 300.0, 50.0)):
    """A flat scene of these R, G, B levels seen through the pattern."""
    return np.asarray(levels, np.float32)[cfa.colour_map(pattern, H, W)]


def test_the_x_trans_pattern_is_recognised_with_its_period():
    assert cfa.is_supported(XT) and cfa.is_xtrans(XT) and cfa.period(XT) == 6
    assert all(cfa.is_supported(b) and cfa.period(b) == 2 for b in cfa.BAYER)
    assert not cfa.is_supported("RGBG" * 9) and not cfa.is_supported("RRGB")
    with pytest.raises(ValueError):
        cfa.period("RGB")


def test_every_3x3_window_of_x_trans_holds_the_three_colours():
    t = np.tile(cfa.channels(XT), (3, 3))
    assert all(set(t[i:i + 3, j:j + 3].ravel()) == {0, 1, 2} for i in range(6) for j in range(6))


def test_the_colour_map_repeats_the_pattern():
    m = cfa.colour_map(XT, 13, 20)
    assert m.shape == (13, 20)
    assert (m[6:12, 6:12] == cfa.channels(XT)).all() and (m[12, :6] == cfa.channels(XT)[0]).all()
    assert [c for _i, _j, c in cfa.positions("GRBG")] == [1, 0, 2, 1]


def test_the_bayer_superpixel_is_the_mean_of_each_colour_of_the_cell():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1000, (10, 12)).astype(np.float32)
    sp = cfa.superpixel(a, "RGGB")
    assert sp.shape == (5, 6, 3)
    assert np.allclose(sp[..., 0], a[0::2, 0::2]) and np.allclose(sp[..., 2], a[1::2, 1::2])
    assert np.allclose(sp[..., 1], (a[0::2, 1::2] + a[1::2, 0::2]) / 2)


def test_the_x_trans_superpixel_gives_each_colour_at_half_resolution():
    a = _mosaic(XT, 36, 48)
    sp = cfa.superpixel(a, XT)
    assert sp.shape == (18, 24, 3)
    assert np.allclose(sp, [100.0, 300.0, 50.0], atol=1e-3)


def test_levelling_takes_the_x_trans_pattern_off_a_flat_sky_and_leaves_bayer_alone():
    a = _mosaic(XT, 60, 60)
    e = cfa.equalised(a, XT)
    assert np.allclose(e, 300.0)
    binned = e.reshape(30, 2, 30, 2).mean((1, 3))
    assert np.ptp(binned) < 1e-3
    b = _mosaic("RGGB", 8, 8)
    assert cfa.equalised(b, "RGGB") is b
    marked = a.copy()
    marked[5, 5] = 70000.0                                     # a saturation mark is left as it is
    assert cfa.equalised(marked, XT, valid_max=65000.0)[5, 5] == 70000.0


def test_bad_x_trans_samples_are_filled_from_their_own_colour():
    a = _mosaic(XT, 36, 36)
    bad = np.zeros(a.shape, bool)
    bad[10:13, 10:13] = True                                    # a block of each colour
    x = torch.from_numpy(np.where(bad, np.nan, a))
    out, filled = db.fill_bad(x, ch_of_k=XT)
    assert bool(filled[11, 11]) and torch.allclose(out, torch.from_numpy(a))


def test_the_calibration_fills_a_bad_x_trans_pixel_from_its_colour():
    from astro_gpu.stages.calibrate import _neighbours
    bad = np.zeros((36, 36), bool)
    bad[14, 15] = True
    ys, xs, ny, nx = _neighbours(bad, XT)
    col = cfa.colour_map(XT, 36, 36)
    assert (col[ny[0], nx[0]] == col[14, 15]).all()
    assert (np.abs(ny[0] - 14) <= 2).all() and (np.abs(nx[0] - 15) <= 2).all()


def test_an_x_trans_frame_is_demosaiced_to_its_three_colours():
    a = torch.from_numpy(_mosaic(XT, 30, 30))
    rgb = db.debayer(a, XT)
    assert rgb.shape == (3, 30, 30)
    assert torch.allclose(rgb[:, 1:-1, 1:-1], torch.tensor([100.0, 300.0, 50.0])[:, None, None])


def test_the_drizzle_weights_of_x_trans_follow_its_colour_counts():
    from astro_gpu.core.coverage import weight_map
    H = W = 36
    coef = np.zeros((1, 2, 10))
    coef[0, 0, :2] = W / 2                                      # the identity: X = W / 2 (1 + u), u = x / (W / 2) - 1
    coef[0, 1, [0, 2]] = H / 2
    w = weight_map(coef, W, H, XT, 1.0, 1.0)
    inner = w[:, 6:30, 6:30]
    assert np.allclose(inner.mean((1, 2)), [8 / 36, 20 / 36, 8 / 36], atol=1e-6)


@pytest.mark.skipif(not RAF.is_file(), reason="the bench's X-Trans frames are not on this machine")
def test_the_pattern_of_a_raf_file_is_read_from_its_visible_image():
    from astro_gpu.io.frames import read_info
    info = read_info(RAF)
    assert info.cfa == XT and (info.width, info.height) == (6032, 4032)
    assert info.camera == "X-T2" and info.pixel_um == 3.91


def test_the_viewer_shows_an_x_trans_mosaic_and_names_its_photosites():
    from astro_gpu.client import stretch as sm
    a = _mosaic(XT, 36, 48)
    sp = sm.superpixel(a, XT)
    assert sp.shape == (3, 18, 24) and np.allclose(sp[:, 5, 7], [100.0, 300.0, 50.0], atol=1e-3)
    assert sm.cfa_letter(XT, 0, 0) == "R" and sm.cfa_letter(XT, 1, 0) == "B" and sm.cfa_letter(XT, 6, 7) == "G"
    assert sm.cfa_letter("RGGB", 1, 1) == "B"


@pytest.mark.skipif(not RAF.is_file(), reason="the bench's X-Trans frames are not on this machine")
def test_markesteijn_demosaics_through_the_raw_file_a_flat_field_exactly():
    H, W = 4032, 6032
    for levels in ((100.0, 300.0, 50.0), (2000.0, 1500.0, 900.0)):     # two frames through the same raw file
        a = torch.from_numpy(_mosaic(XT, H, W, levels))
        a[0, 0] = 0.0                                                    # a span from 0
        rgb = db.debayer(a, XT, template=str(RAF))
        inner = rgb[:, 20:-20, 20:-20]
        assert torch.allclose(inner, torch.tensor(levels)[:, None, None], rtol=1e-3)
    rng = np.random.default_rng(3)
    a = torch.from_numpy(rng.uniform(0, 1000, (H, W)).astype(np.float32))
    rgb = db.debayer(a, XT, template=str(RAF))
    col = torch.from_numpy(cfa.colour_map(XT, H, W).astype(np.int64))
    for c in range(3):
        assert torch.equal(rgb[c][col == c], a[col == c])               # its own samples kept as they were
    with pytest.raises(ValueError):
        db.debayer(a[:100, :100], XT, template=str(RAF))
