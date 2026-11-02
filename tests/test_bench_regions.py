"""Where the bench measures the target's signal-to-noise (tools/bench/regions.py).

The defect these tests guard against: the target and its sky were chosen box by box on ONE of the compared images
(Siril's), then every stack was measured there. A region picked on a noisy image keeps the boxes where that image's own
noise falls well, so it flatters that image (circular analysis; on Iris, 9 % under Siril for our classic stack when
Siril chose, 13 % over when we chose, equal when neither did). The published fix (SExtractor's dual-image mode, Bertin &
Arnouts 1996; Kriegeskorte et al. 2009): choose on an independent image, measure all stacks on the same regions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import regions as RG  # noqa: E402


class ShiftWCS:
    """A pixel grid at an offset (sky = pixel + offset) with a pixel size scale (sky = scale x pixel + offset)."""

    def __init__(self, ox=0.0, oy=0.0, scale=1.0):
        self.ox, self.oy, self.scale = ox, oy, scale

    def all_pix2world(self, x, y, origin):
        return self.scale * np.asarray(x, float) + self.ox, self.scale * np.asarray(y, float) + self.oy

    def all_world2pix(self, ra, dec, origin):
        return (np.asarray(ra, float) - self.ox) / self.scale, (np.asarray(dec, float) - self.oy) / self.scale


BO = 4
N = 160                    # boxes per side
SKY_PER_BOX = 1.0


def scene(x, y):
    """A faint extended nebula (three blobs) in sky coordinates (pixels), peak ~3 sigma of one box."""
    out = np.zeros_like(x, dtype=float)
    for cx, cy, w, a in ((260, 300, 60, 3.0), (420, 360, 45, 2.4), (330, 180, 70, 2.0)):
        out += a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w * w))
    return out


def stack(rng, wcs, sigma=1.0, scale=1.0, bo=BO):
    """Box means of one stack: the scene plus independent noise (one value per box). A pixel of ``scale`` sky units
    holds scale^2 times the flux; its boxes are ``bo`` pixels wide (bo x scale = 4 sky units, the same angular size)."""
    hb = wb = N
    cy, cx = np.mgrid[0:hb, 0:wb] * bo + (bo - 1) / 2.0
    x, y = wcs.all_pix2world(cx, cy, 0)
    v = (scene(x, y) * scale ** 2 + rng.normal(0, sigma * scale ** 2, (hb, wb))).astype(np.float32)
    return RG.BoxMap(v, np.ones((hb, wb)), wcs, 0.0, 0.0, scale ** 2, bo, float(bo))


def ratio(a: RG.BoxMap, b: RG.BoxMap, ref: str, independent=None, first: str = "A"):
    maps = {"A": a, "B": b} if first == "A" else {"B": b, "A": a}
    reg, sel, src = RG.choose(maps, ref, independent)
    assert reg is not None
    ta, tb = RG.on_regions(reg, sel, a), RG.on_regions(reg, sel, b)
    return ta["snr"] / tb["snr"], ta, tb, src


def mean_ratio(trials=24, wcs_b=None, scale_b=1.0, bo_b=BO, how="mean"):
    r = []
    for k in range(trials):
        rng = np.random.default_rng(1000 + k)
        a = stack(rng, ShiftWCS())
        b = stack(rng, wcs_b or ShiftWCS(scale=scale_b), scale=scale_b, bo=bo_b)
        ind = stack(rng, ShiftWCS(), sigma=0.8) if how == "independent" else None
        r.append(ratio(a, b, "A", ind)[0])
    return float(np.mean(r))


def test_two_stacks_with_independent_noise_come_out_equal_on_average():
    """Two stacks of the same scene (same signal, same noise level, independent noise): the SNR ratio is 1 on average,
    whether the regions are chosen on the mean of the two or on a third, independent image."""
    assert mean_ratio(how="mean") == pytest.approx(1.0, abs=0.03)
    assert mean_ratio(how="independent") == pytest.approx(1.0, abs=0.03)


def test_the_order_of_the_list_does_not_matter():
    """Listed first or last, a stack gets the same regions and the same measure (the mean is symmetric)."""
    rng = np.random.default_rng(5)
    a, b = stack(rng, ShiftWCS()), stack(rng, ShiftWCS())
    r1, ta1, tb1, _ = ratio(a, b, "A", first="A")
    r2, ta2, tb2, _ = ratio(a, b, "A", first="B")
    assert r1 == pytest.approx(r2, rel=1e-9)
    assert ta1["snr"] == pytest.approx(ta2["snr"], rel=1e-9)


def test_the_old_choice_on_one_image_favours_it():
    """Control: regions chosen on stack A alone (the old way, Siril's image) put A ahead of an identical B by several
    percent on average; this is the bias the new choice removes."""
    r = []
    for k in range(24):
        rng = np.random.default_rng(1000 + k)
        a, b = stack(rng, ShiftWCS()), stack(rng, ShiftWCS())
        reg = RG.select_regions(a.v, a.f)
        r.append(RG.on_regions(reg, a, a)["snr"] / RG.on_regions(reg, a, b)["snr"])
    assert np.mean(r) > 1.04, np.mean(r)


def test_stacks_on_other_grids_come_out_equal_too():
    """B on a grid shifted by 2 boxes and with pixels twice as large (its flux per pixel in proportion to the area,
    boxes of 2 pixels: the same angular size): the regions found on the mean still give an equal SNR."""
    assert mean_ratio(wcs_b=ShiftWCS(8.0, -12.0, 2.0), scale_b=2.0, bo_b=2, how="mean") == pytest.approx(1.0, abs=0.04)


def test_the_sky_is_the_whole_band_around_the_target():
    """The nearby sky is every good box of the band, not its darker half (that half is chosen on the noise)."""
    rng = np.random.default_rng(3)
    a = stack(rng, ShiftWCS())
    reg = RG.select_regions(a.v, a.f)
    from scipy import ndimage as ndi
    r2 = max(8, int(0.04 * N))
    band = ndi.binary_dilation(reg["target"], iterations=r2) & ~ndi.binary_dilation(reg["target"], iterations=3)
    assert (reg["sky"] == band).all()


def test_a_gradient_in_one_stack_does_not_move_the_result():
    """Raw stacks keep different smooth gradients (flats, sky glow). A strong one in A only must not change the SNR
    ratio on average: the sky under the target is a plane fitted to the nearby sky of each stack."""
    r = []
    for k in range(24):
        rng = np.random.default_rng(1000 + k)
        a, b = stack(rng, ShiftWCS()), stack(rng, ShiftWCS())
        yy, xx = np.mgrid[0:N, 0:N] / N
        a.v = (a.v + 4.0 * xx + 3.0 * yy * yy).astype(np.float32)
        r.append(ratio(a, b, "A")[0])
    assert np.mean(r) == pytest.approx(1.0, abs=0.04), np.mean(r)


def test_saved_maps_give_the_same_measure(tmp_path):
    """The box maps saved by a measure (retarget.py reads them) give the same regions and the same SNR."""
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.crpix = [320.0, 320.0]
    w.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]
    rng = np.random.default_rng(11)
    a, b, ind = (stack(rng, ShiftWCS()) for _ in range(3))
    for m in (a, b, ind):
        m.wcs = w
    maps = {"A": a, "B": b}
    reg, sel, src = RG.choose(maps, "A", ind)
    RG.save_maps(tmp_path / "m.npz", maps, ind, src)
    maps2, ind2, src2 = RG.load_maps(tmp_path / "m.npz")
    reg2, sel2, _ = RG.choose(maps2, "A", ind2)
    assert src2 == src == "independent" and (reg["target"] == reg2["target"]).all()
    assert RG.on_regions(reg, sel, b)["snr"] == pytest.approx(RG.on_regions(reg2, sel2, maps2["B"])["snr"], rel=1e-6)
