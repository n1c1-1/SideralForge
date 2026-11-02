"""Debayering for the Classic stacking method (astro_gpu/core/debayer.py: RCD as Siril runs it, bilinear).

Synthetic CFA frames (RGGB) whose full-colour truth is known: flat colour fields, grey edges and stars, a frame larger
than one RCD tile. And, when the files of the validation are there (F:/astro_essais/classic_check/rcd, made by its
prep.py and Siril 1.4's own debayering of the same frame), the real check against Siril.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from astro_gpu.core import debayer as db

CH = "RGGB"


def grid(H, W):
    return torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")


def mosaic(rgb: torch.Tensor) -> torch.Tensor:
    """The CFA frame a sensor would record of the colour image rgb (3, H, W)."""
    col = db.colour_of(CH, *rgb.shape[1:])
    return torch.gather(rgb, 0, col[None])[0]


def test_short_form_of_the_directional_statistic_is_the_original_v23_one():
    """rcd.cc's sum of three squared high-pass terms is the long quadratic form of RCD v2.3 (Luis Sanz Rodríguez's
    "exact, shorter" formula)."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        c = rng.uniform(0, 1, 9) * rng.choice([1e-3, 1.0])
        a, b = db.rcd_v23_vh_stat(c), db.rcd_short_vh_stat(c)
        assert b == pytest.approx(a, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("method", db.METHODS)
def test_a_flat_colour_field_and_the_recorded_samples_are_kept(method):
    H, W = 64, 72
    rgb = torch.stack([torch.full((H, W), v) for v in (1000.0, 2000.0, 500.0)])
    out = db.debayer(mosaic(rgb), CH, method)
    assert torch.allclose(out, rgb, rtol=0, atol=0.05)
    rng = np.random.default_rng(1)
    noisy = torch.from_numpy(rng.uniform(100, 5000, (H, W)).astype(np.float32))
    out = db.debayer(noisy, CH, method)
    col = db.colour_of(CH, H, W)
    native = torch.gather(out, 0, col[None])[0]
    assert torch.allclose(native, noisy, rtol=2e-6, atol=0.02)


def test_rcd_leaves_far_less_false_colour_than_bilinear_on_grey_edges_and_stars():
    H, W = 96, 96
    yy, xx = grid(H, W)
    scenes = {"edge": 200 + 3000 * (xx > 47.5).float(),
              "slanted edge": 200 + 3000 * torch.sigmoid((xx - 0.3 * yy - 30) * 3),
              "star": 200 + 4000 * torch.exp(-((xx - 48.3) ** 2 + (yy - 47.6) ** 2) / (2 * 1.3 ** 2)),
              "lines": 200 + 1000 * (torch.sin(xx * 0.9) > 0).float()}
    for name, img in scenes.items():
        truth = img.expand(3, H, W)
        err = {m: float((db.debayer(img.clone(), CH, m) - truth)[:, 16:-16, 16:-16].pow(2).mean().sqrt())
               for m in db.METHODS}
        assert err["rcd"] < 0.2 * err["bilinear"], (name, err)


def test_rcd_tiles_are_put_back_in_place():
    """A frame of several tiles: the same content moved by one tile step (176 px) gives the same result moved, and
    the batch size does not matter."""
    rng = np.random.default_rng(7)
    H, W = 420, 460
    yy, xx = grid(H, W)
    img = 500 + 300 * torch.sin(xx / 7) * torch.cos(yy / 9) + torch.from_numpy(rng.normal(0, 30, (H, W)).astype(np.float32))
    for _ in range(40):
        y, x = rng.uniform(20, H - 20), rng.uniform(20, W - 20)
        img += 3000 * torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2))
    a = db.rcd(img, CH, siril_range=False)
    b = db.rcd(img, CH, siril_range=False, tiles_per_batch=1)
    assert torch.equal(a, b)
    s = db.STEP
    c = db.rcd(img[s:, s:].contiguous(), CH, siril_range=False)
    m = 30
    assert torch.allclose(c[:, m:-m, m:-m], a[:, s + m:-m, s + m:-m], rtol=1e-5, atol=1e-3)


def test_bad_samples_are_filled_like_sirils_cosmetic_correction():
    rng = np.random.default_rng(2)
    x = torch.from_numpy(rng.uniform(100, 200, (40, 40)).astype(np.float32))
    x[20, 21] = float("nan")                                   # a green sample of an R-G row
    x[11, 11] = float("nan")                                   # a blue sample, one of its neighbours bad too
    x[13, 13] = float("nan")
    y, miss = db.fill_bad(x)
    assert int(miss.sum()) == 3 and not bool(torch.isnan(y).any())
    nb = [x[20 + dy, 21 + dx] for dy in (-2, 0, 2) for dx in (-2, 0, 2) if dy or dx]
    assert float(y[20, 21]) == pytest.approx(float(torch.stack(nb).mean()), rel=1e-6)
    nb = [x[11 + dy, 11 + dx] for dy in (-2, 0, 2) for dx in (-2, 0, 2) if (dy or dx) and (dy, dx) != (2, 2)]
    assert float(y[11, 11]) == pytest.approx(float(torch.stack(nb).mean()), rel=1e-6)
    assert torch.equal(y[~miss], x[~miss])


def _stars_on_unbalanced_sensor(seed: int = 5):
    """A CFA frame (RGGB, 256 x 256) of a sensor that records the sky and white stars at 0.7 / 1 / 0.4 of green in
    red / green / blue (an unmodified colour camera: heart's Sony A7S records its sky at 0.73 / 1 / 0.40), 36 small
    stars (FWHM 2.6 px) of that colour at random sub-pixel places, noise 5; and every star's true flux (R, G, B)."""
    rng = np.random.default_rng(seed)
    H = W = 256
    yy, xx = grid(H, W)
    level = torch.tensor([0.7, 1.0, 0.4])
    rgb = (300.0 * level)[:, None, None].expand(3, H, W).clone()
    s = 2.6 / 2.355
    stars = []
    for i in range(6):
        for j in range(6):
            y, x = 24 + 40 * i + rng.uniform(0, 1), 24 + 40 * j + rng.uniform(0, 1)
            amp = rng.uniform(2000, 8000)
            g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))
            rgb += (amp * level)[:, None, None] * g
            stars.append((x, y, amp * level.numpy() * 2 * np.pi * s * s))
    cfa = mosaic(rgb) + torch.from_numpy(rng.normal(0, 5, (H, W)).astype(np.float32))
    return cfa, stars, (xx, yy)


def _flux(img, x, y, xx, yy, r=8.0):
    d = torch.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    ap, an = d <= r, (d >= 12) & (d <= 16)
    return np.array([float(img[c][ap].sum() - img[c][an].median() * ap.sum()) for c in range(3)])


def test_white_balance_before_rcd_keeps_the_star_colours():
    """RCD on an unbalanced frame moves green flux into the weak colour of small stars (heart: blue +46 %, red +11 %,
    and the colour stage then found unphysical sensitivity curves). Balanced first and unbalanced after, as raw
    converters demosaic, each colour's flux stays within a few per cent; the balance is the frame's own."""
    cfa, stars, (xx, yy) = _stars_on_unbalanced_sensor()
    bal = db.cfa_balance(cfa, CH)
    assert bal is not None and bal[1] == 1.0
    assert bal[0] == pytest.approx(1 / 0.7, rel=0.02) and bal[2] == pytest.approx(1 / 0.4, rel=0.02)
    plain = db.debayer(cfa, CH, "rcd", balance=None)
    auto = db.debayer(cfa, CH, "rcd")
    r_plain = np.median([_flux(plain, x, y, xx, yy) / t for x, y, t in stars], 0)
    r_auto = np.median([_flux(auto, x, y, xx, yy) / t for x, y, t in stars], 0)
    assert r_plain[2] > 1.25                                    # the defect: blue gains a quarter or more
    assert np.all(np.abs(r_auto - 1) < 0.05), r_auto
    assert torch.equal(db.rcd(cfa, CH, balance=bal), auto)


def test_white_balance_multipliers_are_taken_off_exactly_and_need_a_positive_level():
    rng = np.random.default_rng(4)
    x = torch.from_numpy(rng.uniform(100, 5000, (64, 72)).astype(np.float32))
    a = db.rcd(x, CH)
    b = db.rcd(x, CH, balance=(1.0, 1.0, 1.0))
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-3)
    out = db.rcd(x, CH, balance=(2.0, 1.0, 3.0))
    native = torch.gather(out, 0, db.colour_of(CH, 64, 72)[None])[0]
    assert torch.allclose(native, x, rtol=2e-6, atol=0.02)        # the recorded samples come back unchanged
    dark = x - 6000.0                                              # no positive level in any colour
    assert db.cfa_balance(dark, CH) is None
    assert torch.allclose(db.debayer(dark, CH, "rcd"), db.rcd(dark, CH))
    with pytest.raises(ValueError):
        db.rcd(x, CH, balance=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        db.debayer(x, CH, "rcd", balance="sky")


VALID = Path("F:/astro_essais/classic_check/rcd")


@pytest.mark.slow
@pytest.mark.skipif(not (VALID / "pp_cfa_real.fit").exists(), reason="the validation files are not there")
def test_rcd_matches_sirils_own_debayering_of_a_real_frame():
    """A calibrated frame of heart (Sony A7S, RGGB) debayered by Siril 1.4 (calibrate_single -debayer, RCD) and here:
    the same to float precision but for a few hundred pixels where a rounding flips a direction choice."""
    from astropy.io import fits
    x = np.load(VALID / "cfa_real.npy") / 65535.0
    siril = np.asarray(fits.getdata(VALID / "pp_cfa_real.fit"), np.float32)
    ours = db.rcd(torch.from_numpy(x.astype(np.float32)), CH, balance=None).numpy()
    d = np.abs(ours - siril)[:, 16:-16, 16:-16]
    rel = d / (np.abs(siril[:, 16:-16, 16:-16]) + 1e-4)
    assert float(np.sqrt((d ** 2).mean())) < 1e-6
    assert float(np.median(rel)) < 1e-5 and float((rel > 1e-4).mean()) < 1e-4
