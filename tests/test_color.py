"""Colour calibration safeguards: physical check of the sensitivity curves, camera profile agreement."""
from __future__ import annotations

import numpy as np

from astro_gpu.stages.color import colour_agreement, physical_response


def _bands(r, g, b, lam):
    return np.stack([np.exp(-0.5 * ((lam - c) / w) ** 2) for c, w in (r, g, b)])


def test_physical_response():
    lam = np.arange(336, 1021, 2.0)
    assert physical_response(lam, _bands((615, 35), (540, 35), (465, 30), lam))[0]
    # red channel emulated by a band at 578 nm (degenerate fit seen on a high-latitude field)
    ok, why = physical_response(lam, _bands((578, 15), (540, 35), (465, 30), lam))
    assert not ok and "red channel" in why
    assert not physical_response(lam, _bands((465, 30), (540, 35), (615, 35), lam))[0]


def test_colour_agreement():
    lam = np.arange(336, 1021, 2.0)
    rng = np.random.default_rng(0)
    T = rng.uniform(3500, 9000, 300)
    spec = 1 / (lam[None] ** 5 * (np.exp(1.4388e7 / (lam[None] * T[:, None])) - 1))       # black bodies
    S = _bands((615, 35), (540, 35), (465, 30), lam)
    cam = (spec * np.gradient(lam)) @ S.T * rng.lognormal(0, 0.01, (300, 3))
    assert colour_agreement(spec, lam, cam, S) < 2.5
    other = _bands((640, 30), (530, 35), (455, 30), lam)                                   # another camera
    assert colour_agreement(spec, lam, cam, other) > colour_agreement(spec, lam, cam, S)


def test_spectra_of_two_grids_are_brought_to_one_and_narrow_profiles_are_not_kept(tmp_path, monkeypatch):
    from astro_gpu.core import gaia_archive as ga
    from astro_gpu.stages import color
    wide = (ga.XP_WAVELENGTHS.copy(), np.linspace(1, 2, ga.XP_WAVELENGTHS.size))
    narrow = (ga.GAVO_XP_WAVELENGTHS.copy(), np.linspace(1.1, 1.5, 41))
    rows = ga.common_grid([wide, None, narrow])
    assert rows[1] is None and all(len(r[0]) == 41 for r in (rows[0], rows[2]))
    assert np.allclose(rows[0][1], np.interp(ga.GAVO_XP_WAVELENGTHS, *wide))
    assert ga.common_grid([wide, wide])[0] is wide
    monkeypatch.setattr(color, "PROFILES", tmp_path)
    color.save_profile("Cam X", ga.GAVO_XP_WAVELENGTHS, np.ones((3, 41)), "p", 0.01)
    assert not list(tmp_path.iterdir())
    color.save_profile("Cam X", ga.XP_WAVELENGTHS, np.ones((3, ga.XP_WAVELENGTHS.size)), "p", 0.01)
    assert color.load_profile("Cam X", ga.GAVO_XP_WAVELENGTHS)["S"].shape == (3, 41)


def test_the_generic_colour_sensor_is_physical():
    from astro_gpu.core.gaia_archive import GAVO_XP_WAVELENGTHS, XP_WAVELENGTHS
    from astro_gpu.stages.color import generic_response
    for lam in (XP_WAVELENGTHS, GAVO_XP_WAVELENGTHS):
        S = generic_response(lam)
        assert S.shape == (3, len(lam)) and physical_response(lam, S)[0]


def test_the_stars_white_balance_brings_the_white_reference_to_white():
    """SPCC's white balance: a line through the measured against the catalogue colour ratios, evaluated at the white
    reference. Stars whose colours a demosaiced stack pulled towards grey and tinted orange (red x 1.15, blue x 0.9
    at white, as heart's Classic stack) get factors that bring the white reference back to white; exact colours get
    none; a fifth of wild stars does not move it."""
    from astro_gpu.stages.color import star_white_balance
    rng = np.random.default_rng(3)
    n = 2000
    e = np.c_[rng.lognormal(0, 0.25, n), np.ones(n), rng.lognormal(-0.05, 0.3, n)] * rng.uniform(1, 50, (n, 1))
    wb = star_white_balance(e, e)
    assert np.allclose(wb["k"], 1, atol=1e-9) and wb["stars"] == n
    ratio = e / e[:, 1:2]
    m = np.c_[1.15 * ratio[:, 0] ** 0.8, np.ones(n), 0.9 * ratio[:, 2] ** 0.75] * rng.lognormal(0, 0.08, (n, 3))
    m[: n // 5] *= rng.lognormal(0, 1.0, (n // 5, 3))                                      # wild stars
    wb = star_white_balance(m, e)
    assert wb["k"][1] == 1.0
    assert abs(wb["k"][0] * 1.15 - 1) < 0.03 and abs(wb["k"][2] * 0.9 - 1) < 0.03, wb
    assert star_white_balance(m[:10], e[:10])["k"].tolist() == [1.0, 1.0, 1.0]           # too few stars: none
