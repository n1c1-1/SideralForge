"""Colour calibration: the stars' white balance (one factor per colour), the Gaia spectra on one grid."""
from __future__ import annotations

import numpy as np


def test_spectra_of_two_grids_are_brought_to_one():
    from astro_gpu.core import gaia_archive as ga
    wide = (ga.XP_WAVELENGTHS.copy(), np.linspace(1, 2, ga.XP_WAVELENGTHS.size))
    narrow = (ga.GAVO_XP_WAVELENGTHS.copy(), np.linspace(1.1, 1.5, 41))
    rows = ga.common_grid([wide, None, narrow])
    assert rows[1] is None and all(len(r[0]) == 41 for r in (rows[0], rows[2]))
    assert np.allclose(rows[0][1], np.interp(ga.GAVO_XP_WAVELENGTHS, *wide))
    assert ga.common_grid([wide, wide])[0] is wide


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
