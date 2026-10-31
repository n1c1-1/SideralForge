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
