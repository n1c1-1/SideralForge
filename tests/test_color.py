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
