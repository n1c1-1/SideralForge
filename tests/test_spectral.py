"""The approximate spectral response of the sensor (core/spectral.py): known curves found again from synthetic stars
(within the approximation the broad star spectra allow), and read from a computed project's results
(api.spectral_response)."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import TEST_PROJECT, needs_test_project

from astro_gpu import api
from astro_gpu.core import spectral

LAM = np.arange(336.0, 1021.0, 2.0)


def stars(n=800, seed=1):
    """Black bodies 3000 to 12000 K, reddened up to E(B-V) = 1: spectra as broad and alike as real stars'."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(3000, 12000, n)
    ebv = rng.uniform(0, 1, n)
    lm = LAM * 1e-9
    bb = 1 / (lm[None] ** 5 * (np.exp(1.4388e-2 / (lm[None] * t[:, None])) - 1))
    sp = bb * 10 ** (-0.4 * 3.1 * ebv[:, None] * (550.0 / LAM[None]) ** 1.3)
    return sp / sp.max(1, keepdims=True)


def curves():
    def g(c, w):
        return np.exp(-0.5 * ((LAM - c) / w) ** 2)
    return np.stack([g(610, 35) + 0.25 * g(680, 30), g(535, 40), g(465, 35)])


def test_known_curves_come_back_approximately():
    """The shapes come back within a few hundredths over the visible (what the chart is for); a narrow feature does
    not (the response at H-alpha), which is why the chart says approximate."""
    spec, S = stars(), curves()
    cam = (spec * np.gradient(LAM)) @ S.T
    cam *= 1 + 0.01 * np.random.default_rng(2).standard_normal(cam.shape)    # 1 % photometric noise
    r = spectral.response(spec, LAM, cam)
    truth = S / S.max()
    vis = (LAM >= 420) & (LAM <= 720)
    assert r.stars == len(spec) and r.S.shape == (3, len(LAM)) and r.S.max() == pytest.approx(1.0)
    assert np.median(np.abs(r.S[:, vis] - truth[:, vis])) < 0.03
    assert np.abs(r.S[:, vis] - truth[:, vis]).max() < 0.25
    assert all(0 < x < 0.05 for x in r.residual)                       # the fluxes explained to their noise
    i = np.argmin(abs(LAM - 550))
    assert np.argmax(r.S[:, i]) == 1 and np.argmax(r.S[:, np.argmin(abs(LAM - 465))]) == 2
    sh = r.shares()
    ok = np.isfinite(sh[0])
    assert ok.any() and np.allclose(sh[:, ok].sum(0), 100.0)            # the three shares make the whole signal
    r2 = spectral.response(spec, LAM, cam)
    assert np.array_equal(r.S, r2.S)                                    # nothing random


@needs_test_project
def test_the_response_of_a_computed_project_is_read_from_its_results():
    p = api.open_project(TEST_PROJECT)
    wf = next(w.id for w in p.workflows if api.stage_outputs(p, w.id, "color") is not None)
    r = api.spectral_response(p, wf)
    assert r.stars >= spectral.MIN_STARS and r.S.max() == pytest.approx(1.0) and len(r.lam) == r.S.shape[1]
    red, green, blue = (float(np.interp(656.3, r.lam, r.S[c])) for c in range(3))
    assert red > green and red > blue                                   # H-alpha: red first


def test_no_colour_result_no_curves(tmp_path):
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    p = project2.create(tmp_path / "P", "P", Target(name="M 1"))
    w = api.add_workflow(p, "Classic stack", "Classic stack")
    with pytest.raises(api.NotDone):
        api.spectral_response(p, w.id)
