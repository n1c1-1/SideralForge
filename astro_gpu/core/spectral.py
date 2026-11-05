"""An approximation of the sensor's spectral response, rebuilt from the stars of the field: a diagnostic of the camera,
shown and exported in Inspect > Color. It changes no colour (the Color block is a white balance of the stars,
docs/chantiers/couleur.md) and needs no computation of its own: it reads what the Catalogs block (the stars' Gaia XP
spectra) and the Color block (their fluxes measured in the camera's colours) already wrote.

The method is the Color block's until 3 November. A colour star's flux in a channel is the integral of its spectrum
times the channel's curve; each curve is a non-negative sum of smooth Gaussians whose second differences are
penalised, the positivity and smoothness constraints of the recovery of camera sensitivities from known spectra
(Finlayson, Hordley & Hubel 1998, "Recovering device sensitivities with quadratic programming", IS&T/SID Color Imaging
Conference 6), the stars the fit cannot explain rejected over four passes.

Why only an approximation: star spectra are broad and alike, so many different curves explain the stars equally well.
On synthetic stars with known curves (tests/test_spectral.py) the shapes come back within 0.02 of the maximum in the
median over the visible, but the response at the H-alpha line came out 0.76 for a true 0.59 (about 30 % off). No error
band is drawn: neither resampling the stars nor varying the smoothing measured that error (5 November).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

#: the Gaussians the curves are made of: centres every 15 nm from 380 to 1000 nm, sigma 12 nm
CENTRES = np.arange(380, 1001, 15.0)
WIDTH = 12.0
#: the weight of the smoothness penalty
MU = 3.0
#: fewer colour stars than this: no curves (the Color block's former floor)
MIN_STARS = 50


@dataclass
class SpectralResponse:
    lam: np.ndarray                  # wavelengths (nm), (L,)
    S: np.ndarray                    # the R, G, B curves, relative to their maximum, (3, L)
    stars: int                       # the colour stars the fit used
    residual: tuple[float, float, float]   # median relative misfit of the stars' fluxes, per channel

    def shares(self) -> np.ndarray:
        """The share (%) of each channel in the signal of a pure line at each wavelength (NaN where the sensor sees
        almost nothing)."""
        pos = np.maximum(self.S, 0.0)
        tot = pos.sum(0)
        out = 100 * pos / np.where(tot > 0, tot, 1.0)
        out[:, tot <= 0.03 * float(tot.max() or 1.0)] = np.nan
        return out


def fit_response(spec: np.ndarray, lam: np.ndarray, cam: np.ndarray, mu: float = MU):
    """The three curves S (3, L) such that cam (N, 3) = integral of spec (N, L) times S_c, and the median relative
    misfit of each channel."""
    B = np.exp(-0.5 * ((lam[:, None] - CENTRES[None]) / WIDTH) ** 2)
    Fm = (spec * np.gradient(lam)) @ B
    K = len(CENTRES)
    D = np.diff(np.eye(K), 2, axis=0)
    curves, rel = [], []
    for c in range(3):
        yv = cam[:, c]
        A = Fm * (1.0 / yv)[:, None]                       # each star's equation over its own flux: relative errors
        sc = A.std(0) + 1e-30
        An = A / sc
        Aa = np.vstack([An, np.sqrt(mu) * D / sc * sc.mean()])
        ya = np.concatenate([np.ones(len(yv)), np.zeros(len(D))])
        keep = np.ones(len(yv), bool)
        a = np.zeros(K)
        r = np.zeros(len(yv))
        for _ in range(4):
            rows = np.concatenate([np.where(keep)[0], len(yv) + np.arange(len(D))])
            a, _ = nnls(Aa[rows], ya[rows], maxiter=5000)
            r = An @ a - 1
            s = 1.4826 * np.median(np.abs(r[keep]))
            keep = np.abs(r) < 4 * s
        curves.append(B @ (a / sc))
        rel.append(float(np.median(np.abs(r[keep]))))
    return np.stack(curves), rel


def response(spec: np.ndarray, lam: np.ndarray, cam: np.ndarray) -> SpectralResponse:
    """The approximate curves, over their maximum."""
    S, rel = fit_response(spec, lam, cam)
    return SpectralResponse(np.asarray(lam, float), S / (float(S.max()) or 1.0), len(cam), tuple(rel))
