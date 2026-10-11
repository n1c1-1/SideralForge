"""Data-driven safeguards that make the processing independent of the instrument: sky fit under extended objects,
physical H-alpha signature, fixed pattern of the sensor."""
from __future__ import annotations

import numpy as np
import torch

from astro_gpu.core.imgutil import poly_terms
from astro_gpu.stages.background import fit_sky
from astro_gpu.stages.stack import fixed_pattern


def _scene(galaxy_colour):
    hb, wb = 40, 60
    yb, xb = (np.mgrid[0:hb, 0:wb] + 0.5) * 32 - 0.5
    W, H = wb * 32, hb * 32
    Tp = poly_terms(xb.ravel(), yb.ravel(), W, H, 2)
    rng = np.random.default_rng(1)
    sky = np.stack([Tp @ q for q in rng.normal(0, 3, (3, Tp.shape[1]))]).reshape(3, hb, wb) + 50
    gal = 40 * np.exp(-0.5 * (((xb - W / 2) / 300) ** 2 + ((yb - H / 2) / 200) ** 2))
    Hb = 20 * gal / 40 + rng.normal(0, 0.2, (hb, wb))        # the survey follows the galaxy (its HII regions)
    dens = np.exp(rng.normal(0, 0.4, (hb, wb)))                 # unresolved starlight follows the star density
    U = np.stack([dens * c for c in (0.9, 1.0, 0.8)])
    Y = sky + np.stack([c * gal for c in galaxy_colour]) + rng.normal(0, 0.05, (3, hb, wb))
    return Y, Hb, U, Tp, sky


def test_sky_passes_under_extended_object():
    Y, Hb, U, Tp, sky = _scene((0.9, 1.0, 0.8))               # starlight-coloured galaxy, not H-alpha
    good = np.ones(Y.shape[1:], bool)
    sol, used, how = fit_sky(Y, Hb, U, Tp, good, np.array([1.0, 0.35, 0.18]), 2.5)
    assert "imposed" in how                                    # G/R of the "H-alpha" would be ~2.6: not physical
    assert used.mean() < 0.9                                   # the galaxy core is left out of the sky fit
    for c in range(3):
        model_sky = (Tp @ sol[c][2:]).reshape(Y.shape[1:]) + sol[c][1] * U[c]       # gradient + unresolved light
        naive = (Tp @ np.linalg.lstsq(Tp, Y[c].ravel(), rcond=None)[0]).reshape(Y.shape[1:])
        # the galaxy covers most of the frame here: its bias on the sky is strongly reduced, not cancelled
        assert np.abs(model_sky - sky[c]).mean() < 0.3 * np.abs(naive - sky[c]).mean()


def test_physical_halpha_kept_when_plausible():
    Y, Hb, U, Tp, sky = _scene((1.0, 0.4, 0.2))               # a real H-alpha-coloured emission
    sol, used, how = fit_sky(Y, Hb, U, Tp, np.ones(Y.shape[1:], bool), np.array([1.0, 0.35, 0.18]), 2.5)
    assert how == "fitted"
    k = np.array([s[0] for s in sol])
    assert abs(k[1] / k[0] - 0.4) < 0.1


def test_fixed_pattern_recovered_from_drifting_frames():
    rng = np.random.default_rng(2)
    H, W, n = 128, 192, 40
    pattern = rng.normal(0, 3, (H, W)).astype(np.float32)
    yy, xx = np.mgrid[0:H + 80, 0:W + 80]
    sky = (100 + 0.05 * xx + 0.03 * yy).astype(np.float32)                 # smooth sky with drifting stars
    for x0, y0 in rng.uniform(0, [W + 80, H + 80], (120, 2)):
        sky += 200 * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 1.5 ** 2)
    frames = np.zeros((n, H, W), np.float16)
    coef = np.zeros((n, 2, 10))
    for i in range(n):
        dx, dy = i * 1.5, i * 1.0
        frames[i] = sky[int(dy):int(dy) + H, int(dx):int(dx) + W] + pattern + rng.normal(0, 4, (H, W))
        coef[i, 0, 0], coef[i, 1, 0] = W / 2 + dx, H / 2 + dy
    est, info = fixed_pattern(frames, np.arange(n), coef, torch.device("cpu"), H, W, 24)
    assert info["applied"]
    e = est.numpy()
    corr = np.corrcoef(e[8:-8, 8:-8].ravel(), pattern[8:-8, 8:-8].ravel())[0, 1]
    assert corr > 0.8
    # no drift: not applied
    coef[:, 0, 0], coef[:, 1, 0] = W / 2, H / 2
    assert fixed_pattern(frames, np.arange(n), coef, torch.device("cpu"), H, W, 24)[0] is None


def test_generic_star_guard_any_colour():
    from astro_gpu.stages.render_prep import guard_generic
    rng = np.random.default_rng(4)
    H, W = 160, 160
    blue = np.array([0.6, 1.0, 1.6], np.float32)                         # a reflection nebula: no H-alpha at all
    astro = (20 * blue[:, None, None] + rng.normal(0, 0.3, (3, H, W))).astype(np.float32)
    astro[:, 70:80, 70:80] -= 15                                        # star over-subtracted: a black hole
    astro[0, 30:36, 30:36] += 25                                        # star residual: a red dot
    foot = np.zeros((H, W), np.float32)
    foot[66:84, 66:84] = 1
    foot[26:40, 26:40] = 1
    lum = np.array([0.2126, 0.7152, 0.0722], np.float32)
    w_cam = np.ones(3, np.float32)
    Y = np.tensordot(lum, astro, 1).astype(np.float32)
    out = guard_generic(astro, Y, foot, lum, w_cam, 1.0)
    level = np.tensordot(lum, astro[:, 120:, 120:], 1).mean()
    assert np.tensordot(lum, out[:, 72:78, 72:78], 1).mean() > level - 1.0      # no hole left
    col = out[:, 31:35, 31:35].mean((1, 2))
    assert abs(col[0] / col[1] - 0.6) < 0.1                                     # the red dot takes the nebula colour
    assert np.abs(out[:, 110:, 110:] - astro[:, 110:, 110:]).max() == 0         # untouched off the footprints
