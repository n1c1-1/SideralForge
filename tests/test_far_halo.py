"""The far halo of the stars: no halo above the light the data show beyond the nebula (synthetic, CPU, seconds)."""
from __future__ import annotations

import numpy as np

from astro_gpu.stages import far_halo as fh

D = 8                        # coarse pixel size of the fit grid (px)
N = 200                      # coarse grid: 1600 x 1600 px
XC = YC = 800.0              # the star


def _scene(true_amp, nebula=200.0, seed=0):
    """A star's halo model (unit flux x F), the data: true_amp x that halo + a one-sided nebula in the sector of
    angles 0..90 deg (+ noise), on the coarse grid. Returns rest, unit, F, sig, ok, angle of each coarse pixel."""
    yy, xx = (np.mgrid[0:N, 0:N] + 0.5) * D - 0.5
    r = np.maximum(np.hypot(xx - XC, yy - YC), 60.0)
    unit = (r / 78.0) ** -1.5
    F = np.array([400.0, 500.0, 300.0])
    t = np.degrees(np.arctan2(yy - YC, xx - XC)) % 360
    neb = np.where(t < 90, nebula * np.exp(-r / 600.0), 0.0)
    rng = np.random.default_rng(seed)
    sig = np.array([1.0, 1.0, 1.0])
    rest = np.stack([true_amp * F[c] * unit + neb * (1.0, 0.4, 0.1)[c] + rng.normal(0, sig[c], unit.shape)
                     for c in range(3)])
    return rest, unit, F, sig, np.ones((N, N), bool), t, r


def _factors(rest, unit, F, sig, ok):
    return fh._envelope(rest, unit, F, sig, ok, D, XC, YC, 1.0, 0.0, 78.0, 700.0)


def test_a_halo_above_the_data_is_lowered_to_it_on_the_dark_side():
    rest, unit, F, sig, ok, t, r = _scene(true_amp=0.8)
    env = _factors(rest, unit, F, sig, ok)
    assert env is not None
    yy, xx = (np.mgrid[0:N, 0:N] + 0.5) * D - 0.5
    f = fh._fac_at(env, xx, yy, XC, YC)
    dark = (t > 150) & (t < 330) & (r > 90) & (r < 600)          # far from the nebula's sector
    assert np.all(f <= 1.0 + 1e-6)
    for c in range(3):
        assert abs(float(np.median(f[c][dark])) - 0.8) < 0.04
    # the halo left there no longer digs below the data
    for c in range(3):
        left = rest[c][dark] - f[c][dark] * F[c] * unit[dark]
        assert np.percentile(left, 20) > -3 * sig[c]


def test_a_nebula_on_one_side_neither_raises_nor_lowers_a_right_halo():
    rest, unit, F, sig, ok, t, r = _scene(true_amp=1.0, nebula=400.0)
    env = _factors(rest, unit, F, sig, ok)
    yy, xx = (np.mgrid[0:N, 0:N] + 0.5) * D - 0.5
    f = np.ones((3, N, N)) if env is None else fh._fac_at(env, xx, yy, XC, YC)
    ring = (r > 90) & (r < 600)
    assert np.all(f <= 1.0 + 1e-6)
    assert float(f[:, ring].min()) > 0.95                    # nothing taken from a halo the data hold


def test_the_bounded_halo_does_not_rise_outward_in_any_direction():
    rest, unit, F, sig, ok, t, r = _scene(true_amp=0.8)
    # a dark patch (a dust lane, a hole in the sky) at 300 px in one direction
    yy, xx = (np.mgrid[0:N, 0:N] + 0.5) * D - 0.5
    lane = (np.abs(t - 200) < 20) & (np.abs(r - 300) < 40)
    rest[:, lane] -= 0.5 * F[:, None] * unit[lane]
    env = _factors(rest, unit, F, sig, ok)
    g = env["f"] * np.exp(-1.5 * (env["lr0"] + env["dlr"] * np.arange(env["f"].shape[1])))[None, :, None]
    assert np.all(np.diff(g, axis=1) <= 1e-6)                # factor x shape: non-increasing along each direction


def test_factors_where_the_halo_is_lost_in_the_noise_leave_it_alone():
    rest, unit, F, sig, ok, t, r = _scene(true_amp=0.8)
    assert _factors(rest, unit, F * 1e-4, sig, ok) is None
