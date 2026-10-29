"""The far halo of the stars: no halo above the light the data show beyond the nebula (synthetic, CPU, seconds)."""
from __future__ import annotations

import numpy as np
import pytest

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


def test_a_channel_too_faint_to_bound_takes_the_factor_of_the_others():
    # a very red star 4 times fainter than its catalogue: its halo is measurable in red only
    rest, unit, F, sig, ok, t, r = _scene(true_amp=0.25)
    F2 = np.array([400.0, 4.0, 3.0])
    rest[1:] = np.stack([0.25 * F2[c] * unit + np.random.default_rng(c).normal(0, 1.0, unit.shape) for c in (1, 2)])
    env = _factors(rest, unit, F2, sig, ok)
    yy, xx = (np.mgrid[0:N, 0:N] + 0.5) * D - 0.5
    f = fh._fac_at(env, xx, yy, XC, YC)
    dark = (t > 150) & (t < 330) & (r > 90) & (r < 600)
    assert abs(float(np.median(f[0][dark])) - 0.25) < 0.05
    for c in (1, 2):                                          # not left at 1: the same factor as the red
        assert abs(float(np.median(f[c][dark])) - float(np.median(f[0][dark]))) < 0.02


def _field(seed=0):
    """A 1600 x 1600 px field: eight stars with their true halo, and in the middle a variable star 4 times fainter
    than its catalogue flux, inside a patchy nebula that hides its halo from the per-star fits (it stays at its
    catalogue flux)."""
    import cv2
    H = W = 1600
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    pos = [(300, 300), (800, 250), (1300, 300), (300, 800), (1300, 800), (300, 1300), (800, 1350), (1300, 1300)]
    col = np.array([1.0, 1.2, 0.8])
    F_true = [5e7] * len(pos) + [2e7]
    pos.append((800, 800))
    img = np.full((3, H, W), 100.0, np.float32)
    for (x, y), f in zip(pos, F_true):
        halo = (np.maximum(np.hypot(xx - x, yy - y), 60.0) / 78.0) ** -1.5
        img += (f / 1e6 * col)[:, None, None].astype(np.float32) * halo[None]
    blobs = cv2.GaussianBlur(rng.normal(size=(H, W)).astype(np.float32), (0, 0), 20)
    neb = np.where((blobs > 0) & (np.hypot(xx - 800, yy - 800) < 450), 150.0, 0.0).astype(np.float32)
    img += neb[None] * np.array([1.0, 0.4, 0.2], np.float32)[:, None, None]
    img += rng.normal(0, 1.0, img.shape).astype(np.float32)
    x = np.array([p[0] for p in pos], float)
    y = np.array([p[1] for p in pos], float)
    F_cat = np.outer(col, [5e7] * 8 + [8e7])
    return img, x, y, F_cat, np.full(len(pos), 60.0)


@pytest.mark.slow
def test_the_bound_also_lowers_the_halo_of_a_star_drawn_at_its_catalogue_flux():
    img, x, y, F, r_in = _field()
    _, info = fh.far_halo(img, x, y, F, r_in, envelope=True, log=lambda *a: None)
    # its halo is lost in the nebula: not measured, drawn at the catalogue flux (the bound used to skip it)
    assert "8" not in info["far_halo_own"] and "8" not in info["far_halo_own_amplitude"]
    assert "8" in info["far_halo_bounded"]
    assert max(info["far_halo_bounded"]["8"][:3]) < 0.5           # down toward its true quarter, every channel
    for i in range(8):                                             # the stars with a right halo keep it
        assert str(i) not in info["far_halo_bounded"] or min(info["far_halo_bounded"][str(i)][:3]) > 0.8
    _, off = fh.far_halo(img, x, y, F, r_in, envelope=False, log=lambda *a: None)
    assert not off["far_halo_bounded"]
