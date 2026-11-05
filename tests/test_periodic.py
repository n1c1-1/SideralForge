"""Issue #28: the colour mosaic's pattern and the star network's grid notched in the frequency domain. A synthetic
sky with stars and a nebula, plus a faint pattern every 2 px (one colour) and a grid every 4 px: the patterns go,
the stars keep their flux, the nebula is unchanged; an image without a pattern is returned as it is."""
from __future__ import annotations

import numpy as np

from astro_gpu.core.periodic import lattice_frequencies, mosaic_frequencies, remove_mosaic_pattern, remove_periodic


def _sky(h=512, w=640, seed=2):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.full((3, h, w), 0.1, np.float32) + rng.normal(0, 0.01, (3, h, w)).astype(np.float32)
    img[0] += 0.2 * np.exp(-((yy - 250) ** 2 + (xx - 300) ** 2) / (2 * 60.0 ** 2))
    stars = []
    for _ in range(60):
        y, x = int(rng.integers(20, h - 20)), int(rng.integers(20, w - 20))
        img += (0.8 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.5 ** 2)))[None]
        stars.append((y, x))
    return img, stars


def _peak(a, fy, fx):
    g = a - a.mean()
    P = np.abs(np.fft.fft2(g)) ** 2
    h, w = a.shape
    iy, ix = int(round(fy * h)) % h, int(round(fx * w)) % w
    box = P[max(0, iy - 10):iy + 11, max(0, ix - 10):ix + 11]
    return P[iy, ix] / np.median(box)


def test_the_mosaic_and_grid_patterns_go_the_stars_and_nebula_stay():
    img, stars = _sky()
    h, w = img.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w]
    pat = img.copy()
    pat[0] += 0.004 * ((yy % 2) * 2 - 1)                                   # red: rows every 2 px
    pat[1] += 0.003 * np.cos(2 * np.pi * xx / 4)                            # a grid every 4 px
    assert _peak(pat[0], 0.5, 0) > 100 and _peak(pat[1], 0, 0.25) > 100
    out, taken = remove_mosaic_pattern(pat, 1.0)
    out, taken2 = remove_periodic(out, lattice_frequencies(16))
    assert taken and taken2
    assert _peak(out[0], 0.5, 0) < 5 and _peak(out[1], 0, 0.25) < 5
    for y, x in stars:                                                       # the stars' flux kept
        a, b = img[:, y - 4:y + 5, x - 4:x + 5].sum(), out[:, y - 4:y + 5, x - 4:x + 5].sum()
        assert abs(b - a) / a < 0.01
    assert abs(out[0, 230:270, 280:320].mean() - img[0, 230:270, 280:320].mean()) < 1e-3   # the nebula unchanged


def test_no_pattern_no_change_and_the_lattices():
    img, _s = _sky()
    out, taken = remove_mosaic_pattern(img, 1.0)
    assert taken == [] and np.array_equal(out, img)
    assert set(mosaic_frequencies(1.0)) == {(-0.5, -0.5), (-0.5, 0.0), (0.0, -0.5)}
    assert len(lattice_frequencies(16)) == 16 * 16 - 1
