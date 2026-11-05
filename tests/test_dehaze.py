"""Issue #22: Dehaze for night skies. The identity at Amount 0; a smooth coloured veil over a dark sky with stars and
a nebula smaller than the meshes: the veil goes (the sky flat again, near its own level), the white stays white, the
sky is never crushed, the nebula and the stars keep their contrast; the background map is SExtractor's mode."""
from __future__ import annotations

import numpy as np

from astro_gpu.client.postfilters.dehaze import background_map, dehaze, mesh_levels


def _sky(h=384, w=512, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = np.full((h, w, 3), 0.08, np.float32) + rng.normal(0, 0.004, (h, w, 3)).astype(np.float32)
    neb = 0.25 * np.exp(-((yy - 190) ** 2 + (xx - 250) ** 2) / (2 * 25.0 ** 2))
    sky[..., 0] += neb
    for _ in range(120):
        y, x = rng.integers(0, h), rng.integers(0, w)
        sky += (0.9 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2.0))[..., None]
    veil = (0.18 * xx / w)[..., None] * np.array([1.0, 0.8, 0.5], np.float32)        # an orange glow to the right
    return np.clip(sky + veil, 0, 1), np.clip(sky, 0, 1), neb


def test_amount_zero_is_the_identity():
    x, _clean, _n = _sky()
    assert dehaze(x, 0.0, 128) is x


def test_the_veil_goes_the_white_stays_and_the_sky_is_never_crushed():
    x, clean, neb = _sky()
    out = dehaze(x, 0.9, 64)
    far = neb < 1e-3
    left, right = out[:, :64][far[:, :64]], out[:, -64:][far[:, -64:]]
    before = np.median(x[:, -64:][far[:, -64:]], 0) - np.median(x[:, :64][far[:, :64]], 0)
    after = np.median(right, 0) - np.median(left, 0)
    assert np.all(np.abs(after) < 0.25 * np.abs(before))                       # the veil's slope mostly gone
    assert np.all(np.median(out[far], 0) > 0.002)                               # the sky kept, not black
    assert np.allclose(dehaze(np.ones((64, 64, 3), np.float32), 0.9, 32), 1.0)  # the white stays white
    core = out[190, 250, 0] - np.median(out[far][:, 0])
    assert core > 0.8 * (x[190, 250, 0] - np.median(x[far][:, 0]))             # the nebula's contrast kept


def test_the_mesh_level_is_the_clipped_mode():
    rng = np.random.default_rng(3)
    c = rng.normal(0.1, 0.01, (64, 64))
    c[10:14, 10:14] = 1.0                                                       # a star in the mesh: clipped
    g = mesh_levels(c, 64)
    assert g.shape == (1, 1) and abs(g[0, 0] - 0.1) < 0.003
    assert background_map(np.repeat(c[..., None], 3, 2).astype(np.float32), 32).shape == (64, 64, 3)
