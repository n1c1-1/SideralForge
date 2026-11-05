"""The catalogs stage's selection of the isolated colour stars (no network)."""
from __future__ import annotations

import numpy as np

from astro_gpu.stages.catalogs import CatalogsParams, _isolated_xp


def test_isolated_xp_is_the_same_from_the_npz_file_and_from_arrays_read_once(tmp_path):
    """The stage read gaia_deep.npz's arrays from the archive at every star of its loop (4-5 min of CPU per run); it
    now reads them once. The selection is the same either way, and the one the rule gives: XP stars of the magnitude
    range away from the edges with no star brighter than G + 3 within 40", the brightest first, at most xp_max."""
    rng = np.random.default_rng(3)
    n, W, H, scale = 3000, 2000, 1500, 2.0
    d = dict(source=np.arange(n, dtype=np.int64), x=rng.uniform(-30, W + 30, n), y=rng.uniform(-30, H + 30, n),
             g=rng.uniform(6, 18.5, n), xp=(rng.random(n) < 0.6).astype(float))
    d["xp"][::97] = np.nan
    np.savez(tmp_path / "gaia_deep.npz", **d)
    p = CatalogsParams(xp_max=150)
    with np.load(tmp_path / "gaia_deep.npz") as z:
        from_file = _isolated_xp(z, p, W, H, scale)
        once = _isolated_xp({k: z[k] for k in z.files}, p, W, H, scale)
    # the rule, by brute force
    g, x, y = d["g"], d["x"], d["y"]
    cand = (np.nan_to_num(d["xp"]) == 1) & (g > p.xp_gmin) & (g < p.xp_gmax) & (x > 60) & (x < W - 60) \
        & (y > 60) & (y < H - 60)
    near = np.hypot(x[:, None] - x[None], y[:, None] - y[None]) <= 40.0 / scale
    brighter = g[None] < g[:, None] + 3
    np.fill_diagonal(brighter, False)
    ok = np.where(cand & ~(near & brighter).any(1))[0]
    ok = ok[np.argsort(g[ok])]                                 # spread over the magnitudes (issue #12)
    want = ok[np.linspace(0, len(ok) - 1, p.xp_max).round().astype(int)]
    assert len(want) == p.xp_max                               # the cap is reached: the order matters
    assert np.array_equal(from_file, want) and np.array_equal(once, want)
