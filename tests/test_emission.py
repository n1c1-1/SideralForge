"""The H-alpha emission of the render (render/simple.emission, continuum_ratio): the red above the continuum of the
stars, added or taken out before the stretch; nothing moves at 0, and a pure continuum is never touched."""
from __future__ import annotations

import numpy as np
import torch

from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings, continuum_ratio, emission


def field(seed=0):
    """A linear (3, H, W) field: a sky, stars of a continuum colour (red / green = 1.2), and a patch of emission
    (red only)."""
    rng = np.random.default_rng(seed)
    a = np.full((3, 200, 260), 2.0, np.float32) + 0.05 * rng.standard_normal((3, 200, 260)).astype(np.float32)
    for y, x in rng.integers(10, 190, (60, 2)):
        a[:, y - 1:y + 2, x - 1:x + 2] += np.array([120.0, 100.0, 90.0], np.float32)[:, None, None]
    a[0, 60:140, 150:230] += 1.0
    return a


def test_the_continuum_is_the_stars_colour():
    a = field()
    hp = np.maximum(a - 2.0, 0)
    assert abs(continuum_ratio(hp) - 1.2) < 0.02


def test_emission_adds_the_red_excess_only_where_there_is_one():
    a = field()
    x = torch.from_numpy(a - 2.0)
    assert emission(x, 1.2, 0.0, 1) is x                                   # 0: the image as it is
    y = emission(x, 1.2, 1.0, 1)
    d = (y - x).numpy()
    assert abs(float(d[0, 90:110, 170:210].mean()) - 1.0) < 0.1           # the patch: its excess added once
    assert np.allclose(d[1:], 0) and float(np.abs(d[0, 20:40, 20:60]).mean()) < 0.05   # green, blue, the sky alone
    z = emission(x, 1.2, -1.0, 1).numpy()
    assert abs(float(z[0, 90:110, 170:210].mean())) < 0.1                 # -1: the line taken out


def test_the_engine_renders_the_same_at_0_and_redder_above():
    a = field()
    eng = SimpleRenderEngine.from_array(a, "cpu")
    base = eng.render(SimpleSettings(faint=0.0), 1.0).final.numpy()       # the line, not the faint tones lifted
    same = eng.render(SimpleSettings(faint=0.0, halpha=0.0), 1.0).final.numpy()
    more = eng.render(SimpleSettings(faint=0.0, halpha=2.0), 1.0).final.numpy()
    assert np.array_equal(base, same)
    patch = (slice(None), slice(90, 110), slice(170, 210))
    assert more[patch][0].mean() > base[patch][0].mean() + 0.02
    assert abs(float(eng.continuum()) - 1.2) < 0.05
