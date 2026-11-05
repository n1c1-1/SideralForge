"""Spot healing (client/postfilters/heal.py): a dark dust shadow on a starry sky, healed from its neighbourhood."""
from __future__ import annotations

import time

import numpy as np

from astro_gpu.client.postengine import Composer, clean_spots
from astro_gpu.client.postfilters.heal import find_source, heal, margin

H, W = 300, 400


def sky(seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W], np.float32)[:, None, None]
    base = 0.2 + 0.1 * xx + 0.05 * yy                                     # a gradient across the field
    img = np.repeat(base[..., None], 3, axis=2) * np.array([1.0, 0.9, 0.8]) + rng.normal(0, 0.01, (H, W, 3))
    for _ in range(60):                                                   # stars
        y, x = rng.integers(3, H - 3), rng.integers(3, W - 3)
        img[y - 1:y + 2, x - 1:x + 2] += 0.5
    return img.astype(np.float32)


def test_a_dust_shadow_is_healed_without_an_edge():
    clean = sky()
    img = clean.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.hypot(xx - 200, yy - 150) < 18
    img[blob] *= 0.5                                                      # the shadow: half the light
    dx, dy = find_source(img, 200, 150, 22)
    assert np.hypot(dx, dy) > 40                                          # a source beside it, not on it
    out = heal(img, [[200, 150, 22, dx, dy]], 1.0, (0, 0))
    inner = np.hypot(xx - 200, yy - 150) < 16
    rim = (np.hypot(xx - 200, yy - 150) > 26) & (np.hypot(xx - 200, yy - 150) < 32)
    assert abs(float(np.median(out[inner])) - float(np.median(clean[inner]))) < 0.02     # the level is back
    assert abs(float(np.median(out[inner])) - float(np.median(out[rim]))) < 0.03         # no step at the edge
    assert np.array_equal(out[~(np.hypot(xx - 200, yy - 150) < 22 * 1.3 + 1)],
                          img[~(np.hypot(xx - 200, yy - 150) < 22 * 1.3 + 1)])           # nothing else touched


def test_a_reduced_region_gives_the_same_spot_and_the_margin_covers_the_source():
    img = sky()
    spot = [200.0, 150.0, 20.0, 60.0, -10.0]
    whole = heal(img, [spot], 1.0, (0, 0))
    part = heal(img[100:220, 140:300], [spot], 1.0, (140, 100))         # a region holding the spot and its source
    assert np.allclose(part, whole[100:220, 140:300], atol=1e-5)
    assert margin([spot]) >= 20 * 1.3 + np.hypot(60, -10)
    assert Composer.margin_of([{"kind": "filter", "visible": True, "filter": "heal", "params": {"spots": [spot]}}]) > 64


def test_spots_are_cleaned_and_a_big_spot_is_quick():
    assert clean_spots([[1, 2, 3, 4, 5], ["x"], [1, 2, 0, 0, 0], [1, 2, 3]]) == [[1.0, 2.0, 3.0, 4.0, 5.0]]
    img = np.repeat(sky()[..., :1], 3, axis=2)
    big = np.zeros((900, 900, 3), np.float32) + img[:1, :1]
    t = time.time()
    heal(big, [[450, 450, 150, 330, 0]], 1.0, (0, 0))
    assert time.time() - t < 10


def test_a_spot_on_the_edge_of_the_image_is_healed_too():
    clean = sky(5)
    img = clean.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.hypot(xx - 10, yy - 150) < 25                              # half off the left edge
    img[blob] *= 0.4
    dx, dy = find_source(img, 10, 150, 28)
    assert np.hypot(dx, dy) > 40 and 10 + dx - 28 >= 0                  # a source inside the image
    out = heal(img, [[10, 150, 28, dx, dy]], 1.0, (0, 0))
    assert abs(float(np.median(out[blob])) - float(np.median(clean[blob]))) < 0.03
