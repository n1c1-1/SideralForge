"""The astrometry stage's local shortcuts: tiled star detection, what the camera was solved at before."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astro_gpu.stages import astrometry as A


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "aghome"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "userhome"))


def test_tiled_detection_finds_the_same_stars_at_the_same_places():
    """A large stack is detected in 16 tiles in parallel (172 s -> 26 s on NGC 7822); the star list the solver uses
    (unflagged sources, by decreasing flux) is the one the whole image gives, stars on the tile borders included."""
    rng = np.random.default_rng(2)
    H, W = 1200, 1500
    img = rng.normal(100, 3, (H, W)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    img += (40 * np.exp(-((xx - 700) ** 2 + (yy - 500) ** 2) / (2 * 250.0 ** 2))).astype(np.float32)   # a nebula
    n = 900
    xs, ys = rng.uniform(5, W - 5, n), rng.uniform(5, H - 5, n)
    xs[:40] = W / 4 + rng.uniform(-1.5, 1.5, 40)                       # stars right on a tile border
    ys[40:80] = H / 2 + rng.uniform(-1.5, 1.5, 40)
    flux = 10 ** rng.uniform(2.5, 4.5, n)
    for x, y, f in zip(xs, ys, flux):
        x0, y0 = int(x), int(y)
        sl = np.s_[max(0, y0 - 8):y0 + 9, max(0, x0 - 8):x0 + 9]
        img[sl] += (f / (2 * np.pi * 1.5 ** 2) * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * 1.5 ** 2))
                    ).astype(np.float32)
    whole = A.detect(img.copy(), tile_above=10 ** 9)
    tiled = A.detect(img.copy(), tile_above=1)

    def stars(o):
        o = np.sort(o[o["flag"] == 0], order="flux")[::-1]
        return np.c_[o["x"], o["y"]]
    a, b = stars(whole), stars(tiled)
    assert len(a) > 400 and len(a) == len(b)
    assert np.abs(a - b).max() < 1e-6


def test_the_camera_remembers_its_focal_lengths_and_mirror():
    """An unknown focal length was searched on a grid of 22 scales (8 min on NGC 7822); the focal lengths a camera
    was solved at before are tried first, and the mirror image it was solved with."""
    assert A.focal_hints("ILCE-7S", 8.4) == [] and A.mirror_hint("ILCE-7S", 8.4) == (True, False)
    A.remember_instrument("ILCE-7S", 8.4, 272.1, False)
    A.remember_instrument("ILCE-7S", 8.4, 899.8, False)
    A.remember_instrument("ILCE-7S", 8.4, 273.0, False)                 # the same focal length: moved first
    assert A.focal_hints("ILCE-7S", 8.4) == [273.0, 900.0]
    assert A.mirror_hint("ILCE-7S", 8.4) == (False, True)
    assert A.focal_hints("Canon EOS 6D", 6.55) == [] and A.focal_hints("ILCE-7S", None) == []
