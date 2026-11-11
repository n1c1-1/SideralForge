"""Issue #139: a stop during the Astrometry's own search is answered within about a second, whatever loop the search is
in: the triangle match's hypotheses (a vain match of 100 control points, some 35 s on a laptop, 16 s here), the
similarities of a sparse field, the blind solver's candidates. Measured from the request, made by a callback at a chosen
step of the search, to the stop."""
from __future__ import annotations

import time
from types import SimpleNamespace

import astroalign
import numpy as np
import pytest

from astro_gpu.core.stage import Interrupted
from astro_gpu.stages import astrometry as am
from astro_gpu.stages.astrometry import AstrometryParams
from tests.test_astrometry_search import DEC, RA, SCALE, H, W, _catalogue


class _Stop:
    """A stop requested at the ``at``-th call of a step of the search (``step``), as ``Context.check_stop`` sees it."""

    def __init__(self, at: int):
        self.at, self.n, self.asked, self.looks = at, 0, None, 0

    def step(self):
        self.n += 1
        if self.n == self.at:
            self.asked = time.perf_counter()

    def check_stop(self):
        self.looks += 1
        if self.asked is not None:
            raise Interrupted("stop requested")


def _field(monkeypatch, n=80):
    rng = np.random.default_rng(7)
    o = np.zeros(n, dtype=[("x", float), ("y", float), ("flux", float), ("flag", int)])
    o["x"], o["y"], o["flux"] = rng.uniform(0, W, n), rng.uniform(0, H, n), np.arange(n, 0, -1)
    gs, gm = _catalogue()
    monkeypatch.setattr(am, "detect", lambda g: o)
    monkeypatch.setattr(am, "gaia_bright", lambda *a, **k: (gs, gm))
    monkeypatch.setattr(am, "remember_instrument", lambda *a: None)
    monkeypatch.setattr(am, "mirror_hint", lambda *a: (True, False))
    monkeypatch.setattr(am, "focal_hints", lambda *a: [])
    return o, gs, gm


@pytest.mark.parametrize("at", [1, 250, 1000])
def test_a_stop_in_a_vain_triangle_match_ends_the_stage_within_a_second(tmp_path, monkeypatch, at):
    """The scale known, a field the catalogue does not match: its two mirrors' matches exhaust 4 000 hypotheses each,
    and the stop, asked at the ``at``-th, was only seen at the next trial scale's progress, after them (the stage ended
    on "match no star" some 30 s later)."""
    _field(monkeypatch)
    stop = _Stop(at)
    estimate = astroalign.estimate_transform

    def counted(*a, **k):                   # one call per hypothesis of the RANSAC (its model's fit)
        stop.step()
        return estimate(*a, **k)
    monkeypatch.setattr(astroalign, "estimate_transform", counted)

    def progress(frac, msg="", force=False):
        stop.check_stop()                   # as Context.progress does
    ctx = SimpleNamespace(load_json=lambda stage, name: {"scale": 1.0},
                          load=lambda *a, **k: np.zeros((3, H, W), np.float32), log=lambda m, level="info": None,
                          progress=progress, metric=lambda k, v: None, out=lambda n: tmp_path / n,
                          metric_of=lambda *a, **k: None, check_stop=stop.check_stop)
    with pytest.raises(Interrupted):
        am.Astrometry().run(ctx, AstrometryParams(ra=RA, dec=DEC, scale=SCALE, search_offsets=0.2, blind=False))
    assert stop.asked is not None and time.perf_counter() - stop.asked < 2.0
    assert stop.looks < 50 + stop.n // 10     # throttled: the disk is not read at every hypothesis


def test_the_stop_is_looked_at_between_the_similarities_of_a_sparse_field(monkeypatch):
    """A field of few stars tries every similarity its triangles propose, thousands, without a progress between them."""
    o, gs, gm = _field(monkeypatch, n=40)
    det = np.c_[o["x"], o["y"]]
    stop = _Stop(5)
    log_odds = am.log_odds

    def counted(*a, **k):
        stop.step()
        return log_odds(*a, **k)
    monkeypatch.setattr(am, "log_odds", counted)
    monkeypatch.setattr(am, "STOP_EVERY", 0.0)
    with am._stoppable(SimpleNamespace(check_stop=stop.check_stop)), pytest.raises(Interrupted):
        am._match_sparse(gs, gm, RA, DEC, SCALE, W, H, det, 0.0, W * SCALE / 3600, H * SCALE / 3600, 0.15,
                         (True, False))
    assert stop.n == 5                       # the similarity after the request was not tried


def test_the_blind_solver_looks_at_the_stop_between_its_candidates():
    """Its candidates are verified one by one, thousands on a wide index: ``check`` before each."""
    from itertools import combinations

    from astropy.coordinates import SkyCoord
    from scipy.spatial import cKDTree

    from astro_gpu.core import blindsolve as bs
    rng = np.random.default_rng(2)
    ra, dec = rng.uniform(60, 70, 3000), rng.uniform(5, 15, 3000)
    index = object.__new__(bs.SkyIndex)       # a small index of its own: the cache of the computer is never read
    index.k = 6
    index.combos = np.array(list(combinations(range(index.k), 3)))
    index.ra, index.dec = ra, dec
    index.codes, index.quads = index._build()
    index.xyz = bs._unit(ra, dec)
    index.star_tree, index.code_tree = cKDTree(index.xyz), cKDTree(index.codes)
    w = am._tan(65.0, 10.0, 20.0, W, H)
    px, py = w.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
    ins = (px > 0) & (px < W) & (py > 0) & (py < H)
    xy = np.c_[px[ins], py[ins]][:60]
    calls = []

    def check():
        calls.append(1)
        if len(calls) == 3:
            raise Interrupted("stop requested")
    with pytest.raises(Interrupted):
        bs.solve(xy, W, H, index, check=check)
    assert len(calls) == 3
