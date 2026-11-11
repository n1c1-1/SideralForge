"""Issue #139: the guided search of Astrometry. A stop while the sky tiles come is a stop, not a failure; the bar does
not go back when the blind solver takes over; the shifted centres of the second pass do not match the given centre
again where the first pass did, and still take up the scale that match saw."""
from __future__ import annotations

from types import SimpleNamespace

import astroalign
import numpy as np
import pytest

from astro_gpu.core.stage import Interrupted, StageError
from astro_gpu.stages import astrometry as am
from astro_gpu.stages.astrometry import AstrometryParams

W, H, SCALE, RA, DEC = 3054, 2042, 2.9, 63.43, 12.0


def _catalogue(n=4000, seed=3):
    """(SkyCoord, G) of ``n`` stars over 20 x 20 deg around the centre, brightest first."""
    from astropy.coordinates import SkyCoord
    rng = np.random.default_rng(seed)
    ra = RA + rng.uniform(-10, 10, n) / np.cos(np.radians(DEC))
    dec = DEC + rng.uniform(-10, 10, n)
    return SkyCoord(ra, dec, unit="deg"), np.sort(rng.uniform(6, 12, n))


def _run(tmp_path, monkeypatch, gaia=None, stop_on=None, **par):
    """The stage on 80 detections (a field, not a sparse one): (progress events, logs). ``gaia``: in place of
    ``gaia_bright``; ``stop_on``: the progress messages that find the stop requested."""
    events, logs = [], []
    rng = np.random.default_rng(7)
    o = np.zeros(80, dtype=[("x", float), ("y", float), ("flux", float), ("flag", int)])
    o["x"], o["y"], o["flux"] = rng.uniform(0, W, 80), rng.uniform(0, H, 80), np.arange(80, 0, -1)
    gs, gm = _catalogue()
    monkeypatch.setattr(am, "detect", lambda g: o)
    monkeypatch.setattr(am, "gaia_bright", gaia or (lambda *a, **k: (gs, gm)))
    monkeypatch.setattr(am, "remember_instrument", lambda *a: None)
    monkeypatch.setattr(am, "mirror_hint", lambda *a: (True, False))
    monkeypatch.setattr(am, "focal_hints", lambda *a: [])

    def progress(frac, msg="", force=False):
        events.append((frac, msg))
        if stop_on and stop_on in msg:
            raise Interrupted("stop requested")
    ctx = SimpleNamespace(load_json=lambda stage, name: {"scale": 1.0},
                          load=lambda *a, **k: np.zeros((3, H, W), np.float32),
                          log=lambda m, level="info": logs.append(m), progress=progress, metric=lambda k, v: None,
                          out=lambda n: tmp_path / n, metric_of=lambda *a, **k: None)
    try:
        am.Astrometry().run(ctx, AstrometryParams(ra=RA, dec=DEC, **par))
    finally:
        _run.events, _run.logs = events, logs
    return events, logs


def _no_match(*a, **k):
    raise astroalign.MaxIterError("no match")


@pytest.mark.parametrize("scale", [SCALE, None])
def test_a_stop_while_the_tiles_come_is_a_stop(tmp_path, monkeypatch, scale):
    """It was a failure that blamed the network (scale known), or a trial scale said to have no stars (unknown)."""
    asked = []

    def gaia(ra, dec, radius, glim, log=None, progress=None):
        asked.append(glim)
        progress(0.0, "Gaia stars: tile 1 of 12")
        raise AssertionError("the stop was not checked")
    monkeypatch.setattr(astroalign, "find_transform", _no_match)
    with pytest.raises(Interrupted):
        _run(tmp_path, monkeypatch, gaia=gaia, stop_on="tile", scale=scale, blind=False)
    assert len(asked) == 1 and not any(m.startswith("no Gaia stars") for m in _run.logs)


def test_the_bar_does_not_go_back_when_the_blind_solver_takes_over(tmp_path, monkeypatch):
    from astro_gpu.core import blindsolve
    monkeypatch.setattr(astroalign, "find_transform", _no_match)
    monkeypatch.setattr(blindsolve, "solve_any", lambda *a, **k: None)
    with pytest.raises(StageError, match="could not be identified"):
        _run(tmp_path, monkeypatch, scale=None, search_offsets=0.0, blind=True)
    fr = [f for f, _ in _run.events]
    assert fr[-1] == pytest.approx(0.6) and all(b >= a for a, b in zip(fr, fr[1:]))


def test_the_second_pass_does_not_match_the_given_centre_again(tmp_path, monkeypatch):
    """The scale known: the given centre once (two mirrors), then the 8 shifted centres (two mirrors each)."""
    calls = []

    def find(q, gpos, max_control_points=50):
        calls.append(1)
        raise astroalign.MaxIterError("no match")
    monkeypatch.setattr(astroalign, "find_transform", find)
    with pytest.raises(StageError, match="match no star of the catalogue"):
        _run(tmp_path, monkeypatch, scale=SCALE, search_offsets=0.2, blind=False)
    assert len(calls) == 2 + 8 * 2


def test_the_second_pass_still_tries_the_scale_the_centre_saw(tmp_path, monkeypatch):
    """The scale unknown: the given centre's stars match at twice the first trial scale, so that scale is tried next,
    in the first pass and, with the shifted centres, in the second, as when the second pass matched the centre again."""
    first, hits = [], []

    def find(q, gpos, max_control_points=50):
        key = (tuple(np.round(q[0], 6)), tuple(np.round(gpos[0], 6)), max_control_points)
        if not first:
            first.append(key)
        if key == first[0]:
            hits.append(key)
            return SimpleNamespace(scale=2.0), (np.zeros((am.GOOD_PAIRS + 5, 2)), None)
        raise astroalign.MaxIterError("no match")
    monkeypatch.setattr(astroalign, "find_transform", find)
    with pytest.raises(StageError, match="match no star of the catalogue"):
        _run(tmp_path, monkeypatch, scale=None, search_offsets=0.2, blind=False)
    trials = [m for _, m in _run.events if m.startswith("searching the field at ")]
    s0 = float(trials[0].split(" at ")[1].split('"')[0])
    shifted = [m for m in trials if m.endswith("(shifted centres)")]
    assert trials[1].startswith(f"searching the field at {2 * s0:.2f}")
    assert shifted[0].startswith(f"searching the field at {s0:.2f}") and shifted[1].startswith(
        f"searching the field at {2 * s0:.2f}")
    assert len(hits) == 1                       # the given centre at the first scale, matched once only
