"""Issue #139: the guided search of Astrometry. A stop while the sky tiles come is a stop, not a failure; the bar does
not go back when the blind solver takes over; the shifted centres of the second pass do not match the given centre
again where the first pass did, and still take up the scale that match saw."""
from __future__ import annotations

from types import SimpleNamespace

import astroalign
import numpy as np
import pytest
from skimage.transform import SimilarityTransform

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
            return SimilarityTransform(scale=2.0), (np.zeros((am.GOOD_PAIRS + 5, 2)), None)
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


# ---------------------------------------------------------------------------------------- the lead of NGC 7000
# the run of 11 November: a Sony A7S (4256 x 2848, 8.4 um), focal length unknown, solved at 2.320"/px around RA 314.721
# Dec 44.331, 0.2 deg from NGC 7000 as the shipped catalogues give it; the stars of the 3.58"/px trial matched at once
# and said 2.31"/px, but the trial at that scale around the given centre, on 60 stars, was not certain

NW, NH, NSCALE = 4256, 2848, 2.3202
NRA, NDEC = 314.72086, 44.33148
NRA0, NDEC0 = 314.82142, 44.52878


def _ngc7000(seed, spur, nb, nf=6000):
    """(SkyCoord, G) of a catalogue and the stack's detections, brightest first: ``nb`` stars of G 6 to 11 in the
    image (the bright stars of the 3.58"/px trial's frame are those of the image, as its match says), ``nf`` of G 11
    to 12 over 20 x 20 deg; the image turned by 14.5 deg and mirrored, its brightness 0.1 mag off the catalogue's,
    a share ``spur`` of the 150 brightest detections false (nebula, star halos)."""
    from astropy.coordinates import SkyCoord
    rng = np.random.default_rng(seed)
    w = am._tan(NRA, NDEC, NSCALE, NW, NH)
    xb = rng.uniform([0, 0], [NW, NH], (nb, 2))
    sb = w.pixel_to_world(xb[:, 0], xb[:, 1])
    gb = 11 - np.log10(1 - rng.uniform(0, 1 - 10 ** (-0.35 * 5), nb)) / -0.35
    raf = NRA0 + rng.uniform(-10, 10, nf) / np.cos(np.radians(NDEC0))
    decf = NDEC0 + rng.uniform(-10, 10, nf)
    ra, dec, g = np.r_[sb.ra.deg, raf], np.r_[sb.dec.deg, decf], np.r_[gb, rng.uniform(11, 12, nf)]
    o = np.argsort(g)
    gs, g = SkyCoord(ra[o], dec[o], unit="deg"), g[o]
    px, py = w.world_to_pixel(gs)
    z = ((px - NW / 2) + 1j * (py - NH / 2)) * np.exp(1j * np.deg2rad(14.5)) + NW / 2 + NH / 2 * 1j
    xy = np.c_[z.real, z.imag] + rng.normal(0, 0.3, (len(g), 2))
    xy[:, 1] = NH - 1 - xy[:, 1]
    ins = (xy[:, 0] > 0) & (xy[:, 0] < NW) & (xy[:, 1] > 0) & (xy[:, 1] < NH)
    star = xy[ins][np.argsort(g[ins] + rng.normal(0, 0.1, ins.sum()))]
    ns = int(spur * 150)
    top = np.r_[star[:150 - ns], rng.uniform([0, 0], [NW, NH], (ns, 2))][rng.permutation(150)]
    return gs, g, np.r_[top, star[150 - ns:]]


NGC7000 = dict(seed=1, spur=0.5, nb=400)       # a field whose 3.58"/px trial gives the lead (the first test below)


def test_the_lead_of_the_ngc7000_field_says_its_scale_and_centre():
    """The 3.58"/px trial around the given centre, 60 stars, says 2.32"/px and where the field is, to a few
    hundredths of a degree; matched there with the 100 stars of a known scale, it is certain."""
    gs, gm, det = _ngc7000(**NGC7000)
    seen = []
    am._match(gs, gm, NRA0, NDEC0, 3.58, NW, NH, det[:150], 0.0, NW * 3.58 / 3600, NH * 3.58 / 3600, astroalign, 0.13,
              (True, False), seen, mcp=60)
    s2, ra2, dec2, flip = seen[0]
    assert abs(s2 / NSCALE - 1) < 0.01 and abs(ra2 - NRA) < 0.01 and abs(dec2 - NDEC) < 0.01
    r = am._match(gs, gm, ra2, dec2, s2, NW, NH, det[:150], 0.0, NW * s2 / 3600, NH * s2 / 3600, astroalign, 0.13,
                  (flip, not flip), None, mcp=100)
    assert r is not None and r[2] >= am.GOOD_PAIRS


def test_a_match_that_says_the_scale_is_confirmed_there_at_once(tmp_path, monkeypatch):
    """The lead of the 3.58"/px trial is matched at its scale and centre, on the stars of a known scale, before any
    other scale or shifted centre: the field solved at the second match, where the run went on to 40"/px and 1 500 sky
    tiles before its second pass found it 22 min later. The trial at the lead's scale around the given centre on 60
    stars was not certain in the run (the search went on after it); a synthetic field whose 3.58"/px trial gives the
    lead makes it certain, so its pairs are cut to one short of certain, as the run's were."""
    gs, gm, det = _ngc7000(**NGC7000)
    match = am._match

    def as_in_the_run(gs_, gm_, ra0, dec0, sc, *a, mcp=100, **k):
        r = match(gs_, gm_, ra0, dec0, sc, *a, mcp=mcp, **k)
        if r is not None and mcp == 60 and abs(sc / NSCALE - 1) < 0.01 and (ra0, dec0) == (NRA0, NDEC0):
            r = (r[0], r[1], min(r[2], am.GOOD_PAIRS - 1), r[3])
        return r
    monkeypatch.setattr(am, "_match", as_in_the_run)
    o = np.zeros(len(det), dtype=[("x", float), ("y", float), ("flux", float), ("flag", int)])
    o["x"], o["y"], o["flux"] = det[:, 0], det[:, 1], np.arange(len(det), 0, -1)
    monkeypatch.setattr(am, "detect", lambda g: o)
    monkeypatch.setattr(am, "gaia_bright", lambda *a, **k: (gs, gm))
    monkeypatch.setattr(am, "remember_instrument", lambda *a: None)
    monkeypatch.setattr(am, "mirror_hint", lambda *a: (True, False))
    monkeypatch.setattr(am, "focal_hints", lambda *a: [])
    trials, logs, metrics = [], [], {}

    def progress(frac, msg="", force=False):
        if msg.startswith("searching the field at "):
            trials.append(msg)
            assert len(trials) <= 2, f"the search went on after the lead: {trials}"
    ctx = SimpleNamespace(load_json=lambda stage, name: {"scale": 1.0},
                          load=lambda *a, **k: np.zeros((3, NH, NW), np.float32),
                          log=lambda m, level="info": logs.append(m), progress=progress,
                          metric=lambda k, v: metrics.__setitem__(k, v), out=lambda n: tmp_path / n,
                          metric_of=lambda stage, key, default=None: {"pixel_um": 8.4}.get(key, default))
    am.Astrometry().run(ctx, AstrometryParams(ra=NRA0, dec=NDEC0, pixel_um=8.4, blind=False))
    assert trials[0] == 'searching the field at 3.58"/px' and trials[1].startswith('searching the field at 2.32"/px')
    assert any(m.startswith('stars matched at 3.58"/px trial: the image scale is about 2.32"/px') for m in logs)
    assert abs(metrics["scale_arcsec"] - NSCALE) < 0.002
    off = np.hypot((metrics["ra"] - NRA) * np.cos(np.radians(NDEC)), metrics["dec"] - NDEC) * 3600
    assert off < 2 * NSCALE                     # the centre within 2 px (the mirror's half pixel, the stars' noise)
