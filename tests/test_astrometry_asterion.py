"""Astrometry's first solution by Asterion (stages/astrometry.py ``_first_solution``, core/asterion.py): given, the
scale search, the triangle match and the blind solver are skipped and the fine fit starts from it; none, not installed
or a program that cannot run, FLOAT's own search runs (one line, a warning when the program cannot run); a stop during
it is a stop. Without the program: ``asterion.status`` and ``asterion.solve`` stand in for it on a rendered field whose
catalogue stars and detections come from one WCS. The level 3 test runs the real program on a rendered star field."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astro_gpu.core import asterion
from astro_gpu.core.stage import Interrupted, StageError
from astro_gpu.stages import astrometry as am
from astro_gpu.stages.astrometry import AstrometryParams

W, H, SCALE, RA, DEC = 3000, 2000, 2.5, 97.9, 4.9
DATA = Path(__file__).with_name("data") / "asterion"


def _wcs(ra=RA, dec=DEC, scale=SCALE, rot=20.0, flipped=True):
    """A WCS of FLOAT's frame (pixel origin 0, rows as stored): stored top row first, its parity flipped."""
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [W / 2 + 1, H / 2 + 1]
    t = np.radians(rot)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    w.wcs.cd = scale / 3600 * (R if flipped else R @ np.diag([-1.0, 1.0]))
    return w


def _field(n=600, seed=5):
    """(catalogue SkyCoord, G, detections) of one field: the catalogue over twice the frame, its stars in the frame
    detected where the true WCS puts them, brightest first."""
    from astropy.coordinates import SkyCoord
    rng = np.random.default_rng(seed)
    w = _wcs()
    x, y = rng.uniform(-W / 2, 1.5 * W, n), rng.uniform(-H / 2, 1.5 * H, n)
    gs = w.pixel_to_world(x, y)
    gs = SkyCoord(gs.ra.deg, gs.dec.deg, unit="deg")
    gm = np.sort(rng.uniform(6, 14, n))
    ins = (x > 2) & (x < W - 3) & (y > 2) & (y < H - 3)
    o = np.zeros(int(ins.sum()), dtype=[("x", float), ("y", float), ("flux", float), ("flag", int)])
    o["x"], o["y"] = x[ins], y[ins]
    o["flux"] = 10 ** (-0.4 * gm[ins])
    return gs, gm, o


def _solution(w, matched=250):
    return asterion.Solution(wcs=w, ra=float(w.wcs.crval[0]), dec=float(w.wcs.crval[1]), scale=SCALE, rotation=20.0,
                             up_position_angle=200.0, flipped=True, matched=matched, detected=1000, rms_arcsec=0.3,
                             log_odds=500.0, index="tycho2", seconds=0.1, matches=np.zeros((0, 5)))


def _run(tmp_path, monkeypatch, solve, ok=True, missing=True, check_stop=None, **par):
    """The stage on the rendered field with ``solve`` in place of Asterion's (``ok``: installed): (logs, metrics,
    the WCS written)."""
    logs, metrics = [], {}
    gs, gm, o = _field()
    monkeypatch.setattr(am, "detect", lambda g: o)
    monkeypatch.setattr(am, "gaia_bright", lambda *a, **k: (gs, gm))
    monkeypatch.setattr(am, "remember_instrument", lambda *a: None)
    monkeypatch.setattr(am, "mirror_hint", lambda *a: (True, False))
    monkeypatch.setattr(am, "focal_hints", lambda *a: [])
    monkeypatch.setattr(asterion, "status", lambda *a, **k: asterion.Status(
        ok, why="" if ok else ("the Asterion plate solver is not installed" if missing else "Asterion 0.3.9 is too old"),
        missing=not ok and missing))
    monkeypatch.setattr(asterion, "solve", solve)
    ctx = SimpleNamespace(load_json=lambda stage, name: {"scale": 1.0},
                          load=lambda *a, **k: np.zeros((3, H, W), np.float32),
                          log=lambda m, level="info": logs.append((level, m)), progress=lambda *a, **k: None,
                          metric=metrics.__setitem__, out=lambda n: tmp_path / n, metric_of=lambda *a, **k: None,
                          scratch=tmp_path, check_stop=check_stop or (lambda: None))
    try:
        am.Astrometry().run(ctx, AstrometryParams(**{"ra": RA + 0.3, "dec": DEC - 0.2, **par}))
    finally:
        _run.logs, _run.metrics = logs, metrics
    return logs, metrics, _written(tmp_path / "wcs.fits")


def _written(path):
    import warnings

    from astropy.io import fits
    from astropy.wcs import WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # a header without its image: astropy says so
        return WCS(fits.getheader(path))


def _no_search(monkeypatch):
    def never(*a, **k):
        raise AssertionError("the search ran after Asterion's solution")
    for name in ("_match", "_match_sparse", "_blind"):
        monkeypatch.setattr(am, name, never)


def _far(w1, w2) -> float:
    """The largest distance (px) between the sky two WCS give at the corners and the centre."""
    pts = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1], [W / 2, H / 2]], float)
    return float(np.max(w1.pixel_to_world(pts[:, 0], pts[:, 1]).separation(
        w2.pixel_to_world(pts[:, 0], pts[:, 1])).arcsec)) / SCALE


@pytest.mark.parametrize("scale", [None, SCALE])
def test_asterions_solution_skips_the_search_and_the_fine_fit_starts_from_it(tmp_path, monkeypatch, scale):
    """Its solution 2 px off (a shift and a hair of rotation): no scale search, no triangle match, no blind solver;
    the fine fit on the catalogue lands on the true WCS; the target given as a limit with a radius, the scale when
    known; the line and the metrics."""
    _no_search(monkeypatch)
    asked = {}
    off = _wcs(ra=RA + 2 * SCALE / 3600, rot=20.02)

    def solve(image, work, **kw):
        asked.update(kw, shape=image.shape, work=work)
        return _solution(off)
    logs, metrics, w = _run(tmp_path, monkeypatch, solve, scale=scale)
    assert _far(w, _wcs()) < 0.05
    assert asked["shape"] == (H, W) and asked["work"] == tmp_path               # the plane the stage holds
    assert asked["ra"] == pytest.approx(RA + 0.3) and asked["radius"] == 10.0 and asked["scale"] == scale
    assert asked["check"] is not None and asked["timeout"] == 10.0
    assert metrics["first_solver"] == "asterion" and metrics["first_solution_s"] >= 0
    assert metrics["parity_flipped"] is True and metrics["scale_arcsec"] == pytest.approx(SCALE, rel=1e-3)
    said = [m for _lv, m in logs]
    assert any(m.startswith("first solution by Asterion in ") and m.endswith('2.500"/px, 250 stars matched')
               for m in said)
    assert not any(m.startswith(("triangle match", "searching", "Asterion")) for m in said)


def test_the_position_unknown_asterion_searches_the_whole_sky_when_blind_solving_is_on(tmp_path, monkeypatch):
    _no_search(monkeypatch)
    asked = {}

    def solve(image, work, **kw):
        asked.update(kw)
        return _solution(_wcs())
    _run(tmp_path, monkeypatch, solve, ra=None, dec=None)
    assert asked["ra"] is None and asked["radius"] is None and _run.metrics["first_solver"] == "asterion"


def test_the_position_unknown_and_blind_solving_off_asterion_is_not_asked(tmp_path, monkeypatch):
    def solve(*a, **k):
        raise AssertionError("Asterion asked for the whole sky with Blind solving off")
    with pytest.raises(StageError, match="field centre is unknown"):
        _run(tmp_path, monkeypatch, solve, ra=None, dec=None, blind=False)


def _searched(monkeypatch) -> list:
    calls = []

    def match(*a, **k):
        calls.append(1)
        return None
    monkeypatch.setattr(am, "_match", match)
    return calls


@pytest.mark.parametrize("case", ["none", "not installed", "cannot run", "fails"])
def test_without_a_solution_floats_own_search_runs_after_one_line(tmp_path, monkeypatch, case):
    """No solution, not installed: an information line; a program too old or failing: a warning; then the search."""
    calls = _searched(monkeypatch)

    def solve(*a, **k):
        if case == "fails":
            raise asterion.AsterionError("the Asterion plate solver failed (exit 2): No catalog is installed")
        return None
    with pytest.raises(StageError, match="match no star of the catalogue"):
        _run(tmp_path, monkeypatch, solve, ok=case in ("none", "fails"), missing=case == "not installed",
             scale=SCALE, blind=False)
    assert calls                                                                  # the search ran
    lines = [(lv, m) for lv, m in _run.logs if "Asterion" in m]
    assert len(lines) == 1
    level, msg = lines[0]
    assert msg.endswith(": FLOAT's own search finds the field")
    assert level == ("warning" if case in ("cannot run", "fails") else "info")
    assert {"none": "Asterion found no solution in ", "not installed": "Asterion is not installed",
            "cannot run": "Asterion cannot run (Asterion 0.3.9 is too old)",
            "fails": "Asterion cannot run (the Asterion plate solver failed"}[case] in msg
    assert _run.metrics["first_solver"] == "float"


def test_a_stop_while_asterion_runs_is_a_stop(tmp_path, monkeypatch):
    """The stage's check is handed to the program's watch: a stop raised there ends the stage, no search after it."""
    _no_search(monkeypatch)

    def stop():
        raise Interrupted("stop requested")

    def solve(image, work, check=None, **kw):
        check()
        raise AssertionError("the stop was not raised")
    with pytest.raises(Interrupted):
        _run(tmp_path, monkeypatch, solve, check_stop=stop)


def test_the_lines_are_in_french():
    from astro_gpu import texts
    from astro_gpu.client import i18n
    from astro_gpu.client.services import run
    fr = json.loads((texts.LOCALE / "fr" / "run.json").read_text(encoding="utf-8"))
    for key in ("first solution by Asterion", "first solution by Asterion in {0} s: {1}\"/px, {2} stars matched",
                "Asterion is not installed: FLOAT's own search finds the field",
                "Asterion found no solution in {0} s: FLOAT's own search finds the field",
                "Asterion cannot run ({0}): FLOAT's own search finds the field"):
        assert key in fr
    old = i18n.language()
    try:
        i18n.set_language("fr")
        assert run.engine_text('first solution by Asterion in 0.6 s: 1.926"/px, 308 stars matched') == \
            "première solution par Asterion en 0,6 s : 1,926\"/px, 308 étoiles appariées"
    finally:
        i18n.set_language(old)


# ------------------------------------------------------------------ level 3: the real program in the stage

@pytest.mark.level3
def test_real_the_stage_starts_from_asterions_solution(tmp_path, monkeypatch):
    """The stage on a field rendered from the Heart's Tycho-2 stars (tests/data/asterion), those stars standing for the
    catalogue of the fine fit: Asterion solves it, the search is skipped, the solution is the rendering's."""
    import os

    from astropy.coordinates import SkyCoord
    if not asterion.status().ok:
        pytest.skip(f"Asterion is not installed here: {asterion.status().why}")
    m = asterion.read_matches(DATA / "heart.matches.csv")
    w, h, sc = 1500, 1000, 20.0
    true = _wcs(ra=40.75, dec=61.12, scale=sc, rot=30.0)
    true.wcs.crpix = [w / 2 + 1, h / 2 + 1]
    gs = SkyCoord(m[:, 2], m[:, 3], unit="deg")
    x, y = true.world_to_pixel(gs)
    keep = (x > 3) & (x < w - 4) & (y > 3) & (y < h - 4)
    rng = np.random.default_rng(3)
    img = rng.normal(100, 2, (h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for k, (sx, sy) in enumerate(zip(x[keep], y[keep])):
        sl = np.s_[int(sy) - 6:int(sy) + 7, int(sx) - 6:int(sx) + 7]
        img[sl] += 4000.0 * 0.985 ** k * np.exp(-((xx[sl] - sx) ** 2 + (yy[sl] - sy) ** 2) / (2 * 1.3 ** 2))
    _no_search(monkeypatch)
    monkeypatch.setattr(am, "gaia_bright", lambda *a, **k: (gs, np.linspace(6, 11, len(gs))))
    monkeypatch.setattr(am, "remember_instrument", lambda *a: None)
    logs, metrics = [], {}
    ctx = SimpleNamespace(load_json=lambda stage, name: {"scale": 1.0}, load=lambda *a, **k: np.stack([img] * 3),
                          log=lambda msg, level="info": logs.append(msg), progress=lambda *a, **k: None,
                          metric=metrics.__setitem__, out=lambda n: tmp_path / n, metric_of=lambda *a, **k: None,
                          scratch=tmp_path / "scratch", check_stop=lambda: None)
    (tmp_path / "scratch").mkdir()
    am.Astrometry().run(ctx, AstrometryParams(ra=41.0, dec=61.0))
    got = _written(tmp_path / "wcs.fits")
    gx, gy = got.world_to_pixel(gs[keep])
    assert np.median(np.hypot(gx - x[keep], gy - y[keep])) < 0.3, logs
    assert metrics["first_solver"] == "asterion" and metrics["scale_arcsec"] == pytest.approx(sc, rel=2e-3)
    assert os.listdir(tmp_path / "scratch") == []
