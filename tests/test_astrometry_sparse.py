"""Issue #46: a stack of a few dozen stars is solved on the catalogue, the solution kept only when the stars prove it
(a Bayes factor of 1e9, Lang et al. 2010); a wrong place of the sky is refused and the run says so."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from scipy.spatial import cKDTree

from astro_gpu.core.stage import StageError
from astro_gpu.stages import astrometry as am
from astro_gpu.stages.astrometry import AstrometryParams, log_odds

W, H, SCALE, RA, DEC = 3054, 2042, 0.58, 63.43, 12.0
FW, FH = W * SCALE / 3600, H * SCALE / 3600


def _catalogue(n=320, seed=3):
    """(SkyCoord, G) of ``n`` stars over a patch wider than the frame, brightest first."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform([-0.15 * W, -0.15 * H], [1.15 * W, 1.15 * H], (n, 2))
    s = am._tan(RA, DEC, SCALE, W, H).pixel_to_world(xy[:, 0], xy[:, 1])
    return s, np.sort(rng.uniform(9, 15.4, n))


def _stars(gs, n=37, ang=33.0, shift=(25.0, -40.0), seed=5, mirror=True):
    """The ``n`` brightest catalogue stars in the frame as a stack sees them: turned, shifted, mirrored (the stacks are
    stored top row first), a little noise; positions in the stack's pixels."""
    px, py = am._tan(RA, DEC, SCALE, W, H).world_to_pixel(gs)
    z = (px - W / 2) + 1j * (py - H / 2)
    z = z * np.exp(1j * np.deg2rad(ang)) + W / 2 + H / 2 * 1j + complex(*shift)
    xy = np.c_[z.real, z.imag] + np.random.default_rng(seed).normal(0, 0.3, (len(z), 2))
    if mirror:
        xy[:, 1] = H - 1 - xy[:, 1]
    inside = (xy[:, 0] > 0) & (xy[:, 0] < W) & (xy[:, 1] > 0) & (xy[:, 1] < H)
    return xy[inside][:n]


def _odds(pred, cat, s=1.0 + 0j, t=0j, used=(0, 1, 2)):
    return log_odds(pred, cat, s, t, cKDTree(np.c_[cat.real, cat.imag]), W, H, list(used))


def test_stars_on_the_catalogue_reach_the_published_odds_and_stars_elsewhere_do_not():
    rng = np.random.default_rng(1)
    cat = rng.uniform(0, W, 200) + 1j * rng.uniform(0, H, 200)
    on = cat[:30] + rng.normal(0, 0.3, 30) + 1j * rng.normal(0, 0.3, 30)
    lo, nm = _odds(on, cat)
    assert lo > np.log(am.ODDS_TO_SOLVE) and nm == 30
    away = rng.uniform(0, W, 30) + 1j * rng.uniform(0, H, 30)
    assert _odds(away, cat)[0] < 0


def test_the_stars_of_the_triangle_prove_nothing():
    """A similarity always fits the triangle that proposes it: three stars alone cannot give a solution."""
    rng = np.random.default_rng(2)
    cat = rng.uniform(0, W, 200) + 1j * rng.uniform(0, H, 200)
    assert _odds(cat[:3].copy(), cat)[0] == 0.0
    assert _odds(cat[:5].copy(), cat)[0] < np.log(am.ODDS_TO_SOLVE)   # and two more are not enough


def test_a_catalogue_star_counts_once():
    """Stars piled on one catalogue star (a transform that shrinks the field) are one match, the others weigh against."""
    cat = np.array([500 + 500j, 900 + 300j, 2000 + 1500j])
    pred = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j, *([500 + 500j] * 20)])
    lo, nm = _odds(pred, cat)
    assert nm == 3 + 1 and lo < 0


def test_a_few_dozen_stars_are_found_on_the_catalogue():
    gs, gm = _catalogue()
    det = _stars(gs)
    r = am._match_sparse(gs, gm, RA, DEC, SCALE, W, H, det, 0.0, FW, FH, 0.15, (True, False))
    assert r is not None
    flip, T, nm, _, lo = r
    assert flip is True and nm >= 30 and lo > np.log(am.ODDS_TO_SOLVE) and abs(T.scale - 1) < 0.01


def test_the_same_stars_give_the_same_solution():
    gs, gm = _catalogue()
    det = _stars(gs)
    a, b = (am._match_sparse(gs, gm, RA, DEC, SCALE, W, H, det, 0.0, FW, FH, 0.15, (True, False)) for _ in range(2))
    assert np.array_equal(a[1].params, b[1].params) and a[4] == b[4]


@pytest.mark.parametrize("n", [12, 37])
def test_another_part_of_the_sky_is_refused(n):
    gs, gm = _catalogue(seed=11)
    det = _stars(_catalogue(seed=12)[0], n=n)
    assert am._match_sparse(gs, gm, RA, DEC, SCALE, W, H, det, 0.0, FW, FH, 0.15, (True, False)) is None


# ---------------------------------------------------------------------------------------- the stage


def _stage(tmp_path, monkeypatch, det, gs, gm, **par):
    logs, metrics = [], {}
    o = np.zeros(len(det), dtype=[("x", float), ("y", float), ("flux", float), ("flag", int)])
    o["x"], o["y"], o["flux"] = det[:, 0], det[:, 1], np.arange(len(det), 0, -1)
    monkeypatch.setattr(am, "detect", lambda g: o)
    monkeypatch.setattr(am, "gaia_bright", lambda *a, **k: (gs, gm))
    monkeypatch.setattr(am, "remember_instrument", lambda *a: None)
    monkeypatch.setattr(am, "mirror_hint", lambda *a: (True, False))
    meta = {"scale": 1.0}

    def load_json(stage, name):
        return meta

    ctx = SimpleNamespace(load_json=load_json, load=lambda *a, **k: np.zeros((3, H, W), np.float32),
                          log=lambda m, level="info": logs.append(m), progress=lambda *a, **k: None,
                          metric=lambda k, v: metrics.__setitem__(k, v), out=lambda n: tmp_path / n,
                          metric_of=lambda *a, **k: None)
    am.Astrometry().run(ctx, AstrometryParams(ra=RA, dec=DEC, scale=SCALE, blind=False, search_offsets=0.0, **par))
    return logs, metrics


def test_the_stage_solves_a_sparse_field_without_a_distortion_polynomial(tmp_path, monkeypatch):
    gs, gm = _catalogue()
    logs, metrics = _stage(tmp_path, monkeypatch, _stars(gs), gs, gm)
    assert any(m.startswith("sparse field: the solution holds ") for m in logs)
    assert metrics["stars_used"] >= 25 and metrics["residual_px"] < 0.5
    assert abs(metrics["scale_arcsec"] - SCALE) < 0.01
    h = fits.getheader(tmp_path / "wcs.fits")
    assert "A_ORDER" not in h and h["CTYPE1"] == "RA---TAN"


def test_the_stage_gives_no_solution_rather_than_a_wrong_one(tmp_path, monkeypatch):
    gs, gm = _catalogue(seed=11)
    det = _stars(_catalogue(seed=12)[0])
    with pytest.raises(StageError, match="do not identify it on the sky beyond doubt"):
        _stage(tmp_path, monkeypatch, det, gs, gm)
    assert not (tmp_path / "wcs.fits").exists()


def test_hardly_a_star_is_said_with_the_real_minimum(tmp_path, monkeypatch):
    gs, gm = _catalogue()
    with pytest.raises(StageError, match="only 5 stars found in the stack, at least 6 are needed"):
        _stage(tmp_path, monkeypatch, _stars(gs, n=5), gs, gm)


def test_a_field_with_enough_stars_stays_on_the_usual_match(tmp_path, monkeypatch):
    gs, gm = _catalogue()
    det = _stars(gs, n=60)
    assert len(det) >= am.SPARSE_FIELD

    def never(*a, **k):
        raise AssertionError("the sparse solve is for sparse fields only")
    monkeypatch.setattr(am, "_match_sparse", never)
    monkeypatch.setattr(am, "_match", lambda *a, **k: None)
    with pytest.raises(StageError, match="match no star of the catalogue"):
        _stage(tmp_path, monkeypatch, det, gs, gm)


def test_the_catalogue_of_french_has_the_new_lines():
    from astro_gpu import texts
    fr = json.loads((texts.LOCALE / "fr" / "run.json").read_text(encoding="utf-8"))
    for key in ("the {0} stars of this sparse field do not identify it on the sky beyond doubt, so no solution is given "
                "rather than a wrong one: take Astrometry out of this workflow",
                "sparse field: the solution holds {0} powers of ten over chance, 9 needed ({1} of {2} stars)",
                "reference frame: {0} aligned too few of the frames (another field?), {1} is used",
                "only {0} stars found in the stack, at least {1} are needed to identify the field: add light frames, or "
                "take Astrometry out of this workflow"):
        assert key in fr
