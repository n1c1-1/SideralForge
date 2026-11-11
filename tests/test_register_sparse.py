"""Issue #46: a sparse field (a few dozen stars per frame) aligned from triangles of stars, a frame that matches nothing
rejected, and a field with enough stars left on the usual path."""
from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from astro_gpu.core.stage import StageError
from astro_gpu.stages import register as rg
from astro_gpu.stages.register import apply_poly, fit_sparse

W, H = 3054, 2042


def _sky(n, seed=0):
    rng = np.random.default_rng(seed)
    flux = np.sort(rng.pareto(1.5, n) + 1)[::-1] * 1000
    return np.c_[rng.uniform([0, 0], [W, H], (n, 2)), flux, np.full(n, 2.0)]


def _frame(ref, ang=0.0, shift=(0.0, 0.0), scale=1.0, seed=1, keep=1.0):
    """The reference's stars seen by a frame: the sky turned about the centre, scaled and shifted, a share of the stars
    missed, the positions noisy."""
    rng = np.random.default_rng(seed)
    t = np.deg2rad(ang)
    c = np.array([W / 2, H / 2])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    xy = (ref[:, :2] - c) @ R.T * scale + c + np.array(shift)
    xy += rng.normal(0, 0.25, xy.shape)
    inside = (xy[:, 0] > 0) & (xy[:, 0] < W) & (xy[:, 1] > 0) & (xy[:, 1] < H) & (rng.random(len(xy)) < keep)
    return np.c_[xy, ref[:, 2:]][inside], inside


@pytest.mark.parametrize("ang,shift,scale,n,keep", [(0.0, (30.0, -18.0), 1.0, 53, 0.8), (0.2, (-140.0, 66.0), 1.001, 40, 0.8),
                                                    (180.0, (12.0, 9.0), 1.0, 64, 0.7), (-31.0, (5.0, 5.0), 0.998, 30, 0.9)])
def test_a_field_of_a_few_dozen_stars_is_aligned(ang, shift, scale, n, keep):
    ref = _sky(n)
    cat, inside = _frame(ref, ang, shift, scale, keep=keep)
    r = fit_sparse(cat, ref, cKDTree(ref[:, :2]), W, H)
    assert r is not None
    coef, nmatch, rms, ratio = r
    assert nmatch >= 0.8 * len(cat) and rms < 0.6 and abs(ratio - 1) < 0.01
    assert not coef[:, 3:].any()                             # rotation, shift and scale only
    x, y = apply_poly(coef, cat[:, 0], cat[:, 1], W, H)
    assert np.median(np.hypot(x - ref[inside, 0], y - ref[inside, 1])) < 0.5


def test_the_same_catalogues_give_the_same_transform():
    ref = _sky(50)
    cat, _ = _frame(ref, 0.1, (40.0, 20.0))
    tree = cKDTree(ref[:, :2])
    a, b = fit_sparse(cat, ref, tree, W, H), fit_sparse(cat, ref, tree, W, H)
    assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]


def test_a_mirror_image_is_not_aligned():
    """The triangles' shapes do not see a mirror: the similarity, without one, must not fit it."""
    ref = _sky(50)
    cat = ref.copy()
    cat[:, 0] = W - cat[:, 0]
    assert fit_sparse(cat, ref, cKDTree(ref[:, :2]), W, H) is None


@pytest.mark.parametrize("n", [50, 150])
def test_stars_that_match_nothing_are_rejected(n):
    ref = _sky(60)
    assert fit_sparse(_sky(n, seed=7), ref, cKDTree(ref[:, :2]), W, H) is None


def test_too_few_stars_to_check_a_transform_are_rejected():
    ref = _sky(50)
    cat, _ = _frame(ref)
    assert fit_sparse(cat[:rg.SPARSE_MIN_MATCHES - 1], ref, cKDTree(ref[:, :2]), W, H) is None


# ---------------------------------------------------------------------------------------- the stage


def _run(tmp_path, cats, monkeypatch):
    """Register.run on star catalogues already measured (its checkpoint), on a small calibrated cache."""
    n = len(cats)
    H_, W_ = 8, 8
    cache = tmp_path / "lights.f16"
    np.memmap(cache, dtype=np.float16, mode="w+", shape=(n, H_, W_)).flush()
    info = {"path": str(cache), "n": n, "height": H, "width": W, "cfa": "MONO"}
    monkeypatch.setattr(rg, "phase_shift", lambda *a, **k: None)
    monkeypatch.setattr(rg, "memory_short", lambda *a: 1.0)          # plain reads: no map of the small cache made
    monkeypatch.setattr(rg, "quality_plot", lambda *a, **k: np.zeros((4, 4, 3)))
    ck = tmp_path / "ck"
    ck.mkdir()
    psf = [(np.nan, np.nan, np.nan, 0)] * n
    fwhm, nan = np.full(n, 5.0), np.full(n, np.nan)
    (ck / "catalogs.pkl").write_bytes(pickle.dumps((cats, fwhm, np.ones(n), nan, nan, psf)))
    logs, metrics = [], {}
    meta = {"lights": [{"path": f"f{i}.fits", "time": None} for i in range(n)]}
    ctx = SimpleNamespace(ckpt=ck, load_json=lambda *a: info if a == ("calibrate", "cache.json") else meta,
                          progress=lambda *a, **k: None,
                          log=lambda m, level="info": logs.append((level, m)),
                          metric=lambda k, v: metrics.__setitem__(k, v), preview=lambda *a, **k: None,
                          out=lambda name: tmp_path / name)
    rg.Register().run(ctx, rg.RegisterParams(workers=1))
    return np.load(tmp_path / "reg.npz"), logs, metrics


def _aperture(cat):
    return np.c_[cat, np.full((len(cat), len(rg.APERTURES)), np.nan)]


def test_a_sparse_session_aligns_and_says_what_it_rejected(tmp_path, monkeypatch):
    sky = _sky(28)                                           # the polynomial's 30 matches are out of reach
    cats = [_aperture(_frame(sky, 0.05 * i, (10.0 * i, -6.0 * i), seed=i, keep=0.85)[0]) for i in range(1, 7)]
    cats.append(_aperture(_sky(50, seed=99)))                # not this field, and out of the reference's mid-session
    z, logs, metrics = _run(tmp_path, cats, monkeypatch)
    ok = np.isfinite(z["rms"])
    assert metrics["aligned"] == 6 and not ok[6] and ok.sum() == 6
    assert float(np.median(z["rms"][ok])) < 0.6
    said = [m for _, m in logs]
    assert any(m.startswith("sparse field: ") for m in said)
    assert ("warning", "1 frame(s) could not be aligned and will be ignored, e.g. f6.fits") in logs


def test_a_field_with_enough_stars_stays_on_the_usual_path(tmp_path, monkeypatch):
    """Its results are those of the vote and the polynomial; a frame they reject stays rejected."""
    sky = _sky(600)
    cats = [_aperture(_frame(sky, 0.0, (9.0 * i, 4.0 * i), seed=i)[0]) for i in range(5)]
    cats.append(_aperture(_sky(20, seed=5)))                  # few stars of another field: rejected as before
    def never(*a, **k):
        raise AssertionError("the sparse path is for sparse fields only")
    monkeypatch.setattr(rg, "fit_sparse", never)
    z, logs, metrics = _run(tmp_path, cats, monkeypatch)
    assert metrics["aligned"] == 5 and not np.isfinite(z["rms"][5])
    assert z["coef"][np.isfinite(z["rms"]), :, 3:].any()      # a cubic, as always
    assert not any(m.startswith("sparse field") for _, m in logs)


def test_a_frame_of_another_field_cannot_stay_the_reference(tmp_path, monkeypatch):
    """Two frames of another field, with the most stars and sharpest, in the mid-session the reference is drawn from
    (the two 75 s frames of the bench set, 23.5 degrees away): the reference is a frame of the session's field."""
    sky, other = _sky(28), _sky(120, seed=99)
    cats = [_aperture(_frame(sky, 0.05 * i, (10.0 * i, -6.0 * i), seed=i, keep=0.85)[0]) for i in range(1, 11)]
    cats[4] = _aperture(_frame(other, 0.0, (5.0, 5.0), seed=50)[0])
    cats[5] = _aperture(_frame(other, 0.0, (9.0, 2.0), seed=51)[0])
    assert rg.reference_order(np.array([len(c) for c in cats]), np.full(10, 5.0))[0] in (4, 5)
    z, logs, metrics = _run(tmp_path, cats, monkeypatch)
    assert int(z["ref"]) not in (4, 5) and metrics["aligned"] == 8 and not np.isfinite(z["rms"][[4, 5]]).any()
    assert any(m.startswith("reference frame: f") and "(another field?)" in m and m.endswith(" is used") for _, m in logs)


def test_a_session_aligned_on_its_first_reference_tries_no_other(tmp_path, monkeypatch):
    """The reference an ordinary session always had, and one pass over its frames: nothing is added to it."""
    sky = _sky(600)
    cats = [_aperture(_frame(sky, 0.0, (9.0 * i, 4.0 * i), seed=i)[0]) for i in range(7)]
    calls = []
    real = rg.fit_frame
    monkeypatch.setattr(rg, "fit_frame", lambda *a, **k: calls.append(1) or real(*a, **k))
    nst = np.array([len(c) for c in cats])
    rich = np.where(nst >= np.percentile(nst, 50))[0]
    mid = rich[(rich > 7 * 0.3) & (rich < 7 * 0.7)]
    today = int((mid if len(mid) else rich)[np.argmin(np.full(7, 5.0)[mid if len(mid) else rich])])
    z, logs, _ = _run(tmp_path, cats, monkeypatch)
    assert int(z["ref"]) == today and len(calls) == 7
    assert not any(m.startswith("reference frame:") for _, m in logs)


def test_no_more_references_are_tried_than_the_limit(tmp_path, monkeypatch):
    """Every frame its own field: the session is not aligned, and the search stops."""
    cats = [_aperture(_sky(20, seed=100 + i)) for i in range(12)]
    tried = []
    real = rg.fit_sparse
    monkeypatch.setattr(rg, "fit_sparse", lambda c, r, *a, **k: tried.append(id(r)) or real(c, r, *a, **k))
    z, _, metrics = _run(tmp_path, cats, monkeypatch)
    assert len(set(tried)) == rg.REFERENCE_TRIES and metrics["aligned"] == 1


def test_frames_with_hardly_a_star_still_stop_the_run(tmp_path, monkeypatch):
    sky = _sky(4)
    cats = [_aperture(_frame(sky, seed=i)[0]) for i in range(5)]
    with pytest.raises(StageError, match="too few stars detected per frame"):
        _run(tmp_path, cats, monkeypatch)
