"""Frame selection of stacking v2 (astro_gpu/core/frame_weights.py), on synthetic nights.

A "steady night" spreads every measure on the quantiles of a Gaussian (no frame beyond 2.2 sigma for 40 frames), so
the k = 2.5 filters must keep everything; then a cloud passage, wind, and frames whose weight is under the minimum.
The weights are given (the stack's own, stages/stack.quality_weights); the selection never changes them.
"""
import json
from statistics import NormalDist

import numpy as np
import pytest

from astro_gpu.core import frame_weights as fwm
from astro_gpu.core.frame_weights import (
    K_SIGMA_FILTERS,
    MIN_WEIGHT,
    NO_FILTERS,
    REASON_TEMPLATES,
    Filter,
    parse_filter,
    planes_to_colours,
    robust_sigma,
    select,
)

N = 40


def gauss(n=N, seed=0):
    """The n Gaussian quantiles, shuffled: a sample with no outlier (max 2.24 sigma for 40)."""
    z = np.array([NormalDist().inv_cdf((i + 0.5) / n) for i in range(n)])
    return np.random.default_rng(seed).permutation(z)


def night(n=N):
    """Transparency, noise (ADU), FWHM (px), roundness, sky level of a steady night: independent spreads."""
    return dict(transparency=1.0 + 0.02 * gauss(n, 1), noise=10.0 + 0.3 * gauss(n, 2), fwhm=3.0 + 0.15 * gauss(n, 3),
                roundness=0.9 + 0.02 * gauss(n, 4), background=500.0 + 10.0 * gauss(n, 5))


def weights(m):
    """The stack's weight of each frame (stages/stack.quality_weights' formula, the best = 1)."""
    w = (m["transparency"] / m["noise"]) ** 2 * (np.percentile(m["fwhm"], 5) / m["fwhm"]) ** 2
    return w / w.max()


def sel(m, **kw):
    return select(weights(m), fwhm=m["fwhm"], roundness=m["roundness"], transparency=m["transparency"],
                  background=m["background"], **kw)


# ------------------------------------------------------------------------------------------------- the default
def test_the_default_keeps_every_frame_of_a_steady_night_with_its_weight():
    m = night()
    fs = sel(m)
    assert fs.kept.all() and not fs.exclusions
    np.testing.assert_allclose(fs.weight, weights(m), rtol=0, atol=0)      # the weights are never changed
    assert list(fs.reasons) == [""] * N
    assert fs.suggestion.value == N and fs.suggestion.reason == f"all {N} aligned frames kept"
    assert all(fs.criteria[c].status == "off" for c in fwm.CRITERIA)        # no filter by default (Siril, PixInsight)
    assert fs.auto["min_weight"].value == MIN_WEIGHT == 0.05                # WBPP's minimum weight


def cloudy():
    m = night()
    m["transparency"][10:14] = [0.6, 0.4, 0.3, 0.55]
    return m


def test_the_default_keeps_the_clouded_frames_and_leaves_out_only_those_under_the_minimum_weight():
    m = cloudy()
    m["transparency"][13] = 0.2                                    # weight about 0.04 of the best: under 0.05
    fs = sel(m)
    assert sorted(r.frame for r in fs.exclusions) == [13]
    assert fs.codes[13] == "weight" and fs.reasons[13].endswith("under the minimum of 5 %")
    pct = 100 * weights(m)[13] / weights(m).max()
    assert pct - 0.1 < fs.exclusions[0].args["pct"] <= pct                 # rounded down to a tenth
    assert fs.kept[12] and fs.weight[12] == weights(m)[12] and fs.relative[12] > 0.05     # 0.3: about 0.09, kept
    assert fs.weight[13] == 0 and fs.relative[13] == 0
    assert fs.auto["min_weight"].reason == "1 frame(s) weigh less than 5 % of the best one: excluded"
    assert fs.suggestion.reason == f"{N - 1} of {N} aligned frames kept"
    assert select(weights(m), min_weight=0.0).kept.all()


def test_the_minimum_weight_is_relative_to_the_best_frame_still_in():
    """WBPP normalises the weights of the frames it integrates: a best frame excluded by hand or filtered out does
    not set the scale."""
    w = np.full(10, 0.04)
    w[0] = 1.0
    fs = select(w)
    assert fs.kept.sum() == 1 and fs.kept[0]
    ex = np.zeros(10, bool)
    ex[0] = True
    fs = select(w, exclude=ex)                                     # the others are now the best: all kept
    assert fs.kept[1:].all() and not fs.kept[0] and fs.reasons[0] == "excluded by hand"
    np.testing.assert_allclose(fs.relative[1:], 1.0)


def test_failed_registrations_and_missing_weights_are_left_out_with_a_reason():
    m = night()
    aligned = np.ones(N, bool)
    aligned[3] = False
    w = weights(m)
    w[3] = 0.0                                                     # quality_weights gives 0 to a frame not aligned
    w[5] = np.nan
    fs = select(w, aligned=aligned)
    assert fs.codes[3] == "not_aligned" and fs.reasons[3] == "not aligned"
    assert fs.codes[5] == "no_measure" and fs.reasons[5] == "no measure of its weight"
    assert fs.kept.sum() == N - 2


# ------------------------------------------------------------------------------------------------- the k x sigma option
def test_k_sigma_filters_exclude_a_cloud_passage_with_its_reason():
    m = cloudy()
    fs = sel(m, filters=K_SIGMA_FILTERS)
    out = sorted({r.frame for r in fs.exclusions})
    assert out == [10, 11, 12, 13]
    assert all(c == "transparency" for c in fs.codes[10:14])
    assert fs.reasons[12].startswith("transparency ") and fs.reasons[12].endswith("(30 % of it): clouds or haze?")
    sig = fs.exclusions[[r.frame for r in fs.exclusions].index(12)].args["sigma"]
    med, s = robust_sigma(m["transparency"])
    assert sig == pytest.approx((med - 0.3) / s)
    keep = np.ones(N, bool)
    keep[10:14] = False
    np.testing.assert_allclose(fs.weight[keep], weights(m)[keep])
    assert (fs.weight[~keep] == 0).all()
    assert fs.suggestion.reason == f"{N - 4} of {N} aligned frames kept"
    assert fs.auto["filter_transparency"].reason.startswith("4 frame(s) more than 2.5 σ below the night's median")


def windy():
    m = night()
    m["fwhm"][25:28] *= 1.6
    m["roundness"][25:28] = 0.6
    return m


def test_k_sigma_filters_exclude_the_frames_with_large_stretched_stars():
    m = windy()
    fs = sel(m, filters=K_SIGMA_FILTERS)
    assert sorted({r.frame for r in fs.exclusions}) == [25, 26, 27]
    assert list(fs.codes[25:28]) == ["fwhm,roundness"] * 3
    assert "wind, focus or seeing?" in fs.reasons[25] and "wind or tracking?" in fs.reasons[25]
    med = np.median(m["fwhm"])
    assert fs.reasons[25].startswith("stars ") and \
        f" σ larger than the night's median ({m['fwhm'][25]:.1f} px for {med:.1f} px)" in fs.reasons[25]
    # the default (no filter) keeps them, with the lower weight their stars give them
    d = sel(m)
    assert d.kept[25:28].all() and (d.weight[25:28] < np.median(d.weight)).all()


def test_steady_night_with_the_filters_keeps_everything():
    fs = sel(night(), filters=K_SIGMA_FILTERS)
    assert fs.kept.all()
    a = fs.auto
    assert a["filter_fwhm"].value == "2.5k" and a["filter_fwhm"].reason.startswith("no frame with stars more than 2.5")
    assert a["filter_background"].value == "" and a["filter_background"].reason == "off"


def test_filter_syntax_is_siril_s():
    assert parse_filter("2.5k") == Filter("k", 2.5)
    assert parse_filter("90%") == Filter("pct", 90.0)
    assert parse_filter(" 3,2 ") == Filter("value", 3.2)
    assert parse_filter(4) == Filter("value", 4.0)
    for off in ("", "off", None, "0", "0k", 0):
        assert parse_filter(off).kind == "off"
    assert str(parse_filter("2.5K")) == "2.5k" and str(parse_filter("90 %")) == "90%" and str(Filter()) == ""
    for bad in ("2.5s", "k", "-1k", "150%"):
        with pytest.raises(ValueError):
            parse_filter(bad)
    with pytest.raises(ValueError):
        sel(night(), filters={"eccentricity": "2k"})


def test_percent_and_value_filters():
    m = night()
    fs = sel(m, filters={"fwhm": "90%"}, min_weight=0)
    worst = set(np.argsort(m["fwhm"])[-4:])                         # 36 of 40 kept
    assert {r.frame for r in fs.exclusions} == worst
    assert fs.reasons[max(worst)].endswith("not among the 90 % of frames with the finest stars")
    fv = sel(m, filters={"roundness": "0.89"}, min_weight=0)
    assert {r.frame for r in fv.exclusions} == set(np.where(m["roundness"] < 0.89)[0])
    assert fv.auto["filter_roundness"].reason.endswith("excluded by this filter")


def test_k_sigma_clips_only_the_bad_tail():
    m = night()
    m["fwhm"][5] = 1.5                                             # much sharper: never excluded
    m["roundness"][6] = 1.0                                        # rounder: never excluded
    m["transparency"][7] = 1.5                                     # clearer: never excluded
    m["background"][8] = 100.0                                     # darker sky: never excluded
    f = {**K_SIGMA_FILTERS, "background": "2.5k"}
    fs = sel(m, filters=f, min_weight=0)
    assert fs.kept.all()
    m["background"][9] = 800.0                                     # the moon rises
    fb = sel(m, filters=f, min_weight=0)
    assert [r.frame for r in fb.exclusions] == [9] and fb.reasons[9].endswith("moon, dawn or a light?")


def test_siril_and_mad_limits_agree_on_gaussian_nights():
    """Siril's k x sigma (median + k x standard deviation, recomputed after each clip until stable, bad tail only)
    and ours (median + k x 1.4826 MAD) on Gaussian nights: the docstring's 2.41 to 2.49 sigma at k = 2.5."""
    def siril_limit(v, k):
        v = np.sort(v)
        while True:
            t = np.median(v) + k * np.std(v, ddof=1)
            if (v <= t).all():
                return t
            v = v[v <= t]
    for n in (30, 100, 2000):
        z = gauss(n)
        med, s = robust_sigma(z)
        assert med + 2.5 * s == pytest.approx(2.5, abs=0.01)
        assert 2.40 < siril_limit(z, 2.5) <= 2.5


def test_hand_exclusion_and_alignment_never_move_the_thresholds():
    m = cloudy()
    aligned = np.ones(N, bool)
    aligned[0] = False
    m["fwhm"][0] = 50.0                                            # a failed frame's nonsense: out of the statistics
    ref = sel(m, aligned=aligned, filters=K_SIGMA_FILTERS)
    ex = np.zeros(N, bool)
    ex[[1, 2, 11]] = True
    fs = sel(m, aligned=aligned, exclude=ex, filters=K_SIGMA_FILTERS)
    assert fs.reasons[0] == "not aligned" and fs.codes[0] == "not_aligned"
    assert fs.reasons[1] == fs.reasons[2] == fs.reasons[11] == "excluded by hand"
    for c in ("fwhm", "transparency"):
        assert fs.criteria[c].limit == ref.criteria[c].limit
    assert ref.criteria["fwhm"].median == pytest.approx(np.median(m["fwhm"][1:]))
    others = ~ex & aligned
    assert list(fs.reasons[others]) == list(ref.reasons[others])
    assert ref.reasons[11].endswith("clouds or haze?")


def test_missing_measures():
    m = night()
    fs = select(weights(m), filters=K_SIGMA_FILTERS)               # no FWHM, roundness, transparency: weights only
    assert fs.kept.all() and fs.criteria["fwhm"].status == "no measure"
    assert fs.auto["filter_fwhm"].reason == "no measure for this filter: not applied"
    r = m["roundness"].copy()
    r[6] = np.nan                                                   # not measured: never excluded for it
    fs = select(weights(m), fwhm=m["fwhm"], roundness=r, filters=K_SIGMA_FILTERS)
    assert fs.kept[6] and fs.kept.all()


# ------------------------------------------------------------------------------------------------- texts and outputs
def test_texts_are_templates_with_their_numbers():
    m = windy()
    m["transparency"][10:14] = [0.6, 0.4, 0.3, 0.2]
    fs = sel(m, filters=K_SIGMA_FILTERS)
    used = [r.template for r in fs.exclusions] + [a.template for a in fs.auto.values()] + [fs.suggestion.template]
    assert set(used) <= set(REASON_TEMPLATES)
    fr = fs.reasons_in("fr")
    assert fr[12].count(",") >= 1 and "," in fr[12].split(" σ")[0]     # decimal comma: "30,4 σ"
    assert fs.reasons_in("en") == list(fs.reasons)
    json.dumps(fs.as_dict())
    assert K_SIGMA_FILTERS == {"fwhm": "2.5k", "roundness": "2.5k", "transparency": "2.5k", "background": ""}
    assert NO_FILTERS == {c: "" for c in fwm.CRITERIA}


def test_planes_to_colours():
    planes = np.array([[1.0, 2.0, 4.0, 8.0]])
    np.testing.assert_allclose(planes_to_colours(planes, [0, 1, 1, 2]), [[1.0, 3.0, 8.0]])     # RGGB
