"""Frame weighting and selection of stacking v2 (astro_gpu/core/frame_weights.py), on synthetic nights.

A "steady night" spreads every measure on the quantiles of a Gaussian (no frame beyond 2.2 sigma for 40 frames), so
the default k = 2.5 must keep everything; then a cloud passage, wind, and one frame with a wrong noise measure.
"""
import json
from statistics import NormalDist

import numpy as np
import pytest

from astro_gpu.core import frame_weights as fwm
from astro_gpu.core.frame_weights import (
    DEFAULT_FILTERS,
    NO_FILTERS,
    REASON_TEMPLATES,
    Filter,
    parse_filter,
    planes_to_colours,
    robust_sigma,
    stack_noise,
    weigh_frames,
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


def ivm(t, s):
    w = np.asarray(t, float) ** 2 / np.asarray(s, float) ** 2
    return w / w.max()


# ------------------------------------------------------------------------------------------------- steady night
def test_a_steady_night_keeps_every_frame_weighed_by_inverse_variance():
    m = night()
    fw = weigh_frames(**m)
    assert fw.kept.all() and not fw.exclusions and not fw.notes
    assert fw.weights.shape == (N, 3)
    np.testing.assert_allclose(fw.frame_weight, ivm(m["transparency"], m["noise"]), rtol=1e-12)
    np.testing.assert_allclose(fw.weights[:, 0], fw.weights[:, 2])      # scalar measures: the colours agree
    assert list(fw.reasons) == [""] * N
    assert fw.suggestion.value == N and fw.suggestion.reason == f"all {N} aligned frames kept"
    assert fw.selection_cost["noise_pct"] == 0.0
    assert fw.costs["noise"]["noise_pct"] == 0.0
    a = fw.auto
    assert a["weighting"].value == "noise" and a["min_weight"].value == 0.05
    assert a["filter_fwhm"].value == "2.5k" and a["filter_fwhm"].reason.startswith("no frame with stars more than 2.5")
    assert a["filter_background"].value == "" and a["filter_background"].reason.startswith("off")


def test_inverse_variance_is_per_colour():
    m = night()
    s3 = np.stack([m["noise"] * 1.3, m["noise"], m["noise"] * (1 + 0.1 * gauss(N, 9))], axis=1)
    t3 = np.stack([m["transparency"] * 0.9, m["transparency"], m["transparency"] * 1.05], axis=1)
    fw = weigh_frames(t3, s3, m["fwhm"], m["roundness"])
    assert fw.kept.all()
    for c in range(3):
        np.testing.assert_allclose(fw.weights[:, c], ivm(t3[:, c], s3[:, c]), rtol=1e-12)
    assert not np.allclose(fw.weights[:, 0], fw.weights[:, 2])
    np.testing.assert_allclose(fw.frame_weight, fw.weights.mean(axis=1))


def test_stack_noise_of_the_inverse_variance_is_the_closed_form():
    v = (5.0 + gauss(N)) ** 2
    var = np.repeat(v[:, None], 3, axis=1)
    assert stack_noise(1.0 / var, var) == pytest.approx(1.0 / np.sqrt(np.sum(1.0 / v)))
    assert stack_noise(np.ones_like(var), var) > stack_noise(1.0 / var, var)       # equal weights: noisier


# ------------------------------------------------------------------------------------------------- cloud passage
def cloudy():
    m = night()
    m["transparency"][10:14] = [0.6, 0.4, 0.3, 0.55]
    return m


def test_a_cloud_passage_is_excluded_with_its_reason():
    m = cloudy()
    fw = weigh_frames(**m)
    out = sorted({r.frame for r in fw.exclusions})
    assert out == [10, 11, 12, 13]
    assert all(c == "transparency" for c in fw.codes[10:14])
    assert fw.reasons[12].startswith("transparency ") and fw.reasons[12].endswith("(30 % of it): clouds or haze?")
    sig = fw.exclusions[[r.frame for r in fw.exclusions].index(12)].args["sigma"]
    med, s = robust_sigma(m["transparency"])
    assert sig == pytest.approx((med - 0.3) / s)
    keep = np.ones(N, bool)
    keep[10:14] = False
    np.testing.assert_allclose(fw.frame_weight[keep], ivm(m["transparency"][keep], m["noise"][keep]), rtol=1e-12)
    assert (fw.weights[~keep] == 0).all()
    # what the clouded frames would have brought, measured and said
    assert 0.0 < fw.selection_cost["noise_pct"] < 5.0 and fw.selection_cost["frames"] == 4
    assert fw.suggestion.reason.startswith(f"{N - 4} of {N} frames kept; the 4 left out would have lowered the noise")
    assert fw.auto["filter_transparency"].reason.startswith("4 frame(s) more than 2.5 σ below the night's median")


def test_without_the_filter_the_clouded_frames_are_weighed_and_the_minimum_weight_drops_the_worst():
    m = cloudy()
    m["transparency"][13] = 0.2                                    # weight 0.04 of the best: under 0.05
    fw = weigh_frames(**m, filters={"transparency": ""})
    assert sorted(r.frame for r in fw.exclusions) == [13]
    assert fw.codes[13] == "weight"
    assert fw.reasons[13].endswith("under the minimum of 5 %")
    assert fw.frame_weight[12] == pytest.approx(ivm(m["transparency"], m["noise"])[12])     # 0.3: about 0.09, kept
    assert fw.frame_weight[12] > 0.05
    assert fw.auto["min_weight"].reason == "1 frame(s) weigh less than 5 % of the best one: excluded"
    fw0 = weigh_frames(**m, filters={"transparency": ""}, min_weight=0.0)
    assert fw0.kept.all()


# ------------------------------------------------------------------------------------------------- wind
def windy():
    m = night()
    m["fwhm"][25:28] *= 1.6
    m["roundness"][25:28] = 0.6
    return m


def test_wind_excludes_the_frames_with_large_stretched_stars():
    m = windy()
    fw = weigh_frames(**m)
    assert sorted({r.frame for r in fw.exclusions}) == [25, 26, 27]
    assert list(fw.codes[25:28]) == ["fwhm,roundness"] * 3
    assert "wind, focus or seeing?" in fw.reasons[25] and "wind or tracking?" in fw.reasons[25]
    med = np.median(m["fwhm"])
    assert fw.reasons[25].startswith("stars ") and \
        f" σ larger than the night's median ({m['fwhm'][25]:.1f} px for {med:.1f} px)" in fw.reasons[25]
    # the inverse variance does not look at the stars: the other frames keep their weight
    keep = np.ones(N, bool)
    keep[25:28] = False
    np.testing.assert_allclose(fw.frame_weight[keep], ivm(m["transparency"][keep], m["noise"][keep]), rtol=1e-12)


def test_sharp_stars_has_a_measured_cost():
    m = night()
    m["fwhm"] = 3.0 + 0.5 * gauss(N, 3)                            # a night of changing seeing
    fw = weigh_frames(**m)
    sharp = fw.costs["sharp"]
    assert sharp["noise_pct"] > 0 and sharp["fwhm_pct"] < 0      # noisier, finer
    assert fw.costs["none"]["noise_pct"] > 0                       # equal weights are noisier too
    a = fw.auto["weighting"]
    assert a.value == "noise" and a.template == fwm.A_WEIGHT
    assert a.args == {"noise": sharp["noise_pct"], "finer": -sharp["fwhm_pct"]}
    fs = weigh_frames(**m, weighting="sharp")
    w = m["transparency"] ** 2 / (m["noise"] ** 2 * m["fwhm"] ** 2)
    np.testing.assert_allclose(fs.frame_weight, w / w.max(), rtol=1e-12)
    # alike frames: nothing to choose
    same = weigh_frames(np.ones(N), np.full(N, 10.0), np.full(N, 3.0), np.full(N, 0.9))
    assert same.auto["weighting"].template == fwm.A_WEIGHT_SAME


def test_sharp_stars_is_todays_stack_weight():
    from astro_gpu.stages.stack import select_frames
    m = night()
    m["fwhm"] = 3.0 + 0.5 * gauss(N, 3)
    rms = np.full(N, 0.3)
    stats = np.ones((N, 4, 2))
    w_old, r_old = select_frames(m["transparency"], m["fwhm"], rms, stats, 0.0, 5.0, noise=m["noise"])
    assert (r_old == "").all()
    fw = weigh_frames(**m, weighting="sharp", filters=NO_FILTERS, min_weight=0.0)
    np.testing.assert_allclose(fw.frame_weight, w_old, rtol=1e-12)


def test_no_weighting():
    m = night()
    fw = weigh_frames(**m, weighting="none")
    assert (fw.weights == 1).all()
    assert fw.costs["none"]["noise_pct"] > 0


# ------------------------------------------------------------------------------------------------- one outlier frame
def test_a_wrong_noise_measure_is_brought_back_to_the_median_weight():
    m = night()
    m["noise"][33] = 0.4                                           # 25 x too low: its weight 600 x the others
    fw = weigh_frames(**m)
    assert fw.kept.all()
    assert [(r.frame, r.code) for r in fw.notes] == [(33, "guard")]
    assert fw.notes[0].args["colour"] == "R, G, B" and fw.notes[0].args["ratio"] > 100
    raw = m["transparency"] ** 2 / m["noise"] ** 2
    guarded = raw.copy()
    guarded[33] = np.median(raw)                                   # the median of all the weights, frame 33's included
    np.testing.assert_allclose(fw.frame_weight, guarded / guarded.max(), rtol=1e-12)
    assert fw.as_dict()["guarded"] == [{"frame": 33, "colour": "R, G, B", "ratio": fw.notes[0].args["ratio"]}]
    # without the guard, that one frame would push all the others under the minimum weight
    bare = weigh_frames(**m, guard=False)
    assert bare.kept.sum() == 1 and bare.kept[33]


# ------------------------------------------------------------------------------------------------- the filters
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
        weigh_frames(**night(), filters={"eccentricity": "2k"})
    with pytest.raises(ValueError):
        weigh_frames(**night(), weighting="psf")


def test_percent_and_value_filters():
    m = night()
    fw = weigh_frames(**m, filters={"fwhm": "90%", "roundness": "", "transparency": ""}, min_weight=0)
    worst = set(np.argsort(m["fwhm"])[-4:])                         # 36 of 40 kept
    assert {r.frame for r in fw.exclusions} == worst
    assert fw.reasons[max(worst)].endswith("not among the 90 % of frames with the finest stars")
    fv = weigh_frames(**m, filters={"fwhm": "", "roundness": "0.89", "transparency": ""}, min_weight=0)
    assert {r.frame for r in fv.exclusions} == set(np.where(m["roundness"] < 0.89)[0])
    assert fv.auto["filter_roundness"].reason.endswith("excluded by this filter")


def test_k_sigma_clips_only_the_bad_tail():
    m = night()
    m["fwhm"][5] = 1.5                                             # much sharper: never excluded
    m["roundness"][6] = 1.0                                        # rounder: never excluded
    m["transparency"][7] = 1.5                                     # clearer: never excluded
    m["background"][8] = 100.0                                     # darker sky: never excluded
    fw = weigh_frames(**m, filters={"background": "2.5k"})
    assert fw.kept.all()
    m["background"][9] = 800.0                                     # the moon rises
    fb = weigh_frames(**m, filters={"background": "2.5k"})
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
    ref = weigh_frames(**m, aligned=aligned)
    ex = np.zeros(N, bool)
    ex[[1, 2, 11]] = True
    fw = weigh_frames(**m, aligned=aligned, exclude=ex)
    assert fw.reasons[0] == "not aligned" and fw.codes[0] == "not_aligned"
    assert fw.reasons[1] == fw.reasons[2] == fw.reasons[11] == "excluded by hand"
    for c in ("fwhm", "transparency"):
        assert fw.criteria[c].limit == ref.criteria[c].limit
    assert ref.criteria["fwhm"].median == pytest.approx(np.median(m["fwhm"][1:]))
    others = ~ex & aligned
    assert list(fw.reasons[others]) == list(ref.reasons[others])
    assert ref.reasons[11].endswith("clouds or haze?")


def test_missing_measures():
    m = night()
    fw = weigh_frames(m["transparency"], m["noise"])                # no FWHM, roundness, sky: weights only
    assert fw.kept.all() and fw.criteria["fwhm"].status == "no measure"
    assert fw.auto["filter_fwhm"].reason == "no measure for this filter: not applied"
    assert fw.auto["weighting"].template == fwm.A_WEIGHT_NO_FWHM and "sharp" not in fw.costs
    with pytest.raises(ValueError):
        weigh_frames(m["transparency"], m["noise"], weighting="sharp")
    s = m["noise"].copy()
    s[4] = np.nan
    r = m["roundness"].copy()
    r[6] = np.nan                                                   # not measured: never excluded for it
    fw = weigh_frames(m["transparency"], s, m["fwhm"], r)
    assert fw.codes[4] == "no_measure" and fw.kept[6] and fw.kept.sum() == N - 1


# ------------------------------------------------------------------------------------------------- texts and outputs
def test_texts_are_templates_with_their_numbers():
    m = windy()
    m["transparency"][10:14] = [0.6, 0.4, 0.3, 0.55]
    m["noise"][33] = 0.4
    fw = weigh_frames(**m)
    used = [r.template for r in fw.exclusions + fw.notes] + [a.template for a in fw.auto.values()] + \
        [fw.suggestion.template]
    assert set(used) <= set(REASON_TEMPLATES)
    fr = fw.reasons_in("fr")
    assert fr[12].count(",") >= 1 and "," in fr[12].split(" σ")[0]     # decimal comma: "30,4 σ"
    assert fw.reasons_in("en") == list(fw.reasons)
    json.dumps(fw.as_dict())
    assert DEFAULT_FILTERS == {"fwhm": "2.5k", "roundness": "2.5k", "transparency": "2.5k", "background": ""}


def test_planes_to_colours():
    planes = np.array([[1.0, 2.0, 4.0, 8.0]])
    np.testing.assert_allclose(planes_to_colours(planes, [0, 1, 1, 2]), [[1.0, 3.0, 8.0]])     # RGGB
