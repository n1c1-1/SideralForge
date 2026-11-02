"""Frame selection for the stack (``select``).

Pure functions on the per-frame measures: no file, no GPU, no project. One call::

    fs = select(weight, fwhm=..., roundness=..., transparency=..., background=..., aligned=..., exclude=...,
                filters=NO_FILTERS, min_weight=MIN_WEIGHT)

``weight`` is the stack's own weight of each frame (stages/stack.quality_weights: (transparency / noise)^2 x
sharpness); this module never changes it, it only says which frames are left out, and why.

The state of the art, and where each value comes from
-----------------------------------------------------
* **Keep almost every frame and let the weights do the work (the default).** Siril keeps every frame unless a filter
  is asked for [S2]; PixInsight's SubframeSelector approval expressions are empty by default [P21]; Zackay & Ofek
  [L6]: once the weights are right, "the rejection of images becomes a damaging operation".
* **Minimum weight 0.05 (PixInsight WBPP 2.5.0, its "Minimum weight" default [P17])**: the weights of the frames
  still in are normalised (the best = 1) and those below the minimum are left out. It is the only frame the default
  leaves out besides the frames registration could not align and those excluded by hand.
* **Selection filters, an option off by default: Siril's ``-filter-fwhm / -filter-round / -filter-bkg`` [S2]**, with
  its three forms: a value (the worst value kept), a percentage of the best frames kept, or k x sigma. Siril's
  k x sigma (code: ``generic_compute_accepted_value_with_rejection``, sequence_filtering.c) clips only the bad tail
  (high FWHM and background, low roundness), around the median. Siril's sigma is the standard deviation, recomputed
  after each clip until nothing moves; ours is 1.4826 x MAD in one pass (the Hampel identifier: Hampel 1974, JASA
  69, 383), the robust estimate of the same Gaussian sigma that the outliers it must find cannot inflate. On Gaussian
  nights of 30 to 2000 frames, Siril's limit at k = 2.5 falls at 2.41 to 2.49 sigma, ours at 2.5
  (tests/test_frame_weights.py). Transparency, our measure of clouds, is filtered the same way (low tail).
  PixInsight's SubframeSelector approval expressions (``FWHMSigma < 2``...) are the same idea [P21].
* **k = 2.5 when the filters are on (``K_SIGMA_FILTERS``)**: Siril's official Seestar script (SeeStar_Preprocessing
  v1.3, C. Richard, June 2024, ``seqapplyreg pp_light -filter-round=2.5k``), the only k a tool ships; on FWHM,
  roundness and transparency; the background stays off (a brighter sky only adds noise, which the weights count).
* The statistics of a filter are measured on all the aligned frames, those excluded by hand included, so excluding
  a frame by hand never changes the fate of another through a filter.

Sources:
[S2] Siril 1.4, the ``stack`` command: https://gitlab.com/free-astro/siril-doc/-/raw/1.4/doc/commands/stack.rst
[P17] WBPP 2.5.0 release notes (minimum weight, colour drizzle): https://pixinsight.com/forum/index.php?threads/wbpp-2-5-0-released.19112/
[P21] PixInsight SubframeSelector documentation: https://gitlab.com/pixinsight/Reference-Documentation/-/raw/master/tools/obsolete/SubframeSelector/SubframeSelector.pidoc
[L6] Zackay & Ofek 2017, ApJ 836, 187 (optimal weighting, rejection of frames): https://arxiv.org/abs/1512.06872
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from .params import Auto

MAD_TO_SIGMA = 1.4826
CRITERIA = ("fwhm", "roundness", "transparency", "background")
#: the bad tail of each criterion: True when a high value is bad
HIGH_IS_BAD = {"fwhm": True, "roundness": False, "transparency": False, "background": True}
#: the k x sigma filters, k = 2.5 as in Siril's official Seestar script (-filter-round=2.5k); an option: Siril and
#: SubframeSelector filter nothing by default
K_SIGMA_FILTERS = {"fwhm": "2.5k", "roundness": "2.5k", "transparency": "2.5k", "background": ""}
NO_FILTERS = {c: "" for c in CRITERIA}
#: PixInsight WBPP 2.5.0's default minimum weight (relative to the best frame)
MIN_WEIGHT = 0.05

# ------------------------------------------------------------------------------------------------- reason templates
NOT_ALIGNED = "not aligned"
BY_HAND = "excluded by hand"                    # stack.EXCLUDED
NO_MEASURE = "no measure of its weight"
_K = {
    "fwhm": "stars {sigma:.1f} σ larger than the night's median ({value:.1f} px for {median:.1f} px): "
            "wind, focus or seeing?",
    "roundness": "stars {sigma:.1f} σ less round than the night's median ({value:.2f} for {median:.2f}): "
                 "wind or tracking?",
    "transparency": "transparency {sigma:.1f} σ below the night's median ({pct:.0f} % of it): clouds or haze?",
    "background": "sky {sigma:.1f} σ brighter than the night's median ({pct:+.0f} %): moon, dawn or a light?",
}
#: the same without the ratio to the median, when the median is not above zero
_K_PLAIN = {
    "transparency": "transparency {sigma:.1f} σ below the night's median: clouds or haze?",
    "background": "sky {sigma:.1f} σ brighter than the night's median: moon, dawn or a light?",
}
_PCT = {
    "fwhm": "stars of {value:.1f} px: not among the {keep:g} % of frames with the finest stars",
    "roundness": "stars of roundness {value:.2f}: not among the {keep:g} % of frames with the roundest stars",
    "transparency": "transparency {value:.2f}: not among the {keep:g} % of clearest frames",
    "background": "sky level {value:.4g}: not among the {keep:g} % of frames with the darkest sky",
}
_VALUE = {
    "fwhm": "stars of {value:.1f} px, above the limit of {limit:.1f} px",
    "roundness": "stars of roundness {value:.2f}, below the limit of {limit:.2f}",
    "transparency": "transparency {value:.2f}, below the limit of {limit:.2f}",
    "background": "sky level {value:.4g}, above the limit of {limit:.4g}",
}
LOW_WEIGHT = "weight {pct:.1f} % of the best frame's, under the minimum of {min:.0f} %"

_A_HIT = {
    "fwhm": "{n} frame(s) with stars more than {k:g} σ larger than the night's median (above {limit:.1f} px): "
            "excluded",
    "roundness": "{n} frame(s) with stars more than {k:g} σ less round than the night's median (below "
                 "{limit:.2f}): excluded",
    "transparency": "{n} frame(s) more than {k:g} σ below the night's median transparency (under {pct:.0f} % of "
                    "it): excluded",
    "background": "{n} frame(s) with a sky more than {k:g} σ brighter than the night's median: excluded",
}
_A_NONE = {
    "fwhm": "no frame with stars more than {k:g} σ larger than the night's median ({limit:.1f} px)",
    "roundness": "no frame with stars more than {k:g} σ less round than the night's median ({limit:.2f})",
    "transparency": "no frame more than {k:g} σ below the night's median transparency ({pct:.0f} % of it)",
    "background": "no frame with a sky more than {k:g} σ brighter than the night's median",
}
A_OTHER_HIT = "{n} frame(s) excluded by this filter"
A_OTHER_NONE = "no frame excluded by this filter"
A_OFF = "off"
A_NO_MEASURE = "no measure for this filter: not applied"
A_NO_SPREAD = "all frames alike: nothing to exclude"
A_MIN_HIT = "{n} frame(s) weigh less than {min:.0f} % of the best one: excluded"
A_MIN_NONE = "every frame weighs at least {low:.0f} % of the best one"
A_SUMMARY_KEPT = "{kept} of {n} aligned frames kept"
A_SUMMARY_ALL = "all {n} aligned frames kept"

#: every template this module writes (their French goes to client/locale/fr/engine.json)
REASON_TEMPLATES = tuple(dict.fromkeys(
    [NOT_ALIGNED, BY_HAND, NO_MEASURE, LOW_WEIGHT, A_OTHER_HIT, A_OTHER_NONE, A_OFF, A_NO_MEASURE, A_NO_SPREAD,
     A_MIN_HIT, A_MIN_NONE, A_SUMMARY_KEPT, A_SUMMARY_ALL]
    + [d[c] for d in (_K, _PCT, _VALUE, _A_HIT, _A_NONE) for c in CRITERIA] + list(_K_PLAIN.values())))


@dataclass
class Reason:
    """Why a frame is left out: a code, an English template and its numbers."""
    frame: int
    code: str
    template: str
    args: dict | None = None

    @property
    def text_en(self) -> str:
        return self.template if self.args is None else self.template.format(**self.args)

    def text(self, lang: str = "en") -> str:
        from .. import texts
        return texts.format_reason(self.template, self.args, lang)


# ------------------------------------------------------------------------------------------------- filters
@dataclass(frozen=True)
class Filter:
    """A selection filter in Siril's syntax: ``"2.5k"`` k x sigma, ``"90%"`` the best 90 % kept, ``"4.2"`` the worst
    value kept, ``""`` / ``"off"`` none. A zero amount switches the filter off."""
    kind: str = "off"          # "k", "pct", "value" or "off"
    amount: float = 0.0

    def __str__(self) -> str:
        return {"off": "", "k": f"{self.amount:g}k", "pct": f"{self.amount:g}%"}.get(self.kind, f"{self.amount:g}")


def parse_filter(spec) -> Filter:
    """A ``Filter`` from Siril's syntax (``ValueError`` when it is not one)."""
    if isinstance(spec, Filter):
        return spec
    if spec is None:
        return Filter()
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return Filter("value", float(spec)) if spec else Filter()
    s = str(spec).strip().lower().replace(",", ".")
    if s in ("", "off", "none"):
        return Filter()
    m = re.fullmatch(r"(\d+(?:\.\d*)?|\.\d+)\s*(k|%)?", s)
    if not m:
        raise ValueError(f"not a frame filter: {spec!r} (a value, a percentage like 90% or k x sigma like 2.5k)")
    x = float(m[1])
    if x == 0:
        return Filter()
    kind = {"k": "k", "%": "pct", None: "value"}[m[2]]
    if kind == "pct" and x > 100:
        raise ValueError(f"a percentage of frames to keep is at most 100: {spec!r}")
    return Filter(kind, x)


def robust_sigma(x) -> tuple[float, float]:
    """(median, 1.4826 x MAD) of the finite values: the centre and the Gaussian sigma, robust to a few outliers."""
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan"), float("nan")
    m = float(np.median(v))
    return m, float(MAD_TO_SIGMA * np.median(np.abs(v - m)))


@dataclass
class CriterionStats:
    """What a filter did: its setting, the night's median and sigma, its limit, the frames it excluded."""
    criterion: str
    spec: str
    median: float = float("nan")
    sigma: float = float("nan")
    limit: float = float("nan")
    excluded: list[int] = field(default_factory=list)
    status: str = "off"        # "applied", "off", "no measure", "no spread"


def apply_filter(values, flt: Filter, criterion: str, stats_on, judge) -> tuple[np.ndarray, CriterionStats,
                                                                                   list[Reason]]:
    """The frames ``flt`` excludes on ``criterion``: (bool (n,), its stats, one reason per excluded frame).

    ``values`` (n,) the measure (NaN: not measured, never excluded); the median, sigma and percentiles are taken on
    the frames ``stats_on``, and only the frames ``judge`` can be excluded."""
    v = np.asarray(values, float)
    n = len(v)
    fails = np.zeros(n, bool)
    st = CriterionStats(criterion, str(flt))
    if flt.kind == "off":
        return fails, st, []
    pool = np.asarray(stats_on, bool) & np.isfinite(v)
    if not pool.any():
        st.status = "no measure"
        return fails, st, []
    hi_bad = HIGH_IS_BAD[criterion]
    st.median, st.sigma = robust_sigma(v[pool])
    if flt.kind == "k":
        if not st.sigma > 0:
            st.status = "no spread"
            st.limit = st.median
            return fails, st, []
        st.limit = st.median + flt.amount * st.sigma if hi_bad else st.median - flt.amount * st.sigma
    elif flt.kind == "pct":
        good = np.sort(v[pool]) if hi_bad else np.sort(v[pool])[::-1]
        keep = max(1, int(math.floor(len(good) * flt.amount / 100.0 + 0.5)))
        st.limit = float(good[keep - 1])
    else:
        st.limit = flt.amount
    st.status = "applied"
    with np.errstate(invalid="ignore"):
        bad = v > st.limit if hi_bad else v < st.limit
    fails = bad & np.isfinite(v) & np.asarray(judge, bool)
    reasons = []
    for i in np.where(fails)[0]:
        x = float(v[i])
        if flt.kind == "k":
            sig = (x - st.median) / st.sigma if hi_bad else (st.median - x) / st.sigma
            args = {"sigma": sig, "value": x, "median": st.median}
            tpl = _K[criterion]
            if criterion in ("transparency", "background"):
                if st.median > 0:
                    args["pct"] = 100.0 * x / st.median if criterion == "transparency" \
                        else 100.0 * (x / st.median - 1.0)
                else:                                   # no ratio to a median at or below zero
                    tpl = _K_PLAIN[criterion]
            reasons.append(Reason(int(i), criterion, tpl, args))
        elif flt.kind == "pct":
            reasons.append(Reason(int(i), criterion, _PCT[criterion], {"value": x, "keep": flt.amount}))
        else:
            reasons.append(Reason(int(i), criterion, _VALUE[criterion], {"value": x, "limit": st.limit}))
    st.excluded = [int(i) for i in np.where(fails)[0]]
    return fails, st, reasons


# ------------------------------------------------------------------------------------------------- measures
def per_frame(x, n: int, name: str = "measure") -> np.ndarray | None:
    """(n,) from a measure given per frame or per frame and colour (the mean of the colours); None stays None."""
    if x is None:
        return None
    a = np.asarray(x, float)
    if a.ndim == 2 and a.shape[0] == n:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # a frame with no measure at all: NaN
            a = np.nanmean(a, axis=1)
    if a.shape != (n,):
        raise ValueError(f"{name}: expected shape ({n},) or ({n}, k), got {np.shape(x)}")
    return a


# ------------------------------------------------------------------------------------------------- the result
@dataclass
class FrameSelection:
    weight: np.ndarray             # (n,): the weight given, 0 when left out
    relative: np.ndarray           # (n,): the weight relative to the best frame still in before the minimum (0: out)
    kept: np.ndarray               # (n,) bool
    reasons: np.ndarray            # (n,) object: English reasons joined by "; ", "" for a kept frame
    codes: np.ndarray              # (n,) object: the codes, joined by ","
    exclusions: list[Reason]       # one per frame and cause
    criteria: dict[str, CriterionStats]
    min_weight: float
    auto: dict[str, Auto]          # what each filter and the minimum weight did (filter_*, min_weight)
    suggestion: Auto               # the summary: frames kept

    def as_dict(self) -> dict:
        """A JSON-ready summary (diagnostics.json): thresholds and exclusions."""
        def num(x):
            return None if x is None or not np.isfinite(x) else float(x)
        return {
            "kept": int(self.kept.sum()),
            "min_weight": float(self.min_weight),
            "criteria": {c: {"spec": s.spec, "status": s.status, "median": num(s.median), "sigma": num(s.sigma),
                             "limit": num(s.limit), "excluded": s.excluded} for c, s in self.criteria.items()},
            "exclusions": [{"frame": r.frame, "code": r.code, "reason": r.text_en} for r in self.exclusions],
        }


def select(weight, fwhm=None, roundness=None, transparency=None, background=None, *, aligned=None, exclude=None,
           filters: dict | None = None, min_weight: float = MIN_WEIGHT) -> FrameSelection:
    """The frames left out, with their reasons, and the weights of the others (``weight`` unchanged).

    ``weight`` (n,): the stack's weight of each frame (NaN or <= 0 for an aligned frame: no measure); ``fwhm`` (n,)
    px (Moffat), ``roundness`` (n,) minor / major axis (1 = round), ``transparency`` (n,), ``background`` (n,) or
    (n, 3) the sky level: the measures of the filters (None: that filter cannot apply); ``aligned`` (n,) bool (None:
    all); ``exclude`` (n,) bool: excluded by hand; ``filters`` {criterion: Siril-style spec} over NO_FILTERS (None:
    none, the default); ``min_weight`` relative to the best frame still in (0: none).

    Order: not aligned, by hand, the filters (statistics on all aligned frames), no measure of the weight, the
    minimum weight (WBPP: relative to the best of the frames still in)."""
    w = np.asarray(weight, float)
    n = len(w)
    aligned = np.ones(n, bool) if aligned is None else np.asarray(aligned, bool)
    hand = np.zeros(n, bool) if exclude is None else np.asarray(exclude, bool)
    specs = {**NO_FILTERS, **(filters or {})}
    unknown = set(specs) - set(CRITERIA)
    if unknown:
        raise ValueError(f"unknown frame filter(s) {sorted(unknown)}: {CRITERIA}")

    exclusions: list[Reason] = []
    for i in np.where(~aligned)[0]:
        exclusions.append(Reason(int(i), "not_aligned", NOT_ALIGNED))
    for i in np.where(aligned & hand)[0]:
        exclusions.append(Reason(int(i), "by_hand", BY_HAND))

    # ---- the filters: the night's statistics on every aligned frame; only the frames still in can be excluded
    judge = aligned & ~hand
    measures = {"fwhm": per_frame(fwhm, n, "fwhm"), "roundness": per_frame(roundness, n, "roundness"),
                "transparency": per_frame(transparency, n, "transparency"),
                "background": per_frame(background, n, "background")}
    criteria: dict[str, CriterionStats] = {}
    failed = np.zeros(n, bool)
    for c in CRITERIA:
        flt = parse_filter(specs[c])
        if measures[c] is None:
            criteria[c] = CriterionStats(c, str(flt), status="off" if flt.kind == "off" else "no measure")
            continue
        f, st, rs = apply_filter(measures[c], flt, c, aligned, judge)
        criteria[c] = st
        failed |= f
        exclusions += rs

    # ---- the minimum weight (WBPP): relative to the best frame still in
    measured = np.isfinite(w) & (w > 0)
    for i in np.where(judge & ~failed & ~measured)[0]:
        exclusions.append(Reason(int(i), "no_measure", NO_MEASURE))
    cand = judge & ~failed & measured
    best = float(w[cand].max()) if cand.any() else 0.0
    rel = np.where(cand, w / best, 0.0) if best > 0 else np.zeros(n)
    low = cand & (rel < min_weight) if min_weight > 0 else np.zeros(n, bool)
    lowest = float(rel[cand].min()) if cand.any() else float("nan")
    for i in np.where(low)[0]:                     # rounded down: 4.97 % never reads "5.0 %, under the minimum of 5 %"
        exclusions.append(Reason(int(i), "weight", LOW_WEIGHT, {"pct": math.floor(1000.0 * float(rel[i])) / 10.0,
                                                               "min": 100.0 * min_weight}))
    kept = cand & ~low

    reasons = np.full(n, "", dtype=object)
    codes = np.full(n, "", dtype=object)
    for r in exclusions:
        reasons[r.frame] = r.text_en if not reasons[r.frame] else reasons[r.frame] + "; " + r.text_en
        codes[r.frame] = r.code if not codes[r.frame] else codes[r.frame] + "," + r.code
    auto = _autos(criteria, specs, low, lowest, min_weight)
    out = int((judge & ~kept).sum())
    if out:
        suggestion = Auto(int(kept.sum()), A_SUMMARY_KEPT, {"kept": int(kept.sum()), "n": int(judge.sum())})
    else:
        suggestion = Auto(int(kept.sum()), A_SUMMARY_ALL, {"n": int(judge.sum())})
    return FrameSelection(np.where(kept, w, 0.0), np.where(kept, rel, 0.0), kept, reasons, codes, exclusions,
                          criteria, float(min_weight), auto, suggestion)


def _autos(criteria, specs, low, lowest, min_weight) -> dict[str, Auto]:
    """What each filter and the minimum weight did, as a reason."""
    a: dict[str, Auto] = {}
    for c in CRITERIA:
        st = criteria[c]
        flt = parse_filter(specs[c])
        spec = str(flt)
        key = f"filter_{c}"
        if flt.kind == "off":
            a[key] = Auto(spec, A_OFF)
        elif st.status == "no measure":
            a[key] = Auto(spec, A_NO_MEASURE)
        elif st.status == "no spread":
            a[key] = Auto(spec, A_NO_SPREAD)
        elif flt.kind == "k":
            args = {"n": len(st.excluded), "k": flt.amount, "limit": st.limit}
            if c == "transparency":
                args["pct"] = 100.0 * st.limit / st.median if st.median > 0 else float("nan")
            a[key] = Auto(spec, (_A_HIT if st.excluded else _A_NONE)[c], args)
        else:                                      # a value or a percentage: what it did
            a[key] = Auto(spec, A_OTHER_HIT if st.excluded else A_OTHER_NONE,
                          {"n": len(st.excluded)} if st.excluded else None)
    if low.any():
        a["min_weight"] = Auto(min_weight, A_MIN_HIT, {"n": int(low.sum()), "min": 100.0 * min_weight})
    else:
        a["min_weight"] = Auto(min_weight, A_MIN_NONE, {"low": 100.0 * lowest if np.isfinite(lowest) else 100.0})
    return a
