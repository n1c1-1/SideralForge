"""Frame weighting and frame selection for the stack (stacking v2: docs/spec/STACKING.md, point 4).

Pure functions on the per-frame measures: no file, no GPU, no project. One call does everything::

    fw = weigh_frames(transparency, noise, fwhm, roundness, background, aligned=..., exclude=...,
                      weighting="noise", filters=None, min_weight=0.05)

How stack.py calls it (for the integration agent)
--------------------------------------------------
Behind the stacking-v2 switch only; with the switch off, ``select_frames`` stays as it is (same pixels, same keys).

1. Parameters to declare in ``StackParams`` (the names the ``Auto`` of ``fw.auto`` are keyed on)::

       weighting: str = "noise"            choices WEIGHTINGS = ("noise", "sharp", "none")  (Noise / Sharp stars / None)
       filter_fwhm: str = "2.5k"           Siril syntax: "2.5k" (k x sigma), "90%" (best % kept), "4.2" (a value),
       filter_roundness: str = "2.5k"        "" or "off" (no filter); parse_filter() checks it
       filter_transparency: str = "2.5k"
       filter_background: str = ""
       min_weight: float = 0.05            0 to 0.2

2. In ``Stack.run``, in place of ``select_frames`` (``ch_of_k = cfa_channel_map(info["cfa"])``)::

       from ..core.frame_weights import planes_to_colours, weigh_frames
       fw = weigh_frames(
           transparency=transp,                 # (n,) today; (n, 3) once the per-colour photometric factor exists
           noise=reg["noise"],                  # (n,) pixel noise of each raw frame; (n, 3) when measured per colour
           fwhm=reg["fwhm_moffat"],             # (n,) or (n, 3): core/psf_moffat.py's PSF.fwhm, px
           roundness=reg["roundness_moffat"],   # (n,): PSF.roundness, FWHM minor / major (Siril's r, 1 = round)
           background=planes_to_colours(stats[:, :, 0], ch_of_k),   # sky level per colour, frame_stats.npy
           aligned=np.isfinite(rms), exclude=excl,
           weighting=p.weighting, min_weight=p.min_weight,
           filters={"fwhm": p.filter_fwhm, "roundness": p.filter_roundness,
                    "transparency": p.filter_transparency, "background": p.filter_background})
       weight, reasons = fw.frame_weight, fw.reasons      # same shapes and meaning as select_frames' (n,), (n,)
       W = fw.weights                                     # (n, 3): the weight of each frame in each colour

   ``transparency`` must be the factor the frame is divided by (``val = (val - grad) / transp[i]``) and ``noise`` the
   noise of the raw frame in its own ADU: the weight is then the inverse of the normalised frame's variance.

3. Deposit with the weight of each sample's colour: ``w_i = torch.as_tensor(W[i], dtype=torch.float32,
   device=dev)[chmap]`` and, in ``deposit``, ``ww = wx * wy * w_i[keep]`` (instead of the float ``weight[i]``). With
   a scalar transparency and noise, the three columns of ``W`` are equal and the stack is the scalar-weight one.
   The half-stack noise (``halves_noise``) takes ``w_a``, ``w_b`` per colour: ``W[in_a].sum(0)``, ``W[in_b].sum(0)``.

4. ``frames.json``: ``"weight": fw.frame_weight[i]``, ``"excluded": fw.reasons[i]`` (English), plus
   ``"weights_rgb": W[i]`` and ``"exclusion_codes": fw.codes[i]``; the log counts ``fw.codes`` (not_aligned, by_hand,
   no_measure, fwhm, roundness, transparency, background, weight). ``fw.as_dict()`` goes to diagnostics.json
   (the thresholds, the costs, the guarded weights), for Inspect > Stacking.

5. In ``Stack.auto``: the same call on the same measures with the defaults, then ``a.update(fw.auto)`` (weighting,
   the four filters, min_weight); ``fw.suggestion`` is the one-line summary for the Frames row. The reason
   templates are in ``REASON_TEMPLATES``: their French goes to ``client/locale/fr/engine.json`` (``texts.py`` only
   scans ``stages/*.py``, so ``tests/test_texts.py`` will not ask for them).

The methods, and where they come from (docs/pilot/STATE_OF_THE_ART_STACKING.md: sections 5, 6, 9.3, 9.6)
-----------------------------------------------------------------------------------------------------
* **Weight "noise" (default): the inverse of the variance, per colour**, transparency^2 / noise^2 on frames divided
  by their transparency. It is the optimal weight for an extended source limited by the sky (Zackay & Ofek 2017,
  ApJ 836, 187 [L6]), the IVM weight of Hubble's drizzle (DrizzlePac Handbook [L2]) and Siril's ``-weight=noise``,
  1 / (scale^2 x noise^2) [S2][S6].
* **Weight "sharp" (option "Sharp stars")**: Annis et al. 2014 (ApJ 794, 120), as written by Zackay & Ofek [L6]:
  transparency^2 / (FWHM^2 x noise^2) on normalised frames; the formula stack.py uses today. Its cost against the
  inverse variance is measured here (``costs``): the noise of the stack, sqrt(sum w^2 v) / sum w, and the FWHM of
  the weighted sum of the star images, sqrt(sum w FWHM^2 / sum w) (the second moment of a sum of profiles).
* **Weight "none"**: equal weights, Siril's default [S2].
* **Guard ("Weight overshoot protection", PixInsight 1.9.6 [P9])**: a weight above 10 x the upper quartile of the
  weights is given the median weight, separately for each colour; the other weights never change.
* **Minimum weight 0.05 (WBPP 2.5.0 [P17])**: the weights of the group are normalised (the best frame = 1) and the
  frames below the minimum are left out. Here the test is on the mean of the frame's three colours.
* **Selection filters, Siril's ``-filter-fwhm / -filter-round / -filter-bkg`` [S2]**, with its three forms: a value
  (the worst value kept), a percentage of the best frames kept, or k x sigma. Siril's k x sigma (code:
  ``generic_compute_accepted_value_with_rejection``, sequence_filtering.c) clips only the bad tail (high FWHM and
  background, low roundness), around the median. Siril's sigma is the standard deviation, recomputed after each clip
  until nothing moves; ours is 1.4826 x MAD in one pass (the Hampel identifier: Hampel 1974, JASA 69, 383), the
  robust estimate of the same Gaussian sigma that the outliers it must find cannot inflate. On Gaussian nights of 30
  to 2000 frames, Siril's limit at k = 2.5 falls at 2.41 to 2.49 sigma, ours at 2.5 (tests/test_frame_weights.py).
  Transparency, our measure of clouds, is filtered the same way (low tail).
  PixInsight's SubframeSelector approval expressions (``FWHMSigma < 2``...) are the same idea [P21].
* **Default k = 2.5 on FWHM, roundness and transparency; background off.** No tool filters by default (Siril keeps
  every frame, SubframeSelector's expressions are empty [S2][P21]); the published values are:
    - Siril's official Seestar script (SeeStar_Preprocessing v1.3, C. Richard, June 2024):
      ``seqapplyreg pp_light -filter-round=2.5k``: **2.5**, the only k a tool ships;
    - PixInsight's SubframeSelector documentation (C. Leger, 2017), its one example: ``EccentricitySigma < 2``, in
      mean-absolute-deviation units (about 1.6 Gaussian sigma): the strict end;
    - Zackay & Ofek [L6]: once the weights are right, "the rejection of images becomes a damaging operation": the
      lenient end. 2.5 everywhere follows Siril; the bench measures it against no filter (spec, validation 3).
  The background is off by default: a brighter sky (moon, dawn) only adds noise, which the weights already count;
  ``-filter-bkg`` stays available.
* The statistics of a filter are measured on all the aligned frames, those excluded by hand included, so excluding
  a frame by hand never changes the fate of another (as select_frames does).
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from .params import Auto

COLOURS = "RGB"
MAD_TO_SIGMA = 1.4826
WEIGHTINGS = ("noise", "sharp", "none")
CRITERIA = ("fwhm", "roundness", "transparency", "background")
#: the bad tail of each criterion: True when a high value is bad
HIGH_IS_BAD = {"fwhm": True, "roundness": False, "transparency": False, "background": True}
#: Siril's k in its official Seestar script (-filter-round=2.5k), used for every criterion
DEFAULT_K = 2.5
DEFAULT_FILTERS = {"fwhm": "2.5k", "roundness": "2.5k", "transparency": "2.5k", "background": ""}
NO_FILTERS = {c: "" for c in CRITERIA}
#: WBPP 2.5.0's default minimum weight (relative to the best frame)
MIN_WEIGHT = 0.05
#: PixInsight 1.9.6: a weight above this many times the upper quartile is given the median weight
GUARD_FACTOR = 10.0

# ------------------------------------------------------------------------------------------------- reason templates
NOT_ALIGNED = "not aligned"
BY_HAND = "excluded by hand"                    # stack.EXCLUDED
NO_MEASURE = "no measure of its noise or transparency"
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
GUARDED = "weight {ratio:.0f} × the upper quartile in {colour}: given the median weight (a wrong noise measure?)"

A_WEIGHT = ("inverse variance, the optimum for nebulae; Sharp stars would add {noise:.1f} % of noise for stars "
            "{finer:.1f} % finer")
A_WEIGHT_SAME = "inverse variance, the optimum for nebulae; the frames are alike: Sharp stars would change nothing"
A_WEIGHT_NO_FWHM = "inverse variance, the optimum for nebulae"
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
A_BACKGROUND_OFF = "off: a brighter sky only adds noise, which the weights already count"
A_OFF = "off"
A_NO_MEASURE = "no measure for this filter: not applied"
A_NO_SPREAD = "all frames alike: nothing to exclude"
A_MIN_HIT = "{n} frame(s) weigh less than {min:.0f} % of the best one: excluded"
A_MIN_NONE = "every frame weighs at least {low:.0f} % of the best one"
A_SUMMARY = "{kept} of {n} frames kept; the {out} left out would have lowered the noise by {pct:.1f} %"
A_SUMMARY_ALL = "all {n} aligned frames kept"

#: every template this module writes (their French goes to client/locale/fr/engine.json)
REASON_TEMPLATES = tuple(dict.fromkeys(
    [NOT_ALIGNED, BY_HAND, NO_MEASURE, LOW_WEIGHT, GUARDED, A_WEIGHT, A_WEIGHT_SAME, A_WEIGHT_NO_FWHM, A_OTHER_HIT,
     A_OTHER_NONE, A_BACKGROUND_OFF, A_OFF, A_NO_MEASURE, A_NO_SPREAD, A_MIN_HIT, A_MIN_NONE, A_SUMMARY,
     A_SUMMARY_ALL]
    + [d[c] for d in (_K, _PCT, _VALUE, _A_HIT, _A_NONE) for c in CRITERIA] + list(_K_PLAIN.values())))


@dataclass
class Reason:
    """Why a frame is left out, or what was done to its weight: a code, an English template and its numbers."""
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


# ------------------------------------------------------------------------------------------------- weights
def per_colour(x, n: int, name: str = "measure") -> np.ndarray:
    """(n, 3) float from a measure given per frame (n,) or per frame and colour (n, 3)."""
    a = np.asarray(x, float)
    if a.shape == (n,):
        return np.repeat(a[:, None], 3, axis=1)
    if a.shape == (n, 3):
        return a.copy()
    raise ValueError(f"{name}: expected shape ({n},) or ({n}, 3), got {a.shape}")


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


def planes_to_colours(planes, ch_of_k) -> np.ndarray:
    """(n, 3) from a measure per Bayer plane (n, 4) (frame_stats.npy[:, :, 0]: the sky level), the planes of one
    colour averaged (``ch_of_k``: the colour of each plane, io.frames.cfa_channel_map)."""
    p = np.asarray(planes, float)
    ch = np.asarray(ch_of_k)
    return np.stack([p[:, ch == c].mean(axis=1) if (ch == c).any() else p.mean(axis=1) for c in range(3)], axis=1)


def scheme_weights(transparency, noise, fwhm=None, scheme: str = "noise") -> np.ndarray:
    """Raw weights (n, 3), up to a constant, of frames normalised by their transparency:

    * ``noise``: transparency^2 / noise^2, the inverse of the normalised frame's variance [L6][L2][S2];
    * ``sharp``: the same / FWHM^2, Annis et al. 2014 [L6];
    * ``none``: 1.

    A weight that cannot be computed (no measure, zero noise) is NaN."""
    if scheme not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {scheme!r}: one of {WEIGHTINGS}")
    n = len(np.asarray(transparency))
    t, s = per_colour(transparency, n, "transparency"), per_colour(noise, n, "noise")
    if scheme == "none":
        return np.ones((n, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        w = t ** 2 / s ** 2
        if scheme == "sharp":
            if fwhm is None:
                raise ValueError("the Sharp stars weighting needs the FWHM of every frame")
            w = w / per_colour(fwhm, n, "fwhm") ** 2
    return np.where(np.isfinite(w) & (w > 0), w, np.nan)


def guard_weights(w, among) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """PixInsight 1.9.6's weight overshoot protection [P9]: per colour, a weight of the frames ``among`` above
    10 x the upper quartile of their weights gets their median weight; the others never change. Returns the weights
    and the guarded (frame, colour, weight / upper quartile)."""
    w = np.array(w, float)
    among = np.asarray(among, bool)
    hits = []
    for c in range(w.shape[1]):
        v = w[among, c]
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        q3, med = float(np.percentile(v, 75)), float(np.median(v))
        if not q3 > 0:
            continue
        for i in np.where(among & (w[:, c] > GUARD_FACTOR * q3))[0]:
            hits.append((int(i), c, float(w[i, c] / q3)))
            w[i, c] = med
    return w, hits


def stack_noise(w, var) -> float:
    """Noise of the weighted mean of frames of variance ``var`` (n, 3), sqrt(sum w^2 var) / sum w, the mean of the
    colours (only the frames with a weight count). For the inverse variance it is 1 / sqrt(sum 1 / var)."""
    w, var = np.asarray(w, float), np.asarray(var, float)
    out = []
    for c in range(w.shape[1]):
        m = np.isfinite(w[:, c]) & (w[:, c] > 0) & np.isfinite(var[:, c])
        if m.any():
            out.append(math.sqrt(float(np.sum(w[m, c] ** 2 * var[m, c]))) / float(np.sum(w[m, c])))
    return float(np.mean(out)) if out else float("nan")


def stack_fwhm(w, fwhm) -> float:
    """FWHM of the stacked stars: sqrt(sum w FWHM^2 / sum w), the second moment of the weighted sum of the frames'
    star images (exact for the second moment, the FWHM of a Gaussian), the mean of the colours."""
    w, f = np.asarray(w, float), np.asarray(fwhm, float)
    out = []
    for c in range(w.shape[1]):
        m = np.isfinite(w[:, c]) & (w[:, c] > 0) & np.isfinite(f[:, c])
        if m.any():
            out.append(math.sqrt(float(np.sum(w[m, c] * f[m, c] ** 2) / np.sum(w[m, c]))))
    return float(np.mean(out)) if out else float("nan")


def _restricted(w, among) -> np.ndarray:
    out = np.where(np.asarray(among, bool)[:, None], w, 0.0)
    return np.where(np.isfinite(out), out, 0.0)


# ------------------------------------------------------------------------------------------------- the result
@dataclass
class FrameWeights:
    weighting: str
    weights: np.ndarray            # (n, 3): relative weight per colour (best kept frame = 1), 0 when left out
    frame_weight: np.ndarray       # (n,): the mean of the colours (frames.json "weight")
    kept: np.ndarray               # (n,) bool
    reasons: np.ndarray            # (n,) object: English reasons joined by "; ", "" for a kept frame
    codes: np.ndarray              # (n,) object: the codes, joined by ","
    exclusions: list[Reason]       # one per frame and cause
    notes: list[Reason]            # weights changed by the guard (the frame is kept)
    criteria: dict[str, CriterionStats]
    #: per weighting, against the inverse variance on the same frames: noise_pct (+ = noisier), fwhm_pct (- = finer)
    costs: dict[str, dict[str, float]]
    #: what the left-out frames cost: noise_pct (+ = the stack is noisier without them), mag, frames
    selection_cost: dict[str, float]
    auto: dict[str, Auto]          # per parameter of the stack (weighting, filter_*, min_weight)
    suggestion: Auto               # the summary: frames kept and what the others would have brought

    def reasons_in(self, lang: str = "en") -> list[str]:
        """Every frame's reason in ``lang`` ("" when kept)."""
        by = {}
        for r in self.exclusions:
            by.setdefault(r.frame, []).append(r.text(lang))
        return ["; ".join(by.get(i, [])) for i in range(len(self.kept))]

    def as_dict(self) -> dict:
        """A JSON-ready summary (diagnostics.json): thresholds, costs, guarded weights, exclusions."""
        def num(x):
            return None if x is None or not np.isfinite(x) else float(x)
        return {
            "weighting": self.weighting,
            "kept": int(self.kept.sum()),
            "criteria": {c: {"spec": s.spec, "status": s.status, "median": num(s.median), "sigma": num(s.sigma),
                             "limit": num(s.limit), "excluded": s.excluded} for c, s in self.criteria.items()},
            "costs": {k: {m: num(x) for m, x in v.items()} for k, v in self.costs.items()},
            "selection_cost": {k: num(v) for k, v in self.selection_cost.items()},
            "guarded": [{"frame": r.frame, "colour": r.args["colour"], "ratio": r.args["ratio"]} for r in self.notes],
            "exclusions": [{"frame": r.frame, "code": r.code, "reason": r.text_en} for r in self.exclusions],
        }


def weigh_frames(transparency, noise, fwhm=None, roundness=None, background=None, *, aligned=None, exclude=None,
                 weighting: str = "noise", filters: dict | None = None, min_weight: float = MIN_WEIGHT,
                 guard: bool = True) -> FrameWeights:
    """Weights per frame and colour, the frames left out with their reasons, and the automatic values.

    ``transparency`` (n,) or (n, 3): the factor each frame is divided by to match the reference; ``noise`` (n,) or
    (n, 3): the pixel noise of the raw frame (its ADU); ``fwhm`` (n,) or (n, 3), px (Moffat); ``roundness`` (n,):
    minor / major axis (1 = round; from an eccentricity e: sqrt(1 - e^2)); ``background`` (n,) or (n, 3): the sky
    level; ``aligned`` (n,) bool (None: all); ``exclude`` (n,) bool: excluded by hand; ``weighting`` one of
    WEIGHTINGS; ``filters`` {criterion: Siril-style spec}, merged over DEFAULT_FILTERS (NO_FILTERS: none);
    ``min_weight`` relative to the best frame (0: none); ``guard`` the weight overshoot protection.

    Order: not aligned, by hand, the filters (statistics on all aligned frames), the weights, the guard, the
    normalisation (best kept frame = 1, per colour), the minimum weight."""
    if weighting not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {weighting!r}: one of {WEIGHTINGS}")
    n = len(np.asarray(transparency))
    T = per_colour(transparency, n, "transparency")
    S = per_colour(noise, n, "noise")
    F3 = per_colour(fwhm, n, "fwhm") if fwhm is not None else None
    aligned = np.ones(n, bool) if aligned is None else np.asarray(aligned, bool)
    hand = np.zeros(n, bool) if exclude is None else np.asarray(exclude, bool)
    specs = {**DEFAULT_FILTERS, **(filters or {})}
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

    # ---- the weights
    raw = {s: scheme_weights(T, S, F3, s) for s in WEIGHTINGS if s != "sharp" or F3 is not None}
    if weighting not in raw:
        raise ValueError("the Sharp stars weighting needs the FWHM of every frame")
    measured = np.isfinite(raw[weighting]).all(axis=1)          # always true for "none"
    for i in np.where(judge & ~failed & ~measured)[0]:
        exclusions.append(Reason(int(i), "no_measure", NO_MEASURE))
    cand = judge & ~failed & measured
    comparable = np.all([np.isfinite(x).all(axis=1) for x in raw.values()], axis=0)    # every weighting computable

    w = _restricted(raw[weighting], cand)
    notes: list[Reason] = []
    if guard and weighting != "none":
        w, hits = guard_weights(np.where(cand[:, None], w, np.nan), cand)
        w = np.where(np.isfinite(w), w, 0.0)
        by: dict[int, list] = {}
        for i, c, r in hits:                       # one note per frame, its guarded colours together
            by.setdefault(i, []).append((c, r))
        notes = [Reason(i, "guard", GUARDED, {"ratio": max(r for _, r in cr),
                                              "colour": ", ".join(COLOURS[c] for c, _ in sorted(cr))})
                 for i, cr in sorted(by.items())]

    def relative(w, among):
        mx = np.where(among[:, None], w, 0.0).max(axis=0) if among.any() else np.zeros(3)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(among[:, None] & (mx > 0), w / np.where(mx > 0, mx, 1.0), 0.0)

    rel = relative(w, cand)
    low = cand & (rel.mean(axis=1) < min_weight) if min_weight > 0 else np.zeros(n, bool)
    lowest = float(rel[cand].mean(axis=1).min()) if cand.any() else float("nan")
    for i in np.where(low)[0]:
        exclusions.append(Reason(int(i), "weight", LOW_WEIGHT, {"pct": 100.0 * float(rel[i].mean()),
                                                               "min": 100.0 * min_weight}))
    kept = cand & ~low
    rel = relative(rel, kept)

    # ---- what each weighting costs on the frames kept, and what the left-out frames would have brought
    var = np.where(T > 0, S ** 2 / np.where(T > 0, T, 1.0) ** 2, np.nan)

    def guarded(s, among):
        x = _restricted(raw[s], among)
        if s != "none" and guard:
            x, _ = guard_weights(np.where(among[:, None], x, np.nan), among)
            x = np.where(np.isfinite(x), x, 0.0)
        return x

    costs: dict[str, dict[str, float]] = {}
    same = kept & comparable
    if same.any():
        base_n = stack_noise(guarded("noise", same), var)
        base_f = stack_fwhm(guarded("noise", same), F3) if F3 is not None else float("nan")
        for s in raw:
            x = guarded(s, same)
            costs[s] = {"noise_pct": 100.0 * (stack_noise(x, var) / base_n - 1.0),
                        "fwhm_pct": 100.0 * (stack_fwhm(x, F3) / base_f - 1.0) if F3 is not None else float("nan")}
    # the frames the filters and the minimum weight left out, against all the frames that could be weighed
    noisy = np.isfinite(raw["noise"]).all(axis=1)
    pool = judge & noisy
    out = int((judge & ~kept).sum())
    sel = {"frames": float(out), "noise_pct": 0.0, "mag": 0.0}
    if (pool & ~kept).any() and (kept & noisy).any():
        ratio = stack_noise(guarded("noise", kept & noisy), var) / stack_noise(guarded("noise", pool), var)
        sel = {"frames": float(out), "noise_pct": 100.0 * (ratio - 1.0), "mag": 2.5 * math.log10(ratio)}

    # ---- the per-frame outputs
    reasons = np.full(n, "", dtype=object)
    codes = np.full(n, "", dtype=object)
    for r in exclusions:
        reasons[r.frame] = r.text_en if not reasons[r.frame] else reasons[r.frame] + "; " + r.text_en
        codes[r.frame] = r.code if not codes[r.frame] else codes[r.frame] + "," + r.code
    auto = _autos(criteria, specs, costs, F3 is not None, low, lowest, min_weight)
    if out:
        suggestion = Auto(int(kept.sum()), A_SUMMARY, {"kept": int(kept.sum()), "n": int(judge.sum()), "out": out,
                                                       "pct": sel["noise_pct"]})
    else:
        suggestion = Auto(int(kept.sum()), A_SUMMARY_ALL, {"n": int(judge.sum())})
    return FrameWeights(weighting, rel, rel.mean(axis=1), kept, reasons, codes, exclusions, notes, criteria, costs,
                        sel, auto, suggestion)


def _autos(criteria, specs, costs, have_fwhm, low, lowest, min_weight) -> dict[str, Auto]:
    """The automatic value and reason of each parameter (the default values: the reasons say what they do here)."""
    a: dict[str, Auto] = {}
    sharp = costs.get("sharp")
    if not have_fwhm or sharp is None:
        a["weighting"] = Auto("noise", A_WEIGHT_NO_FWHM)
    elif sharp["noise_pct"] < 0.05 and abs(sharp["fwhm_pct"]) < 0.05:
        a["weighting"] = Auto("noise", A_WEIGHT_SAME)
    else:
        a["weighting"] = Auto("noise", A_WEIGHT, {"noise": sharp["noise_pct"], "finer": -sharp["fwhm_pct"]})
    for c in CRITERIA:
        st, spec = criteria[c], DEFAULT_FILTERS[c]
        flt = parse_filter(specs[c])
        key = f"filter_{c}"
        if parse_filter(spec).kind == "off":
            a[key] = Auto(spec, A_BACKGROUND_OFF if c == "background" else A_OFF)
        elif st.status == "no measure":
            a[key] = Auto(spec, A_NO_MEASURE)
        elif st.status == "no spread":
            a[key] = Auto(spec, A_NO_SPREAD)
        elif flt.kind == "k" and str(flt) == spec:
            args = {"n": len(st.excluded), "k": flt.amount, "limit": st.limit}
            if c == "transparency":
                args["pct"] = 100.0 * st.limit / st.median if st.median > 0 else float("nan")
            a[key] = Auto(spec, (_A_HIT if st.excluded else _A_NONE)[c], args)
        else:                                      # another setting than the default is in use: what it did
            a[key] = Auto(spec, A_OTHER_HIT if st.excluded else A_OTHER_NONE,
                          {"n": len(st.excluded)} if st.excluded else None)
    if low.any():
        a["min_weight"] = Auto(MIN_WEIGHT, A_MIN_HIT, {"n": int(low.sum()), "min": 100.0 * min_weight})
    else:
        a["min_weight"] = Auto(MIN_WEIGHT, A_MIN_NONE, {"low": 100.0 * lowest if np.isfinite(lowest) else 100.0})
    return a
