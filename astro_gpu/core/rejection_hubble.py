"""Outlier rejection by the Hubble method (STScI DrizzlePac / AstroDrizzle ``driz_cr``) for the Bayer drizzle.

The method, as AstroDrizzle runs it [L2][L3][L4]:

1. a **median image** on the output grid, built from the frames drizzled one by one (``driz_separate`` then
   ``create_median``); ``combine_type`` ``median`` or ``minmed`` (the median, or the minimum where the median is
   significantly above it, which protects against outliers present in several of a few frames);
2. that median **blotted** back onto each frame's own pixel grid, and its derivative ``qderiv`` (per input pixel: the
   largest absolute difference to its 4 nearest neighbours);
3. ``driz_cr``: a pixel is an outlier when |frame - blot| > SNR1 x noise + scale1 x deriv; then each pixel next to an
   outlier (its 3 x 3 neighbourhood) is one when |frame - blot| > SNR2 x noise + scale2 x deriv; the noise is the
   sensor's, sqrt(gain x |blot + sky| + read noise^2) / gain (in ADU; gain in e-/ADU, read noise in e-).
   The slope term lets star edges move by a fraction of a pixel (registration errors, seeing) without being rejected.

Published defaults, used here: driz_cr_snr = "3.5 3.0", driz_cr_scale = "1.2 0.7", driz_cr_grow = 1 (nothing more than
the 3 x 3 neighbourhood test), driz_cr_ctegrow = 0 (no CTE tail: a CCD readout feature, not a CMOS one) [L4];
combine_nsigma = "4 3" and combine_grow = 1 for minmed, combine_nhigh = 0 for the median [L3]. The choice of the
prediction follows the project's spec (docs/spec/STACKING.md, point 2, after the DrizzlePac advice [L3]: minmed is
"highly recommended for three images, and is good for four to six", the median "works equally well as minmed down to
approximately four images"): per output point and per colour, **minmed when 3 to 5 samples cover the point, the median
from 6 on, and no rejection under 3 samples** (reported: ``Prediction.summary`` and ``FrameRejection.counts``).

The equivalences with our Bayer drizzle (stages/stack.py), where every raw sample is dropped at its exact registered
position (Xo, Yo) on the output grid, one colour plane per CFA colour:

- *single drizzle + median*: per colour, a frame's value at an output pixel is the sample of that colour whose drop
  (side pixfrac x scale output pixels, the same square drop as the final drizzle) covers the pixel's centre. A frame's
  drops of one colour never overlap (same-colour samples are 2 sensor pixels apart, or sqrt(2) for green, while the drop
  is at most one sensor pixel wide), so each point gets at most one sample per frame: the median of a point is the
  median of the real samples around it, and their number is the "samples covering the point" of the spec (at scale 1,
  drop 1: a quarter of the frames in red and blue, half in green). AstroDrizzle's ``combine_maskpt`` (leaving out the
  pixels a single drizzle barely covers) has no other equivalent: a sample counts or does not.
- *blot*: the colour's median image interpolated at the sample's position (Xo, Yo) (AstroDrizzle's ``blot_interp`` is
  poly5 by default, poly3 among its options; here torch's bicubic, the nearest it offers; bilinear as an option).
- *qderiv*: the derivative of the blotted image of the sample's own colour on the frame's grid: that colour's median
  interpolated at the positions of the sample's 4 sensor neighbours **as if they were of its colour** (its position
  plus or minus one sensor pixel along the frame's axes, the step taken from the same-colour samples two pixels away,
  so that the colour registration of stack.py, which moves each colour differently, does not enter the derivative).
- *3 x 3 neighbourhood of driz_cr*: on the sensor grid, all colours together (a cosmic ray or a trail hits adjacent
  sensor pixels whatever their colour), exactly as AstroDrizzle does on the frame's own grid.
- *noise*: the sensor model of stack.py (``gain`` in e-/ADU and ``read_noise`` in ADU of the calibrated frames, as
  ``Stack.auto`` measures them): the predicted raw value of the sample, blot x t + grad (t: the frame's photometric
  scale, grad: its additive sky difference, both from stack.py's normalisation), has the variance |raw| / gain +
  read_noise^2 in the frame's ADU, divided by t^2 in the normalised units the samples are compared in. This is
  sqrt(gain |blot + sky| + rn_e^2) / gain of driz_cr written with the read noise in ADU (rn_e = gain x read_noise).
  As in AstroDrizzle, the noise of the median itself is not added, and the flat field is not in the model.
- *minmed's sigma*: the documentation's sqrt(M + S + R^2) as AstroDrizzle's ``min_med`` code computes it, summed over
  the frames covering the point: sqrt(sum_i ((med t_i + o_i) / gain + read_noise^2)), against the median and the
  minimum weighted by the summed exposure sum_i t_i (our t_i plays the exposure time, o_i the frame's sky difference);
  the median of minmed leaves out the highest sample first (nhigh = 1, as ``min_med`` does). The minimum replaces the
  median where med - min > 4 sigma, or 3 sigma next to a point where it did (combine_grow = 1). As DrizzlePac warns,
  this biases the prediction low: on clean sky without the slope term, 1 % of the samples at minmed points are
  rejected; with the published slope term, 0.004 to 0.008 % (tests).
- *minmed and low outliers*: minmed takes the minimum, so an unmasked low outlier (a dead pixel) at a point with 3 to 5
  samples becomes the prediction there, and the hole's slope hides the samples around it (tests: 317 of 400 dead
  samples caught, 400 of 400 with the median everywhere). AstroDrizzle relies on the bad-pixel masks (DQ) for those:
  here stack.py's ``bad`` (the hot, cold and dusty pixels the calibration flags) must stay out of the store
  (``keep = ~bad``).
- *memory*: AstroDrizzle keeps every single drizzle on disk. Here the median holds at most K frames (``median_frames``:
  all of them when they fit the memory budget, else K spread evenly over the session: with 30 frames at scale 1, about
  7 red samples per point, the median's noise is about half a sample's). Every frame is then tested against it.

Cost, 30 synthetic frames of 6000 x 4000 at scale 1, frame loading not counted: 11 s on the RTX 4060
Ti (store 2.8 s, median 1.4 s, driz_cr 6.4 s; 6.9 GB at peak with the store on the GPU), about 4 min on the CPU (driz_cr
5.4 s per frame, mostly the bicubic blot). The GPU and the CPU give the same masks, bit for bit.

How stages/stack.py calls it (the integration; ``frame_model``, ``deposit``, ``used``, ``transp``, ``norm``, ``chmap``,
``S``, ``PIX``, ``Ho``, ``Wo``, ``dev`` are stack.py's own names):

- **pass 1** (unchanged): the weighted mean and the colour registration ``chan_fix`` measured on it. The Hubble rejection
  needs the final sample positions, so it comes after.
- **pass 2** (replaces the "cleaned prediction" pass): the median store, then the prediction::

      from ..core import rejection_hubble as rh
      pick = rh.median_frames(len(used), Ho, Wo)               # positions in ``used``; all frames if they fit
      store = rh.new_store(len(pick), Ho, Wo, device="cpu")    # 6 bytes per output pixel and frame; dev if room
      for k, j in enumerate(pick):
          val, Xo, Yo, sat, bad, grad, _, _ = frame_model(used[j])
          rh.add_to_store(store, k, val, Xo, Yo, chmap, ~bad, S, PIX)
      offs = np.array([norm[int(used[j])][0][:, 0] for j in pick])    # the sky difference of each frame (frame ADU)
      pred = rh.build_prediction(store, transp[used[pick]], offs, p.gain, p.read_noise, device=dev)
      del store
      ctx.log(pred.describe())

- **pass 3** (the rejection, replacing ``judge`` / ``reject``; the half-stacks and the rest unchanged)::

      tally = rh.RejectionTally(Ho, Wo, dev)
      for i in used:
          val, Xo, Yo, sat, bad, grad, _, near = frame_model(i)
          r = rh.driz_cr(val, Xo, Yo, chmap, pred, float(transp[i]), grad, p.gain, p.read_noise, exempt=sat)
          hi, lo = r.high, r.low                           # keep = ~bad & ~hi & ~lo, as today
          rej_frames[int(i)] = tally.add(Xo, Yo, r)      # (low, high, untested) of the frame, for diagnostics.json
          ...deposit as today...
      tally.save(ctx.out(REJECTION))                     # same keys as RejectionMap (low, high, factor, grid) + untested
      s = tally.summary(); ctx.metric("rejected_pct", s["rejected_pct"]); ctx.metric("untested_pct", s["untested_pct"])

Inputs: the arrays of ``frame_model`` (sensor grid H x W, on any device; ``chmap`` the CFA colour 0/1/2 of each sensor
pixel), the frame's scale t (``transp[i]``) and sky difference ``grad`` (H x W, frame ADU), the sensor model
(``p.gain``, ``p.read_noise``). Outputs: per frame, the rejected samples (``FrameRejection.high`` / ``.low``, boolean
H x W: the per-frame bad-pixel masks) and the samples left untested (fewer than 3 samples around them); for the stack,
the rejection rate in % of the samples landing on it, and the low / high / untested maps on the output grid
(``RejectionTally``), for the Inspect mode next to Siril's 0.1-0.5 % mark [S12].

Sources (docs/pilot/STATE_OF_THE_ART_STACKING.md):
[L2] DrizzlePac Handbook v3, STScI, ch. 4 (the AstroDrizzle steps: separate drizzle, median, blot, driz_cr):
     https://hst-docs.stsci.edu/drizzpac
[L3] AstroDrizzle, "Create Median Image" (combine_type, combine_nsigma "4 3", combine_grow 1, combine_nhigh 0, minmed's
     sigma = sqrt(M + S + R^2)): https://drizzlepac.readthedocs.io/en/deployment/median.html ; code: drizzlepac/minmed.py
     (``min_med``), https://github.com/spacetelescope/drizzlepac
[L4] AstroDrizzle, ``driz_cr`` (driz_cr_snr "3.5 3.0", driz_cr_scale "1.2 0.7", driz_cr_grow 1, driz_cr_ctegrow 0):
     https://drizzlepac.readthedocs.io/en/latest/api/drizzlepac.drizCR.drizCR.html ; code: drizzlepac/drizCR.py
     (``_driz_cr``) and drizzlepac/quickDeriv.py (``qderiv``); blot: drizzlepac/ablot.py (blot_interp "poly5").
[S12] Siril tutorial, the rejection rate to aim at (0.1 to 0.5 %): https://siril.org/tutorials/tuto-manual/
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

#: colour names of the planes, for the summaries
COLOURS = ("red", "green", "blue")
#: a float16 store keeps values below this (the calibrated cache is float16 below 60000)
F16_MAX = 65000.0
#: the rejection maps are reduced by a whole factor until their longer side is at most this (as stack.py's)
REJECTION_SIDE = 4096


@dataclass(frozen=True)
class HubbleParams:
    """The method's settings, each with its published default."""
    #: driz_cr_snr: the noise multiples of the first test and of the neighbours' test [L4]
    snr: tuple[float, float] = (3.5, 3.0)
    #: driz_cr_scale: the multiples of the blotted median's derivative of the two tests [L4]
    scale: tuple[float, float] = (1.2, 0.7)
    #: driz_cr_grow: side of the box the final mask is grown by (1: nothing beyond the 3 x 3 neighbour test) [L4]
    grow: int = 1
    #: combine_nsigma: minmed takes the minimum where the median is this many sigmas above it (the second: next to a
    #: point where it did) [L3]
    nsigma: tuple[float, float] = (4.0, 3.0)
    #: combine_grow: radius of the box around a point where minmed took the minimum [L3]
    combine_grow: int = 1
    #: combine_nhigh: highest samples left out of the median (minmed always leaves out one, as min_med does) [L3]
    nhigh: int = 0
    #: fewer samples than this covering a point: no prediction good enough to reject anything there (the spec)
    min_samples: int = 3
    #: up to this many samples: minmed; above: the median (the spec, after [L3])
    minmed_max: int = 5
    #: the blot's interpolation: "bicubic" (closest to AstroDrizzle's poly5 / poly3) or "bilinear"
    interp: str = "bicubic"


HUBBLE = HubbleParams()


# ------------------------------------------------------------------------------------------------ the median store
def median_frames(n: int, Ho: int, Wo: int, budget_bytes: float = 4 * 2**30, max_frames: int = 255) -> np.ndarray:
    """Which of the ``n`` frames (positions 0..n-1) make the median: all of them when their store fits
    ``budget_bytes`` (6 bytes per output pixel and frame: a float16 per colour), else as many as fit, spread evenly
    over the session; at most ``max_frames`` (the counts are bytes), at least min(n, 3)."""
    k = int(min(n, max_frames, max(min(n, 3), budget_bytes // (6 * Ho * Wo))))
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


def new_store(k: int, Ho: int, Wo: int, device="cpu") -> torch.Tensor:
    """An empty median store for ``k`` frames: (k, 3, Ho, Wo) float16, NaN where a frame has no sample."""
    return torch.full((k, 3, Ho, Wo), float("nan"), dtype=torch.float16, device=device)


def add_to_store(store: torch.Tensor, slot: int, val: torch.Tensor, Xo: torch.Tensor, Yo: torch.Tensor,
                 chmap: torch.Tensor, keep: torch.Tensor, scale: float, pixfrac: float) -> None:
    """Put one frame in the median store (the equivalent of its single drizzle): each kept sample ``val`` (normalised,
    sensor grid H x W) at every output pixel of its colour whose centre its drop covers (drop: side ``pixfrac`` x
    ``scale`` output pixels around (Xo, Yo), half-open, so that a pixel centre is in one drop at most per colour). A
    centre a rounding error puts in two drops of one colour keeps the larger value (deterministic on any device)."""
    _, C, Ho, Wo = store.shape
    dev = val.device
    h = pixfrac * scale / 2.0
    m = keep & torch.isfinite(val)
    v, x, y, c = val[m].float(), Xo[m], Yo[m], chmap[m].long()
    tmp = torch.full((C * Ho * Wo,), float("-inf"), device=dev)
    n = max(1, int(np.ceil(2 * h - 1e-9)))
    j0, i0 = torch.ceil(x - h), torch.ceil(y - h)
    for di in range(n):
        iy = i0 + di
        oky = (iy < y + h) & (iy >= 0) & (iy < Ho)
        for dj in range(n):
            jx = j0 + dj
            ok = oky & (jx < x + h) & (jx >= 0) & (jx < Wo)
            idx = c[ok] * (Ho * Wo) + iy[ok].long() * Wo + jx[ok].long()
            tmp.scatter_reduce_(0, idx, v[ok], reduce="amax")
    out = torch.where(torch.isfinite(tmp), tmp.clamp(-F16_MAX, F16_MAX), torch.full_like(tmp, float("nan")))
    store[slot] = out.view(C, Ho, Wo).to(store.device, torch.float16)


# ------------------------------------------------------------------------------------------------ the prediction
@dataclass
class Prediction:
    """The median image per colour (``pred``, 3 x Ho x Wo float32, holes filled from their neighbours so that it can
    be interpolated anywhere), the number of samples behind each point (``count``, uint8) and what was used where
    (``summary``)."""
    pred: torch.Tensor
    count: torch.Tensor
    params: HubbleParams
    summary: dict = field(default_factory=dict)

    def describe(self) -> str:
        """One line for the log, e.g. "samples per point: red 9, green 18, blue 9 (median); 0.2 % of the red points
        have fewer than 3: not tested"."""
        s = self.summary
        parts = [f"{c} {s[c]['samples_per_point']:g}" for c in COLOURS]
        kinds = sorted({s[c]["prediction"] for c in COLOURS})
        txt = f"Hubble rejection: samples per point {', '.join(parts)} ({' / '.join(kinds)})"
        low = [f"{s[c]['too_few_pct']:.2g} % of the {c} points" for c in COLOURS if s[c]["too_few_pct"] >= 0.05]
        if low:
            txt += f"; fewer than {self.params.min_samples} samples (not tested) at " + ", ".join(low)
        return txt


def build_prediction(store: torch.Tensor, t, offset=None, gain: float = 1.0, read_noise: float = 0.0,
                     params: HubbleParams = HUBBLE, device=None, tile_elems: int = 1 << 26) -> Prediction:
    """The prediction from a filled median store (k, 3, Ho, Wo): per point and colour, the median of its n samples
    (n >= minmed_max + 1, leaving out the ``nhigh`` highest), minmed (min_samples <= n <= minmed_max), or the median of
    what there is (n < min_samples: a value to interpolate from, nothing is tested there).

    ``t`` (k,): the photometric scale of each stored frame (stack.py's transparency: the frame's ADU per normalised
    unit); ``offset`` (k, 3) or None: its additive sky difference per colour in its own ADU (stack.py: the constant
    term of its gradient); with ``gain`` (e-/ADU) and ``read_noise`` (ADU), the noise minmed compares with. Computed
    in tiles of about ``tile_elems`` values on ``device`` (default: the store's)."""
    K, C, Ho, Wo = store.shape
    dev = torch.device(device) if device is not None else store.device
    p = params
    tk = torch.as_tensor(np.asarray(t, np.float64), dtype=torch.float32, device=dev).view(K)
    ok = torch.zeros(K, C, device=dev) if offset is None else \
        torch.as_tensor(np.asarray(offset, np.float64), dtype=torch.float32, device=dev).view(K, C)
    pred = torch.empty(C, Ho, Wo, device=dev)
    count = torch.empty(C, Ho, Wo, dtype=torch.uint8, device=dev)
    zmin = torch.full((C, Ho, Wo), float("nan"), device=dev)      # minmed: (median - minimum) / sigma
    vmin = torch.full((C, Ho, Wo), float("nan"), device=dev)      # minmed: the minimum
    rows = max(1, tile_elems // max(1, K * C * Wo))
    for r0 in range(0, Ho, rows):
        r1 = min(Ho, r0 + rows)
        blk = store[:, :, r0:r1].to(dev).float()                      # (K, C, r, Wo)
        fin = torch.isfinite(blk)
        n = fin.sum(0)                                                # (C, r, Wo)
        s = torch.sort(blk.permute(1, 2, 3, 0), dim=-1).values        # NaN last
        del blk

        def kth(k):
            return s.gather(-1, k.clamp(0, K - 1).unsqueeze(-1)).squeeze(-1)

        m = (n - p.nhigh).clamp(min=1)
        med = 0.5 * (kth((m - 1) // 2) + kth(m // 2))
        mm = (n >= p.min_samples) & (n <= p.minmed_max)
        m2 = (n - 1).clamp(min=1)                                     # min_med: the highest left out
        med2 = 0.5 * (kth((m2 - 1) // 2) + kth(m2 // 2))
        mn = s[..., 0]
        del s
        f = fin.float()
        T = torch.einsum("k,kcrw->crw", tk, f)
        O = torch.einsum("kc,kcrw->crw", ok, f)
        del f, fin
        var = ((med2 * T + O) / gain + n * read_noise ** 2).clamp(min=0)
        z = (med2 - mn) * T / var.sqrt().clamp(min=1e-12)
        nanv = torch.full_like(med, float("nan"))
        pred[:, r0:r1] = torch.where(n == 0, nanv, torch.where(mm, med2, med))
        zmin[:, r0:r1] = torch.where(mm, z, nanv)
        vmin[:, r0:r1] = torch.where(mm, mn, nanv)
        count[:, r0:r1] = n.clamp(max=255).to(torch.uint8)
    # minmed: the minimum where the median is nsigma1 above it, or nsigma2 next to such a point (combine_grow)
    first = zmin > p.nsigma[0]
    near = torch.stack([_dilate(first[c], int(p.combine_grow)) for c in range(C)])
    use_min = zmin > torch.where(near, torch.full_like(zmin, p.nsigma[1]), torch.full_like(zmin, p.nsigma[0]))
    pred = torch.where(use_min, vmin, pred)
    del zmin, vmin, first, near
    summary = _summary(count, use_min, p)
    del use_min
    return Prediction(_fill_holes(pred), count, p, summary)


def _summary(count: torch.Tensor, use_min: torch.Tensor, p: HubbleParams) -> dict:
    out = {}
    for c, name in enumerate(COLOURS):
        n = count[c]
        tot = n.numel()
        sub = n[::max(1, n.shape[0] // 1024), ::max(1, n.shape[1] // 1024)]
        cov = sub[sub > 0]
        spp = float(cov.float().median()) if cov.numel() else 0.0
        k_med = int((n > p.minmed_max).sum())
        k_mm = int(((n >= p.min_samples) & (n <= p.minmed_max)).sum())
        k_few = int(((n > 0) & (n < p.min_samples)).sum())
        k_used = int(use_min[c].sum())
        out[name] = {"samples_per_point": spp,
                     "prediction": "median" if spp > p.minmed_max else ("minmed" if spp >= p.min_samples else "none"),
                     "median_pct": round(100 * k_med / tot, 3), "minmed_pct": round(100 * k_mm / tot, 3),
                     "too_few_pct": round(100 * k_few / tot, 3), "empty_pct": round(100 * int((n == 0).sum()) / tot, 3),
                     "minimum_used_pct": round(100 * k_used / max(k_mm, 1), 3)}
    return out


def _dilate(mask: torch.Tensor, r: int, r_after: int | None = None) -> torch.Tensor:
    """``mask`` (H, W) grown by a box: a pixel is set when a set pixel lies within ``r`` before it and ``r_after``
    (default ``r``) after it on each axis (outside the image: nothing). Works on the set pixels only (they are few)."""
    ra = r if r_after is None else r_after
    if r <= 0 and ra <= 0:
        return mask.clone()
    H, W = mask.shape
    pts = torch.nonzero(mask)
    out = torch.zeros_like(mask)
    if pts.numel() == 0:
        return out
    for dy in range(-ra, r + 1):
        y = pts[:, 0] + dy
        for dx in range(-ra, r + 1):
            x = pts[:, 1] + dx
            ok = (y >= 0) & (y < H) & (x >= 0) & (x < W)
            out[y[ok], x[ok]] = True
    return out


def _fill_holes(pred: torch.Tensor, rounds: int = 3) -> torch.Tensor:
    """The points without a sample (NaN) filled with the mean of their neighbours that have one, ``rounds`` times;
    what is still empty, with the colour's median. Only for the interpolation: those points are never tested."""
    out = pred.clone()
    for c in range(out.shape[0]):
        x = out[c]
        v = torch.isfinite(x)
        if bool(v.all()):
            continue
        if not bool(v.any()):
            out[c] = 0.0
            continue
        for _ in range(rounds):
            vf = v.float()[None, None]
            num = F.avg_pool2d(torch.where(v, x, torch.zeros_like(x))[None, None], 3, 1, 1, count_include_pad=False)
            den = F.avg_pool2d(vf, 3, 1, 1, count_include_pad=False)
            new = (den[0, 0] > 0) & ~v
            x = torch.where(new, num[0, 0] / den[0, 0].clamp(min=1e-12), x)
            v = v | new
        out[c] = torch.where(v, x, x[v].median())
    return out


# ------------------------------------------------------------------------------------------------ driz_cr
@dataclass
class FrameRejection:
    """One frame judged (sensor grid H x W, boolean): the samples rejected above (``high``) and below (``low``) the
    prediction, those tested and those on the stack but left untested (fewer than ``min_samples`` samples around
    them); ``counts``: the same counted, with ``on_stack`` (the samples landing on a point that has a prediction: the
    denominator of the rejection rate), ``exempt`` (on the stack but never judged: saturated...) and ``z_spread`` (the
    robust spread of the tested residuals in noise sigmas: about 1 when the sensor model is right); with
    ``diagnostics``: ``z``, the residuals in noise sigmas, and ``limit``, the first test's threshold in noise sigmas
    (SNR1 + scale1 x deriv / noise), H x W, 0 where not tested."""
    high: torch.Tensor
    low: torch.Tensor
    tested: torch.Tensor
    untested: torch.Tensor
    counts: dict
    z: torch.Tensor | None = None
    limit: torch.Tensor | None = None


def _steps(A: torch.Tensor, dim: int) -> torch.Tensor:
    """d A / d (sensor pixel) along ``dim``, from the samples two pixels away (same CFA colour, so the same colour
    registration), at every pixel (the nearest available difference at the edges)."""
    n = A.shape[dim]
    if n < 3:
        return torch.zeros_like(A)
    d = (A.narrow(dim, 2, n - 2) - A.narrow(dim, 0, n - 2)) / 2.0
    return torch.cat([d.narrow(dim, 0, 1), d, d.narrow(dim, n - 3, 1)], dim)


def _interp(plane: torch.Tensor, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
    """``plane`` (Ho, Wo) at the output positions (x, y) (pixel centres at integers)."""
    Ho, Wo = plane.shape
    g = torch.stack([(x + 0.5) / Wo * 2 - 1, (y + 0.5) / Ho * 2 - 1], -1).view(1, 1, -1, 2)
    return F.grid_sample(plane[None, None], g, mode=mode, padding_mode="border", align_corners=False).view(-1)


def driz_cr(val: torch.Tensor, Xo: torch.Tensor, Yo: torch.Tensor, chmap: torch.Tensor, pred: Prediction,
            t: float, grad, gain: float, read_noise: float, exempt: torch.Tensor | None = None,
            diagnostics: bool = False, batch: int = 1 << 22) -> FrameRejection:
    """The samples of one frame rejected by ``driz_cr`` against the prediction ``pred``.

    ``val``: the frame normalised as stack.py deposits it (H x W, ADU of the reference); ``Xo``, ``Yo``: where each
    sample lands on the output grid; ``chmap``: its colour (0, 1, 2); ``t``: the frame's photometric scale and ``grad``
    its additive sky difference (H x W or a number), both in the frame's own ADU (raw = val x t + grad); ``gain``
    (e-/ADU), ``read_noise`` (ADU): the sensor model; ``exempt``: samples never rejected and never counted as outliers
    for their neighbours (stack.py: the saturated ones). NaN samples are not on the stack. Works on ``pred``'s device,
    the interpolations in batches of ``batch`` samples; ``diagnostics``: keep the residuals and thresholds in
    noise sigmas (``FrameRejection.z``, ``.limit``)."""
    p = pred.params
    dev = pred.pred.device
    val, Xo, Yo = val.to(dev).float(), Xo.to(dev).float(), Yo.to(dev).float()
    chmap = chmap.to(dev).long()
    H, W = val.shape
    C, Ho, Wo = pred.pred.shape
    if not torch.is_tensor(grad):
        grad = torch.full_like(val, float(grad))
    grad = grad.to(dev).float()
    exempt = torch.zeros_like(val, dtype=torch.bool) if exempt is None else exempt.to(dev).bool()

    jx, iy = torch.floor(Xo + 0.5).long(), torch.floor(Yo + 0.5).long()
    fin = torch.isfinite(val) & (jx >= 0) & (jx < Wo) & (iy >= 0) & (iy < Ho)
    cnt = torch.zeros(H, W, dtype=torch.uint8, device=dev)
    cnt[fin] = pred.count.view(-1)[chmap[fin] * (Ho * Wo) + iy[fin] * Wo + jx[fin]]
    del jx, iy
    on_stack = fin & (cnt > 0)
    testable = on_stack & (cnt >= p.min_samples) & ~exempt
    untested = on_stack & (cnt < p.min_samples) & ~exempt
    del fin, cnt

    # the blot (B0) and its derivative (D): the sample's colour interpolated at its position and at its 4 neighbours'
    sx = (_steps(Xo, 1), _steps(Yo, 1))         # one sensor pixel along x, in output pixels (dx, dy)
    sy = (_steps(Xo, 0), _steps(Yo, 0))         # one sensor pixel along y
    B0 = torch.zeros(H * W, device=dev)
    D = torch.zeros(H * W, device=dev)
    flat = [a.reshape(-1) for a in (Xo, Yo, sx[0], sx[1], sy[0], sy[1])]
    for c in range(C):
        idx = torch.nonzero((testable & (chmap == c)).view(-1)).view(-1)
        for b0 in range(0, idx.numel(), batch):
            b = idx[b0:b0 + batch]
            x, y, ax, ay, bx, by = (a[b] for a in flat)
            px = torch.cat([x, x + ax, x - ax, x + bx, x - bx])
            py = torch.cat([y, y + ay, y - ay, y + by, y - by])
            v = _interp(pred.pred[c], px, py, p.interp).view(5, -1)
            B0[b] = v[0]
            D[b] = (v[1:] - v[0]).abs().amax(0)
    del sx, sy, flat
    B0, D = B0.view(H, W), D.view(H, W)

    r = torch.where(testable, val - B0, torch.zeros_like(val))
    t1 = r.abs()
    ta = ((B0 * t + grad).abs() / gain + read_noise ** 2).sqrt() / t        # the sensor noise, normalised units
    fail1 = testable & (t1 > p.scale[0] * D + p.snr[0] * ta)
    fail2 = testable & (t1 > p.scale[1] * D + p.snr[1] * ta)
    rej = fail2 & _dilate(fail1, 1)                     # failing test 2, and test 1 failed by itself or a neighbour
    if p.grow > 1:                                      # driz_cr_grow: a k x k box placed as scipy's 'same' places it
        k = int(p.grow)
        rej = _dilate(rej, k // 2, (k - 1) // 2) & testable
    high, low = rej & (r > 0), rej & (r < 0)
    z = r / ta
    zt = z[testable]
    zt = zt[::max(1, zt.numel() // 1_000_000)]
    spread = float(1.4826 * (zt - zt.median()).abs().median()) if zt.numel() else float("nan")
    counts = {"on_stack": int(on_stack.sum()), "tested": int(testable.sum()), "untested": int(untested.sum()),
              "exempt": int((on_stack & exempt).sum()), "high": int(high.sum()), "low": int(low.sum()),
              "z_spread": round(spread, 4)}
    if not diagnostics:
        return FrameRejection(high, low, testable, untested, counts)
    limit = torch.where(testable, (p.scale[0] * D + p.snr[0] * ta) / ta, torch.zeros_like(ta))
    return FrameRejection(high, low, testable, untested, counts, z, limit)


# ------------------------------------------------------------------------------------------------ the stack's tally
class RejectionTally:
    """The rejected samples of all frames counted on the output grid (each at the output pixel nearest to where it
    lands, the grid reduced by a whole factor ``f`` so that its longer side is at most REJECTION_SIDE), low, high and
    untested apart, and the totals: the rejection rate in % of the samples landing on the stack. The file it saves has
    the keys of stack.py's RejectionMap (low, high, factor, grid), plus ``untested``."""

    def __init__(self, Ho: int, Wo: int, device="cpu", side: int = REJECTION_SIDE):
        self.f = max(1, -(-max(Ho, Wo) // side))
        self.Ho, self.Wo = Ho, Wo
        self.Hr, self.Wr = -(-Ho // self.f), -(-Wo // self.f)
        self.maps = {k: torch.zeros(self.Hr * self.Wr, dtype=torch.int64, device=device)
                     for k in ("low", "high", "untested")}
        self.totals = {"on_stack": 0, "tested": 0, "untested": 0, "exempt": 0, "high": 0, "low": 0, "frames": 0}
        self.z_spread: list[float] = []          # per frame: the noise model's check (about 1)

    def add(self, Xo: torch.Tensor, Yo: torch.Tensor, r: FrameRejection) -> tuple[int, int, int]:
        """Count one frame; returns its (low, high, untested) counts."""
        for k, mask in (("low", r.low), ("high", r.high), ("untested", r.untested)):
            acc = self.maps[k]
            mask = mask.to(acc.device)
            jx = torch.floor(Xo.to(acc.device)[mask] + 0.5).long()
            iy = torch.floor(Yo.to(acc.device)[mask] + 0.5).long()
            ok = (jx >= 0) & (jx < self.Wo) & (iy >= 0) & (iy < self.Ho)
            acc += torch.bincount((iy[ok] // self.f) * self.Wr + jx[ok] // self.f, minlength=acc.numel())
        for k in ("on_stack", "tested", "untested", "exempt", "high", "low"):
            self.totals[k] += r.counts[k]
        self.totals["frames"] += 1
        self.z_spread.append(r.counts["z_spread"])
        return r.counts["low"], r.counts["high"], r.counts["untested"]

    def summary(self) -> dict:
        """The rates in % of the samples on the stack: rejected (low + high), low, high, untested; Siril's mark."""
        n = max(self.totals["on_stack"], 1)
        tt = self.totals
        return {"rejected_pct": round(100 * (tt["low"] + tt["high"]) / n, 4),
                "low_pct": round(100 * tt["low"] / n, 4), "high_pct": round(100 * tt["high"] / n, 4),
                "untested_pct": round(100 * tt["untested"] / n, 4), "exempt_pct": round(100 * tt["exempt"] / n, 4),
                "samples_on_stack": tt["on_stack"], "frames": tt["frames"], "reference_pct": [0.1, 0.5],
                "z_spread_median": round(float(np.nanmedian(self.z_spread)), 4) if self.z_spread else None}

    def arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for k, a in self.maps.items():
            v = a.view(self.Hr, self.Wr).cpu().numpy()
            out[k] = v.astype(np.uint16 if v.max(initial=0) < 65536 else np.uint32)
        return out

    def save(self, path) -> None:
        np.savez_compressed(path, **self.arrays(), factor=self.f, grid=np.array([self.Wo, self.Ho]))
