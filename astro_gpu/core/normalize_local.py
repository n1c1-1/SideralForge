"""Local normalisation per colour, for the stack (stacking v2: docs/spec/STACKING.md, point 5).

Every frame, colour by colour, is mapped onto a REFERENCE::

    frame_c(x) = scale_c * reference_c(P(x)) + Q_c(P(x))   ->   normalised_c(x) = (frame_c(x) - Q_c(P(x))) / scale_c

P is the frame's registration polynomial (reg.npz ``coef``: frame sensor -> reference sensor), Q_c a smooth additive
background of degree 2 and scale_c a multiplicative photometric scale, both per colour (c = 0 red, 1 green, 2 blue).
Pure functions on what stack.py has (the calibrated CFA frames, the registration, a weight per frame, the saturation
level); torch, on the device given: the GPU in stack.py, the CPU in the tests.

How stack.py calls it (for the integration agent)
-------------------------------------------------
Behind the stacking-v2 switch only (a parameter ``normalisation``: "local" (default) | "global"); switch off = today's
``frame_norm`` and ``transp[i]``, untouched.

1. Once, after the frame selection and BEFORE the frame weights (core/frame_weights.py needs the per-colour scale and
   noise): one pass over the used frames (the best ones are read twice more for the reference)::

       from ..core import normalize_local as nl

       def load(i):                    # the calibrated CFA frame (H, W), fixed pattern removed as frame_model does
           fr = torch.from_numpy(np.asarray(mm[i], np.float32)).to(dev)
           return fr if fpn is None else torch.where(fr > VALID_MAX, fr, fr - fpn)

       res = nl.normalize(load, coef, weight, ch_of_k, H, W, fwhm=float(np.percentile(fwhm[used], 90)),
                          frames=used, device=dev, sat_level=satval, mode=p.normalisation,
                          ref_frame=ref, progress=lambda f, msg: ctx.progress(0.1 * f, msg))

   - ``weight`` (n,) only picks the frames of the reference: the N_BEST = 20 highest, as WBPP. The weight that
     select_frames gives today ((transparency / noise)^2 x sharpness, 0 = left out) is the right one.
   - ``fwhm``: sensor pixels, the larger stars of the night; the aperture radius is APERTURE_FWHM = 3 FWHM.
   - ``satval``: the per-pixel saturation level stack.py computes (or a float, or None).
   - ``ref``: reg.npz's reference frame; only the "global" mode uses it (Siril's sequence reference).
   - Result, NaN rows for the frames not measured: ``res.scale`` (n, 3) the factor each colour is divided by,
     ``res.Q`` (n, 3, 6) the background in the frame's ADU, ``res.photometric`` (n, 3) the star flux ratio per colour
     (= ``res.scale`` in local mode), ``res.noise`` (n, 3) the pixel noise of the raw frame, ``res.sky`` (n, 3),
     ``res.ok`` (n,): False when a frame has fewer than MIN_STARS stars in every colour (leave it out, reason
     "no photometry", or fall back to today's ``frame_norm`` for it).

2. Frame weights: ``weigh_frames(transparency=res.scale, noise=res.noise, ...)`` (both (n, 3)).

3. In ``frame_model(i)``, in place of ``frame_norm`` and of the division by ``transp[i]``::

       X, Y = poly_eval(coef[i], xx, yy)                              # as today (reference sensor coordinates)
       grad = nl.background_at(res.Q[i], X, Y, chmap, W, H)          # (H, W), the frame's ADU (Q's terms = quad_terms)
       t = nl.scale_at(res.scale[i], chmap)                          # (H, W), the scale of each sample's colour
       val = (val - grad) / t
       bkg_level = res.sky[i]

   ``judge``/``reject`` take ``t`` where they took ``float(transp[i])``: ``raw = (pred * t + grad)`` and ``/ t ** 2``
   are elementwise, so a tensor works unchanged.

4. diagnostics.json: per frame ``res.frames[i].as_dict()`` (``scale``, ``offset``, ``gradient`` as today, plus
   ``photometric``, ``noise``, ``stars``, ``spread_sigma``); at the top ``"normalisation": res.as_dict()`` (the frames
   of the reference, its stars, the diagnostic).

5. The automatic suggestion: ``Auto("local", res.diagnostic["reason"], res.diagnostic["params"])`` (local by default,
   as WBPP; the reason gives the measured spread, e.g. "backgrounds differ by up to 4.2 σ from edge to edge between
   frames; blue against green varies by 18 %"). French of REASON_TEMPLATES -> client/locale/fr/engine.json.

The methods (docs/pilot/STATE_OF_THE_ART_STACKING.md, sections 3 and 9.4; sources [..] listed there)
--------------------------------------------------------------------------------------------------
* **Local normalisation**, the WBPP default since 2.4 [P15][P16]: PixInsight's LocalNormalization models, per frame,
  the background and the scale against a reference [P4][P11]. Our background model is APP's LNC of degree 2 [A2]: a
  polynomial of degree 2 per colour, fitted (least squares, 3-sigma clipping) on 32 x 32 block medians of
  frame - scale x reference, the reference sampled where the frame's samples land (no resampling of the frame).
* **The photometric scale per colour, from the stars**: PixInsight's LocalNormalization takes its scale from the
  stars' flux ("PSF flux", the default scale evaluation [P4][P11]); Siril normalises each channel apart [S1][S6]. One
  scale for the three colours (today) leaves colour shifts between frames: the sky absorbs blue more than red,
  and more at low altitude (Bouguer's law, extinction k_B > k_G > k_R). Here: aperture photometry on the frame's own
  samples of the colour (no interpolation), the sky from the median of an annulus (Stetson 1987, PASP 99, 191,
  DAOPHOT), an aperture of radius 3 FWHM: a total-flux aperture (the growth curve, Stetson 1990, PASP 102, 932),
  leaving out under 1 % of a Moffat star of beta >= 3 (computed), so that the ratio does not follow the seeing
  (a fixed small aperture did, on Iris: register.py); stars detected on the reference with an aperture S/N of at
  least 40 (PixInsight's minimum detection SNR for LocalNormalization [P5]), none touching a value above 0.85 of
  saturation (LocalNormalization's high clipping level [P11], stack.py's NEAR_SATURATION); the scale = the median
  of the frame / reference flux ratios, with at least 10 stars (as the registration's transparency).
* **The reference = an integration of the best frames**, as WBPP (the 20 best by weight [P15]), not one frame: the
  best frames are put on the reference frame's grid (bilinear, by their inverse registration), then (1) their median
  after an additive offset per colour, on which stars are found, then (2) each of them normalised locally onto that
  median, combined by a MAD-clipped mean (3 sigma, Siril's "MAD clipping" [S2]; the median under 6 frames). The
  reference lives on the super-pixel grid of the registration's reference frame: pixel (j, i) = sensor (2i + 0.5,
  2j + 0.5), one image per colour.
* **Option "global" (like Siril)**: additive with scaling [S1][S2], per colour, against the reference frame (Siril's
  sequence reference image), location and scale by Siril's IKSS estimator as its code runs today (IKSSlite in
  src/algos/statistics_float.c: one clipping at median +- 6 MAD, location = median of the rest, scale = 0.991 x the
  square root of their biweight midvariance) and Siril's coefficients (src/stacking/normalization.c:
  normalised = (x - location_i) x scale_ref / scale_i + location_ref). The local measures are still made, for the
  per-colour transparency (weights) and the diagnostic.
* **Diagnostic** (the reason of the automatic suggestion, spec 9.6): per frame and colour, how far its background
  model departs from the night's median one, from edge to edge (the constant part removed: any normalisation
  removes it), in units of the frame's pixel noise; and how much the blue / green and red / green transparencies vary.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

#: values above this in the calibrated cache are saturated (astro_gpu.stages.calibrate.VALID_MAX, repeated here so
#: that a core module does not import a stage)
VALID_MAX = 60000.0
#: a value above this fraction of the saturation level is near saturation: no photometry on it (LocalNormalization's
#: high clipping level [P11]; stack.py's NEAR_SATURATION)
NEAR_SATURATION = 0.85
#: frames integrated into the reference (WBPP: the 20 best [P15])
N_BEST = 20
#: least aperture S/N, on the reference, of a star used for the photometric scale (LocalNormalization [P5])
MIN_SNR = 40.0
#: fewest stars for a scale (the registration's transparency asks the same)
MIN_STARS = 10
#: most stars measured (the brightest)
MAX_STARS = 3000
#: aperture radius, in FWHM (nearly all the light whatever the seeing: the growth curve, Stetson 1990)
APERTURE_FWHM = 3.0
#: sky annulus, from the aperture radius + the first to + the second (plane pixels; register.py's bkgann)
ANNULUS = (2.0, 6.0)
#: side of the blocks (plane pixels) whose medians the background is fitted on (stack.py's)
BLOCK = 32
#: clipping of the background fit and of the reference's combination, in sigmas
CLIP = 3.0
#: Siril's IKSSlite: one clipping at median +- this many MAD; at most this many samples per colour
IKSS_CLIP = 6.0
IKSS_MAX = 2_000_000
MODES = ("local", "global")
COLOURS = ("red", "green", "blue")
#: the reasons given with the automatic suggestion (French: client/locale/fr/engine.json)
REASON_TEMPLATES = {
    "spread": "backgrounds differ by up to {spread:.1f} σ from edge to edge between frames ({colour}); "
              "blue against green varies by {colour_pct:.0f} %",
    "flat": "backgrounds stay within {spread:.1f} σ from edge to edge between frames; blue against green varies by "
            "{colour_pct:.0f} %",
    "few": "{n} frame(s) measured: nothing to compare",
}

KS = ((0, 0), (0, 1), (1, 0), (1, 1))          # (row, column) of the 2 x 2 positions k = 2 * row + column


# ------------------------------------------------------------------------------------------------ geometry
def _poly(c, u, v):
    """The cubic of register.poly_terms with coefficients c (10,), on normalised coordinates u, v."""
    return (c[0] + c[1] * u + c[2] * v + c[3] * u * u + c[4] * u * v + c[5] * v * v + c[6] * u ** 3
            + c[7] * u * u * v + c[8] * u * v * v + c[9] * v ** 3)


def poly_apply(coef, x: torch.Tensor, y: torch.Tensor, W: int, H: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sensor positions (x, y) of a frame -> the reference's, by its registration polynomial ``coef`` (2, 10)
    (register.apply_poly, in torch)."""
    c = np.asarray(coef, np.float64)
    u, v = x / (W / 2) - 1.0, y / (H / 2) - 1.0
    return _poly([float(a) for a in c[0]], u, v), _poly([float(a) for a in c[1]], u, v)


def _poly_terms_np(x, y, W, H):
    u, v = x / (W / 2) - 1.0, y / (H / 2) - 1.0
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3], -1)


def poly_inverse(coef, X: torch.Tensor, Y: torch.Tensor, W: int, H: int, iters: int = 3, chunk: int = 1 << 21):
    """Reference positions (X, Y) -> the frame's sensor positions: an inverse cubic fitted on a grid, then Newton
    steps on the polynomial itself (float64; 1e-6 px after three steps), by chunks of ``chunk`` positions."""
    c = np.asarray(coef, np.float64)
    gx, gy = np.meshgrid(np.linspace(-0.25 * W, 1.25 * W, 25), np.linspace(-0.25 * H, 1.25 * H, 25))
    gx, gy = gx.ravel(), gy.ravel()
    T = _poly_terms_np(gx, gy, W, H)
    A = _poly_terms_np(T @ c[0], T @ c[1], W, H)
    inv = np.stack([np.linalg.lstsq(A, gx, rcond=None)[0], np.linalg.lstsq(A, gy, rcond=None)[0]])
    Xf, Yf = X.reshape(-1), Y.reshape(-1)
    xs, ys = [], []
    for s in range(0, Xf.numel(), chunk):
        x, y = _newton(c, inv, Xf[s:s + chunk].double(), Yf[s:s + chunk].double(), W, H, iters)
        xs.append(x)
        ys.append(y)
    if not xs:
        return X.clone(), Y.clone()
    return torch.cat(xs).reshape(X.shape).to(X.dtype), torch.cat(ys).reshape(Y.shape).to(Y.dtype)


def _newton(c, inv, Xd, Yd, W, H, iters):
    x, y = poly_apply(inv, Xd, Yd, W, H)
    cx, cy = [float(a) for a in c[0]], [float(a) for a in c[1]]
    for _ in range(iters):
        u, v = x / (W / 2) - 1.0, y / (H / 2) - 1.0
        rx, ry = Xd - _poly(cx, u, v), Yd - _poly(cy, u, v)
        # the Jacobian of the polynomial, per sensor pixel
        a = (cx[1] + 2 * cx[3] * u + cx[4] * v + 3 * cx[6] * u * u + 2 * cx[7] * u * v + cx[8] * v * v) * (2 / W)
        b = (cx[2] + cx[4] * u + 2 * cx[5] * v + cx[7] * u * u + 2 * cx[8] * u * v + 3 * cx[9] * v * v) * (2 / H)
        e = (cy[1] + 2 * cy[3] * u + cy[4] * v + 3 * cy[6] * u * u + 2 * cy[7] * u * v + cy[8] * v * v) * (2 / W)
        d = (cy[2] + cy[4] * u + 2 * cy[5] * v + cy[7] * u * u + 2 * cy[8] * u * v + 3 * cy[9] * v * v) * (2 / H)
        det = a * d - b * e
        x = x + (d * rx - b * ry) / det
        y = y + (a * ry - e * rx) / det
    return x, y


def quad_terms(X: torch.Tensor, Y: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """(6, ...) the terms of the background model at reference positions (X, Y): stack.py's ``quad_terms``."""
    a, b = X / (W / 2) - 1.0, Y / (H / 2) - 1.0
    return torch.stack([torch.ones_like(a), a, b, a * a, a * b, b * b])


def background_at(Q, X: torch.Tensor, Y: torch.Tensor, chmap: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """The additive background (the frame's ADU) of each sample: Q (3, 6) of its colour (``chmap``) at its reference
    position (X, Y)."""
    q = torch.as_tensor(np.asarray(Q, np.float64), dtype=X.dtype, device=X.device)[chmap]
    a, b = X / (W / 2) - 1.0, Y / (H / 2) - 1.0
    return q[..., 0] + q[..., 1] * a + q[..., 2] * b + q[..., 3] * a * a + q[..., 4] * a * b + q[..., 5] * b * b


def scale_at(scale, chmap: torch.Tensor) -> torch.Tensor:
    """The scale (3,) of each sample's colour (``chmap``): what its value minus its background is divided by."""
    return torch.as_tensor(np.asarray(scale, np.float64), dtype=torch.float32, device=chmap.device)[chmap]


# ------------------------------------------------------------------------------------------------ pixels
def _as_tensor(fr, dev) -> torch.Tensor:
    if isinstance(fr, torch.Tensor):
        return fr.to(dev, torch.float32)
    return torch.from_numpy(np.asarray(fr, np.float32)).to(dev)


def _planes(fr: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(4, H/2, W/2) the four Bayer planes, k = 2 * row + column; plane pixel (j, i) = sensor (2i + kx, 2j + ky)."""
    h2, w2 = H // 2, W // 2
    return torch.stack([fr[ky::2, kx::2][:h2, :w2] for ky, kx in KS])


def _sat_planes(sat_level, H: int, W: int, dev):
    if sat_level is None:
        return None
    if np.ndim(sat_level) == 0:
        return float(sat_level)
    return _planes(_as_tensor(sat_level, dev), H, W)


def _near(pl: torch.Tensor, satp, valid_max: float) -> torch.Tensor:
    """Saturated (above ``valid_max``: the cache's mark) or above NEAR_SATURATION of the saturation level."""
    m = pl > valid_max
    if satp is not None:
        m |= pl > NEAR_SATURATION * satp
    return m


def _bilinear(st: torch.Tensor, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """(C, *px.shape): the C images ``st`` (C, h, w) sampled bilinearly at pixel coordinates (px, py), 0 outside."""
    C, h, w = st.shape
    gx = px.float() / max(w - 1, 1) * 2 - 1
    gy = py.float() / max(h - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1).reshape(1, 1, -1, 2)
    out = F.grid_sample(st[None].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[0, :, 0].reshape(C, *px.shape)


def _colour_on(pl, valid, near, ch_of_k, c: int, x: torch.Tensor, y: torch.Tensor):
    """The frame's colour c at its sensor positions (x, y): each of its planes of that colour sampled bilinearly from
    its valid samples, averaged (value, NaN where too few); and whether a near-saturated sample is involved."""
    ks = [k for k in range(4) if ch_of_k[k] == c]
    num = den = sat = 0
    for k in ks:
        ky, kx = KS[k]
        st = torch.stack([torch.where(valid[k], pl[k], torch.zeros((), device=pl.device)), valid[k].float(),
                          near[k].float()])
        s = _bilinear(st, (x - kx) / 2, (y - ky) / 2)
        num, den, sat = num + s[0], den + s[1], sat + s[2]
    val = torch.where(den > 0.5 * len(ks), num / den.clamp(min=1e-6), torch.full_like(num, float("nan")))
    return val, sat > 0


def _fill_bad(pl: torch.Tensor, bad: torch.Tensor) -> torch.Tensor:
    """Bad samples (NaN: hot pixels, defects) replaced by the mean of their valid neighbours in the same plane, for
    the photometry only (a cosmetic correction)."""
    v = torch.where(bad, torch.zeros_like(pl), pl)
    m = (~bad).float()
    num = F.avg_pool2d(v[:, None], 3, 1, 1)[:, 0]
    den = F.avg_pool2d(m[:, None], 3, 1, 1)[:, 0]
    fill = torch.where(den > 0, num / den.clamp(min=1e-6), torch.full_like(pl, float("nan")))
    return torch.where(bad, fill, pl)


def _smooth(img: torch.Tensor, valid: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """The smooth background of an image: medians of its valid pixels in blocks, bilinearly interpolated
    (stack.py's ``smooth_background``)."""
    h, w = img.shape
    b = min(block, max(4, min(h, w) // 4))
    hb, wb = -(-h // b), -(-w // b)
    x = torch.where(valid, img, torch.full_like(img, float("nan")))
    x = F.pad(x, (0, wb * b - w, 0, hb * b - h), value=float("nan"))
    med = torch.nanmedian(x.reshape(hb, b, wb, b).permute(0, 2, 1, 3).reshape(hb, wb, b * b), dim=-1).values
    fin = torch.isfinite(med)
    if not fin.any():
        return torch.zeros_like(img)
    med = torch.where(fin, med, torch.nanmedian(med[fin]))
    up = F.interpolate(med[None, None], size=(hb * b, wb * b), mode="bilinear", align_corners=False)
    return up[0, 0, :h, :w]


def _mad_sigma(x: torch.Tensor) -> float:
    x = x[torch.isfinite(x)]
    if x.numel() < 10:
        return float("nan")
    return float(1.4826 * (x - x.median()).abs().median())


def _plane_noise(p: torch.Tensor, ok: torch.Tensor) -> float:
    """Pixel noise of a plane: adjacent samples differ by their noise only (register.py's measure)."""
    d = (p[::2, 1:] - p[::2, :-1])[ok[::2, 1:] & ok[::2, :-1]]
    return _mad_sigma(d) / math.sqrt(2) if d.numel() >= 100 else float("nan")


def aperture(img: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, r: float, sat: torch.Tensor | None = None):
    """Aperture photometry of stars at pixel coordinates (cx, cy) of ``img`` (h, w): the sum, in a circle of radius
    ``r`` (soft edge: the pixel's share inside), of the values minus the sky, the median of the annulus ANNULUS around
    (Stetson 1987). Returns (flux, its error from the annulus' noise, ok): not ok when the window leaves the image,
    a value in the aperture is NaN or ``sat``, or less than half the annulus is finite."""
    h, w = img.shape
    m = cx.numel()
    if m == 0:
        z = torch.zeros(0, device=img.device)
        return z, z, z.bool()
    R = int(math.ceil(r + ANNULUS[1]))
    off = torch.arange(-R, R + 1, device=img.device)
    oy, ox = torch.meshgrid(off, off, indexing="ij")
    ix, iy = torch.round(cx).long(), torch.round(cy).long()
    xx, yy = ix[:, None] + ox.reshape(1, -1), iy[:, None] + oy.reshape(1, -1)
    inb = (ix - R >= 0) & (ix + R < w) & (iy - R >= 0) & (iy + R < h)
    xx, yy = xx.clamp(0, w - 1), yy.clamp(0, h - 1)
    v = img[yy, xx]
    d = torch.hypot(xx.float() - cx.float()[:, None], yy.float() - cy.float()[:, None])
    wap = (r + 0.5 - d).clamp(0, 1)
    ann = (d >= r + ANNULUS[0]) & (d <= r + ANNULUS[1])
    fin = torch.isfinite(v)
    va = torch.where(ann & fin, v, torch.full_like(v, float("nan")))
    sky = torch.nanmedian(va, dim=1).values
    sky_sig = 1.4826 * torch.nanmedian((va - sky[:, None]).abs(), dim=1).values
    nann = (ann & fin).sum(1)
    inap = wap > 0
    flux = torch.where(inap & fin, wap * (v - sky[:, None]), torch.zeros_like(v)).sum(1)
    npix = wap.sum(1)
    err = sky_sig * torch.sqrt(npix + npix ** 2 * (math.pi / 2) / nann.clamp(min=1))
    ok = inb & (nann >= 0.5 * ann.sum(1)) & ~(inap & ~fin).any(1) & torch.isfinite(sky)
    if sat is not None:
        ok &= ~(inap & sat[yy, xx]).any(1)
    return flux, err, ok


# ------------------------------------------------------------------------------------------------ Siril's IKSS
def _median(x: torch.Tensor) -> float:
    """The median, the mean of the two middle values for an even count (gsl's, as Siril)."""
    n = x.numel()
    lo = x.kthvalue((n - 1) // 2 + 1).values
    hi = lo if n % 2 else x.kthvalue(n // 2 + 1).values
    return float(0.5 * (lo + hi))


def _maxpool(x: torch.Tensor, r: int) -> torch.Tensor:
    """The maximum over a (2r + 1)-pixel square around each pixel of an image (h, w), separable."""
    y = F.max_pool2d(x[None, None], (1, 2 * r + 1), 1, (0, r))
    return F.max_pool2d(y, (2 * r + 1, 1), 1, (r, 0))[0, 0]


def _bwmv(x: torch.Tensor, m: float, mad: float) -> float:
    """Biweight midvariance (Siril's siril_stats_float_bwmv: 9 MAD, n over all the values)."""
    if mad <= 0:
        return 0.0
    y = (x - m) / (9 * mad)
    a = y.abs() < 1
    y2 = y * y
    up = float(torch.where(a, (x - m) ** 2 * (1 - y2) ** 4, torch.zeros_like(x)).sum())
    down = float(torch.where(a, (1 - y2) * (1 - 5 * y2), torch.zeros_like(x)).sum())
    return x.numel() * up / down ** 2 if down else 0.0


def ikss(x) -> tuple[float, float]:
    """Location and scale of the values ``x`` by Siril's IKSS estimator as it runs today (IKSSlite): the median and
    raw MAD of the good values (finite, non-zero), one clipping at median +- IKSS_CLIP MAD, then the median of the
    rest (location) and 0.991 x the square root of their biweight midvariance (scale; 0.991 makes it the sigma of a
    normal distribution). At most IKSS_MAX values (a regular subsample). (0, 0) without values."""
    x = torch.as_tensor(x).reshape(-1).double()
    x = x[torch.isfinite(x) & (x != 0)]
    if x.numel() > IKSS_MAX:
        x = x[:: int(math.ceil(x.numel() / IKSS_MAX))]
    if x.numel() == 0:
        return 0.0, 0.0
    m = _median(x)
    mad = _median((x - m).abs())
    k = x[(x >= m - IKSS_CLIP * mad) & (x <= m + IKSS_CLIP * mad)]
    if k.numel() == 0:
        return m, 0.0
    loc = _median(k)
    mad = _median((k - loc).abs())
    if mad == 0:
        return loc, 0.0
    return loc, math.sqrt(_bwmv(k, loc, mad)) * 0.991


# ------------------------------------------------------------------------------------------------ results
@dataclass
class Reference:
    """The reference every frame is normalised onto: an integration of the best frames (``best``), per colour, on the
    super-pixel grid of the registration's reference frame (``image`` (3, H/2, W/2): pixel (j, i) = sensor
    (2i + 0.5, 2j + 0.5); NaN where no frame covers), ``sat`` where a frame was near saturation, the stars measured
    on it (``stars`` (m, 2) reference sensor coordinates, ``flux`` (m, 3), NaN where unusable in that colour)."""
    image: torch.Tensor
    sat: torch.Tensor
    H: int
    W: int
    ch_of_k: list
    radius: float                        # aperture radius, plane pixels
    best: np.ndarray
    mode: str = "local"
    stars: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    flux: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    noise: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    sat_planes: object = None
    valid_max: float = VALID_MAX
    ref_frame: int | None = None
    gstats: np.ndarray | None = None     # (3, 2) IKSS location and scale of the reference frame ("global")
    _st: dict = field(default_factory=dict, repr=False)

    @property
    def device(self):
        return self.image.device

    def sample(self, c: int, X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Colour c of the reference at reference sensor positions (X, Y): (value, NaN where not covered; whether a
        near-saturated pixel is involved)."""
        if c not in self._st:
            img = self.image[c]
            fin = torch.isfinite(img)
            self._st[c] = torch.stack([torch.where(fin, img, torch.zeros_like(img)), fin.float(), self.sat.float()])
        s = _bilinear(self._st[c], (X - 0.5) / 2, (Y - 0.5) / 2)
        val = torch.where(s[1] > 0.5, s[0] / s[1].clamp(min=1e-6), torch.full_like(s[0], float("nan")))
        return val, s[2] > 0

    def as_dict(self) -> dict:
        return {"mode": self.mode, "frames": [int(i) for i in self.best], "reference_frame": self.ref_frame,
                "stars": [int(np.isfinite(self.flux[:, c]).sum()) for c in range(3)],
                "aperture_px": round(2 * self.radius, 2), "noise": [round(float(v), 4) for v in self.noise]}


@dataclass
class FrameNorm:
    """The normalisation of one frame: normalised = (raw - Q[c] . quad_terms(X, Y)) / scale[c] (``Q``, ``scale``: the
    mode's); the local measures whatever the mode (``photometric``: the star flux ratio per colour, the frame's
    transparency in that colour; ``local_Q``), the pixel noise and sky level of the raw frame, the stars used."""
    index: int
    mode: str
    scale: np.ndarray
    Q: np.ndarray
    photometric: np.ndarray
    scale_err: np.ndarray
    local_Q: np.ndarray
    noise: np.ndarray
    sky: np.ndarray
    stars: np.ndarray
    ok: bool
    note: str = ""
    spread_sigma: np.ndarray | None = None     # set by diagnose

    def as_dict(self) -> dict:
        def r(a, n=5):
            return [None if not np.isfinite(v) else round(float(v), n) for v in np.ravel(a)]
        d = {"scale": r(self.scale, 6), "offset": r(self.Q[:, 0], 3), "gradient": np.asarray(self.Q).tolist(),
             "photometric": r(self.photometric, 6), "photometric_err": r(self.scale_err, 6), "noise": r(self.noise, 3),
             "sky": r(self.sky, 3), "stars": [int(v) for v in self.stars], "normalisation_ok": bool(self.ok)}
        if self.spread_sigma is not None:
            d["spread_sigma"] = r(self.spread_sigma, 3)
        if self.note:
            d["normalisation_note"] = self.note
        return d


@dataclass
class Normalisation:
    """Every frame's normalisation (``frames``: index -> FrameNorm), the reference, the diagnostic; arrays over all
    the n frames, NaN rows for those not measured."""
    reference: Reference
    frames: dict
    n: int
    diagnostic: dict

    def _arr(self, name: str, shape: tuple) -> np.ndarray:
        out = np.full((self.n, *shape), np.nan)
        for i, fn in self.frames.items():
            out[i] = getattr(fn, name)
        return out

    @property
    def scale(self) -> np.ndarray:
        return self._arr("scale", (3,))

    @property
    def Q(self) -> np.ndarray:
        return self._arr("Q", (3, 6))

    @property
    def photometric(self) -> np.ndarray:
        return self._arr("photometric", (3,))

    @property
    def noise(self) -> np.ndarray:
        return self._arr("noise", (3,))

    @property
    def sky(self) -> np.ndarray:
        return self._arr("sky", (3,))

    @property
    def ok(self) -> np.ndarray:
        out = np.zeros(self.n, bool)
        for i, fn in self.frames.items():
            out[i] = fn.ok
        return out

    def as_dict(self) -> dict:
        return {**self.reference.as_dict(), "diagnostic": {k: v for k, v in self.diagnostic.items()
                                                           if k not in ("spread_sigma",)}}


# ------------------------------------------------------------------------------------------------ the reference
def _detect(ref: Reference) -> None:
    """The stars of the reference: local maxima of its luminance, the brightest in their annulus, 5 sigma above the
    smooth background, nowhere near saturation; their flux per colour, kept where the S/N reaches MIN_SNR."""
    R = ref.image
    fin = torch.isfinite(R).all(0)
    for c in range(3):
        ok = torch.isfinite(R[c])
        ref.noise[c] = _mad_sigma((R[c] - _smooth(R[c], ok))[::2, ::2][ok[::2, ::2]])
    L = torch.where(fin, R.sum(0), torch.zeros_like(R[0]))
    hp = torch.where(fin, L - _smooth(L, fin), torch.full_like(L, float("nan")))
    sig = _mad_sigma(hp[::2, ::2])
    rw = int(math.ceil(ref.radius + ANNULUS[1]))
    z = torch.where(fin, hp, torch.full_like(hp, -float("inf")))
    mx = _maxpool(z, rw)
    peak = (z == mx) & fin & (hp > 5 * sig)
    peak[: rw + 1] = False
    peak[-rw - 1:] = False
    peak[:, : rw + 1] = False
    peak[:, -rw - 1:] = False
    if ref.sat.any():
        ra = int(math.ceil(ref.radius)) + 1
        peak &= ~(_maxpool(ref.sat.float(), ra) > 0)
    ys, xs = torch.nonzero(peak, as_tuple=True)
    if len(ys) > MAX_STARS:
        keep = torch.argsort(hp[ys, xs], descending=True)[:MAX_STARS]
        ys, xs = ys[keep], xs[keep]
    # the centre: the centroid of the 3 x 3 pixels around the peak
    o = torch.arange(-1, 2, device=R.device)
    oy, ox = torch.meshgrid(o, o, indexing="ij")
    wv = hp[(ys[:, None] + oy.reshape(1, -1)).clamp(0, hp.shape[0] - 1),
            (xs[:, None] + ox.reshape(1, -1)).clamp(0, hp.shape[1] - 1)].clamp(min=0).nan_to_num(0)
    s = wv.sum(1).clamp(min=1e-12)
    cx = xs.float() + (wv * ox.reshape(1, -1)).sum(1) / s
    cy = ys.float() + (wv * oy.reshape(1, -1)).sum(1) / s
    flux = torch.full((len(ys), 3), float("nan"), device=R.device)
    for c in range(3):
        f, e, ok = aperture(R[c], cx, cy, ref.radius, ref.sat)
        good = ok & (f > 0) & (f >= MIN_SNR * e)
        flux[:, c] = torch.where(good, f, flux[:, c])
    keep = torch.isfinite(flux).any(1)
    ref.stars = torch.stack([2 * cx + 0.5, 2 * cy + 0.5], 1)[keep].double().cpu().numpy()
    ref.flux = flux[keep].double().cpu().numpy()


def _combine(S: np.ndarray, dev, how: str, shift=None, Q=None, scale=None, W=0, H=0) -> torch.Tensor:
    """Pixel-wise combination of the frames S (N, 3, h2, w2) on the reference grid (inf: near saturation, NaN: none),
    by row bands: ``median`` of S - shift (N, 3), or the frames normalised ((S - Q) / scale; Q (N, 3, 6) at the
    grid's reference positions) combined by a MAD-clipped mean (the median under 6 frames)."""
    N, _, h2, w2 = S.shape
    out = torch.empty((3, h2, w2), device=dev)
    band = max(1, int(2 ** 24 // max(1, N * 3 * w2)))
    u = (2 * torch.arange(w2, device=dev, dtype=torch.float32) + 0.5)
    for y0 in range(0, h2, band):
        x = torch.from_numpy(np.asarray(S[:, :, y0:y0 + band], np.float32)).to(dev)
        x = torch.where(torch.isinf(x), torch.full_like(x, float("nan")), x)
        if how == "median":
            x = x - torch.as_tensor(shift, dtype=torch.float32, device=dev)[:, :, None, None]
            out[:, y0:y0 + band] = torch.nanmedian(x, dim=0).values
            continue
        v = (2 * torch.arange(y0, y0 + x.shape[2], device=dev, dtype=torch.float32) + 0.5)
        V, U = torch.meshgrid(v, u, indexing="ij")
        T = quad_terms(U, V, W, H)                                                    # (6, b, w2)
        q = torch.as_tensor(Q, dtype=torch.float32, device=dev)                      # (N, 3, 6)
        s = torch.as_tensor(scale, dtype=torch.float32, device=dev)
        x = (x - torch.einsum("nct,tij->ncij", q, T)) / s[:, :, None, None]
        if N < 6:
            out[:, y0:y0 + band] = torch.nanmedian(x, dim=0).values
            continue
        for _ in range(3):                         # Siril's MAD clipping, 3 sigma: median and MAD of what is left
            med = torch.nanmedian(x, dim=0).values
            mad = 1.4826 * torch.nanmedian((x - med).abs(), dim=0).values
            out_ = (x - med).abs() > CLIP * mad
            if not (out_ & (mad > 0)).any():
                break
            x = torch.where(out_ & (mad > 0), torch.full_like(x, float("nan")), x)
        out[:, y0:y0 + band] = torch.nanmean(x, dim=0)
    return out


def build_reference(load: Callable, coef, weight, ch_of_k, H: int, W: int, fwhm: float, *, device="cpu",
                    sat_level=None, n_best: int = N_BEST, mode: str = "local", ref_frame: int | None = None,
                    valid_max: float = VALID_MAX, progress: Callable | None = None) -> Reference:
    """The reference: an integration of the ``n_best`` frames of highest ``weight`` (WBPP: 20), see the module's
    docstring. ``load(i)``: frame i, CFA (H, W), saturated samples above ``valid_max``, bad ones NaN; ``coef`` (n, 2,
    10) the registration polynomials; ``fwhm``: sensor pixels (the aperture is APERTURE_FWHM of it); ``sat_level``:
    the saturation level, a float or a CFA map (H, W), or None (only ``valid_max``)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: one of {MODES}")
    dev = torch.device(device)
    progress = progress or (lambda f, msg: None)
    weight = np.asarray(weight, float)
    order = np.argsort(-np.nan_to_num(weight, nan=-1.0), kind="stable")
    order = order[np.isfinite(weight[order]) & (weight[order] > 0)]
    if not len(order):
        raise ValueError("no frame with a positive weight for the normalisation reference")
    best = np.sort(order[:n_best])
    h2, w2 = H // 2, W // 2
    radius = max(2.0, APERTURE_FWHM * float(fwhm) / 2)
    satp = _sat_planes(sat_level, H, W, dev)
    jy, jx = torch.meshgrid(torch.arange(h2, device=dev, dtype=torch.float64),
                            torch.arange(w2, device=dev, dtype=torch.float64), indexing="ij")
    U, V = 2 * jx + 0.5, 2 * jy + 0.5
    S = np.empty((len(best), 3, h2, w2), np.float16)
    for n_, j in enumerate(best):
        pl = _planes(_as_tensor(load(int(j)), dev), H, W)
        near = _near(pl, satp, valid_max)
        valid = torch.isfinite(pl) & ~near
        x, y = poly_inverse(coef[j], U, V, W, H)
        x, y = x.float(), y.float()
        for c in range(3):
            val, sat = _colour_on(pl, valid, near, ch_of_k, c, x, y)
            S[n_, c] = torch.where(sat, torch.full_like(val, float("inf")), val).cpu().numpy().astype(np.float16)
        del pl, near, valid, x, y
        progress(0.5 * (n_ + 1) / len(best), f"normalisation reference: {n_ + 1}/{len(best)} frames")
    sat = torch.from_numpy(np.isinf(S).any(axis=(0, 1))).to(dev)
    # (1) the median of the frames, each with an additive offset per colour
    off = np.array([[np.nanmedian(np.where(np.isfinite(S[n_, c]), S[n_, c], np.nan).astype(np.float32))
                     for c in range(3)] for n_ in range(len(best))])
    R0 = _combine(S, dev, "median", shift=off) + torch.as_tensor(np.median(off, 0), dtype=torch.float32,
                                                                 device=dev)[:, None, None]
    ref = Reference(R0, sat, H, W, list(ch_of_k), radius, best, mode, sat_planes=satp, valid_max=valid_max)
    _detect(ref)
    # (2) each frame normalised locally onto that median, combined by a clipped mean
    if len(best) >= 2:
        Qs, ss, keep = [], [], []
        for n_, j in enumerate(best):
            fn = measure_frame(load(int(j)), coef[j], ref, index=int(j))
            if fn.ok:
                Qs.append(fn.Q)
                ss.append(fn.scale)
                keep.append(n_)
            progress(0.5 + 0.5 * (n_ + 1) / len(best), f"normalisation reference: {n_ + 1}/{len(best)} frames")
        if len(keep) >= 2:
            R = _combine(S[keep], dev, "mean", Q=np.array(Qs), scale=np.array(ss), W=W, H=H)
            ref = Reference(R, sat, H, W, list(ch_of_k), radius, best[keep], mode, sat_planes=satp,
                            valid_max=valid_max)
            _detect(ref)
    del S
    if mode == "global":
        rf = int(best[np.argmax(weight[best])]) if ref_frame is None else int(ref_frame)
        pl = _planes(_as_tensor(load(rf), dev), H, W)
        valid = torch.isfinite(pl) & ~_near(pl, satp, valid_max)
        ref.ref_frame = rf
        ref.gstats = np.array([ikss(_colour_values(pl, valid, ch_of_k, c)) for c in range(3)])
    return ref


def _colour_values(pl, valid, ch_of_k, c: int) -> torch.Tensor:
    return torch.cat([pl[k][valid[k]] for k in range(4) if ch_of_k[k] == c])


# ------------------------------------------------------------------------------------------------ one frame
def _robust_fit(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least squares with CLIP-sigma clipping (MAD), three rounds (stack.py's fit of the background)."""
    m = np.ones(len(b), bool)
    q = np.zeros(A.shape[1])
    for _ in range(3):
        q = np.linalg.lstsq(A[m], b[m], rcond=None)[0]
        r = b - A @ q
        s = 1.4826 * np.median(np.abs(r[m]))
        m = np.abs(r) < CLIP * s + 1e-9
        if m.sum() < A.shape[1] * 2:
            break
    return q


def _photometry(pl, bad, near, coef_i, ref: Reference):
    """Per colour: the median frame / reference flux ratio of the stars, its error, the number of stars."""
    H, W = ref.H, ref.W
    scale, err, nst = np.full(3, np.nan), np.full(3, np.nan), np.zeros(3, int)
    if not len(ref.stars):
        return scale, err, nst
    su = torch.as_tensor(ref.stars[:, 0], dtype=torch.float64, device=pl.device)
    sv = torch.as_tensor(ref.stars[:, 1], dtype=torch.float64, device=pl.device)
    x, y = poly_inverse(coef_i, su, sv, W, H)
    filled = _fill_bad(pl, bad)
    for c in range(3):
        fs, ok = [], torch.ones(len(su), dtype=torch.bool, device=pl.device)
        for k in [k for k in range(4) if ref.ch_of_k[k] == c]:
            ky, kx = KS[k]
            f, _, o = aperture(filled[k], (x - kx) / 2, (y - ky) / 2, ref.radius, near[k])
            fs.append(f)
            ok &= o
        f = (sum(fs) / len(fs)).double().cpu().numpy()
        ok = ok.cpu().numpy() & np.isfinite(ref.flux[:, c]) & (f > 0)
        r = f[ok] / ref.flux[ok, c]
        nst[c] = len(r)
        if len(r) >= MIN_STARS:
            scale[c] = float(np.median(r))
            err[c] = float(1.2533 * 1.4826 * np.median(np.abs(r - scale[c])) / math.sqrt(len(r)))
    return scale, err, nst


def _local_background(pl, valid, coef_i, ref: Reference, scale) -> np.ndarray:
    """Q (3, 6): per colour, the degree-2 fit (reference coordinates) of the block medians of frame - scale x
    reference, the reference sampled where each sample of the frame lands."""
    H, W = ref.H, ref.W
    _, h2, w2 = pl.shape
    dev = pl.device
    B = BLOCK if min(h2, w2) >= 4 * BLOCK else max(4, min(h2, w2) // 4)
    nby, nbx = h2 // B, w2 // B
    gy, gx = torch.meshgrid(torch.arange(h2, device=dev, dtype=torch.float32),
                            torch.arange(w2, device=dev, dtype=torch.float32), indexing="ij")
    by, bx = torch.meshgrid((torch.arange(nby, device=dev, dtype=torch.float32) + 0.5) * B - 0.5,
                            (torch.arange(nbx, device=dev, dtype=torch.float32) + 0.5) * B - 0.5, indexing="ij")
    rows: dict = {c: [] for c in range(3)}
    tgt: dict = {c: [] for c in range(3)}
    for k in range(4):
        c = ref.ch_of_k[k]
        if not np.isfinite(scale[c]):
            continue
        ky, kx = KS[k]
        U, V = poly_apply(coef_i, 2 * gx + kx, 2 * gy + ky, W, H)
        rv, rs = ref.sample(c, U, V)
        ok = valid[k] & ~rs & torch.isfinite(rv)
        d = torch.where(ok, pl[k] - float(scale[c]) * rv, torch.full_like(rv, float("nan")))
        blk = d[: nby * B, : nbx * B].reshape(nby, B, nbx, B).permute(0, 2, 1, 3).reshape(nby, nbx, B * B)
        cnt = torch.isfinite(blk).sum(-1)
        med = torch.nanmedian(blk, dim=-1).values
        keep = cnt >= B * B // 2
        bu, bv = poly_apply(coef_i, 2 * bx + kx, 2 * by + ky, W, H)
        rows[c].append(quad_terms(bu[keep], bv[keep], W, H).T.double().cpu().numpy())
        tgt[c].append(med[keep].double().cpu().numpy())
    Q = np.full((3, 6), np.nan)
    for c in range(3):
        if rows[c]:
            A, b = np.concatenate(rows[c]), np.concatenate(tgt[c])
            if len(b) >= 12:
                Q[c] = _robust_fit(A, b)
    return Q


def measure_frame(fr, coef_i, ref: Reference, index: int = -1) -> FrameNorm:
    """The normalisation of one frame (``fr``: CFA (H, W), numpy or torch; ``coef_i`` (2, 10)) onto ``ref``: its
    photometric scale per colour (stars), its local background Q (3, 6), and, in "global" mode, Siril's additive
    with scaling (IKSS) as the applied ``scale`` and ``Q``."""
    H, W = ref.H, ref.W
    pl = _planes(_as_tensor(fr, ref.device), H, W)
    bad = torch.isnan(pl)
    near = _near(pl, ref.sat_planes, ref.valid_max)
    valid = ~bad & ~near
    noise_k = [_plane_noise(pl[k], valid[k]) for k in range(4)]
    noise, sky = np.full(3, np.nan), np.full(3, np.nan)
    for c in range(3):
        ks = [k for k in range(4) if ref.ch_of_k[k] == c]
        noise[c] = math.sqrt(np.mean([noise_k[k] ** 2 for k in ks]))
        v = torch.cat([pl[k][::4, ::4][valid[k][::4, ::4]] for k in ks])
        sky[c] = float(v.median()) if v.numel() else float("nan")
    phot, err, nst = _photometry(pl, bad, near, coef_i, ref)
    note = ""
    ok = bool(np.isfinite(phot).any())
    if ok and not np.isfinite(phot).all():
        miss = [COLOURS[c] for c in range(3) if not np.isfinite(phot[c])]
        note = f"fewer than {MIN_STARS} stars in {', '.join(miss)}: the other colours' scale used"
        phot = np.where(np.isfinite(phot), phot, float(np.exp(np.nanmean(np.log(phot)))))
    elif not ok:
        note = f"fewer than {MIN_STARS} stars measured in every colour: no photometric scale"
    local_Q = _local_background(pl, valid, coef_i, ref, phot) if ok else np.full((3, 6), np.nan)
    if ok and not np.isfinite(local_Q).all():
        ok, note = False, "too little sky compared with the reference for its background"
    scale, Q = phot.copy(), local_Q.copy()
    if ref.mode == "global" and ref.gstats is not None:
        scale, Q = np.full(3, np.nan), np.zeros((3, 6))
        for c in range(3):
            loc, sc = ikss(_colour_values(pl, valid, ref.ch_of_k, c))
            gl, gs = ref.gstats[c]
            scale[c] = sc / gs if gs > 0 and sc > 0 else 1.0          # Siril: a zero scale -> factor 1
            Q[c, 0] = loc - scale[c] * gl
    return FrameNorm(int(index), ref.mode, scale, Q, phot, err, local_Q, noise, sky, nst, ok, note)


# ------------------------------------------------------------------------------------------------ the night
def diagnose(frames: dict, ref: Reference) -> dict:
    """The spread the normalisation corrects, for the automatic suggestion's reason: per frame and colour, the range
    from edge to edge (a 9 x 9 grid over the field) of its background model in reference units (local_Q /
    photometric) minus the night's median one, in units of its pixel noise (noise / photometric) (``spread_sigma``,
    also set on each FrameNorm); the variation of the blue / green and red / green transparencies (%)."""
    ok = {i: fn for i, fn in frames.items() if fn.ok}
    if len(ok) < 2:
        return {"spread_sigma": {}, "max_sigma": None, "reason": REASON_TEMPLATES["few"], "params": {"n": len(ok)}}
    idx = sorted(ok)
    gu, gv = np.meshgrid(np.linspace(0, ref.W - 1, 9), np.linspace(0, ref.H - 1, 9))
    T = quad_terms(torch.from_numpy(gu.ravel()), torch.from_numpy(gv.ravel()), ref.W, ref.H).numpy()    # (6, 81)
    g = np.array([[ok[i].local_Q[c] @ T / ok[i].photometric[c] for c in range(3)] for i in idx])        # (m, 3, 81)
    sig = np.array([ok[i].noise / ok[i].photometric for i in idx])                                       # (m, 3)
    spread = np.ptp(g - np.nanmedian(g, 0), axis=2) / sig
    for j, i in enumerate(idx):
        ok[i].spread_sigma = spread[j]
    jm, cm = np.unravel_index(int(np.nanargmax(spread)), spread.shape)
    s = np.array([ok[i].photometric for i in idx])
    pct = {nm: float(100 * (np.nanmax(r) / np.nanmin(r) - 1)) for nm, r in (("blue_green", s[:, 2] / s[:, 1]),
                                                                            ("red_green", s[:, 0] / s[:, 1]))}
    mx = float(spread[jm, cm])
    params = {"spread": round(mx, 2), "colour": COLOURS[cm], "frame": int(idx[jm]),
              "colour_pct": round(pct["blue_green"], 1), "red_green_pct": round(pct["red_green"], 1),
              "frames_over_1_sigma": int((np.nanmax(spread, 1) > 1).sum()), "frames": len(idx)}
    return {"spread_sigma": {int(i): [round(float(v), 3) for v in spread[j]] for j, i in enumerate(idx)},
            "max_sigma": round(mx, 3), "median_sigma": [round(float(v), 3) for v in np.nanmedian(spread, 0)],
            "transparency_pct": {nm: round(v, 2) for nm, v in pct.items()},
            "reason": REASON_TEMPLATES["spread" if mx >= 1 else "flat"], "params": params}


def normalize(load: Callable, coef, weight, ch_of_k, H: int, W: int, fwhm: float, *, frames=None, device="cpu",
              sat_level=None, n_best: int = N_BEST, mode: str = "local", ref_frame: int | None = None,
              valid_max: float = VALID_MAX, progress: Callable | None = None) -> Normalisation:
    """The whole normalisation in one call: the reference (``build_reference``), every frame of ``frames`` (default:
    those of positive weight) measured on it (``measure_frame``), the diagnostic (``diagnose``)."""
    progress = progress or (lambda f, msg: None)
    weight = np.asarray(weight, float)
    frames = np.flatnonzero(np.nan_to_num(weight) > 0) if frames is None else np.asarray(frames, int)
    ref = build_reference(load, coef, weight, ch_of_k, H, W, fwhm, device=device, sat_level=sat_level,
                          n_best=n_best, mode=mode, ref_frame=ref_frame, valid_max=valid_max,
                          progress=lambda f, msg: progress(0.4 * f, msg))
    out = {}
    for c_, i in enumerate(frames):
        out[int(i)] = measure_frame(load(int(i)), coef[i], ref, index=int(i))
        progress(0.4 + 0.6 * (c_ + 1) / len(frames), f"normalisation: {c_ + 1}/{len(frames)} frames")
    return Normalisation(ref, out, len(weight), diagnose(out, ref))
