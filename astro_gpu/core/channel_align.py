"""Colour channel alignment that follows the night: where the red and blue images of the stars fall relative to the
green one, per block of consecutive frames, with its field-dependent part (the corners), returned per frame as a
correction of the raw sample positions that the Bayer drizzle applies before the registration polynomial. Nothing
is interpolated: each red or blue sample is simply deposited where its green counterpart would have been.

HOW TO CALL IT (for the integration agent; stacking v2, switch "colour alignment: per block")
=================================================================================================

1. Measure, per frame, in the registration workers (register.py ``_catalog``: the raw frame and its star list are
   both in hand there; 0.2 s for a 12 MP frame with 1 100 isolated stars, against 1.1 s for the detection)::

       from ..core import channel_align
       chan = channel_align.measure_frame(_G["mm"][idx], cat, _G["ch_of_k"], fwhm)   # after `cat` and `fwhm`

   ``_G["ch_of_k"]`` = ``cfa_channel_map(info["cfa"])`` passed through ``_init``. Return ``chan`` (an (m, 8) float32
   array, see the COLUMNS below) with the catalogue and keep it in catalogs.pkl (a 6-tuple: an older 5-tuple
   checkpoint is measured again). A frame without stars gives an empty array. Optionally pass ``sat_level=0.5 *
   satval`` (as the old ChannelOffsets did) when the saturation map is at hand; the default leaves out the stars
   with a sample marked saturated (> VALID_MAX).

2. Fit the night in ``Register.run``, after the polynomials (``coef``, ``rms``)::

       times = channel_align.times_s([m.get("time") for m in meta])
       align = channel_align.fit_night(chans, W, H, times=times, coef=coef, use=np.isfinite(rms))
       np.savez(ctx.out("chan.npz"), **align.to_npz())
       ctx.metric("channel_alignment", align.summary()); ctx.log(align.describe())

   (bump ``Register.version``). ``align.applied`` is False under MIN_STARS (50) isolated stars in the night.

3. Apply it in the drizzle (stack.py ``frame_model``), behind the switch; the old path stays untouched::

       align = channel_align.ChannelAlignment.from_npz(np.load(ctx.input("register", "chan.npz")))  # once
       ...
       if align is not None and align.applied:
           dx, dy = align.shift(i, xx, yy, chmap)        # sensor px, zero on green; cached per block
           X, Y = poly_eval(coef[i], xx - dx, yy - dy)  # instead of poly_eval(coef[i], xx, yy)

   and, with the switch on, skip the old ``ChannelOffsets.measure`` / ``model.shift_channels`` / ``chan_fix.at``
   steps: the samples of pass 1 already land corrected, so the first mean needs no shift. ``frame_norm`` (smooth
   backgrounds) needs no correction. ``shift`` accepts torch tensors (on their device) or numpy arrays.

   If register.py must not change: call ``measure_frame(mm[i], cat_i, ch_of_k, fwhm[i])`` in a pre-pass of stack.py
   with register's catalogues, then step 2 there; it costs one more read of each frame.

4. Optional check against the physics, once the WCS is known (astrometry; for the report or the bench)::

       model = channel_align.dispersion_offsets(times, lat, lon, ra, dec, cd_sensor, coef=coef, W=W, H=H)
       channel_align.check_dispersion(align, model)   # {"ratio": ~1 when the measure follows the refraction, ...}

   ``cd_sensor``: the CD matrix (deg per px) of the reference frame in sensor pixels, x = column, y = row (array
   order): the stack's CD times the drizzle scale. ``lat``/``lon``: the site (FITS SITELAT/SITELONG), degrees, east
   positive; ``times``: UTC seconds (``times_s`` of DATE-OBS; the EXIF time of a camera RAW is local time: subtract
   its time zone). The blocks themselves need no site and no time zone.

THE PHYSICS AND WHAT THE OTHER TOOLS DO
=======================================

- Atmospheric dispersion. Refraction lifts a star towards the zenith by R(lambda) ~ 206265 (n(lambda) - 1) tan z
  arcsec, more in blue than in red: the colours of a star are spread along the vertical circle, by
  dR = 206265 [n(lambda) - n(lambda_G)] tan z, the blue image towards the zenith (Filippenko 1982, eq. 4). The
  refractivity of air is Edlen's (1953) formula at 15 C and 760 mm Hg, corrected for pressure, temperature and water
  vapour (Filippenko 1982, eqs. 1-3; ``air_refractivity``). Magnitude for a Bayer camera at sea level: blue-green
  about 0.4" x tan z, red-green about -0.26" x tan z. It changes with the altitude through the night (tan z), and its
  direction, the parallactic angle q, turns relative to an equatorial mount (Meeus 1998, eq. 14.1). Over a field of a
  degree it varies by a few per cent only (d tan z / dz = sec^2 z): a uniform shift per block of frames.
- Lateral chromatic aberration (refractors, correctors): a wavelength-dependent radial distortion, r (k1 + k3 r^2)
  about the optical axis (the Brown-Conrady model, Brown 1966), fixed in the sensor and growing towards the corners.
  A newtonian has none. Both together: per colour, a cubic polynomial over the sensor (the basis of register.py's
  ``poly_terms``) whose higher terms are fixed for the night and whose constant term (the shift) follows the night.
- PixInsight WBPP 2.5 "Separated RGB" [P17]: the R, G and B images of each debayered frame are registered apart,
  "all on the best Green frame", with distortion correction "to minimize the chromatic aberration"; each channel is
  then resampled (interpolated) by the registration, and "Drizzle is not currently available for separated channels".
- PixInsight ChannelMatch [P22]: one fixed X/Y offset per channel, set by hand, applied by a bicubic spline
  interpolation when fractional (``BicubicSplinePixelInterpolation`` in ChannelMatchInstance.cpp); no field part.
- Siril ``rgbalign`` (1.5, in development) [S10]: aligns the channels of the stacked image (global star registration,
  PSF, DFT or KOMBAT) to correct "the residual colour fringing left by differential atmospheric refraction".
- Here: WBPP's per-frame, per-colour registration on green, but measured per block of frames (enough stars for a
  precise measure) and applied inside the Bayer drizzle as a shift of the raw sample positions: no debayering, no
  resampling.

THE MEASURE
===========

- Per star and per colour, the windowed centroid of SExtractor (XWIN_IMAGE; Bertin, SExtractor manual "windowed
  positional parameters"; SEP ``winpos``, Barbary 2016, the one register.py uses): x <- x + 2 sum(w I (x_i - x)) /
  sum(w I), w = exp(-r^2 / 2 s^2), s = FWHM / 2.3548, inside 4 s, at most 16 iterations, until the step is below
  1e-4 px; its error 4 sum(w^2 sigma^2 (x_i - x)^2) / (sum w I)^2 (ERRX2WIN). Each colour on its own raw samples, at
  their exact positions: the 2-pixel grid of red and blue biases a single star by its sub-pixel phase, which is
  random from star to star and from frame to frame (dithering), so it averages out.
- Stars: the registration catalogue, unsaturated and isolated (no other star above a tenth of its flux within two
  stamp radii, the old ChannelOffsets rule), on the local background of an annulus.
- Fit: per colour and axis, weighted least squares of the offset (colour - green) on the cubic over the sensor, with
  one constant per frame (the exact "within" solution of the per-frame intercepts), iterated 3-sigma clipping (as the
  old ChannelOffsets fit). The weights: 1 / (a v + b), v the formal variance of each star, a and b fitted on the
  residuals (regression of squared residuals on the variance, Carroll and Ruppert 1988, ch. 3). The floor b is the
  sampling noise of the 2-pixel red and blue grid (about 0.09 px a star at FWHM 3 px), which does not shrink with
  the brightness: without it the brightest stars took all the weight and the corners were off by 0.17 px.
- Blocks: consecutive frames in time, a fresh block at each meridian flip (the field turned by more than 90 degrees in
  the registration). Their duration is chosen by leave-one-out cross-validation of the per-frame shifts (Stone 1974),
  with the one-standard-error rule (Breiman et al. 1984; Hastie, Tibshirani and Friedman 2009, 7.10): the longest
  blocks whose prediction error is within one standard error of the best. Short blocks are limited by the noise (the
  number of stars), long ones by the change of the dispersion; a block never holds fewer than MIN_STARS (50) star
  measurements nor fewer than 2 frames. A night whose offsets do not move gets one block: the whole night.
  Applied when a frame has at least MIN_STARS isolated stars.
- Measured (tests/test_channel_align.py, synthetic 6-h night, dec -15 from latitude 45): error 0.02-0.03 px at the
  centre and corners per block, against 0.23 (red) and 0.32 px (blue) for one correction for the night. On the bench
  (Heart, Base, read-only, 100 frames over 5 h, refractor): 5 blocks of ~49 min; red 0.73 px in the worst corner;
  the blue shift at the centre drifted by 0.06 px over the night, following the refraction model (ratio 0.94 +- 0.03,
  74 % of the change explained, for an assumed site in France and summer time).

Sources: Filippenko A. V. 1982, PASP 94, 715; Edlen B. 1953, JOSA 43, 339; Meeus J. 1998, Astronomical Algorithms,
2nd ed., eqs. 12.4, 13.6, 14.1; Brown D. C. 1966, Photogrammetric Engineering 32, 444; Bertin E. & Arnouts S. 1996,
A&AS 117, 393 and the SExtractor manual; Barbary K. 2016, JOSS 1, 58 (SEP); Carroll R. J. & Ruppert D. 1988,
Transformation and Weighting in Regression, ch. 3; Stone M. 1974, JRSS B 36, 111; Breiman
L. et al. 1984, Classification and Regression Trees; Hastie T., Tibshirani R. & Friedman J. 2009, The Elements of
Statistical Learning, 7.10; [P17] https://pixinsight.com/forum/index.php?threads/wbpp-2-5-0-released.19112/ ; [P22]
https://gitlab.com/pixinsight/PCL/-/raw/master/src/modules/processes/Geometry/ChannelMatchInstance.cpp ; [S10]
https://gitlab.com/free-astro/siril-doc/-/raw/main/doc/commands/rgbalign.rst
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import UTC, datetime

import numpy as np
from scipy.spatial import cKDTree

#: fewest isolated star measurements for a correction, and per block (the spec: "activé dès 50 étoiles isolées")
MIN_STARS = 50
#: SExtractor / SEP windowed centroid: aperture of 4 window sigmas, step factor 2, 16 iterations, 1e-4 px
WIN_NSIG = 4.0
WIN_FAC = 2.0
WIN_NITER = 16
WIN_STEPMIN = 1e-4
#: width (px) of the annulus beyond the aperture where the local background of each colour is measured
ANNULUS_PX = 3.0
#: a neighbour above this fraction of a star's flux within two stamp radii makes it not isolated
ISOLATION_FLUX = 0.1
#: iterated clipping of the field fit (the old ChannelOffsets fit: 5 passes at 3 sigma)
CLIP_SIGMA = 3.0
CLIP_ITERS = 5
#: a turn of the field by more than this between two frames is a meridian flip: a new block
FLIP_DEG = 90.0

#: the columns of ``measure_frame``'s array: green position (sensor px), red - green, blue - green (sensor px), and
#: the variance of one axis of each difference (px^2)
X, Y, RX, RY, BX, BY, VR, VB = range(8)
NCOL = 8


# ---------------------------------------------------------------------------------------------- the measure, per frame
def measure_frame(frame, cat, ch_of_k, fwhm: float, sat_level: float = 60000.0, gain: float | None = None,
                  max_stars: int = 1500) -> np.ndarray:
    """The offsets of the red and blue images of the isolated stars of a frame from the green one.

    frame: the calibrated CFA frame (H, W) (a row of the float16 cache memmap is fine: only the stamps are read);
    cat: the frame's star list, (N, >= 2) sensor coordinates x, y (register's catalogue; column 2, the flux, ranks
    the stars and tests their isolation); ch_of_k: the channel (0 R, 1 G, 2 B) of each 2x2 position k = 2 (y % 2) +
    (x % 2) (``cfa_channel_map``); fwhm: the stars' FWHM (sensor px); sat_level: a star with a sample above it in its
    aperture is left out; gain: e-/ADU of the frame for the shot noise of the errors (None: estimated from the
    annuli, sky level / sky variance, as stack.py's auto does).

    Returns an (m, 8) float32 array, columns X, Y (green centroid), RX, RY (red - green), BX, BY (blue - green),
    VR, VB (variance of one axis of each difference), sensor pixels."""
    H, W = frame.shape[:2]
    cat = np.asarray(cat, float)
    if cat.ndim != 2 or len(cat) == 0:
        return np.zeros((0, NCOL), np.float32)
    x, y = cat[:, 0], cat[:, 1]
    flux = cat[:, 2] if cat.shape[1] > 2 else np.ones(len(cat))
    sig = max(float(fwhm), 1.0) / 2.3548           # SExtractor: half-light radius >= 0.5 px
    ra = WIN_NSIG * sig
    rb = ra + ANNULUS_PX
    R = int(math.ceil(rb)) + 1
    fin = np.isfinite(x) & np.isfinite(y) & np.isfinite(flux)
    ok = fin & (flux > 0) & (x >= R) & (x <= W - 1 - R) & (y >= R) & (y <= H - 1 - R)
    # isolated: no other star above a tenth of the flux within two stamp radii
    fi = np.where(fin)[0]
    if len(fi) > 1:
        pairs = cKDTree(np.c_[x[fi], y[fi]]).query_pairs(2 * rb, output_type="ndarray")
        if len(pairs):
            a, b = fi[pairs[:, 0]], fi[pairs[:, 1]]
            ok[a[flux[b] > ISOLATION_FLUX * flux[a]]] = False
            ok[b[flux[a] > ISOLATION_FLUX * flux[b]]] = False
    idx = np.where(ok)[0]
    idx = idx[np.argsort(-flux[idx], kind="stable")][:max_stars]
    if len(idx) == 0:
        return np.zeros((0, NCOL), np.float32)
    cx, cy = np.rint(x[idx]).astype(np.int64), np.rint(y[idx]).astype(np.int64)
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    keep = ox ** 2 + oy ** 2 <= rb ** 2
    ox, oy = ox[keep], oy[keep]
    n = len(idx)
    pos = np.full((3, n, 2), np.nan)               # colour, star, (x, y) relative to (cx, cy)
    sumw = np.zeros((3, n))                        # sum w I
    eb = np.zeros((3, n, 2))                       # sum w^2 d^2 (background part of the error)
    es = np.zeros((3, n, 2))                       # sum w^2 d^2 I+ (shot part)
    sky = np.full((3, n), np.nan)
    skyvar = np.full((3, n), np.nan)
    bad = np.zeros(n, bool)
    for py in (0, 1):
        for px in (0, 1):
            g = np.where((cx % 2 == px) & (cy % 2 == py))[0]
            if len(g) == 0:
                continue
            chan = np.asarray(ch_of_k)[2 * ((py + oy) % 2) + (px + ox) % 2]
            for c in (1, 0, 2):                    # green first: red and blue start from its centroid
                m = chan == c
                oxc, oyc = ox[m].astype(float), oy[m].astype(float)
                v = np.asarray(frame[cy[g, None] + oy[m][None], cx[g, None] + ox[m][None]], np.float32)
                valid = np.isfinite(v)
                r2 = oxc ** 2 + oyc ** 2
                ann = r2 > ra ** 2
                bad[g] |= ((v > sat_level) & (r2 <= ra ** 2)[None]).any(1)
                va = np.where(valid & ann[None], v, np.nan)
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)          # an all-NaN annulus: the star is out
                    b = np.nanmedian(va, axis=1)
                    s = 1.4826 * np.nanmedian(np.abs(va - b[:, None]), axis=1)
                bad[g] |= ~np.isfinite(b)
                b = np.nan_to_num(b)
                I = np.where(valid, v - b[:, None], 0.0)
                if c == 1:
                    x0, y0 = x[idx[g]] - cx[g], y[idx[g]] - cy[g]
                else:
                    x0, y0 = pos[1, g, 0], pos[1, g, 1]
                    x0, y0 = np.nan_to_num(x0), np.nan_to_num(y0)
                xc, yc, sw, w2dx, w2dy, w2dxI, w2dyI = _winpos(I, valid, oxc, oyc, x0, y0, sig, ra)
                pos[c, g, 0], pos[c, g, 1] = xc, yc
                sumw[c, g] = sw
                eb[c, g, 0], eb[c, g, 1] = w2dx, w2dy
                es[c, g, 0], es[c, g, 1] = w2dxI, w2dyI
                sky[c, g], skyvar[c, g] = b, s ** 2
    if gain is None:                               # sky level / sky variance of the green annuli
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lev, var = np.nanmedian(sky[1]), np.nanmedian(skyvar[1])
        gain = float(lev / var) if np.isfinite(lev) and np.isfinite(var) and lev > 0 and var > 0 else np.inf
    inv_g = 0.0 if not np.isfinite(gain) or gain <= 0 else 1.0 / gain
    with np.errstate(all="ignore"):
        var = 4.0 * (skyvar[:, :, None] * eb + inv_g * es) / sumw[:, :, None] ** 2     # (3, n, 2)
    moved = np.hypot(pos[1, :, 0] - (x[idx] - cx), pos[1, :, 1] - (y[idx] - cy))
    good = ~bad & (sumw > 0).all(0) & np.isfinite(pos).all((0, 2)) & np.isfinite(var).all((0, 2))
    good &= (var > 0).all((0, 2)) & (moved < ra)
    for c in (0, 2):
        good &= np.hypot(pos[c, :, 0] - pos[1, :, 0], pos[c, :, 1] - pos[1, :, 1]) < ra
    out = np.zeros((int(good.sum()), NCOL), np.float32)
    out[:, X], out[:, Y] = cx[good] + pos[1, good, 0], cy[good] + pos[1, good, 1]
    out[:, RX], out[:, RY] = pos[0, good, 0] - pos[1, good, 0], pos[0, good, 1] - pos[1, good, 1]
    out[:, BX], out[:, BY] = pos[2, good, 0] - pos[1, good, 0], pos[2, good, 1] - pos[1, good, 1]
    vg = var[1, good].mean(1)
    out[:, VR], out[:, VB] = var[0, good].mean(1) + vg, var[2, good].mean(1) + vg
    return out


def _winpos(I, valid, ox, oy, x0, y0, sig, ra):
    """SExtractor's windowed centroid of each row of samples I (n, p) at offsets (ox, oy) (p,), from (x0, y0) (n,).
    Returns the centroid, sum w I, and the sums of w^2 d^2 and w^2 d^2 I+ per axis (for ERRX2WIN)."""
    x, y = x0.astype(float).copy(), y0.astype(float).copy()
    inv2 = 0.5 / sig ** 2
    vf = valid.astype(float)
    live = np.ones(len(x), bool)
    for _ in range(WIN_NITER):
        dx = ox[None] - x[:, None]
        dy = oy[None] - y[:, None]
        r2 = dx * dx + dy * dy
        w = np.exp(-r2 * inv2) * vf * (r2 <= ra * ra)
        wi = w * I
        s = wi.sum(1)
        pos = s > 0
        safe = np.where(pos, s, 1.0)
        sx = np.where(pos & live, WIN_FAC * (wi * dx).sum(1) / safe, 0.0)
        sy = np.where(pos & live, WIN_FAC * (wi * dy).sum(1) / safe, 0.0)
        x += sx
        y += sy
        live &= np.hypot(sx, sy) >= WIN_STEPMIN
        if not live.any():
            break
    dx = ox[None] - x[:, None]
    dy = oy[None] - y[:, None]
    r2 = dx * dx + dy * dy
    w = np.exp(-r2 * inv2) * vf * (r2 <= ra * ra)
    s = (w * I).sum(1)
    w2 = w * w
    Ip = np.clip(I, 0, None)
    return (x, y, s, (w2 * dx * dx).sum(1), (w2 * dy * dy).sum(1), (w2 * dx * dx * Ip).sum(1),
            (w2 * dy * dy * Ip).sum(1))


# ------------------------------------------------------------------------------------------------ the field polynomial
def nterms(order: int) -> int:
    return (order + 1) * (order + 2) // 2


def poly_terms(x, y, W: int, H: int, order: int = 3) -> np.ndarray:
    """Monomials up to ``order`` on sensor coordinates normalised to [-1, 1]: register.py's basis
    (1, u, v, u^2, u v, v^2, u^3, u^2 v, u v^2, v^3)."""
    u = np.asarray(x, float) / (W / 2) - 1.0
    v = np.asarray(y, float) / (H / 2) - 1.0
    t = [np.ones_like(u)]
    for d in range(1, order + 1):
        for k in range(d + 1):
            t.append(u ** (d - k) * v ** k)
    return np.stack(t, -1)


def poly_shift(q, x, y, chmap, W: int, H: int):
    """The offset (dx, dy) (sensor px) of each sample at (x, y) in its own channel (``chmap``: 0 R, 1 G, 2 B) from
    the coefficients q (3, 2, T) (zero on green). Works on numpy arrays or torch tensors (on their device), one
    monomial at a time (no stack of terms in memory)."""
    q = np.asarray(q, float)
    T = q.shape[-1]
    order = int(round((math.sqrt(8 * T + 1) - 3) / 2))
    is_np = isinstance(x, np.ndarray)
    if is_np:
        where = np.where
        zeros = np.zeros_like(np.asarray(x, np.float64))
    else:
        import torch
        where = torch.where
        zeros = torch.zeros_like(x, dtype=torch.float32)
    u = x / (W / 2) - 1.0
    v = y / (H / 2) - 1.0
    acc = {(c, a): zeros.clone() if not is_np else zeros.copy() for c in (0, 2) for a in (0, 1)}
    t = 0
    for d in range(order + 1):
        for k in range(d + 1):
            m = None if d == 0 else (u ** (d - k)) * (v ** k)
            for c in (0, 2):
                for a in (0, 1):
                    co = float(q[c, a, t])
                    if co != 0.0:
                        acc[(c, a)] = acc[(c, a)] + (co if m is None else co * m)
            t += 1
    dx = where(chmap == 0, acc[(0, 0)], where(chmap == 2, acc[(2, 0)], zeros))
    dy = where(chmap == 0, acc[(0, 1)], where(chmap == 2, acc[(2, 1)], zeros))
    return dx, dy


# -------------------------------------------------------------------------------------------------- the fit, the night
def _variance_function(v, e):
    """The variance of the residuals ``e`` as a x v + b (v: the formal variance of each measure; b: a floor, the
    sampling noise of the 2-pixel red and blue grid, which does not shrink with the brightness), fitted on the robust
    variances of 8 bins of v (the regression of squared residuals on the variance covariates, Carroll and Ruppert
    1988, ch. 3), a, b >= 0."""
    ok = np.isfinite(v) & np.isfinite(e) & (v > 0)
    v, e = v[ok], e[ok]
    if len(v) < 80:
        s2 = float((1.4826 * np.median(np.abs(e / np.sqrt(v)))) ** 2) if len(v) >= 3 else 1.0
        return s2, 0.0
    o = np.argsort(v)
    vb, sb = [], []
    for q in np.array_split(o, 8):
        vb.append(np.median(v[q]))
        sb.append((1.4826 * np.median(np.abs(e[q] - np.median(e[q])))) ** 2)
    vb, sb = np.array(vb), np.array(sb)
    a, b = np.linalg.lstsq(np.c_[vb, np.ones_like(vb)], sb, rcond=None)[0]
    if b < 0:
        a, b = float(np.sum(sb * vb) / np.sum(vb * vb)), 0.0
    if a < 0:
        a, b = 0.0, float(np.mean(sb))
    return max(float(a), 1e-12), max(float(b), 0.0)


def _within_fit(fid, Z, yv, v, nf):
    """Weighted least squares of yv = a[fid] + Z beta with one constant per frame (the exact within-frame solution);
    the weights 1 / (a v + b) from the formal variances ``v`` and the residuals (``_variance_function``), iterated
    with 3-sigma clipping. Returns (a (nf,), beta, kept weight per frame (1 / variance of a), kept rows per frame)."""
    v = np.asarray(v, float)
    keep = np.isfinite(yv) & np.isfinite(v) & (v > 0)
    beta = np.zeros(Z.shape[1])
    a = np.full(nf, np.nan)
    va, vb = 1.0, 0.0
    for _ in range(CLIP_ITERS):
        w = np.where(np.isfinite(v) & (v > 0), 1.0 / (va * np.where(v > 0, v, 1.0) + vb), 0.0)
        ww = np.where(keep, w, 0.0)
        Wf = np.bincount(fid, ww, nf)
        has = Wf > 0
        sW = np.where(has, Wf, 1.0)
        ybar = np.bincount(fid, ww * np.nan_to_num(yv), nf) / sW
        if Z.shape[1]:
            Zbar = np.stack([np.bincount(fid, ww * Z[:, j], nf) for j in range(Z.shape[1])], 1) / sW[:, None]
            rows = keep & has[fid]
            sq = np.sqrt(ww[rows])
            beta = np.linalg.lstsq((Z[rows] - Zbar[fid[rows]]) * sq[:, None],
                                   (yv[rows] - ybar[fid[rows]]) * sq, rcond=None)[0]
            a = np.where(has, ybar - Zbar @ beta, np.nan)
            e = yv - a[fid] - Z @ beta
        else:
            a = np.where(has, ybar, np.nan)
            e = yv - a[fid]
        fin = np.isfinite(e) & np.isfinite(v) & (v > 0)
        if fin.sum() < 3:
            break
        va, vb = _variance_function(v[fin], e[fin])
        sd = np.sqrt(va * np.where(v > 0, v, np.inf) + vb)
        keep = fin & (np.abs(e) < CLIP_SIGMA * sd)
    w = np.where(np.isfinite(v) & (v > 0), 1.0 / (va * np.where(v > 0, v, 1.0) + vb), 0.0)
    ww = np.where(keep, w, 0.0)
    return a, beta, np.bincount(fid, ww, nf), np.bincount(fid, keep.astype(float), nf)


def _jacobian(coef, W: int, H: int) -> np.ndarray:
    """The linear part (n, 2, 2) of register's polynomials at the sensor centre: d(reference) / d(frame)."""
    c = np.asarray(coef, float)
    return np.stack([np.stack([c[:, 0, 1] * 2 / W, c[:, 0, 2] * 2 / H], -1),
                     np.stack([c[:, 1, 1] * 2 / W, c[:, 1, 2] * 2 / H], -1)], 1)


def _segments(order_f, coef, use, W, H):
    """A segment id per frame (in time order ``order_f``): a new one at each meridian flip (the registration turned
    the field by more than FLIP_DEG between two used frames)."""
    n = len(order_f)
    seg = np.zeros(n, int)
    if coef is None:
        return seg
    J = _jacobian(coef, W, H)
    ang = np.degrees(np.arctan2(J[:, 1, 0] - J[:, 0, 1], J[:, 0, 0] + J[:, 1, 1]))
    cur, last = 0, None
    for j, i in enumerate(order_f):
        if use[i] and np.isfinite(ang[i]):
            if last is not None and abs((ang[i] - last + 180.0) % 360.0 - 180.0) > FLIP_DEG:
                cur += 1
            last = ang[i]
        seg[j] = cur
    return seg


def _partition(t, seg, stars, D, min_stars):
    """Blocks of consecutive frames (time order) of about ``D`` (seconds): a block closes once it spans D and holds
    at least ``min_stars`` star measurements and 2 frames; a short last block joins the previous one of its segment;
    never across a segment. Returns a block id per frame."""
    blk = np.zeros(len(t), int)
    b = -1
    for s_ in np.unique(seg):
        idx = np.where(seg == s_)[0]
        b += 1
        first_b = b
        t0, ns, nfr = t[idx[0]], 0, 0
        for i in idx:
            if nfr >= 2 and ns >= min_stars and t[i] - t0 >= D:
                b += 1
                t0, ns, nfr = t[i], 0, 0
            blk[i] = b
            ns += stars[i]
            nfr += 1
        if b > first_b and (nfr < 2 or ns < min_stars):        # the short tail joins the block before it
            blk[blk == b] = b - 1
            b -= 1
    return blk


def _block_means(blk, a, var):
    """Inverse-variance means of the per-frame shifts a (n, K) in each block, and their variances."""
    nb = int(blk.max()) + 1
    w = np.where(np.isfinite(var) & (var > 0) & np.isfinite(a), 1.0 / np.where(var > 0, var, 1.0), 0.0)
    W_ = np.stack([np.bincount(blk, w[:, k], nb) for k in range(a.shape[1])], 1)
    S_ = np.stack([np.bincount(blk, w[:, k] * np.nan_to_num(a[:, k]), nb) for k in range(a.shape[1])], 1)
    with np.errstate(all="ignore"):
        return S_ / W_, 1.0 / W_, W_, S_, w


def _cv(blk, a, var):
    """Leave-one-out error of each frame's shift predicted by the other frames of its block (squared px, summed over
    the four components); NaN for a frame alone in its block."""
    _, _, W_, S_, w = _block_means(blk, a, var)
    Wl = W_[blk] - w
    with np.errstate(all="ignore"):
        loo = (S_[blk] - w * np.nan_to_num(a)) / Wl
    return np.where(Wl > 0, (a - loo) ** 2, np.nan).sum(1)          # NaN unless all four components are there


class ChannelAlignment:
    """The per-frame correction of the red and blue sample positions (sensor px, before the registration polynomial):
    ``coefs`` (n, 3, 2, T) on ``poly_terms`` (green zero), the block of each frame and what was measured."""

    def __init__(self, coefs, W, H, order, block_of, blocks, info, frame_shift=None, frame_var=None):
        self.coefs = None if coefs is None else np.asarray(coefs, float)
        self.W, self.H, self.order = int(W), int(H), int(order)
        self.block_of = np.asarray(block_of, int)
        self.blocks = blocks
        self.info = info
        self.frame_shift = frame_shift              # (n, 4) per-frame centre shift R x, R y, B x, B y (NaN: none)
        self.frame_var = frame_var
        self.applied = coefs is not None
        self._cache = None

    # ---- applying it
    def shift(self, i: int, x, y, chmap):
        """(dx, dy), sensor px, of the samples of frame ``i`` at sensor positions (x, y) in their channel ``chmap``
        (0 R, 1 G, 2 B): deposit each at ``poly(coef[i], x - dx, y - dy)``. Cached while the coefficients and the
        grids stay the same (consecutive frames of a block)."""
        q = self.coefs[int(i)]
        key = (q.tobytes(), id(x), id(y), id(chmap))
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        r = poly_shift(q, x, y, chmap, self.W, self.H)
        self._cache = (key, r, (x, y, chmap))      # the grids held: their ids stay theirs
        return r

    def at(self, i: int, x, y) -> np.ndarray:
        """The offsets (3, m, 2) of the colours from green at sensor points (x, y) of frame ``i`` (green zero)."""
        T = poly_terms(np.atleast_1d(x), np.atleast_1d(y), self.W, self.H, self.order)
        return np.einsum("ckt,mt->cmk", self.coefs[int(i)], T)

    # ---- telling it
    def summary(self) -> dict:
        return self.info

    def describe(self) -> str:
        s = self.info
        if not self.applied:
            return f"colours not registered on each other: {s.get('stars', 0)} isolated star measurements only"
        c, e, mv = s["centre_px"], s["corner_max_px"], s["night_range_px"]
        nb = s["blocks"]
        size = f"{s['block_minutes']:.0f} min" if s.get("timed") else f"{s.get('block_frames', 0):.0f} frames"
        why = " (the offsets did not move beyond their noise)" if s.get("block_choice") != "given" else ""
        how = f"one correction for the whole night{why}" if nb <= s.get("segments", 1) else f"{nb} blocks of about {size}"
        return (f"colours registered on each other, {how}, {s['stars']} star measurements: red "
                f"{np.hypot(*c['R']):.2f} px at the centre, up to {e['R']:.2f} px in the corners; blue "
                f"{np.hypot(*c['B']):.2f} px, up to {e['B']:.2f} px; over the night the shift moved by up to "
                f"{mv['R']:.2f} px (red) and {mv['B']:.2f} px (blue)")

    # ---- storing it
    def to_npz(self) -> dict:
        d = {"W": self.W, "H": self.H, "order": self.order, "block_of": self.block_of,
             "applied": self.applied, "info": json.dumps(self.info), "blocks": json.dumps(self.blocks)}
        if self.applied:
            d["coefs"] = self.coefs
        if self.frame_shift is not None:
            d["frame_shift"], d["frame_var"] = self.frame_shift, self.frame_var
        return d

    @classmethod
    def from_npz(cls, z) -> ChannelAlignment | None:
        """From ``np.load`` of ``to_npz``'s file; None when nothing was applied."""
        if not bool(z["applied"]):
            return None
        return cls(z["coefs"], int(z["W"]), int(z["H"]), int(z["order"]), z["block_of"], json.loads(str(z["blocks"])),
                   json.loads(str(z["info"])), z["frame_shift"] if "frame_shift" in z.files else None,
                   z["frame_var"] if "frame_var" in z.files else None)


def fit_night(meas, W: int, H: int, times=None, coef=None, use=None, order: int = 3, field: str = "night",
              block_s: float | None = None, min_stars: int = MIN_STARS) -> ChannelAlignment:
    """The colour correction of every frame from the per-frame measures (``measure_frame``; None or empty for a
    frame without).

    times: per frame, seconds (``times_s``); None or any missing: the frame order (block_s then counts frames);
    coef: register's polynomials (n, 2, 10), to start a new block at each meridian flip; use: the frames to fit
    (default: those with measures); order: of the field polynomial (3: register's cubic; 2: the old quadratic);
    field: "night" (the field part, lateral chromatic aberration, fixed in the sensor for the night, the shift per
    block) or "block" (each block its own polynomial); block_s: the block duration (None: cross-validated; np.inf:
    one correction for the whole night, the old behaviour but in sensor coordinates)."""
    n = len(meas)
    T = nterms(order)
    sizes = np.array([0 if m is None else len(m) for m in meas])
    use = sizes > 0 if use is None else (np.asarray(use, bool) & (sizes > 0))
    t = None if times is None else np.asarray(times, float)
    timed = bool(t is not None and len(t) == n and np.isfinite(t[use]).all() and use.any())
    if not timed:
        t = np.arange(n, dtype=float)
    elif not np.isfinite(t).all():                         # a frame left out without a time: between its neighbours
        k = np.isfinite(t)
        t = np.where(k, t, np.interp(np.arange(n), np.where(k)[0], t[k]))
    order_f = np.argsort(t, kind="stable")
    seg_all = _segments(order_f, coef, use, W, H)
    fr = [i for i in order_f if use[i]]                    # the fitted frames, in time order
    total = int(sizes[fr].sum()) if fr else 0
    base = {"applied": False, "stars": total, "frames": len(fr), "order": order, "field": field,
            "stars_per_frame": int(np.median(sizes[fr])) if fr else 0}
    if not fr or int(sizes[fr].max()) < min_stars:        # the spec: from 50 isolated stars (distinct, in a frame)
        return ChannelAlignment(None, W, H, order, np.zeros(n, int), [], base)
    nf = len(fr)
    data = np.concatenate([np.asarray(meas[i], float) for i in fr])
    fid = np.repeat(np.arange(nf), sizes[fr])
    Zall = poly_terms(data[:, X], data[:, Y], W, H, order)
    Z = Zall[:, 1:]
    comps = ((RX, VR), (RY, VR), (BX, VB), (BY, VB))
    a = np.full((nf, 4), np.nan)
    var = np.full((nf, 4), np.nan)
    beta_n = np.zeros((4, T - 1))
    kept = np.zeros((nf, 4))
    for k, (col, vc) in enumerate(comps):
        ak, bk, Wk, nk = _within_fit(fid, Z, data[:, col], data[:, vc], nf)
        a[:, k], beta_n[k], kept[:, k] = ak, bk, nk
        with np.errstate(all="ignore"):
            var[:, k] = np.where(Wk > 0, 1.0 / Wk, np.nan)
    stars = kept.min(1).astype(int)
    tf = t[fr]
    pos_in_order = {int(i): j for j, i in enumerate(order_f)}
    seg = np.array([seg_all[pos_in_order[int(i)]] for i in fr])
    nseg = len(np.unique(seg))

    # ---- the block duration: leave-one-out cross-validation, one-standard-error rule
    span = float(tf.max() - tf.min()) if nf > 1 else 0.0
    cad = float(np.median(np.diff(tf))) if nf > 1 else 1.0
    cv_curve = []
    if block_s is None:
        cands = [np.inf]
        if span > 0 and cad > 0:
            cands = list(np.geomspace(max(cad, 1e-9) * 1.5, span, 24)) + [np.inf]
        res = []
        for D in cands:
            blk = _partition(tf, seg, stars, D, min_stars)
            e = _cv(blk, a, var)
            m = np.isfinite(e)
            if m.sum() < 2:
                continue
            res.append((D, float(e[m].mean()), float(e[m].std() / np.sqrt(m.sum())), int(blk.max()) + 1))
        if res:
            best = min(res, key=lambda r: r[1])
            ok_ = [r for r in res if r[1] <= best[1] + best[2]]
            D_star = max(ok_, key=lambda r: (r[0], -r[3]))[0]
            cv_curve = [[None if not np.isfinite(r[0]) else round(float(r[0]), 1), round(r[1], 6), round(r[2], 6),
                         r[3]] for r in res]
        else:
            D_star = np.inf
    else:
        D_star = float(block_s)
    blk = _partition(tf, seg, stars, D_star, min_stars)
    nb = int(blk.max()) + 1
    A, Avar, _, _, _ = _block_means(blk, a, var)               # (nb, 4)
    beta_b = np.repeat(beta_n[None], nb, 0)                   # (nb, 4, T-1)
    if field == "block":
        for b in range(nb):
            rows = np.isin(fid, np.where(blk == b)[0])
            fb = np.unique(fid[rows])
            remap = np.searchsorted(fb, fid[rows])
            for k, (col, vc) in enumerate(comps):
                ak, bk, Wk, _ = _within_fit(remap, Z[rows], data[rows, col], data[rows, vc], len(fb))
                wv = np.where(Wk > 0, Wk, 0.0)
                beta_b[b, k] = bk
                if wv.sum() > 0:
                    A[b, k] = float(np.nansum(wv * ak) / wv.sum())
                    Avar[b, k] = 1.0 / wv.sum()
    night_blk = _partition(tf, seg, stars, np.inf, min_stars)
    An, _, _, _, _ = _block_means(night_blk, a, var)

    # ---- per frame (every frame: the block of the nearest fitted frame in time, in its segment)
    block_of = np.zeros(n, int)
    for j, i in enumerate(order_f):
        same = np.where(seg == seg_all[j])[0]
        cand = same if len(same) else np.arange(nf)
        block_of[i] = blk[cand[np.argmin(np.abs(tf[cand] - t[i]))]]
    coefs = np.zeros((n, 3, 2, T))
    for k, (c, ax) in enumerate(((0, 0), (0, 1), (2, 0), (2, 1))):
        coefs[:, c, ax, 0] = A[block_of, k]
        coefs[:, c, ax, 1:] = beta_b[block_of, k]
    fshift = np.full((n, 4), np.nan)
    fvar = np.full((n, 4), np.nan)
    fshift[fr], fvar[fr] = a, var

    # ---- what to tell
    corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], float)
    Tc = poly_terms(corners[:, 0], corners[:, 1], W, H, order)
    Tm = poly_terms([W / 2], [H / 2], W, H, order)
    centre, corner = {}, {}
    wf = np.array([np.sum(blk == b) for b in range(nb)], float)
    for nm, k0 in (("R", 0), ("B", 2)):
        cc = np.einsum("bkt,mt->bmk", np.concatenate([A[:, k0:k0 + 2, None], beta_b[:, k0:k0 + 2]], 2), Tm)[:, 0]
        centre[nm] = [round(float(v), 3) for v in np.average(cc, axis=0, weights=wf)]
        ce = np.einsum("bkt,mt->bmk", np.concatenate([A[:, k0:k0 + 2, None], beta_b[:, k0:k0 + 2]], 2), Tc)
        corner[nm] = round(float(np.hypot(ce[..., 0], ce[..., 1]).max()), 3)
    first = np.array([np.where(blk == b)[0][0] for b in range(nb)])
    dblk = A - An[night_blk[first]]                            # per block: its shift minus the whole night's
    moved = {nm: round(float(np.hypot(dblk[:, k0], dblk[:, k0 + 1]).max()), 3) for nm, k0 in (("R", 0), ("B", 2))}
    diff = A[blk] - An[night_blk]                              # per fitted frame: block minus whole night
    vs_night = {nm: round(float(np.sqrt(np.nanmean(np.sum(diff[:, k0:k0 + 2] ** 2, 1)))), 4)
                for nm, k0 in (("R", 0), ("B", 2))}
    blocks = []
    for b in range(nb):
        m = np.where(blk == b)[0]
        blocks.append({"first": int(fr[m[0]]), "last": int(fr[m[-1]]), "frames": int(len(m)),
                       "t0": float(tf[m[0]]), "t1": float(tf[m[-1]]), "stars": int(stars[m].sum()),
                       "shift_px": [round(float(v), 4) for v in A[b]],
                       "error_px": [round(float(np.sqrt(v)), 4) if np.isfinite(v) else None for v in Avar[b]]})
    dur = [bl["t1"] - bl["t0"] + (cad if nf > 1 else 0.0) for bl in blocks]
    info = {**base, "applied": True, "segments": nseg, "blocks": nb, "timed": timed,
            "block_choice": "cross-validated" if block_s is None else "given",
            "block_s": None if not np.isfinite(D_star) else round(float(D_star), 1),
            "block_minutes" if timed else "block_frames": round(float(np.median(dur)) / (60.0 if timed else 1.0), 1),
            "stars_per_block_min": int(min(
                bl["stars"] for bl in blocks)), "centre_px": centre, "corner_max_px": corner,
            "night_range_px": moved, "vs_whole_night_rms_px": vs_night,
            "precision_px": round(float(np.sqrt(np.nanmedian(Avar))), 4), "cv": cv_curve}
    return ChannelAlignment(coefs, W, H, order, block_of, blocks, info, fshift, fvar)


# ------------------------------------------------------------------------------- the physics: dispersion vs altitude
def times_s(stamps) -> np.ndarray:
    """UTC seconds (Unix) of ISO time stamps (FITS DATE-OBS; naive = UTC); NaN where missing or unreadable."""
    out = np.full(len(stamps), np.nan)
    for i, s in enumerate(stamps):
        if not s:
            continue
        try:
            d = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
        except ValueError:
            continue
        if d.tzinfo is None:
            d = d.replace(tzinfo=UTC)
        out[i] = d.timestamp()
    return out


def air_refractivity(lam_um, pressure_mmhg: float = 760.0, temperature_c: float = 15.0,
                     vapour_mmhg: float = 0.0):
    """n - 1 of air at wavelength ``lam_um`` (micrometres): Edlen (1953) at 15 C and 760 mm Hg, corrected for the
    pressure and temperature and for water vapour of partial pressure ``vapour_mmhg`` (Filippenko 1982, eqs. 1-3)."""
    s2 = 1.0 / np.asarray(lam_um, float) ** 2
    n1 = 1e-6 * (64.328 + 29498.1 / (146.0 - s2) + 255.4 / (41.0 - s2))
    P, Tc = float(pressure_mmhg), float(temperature_c)
    n1 = n1 * P * (1.0 + (1.049 - 0.0157 * Tc) * 1e-6 * P) / (720.883 * (1.0 + 0.003661 * Tc))
    return n1 - 1e-6 * (0.0624 - 0.000680 * s2) / (1.0 + 0.003661 * Tc) * float(vapour_mmhg)


def zenith_parallactic(times_unix, lat_deg: float, lon_deg: float, ra_deg: float, dec_deg: float):
    """Zenith distance z and parallactic angle q (degrees; q: position angle of the zenith seen from the target,
    from north through east) at UTC seconds ``times_unix`` (Meeus 1998, eqs. 12.4, 13.6, 14.1)."""
    jd = 2440587.5 + np.asarray(times_unix, float) / 86400.0
    T = (jd - 2451545.0) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T ** 2 - T ** 3 / 38710000.0
    h = np.radians((gmst + lon_deg - ra_deg) % 360.0)
    phi, dec = math.radians(lat_deg), math.radians(dec_deg)
    cz = math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * np.cos(h)
    z = np.degrees(np.arccos(np.clip(cz, -1, 1)))
    q = np.degrees(np.arctan2(np.sin(h), math.tan(phi) * math.cos(dec) - math.sin(dec) * np.cos(h)))
    return z, q


def dispersion_offsets(times_unix, lat_deg: float, lon_deg: float, ra_deg: float, dec_deg: float, cd,
                       coef=None, W: int | None = None, H: int | None = None,
                       wavelengths_um=(0.600, 0.530, 0.465), pressure_hpa: float | None = None,
                       elevation_m: float = 0.0, temperature_c: float = 10.0, vapour_mmhg: float = 8.0) -> dict:
    """The offsets of the R, G, B images from the green one that atmospheric dispersion predicts, per frame, in each
    frame's sensor pixels: ``{"offsets": (n, 3, 2), "zenith_deg", "parallactic_deg"}``.

    dR_c = 206265 [n(lambda_c) - n(lambda_G)] tan z arcsec towards the zenith (Filippenko 1982, eq. 4), the zenith at
    position angle q; turned into the reference frame's pixels by the inverse of ``cd`` (2x2, deg per sensor px,
    x = column, y = row; the first world axis increasing to the east), then into each frame's by the inverse of the
    linear part of its registration (``coef``, register's (n, 2, 10)). wavelengths_um: the effective wavelengths of the
    camera's R, G, B filters (a Bayer CMOS for a sun-like star by default); the pressure from the site's elevation by
    the standard atmosphere (ISO 2533) when not given; temperature and water vapour: Filippenko's typical night."""
    t = np.asarray(times_unix, float)
    if pressure_hpa is None:
        pressure_hpa = 1013.25 * (1.0 - 2.25577e-5 * float(elevation_m)) ** 5.25588
    P = float(pressure_hpa) * 0.750062
    n1 = air_refractivity(np.asarray(wavelengths_um, float), P, temperature_c, vapour_mmhg)
    z, q = zenith_parallactic(t, lat_deg, lon_deg, ra_deg, dec_deg)
    with np.errstate(all="ignore"):
        tz = np.where(z < 90.0, np.tan(np.radians(z)), np.nan)
    dR = 206264.806 * (n1[None, :] - n1[1]) * tz[:, None]               # (n, 3) arcsec, towards the zenith
    qq = np.radians(q)
    en = np.stack([dR * np.sin(qq)[:, None], dR * np.cos(qq)[:, None]], -1) / 3600.0      # (n, 3, 2) east, north deg
    pix = np.einsum("ij,ncj->nci", np.linalg.inv(np.asarray(cd, float)), en)
    if coef is not None:
        J = _jacobian(coef, W, H)
        okJ = np.abs(np.linalg.det(J)) > 1e-9
        Ji = np.where(okJ[:, None, None], np.linalg.inv(np.where(okJ[:, None, None], J, np.eye(2))), np.nan)
        pix = np.einsum("nij,ncj->nci", Ji, pix)
    return {"offsets": pix, "zenith_deg": z, "parallactic_deg": q}


def check_dispersion(align: ChannelAlignment, model: dict) -> dict | None:
    """How well the measured per-frame shifts follow the dispersion model: weighted fit of measured = c + s x
    predicted over the frames (c: the fixed part, the lateral chromatic aberration at the centre). ``ratio`` (s) near
    1: the measure follows the refraction; ``explained``: the part of the measured change over the night the model
    accounts for; residual and change in px. None without measures or with fewer than 3 frames."""
    if align is None or not align.applied or align.frame_shift is None:
        return None
    a, v = align.frame_shift, align.frame_var
    p = np.asarray(model["offsets"], float)[:, (0, 0, 2, 2), (0, 1, 0, 1)]          # (n, 4) as R x, R y, B x, B y
    m = np.isfinite(a).all(1) & np.isfinite(p).all(1) & np.isfinite(v).all(1)
    if m.sum() < 3:
        return None
    a, v, p = a[m], v[m], p[m]
    w = 1.0 / v
    rows, rhs, wt = [], [], []
    for k in range(4):
        r = np.zeros((len(a), 5))
        r[:, k] = 1.0
        r[:, 4] = p[:, k]
        rows.append(r)
        rhs.append(a[:, k])
        wt.append(w[:, k])
    A_, b_, w_ = np.concatenate(rows), np.concatenate(rhs), np.concatenate(wt)
    sw = np.sqrt(w_)
    sol, *_ = np.linalg.lstsq(A_ * sw[:, None], b_ * sw, rcond=None)
    res = b_ - A_ @ sol
    cov = np.linalg.pinv((A_ * w_[:, None]).T @ A_) * (np.sum(w_ * res ** 2) / max(1, len(b_) - 5))
    cfix = sol[:4]
    change = a - np.average(a, axis=0, weights=w)
    tot = float(np.sqrt(np.mean(np.sum(change ** 2, 1))))
    resid = float(np.sqrt(np.mean(np.sum(res.reshape(4, -1).T ** 2, 1))))
    pc = p - p.mean(0)
    with np.errstate(all="ignore"):
        corr = float(np.sum(change * pc) / np.sqrt(np.sum(change ** 2) * np.sum(pc ** 2)))
    return {"ratio": round(float(sol[4]), 3), "ratio_error": round(float(np.sqrt(cov[4, 4])), 3),
            "fixed_px": [round(float(x), 3) for x in cfix], "change_rms_px": round(tot, 4),
            "residual_rms_px": round(resid, 4), "correlation": round(corr, 3),
            "explained": round(1.0 - (resid / tot) ** 2, 3) if tot > 0 else None, "frames": int(m.sum())}
