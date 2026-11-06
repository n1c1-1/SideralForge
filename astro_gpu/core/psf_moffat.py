"""Star size (FWHM), roundness and Moffat beta, by fitting an elliptical Moffat profile to isolated, unsaturated stars
at native resolution: on the samples of one colour of a raw CFA frame, at their true positions, or on one plane of a
stack. Entry points: ``measure_psf``, ``measure_psf_rgb``, ``find_stars``.

The method of Siril's PSF (star finder, Dynamic PSF) and of PixInsight's DynamicPSF / SubframeSelector [S3][P21]:

    I(x, y) = B + A (1 + Q)^-beta,   Q = a dx^2 + 2 b dx dy + c dy^2,   dx = x - x0, dy = y - y0

an ellipse at any angle, fitted by Levenberg-Marquardt on the values at the pixel centres (Siril: GSL's
Levenberg-Marquardt on B, A, x0, y0, two widths, the angle and beta, beta <= 10). Per star: the FWHM along each axis is
2 alpha sqrt(2^(1/beta) - 1) (Siril's ``FWHM_from_s``), alpha = 1 / sqrt(eigenvalue of [[a, b], [b, c]]); roundness =
FWHM_minor / FWHM_major (Siril's r); eccentricity = sqrt(1 - r^2) (SubframeSelector). A star is kept as Siril's star
finder keeps it (``reject_star`` and the defaults of ``settings.c``): roundness >= 0.5, beta >= 1.5, RMS residual <= 0.2
x the amplitude; and, as both tools do for their statistics, no saturated sample and no other catalogued star in its
window. The value of the image is the median over the kept stars (SubframeSelector weights each star by its residual
instead; the median is the robust equivalent). ``fwhm`` is the geometric mean sqrt(major x minor), the width of the
circle of the same area.

Why: a half-flux radius on 2x2 super-pixels overestimates the FWHM of undersampled stars about twofold; the Moffat
fit on the samples of one colour does not. Validated on synthetic Moffat stars of known FWHM (1.3 to 5 px, beta 2.5
to 4.5): within 2 % on a plane, 3 % on the red, green or blue samples of a CFA frame (tests/test_psf_moffat.py).

Callers
-------
* register.py, in the worker ``_catalog(idx)`` (one calibrated frame): ``measure_psf`` on the green samples of the
  CFA frame at the catalogued star positions; ``psf.fwhm`` (sensor px), ``psf.roundness`` and ``psf.beta`` go into
  reg.npz as ``fwhm_moffat``, ``roundness_moffat``... (NaN when ``psf.ok`` is False: fewer than MIN_STARS stars
  fitted). About 0.1 to 0.2 s per frame and colour for 150 stars.
* stack.py, ``Stack.auto``: the median over the aligned frames of ``fwhm_moffat`` is the FWHM of the drizzle rule
  (``core/coverage.suggest_drizzle``). Frame selection (k x sigma on FWHM, roundness) reads the same arrays.
* On a stack (Inspect)::

      xy = find_stars(stack[1])                                        # green plane, brightest first
      per_colour = measure_psf_rgb(stack, xy, saturation=near_saturation_mask)

  gives one result per colour, in output pixels (divide by the drizzle scale for sensor pixels).
Pure numpy (scipy for the neighbour search, ``sep`` only in ``find_stars``); no torch, no GPU.

Sources:
[S3] Siril 1.4, registration: https://gitlab.com/free-astro/siril-doc/-/raw/1.4/doc/preprocessing/registration.rst
[P21] PixInsight SubframeSelector documentation: https://gitlab.com/pixinsight/Reference-Documentation/-/raw/master/tools/obsolete/SubframeSelector/SubframeSelector.pidoc
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

CHANNELS = "RGB"
#: Siril's star finder defaults (src/core/settings.c, starfinder_conf): minimum roundness and minimum Moffat beta
MIN_ROUNDNESS = 0.5
MIN_BETA = 1.5
#: Siril's upper bound of beta in the fit (MOFFAT_BETA_UBOUND)
MAX_BETA = 10.0
#: Siril's reject_star: a fit whose RMS residual exceeds 0.2 x its amplitude is not a star (sigma = 1, the default)
MAX_REL_RMSE = 0.2
#: fewest fitted stars for a measure
MIN_STARS = 10
#: Levenberg-Marquardt iterations at most
MAX_ITER = 60
#: positions looked at, in the order given (the brightest are often saturated: the fits take the brightest of the others)
MAX_CANDIDATES = 5000


@dataclass
class PSF:
    """The star profile of an image (one colour): medians over the fitted stars, and the stars themselves."""

    fwhm: float                 # median sqrt(FWHM_major x FWHM_minor), pixels of the image given
    fwhm_major: float
    fwhm_minor: float
    roundness: float            # median FWHM_minor / FWHM_major (Siril's r; 1 = round)
    eccentricity: float         # median sqrt(1 - r^2) (PixInsight)
    beta: float                 # median Moffat beta
    n: int                      # stars kept
    tried: int                  # stars fitted (isolated and unsaturated candidates)
    radius: int                 # half-size of the fit window (pixels)
    channel: str                # "R", "G", "B" or "" (a plane given as it is)
    stars: dict = field(default_factory=dict, repr=False)   # per kept star: x, y, fwhm, fwhm_major, ... (arrays)

    @property
    def ok(self) -> bool:
        return self.n >= MIN_STARS

    def as_dict(self) -> dict:
        """The scalars, for a JSON file or a metric."""
        return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k != "stars"}


def _channel_index(channel) -> int:
    if isinstance(channel, str):
        return CHANNELS.index(channel.upper())
    return int(channel)


def _empty(radius: int, channel: str, tried: int = 0) -> PSF:
    nan = float("nan")
    return PSF(nan, nan, nan, nan, nan, nan, 0, tried, radius, channel, {})


def radius_for(fwhm: float) -> int:
    """Half-size of the fit window for stars of this FWHM: about two FWHM on each side, 5 to 30 pixels (at least 30
    samples of a red or blue CFA plane, eight parameters to fit)."""
    if not np.isfinite(fwhm):
        return 8
    return int(np.clip(np.ceil(2.0 * fwhm) + 1, 5, 30))


def measure_psf(image, xy, *, cfa: str | None = None, channel="G", saturation=None, radius: int | None = None,
                fwhm_hint: float | None = None, max_stars: int = 200, isolation: float | None = None) -> PSF:
    """FWHM, roundness and beta of the stars at ``xy`` by Moffat fits (see the module docstring).

    image      (H, W) plane, or (3, H, W) stack (``channel`` picks the plane). With ``cfa`` ("RGGB"...), ``image`` is a
               raw CFA frame and only the samples of ``channel`` are fitted, at their own positions.
    xy         (N, 2) star positions (x, y) in the image's pixels, brightest first (a catalogue order); about 1 px
               accuracy is enough, the fit finds the centre.
    saturation a level (a sample >= it is saturated) or a boolean mask of the image's shape (True = saturated or near
               it); a star with a saturated sample in its window is left out. NaN samples are ignored.
    radius     half-size of the fit window; by default from ``fwhm_hint`` or from a first pass on the 60 brightest stars.
    isolation  a star with another position of ``xy`` closer than this is left out (default: 2 x radius).
    """
    ci = _channel_index(channel)
    img = np.asarray(image)
    if img.ndim == 3:
        if cfa:
            raise ValueError("a CFA frame is one plane (H, W)")
        plane = img[ci]
    elif img.ndim == 2:
        plane = img
    else:
        raise ValueError(f"image of shape {img.shape}: expected (H, W) or (3, H, W)")
    name = CHANNELS[ci] if (cfa or img.ndim == 3) else ""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if radius is None:
        if fwhm_hint is not None and np.isfinite(fwhm_hint):
            radius = radius_for(fwhm_hint)
        else:                                    # a first pass on the brightest stars; a larger window for big stars
            radius = 8
            for r0 in (8, 16):
                first = _measure(plane, xy, cfa, ci, saturation, r0, min(60, max_stars), isolation, name)
                if first.n >= 5:
                    radius = radius_for(first.fwhm)
                    break
    return _measure(plane, xy, cfa, ci, saturation, int(radius), max_stars, isolation, name)


def measure_psf_rgb(image, xy, *, cfa: str | None = None, **kw) -> dict[str, PSF]:
    """``measure_psf`` for the three colours: of a CFA frame (``cfa``) or of a stack (3, H, W)."""
    return {c: measure_psf(image, xy, cfa=cfa, channel=c, **kw) for c in CHANNELS}


def find_stars(plane, *, cfa: str | None = None, k: float = 8.0, max_stars: int = 2000) -> np.ndarray:
    """Positions (N, 2) of the stars of a plane, brightest first: SExtractor's detection (``sep``) above ``k`` x the
    background noise, unflagged objects only. With ``cfa``, a raw CFA frame: detected on 2x2 super-pixels as
    register.py does, positions given in sensor pixels."""
    import sep
    a = np.asarray(plane, np.float32)
    if cfa:
        from .cfa import equalised
        a = equalised(a, cfa)                        # X-Trans: its colours levelled before the 2 x 2 cells
        H, W = a.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)           # a super-pixel of bad pixels only: NaN
            a = np.nanmean(a[: H - H % 2, : W - W % 2].reshape(H // 2, 2, W // 2, 2), axis=(1, 3))
    a = np.ascontiguousarray(a, dtype=np.float32)
    mask = ~np.isfinite(a)
    if mask.all():
        return np.zeros((0, 2))
    a = np.where(mask, np.nanmedian(a), a).astype(np.float32)
    bkg = sep.Background(a, mask=mask, bw=64, bh=64)
    x = a - bkg.back()
    sep.set_extract_pixstack(3_000_000)
    from .imgutil import sep_extract
    o = sep_extract(x, k, err=bkg.globalrms, minarea=4, mask=mask, deblend_cont=0.005)
    o = np.sort(o[o["flag"] == 0], order="flux")[::-1][:max_stars]
    xy = np.c_[o["x"], o["y"]].astype(float)
    return 2.0 * xy + 0.5 if cfa else xy


# ------------------------------------------------------------------------------------------------------ the fits
def _measure(plane, xy, cfa, ci, saturation, R, max_stars, isolation, name) -> PSF:
    from scipy.spatial import cKDTree

    H, W = plane.shape
    fin = np.isfinite(xy).all(1)
    if not fin.any():
        return _empty(R, name)
    xi = np.rint(xy[:, 0]).astype(np.int64)
    yi = np.rint(xy[:, 1]).astype(np.int64)
    inside = fin & (xi >= R) & (xi < W - R) & (yi >= R) & (yi < H - R)
    iso = 2.0 * R if isolation is None else float(isolation)
    tree = cKDTree(xy[fin])
    near = np.zeros(len(xy), np.int64)
    near[fin] = tree.query_ball_point(xy[fin], r=iso, return_length=True)
    cand = np.where(inside & (near <= 1))[0][:MAX_CANDIDATES]
    if len(cand) == 0:
        return _empty(R, name)

    d = np.arange(-R, R + 1)
    YY = yi[cand][:, None, None] + d[None, :, None]
    XX = xi[cand][:, None, None] + d[None, None, :]
    V = np.asarray(plane[YY, XX], np.float64)
    valid = np.isfinite(V)
    if cfa:
        from .cfa import channels
        g = channels(cfa)
        P = g.shape[0]
        valid &= g[YY % P, XX % P] == ci
    if saturation is not None:
        if np.ndim(saturation) == 0:
            sat = valid & (np.nan_to_num(V, nan=-np.inf) >= float(saturation))
        else:
            sat = valid & np.asarray(saturation, bool)[YY, XX]
        good = ~sat.any(axis=(1, 2))
    else:
        good = np.ones(len(cand), bool)
    S = 2 * R + 1
    n_par = 8
    good &= valid.reshape(len(cand), -1).sum(1) >= 3 * n_par
    ring = np.maximum(np.abs(d)[:, None], np.abs(d)[None, :]) >= R - 1     # two pixels wide: a CFA colour is in it
    inner = (np.abs(d)[:, None] <= max(2, R // 2)) & (np.abs(d)[None, :] <= max(2, R // 2))
    Vn = np.where(valid, V, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)               # an empty window: NaN, left out below
        b0 = np.nanmedian(np.where(ring[None], Vn, np.nan).reshape(len(cand), -1), axis=1)
        peak = np.nanmax(np.where(inner[None], Vn, np.nan).reshape(len(cand), -1), axis=1) - b0
    good &= np.isfinite(b0) & np.isfinite(peak) & (peak > 0)
    sel = np.where(good)[0]
    sel = sel[np.argsort(-peak[sel], kind="stable")][:max_stars]
    if len(sel) == 0:
        return _empty(R, name)
    V, valid, b0, peak = V[sel], valid[sel], b0[sel], peak[sel]
    cx, cy = xi[cand][sel], yi[cand][sel]
    n = len(sel)

    # first guesses: the background of the ring, the centre from the brightest sample, the width from the area above
    # half the peak (counted on the samples of the colour, so on a quarter of the pixels for red and blue)
    dx = np.broadcast_to(d[None, :], (S, S)).ravel().astype(np.float64)
    dy = np.broadcast_to(d[:, None], (S, S)).ravel().astype(np.float64)
    v = np.where(valid, V, 0.0).reshape(n, -1)
    m = valid.reshape(n, -1)
    above = np.where(m, v - b0[:, None], -np.inf)
    j = np.argmax(np.where(np.abs(dx)[None] + np.abs(dy)[None] <= R, above, -np.inf), axis=1)
    w = np.clip(above, 0, None) * ((dx[None] - dx[j][:, None]) ** 2 + (dy[None] - dy[j][:, None]) ** 2 <= 4.0)
    sw = w.sum(1)
    x0 = np.where(sw > 0, (w * dx[None]).sum(1) / np.maximum(sw, 1e-30), dx[j])
    y0 = np.where(sw > 0, (w * dy[None]).sum(1) / np.maximum(sw, 1e-30), dy[j])
    density = m.mean(1)
    n_half = ((above > 0.5 * peak[:, None]) & m).sum(1)
    f0 = np.clip(2.0 * np.sqrt(np.maximum(n_half, 1) / density / np.pi), 1.0, R)
    beta0 = 2.5
    alpha0 = f0 / (2.0 * np.sqrt(2.0 ** (1.0 / beta0) - 1.0))
    p0 = np.stack([b0, peak, x0, y0, 1 / alpha0 ** 2, np.zeros(n), 1 / alpha0 ** 2, np.full(n, beta0)], 1)

    p, cost = _levenberg_marquardt(v, m, dx, dy, p0)

    # the profile of each star
    B, A, X0, Y0, a, b, c, beta = p.T
    with np.errstate(all="ignore"):
        tr, det = a + c, a * c - b * b
        disc = np.sqrt(np.maximum((a - c) ** 2 / 4 + b * b, 0))
        lmax, lmin = tr / 2 + disc, tr / 2 - disc
        kf = 2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0)
        fmaj = kf / np.sqrt(lmin)
        fmin = kf / np.sqrt(lmax)
        rnd = fmin / fmaj
        ang = np.degrees(0.5 * np.arctan2(2 * b, a - c)) + 90.0      # the major axis (the smaller eigenvalue)
        ang = (ang + 90.0) % 180.0 - 90.0
        rmse = np.sqrt(cost / np.maximum(m.sum(1) - n_par, 1))
        rel = rmse / A
    keep = (np.isfinite(fmaj) & np.isfinite(fmin) & (A > 0) & (det > 0) & (lmin > 0) & (fmin > 0.5) & (fmaj <= R)
            & (rnd >= MIN_ROUNDNESS) & (beta >= MIN_BETA) & (rel <= MAX_REL_RMSE)
            & (np.abs(X0) <= R / 2) & (np.abs(Y0) <= R / 2))
    if not keep.any():
        return _empty(R, name, n)
    fw = np.sqrt(fmaj * fmin)
    stars = {"x": (cx + X0)[keep], "y": (cy + Y0)[keep], "fwhm": fw[keep], "fwhm_major": fmaj[keep],
             "fwhm_minor": fmin[keep], "roundness": rnd[keep], "angle": ang[keep], "beta": beta[keep],
             "amplitude": A[keep], "background": B[keep], "rmse": rmse[keep]}
    med = {k: float(np.median(stars[k])) for k in ("fwhm", "fwhm_major", "fwhm_minor", "roundness", "beta")}
    ecc = float(np.median(np.sqrt(np.clip(1 - stars["roundness"] ** 2, 0, 1))))
    return PSF(med["fwhm"], med["fwhm_major"], med["fwhm_minor"], med["roundness"], ecc, med["beta"], int(keep.sum()),
               n, R, name, stars)


def _model(p, dx, dy, jac: bool):
    """The Moffat profile of each star at the window's offsets (dx, dy), and its Jacobian (n, P, 8)."""
    B, A, x0, y0, a, b, c, beta = (p[:, k:k + 1] for k in range(8))
    X = dx[None] - x0
    Y = dy[None] - y0
    u = 1.0 + a * X * X + 2.0 * b * X * Y + c * Y * Y
    with np.errstate(all="ignore"):
        lu = np.log(u)
        pw = np.exp(-beta * lu)
    f = B + A * pw
    if not jac:
        return f, None
    g = -A * beta * pw / u                                         # d f / d u
    J = np.stack([np.ones_like(f), pw, -2.0 * g * (a * X + b * Y), -2.0 * g * (b * X + c * Y),
                  g * X * X, 2.0 * g * X * Y, g * Y * Y, -A * pw * lu], -1)
    return f, J


def _levenberg_marquardt(v, m, dx, dy, p0):
    """Least squares of every star at once (Marquardt's damping on the diagonal, one damping per star). Returns the
    parameters (n, 8) and the sums of squared residuals."""
    p = p0.copy()
    n = len(p)
    lam = np.full(n, 1e-3)
    f, J = _model(p, dx, dy, True)
    r = np.where(m, v - f, 0.0)
    cost = (r * r).sum(1)
    cost[~np.isfinite(cost)] = np.inf
    active = np.isfinite(cost)
    eye = np.eye(8)
    for _ in range(MAX_ITER):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        Jm = J[idx] * m[idx][..., None]
        JtJ = np.matmul(Jm.transpose(0, 2, 1), Jm)
        grad = np.matmul(Jm.transpose(0, 2, 1), r[idx][..., None])[..., 0]
        dg = np.maximum(np.diagonal(JtJ, axis1=1, axis2=2), 1e-12)
        aug = JtJ + (lam[idx][:, None] * dg)[:, :, None] * eye + 1e-12 * eye
        try:
            step = np.linalg.solve(aug, grad[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = np.stack([np.linalg.lstsq(aug[k], grad[k], rcond=None)[0] for k in range(len(idx))])
        pn = p[idx] + step
        pn[:, 7] = np.clip(pn[:, 7], 0.3, MAX_BETA)
        fn, _ = _model(pn, dx, dy, False)
        rn = np.where(m[idx], v[idx] - fn, 0.0)
        cn = (rn * rn).sum(1)
        ok = (pn[:, 1] > 0) & (pn[:, 4] > 0) & (pn[:, 6] > 0) & (pn[:, 4] * pn[:, 6] - pn[:, 5] ** 2 > 0)
        cn[~ok | ~np.isfinite(cn)] = np.inf
        better = cn < cost[idx]
        gain = np.where(better, (cost[idx] - cn) / np.maximum(cost[idx], 1e-300), 0.0)
        bi = idx[better]
        p[bi] = pn[better]
        cost[bi] = cn[better]
        lam[idx] = np.clip(np.where(better, lam[idx] * 0.3, lam[idx] * 5.0), 1e-10, 1e10)
        if len(bi):
            fb, Jb = _model(p[bi], dx, dy, True)
            J[bi] = Jb
            r[bi] = np.where(m[bi], v[bi] - fb, 0.0)
        done = (better & (gain < 1e-9)) | (lam[idx] >= 1e10)
        active[idx[done]] = False
    return p, cost
