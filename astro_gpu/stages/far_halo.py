"""The far halo of the stars: light scattered by the optics, the same for every star, measured once on the image.

Far from its core, every star is surrounded by a faint glare (scattering by the mirror, the sensor window, dust on
the optics). Around a bright star it spreads over a large part of the field: Alnitak lights a quarter of the
Horsehead frame. It is part of the star, proportional to its flux, so it is modelled as such:

* one radial shape S(r) for the whole instrument, continuous, falling at least as 1/r (1/r times a sum of
  non-negative ramps in log r), fitted jointly on every bright star of the image together with a smooth sky
  (low-order polynomial), robustly (nebula and residuals are clipped as outliers); each star contributes its flux
  times S, in its own colour (the glare of a blue star is blue: scattering by the optics is achromatic at this
  precision; shapes per channel took the red emission for glare);
* the glare may be elongated (it is, on the Newtonian of the Horsehead): the shape is a function of an elliptical
  radius whose axis ratio and angle are measured first, from the second angular harmonic of the same fit;
* inside the radius that the star model already describes (its wings, or its saturated core), the halo is
  continued flat: what is left there is the level of the halo at that radius;
* a star whose halo is measurable against its surroundings gets its own amplitude (a variable star, a wrong
  catalogue flux); a star whose glare dominates them also its own elongation, orientation and radial slope;
* optionally, no halo exceeds the light the data show around its star beyond the nebula (a nebula only adds
  light): per channel, radius and direction, the halo is lowered to the lower envelope of the data around the star
  (see _envelope). Least squares had let a nebula rising toward its star pull the halo above the data on the other
  side: a dark ring around sigma Orionis, dark lobes beside Alnitak; and a star fainter than its catalogue flux (a
  variable) kept a halo above the data: a dark disc.

There is no cut at a fixed radius and no local reference level, which drew a dark disc in a bright ring around
every bright star (the 'eye' of sigma Orionis).
"""
from __future__ import annotations

import cv2
import numpy as np
from numpy.polynomial import legendre as L
from scipy.optimize import lsq_linear

WING_OUT = 60.0      # px, the star model of the stars stage includes the wings out to this radius
ALPHA_MIN = 1.0      # light scattered by optics falls at least as 1/r (measured: r^-1.5 to r^-1.9 on a Newtonian);
                     # a shape allowed to level off took the dust around the stars for their glare (Iris)


ENVELOPE_Q = 0.2     # the lower part of the star's surroundings that bounds its halo (see _envelope)
ENVELOPE_SNR = 10.0  # a halo is bounded only where it stands this many times above the noise of its surroundings


def _ell_polar(X, Y, xc, yc, q, phi):
    """Elliptical radius and angle (0..2 pi) of the points X, Y around (xc, yc), axis ratio q, major axis at phi."""
    cp, sp = np.cos(phi), np.sin(phi)
    dx, dy = X - xc, Y - yc
    a, b = dx * cp + dy * sp, -dx * sp + dy * cp
    return np.sqrt(a * a / q + b * b * q), np.arctan2(b * np.sqrt(q), a / np.sqrt(q)) % (2 * np.pi)


def _envelope(rest, unit, Fi, sig, ok, D, xc, yc, q_, phi_, r0, rmax, q=ENVELOPE_Q, nr=64, nt=72, win_r=3, win_t=11,
              snr_min=ENVELOPE_SNR):
    """Factors (<= 1, per channel, radius and angle) that keep a star's halo under the light the data show around it
    beyond the nebula.

    A nebula only adds light: the halo of a star cannot be brighter than the darker part of its surroundings. The
    fits treat the nebula as noise (and the amplitude of a dominant star is fitted with a free local level), so a
    nebula that rises toward its star - sigma Orionis in IC 434, the Flame beside Alnitak - pulled the halo above the
    data on the dark side: dark rings and lobes around the star once the halo is removed. Around the star, on a grid
    of its elliptical radius (log) and angle: the data divided by the model (what the data allow of this halo at each
    point), per channel; its lower ``q`` over a window of win_r radii x win_t angles (~55 deg: a nebula that fills
    less of it does not count) with its noise added back, at most 1; smoothed; then the largest halo under it that
    does not rise outward, along each direction. Per channel: one factor for the three (the strictest channel) left
    blue glare around sigma Orionis where the red bound it; the three channels together left Alnitak's red
    over-subtracted. Per direction: one factor per ring lowered all of Alnitak's ring for a lobe on one side (the
    Flame on the other had pulled the elongation of its halo). A channel where the halo is too faint to be bounded
    takes the factor of the others (the halo of a very red star is measurable in red only).

    Only where the halo stands well above the darker side of its surroundings (snr_min x sig): fainter, the bound
    followed the dark patches of the sky and cut the halo in wedges. Beyond, the factor of the last such radius,
    the same in every direction (its median), reached smoothly (the halo still does not rise outward).

    rest (3, h, w): data minus sky minus the other stars' halos, on the coarse grid (pixels of D px); unit (h, w):
    this star's halo per unit flux; Fi (3,): its flux; sig (3,): how far below its local median the background of a
    coarse pixel goes (noise and dark structure); ok (h, w): usable pixels;
    (xc, yc, q_, phi_): the star and the ellipse of its halo; r0, rmax: the radii bounded.
    Returns the factors {'lr0', 'dlr', 'f' (3, nr, nt)} on the grid, or None where the halo stands nowhere above the
    noise."""
    from statistics import NormalDist
    z = -NormalDist().inv_cdf(q)
    Fi, sig = np.asarray(Fi, float), np.asarray(sig, float)
    if rmax <= r0 * 1.2:
        return None
    lr = np.linspace(np.log(r0), np.log(rmax), nr)
    t = np.arange(nt) * (2 * np.pi / nt)
    re = np.exp(lr)[:, None]
    a, b = re * np.sqrt(q_) * np.cos(t)[None], re / np.sqrt(q_) * np.sin(t)[None]
    cp, sp = np.cos(phi_), np.sin(phi_)
    mx = ((xc + a * cp - b * sp + 0.5) / D - 0.5).astype(np.float32)
    my = ((yc + a * sp + b * cp + 0.5) / D - 0.5).astype(np.float32)

    def polar(img, fill):
        return cv2.remap(np.ascontiguousarray(img, np.float32), mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    up = polar(unit, np.nan)
    good = (polar(ok.astype(np.float32), 0.0) > 0.99) & np.isfinite(up) & (up > 0)
    urow = np.array([np.median(up[k][good[k]]) if good[k].any() else np.nan for k in range(nr)])
    if not np.isfinite(urow).any():
        return None
    # windows: win_r radii x win_t angles (periodic)
    ir = np.clip(np.arange(nr)[:, None] + np.arange(-(win_r // 2), win_r // 2 + 1)[None], 0, nr - 1)
    it = (np.arange(nt)[:, None] + np.arange(-(win_t // 2), win_t // 2 + 1)[None]) % nt
    f = np.ones((3, nr, nt))
    measured = []
    for c in range(3):
        snr = urow * Fi[c] / sig[c]
        sig_rows = np.isfinite(snr) & (snr >= snr_min)
        if not sig_rows.any():
            continue
        ratio = np.where(good, polar(rest[c], np.nan) / (Fi[c] * np.where(good, up, 1.0)), np.nan)
        win = ratio[ir[:, None, :, None], it[None, :, None, :]].reshape(nr, nt, -1)
        cnt = np.isfinite(win).sum(-1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lim = np.nanquantile(win, q, axis=-1) + z / np.maximum(snr, 1e-9)[:, None]
        lim[(cnt < max(5, win.shape[-1] // 3)) | ~sig_rows[:, None]] = np.nan
        # cells without a measure: the median of their ring; inside the first measured ring, that ring; beyond the
        # last one, its median in every direction
        for k in range(nr):
            if np.isnan(lim[k]).all():
                continue
            lim[k][np.isnan(lim[k])] = np.nanmedian(lim[k])
        rows = np.where(~np.isnan(lim).all(1))[0]
        if not len(rows):
            continue
        lim[:rows[0]] = lim[rows[0]]
        lim[rows[-1] + 1:] = np.median(lim[rows[-1]])
        for k in range(rows[0], rows[-1]):
            if np.isnan(lim[k]).all():
                lim[k] = lim[rows[rows <= k][-1]]
        fc = np.clip(lim, 0.0, 1.0).astype(np.float32)
        pad = 6
        fc = cv2.GaussianBlur(np.concatenate([fc[:, -pad:], fc, fc[:, :pad]], 1), (0, 0), sigmaX=1.5, sigmaY=1.0,
                              borderType=cv2.BORDER_REPLICATE)[:, pad:-pad]
        f[c] = fc
        measured.append(c)
    if not measured:
        return None
    # a channel too faint to bound takes the factor of the channels bounded: the glare has the colour of its star,
    # and a wrong catalogue flux (a variable star) is wrong in every channel. Left at 1, the green and blue halos of
    # a very red star (T Cephei) stayed over-subtracted around it once its red was bounded: a red disc
    if len(measured) < 3:
        fm = f[measured].min(0)
        for c in range(3):
            if c not in measured:
                f[c] = fm
    u_ = np.where(np.isfinite(urow), urow, np.nan)
    u_ = np.interp(np.arange(nr), np.where(np.isfinite(u_))[0], u_[np.isfinite(u_)])[None, :, None]
    f = np.minimum.accumulate(u_ * f, axis=1) / np.maximum(u_, 1e-30)   # never rising outward, in each direction
    live = u_[0, :, 0] > 1e-3 * u_.max()               # where the shape has ended, the halo is nothing anyway
    if not live.all() and live.any():
        f[:, ~live] = f[:, np.where(live)[0][-1]][:, None]
    return {"lr0": float(lr[0]), "dlr": float(lr[1] - lr[0]), "q": float(q_), "phi": float(phi_),
            "f": np.clip(f, 0.0, 1.0).astype(np.float32)}


def _fac_at(fac, X, Y, xc, yc):
    """The factors of _envelope at the points X, Y (3,) + X.shape (inside the first radius: the first ring's)."""
    shp = np.shape(X)
    re, t = _ell_polar(np.ravel(X), np.ravel(Y), xc, yc, fac["q"], fac["phi"])
    nr, nt = fac["f"].shape[1:]
    npx, wd = re.size, 4096                                     # remap wants images under 32767 px a side
    pad = (-npx) % wd
    my = np.clip((np.log(np.maximum(re, 1e-3)) - fac["lr0"]) / fac["dlr"], 0, nr - 1).astype(np.float32)
    mx = (t / (2 * np.pi) * nt).astype(np.float32)
    my, mx = np.r_[my, np.zeros(pad, np.float32)].reshape(-1, wd), np.r_[mx, np.zeros(pad, np.float32)].reshape(-1, wd)
    out = []
    for c in range(3):
        fc = np.concatenate([fac["f"][c], fac["f"][c][:, :1]], 1)          # periodic in angle
        v = cv2.remap(np.ascontiguousarray(fc), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out.append(v.ravel()[:npx].reshape(shp))
    return np.stack(out)


def _ramps(r, lk):
    """(n, ...) non-increasing ramps: 1 inside knot j, 0 beyond knot j + 1, linear in log r in between."""
    lr = np.log(np.maximum(r, np.exp(lk[0])))
    return np.clip((lk[1:, None] - lr.ravel()[None]) / (lk[1:] - lk[:-1])[:, None], 0, 1).reshape(
        (len(lk) - 1,) + r.shape).astype(np.float32)


def _hats(r, lk):
    """(n, ...) hat functions on the knots (log r), for the angular harmonic."""
    lr = np.log(np.maximum(r, 1.0)).ravel()
    out = np.zeros((len(lk), lr.size), np.float32)
    for j in range(len(lk)):
        lo = lk[j - 1] if j else lk[0] - 0.3
        hi = lk[j + 1] if j + 1 < len(lk) else lk[-1] + 1e-6
        out[j] = np.clip(np.minimum((lr - lo) / (lk[j] - lo), (hi - lr) / (hi - lk[j])), 0, 1)
    return out.reshape((len(lk),) + r.shape)


def _solve(Pf, B, yv, m, nonneg):
    """Joint least squares over the channels: halo coefficients shared, background per channel; m: rows kept per
    channel; nonneg: number of leading halo coefficients bounded below by 0. Normal equations (small system)."""
    nP, nB = Pf.shape[1], B.shape[0]
    n = nP + 3 * nB
    AtA = np.zeros((n, n))
    Atb = np.zeros(n)
    for c in range(3):
        mc = m[c]
        sig = 1.4826 * np.median(np.abs(yv[c][mc] - np.median(yv[c][mc]))) + 1e-9
        A = np.concatenate([Pf[c][:, mc], B[:, mc]]).T.astype(np.float64) / sig
        idx = np.r_[np.arange(nP), nP + c * nB + np.arange(nB)]
        AtA[np.ix_(idx, idx)] += A.T @ A
        Atb[idx] += A.T @ (yv[c][mc] / sig)
    d = np.sqrt(np.diag(AtA)) + 1e-12
    M = AtA / d[:, None] / d[None]
    R = np.linalg.cholesky(M + 1e-10 * np.eye(n)).T
    z = np.linalg.solve(R.T, Atb / d)
    lb = np.r_[np.zeros(nonneg), -np.inf * np.ones(n - nonneg)]
    return lsq_linear(R, z, bounds=(lb, np.inf)).x / d


def _robust(Pf, B, yv, use, nonneg, iters=8, clip=2.5):
    nP, nB = Pf.shape[1], B.shape[0]
    m = [use.copy() for _ in range(3)]
    for _ in range(iters):
        s = _solve(Pf, B, yv, m, nonneg)
        new = []
        for c in range(3):
            res = yv[c] - Pf[c].T @ s[:nP] - B.T @ s[nP + c * nB:nP + (c + 1) * nB]
            sd = 1.4826 * np.median(np.abs(res[m[c]] - np.median(res[m[c]])))
            new.append(use & (np.abs(res - np.median(res[m[c]])) < clip * sd))
        if all((a == b).all() for a, b in zip(new, m)):
            break
        m = new
    return s, m


def _amp_fit(unit, Fi, rest, wr, iters=4, clip=2.5, trim=0.8):
    """Amplitude of a star's halo (unit shape x its flux per channel) plus one level per channel, each pixel weighted
    by wr (the same weight for every ring in log r: the shape is modelled in log r, and the outer rings would
    otherwise outweigh the inner ones by their area); robust (the nebula next to the star - the Flame beside
    Alnitak - is clipped). Returns (amplitude, score): the score is the mean of the best explained ``trim`` of the
    squared residuals, the same measure for every candidate shape (pixels kept by a fit of one candidate would
    have been chosen against the others)."""
    nr_ = len(unit)
    sw = np.tile(np.sqrt(wr), 3)
    X = np.zeros((3 * nr_, 4))
    X[:, 0] = (Fi[:, None] * unit[None]).ravel()
    for c in range(3):
        X[c * nr_:(c + 1) * nr_, 1 + c] = 1
    X *= sw[:, None]
    yv = rest.ravel() * sw
    keep = np.ones(len(yv), bool)
    for _ in range(iters):
        sol = np.linalg.lstsq(X[keep], yv[keep], rcond=None)[0]
        res = yv - X @ sol
        sd = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-12
        keep = np.abs(res - np.median(res[keep])) < clip * sd
    r2 = np.sort(res ** 2)
    return float(sol[0]), float(r2[:int(trim * len(r2))].mean())


def far_halo(sl, x, y, F, r_in, deg=3, nknots=20, envelope=True, log=print):
    """Model of the far halo of the stars (C, H, W) to subtract from the starless image ``sl``, and metrics.

    x, y: positions of the bright stars; F (3, n): their fluxes (camera units, the colour of each star); r_in (n):
    the radius inside which the star model already holds the star (wings, filled core): the halo is flat inside.
    """
    C, H, W = sl.shape
    n = len(x)
    if n < 3:
        return None, {"far_halo": "fewer than 3 bright stars"}
    D = max(2, int(round(np.sqrt(H * W / 3.5e5))))     # the halo is smooth beyond 1.3 WING_OUT: ~8 px sampling
    h, w = H // D, W // D
    img = np.stack([cv2.resize(np.nan_to_num(np.asarray(sl[c], np.float32)), (w, h), interpolation=cv2.INTER_AREA)
                    for c in range(3)])
    yy, xx = (np.mgrid[0:h, 0:w] + 0.5) * D - 0.5
    rmax = 0.4 * np.hypot(W, H)
    # the shape starts where it is measured: beyond the region excluded around the stars the model holds
    lk = np.log(np.geomspace(1.3 * WING_OUT, rmax, nknots + 1))
    F = np.asarray(F, np.float64) / 1e6
    r_in = np.maximum(np.asarray(r_in, float), WING_OUT)
    use = np.ones((h, w), bool)
    for i in range(n):                               # what the star model holds is not used
        use &= np.hypot(xx - x[i], yy - y[i]) > 1.3 * r_in[i]
    use = use.ravel()
    u, v = xx / W * 2 - 1, yy / H * 2 - 1
    B = np.stack([L.legval(u, np.eye(deg + 1)[a]) * L.legval(v, np.eye(deg + 1)[b])
                  for a in range(deg + 1) for b in range(deg + 1 - a)]).reshape(-1, h * w)
    yv = img.reshape(3, -1).astype(np.float64)
    nr = nknots

    Q, PHI = np.ones(n), np.zeros(n)                 # elongation of each star's halo (the instrument's by default)
    r_first = float(np.exp(lk[0]))

    def pw(r):
        """the shape is (r / r_first)^-ALPHA_MIN times a non-increasing function: it falls at least that fast"""
        return ((r / r_first) ** -ALPHA_MIN).astype(np.float32)
    BETA, RREF = np.zeros(n), np.ones(n)             # radial slope of its own, relative to the instrument's shape

    def ell_r(i, X_, Y_, q, phi):
        cp, sp = np.cos(phi), np.sin(phi)
        dx, dy = X_ - x[i], Y_ - y[i]
        a, b = dx * cp + dy * sp, -dx * sp + dy * cp
        return np.maximum(np.sqrt(a * a / q + b * b * q), r_in[i])

    def star_basis(i, harmonic=False):
        """(nb, h w) shape functions of star i (unit flux)."""
        r = ell_r(i, xx, yy, Q[i], PHI[i])
        R_ = _ramps(r, lk) * pw(r)[None]
        if BETA[i]:
            R_ = R_ * ((r / RREF[i]) ** BETA[i]).astype(np.float32)[None]
        if harmonic:
            t = np.arctan2(yy - y[i], xx - x[i])
            Hh = _hats(r, lk) * pw(r)[None]
            R_ = np.concatenate([R_, Hh * np.cos(2 * t)[None], Hh * np.sin(2 * t)[None]])
        return R_.reshape(len(R_), -1)

    def basis(kap, harmonic=False):
        P = None
        for i in range(n):
            R_ = star_basis(i, harmonic)
            if P is None:
                P = np.zeros((3,) + R_.shape, np.float32)
            for c in range(3):
                P[c] += R_ * np.float32(kap[i] * F[c, i])
        return P

    kap = np.ones(n)
    # 1) elongation: second angular harmonic of a round fit, where the halo is well measured
    s, _ = _robust(basis(kap, True), B, yv, use, nr)
    S0 = np.r_[np.cumsum(s[:nr][::-1])[::-1], 0.0]
    c2, s2 = s[nr:2 * nr + 1], s[2 * nr + 1:3 * nr + 2]
    rk = np.exp(lk)
    slope = -np.gradient(np.log(np.maximum(S0, 1e-12)), lk) + ALPHA_MIN
    # where the glare dominates: beyond the spikes and the residuals of the cores, while still well above zero
    good = (S0 * pw(rk) > 0.02 * S0[0]) & (rk > 2.5 * WING_OUT) & (rk < 0.25 * rmax)
    q, phi = 1.0, 0.0
    if good.sum() >= 3:
        wgt = S0[good] * pw(rk[good]) * rk[good] ** 2      # the light of the halo in each ring
        A2 = np.sum(wgt * c2[good]) / np.sum(wgt * S0[good])
        B2 = np.sum(wgt * s2[good]) / np.sum(wgt * S0[good])
        e2 = np.hypot(A2, B2)                               # amplitude of cos 2(t - phi) relative to the round part
        alpha = float(np.median(np.clip(slope[good], 0.5, 4)))
        eps = min(2 * e2 / alpha, 0.6)                      # S(r_e) ~ S(r) (1 + alpha eps/2 cos 2(t - phi))
        phi = 0.5 * np.arctan2(B2, A2)
        q = 1 + eps
    Q[:], PHI[:] = q, phi
    # 2) the shape on the elliptical radius
    Pf = basis(kap)
    s, m = _robust(Pf, B, yv, use, nr)
    # 3) the stars whose glare dominates their surroundings: their own amplitude, elongation and orientation
    model = np.stack([Pf[c].T @ s[:nr] for c in range(3)])
    Sk = np.r_[np.cumsum(s[:nr][::-1])[::-1], 0.0]
    strong, measured = [], []
    for i in range(n):
        r = np.hypot(xx - x[i], yy - y[i]).ravel()
        ring = (r > 1.3 * r_in[i]) & (r < 4 * r_in[i]) & use
        if ring.sum() < 50:
            continue
        re0 = ell_r(i, xx.ravel()[ring], yy.ravel()[ring], q, phi)
        unit = np.interp(np.log(re0), lk, Sk, right=0.0) * pw(re0)
        mi = F[:, i][:, None] * unit[None]                  # this star's halo in the ring, per channel
        rest = yv[:, ring] - model[:, ring] + mi            # its halo + its surroundings
        # what can be measured of a star's halo depends on how much it varies across the ring compared with what
        # is left around it (nebula, neighbours). Lost in its surroundings: the instrument's shape at the catalogue
        # flux. Measurable: its own amplitude, which may be anything from none to a little more (a variable star -
        # T Cephei by the Iris, fainter than its Gaia magnitude - would otherwise lose a dark disc). Dominant: its
        # own amplitude, elongation, orientation and radial slope
        spread = 1.4826 * np.median(np.abs(rest[1] - mi[1] - np.median(rest[1] - mi[1])))
        contrast = (np.percentile(mi[1], 95) - np.percentile(mi[1], 5)) / max(spread, 1e-9)
        if contrast < 3:
            continue
        xr, yr = xx.ravel()[ring], yy.ravel()[ring]
        rr = np.hypot(xr - x[i], yr - y[i])
        r_ref = float(np.median(rr))
        wr = (r_ref / rr) ** 2
        if contrast < 10:
            k, _ = _amp_fit(unit, F[:, i], rest, wr)
            kap[i] = float(np.clip(k, 0.0, 1.5))
            measured.append(i)
            continue
        best = None
        for qi in (1.0, 1.1, 1.2, 1.3, 1.45, 1.6):
            for ph in (np.radians(np.arange(-80, 100, 10)) if qi > 1 else [0.0]):
                re = ell_r(i, xr, yr, qi, ph)
                u0 = np.interp(np.log(re), lk, Sk, right=0.0) * pw(re)
                for beta in (-0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3):
                    k, sc = _amp_fit(u0 * (re / r_ref) ** beta, F[:, i], rest, wr)
                    if best is None or sc < best[0]:
                        best = (sc, k, qi, ph, beta)
        _, k, Q[i], PHI[i], BETA[i] = best
        RREF[i] = r_ref
        kap[i] = float(np.clip(k, 0.0, 2.0))
        strong.append(i)
    if strong or measured:
        Pf = basis(kap)
        s, m = _robust(Pf, B, yv, use, nr)
    S = np.r_[np.cumsum(s[:nr][::-1])[::-1], 0.0]
    # 4) no halo above the light the data show beyond the nebula (see _envelope), the brightest halos first, for
    # every star: the halos fitted on their own surroundings (a nebula can pull them up) and the halos of the stars
    # lost in their surroundings, drawn at their catalogue flux. Those had no bound: a variable star fainter than its
    # Gaia magnitude (T Cephei by the Iris, on 300 frames) kept a halo 4 times the light around it - a dark disc
    fac = [None] * n
    if envelope:
        nB = B.shape[0]
        rest = yv - np.stack([B.T @ s[nr + c * nB:nr + (c + 1) * nB] for c in range(3)])
        peak = np.zeros(n)                               # each halo's highest level per unit flux
        for i in range(n):
            u_ = (star_basis(i).T @ s[:nr]).astype(np.float32)
            rest -= kap[i] * F[:, i][:, None] * u_[None]
            peak[i] = float(u_.max())
        # pixels with data (a frame edge without coverage reads as a deficit of light)
        cov = cv2.resize((np.isfinite(sl[1]) & (np.asarray(sl[1]) != 0)).astype(np.float32), (w, h),
                         interpolation=cv2.INTER_AREA).ravel()
        ok = (use & np.isfinite(yv).all(0) & (cov > 0.999)).reshape(h, w)
        # how far below its local median the background goes (noise and dark structure: dust, the sky's own
        # errors), per channel: the median over blocks of 16 x 16 coarse pixels of their median minus their 16th
        # percentile (the nebula, only above, does not count)
        sig = np.zeros(3)
        r3 = rest.reshape(3, h, w)
        for c in range(3):
            lows = []
            for by in range(0, h - 15, 16):
                for bx in range(0, w - 15, 16):
                    m_ = ok[by:by + 16, bx:bx + 16]
                    if m_.mean() > 0.5:
                        v_ = r3[c, by:by + 16, bx:bx + 16][m_]
                        lows.append(np.percentile(v_, 50) - np.percentile(v_, 16))
            sig[c] = float(np.median(lows)) if lows else 1.0
        # only a halo that can stand above the noise of its surroundings can be bounded (see _envelope)
        cand = [i for i in range(n) if (kap[i] * F[:, i] * peak[i] >= ENVELOPE_SNR * sig).any()]
        for i in sorted(cand, key=lambda j: -kap[j] * F[1, j]):
            u_ = (star_basis(i).T @ s[:nr]).astype(np.float32)
            hi = kap[i] * F[:, i][:, None] * u_[None]
            rest += hi                                   # this star's halo and its surroundings
            env = _envelope(rest.reshape(3, h, w), u_.reshape(h, w), kap[i] * F[:, i], sig, ok, D, x[i], y[i],
                            Q[i], PHI[i], 1.3 * r_in[i], rmax)
            if env is not None and env["f"].min() < 0.999:
                fac[i] = env
                hi = hi * _fac_at(env, xx.ravel(), yy.ravel(), x[i], y[i])
            rest -= hi
        del rest, r3
    # per star: the lowest factor of each channel, its radius
    bounded = {}
    for i in range(n):
        if fac[i] is not None:
            fmin = fac[i]["f"].min(axis=(1, 2))
            k = np.unravel_index(np.argmin(fac[i]["f"].min(0)), fac[i]["f"].shape[1:])[0]
            bounded[i] = [round(float(v), 2) for v in fmin] + [int(round(np.exp(fac[i]["lr0"] + k * fac[i]["dlr"])))]
    # full-resolution model: the same sum, evaluated per star on the image grid (coarse in r, so done at 1/2)
    h2, w2 = H // 2, W // 2
    Y2, X2 = (np.mgrid[0:h2, 0:w2] + 0.5) * 2 - 0.5
    out = np.zeros((3, h2, w2), np.float32)
    dom = np.zeros((3, h2, w2), np.float32)          # the halo of the stars that dominate their surroundings
    for i in range(n):
        re = ell_r(i, X2, Y2, Q[i], PHI[i])
        prof = np.interp(np.log(re), lk, S, right=0.0).astype(np.float32) * pw(re)
        if BETA[i]:
            prof *= ((re / RREF[i]) ** BETA[i]).astype(np.float32)
        fi = None if fac[i] is None else _fac_at(fac[i], X2, Y2, x[i], y[i])
        for c in range(3):
            pc = prof if fi is None else prof * fi[c]
            out[c] += pc * np.float32(kap[i] * F[c, i])
            if i in strong:
                dom[c] += pc * np.float32(kap[i] * F[c, i])
    full = np.stack([cv2.resize(out[c], (W, H), interpolation=cv2.INTER_LINEAR) for c in range(3)])
    info = {"far_halo_axis_ratio": round(q, 3), "far_halo_angle_deg": round(float(np.degrees(phi)), 1),
            "far_halo_profile": {str(int(round(rk[j]))): round(float(S[j] * pw(rk[j])), 5) for j in range(0, nr + 1, 2)},
            "far_halo_own": {str(i): [round(float(kap[i]), 2), round(float(Q[i]), 2), round(float(np.degrees(PHI[i]))),
                                      round(float(BETA[i]), 2)] for i in strong},
            "far_halo_own_amplitude": {str(i): round(float(kap[i]), 2) for i in measured},
            "far_halo_kept_pct": round(100 * float(np.mean([mm.mean() for mm in m])), 1),
            "far_halo_bounded": {str(i): list(v) for i, v in bounded.items()},
            "_dominant": np.stack([cv2.resize(dom[c], (W, H), interpolation=cv2.INTER_LINEAR) for c in range(3)])
            if strong else None,
            "_fit": dict(x=np.asarray(x, float), y=np.asarray(y, float), F=F, kap=kap, Q=Q, PHI=PHI, BETA=BETA,
                         RREF=RREF, r_in=r_in, lk=lk, S=S, r_first=r_first, strong=strong, measured=measured,
                         sky=np.asarray(s[nr:], float).reshape(3, -1), deg=deg, W=W, H=H, fac=fac)}
    log(f"far halo of the stars: shape measured on {n} stars out to {rmax:.0f} px, elongation {q:.2f} at "
        f"{np.degrees(phi):.0f} deg; {len(strong)} stars dominating their surroundings with their own amplitude and "
        f"elongation, {len(measured)} with their own amplitude; {len(bounded)} halos lowered to the light the data "
        f"show around them" + (f" (down to {min(min(v[:3]) for v in bounded.values()):.2f} x)" if bounded else ""))
    return full, info


def star_halo(fit, i, X, Y):
    """The far halo of star i of a fit (``info['_fit']`` of far_halo) at the points X, Y: (3,) + X.shape."""
    x, y, lk, S = fit["x"], fit["y"], fit["lk"], fit["S"]
    q, phi = fit["Q"][i], fit["PHI"][i]
    cp, sp = np.cos(phi), np.sin(phi)
    dx, dy = X - x[i], Y - y[i]
    a, b = dx * cp + dy * sp, -dx * sp + dy * cp
    re = np.maximum(np.sqrt(a * a / q + b * b * q), fit["r_in"][i])
    prof = np.interp(np.log(re), lk, S, right=0.0) * (re / fit["r_first"]) ** -ALPHA_MIN
    if fit["BETA"][i]:
        prof = prof * (re / fit["RREF"][i]) ** fit["BETA"][i]
    fac = fit.get("fac")
    fi = None if fac is None or fac[i] is None else _fac_at(fac[i], X, Y, x[i], y[i])
    return np.stack([prof * fit["kap"][i] * fit["F"][c, i] * (1.0 if fi is None else fi[c])
                     for c in range(3)]).astype(np.float32)


def sky_of(fit, X, Y):
    """The smooth sky of a fit at the points X, Y: (3,) + X.shape."""
    deg = fit["deg"]
    u, v = X / fit["W"] * 2 - 1, Y / fit["H"] * 2 - 1
    B = [L.legval(u, np.eye(deg + 1)[a]) * L.legval(v, np.eye(deg + 1)[b])
         for a in range(deg + 1) for b in range(deg + 1 - a)]
    return np.stack([sum(fit["sky"][c, k] * B[k] for k in range(len(B))) for c in range(3)]).astype(np.float32)


def own_rays(res, x, y, r_in, halo, fac=2.0, across=12.0, rmax=1200, nt=1440):
    """The fine rays of the stars whose glare dominates their surroundings: star light that neither the shared star
    shape (spikes) nor the far halo holds, and that the star removal smeared into blotches stretched as nebula.

    A ray is radial: around its star it keeps the same angle over a long range of radius, and it is narrow across.
    So, on the residual image (stars and far halo out) around each such star, in polar coordinates: each ring minus
    its median (what the halo leaves, broad), the running median along the radius over a factor ``fac`` (what keeps
    its angle that long: structures of the nebula, the neighbouring stars do not), minus the running median along
    the arc over +-``across`` px (what is broader across than a ray goes), per channel (the rays are chromatic).
    What is not above twice its own noise is left out. Inside ``r_in`` (the star model) nothing is taken. The rays
    are part of the star's light, so they follow its glare (``halo``, the far halo image): their typical contrast to
    it is measured between r_in and 2 r_in, and farther out only what that contrast makes significant is kept, never
    more than four times it (beyond, the structure of the nebula passed for rays: radial streaks far from the star).
    Returns the rays (C, H, W), to be added to the far halo (removed from the nebula, drawn back as glare)."""
    from scipy.ndimage import median_filter
    C, H, W = res.shape
    out = np.zeros((C, H, W), np.float32)
    dth = 2 * np.pi / nt
    hp = res[1, ::3, ::3] - cv2.GaussianBlur(res[1, ::3, ::3], (0, 0), 2)
    sig = 1.4826 * float(np.nanmedian(np.abs(hp - np.nanmedian(hp))))
    for i in range(len(x)):
        R = int(rmax)
        r0 = float(r_in[i])
        if R <= 2 * r0:
            continue
        rg = np.arange(R, dtype=np.float32)[:, None]
        th = (np.arange(nt) * dth).astype(np.float32)[None]
        px, py = (x[i] + rg * np.cos(th)).astype(np.float32), (y[i] + rg * np.sin(th)).astype(np.float32)
        lg = np.linspace(np.log(r0), np.log(R - 1), 400)
        win = max(3, int(round(np.log(fac) / (lg[1] - lg[0]))) | 1)
        r_lin = np.arange(R, dtype=np.float64)
        E = np.zeros((C, R, nt), np.float32)
        h_r = [np.nan_to_num(np.nanmedian(cv2.remap(np.ascontiguousarray(halo[c]), px, py, cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan), 1))[:, None]
               for c in range(C)]                             # the glare of the star along the radius
        kappa = None
        for c in (1, 0, 2):
            v = cv2.remap(np.ascontiguousarray(res[c]), px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=np.nan)
            v = v - np.nanmedian(v, 1, keepdims=True)
            v = np.nan_to_num(v)
            A = np.stack([np.interp(lg, np.log(np.maximum(r_lin, 1)), v[:, j]) for j in range(nt)], 1)
            A = median_filter(A, size=(win, 1), mode="nearest")
            B = np.zeros((R, nt), np.float32)
            k0 = int(np.ceil(r0))
            B[k0:] = np.stack([np.interp(np.log(r_lin[k0:]), lg, A[:, j]) for j in range(nt)], 1)
            for k in range(k0, R):                         # arc high-pass: narrower across than +-across px
                nb = int(round(across / (k * dth)))
                if nb < nt // 2:
                    row = B[k]
                    ext = np.concatenate([row[-nb:], row, row[:nb]])
                    B[k] = row - median_filter(ext, size=2 * nb + 1, mode="nearest")[nb:nb + nt]
            # noise of a median of the ``win`` samples of the radial window (one interpolated pixel each), and a smooth
            # significance gate (a looser bound let the noise far from the star through: dark radial streaks)
            s_e = 1.25 * sig / np.sqrt(win)
            if kappa is None:                              # contrast of the rays to the glare, near the star (green)
                near = slice(int(np.ceil(r0)), min(int(2 * r0), R))
                kappa = float(np.median(np.abs(B[near]) / np.maximum(h_r[c][near], 1e-6)))
            expect = kappa * h_r[c]
            gate = np.clip((np.abs(B) / s_e - 2.0) / 2.0, 0, 1) * np.clip((expect / s_e - 2.0) / 2.0, 0, 1)
            B = np.clip(B, -4 * expect, 4 * expect)
            ramp = (np.clip((r_lin - r0) / 10.0, 0, 1) * np.clip((R - r_lin) / (0.2 * R), 0, 1))[:, None]
            E[c] = B * gate * ramp
        x0, x1 = max(0, int(x[i]) - R), min(W, int(x[i]) + R + 2)
        y0, y1 = max(0, int(y[i]) - R), min(H, int(y[i]) + R + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = (xx - x[i]).astype(np.float32), (yy - y[i]).astype(np.float32)
        r = np.hypot(dx, dy)
        t = ((np.arctan2(dy, dx) % (2 * np.pi)) / dth).astype(np.float32)
        w = (r < R - 1).astype(np.float32)
        for c in range(C):
            out[c, y0:y1, x0:x1] += w * cv2.remap(E[c], t, r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return out
