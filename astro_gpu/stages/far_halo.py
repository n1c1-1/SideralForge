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
  catalogue flux); a star whose glare dominates them also its own elongation, orientation and radial slope.

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
    AtA = np.zeros((n, n)); Atb = np.zeros(n)
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


def far_halo(sl, x, y, F, r_in, deg=3, nknots=20, log=print):
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
    # full-resolution model: the same sum, evaluated per star on the image grid (coarse in r, so done at 1/2)
    h2, w2 = H // 2, W // 2
    Y2, X2 = (np.mgrid[0:h2, 0:w2] + 0.5) * 2 - 0.5
    out = np.zeros((3, h2, w2), np.float32)
    for i in range(n):
        re = ell_r(i, X2, Y2, Q[i], PHI[i])
        prof = np.interp(np.log(re), lk, S, right=0.0).astype(np.float32) * pw(re)
        if BETA[i]:
            prof *= ((re / RREF[i]) ** BETA[i]).astype(np.float32)
        for c in range(3):
            out[c] += prof * np.float32(kap[i] * F[c, i])
    full = np.stack([cv2.resize(out[c], (W, H), interpolation=cv2.INTER_LINEAR) for c in range(3)])
    info = {"far_halo_axis_ratio": round(q, 3), "far_halo_angle_deg": round(float(np.degrees(phi)), 1),
            "far_halo_profile": {str(int(round(rk[j]))): round(float(S[j] * pw(rk[j])), 5) for j in range(0, nr + 1, 2)},
            "far_halo_own": {str(i): [round(float(kap[i]), 2), round(float(Q[i]), 2), round(float(np.degrees(PHI[i]))),
                                      round(float(BETA[i]), 2)] for i in strong},
            "far_halo_own_amplitude": {str(i): round(float(kap[i]), 2) for i in measured},
            "far_halo_kept_pct": round(100 * float(np.mean([mm.mean() for mm in m])), 1)}
    log(f"far halo of the stars: shape measured on {n} stars out to {rmax:.0f} px, elongation {q:.2f} at "
        f"{np.degrees(phi):.0f} deg; {len(strong)} stars dominating their surroundings with their own amplitude and "
        f"elongation, {len(measured)} with their own amplitude")
    return full, info
