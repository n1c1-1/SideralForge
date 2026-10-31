"""The angular part of the star shape (diamond core, diffraction spikes, secondary rays), measured once on the image.

A star is its flux times the shape the optics give it. The round part of that shape is the fitted PSF and its radial
wings; what is not round - the diamond of a Newtonian's star, its diffraction spikes, the rays of the mirror clips - is
measured here, the same for every star:

* around each bright star the residual image (image minus the star model) is resampled on a polar grid (1 px in
  radius, 0.25 degree in angle); each ring minus its median over the angle keeps the angular structure only (the
  level of the nebula and any round halo go);
* per unit flux, the stars are combined with weights flux^2, capped at the 5th brightest star reaching each radius so
  that no single star (and the nebula around it) dominates, with sigma clipping;
* the shape is smoothed along the radius in proportion to it and along the arc at the pixel scale: far from the
  core a point spread function is smooth along the radius, nebula and noise are not radial;
* the stars are split into two halves measured independently: the shape of the instrument is what every star shares,
  so only what both halves show is kept; structure broader across than three spike widths is nebula, not a spike;
  what is not significant is dropped.

The flux scale is calibrated on the unsaturated stars (photometry); the flux of a saturated star, which neither its
core nor the catalogue gives reliably (a variable or very red star, a core flattened by saturation), is measured on
its own spikes, which are not saturated.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter

RMAX = 600            # px, longest spike measured
NT = 1440             # angular samples (0.25 degree)
DTH = 2 * np.pi / NT
R_IN = 8              # the shape model starts here (inside, the local core templates hold the star shape)
CORE_R = 60.0         # px, core of a bright star (its model and its residual dominate there)


def _polar(a, xc, yc, R, fill=np.nan):
    rg = np.arange(R, dtype=np.float32)[:, None]
    th = (np.arange(NT) * DTH).astype(np.float32)[None]
    px = (xc + rg * np.cos(th)).astype(np.float32)
    py = (yc + rg * np.sin(th)).astype(np.float32)
    return cv2.remap(a, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill), px, py


def _smooth(E, sr=0.05, sa=0.7):
    """Smoothing along the radius (sigma ``sr`` x r) and along the arc (sigma ``sa`` px)."""
    out = E.copy()
    r4 = np.arange(4, RMAX, dtype=np.float64)
    lg = np.linspace(np.log(4), np.log(RMAX - 1), 500)
    A = np.stack([np.interp(lg, np.log(r4), E[4:, j]) for j in range(NT)], 1).astype(np.float32)
    A = cv2.GaussianBlur(A, (1, 0), sigmaX=1e-3, sigmaY=sr / (lg[1] - lg[0]), borderType=cv2.BORDER_REPLICATE)
    B = np.stack([np.interp(np.log(r4), lg, A[:, j]) for j in range(NT)], 1)
    for k, r in enumerate(r4):
        s = sa / (r * DTH)
        if s > 0.3:
            row = np.tile(B[k], 3)[None].astype(np.float32)
            B[k] = cv2.GaussianBlur(row, (0, 1), sigmaX=s, sigmaY=1e-3)[0, NT:2 * NT]
    out[4:] = B
    return out


def _arc_highpass(E, L):
    """At each radius, minus the running median over an arc of +-L px: what is broader across than that goes."""
    out = E.copy()
    for k in range(R_IN, E.shape[0]):
        nb = int(round(L / (k * DTH)))
        row = E[k]
        if nb >= NT // 2 or not np.isfinite(row).all():
            continue
        ext = np.concatenate([row[-nb:], row, row[:nb]])
        out[k] = row - median_filter(ext, size=2 * nb + 1, mode="nearest")[nb:nb + NT]
    return out


def _combine(stk, members, Rs, Fw, sig_c, c, ref=None):
    """Weighted, clipped mean per unit flux of the polar samples of ``members`` -> (mean, noise)."""
    Rarr = np.array([Rs[i] for i in members])
    Farr = np.array([Fw[i] for i in members])
    cap = np.ones(RMAX)
    for r in range(RMAX):
        f = np.sort(Farr[Rarr > r])
        cap[r] = f[-min(5, len(f))] if len(f) else 1.0
    num, den, var = np.zeros((RMAX, NT)), np.zeros((RMAX, NT)), np.zeros((RMAX, NT))
    for i in members:
        v, fl = stk[i][c]
        R = v.shape[0]
        vv = v / fl
        s_k = sig_c / fl
        ok = np.isfinite(vv)
        if ref is not None:
            ok &= np.abs(vv - ref[:R]) < 3 * s_k + 0.3 * np.abs(ref[:R])
        w = np.where(ok, np.minimum(Fw[i], cap[:R, None]) ** 2, 0.0)
        num[:R] += np.nan_to_num(vv) * w
        den[:R] += w
        var[:R] += w ** 2 * s_k ** 2
    E = num / np.maximum(den, 1e-30)
    S = np.where(den > 0, np.sqrt(var) / np.maximum(den, 1e-30), np.inf)
    return E, S


def _fit_amplitude(v, e, s_min=100):
    """Robust (Huber) least-squares amplitude of the template ``e`` in the samples ``v`` on the template's support."""
    emax = e.max(1, keepdims=True)
    sup = (e > 0.2 * np.maximum(emax, 1e-30)) & np.isfinite(v)
    if sup.sum() < s_min:
        return np.nan, 0.0
    vs, es = v[sup], e[sup]
    a = float((vs * es).sum() / (es ** 2).sum())
    for _ in range(5):
        r = vs - a * es
        s = 1.4826 * np.median(np.abs(r)) + 1e-12
        w = 1.0 / np.maximum(1.0, np.abs(r) / (2.0 * s))
        a = float((w * vs * es).sum() / (w * es ** 2).sum())
    s = 1.4826 * np.median(np.abs(vs - a * es)) + 1e-12
    return a, a * np.sqrt((es ** 2).sum()) / s


def measure(res, img, x, y, g, flux, pred, log=print):
    """Angular star shape (C, RMAX, NT) per unit flux, the flux of every star for that shape (C, N) and metrics;
    (None, None, info) when too few bright stars. ``g`` is shifted by the setup's magnitude offset."""
    C, H, W = res.shape
    info = {}
    sig = []
    for c in range(C):
        hp = res[c] - cv2.GaussianBlur(res[c], (0, 0), 2)
        sig.append(float(1.4826 * np.median(np.abs(hp[::7, ::7]))))
    sat = []
    for c in range(C):
        lvl = float(np.percentile(img[c][::4, ::4], 99.999))
        sat.append(cv2.dilate((img[c] > 0.7 * lvl).astype(np.uint8), np.ones((7, 7), np.uint8)))
    xi = np.clip(np.round(x).astype(int), 0, W - 1)
    yi = np.clip(np.round(y).astype(int), 0, H - 1)
    issat = np.zeros(len(x), bool)
    for c in range(C):
        issat |= sat[c][yi, xi] > 0
    calm = (g > 10.5) & (g < 12.5) & ~issat & (pred[1] > 0)
    if calm.sum() < 10:
        return None, None, {"angular_shape": "too few stars"}
    cal = np.array([np.median(flux[c, calm] / np.maximum(pred[c, calm], 1e-9)) for c in range(C)])
    fl = np.where(issat[None], pred * cal[:, None], flux).astype(np.float64)
    sel = np.where((g < 11.0) & (fl[1] > 0))[0]
    # a star in the core of a much brighter one has no shape of its own there: its rings, per unit of its flux, carry
    # the residual of the bright core multiplied by the ratio of the fluxes (the companions of sigma Orionis spoilt
    # the half of the stars they fell in, and the cross-check then rejected the shape)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[x, y])
    sel = np.array([i for i in sel if not any(j != i and g[j] < g[i] - 1.5
                                               for j in tree.query_ball_point([x[i], y[i]], CORE_R))], int)
    if len(sel) < 10:
        return None, None, {"angular_shape": "too few bright stars"}
    ref = np.median(flux[1][(g > 10.8) & (g < 11.2) & ~issat]) if ((g > 10.8) & (g < 11.2) & ~issat).any() \
        else np.median(fl[1, sel])
    Rs = {int(i): int(np.clip(60 * np.sqrt(max(fl[1, i], 1.0) / max(ref, 1e-9)), 40, RMAX)) for i in sel}
    stk = {}
    for i in sel:
        i = int(i)
        row = []
        for c in range(C):
            v, px, py = _polar(res[c], x[i], y[i], Rs[i])
            m = cv2.remap(sat[c].astype(np.float32), px, py, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=1) > 0
            v[m] = np.nan
            row.append([(v - np.nanmedian(v, 1, keepdims=True)).astype(np.float32), fl[c, i]])
        stk[i] = row
    sel = [int(i) for i in sel]

    def build():
        Fw = {i: stk[i][1][1] for i in sel}
        order = sorted(sel, key=lambda i: -Fw[i])
        halves = [order[0::2], order[1::2]]
        # one shape for the three channels, measured on green (the best signal): a diffraction pattern is the same
        # in every colour at this sampling (spike widths measured equal), so the shape takes the colour of its star;
        # separate shapes per channel kept different noise in each and drew coloured streaks
        Eh, Sh = [], []
        for hm in halves:
            E0, _ = _combine(stk, hm, Rs, Fw, sig[1], 1)
            E1, _ = _combine(stk, hm, Rs, Fw, sig[1], 1, E0)
            E2, S2 = _combine(stk, hm, Rs, Fw, sig[1], 1, E1)
            nwin = np.maximum(2.5 * 0.05 * np.arange(RMAX), 1.0) * 1.75
            Eh.append(_smooth(E2.astype(np.float32))[None])
            Sh.append((S2 / np.sqrt(nwin)[:, None])[None])
        # a spike is a line: at each radius, structure broader across than three spike widths is not star shape
        prof = np.nanmean(0.5 * (Eh[0][0] + Eh[1][0])[20:40], 0)
        j0 = int(np.argmax(prof))
        arc = (np.arange(NT) - j0 + NT // 2) % NT - NT // 2
        cut = prof > 0.5 * prof[j0]
        width = float(30 * DTH * max(1, int(np.sum(cut & (np.abs(arc) < 60)))))
        hA, hB = (np.nan_to_num(_arc_highpass(Eh[k][0], 3.0 * width)) for k in (0, 1))
        # noise of one half: the pixel noise, or - what really limits it - how much the two halves differ (their
        # neighbouring stars and nebula differ), averaged around each point
        s_emp = np.sqrt(np.clip(_smooth(0.5 * (hA - hB) ** 2, 0.10, 3.0), 0, None))
        Sn = np.maximum(np.maximum(Sh[0][0], Sh[1][0]), s_emp)
        # where the shape is: both halves show it above twice their noise (soft edge, widened by a pixel of arc so
        # that a spike is not cut along its sides); its value there: the mean of the halves (the smaller of the two
        # would be biased low and thin the faint ends of the spikes, which then stayed in the nebula)
        sup = np.clip((np.minimum(hA, hB) / np.maximum(Sn, 1e-30) - 1.5) / 1.0, 0, 1).astype(np.float32)
        for k in range(R_IN, RMAX):
            nb = max(1, int(round(1.0 / (k * DTH))))
            sup[k] = np.max(np.stack([np.roll(sup[k], d) for d in range(-nb, nb + 1)]), 0)
        E = (np.clip(0.5 * (hA + hB), 0, None) * sup).astype(np.float32)
        E[:R_IN] = 0
        # is it a shape the stars share? Cross-validation: the shape measured on one half must be found in the
        # stars of the other half - the significance (in standard errors) of its amplitude there, at their fluxes
        cv = []
        for h_own, S_own, other in ((hA, Sh[0][0], halves[1]), (hB, Sh[1][0], halves[0])):
            T = np.clip(h_own, 0, None) * np.clip((h_own / np.maximum(S_own, 1e-30) - 2.0) / 1.0, 0, 1)
            T[:12] = 0
            vs, ts = [], []
            for i in other:
                v = stk[i][1][0][:min(150, Rs[i])]
                t = T[:v.shape[0]] * Fw[i]
                ok = np.isfinite(v) & (t > 0)
                if ok.sum():
                    vs.append(v[ok])
                    ts.append(t[ok])
            if not vs:
                cv.append(0.0)
                continue
            v_, t_ = np.concatenate(vs).astype(np.float64), np.concatenate(ts).astype(np.float64)
            a_ = float((v_ * t_).sum() / (t_ ** 2).sum())
            s_ = float(np.sqrt(np.mean((v_ - a_ * t_) ** 2) / (t_ ** 2).sum()))
            cv.append(a_ / max(s_, 1e-30))
        return np.repeat(E[None], C, 0), width, cv

    def fit(E, i, c):
        v = stk[i][c][0][:min(150, Rs[i])].copy()
        v[:12] = np.nan
        return _fit_amplitude(v, E[c][:v.shape[0]])

    E, width, cv = build()
    info["angular_shape_cross_check_sigma"] = [round(v, 1) for v in cv]
    if min(cv) < 5.0:
        log(f"star shape: no angular shape shared by the stars (the shape measured on one half of the bright stars "
            f"is not found in the other half: amplitude {min(cv):.1f} standard errors): round stars only")
        info["angular_shape"] = "none"
        return None, None, info
    # flux scale: the unsaturated stars measured with S/N > 20 on their own shape must match their photometry
    ratios = []
    for i in sel:
        if not issat[i]:
            a, snr = fit(E, i, 1)
            if np.isfinite(a) and snr > 20:
                ratios.append(a / fl[1, i])
    scale = float(np.median(ratios)) if len(ratios) >= 5 else 1.0
    E *= scale
    # saturated stars: flux measured on their own shape, then the shape measured again with those fluxes
    n_sat = 0
    for i in sel:
        if issat[i]:
            for c in range(C):
                a, snr = fit(E, i, c)
                if np.isfinite(a) and snr > 20 and a > 0:
                    stk[i][c][1] = a
                    n_sat += c == 1
    E, width, cv = build()
    info["angular_shape_cross_check_sigma"] = [round(v, 1) for v in cv]
    ratios = []
    for i in sel:
        if not issat[i]:
            a, snr = fit(E, i, 1)
            if np.isfinite(a) and snr > 20:
                ratios.append(a / fl[1, i])
    scale2 = float(np.median(ratios)) if len(ratios) >= 5 else 1.0
    if not 1 / 3 <= scale * scale2 <= 3:
        # the shape is not the stars' own light: its amplitude has to be multiplied by thousands to match their
        # photometry (40 frames of the Heart, a refractor without spikes: x2196, 59 % of the light "beyond 12 px",
        # and drawing every star out to hundreds of pixels never ended). The user's projects: x0.96-1.28
        log(f"star shape: the angular shape found needs a flux scale of {scale * scale2:.3g} to match the stars' "
            f"photometry (about 1 expected): it is not the stars' light, round stars only")
        info["angular_shape"] = "none"
        info["angular_flux_scale_rejected"] = round(scale * scale2, 3)
        return None, None, info
    E *= scale2
    fshape = flux.astype(np.float64).copy()
    for i in np.where(issat)[0]:
        fshape[:, i] = fl[:, i]
    for i in sel:
        if issat[i]:
            for c in range(C):
                a, snr = fit(E, i, c)
                fshape[c, i] = a if (np.isfinite(a) and snr > 20 and a > 0) else fl[c, i]
    rr = np.arange(RMAX)[:, None]
    info["angular_shape_stars"] = len(sel)
    info["angular_shape_saturated_measured"] = int(n_sat)
    info["spike_width_px"] = round(width, 1)
    info["angular_light_pct"] = [round(100 * float((E[c][12:] * rr[12:] * DTH).sum()), 3) for c in range(C)]
    info["angular_flux_scale"] = round(scale * scale2, 3)
    info["angular_calibration_spread"] = round(float(1.4826 * np.median(np.abs(np.array(ratios) - np.median(ratios))))
                                               / max(abs(float(np.median(ratios))), 1e-9), 2) if ratios else None
    log(f"star shape: angular part (diamond, spikes) measured on {len(sel)} bright stars, spike width "
        f"{width:.1f} px, {info['angular_light_pct'][1]:.2f} % of a star's light beyond 12 px; the shape of each half "
        f"is found at {min(cv):.0f} standard errors in the other half; flux of {n_sat} saturated stars measured on "
        f"their spikes")
    return E, fshape.astype(np.float32), info


def core_angular(P):
    """Angular part (polar, clipped >= 0) of a centred core template P (C, K, K), measured on green and used for every
    channel like the outer shape -> (C, K // 2 + 1, NT)."""
    C, K, _ = P.shape
    R = K // 2 + 1
    v, _, _ = _polar(P[1].astype(np.float32), K // 2, K // 2, R, fill=0.0)
    return np.repeat(np.clip(v - np.median(v, 1, keepdims=True), 0, None)[None], C, 0).astype(np.float32)


def render(E, x, y, f, H, W, sig, r_from=None, ramp=3.0, out=None, frac=0.05, idx=None):
    """Sum over the stars of flux x angular shape (C, H, W), each star out to where its shape falls below ``frac``
    of the noise ``sig`` (green). ``r_from``: per-star radius where the shape starts (ramp of ``ramp`` px)."""
    C = E.shape[0]
    out = np.zeros((C, H, W), np.float32) if out is None else out
    emax = E[1].max(1)
    idx = np.arange(len(x)) if idx is None else idx
    n = 0
    for i in idx:
        vis = f[1, i] * emax > frac * sig
        if not vis.any() or f[1, i] <= 0:
            continue
        R = int(np.where(vis)[0].max()) + 1
        x0, x1 = max(0, int(x[i]) - R), min(W, int(x[i]) + R + 2)
        y0, y1 = max(0, int(y[i]) - R), min(H, int(y[i]) + R + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = (xx - x[i]).astype(np.float32), (yy - y[i]).astype(np.float32)
        r = np.hypot(dx, dy)
        t = ((np.arctan2(dy, dx) % (2 * np.pi)) / DTH).astype(np.float32)
        w = (r < min(R, RMAX - 1)).astype(np.float32)
        if r_from is not None:
            w *= np.clip((r - r_from[i]) / ramp, 0, 1)
        for c in range(C):
            v = cv2.remap(E[c], t, r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)   # the angle wraps around
            out[c, y0:y1, x0:x1] += f[c, i] * v * w
        n += 1
    return out, n
