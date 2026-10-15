"""Learned star removal network: the library behind the 'star_net' stage (recipe "run E" of the own-data experiment).

A U-Net learns to remove the stars of ONE stack, from that stack alone:

* input  = the stack minus the measured far halo (glare) of the stars, with real bright-star stamps of the same stack
           pasted at random places (scaled, re-saturated at their own plateau, with their photon noise); in half of
           the patches the patch's own nebula is amplified 1-40 times (bright nebula is rare away from bright stars);
* target = the model-based starless image + the input's own noise (the network removes stars, it does not denoise),
           trusted only away from the bright stars and outside the cores of the real stars.

A smooth glow centred on half of the pasted bright stars goes into the input and the target alike: a nebula peaked
on its star is nebula (the network had taken the bright core of NGC 2023, around a G 7.8 star drawn at rendering,
for star light).

Everything is measured on the project's own data: background and noise, photon gain, saturation plateaus, how far
the stars' light reaches (halo and spikes), two hold-out regions (never training targets, they measure the network),
the colour of the frame's brightest extended nebula.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter

from ..core.stage import StageError
from .denoise import UNet

K_ASINH = 4.0                             # asinh softening, in noise sigmas
WIDTHS = (32, 64, 128, 192, 256, 320)     # 5 down-samplings: receptive field ~ 400 px
PATCH = 384


# ------------------------------------------------------------------------------------------------ star extent

def radius_of(g, fit):
    """Radius (px) reached by the light of a star of Gaia magnitude g: the law measured on the frame's bright stars
    (R = exp(a + b G), 12..600 px); without a measurement (fewer than 5 usable bright stars) a fixed law
    (40 px at G 11 to 250 px at G 6, 12..400 px)."""
    g = np.asarray(g, float)
    if fit is None:
        return np.clip(40.0 * 6.25 ** ((11.0 - g) / 5.0), 12, 400)
    a, b = fit
    return np.clip(np.exp(a + b * g), 12, 600)


def taper(R):
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    r = np.hypot(yy, xx) / R
    w = np.where(r <= 0.6, 1.0, 0.5 * (1 + np.cos(np.pi * np.clip((r - 0.6) / 0.4, 0, 1))))
    return w.astype(np.float32), r


class Difference:
    """a - b, computed window by window: Difference(a, b)[sl] == a[sl] - b[sl] (float32)."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __getitem__(self, sl):
        return np.asarray(self.a[sl], np.float32) - np.asarray(self.b[sl], np.float32)


def measure_extent(SL, sg, x, y, g, dirs, H, W, rmax=1000, nmax=60):
    """Extent of the light of the brightest stars (G < 11, no brighter star within 100 px), measured on the star
    layer SL (indexable as SL[:, y0:y1, x0:x1]) = denoised stack - far halo - starless: the radius where the ring
    median (relative to its level in the outer third of the window), and the mean over the spikes (measured
    directions) of the median along each spike minus the median 6-14 px beside it, fall below 0.25 noise sigma
    (RGB sum, 2 px Gaussian, 5-ring running median, 4 px rings). Returns {index: (r_halo, r_spikes)}."""
    out = {}
    for i in np.argsort(g):
        if g[i] >= 11 or len(out) >= nmax:
            break
        xi, yi = int(round(x[i])), int(round(y[i]))
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        br = g < g[i]
        if br.any() and np.hypot(x[br] - x[i], y[br] - y[i]).min() < 100:
            continue
        y0, y1, x0, x1 = max(0, yi - rmax), min(H, yi + rmax + 1), max(0, xi - rmax), min(W, xi + rmax + 1)
        src = np.asarray(SL[:, y0:y1, x0:x1], np.float32)
        s = cv2.GaussianBlur((src / sg[:, None, None]).sum(0) / math.sqrt(3), (0, 0), 2)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - x[i], yy - y[i]
        r = np.hypot(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx)) % 360
        nb = int(min(r.max(), rmax) // 4)
        if nb < 8:
            continue
        lab = np.zeros(r.shape, np.int8)      # 1+k on spike k, -(1+k) beside it, 0 elsewhere
        for k, t in enumerate(dirs):
            da = np.abs((ang - t + 180) % 360 - 180)
            perp = r * np.sin(np.radians(np.minimum(da, 90)))
            near = da < 30
            lab[near & (perp <= 2.5)] = k + 1
            lab[near & (perp >= 6) & (perp <= 14)] = -(k + 1)
        rb = (r // 4).astype(int).ravel()
        order = np.argsort(rb, kind="stable")
        fs, ss, ls = rb[order], s.ravel()[order], lab.ravel()[order]
        starts = np.searchsorted(fs, np.arange(nb + 1))
        med = np.zeros(nb)
        spk = np.zeros(nb)
        for kb in range(3, nb):
            seg, lb = ss[starts[kb]:starts[kb + 1]], ls[starts[kb]:starts[kb + 1]]
            if seg.size < 20:
                continue
            med[kb] = np.median(seg[lb == 0]) if (lb == 0).sum() > 10 else np.median(seg)
            ex = [np.median(seg[lb == k + 1]) - np.median(seg[lb == -(k + 1)]) for k in range(len(dirs))
                  if (lb == k + 1).sum() >= 4 and (lb == -(k + 1)).sum() >= 8]
            if ex:
                spk[kb] = np.mean(ex)

        def first_below(v, thr=0.25):
            v = median_filter(v, size=5, mode="nearest")
            b = np.where(v[3:] < thr)[0]
            return 4 * (int(b[0]) + 3) if b.size else 4 * len(v)
        # the halo against the layer's own level far out: the layer holds unresolved light and the starless image's
        # offsets, which do not belong to this star
        ref = np.median(med[max(4, 2 * nb // 3):]) if nb > 12 else 0.0
        out[int(i)] = (first_below(med - ref), first_below(spk) if len(dirs) else 0)
    return out


def fit_extent(ext, g):
    """R = 1.15 x max(halo, spikes) radius; robust log-linear fit of R against G. Returns (a, b) or None."""
    idx = np.array(list(ext.keys()), int)
    if idx.size < 5:
        return None
    G = g[idx]
    R = 1.15 * np.array([max(v) for v in ext.values()], float)
    ok = R > 12
    if ok.sum() < 5:
        return None
    G, R = G[ok], np.log(R[ok])
    keep = np.ones(G.size, bool)
    for _ in range(3):
        b, a = np.polyfit(G[keep], R[keep], 1)
        res = R - (a + b * G)
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
        keep = np.abs(res) < 2.5 * s
        if keep.sum() < 5:
            return None
    if b >= 0:
        return None
    return float(a), float(b)


# ------------------------------------------------------------------------------------------- measurements

def noise_stats(S, A, B, L):
    """Sky level (median of the starless image), pixel noise of the stack (from the half-stacks), saturation level
    (99.999th percentile of the stack), per channel."""
    bk = np.array([np.median(np.asarray(S[c, ::4, ::4])) for c in range(3)], np.float32)
    sg = []
    for c in range(3):
        n = (np.asarray(A[c, ::4, ::4], np.float32) - np.asarray(B[c, ::4, ::4], np.float32)) / 2
        sg.append(1.4826 * np.median(np.abs(n - np.median(n))))
    sat99 = np.array([np.percentile(np.asarray(L[c, ::2, ::2]), 99.999) for c in range(3)], np.float32)
    return bk, np.maximum(np.array(sg, np.float32), 1e-6), sat99


def photon_gain(NL, S, A, B, x, y, g, H, W, log=print):
    """Photon noise gain: var((A - B) / 2) = a + b s + c s^2 around G 10-13 stars (s = star light); b is the variance
    per ADU of star light given to the pasted stars (the c term is the PSF difference between the half-stacks)."""
    from scipy.optimize import nnls
    sel = np.where((g > 10) & (g < 13) & (x > 50) & (y > 50) & (x < W - 50) & (y < H - 50))[0][:1500]
    gain = []
    for c in range(3):
        ss, vv = [], []
        for i in sel:
            xi, yi = int(round(x[i])), int(round(y[i]))
            s_ = (c, slice(yi - 6, yi + 7), slice(xi - 6, xi + 7))
            ss.append((np.asarray(NL[s_], np.float32) - np.asarray(S[s_], np.float32)).ravel())
            vv.append((((np.asarray(A[s_], np.float32) - np.asarray(B[s_], np.float32)) / 2) ** 2).ravel())
        if not ss:
            gain.append(0.03)
            continue
        s, v = np.concatenate(ss), np.concatenate(vv)
        m = s < 2500
        coef = nnls(np.c_[np.ones(m.sum()), s[m], s[m] ** 2], v[m])[0]
        gain.append(max(float(coef[1]), 1e-4))
    log(f"photon noise gain (variance per ADU of star light, R G B): {np.round(gain, 4).tolist()}")
    return np.array(gain, np.float32)


def fill_radii(ext, g, fit):
    """Radius of the area each star dominates: its measured extent (1.15 x the larger of its halo and spike radii)
    when it was measured, else the fitted law."""
    out = radius_of(g, fit).astype(float)
    for i, v in ext.items():
        out[i] = max(12.0, 1.15 * max(v))
    return out


def stamp_library(NLh, S, L, x, y, g, R_all, sat99, H, W, seed=0, nfaint=2000, fill=None, filled=None):
    """Real star stamps cut from the stack: star layer (denoised stack - far halo - background) around every star
    G < 11 with no brighter star within 60 px ("bright"), and up to `nfaint` isolated stars G 11-15 ("faint");
    the median of the outer ring removed, cosine taper, negatives clipped. Saturated cores are flagged with their own
    plateau level (re-saturation when pasted).

    Background: the starless image S, except for the bright stamps when `fill` (radius per star, see fill_radii) and
    `filled` (S's filled cores, bool (H, W)) are given. A bright stamp is then the star as the input shows it,
    without the nebula, inside its fill radius (the window ends 25 %, at least 16 px, beyond it):
      - interpolated background: membrane fill of S from the ring just outside the radius;
      - bright part: light above the higher of that background and S (a model-based nebula estimate) outside S's
        filled cores (dilated 4 px), above the interpolated background inside them. S's holes under the star do not
        become "star light", and a nebula that S keeps (a reflection nebula around its star) stays out of the stamp;
      - dark part: where the input is below the interpolated background at a pixel and at its mirror through the
        star, the smaller darkness of the two. The far-halo subtraction leaves point-symmetric dark rings and lobes
        around the brightest stars; a network that never saw them in a stamp dug craters there. Nebula lanes, which
        are not point-symmetric, stay out of the stamp.
    Faint stars inside a stamp stay in it. A stamp is saturated as the rendering sees it (3 x 3 max above 0.7 x the
    99.999th percentile of the frame's 4 x 4 subsample)."""
    from ..core.imgutil import membrane_fill
    lvl = np.array([np.percentile(np.asarray(L[c, ::4, ::4]), 99.999) for c in range(L.shape[0])])
    bright, faint = [], []
    for i in np.argsort(g):
        if g[i] >= 15:
            break
        xi, yi = int(round(x[i])), int(round(y[i]))
        inside = fill is not None and filled is not None and g[i] < 11
        R = int(round(fill[i] + max(16.0, 0.25 * fill[i]))) if inside else int(R_all[i])
        Re = min(R, xi, yi, W - 1 - xi, H - 1 - yi)
        if Re < 0.75 * R or Re < 10:
            continue
        brighter = g < g[i]
        dd = np.hypot(x[brighter] - x[i], y[brighter] - y[i])
        if dd.size and dd.min() < (60 if g[i] < 11 else 30):
            continue
        if g[i] >= 11:      # faint stamps must not hold part of a bright star
            bb = g < 11
            if np.any(np.hypot(x[bb] - x[i], y[bb] - y[i]) < R_all[bb] + Re):
                continue
        sl = (slice(None), slice(yi - Re, yi + Re + 1), slice(xi - Re, xi + Re + 1))
        w, r = taper(Re)
        nl, s_ = np.asarray(NLh[sl], np.float32), np.asarray(S[sl], np.float32)
        if inside:
            hole = r * Re < min(fill[i], 0.85 * Re)
            bg = np.stack([membrane_fill(s_[c], hole) for c in range(3)])
            ring = (r >= 0.85) & (r <= 1.0)
            ped = np.median((nl - bg)[:, ring], axis=1)[:, None, None]
            core = cv2.dilate(np.asarray(filled[sl[1:]]).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            pos = np.clip(nl - np.where(core[None], bg, np.maximum(bg, s_)) - ped, 0, None)
            neg = np.clip(nl - bg - ped, None, 0)
            neg = np.where(hole[None], np.maximum(neg, neg[:, ::-1, ::-1]), 0)
            st = ((pos + neg) * w).astype(np.float32)
        else:
            win = nl - s_
            ring = (r >= 0.85) & (r <= 1.0)
            ped = np.median(win[:, ring], axis=1)
            st = np.clip((win - ped[:, None, None]) * w, 0, None).astype(np.float32)
        core = np.asarray(L[:, yi - 4:yi + 5, xi - 4:xi + 5]).reshape(3, -1).max(1)
        # saturated as the rendering sees it (3 x 3 max above 0.7 x the 99.999th percentile of the frame): above
        # 0.75 x it, Alnitak's plateau and most of the Rosette's saturated stars passed for unsaturated
        sat_ = np.asarray(L[:, yi - 1:yi + 2, xi - 1:xi + 2]).reshape(3, -1).max(1) > 0.7 * lvl
        rec = dict(x=float(x[i]), y=float(y[i]), g=float(g[i]), R=Re, sat=sat_,
                   plateau=core.astype(np.float32), t=st.astype(np.float16) if g[i] >= 11 else st)
        (bright if g[i] < 11 else faint).append(rec)
    if len(faint) > nfaint:
        rng = np.random.default_rng(seed)
        faint = [faint[k] for k in sorted(rng.choice(len(faint), nfaint, replace=False))]
    return {"bright": bright, "faint": faint}


def down4(a):
    """(C, H, W) -> (C, H/4, W/4) block means."""
    c, h, w = a.shape
    h4, w4 = h // 4, w // 4
    return np.asarray(a[:, :h4 * 4, :w4 * 4], np.float32).reshape(c, h4, 4, w4, 4).mean((2, 4))


def box_scores(lum, zone, size, step, margin):
    """Mean, std of lum and zone fraction over every size x size box (top-left on a step grid), via integral images."""
    def integ(a):
        return np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    I1, I2, IZ = integ(lum.astype(np.float64)), integ(lum.astype(np.float64) ** 2), integ(zone.astype(np.float64))
    h, w = lum.shape
    ys = np.arange(margin, h - size - margin + 1, step)
    xs = np.arange(margin, w - size - margin + 1, step)
    Y, X = np.meshgrid(ys, xs, indexing="ij")

    def bsum(I):
        return I[Y + size, X + size] - I[Y, X + size] - I[Y + size, X] + I[Y, X]
    n = size * size
    m = bsum(I1) / n
    sd = np.sqrt(np.maximum(bsum(I2) / n - m ** 2, 0))
    return Y, X, m, sd, bsum(IZ) / n


def reliability(x, y, g, R_all, H, W):
    """Where the starless image is trusted as a target: not within max(150 px, 1.2 R) of the stars brighter than
    G 10, nor within 1.2 R of the G 10-10.5 stars, nor at the frame edge. Returns the weight map (H, W) uint8 and the
    bright-star zones at 1/4 scale (30 px wider)."""
    wm = np.ones((H, W), np.uint8)
    zone4 = np.zeros((H // 4, W // 4), np.uint8)
    for i in np.where(g < 10.5)[0]:
        rz = max(150.0, 1.2 * R_all[i]) if g[i] < 10 else 1.2 * R_all[i]
        c_ = (int(round(x[i])), int(round(y[i])))
        cv2.circle(wm, c_, int(rz), 0, -1)
        cv2.circle(zone4, (c_[0] // 4, c_[1] // 4), int((rz + 30) / 4), 1, -1)
    e = min(16, H // 8, W // 8)
    wm[:e], wm[H - e:], wm[:, :e], wm[:, W - e:] = 0, 0, 0, 0
    return wm, zone4


def hold_out_boxes(lum4, zone4, H, W):
    """Two regions never used as training targets: the most structured 384 px box (std of the 8 px-smoothed
    luminance) outside every bright-star zone, and the most structured 512 px box among the faintest quarter of
    the admissible ones, 256 px away from the first. Sizes shrink on small frames. Returns (box1, box2), boxes as
    (x0, y0, x1, y1), or None when the frame is too small."""
    s1, s2 = min(384, H // 4, W // 4) // 4 * 4, min(512, H // 3, W // 3) // 4 * 4
    if s1 < 64 or s2 < 64:
        return None
    lumb = cv2.GaussianBlur(lum4, (0, 0), 2)
    margin = max(1, min(16, lum4.shape[0] // 16, lum4.shape[1] // 16))

    def pick(size, extra=None, faint=False):
        Y, X, mean_, sd_, zf = box_scores(lumb, zone4, size // 4, max(1, margin), margin)
        if Y.size == 0:
            return None
        ok = zf == 0
        if not ok.any():
            ok = zf <= np.percentile(zf, 5)
        if extra is not None:
            ok &= extra(Y, X)
        if faint and ok.any():
            ok &= mean_ <= np.percentile(mean_[ok], 25)
        if not ok.any():
            return None
        k = int(np.argmax(np.where(ok, sd_, -1)))
        y0, x0 = int(Y.flat[k]) * 4, int(X.flat[k]) * 4
        return (x0, y0, x0 + size, y0 + size)
    b1 = pick(s1)
    if b1 is None:
        return None

    def apart(Y, X):
        x0, y0, x1, y1 = b1
        bx, by = X * 4, Y * 4
        return (bx + s2 < x0 - s1 // 1.5) | (bx > x1 + s1 // 1.5) | (by + s2 < y0 - s1 // 1.5) | (by > y1 + s1 // 1.5)
    b2 = pick(s2, apart, faint=True)
    return None if b2 is None else (b1, b2)


def nebula_colour(S4, lum4, bk, sg, x, y, g, R_all, filled):
    """Colour of the frame's brightest extended nebula (for the brightness augmentation): median colour (noise units)
    of the brightest 0.5 % of the 16 px-smoothed luminance at 1/4 scale, away from the filled star cores and from
    the inner half of the extent of the stars brighter than G 10.5. Returns (colour / its mean, description)."""
    inner4 = np.zeros(lum4.shape, np.uint8)
    for i in np.where(g < 10.5)[0]:
        cv2.circle(inner4, (int(round(x[i])) // 4, int(round(y[i])) // 4), int(0.5 * R_all[i] / 4) + 1, 1, -1)
    core4 = down4(np.asarray(filled)[None].astype(np.float32))[0] > 0
    core4 = cv2.dilate(core4.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    lumc = cv2.GaussianBlur(lum4, (0, 0), 4)
    okc = ~core4 & (inner4 == 0)
    if okc.sum() < 100:
        okc = np.ones_like(okc)
    selc = okc & (lumc >= np.percentile(lumc[okc], 99.5))
    col = np.array([np.median((S4[c][selc] - bk[c]) / sg[c]) for c in range(3)])
    if not np.all(np.isfinite(col)) or col.mean() <= 0:
        col = np.ones(3)
    yy, xx = np.nonzero(selc)
    where = f"level {float(np.median(lumc[selc])):.1f} sigma around ({4 * xx.mean():.0f},{4 * yy.mean():.0f})"
    return np.clip(col / col.mean(), 0.2, 5.0).astype(np.float32), where


def star_cores(filled, x, y, g, R_all):
    """Cores of the real stars, where the starless image is a fill and the denoised stack no faithful noise
    reference: the filled cores dilated 3 px, and discs of max(4, 0.12 R) px around every star G < 16."""
    cm = cv2.dilate(np.asarray(filled).astype(np.uint8), np.ones((7, 7), np.uint8))
    for xs, ys_, gs, rs in zip(x, y, g, R_all):
        if gs < 16:
            cv2.circle(cm, (int(round(xs)), int(round(ys_))), int(max(4, 0.12 * rs)), 1, -1)
    return cm


def patch_centres(WM, x, y, g, H, W, P):
    """Admissible patch centres (every 4 px): at least 150 px from every star brighter than G 10 (scaled down on
    small frames), the patch inside the frame, a trusted target at the centre."""
    ok = np.ones((H, W), np.uint8)
    rr = int(min(150, P // 2))
    for i in np.where(g < 10)[0]:
        cv2.circle(ok, (int(round(x[i])), int(round(y[i]))), rr, 0, -1)
    h2 = P // 2 + 1
    ok[:h2], ok[H - h2:], ok[:, :h2], ok[:, W - h2:] = 0, 0, 0, 0
    cys, cxs = np.nonzero(ok[::4, ::4])
    cys, cxs = cys * 4, cxs * 4
    keep = WM[cys, cxs] > 0
    return cys[keep], cxs[keep]


# --------------------------------------------------------------------------------------------- synthesis

class Synth:
    """Pastes real star stamps into (3, P, P) patches (on the training device)."""

    def __init__(self, lib, sat99, gain, sg, dev, glow=False, neb_col=None):
        self.dev = dev
        # glow: a smooth glow centred on half of the pasted bright stars, in the input and the target alike: a nebula
        # peaked on its star is nebula (the network had taken the bright core of NGC 2023 around HD 37903, a G 7.8
        # star drawn at rendering, for star light: lost)
        self.glow = glow
        nc = np.asarray(neb_col if neb_col is not None else [1.0, 1.0, 1.0], np.float64)
        self.neb_col = nc / nc.mean()
        self.sat99 = np.asarray(sat99, np.float32)
        self.gain = torch.as_tensor(np.asarray(gain, np.float32), device=dev)
        self.sg = torch.as_tensor(np.asarray(sg, np.float32), device=dev)
        self.bright = [dict(b, t=torch.from_numpy(np.asarray(b["t"], np.float32)).to(dev)) for b in lib["bright"]]
        self.faint = [dict(f, t=torch.from_numpy(np.asarray(f["t"], np.float32)).to(dev)) for f in lib["faint"]]
        gb = np.array([b["g"] for b in self.bright], float)
        w = 10 ** (-0.15 * (gb - 11)) if gb.size else np.ones(0)
        self.pb = w / w.sum() if w.size else w
        self.n_bright = len(self.bright)

    def plan(self, rng, P, n_bright=None):
        """1-6 bright stamps (bright ones drawn more often), scale 0.3-2 (saturated: 1-2), anywhere from -0.3 R to
        P + 0.3 R, 8 % colour jitter, clipped at their plateau, half of them (with ``glow``) on a smooth glow centred
        on them; and 0-24 faint stamps."""
        plan = []
        nb = (rng.integers(1, 7) if n_bright is None else n_bright) if self.n_bright else 0
        for _ in range(nb):
            i = int(rng.choice(self.n_bright, p=self.pb))
            b = self.bright[i]
            m = 0.3 * b["R"]
            sat = bool(np.asarray(b["sat"]).any())
            s = rng.uniform(1.0, 2.0) if sat else rng.uniform(0.3, 2.0)
            cy, cx = rng.uniform(-m, P + m, 2).astype(int)
            col = np.exp(rng.normal(0, 0.08, 3))
            clip = np.where(b["sat"], b["plateau"], self.sat99 * rng.uniform(1.0, 1.35))
            if self.glow and rng.random() < 0.5:
                # 2-300 noise sigmas at the centre, scale 15-150 px, exponential or Lorentzian, the nebula's colour or
                # the star's (a reflection nebula)
                amp = float(np.exp(rng.uniform(np.log(2.0), np.log(300.0))))
                gcol = (self.neb_col if rng.random() < 0.5 else np.ones(3)) * np.exp(rng.normal(0, 0.15, 3))
                plan.insert(0, ("g", None, cy, cx, (amp, float(rng.uniform(15, 150)), int(rng.integers(2))), gcol,
                                None))
            plan.append(("b", i, cy, cx, s, col, clip))
        nf = (rng.integers(0, 25) if n_bright is None else 0) if self.faint else 0
        for _ in range(nf):
            i = int(rng.integers(len(self.faint)))
            cy, cx = rng.integers(0, P, 2)
            plan.append(("f", i, cy, cx, rng.uniform(0.3, 2.0), np.exp(rng.normal(0, 0.08, 3)), None))
        return plan

    def apply(self, X, plan, T=None):
        """X (3, P, P) in ADU, modified in place: the stamps added (light never lowers a pixel already brighter than
        the clip level), then their photon noise (none on clipped pixels). Returns the pasted light. The glows (see
        plan) go into the target T too (modified in place), when it is given: the network learns to leave them."""
        P = X.shape[-1]
        light = torch.zeros_like(X)
        clipped = torch.zeros_like(X, dtype=torch.bool)
        for kind, i, cy, cx, s, col, clip in plan:
            if kind == "g":         # a glow centred on the star that follows: nebula, in the input and the target
                amp, sc, shape = s
                ar = torch.arange(P, device=self.dev, dtype=torch.float32)
                r = torch.sqrt((ar[:, None] - cy) ** 2 + (ar[None, :] - cx) ** 2)
                prof = torch.exp(-r / sc) if shape == 0 else 1.0 / (1.0 + (r / sc) ** 2)
                G = (amp * prof[None] * torch.tensor(col, device=self.dev, dtype=torch.float32).view(3, 1, 1)
                     * self.sg.view(3, 1, 1))
                G = G + torch.randn_like(G) * torch.sqrt(self.gain.view(3, 1, 1) * G)
                X += G
                if T is not None:
                    T += G
                continue
            st_ = (self.bright if kind == "b" else self.faint)[i]
            t = st_["t"]
            R = t.shape[-1] // 2
            y0, x0 = cy - R, cx - R
            ys0, xs0 = max(0, -y0), max(0, -x0)
            ys1, xs1 = min(t.shape[-2], P - y0), min(t.shape[-1], P - x0)
            if ys1 <= ys0 or xs1 <= xs0:
                continue
            sub = t[:, ys0:ys1, xs0:xs1] * torch.tensor(s * col, device=self.dev, dtype=torch.float32).view(3, 1, 1)
            sy, sx = slice(y0 + ys0, y0 + ys1), slice(x0 + xs0, x0 + xs1)
            reg = X[:, sy, sx]
            new = reg + sub
            if clip is not None:
                lim = torch.maximum(torch.tensor(clip, device=self.dev, dtype=torch.float32).view(3, 1, 1), reg)
                cl = (new > lim) & (sub > 0)
                new = torch.where(cl, lim, new)
                clipped[:, sy, sx] |= cl
            light[:, sy, sx] += new - reg
            X[:, sy, sx] = new
        noise = torch.randn_like(X) * torch.sqrt(self.gain.view(3, 1, 1) * light.clamp(min=0))
        X += torch.where(clipped, torch.zeros_like(noise), noise)
        return light


def nebula_boost(Dn, bk, sg, neb_col, valid=None, p=0.5, kmax=40.0):
    """Extra nebula light for a batch of starless patches Dn (B, 3, P, P): smoothed luminance above the sky (noise
    units, 6 px Gaussian, so that star-sized defects are not amplified, interpolated over the filled star cores
    where `valid` is 0) x (k - 1), k log-uniform in 1..kmax for a fraction p of the patches, coloured like neb_col
    with 15 % jitter. Returns the added light in ADU."""
    dev, nb = Dn.device, Dn.shape[0]
    lum = ((Dn - bk.view(1, 3, 1, 1)) / sg.view(1, 3, 1, 1)).mean(1, keepdim=True)
    gk = torch.exp(-0.5 * (torch.arange(-18, 19, device=dev, dtype=torch.float32) / 6.0) ** 2)
    gk = gk / gk.sum()
    if valid is None:
        valid = torch.ones_like(lum)

    def blur(a):
        pad = min(18, a.shape[-1] - 1, a.shape[-2] - 1)
        k1 = gk[18 - pad:18 + pad + 1] / gk[18 - pad:18 + pad + 1].sum()
        a = F.conv2d(F.pad(a, (pad, pad, 0, 0), mode="reflect"), k1.view(1, 1, 1, -1))
        return F.conv2d(F.pad(a, (0, 0, pad, pad), mode="reflect"), k1.view(1, 1, -1, 1))
    den = blur(valid)
    lum = torch.where(den > 0.05, blur(lum * valid) / den.clamp(min=0.05), torch.zeros_like(den)).clamp(min=0)
    col = torch.as_tensor(np.asarray(neb_col, np.float32), device=dev).view(1, 3, 1, 1) * \
        torch.exp(0.15 * torch.randn(nb, 3, 1, 1, device=dev))
    kk = torch.exp(torch.rand(nb, 1, 1, 1, device=dev) * math.log(kmax))
    amp = torch.where(torch.rand(nb, 1, 1, 1, device=dev) < p, kk - 1, torch.zeros_like(kk))
    return amp * lum * col * sg.view(1, 3, 1, 1)


# -------------------------------------------------------------------------------------------------- network

def normalise(x, bk, sg):
    return torch.asinh((x - bk.view(1, 3, 1, 1)) / (K_ASINH * sg.view(1, 3, 1, 1)))


def denormalise(n, bk, sg):
    return torch.sinh(n) * (K_ASINH * sg.view(1, 3, 1, 1)) + bk.view(1, 3, 1, 1)


def make_net(dev):
    return UNet(cin=3, cout=3, w=WIDTHS).to(dev)


def forward(net, x, bk, sg):
    """Linear ADU in -> starless in the asinh space (a residual: input + network output)."""
    n = normalise(x, bk, sg)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        o = net(n).float()
    return n + o


def val_items(boxes, synth, P=256, seed=123):
    """Fixed validation patches inside the hold-out boxes, each with two pasted bright stars."""
    rng = np.random.default_rng(seed)
    items = []
    for (x0, y0, x1, y1) in boxes or ():
        P_ = min(P, (x1 - x0) // 32 * 32, (y1 - y0) // 32 * 32)
        if P_ < 64:
            continue
        for cy in np.linspace(y0 + P_ // 2, y1 - P_ // 2, 2).astype(int):
            for cx in np.linspace(x0 + P_ // 2, x1 - P_ // 2, 2 if x1 - x0 > 1.2 * P_ else 1).astype(int):
                ya, xa = cy - P_ // 2, cx - P_ // 2
                w = np.zeros((P_, P_), np.float32)
                w[max(0, y0 - ya):min(P_, y1 - ya), max(0, x0 - xa):min(P_, x1 - xa)] = 1
                items.append((ya, xa, P_, w, synth.plan(rng, P_, n_bright=2)))
    return items


@torch.no_grad()
def run_val(net, items, synth, L, T, bk, sg, dev):
    """Mean L1 loss (asinh space) on the hold-out patches, with the pasted stars and without them."""
    if not items:
        return float("nan"), float("nan")
    net.eval()
    tot, tot0 = 0.0, 0.0
    for ya, xa, P_, w, plan in items:
        X = torch.from_numpy(np.ascontiguousarray(L[:, ya:ya + P_, xa:xa + P_], np.float32)).to(dev)
        Tp = torch.from_numpy(np.ascontiguousarray(T[:, ya:ya + P_, xa:xa + P_], np.float32)).to(dev)
        nt0 = normalise(Tp[None], bk, sg)
        wt = torch.from_numpy(w).to(dev)[None, None]
        out0 = forward(net, X[None], bk, sg)
        torch.manual_seed(0)
        synth.apply(X, plan, Tp)
        out = forward(net, X[None], bk, sg)
        tot += float(((out - normalise(Tp[None], bk, sg)).abs() * wt).sum() / (3 * wt.sum()))
        tot0 += float(((out0 - nt0).abs() * wt).sum() / (3 * wt.sum()))
    net.train()
    return tot / len(items), tot0 / len(items)


def train_net(data, synth, dev, minutes, batch=6, patch=PATCH, lr=3e-4, seed=0, progress=None, log=print,
              ckpt: Path | None = None, max_steps=None, report_every=20):
    """Train for a wall-clock budget (warm-up 300 steps, cosine decay over the budget), L1 loss in the asinh space,
    weight 0 where the target is not trusted and x5 on the pasted stars' footprints.

    data: L (input, (3, H, W)), T (target), S_den (denoised starless, for the brightness augmentation), WM (weight
    map (H, W)), VAL (bool (H, W), outside the filled cores), cys / cxs (patch centres), bk, sg, neb_col, items
    (validation patches). progress(frac, msg), called every `report_every` steps, may raise to stop: the state is
    checkpointed to `ckpt` (every 1000 steps and when stopped) and training resumes from it. Returns (net, history,
    steps)."""
    L, T, S_den, WM, VAL = data["L"], data["T"], data["S_den"], data["WM"], data["VAL"]
    cys, cxs = data["cys"], data["cxs"]
    if len(cys) == 0:
        raise StageError("no place to train the star removal network: the frame is too small or covered by the "
                         "surroundings of bright stars")
    bk = torch.as_tensor(np.asarray(data["bk"], np.float32), device=dev)
    sg = torch.as_tensor(np.asarray(data["sg"], np.float32), device=dev)
    P = patch
    net = make_net(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    it, run_l, hist, elapsed0 = 0, None, [], 0.0
    if ckpt is not None and Path(ckpt).exists():
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        it, run_l, hist, elapsed0 = ck["it"], ck["run"], ck["hist"], ck["elapsed"]
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        log(f"training resumed at step {it} ({elapsed0 / 60:.1f} of {minutes:.0f} min done)")
    budget = minutes * 60.0
    t0 = time.time()

    def save_ckpt():
        if ckpt is not None:
            tmp = Path(ckpt).with_suffix(".tmp")
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it, "run": run_l, "hist": hist,
                        "elapsed": elapsed0 + time.time() - t0, "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state()}, tmp)
            tmp.replace(ckpt)

    while True:
        el = elapsed0 + time.time() - t0
        frac = el / budget
        if frac >= 1 or (max_steps is not None and it >= max_steps):
            break
        lr_ = lr * min(1.0, (it + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
        for pg in opt.param_groups:
            pg["lr"] = lr_
        k = rng.integers(len(cys), size=batch)
        sl = [(slice(None), slice(cys[j] - P // 2, cys[j] - P // 2 + P), slice(cxs[j] - P // 2, cxs[j] - P // 2 + P))
              for j in k]
        X = torch.from_numpy(np.stack([np.ascontiguousarray(L[s], np.float32) for s in sl])).to(dev)
        Tt = torch.from_numpy(np.stack([np.ascontiguousarray(T[s], np.float32) for s in sl])).to(dev)
        Wt = torch.from_numpy(np.stack([np.ascontiguousarray(WM[s[1:]]) for s in sl])).to(dev).float()[:, None]
        # half of the patches get their own nebula amplified (with its photon noise) in the input and the target
        Dn = torch.from_numpy(np.stack([np.ascontiguousarray(S_den[s], np.float32) for s in sl])).to(dev)
        Vn = torch.from_numpy(np.stack([np.ascontiguousarray(VAL[s[1:]])[None] for s in sl]).astype(np.float32)).to(dev)
        delta = nebula_boost(Dn, bk, sg, data["neb_col"], Vn)
        nz = torch.randn_like(delta) * torch.sqrt(synth.gain.view(1, 3, 1, 1) * delta)
        X += delta + nz
        Tt += delta + nz
        del Dn, Vn, delta, nz
        light = torch.stack([synth.apply(X[b], synth.plan(rng, P), Tt[b]) for b in range(batch)])
        fp = F.max_pool2d(((light.abs() / sg.view(1, 3, 1, 1)).amax(1, keepdim=True) > 1.0).float(), 7, 1, 3)
        wt = Wt * (1 + 4 * fp)
        out = forward(net, X, bk, sg)
        loss = ((out - normalise(Tt, bk, sg)).abs() * wt).sum() / (3 * wt.sum().clamp(min=1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        it += 1
        run_l = loss.item() if run_l is None else 0.98 * run_l + 0.02 * loss.item()
        if it % 500 == 0 and data.get("items"):
            v, v0 = run_val(net, data["items"], synth, L, T, bk, sg, dev)
            hist.append(dict(step=it, minutes=round(el / 60, 2), loss=run_l, val=v, val0=v0))
            log(f"step {it}, {el / 60:.1f} of {minutes:.0f} min: loss {run_l:.4f}, hold-out regions {v:.4f} with "
                f"pasted stars, {v0:.4f} without")
        if it % 1000 == 0:
            save_ckpt()
        if progress is not None and it % report_every == 0:
            try:
                progress(min(frac, 1.0), f"training step {it}, loss {run_l:.4f}")
            except BaseException:
                save_ckpt()          # stopped: the next run resumes from here
                raise
    v, v0 = run_val(net, data.get("items"), synth, L, T, bk, sg, dev)
    hist.append(dict(step=it, minutes=round((elapsed0 + time.time() - t0) / 60, 2), loss=run_l, val=v, val0=v0))
    return net, hist, it


@torch.no_grad()
def apply_tiled(net, img, bk, sg, dev, tile=512, ov=128, batch=4, progress=None):
    """img (3, H, W) linear -> starless (3, H, W) linear: reflect-padded, overlapping tiles, linear feathering."""
    net.eval()
    bk = torch.as_tensor(np.asarray(bk, np.float32), device=dev)
    sg = torch.as_tensor(np.asarray(sg, np.float32), device=dev)
    H, W = img.shape[1:]
    pad = ov // 2
    step = tile - ov
    Hp = int(math.ceil(max(H + 2 * pad - tile, 0) / step)) * step + tile
    Wp = int(math.ceil(max(W + 2 * pad - tile, 0) / step)) * step + tile
    xp = np.pad(np.asarray(img, np.float32), ((0, 0), (pad, Hp - H - pad), (pad, Wp - W - pad)), mode="reflect")
    num = np.zeros((3, Hp, Wp), np.float32)
    den = np.zeros((Hp, Wp), np.float32)
    r = np.minimum(np.arange(tile) + 0.5, tile - np.arange(tile) - 0.5)
    w1 = np.clip(r / max(ov, 1), 1e-3, 1.0).astype(np.float32)
    wt = np.outer(w1, w1)
    pos = [(yy, xx) for yy in range(0, Hp - tile + 1, step) for xx in range(0, Wp - tile + 1, step)]
    for k in range(0, len(pos), batch):
        pb = pos[k:k + batch]
        t = torch.from_numpy(np.stack([xp[:, yy:yy + tile, xx:xx + tile] for yy, xx in pb])).to(dev)
        o = denormalise(forward(net, t, bk, sg), bk, sg).float().cpu().numpy()
        for (yy, xx), oo in zip(pb, o):
            num[:, yy:yy + tile, xx:xx + tile] += oo * wt
            den[yy:yy + tile, xx:xx + tile] += wt
        if progress is not None:
            progress((k + len(pb)) / len(pos), "removing the stars from the full frame")
    return (num / den)[:, pad:pad + H, pad:pad + W]
