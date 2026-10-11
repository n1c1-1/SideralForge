"""Starless image by subtraction of the Gaia star model (no neural star removal), cleaning and denoising."""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch

from ..core.imgutil import fill_holes, starlet
from ..core.params import ADVANCED, P, StageParams
from ..core.stage import Context, Stage
from .denoise import apply_net, load_model, norm_params


class StarlessParams(StageParams):
    mask_k: float = P(40.0, label="Unreliable core threshold", level=ADVANCED, ge=5, le=500, unit="sigma",
                      help="Star cores where the model exceeds this many noise sigmas (and saturated cores) are "
                           "treated as missing data and filled from their surroundings instead of trusting the "
                           "subtraction of a very bright, imperfectly modelled star.")
    wiener_k: float = P(1.0, label="Denoising strength", ge=0.2, le=4.0, step=0.1,
                        help="Strength of the multi-scale Wiener denoiser: at every scale, a structure is kept in "
                             "proportion to how consistently it appears in the two independent half-stacks.",
                        tip="1 is the statistically neutral value. Higher = smoother, lower = more grain.")
    residual_k: float = P(1.5, label="Star residual suppression", level=ADVANCED, ge=0, le=5, step=0.1,
                          help="How strongly the small-scale residuals of the star subtraction (common to both "
                               "half-stacks, so not caught by the Wiener filter) are suppressed. They are what makes "
                               "a starless background look mottled.")
    residual_scales: int = P(5, label="Residual scales", level=ADVANCED, ge=1, le=7,
                             help="Number of fine wavelet scales (1, 2, 4... px) where star residuals are modelled.")
    nebula_gate: float = P(0.85, label="Nebula protection", level=ADVANCED, ge=0, le=1, step=0.05,
                           help="In bright emission, the residual suppression is reduced by this fraction (the "
                                "nebula signal dominates the residuals there).")
    chroma_min_scale: int = P(3, label="Finest colour scale", level=ADVANCED, ge=0, le=6,
                              help="Colour information is discarded below wavelet scale 2^n px (fine colour is pure "
                                   "noise); 3 = 8 px.")
    lift_gmax: float = P(17.5, label="Dip repair limit", level=ADVANCED, ge=10, le=21, step=0.5, unit="G mag",
                         help="Around stars brighter than this, the background is not allowed to dip below its "
                              "local surroundings (over-subtracted star = dark hole).")
    halo_gmax: float = P(10.5, label="Optical halo limit", level=ADVANCED, ge=4, le=15, step=0.5, unit="G mag",
                         help="Stars brighter than this get their optical halo (blue defocus of the refractor, "
                              "filter reflections) measured and removed from the background, out to a radius growing "
                              "with the star flux.")


class Starless(Stage):
    name = "starless"
    title = "Star removal and denoising"
    description = ("Subtracts the complete star model (measured + faint stars) from the two half-stacks, fills the "
                   "few cores that cannot be trusted (saturated stars), removes residual and optical halos and "
                   "repairs over-subtracted dips. Then a multi-scale Wiener filter driven by the two independent "
                   "half-stacks keeps every structure that is really present in both, and removes the rest. No "
                   "neural star remover is used: faint nebulosity under the stars is preserved.")
    deps = ("stars", "optics", "denoise", "stack", "color")
    Params = StarlessParams
    version = "1"

    def run(self, ctx: Context, p: StarlessParams) -> None:
        dm = float(ctx.metric_of("optics", "mag_offset", 0.0))
        fs = np.load(ctx.input("stars", "star_fluxes.npz"))
        x, y, g, fl = fs["x"], fs["y"], fs["g"] - dm, fs["flux"]          # g shifted: thresholds stay meaningful
        M = ctx.load("color", "M.npy")
        model = ctx.load("stars", "stars_model.npy").astype(np.float32) + ctx.load("stars", "stars_model_faint.npy")

        ctx.progress(0.02, "subtracting the star model")
        A = ctx.load("optics", "lat_A.npy").astype(np.float32) - model
        B = ctx.load("optics", "lat_B.npy").astype(np.float32) - model
        full = ctx.load("optics", "lat_full.npy", mmap=True)
        mask = core_mask(full, model, A, B, p.mask_k)
        ctx.metric("filled_pct", round(100 * float(mask.mean()), 2))
        rng = np.random.default_rng(1)
        for X in (A, B):
            for c in range(3):
                hp = X[c] - cv2.GaussianBlur(X[c], (0, 0), 2)
                sgx = 1.4826 * np.median(np.abs(hp[::7, ::7])) * 1.3
                f = fill_holes(X[c], mask)
                X[c][mask] = f[mask] + rng.normal(0, sgx, int(mask.sum())).astype(np.float32)
        ctx.save("sl_mask.npy", mask)
        rawfill = 0.5 * (A + B)

        ctx.progress(0.15, "cleaning residual halos and dips")
        cleanraw, _ = clean(rawfill, x, y, g, fl, smooth=True, halos=True, lift_gmax=15.0, log=ctx.log)
        delta = cleanraw - rawfill
        del cleanraw

        ctx.progress(0.35, "multi-scale Wiener denoising")
        wms = wiener(A + delta, B + delta, M, p, model, log=ctx.log)
        ctx.progress(0.55, "final cleaning")
        final0, _ = clean(wms, x, y, g, fl, smooth=False, halos=False, lift_gmax=p.lift_gmax, log=ctx.log)
        del wms
        sl_final, nh = bright_halos(final0, x, y, g, p.halo_gmax)
        ctx.log(f"optical halos removed around {nh} bright stars")
        ctx.save("sl_final.npy", sl_final)
        del final0

        ctx.progress(0.7, "Noise2Noise detail layer")
        net, bk, sg = load_model(ctx)
        Wt = ctx.load("stack", "drz_weight.npy").astype(np.float32)
        _, _, nm = norm_params(ctx.load("stack", "drz_A.npy").astype(np.float32), Wt)
        sl_n2n = 0.5 * (apply_net(net, A + delta, nm, bk, sg, ctx.device) + apply_net(net, B + delta, nm, bk, sg, ctx.device))
        del net
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()
        ctx.progress(0.85, "cleaning the detail layer")
        n2n_c0, _ = clean(sl_n2n, x, y, g, fl, smooth=False, halos=False, lift_gmax=p.lift_gmax, log=ctx.log)
        n2n_final, _ = bright_halos(n2n_c0, x, y, g, p.halo_gmax)
        ctx.save("n2n_final.npy", n2n_final)
        from ..core.imgutil import asinh_preview
        ctx.preview("starless", asinh_preview(sl_final, M))


def core_mask(full, model, A, B, K):
    """Unreliable pixels: saturation, very bright model cores, significant residuals on star footprints."""
    raw = 0.5 * (A + B)
    mask = np.zeros(model.shape[1:], bool)
    for c in range(3):
        fc = np.asarray(full[c], np.float32)
        hp = fc - cv2.GaussianBlur(fc, (0, 0), 2)
        sig = 1.4826 * np.median(np.abs(hp[::7, ::7])) * 1.3
        lvl = np.percentile(fc, 99.999)
        rs = cv2.GaussianBlur(raw[c], (0, 0), 1.0)
        keep = (model[c] < 1.0 * sig).astype(np.float32)
        bgl = cv2.GaussianBlur(rs * keep, (0, 0), 6) / np.maximum(cv2.GaussianBlur(keep, (0, 0), 6), 1e-3)
        z = (rs - bgl) / (sig * 0.55)
        mask |= (fc > 0.7 * lvl) | (model[c] > K * sig) | ((np.abs(z) > 4) & (model[c] > 2 * sig))
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0


def clean(sl, x, y, g, fl, smooth, halos, lift_gmax, log=print):
    """Residual halo subtraction (optional), significant core dips/bumps filled, one-sided dip repair.

    ``g`` must already be shifted by the magnitude offset of the setup.
    """
    sl = sl.astype(np.float32).copy()
    H, W = sl.shape[1:]
    t0 = time.time()
    if halos:
        sl = _residual_halos(sl, x, y, g, fl, W, H, log)
    # 2. significant dips / bumps at the centre of stars G < 15
    Y = sl.mean(0)
    hp = Y - cv2.GaussianBlur(Y, (0, 0), 2)
    sig = 1.4826 * np.median(np.abs(hp[::7, ::7]))
    fixm = np.zeros((H, W), np.uint8)
    oy, ox = np.mgrid[-10:11, -10:11]
    rd = np.hypot(ox, oy)
    core, ring = rd < 4, (rd >= 6) & (rd <= 10)
    nfix = 0
    for i in np.where((g < 15) & (x > 40) & (x < W - 40) & (y > 40) & (y < H - 40))[0]:
        xc, yc = int(round(x[i])), int(round(y[i]))
        cut = Y[yc - 10:yc + 11, xc - 10:xc + 11]
        d = cut[core].mean() - np.median(cut[ring])
        if g[i] < 9.5 or abs(d) > 3 * sig / np.sqrt(core.sum()) * 3:
            rfix = int(np.clip(3 + 1.0 * (14.5 - g[i]), 3, 9)) if g[i] >= 9.5 else int(np.clip(9 + 4.5 * (9.5 - g[i]), 9, 34))
            cv2.circle(fixm, (xc, yc), rfix, 1, -1)
            nfix += 1
    oy2, ox2 = np.mgrid[-18:19, -18:19]
    rd2 = np.hypot(ox2, oy2)
    core2, ring2 = rd2 < 7, (rd2 >= 12) & (rd2 <= 18)
    Ys = cv2.GaussianBlur(Y, (0, 0), 1.5)
    for i in np.where((g < 14) & (x > 40) & (x < W - 40) & (y > 40) & (y < H - 40))[0]:
        xc, yc = int(round(x[i])), int(round(y[i]))
        cut = Ys[yc - 18:yc + 19, xc - 18:xc + 19]
        rg = cut[ring2]
        d = cut[core2].mean() - np.median(rg)
        if d < -max(2.5 * 1.4826 * np.median(np.abs(rg - np.median(rg))) / np.sqrt(core2.sum() / 9), 0.15):
            cv2.circle(fixm, (xc, yc), 10, 1, -1)
    m = fixm > 0
    wgt = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
    for c in range(3):
        sl[c] = sl[c] * (1 - wgt) + fill_holes(sl[c], m) * wgt
    # 3. one-sided dip repair: inside the disc of each star the background stays above its surroundings - 0.5 sigma
    disc = np.zeros((H, W), np.uint8)
    for i in np.where((g < lift_gmax) & (x > 0) & (x < W) & (y > 0) & (y < H))[0]:
        rr_ = int(np.clip(8 + 1.5 * (14 - g[i]), 7 if g[i] < 15 else 5, 18))
        cv2.circle(disc, (int(round(x[i])), int(round(y[i]))), rr_, 1, -1)
    dmk = disc > 0
    kp = (~dmk).astype(np.float32)
    wdisc = np.clip(cv2.GaussianBlur(dmk.astype(np.float32), (0, 0), 2.0) * 1.6, 0, 1)
    for c in range(3):
        hpc = sl[c] - cv2.GaussianBlur(sl[c], (0, 0), 2)
        sgc = 1.4826 * np.median(np.abs(hpc[::7, ::7]))
        bann = cv2.GaussianBlur(sl[c] * kp, (0, 0), 6) / np.maximum(cv2.GaussianBlur(kp, (0, 0), 6), 1e-3)
        floor_ = bann - 0.5 * sgc * (0.25 if smooth else 1.0)
        base_ = cv2.GaussianBlur(sl[c], (0, 0), 2.0) if smooth else sl[c]
        sl[c] = sl[c] + wdisc * (np.maximum(base_, floor_) - base_)
    # bright stars (G < 11): over-subtracted wings = large dark discs -> floor at the outer ring (not in emission)
    Egb = cv2.GaussianBlur(sl[0] - 0.87 * sl[1], (0, 0), 12)
    qb1, qb2 = np.percentile(Egb[::4, ::4], [55, 80])
    gate_b = np.clip((qb2 - Egb) / max(qb2 - qb1, 1e-6), 0, 1).astype(np.float32)
    before = sl.copy()
    yyb, xxb = np.mgrid[-90:91, -90:91]
    rdb = np.hypot(xxb, yyb)
    for i in np.where((g < 11) & (x > 95) & (x < W - 95) & (y > 95) & (y < H - 95))[0]:
        rr_ = float(np.clip(22 + 10 * (11 - g[i]), 22, 70))
        xc, yc = int(round(x[i])), int(round(y[i]))
        win = (slice(yc - 90, yc + 91), slice(xc - 90, xc + 91))
        wd_ = np.clip((rr_ + 4 - rdb) / 16, 0, 1) * gate_b[win]
        ann = (rdb >= rr_ + 4) & (rdb < rr_ + 18)
        for c in range(3):
            cut = sl[c][win]
            cs_ = cv2.GaussianBlur(cut, (0, 0), 2.0) if smooth else cut
            hpc = cs_ - cv2.GaussianBlur(cs_, (0, 0), 2)
            sgc = 1.4826 * np.median(np.abs(hpc[ann]))
            fl_ = np.percentile(cs_[ann], 20) - 0.5 * sgc
            sl[c][win] = cut + wd_ * (np.maximum(cs_, fl_) - cs_)
    for c in range(3):
        dl = np.clip(sl[c] - before[c], 0, None)
        sl[c] = before[c] + cv2.GaussianBlur(dl, (0, 0), 6)
    log(f"cleaning: {nfix} star cores repaired, dip repair on {100 * dmk.mean():.1f} % of pixels "
        f"({time.time() - t0:.0f} s)")
    return sl, m


def _residual_halos(sl, x, y, g, fl, W, H, log, RH=30):
    """Mean residual halo per channel and magnitude bin, normalised by flux, subtracted for every star G < 14.5."""
    from scipy.spatial import cKDTree
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yy, xx = np.mgrid[-RH:RH + 1, -RH:RH + 1]
    Rr = np.hypot(xx, yy)
    rb = np.arange(0, RH + 2, 1.0)
    rc = 0.5 * (rb[:-1] + rb[1:])
    grid_r = np.arange(0, RH + 1.01, 0.05)
    t = cKDTree(np.c_[x, y])
    bins = [(9.5, 10.5), (10.5, 11.5), (11.5, 12.5), (12.5, 13.5), (13.5, 14.5)]
    hb = np.zeros((len(bins), 3, len(rb) - 1))
    for kb, (lo, hi) in enumerate(bins):
        cand = np.where((g > lo) & (g < hi) & (x > RH + 5) & (x < W - RH - 5) & (y > RH + 5) & (y < H - RH - 5))[0]
        cand = cand[np.random.default_rng(1).permutation(len(cand))[:20000]]
        iso = np.array([i for i in cand if all((j == i) or (g[j] > g[i] + 2.5) for j in t.query_ball_point([x[i], y[i]], 12))])
        iso = iso[np.random.default_rng(0).permutation(len(iso))[:600]] if len(iso) else iso
        for c in range(3):
            profs = []
            for i in iso:
                xi, yi = int(round(x[i])), int(round(y[i]))
                cut = sl[c, yi - RH:yi + RH + 1, xi - RH:xi + RH + 1]
                bg = np.median(cut[Rr > RH - 3])
                pr_ = np.array([cut[(Rr >= rb[k]) & (Rr < rb[k + 1])].mean() - bg for k in range(len(rb) - 1)])
                profs.append(pr_ / max(fl[c, i], 1e-3))
            if profs:
                hh = np.median(np.array(profs), 0)
                hh[hh < 0] = 0
                hb[kb, c] = hh
    sel = np.where((g < bins[-1][1]) & (g >= bins[0][0]) & (x > -RH) & (x < W + RH) & (y > -RH) & (y < H + RH))[0]
    tabs = [torch.from_numpy(np.stack([np.interp(grid_r, rc, hb[k, c], right=0) for k in range(len(bins))])
                             .astype(np.float32)).to(dev) for c in range(3)]
    models = [torch.zeros(H * W, device=dev) for _ in range(3)]
    xxt = torch.from_numpy(xx.astype(np.float32)).to(dev)[None]
    yyt = torch.from_numpy(yy.astype(np.float32)).to(dev)[None]
    for k0 in range(0, len(sel), 8000):                      # ~1 GB per batch on the GPU
        ss = sel[k0:k0 + 8000]
        kb = np.clip(np.searchsorted([b_[1] for b_ in bins], g[ss]), 0, len(bins) - 1)
        xi, yi = np.round(x[ss]).astype(np.int64), np.round(y[ss]).astype(np.int64)
        py = torch.from_numpy(yi[:, None, None] + yy[None]).to(dev)
        px = torch.from_numpy(xi[:, None, None] + xx[None]).to(dev)
        ok = ((py >= 0) & (py < H) & (px >= 0) & (px < W)).float()
        idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1)).reshape(-1)
        dxs = torch.from_numpy((xi - x[ss]).astype(np.float32)).to(dev)
        dys = torch.from_numpy((yi - y[ss]).astype(np.float32)).to(dev)
        rr = torch.sqrt((xxt + dxs[:, None, None]) ** 2 + (yyt + dys[:, None, None]) ** 2)
        ridx = (rr / 0.05).long().clamp(max=len(grid_r) - 1)
        kb_t = torch.from_numpy(kb).to(dev)[:, None, None].expand_as(ridx)
        for c in range(3):
            fc = torch.from_numpy(fl[c, ss].astype(np.float32)).to(dev)
            models[c].index_add_(0, idx, (fc[:, None, None] * tabs[c][kb_t, ridx] * ok).reshape(-1))
    for c in range(3):
        sl[c] -= models[c].reshape(H, W).cpu().numpy()
    log(f"residual halos subtracted for {len(sel)} stars")
    return sl


def wiener(A_cam, B_cam, M, p, star_model, log=print):
    """Multi-scale Wiener filter on Y, R-Y, B-Y driven by the two half-stacks, with a star-residual model."""
    t0 = time.time()
    Minv = np.linalg.inv(M)
    NJ = 7

    def to_ycc(cam):
        lin = np.tensordot(M, cam, 1)
        Yv = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
        return np.stack([Yv, lin[0] - Yv, lin[2] - Yv]).astype(np.float32)

    def from_ycc(ycc):
        Yv, cr, cb = ycc
        R, Bv = cr + Yv, cb + Yv
        G = (Yv - 0.2126 * R - 0.0722 * Bv) / 0.7152
        return np.tensordot(Minv, np.stack([R, G, Bv]), 1).astype(np.float32)

    A = to_ycc(A_cam)
    B = to_ycc(B_cam)
    lm = np.tensordot(M, star_model, 1)
    Ym = (0.2126 * lm[0] + 0.7152 * lm[1] + 0.0722 * lm[2]).astype(np.float32)
    del lm
    wm, _ = starlet(Ym, NJ)
    Ysm = cv2.GaussianBlur(0.5 * (A[0] + B[0]), (0, 0), 20)
    mg = 300 if min(Ysm.shape) > 1200 else min(Ysm.shape) // 8
    inner = Ysm[mg:-mg, mg:-mg]
    faint = Ysm < np.percentile(inner, 35)
    faint[:mg] = faint[-mg:] = False
    faint[:, :mg] = faint[:, -mg:] = False
    q1, q2 = np.percentile(inner, [75, 97])
    neb_gate = p.nebula_gate * np.clip((Ysm - q1) / (q2 - q1), 0, 1)
    out = np.zeros_like(A)
    for c in range(3):
        wa, ca = starlet(A[c], NJ)
        wb, cb = starlet(B[c], NJ)
        res = 0.5 * (ca + cb)
        msg = []
        for j in range(NJ):
            S = 0.5 * (wa[j] + wb[j])
            D = 0.5 * (wa[j] - wb[j])
            s = max(3.0, 1.5 * 2 ** j)
            ps = cv2.GaussianBlur(S * S, (0, 0), s)
            pd = cv2.GaussianBlur(D * D, (0, 0), s)
            if p.residual_k > 0 and j <= p.residual_scales:
                pm = cv2.GaussianBlur(wm[j] * wm[j], (0, 0), s)
                ex = (ps - pd)[faint][::3]
                pmf = pm[faint][::3]
                qs = np.quantile(pmf, np.linspace(0, 1, 25))
                idx = np.clip(np.searchsorted(qs, pmf) - 1, 0, 23)
                med = np.array([np.median(ex[idx == k]) if (idx == k).any() else 0 for k in range(24)])
                ctr = np.array([np.median(pmf[idx == k]) if (idx == k).any() else 0 for k in range(24)])
                resid_pow = np.maximum.accumulate(np.clip(med - med[:3].mean(), 0, None))
                rmap = np.interp(pm, ctr, resid_pow) * p.residual_k * (1 - neb_gate)
                gj = np.clip(1 - (p.wiener_k * pd + rmap) / np.maximum(ps, 1e-12), 0, 1)
            else:
                gj = np.clip(1 - p.wiener_k * pd / np.maximum(ps, 1e-12), 0, 1)
            if c > 0 and j < p.chroma_min_scale:
                gj = gj * 0.0
            res += gj * S
            msg.append(f"{2 ** j}px:{np.median(gj):.2f}")
        out[c] = res
        log(f"Wiener {('Y', 'R-Y', 'B-Y')[c]} median gain per scale " + " ".join(msg) + f" ({time.time() - t0:.0f} s)")
    return from_ycc(out)


def bright_halos(sl, x, y, g, gmax):
    """Remove the optical halo of every bright star from the starless image (radial profile per star and channel).

    The halo radius grows with the square root of the flux (315 px at G = 5 down to 25 px at G = gmax); the
    background is measured just beyond it. No removal inside bright emission (the profile would measure the nebula).
    """
    sl = sl.astype(np.float32).copy()
    Eg = cv2.GaussianBlur(sl[0] - 0.87 * sl[1], (0, 0), 12)
    q1, q2 = np.percentile(Eg[::4, ::4], [55, 80])
    gate = np.clip((q2 - Eg) / max(q2 - q1, 1e-6), 0, 1).astype(np.float32)
    H, W = sl.shape[1:]
    sel = np.where((g < gmax) & (x > -40) & (x < W + 40) & (y > -40) & (y < H + 40))[0]
    sel = sel[np.argsort(g[sel])]
    n = 0
    for i in sel:
        rmax = int(np.clip(25 * 10 ** (0.2 * (gmax - g[i])), 25, 320))
        R_ = rmax + 20
        xc, yc = int(round(x[i])), int(round(y[i]))
        y0, y1, x0, x1 = max(0, yc - R_), min(H, yc + R_ + 1), max(0, xc - R_), min(W, xc + R_ + 1)
        if y1 - y0 < 20 or x1 - x0 < 20:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(xx - x[i], yy - y[i])
        rb = np.arange(0, rmax + 20, 2.0)
        idx = np.clip((rr / 2.0).astype(int), 0, len(rb) - 2)
        flat_idx = idx.ravel()
        counts = np.bincount(flat_idx, minlength=len(rb) - 1)
        order = np.argsort(flat_idx, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(counts)])
        for c in range(3):
            patch = sl[c, y0:y1, x0:x1]
            vals = patch.ravel()[order]
            med = np.array([np.percentile(vals[bounds[k]:bounds[k + 1]], 35) if counts[k] > 5 else np.nan
                            for k in range(len(rb) - 1)])
            bgv = np.nanmedian(med[-8:])
            prof = np.minimum.accumulate(np.nan_to_num(np.clip(med - bgv, 0, None)))
            rcv = rb[:-1] + 1.0
            prof = prof * np.clip((rmax - rcv) / (0.4 * rmax), 0, 1) ** 2
            h = np.interp(rr, rcv, prof, right=0)
            h = np.minimum(h, 0.9 * np.clip(patch - bgv, 0, None))
            sl[c, y0:y1, x0:x1] = patch - h * gate[y0:y1, x0:x1]
        n += 1
    return sl, n
