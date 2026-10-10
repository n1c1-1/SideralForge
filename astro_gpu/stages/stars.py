"""Stars: forced PSF photometry of every Gaia star, sub-pixel position refinement, faint-star model."""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..core.params import ADVANCED, P, StageParams
from ..core.stage import Context, Stage
from .optics import COL0, basis, sample_E


class StarsParams(StageParams):
    color_prior: float = P(0.3, label="Colour prior strength", level=ADVANCED, ge=0.02, le=5.0, step=0.05,
                           help="How much the green and blue fluxes of a star may depart from the value expected "
                                "from its red flux and its Gaia colour (relative tolerance).",
                           tip="Small values make star colours follow Gaia closely and prevent failed fits (green "
                               "dots, black holes); large values trust the measurement more.")
    refine_passes: int = P(2, label="Position refinement passes", level=ADVANCED, ge=0, le=5,
                           help="Passes of sub-pixel position refinement (Gaia positions are excellent but the stack "
                                "registration leaves small local offsets).")
    solver_passes: int = P(4, label="Background / flux iterations", level=ADVANCED, ge=1, le=10,
                           help="Alternations between the local background estimate and the joint flux fit.")
    cg_iterations: int = P(60, label="Solver iterations", level=ADVANCED, ge=10, le=500,
                           help="Conjugate-gradient iterations of the joint flux fit of all stars.")
    faint_model: bool = P(True, label="Model faint stars",
                          help="Render the Gaia stars too faint to be measured from their catalogue magnitude, and "
                               "move them from the nebula layer to the star layer.")
    extended_psf: bool = P(True, label="Extended PSF (halos, spikes)",
                           help="The far light of bright stars - halo, diffraction spikes of a Newtonian's spider, "
                                "reflections - is measured on your own data: the residuals around many bright isolated "
                                "stars, normalised by their flux, are combined into a 2-D template per colour, which is "
                                "then subtracted around every bright star. Nothing is assumed about the optics.")
    extended_radius: int = P(0, label="Extended PSF radius", level=ADVANCED, ge=0, le=600, unit="px",
                             help="Radius of the extended PSF template. 0 = measured on the data (where the average "
                                  "halo of the brightest stars falls into the noise).")


class Stars(Stage):
    name = "stars"
    title = "Star photometry"
    description = ("Measures every Gaia star of the field simultaneously with the real PSF at its known position "
                   "(forced photometry): blended stars are separated, and the flux of each colour channel is "
                   "constrained by the Gaia colour of the star so that a failed fit cannot create green dots or "
                   "black holes. Positions are then refined to a hundredth of a pixel. Stars too faint to be "
                   "measured are modelled from their Gaia magnitude. The result is a complete model of the starlight "
                   "of the image, used to separate the stars from the nebula and to redraw them.")
    deps = ("optics", "catalogs")
    Params = StarsParams
    version = "11"         # 9: physical template; 11: per-channel amplitudes kept (one amplitude tried, worse)

    def run(self, ctx: Context, p: StarsParams) -> None:
        dev = ctx.device
        img = ctx.load("optics", "lat_full.npy").astype(np.float32)
        H, W = img.shape[1:]
        pm = torch.load(ctx.input("optics", "psf_model.pt"), weights_only=False)
        dm = float(ctx.metric_of("optics", "mag_offset", 0.0))
        d = np.load(ctx.input("catalogs", "gaia_deep.npz"))
        x0, y0, g0 = d["x"], d["y"], d["g"]
        col0 = np.nan_to_num(d["bp"] - d["rp"], nan=COL0)
        ok = (x0 > -15) & (x0 < W + 15) & (y0 > -15) & (y0 < H + 15) & np.isfinite(g0)
        x, y, g, col = x0[ok].astype(np.float64), y0[ok].astype(np.float64), g0[ok], col0[ok]
        sel = np.searchsorted(np.where(ok)[0], pm["sel"])
        npass = 1 + p.refine_passes
        tdx, tdy = np.zeros(len(x)), np.zeros(len(x))
        model = fluxes = Fp = None
        for k in range(npass):
            if k > 0:
                ctx.progress(k / npass - 0.02, f"refining star positions (pass {k})")
                dx, dy, msg = refine_positions(img, model, x, y, g, col, fluxes, pm, W, H, dm, dev)
                x, y = x + dx, y + dy
                tdx, tdy = tdx + dx, tdy + dy
                ctx.log(msg)

            def prog(f, msg, k=k):
                ctx.progress((k + f) / npass * 0.85, msg)
            model, fluxes, Fp, info = forced_photometry(img, pm, x, y, g, col, sel, dm, p, W, H, dev, prog)
        ctx.save("stars_model.npy", model)
        np.savez(ctx.out("star_fluxes.npz"), x=x, y=y, g=g, col=col, flux=fluxes, pred=Fp, dx=tdx, dy=tdy)
        if p.extended_psf:
            ctx.progress(0.86, "extended PSF (halos, spikes)")
            ext, einfo = extended_psf(img, model, x, y, g, Fp, fluxes, dm, W, H, p.extended_radius, ctx.log)
            tpl = einfo.pop("_template", None)
            for k_, v in einfo.items():
                ctx.metric(k_, v)
            if ext is not None:
                ctx.save("stars_model_ext.npy", ext.astype(np.float32))
                ctx.save("ext_psf_template.npy", tpl)
            else:
                ctx.save("stars_model_ext.npy", np.zeros((3, 1, 1), np.float32))
        else:
            ctx.save("stars_model_ext.npy", np.zeros((3, 1, 1), np.float32))
        for k_, v in info.items():
            ctx.metric(k_, v)
        q = star_qc(g, fluxes, Fp)
        for k_, v in q.items():
            ctx.metric(k_, v)
        ctx.log(f"{len(x)} stars measured; channel failures (< 35 % of expected, G < 17): green "
                f"{q['collapsed_green']}, blue {q['collapsed_blue']}")
        # faint stars
        f = np.load(ctx.input("catalogs", "gaia_faint.npz"))
        g_deep = float(np.nanmax(g)) if len(g) else 18.5
        if p.faint_model and len(f["g"]):
            ctx.progress(0.9, "modelling the faint stars")
            fm, ff = faint_model(f, x, y, g, col, Fp, pm, W, H, g_deep, dev)
            ctx.save("stars_model_faint.npy", fm)
            np.savez(ctx.out("faint_fluxes.npz"), **ff)
            ctx.metric("faint_stars", int(len(ff["g"])))
            ctx.metric("faint_light_adu", [round(float(fm[c].mean()), 3) for c in range(3)])
        else:
            ctx.save("stars_model_faint.npy", np.zeros_like(model))
            np.savez(ctx.out("faint_fluxes.npz"), x=np.zeros(0), y=np.zeros(0), g=np.zeros(0),
                     flux=np.zeros((3, 0), np.float32), col0=COL0)
        yy, xx = H // 2, W // 2
        crop = np.s_[:, yy - 300:yy + 300, xx - 300:xx + 300]
        from ..core.imgutil import asinh_preview
        ctx.preview("star_model", np.concatenate([asinh_preview(img[crop]), np.ones((3, 600, 6)) * 0.2,
                                                  asinh_preview(img[crop] - model[crop])], 2))


def predict_flux(g, col, sel, flux_fit):
    """log10(F_c) + 0.4 G = cubic(colour), fitted on the PSF stars; prediction for every star."""
    out, disp = [], []
    for c in range(3):
        yv = np.log10(np.clip(flux_fit[c], 1e-3, None)) + 0.4 * g[sel]
        xc = col[sel]
        ok = np.isfinite(yv)
        s = 0.0
        pfit = np.zeros(4)
        for _ in range(3):
            pfit = np.polyfit(xc[ok], yv[ok], 3)
            r = yv - np.polyval(pfit, xc)
            s = 1.4826 * np.median(np.abs(r[ok]))
            ok = np.abs(r) < 3 * s
        out.append(10 ** (np.polyval(pfit, np.clip(col, -0.5, 4.5)) - 0.4 * g))
        disp.append(round(100 * (10 ** s - 1), 1))
    return np.array(out), disp


def stamps(E, xs, ys, cols, R, W, H, dev, xi=None, yi=None):
    """Normalised PSF of each star sampled on integer pixels around round(x): (N, 2R+1, 2R+1) half floats."""
    if xi is None:
        xi, yi = np.round(xs).astype(np.int64), np.round(ys).astype(np.int64)
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    ring = torch.from_numpy(np.hypot(ox, oy) >= R - 1.5).to(dev)
    rr = torch.from_numpy(np.hypot(ox, oy).astype(np.float32)).to(dev)
    taper = torch.clamp((R + 0.5 - rr) / 2.0, 0, 1)
    out = []
    for k in range(0, len(xs), 20000):
        sl = slice(k, k + 20000)
        dx = torch.from_numpy((xi[sl, None, None] + ox[None] - xs[sl, None, None]).astype(np.float32)).to(dev)
        dy = torch.from_numpy((yi[sl, None, None] + oy[None] - ys[sl, None, None]).astype(np.float32)).to(dev)
        u = torch.from_numpy((xs[sl] / (W / 2) - 1).astype(np.float32)).to(dev)
        v = torch.from_numpy((ys[sl] / (H / 2) - 1).astype(np.float32)).to(dev)
        cc = torch.from_numpy((cols[sl] - COL0).astype(np.float32)).to(dev)
        Pk = (sample_E(E, dx, dy) * basis(u, v, cc)[:, None, None, :]).sum(-1)
        edge = (Pk * ring).sum((1, 2)) / ring.sum()       # PSF must reach 0 at the stamp edge (no dark squares)
        Pk = (Pk - edge[:, None, None]).clamp(min=0) * taper
        Pk = Pk / Pk.sum((1, 2), keepdim=True).clamp(min=1e-9)
        out.append(Pk.half())
    return torch.cat(out) if out else torch.zeros(0, 2 * R + 1, 2 * R + 1, device=dev, dtype=torch.half), xi, yi


class Group:
    """Stars sharing a stamp size: render (flux -> image) and gather (image -> per-star projection).

    Memory: int32 pixel indices + half-precision profiles (the off-image part is zeroed in the profiles), built
    in batches, so that a million stars fit on an 8 GB GPU.
    """

    def __init__(self, Pst, xi, yi, R, W, H, dev, batch=50000):
        oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
        oyt = torch.from_numpy(oy.astype(np.int32)).to(dev)
        oxt = torch.from_numpy(ox.astype(np.int32)).to(dev)
        idx_parts, p_parts = [], []
        for k in range(0, len(xi), batch):
            py = torch.from_numpy(yi[k:k + batch].astype(np.int32)).to(dev)[:, None, None] + oyt[None]
            px = torch.from_numpy(xi[k:k + batch].astype(np.int32)).to(dev)[:, None, None] + oxt[None]
            inside = (py >= 0) & (py < H) & (px >= 0) & (px < W)
            idx_parts.append(py.clamp(0, H - 1) * W + px.clamp(0, W - 1))
            p_parts.append(Pst[k:k + batch] * inside.to(Pst.dtype))
            del py, px, inside
        self.idx = torch.cat(idx_parts) if idx_parts else torch.zeros(0, 2 * R + 1, 2 * R + 1, dtype=torch.int32, device=dev)
        self.P = torch.cat(p_parts) if p_parts else torch.zeros(0, 2 * R + 1, 2 * R + 1, dtype=torch.half, device=dev)

    def render(self, Fl, out):
        out.index_add_(0, self.idx.reshape(-1), (Fl[:, None, None] * self.P.float()).reshape(-1))

    def gather(self, img_flat):
        return (img_flat[self.idx] * self.P.float()).sum((1, 2))


def forced_photometry(img, pm, x, y, g, col, sel, dm, p, W, H, dev, progress):
    Fp, disp = predict_flux(g, col, sel, pm["flux"])
    bright = g < 13.0 + dm
    t0 = time.time()
    model_all = np.zeros_like(img)
    fluxes = np.zeros((3, len(x)), np.float32)
    scale_R = None
    info = {"prediction_scatter_pct": disp}
    for c in range(3):
        E = pm["E"][c].to(dev)
        d = torch.from_numpy(img[c]).to(dev).reshape(-1)
        lvl = np.percentile(img[c], 99.999)
        sat = torch.from_numpy(img[c] > 0.7 * lvl).to(dev)
        sat = F.max_pool2d(sat.float()[None, None], 5, 1, 2)[0, 0].reshape(-1) > 0
        hp = img[c] - cv2.GaussianBlur(img[c], (0, 0), 2)
        sig = float(1.4826 * np.median(np.abs(hp[::7, ::7])) * 1.3)
        use = (Fp[c] * 0.12 > 0.25 * sig) | bright
        groups = []
        for R, m in ((10, use & bright), (7, use & ~bright)):
            ii = np.where(m)[0]
            Pst, xi, yi = stamps(E, x[ii], y[ii], col[ii], R, W, H, dev)
            groups.append((ii, Group(Pst, xi, yi, R, W, H, dev)))
        # radial wings for bright stars (median profile normalised to the predicted flux of a G = 10 star)
        wr, wp = pm["wing_r"], pm["wing_prof"][c]
        m10 = (g > 9.9) & (g < 10.1)
        fp10 = np.median(Fp[c][m10]) if m10.any() else 1.0
        wing_frac = np.clip(wp / fp10, 0, None)
        wing_frac[wr <= 9] = 0
        ib = np.where(g < 11.0 + dm)[0]
        yy, xx = np.mgrid[-60:61, -60:61]
        wing_stamp = torch.from_numpy(np.interp(np.hypot(xx, yy), wr, wing_frac, right=0).astype(np.float32)).to(dev)
        wgrp = Group(wing_stamp[None].expand(len(ib), -1, -1).half(), np.round(x[ib]).astype(np.int64),
                     np.round(y[ib]).astype(np.int64), 60, W, H, dev)
        # prior: red = Gaia prediction; green and blue = red scale of the star x Gaia colour
        ctr = Fp[c].astype(np.float32) if (c == 0 or scale_R is None) else (scale_R * Fp[c]).astype(np.float32)
        Fl = torch.from_numpy(ctr).to(dev)
        Fpt = Fl.clone()
        mdl_pred = torch.zeros(H * W, device=dev)
        for ii, G in groups:
            G.render(Fpt[ii], mdl_pred)
        sig_prior = (0.5 * Fpt if c == 0 else p.color_prior * Fpt) + 0.5 * sig / 0.12
        satnp = sat.reshape(H, W).cpu().numpy()
        core_sat = torch.from_numpy(satnp[np.clip(np.round(y).astype(int), 0, H - 1),
                                          np.clip(np.round(x).astype(int), 0, W - 1)]).to(dev)
        lam = 1.0 / sig_prior ** 2

        def Aop(f):
            m = torch.zeros(H * W, device=dev)
            for ii, G in groups:
                G.render(f[ii], m)
            wgrp.render(f[ib], m)
            wm = wgt * m
            out = torch.zeros_like(f)
            for ii, G in groups:
                out[ii] += G.gather(wm)
            out[ib] += wgrp.gather(wm)
            return out + lam * f

        for outer in range(p.solver_passes):
            mdl = torch.zeros(H * W, device=dev)
            for ii, G in groups:
                G.render(Fl[ii], mdl)
            wgrp.render(Fl[ib], mdl)
            var = sig ** 2 + 0.05 * mdl.clamp(min=0) + (0.03 * mdl) ** 2
            wgt = torch.where(sat, torch.zeros_like(var), 1.0 / var)
            res = (d - mdl).reshape(H, W)
            keep = (torch.maximum(mdl, mdl_pred).reshape(H, W) < 0.5 * sig).float() * (~sat.reshape(H, W)).float()
            num = cv2.GaussianBlur((res * keep).cpu().numpy(), (0, 0), 4)
            den = cv2.GaussianBlur(keep.cpu().numpy(), (0, 0), 4)
            bb = num / np.maximum(den, 1e-3)
            bb = np.where(den > 0.02, bb, cv2.GaussianBlur(bb, (0, 0), 20))
            b = torch.from_numpy(bb.astype(np.float32)).to(dev).reshape(-1)
            rhs_img = wgt * (d - b)
            rhs = torch.zeros_like(Fl)
            for ii, G in groups:
                rhs[ii] += G.gather(rhs_img)
            rhs[ib] += wgrp.gather(rhs_img)
            rhs = rhs + lam * Fpt
            f = Fl.clone()
            r = rhs - Aop(f)
            pv = r.clone()
            rs = (r * r).sum()
            for _ in range(p.cg_iterations):
                Ap = Aop(pv)
                a = rs / (pv * Ap).sum()
                f = f + a * pv
                r = r - a * Ap
                rsn = (r * r).sum()
                pv = r + (rsn / rs) * pv
                rs = rsn
            Fl = torch.minimum(f.clamp(min=0), torch.where(core_sat, 4.0 * Fpt, torch.full_like(Fpt, 1e12)))
            progress((c + (outer + 1) / p.solver_passes) / 3, f"photometry channel {'RGB'[c]}, pass {outer + 1}/"
                                                             f"{p.solver_passes} ({time.time() - t0:.0f} s)")
        Fn = Fl.cpu().numpy()
        if c > 0 and scale_R is not None:
            # guard: a channel below 35 % (or far above) of the expected value is a failed fit
            exp_c = scale_R * Fp[c]
            bad = ((Fn < 0.35 * exp_c) | ((Fn > 3.0 * exp_c) & (g > 13.0 + dm))) & (g < 17.5 + dm)
            info[f"replaced_{'RGB'[c]}"] = int(bad.sum())
            Fn = np.where(bad, exp_c, Fn).astype(np.float32)
            Fl = torch.from_numpy(Fn).to(dev)
        mdl = torch.zeros(H * W, device=dev)
        for ii, G in groups:
            G.render(Fl[ii], mdl)
        wgrp.render(Fl[ib], mdl)
        model_all[c] = mdl.reshape(H, W).cpu().numpy()
        fluxes[c] = Fn
        if c == 0:
            snr_ok = Fn > 3 * sig / 0.12 * 0.2
            scale_R = np.where(snr_ok, np.clip(Fn / np.maximum(Fp[0], 1e-6), 0.3, 3.0), 1.0)
        groups = wgrp = None  # free GPU memory before the next channel
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    info["modelled_stars"] = int(len(x))
    return model_all, fluxes, Fp, info


def refine_positions(img, mdl, x, y, g, col, fl, pm, W, H, dm, dev):
    """Linearised sub-pixel position fit on the residual: r ~ c0 + a P + bx dP/dx + by dP/dy."""
    R, h = 6, 0.2
    sel = np.where((g < 15.0 + dm) & (g > 8.0 + dm) & (x > R + 2) & (x < W - R - 3) & (y > R + 2) & (y < H - R - 3))[0]
    xi, yi = np.round(x[sel]).astype(np.int64), np.round(y[sel]).astype(np.int64)
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    acc_dx, acc_dy, acc_w = np.zeros(len(sel)), np.zeros(len(sel)), np.zeros(len(sel))
    meds = []
    for c in range(3):
        lvl = np.percentile(img[c, ::4, ::4], 99.999)
        res_img = torch.from_numpy(img[c] - mdl[c]).to(dev)
        raw_img = torch.from_numpy(img[c]).to(dev)
        E = pm["E"][c].to(dev)

        def st(dx_, dy_):
            return stamps(E, x[sel] + dx_, y[sel] + dy_, col[sel], R, W, H, dev, xi, yi)[0].float()
        P0 = st(0, 0)
        dPx = (st(h, 0) - st(-h, 0)) / (2 * h)
        dPy = (st(0, h) - st(0, -h)) / (2 * h)
        py = torch.from_numpy(yi[:, None, None] + oy[None]).to(dev)
        px = torch.from_numpy(xi[:, None, None] + ox[None]).to(dev)
        r = res_img[py, px]
        sat = (raw_img[py, px] > 0.7 * lvl).any(-1).any(-1)
        Fc = torch.from_numpy(fl[c, sel].astype(np.float32)).to(dev)
        A = torch.stack([torch.ones_like(r), P0 * Fc[:, None, None], dPx * Fc[:, None, None],
                         dPy * Fc[:, None, None]], -1).reshape(len(sel), -1, 4)
        b = r.reshape(len(sel), -1, 1)
        AtA = A.transpose(1, 2) @ A + 1e-6 * torch.eye(4, device=dev)
        sol = torch.linalg.solve(AtA, A.transpose(1, 2) @ b)[..., 0]
        dx = sol[:, 2].cpu().numpy()
        dy = sol[:, 3].cpu().numpy()
        snr = (Fc / (1e-3 + r.std((1, 2)) * (2 * R + 1))).cpu().numpy()
        w = np.where(sat.cpu().numpy() | ~np.isfinite(dx) | ~np.isfinite(dy), 0, np.clip(snr, 0, 200) ** 2)
        acc_dx += w * np.nan_to_num(dx)
        acc_dy += w * np.nan_to_num(dy)
        acc_w += w
        meds.append(float(np.median(np.hypot(dx, dy)[w > 0])) if (w > 0).any() else 0.0)
    dx = np.clip(acc_dx / np.maximum(acc_w, 1e-9), -0.8, 0.8)
    dy = np.clip(acc_dy / np.maximum(acc_w, 1e-9), -0.8, 0.8)
    dx[acc_w == 0] = 0
    dy[acc_w == 0] = 0
    out_x, out_y = np.zeros(len(x)), np.zeros(len(x))
    out_x[sel], out_y[sel] = dx, dy
    msg = (f"{len(sel)} star positions refined, median correction {np.median(np.hypot(dx, dy)):.3f} px "
           f"(per channel {', '.join(f'{m:.3f}' for m in meds)})")
    return out_x, out_y, msg


def _shift(a, dx, dy):
    """Sub-pixel shift of a 2-D array (bilinear), value at (x, y) taken from (x - dx, y - dy)."""
    Mt = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, Mt, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderValue=np.nan)


def _star_mask(tree, x, y, gd, i, xc, yc, R, shape, gmax=17.0, k=3.0):
    """Discs over the stars (brighter than gmax) of the cutout centred on star i (i = None: on xc, yc)."""
    msk = np.zeros(shape, np.uint8)
    ctr = [xc, yc] if i is None else [x[i], y[i]]
    for j in tree.query_ball_point(ctr, R * 1.42):
        if j != i and gd[j] < gmax:
            cv2.circle(msk, (int(round(x[j] - xc + R)), int(round(y[j] - yc + R))),
                       int(np.clip(3 + k * (gmax - gd[j]), 3, 60)), 1, -1)
    return msk > 0


def field_sky(img, block=32, degree=4):
    """Smooth sky of the whole field per channel: polynomial of the given degree fitted on block medians, objects
    brighter than the sky clipped iteratively (one-sided: nebulae and residual starlight are positive)."""
    C, H, W = img.shape
    nby, nbx = H // block, W // block
    by, bx = (np.arange(nby) + 0.5) * block, (np.arange(nbx) + 0.5) * block
    v_, u_ = np.meshgrid(by / H * 2 - 1, bx / W * 2 - 1, indexing="ij")
    terms = [u_ ** i * v_ ** j for i in range(degree + 1) for j in range(degree + 1 - i)]
    A = np.stack([t.ravel() for t in terms], 1)
    out = np.empty_like(img)
    for c in range(C):
        b = np.median(img[c][:nby * block, :nbx * block].reshape(nby, block, nbx, block), axis=(1, 3)).ravel()
        ok = np.isfinite(b)
        for _ in range(8):
            q = np.linalg.lstsq(A[ok], b[ok], rcond=None)[0]
            r = b - A @ q
            s = 1.4826 * np.median(np.abs(r[ok] - np.median(r[ok])))
            ok = np.isfinite(b) & (r < 2.0 * s) & (r > -4.0 * s)
        grid = (A @ q).reshape(nby, nbx).astype(np.float32)
        out[c] = cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)
    return out


def _non_increasing(v):
    """Least-squares non-increasing fit of a sequence (pool adjacent violators): a dip is averaged with its
    neighbours instead of capping everything beyond it."""
    blocks = []                                   # [sum, count]
    for val in np.asarray(v, np.float64):
        blocks.append([val, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s_, n_ = blocks.pop()
            blocks[-1][0] += s_
            blocks[-1][1] += n_
    return np.concatenate([np.full(n_, s_ / n_) for s_, n_ in blocks])


def _ring_plane(v, ring, xx, yy):
    """Plane a + b x + c y fitted on the ring pixels of v (one 3-sigma clipping pass); the median if too few."""
    z = v[ring]
    if z.size < 30:
        return float(np.median(v[ring])) if z.size else 0.0
    A = np.c_[np.ones(z.size), xx[ring], yy[ring]].astype(np.float64)
    ok = np.ones(z.size, bool)
    for _ in range(2):
        coef = np.linalg.lstsq(A[ok], z[ok], rcond=None)[0]
        d = z - A @ coef
        s_ = 1.4826 * np.median(np.abs(d[ok] - np.median(d[ok])))
        ok = np.abs(d) < 3 * max(s_, 1e-12)
        if ok.sum() < 30:
            return float(np.median(z))
    return (coef[0] + coef[1] * xx + coef[2] * yy).astype(np.float32)


def _template(img, res_src, x, y, gd, flux, idx, R, tree, gmask, subpixel):
    """Median of the residual cutouts around the stars ``idx``, background removed, other stars masked, centred
    to the sub-pixel when ``subpixel``, divided by the flux of the star: (3, 2R+1, 2R+1).

    The background of each cutout is a plane fitted on its outer ring: over a large radius the sky is not flat (the
    Milky Way, a nebula), and a gradient shared by the field would not average out over the stars - it would end up
    in the template and be subtracted around every bright star."""
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy)
    outer = rr > R - 4
    cuts = np.full((len(idx), 3, 2 * R + 1, 2 * R + 1), np.nan, np.float32)
    for n, i in enumerate(idx):
        xc, yc = int(round(x[i])), int(round(y[i]))
        win = np.s_[:, yc - R:yc + R + 1, xc - R:xc + R + 1]
        res = res_src(win)
        msk = _star_mask(tree, x, y, gd, i, xc, yc, R, rr.shape, gmask)
        ring = outer & ~msk
        for c in range(3):
            v = res[c].astype(np.float32)
            v = v - _ring_plane(v, ring, xx, yy)
            v[msk] = np.nan
            if subpixel:
                v = _shift(v, xc - x[i], yc - y[i])
            cuts[n, c] = v / max(float(flux[c][i]), 1e-6)
    return np.nan_to_num(np.nanmedian(cuts, axis=0)), rr


def _add(model, tpl, x, y, scale, idx, subpixel):
    """Add scale[c, i] x template at the position of every star of ``idx`` (in place)."""
    H, W = model.shape[1:]
    R = tpl.shape[1] // 2
    for i in idx:
        xc, yc = int(round(x[i])), int(round(y[i]))
        ya, yb, xa, xb = max(0, yc - R), min(H, yc + R + 1), max(0, xc - R), min(W, xc + R + 1)
        if ya >= yb or xa >= xb:
            continue
        ty, tx = slice(ya - (yc - R), yb - (yc - R)), slice(xa - (xc - R), xb - (xc - R))
        for c in range(3):
            t = _shift(tpl[c], x[i] - xc, y[i] - yc) if subpixel else tpl[c]
            model[c, ya:yb, xa:xb] += float(scale[c, i]) * np.nan_to_num(t[ty, tx])


def extended_psf(img, model, x, y, g, Fp, fluxes, dm, W, H, radius, log=print, rmax=400):
    """Empirical correction of the star model, measured on the data, in two parts.

    1. Intermediate PSF (24 px): the residuals (image - model) around hundreds of unsaturated isolated stars,
       centred to the sub-pixel and divided by their measured flux, are median-combined. It holds what the core
       model misses (wider seeing halo, aberrations) and is added around every star where it exceeds the noise.
    2. Extended PSF (halo, diffraction spikes, reflections): the same on bright isolated stars, out to the radius
       where their average halo falls into the noise; its amplitude is fitted star by star (saturated cores make the
       residual not exactly proportional to the flux).
    Returns (correction model (3, H, W) or None, info).
    """
    from scipy.spatial import cKDTree
    gd = g - dm
    hp = img[1] - cv2.GaussianBlur(img[1], (0, 0), 2)
    sig = float(1.4826 * np.median(np.abs(hp[::7, ::7])))
    tree = cKDTree(np.c_[x, y])
    corr = np.zeros_like(img)
    info = {}
    # ---- 1. intermediate PSF
    Rm = 24
    cand = np.where((gd > 9.5) & (gd < 13.0) & (x > Rm + 2) & (x < W - Rm - 2) & (y > Rm + 2) & (y < H - Rm - 2)
                    & (fluxes[1] > 0))[0]
    iso = np.array([i for i in cand if all(j == i or gd[j] > gd[i] + 3 for j in tree.query_ball_point([x[i], y[i]], Rm))],
                   int)[:600]
    if len(iso) >= 30:
        tmid, rr = _template(img, lambda w: img[w] - model[w], x, y, gd, fluxes, iso, Rm, tree, 17.0, True)
        # the core is the photometric fit's job (it would be counted twice): from 4 px out, smoothly
        tmid *= (np.clip((Rm - 2 - rr) / 4.0, 0, 1) * np.clip((rr - 3.0) / 2.0, 0, 1))[None].astype(np.float32)
        peak = np.abs(tmid[1]).max()
        use = np.where(fluxes[1] * peak > 0.3 * sig)[0]
        _add(corr, tmid, x, y, fluxes, use, True)
        info["mid_psf_stars"] = int(len(iso))
        info["mid_psf_applied"] = int(len(use))
        log(f"intermediate PSF measured on {len(iso)} isolated stars, added around {len(use)} stars")
    # ---- 2. extended PSF
    R0 = rmax
    cand = np.where((gd > 5.0) & (gd < 9.5) & (x > R0) & (x < W - R0) & (y > R0) & (y < H - R0))[0]
    if len(cand) < 8:
        R0 = rmax // 2
        cand = np.where((gd > 5.0) & (gd < 10.0) & (x > R0) & (x < W - R0) & (y > R0) & (y < H - R0))[0]
    iso = [i for i in cand if all(j == i or gd[j] > gd[i] + 1.5 for j in tree.query_ball_point([x[i], y[i]], R0 * 0.6))]
    iso = np.array(sorted(iso, key=lambda i: gd[i])[:120], int)
    if len(iso) < 6:
        info["extended_psf"] = f"not measured: only {len(iso)} bright isolated stars"
        return (corr if info.get("mid_psf_stars") else None), info
    # the sky of the field is curved (vignetting residuals, light pollution): over a radius of hundreds of pixels a
    # plane per cutout is not enough, the curvature would look like a (negative) halo around every star. A smooth
    # sky of the whole field is removed first (measured on blocks, stars subtracted, nebulae clipped)
    sky = field_sky(img - model)
    text, rr = _template(img, lambda w: img[w] - sky[w] - model[w] - corr[w], x, y, gd, Fp, iso, R0, tree, 16.0, False)
    f_ref = float(np.median([Fp[1][i] for i in iso[:5]]))
    rb = np.arange(0, R0, 4)
    prof = np.array([np.mean(text[1][(rr >= r) & (rr < r + 4)]) for r in rb]) * f_ref
    noise_r = sig / np.sqrt(np.maximum(2 * np.pi * (rb + 2) * 4 / 9, 1))
    above = np.where(prof > 3 * noise_r)[0]
    Rt = int(radius) if radius > 0 else int(np.clip(rb[above.max()] + 8 if len(above) else 40, 30, R0))
    r_in = 8.0
    text = text * (np.clip((Rt - rr) / max(0.25 * Rt, 1), 0, 1) * np.clip((rr - r_in) / 4.0, 0, 1))[None]
    far = np.clip((rr - 30) / 30, 0, 1).astype(np.float32)
    text = (text * (1 - far) + np.stack([cv2.GaussianBlur(t, (0, 0), 1.5) for t in text]) * far).astype(np.float32)
    text = text[:, R0 - Rt:R0 + Rt + 1, R0 - Rt:R0 + Rt + 1]
    rr_t = rr[R0 - Rt:R0 + Rt + 1, R0 - Rt:R0 + Rt + 1]
    peak = np.abs(text[1][(rr_t > r_in + 4) & (rr_t < r_in + 12)]).mean()
    use = np.where(Fp[1] * peak > 0.3 * sig)[0]
    # the optical halo of a star is proportional to its flux (scattering, diffraction): the template (spikes, ghosts,
    # radial halo) is subtracted with the predicted flux of each star, so that light in excess of the instrument's
    # halo (a reflection nebula around the star) stays. Near the core (< 30 px) the radial part depends on
    # saturation and is left to the per-star measurement of the starless stage; the non-radial part is kept there.
    rb1 = np.round(rr_t).astype(int)
    w_rad = np.clip((rr_t - 10.0) / 8.0, 0, 1).astype(np.float32)
    order = np.argsort(rb1.ravel(), kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(rb1.ravel()))])
    meds = []
    for c in range(3):
        vals = text[c].ravel()[order]
        meds.append(np.array([np.median(vals[bounds[k]:bounds[k + 1]]) if bounds[k + 1] > bounds[k] else 0.0
                              for k in range(len(bounds) - 1)]))
    # scattered light is positive and fades with the distance: beyond the maximum of the radial part (12-40 px, the
    # same radius for the three channels so that the colour is kept) it is replaced by its closest non-increasing,
    # non-negative profile (what is left of the measurement noise would otherwise be subtracted as light around
    # every bright star - a negative halo adds a bright disc)
    tot = np.sum(meds, axis=0)
    k0 = min(12 + int(np.argmax(tot[12:41])), len(tot) - 1) if len(tot) > 41 else min(12, len(tot) - 1)
    for c in range(3):
        v = text[c]
        med = meds[c]
        phys = med.copy()
        phys[k0:] = np.clip(_non_increasing(med[k0:]), 0, None)
        # the rest (spikes, rings, reflections) is kept with its point symmetry only: the odd part of a template is
        # what remains of the sky gradients of the cutouts, not light of the instrument
        nr = v - med[rb1]
        nr = 0.5 * (nr + nr[::-1, ::-1])
        text[c] = nr + phys[rb1] * w_rad
    # amplitude per star and per channel, measured on ring medians beyond 50 px and capped: never more than the
    # instrument's halo (Gaia fluxes of very bright stars are unreliable; light in excess is nebula, kept; the
    # colour of the scattered light may differ from the star's). (One amplitude for the three channels was tried:
    # it leaves large coloured halos around the brightest stars, worse than the small coloured residual this leaves
    # around medium ones.)
    rings = np.arange(50, min(Rt, 220), 6)
    ridx = [(rr_t >= r) & (rr_t < r + 6) for r in rings]
    t_r = np.array([[np.median(text[c][m]) for m in ridx] for c in range(3)])
    amp = np.ones((3, len(x)))
    nfit = 0
    for i in use:
        if gd[i] > 10.5:
            continue
        xc, yc = int(round(x[i])), int(round(y[i]))
        if xc - Rt < 0 or yc - Rt < 0 or xc + Rt >= W or yc + Rt >= H:
            continue
        win = np.s_[yc - Rt:yc + Rt + 1, xc - Rt:xc + Rt + 1]
        free = ~_star_mask(tree, x, y, gd, i, xc, yc, Rt, rr_t.shape, 16.0)
        for c in range(3):
            res = img[c][win] - sky[c][win] - model[c][win] - corr[c][win]
            bk = np.median(res[(rr_t > Rt - 6) & free])
            r_r = np.array([np.median(res[m & free]) - bk if (m & free).sum() > 30 else np.nan for m in ridx])
            tt = t_r[c] * float(Fp[c][i])
            ok = np.isfinite(r_r) & (np.abs(tt) > sig / 3)
            if ok.sum() >= 4:                     # median ratio over the rings: robust to a ring of nebula
                amp[c, i] = float(np.clip(np.median(r_r[ok] / tt[ok]), 0.0, 1.3))
        nfit += 1
    _add(corr, text, x, y, Fp * amp, use, False)
    info.update({"extended_psf_stars": int(len(iso)), "extended_psf_radius": Rt, "extended_psf_applied": int(len(use)),
                 "_template": text})
    log(f"extended PSF measured on {len(iso)} bright isolated stars, radius {Rt} px (spikes, ghosts, halo beyond "
        f"10 px), subtracted around {len(use)} stars ({nfit} amplitudes measured, median "
        f"{np.median(amp[1][use]):.2f})")
    return corr, info


def star_qc(g, fl, pr):
    m = g < 17
    s = fl[0, m] / np.maximum(pr[0, m], 1e-6)
    out = {}
    for c, n in ((1, "green"), (2, "blue")):
        q = fl[c, m] / np.maximum(s * pr[c, m], 1e-6)
        out[f"collapsed_{n}"] = int((q < 0.35).sum())
        out[f"excess_{n}"] = int(((q > 3) & (g[m] > 13)).sum())
    return out


def faint_model(f, x, y, g, col, Fp, pm, W, H, g_deep, dev):
    """Faint stars (not fitted): flux from G with the mean colour of the field, rendered with the real PSF."""
    fx, fy, fg = f["x"].astype(np.float64), f["y"].astype(np.float64), f["g"].astype(np.float64)
    ok = np.isfinite(fg) & (fx > -8) & (fx < W + 8) & (fy > -8) & (fy < H + 8)
    fx, fy, fg = fx[ok], fy[ok], fg[ok]
    m = (g > g_deep - 1.0) & (g < g_deep)
    col0 = float(np.median(col[m])) if m.any() else COL0
    model = np.zeros((3, H, W), np.float32)
    flux = np.zeros((3, len(fx)), np.float32)
    for c in range(3):
        kc = float(np.median(Fp[c, m] / 10 ** (-0.4 * g[m]))) if m.any() else 0.0
        fc = (kc * 10 ** (-0.4 * fg)).astype(np.float32)
        flux[c] = fc
        E = pm["E"][c].to(dev)
        out = torch.zeros(H * W, device=dev)
        for k in range(0, len(fx), 200000):
            sl = slice(k, k + 200000)
            Pst, xi, yi = stamps(E, fx[sl], fy[sl], np.full(len(fx[sl]), col0), 5, W, H, dev)
            Group(Pst, xi, yi, 5, W, H, dev).render(torch.from_numpy(fc[sl]).to(dev), out)
            del Pst
        model[c] = out.reshape(H, W).cpu().numpy()
    return model, {"x": fx, "y": fy, "g": fg, "flux": flux, "col0": col0}
