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
    version = "1"

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
