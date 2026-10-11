"""Render preparation: the heavy, rarely changed part of the rendering, computed once.

Produces the calibrated nebula layer (fine detail injected, emission / continuum separated, star footprints
guarded against artefacts, faint areas smoothed at the scale confirmed by the data), the star tables and the real
radial profile of the stars. The interactive rendering engine works from these files.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy.ndimage import median_filter

from ..core.params import ADVANCED, P, StageParams
from ..core.stage import Context, Stage
from .optics import basis, sample_E


class RenderPrepParams(StageParams):
    cover_min: float = P(0.6, label="Edge coverage", level=ADVANCED, ge=0.1, le=1.0, step=0.05,
                         help="The frame is cropped where fewer than this fraction of the typical number of frames "
                              "overlap (edges uncovered because of the drift between frames).")
    edge_frac: float = P(0.995, label="Edge cleanliness", level=ADVANCED, ge=0.9, le=1.0, step=0.001,
                         help="Edges are trimmed until this fraction of every border row/column is well covered.")
    fine_k: float = P(1.0, label="Fine nebula detail", ge=0.0, le=2.0, step=0.05,
                      help="Re-injects the finest scales of the Noise2Noise image in the bright emission, where the "
                           "network keeps filaments that the Wiener filter smooths.",
                      tip="0 = only the Wiener result (smoothest), 1 = full detail.")
    fine_sigma: float = P(3.0, label="Fine detail scale", level=ADVANCED, ge=0.5, le=10, unit="px",
                          help="Scales below this are taken from the Noise2Noise image.")
    fine_lo_pct: float = P(80, label="Fine detail start", level=ADVANCED, ge=0, le=100, unit="percentile",
                           help="Emission brightness (percentile) where detail injection starts.")
    fine_hi_pct: float = P(95, label="Fine detail full", level=ADVANCED, ge=0, le=100, unit="percentile",
                           help="Emission brightness (percentile) where detail injection is complete.")
    cont_smooth: float = P(24.0, label="Continuum smoothing", level=ADVANCED, ge=0, le=200, unit="px",
                           help="The nebula is split per pixel into emission (H-alpha signature) and continuum "
                                "(starlight colour). The continuum's small scales are star residuals and are "
                                "smoothed at this scale; the emission keeps full resolution. 0 disables the split.")
    resid_smooth: float = P(6.0, label="Residual colour smoothing", level=ADVANCED, ge=0, le=50, unit="px",
                            help="Smoothing of what is neither H-alpha nor starlight (reflection, OIII).")
    star_guard: bool = P(True, label="Star artefact guard",
                         help="At star positions, emission may not dip below its local level and compact colour "
                              "excesses are pulled back: an imperfect star subtraction can create neither a black "
                              "hole nor a coloured dot. Real dark globules (which darken emission and continuum "
                              "together) are preserved.")
    guard_sigma: float = P(1.0, label="Guard tolerance", level=ADVANCED, ge=0, le=5, step=0.1, unit="sigma",
                           help="How far below the local emission level a star footprint may go.")
    faint_smooth: float = P(32.0, label="Faint area smoothing", ge=0, le=200, unit="px",
                            help="In faint areas, structures smaller than this are not confirmed by the data (noise "
                                 "and residuals) and are smoothed; bright nebulosity keeps full resolution.",
                            tip="Lower values show more fine structure in the faint background, with more mottle.")
    faint_lo_pct: float = P(55, label="Faint area limit", level=ADVANCED, ge=0, le=100, unit="percentile",
                            help="Below this emission percentile the faint-area smoothing is complete.")
    faint_hi_pct: float = P(85, label="Bright area limit", level=ADVANCED, ge=0, le=100, unit="percentile",
                            help="Above this emission percentile the image keeps full resolution.")


class RenderPrep(Stage):
    name = "render_prep"
    title = "Rendering preparation"
    description = ("Prepares the layers used by the interactive rendering: the calibrated nebula with its emission "
                   "and continuum separated, protected against star-subtraction artefacts, with faint areas smoothed "
                   "only at scales the data does not confirm; the table of stars to draw with their measured fluxes "
                   "and colours; and the real radial profile of your stars (core + wings) used to redraw them.")
    deps = ("background", "starless", "stars", "color", "optics", "stack")
    Params = RenderPrepParams
    version = "1"

    def run(self, ctx: Context, p: RenderPrepParams) -> None:
        dm = float(ctx.metric_of("optics", "mag_offset", 0.0))
        S = float(ctx.load_json("stack", "drz_meta.json")["scale"])
        wt = ctx.load("stack", "drz_weight.npy", mmap=True)
        cov = np.asarray(wt[1], np.float32)
        good = cov > p.cover_min * np.median(cov[cov > 0])
        H, W = good.shape
        y0, y1, x0, x1 = crop_box(good, p.edge_frac)
        ctx.metric("crop", [int(x0), int(x1), int(y0), int(y1)])
        ctx.log(f"useful frame x {x0}-{x1}, y {y0}-{y1} ({100 * (y1 - y0) * (x1 - x0) / (H * W):.1f} % of the stack)")
        astro = ctx.load("background", "astro.npy").astype(np.float32)
        M = ctx.load("color", "M.npy")
        fs = np.load(ctx.input("stars", "star_fluxes.npz"))
        flat = ctx.load("background", "flat.npy")
        coefs = ctx.load_json("background", "coefs.json")["fit"]
        spec = np.load(ctx.input("color", "specresp.npz"))
        if coefs["0"]["k_ha"] is not None:
            e_ = np.array([coefs[str(c)]["k_ha"] for c in range(3)])
        else:
            e_ = spec["cam_hii"].astype(np.float64)
        e_ = e_ / e_[0]
        ctx.progress(0.05, "fine detail")
        if p.fine_k > 0:
            n2 = ctx.load("starless", "n2n_final.npy").astype(np.float32) / flat
            b0 = ctx.load("starless", "sl_final.npy").astype(np.float32) / flat
            det = np.stack([(n2[c] - cv2.GaussianBlur(n2[c], (0, 0), p.fine_sigma))
                            - (b0[c] - cv2.GaussianBlur(b0[c], (0, 0), p.fine_sigma)) for c in range(3)])
            del n2, b0
            Eg0 = cv2.GaussianBlur(astro[0] - 0.87 * astro[1], (0, 0), 8)
            q1, q2 = np.percentile(Eg0[good], [p.fine_lo_pct, p.fine_hi_pct])
            wf = cv2.GaussianBlur(np.clip((Eg0 - q1) / max(q2 - q1, 1e-6), 0, 1).astype(np.float32), (0, 0), 6)
            astro = astro + p.fine_k * wf[None] * det
            del det
        if p.cont_smooth > 0:
            ctx.progress(0.2, "emission / continuum separation")
            mk = (fs["g"] - dm > 11) & (fs["g"] - dm < 14)
            k_ = np.array([np.median(fs["flux"][c, mk] / np.maximum(fs["flux"][1, mk], 1e-3)) for c in range(3)])
            Am = np.stack([e_, k_], 1)
            EC = np.tensordot(np.linalg.pinv(Am), astro, 1)
            resid = astro - np.tensordot(Am, EC, 1)
            Cs = cv2.GaussianBlur(EC[1], (0, 0), p.cont_smooth)
            rs_ = np.stack([cv2.GaussianBlur(c, (0, 0), p.resid_smooth) for c in resid]) if p.resid_smooth > 0 else resid
            Ef = EC[0]
            if p.star_guard:
                ctx.progress(0.3, "star artefact guard")
                Ef, rs_, info = guard(ctx, EC, rs_, fs, dm, H, W, p.guard_sigma)
                for k, v in info.items():
                    ctx.metric(k, v)
            if p.faint_smooth > 0:
                ctx.progress(0.6, "faint area smoothing")
                Esm = cv2.GaussianBlur(Ef, (0, 0), p.faint_smooth)
                Eg = cv2.GaussianBlur(Ef, (0, 0), 8)
                q1, q2 = np.percentile(Eg[good], [p.faint_lo_pct, p.faint_hi_pct])
                wE = cv2.GaussianBlur(np.clip((Eg - q1) / max(q2 - q1, 1e-6), 0, 1).astype(np.float32), (0, 0), 6)
                Ef = wE * Ef + (1 - wE) * Esm
                rs_ = wE[None] * rs_ + (1 - wE[None]) * np.stack([cv2.GaussianBlur(c, (0, 0), p.faint_smooth)
                                                                  for c in resid])
            astro = np.tensordot(Am, np.stack([Ef, Cs]), 1) + rs_
            ctx.metric("emission_signature", np.round(e_, 3).tolist())
            ctx.metric("continuum_colour", np.round(k_, 3).tolist())
        ctx.save("astro_prep.npy", astro.astype(np.float32))
        np.save(ctx.out("valid.npy"), good)
        ctx.progress(0.75, "star tables")
        mm_ = (fs["g"] - dm > 11) & (fs["g"] - dm < 14)
        alpha = float(np.median(fs["flux"][0, mm_] / np.maximum(fs["flux"][1, mm_], 1e-3)))
        vh = np.linalg.inv(M) @ (M @ (spec["cam_hii"] / spec["cam_hii"][0]))
        ff = np.load(ctx.input("stars", "faint_fluxes.npz"))
        X = np.concatenate([fs["x"], ff["x"]])
        Y = np.concatenate([fs["y"], ff["y"]])
        G = np.concatenate([fs["g"], ff["g"]]) - dm
        F = np.concatenate([fs["flux"].astype(np.float64), ff["flux"].astype(np.float64)], 1)
        Fpred = np.concatenate([fs["pred"].astype(np.float64), ff["flux"].astype(np.float64)], 1)
        nb = len(fs["g"])
        xi, yi = np.clip(np.round(X).astype(int), 0, W - 1), np.clip(np.round(Y).astype(int), 0, H - 1)
        gb = fs["g"] - dm
        for c in range(3):
            F[c] /= flat[c][yi, xi]
            mm2 = (gb > 15.5) & (gb < 17.5)
            Fpred[c, :nb] *= np.median(fs["flux"][c, mm2]) / np.median(fs["pred"][c, mm2])
            Fpred[c] /= flat[c][yi, xi]
        # prediction-based colour for faint / saturated stars, scaled to measured fluxes (G 11-13)
        pr = np.concatenate([fs["pred"].astype(np.float64), ff["flux"].astype(np.float64)], 1)
        m13 = (gb > 11) & (gb < 13)
        for c in range(3):
            pr[c] /= np.median(fs["pred"][c][m13] / np.maximum(fs["flux"][c][m13], 1e-9))
        np.savez(ctx.out("stars.npz"), x=X, y=Y, g=G, flux=F.astype(np.float32), pred=pr.astype(np.float32),
                 glow_flux=Fpred.astype(np.float32), n_measured=nb)
        ctx.progress(0.85, "real star profile")
        r_all, prof = star_profile(torch.load(ctx.input("optics", "psf_model.pt"), weights_only=False))
        ctx.save_json("prep.json", {"crop": [int(x0), int(x1), int(y0), int(y1)], "alpha": alpha,
                                    "vh": vh.tolist(), "M": M.tolist(), "drizzle_scale": S, "width": W,
                                    "height": H, "mag_offset": dm, "e": e_.tolist()})
        np.savez(ctx.out("profile.npz"), r=r_all, prof=prof)
        from ..core.imgutil import asinh_preview
        ctx.preview("nebula_layer", asinh_preview(astro, M, k=30))


def crop_box(good: np.ndarray, edge_frac: float):
    """Trim the least covered border until the four borders are well covered."""
    H, W = good.shape
    y0, y1, x0, x1 = 0, H, 0, W
    while y1 - y0 > 10 and x1 - x0 > 10:
        sub = good[y0:y1, x0:x1]
        e = {"t": sub[0].mean(), "b": sub[-1].mean(), "l": sub[:, 0].mean(), "r": sub[:, -1].mean()}
        k_, v_ = min(e.items(), key=lambda kv: kv[1])
        if v_ >= edge_frac:
            break
        if k_ == "t":
            y0 += 1
        elif k_ == "b":
            y1 -= 1
        elif k_ == "l":
            x0 += 1
        else:
            x1 -= 1
    return y0, y1, x0, x1


def guard(ctx, EC, rs_, fs, dm, H, W, guard_sigma, art_k=2.5, rsig=8.0, tol_k=1.5):
    """Emission floor and compact colour clamp on star footprints and on the artefact signature.

    Footprints: a disc around every medium / faint star (11.5 < G < 17.5), every pixel showing the artefact
    signature (compact emission dip together with a compact continuum excess) and every filled core.
    """
    disc_ = np.zeros((H, W), np.uint8)
    gx_b, gy_b, gg_b = fs["x"], fs["y"], fs["g"] - dm
    for i_ in np.where((gg_b > 11.5) & (gg_b < 17.5) & (gx_b > -5) & (gx_b < W + 5) & (gy_b > -5) & (gy_b < H + 5))[0]:
        cv2.circle(disc_, (int(round(gx_b[i_])), int(round(gy_b[i_]))), int(round(2.5 + 0.7 * (17.5 - gg_b[i_]))), 1, -1)
    foot = disc_ > 0
    Er, Cr = EC[0], EC[1]
    Ed = cv2.GaussianBlur(Er, (0, 0), 1.0) - median_filter(Er, size=15)
    Cd = cv2.GaussianBlur(Cr, (0, 0), 1.0) - median_filter(Cr, size=15)
    sEd = 1.4826 * np.median(np.abs(Ed[::5, ::5]))
    sCd = 1.4826 * np.median(np.abs(Cd[::5, ::5]))
    art = (Ed < -art_k * sEd) & (Cd > art_k * sCd)
    art = cv2.dilate(art.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    msk_ = ctx.load("starless", "sl_mask.npy")
    msk_ = cv2.dilate(msk_.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    foot = foot | art | msk_

    def nfill(img):
        hh_, ww_ = img.shape
        b_ = cv2.resize(img, (ww_ // 2, hh_ // 2), interpolation=cv2.INTER_AREA)
        b_ = median_filter(b_, size=9, mode="nearest")
        return cv2.resize(cv2.GaussianBlur(b_, (0, 0), 1.0), (ww_, hh_), interpolation=cv2.INTER_LINEAR)
    Ef = EC[0]
    E_loc = nfill(Ef)
    hpE = Ef - cv2.GaussianBlur(Ef, (0, 0), 2)
    sE = 1.4826 * np.median(np.abs(hpE[::7, ::7]))
    wf_ = cv2.GaussianBlur(foot.astype(np.float32), (0, 0), 1.0)
    Ef = Ef + wf_ * (np.maximum(Ef, E_loc - guard_sigma * sE) - Ef)
    r_loc = np.stack([cv2.GaussianBlur(c, (0, 0), rsig) for c in rs_])
    wa_ = cv2.GaussianBlur(art.astype(np.float32), (0, 0), 1.0)[None]
    rs_ = rs_ - wa_ * np.clip(rs_ - r_loc, 0, None)
    tol = np.array([1.4826 * np.median(np.abs((c - cv2.GaussianBlur(c, (0, 0), 2))[::7, ::7])) for c in rs_])[:, None, None]
    rs_c = np.clip(rs_, r_loc - tol_k * tol, r_loc + tol_k * tol)
    rs_ = rs_ + wf_[None] * (rs_c - rs_)
    return Ef, rs_, {"guard_artefact_pct": round(100 * float(art.mean()), 2),
                     "guard_footprint_pct": round(100 * float(foot.mean()), 1)}


def star_profile(pm):
    """Real azimuthally averaged star profile: PSF core (green, field centre) + measured wings, r = 0..60 px."""
    dev = pm["E"].device
    RC = 12
    rb = np.arange(0, RC + 0.01, 0.25)
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    dxp = torch.from_numpy((rb[:, None] * np.cos(th)[None]).astype(np.float32))[None].to(dev)
    dyp = torch.from_numpy((rb[:, None] * np.sin(th)[None]).astype(np.float32))[None].to(dev)
    z = torch.zeros(1, device=dev)
    polar = (sample_E(pm["E"][1].to(dev), dxp, dyp) * basis(z, z, z)[:, None, None, :]).sum(-1)[0].cpu().numpy()
    core = polar.mean(1)
    core = np.clip(core - core[-4:].mean(), 0, None)
    wr, wp = pm["wing_r"], pm["wing_prof"][1]
    k_ = np.interp(6, rb, core) / max(np.interp(6, wr, wp), 1e-12)
    r_all = np.arange(0, 61, 0.25)
    prof = np.where(r_all <= 6, np.interp(r_all, rb, core), k_ * np.interp(r_all, wr, wp))
    return r_all, np.clip(prof, 0, None)
