"""Render preparation: the heavy, rarely changed part of the rendering, computed once.

Produces the layers that the interactive engine mixes live: the calibrated nebula at full resolution with the star
footprints guarded against artefacts (whatever the colour of the object: emission, reflection, galaxy), the
fine-detail layer from the Noise2Noise image, the fine colour confirmed by both half-stacks, the optional colour
components (rest of the colour, OIII) used by the colour sliders; the star tables and the real radial profile of the
stars.
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
    min_exposure: float = P(0.15, label="Minimum effective exposure", level=ADVANCED, ge=0, le=1, step=0.01,
                            help="Coverage x vignetting^2 (the signal-to-noise squared, relative to the field centre) "
                                 "below which the image is not used: in strongly vignetted corners the noise is "
                                 "amplified and flat-field errors dominate. 0.15 cuts where the flat falls below ~40 % "
                                 "of the centre; instruments with mild vignetting are not affected.")
    fine_sigma: float = P(3.0, label="Fine detail scale", level=ADVANCED, ge=0.5, le=10, unit="px",
                          help="Scales below this are taken from the Noise2Noise image for the fine-detail layer "
                               "(its amount is set in the final adjustment tab).")
    resid_smooth: float = P(6.0, label="Residual colour smoothing", level=ADVANCED, ge=0, le=50, unit="px",
                            help="Smoothing of the colour component that is neither H-alpha nor starlight, used by the "
                                 "'Blue / reflection tones' slider.")
    star_guard: bool = P(True, label="Star artefact guard",
                         help="On the footprint of each star, the brightness may not dip below the local level and "
                              "the colour is taken from the surroundings: an imperfect star subtraction can create "
                              "neither a black hole nor a coloured dot. The brightness keeps full resolution.")
    guard_sigma: float = P(1.0, label="Guard tolerance", level=ADVANCED, ge=0, le=5, step=0.1, unit="sigma",
                           help="How far below the local brightness a star footprint may go.")
    oiii_scale: float = P(4.0, label="OIII map base smoothing", level=ADVANCED, ge=0, le=50, unit="px",
                          help="The OIII emission (teal, 496/501 nm) is separated from H-alpha and starlight with the "
                               "spectral response of your camera. With a colour camera the three signatures are close "
                               "and the separation is noisy: the map is stored smoothed at this scale (more smoothing "
                               "is set live in the final adjustment tab). 0 disables the OIII layer.")


class RenderPrep(Stage):
    name = "render_prep"
    title = "Rendering preparation"
    description = ("Prepares the layers used by the interactive rendering: the calibrated nebula at full resolution, "
                   "protected against star-subtraction artefacts; a fine-detail layer from the Noise2Noise image; the "
                   "fine colour confirmed by both half-stacks; the colour components used by the colour sliders "
                   "(OIII, reflection); the table of stars to draw with their measured fluxes and colours; and the "
                   "real radial profile of your stars. How much of each layer is shown is decided live in the final "
                   "adjustment tab.")
    deps = ("background", "starless", "stars", "color", "optics", "stack")
    Params = RenderPrepParams
    version = "7"          # 7: full-resolution base layer, colour-agnostic star guard (no emission/continuum split)

    def run(self, ctx: Context, p: RenderPrepParams) -> None:
        dm = float(ctx.metric_of("optics", "mag_offset", 0.0))
        S = float(ctx.load_json("stack", "drz_meta.json")["scale"])
        wt = ctx.load("stack", "drz_weight.npy", mmap=True)
        cov = np.asarray(wt[1], np.float32)
        good = cov > p.cover_min * np.median(cov[cov > 0])
        H, W = good.shape
        fp = ctx.project.stage_dir("calibrate") / "master_flat.npy"
        if p.min_exposure > 0 and fp.exists():
            fl = np.asarray(np.load(fp, mmap_mode="r"), np.float32)
            fl = cv2.resize(fl, (fl.shape[1] // 2, fl.shape[0] // 2), interpolation=cv2.INTER_AREA)   # Bayer cells
            fl = cv2.GaussianBlur(fl / np.percentile(fl, 99), (0, 0), 4)
            eff = cov / np.median(cov[cov > 0]) * cv2.resize(fl, (W, H), interpolation=cv2.INTER_LINEAR) ** 2
            vig = good & (eff < p.min_exposure)
            good &= ~vig
            ctx.metric("vignetted_pct", round(100 * float(vig.mean()), 2))
            if vig.any():
                ctx.log(f"{100 * vig.mean():.1f} % of the field left out: effective exposure below "
                        f"{p.min_exposure:.2f} (strong vignetting)")
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
        mk = (fs["g"] - dm > 11) & (fs["g"] - dm < 14)
        k_ = np.array([np.median(fs["flux"][c, mk] / np.maximum(fs["flux"][1, mk], 1e-3)) for c in range(3)])
        Am = np.stack([e_, k_], 1)
        lum = np.array([0.2126, 0.7152, 0.0722]) @ M                # camera -> display luminance
        w_cam = np.linalg.solve(M, np.ones(3))                       # camera vector of neutral light, luminance 1
        Yl = np.tensordot(lum, astro, 1).astype(np.float32)
        # brightness map gating the fine-detail injection and the faint-area blend: the whole signal, whatever its
        # colour (an H-alpha-only gate would treat reflection nebulae and galaxies as empty sky)
        ctx.save("Eg0.npy", cv2.GaussianBlur(Yl, (0, 0), 8).astype(np.float32))
        ctx.progress(0.1, "star footprints")
        foot_w = footprints(ctx, fs, dm, H, W) if p.star_guard else np.zeros((H, W), np.float32)
        ctx.metric("guard_footprint_pct", round(100 * float((foot_w > 0.5).mean()), 1))
        ctx.progress(0.25, "nebula layer")
        N = guard_generic(astro, Yl, foot_w, lum, w_cam, p.guard_sigma) if p.star_guard else astro
        ctx.save("N.npy", N.astype(np.float32))
        preview = N
        del Yl
        # optional colour components for the colour sliders: the rest (neither H-alpha nor starlight) and OIII
        ctx.progress(0.45, "colour components")
        EC = np.tensordot(np.linalg.pinv(Am), astro, 1)
        resid = astro - np.tensordot(Am, EC, 1)
        del EC
        R = np.stack([cv2.GaussianBlur(c, (0, 0), p.resid_smooth) for c in resid]) if p.resid_smooth > 0 else resid
        ctx.save("R.npy", (R * (1 - foot_w)[None]).astype(np.float32))
        del resid, R
        ctx.metric("emission_signature", np.round(e_, 3).tolist())
        ctx.metric("continuum_colour", np.round(k_, 3).tolist())
        o_ = spec["cam_oiii"].astype(np.float64)
        o_ = o_ / max(o_[1], 1e-9)
        B3 = np.stack([e_, o_, k_], 1)
        s_o = None
        if p.oiii_scale > 0 and np.linalg.cond(B3) < 200:
            ctx.progress(0.6, "OIII layer")
            O = cv2.GaussianBlur(np.tensordot(np.linalg.inv(B3)[1], astro, 1).astype(np.float32), (0, 0), p.oiii_scale)
            Eg_ = cv2.GaussianBlur(astro[0] - 0.87 * astro[1], (0, 0), 8)
            sky_ = good & (Eg_ < np.percentile(Eg_[good], 50))
            s_o = float(1.4826 * np.median(np.abs(O[sky_] - np.median(O[sky_]))))
            ctx.save("O.npy", (O * (1 - foot_w)).astype(np.float32))
            ctx.metric("oiii_noise", round(s_o, 3))
            ctx.metric("oiii_peak_snr", round(float(np.percentile(O[good], 99.9)) / max(s_o, 1e-9), 1))
            ctx.log(f"OIII layer: peak {np.percentile(O[good], 99.9) / max(s_o, 1e-9):.0f} sigma above the sky "
                    f"(condition number of the separation {np.linalg.cond(B3):.0f})")
            del O, Eg_
        o_disp = oiii_display_vector(M, o_)
        # fine-detail layer: Noise2Noise small scales minus Wiener small scales, kept as brightness (its colour is
        # noise), off on star footprints
        ctx.progress(0.65, "fine detail layer")
        n2 = ctx.load("starless", "n2n_final.npy").astype(np.float32) / flat
        b0 = ctx.load("starless", "sl_final.npy").astype(np.float32) / flat
        det = np.stack([(n2[c] - cv2.GaussianBlur(n2[c], (0, 0), p.fine_sigma))
                        - (b0[c] - cv2.GaussianBlur(b0[c], (0, 0), p.fine_sigma)) for c in range(3)])
        del n2, b0
        ctx.save("dY.npy", (np.tensordot(lum, det, 1) * (1 - foot_w)).astype(np.float16))
        del det
        cf = ctx.project.stage_dir("starless") / "chroma_fine.npy"
        if cf.exists():                           # star residuals are common to both halves: off on footprints too
            ctx.save("chroma_fine.npy", (np.load(cf).astype(np.float32) / flat * (1 - foot_w)[None]).astype(np.float16))
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
                                    "height": H, "mag_offset": dm, "e": e_.tolist(), "Am": Am.tolist(),
                                    "layers": 3, "w_cam": w_cam.tolist(), "o": o_.tolist(),
                                    "o_display": o_disp.tolist(), "oiii_sigma": s_o, "oiii_scale": p.oiii_scale})
        np.savez(ctx.out("profile.npz"), r=r_all, prof=prof)
        from ..core.imgutil import asinh_preview
        ctx.preview("nebula_layer", asinh_preview(preview, M, k=30))


def oiii_display_vector(M: np.ndarray, o_cam: np.ndarray) -> np.ndarray:
    """Camera-space vector whose display colour is the true colour of the [OIII] 495.9 + 500.7 nm lines.

    The colour of the doublet is outside the sRGB gamut; it is brought in by adding white (hue kept). The vector has
    the same luminance as the OIII signature of the camera, so a boost adds the brightness the OIII really has.
    """
    import colour

    from .color import XYZ_TO_SRGB
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    wl, v = cmfs.wavelengths, cmfs.values
    xyz = sum(w * np.array([np.interp(lam, wl, v[:, i]) for i in range(3)]) for lam, w in ((500.7, 1.0), (495.9, 0.33)))
    rgb = XYZ_TO_SRGB @ xyz
    rgb = rgb - min(rgb.min(), 0.0)
    lum = np.array([0.2126, 0.7152, 0.0722])
    rgb = rgb / (lum @ rgb) * (lum @ (M @ o_cam))
    return np.linalg.solve(M, rgb)


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


def footprints(ctx, fs, dm, H, W) -> np.ndarray:
    """Weight (0..1) of the star footprints: a disc around every medium / faint star (11.5 < G < 17.5) and every
    filled star core of the starless stage."""
    disc_ = np.zeros((H, W), np.uint8)
    gx_b, gy_b, gg_b = fs["x"], fs["y"], fs["g"] - dm
    for i_ in np.where((gg_b > 11.5) & (gg_b < 17.5) & (gx_b > -5) & (gx_b < W + 5) & (gy_b > -5) & (gy_b < H + 5))[0]:
        cv2.circle(disc_, (int(round(gx_b[i_])), int(round(gy_b[i_]))), int(round(2.5 + 0.7 * (17.5 - gg_b[i_]))), 1, -1)
    msk_ = ctx.load("starless", "sl_mask.npy")
    msk_ = cv2.dilate(msk_.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    return cv2.GaussianBlur(((disc_ > 0) | msk_).astype(np.float32), (0, 0), 1.0)


def guard_generic(astro, Yl, foot_w, lum, w_cam, guard_sigma):
    """Star footprints of the calibrated nebula: the brightness keeps full resolution but may not dip below the local
    level by more than ``guard_sigma`` noise sigmas; the colour is the colour of the surroundings (footprints
    excluded). Works for any colour of the object: no emission / continuum hypothesis."""
    def nfill(img):
        hh_, ww_ = img.shape
        b_ = cv2.resize(img, (ww_ // 2, hh_ // 2), interpolation=cv2.INTER_AREA)
        b_ = median_filter(b_, size=9, mode="nearest")
        return cv2.resize(cv2.GaussianBlur(b_, (0, 0), 1.0), (ww_, hh_), interpolation=cv2.INTER_LINEAR)
    hp = Yl - cv2.GaussianBlur(Yl, (0, 0), 2)
    sY = 1.4826 * float(np.median(np.abs(hp[::7, ::7])))
    Yg = np.maximum(Yl, nfill(Yl) - guard_sigma * sY)
    kp = (foot_w < 0.5).astype(np.float32)
    den = np.maximum(cv2.GaussianBlur(kp, (0, 0), 6), 1e-3)
    As = np.stack([cv2.GaussianBlur(c * kp, (0, 0), 6) / den for c in astro])     # colour of the surroundings
    Ys = np.tensordot(lum, As, 1)
    Nfp = As + (Yg - Ys)[None] * np.asarray(w_cam, np.float32)[:, None, None]
    return (astro + foot_w[None] * (Nfp - astro)).astype(np.float32)


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
