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
    cont_smooth: float = P(24.0, label="Continuum smoothing under star light", level=ADVANCED, ge=0, le=60, unit="px",
                           help="Where the light of the stars dominates the signal, the fine structure of the "
                                "star-coloured (continuum) component is star residual: it is smoothed at this scale. "
                                "Where the nebula dominates, the continuum is kept at full resolution.")
    resid_smooth: float = P(6.0, label="Residual colour smoothing", level=ADVANCED, ge=0, le=50, unit="px",
                            help="Smoothing of the colour component that is neither H-alpha nor starlight, used by the "
                                 "'Blue / reflection tones' slider.")
    star_guard: bool = P(True, label="Star artefact guard",
                         help="On the footprint of each star, the brightness may not dip below the local level and "
                              "the colour is taken from the surroundings: an imperfect star subtraction can create "
                              "neither a black hole nor a coloured dot. The brightness keeps full resolution.")
    guard_sigma: float = P(3.0, label="Guard tolerance", level=ADVANCED, ge=0, le=10, step=0.1, unit="sigma",
                           help="How far below the brightness of the surroundings a star footprint may go before it is "
                                "lifted (deeper = a hole left by the star subtraction, not noise).")
    n2n_base: float = P(0.0, label="Noise2Noise in the base", level=ADVANCED, ge=0, le=1, step=0.05,
                        help="The fine scales of the nebula are taken from the Noise2Noise network trained on your data: "
                             "a local Wiener filter leaves islands of unfiltered pixel noise ('pepper') wherever its "
                             "local noise estimate falls short, the network does not. 0 = Wiener everywhere (the 'Fine "
                             "detail' slider then injects the network's detail).")
    n2n_scale: float = P(16.0, label="Noise2Noise up to", level=ADVANCED, ge=0, le=200, step=1, unit="px",
                         help="Scales smaller than this come from the network, larger ones from the Wiener filter (its "
                              "large-scale level is the best characterised). 0 = the network at every scale.")
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
    version = "26"         # 25: real star light without the far halo, the far halo as a layer of its own (glow);
                           # 26: footprint of a saturated star at least its unreliable core

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
        ctx.progress(0.2, "nebula layer")
        # two denoisers are available: the multi-scale Wiener filter and the Noise2Noise network trained on these data.
        # A local Wiener filter leaves islands of unfiltered pixel noise wherever its local noise estimate falls short
        # (dark and bright pepper, worst at the edges of a nebula); the network does not. The fine scales come from the
        # network, the large ones from the Wiener filter (bright-star zones are interpolated identically in both)
        wstar = star_light_weight(ctx, lum, astro.shape[1:]) * (1 - foot_w)
        ctx.metric("fine_layers_off_pct", round(100 * float((wstar < 0.5).mean()), 1))
        n2 = ctx.load("starless", "n2n_final.npy").astype(np.float32) / flat
        b0 = ctx.load("starless", "sl_final.npy").astype(np.float32) / flat
        base = astro.copy()
        for c in range(3):
            d = n2[c] - b0[c]
            if p.n2n_scale > 0:
                d -= cv2.GaussianBlur(d, (0, 0), p.n2n_scale)
            base[c] += p.n2n_base * d
        del d
        N = base
        # where the light of the stars dominates the signal (dense fields), the fine structure of the continuum
        # (star-coloured light) is star residual, not nebula: the continuum is smoothed there at cont_smooth, as the
        # first version of this stage did everywhere - but only where that is safe (see continuum_smoothing_allowed):
        # a structured continuum (reflection nebula, galaxy, dust) is the signal and is kept at full resolution.
        # This comes before the star guard: the guard gives the brightness it keeps on a star footprint a neutral
        # colour, i.e. star-coloured light plus a negative emission; smoothing the star-coloured part afterwards left
        # the negative part, so a bright residual came out as a dark, slightly coloured hole (the grit of Heart)
        if p.cont_smooth > 0:
            EC_N = np.tensordot(np.linalg.pinv(Am), N, 1)
            Cs = cv2.GaussianBlur(EC_N[1].astype(np.float32), (0, 0), p.cont_smooth)
            sm = (1 - wstar) * continuum_smoothing_allowed(EC_N[1].astype(np.float32), wstar)
            N = N - Am[:, 1][:, None, None] * ((EC_N[1] - Cs) * sm)[None]
            ctx.metric("continuum_smoothed_pct", round(100 * float((sm > 0.5).mean()), 1))
            del EC_N, Cs, sm
        if p.star_guard:
            N = guard_generic(N, np.tensordot(lum, N, 1).astype(np.float32), foot_w, lum, w_cam, p.guard_sigma)
        # saturated stars: shown by their real light; the nebula under them interpolated where that is safe
        ctx.progress(0.42, "real light of the saturated stars")
        N, fp_all, fp_fill, Sreal = saturated_star_light(ctx, fs, N, flat, lum, ctx.log)
        ctx.save("N.npy", N.astype(np.float32))
        preview = N
        del Yl, base
        # optional colour components for the colour sliders: the rest (neither H-alpha nor starlight) and OIII
        ctx.progress(0.45, "colour components")
        EC = np.tensordot(np.linalg.pinv(Am), astro, 1)
        resid = astro - np.tensordot(Am, EC, 1)
        del EC
        R = np.stack([cv2.GaussianBlur(c, (0, 0), p.resid_smooth) for c in resid]) if p.resid_smooth > 0 else resid
        ctx.save("R.npy", (R * ((1 - foot_w) * (1 - fp_fill))[None]).astype(np.float32))
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
            ctx.save("O.npy", (O * (1 - foot_w) * (1 - fp_fill)).astype(np.float32))
            ctx.metric("oiii_noise", round(s_o, 3))
            ctx.metric("oiii_peak_snr", round(float(np.percentile(O[good], 99.9)) / max(s_o, 1e-9), 1))
            ctx.log(f"OIII layer: peak {np.percentile(O[good], 99.9) / max(s_o, 1e-9):.0f} sigma above the sky "
                    f"(condition number of the separation {np.linalg.cond(B3):.0f})")
            del O, Eg_
        o_disp = oiii_display_vector(M, o_)
        # fine-detail layer: Noise2Noise small scales minus Wiener small scales, kept as brightness (its colour is
        # noise), off on star footprints
        ctx.progress(0.65, "fine detail layer")
        # with the network in the base, the layer holds its fine scales and the 'Fine detail' slider scales them
        # (1 = as measured); otherwise it holds network minus Wiener, injected into the Wiener base
        fine_scale = p.n2n_base > 0
        det = np.stack([(n2[c] - cv2.GaussianBlur(n2[c], (0, 0), p.fine_sigma))
                        - (0 if fine_scale else b0[c] - cv2.GaussianBlur(b0[c], (0, 0), p.fine_sigma))
                        for c in range(3)])
        del b0
        # the fine layers are not reliable where the light of the stars dominates (residual spikes, halo streaks)
        ctx.save("dY.npy", (np.tensordot(lum, det, 1) * wstar * (1 - fp_fill)).astype(np.float16))
        del det
        # sharpening increment: the sharpening network minus the plain one (brightness), off on star light
        sp = ctx.project.stage_dir("starless") / "n2n_sharp_final.npy"
        if sp.exists():
            n2s = np.load(sp).astype(np.float32) / flat
            ctx.save("dS.npy", (np.tensordot(lum, n2s - n2, 1) * wstar).astype(np.float16))
            del n2s
        del n2
        cf = ctx.project.stage_dir("starless") / "chroma_fine.npy"
        if cf.exists():                           # star residuals are common to both halves: off on star light too
            ctx.save("chroma_fine.npy", (np.load(cf).astype(np.float32) / flat * (wstar * (1 - fp_fill))[None])
                     .astype(np.float16))
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
        # stars inside the footprint of a saturated star are shown by its real light, not drawn again
        real = (fp_all[np.clip(np.round(Y).astype(int), 0, H - 1), np.clip(np.round(X).astype(int), 0, W - 1)] > 0.5)             & (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        # every star gets the part of its shape that a round drawn star lacks (diamond, diffraction spikes), measured
        # on the image by the star stage (flux x measured shape), outside the saturated stars' own light. A saturated
        # star's spikes continue its real light; a drawn star's shape is a layer of its own, drawn by the rendering
        # like the star's wings (same attenuation as the star size settings give them)
        shp = ctx.project.stage_dir("stars") / "star_shape.npz"
        if shp.exists() and ctx.metric_of("starless", "spike_directions", []):
            from .star_shape import render as render_shape
            sh = np.load(shp)
            rf = real[:len(fs["x"])]
            S_sh, n_r = render_shape(sh["E_render"], fs["x"], fs["y"], sh["flux"], H, W, float(sh["sig"]),
                                     idx=np.where(rf)[0])
            Sreal += S_sh / flat * (1 - fp_all)[None]
            S_sh, n_d = render_shape(sh["E_render"], fs["x"], fs["y"], sh["flux"], H, W, float(sh["sig"]),
                                     idx=np.where(~rf)[0])
            ctx.save("Sshape.npy", (S_sh / flat * (1 - fp_all)[None]).astype(np.float16))
            ctx.metric("stars_with_measured_shape", int(n_r + n_d))
            del S_sh, sh
        ctx.save("Sreal.npy", Sreal.astype(np.float16))
        hf = ctx.project.stage_dir("stars") / "stars_model_halo.npy"
        if hf.exists():    # the far halo of the stars, taken out of the nebula: the rendering draws it as their glow
            hal = np.load(hf, mmap_mode="r")
            ctx.save("Shalo.npy", np.stack([np.asarray(hal[c], np.float32) / flat[c] for c in range(3)]).astype(np.float16))
            del hal
        ctx.metric("stars_real_light", int(real.sum()))
        del Sreal
        np.savez(ctx.out("stars.npz"), x=X, y=Y, g=G, flux=F.astype(np.float32), pred=pr.astype(np.float32),
                 glow_flux=Fpred.astype(np.float32), n_measured=nb, real=real)
        # depth of the data: magnitude where the measured stars reach S/N = 10. The star settings of the rendering
        # (faintest drawn star, sizes, brightness) refer to a field where this is G = 15.5; deeper or finer-sampled
        # data shift them, so that "the faintest drawn star" means the same thing for every instrument
        depth = star_depth(ctx, fs, dm)
        depth_shift = float(np.clip(round(((depth if depth else 15.5) - 15.5) * 4) / 4, -3, 6))
        ctx.metric("star_depth_snr10", depth)
        ctx.metric("star_depth_shift", depth_shift)
        ctx.progress(0.85, "real star profile")
        r_all, prof = star_profile(torch.load(ctx.input("optics", "psf_model.pt"), weights_only=False))
        ctx.save_json("prep.json", {"crop": [int(x0), int(x1), int(y0), int(y1)], "alpha": alpha,
                                    "vh": vh.tolist(), "M": M.tolist(), "drizzle_scale": S, "width": W,
                                    "height": H, "mag_offset": dm, "e": e_.tolist(), "Am": Am.tolist(),
                                    "layers": 3, "w_cam": w_cam.tolist(), "o": o_.tolist(), "depth_shift": depth_shift,
                                    "o_display": o_disp.tolist(), "oiii_sigma": s_o, "oiii_scale": p.oiii_scale,
                                    "fine_layer": "scale" if fine_scale else "add"})
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


def star_depth(ctx, fs, dm) -> float | None:
    """Gaia G magnitude where the median S/N of the measured stars falls to 10 (aperture of radius one FWHM)."""
    try:
        S = float(ctx.load_json("stack", "drz_meta.json")["scale"])
        fwhm = float(ctx.metric_of("register", "fwhm_px")) * S
        img = ctx.load("optics", "lat_full.npy", mmap=True)
        a = np.asarray(img[1, ::2, ::2], np.float32)
        sig = 1.4826 * float(np.median(np.abs(a - cv2.GaussianBlur(a, (0, 0), 2))))
        snr = fs["flux"][1] / (sig * np.sqrt(np.pi * fwhm ** 2))
        g = fs["g"] - dm
        for b in np.arange(9.0, 21.0, 0.25):
            m = (g >= b) & (g < b + 0.25)
            if m.sum() > 20 and np.median(snr[m]) < 10:
                return float(b)
    except Exception:  # noqa: BLE001
        return None
    return None


def _ring_median(img, xc, yc, r_):
    H, W = img.shape
    a = np.linspace(0, 2 * np.pi, max(32, int(2 * np.pi * r_)), endpoint=False)
    px = np.round(xc + r_ * np.cos(a)).astype(int)
    py = np.round(yc + r_ * np.sin(a)).astype(int)
    m = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return float(np.median(img[py[m], px[m]])) if m.sum() >= 16 else np.nan


def saturated_star_light(ctx, fs, N, flat, lum, log=print, rmax=400):
    """Stars whose core is saturated cannot be measured nor modelled: when the instrument has diffraction spikes,
    which a drawn star cannot reproduce, they are shown by their real light (without spikes the drawn stars are
    kept: the real light of a saturated star is then only its blooming and halo).

    Footprint of such a star: out to where its light (image minus starless) falls below twice the noise. Under the
    footprint the nebula is
    interpolated from its edge (membrane fill: the level follows the edge) and everything else - core, halo,
    diffraction spikes - is star light, image minus nebula, as a manual star separation does. If the star sits in a
    structured nebula (band-pass structure around the footprint well above the flattest areas of the image), an
    interpolation would erase it: there the starless image is kept and the star light is image minus starless.
    Faint pixels of the star light fade out smoothly so that the noise of the image is not carried into the star
    layer. Returns the nebula base with the footprints interpolated, the footprint weight of every saturated star,
    the weight of the interpolated ones and the star light (3, H, W, camera space, divided by the flat).
    """
    from ..core.imgutil import membrane_fill
    H, W = flat.shape[1:]
    if not ctx.metric_of("starless", "spike_directions", []):
        # without diffraction spikes the drawing of a star loses nothing of it: every star is drawn
        log("no diffraction spikes in this instrument: every star is drawn from the tables")
        return N, np.zeros((H, W), np.float32), np.zeros((H, W), np.float32), np.zeros_like(N)
    full = ctx.load("optics", "lat_full.npy", mmap=True)
    sl = ctx.load("starless", "sl_final.npy", mmap=True)
    H, W = flat.shape[1:]
    xs, ys = fs["x"], fs["y"]
    ok = (xs > 2) & (xs < W - 3) & (ys > 2) & (ys < H - 3)
    xi, yi = np.round(xs[ok]).astype(int), np.round(ys[ok]).astype(int)
    sat = np.zeros(ok.sum(), bool)
    for c in range(3):
        a = np.asarray(full[c])
        lvl = np.percentile(a[::4, ::4], 99.999)
        v = np.max([a[yi + dy, xi + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 0)
        sat |= v > 0.7 * lvl
    idx_sat = np.where(ok)[0][sat]
    fp_all = np.zeros((H, W), np.float32)
    fp_fill = np.zeros((H, W), np.float32)
    if not len(idx_sat):
        return N, fp_all, fp_fill, np.zeros_like(N)
    slf = np.stack([np.asarray(sl[c], np.float32) / flat[c] for c in range(3)])
    S = np.stack([np.asarray(full[c], np.float32) / flat[c] for c in range(3)]) - slf
    hf = ctx.project.stage_dir("stars") / "stars_model_halo.npy"
    if hf.exists():        # the far halo of the stars is not drawn back: their real light is core, spikes and rays
        hal = np.load(hf, mmap_mode="r")
        S -= np.stack([np.asarray(hal[c], np.float32) / flat[c] for c in range(3)])
        del hal
    S = np.stack([cv2.GaussianBlur(S[c], (0, 0), 1.0) for c in range(3)])
    Ys = np.tensordot(lum, S, 1).astype(np.float32)
    YL = np.tensordot(lum, N, 1).astype(np.float32)
    hp = Ys[::3, ::3] - cv2.GaussianBlur(Ys[::3, ::3], (0, 0), 2)
    sig = 1.4826 * float(np.median(np.abs(hp - np.median(hp))))
    # structure of the nebula (band-pass 4-16 px) and its level in the flattest areas
    YN = YL[::2, ::2]
    bp2 = (cv2.GaussianBlur(YN, (0, 0), 1) - cv2.GaussianBlur(YN, (0, 0), 4)) ** 2
    floor = float(np.percentile(cv2.GaussianBlur(bp2, (0, 0), 24), 5))
    N2 = N.copy()
    n_fill = 0
    # with the far halo removed, the footprint must still cover the region where the star model cannot be trusted
    # (the core the far halo continues flat, x 1.3): else the core residue spills out of it and makes its
    # surroundings look structured, and the nebula is no longer interpolated under the star
    from .far_halo import WING_OUT
    from .starless import _filled_radius
    smask = ctx.load("starless", "sl_mask.npy", mmap=True) if hf.exists() else None
    for i in idx_sat[np.argsort(fs["g"][idx_sat])]:
        R = 4
        for r_ in range(4, rmax, 2):
            if not _ring_median(Ys, xs[i], ys[i], r_) >= 2 * sig:
                break
            R = r_
        if smask is not None:
            R = max(R, int(1.3 * max(WING_OUT, _filled_radius(np.asarray(smask), xs[i], ys[i], rmax=rmax) + 4)))
        R = int(R * 1.15) + 4
        x0_, x1_ = max(0, int(xs[i]) - R - 40), min(W, int(xs[i]) + R + 41)
        y0_, y1_ = max(0, int(ys[i]) - R - 40), min(H, int(ys[i]) + R + 41)
        yy, xx = np.mgrid[y0_:y1_, x0_:x1_]
        rr = np.hypot(xx - xs[i], yy - ys[i])
        w_ = np.clip((R - rr) / (0.15 * R), 0, 1).astype(np.float32)
        fp_all[y0_:y1_, x0_:x1_] = np.maximum(fp_all[y0_:y1_, x0_:x1_], w_)
        ann = (rr >= R) & (rr < R + 40)
        q = np.sqrt(float(np.median(bp2[np.minimum(yy[ann] // 2, bp2.shape[0] - 1),
                                        np.minimum(xx[ann] // 2, bp2.shape[1] - 1)])) / max(floor, 1e-12))
        if q < 2.5:
            hole = rr < R
            for c in range(3):
                win = N2[c, y0_:y1_, x0_:x1_]
                N2[c, y0_:y1_, x0_:x1_] = win * (1 - w_) + membrane_fill(win, hole) * w_
            fp_fill[y0_:y1_, x0_:x1_] = np.maximum(fp_fill[y0_:y1_, x0_:x1_], w_)
            n_fill += 1
    del Ys, YL
    # star light: image minus nebula (the interpolated nebula where it was interpolated, the starless elsewhere)
    Sr = S + np.stack([cv2.GaussianBlur(N[c] - N2[c], (0, 0), 1.0) for c in range(3)])
    del S, slf
    fade = np.clip((np.tensordot(lum, Sr, 1) / max(sig, 1e-9) - 1.5) / 2.0, 0, 1)
    Sr *= (fade * fp_all)[None]
    log(f"saturated stars: {len(idx_sat)} shown by their own light ({100 * float((fp_all > 0.5).mean()):.2f} % of the "
        f"field); nebula interpolated under {n_fill} of them, kept under the others (structured nebula around)")
    return N2, fp_all, fp_fill, Sr


def continuum_smoothing_allowed(C, wstar) -> np.ndarray:
    """Where smoothing the continuum C near the stars is safe (1) or would erase real structure (0).

    Safe where (a) the field is too dense to see the continuum between the stars - star light above the noise on
    most pixels: smoothing hides their residual mottling - or (b) between the stars the continuum is flat at the
    scale of the star residuals (4-16 px): its local band-pass power over the star-free pixels is compared with the
    flattest areas of the image. A continuum with structure of its own between the stars (a reflection nebula, a
    galaxy, dust) is the signal: smoothed discs around its stars would show as bubbles."""
    H, W = C.shape
    c2 = C[::2, ::2]
    free = (wstar[::2, ::2] > 0.9).astype(np.float32)
    bp = cv2.GaussianBlur(c2, (0, 0), 2) - cv2.GaussianBlur(c2, (0, 0), 8)
    den = cv2.GaussianBlur(free, (0, 0), 24)
    ok = den > 0.15
    if ok.mean() < 0.02:
        return np.ones((H, W), np.float32)
    pf = cv2.GaussianBlur(bp * bp * free, (0, 0), 24) / np.maximum(den, 1e-3)
    q = np.sqrt(pf / max(float(np.percentile(pf[ok], 5)), 1e-12))
    flat = np.where(ok, np.clip(2.5 - q, 0, 1), 1.0)
    dense = np.clip((0.4 - den) / 0.25, 0, 1)
    allow = cv2.GaussianBlur(np.maximum(flat, dense).astype(np.float32), (0, 0), 8)
    return cv2.resize(allow, (W, H), interpolation=cv2.INTER_LINEAR)


def star_light_weight(ctx, lum, shape) -> np.ndarray:
    """1 where the light of the stars (model, extended PSF, removed halos) is below 1 noise sigma of the brightness,
    0 above 3 sigmas. (Tried on 2026-10-11: "dominates" relative to the nebula brightness instead of the noise -
    it brought nothing more on a reflection nebula and exposed the star residual mottling in bright emission.)"""
    H, W = shape
    L = np.zeros((H, W), np.float32)
    for st, name in (("stars", "stars_model.npy"), ("stars", "stars_model_ext.npy"), ("starless", "halo_removed.npy")):
        f = ctx.project.stage_dir(st) / name
        if f.exists():
            a = np.load(f, mmap_mode="r")
            if a.shape[1:] == (H, W):
                L += np.tensordot(lum, np.asarray(a, np.float32), 1)
    if not L.any():
        return np.ones((H, W), np.float32)
    astro = ctx.load("background", "astro.npy", mmap=True)
    Y = np.tensordot(lum, np.asarray(astro[:, ::2, ::2], np.float32), 1)
    sY = 1.4826 * float(np.median(np.abs(Y - cv2.GaussianBlur(Y, (0, 0), 2))))
    L = cv2.GaussianBlur(np.abs(L), (0, 0), 3)
    w = np.clip((3 * sY - L) / (2 * sY), 0, 1).astype(np.float32)
    return cv2.GaussianBlur(w, (0, 0), 2)


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
    """Star footprints of the calibrated nebula: the brightness keeps full resolution and is only lifted where it dips
    more than ``guard_sigma`` noise sigmas below the surroundings (a real hole, not the noise); the colour is the
    colour of the surroundings, interpolated across footprints of any size. No emission / continuum hypothesis."""
    from ..core.imgutil import fill_holes
    hole = foot_w >= 0.5
    if not hole.any():
        return astro.astype(np.float32)
    hp = Yl - cv2.GaussianBlur(Yl, (0, 0), 2)
    sY = 1.4826 * float(np.median(np.abs(hp[::7, ::7])))
    As = np.stack([cv2.GaussianBlur(fill_holes(c, hole), (0, 0), 3) for c in astro])   # surroundings, any hole size
    Ys = np.tensordot(lum, As, 1)
    Yg = np.maximum(Yl, Ys - guard_sigma * sY)
    Nfp = As + (Yg - Ys)[None] * np.asarray(w_cam, np.float32)[:, None, None]   # brightness Yg, colour of As
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
