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
    real_g: float = P(12.0, label="Spikes measured down to", level=ADVANCED, ge=0, le=16, step=0.5, unit="G mag",
                      help="With diffraction spikes, the spikes of the stars brighter than this are measured on the "
                           "image (their own light, the nebula interpolated across them), for the rendering setting "
                           "'Photographed spikes': the spike shape measured once for all stars never matches each "
                           "star's own spikes. 0 = none (every spike drawn).")
    spike_snr: float = P(2.0, label="Spikes kept from", level=ADVANCED, ge=1, le=50, step=0.5, unit="x noise",
                         help="A spike is shown as photographed out to where it stands this many times the noise above "
                              "the light beside it; beyond, the drawn star shape continues it (at 5 x noise the strips "
                              "stopped short of the dark blobs the shared spike shape leaves along a G 6.9 spike).")
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
    diffuse_scale: float = P(0.2, label="Diffuse glow scale", level=ADVANCED, ge=0.05, le=1.0, step=0.01,
                             help="Size of the window, as a fraction of the shorter side of the frame, in which the "
                                  "diffuse glow of the field (the level a gradient removal would take as background) "
                                  "is measured, as a low percentile of the nebula. Structures smaller than the window "
                                  "stand above it. How much of it is removed is set live ('Diffuse glow removed').")
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
    version = "41"         # 29: the star guard wherever star light dominates (dark moats around G 9.5-11.5 stars);
                           # 30: the diffuse part of the star glare; 31: the glare of the dominant stars only, the
                           # measured emission colour also in the emission / continuum / OIII separation; 32: the
                           # star guard back to the footprints (its extension drew dark discs on the Heart); 33: the
                           # real light of a saturated star = image minus the nebula rendered under it (dark notches);
                           # 34: under a saturated star in structured nebula, the nebula kept at large scales only;
                           # 35: only where the star's model light exceeds the nebula (34 washed out the Iris);
                           # 36: every star brighter than real_g shown as photographed, nebula interpolated across spikes;
                           # 38: under the spike strips only the spikes taken from the image, the stars there drawn;
                           # 40: their light averaged along the spike direction; 41: the spikes drawn in the base
                           # layers, the photographed ones in layers of their own (chosen at rendering)

    learned = False        # RenderPrepLearned: the same layers from the learned starless image

    def run(self, ctx: Context, p: RenderPrepParams) -> None:
        if self.learned and not (ctx.project.stage_dir("starless") / "sl_final_learned.npy").exists():
            ctx.log("no learned star removal for this project (stage 'Star removal and denoising', setting 'Learned "
                    "star layer'): nothing to prepare")
            ctx.save_json("none.json", {"learned": False})
            return
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
        M, e_meas, w_e = emission_anchor(astro, M, spec["cam_hii"].astype(np.float64), fs, dm)
        sl_name, n2_name = "sl_final.npy", "n2n_final.npy"
        if self.learned:
            # the learned starless image with the same sky and large-scale corrections as the model one (the colour
            # matrix above is measured on the model one: both renderings share their colours)
            sl_name, n2_name = "sl_final_learned.npy", "n2n_final_learned.npy"
            for c in range(3):
                astro[c] += (np.asarray(ctx.load("starless", sl_name, mmap=True)[c], np.float32)
                             - np.asarray(ctx.load("starless", "sl_final.npy", mmap=True)[c], np.float32)) / flat[c]
        if e_meas is not None:
            ctx.metric("emission_colour_measured", np.round(e_meas, 3).tolist())
            ctx.metric("emission_anchor_weight", round(w_e, 2))
            ctx.log(f"line emission measured on the nebula: camera colour R 1 : G {e_meas[1]:.2f} : B {e_meas[2]:.2f}"
                    + (f"; rendered with the colour of an HII spectrum (weight {w_e:.2f}), the star colours unchanged"
                       if w_e > 0 else " - within the colours of the stars: no line emission, matrix unchanged"))
        if coefs["0"]["k_ha"] is not None:
            e_ = np.array([coefs[str(c)]["k_ha"] for c in range(3)])
        else:
            e_ = spec["cam_hii"].astype(np.float64)
        e_ = e_ / e_[0]
        if e_meas is not None and w_e > 0:
            # the emission as measured on the nebula also separates it from the continuum and the OIII: with the
            # reconstructed response, the green of the emission passed for OIII wherever there is H-alpha
            e_ = (1 - w_e) * e_ + w_e * e_meas
            ctx.metric("emission_signature_source", "measured on the nebula")
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
        # (the learned starless image has no star-subtraction artefacts to guard against)
        foot_w = footprints(ctx, fs, dm, H, W) if p.star_guard and not self.learned else np.zeros((H, W), np.float32)
        ctx.metric("guard_footprint_pct", round(100 * float((foot_w > 0.5).mean()), 1))
        ctx.progress(0.2, "nebula layer")
        # two denoisers are available: the multi-scale Wiener filter and the Noise2Noise network trained on these data.
        # A local Wiener filter leaves islands of unfiltered pixel noise wherever its local noise estimate falls short
        # (dark and bright pepper, worst at the edges of a nebula); the network does not. The fine scales come from the
        # network, the large ones from the Wiener filter (bright-star zones are interpolated identically in both)
        # (also with the learned star removal: its Noise2Noise layer is not reliable where star light dominates either -
        # the two half-stacks differ systematically around bright cores, and the network left values of 12 to 64 in
        # a nebula of 36 around the Iris G 6.9 star)
        wsl = star_light_weight(ctx, lum, astro.shape[1:])
        wstar = wsl * (1 - foot_w)
        ctx.metric("fine_layers_off_pct", round(100 * float((wstar < 0.5).mean()), 1))
        n2 = ctx.load("starless", n2_name).astype(np.float32) / flat
        b0 = ctx.load("starless", sl_name).astype(np.float32) / flat
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
        if p.star_guard and not self.learned:
            # (tried 2026-10-14: the guard also wherever the light of a star dominates the noise, for the dark moats
            # around G 9.5-11.5 stars - in a dense field that is 98 % of it (Heart), and the guard drew dark discs)
            N = guard_generic(N, np.tensordot(lum, N, 1).astype(np.float32), foot_w, lum, w_cam, p.guard_sigma)
        del wsl
        # saturated stars: shown by their real light; the nebula under them interpolated where that is safe
        ctx.progress(0.42, "real light of the saturated stars")
        N, fp_all, fp_fill, Sreal, strips = saturated_star_light(ctx, fs, N, flat, lum, ctx.log, astro=astro,
                                                         g_real=p.real_g if p.real_g > 0 else None, dm=dm, spike_snr=p.spike_snr,
                                                                 starless=sl_name, interpolate=not self.learned)
        ctx.save("N.npy", N.astype(np.float32))
        if self.learned:
            # every star as photographed (the learned starless image is trusted everywhere): image minus the nebula
            # rendered, the far halo excepted (drawn back as the glare), the noise faded out like the saturated stars'
            full = ctx.load("optics", "lat_full.npy", mmap=True)
            hf = ctx.project.stage_dir("stars") / "stars_model_halo.npy"
            hal = np.load(hf, mmap_mode="r") if hf.exists() else None
            Sall = np.stack([np.asarray(full[c], np.float32) / flat[c] - b0[c]
                             - (np.asarray(hal[c], np.float32) / flat[c] if hal is not None else 0)
                             + cv2.GaussianBlur(astro[c] - N[c], (0, 0), 1.0) for c in range(3)])
            del full, hal
            Ya = np.tensordot(lum, Sall, 1)
            hp_ = Ya[::3, ::3] - cv2.GaussianBlur(Ya[::3, ::3], (0, 0), 2)
            sg_ = 1.4826 * float(np.median(np.abs(hp_ - np.median(hp_))))
            Sall *= np.clip((cv2.GaussianBlur(Ya, (0, 0), 1.0) / max(sg_, 1e-9) - 1.5) / 2.0, 0, 1)[None]
            ctx.save("Sall.npy", (Sall * good[None]).astype(np.float16))
            del Sall, Ya, hp_
        if strips is not None:              # the photographed spikes, chosen at rendering
            for nm_, a_ in strips.items():
                ctx.save(f"{nm_}.npy", a_.astype(np.float16))
            del strips
        ctx.save("Dglow.npy", diffuse_glow(N, good, p.diffuse_scale).astype(np.float16))
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
            ctx.save("dS.npy", (np.tensordot(lum, n2s - n2, 1) * wstar * (1 - fp_fill)).astype(np.float16))
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
        hf = ctx.project.stage_dir("stars") / "stars_model_glow.npy"
        if not hf.exists():
            hf = ctx.project.stage_dir("stars") / "stars_model_halo.npy"
        # the glare of the stars that dominate their surroundings (with their rays), taken out of the nebula: the
        # rendering draws it back as their glow. Earlier star stages saved only the glare of all the stars
        hal = np.load(hf, mmap_mode="r") if hf.exists() else None
        if hal is not None and hal.shape[1:] == (H, W):
            Sh = np.stack([np.asarray(hal[c], np.float32) / flat[c] for c in range(3)])
            ctx.save("Shalo.npy", Sh.astype(np.float16))
            # its diffuse part, measured like the diffuse glow of the nebula: removed with it ('Diffuse glow removed'),
            # the glare then keeps its part near the stars (far out it is a gradient, as a gradient removal sees it)
            ctx.save("ShaloL.npy", diffuse_glow(Sh, good, p.diffuse_scale).astype(np.float16))
            del Sh
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


def emission_anchor(astro, M, cam_hii, fs, dm, block=64):
    """Colour matrix whose rendering of the line emission of this field is the colour of an HII spectrum.

    The colour calibration fits the matrix on the stars and anchors the HII colour on the camera's response to an HII
    spectrum, reconstructed from the stars: its value at the lines is poorly determined (the green / red response at
    H-alpha came out 0.04 to 0.43 on fields of the same camera), and the emission really recorded also holds lines
    the eye does not see (UV, near infrared). The emission is measured on the nebula instead: the colour of the
    structure of its brightest, most coherent parts (band-pass 4-32 px, so free of any zero point), the reddest
    fifth of them. The matrix is changed only across the plane of the star colours (rank-one update along its
    normal), so every star keeps its calibrated colour; the measured emission gets the colour the calibration gives
    the HII spectrum. Applied in full when the emission is clearly redder than the reddest stars of the field, not
    at all when it is within their colours (no emission: reflection nebula, galaxy).
    Returns (matrix, measured emission colour R = 1 or None, weight)."""
    g = fs["g"] - dm
    F = fs["flux"].astype(np.float64)
    m = (g > 9) & (g < 13.5) & (F > 0).all(0)
    if m.sum() < 50 or not np.isfinite(cam_hii).all() or cam_hii[0] <= 0:
        return M, None, 0.0
    v = F[:, m] / np.linalg.norm(F[:, m], axis=0)
    n = np.linalg.svd(v.T, full_matrices=False)[2][2]          # normal of the plane of the star colours
    g_red = float(np.percentile(F[1, m] / F[0, m], 0.5))       # the reddest stars, G / R
    a = np.stack([np.asarray(astro[c, ::2, ::2], np.float32) for c in range(3)])
    bp = np.stack([cv2.GaussianBlur(c, (0, 0), 2) - cv2.GaussianBlur(c, (0, 0), 16) for c in a])
    del a
    H, W = bp.shape[1:]
    rows = []
    for y in range(0, H - block + 1, block):
        for x in range(0, W - block + 1, block):
            r, gg, b = (bp[c, y:y + block, x:x + block].ravel().astype(np.float64) for c in range(3))
            vr = float(r @ r)
            if vr <= 0 or not np.isfinite(vr):
                continue
            rows.append((vr / r.size, (gg @ r) / vr, (b @ r) / vr, np.corrcoef(r, gg)[0, 1]))
    if len(rows) < 20:
        return M, None, 0.0
    rows = np.array(rows)
    top = rows[(rows[:, 0] > np.percentile(rows[:, 0], 80)) & (rows[:, 3] > 0.8)]
    if len(top) < 10:
        return M, None, 0.0
    red = top[top[:, 1] <= np.percentile(top[:, 1], 20)]
    e = np.array([1.0, float(np.median(red[:, 1])), float(np.median(red[:, 2]))])
    w = float(np.clip((g_red - e[1]) / (0.25 * g_red), 0, 1))
    if w <= 0 or abs(n @ e) < 1e-3:
        return M, e, 0.0
    t = M @ (cam_hii / cam_hii[0])
    return M + w * np.outer(t - M @ e, n) / (n @ e), e, w


def diffuse_glow(N, good, frac, q=3.0) -> np.ndarray:
    """Large-scale lower envelope of the nebula per channel: the q-th percentile in a window of ``frac`` times the
    shorter side of the frame, smoothed at half the window - the diffuse glow on which the structures stand (diffuse
    H-alpha, galactic light), that a gradient removal takes as background: at the darkest areas, as its samples are
    (at the 20th percentile, a fifth of the field fell below it and came out as black holes)."""
    from scipy.ndimage import percentile_filter
    C, H, W = N.shape
    f = max(4, min(H, W) // 160)
    h, w = H // f, W // f
    ok = cv2.resize(good.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5
    win = max(3, int(round(frac * min(H, W) / f)) | 1)
    out = np.zeros((C, H, W), np.float32)
    for c in range(C):
        s = cv2.resize(np.asarray(N[c], np.float32), (w, h), interpolation=cv2.INTER_AREA)
        if ok.any() and not ok.all():
            from ..core.imgutil import membrane_fill
            s = membrane_fill(s, ~ok)
        e = cv2.GaussianBlur(percentile_filter(s, q, size=win, mode="reflect"), (0, 0), win / 2)
        out[c] = cv2.resize(e, (W, H), interpolation=cv2.INTER_CUBIC)
    return out


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


def _ring_median(img, xc, yc, r_, q=50):
    H, W = img.shape
    a = np.linspace(0, 2 * np.pi, max(32, int(2 * np.pi * r_)), endpoint=False)
    px = np.round(xc + r_ * np.cos(a)).astype(int)
    py = np.round(yc + r_ * np.sin(a)).astype(int)
    m = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return float(np.percentile(img[py[m], px[m]], q)) if m.sum() >= 16 else np.nan


def saturated_star_light(ctx, fs, N, flat, lum, log=print, rmax=400, astro=None, g_real=None, dm=0.0, spike_snr=2.0,
                         starless="sl_final.npy", interpolate=True):
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
        z = np.zeros((H, W), np.float32)
        return N, z, z.copy(), np.zeros_like(N), None
    full = ctx.load("optics", "lat_full.npy", mmap=True)
    sl = ctx.load("starless", starless, mmap=True)
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
    if not len(idx_sat) and g_real is None:
        return N, fp_all, fp_fill, np.zeros_like(N), None
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
    models = [np.load(f, mmap_mode="r") for f in (ctx.project.stage_dir("stars") / "stars_model.npy",
                                                   ctx.project.stage_dir("stars") / "stars_model_ext.npy") if f.exists()]
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
        if not interpolate:          # learned starless image: the nebula under the star is trusted as it is (its fine
            fp_fill[y0_:y1_, x0_:x1_] = np.maximum(fp_fill[y0_:y1_, x0_:x1_], w_)    # detail layers are not)
            continue
        if q < 2.5:
            hole = rr < R
            for c in range(3):
                win = N2[c, y0_:y1_, x0_:x1_]
                N2[c, y0_:y1_, x0_:x1_] = win * (1 - w_) + membrane_fill(win, hole) * w_
            fp_fill[y0_:y1_, x0_:x1_] = np.maximum(fp_fill[y0_:y1_, x0_:x1_], w_)
            n_fill += 1
        else:
            # structured nebula around: kept - but where the star's model light exceeds the nebula, only at the scales
            # the model leaves intact: between its rays the symmetric star model over-subtracts, and left dark notches
            # in a ring around the star (Flaming Star). The finer scales go to the star light (image minus this
            # nebula), which gives the image back. Where the nebula dominates (the Iris around its star) it is kept as
            # is: smoothing it there washed out its structure
            sm = [cv2.GaussianBlur(N2[c, y0_:y1_, x0_:x1_], (0, 0), max(R / 8.0, 2.0)) for c in range(3)]
            Mw = sum(np.tensordot(lum, np.asarray(m_[:, y0_:y1_, x0_:x1_], np.float32) / flat[:, y0_:y1_, x0_:x1_], 1)
                     for m_ in models) if models else 0.0
            Nb = np.maximum(np.tensordot(lum, np.stack(sm), 1), 3 * sig)
            dom = np.clip((Mw / Nb - 0.5) / 1.5, 0, 1) * w_
            for c in range(3):
                win = N2[c, y0_:y1_, x0_:x1_]
                N2[c, y0_:y1_, x0_:x1_] = win + dom * (sm[c] - win)
    spiked, n_spk = np.zeros(0, int), 0
    sw, axes, gmap = None, [], None
    N2s, fp_fill_s = N2, fp_fill                   # the nebula with the saturated stars only: drawn spikes
    if g_real is not None:
        N2s, fp_fill_s = N2.copy(), fp_fill.copy()
        spiked, n_spk, sw, axes, gmap = _bright_stars_real(ctx, fs, N2, fp_all, fp_fill, Ys, sig, xs, ys, ok, idx_sat,
                                                           g_real, dm, rmax, spike_snr=spike_snr,
                                                           interpolate=interpolate)
    del Ys, YL
    # star light: the image minus the nebula layer actually rendered under the star, so that nebula + star light give
    # the image back. Image minus the starless image plus the change made to the nebula (N - N2) missed what the
    # preparation changed in the starless image itself (smoothing, guard): where the core fill of the starless image
    # was brighter than the image between the spikes, the star light came out negative - dark notches in a diamond
    # around the bright stars
    ref = N if astro is None else astro
    fp_sat = fp_all.copy()
    # the base (spikes drawn): the saturated stars' light against the nebula without the spike strips
    Sb = S + np.stack([cv2.GaussianBlur(ref[c] - N2s[c], (0, 0), 1.0) for c in range(3)])
    fade = np.clip((np.tensordot(lum, Sb, 1) / max(sig, 1e-9) - 1.5) / 2.0, 0, 1)
    Sb *= (fade * fp_sat)[None]
    strips = None
    if sw is not None:
        Sr = S + np.stack([cv2.GaussianBlur(ref[c] - N2[c], (0, 0), 1.0) for c in range(3)])
    del S, slf
    if sw is not None:
        # under a spike strip only the spike is the image's: the round light of every star there (the spike's own
        # star beyond its drawn core, its neighbours) is drawn from the tables - taken from the image it came out as
        # soft blobs (raw light, wider than the drawn stars) and the neighbours were shown twice
        # (the cores themselves, where the star removal filled the image, are left out of the image's light: the
        # core model never matches a core closely enough - a G 11 core came back at full brightness over the drawn one)
        ws = sw * (1 - fp_sat)
        cm = ctx.project.stage_dir("starless") / "sl_mask.npy"
        q = np.ones((H, W), np.float32)
        if cm.exists():
            cmk = cv2.dilate(np.load(cm).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
            q = 1 - cv2.GaussianBlur(cmk, (0, 0), 1.5)
            ws *= q
            del cmk
        cf = [ctx.project.stage_dir("stars") / n for n in ("stars_model.npy", "stars_model_faint.npy")]
        ef = ctx.project.stage_dir("stars") / "stars_model_ext.npy"
        for c in range(3):
            core = sum(np.asarray(np.load(f, mmap_mode="r")[c], np.float32) for f in cf if f.exists())
            if ef.exists():
                core = core - np.asarray(np.load(ef, mmap_mode="r")[c], np.float32)
            Sr[c] -= cv2.GaussianBlur(core / flat[c], (0, 0), 1.0) * ws
            del core
        # a spike is a line: under its strip the light is averaged along the spike's direction only, over the strip's
        # width (sharp across it; the raw light left brownish blotches along the faint spikes). The star cores left out
        # above do not enter the average (weights q)
        Sraw = Sr.copy()
        for th, m, L in axes:
            kern = np.zeros((L, L), np.float32)
            c0, d = L // 2, (L // 2) * np.array([np.cos(th), np.sin(th)])
            cv2.line(kern, (int(round(c0 - d[0])), int(round(c0 - d[1]))), (int(round(c0 + d[0])), int(round(c0 + d[1]))),
                     1.0, 1, cv2.LINE_AA)
            wk = cv2.GaussianBlur((m > 0).astype(np.float32), (0, 0), 1.5) * (1 - fp_sat)
            den = np.maximum(cv2.filter2D(q, -1, kern, borderType=cv2.BORDER_REFLECT), 1e-3)
            for c in range(3):
                Sd = cv2.filter2D(Sr[c] * q, -1, kern, borderType=cv2.BORDER_REFLECT) / den
                Sr[c] += wk * (Sd - Sr[c])
            del wk, den
        del axes, q
        # the strip layers, chosen at rendering ('Photographed spikes'): nebula interpolated across the strips (not
        # inside the saturated stars' footprints: their light was measured against the base nebula), the spikes'
        # light averaged along them and raw, where the strips stand, and the magnitude of the star of each strip
        strips = {"Wsp": sw if interpolate else np.zeros_like(sw), "Wsl": ws, "Gsp": gmap,
                  "dNsp": (N2 - N2s) * (1 - fp_sat)[None]}
        for k_, a_ in (("Ssp", Sr), ("Ssp_raw", Sraw)):
            fd = np.clip((np.tensordot(lum, a_, 1) / max(sig, 1e-9) - 1.5) / 2.0, 0, 1)
            a_ *= (fd * ws)[None]
            strips[k_] = a_
        del Sr, Sraw, ws
    log(f"saturated stars: {len(idx_sat)} shown by their own light ({100 * float((fp_sat > 0.5).mean()):.2f} % of the "
        f"field); nebula interpolated under {n_fill} of them, kept under the others (structured nebula around)")
    if g_real is not None:
        log(f"diffraction spikes shown as photographed for {len(spiked)} more stars brighter than G {g_real:.1f} "
            f"({n_spk} spikes, {100 * float((sw > 0.5).mean()):.2f} % of the field, the nebula interpolated across "
            f"them, the choice made at rendering); their cores and the stars along them stay drawn")
    return N2s, fp_sat, fp_fill_s, Sb, strips


def _bright_stars_real(ctx, fs, N2, fp_all, fp_fill, Ys, sig, xs, ys, ok, idx_sat, g_real, dm, rmax, spike_snr=2.0,
                       interpolate=True):
    """The diffraction spikes of every star brighter than g_real are shown by their own light, as photographed.

    The star removal subtracted from the nebula one spike shape measured for all the stars and the rendering drew it
    back; a star's own spikes differ from it (dark blobs along them where it was too strong, segments where it was
    weak). Along each spike direction, from the edge of the star's core out to where the light along the spike stops
    standing twice the noise above the light beside it on both sides (Ys: image minus starless; a nebula rises on both
    alike - measured against zero, the strips ran 400 px through the Rosette), a strip: under it the nebula is interpolated across the strip
    and the star light is image minus that nebula. The round core of a star that is not saturated stays drawn (compact,
    its size set by the star settings: its own light, wings included, made the medium stars look large and soft).
    N2, fp_all and fp_fill (interpolated: the fine detail and colour layers, taken from the starless images where the
    spike model was subtracted, are off there) are updated in place. Returns the indices of the stars given real spikes,
    the number of strips, their weight and, per spike direction, (direction, strips, strip width)."""
    from ..core.imgutil import membrane_fill
    H, W = fp_all.shape
    dirs = np.radians(np.asarray(ctx.metric_of("starless", "spike_directions", []), float))
    S_ = float(ctx.load_json("stack", "drz_meta.json")["scale"])
    fwhm = float(ctx.metric_of("register", "fwhm_px", 2.0) or 2.0) * S_
    hw = int(max(2, round(0.9 * fwhm)))                     # half-width of a spike strip
    r0 = 2.5 * fwhm                                         # the drawn core of a star that is not saturated
    g = fs["g"] - dm
    sat = set(int(i) for i in idx_sat)
    bright = np.where(ok & (g < g_real))[0]
    strip = np.zeros((H, W), np.uint8)
    strip_d = [np.zeros((H, W), np.uint8) for _ in dirs]           # the strips of each spike direction
    gmap = np.full((H, W), 99.0, np.float32)                        # magnitude of the brightest star of each strip
    with_spikes, n_spk = [], 0
    r = np.arange(1, rmax, dtype=np.float32)
    off = np.arange(-1, 2, dtype=np.float32)
    for i in bright[np.argsort(g[bright])]:
        xc, yc = float(xs[i]), float(ys[i])
        got = False
        lines = []
        for k_, th in enumerate(dirs):
            ux, uy = float(np.cos(th)), float(np.sin(th))
            px = (xc + r * ux)[:, None] - uy * off[None]
            py = (yc + r * uy)[:, None] + ux * off[None]
            v = cv2.remap(Ys, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # beside the spike, on both sides (a nebula or the star's round wings rise there too: only what stands
            # above them along the direction is the spike)
            sd = np.array([-5.0 * hw, 5.0 * hw], np.float32)
            qx = (xc + r * ux)[:, None] - uy * sd[None]
            qy = (yc + r * uy)[:, None] + ux * sd[None]
            vs = cv2.remap(Ys, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            above = ((np.median(v, 1) - np.max(vs, 1)) >= spike_snr * sig).astype(np.float32)
            # the spike ends where less than 60 % of the last 15 px stand above spike_snr x the noise (a saturated core or a
            # neighbour may interrupt it near the star)
            frac = np.convolve(above, np.ones(15) / 15, mode="full")[:len(r)]
            good = np.where(frac >= 0.6)[0]
            L = int(r[good[-1]]) if len(good) else 0
            start = 0.0 if int(i) in sat else r0
            if L <= start + 3:
                continue
            p0 = (int(round(xc + start * ux)), int(round(yc + start * uy)))
            p1 = (int(round(xc + (L + hw) * ux)), int(round(yc + (L + hw) * uy)))
            cv2.line(strip, p0, p1, 255, thickness=2 * hw + 1)
            cv2.line(strip_d[k_], p0, p1, 255, thickness=2 * hw + 1)
            lines.append((p0, p1))
            n_spk += 1
            got = True
        if lines:
            pts = np.array([p for l_ in lines for p in l_])
            bx0, by0 = np.maximum(pts.min(0) - hw - 2, 0)
            bx1, by1 = np.minimum(pts.max(0) + hw + 3, (W, H))
            if bx1 > bx0 and by1 > by0:
                loc = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
                for p0, p1 in lines:
                    cv2.line(loc, (p0[0] - bx0, p0[1] - by0), (p1[0] - bx0, p1[1] - by0), 1, thickness=2 * hw + 1)
                gm = gmap[by0:by1, bx0:bx1]
                np.minimum(gm, np.where(loc > 0, np.float32(g[i]), np.float32(99)), out=gm)
        if got and int(i) not in sat:
            with_spikes.append(int(i))
    if n_spk and not interpolate:   # learned starless image: the nebula across the spikes is trusted as it is
        sw = cv2.GaussianBlur((strip > 0).astype(np.float32), (0, 0), 1.5)
    elif n_spk:
        hole = strip > 0
        sw = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 1.5)
        nlab, lab, st, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
        it = int(np.clip(8 * hw * hw, 40, 400))         # a Laplace fill across a thin strip converges with its width
        for k in range(1, nlab):
            x, y, w, h, _ = st[k]
            x0, y0, x1, y1 = max(0, x - 4), max(0, y - 4), min(W, x + w + 4), min(H, y + h + 4)
            hk = lab[y0:y1, x0:x1] == k
            wk = sw[y0:y1, x0:x1]
            for c in range(3):
                win = N2[c, y0:y1, x0:x1]
                N2[c, y0:y1, x0:x1] = win + wk * (membrane_fill(win, hk, iters=(it, it, it, it)) - win)
        np.maximum(fp_fill, sw, out=fp_fill)
    else:
        sw = np.zeros((H, W), np.float32)
    return np.array(with_spikes, int), n_spk, sw, [(th, m, 2 * hw + 1) for th, m in zip(dirs, strip_d) if m.any()], gmap


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


class RenderPrepLearned(RenderPrep):
    name = "render_prep_learned"
    title = "Rendering preparation, learned star removal"
    description = ("The same rendering layers from the starless image of the learned star removal (when the project "
                   "has one: stage 'Star removal and denoising', setting 'Learned star layer'), without the repairs of "
                   "the star model (guard, interpolation under the stars and across their spikes). The rendering "
                   "setting 'Learned star removal' chooses between the two preparations.")
    learned = True
