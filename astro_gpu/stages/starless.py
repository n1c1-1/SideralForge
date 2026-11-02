"""Starless image by subtraction of the Gaia star model (no neural star removal), cleaning and denoising."""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch

from ..core.imgutil import fill_holes, starlet
from ..core.params import ADVANCED, Auto, P, StageParams
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
    learned_stars: str = P("", label="Learned star layer", level=ADVANCED,
                           help="Optional: the star layer removed by a star-removal network trained on this project's "
                                "own stack (.npy, 3 x H x W, stack units, the far halo of the stars excluded). It is "
                                "subtracted from the two half-stacks instead of the star model, and the result goes "
                                "through the same denoising: a second starless image, chosen at rendering ('Learned "
                                "star removal'). Empty = the layer of the 'Learned star removal network' stage when it "
                                "made one, else the star model only; 'none' = the star model only.")
    halo_gmax: float = P(10.5, label="Optical halo limit", level=ADVANCED, ge=4, le=15, step=0.5, unit="G mag",
                         help="Stars brighter than this get their optical halo (blue defocus of the refractor, "
                              "filter reflections) measured and removed from the background, out to a radius growing "
                              "with the star flux.")
    sharpen: bool = P(True, label="Sharpening network",
                      help="Fine-tunes the Noise2Noise network into a sharpening network: the star shape measured on "
                           "the stack is undone towards a sharper target the data can reach (two thirds of the core "
                           "width, never below 2 px). Learned on your own half-stacks; drives the rendering setting "
                           "'Sharpening (measured star shape)', for both star removals.")
    sharpen_steps: int = P(4000, label="Sharpening training steps", level=ADVANCED, ge=500, le=20000, step=500,
                           help="Fine-tuning steps of the sharpening network (a few minutes on a GPU).")


class Starless(Stage):
    name = "starless"
    title = "Star removal and denoising"
    description = ("Subtracts the complete star model (measured + faint stars) from the two half-stacks, fills the "
                   "few cores that cannot be trusted (saturated stars), removes residual and optical halos and "
                   "repairs over-subtracted dips. Then a multi-scale Wiener filter driven by the two independent "
                   "half-stacks keeps every structure that is really present in both, and removes the rest. No "
                   "neural star remover is used: faint nebulosity under the stars is preserved.")
    deps = ("stars", "optics", "denoise", "stack", "color", "catalogs")
    Params = StarlessParams
    version = "24"
    # 21: the residual halo bound measured with the far halo; 22: residual cleaning only where the star model holds
    # the star; 23: the fainter stars' cleaning bounded by their typical residual; 24: the sharpening network, for both
    # images (the optional learned star layer is a parameter, not a version)

    def auto(self, ctx: Context) -> dict[str, Auto]:
        """The learned star layer of the 'star_net' stage, when that stage ran and wrote one (an explicit path set by
        the user still wins). The value carries the run of 'star_net' that made the layer, so that a new layer makes
        this stage run again; without a layer nothing changes (projects that do not use it stay up to date)."""
        from .star_net import stage_layer
        v = stage_layer(ctx.project)
        if v is None:
            return {}
        return {"learned_stars": Auto(v, "star layer of the learned star removal network (stage 'star_net')")}

    def run(self, ctx: Context, p: StarlessParams) -> None:
        r = model_starless(ctx, p)
        x, y, g, fl, fs, M, model, halo = r["x"], r["y"], r["g"], r["fl"], r["fs"], r["M"], r["model"], r["halo"]
        A, B, delta, mask, sl_final, r_core = r["A"], r["B"], r["delta"], r["mask"], r["sl_final"], r["r_core"]
        dirs = r["dirs"]
        del r
        ctx.metric("filled_pct", round(100 * float(mask.mean()), 2))
        ctx.save("sl_mask.npy", mask)
        ctx.metric("spike_directions", [round(d) for d in dirs])
        ctx.save("sl_final.npy", sl_final)

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
        n2n_final, _ = bright_halos(n2n_c0, x, y, g, fs["pred"], p.halo_gmax,
                                    ref=None if halo is None else n2n_c0 + halo, r_core=r_core)
        ctx.save("n2n_final.npy", n2n_final)
        from ..core.imgutil import asinh_preview
        ctx.preview("starless", asinh_preview(sl_final, M))
        del n2n_final, n2n_c0, sl_n2n
        sharp = None
        if p.sharpen:
            # the same detail layer from the sharpening network, cleaned the same way: the rendering takes the
            # difference of the two (off on star light) as its sharpening increment
            sharp = self._sharpen_net(ctx, p, A + delta, B + delta, nm)
            n2s = 0.5 * (apply_net(sharp[0], A + delta, nm, sharp[1], sharp[2], ctx.device)
                         + apply_net(sharp[0], B + delta, nm, sharp[1], sharp[2], ctx.device))
            s_c0, _ = clean(n2s, x, y, g, fl, smooth=False, halos=False, lift_gmax=p.lift_gmax, log=ctx.log)
            s_final, _ = bright_halos(s_c0, x, y, g, fs["pred"], p.halo_gmax,
                                      ref=None if halo is None else s_c0 + halo, r_core=r_core)
            ctx.save("n2n_sharp_final.npy", s_final)
            del n2s, s_c0, s_final
            if ctx.device.type == "cuda":
                torch.cuda.empty_cache()
        del A, B, delta
        if p.learned_stars:
            self._learned(ctx, p, M, model, halo, Wt, sharp)

    @staticmethod
    def _sharpen_net(ctx: Context, p: StarlessParams, A, B, nm):
        """Non-blind sharpening: the star shape measured on this stack, undone towards a target it can reach, by the
        Noise2Noise network fine-tuned on the two half-stacks (its output, blurred back to the measured shape, must
        predict the other half). Returns (network, black level, scale) as load_model does."""
        from ..core.sharpen import PSFField, finetune_sharp, target_fwhm, target_to_stack_kernels
        ctx.progress(0.86, "sharpening network")
        cat = np.load(ctx.input("catalogs", "gaia_deep.npz"))
        gsat = float(ctx.metric_of("optics", "g_sat", 10.9))
        full = ctx.load("optics", "lat_full.npy", mmap=True)
        psf = PSFField(full, cat["x"], cat["y"], cat["g"], gsat, log=ctx.log)
        tf = target_fwhm(psf)
        Ks, worst = target_to_stack_kernels(psf, tf)
        ctx.metric("sharpen_target_fwhm", tf)
        ctx.metric("sharpen_kernel_error_pct", round(100 * worst, 1))
        ctx.log(f"sharpening target {tf} px (star core {psf.fwhm(1, psf.W / 2, psf.H / 2):.1f} px at the centre); "
                f"worst reconstruction of the measured shape {100 * worst:.1f} %")
        del full
        net, bk, sg = load_model(ctx)
        net, loss = finetune_sharp(net, A, B, nm, bk, sg, Ks, ctx.device, steps=p.sharpen_steps,
                                   progress=lambda fr, m: ctx.progress(0.86 + 0.04 * fr, m))
        ctx.metric("sharpen_loss", round(loss, 4))
        torch.save({"net": net.state_dict(), "bk": bk, "sg": sg, "target_fwhm": tf}, ctx.out("n2n_sharp_model.pt"))
        return net, bk, sg

    @staticmethod
    def _learned(ctx: Context, p: StarlessParams, M, model, halo, Wt, sharp=None) -> None:
        """The second starless image: the learned star layer subtracted from the two half-stacks, then the same
        Wiener filter and Noise2Noise detail layer. None of the star-model repairs (core fill, residual halos, dip
        repair, bright halos): they correct the model's errors, which this layer does not have."""
        from .star_net import layer_file
        f = layer_file(p.learned_stars)
        if f is None:
            return
        if not f.exists():
            ctx.log(f"learned star layer not found: {f} - only the star model is used", "warning")
            return
        SL = np.load(f, mmap_mode="r")
        if SL.shape != model.shape:
            ctx.log(f"learned star layer ignored: shape {SL.shape} differs from {model.shape}", "warning")
            return
        ctx.progress(0.9, "learned star removal")
        SL = np.array(SL, np.float32)                 # a writable copy of the memory-mapped layer
        # only the significant star light: the layer the network removes holds a little of the stack's noise (16 % on
        # the Iris), common to both half-stacks - the Wiener filter kept it as signal and the starless image came out
        # twice as grainy as the model one. Below 3 x its own noise the layer is zeroed (98.6 % of the star light kept).
        # On both sides of zero: around the dominant stars the layer is negative where it gives back the light the far
        # halo model over-subtracted (down to -83 noise sigmas 40-250 px from Alnitak) - cutting it dug a dark ring
        sb = cv2.GaussianBlur(SL.mean(0), (0, 0), 1.0)
        s0 = 1.4826 * float(np.median(np.abs(sb[::4, ::4] - np.median(sb[::4, ::4]))))
        w = np.clip((np.abs(sb) / max(s0, 1e-9) - 3.0) / 3.0, 0, 1)
        w = cv2.GaussianBlur(cv2.dilate(w, np.ones((3, 3), np.uint8)), (0, 0), 1.0)
        ctx.metric("learned_layer_kept_pct", round(100 * float((w > 0.5).mean()), 1))
        SL *= w[None]
        del sb, w
        A = ctx.load("optics", "lat_A.npy").astype(np.float32) - SL
        B = ctx.load("optics", "lat_B.npy").astype(np.float32) - SL
        del SL
        if halo is not None:
            A -= halo
            B -= halo
        # saturated cores are missing data here too: the two half-stacks saturate differently (+-40 ADU at the core of
        # the Iris G 6.9 star), which left a hole and a ring in the Noise2Noise layer. Filled from their surroundings,
        # as the star model's path does
        full = ctx.load("optics", "lat_full.npy", mmap=True)
        sat = np.zeros(A.shape[1:], bool)
        for c in range(3):
            fc = np.asarray(full[c], np.float32)
            sat |= fc > 0.7 * np.percentile(fc, 99.999)
        del fc
        sat = cv2.dilate(sat.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        rng = np.random.default_rng(1)
        for X in (A, B):
            for c in range(3):
                hp = X[c] - cv2.GaussianBlur(X[c], (0, 0), 2)
                sgx = 1.4826 * np.median(np.abs(hp[::7, ::7])) * 1.3
                fh = fill_holes(X[c], sat)
                X[c][sat] = fh[sat] + rng.normal(0, sgx, int(sat.sum())).astype(np.float32)
        del hp, fh, full
        wl = wiener(A, B, M, p, model, log=ctx.log)
        ctx.save("sl_final_learned.npy", wl)
        net, bk, sg = load_model(ctx)
        _, _, nm = norm_params(ctx.load("stack", "drz_A.npy").astype(np.float32), Wt)
        n2l = 0.5 * (apply_net(net, A, nm, bk, sg, ctx.device) + apply_net(net, B, nm, bk, sg, ctx.device))
        del net
        # the filled cores hold no detail: the network turned the fill's noise into a checkered pattern (the Wiener
        # filter removes it, independent in the two halves) - the detail layer takes the Wiener result there
        wsat = cv2.GaussianBlur(cv2.dilate(sat.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                                .astype(np.float32), (0, 0), 2.0)
        n2l += wsat[None] * (wl - n2l)
        ctx.save("n2n_final_learned.npy", n2l)
        del n2l
        if sharp is not None:                      # the sharpening network on this image too, filled cores alike
            n2s = 0.5 * (apply_net(sharp[0], A, nm, sharp[1], sharp[2], ctx.device)
                         + apply_net(sharp[0], B, nm, sharp[1], sharp[2], ctx.device))
            n2s += wsat[None] * (wl - n2s)
            ctx.save("n2n_sharp_final_learned.npy", n2s)
            del n2s
        del wl, wsat
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()
        ctx.log(f"learned star removal: second starless image from {f.name} (chosen at rendering)")


def model_starless(ctx: Context, p: StarlessParams, prog=None) -> dict:
    """The model-based starless image of the two half-stacks: the star model (+ far halo) subtracted, the unreliable
    cores filled, residual halos and dips cleaned, the multi-scale Wiener filter, the final cleaning and the bright
    stars' optical halos. Shared by this stage and by 'star_net' (its training target). Nothing is saved here.
    Returns a dict: x, y, g, fl, fs, M, model, halo, A, B, delta, mask, sl_final, r_core, dirs."""
    prog = prog or ctx.progress
    dm = float(ctx.metric_of("optics", "mag_offset", 0.0))
    fs = np.load(ctx.input("stars", "star_fluxes.npz"))
    x, y, g, fl = fs["x"], fs["y"], fs["g"] - dm, fs["flux"]          # g shifted: thresholds stay meaningful
    M = ctx.load("color", "M.npy")
    model = ctx.load("stars", "stars_model.npy").astype(np.float32) + ctx.load("stars", "stars_model_faint.npy")
    # the measured angular shape of the stars (spikes) is in the model; the unreliable-core mask is built on the
    # core model only (spike pixels are subtracted, never cut out: cutting strips left straight traces)
    exf = ctx.project.stage_dir("stars") / "stars_model_ext.npy"
    model_core = model - np.load(exf).astype(np.float32) if exf.exists() else model

    prog(0.02, "subtracting the star model")
    A = ctx.load("optics", "lat_A.npy").astype(np.float32) - model
    B = ctx.load("optics", "lat_B.npy").astype(np.float32) - model
    hf = ctx.project.stage_dir("stars") / "stars_model_halo.npy"
    halo = None
    if hf.exists():            # the far halo of the stars (glare of the optics) leaves with them; it is kept out
        halo = np.load(hf)     # of the core model, which decides what is filled and where residuals are guarded
        A -= halo
        B -= halo
    full = ctx.load("optics", "lat_full.npy", mmap=True)
    mask = core_mask(full, model_core, A, B, p.mask_k)
    del model_core
    rng = np.random.default_rng(1)
    for X in (A, B):
        for c in range(3):
            hp = X[c] - cv2.GaussianBlur(X[c], (0, 0), 2)
            sgx = 1.4826 * np.median(np.abs(hp[::7, ::7])) * 1.3
            f = fill_holes(X[c], mask)
            X[c][mask] = f[mask] + rng.normal(0, sgx, int(mask.sum())).astype(np.float32)
    rawfill = 0.5 * (A + B)

    prog(0.15, "cleaning residual halos and dips")
    cleanraw, _ = clean(rawfill, x, y, g, fl, smooth=True, halos=True, lift_gmax=15.0, log=ctx.log)
    delta = cleanraw - rawfill
    del cleanraw

    prog(0.35, "multi-scale Wiener denoising")
    wms = wiener(A + delta, B + delta, M, p, model, log=ctx.log)
    prog(0.55, "final cleaning")
    final0, _ = clean(wms, x, y, g, fl, smooth=False, halos=False, lift_gmax=p.lift_gmax, log=ctx.log)
    del wms
    # what is left around the bright stars (their cores, where the star model fails) is removed star by star,
    # bounded by the typical halo of their brightness - measured with the far halo, which is what it describes
    r_core = None
    if halo is not None:     # the region the star model holds, as the far halo sees it (flat inside, x 1.3)
        from .far_halo import WING_OUT
        r_core = np.zeros(len(x))
        for i in np.where(g < p.halo_gmax)[0]:
            r_core[i] = 1.3 * max(WING_OUT, _filled_radius(mask, x[i], y[i], rmax=800) + 4)
    sl_final, nh = bright_halos(final0, x, y, g, fs["pred"], p.halo_gmax,
                                ref=None if halo is None else final0 + halo, r_core=r_core)
    ctx.log(f"optical halos removed around {nh} bright stars")
    del final0
    prog(0.62, "diffraction spikes")
    # the spikes left with the star model (measured star shape); their directions tell the rendering that the
    # optics have spikes (saturated stars are then shown by their real light)
    with_spikes = sl_final + np.load(exf).astype(np.float32) if exf.exists() else sl_final
    dirs, _ = spike_directions(with_spikes, mask, x, y, g, log=ctx.log)
    del with_spikes
    return dict(x=x, y=y, g=g, fl=fl, fs=fs, M=M, model=model, halo=halo, A=A, B=B, delta=delta, mask=mask,
                sl_final=sl_final, r_core=r_core, dirs=dirs)


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
        # a residual is a model error only if the model could be that wrong: beyond 60 % of the model it is the
        # star's real halo or the nebula, not missing data
        plausible = np.abs(rs - bgl) < 0.6 * model[c] + 2 * sig
        mask |= (fc > 0.7 * lvl) | (model[c] > K * sig) | ((np.abs(z) > 4) & (model[c] > 2 * sig) & plausible)
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
        if abs(d) > 3 * sig / np.sqrt(core.sum()) * 3:
            rfix = int(np.clip(3 + 1.0 * (14.5 - g[i]), 3, 9)) if g[i] >= 9.5 else int(np.clip(9 + 4.5 * (9.5 - g[i]), 9, 34))
            cv2.circle(fixm, (xc, yc), rfix, 1, -1)
            nfix += 1
    oy2, ox2 = np.mgrid[-18:19, -18:19]
    rd2 = np.hypot(ox2, oy2)
    core2, ring2 = rd2 < 7, (rd2 >= 12) & (rd2 <= 18)
    Ys = cv2.GaussianBlur(Y, (0, 0), 1.5)
    dipped = set()
    for i in np.where((g < 14) & (x > 40) & (x < W - 40) & (y > 40) & (y < H - 40))[0]:
        xc, yc = int(round(x[i])), int(round(y[i]))
        cut = Ys[yc - 18:yc + 19, xc - 18:xc + 19]
        rg = cut[ring2]
        d = cut[core2].mean() - np.median(rg)
        if d < -max(2.5 * 1.4826 * np.median(np.abs(rg - np.median(rg))) / np.sqrt(core2.sum() / 9), 0.15):
            cv2.circle(fixm, (xc, yc), 10, 1, -1)
            dipped.add(int(i))
    m = fixm > 0
    wgt = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
    for c in range(3):
        sl[c] = sl[c] * (1 - wgt) + fill_holes(sl[c], m) * wgt
    # 3. one-sided dip repair: inside the disc of each star the background stays above its surroundings - 0.5 sigma
    # only around the stars whose centre measurably dips (and the faint ones, whose dip is below the test but whose
    # disc is small): in a dense field the discs of all the stars would cover most of the image and flatten the real
    # dark structures of the nebula into plateaus
    disc = np.zeros((H, W), np.uint8)
    for i in np.where((g < lift_gmax) & (x > 0) & (x < W) & (y > 0) & (y < H))[0]:
        if g[i] < 14 and int(i) not in dipped:
            continue
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
    # bright stars (G < 11): over-subtracted wings = large dark discs -> floor at the outer ring (not in emission).
    # Inside a structured nebula the floor is applied only where the wings are measurably over-subtracted (the disc
    # darker than the ring around it): otherwise it flattened the dark lanes around the star into a plateau and a
    # ring. On a smooth background it stays as it was (it also hides the dark marks of badly modelled stars)
    Egb = cv2.GaussianBlur(sl[0] - 0.87 * sl[1], (0, 0), 12)
    qb1, qb2 = np.percentile(Egb[::4, ::4], [55, 80])
    gate_b = np.clip((qb2 - Egb) / max(qb2 - qb1, 1e-6), 0, 1).astype(np.float32)
    Yg = sl.mean(0)[::2, ::2]
    bpg = cv2.GaussianBlur(Yg, (0, 0), 0.75) - cv2.GaussianBlur(Yg, (0, 0), 3)
    struct = np.sqrt(cv2.GaussianBlur(bpg * bpg, (0, 0), 15))
    struct_floor = float(np.percentile(struct, 10))
    before = sl.copy()
    yyb, xxb = np.mgrid[-90:91, -90:91]
    rdb = np.hypot(xxb, yyb)
    for i in np.where((g < 11) & (x > 95) & (x < W - 95) & (y > 95) & (y < H - 95))[0]:
        rr_ = float(np.clip(22 + 10 * (11 - g[i]), 22, 70))
        xc, yc = int(round(x[i])), int(round(y[i]))
        win = (slice(yc - 90, yc + 91), slice(xc - 90, xc + 91))
        wd_ = np.clip((rr_ + 4 - rdb) / 16, 0, 1) * gate_b[win]
        ann = (rdb >= rr_ + 4) & (rdb < rr_ + 18)
        a2 = (rdb >= rr_ + 4) & (rdb < 88)
        ys_ = np.clip((yc + yyb[a2]) // 2, 0, struct.shape[0] - 1)
        xs_ = np.clip((xc + xxb[a2]) // 2, 0, struct.shape[1] - 1)
        structured = float(np.median(struct[ys_, xs_])) > 3 * struct_floor   # nebula structure around the star
        for c in range(3):
            cut = sl[c][win]
            cs_ = cv2.GaussianBlur(cut, (0, 0), 2.0) if smooth else cut
            hpc = cs_ - cv2.GaussianBlur(cs_, (0, 0), 2)
            sgc = 1.4826 * np.median(np.abs(hpc[ann]))
            inner = (rdb > 0.4 * rr_) & (rdb < rr_)
            if structured and np.median(cs_[inner]) >= np.median(cs_[ann]) - sgc:
                continue                                        # structured nebula, no dark disc: nothing to repair
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


_HALO_STEP = 2.0       # radial bin, px
_HALO_RMAX = 700       # farthest radius ever examined, px
_HALO_PCT = 35         # percentile of each annulus taken as the halo level (version 4 value)
_HALO_K = 3.0          # a star may have up to K times the typical halo of its brightness (saturated stars do)


def _radial_profiles(sl, xi, yi, rsearch):
    """Low-percentile radial profile of the three channels around (xi, yi) in 2 px annuli out to rsearch + 20 px.
    Returns the patch bounds, the radius map, the bin centres and the profiles (nan where the annulus is empty)."""
    H, W = sl.shape[1:]
    R_ = rsearch + 20
    xc, yc = int(round(xi)), int(round(yi))
    y0, y1, x0, x1 = max(0, yc - R_), min(H, yc + R_ + 1), max(0, xc - R_), min(W, xc + R_ + 1)
    if y1 - y0 < 20 or x1 - x0 < 20:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - xi, yy - yi)
    rb = np.arange(0, rsearch + 20, _HALO_STEP)
    idx = np.clip((rr / _HALO_STEP).astype(int), 0, len(rb) - 2)
    flat_idx = idx.ravel()
    counts = np.bincount(flat_idx, minlength=len(rb) - 1)
    order = np.argsort(flat_idx, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(counts)])
    rcv = rb[:-1] + _HALO_STEP / 2
    meds = []
    for c in range(3):
        vals = sl[c, y0:y1, x0:x1].ravel()[order]
        meds.append(np.array([np.percentile(vals[bounds[k]:bounds[k + 1]], _HALO_PCT) if counts[k] > 5 else np.nan
                              for k in range(len(rb) - 1)]))
    return (y0, y1, x0, x1), rr, rcv, meds


def _decreasing(prof, kb, kmax=None):
    """Non-increasing profile from its peak inside kb bins (the filled core may be lower than the halo); the peak is
    looked for within kmax bins only (just beyond the filled core)."""
    prof = np.nan_to_num(np.clip(prof, 0, None))
    k0 = int(np.argmax(prof[:max(min(kb, kmax if kmax is not None else kb), 1)]))
    prof[:k0] = prof[k0]
    return np.minimum.accumulate(prof)


def bright_halos(sl, x, y, g, pred, gmax, mask=None, ref=None, r_core=None):
    """Remove the optical halo of every bright star from the starless image.

    The halo of a star cannot be told from a nebula centred on it by looking at that star alone (the Iris nebula is
    centred on its illuminating star). So the halo shape is measured on all the bright stars of the field: the
    radial profile of each (low percentile of every annulus, so spikes and neighbours do not count), divided by the
    star's catalogue flux, and the median over the stars gives the instrument's halo template per channel and radius.
    Each star then loses min(template x its flux, its own measured profile): never more than a halo of its brightness
    has, never more than is there. The extent of a star's halo is where its template falls into the scatter of its
    far background; the background is measured just beyond. No removal inside bright emission. With fewer than five
    bright stars the template is unknown and each star's own profile is used within the nominal radius (315 px at
    G = 5 down to 25 px at G = gmax). ``ref``: the image the template is measured on, when ``sl`` has already lost
    the far halo of the stars (the template of what is left would be nearly zero and bound nothing). ``r_core``
    (per star): with the far halo removed, only what the star model was meant to hold is left to clean - out to that
    radius, beyond it the far halo holds the glare (cleaning further erased the nebula around AE Aurigae).
    """
    sl = sl.astype(np.float32).copy()
    Eg = cv2.GaussianBlur(sl[0] - 0.87 * sl[1], (0, 0), 12)
    q1, q2 = np.percentile(Eg[::4, ::4], [55, 80])
    gate = np.clip((q2 - Eg) / max(q2 - q1, 1e-6), 0, 1).astype(np.float32)
    H, W = sl.shape[1:]
    sel = np.where((g < gmax) & (x > -40) & (x < W + 40) & (y > -40) & (y < H + 40))[0]
    sel = sel[np.argsort(g[sel])]
    nb = len(np.arange(0, _HALO_RMAX + 20, _HALO_STEP)) - 1          # bins of the longest profile
    # flux scale of each star: the catalogue magnitude (the fitted fluxes and the colour prediction both fail on
    # saturated stars), with the star's colour from the prediction
    near10 = (g > 9.5) & (g < 10.5)
    m10 = float(np.median(pred[1, near10])) if near10.sum() >= 5 else float(np.median(pred[1]))
    amp = m10 * 10 ** (-0.4 * (g - 10.0)) * pred / np.maximum(pred[1], 1e-3)[None]
    # pass 1: profiles of every bright star; the template comes from the brightest eight (five at least), the template
    # of the other stars from themselves
    stars = []
    nt = min(8, len(sel))
    norm = np.full((nt, 3, nb), np.nan, np.float32)
    norm_o = np.full((max(len(sel) - nt, 1), 3, nb), np.nan, np.float32)
    for j, i in enumerate(sel):
        rnom = int(np.clip(25 * 10 ** (0.2 * (gmax - g[i])), 25, 320))
        rsearch = min(3 * rnom, _HALO_RMAX) if j < nt else rnom     # only the brightest are followed beyond the nominal radius
        rp = _radial_profiles(sl, x[i], y[i], rsearch)
        if rp is None:
            stars.append(None)
            continue
        stars.append((rsearch, rp))
        if j >= nt:
            # what is left around a fainter star, per unit of its flux, measured on the image cleaned (the far halo
            # out): the typical residual of the star model, which a nebula centred on one star exceeds
            _, _, rcv, meds = rp
            for c in range(3):
                m = min(len(rcv), nb)
                far = np.nanmedian(meds[c][max(m - 5, 0):m])
                norm_o[j - nt, c, :m] = (meds[c][:m] - far) / max(float(amp[c, i]), 1e-3)
            continue
        _, _, rcv, meds = rp if ref is None else _radial_profiles(ref, x[i], y[i], rsearch)
        for c in range(3):
            far = meds[c][-20:]
            m = min(len(rcv), nb)
            norm[j, c, :m] = (meds[c][:m] - np.nanmedian(far)) / max(float(amp[c, i]), 1e-3)
    enough = np.isfinite(norm[:, 1, :]).sum(0) >= 5
    template = np.where(enough[None], np.nanmedian(np.where(np.isfinite(norm), norm, np.nan), 0), 0.0) \
        if enough.any() else None
    if template is not None:
        template = np.stack([_decreasing(template[c], nb) for c in range(3)])
    enough_o = np.isfinite(norm_o[:, 1, :]).sum(0) >= 5
    template_o = np.stack([_decreasing(np.where(enough_o, np.nanmedian(norm_o[:, c], 0), 0.0), nb)
                           for c in range(3)]) if enough_o.any() else None
    n = 0
    for j, i in enumerate(sel):
        if stars[j] is None:
            continue
        rsearch, ((y0, y1, x0, x1), rr, rcv, meds) = stars[j]
        m = len(rcv)
        own = [meds[c] for c in range(3)]
        extended = template is not None and j < nt
        if extended:
            halo = [_HALO_K * float(amp[c, i]) * template[c, :m] for c in range(3)]   # bound: K x the typical halo
            # extent: last radius where the typical halo of this brightness stands above the scatter of the far
            # background (max over channels). Beyond it a sky gradient would be taken for halo
            rmax = 25
            for c in range(3):
                far = own[c][-20:]
                scatter = 1.4826 * np.nanmedian(np.abs(far - np.nanmedian(far))) + 1e-6
                above = np.where(np.nan_to_num(halo[c]) / _HALO_K > 3 * scatter)[0]
                if len(above):
                    rmax = max(rmax, int(rcv[above[-1]]))
            rmax = min(rmax, rsearch - 20)
        else:                                                   # every other star: within the nominal radius
            rmax = rsearch
            if template_o is not None:
                # bound: K x the typical residual of these stars (unbounded, the core of a reflection nebula around its
                # star - NGC 2023, IC 435 - was taken for halo and erased)
                halo = [_HALO_K * float(amp[c, i]) * template_o[c, :m] for c in range(3)]
        if r_core is not None:
            rmax = min(rmax, max(int(r_core[i]), 25))
        kb = int(np.searchsorted(rcv, rmax))
        kfill = None
        if mask is not None:                                    # radius of the filled core of this star
            mf = mask[y0:y1, x0:x1].ravel()
            fr = np.array([mf[np.abs(rr.ravel() - r_) < _HALO_STEP / 2].mean() if k_ < 200 else 0.0
                           for k_, r_ in enumerate(rcv[:200])])
            kfill = int(np.argmax(fr < 0.5)) + int(6 / _HALO_STEP) if (fr < 0.5).any() else None
        for c in range(3):
            patch = sl[c, y0:y1, x0:x1]
            bgv = float(np.nanmedian(own[c][kb:kb + 10]))       # just beyond the end of the halo
            if extended or template_o is not None:
                prof = np.minimum(_decreasing(halo[c], kb), _decreasing(own[c] - bgv, kb, kfill))
            else:
                prof = np.minimum.accumulate(np.nan_to_num(np.clip(own[c] - bgv, 0, None)))
            prof = prof * np.clip((rmax - rcv) / (0.4 * rmax), 0, 1) ** 2
            h = np.interp(rr, rcv, prof, right=0)
            h = np.minimum(h, 0.9 * np.clip(patch - bgv, 0, None))
            sl[c, y0:y1, x0:x1] = patch - h * gate[y0:y1, x0:x1]
        n += 1
    return sl, n


def _filled_radius(mask, xc, yc, rmax=200):
    """Radius beyond which less than half of each ring around (xc, yc) is filled core."""
    H, W = mask.shape
    for r_ in range(3, rmax, 2):
        a = np.linspace(0, 2 * np.pi, max(16, int(2 * np.pi * r_)), endpoint=False)
        px = np.clip(np.round(xc + r_ * np.cos(a)).astype(int), 0, W - 1)
        py = np.clip(np.round(yc + r_ * np.sin(a)).astype(int), 0, H - 1)
        if mask[py, px].mean() < 0.5:
            return r_
    return rmax


def spike_directions(sl, mask, x, y, g, gmax=10.0, log=print):
    """Directions (degrees) and transverse profile of the diffraction spikes, measured on the bright stars of the
    field: the azimuthal profiles just outside their filled cores, normalised and summed, peak where the spikes are.
    A spider gives four spikes 90 degrees apart; a refractor gives none (empty list)."""
    H, W = sl.shape[1:]
    Y = sl.sum(0)
    acc = np.zeros(360)
    n = 0
    for i in np.where((g < gmax) & (x > 160) & (x < W - 160) & (y > 160) & (y < H - 160))[0]:
        r0 = _filled_radius(mask, x[i], y[i]) + 4
        if r0 > 100:
            continue
        rr = np.arange(r0, r0 + 50, 1.0)
        a = np.radians(np.arange(360))
        px = x[i] + rr[:, None] * np.cos(a)[None]
        py = y[i] + rr[:, None] * np.sin(a)[None]
        v = cv2.remap(Y, px.astype(np.float32), py.astype(np.float32), cv2.INTER_LINEAR)
        v = v - np.median(v, 1, keepdims=True)
        prof = np.median(v, 0)
        s = 1.4826 * np.median(np.abs(prof - np.median(prof))) + 1e-9
        acc += np.clip((prof - np.median(prof)) / s, -5, 20)
        n += 1
    if n < 3:
        return [], None
    acc /= np.sqrt(n)
    k0 = int(np.argmax(acc))
    dirs = []
    for k in range(4):
        c = (k0 + 90 * k) % 360
        win = [(c + d) % 360 for d in range(-4, 5)]
        best = max(win, key=lambda a_: acc[a_])
        # a diffraction spike is point-symmetric: the spike at 180 degrees must stand out as well. A one-sided
        # excess is a leftover of the star removal (Heart, refractor: 67 and 153 deg without their opposites,
        # 21 sigma summed over 617 stars - the rendering then showed every saturated star as photographed)
        opp = max([(c + 180 + d) % 360 for d in range(-4, 5)], key=lambda a_: acc[a_])
        if acc[best] > 6 and acc[opp] > 6:
            dirs.append(float(best))
    if len(dirs) < 2:
        log(f"diffraction spikes: none detected (peak {acc[k0]:.1f} sigma over {n} bright stars)")
        return [], None
    # transverse profile of a spike, measured on the bright stars over every direction (half-pixel steps),
    # normalised to its peak: the shape every spike model uses
    tw = np.arange(-14.0, 14.25, 0.5)
    side = np.abs(tw) >= 9
    profs = []
    for i in np.where((g < gmax) & (x > 160) & (x < W - 160) & (y > 160) & (y < H - 160))[0][:60]:
        r0 = _filled_radius(mask, x[i], y[i]) + 4
        if r0 > 100:
            continue
        rr = np.arange(r0, r0 + 80, 1.0)
        for d in dirs:
            u = np.array([np.cos(np.radians(d)), np.sin(np.radians(d))])
            nrm = np.array([-u[1], u[0]])
            px = (x[i] + rr[:, None] * u[0] + tw[None] * nrm[0]).astype(np.float32)
            py = (y[i] + rr[:, None] * u[1] + tw[None] * nrm[1]).astype(np.float32)
            v = cv2.remap(Y, px, py, cv2.INTER_LINEAR)
            pr = np.median(v - np.median(v[:, side], 1, keepdims=True), 0)
            pk = pr[np.abs(tw) <= 2].max()
            noise = 1.4826 * np.median(np.abs(pr[side] - np.median(pr[side]))) + 1e-9
            if pk > 5 * noise:
                profs.append(pr / pk)
    if len(profs) < 4:
        log("diffraction spikes: directions found but their profile could not be measured, left in place")
        return [], None
    T = np.median(np.array(profs), 0)
    T = np.clip(0.5 * (T + T[::-1]), 0, None)
    T[side] = 0.0
    hw = float(np.abs(tw)[T > 0.05].max()) if (T > 0.05).any() else 2.0
    T[np.abs(tw) > hw + 1] = 0.0
    T /= max(T.max(), 1e-9)
    log(f"diffraction spikes: {len(dirs)} directions " + ", ".join(f"{d:.0f}" for d in dirs) +
        f" deg (peak {acc[k0]:.1f} sigma over {n} bright stars), profile measured on {len(profs)} spikes, "
        f"half width {hw:.1f} px")
    return dirs, {"tw": tw, "T": T, "hw": hw}
