"""Weighted Bayer drizzle on the GPU with predictive outlier rejection and two independent half-stacks."""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..core.imgutil import asinh_preview
from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage, StageError
from ..io.frames import cfa_channel_map
from .calibrate import VALID_MAX, open_cache

#: the level of a parameter the project sets itself: never shown in a settings form
INTERNAL = "internal"
EXCLUDED = "excluded by hand"
#: what the stack keeps for the Inspect mode (extra files: an older result lacks them): the normalisation, half and
#: rejected samples of each used frame and the noise of the result (DIAG); the rejected samples counted on the output
#: grid (REJECTION)
DIAG = "diagnostics.json"
REJECTION = "rejection.npz"
#: per output pixel, the share (0-255) of its samples near saturation (NEAR_SATURATION; layer 0: the colour
#: calibration leaves those stars out) and of its clipped samples (layer 1: the simple output gives those cores the
#: colour of their surroundings), the most of the three colours
SATURATION = "saturation.npy"
#: a raw value above this fraction of the saturation level is near saturation (the response bends before it clips)
NEAR_SATURATION = 0.85
#: the rejection maps are reduced by a whole factor until their longer side is at most this
REJECTION_SIDE = 4096


class StackParams(StageParams):
    scale: float = P(1.5, label="Drizzle scale", ge=1.0, le=3.0, step=0.25, unit="x",
                     help="Output resolution relative to the sensor. Each raw pixel is dropped onto a finer grid at "
                          "its exact registered position, so under-sampled stars become rounder and finer.",
                     tip="1.5 for most setups with dozens of frames; 2 only with many well-dithered frames and "
                         "small stars; 1 when stars are large (> 5 px) or frames are few. Higher values cost memory "
                         "and add noise.")
    pixfrac: float = P(0.7, label="Drop size", level=ADVANCED, ge=0.3, le=1.0, step=0.05,
                       help="Size of the deposited drop relative to the input pixel.",
                       tip="Smaller drops = sharper but noisier, with a risk of holes; 0.6-0.8 is the usual range.")
    min_transparency: float = P(0.6, label="Minimum transparency", ge=0.0, le=1.0, step=0.05,
                                help="Frames whose star flux falls below this fraction of the best frames (clouds, "
                                     "haze, dew) are excluded.",
                                tip="0 keeps everything (low-transparency frames still get a low weight). Raise it "
                                    "if hazy frames bring gradients or bloated halos.")
    max_fwhm_ratio: float = P(2.0, label="Maximum star size", level=ADVANCED, ge=1.0, le=5.0, step=0.1, unit="x median",
                              help="Frames whose stars are larger than this multiple of the median size (wind, "
                                   "focus drift, tracking error) are excluded.")
    kappa: float = P(5.0, label="Rejection threshold", level=ADVANCED, ge=2.0, le=20.0, step=0.5, unit="sigma",
                     help="A pixel deviating from the predicted sky by more than this many noise sigmas is rejected "
                          "(satellites, planes, cosmic rays, hot pixels). Bright outliers are dilated by one pixel "
                          "to remove the edges of trails.",
                     tip="Lower (3-4) removes more trails but may eat star cores in poor seeing.")
    core_tolerance: float = P(0.5, label="Star core tolerance", level=ADVANCED, ge=0.0, le=2.0, step=0.1,
                              help="Extra relative tolerance on bright pixels, so that sharp star cores (which vary "
                                   "with seeing and sub-pixel position) are not rejected.")
    half_block: int = P(20, label="Half-stack block", level=ADVANCED, ge=1, le=200, unit="frames",
                        help="Frames are dealt alternately in blocks of this size to two independent half-stacks A "
                             "and B. Their difference measures the noise, which drives the Noise2Noise denoiser and "
                             "the multi-scale Wiener filter.")
    gain: float = P(0.13, label="Sensor gain", level=ADVANCED, gt=0, unit="e-/ADU",
                    help="Electrons per ADU in the calibrated frames, used for the noise model of the rejection.")
    read_noise: float = P(11.9, label="Read noise", level=ADVANCED, ge=0, unit="ADU",
                          help="Noise of a frame without light, used for the noise model of the rejection.")
    pattern_removal: bool = P(False, label="Fixed pattern removal", level=ADVANCED,
                              help="The sensor's fixed pattern (residual dark current, hot or cold columns, amplifier "
                                   "patterns not removed by the darks - or everything when there are no darks) stays "
                                   "at the same place on the sensor while the sky drifts between frames. Stacked, it "
                                   "can become 'walking noise': streaks along the drift. The pattern is measured "
                                   "as the median of the frames in sensor coordinates (stars and sky removed) and "
                                   "subtracted from every frame. Skipped automatically when the drift is too small "
                                   "to tell the pattern from the sky.",
                              tip="Off by default: on the Rosette the estimate also held the tracks of the faint stars "
                                  "across the sensor, and subtracting it drew the very streaks it was meant to remove "
                                  "(without it the stacked background has none; it was neutral on Heart).")
    pattern_frames: int = P(48, label="Frames for the pattern reference", level=ADVANCED, ge=8, le=400,
                            help="Number of frames, evenly spread over the session, whose median is the first estimate "
                                 "of the fixed pattern; the final estimate uses all frames around it.")
    colour_from_luminance: bool = P(True, label="Fine detail from all colours", level=ADVANCED,
                                    help="A colour sensor samples red and blue on one pixel in four, green on one in "
                                         "two: at the finest scale each colour alone is noisy, while the three together "
                                         "(the luminance) are sampled on every pixel. As a debayering does, the noise "
                                         "finer than about a pixel of the sensor is taken from the luminance (all the "
                                         "samples, each colour weighted by its noise), in the colour of its "
                                         "surroundings; the stars and each colour's own light at that scale are kept. "
                                         "Less colour grain, the same sharpness and star colours.",
                                    tip="Off: each colour keeps its own finest detail (with many well-dithered frames "
                                        "the drizzle resolves it, at the cost of more colour grain).")
    exclude: list[str] = P([], label="Frames excluded by hand", level=INTERNAL,
                           help="Light frames left out of the stack whatever their quality (their files). Set by the "
                                "project from the frames the user excluded; not a setting of its own.")


class Stack(Stage):
    name = "stack"
    title = "Stacking (drizzle)"
    description = ("Combines the calibrated frames directly from their raw colour pixels: every red, green or blue "
                   "pixel is projected through its registration polynomial onto a finer output grid and deposited as "
                   "a small square drop. No debayering interpolation is ever done. Frames are normalised for "
                   "transparency and their additive gradient, weighted by (transparency/noise)^2 x sharpness, and a "
                   "second pass rejects outliers against the first-pass prediction. Two half-stacks are produced for "
                   "noise-aware denoising.")
    deps = ("calibrate", "register")
    Params = StackParams
    # 2: fixed pattern (walking noise) removal; 3: off by default (it drew walking noise); 4: the rejection judged
    # against a prediction cleaned of the strong outliers, with a noise model calibrated on each frame's sky and
    # the star-core allowance on the signal above the background (it was on the sky too)
    version = "4"

    def auto(self, ctx: Context) -> dict[str, Auto]:
        a: dict[str, Auto] = {}
        reg = np.load(ctx.input("register", "reg.npz"))
        n = len(reg["fwhm"])
        hb = int(max(1, min(20, n // 4)))
        a["half_block"] = Auto(hb, "{n} frames: blocks of {hb} give two half-stacks of about {half} frames" if hb == 20
                               else "{n} frames: blocks of {hb} give two half-stacks of about {half} frames "
                                    "(blocks of 20 need 80 frames)", {"n": n, "hb": hb, "half": n // 2})
        f = float(np.median(reg["fwhm"]))
        if f < 3.0 and n >= 150:
            a["scale"] = Auto(2.0, "small stars (FWHM {fwhm:.1f} px) and {n} frames", {"fwhm": f, "n": n})
        elif f < 5.5 and n >= 40:
            a["scale"] = Auto(1.5, "FWHM {fwhm:.1f} px with {n} frames: under-sampled, enough frames to drizzle",
                              {"fwhm": f, "n": n})
        else:
            a["scale"] = Auto(1.0, "FWHM {fwhm:.1f} px / {n} frames: drizzling would add noise without detail",
                              {"fwhm": f, "n": n})
        ok = np.isfinite(reg["rms"])
        t = reg["transp"]
        frac = float(np.mean(t[ok] < 0.6 * np.percentile(t[ok], 90)))
        if frac > 0.5:
            a["min_transparency"] = Auto(0.3, "{pct:.0f} % of frames are below 60 % transparency: only the "
                                              "worst (below 30 %) are excluded", {"pct": 100 * frac})
        elif frac > 0:
            a["min_transparency"] = Auto(0.6, "{pct:.0f} % of frames lost more than 40 % of the signal "
                                              "(clouds or haze): excluded", {"pct": 100 * frac})
        else:
            a["min_transparency"] = Auto(0.6, "transparency is stable: no frame falls below 60 %")
        st = np.load(ctx.input("calibrate", "frame_stats.npy"))
        m, s = st[:, :, 0].mean(1), st[:, :, 1].mean(1)
        rn = ctx.metric_of("calibrate", "dark_frame_noise")
        scale = float(ctx.metric_of("calibrate", "cache_scale", 1.0))
        if rn:
            rn_c = rn * scale
            g = float(np.median(m / np.maximum(s ** 2 - rn_c ** 2, 1e-3)))
            a["read_noise"] = Auto(round(rn_c, 2), "noise of a single dark frame")
            a["gain"] = Auto(round(g, 4), "sky level vs sky noise of the light frames, read noise removed")
        else:
            g = float(np.median(m / np.maximum(s ** 2, 1e-3)))
            a["gain"] = Auto(round(g, 4), "sky level vs sky noise of the light frames (no darks: read noise ignored)")
        return a

    def run(self, ctx: Context, p: StackParams) -> None:
        t0 = time.time()
        dev = ctx.device
        mm, info = open_cache(ctx)
        n, H, W = info["n"], info["height"], info["width"]
        ch_of_k = cfa_channel_map(info["cfa"])
        reg = np.load(ctx.input("register", "reg.npz"))
        coef, transp, fwhm, ref = reg["coef"], reg["transp"].astype(float), reg["fwhm"], int(reg["ref"])
        rms = reg["rms"]
        stats = ctx.load("calibrate", "frame_stats.npy")
        dark = ctx.load("calibrate", "master_dark.npy")
        flat = ctx.load("calibrate", "master_flat.npy")
        satval = torch.from_numpy(((info["white_level"] - dark) / flat * info["scale"]).astype(np.float32)).to(dev)
        del dark, flat
        S, PIX = p.scale, p.pixfrac
        Ho, Wo = int(round(H * S)), int(round(W * S))

        meta = ctx.load_json("inventory", "frames.json")["lights"]
        excl = excluded_mask([f["path"] for f in meta], p.exclude)
        if len(p.exclude) > int(excl.sum()):
            ctx.log(f"{len(p.exclude) - int(excl.sum())} frame(s) excluded by hand are not among the light frames",
                    "warning")
        sel_f = ctx.project.stage_dir("register") / "selection.npz"
        if sel_f.exists():                       # a selection made by its own block (frame selection)
            sel = np.load(sel_f)
            weight, reasons = sel["weight"].astype(float), sel["reasons"].astype(object)
            ctx.log("frame selection given by the selection block")
            if excl.any():
                reasons[(reasons == "") & excl] = EXCLUDED
                weight[reasons != ""] = 0.0
        else:
            weight, reasons = select_frames(transp, fwhm, rms, stats, p.min_transparency, p.max_fwhm_ratio,
                                            exclude=excl, noise=reg["noise"] if "noise" in reg.files else None)
        used = np.where(weight > 0)[0]
        if len(used) < 2:
            raise StageError("fewer than 2 usable frames after selection; relax the selection thresholds")
        ctx.save_json("frames.json", [{"path": meta[i]["path"], "time": meta[i].get("time"), "weight": float(weight[i]),
                                       "transparency": float(transp[i]), "fwhm": float(fwhm[i]),
                                       "excluded": reasons[i]} for i in range(n)])
        for r in ("low transparency", "large stars", "not aligned", EXCLUDED):
            k = int((reasons == r).sum())
            if k:
                ctx.log(f"{k} frame(s) excluded: {r}")
        texp = ctx.metric_of("inventory", "exposure_s") or 0
        ctx.metric("frames_used", int(len(used)))
        ctx.metric("integration_used_h", round(len(used) * texp / 3600, 2))
        ctx.metric("grid", [Wo, Ho])

        yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        kmap = 2 * (yy.long() % 2) + (xx.long() % 2)
        chmap = torch.tensor(ch_of_k, device=dev)[kmap]
        h = PIX * S / 2.0
        nby, nbx = (H // 2) // 32, (W // 2) // 32

        def poly_eval(c, x, y):
            u = x / (W / 2) - 1.0
            v = y / (H / 2) - 1.0
            terms = [torch.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3]
            return (sum(float(c[0, i]) * t for i, t in enumerate(terms)),
                    sum(float(c[1, i]) * t for i, t in enumerate(terms)))

        def quad_terms(u, v):
            a = u / (W / 2) - 1.0
            b = v / (H / 2) - 1.0
            return np.stack([np.ones_like(a), a, b, a * a, a * b, b * b], -1)

        def block_bkg(frame):
            out, cx, cy = [], [], []
            for k in range(4):
                ky, kx = divmod(k, 2)
                pl = frame[ky::2, kx::2][: nby * 32, : nbx * 32]
                b = pl.reshape(nby, 32, nbx, 32).permute(0, 2, 1, 3).reshape(nby, nbx, 1024)
                b = torch.where(b > VALID_MAX, torch.nan, b)
                out.append(torch.nanmedian(b, dim=-1).values)
            jj = (torch.arange(nbx, device=dev) + 0.5) * 32 - 0.5
            ii = (torch.arange(nby, device=dev) + 0.5) * 32 - 0.5
            gy, gx = torch.meshgrid(ii, jj, indexing="ij")
            for k in range(4):
                ky, kx = divmod(k, 2)
                cx.append(2 * gx + kx)
                cy.append(2 * gy + ky)
            return torch.stack(out), torch.stack(cx), torch.stack(cy)

        fpn = None
        if p.pattern_removal:
            ctx.progress(0.0, "measuring the fixed pattern of the sensor")
            fpn, finfo = fixed_pattern(mm, used, coef, dev, H, W, p.pattern_frames,
                                       lambda f: ctx.progress(0.1 * f, "measuring the fixed pattern of the sensor"))
            ctx.metric("fixed_pattern", finfo)
            ctx.log(finfo["msg"])
            if fpn is not None:
                g = fpn[0::2, 1::2].cpu().numpy()                                # one green plane
                sg = 1.4826 * float(np.median(np.abs(g - np.median(g)))) + 1e-9
                ctx.preview("fixed_pattern", np.clip(0.5 + g / (8 * sg), 0, 1))

        fr = torch.from_numpy(np.asarray(mm[ref], np.float32)).to(dev)
        if fpn is not None:
            fr = torch.where(fr > VALID_MAX, fr, fr - fpn)
        Bref_np = block_bkg(fr)[0].cpu().numpy()
        from scipy.ndimage import map_coordinates

        def ref_bkg_at(k, u, v):
            ky, kx = divmod(k, 2)
            pj = ((u - kx) / 2 + 0.5) / 32 - 0.5
            pi = ((v - ky) / 2 + 0.5) / 32 - 0.5
            return map_coordinates(Bref_np[k], [pi.ravel(), pj.ravel()], order=1, mode="nearest").reshape(pi.shape)

        norm: dict[int, tuple[np.ndarray, np.ndarray]] = {}       # frame -> (gradient Q (3, 6), sky level (3,))

        def frame_norm(i, fr):
            """The frame's additive sky difference to the reference (a quadratic per channel, Q (3, 6)) and its sky
            level per channel; measured once (the first pass), then reused."""
            if int(i) in norm:
                return norm[int(i)]
            Bi, cx, cy = block_bkg(fr)
            Q = np.zeros((3, 6))
            bkg_level = np.zeros(3)
            for c in range(3):
                ks = [k for k in range(4) if ch_of_k[k] == c]
                rows, tg = [], []
                for k in ks:
                    u, v = poly_eval(coef[i], cx[k], cy[k])
                    u, v = u.cpu().numpy(), v.cpu().numpy()
                    d = Bi[k].cpu().numpy() - transp[i] * ref_bkg_at(k, u, v)
                    ok = np.isfinite(d)
                    rows.append(quad_terms(u[ok], v[ok]))
                    tg.append(d[ok])
                A, b = np.concatenate(rows), np.concatenate(tg)
                m = np.ones(len(b), bool)
                for _ in range(3):
                    q = np.linalg.lstsq(A[m], b[m], rcond=None)[0]
                    r = b - A @ q
                    s = 1.4826 * np.median(np.abs(r[m]))
                    m = np.abs(r) < 3 * s + 1e-9
                Q[c] = q
                bkg_level[c] = np.nanmedian(np.concatenate([Bi[k].cpu().numpy().ravel() for k in ks]))
            norm[int(i)] = (Q, bkg_level)
            return norm[int(i)]

        def frame_model(i):
            """The frame normalised to the reference (value, ADU of the reference), where its samples land on the
            output grid (Xo, Yo), its saturated and bad samples, its additive sky difference in its own ADU (grad,
            for the noise model) and its sky level per channel."""
            fr = torch.from_numpy(np.asarray(mm[i], np.float32)).to(dev)
            if fpn is not None:
                fr = torch.where(fr > VALID_MAX, fr, fr - fpn)
            X, Y = poly_eval(coef[i], xx, yy)
            Q, bkg_level = frame_norm(i, fr)
            a = X / (W / 2) - 1.0
            b = Y / (H / 2) - 1.0
            terms = torch.stack([torch.ones_like(a), a, b, a * a, a * b, b * b])
            Qs = torch.from_numpy(Q).to(dev, torch.float32)[chmap]
            grad = (Qs.permute(2, 0, 1) * terms).sum(0)
            sat = fr > VALID_MAX
            bad = torch.isnan(fr)
            near = sat | (fr > NEAR_SATURATION * satval)
            val = torch.where(sat, satval, fr)
            val = (val - grad) / transp[i]
            Xo, Yo = (X + 0.5) * S - 0.5, (Y + 0.5) * S - 0.5
            if chan_fix is not None:                  # each colour's own image placed on the green one
                dx, dy = chan_fix.at(Xo, Yo, chmap)
                Xo, Yo = Xo - dx, Yo - dy
            return val, Xo, Yo, sat, bad, grad, bkg_level, near

        def deposit(num, den, val, Xo, Yo, w, keep):
            v = val[keep]
            xo, yo, c = Xo[keep], Yo[keep], chmap[keep]
            jx0 = torch.floor(xo - h + 0.5)
            iy0 = torch.floor(yo - h + 0.5)
            for di in range(3):
                iy = iy0 + di
                wy = (torch.minimum(yo + h, iy + 0.5) - torch.maximum(yo - h, iy - 0.5)).clamp(min=0)
                for dj in range(3):
                    jx = jx0 + dj
                    wx = (torch.minimum(xo + h, jx + 0.5) - torch.maximum(xo - h, jx - 0.5)).clamp(min=0)
                    ww = wx * wy * w
                    ok = (ww > 0) & (jx >= 0) & (jx < Wo) & (iy >= 0) & (iy < Ho)
                    idx = c[ok] * (Ho * Wo) + iy[ok].long() * Wo + jx[ok].long()
                    num.index_add_(0, idx, v[ok] * ww[ok])
                    if den is not None:
                        den.index_add_(0, idx, ww[ok])

        chan_fix = None
        # ---------------- pass 1: weighted mean (the first prediction)
        num = torch.zeros(3 * Ho * Wo, device=dev)
        den = torch.zeros(3 * Ho * Wo, device=dev)
        for c_, i in enumerate(used):
            val, Xo, Yo, sat, bad, _, _, _ = frame_model(i)
            deposit(num, den, val, Xo, Yo, float(weight[i]), ~bad)
            ctx.progress(0.3 * (c_ + 1) / len(used), f"pass 1: {c_ + 1}/{len(used)} frames")
        model = Prediction(num, den, len(used), Ho, Wo, dev)
        del num, den
        # ---------------- the colours registered on each other: the offsets of the red and blue images of the stars
        # from the green one (lateral chromatic aberration, atmospheric dispersion), measured on the first mean
        chan_fix = ChannelOffsets.measure(model, float(satval.median()), S, float(np.median(fwhm)))
        ctx.metric("channel_offsets", chan_fix.summary())
        ctx.log(chan_fix.describe())
        if chan_fix.applied:
            model.shift_channels(chan_fix)
        else:
            chan_fix = None

        def judge(i, model, kappa, edge, diag=None):
            """The samples of frame i rejected against the prediction ``model`` (``reject``)."""
            val, Xo, Yo, sat, bad, grad, _, near = frame_model(i)
            pr = model.sample(Xo, Yo, chmap)
            hi, lo, inside = reject(val, pr, sat, bad, grad, chmap, float(transp[i]), p, kappa, edge, diag)
            return val, Xo, Yo, bad, hi, lo, inside, near, sat

        # ---------------- pass 2: the strong outliers (satellites, planes, cosmic rays, hot pixels) left out of the
        # prediction, so that the final pass does not judge the other frames against a mean they pulled up or down
        num = torch.zeros(3 * Ho * Wo, device=dev)
        den = torch.zeros(3 * Ho * Wo, device=dev)
        for c_, i in enumerate(used):
            val, Xo, Yo, bad, hi, lo, _, _, _ = judge(i, model, CLEAN_FACTOR * p.kappa, False)
            deposit(num, den, val, Xo, Yo, float(weight[i]), ~bad & ~hi & ~lo)
            ctx.progress(0.3 + 0.3 * (c_ + 1) / len(used), f"pass 2: {c_ + 1}/{len(used)} frames")
        model = model.cleaned(num, den)
        del num, den

        # ---------------- pass 3: rejection against the cleaned prediction + half-stacks
        accA = [torch.zeros(3 * Ho * Wo, device=dev) for _ in range(2)]
        accB = [torch.zeros(3 * Ho * Wo, device=dev) for _ in range(2)]
        rej_tot = tested = 0
        rmap = RejectionMap(Ho, Wo, dev)
        rej_frames: dict[int, tuple[int, int, int]] = {}
        judged: dict[int, dict] = {}
        near_acc = torch.zeros(3 * Ho * Wo, device=dev)     # the weight of the samples near saturation
        clip_acc = torch.zeros(3 * Ho * Wo, device=dev)     # and of the clipped ones
        for c_, i in enumerate(used):
            rows: list = []
            val, Xo, Yo, bad, hi, lo, inside, near, sat = judge(i, model, p.kappa, True, rows)
            judged[int(i)] = rows[0]
            keep = ~bad & ~hi & ~lo
            rej_tot += int((hi | lo).sum())
            tested += int((inside & ~bad).sum())
            rej_frames[int(i)] = rmap.add(Xo, Yo, hi, lo)
            acc = accA if (i // p.half_block) % 2 == 0 else accB
            deposit(acc[0], acc[1], val, Xo, Yo, float(weight[i]), keep)
            if near.any():
                deposit(near_acc, None, near.float(), Xo, Yo, float(weight[i]), keep & near)
                deposit(clip_acc, None, sat.float(), Xo, Yo, float(weight[i]), keep & sat)
            ctx.progress(0.6 + 0.4 * (c_ + 1) / len(used), f"pass 3: {c_ + 1}/{len(used)} frames, "
                                                          f"rejected {100 * rej_tot / max(tested, 1):.2f} %")
        del model

        def finish(a):
            img = (a[0] / a[1].clamp(min=1e-12)).view(3, Ho, Wo)
            return torch.where(a[1].view(3, Ho, Wo) > 0, img, torch.zeros_like(img)).cpu().numpy()

        full = [accA[0] + accB[0], accA[1] + accB[1]]
        in_a = {int(i): (int(i) // p.half_block) % 2 == 0 for i in used}
        w_a = float(sum(weight[i] for i in used if in_a[int(i)]))
        w_b = float(sum(weight[i] for i in used if not in_a[int(i)]))
        step = noise_step(Ho, Wo)
        fine = None
        if p.colour_from_luminance:
            # the noise of each colour, from the half-stacks, weighs it in the luminance
            hs = [finish(acc)[:, ::step, ::step] for acc in (accA, accB)]
            nz = halves_noise(hs[0], hs[1], w_a, w_b)
            del hs
            sig = np.array([nz[c] for c in "RGB"], float)
            if np.isfinite(sig).all() and (sig > 0).all():
                fine = (sig, S)
                ctx.log("fine detail from the luminance; noise of the colours " +
                        ", ".join(f"{c} {v:.2f}" for c, v in zip("RGB", sig)))
        cover = (full[1].view(3, Ho, Wo) > 0).all(0)
        drz = finish(full)
        if fine is not None:
            drz = luminance_detail(drz, cover, *fine, dev)
        ctx.save("drz_full.npy", drz)
        halves = []                              # a sample of each half-stack: the noise of the result
        for name, acc in (("drz_A.npy", accA), ("drz_B.npy", accB)):
            img = finish(acc)
            if fine is not None:
                img = luminance_detail(img, cover, *fine, dev)
            ctx.save(name, img)
            halves.append(img[:, ::step, ::step].copy())
            del img
        ctx.save("drz_weight.npy", full[1].view(3, Ho, Wo).cpu().numpy())
        frac = torch.stack([(a / full[1].clamp(min=1e-12)).view(3, Ho, Wo).amax(0) for a in (near_acc, clip_acc)])
        ctx.save(SATURATION, torch.ceil(255 * frac.clamp(0, 1)).to(torch.uint8).cpu().numpy())
        ctx.metric("near_saturation_pct", round(100 * float((frac[0] > 0).float().mean()), 4))
        del near_acc, clip_acc, frac
        rej_pct = 100 * rej_tot / max(tested, 1)          # of the samples landing on the stack (the others: no data)
        ctx.metric("rejected_pct", round(rej_pct, 3))
        ctx.save_json("drz_meta.json", {"scale": S, "pixfrac": PIX, "n": int(len(used)), "reject_pct": rej_pct,
                                        "width": Wo, "height": Ho, "sensor": [W, H]})
        rmap.save(ctx.out(REJECTION))
        frames_d = []
        for i in used:
            q, sky = norm[int(i)]
            lo, hi_n, outside = rej_frames.get(int(i), (0, 0, 0))
            frames_d.append({"index": int(i), "path": meta[i]["path"], "weight": float(weight[i]),
                             "transparency": float(transp[i]), "scale": [float(1.0 / transp[i])] * 3,
                             "offset": [float(v) for v in q[:, 0]], "gradient": q.tolist(), "sky": sky.tolist(),
                             "half": "A" if in_a[int(i)] else "B", "rejected_low": lo, "rejected_high": hi_n,
                             "rejected_outside": outside, **judged.get(int(i), {})})
        noise = {c: (v if np.isfinite(v) else None) for c, v in halves_noise(halves[0], halves[1], w_a, w_b).items()}
        ctx.save_json(DIAG, {"frames": frames_d, "noise": noise,
                             "noise_sample_step": step, "half_weights": [w_a, w_b], "kappa": p.kappa,
                             "rejection": {"file": REJECTION, "factor": rmap.f, "grid": [Wo, Ho],
                                           "shape": [rmap.Hr, rmap.Wr]}})
        del halves
        ctx.log(f"drizzle {Wo}x{Ho} from {len(used)} frames in {(time.time() - t0) / 60:.1f} min, "
                f"{rej_pct:.2f} % pixels rejected")
        ctx.preview("stack", asinh_preview(drz))


#: the first rejection pass leaves out of the prediction only the samples this many times farther than the threshold
CLEAN_FACTOR = 2.0
#: fewer frames than this covering a place: nothing is rejected there (too few to tell an outlier from the others)
MIN_COVER = 3.0
#: a neighbour of a high outlier deviating by more than this many sigmas goes with it (the edges of a trail)
EDGE_SIGMA = 1.5
#: side (output pixels) of the blocks whose medians make the smooth background of the prediction
BKG_BLOCK = 64
#: allowance, as a fraction of the extended signal above the sky floor, for the error of a frame's photometric scale
SCALE_TOLERANCE = 0.05


def smooth_background(M: torch.Tensor, valid: torch.Tensor, block: int = BKG_BLOCK) -> tuple[torch.Tensor, list]:
    """Smooth background of a stack (3, H, W): the medians of its covered pixels in blocks of ``block`` pixels,
    bilinearly interpolated (the stars and other small structures above it); and the sky floor of each channel (the
    10th percentile of the block medians)."""
    C, H, W = M.shape
    hb, wb = -(-H // block), -(-W // block)
    out = torch.zeros_like(M)
    floors = []
    for c in range(C):
        x = torch.where(valid[c], M[c], torch.nan)
        x = F.pad(x, (0, wb * block - W, 0, hb * block - H), value=float("nan"))
        b = x.reshape(hb, block, wb, block).permute(0, 2, 1, 3).reshape(hb, wb, block * block)
        med = torch.nanmedian(b, dim=-1).values
        fin = torch.isfinite(med)
        if not fin.any():
            floors.append(0.0)
            continue
        floors.append(float(torch.quantile(med[fin], 0.1)))
        med = torch.where(fin, med, torch.nanmedian(med[fin]))
        up = F.interpolate(med[None, None], size=(hb * block, wb * block), mode="bilinear", align_corners=False)
        out[c] = up[0, 0, :H, :W]
    return out, floors


class Prediction:
    """What a frame is judged against: the weighted mean of the frames (``num`` / ``den``, flat 3 x Ho x Wo), how many
    frames cover each place (``cover``: the deposited weight relative to its median, times the number of frames), the
    smooth background of the mean and its sky floor; sampled where a frame's samples land (``sample``)."""

    def __init__(self, num: torch.Tensor, den: torch.Tensor, n: int, Ho: int, Wo: int, dev):
        d = den.view(3, Ho, Wo)
        self.Ho, self.Wo = Ho, Wo
        self.valid = d > 0
        self.M = torch.where(self.valid, num.view(3, Ho, Wo) / d.clamp(min=1e-12), torch.zeros_like(d))
        self.cover = torch.zeros_like(d)
        for c in range(3):
            pos = d[c][self.valid[c]]
            if pos.numel():
                med = torch.median(pos[::max(1, pos.numel() // 4_000_000)])
                self.cover[c] = n * d[c] / med
        self.bkg, self.floor = smooth_background(self.M, self.valid)

    def shift_channels(self, fix: "ChannelOffsets") -> None:
        """The red and blue images of the mean moved onto the green one (``fix``), where the following passes place
        their samples."""
        yy, xx = torch.meshgrid(torch.arange(self.Ho, device=self.M.device, dtype=torch.float32),
                                torch.arange(self.Wo, device=self.M.device, dtype=torch.float32), indexing="ij")
        for c in (0, 2):
            dx, dy = fix.at(xx, yy, torch.full_like(xx, c, dtype=torch.long))
            grid = torch.stack([(xx + dx + 0.5) / self.Wo * 2 - 1, (yy + dy + 0.5) / self.Ho * 2 - 1], -1)[None]
            del dx, dy
            num = F.grid_sample((self.M[c] * self.valid[c])[None, None], grid, mode="bilinear", align_corners=False)
            w = F.grid_sample(self.valid[c].float()[None, None], grid, mode="bilinear", align_corners=False)
            self.M[c] = torch.where(w[0, 0] > 0.5, num[0, 0] / w[0, 0].clamp(min=1e-6), self.M[c])

    def cleaned(self, num: torch.Tensor, den: torch.Tensor) -> "Prediction":
        """The same prediction with the mean of the samples ``num`` / ``den`` kept by a rejection (its coverage and
        background are the first mean's)."""
        d = den.view(3, self.Ho, self.Wo)
        self.M = torch.where(d > 0, num.view(3, self.Ho, self.Wo) / d.clamp(min=1e-12), self.M)
        return self

    def sample(self, Xo: torch.Tensor, Yo: torch.Tensor, chmap: torch.Tensor) -> dict[str, torch.Tensor]:
        """The prediction (``pred``), coverage, background and sky floor of each sample landing at (Xo, Yo), in the
        sample's own channel (``chmap``); the mean is interpolated from the covered pixels only."""
        grid = torch.stack([(Xo + 0.5) / self.Wo * 2 - 1, (Yo + 0.5) / self.Ho * 2 - 1], -1).unsqueeze(0)

        def at(a: torch.Tensor) -> torch.Tensor:
            v = F.grid_sample(a.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
            return torch.gather(v, 0, chmap.unsqueeze(0))[0]

        vf = self.valid.float()
        w = at(vf)
        pred = at(self.M * vf) / w.clamp(min=1e-6)
        floor = torch.tensor(self.floor, device=Xo.device, dtype=torch.float32)[chmap]
        return {"pred": pred, "cover": torch.where(w > 0.5, at(self.cover), torch.zeros_like(w)), "bkg": at(self.bkg),
                "floor": floor}


class ChannelOffsets:
    """Where the red and blue images of the stars fall relative to the green one over the field: a quadratic surface
    of the offset (output pixels) per channel, fitted on the isolated unsaturated stars of a stack. Lateral chromatic
    aberration grows towards the edges; atmospheric dispersion shifts the whole field (red up, blue down, by up to a
    pixel low over the horizon). Left alone, the colours of a star land apart: coloured edges, elongated stars."""

    #: fewest stars for a fit; below, nothing is corrected
    MIN_STARS = 50

    def __init__(self, coefs, W: int, H: int, n: int, centre=None, edge=None):
        self.coefs, self.W, self.H, self.n = coefs, W, H, n          # coefs (3, 2, 6): channel, (x, y), terms
        self.centre, self.edge = centre, edge
        self.applied = coefs is not None

    @staticmethod
    def terms(x: torch.Tensor, y: torch.Tensor, W: int, H: int) -> torch.Tensor:
        u, v = x / (W / 2) - 1.0, y / (H / 2) - 1.0
        return torch.stack([torch.ones_like(u), u, v, u * u, u * v, v * v])

    @classmethod
    def measure(cls, model: "Prediction", sat_level: float, S: float, fwhm_px: float) -> "ChannelOffsets":
        """The offsets measured on the stars of the mean ``model``: windowed centroids of each colour on the peaks
        of the green image well above the noise, below half the saturation and alone in their neighbourhood."""
        M, valid = model.M, model.valid.all(0)
        C, H, W = M.shape
        dev = M.device
        sw = S                                                  # centroid window: one sensor pixel (the core)
        r = int(np.ceil(3 * sw))
        G = M[1] - model.bkg[1]
        hp = G[::3, ::3][valid[::3, ::3]]
        noise = float(1.4826 * (hp - hp.median()).abs().median()) if hp.numel() else 0.0
        peak = (F.max_pool2d(G[None, None], 2 * r + 1, 1, r)[0, 0] == G) & valid
        peak &= (G > 50 * noise) & (M.max(0).values < 0.5 * sat_level)
        peak[:2 * r] = False
        peak[-2 * r:] = False
        peak[:, :2 * r] = False
        peak[:, -2 * r:] = False
        # alone: no other peak above a tenth of its height within three windows
        tall = F.max_pool2d(torch.where(peak, G, torch.zeros_like(G))[None, None], 6 * r + 1, 1, 3 * r)[0, 0]
        second = F.max_pool2d(torch.where(peak & (G < tall), G, torch.zeros_like(G))[None, None], 6 * r + 1, 1,
                              3 * r)[0, 0]
        peak &= (G >= tall) & (second < 0.1 * G)
        ys, xs = torch.nonzero(peak, as_tuple=True)
        if len(ys) > 20000:
            keep = torch.argsort(G[ys, xs], descending=True)[:20000]
            ys, xs = ys[keep], xs[keep]
        dd = torch.arange(-r, r + 1, device=dev)
        oy, ox = torch.meshgrid(dd, dd, indexing="ij")
        oy, ox = oy.reshape(-1), ox.reshape(-1)
        yy, xx = ys[:, None] + oy[None], xs[:, None] + ox[None]
        oy, ox = oy.float(), ox.float()
        cen = torch.zeros(3, len(ys), 2, device=dev)
        good = torch.ones(len(ys), dtype=torch.bool, device=dev)
        for c in range(3):
            a = (M[c][yy, xx] - model.bkg[c][ys, xs][:, None]).clamp(min=0)
            cx = torch.zeros(len(ys), device=dev)
            cy = torch.zeros(len(ys), device=dev)
            for _ in range(5):
                w = a * torch.exp(-0.5 * ((ox[None] - cx[:, None]) ** 2 + (oy[None] - cy[:, None]) ** 2) / sw ** 2)
                s = w.sum(1)
                good &= s > 0
                cx = ((w * ox[None]).sum(1) / s.clamp(min=1e-12)).clamp(-r, r)
                cy = ((w * oy[None]).sum(1) / s.clamp(min=1e-12)).clamp(-r, r)
            cen[c, :, 0], cen[c, :, 1] = xs.float() + cx, ys.float() + cy
        n = int(good.sum())
        if n < cls.MIN_STARS:
            return cls(None, W, H, n)
        X, Y = cen[1, good, 0], cen[1, good, 1]
        A = cls.terms(X, Y, W, H).T.double().cpu().numpy()
        coefs = np.zeros((3, 2, 6))
        for c in (0, 2):
            for k in range(2):
                b = (cen[c, good, k] - cen[1, good, k]).double().cpu().numpy()
                m = np.ones(len(b), bool)
                for _ in range(5):
                    q = np.linalg.lstsq(A[m], b[m], rcond=None)[0]
                    e = b - A @ q
                    sg = 1.4826 * np.median(np.abs(e[m])) + 1e-6
                    m = np.abs(e) < 3 * sg
                coefs[c, k] = q
        pts = torch.tensor([[W / 2, H / 2]] + [[x, y] for x in (0, W) for y in (0, H)], dtype=torch.float64)
        T = cls.terms(pts[:, 0], pts[:, 1], W, H).numpy().T
        off = np.einsum("ckt,pt->cpk", coefs, T)               # channel, point, (dx, dy)
        centre = {nm: [round(float(v), 3) for v in off[c, 0]] for c, nm in ((0, "R"), (2, "B"))}
        edge = {nm: round(float(np.hypot(off[c, 1:, 0], off[c, 1:, 1]).max()), 3) for c, nm in ((0, "R"), (2, "B"))}
        return cls(coefs, W, H, n, centre, edge)

    def at(self, x: torch.Tensor, y: torch.Tensor, chmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The offset (output pixels) of each sample in its own channel, where it lands (x, y)."""
        T = self.terms(x, y, self.W, self.H)                               # (6, ...)
        q = torch.from_numpy(self.coefs).to(x.device, torch.float32)       # (3, 2, 6)
        dx = sum(q[:, 0, t][chmap] * T[t] for t in range(6))
        dy = sum(q[:, 1, t][chmap] * T[t] for t in range(6))
        return dx, dy

    def summary(self) -> dict:
        return {"stars": self.n, "applied": self.applied, "centre_px": self.centre, "corner_max_px": self.edge}

    def describe(self) -> str:
        if not self.applied:
            return f"colours not registered on each other: {self.n} isolated stars only"
        return (f"colours registered on each other ({self.n} stars): red {self.centre['R']} px at the centre, up to "
                f"{self.edge['R']} px in the corners; blue {self.centre['B']} px, up to {self.edge['B']} px")


def reject(val, pr: dict, sat, bad, grad, chmap, t: float, p: "StackParams", kappa: float, edge: bool,
           diag: list | None = None):
    """The samples of a normalised frame (``val``, ADU of the reference) deviating from the prediction ``pr``
    (``Prediction.sample``) by more than ``kappa`` sigmas, high and low; ``edge``: with the neighbours of the high
    ones above EDGE_SIGMA. Returns (high, low, inside: the samples with a prediction).

    The sigma of a sample: the shot and read noise of the predicted value in the frame's own ADU (``grad``: the
    frame's sky difference to the reference; gain and read noise), scaled so that the frame's sky samples have unit
    spread (the model's gain matters only through its shape), widened where few frames cover the place; plus two
    allowances for what differs between good frames: the seeing (``core_tolerance`` x the signal above the smooth
    background: star cores) and the photometric scale of the frame (SCALE_TOLERANCE x the extended signal above the
    sky floor). Saturated and bad samples, and those covered by fewer than MIN_COVER frames, are never rejected."""
    pred, cover = pr["pred"], pr["cover"]
    inside = cover > 0
    test = (cover >= MIN_COVER) & ~bad & ~sat & torch.isfinite(val)
    r = val - pred
    raw = (pred * t + grad).clamp(min=0)
    var_n = (raw / p.gain + p.read_noise ** 2) / t ** 2 * (1 + 1 / cover.clamp(min=1))
    allow = (p.core_tolerance * (pred - pr["bkg"]).clamp(min=0)) ** 2 + \
            (SCALE_TOLERANCE * (pr["bkg"] - pr["floor"]).clamp(min=0)) ** 2
    z0 = r / var_n.sqrt()
    sky = test & (allow < 0.25 * var_n)
    s = torch.ones(3, device=val.device)
    for c in range(3):
        v = z0[sky & (chmap == c)]
        v = v[::max(1, v.numel() // 2_000_000)]
        if v.numel() >= 1000:
            s[c] = 1.4826 * (v - v.median()).abs().median()
    s = s.clamp(0.25, 4.0)
    z = r / (var_n * s[chmap] ** 2 + allow).sqrt()
    hi = (z > kappa) & test
    if edge:
        near = F.max_pool2d(hi.float()[None, None], 3, 1, 1)[0, 0] > 0
        hi = hi | (near & (z > EDGE_SIGMA) & test)
    lo = (z < -kappa) & test
    if diag is not None:
        row = {"noise_scale": [round(float(x), 3) for x in s]}
        diag.append(row)
    return hi, lo, inside


#: the scale (sensor pixels, Gaussian sigma) below which a stack's detail comes from its luminance
DETAIL_SCALE = 1.0


def luminance_detail(img: np.ndarray, cover: torch.Tensor, sig: np.ndarray, S: float, dev) -> np.ndarray:
    """The detail of a stack (3, H, W) finer than DETAIL_SCALE sensor pixels taken from its luminance, in each
    place's colour; each colour's own light at that scale kept.

    Each colour is split at a Gaussian of DETAIL_SCALE sensor pixels (x ``S`` output pixels): its fine part ``hp``
    is replaced by the fine part of the luminance (the colours' fine parts weighted by their colour ``lp`` over their
    noise ``sig`` squared: the least noisy estimate of a detail of that colour) times the colour, where that detail is
    noise: the change is weighted by noise / (noise + detail), the detail's power measured on the luminance's fine
    part around the pixel, its noise from the colours' noise (a constant fitted on the image). Then what the change
    did at the scale of the split is taken back: each colour keeps its own light there (the aperture photometry of a
    star, its colour). A star keeps its own fine colour and shape; the finest noise of the sky becomes the
    luminance's, shared by the three colours: less colour grain, the same sharpness. Without the weighting the
    colours of Heart's stars moved by 0.01 mag; without taking back, by 0.08. As a debayering does per frame (the
    colour ratios vary slowly over a pixel), once on the stack. Outside ``cover`` the stack is unchanged."""
    x = torch.from_numpy(np.ascontiguousarray(img)).to(dev)
    vf = cover.float()
    sg = DETAIL_SCALE * float(S)
    nrm = _gauss(vf, sg).clamp(min=1e-6)
    lp = torch.stack([_gauss(x[c] * vf, sg) / nrm for c in range(3)])
    hp = x - lp
    k = lp.clamp(min=0)
    w = k / torch.tensor(sig, device=dev, dtype=torch.float32)[:, None, None] ** 2
    q = (w * k).sum(0).clamp(min=1e-12)
    lum = (w * hp).sum(0) / q
    del w
    # the detail's signal to noise: its power around the pixel over its noise (1 / q times a constant, the median of
    # lum^2 q over the image, where noise dominates, / 0.4549, the median of a chi-square of one degree)
    c0 = float((lum ** 2 * q)[cover][::7].median()) / 0.4549
    snr2 = _gauss(lum ** 2 * vf, sg) / nrm * q / max(c0, 1e-30) - 1
    alpha = 1 / (1 + snr2.clamp(min=0))
    change = alpha[None] * (k * lum[None] - hp)
    del lp, hp, lum, q, snr2, alpha
    change -= torch.stack([_gauss(change[c] * vf, sg) / nrm for c in range(3)])
    out = torch.where(cover[None], x + change, x)
    return out.cpu().numpy()


class RejectionMap:
    """The samples the second pass rejects, counted on the output grid, low and high apart: each sample at the output
    pixel nearest to where it lands, the grid reduced by a whole factor ``f`` so that its longer side is at most
    REJECTION_SIDE. Counts only: it never touches what is deposited."""

    def __init__(self, Ho: int, Wo: int, dev):
        self.f = max(1, -(-max(Ho, Wo) // REJECTION_SIDE))
        self.Ho, self.Wo = Ho, Wo
        self.Hr, self.Wr = -(-Ho // self.f), -(-Wo // self.f)
        self.low = torch.zeros(self.Hr * self.Wr, dtype=torch.int64, device=dev)
        self.high = torch.zeros_like(self.low)

    def _count(self, acc: torch.Tensor, Xo: torch.Tensor, Yo: torch.Tensor, mask: torch.Tensor) -> tuple[int, int]:
        jx = torch.floor(Xo[mask] + 0.5).long()
        iy = torch.floor(Yo[mask] + 0.5).long()
        ok = (jx >= 0) & (jx < self.Wo) & (iy >= 0) & (iy < self.Ho)
        acc += torch.bincount((iy[ok] // self.f) * self.Wr + jx[ok] // self.f, minlength=acc.numel())
        n = int(ok.sum())
        return n, int(mask.sum()) - n

    def add(self, Xo: torch.Tensor, Yo: torch.Tensor, high: torch.Tensor, low: torch.Tensor) -> tuple[int, int, int]:
        """Count one frame's rejected samples (boolean masks on the sensor). Returns (low, high, outside): the low and
        high ones that land on the grid, and those that land outside it (none since version 4: a sample with nothing
        to be compared with is never rejected)."""
        lo, out_lo = self._count(self.low, Xo, Yo, low)
        hi, out_hi = self._count(self.high, Xo, Yo, high)
        return lo, hi, out_lo + out_hi

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        out = []
        for a in (self.low, self.high):
            v = a.view(self.Hr, self.Wr).cpu().numpy()
            out.append(v.astype(np.uint16 if v.max(initial=0) < 65536 else np.uint32))
        return out[0], out[1]

    def save(self, path) -> None:
        low, high = self.arrays()
        np.savez_compressed(path, low=low, high=high, factor=self.f, grid=np.array([self.Wo, self.Ho]))


def noise_step(Ho: int, Wo: int) -> int:
    """The stride of the sample the noise of a stack is measured on (about 4 million pixels per channel)."""
    return max(1, int(np.sqrt(Ho * Wo / 4e6)))


def halves_noise(A: np.ndarray, B: np.ndarray, w_a: float, w_b: float) -> dict[str, float]:
    """Pixel noise (robust sigma, ADU of the stack) of each channel of the full stack, from its two independent
    half-stacks (3, h, w), zero where empty: A - B holds no signal, only the noise of both halves, and the full stack,
    their weighted mean, has sigma(A - B) x sqrt(w_a w_b) / (w_a + w_b) (the frame weights are inverse variances).
    NaN without two halves."""
    f = float(np.sqrt(w_a * w_b) / (w_a + w_b)) if w_a > 0 and w_b > 0 else float("nan")
    out = {}
    for c, name in enumerate("RGB"):
        a, b = np.asarray(A[c], np.float64), np.asarray(B[c], np.float64)
        d = (a - b)[(a != 0) & (b != 0) & np.isfinite(a) & np.isfinite(b)]
        out[name] = float(1.4826 * np.median(np.abs(d - np.median(d))) * f) if d.size else float("nan")
    return out


def select_frames(transp, fwhm, rms, stats, min_transparency: float, max_fwhm_ratio: float, exclude=None,
                  noise=None):
    """Weight of every frame, (transparency / noise)^2 x sharpness, and the reason a frame is left out ('' = kept):
    not aligned, excluded by hand (``exclude``: a boolean per frame), transparency below ``min_transparency`` x the
    best frames, stars larger than ``max_fwhm_ratio`` x the median. The thresholds are measured on all the frames,
    so excluding a frame by hand never changes the fate of another. The noise: the registration's (``noise``, reg.npz
    since version 2: the pixel noise alone), else the spread of the frame's pixels (``stats``, frame_stats.npy:
    the nebulae and stars of the field in it too)."""
    n = len(transp)
    sig = stats[:, :, 1].mean(1)
    if noise is not None and np.isfinite(noise).all() and (np.asarray(noise) > 0).all():
        sig = np.asarray(noise, float)
    weight = (transp / sig) ** 2 * (np.percentile(fwhm, 5) / fwhm) ** 2
    weight = np.where(np.isfinite(rms) & np.isfinite(weight), weight, 0.0)
    weight /= weight.max()
    tmin = min_transparency * np.percentile(transp[np.isfinite(rms)], 90)
    reasons = np.full(n, "", dtype=object)
    reasons[~np.isfinite(rms)] = "not aligned"
    if exclude is not None:
        reasons[(reasons == "") & np.asarray(exclude, bool)] = EXCLUDED
    reasons[(reasons == "") & (transp < tmin)] = "low transparency"
    reasons[(reasons == "") & (fwhm > max_fwhm_ratio * np.median(fwhm))] = "large stars"
    weight[reasons != ""] = 0.0
    return weight, reasons


def excluded_mask(paths: list[str], exclude: list[str]) -> np.ndarray:
    """Which of the frames (their files, as the inventory lists them) are among the ``exclude`` files. Paths are
    compared normalised (case, separators)."""
    def norm(f: str) -> str:
        return os.path.normcase(os.path.normpath(str(f)))
    ex = {norm(f) for f in exclude}
    return np.array([norm(f) in ex for f in paths], bool)


def fixed_pattern(mm, used, coef, dev, H, W, nframes, progress=lambda f: None):
    """Sensor fixed pattern, per Bayer plane, in sensor coordinates.

    The sky drifts across the sensor while the pattern stays: stars and sky structures smaller than the drift fall
    on a pixel in a few frames only. Pass 1: median over ``nframes`` frames spread over the session (frames minus
    their smooth sky). Pass 2: mean over all used frames of the values within 3 sigma of that median (stars, cosmic
    rays and trails rejected), streamed, so that the estimate is precise enough not to add its own noise. Scales
    above a quarter of the drift are not kept (larger sky structures are not averaged out).
    Returns (pattern (H, W) on ``dev`` or None, info).
    """
    sel = used[np.unique(np.linspace(0, len(used) - 1, min(nframes, len(used))).round().astype(int))]
    tx, ty = coef[used, 0, 0], coef[used, 1, 0]                            # field-centre position (pixels)
    drift = float(np.hypot(np.ptp(tx), np.ptp(ty)))
    if drift < 12 or len(used) < 8:
        return None, {"applied": False, "drift_px": round(drift, 1), "frames": int(len(used)),
                      "msg": f"fixed pattern not measured: the sky drifted by {drift:.0f} px only over the session "
                             f"(too little to tell the pattern from the sky)"}
    Hh, Wh = H // 2, W // 2

    def residuals(i):
        """(4, Hh, Wh) frame minus its smooth sky per Bayer plane (NaN where saturated / bad), noise per plane."""
        fr = torch.from_numpy(np.asarray(mm[i], np.float32)).to(dev)
        out, sg = [], []
        for k in range(4):
            ky, kx = divmod(k, 2)
            pl = fr[ky::2, kx::2][:Hh, :Wh]
            pl = torch.where(pl > VALID_MAX, torch.nan, pl)
            sky = _plane_sky(pl)
            r = pl - sky
            sub = r[::7, ::7]
            sg.append(1.4826 * torch.nanmedian((sub - torch.nanmedian(sub)).abs()))
            out.append(r)
        return torch.stack(out), torch.stack(sg)

    planes = np.zeros((len(sel), 4, Hh, Wh), np.float16)
    for j, i in enumerate(sel):
        planes[j] = residuals(i)[0].cpu().numpy().astype(np.float16)
        progress(0.3 * (j + 1) / len(sel))
    med = torch.zeros(4, Hh, Wh, device=dev)
    for y0 in range(0, Hh, 128):
        blk = torch.from_numpy(planes[:, :, y0:y0 + 128].astype(np.float32)).to(dev)
        med[:, y0:y0 + 128] = torch.nanmedian(blk, dim=0).values.nan_to_num(0.0)
    del planes
    acc = torch.zeros(4, Hh, Wh, device=dev)
    cnt = torch.zeros(4, Hh, Wh, device=dev)
    for j, i in enumerate(used):
        r, sg = residuals(i)
        keep = ((r - med).abs() < 3 * sg[:, None, None]) & torch.isfinite(r)
        acc += torch.where(keep, r, torch.zeros_like(r))
        cnt += keep.float()
        progress(0.3 + 0.7 * (j + 1) / len(used))
    pat = torch.where(cnt > 0, acc / cnt.clamp(min=1), med)
    sig = float(np.clip(drift / 2 / 4, 2.0, 16.0))
    out = torch.zeros(H, W, device=dev)
    amp = []
    for k in range(4):
        ky, kx = divmod(k, 2)
        hp = pat[k] - _gauss(pat[k], sig)
        sub = hp[::3, ::3]
        amp.append(float(1.4826 * torch.median((sub - torch.median(sub)).abs())))
        out[ky::2, kx::2][:Hh, :Wh] = hp
    info = {"applied": True, "drift_px": round(drift, 1), "frames": int(len(used)), "scale_px": round(2 * sig, 1),
            "amplitude_adu": [round(a, 3) for a in amp],
            "msg": f"fixed pattern measured on {len(used)} frames (sky drift {drift:.0f} px): amplitude "
                   f"{np.mean(amp):.2f} ADU per pixel, removed from every frame"}
    return out, info


def _plane_sky(pl: torch.Tensor, b: int = 32) -> torch.Tensor:
    """Smooth sky of a Bayer plane: medians of b x b blocks, bilinearly interpolated."""
    Hh, Wh = pl.shape
    nby, nbx = Hh // b, Wh // b
    blk = pl[: nby * b, : nbx * b].reshape(nby, b, nbx, b).permute(0, 2, 1, 3).reshape(nby, nbx, b * b)
    m = torch.nanmedian(blk, dim=-1).values
    m = torch.where(torch.isfinite(m), m, torch.nanmedian(m))
    up = F.interpolate(m[None, None], size=(nby * b, nbx * b), mode="bilinear", align_corners=False)
    return F.pad(up, (0, Wh - nbx * b, 0, Hh - nby * b), mode="replicate")[0, 0]


def _gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = int(4 * sigma + 0.5)
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    y = F.pad(x[None, None], (r, r, r, r), mode="reflect")
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    return F.conv2d(y, k.view(1, 1, -1, 1))[0, 0]
