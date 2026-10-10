"""Weighted Bayer drizzle on the GPU with predictive outlier rejection and two independent half-stacks."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from ..core.imgutil import asinh_preview
from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage, StageError
from ..io.frames import cfa_channel_map
from .calibrate import VALID_MAX, open_cache


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
    version = "1"

    def auto(self, ctx: Context) -> dict[str, Auto]:
        a: dict[str, Auto] = {}
        reg = np.load(ctx.input("register", "reg.npz"))
        n = len(reg["fwhm"])
        f = float(np.median(reg["fwhm"]))
        if f < 3.0 and n >= 150:
            a["scale"] = Auto(2.0, f"small stars (FWHM {f:.1f} px) and {n} frames")
        elif f < 5.5 and n >= 40:
            a["scale"] = Auto(1.5, f"FWHM {f:.1f} px with {n} frames: under-sampled, enough frames to drizzle")
        else:
            a["scale"] = Auto(1.0, f"FWHM {f:.1f} px / {n} frames: drizzling would add noise without detail")
        ok = np.isfinite(reg["rms"])
        t = reg["transp"]
        frac = float(np.mean(t[ok] < 0.6 * np.percentile(t[ok], 90)))
        if frac > 0.5:
            a["min_transparency"] = Auto(0.3, f"{100 * frac:.0f} % of frames are below 60 % transparency: only the "
                                              "worst (below 30 %) are excluded")
        elif frac > 0:
            a["min_transparency"] = Auto(0.6, f"{100 * frac:.0f} % of frames lost more than 40 % of the signal "
                                              "(clouds or haze): excluded")
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

        sig = stats[:, :, 1].mean(1)
        weight = (transp / sig) ** 2 * (np.percentile(fwhm, 5) / fwhm) ** 2
        weight = np.where(np.isfinite(rms) & np.isfinite(weight), weight, 0.0)
        weight /= weight.max()
        tmin = p.min_transparency * np.percentile(transp[np.isfinite(rms)], 90)
        reasons = np.full(n, "", dtype=object)
        reasons[~np.isfinite(rms)] = "not aligned"
        reasons[(reasons == "") & (transp < tmin)] = "low transparency"
        reasons[(reasons == "") & (fwhm > p.max_fwhm_ratio * np.median(fwhm))] = "large stars"
        weight[reasons != ""] = 0.0
        used = np.where(weight > 0)[0]
        if len(used) < 2:
            raise StageError("fewer than 2 usable frames after selection; relax the selection thresholds")
        meta = ctx.load_json("inventory", "frames.json")["lights"]
        ctx.save_json("frames.json", [{"path": meta[i]["path"], "time": meta[i].get("time"), "weight": float(weight[i]),
                                       "transparency": float(transp[i]), "fwhm": float(fwhm[i]),
                                       "excluded": reasons[i]} for i in range(n)])
        for r in ("low transparency", "large stars", "not aligned"):
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

        fr = torch.from_numpy(np.asarray(mm[ref], np.float32)).to(dev)
        Bref_np = block_bkg(fr)[0].cpu().numpy()
        from scipy.ndimage import map_coordinates

        def ref_bkg_at(k, u, v):
            ky, kx = divmod(k, 2)
            pj = ((u - kx) / 2 + 0.5) / 32 - 0.5
            pi = ((v - ky) / 2 + 0.5) / 32 - 0.5
            return map_coordinates(Bref_np[k], [pi.ravel(), pj.ravel()], order=1, mode="nearest").reshape(pi.shape)

        def frame_model(i):
            fr = torch.from_numpy(np.asarray(mm[i], np.float32)).to(dev)
            X, Y = poly_eval(coef[i], xx, yy)
            Bi, cx, cy = block_bkg(fr)
            Q = torch.zeros(3, 6, dtype=torch.float64)
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
                Q[c] = torch.from_numpy(q)
                bkg_level[c] = np.nanmedian(np.concatenate([Bi[k].cpu().numpy().ravel() for k in ks]))
            a = X / (W / 2) - 1.0
            b = Y / (H / 2) - 1.0
            terms = torch.stack([torch.ones_like(a), a, b, a * a, a * b, b * b])
            Qs = Q.to(dev, torch.float32)[chmap]
            grad = (Qs.permute(2, 0, 1) * terms).sum(0)
            sat = fr > VALID_MAX
            bad = torch.isnan(fr)
            val = torch.where(sat, satval, fr)
            val = (val - grad) / transp[i]
            return val, (X + 0.5) * S - 0.5, (Y + 0.5) * S - 0.5, sat, bad, bkg_level

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
                    den.index_add_(0, idx, ww[ok])

        # ---------------- pass 1: weighted mean
        num = torch.zeros(3 * Ho * Wo, device=dev)
        den = torch.zeros(3 * Ho * Wo, device=dev)
        for c_, i in enumerate(used):
            val, Xo, Yo, sat, bad, _ = frame_model(i)
            deposit(num, den, val, Xo, Yo, float(weight[i]), ~bad)
            ctx.progress(0.5 * (c_ + 1) / len(used), f"pass 1: {c_ + 1}/{len(used)} frames")
        M = (num / den.clamp(min=1e-12)).view(3, Ho, Wo)
        M = torch.where(den.view(3, Ho, Wo) > 0, M, torch.zeros_like(M))
        del num, den

        # ---------------- pass 2: predictive rejection + half-stacks
        accA = [torch.zeros(3 * Ho * Wo, device=dev) for _ in range(2)]
        accB = [torch.zeros(3 * Ho * Wo, device=dev) for _ in range(2)]
        Mb = M.unsqueeze(0)
        rej_tot = 0
        for c_, i in enumerate(used):
            val, Xo, Yo, sat, bad, bkg = frame_model(i)
            gx = (Xo + 0.5) / Wo * 2 - 1
            gy = (Yo + 0.5) / Ho * 2 - 1
            grid = torch.stack([gx, gy], -1).unsqueeze(0)
            pred_all = F.grid_sample(Mb, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
            pred = torch.gather(pred_all, 0, chmap.unsqueeze(0))[0]
            bk = torch.tensor(bkg, device=dev, dtype=torch.float32)[chmap]
            var = (((pred * transp[i] + bk).clamp(min=1) / p.gain + p.read_noise ** 2) / transp[i] ** 2
                   + (p.core_tolerance * pred.clamp(min=0)) ** 2)
            z = (val - pred) / var.sqrt()
            hi = (z > p.kappa) & ~sat
            hi = F.max_pool2d(hi.float()[None, None], 3, 1, 1)[0, 0] > 0
            rej = hi | ((z < -p.kappa) & ~sat)
            keep = ~bad & ~rej
            rej_tot += int(rej.sum())
            acc = accA if (i // p.half_block) % 2 == 0 else accB
            deposit(acc[0], acc[1], val, Xo, Yo, float(weight[i]), keep)
            ctx.progress(0.5 + 0.5 * (c_ + 1) / len(used), f"pass 2: {c_ + 1}/{len(used)} frames, "
                                                          f"rejected {100 * rej_tot / ((c_ + 1) * H * W):.2f} %")

        def finish(a):
            img = (a[0] / a[1].clamp(min=1e-12)).view(3, Ho, Wo)
            return torch.where(a[1].view(3, Ho, Wo) > 0, img, torch.zeros_like(img)).cpu().numpy()

        full = [accA[0] + accB[0], accA[1] + accB[1]]
        drz = finish(full)
        ctx.save("drz_full.npy", drz)
        ctx.save("drz_A.npy", finish(accA))
        ctx.save("drz_B.npy", finish(accB))
        ctx.save("drz_weight.npy", full[1].view(3, Ho, Wo).cpu().numpy())
        rej_pct = 100 * rej_tot / (len(used) * H * W)
        ctx.metric("rejected_pct", round(rej_pct, 3))
        ctx.save_json("drz_meta.json", {"scale": S, "pixfrac": PIX, "n": int(len(used)), "reject_pct": rej_pct,
                                        "width": Wo, "height": Ho, "sensor": [W, H]})
        ctx.log(f"drizzle {Wo}x{Ho} from {len(used)} frames in {(time.time() - t0) / 60:.1f} min, "
                f"{rej_pct:.2f} % pixels rejected")
        ctx.preview("stack", asinh_preview(drz))
