"""Physical background: star flat, sky model anchored on the H-alpha survey and on Gaia unresolved starlight."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from ..core.imgutil import bin_median, normalized_blur, poly_terms, robust_lstsq
from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage


class BackgroundParams(StageParams):
    sky_order: int = P(3, label="Sky gradient complexity", ge=1, le=5,
                       help="Degree of the smooth polynomial describing light pollution and sky gradients.",
                       tip="2-3 for most fields; higher for very wide fields with complex gradients. Too high can "
                           "absorb large faint nebulae when no H-alpha reference is used.")
    bin_px: int = P(32, label="Sky sampling", level=ADVANCED, ge=8, le=256, unit="px",
                    help="Size of the blocks in which the background is sampled for the fit.")
    star_flat: bool = P(True, label="Star flat correction",
                        help="Measures the residual error of your flat field directly on the stars: their measured "
                             "flux divided by the flux predicted from Gaia gives a smooth map of the real "
                             "sensitivity of the instrument across the field, which corrects the image.")
    large_scale_px: float = P(200.0, label="Large-scale correction", level=ADVANCED, ge=0, le=2000, unit="px",
                              help="Above this scale, the H-alpha emission of the image is matched to the survey "
                                   "(multi-scale gradient correction). 0 disables it.")


class Background(Stage):
    name = "background"
    title = "Physical sky background"
    description = ("Separates the real sky from the instrument and the atmosphere. 1) Star flat: Gaia predicts the "
                   "flux of thousands of stars, so the ratio measured/predicted maps the residual vignetting and flat "
                   "errors, which are corrected. 2) Sky: the background is modelled as H-alpha emission (shaped like "
                   "the professional survey) + unresolved starlight (shaped like the faint Gaia stars) + a smooth "
                   "gradient. Only the gradient is removed. 3) The zero point of the continuum is set by the light "
                   "of the stars too faint to be resolved, extrapolated from Gaia counts - not by guessing a black "
                   "point. Faint emission (IFN-like structures) is therefore kept with its true brightness.")
    deps = ("starless", "stars", "optics", "surveys", "astrometry", "stack", "color", "catalogs")
    Params = BackgroundParams
    version = "1"

    def auto(self, ctx: Context) -> dict[str, Auto]:
        try:
            sc = float(ctx.metric_of("astrometry", "scale_arcsec"))
            px = round(14 * 60 / sc / 10) * 10
            return {"large_scale_px": Auto(float(px), f"14 arcmin at {sc:.2f}\"/px")}
        except Exception:  # noqa: BLE001
            return {}

    def run(self, ctx: Context, p: BackgroundParams) -> None:
        dm = float(ctx.metric_of("optics", "mag_offset", 0.0))
        fs = np.load(ctx.input("stars", "star_fluxes.npz"))
        sl = ctx.load("starless", "sl_final.npy").astype(np.float32)
        H, W = sl.shape[1:]
        survey = ctx.load_json("surveys", "survey.json")["survey"]
        ha = ctx.load("surveys", "ref_ha.npy") if survey != "none" else None
        g_faint = float(np.max(np.load(ctx.input("catalogs", "gaia_faint.npz"))["g"], initial=0.0))
        g_deep = float(np.nanmax(fs["g"]))
        ctx.progress(0.05, "star flat")
        if p.star_flat:
            flat, finfo = star_flat(fs, ctx.load("optics", "lat_full.npy", mmap=True), W, H, dm)
            ctx.metric("flat_error_pct", [round(100 * float(flat.min() - 1), 1), round(100 * float(flat.max() - 1), 1)])
            ctx.log(f"star flat: the flat field was wrong by {100 * (flat.min() - 1):+.1f} to {100 * (flat.max() - 1):+.1f} %")
        else:
            flat, finfo = np.ones_like(sl), {}
        ctx.save("flat.npy", flat)
        sl /= flat
        ctx.progress(0.3, "sky model")
        U = unresolved_template(fs, W, H, p.bin_px, g_deep)
        wt = ctx.load("stack", "drz_weight.npy", mmap=True)
        cover = bin_median(np.asarray(wt[1], np.float32), p.bin_px)
        Hb = bin_median(ha, p.bin_px) if ha is not None else None
        hb, wb = cover.shape
        yb, xb = (np.mgrid[0:hb, 0:wb] + 0.5) * p.bin_px - 0.5
        Tp = poly_terms(xb.ravel(), yb.ravel(), W, H, p.sky_order)
        good = cover > 0.5 * np.nanmedian(cover)
        if Hb is not None:
            good &= np.isfinite(Hb)
        sky, coefs = [], {}
        yy, xx = np.mgrid[0:H:16, 0:W:16].astype(np.float64)
        Tfull = poly_terms(xx, yy, W, H, p.sky_order)
        for c in range(3):
            Y = bin_median(sl[c], p.bin_px)
            cols = ([Hb.ravel()] if Hb is not None else []) + [U[c].ravel()]
            A = np.c_[np.stack(cols, 1), Tp]
            ok = good.ravel() & np.isfinite(Y.ravel()) & np.isfinite(A).all(1)
            p0 = np.linalg.lstsq(A[ok], Y.ravel()[ok], rcond=None)[0]
            s0 = 1.4826 * np.median(np.abs(A[ok] @ p0 - Y.ravel()[ok]))
            q = least_squares(lambda v: A[ok] @ v - Y.ravel()[ok], p0, loss="huber", f_scale=1.5 * s0).x
            nh = 1 if Hb is not None else 0
            coefs[c] = {"k_ha": float(q[0]) if nh else None, "u": float(q[nh]), "poly": q[nh + 1:].tolist()}
            P_ = (Tfull @ q[nh + 1:]).astype(np.float32)
            sky.append(cv2.resize(P_, (W, H), interpolation=cv2.INTER_CUBIC))
            msg = f"channel {'RGB'[c]}: " + (f"H-alpha gain {q[0]:.4f} ADU/R, " if nh else "") + \
                  f"gradient {np.percentile(P_, 1):.1f}..{np.percentile(P_, 99):.1f} ADU"
            ctx.log(msg)
        sky = np.stack(sky)
        astro = (sl - sky).astype(np.float32)
        del sl
        # continuum zero point: unresolved starlight below the faintest removed star, extrapolated from Gaia counts
        gg, prr = fs["g"], fs["pred"]
        b1 = prr[:, (gg >= g_deep - 1.0) & (gg < g_deep - 0.5)].sum(1)
        b2 = prr[:, (gg >= g_deep - 0.5) & (gg < g_deep)].sum(1)
        rr = float(np.median(b2 / np.maximum(b1, 1e-9)))
        rr = min(rr, 0.95)
        f_ext = (rr / (1 - rr)) * b2 / np.maximum(b1 + b2, 1e-9)
        zg = max(g_faint, g_deep)
        f_ext = f_ext * rr ** ((zg - g_deep) / 0.5)
        low = good & (Hb < np.nanpercentile(Hb[good], 25)) if Hb is not None else good
        offs = []
        for c in range(3):
            Yc = bin_median(astro[c], p.bin_px)
            if Hb is not None:
                Yc = Yc - coefs[c]["k_ha"] * Hb
            want = f_ext[c] * U[c] / (p.bin_px ** 2)
            offs.append(float(np.nanmedian((want - Yc)[low])))
            astro[c] += offs[-1]
        ctx.metric("continuum_zero_offsets", [round(o, 2) for o in offs])
        ctx.metric("unresolved_light_adu", [round(float(np.nanmedian(f_ext[c] * U[c] / p.bin_px ** 2)), 2) for c in range(3)])
        if Hb is not None:
            k = np.array([coefs[c]["k_ha"] for c in range(3)])
            ctx.metric("ha_signature", [1.0, round(k[1] / k[0], 3), round(k[2] / k[0], 3)])
        # large-scale correction (MGC-like) towards the survey
        if Hb is not None and p.large_scale_px > 0:
            ctx.progress(0.7, "large-scale correction")
            Ab = [bin_median(astro[c], p.bin_px).ravel() for c in range(3)]
            Hn = Hb.ravel()
            okb = good.ravel() & np.isfinite(Hn) & np.isfinite(Ab[0])
            faint = okb & (Hn < np.nanpercentile(Hn[okb], 60))
            q = np.linalg.lstsq(np.c_[Hn[faint], Ab[1][faint], Ab[2][faint], np.ones(faint.sum())], Ab[0][faint],
                                rcond=None)[0]
            Hm = astro[0] - q[1] * astro[1] - q[2] * astro[2]
            valid = np.isfinite(ha)
            m = valid.astype(np.float32)
            dH = normalized_blur(q[0] * np.where(valid, ha, 0).astype(np.float32), m, p.large_scale_px) - \
                normalized_blur(Hm, m, p.large_scale_px)
            covm = cv2.resize(cv2.GaussianBlur(cv2.resize(m, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA),
                                               (0, 0), 10), (W, H))
            dH *= np.clip(covm * 1.5, 0, 1)
            k = np.array([coefs[c]["k_ha"] for c in range(3)])
            for c in range(3):
                astro[c] += dH * k[c] / k[0]
            ctx.metric("large_scale_correction_adu", [round(float(np.percentile(dH[::8, ::8], v)), 2) for v in (1, 50, 99)])
            ctx.save_json("ha_extraction.json", {"q": q.tolist()})
        ctx.save("astro.npy", astro)
        ctx.save_json("coefs.json", {"fit": coefs, "flat": finfo, "survey": survey})
        ctx.save("sky.npy", sky)
        ctx.metric("emission_dips", emission_dips(astro, fs, W, H, dm))
        from ..core.imgutil import asinh_preview
        ctx.preview("background", asinh_preview(astro, ctx.load("color", "M.npy"), k=30))


def star_flat(fs, img, W, H, dm):
    x, y, g, fl, pr = fs["x"], fs["y"], fs["g"] - dm, fs["flux"], fs["pred"]
    m = (g > 9.5) & (g < 13.0) & (x > 50) & (x < W - 50) & (y > 50) & (y < H - 50)
    yy, xx = np.mgrid[0:H:16, 0:W:16].astype(np.float64)
    maps, info = [], {}
    for c in (0, 1):
        r = np.log(np.clip(fl[c, m], 1e-3, None) / pr[c, m])
        q, s, ok = robust_lstsq(poly_terms(x[m], y[m], W, H, 3), r)
        maps.append(poly_terms(xx, yy, W, H, 3) @ q)
        info[c] = {"n": int(ok.sum()), "scatter": float(s)}
    # blue: wide aperture on bright isolated stars as a check; the map uses the mean R/G shape (the very spread
    # blue PSF of the corners biases blue PSF photometry)
    from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry
    t = cKDTree(np.c_[x, y])
    cand = np.where((g > 8.5) & (g < 11.0) & (x > 60) & (x < W - 60) & (y > 60) & (y < H - 60))[0]
    iso = np.array([i for i in cand if all((j == i) or (g[j] > g[i] + 3.5) for j in t.query_ball_point([x[i], y[i]], 22))],
                   int)
    mrg = 0.5 * (maps[0] + maps[1])
    if len(iso) > 20:
        xy = np.c_[x[iso], y[iso]]
        im = np.asarray(img[2], np.float32)
        ap, an = CircularAperture(xy, r=18), CircularAnnulus(xy, r_in=24, r_out=32)
        flb = aperture_photometry(im, ap)["aperture_sum"].data - ApertureStats(im, an).median * ap.area
        r = np.log(np.clip(flb, 1e-3, None) / pr[2, iso])
        q, s, ok = robust_lstsq(poly_terms(xy[:, 0], xy[:, 1], W, H, 2), r)
        mb = poly_terms(xx, yy, W, H, 2) @ q
        info[2] = {"n": int(ok.sum()), "scatter": float(s),
                   "aperture_check_pct": float(100 * np.std((mb - np.median(mb)) - (mrg - np.median(mrg))))}
    maps.append(mrg)
    flat = [cv2.resize(np.exp(mp - np.median(mp)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) for mp in maps]
    return np.stack(flat), info


def unresolved_template(fs, W, H, b, g_deep):
    x, y, g, pr = fs["x"], fs["y"], fs["g"], fs["pred"]
    m = (g > g_deep - 1.0) & (g <= g_deep)
    out = []
    for c in range(3):
        acc = np.zeros((H // b, W // b))
        xi = np.clip((x[m] / b).astype(int), 0, W // b - 1)
        yi = np.clip((y[m] / b).astype(int), 0, H // b - 1)
        np.add.at(acc, (yi, xi), pr[c, m])
        out.append(cv2.GaussianBlur(acc.astype(np.float32), (0, 0), 2.0))
    return np.stack(out)


def emission_dips(astro, fs, W, H, dm):
    """Number of stars (G < 17) sitting in a compact emission dip deeper than 4 sigma (should stay low)."""
    E = astro[0] - 0.87 * astro[1]
    Es = cv2.GaussianBlur(E, (0, 0), 1.0)
    loc = cv2.GaussianBlur(E, (0, 0), 6)
    hp = E - cv2.GaussianBlur(E, (0, 0), 2)
    sg = 1.4826 * np.median(np.abs(hp[::7, ::7]))
    g, x, y = fs["g"] - dm, fs["x"], fs["y"]
    sel = np.where((g < 17) & (x > 10) & (x < W - 10) & (y > 10) & (y < H - 10))[0]
    xi, yi = np.round(x[sel]).astype(int), np.round(y[sel]).astype(int)
    d = (Es[yi, xi] - loc[yi, xi]) / sg
    return {"below_4sigma": int((d < -4).sum()), "below_8sigma": int((d < -8).sum()), "stars": int(len(sel))}
