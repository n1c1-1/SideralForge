"""Physical background: star flat, sky model anchored on the H-alpha survey and on Gaia unresolved starlight."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree

from ..core.imgutil import bin_median, normalized_blur, poly_terms, robust_lstsq
from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage

HALO_PRECISION = 0.2      # precision of the removal of the far halo of the stars, relative to it (around Alnitak the
                          # data depart from the model by -5 % at 90 px to +25 % at 300 px)


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
    object_clip: float = P(2.5, label="Object rejection", level=ADVANCED, ge=1.0, le=10, step=0.1, unit="sigma",
                           help="Blocks brighter than the sky model by more than this are taken as objects (galaxy, "
                                "reflection nebula, dust glow not described by the survey) and left out of the sky fit, "
                                "iteratively. The sky then passes under the data instead of through extended objects, "
                                "which would leave the darker areas below zero.")


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
    version = "5"          # 3: zero-point floor on the darkest areas; 4: not set where the removed star glare dominates;
                           # 5: star flat from wide apertures when the star models' fluxes scatter twice as much;
                           # in an emission field, objects rejected on green and blue when the H-alpha amplitude
                           # collapses

    def auto(self, ctx: Context) -> dict[str, Auto]:
        try:
            sc = float(ctx.metric_of("astrometry", "scale_arcsec"))
            px = round(14 * 60 / sc / 10) * 10
            return {"large_scale_px": Auto(float(px), "14 arcmin at {scale:.2f}\"/px", {"scale": sc})}
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
            fw = float(ctx.metric_of("register", "fwhm_px")) * float(ctx.load_json("stack", "drz_meta.json")["scale"])
            flat, finfo = star_flat(fs, ctx.load("optics", "lat_full.npy", mmap=True), W, H, dm, fw)
            ctx.metric("flat_error_pct", [round(100 * float(flat.min() - 1), 1), round(100 * float(flat.max() - 1), 1)])
            ctx.metric("star_flat_photometry", finfo["photometry"])
            if finfo["photometry"] == "aperture":
                ctx.log(f"star flat from wide apertures (radius {finfo['aperture_px']:.0f} px): the fluxes of the star "
                        f"models scatter {finfo['scatter_psf']:.3f} around the flat, the apertures "
                        f"{finfo['scatter_aperture']:.3f} (star shapes change across the field)")
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
        Yb = np.stack([bin_median(sl[c], p.bin_px) for c in range(3)])
        h_cam = np.load(ctx.input("color", "specresp.npz"))["cam_hii"].astype(np.float64)
        sol, used, how = fit_sky(Yb, Hb, U, Tp, good, h_cam / h_cam[0], p.object_clip)
        ctx.metric("sky_fit_blocks_pct", round(100 * float(used.sum()) / max(int(good.sum()), 1), 1))
        ctx.metric("ha_fit", how)
        ctx.log(f"sky fitted on {100 * used.sum() / max(good.sum(), 1):.0f} % of the field (the rest is taken as "
                f"objects); H-alpha signature: {how}")
        nh = 1 if Hb is not None else 0
        for c in range(3):
            q = sol[c]
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
        # what can set the floor below: blocks measured, not interpolated (the filled star cores), and where the far
        # halo of the stars removed there is small enough that the precision of its removal cannot leave them darker
        # than nothing (on Horsehead these blocks raised the floor by ~5 ADU, unequally in the three channels: a blue
        # cast on the whole sky)
        fill = bin_median(ctx.load("starless", "sl_mask.npy").astype(np.float32), p.bin_px)
        measured = good & (fill < 0.1)
        hf = ctx.project.stage_dir("stars") / "stars_model_halo.npy"
        hb = None
        if hf.exists():
            hal = np.load(hf, mmap_mode="r")
            hb = np.stack([bin_median(np.asarray(hal[c], np.float32), p.bin_px) for c in range(3)])
            del hal
        offs = []
        for c in range(3):
            Yc = bin_median(astro[c], p.bin_px)
            if Hb is not None:
                Yc = Yc - coefs[c]["k_ha"] * Hb
            want = f_ext[c] * U[c] / (p.bin_px ** 2)
            off = float(np.nanmedian((want - Yc)[low]))
            # physical floor: the sky cannot be darker than nothing - no significant area may end below zero
            floor = -float(np.nanpercentile(Yc[measured], 0.2))   # the darkest areas, not only 98 % of the field
            if hb is not None:
                for _ in range(10):
                    ok_ = measured & (HALO_PRECISION * hb[c] <= max(floor, 0.0))
                    if ok_.sum() < 0.5 * good.sum():
                        break
                    f2 = -float(np.nanpercentile(Yc[ok_], 0.2))
                    if f2 >= floor - 1e-3:
                        break
                    floor = f2
            if floor > off:
                ctx.log(f"channel {'RGB'[c]}: continuum zero point raised by {floor - off:.2f} ADU so that the darkest "
                        f"areas (dark nebulae) stay above zero")
                off = floor
            offs.append(off)
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


#: an emission field: the H-alpha survey reaches this brightness (99th percentile, rayleigh) - the user's emission
#: fields 400-1250 R, the galaxies and reflection nebulae 3-110 R
EMISSION_FIELD_R = 200.0


def fit_sky(Yb, Hb, U, Tp, good, h_cam, clip, iters=10):
    """Sky model per channel: Yb = k_ha Hb + u U + polynomial, with k_ha >= 0 and u >= 0.

    Blocks brighter than the model by more than ``clip`` robust sigmas are objects and are left out, iteratively
    (the sky passes under extended objects). The H-alpha colour must be compatible with the camera's HII signature
    ``h_cam`` (R = 1); otherwise the survey is correlated with something else in the field (reflection nebula,
    galaxy) and the H-alpha term is refitted with the camera signature imposed.
    Returns (coefficients per channel, blocks used, description of the H-alpha fit).
    """
    nh = 1 if Hb is not None else 0
    npoly = Tp.shape[1]
    base = good.ravel() & np.isfinite(Yb).all(0).ravel() & np.isfinite(U).all(0).ravel()
    if nh:
        base &= np.isfinite(Hb.ravel())

    def design(c):
        cols = ([Hb.ravel()] if nh else []) + [U[c].ravel()]
        return np.c_[np.stack(cols, 1), Tp]

    def solve(mask, shared):
        """Per-channel bounded least squares, or one shared H-alpha amplitude a (k_ha[c] = a h_cam[c])."""
        if not shared:
            out = []
            for c in range(3):
                A = design(c)
                lb = np.r_[np.zeros(nh + 1), -np.inf * np.ones(npoly)]
                out.append(lsq_linear(A[mask], Yb[c].ravel()[mask], bounds=(lb, np.inf)).x)
            return out
        rows, ys = [], []
        n = 1 + 3 * (1 + npoly)
        for c in range(3):
            A = design(c)[mask]
            sc = 1.4826 * np.median(np.abs(Yb[c].ravel()[mask] - np.median(Yb[c].ravel()[mask]))) + 1e-9
            M_ = np.zeros((A.shape[0], n))
            M_[:, 0] = A[:, 0] * h_cam[c]
            M_[:, 1 + c * (1 + npoly):1 + (c + 1) * (1 + npoly)] = A[:, 1:]
            rows.append(M_ / sc)
            ys.append(Yb[c].ravel()[mask] / sc)
        lb = -np.inf * np.ones(n)
        lb[0] = 0
        for c in range(3):
            lb[1 + c * (1 + npoly)] = 0
        x = lsq_linear(np.vstack(rows), np.concatenate(ys), bounds=(lb, np.inf)).x
        return [np.r_[x[0] * h_cam[c], x[1 + c * (1 + npoly):1 + (c + 1) * (1 + npoly)]] for c in range(3)]

    def plausible(sol):
        if not nh or sol[0][0] <= 0:
            return True
        k = np.array([s[0] for s in sol])
        return k[1] / k[0] <= 2.0 * h_cam[1] and k[2] / k[0] <= 2.5 * h_cam[2]

    def clip_objects(sol, mask, channels=(0, 1, 2)):
        keep = base.copy()
        for c in channels:
            r = Yb[c].ravel() - design(c) @ sol[c]
            s = 1.4826 * np.median(np.abs(r[mask] - np.median(r[mask]))) + 1e-9
            obj = (r - np.median(r[mask]) > clip * s).reshape(Yb.shape[1:])
            obj = cv2.dilate(obj.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            keep &= ~obj.ravel()
        return keep if keep.sum() >= 0.1 * base.sum() else mask

    def fit(channels):
        shared = False
        mask = base
        sol = solve(mask, shared)
        k_all = float(sol[0][0]) if nh else 0.0
        for _ in range(iters):
            if not plausible(sol):
                shared = True
                sol = solve(mask, shared)
                if _ == 0 and nh:
                    k_all = float(sol[0][0])
            new = clip_objects(sol, mask, channels)
            if (new == mask).all():
                break
            mask = new
            sol = solve(mask, shared)
        if not plausible(sol):
            shared = True
            sol = solve(mask, shared)
        return sol, mask, shared, k_all

    sol, mask, shared, k_all = fit((0, 1, 2))
    kept = ""
    if nh and k_all > 0 and sol[0][0] < 0.3 * k_all and np.nanpercentile(Hb.ravel()[base], 99) >= EMISSION_FIELD_R:
        # in an emission field the rejection took the H-alpha nebula for objects, down to less than 30 % of the
        # amplitude found on all the blocks: objects (galaxies, reflection nebulae) are bright in green and blue, an
        # H-alpha excess is red (Canon 40D on the Rosette: 0.019 -> 0.0021 ADU/R with every channel, 0.015 with
        # green and blue; the red excess of the image against the survey says 0.012-0.016). The fields where the
        # survey is faint are left alone: there it correlates with the galaxy or the reflection nebula, and the
        # rejection is right to lower it (Iris, M31, M33: 5 %, 0 %, 0 % of the first amplitude kept).
        sol2, mask2, shared2, _ = fit((1, 2))
        if sol2[0][0] > sol[0][0]:
            kept = (f"; the rejection on all channels left {100 * sol[0][0] / k_all:.0f} % of the H-alpha amplitude "
                    f"of the whole field: objects rejected on green and blue only")
            sol, mask, shared = sol2, mask2, shared2
    how = "none (no survey)" if not nh else ("camera HII signature imposed (the survey correlates with other "
                                             "structures in this field)" if shared else "fitted") + kept
    return sol, mask.reshape(Yb.shape[1:]), how


def star_flat(fs, img, W, H, dm, fwhm=None):
    """Smooth map (3, H, W) of the stars' measured / Gaia-predicted flux: the residual error of the flat field.

    Red and green from the star models' fluxes, or from wide apertures (radius 3 FWHM, 18 px at least) on isolated
    stars when the models' fluxes scatter at least twice as much around the map: where the star shapes change across
    the field (coma of a Newtonian without corrector) the models lose a part of the light of the stars that does not
    fit them, which then passes for a loss of sensitivity (Canon 6D on a Newton 200/1000: the models gave -52 %, the
    apertures -11 % with a scatter of 0.015 against 0.068, and the sky over-corrected became a bright band)."""
    x, y, g, fl, pr = fs["x"], fs["y"], fs["g"] - dm, fs["flux"], fs["pred"]
    m = (g > 9.5) & (g < 13.0) & (x > 50) & (x < W - 50) & (y > 50) & (y < H - 50)
    yy, xx = np.mgrid[0:H:16, 0:W:16].astype(np.float64)
    maps, info = [], {}
    for c in (0, 1):
        r = np.log(np.clip(fl[c, m], 1e-3, None) / pr[c, m])
        q, s, ok = robust_lstsq(poly_terms(x[m], y[m], W, H, 3), r)
        maps.append(poly_terms(xx, yy, W, H, 3) @ q)
        info[c] = {"n": int(ok.sum()), "scatter": float(s)}
    from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry
    t = cKDTree(np.c_[x, y])
    info["photometry"], info["scatter_psf"] = "star models", info[1]["scatter"]
    R = max(18.0, 3.0 * fwhm) if fwhm else 18.0
    cand = np.where((g > 9.5) & (g < 12.5) & (x > R + 20) & (x < W - R - 20) & (y > R + 20) & (y < H - R - 20))[0]
    iso = np.array([i for i in cand if all((j == i) or (g[j] > g[i] + 3.0)
                                           for j in t.query_ball_point([x[i], y[i]], 1.3 * R))], int)
    if len(iso) >= 40:
        xy = np.c_[x[iso], y[iso]]
        ap, an = CircularAperture(xy, r=R), CircularAnnulus(xy, r_in=1.3 * R, r_out=1.8 * R)
        amaps, sc = [], []
        for c in (0, 1):
            im = np.asarray(img[c], np.float32)
            fa = aperture_photometry(im, ap)["aperture_sum"].data - ApertureStats(im, an).median * ap.area
            q, s, ok = robust_lstsq(poly_terms(xy[:, 0], xy[:, 1], W, H, 3), np.log(np.clip(fa, 1e-3, None) / pr[c, iso]))
            amaps.append(poly_terms(xx, yy, W, H, 3) @ q)
            sc.append(float(s))
        info["scatter_aperture"], info["aperture_px"] = sc[1], R
        if info[1]["scatter"] > 2.0 * sc[1]:
            maps, info["photometry"] = amaps, "aperture"
            for c in (0, 1):
                info[c] = {"n": int(len(iso)), "scatter": sc[c]}
    # blue: wide aperture on bright isolated stars as a check; the map uses the mean R/G shape (the very spread
    # blue PSF of the corners biases blue PSF photometry)
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
