"""Physical background of the starless image: the star flat, the unresolved starlight and the zero point.

The sky gradient is the Gradient block's (stages/gradient.py), removed right after stacking: this stage finds only
what is left of the sky as a level, per channel: a constant plus the light of the stars too faint to be resolved
(shaped like the faint Gaia stars), fitted on the blocks of sky (the objects left out, iteratively).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree

from ..core.imgutil import bin_median, poly_terms, robust_lstsq
from ..core.params import ADVANCED, P, StageParams
from ..core.stage import Context, Stage

HALO_PRECISION = 0.2      # precision of the removal of the far halo of the stars, relative to it (around Alnitak the
                          # data depart from the model by -5 % at 90 px to +25 % at 300 px)


class BackgroundParams(StageParams):
    bin_px: int = P(32, label="Sky sampling", level=ADVANCED, ge=8, le=256, unit="px",
                    help="Size of the blocks in which the background is sampled for the fit.")
    star_flat: bool = P(True, label="Star flat correction",
                        help="Measures the residual error of your flat field directly on the stars: their measured "
                             "flux divided by the flux predicted from Gaia gives a smooth map of the real "
                             "sensitivity of the instrument across the field, which corrects the image.")
    object_clip: float = P(2.5, label="Object rejection", level=ADVANCED, ge=1.0, le=10, step=0.1, unit="sigma",
                           help="Blocks brighter than the sky model by more than this are taken as objects (nebula, "
                                "galaxy, reflection nebula, dust glow) and left out of the sky fit, iteratively. The "
                                "sky then passes under the data instead of through extended objects, which would leave "
                                "the darker areas below zero.")


class Background(Stage):
    name = "background"
    title = "Physical sky background"
    description = ("Separates the real sky from the instrument on the starless image (the sky gradient was removed "
                   "after stacking). 1) Star flat: Gaia predicts the flux of thousands of stars, so the ratio "
                   "measured/predicted maps the residual vignetting and flat errors, which are corrected. 2) Sky "
                   "level: a constant plus the unresolved starlight (shaped like the faint Gaia stars). 3) The zero "
                   "point of the continuum is set by the light of the stars too faint to be resolved, extrapolated "
                   "from Gaia counts - not by guessing a black point. Faint emission (IFN-like structures) is "
                   "therefore kept with its true brightness.")
    deps = ("starless", "stars", "optics", "astrometry", "stack", "color", "catalogs")
    Params = BackgroundParams
    version = "6"          # 3: zero-point floor on the darkest areas; 4: not set where the removed star glare dominates;
                           # 5: star flat from wide apertures when the star models' fluxes scatter twice as much;
                           # 6: the sky gradient is the Gradient block's: a level here, no H-alpha survey

    def run(self, ctx: Context, p: BackgroundParams) -> None:
        dm = float(ctx.metric_of("optics", "mag_offset", 0.0))
        fs = np.load(ctx.input("stars", "star_fluxes.npz"))
        sl = ctx.load("starless", "sl_final.npy").astype(np.float32)
        H, W = sl.shape[1:]
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
        good = cover > 0.5 * np.nanmedian(cover)
        Yb = np.stack([bin_median(sl[c], p.bin_px) for c in range(3)])
        sol, used = fit_sky(Yb, U, good, p.object_clip)
        ctx.metric("sky_fit_blocks_pct", round(100 * float(used.sum()) / max(int(good.sum()), 1), 1))
        ctx.log(f"sky level fitted on {100 * used.sum() / max(good.sum(), 1):.0f} % of the field (the rest is taken "
                f"as objects)")
        coefs = {c: {"u": float(sol[c][0]), "level": float(sol[c][1])} for c in range(3)}
        sky = np.repeat(np.array([sol[c][1] for c in range(3)], np.float32)[:, None, None], H, 1).repeat(W, 2)
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
        low = good
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
        ctx.save("astro.npy", astro)
        ctx.save_json("coefs.json", {"fit": coefs, "flat": finfo})
        ctx.save("sky.npy", sky)
        ctx.metric("emission_dips", emission_dips(astro, fs, W, H, dm))
        from ..core.imgutil import asinh_preview
        ctx.preview("background", asinh_preview(astro, ctx.load("color", "M.npy"), k=30))


def fit_sky(Yb, U, good, clip, iters=10):
    """Sky level per channel on the blocks ``Yb`` (3, hb, wb): Yb = u U + level, u >= 0 (U: the unresolved starlight
    template). Blocks brighter than the model by more than ``clip`` robust sigmas in any channel are objects and are
    left out (dilated by a block), iteratively, so the level passes under the extended objects.
    Returns ([u, level] per channel, the blocks used)."""
    base = good.ravel() & np.isfinite(Yb).all(0).ravel() & np.isfinite(U).all(0).ravel()

    def solve(mask):
        out = []
        for c in range(3):
            A = np.c_[U[c].ravel(), np.ones(U[c].size)]
            out.append(lsq_linear(A[mask], Yb[c].ravel()[mask], bounds=([0, -np.inf], np.inf)).x)
        return out

    mask = base
    sol = solve(mask)
    for _ in range(iters):
        keep = base.copy()
        for c in range(3):
            r = Yb[c].ravel() - (sol[c][0] * U[c].ravel() + sol[c][1])
            sd = 1.4826 * np.median(np.abs(r[mask] - np.median(r[mask]))) + 1e-9
            obj = (r - np.median(r[mask]) > clip * sd).reshape(Yb.shape[1:])
            obj = cv2.dilate(obj.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            keep &= ~obj.ravel()
        new = keep if keep.sum() >= 0.1 * base.sum() else mask
        if (new == mask).all():
            break
        mask = new
        sol = solve(mask)
    return sol, mask.reshape(Yb.shape[1:])


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
