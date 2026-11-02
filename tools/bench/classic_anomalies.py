"""The two anomalies of the classic stack (docs/pilot/CLASSIC_ANOMALIES.md), measured on the bench's images.

    python tools/bench/classic_anomalies.py depth CASE...     -> F:/astro_essais/bench/<case>/anomalies_depth.json
    python tools/bench/classic_anomalies.py colour CASE...    -> F:/astro_essais/bench/<case>/anomalies_colour.json

``depth``: on the stacks (before the colour stage, as the bench's depth), for Siril classic, our drizzle, our classic
stack with and without the white balance before the debayering, on the sky they share:
  - the bench's depth reproduced (measure.photometry: zero point in an aperture of 2.5 FWHM, noise of empty apertures
    of radius 1 FWHM made of whole pixels, depth = zp - 2.5 log10(5 sigma));
  - the same with each other image's aperture radius (how much of a difference is the radius alone);
  - a continuous depth: exact circular apertures (photutils, method "exact") of one radius in arcseconds for all
    images, with the aperture correction measured on the bright stars (the share of a star's flux the small aperture
    holds), the empty-aperture depth of the deep surveys (e.g. Skelton et al. 2014, ApJS 214, 24, section 3.4);
  - per colour: the background noise at 1, 2, 4 and 8 px boxes, the flux scale (zero point) and the share of a star's
    flux within 1 FWHM (its profile), and the Gaussian FWHM of the stars.
``colour``: the colour stage stage by stage (the sensitivity curves, why they are not physical, the generic curves'
agreement, the matrix and how it amplifies the noise) and the star colours of the stacks before the colour stage.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossed as X  # noqa: E402  (sets the bench environment first)
import cases as C  # noqa: E402
import measure as M  # noqa: E402

LEVEL = "express"
SIRIL = X.SIRIL
NAMES = [SIRIL, "PI-own", "CL", "CL-off"]
CASES = ["heart", "iris", "m51", "m8", "seestar", "rosette"]


def dirs_of(p, n: str) -> dict[str, Path]:
    w = X.wf(p, n)
    return {s: Path(o.dir) for s in X.api.chain(p, w.id) if (o := X.api.stage_outputs(p, w.id, s)) is not None}


def exact_sums(img: np.ndarray, x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    from photutils.aperture import CircularAperture, aperture_photometry
    ap = CircularAperture(np.c_[x, y], r=r)
    return np.asarray(aperture_photometry(img, ap, method="exact")["aperture_sum"].data, float)


def depth(case: str) -> dict:
    t0 = time.time()
    today = json.loads((C.BENCH / case / "today_express.json").read_text(encoding="utf-8"))
    shapes_all = {n: e["stars"] for n, e in today["by_image"].items()}
    fwhm_as = max(s.get("fwhm_arcsec", 5) for s in shapes_all.values())
    gmin = max(s.get("max_g_unsaturated_bright", 10) for s in shapes_all.values())
    sv = C.norm_variant(case, None)
    p, _ = X.project(case, LEVEL)
    dirs = {n: dirs_of(p, n) for n in NAMES if n != SIRIL}

    def load(n):
        return M.load_siril(case, LEVEL, "stack_linear", sv) if n == SIRIL else M.load_ours(dirs[n], "stack")
    imgs = {n: load(n) for n in NAMES}
    ref = imgs[SIRIL]
    common = ref.cover.copy()
    for n in NAMES[1:]:
        common &= M.common_cover(ref, imgs[n])
    ref.cover = common
    for n in NAMES[1:]:
        imgs[n].cover &= M.common_cover(imgs[n], ref)
    st = M.gaia(dirs["PI-own"], imgs["PI-own"])
    ins = np.logical_and.reduce([im.inside(*im.pix(st["ra"], st["dec"]), 40) for im in imgs.values()])
    bright = ins & (st["g"] < 14) & (st["g"] > 9)
    iso_f = M.isolated(st, ins & (st["g"] > gmin), 3 * fwhm_as / 3600, 1.5)
    iso_z = M.isolated(st, ins & (st["g"] > gmin + 0.3) & (st["g"] < gmin + 3.5), 30 / 3600, 2.5)
    fw = {n: shapes_all[n]["fwhm_px"] for n in NAMES}
    # one aperture for every image: the radius of Siril classic's FWHM in arcseconds (the continuous depth), and the
    # big aperture of the zero point at 2.5 times the widest FWHM
    r_as = fw[SIRIL] * ref.scale
    big_as = 2.5 * max(fw[n] * imgs[n].scale for n in NAMES)
    rng = np.random.default_rng(1)
    out = {"case": case, "r_fixed_arcsec": r_as, "big_arcsec": big_as, "images": {}}
    for n in NAMES:
        im = imgs[n]
        M.refine_shift(im, st, bright)
        mask = M.star_mask(im, st, fwhm_as)
        zps = M.normalise_channels(im, st, iso_z, fw[n])
        ph = M.photometry(im, st, iso_f, fw[n], mask, gmin, rng=np.random.default_rng(1))
        R = ph["R"]
        e = {"fwhm_px": fw[n], "scale": im.scale, "r_det_px": ph["r_det_px"],
             "aperture_pixels": int(len(M.disk_offsets(ph["r_det_px"])[0])), "zp": ph["zp"],
             "sigma_ap": ph["sigma_ap"], "depth_bench": ph["depth5_mag"], "channel_zp": zps}
        # the empty apertures of the bench (same draw)
        free = im.cover & ~mask
        r_max = max(max(fw.values()) * 1.6, r_as / im.scale) + 2
        free_e = M._erode(free, int(r_max))
        yy, xx = np.nonzero(free_e[::4, ::4])
        pick = rng.choice(len(yy), min(len(yy), 6000), replace=False)
        ex, ey = xx[pick] * 4.0, yy[pick] * 4.0
        # the bench's depth with each image's radius
        e["depth_with_radius_of"] = {}
        for m_ in NAMES:
            r = max(1.5, fw[m_])
            s = M.rstd(M.gather(R, ex, ey, M.disk_offsets(r)).sum(1))
            e["depth_with_radius_of"][m_] = {"r": r, "pixels": int(len(M.disk_offsets(r)[0])),
                                             "depth": ph["zp"] - 2.5 * np.log10(5 * s)}
        # continuous: exact apertures, one radius in arcsec, aperture-corrected
        x, y = im.pix(st["ra"], st["dec"])
        zsel = iso_z & im.inside(x, y, 40)
        xs, ys = x[zsel], y[zsel]
        r_px, big_px = r_as / im.scale, big_as / im.scale
        ann_in, ann_out = 1.4 * big_px, 2.0 * big_px
        from photutils.aperture import CircularAnnulus, ApertureStats
        loc = ApertureStats(R, CircularAnnulus(np.c_[xs, ys], ann_in, ann_out)).median
        f_big = exact_sums(R, xs, ys, big_px) - loc * np.pi * big_px ** 2
        f_small = exact_sums(R, xs, ys, r_px) - loc * np.pi * r_px ** 2
        ok = (f_big > 0) & (f_small > 0)
        zp_c = float(np.median(st["g"][zsel][ok] + 2.5 * np.log10(f_big[ok])))
        frac = float(np.median(f_small[ok] / f_big[ok]))
        s_c = M.rstd(exact_sums(R, ex, ey, r_px))
        e.update({"zp_exact": zp_c, "aperture_fraction": frac, "sigma_exact": s_c,
                  "depth_continuous": zp_c - 2.5 * np.log10(5 * s_c / frac)})
        # own-FWHM exact aperture, corrected (the bench's idea without the whole pixels)
        r_own = max(1.5, fw[n])
        f_own = exact_sums(R, xs, ys, r_own) - loc * np.pi * r_own ** 2
        fr_own = float(np.median(f_own[ok] / f_big[ok]))
        s_own = M.rstd(exact_sums(R, ex, ey, r_own))
        e["depth_own_radius_exact"] = zp_c - 2.5 * np.log10(5 * s_own / fr_own)
        e["depth_own_radius_exact_uncorrected"] = zp_c - 2.5 * np.log10(5 * s_own)
        # per colour: noise by scale, star profile (share of the flux within 1 FWHM of Siril), Gaussian FWHM
        per = []
        for c in range(3):
            ch = im.rgb[c]
            B = M.background(ch, free, max(32, int(8 * fw[n])))
            Rc = ch - B
            nz = {}
            for b in (1, 2, 4, 8):
                bex = b * ref.scale / im.scale
                bb = max(1, int(round(bex)))
                s = M.box_noise(im, Rc, mask, bb, bex)["sigma_box"]
                nz[f"{b}px"] = float(25.0 - 2.5 * np.log10(s / im.pixarea))
            lc = ApertureStats(Rc, CircularAnnulus(np.c_[xs, ys], ann_in, ann_out)).median
            fb = exact_sums(Rc, xs, ys, big_px) - lc * np.pi * big_px ** 2
            fs = exact_sums(Rc, xs, ys, r_px) - lc * np.pi * r_px ** 2
            okc = (fb > 0) & (fs > 0)
            fws = []
            order = np.argsort(st["g"][zsel])
            for k in order[:400]:
                g_ = M.gauss_fit(Rc, xs[k], ys[k], max(4, int(round(2.2 * fw[n]))))
                if g_ is not None:
                    fws.append(np.sqrt(g_["fwhm_major"] * g_["fwhm_minor"]) * im.scale)
            zpc = float(np.median(st["g"][zsel][okc] + 2.5 * np.log10(fb[okc])))
            fcc = float(np.median(fs[okc] / fb[okc]))
            sc_ = M.rstd(exact_sums(Rc, ex, ey, r_px))
            per.append({"noise_sb": nz, "zp": zps[c], "flux_within_r": fcc,
                        "depth_continuous": zpc - 2.5 * np.log10(5 * sc_ / fcc),
                        "fwhm_gauss_arcsec": float(np.median(fws)) if fws else None, "fwhm_stars": len(fws)})
        e["per_colour"] = per
        out["images"][n] = e
        print(f"{case} {n}: bench depth {e['depth_bench']:.3f} (r {e['r_det_px']:.3f} px, {e['aperture_pixels']} px) "
              f"continuous {e['depth_continuous']:.3f} own-r exact {e['depth_own_radius_exact']:.3f} "
              f"frac {frac:.3f} ({time.time() - t0:.0f} s)", flush=True)
        for c, pc in zip("RGB", per):
            print(f"   {c}: depth {pc['depth_continuous']:.3f} noise {pc['noise_sb']} flux<r {pc['flux_within_r']:.3f} fwhm {pc['fwhm_gauss_arcsec']}",
                  flush=True)
        imgs[n] = None
        del im, R, ph
    (C.BENCH / case / "anomalies_depth.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def colour_errors(F: np.ndarray, ref: np.ndarray) -> dict:
    """For star fluxes F (N, 3) in output colours and the colours their Gaia XP spectra have in the same space
    (``ref`` (N, 3), sRGB linear, the white reference neutral): for B-R and G-R, the line through the measured index
    -2.5 log10(B / R) against the expected one; its robust scatter over its slope (as measure.colours, in magnitudes
    of the expected index) and its slope (1 = the catalogue's spread of colours, lower = colours pulled towards
    grey), and the offset at an expected 0 (the white point)."""
    out = {}
    ok = (F > 0).all(1) & (ref > 0).all(1)
    for name, j in (("b_r", 2), ("g_r", 1)):
        c = -2.5 * np.log10(F[ok, j] / F[ok, 0])
        col = -2.5 * np.log10(ref[ok, j] / ref[ok, 0])
        a, b0 = np.polyfit(col, c, 1)
        for _ in range(3):
            res = c - (a * col + b0)
            keep = np.abs(res) < 4 * M.rstd(res)
            a, b0 = np.polyfit(col[keep], c[keep], 1)
        res = c - (a * col + b0)
        out[name] = {"error": M.rstd(res) / abs(a), "slope": float(a), "offset": float(b0)}
    out["stars"] = int(ok.sum())
    return out


def colour(case: str) -> dict:
    """The colour stage, stage by stage, for our drizzle and both classic stacks."""
    from astro_gpu.stages import color as K
    p, _ = X.project(case, LEVEL)
    out = {"case": case, "images": {}}
    flux_by = {}
    for n in ["PI-own", "CL", "CL-off"]:
        d = dirs_of(p, n)
        man = json.loads((d["color"] / "manifest.json").read_text(encoding="utf-8"))
        met, par = man["metrics"], man["params"]
        stack = np.load(d["stack"] / "drz_full.npy").astype(np.float32)
        xp = np.load(d["catalogs"] / "gaia_xp.npz")
        sat = np.load(d["stack"] / "saturation.npy")[0] if (d["stack"] / "saturation.npy").exists() else None
        table, why = K.colour_star_table(stack, xp["x"], xp["y"], par["aperture"], sat)
        idx = np.where(why == "")[0]
        cam = table[idx]
        lam = xp["lam"]
        S, rel = K.fit_response(xp["flux"][idx], lam, cam)
        good, reason = K.physical_response(lam, S)
        vis = (lam >= 380) & (lam <= 720)
        cen = [float((lam[vis] * S[c, vis]).sum() / max(S[c, vis].sum(), 1e-30)) for c in range(3)]
        gen = K.generic_response(lam)
        Mc = np.load(d["color"] / "M.npy")
        prep = json.loads((d["render_simple"] / "prep_simple.json").read_text(encoding="utf-8"))
        Mr = np.array(prep["M"])
        import colour as CS
        W_ref = K.white_xyz(par["white_temp"])
        W_d65 = CS.xy_to_XYZ(CS.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
        to_out = K.XYZ_TO_SRGB @ K.bradford(W_ref, W_d65)
        bp_rp = np.asarray(xp["xyz"], float)[idx] @ to_out.T          # the expected output colours
        # the noise of the sky of the stack, camera colours: covariance of the small-scale part (3 x 3 px means
        # removed), on a central sky region
        H, W = stack.shape[1:]
        cut = stack[:, H // 3:2 * H // 3, W // 3:2 * W // 3].astype(np.float64)
        from scipy import ndimage as ndi
        hp = np.stack([c - ndi.median_filter(c, 7) for c in cut]).reshape(3, -1)
        lo, hi = np.percentile(hp.sum(0), [5, 95])
        keep = (hp.sum(0) > lo) & (hp.sum(0) < hi)
        Sig = np.cov(hp[:, keep])

        def gain(Mx):
            """per output colour: its noise against a diagonal calibration giving white the same output (Siril's
            PCC: one factor per channel), for the same white star signal"""
            wc = np.linalg.solve(Mx, np.ones(3))                   # the camera colour the matrix makes white
            D = np.diag(1.0 / wc)
            return (np.sqrt(np.diag(Mx @ Sig @ Mx.T)) / np.sqrt(np.diag(D @ Sig @ D.T))).tolist()
        e = {"response_source": met.get("response_source"), "fit_agreement_pct": met.get("fit_agreement_pct"),
             "generic_agreement_pct": met.get("generic_agreement_pct"),
             "profile_agreement_pct": met.get("profile_agreement_pct"),
             "free_fit_physical": bool(good), "free_fit_reason": reason, "free_fit_centroids_nm": cen,
             "free_fit_agreement_pct": K.colour_agreement(xp["flux"][idx], lam, cam, S),
             "generic_agreement_recomputed_pct": K.colour_agreement(xp["flux"][idx], lam, cam, gen),
             "matrix_colour": Mc.tolist(), "matrix_output": Mr.tolist(),
             "noise_gain_colour_matrix": gain(Mc), "noise_gain_output_matrix": gain(Mr),
             "emission": prep.get("emission"), "emission_weight": prep.get("emission_anchor_weight"),
             "stars": int(len(idx)), "sky_cov": Sig.tolist()}
        e["colours_camera"] = colour_errors(cam, bp_rp)
        e["colours_colour_matrix"] = colour_errors(cam @ Mc.T, bp_rp)
        e["colours_output_matrix"] = colour_errors(cam @ Mr.T, bp_rp)
        wc = np.linalg.solve(Mc, np.ones(3))
        e["colours_diagonal"] = colour_errors(cam / wc[None], bp_rp)
        out["images"][n] = e
        flux_by[n] = dict(zip(xp["source"][idx].tolist(), cam.tolist())) if "source" in xp.files else {}
        print(f"{case} {n}: source {e['response_source']}, free fit physical {good} ({reason}) centroids "
              f"{np.round(cen, 0)}, agreement fit {e['free_fit_agreement_pct']:.1f} % generic "
              f"{e['generic_agreement_recomputed_pct']:.1f} %", flush=True)
        print(f"   noise gain colour matrix {np.round(e['noise_gain_colour_matrix'], 2)} output "
              f"{np.round(e['noise_gain_output_matrix'], 2)}; B-R/V-R errors camera "
              f"{e['colours_camera']['b_r']['error']:.3f}/{e['colours_camera']['g_r']['error']:.3f} (slopes "
              f"{e['colours_camera']['b_r']['slope']:.3f}/{e['colours_camera']['g_r']['slope']:.3f}), matrix "
              f"{e['colours_output_matrix']['b_r']['error']:.3f}/{e['colours_output_matrix']['g_r']['error']:.3f}, "
              f"diagonal {e['colours_diagonal']['b_r']['error']:.3f}/{e['colours_diagonal']['g_r']['error']:.3f}",
              flush=True)
        del stack
    # the same stars in both classic stacks: what the balance changes in each colour (flux ratio off / on)
    a, b = flux_by.get("CL-off", {}), flux_by.get("CL", {})
    common = [k for k in a if k in b]
    if common:
        fa, fb = np.array([a[k] for k in common]), np.array([b[k] for k in common])
        r = fa / fb
        rn = r / r[:, 1:2]
        out["off_over_on"] = {"stars": len(common), "median": np.median(r, 0).tolist(),
                              "median_vs_green": np.median(rn, 0).tolist()}
        print(f"{case} same stars, classic without / with the balance, flux per colour {np.round(np.median(r, 0), 3)},"
              f" relative to green {np.round(np.median(rn, 0), 3)} ({len(common)} stars)", flush=True)
    (C.BENCH / case / "anomalies_colour.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def demo(case: str, out_name: str | None = None, before: dict[str, str] | None = None) -> Path:
    """A small board as the bench's (two_methods.board: Siril classic's pixels, one flux scale from the same Gaia
    stars, each colour's own sky, one asinh stretch), on the bench's faint area and star field: Siril classic, our
    drizzle, our classic stack with the balance, without it, and without it calibrated by one factor per colour (the
    camera colours of our output, the colour stage's white: what Siril's photometric calibration does).
    ``before``: {configuration: an older render_simple folder} to show an earlier result (and its matrix)."""
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage as ndi
    import sheet as SH
    import two_methods as TM
    sv = C.norm_variant(case, None)
    p, _ = X.project(case, LEVEL)
    cols = [(SIRIL, "Siril classique"), ("PI-own", "Notre drizzle"), ("CL", "Notre classique (équilibré)"),
            ("CL-off", "Notre classique sans équilibrage"),
            ("CL-off/diag", "Le même, un facteur par couleur")]
    ref = M.load_siril(case, LEVEL, "color_linear", sv)
    imgs = {SIRIL: ref}
    before = before or {}

    def dirs_b(n):
        d_ = dirs_of(p, n)
        if n in before:
            d_["render_simple"] = Path(before[n])
        return d_
    for n in ("PI-own", "CL", "CL-off"):
        imgs[n] = M.load_ours(dirs_b(n), "lin")
    d = dirs_b("CL-off")
    Mr = np.array(json.loads((d["render_simple"] / "prep_simple.json").read_text(encoding="utf-8"))["M"])
    wc = np.linalg.solve(Mr, np.ones(3))
    T = (np.diag(1.0 / wc) @ np.linalg.inv(Mr)).astype(np.float32)
    im = imgs["CL-off"]
    imgs["CL-off/diag"] = M.Img("diag", np.tensordot(T, im.rgb, 1), im.wcs, im.cover)
    m = TM.load(case)
    st = M.gaia(dirs_of(p, "PI-own"), imgs["PI-own"])
    x, y = ref.pix(st["ra"], st["dec"])
    sel = ref.inside(x, y, 60) & (st["g"] > 11.5) & (st["g"] < 14.5)
    sel = M.isolated(st, sel, 30 / 3600, 2.5)
    fw = float(m["by_image"][SIRIL]["stars"].get("fwhm_px", 3.0))

    def star_flux(im):
        xx, yy = im.pix(st["ra"][sel], st["dec"][sel])
        k = im.scale / ref.scale
        bo, ro = M.disk_offsets(2.5 * fw / k), M.ring_offsets(3.5 * fw / k, 5.0 * fw / k)
        ok = im.inside(xx, yy, 40)
        o = np.full(len(xx), np.nan)
        loc = np.median(M.gather(im.L, xx[ok], yy[ok], ro), 1)
        o[ok] = M.gather(im.L, xx[ok], yy[ok], bo).sum(1) - loc * len(bo[0])
        return o
    f_ref = star_flux(ref)
    kf = {}
    for n, im in imgs.items():
        f = star_flux(im)
        ok = np.isfinite(f) & np.isfinite(f_ref) & (f > 0) & (f_ref > 0)
        kf[n] = float(np.median(f_ref[ok] / f[ok]))
    # the bench board's two places (recomputed as two_methods.board does)
    CROP, blk = 400, 128
    stars = ref.inside(x, y, 0) & (st["g"] < 15)
    hist, _, _ = np.histogram2d(y[stars], x[stars], bins=[np.arange(0, (ref.H // blk) * blk + 1, blk),
                                                         np.arange(0, (ref.W // blk) * blk + 1, blk)])
    full = SH._bin(ref.cover.astype(np.float32), blk)
    win = max(2, CROP // blk + 1)
    sm = np.where(ndi.minimum_filter(full, size=win) > 0.98, ndi.uniform_filter(hist, size=win), -1)
    iy, ix = np.unravel_index(np.argmax(sm), sm.shape)
    star_c = ((ix + 0.5) * blk, (iy + 0.5) * blk)
    Lm = ndi.median_filter(SH._bin(ref.L, 16), 7)
    mm = np.zeros_like(Lm, bool)
    mm[int(0.15 * Lm.shape[0]):int(0.85 * Lm.shape[0]), int(0.15 * Lm.shape[1]):int(0.85 * Lm.shape[1])] = True
    mm &= SH._bin(ref.cover.astype(np.float32), 16) > 0.99
    target = np.nanpercentile(np.where(mm, Lm, np.nan), 15)
    iy, ix = np.unravel_index(np.nanargmin(np.abs(np.where(mm, Lm, np.inf) - target)), Lm.shape)
    faint_c = ((ix + 0.5) * 16, (iy + 0.5) * 16)

    def crop(n, c):
        im = imgs[n]
        gx, gy = np.meshgrid(np.arange(CROP) - CROP / 2, np.arange(CROP) - CROP / 2)
        ra, dec = ref.wcs.all_pix2world(c[0] + gx.ravel(), c[1] + gy.ravel(), 0)
        px, py = im.wcs.all_world2pix(ra, dec, 0)
        px, py = px.reshape(gx.shape) + im.dx, py.reshape(gx.shape) + im.dy
        x0, x1 = max(0, int(np.floor(px.min())) - 6), min(im.W, int(np.ceil(px.max())) + 7)
        y0, y1 = max(0, int(np.floor(py.min())) - 6), min(im.H, int(np.ceil(py.max())) + 7)
        o = np.stack([ndi.map_coordinates(np.asarray(im.rgb[k, y0:y1, x0:x1]), [py - y0, px - x0], order=3,
                                          mode="nearest") for k in range(3)], -1)
        return o * (kf[n] / im.pixarea * ref.pixarea)
    places = [("Une zone faible", faint_c), ("Un champ d'étoiles", star_c)]
    gap, top, lab = 8, 34, 24
    Wt = len(cols) * (CROP + gap) + gap
    Ht = top + len(places) * (CROP + lab + gap) + 6
    img = Image.new("RGB", (Wt, Ht), (24, 24, 24))
    dr = ImageDraw.Draw(img)
    f2 = ImageFont.truetype(SH.FONT, 16)
    for i, (_n, label) in enumerate(cols):
        dr.text((gap + i * (CROP + gap) + 4, 8), label, fill=(235, 235, 235), font=f2)
    y0 = top
    stats = {}
    for title, c in places:
        row = {n: crop(n, c) for n, _ in cols}
        r = row[SIRIL]
        lum = r.mean(-1)
        sky = float(np.median(lum))
        sig = float(1.4826 * np.median(np.abs(lum - sky)))
        white = float(np.percentile(lum, 99.7) - sky)
        a = max(2.0 * sig, 1e-9)
        norm = np.arcsinh(max(white, 10 * a) / a)
        dr.text((gap, y0 + 2), title, fill=(200, 200, 200), font=f2)
        y0 += lab
        for i, (n, _l) in enumerate(cols):
            dd = row[n]
            skyn = np.median(dd.reshape(-1, 3), axis=0)
            v = np.arcsinh(np.clip(dd - skyn + 1.0 * sig, 0, None) / a) / norm
            stats[f"{title}|{n}"] = {"noise_rgb_vs_siril_L": [float(1.4826 * np.median(np.abs(dd[..., k] - skyn[k])) / sig)
                                                            for k in range(3)],
                                     "black_pixels_pct": float(100 * np.mean((dd - skyn + sig) <= 0))}
            img.paste(Image.fromarray((np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8)), (gap + i * (CROP + gap), y0))
        y0 += CROP + gap
    outd = C.REPO / "docs" / "pilot" / "shots" / "2026-11-02" / "anomalies"
    outd.mkdir(parents=True, exist_ok=True)
    out = outd / (out_name or f"{case}_couleur.png")
    img.save(out)
    for k, v in stats.items():
        print(k, {a: np.round(b, 2) for a, b in v.items()})
    print("board", out, "flux scales", kf, flush=True)
    return out


def rerun(case: str, device: str = "cpu") -> int:
    """Re-run the bench's workflows of today (after a change of the colour stage): only the stages after the stack;
    refuses if the plan holds a calibration, registration or stack job (those belong to GPU runs)."""
    from astro_gpu.cli2 import follow
    p, _ = X.project(case, LEVEL)
    ids = [X.wf(p, n).id for n in X.TODAY if X.wf(p, n) is not None]
    if X.api.running(p) is not None:
        print(case, "a run is live: not started", flush=True)
        return 1
    plan = X.api.plan_workflows(p, ids, device)
    stages = sorted({j.stage for j in plan.jobs})
    print(case, "plan:", len(plan.jobs), "jobs", stages, flush=True)
    if {"calibrate", "register", "stack"} & set(stages):
        print(case, "the plan holds heavy stages: not started", flush=True)
        return 1
    if not plan:
        return 0
    info = X.api.start_run(p, plan, device)
    st = follow(p, info.offset, info.run_id, info.pid, poll=3.0, echo=lambda s: None)
    X.api.refresh(p)
    print(case, "run", st, flush=True)
    return 0 if "fail" not in str(st).lower() else 1


def measure_after(case: str) -> None:
    """two_methods.py's measure (the same function, the same images), written next to its file, not over it:
    F:/astro_essais/bench/<case>/today_express_colour5.json."""
    import two_methods as TM
    r = X.measure_crossed(case, LEVEL, list(TM.OURS), {TM.SIRIL_DRZ: f"{LEVEL}_drizzle"})
    (C.BENCH / case / "today_express_colour5.json").write_text(json.dumps(r, indent=1, default=float),
                                                               encoding="utf-8")
    print(f"written {case} ({r['seconds']:.0f} s)", flush=True)


def compare(case: str) -> None:
    """Before (today_express.json, colour stage 4) and after (today_express_colour5.json), the bench's rows."""
    import two_methods as TM
    a = json.loads((C.BENCH / case / "today_express.json").read_text(encoding="utf-8"))
    f = C.BENCH / case / "today_express_colour5.json"
    if not f.exists():
        print(case, "not measured after")
        return
    b = json.loads(f.read_text(encoding="utf-8"))
    print(f"== {case}")
    for label, key in TM.ROWS:
        if key in ("frames", "reject", "time_pre"):
            continue
        cells = []
        for n in (SIRIL, "PI-own", "CL", "CL-off"):
            va, vb = TM.cell_value(a, case, n, key), TM.cell_value(b, case, n, key)
            fa = "–" if va is None else f"{va:.3f}"
            fb = "–" if vb is None else f"{vb:.3f}"
            cells.append(f"{n} {fa}" + ("" if fa == fb else f" -> {fb}"))
        print(f"  {key:6s} " + " | ".join(cells))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "measure-after":
        for c in sys.argv[2:]:
            measure_after(c)
        sys.exit(0)
    if cmd == "compare":
        for c in sys.argv[2:]:
            compare(c)
        sys.exit(0)
    if cmd == "rerun":
        sys.exit(max([rerun(c) for c in sys.argv[2:]] or [0]))
    if cmd == "demo":
        for c in sys.argv[2:]:
            demo(c)
        sys.exit(0)
    if cmd == "demo-before":           # demo-before CASE OUT CONFIG=render_simple_dir ...
        demo(sys.argv[2], sys.argv[3], dict(a.split("=", 1) for a in sys.argv[4:]))
        sys.exit(0)
    if cmd == "depth":
        for c in sys.argv[2:]:
            depth(c)
        sys.exit(0)
    if cmd == "colour":
        for c in sys.argv[2:]:
            colour(c)
        sys.exit(0)
    sys.exit(2)
