"""Four stacks of a case on the same sky, with the bench's own measures (measure.py, as crossed.py does): Siril's
classic stack, Siril's Bayer drizzle (run_siril_drizzle.py), our drizzle and our classic stack, both read from the
existing crossed project (v2x_<level>: chains "OLD" and "S7"; nothing is re-run).

    CUDA_VISIBLE_DEVICES=-1 python tools/bench/siril/measure_drizzle.py CASE [LEVEL]

Writes F:/astro_essais/bench/<case>/drizzle_siril_<level>.json (same layout as crossed_<level>.json: by_image).
This is crossed.measure_crossed with two Siril stacks instead of one; the measures are the same functions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crossed as X  # noqa: E402  (sets the bench environment before astro_gpu is imported)
import measure as M  # noqa: E402
import cases as C  # noqa: E402

SIRIL, SIRIL_DRZ = "Siril", "Siril drizzle"
OURS = ["OLD", "S7"]                                       # our drizzle, our classic stack


class Geo(M.Img):
    def __init__(self, name, wcs, cover, H, W):            # the geometry of an image, no pixels
        self.name, self.wcs, self.cover, self.H, self.W = name, wcs, cover, H, W
        self.scale = float(np.abs(np.linalg.det(wcs.pixel_scale_matrix)) ** 0.5 * 3600)
        self.pixarea = self.scale ** 2
        self.dx = self.dy = 0.0


def measure(case: str, level: str) -> dict:
    t0 = time.time()
    sv = C.norm_variant(case, None)
    p, _ = X.project(case, level)
    dirs: dict[str, dict[str, Path]] = {}
    for n in OURS:
        w = X.wf(p, n)
        d = {}
        for s in X.api.chain(p, w.id):
            o = X.api.stage_outputs(p, w.id, s)
            if o is not None:
                d[s] = Path(o.dir)
        dirs[n] = d
    first = "OLD"
    sdir = {SIRIL: level, SIRIL_DRZ: f"{level}_drizzle"}      # the Siril folders' level names

    ref = M.load_siril(case, level, "stack_linear", sv)
    geos: dict[str, Geo] = {}
    d_im = M.load_siril(case, sdir[SIRIL_DRZ], "stack_linear", sv)
    geos[SIRIL_DRZ] = Geo(SIRIL_DRZ, d_im.wcs, d_im.cover, d_im.H, d_im.W)
    del d_im
    for n in OURS:
        im = M.load_ours(dirs[n], "stack")
        geos[n] = Geo(n, im.wcs, im.cover, im.H, im.W)
        del im
    common_ref = ref.cover.copy()
    for g in geos.values():
        common_ref &= M.common_cover(ref, g)
    refc = Geo("ref", ref.wcs, common_ref, ref.H, ref.W)
    for g in geos.values():
        g.cover = g.cover & M.common_cover(g, refc)
    ref.cover = common_ref
    print(f"{case} {level}: 4 stacks, common sky ({time.time() - t0:.0f} s)", flush=True)
    st = M.gaia(dirs[first], geos[first])
    allg = {SIRIL: ref, **geos}
    inside = []
    for g in allg.values():
        x, y = g.pix(st["ra"], st["dec"])
        inside.append(g.inside(x, y, 40))
    common = np.logical_and.reduce(inside)
    bright = common & (st["g"] < 14) & (st["g"] > 9)
    iso_shape = M.isolated(st, common & (st["g"] < 16.5) & (st["g"] > 8), 40 / 3600, 3.0)

    def load(n: str, which: str = "stack"):
        if n == SIRIL_DRZ:
            im = M.load_siril(case, sdir[n], "stack_linear" if which == "stack" else "color_linear", sv)
        else:
            im = M.load_ours(dirs[n], which)
        if which == "stack":
            im.cover = geos[n].cover
            im.dx, im.dy = geos[n].dx, geos[n].dy
        return im
    shapes = {}
    M.refine_shift(ref, st, bright)
    shapes[SIRIL] = M.star_shapes(ref, st, iso_shape, ref)
    for n in [SIRIL_DRZ] + OURS:
        im = load(n)
        M.refine_shift(im, st, bright)
        geos[n].dx, geos[n].dy = im.dx, im.dy
        shapes[n] = M.star_shapes(im, st, iso_shape, ref)
        del im
    print(f"stars fitted ({time.time() - t0:.0f} s)", flush=True)
    fwhm_as = max(s.get("fwhm_arcsec", 5) for s in shapes.values())
    gmin = max(s.get("max_g_unsaturated_bright", 10) for s in shapes.values())
    iso_f = M.isolated(st, common & (st["g"] > gmin), 3 * fwhm_as / 3600, 1.5)
    iso_z = M.isolated(st, common & (st["g"] > gmin + 0.3) & (st["g"] < gmin + 3.5), 30 / 3600, 2.5)
    iso_c = M.isolated(st, common & np.isfinite(st["bp"]) & (st["g"] > gmin + 0.3) & (st["g"] < gmin + 5),
                       30 / 3600, 2.5)
    dr = M.drift_direction(dirs[first])
    sc = M.load_siril(case, level, "color_linear", sv)
    bs = 4
    ref_mask = M.star_mask(ref, st, fwhm_as)
    M.normalise_channels(ref, st, iso_z, shapes[SIRIL].get("fwhm_px", 3.0))
    ref_ph = M.photometry(ref, st, iso_f, shapes[SIRIL].get("fwhm_px", 3.0), ref_mask, gmin)
    out: dict = {}
    snrs, fsel = {}, {}
    for n in [SIRIL, SIRIL_DRZ] + OURS:
        fw = shapes[n].get("fwhm_px", 3.0)
        if n == SIRIL:
            im, mask, ph = ref, ref_mask, ref_ph
        else:
            im = load(n)
            mask = M.star_mask(im, st, fwhm_as)
            M.normalise_channels(im, st, iso_z, fw)
            ph = M.photometry(im, st, iso_f, fw, mask, gmin)
        e: dict = {"stars": shapes[n]}
        snr = np.full(len(st["g"]), np.nan)
        snr[ph["fsel"]] = ph["snr"]
        snrs[n], fsel[n] = snr, ph["fsel"]
        bex = bs * ref.scale / im.scale
        b = max(2, int(round(bex)))
        bn = M.box_noise(im, ph["R"], mask, b, bex)
        zp = ph["zp"]
        bn["sb_mag_arcsec2"] = float(zp - 2.5 * np.log10(bn["sigma_box"] / im.pixarea))
        bn["sb_large_mag_arcsec2"] = float(zp - 2.5 * np.log10(
            M.box_noise(im, ph["R"], mask, max(2, int(round(4 * bex))), 4 * bex)["sigma_box"] / im.pixarea))
        sky = float(np.median(ph["B"][im.cover & ~mask]))
        bn["sky_sb"] = float(zp - 2.5 * np.log10(sky / im.pixarea)) if sky > 0 else float("nan")
        bn.update({k: ph[k] for k in ("zp", "zp_stars", "r_det_px", "depth5_mag")})
        e["noise"] = bn
        t = M.target_snr(sc, ref, im, ref_mask, mask, bs, b, bex, ref_ph["zp"], zp)
        e["target"] = {"snr": t.get("snr_ours"), "signal_sb": t.get("signal_sb_ours"),
                       "noise_sb": t.get("noise_sb_ours"), "boxes": t.get("target_boxes_ours"),
                       "ref_snr": t.get("snr_siril")}
        ang = M.angle_in(im, geos[first], dr["v"])
        e["streaks"] = M.streaks(im, ph["R"], mask, ph["B"], ang, zp)
        e["coloured_spots"] = M.coloured_spots(im, mask, fw)
        rs = M.residuals(im, mask, fw)
        e["residuals"] = {"trails": len(rs["trails"]), "area_deg2": rs["area_deg2"],
                          "trail_length_arcmin": float(sum(x["length_arcmin"] for x in rs["trails"])),
                          "hot_spots_per_deg2": len(rs["spots"]["hot"]) / max(rs["area_deg2"], 1e-6),
                          "cold_spots_per_deg2": len(rs["spots"]["cold"]) / max(rs["area_deg2"], 1e-6)}
        if n == SIRIL:
            im2 = M.load_siril(case, level, "color_linear", sv)
        else:
            im2 = load(n, "lin")
        im2.dx, im2.dy, im2.cover = im.dx, im.dy, im.cover & im2.cover
        e["colours"] = M.colours(im2, st, iso_c, fw)
        M.normalise_channels(im2, st, iso_z, fw)
        ph2 = M.photometry(im2, st, iso_f, fw, mask, gmin)
        e["flatness"] = M.flatness(im2, mask, ph2["zp"], bn["sky_sb"])
        del im2, ph2
        if n in (SIRIL, SIRIL_DRZ):
            e["engine"] = {"rejected_pct": X._siril_rejection(case, sdir[n])}
        else:
            e["engine"] = X.engine_info(p, n)
        out[n] = e
        print(f"{n}: depth {bn['depth5_mag']:.2f} noise {bn['sb_mag_arcsec2']:.2f}/{bn['sb_large_mag_arcsec2']:.2f} "
              f"snr {e['target']['snr']} fwhm {shapes[n].get('fwhm_centre_arcsec')} "
              f"spots {e['coloured_spots']['all_per_deg2']:.0f} "
              f"B-R {(e['colours'].get('b_r') or {}).get('scatter_bp_rp')} ({time.time() - t0:.0f} s)", flush=True)
        if n != SIRIL:
            del im, mask, ph
    common_f = np.logical_and.reduce([fsel[n] for n in out])
    for n in out:
        s = snrs[n][common_f]
        out[n]["faint_stars"] = {"found": int((s >= 5).sum()), "of": int(len(s)),
                                 "g50": M._g50(s >= 5, st["g"][common_f])}
    return {"case": case, "level": level, "images": list(out), "first": first, "box_arcsec": bs * ref.scale,
            "gaia_stars": int(len(st["g"])), "by_image": out, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    case = sys.argv[1]
    level = sys.argv[2] if len(sys.argv) > 2 else "express"
    r = measure(case, level)
    f = C.BENCH / case / f"drizzle_siril_{level}.json"
    f.write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
    print(f"written {f} ({r['seconds']:.0f} s)")
