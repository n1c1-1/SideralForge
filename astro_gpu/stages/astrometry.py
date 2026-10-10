"""Astrometry: plate solving of the stack directly on Gaia DR3 (TAN-SIP world coordinate system)."""
from __future__ import annotations

import numpy as np

from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage, StageError

GOOD_PAIRS = 15     # matched star pairs that make a triangle match certain (false matches raise instead)


class AstrometryParams(StageParams):
    target: str = P("", label="Target name",
                    help="Name of the main object (e.g. 'NGC 7822', 'IC 1805', 'Heart Nebula'), resolved with SIMBAD "
                         "to get the approximate centre of the field.",
                    tip="Not needed when RA/Dec are given or when the FITS files contain the pointing.")
    ra: float | None = P(None, label="Centre RA", unit="deg", ge=0, le=360,
                         help="Approximate right ascension of the field centre (degrees). Within ~20 % of the field "
                              "width is enough.")
    dec: float | None = P(None, label="Centre Dec", unit="deg", ge=-90, le=90,
                          help="Approximate declination of the field centre (degrees).")
    focal_mm: float | None = P(None, label="Focal length", unit="mm", gt=0,
                               help="Effective focal length of the instrument (with reducer / flattener). With the pixel "
                                    "size of the camera it gives the image scale, which makes the astrometry immediate.",
                               tip="Read from the files when the camera records it; telescopes usually do not report "
                                   "it. Leave empty to let the solver search the scale (a few minutes).")
    pixel_um: float | None = P(None, label="Pixel size", unit="um", gt=0, level=ADVANCED,
                               help="Physical size of a sensor pixel, from the camera database or the FITS header.")
    scale: float | None = P(None, label="Image scale", unit="arcsec/px", gt=0, level=ADVANCED,
                            help="Approximate size of a pixel of the stacked image on the sky. Computed from the focal "
                                 "length and the pixel size when they are known; otherwise searched automatically.",
                            tip="206.265 x pixel size (um) / focal length (mm), divided by the drizzle scale.")
    blind: bool = P(True, label="Blind solving", level=ADVANCED,
                    help="When the field centre is unknown (or the guided search fails), identify the field on the "
                         "whole sky from its brightest stars (geometric hashing of star quads, like astrometry.net).",
                    tip="Works for fields wider than about 1 deg; the first use downloads two bright-star catalogues.")
    search_offsets: float = P(0.2, label="Centre search range", level=ADVANCED, ge=0.0, le=1.0, step=0.05,
                              unit="x field", help="How far from the given centre the solver looks for the field.")
    sip_order: int = P(3, label="Distortion order", level=ADVANCED, ge=0, le=5,
                       help="Order of the SIP polynomial describing the optical distortion.",
                       tip="3 for refractors with reducers / wide fields; 2 for small fields; 0 = pure tangent plane.")


class Astrometry(Stage):
    name = "astrometry"
    title = "Astrometry"
    description = ("Finds exactly where the image points on the sky by matching its stars to Gaia DR3 (triangle "
                   "matching, then a tangent-plane + distortion fit on thousands of stars). Every later step that "
                   "uses the sky (Gaia photometry, colour calibration, sky surveys) relies on this solution; the "
                   "typical residual is a few hundredths of a pixel.")
    deps = ("inventory", "stack")
    Params = AstrometryParams
    version = "1"
    network = True

    def auto(self, ctx: Context) -> dict[str, Auto]:
        a: dict[str, Auto] = {}
        tgt = ctx.project.config.target
        if tgt.ra is not None and tgt.dec is not None:
            a["ra"], a["dec"] = Auto(tgt.ra, "project target"), Auto(tgt.dec, "project target")
        elif ctx.metric_of("inventory", "pointing_ra") is not None:
            a["ra"] = Auto(ctx.metric_of("inventory", "pointing_ra"), "telescope pointing in the FITS headers")
            a["dec"] = Auto(ctx.metric_of("inventory", "pointing_dec"), "telescope pointing in the FITS headers")
        name = tgt.name or ctx.metric_of("inventory", "object") or ctx.project.config.name
        if name:
            a["target"] = Auto(name, "project / FITS object name")
        pix, foc = ctx.metric_of("inventory", "pixel_um"), ctx.metric_of("inventory", "focal_mm")
        if pix:
            a["pixel_um"] = Auto(pix, f"camera {ctx.metric_of('inventory', 'camera')}")
        if foc:
            a["focal_mm"] = Auto(foc, "recorded in the files")
        if ctx.project.config.target.focal_mm:
            a["focal_mm"] = Auto(ctx.project.config.target.focal_mm, "entered when the project was created")
        return a

    def run(self, ctx: Context, p: AstrometryParams) -> None:
        import astroalign as aa
        import sep
        from astropy.io import fits
        from astropy.wcs.utils import fit_wcs_from_points
        from scipy.spatial import cKDTree

        ra0, dec0 = p.ra, p.dec
        S_drz = float(ctx.load_json("stack", "drz_meta.json")["scale"])
        if p.scale is None and p.focal_mm and p.pixel_um:
            p = p.model_copy(update={"scale": 206.265 * p.pixel_um / p.focal_mm / S_drz})
            ctx.log(f"image scale {p.scale:.3f}\"/px from focal {p.focal_mm} mm and pixel {p.pixel_um} um")
        if (ra0 is None or dec0 is None) and p.target:
            try:
                ra0, dec0 = resolve_name(p.target)
                ctx.log(f"'{p.target}' resolved by SIMBAD: RA {ra0:.4f} Dec {dec0:.4f}")
            except StageError as e:
                ctx.log(str(e), "warning")
        d = ctx.load("stack", "drz_full.npy", mmap=True)
        g = np.array(d[1], dtype=np.float32, copy=True)
        H, W = g.shape
        sep.set_extract_pixstack(3000000)
        bk = sep.Background(g, bw=64, bh=64)
        o = sep.extract(g - bk.back(), 15, err=bk.globalrms, minarea=6)
        o_all = np.sort(o, order="flux")[::-1]
        brightdet = np.c_[o_all["x"], o_all["y"]][:150]
        o = np.sort(o[o["flag"] == 0], order="flux")[::-1]
        det = np.c_[o["x"], o["y"]]
        ctx.log(f"{len(det)} stars detected in the stack")
        if len(det) < 50:
            raise StageError("too few stars detected in the stack to solve the field")
        if ra0 is None or dec0 is None:
            if not p.blind:
                raise StageError("the approximate field centre is unknown: enter the target name or its RA/Dec")
            ra0, dec0, sc_b = _blind(ctx, brightdet, W, H)
            p = p.model_copy(update={"scale": sc_b})

        # triangle matching is scale-invariant: a coarse grid of scales is enough when the scale is unknown
        scales = [p.scale] if p.scale else list(np.geomspace(0.4, 40, 22))
        tol = 0.15 if p.scale else 0.13
        pix = p.pixel_um
        if not p.scale:
            # most plausible first: a ~500 mm focal length when the pixel size is known, else ~3"/px
            typical = 206.265 * pix / 500.0 / S_drz if pix else 3.0
            scales.sort(key=lambda v: abs(np.log(v / typical)))
        best = None
        # pass 1: every scale with the given centre only (fast); pass 2: also shifted centres
        passes = [0.0, p.search_offsets] if p.search_offsets > 0 else [0.0]
        for ip, off in enumerate(passes):
            for k, sc in enumerate(scales):
                ctx.progress(0.6 * (ip + k / len(scales)) / len(passes),
                             f"searching the field at {sc:.2f}\"/px" + (" (shifted centres)" if off else ""))
                fw, fh = W * sc / 3600, H * sc / 3600
                if not p.scale and (fw > 40 or fw < 0.1):
                    continue
                area = fw * fh
                glim = float(np.clip(12.5 - 1.25 * np.log10(max(area, 1e-3) / 37.0), 10.5, 16.0))
                radius = 0.5 * np.hypot(fw, fh) * (1 + 2 * p.search_offsets) + 0.2
                gs, gm = gaia_bright(ra0, dec0, radius, glim)
                if len(gm) < 20:
                    continue
                res = _match(gs, gm, ra0, dec0, sc, W, H, brightdet, off, fw, fh, aa, tol)
                if res is not None and (best is None or res[2] > best[2]):
                    best = (*res, sc, glim, radius)
                    if res[2] >= GOOD_PAIRS:
                        break
            if best is not None and best[2] >= GOOD_PAIRS:
                break
        if best is None and p.blind:
            ctx.log("guided search failed: trying the all-sky blind solver", "warning")
            ra0, dec0, sc_b = _blind(ctx, brightdet, W, H)
            fw, fh = W * sc_b / 3600, H * sc_b / 3600
            glim = float(np.clip(12.5 - 1.25 * np.log10(max(fw * fh, 1e-3) / 37.0), 10.5, 16.0))
            radius = 0.5 * np.hypot(fw, fh) * (1 + 2 * p.search_offsets) + 0.2
            gs, gm = gaia_bright(ra0, dec0, radius, glim)
            res = _match(gs, gm, ra0, dec0, sc_b, W, H, brightdet, 0.0, fw, fh, aa, 0.15)
            if res is not None:
                best = (*res, sc_b, glim, radius)
        if best is None:
            raise StageError("could not match the stars with Gaia: check the target / coordinates and the image "
                             "scale (focal length, pixel size)")
        flip, T, nm, crval, sc, glim, radius = best
        w0 = _tan(crval[0], crval[1], sc, W, H)
        ctx.log(f"triangle match: {nm} stars, scale {sc * T.scale:.3f}\"/px, mirror {flip}")
        q = det[:4000].copy()
        if flip:
            q[:, 1] = H - 1 - q[:, 1]
        pred = T(q)
        sky = w0.pixel_to_world(pred[:, 0], pred[:, 1])
        w1 = fit_wcs_from_points((det[:4000, 0], det[:4000, 1]), sky, projection="TAN", sip_degree=p.sip_order or None)
        gs, gm = gaia_bright(ra0, dec0, radius, glim)
        res = None
        for it, tol in enumerate((3.0, 1.5, 1.0)):
            ctx.progress(0.7 + 0.1 * it, "refining the solution")
            gx, gy = w1.world_to_pixel(gs)
            ins = (gx > 5) & (gx < W - 5) & (gy > 5) & (gy < H - 5)
            tr = cKDTree(det[:8000])
            dd, j = tr.query(np.c_[gx[ins], gy[ins]], distance_upper_bound=tol)
            m = np.isfinite(dd)
            w1 = fit_wcs_from_points((det[:8000][j[m], 0], det[:8000][j[m], 1]), gs[ins][m], projection="TAN",
                                     sip_degree=p.sip_order or None)
            gx2, gy2 = w1.world_to_pixel(gs[ins][m])
            res = np.hypot(gx2 - det[:8000][j[m], 0], gy2 - det[:8000][j[m], 1])
        fits.PrimaryHDU(header=w1.to_header(relax=True)).writeto(ctx.out("wcs.fits"), overwrite=True)
        c = w1.pixel_to_world(W / 2, H / 2)
        scale = abs(w1.proj_plane_pixel_scales()[0].to_value("arcsec"))
        north = w1.pixel_to_world(W / 2, H / 2 - 100)
        ang = float(c.position_angle(north).deg)
        S = float(ctx.load_json("stack", "drz_meta.json")["scale"])
        ctx.metric("ra", round(float(c.ra.deg), 5))
        ctx.metric("dec", round(float(c.dec.deg), 5))
        ctx.metric("scale_arcsec", round(scale, 4))
        ctx.metric("sensor_scale_arcsec", round(scale * S, 4))
        ctx.metric("field_deg", [round(W * scale / 3600, 3), round(H * scale / 3600, 3)])
        ctx.metric("up_position_angle", round(ang, 2))
        ctx.metric("parity_flipped", bool(np.linalg.det(w1.pixel_scale_matrix) > 0))
        ctx.metric("residual_px", round(float(np.median(res)), 3))
        ctx.metric("stars_used", int(len(res)))
        pix = ctx.metric_of("inventory", "pixel_um")
        if pix:
            ctx.metric("focal_estimate_mm", round(206.265 * pix / (scale * S), 1))
        ctx.log(f"solved: centre RA {c.ra.deg:.4f} Dec {c.dec.deg:.4f}, {scale:.3f}\"/px, field "
                f"{W * scale / 3600:.2f} x {H * scale / 3600:.2f} deg, residual {np.median(res):.3f} px on {len(res)} stars")


def _blind(ctx, xy, W, H):
    """All-sky blind solve of the brightest stars: (ra, dec, scale) of the field centre."""
    from ..core.blindsolve import solve_any
    ctx.progress(0.05, "all-sky blind solving (first use builds the star index, a few minutes)")
    r = solve_any(xy[:80], W, H)
    if r is None:
        raise StageError("the field could not be identified on the sky: enter the target name or its coordinates")
    ctx.log(f"blind solve: centre RA {r['ra']:.3f} Dec {r['dec']:+.3f}, {r['scale']:.2f}\"/px, "
            f"{r['matches']} catalogue stars matched (chance level {r['chance']})")
    ctx.metric("blind_solved", True)
    return r["ra"], r["dec"], r["scale"]


def _tan(ra, dec, scale, W, H):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [W / 2, H / 2]
    w.wcs.cdelt = [-scale / 3600, scale / 3600]
    return w


def _match(gs, gm, ra0, dec0, sc, W, H, brightdet, off, fw, fh, aa, tol):
    best = None
    dra_l = [0.0] + ([-off * fw, off * fw] if off > 0 else [])
    ddec_l = [0.0] + ([-off * fh, off * fh] if off > 0 else [])
    for dra in dra_l:
        for ddec in ddec_l:
            dec_c = float(np.clip(dec0 + ddec, -89.9, 89.9))
            w0 = _tan(ra0 + dra / np.cos(np.radians(dec_c)), dec_c, sc, W, H)
            px, py = w0.world_to_pixel(gs)
            inb = (px > -0.15 * W) & (px < 1.15 * W) & (py > -0.15 * H) & (py < 1.15 * H)
            if inb.sum() < 20:
                continue
            order = np.argsort(gm[inb])
            gpos = np.c_[px[inb], py[inb]][order][:150]
            for flip in (False, True):
                q = brightdet.copy()
                if flip:
                    q[:, 1] = H - 1 - q[:, 1]
                try:
                    T, (a, _) = aa.find_transform(q, gpos, max_control_points=100)
                except Exception:  # noqa: BLE001 - no match for this hypothesis
                    continue
                if abs(np.log(T.scale)) < tol and (best is None or len(a) > best[2]):
                    best = (flip, T, len(a), list(w0.wcs.crval))
        if best is not None and best[2] >= GOOD_PAIRS:
            break
    return best


_GAIA_CACHE: dict = {}


def gaia_bright(ra, dec, radius, glim):
    """Gaia DR3 stars brighter than glim within radius (deg) of (ra, dec): (SkyCoord, G)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier
    key = (round(ra, 2), round(dec, 2), round(radius, 1), round(glim, 1))
    if key not in _GAIA_CACHE:
        v = Vizier(columns=["RA_ICRS", "DE_ICRS", "Gmag"], column_filters={"Gmag": f"<{glim:.2f}"}, row_limit=-1,
                   timeout=600)
        r = v.query_region(SkyCoord(ra, dec, unit="deg"), radius=radius * u.deg, catalog="I/355/gaiadr3")
        if not len(r):
            _GAIA_CACHE[key] = (SkyCoord([], [], unit="deg"), np.zeros(0))
        else:
            t = r[0]
            _GAIA_CACHE[key] = (SkyCoord(t["RA_ICRS"], t["DE_ICRS"], unit="deg"), np.asarray(t["Gmag"], float))
    return _GAIA_CACHE[key]


def resolve_name(name: str) -> tuple[float, float]:
    from astroquery.simbad import Simbad
    candidates = [name, name.replace("_", " "), name.replace("_nebula", " Nebula").replace("_", " ")]
    for c in candidates:
        try:
            t = Simbad.query_object(c)
        except Exception:  # noqa: BLE001
            t = None
        if t is not None and len(t):
            cols = {k.lower(): k for k in t.colnames}
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            if "ra" in cols and np.issubdtype(np.asarray(t[cols["ra"]]).dtype, np.floating):
                return float(t[cols["ra"]][0]), float(t[cols["dec"]][0])
            sc = SkyCoord(t[cols.get("ra", "RA")][0], t[cols.get("dec", "DEC")][0], unit=(u.hourangle, u.deg))
            return float(sc.ra.deg), float(sc.dec.deg)
    raise StageError(f"SIMBAD does not know '{name}': enter the RA/Dec of the field centre")
