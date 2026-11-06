"""Colour calibration: the photometric white balance of the stars, one factor per colour channel.

The colours of a few thousand isolated stars of the field are measured on the stack (aperture photometry, local
background) and compared with the colours their Gaia XP spectra give in the output's RGB (linear sRGB, the white
reference ``white_temp`` rendered neutral). For red and blue, a robust straight line through the stars (measured
colour ratio against the catalogue's) gives the factor that brings the white reference to neutral; green is 1. The
camera's channels are not mixed: this is the white balance of the photometric calibrations of Siril and PixInsight
(PCC, SPCC). An earlier version also reconstructed the sensor's sensitivity curves from the stars and fitted a colour
matrix on them; the matrix mixed the channels arbitrarily where the stars do not constrain it (emission lines: the
H-alpha of the Rosette and of the Heart came out magenta) and was taken out.

Outputs: M.npy (3, 3) camera -> linear sRGB (diagonal), color_stars.npz (every catalogue star: measured, expected,
used or why not, for the Inspect mode).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage, StageError
from .gradient import image_in

XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660, 1.8760108, 0.0415560],
                        [0.0556434, -0.2040259, 1.0572252]])
#: the fit of every catalogue star, for the Inspect mode (``star_fit``)
STARS = "color_stars.npz"
#: what the stage applies
APPLIED = "one factor per colour"


class ColorParams(StageParams):
    aperture: float = P(5.0, label="Photometry aperture", level=ADVANCED, ge=1.5, le=30, step=0.5, unit="px",
                        help="Radius of the circular aperture used to measure the colour stars on the stack.",
                        tip="About twice the star FWHM; larger apertures are safer with poor seeing but noisier.")
    white_temp: float = P(5778.0, label="White reference", ge=3000, le=12000, step=100, unit="K",
                          help="Colour temperature rendered as neutral white. 5778 K is the Sun: a G2V star appears "
                               "white, hotter stars blue, cooler stars orange, as in professional colour images.",
                          tip="This is the physical calibration; the final rendering tab can still warm or cool "
                              "the image without touching this.")


class Color(Stage):
    name = "color"
    title = "Colour calibration"
    description = ("Measures a few thousand isolated stars whose spectrum is known from Gaia XP and sets one factor "
                   "per colour so that their colours come out right, the chosen white reference neutral: the "
                   "photometric white balance of the usual colour calibrations. The colours of the camera are not "
                   "mixed.")
    deps = ("catalogs", "stack", "register")
    Params = ColorParams
    # 2: physical check, camera profiles, synthetic spectra in the matrix fit; 3: saturation from the stack's map;
    # 4: the white point set by the stars' white balance; 5: one factor per colour when no curves are trusted;
    # 6: one factor per colour always: the reconstructed curves and their colour matrix are taken out
    version = "6"

    def auto(self, ctx: Context) -> dict[str, Auto]:
        try:
            S = float(ctx.load_json("stack", "drz_meta.json")["scale"])
            f = float(ctx.metric_of("register", "fwhm_px"))
            return {"aperture": Auto(round(max(4.0, 0.7 * f * S), 1), "star FWHM {fwhm:.1f} px x drizzle {scale}",
                                    {"fwhm": f, "scale": S})}
        except Exception:  # noqa: BLE001
            return {}

    def run(self, ctx: Context, p: ColorParams) -> None:
        import colour
        img = image_in(ctx).img                       # its sky removed, the Color blocks before it applied (#47)
        xp = np.load(ctx.input("catalogs", "gaia_xp.npz"))
        table, star_why = colour_star_table(img, xp["x"], xp["y"], p.aperture, saturation_map(ctx))
        idx = np.where(star_why == "")[0]
        cam = table[idx]
        if len(idx) < 50:
            raise StageError(f"only {len(idx)} colour stars could be measured (need 50)")
        ctx.progress(0.5, f"white balance on {len(idx)} stars")
        W_ref = white_xyz(p.white_temp)
        W_d65 = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
        to_out = XYZ_TO_SRGB @ bradford(W_ref, W_d65)          # XYZ -> the output's RGB, the white reference neutral
        expected = xp["xyz"][idx] @ to_out.T
        wb = star_white_balance(cam, expected)
        M = np.diag(wb["k"]) / float(np.mean(1.0 / wb["k"]))   # a grey of the camera keeps its brightness
        ctx.save("M.npy", M)
        np.savez(ctx.out(STARS), source_kind=APPLIED, **star_fit(xp, table, star_why, idx, M, to_out))
        q = cam @ M.T
        q = q / q.sum(1, keepdims=True)
        e = expected / expected.sum(1, keepdims=True)
        rs = np.hypot(*(q - e)[:, [0, 2]].T)
        ctx.metric("stars", int(len(idx)))
        ctx.metric("white_balance", [round(float(v), 4) for v in wb["k"]])
        ctx.metric("white_balance_stars", wb["stars"])
        ctx.metric("chromaticity_residual", round(float(np.median(rs)), 4))
        ctx.metric("matrix", np.round(M, 4).tolist())
        ctx.log(f"white balance from {wb['stars']} Gaia XP stars: red x {wb['k'][0]:.3f}, blue x {wb['k'][2]:.3f} "
                f"(green 1); colour residual {np.median(rs):.4f}")


def star_white_balance(measured, expected) -> dict:
    """The white balance of the stars, as the spectrophotometric colour calibrations make it (PixInsight's SPCC
    [P21], Siril's SPCC [S13]): for red and blue, a robust straight line through the stars' measured colour ratio
    (c / green) against the ratio their catalogue spectrum gives, evaluated at the ratio of the white reference; the
    factor brings that value to the white reference's. Here the ratios are taken after the colour matrix, against the
    Gaia XP spectra put in the same output space (``measured``, ``expected`` (N, 3); the white reference is (1, 1, 1)
    there), so the white point no longer rests on the matrix's least-squares fit alone: the stars of a demosaiced
    stack have their colours pulled towards grey (RCD takes the fine detail of red and blue from green), which makes
    the fitted sensitivity curves overlap and moved heart's white from BP-RP 0.88 to 0.71. The line is Theil-Sen's
    (the median of the pairwise slopes, Sen 1968), robust to the per-star scatter. Returns {"k": (3,) factors (green
    1), "stars": the stars used, "at_white": the measured red and blue ratios at the white reference}.

    [P21] PixInsight SpectrophotometricColorCalibration documentation (2022): the calibration factors are the
    fitted functions evaluated at the white reference; [S13] Siril 1.2 documentation, Spectrophotometric Color
    Calibration."""
    from scipy.stats import theilslopes
    m, e = np.asarray(measured, float), np.asarray(expected, float)
    ok = np.isfinite(m).all(1) & np.isfinite(e).all(1) & (m > 0).all(1) & (e > 0).all(1)
    k = np.ones(3)
    at = [1.0, 1.0]
    if ok.sum() < 20:
        return {"k": k, "stars": int(ok.sum()), "at_white": at}
    for j, c in enumerate((0, 2)):
        x, y = e[ok, c] / e[ok, 1], m[ok, c] / m[ok, 1]
        slope, intercept = theilslopes(y, x)[:2]
        v = float(intercept + slope * 1.0)
        if np.isfinite(v) and v > 0:
            k[c] = 1.0 / v
            at[j] = v
    return {"k": k, "stars": int(ok.sum()), "at_white": at}


def saturation_map(ctx: Context, clipped: bool = False) -> np.ndarray | None:
    """The stack's map of the pixels with samples near saturation (``stack.SATURATION``, layer 0), or with clipped
    samples (``clipped``, layer 1); None for a stack made before it kept one."""
    from .stack import SATURATION
    try:
        f = Path(ctx.input("stack", SATURATION))
    except StageError:
        return None
    return np.load(f)[1 if clipped else 0] if f.is_file() else None


def colour_stars(img, x, y, aperture, sat=None):
    """The colour stars of a stack: the catalogue stars at (x, y) away from the edges, measured in the aperture,
    positive in every channel and unsaturated (``colour_star_table``). Returns (their indices in x, their camera
    fluxes (N, 3))."""
    flux, why = colour_star_table(img, x, y, aperture, sat)
    ok = why == ""
    return np.where(ok)[0], flux[ok]


#: why a catalogue star is not a colour star (``colour_star_table``)
STAR_REASONS = {"edge": "within 50 px of the edge of the stack",
                "not_finite": "no measure (empty pixels in its aperture)",
                "not_positive": "no positive flux in every channel (faint or on a dark background)",
                "saturated": "saturated or near saturation in at least one frame"}


def colour_star_table(img, x, y, aperture, sat=None):
    """Every catalogue star at (x, y) as ``colour_stars`` judges it: (camera fluxes (N, 3), NaN for the stars too close
    to the edge to be measured; why each one is left out, '' for a colour star, else a key of STAR_REASONS).

    Saturated: with the stack's map ``sat`` (``saturation_map``), a sample near saturation within the aperture (one
    pixel more); without it (a stack made before it kept one), a peak above 45 % of the brightest pixels."""
    Hh, Ww = img.shape[1:]
    m = (x > 50) & (x < Ww - 50) & (y > 50) & (y < Hh - 50)
    flux, peak = star_photometry(img, np.c_[x[m], y[m]], aperture)
    if sat is not None and sat.shape == (Hh, Ww):
        from scipy.ndimage import maximum_filter
        r = int(np.ceil(aperture)) + 1
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        near = maximum_filter(np.asarray(sat) > 0, footprint=xx ** 2 + yy ** 2 <= r * r)
        unsat = ~near[np.round(y[m]).astype(int), np.round(x[m]).astype(int)]
    else:
        chmax = np.array([np.nanpercentile(img[c], 99.999) for c in range(3)])
        unsat = (peak < 0.45 * chmax).all(1)
    ok = (flux > 0).all(1) & unsat & np.isfinite(flux).all(1)
    why_m = np.where(~np.isfinite(flux).all(1), "not_finite",
                     np.where(~(flux > 0).all(1), "not_positive", np.where(~ok, "saturated", "")))
    out = np.full((len(x), 3), np.nan)
    out[m] = flux
    why = np.full(len(x), "edge", dtype=object)
    why[m] = why_m
    return out, why.astype(str)


def star_fit(xp, table, why, idx, M, to_out) -> dict[str, np.ndarray]:
    """Every catalogue star of ``xp`` (gaia_xp.npz) as the stage saw it: position (stack pixels), measured camera
    fluxes ``flux`` (NaN when not measured) and ``reason`` ('' for a colour star), ``used`` (the colour stars ``idx``),
    the colour of its spectrum (``cat_xy``, rg chromaticity in the output's RGB) and the one the white balance ``M``
    gives from its measured colour (``pred_xy``), their distance ``resid``, and the camera fluxes the white balance
    expects from the spectrum (``cam_pred``)."""
    x = np.asarray(xp["x"], float)
    n = len(x)
    used = np.zeros(n, bool)
    used[idx] = True
    out_rgb = np.asarray(xp["xyz"], float) @ np.asarray(to_out, float).T
    cat = out_rgb / out_rgb.sum(1, keepdims=True)
    cam = np.asarray(table, float)[idx]
    q = cam @ np.asarray(M, float).T
    q = q / q.sum(1, keepdims=True)
    pred = np.full((n, 2), np.nan)
    pred[idx] = q[:, :2]
    resid = np.full(n, np.nan)
    resid[idx] = np.hypot(*(pred[idx] - cat[idx, :2]).T)
    return {"x": x, "y": np.asarray(xp["y"], float), "source": np.asarray(xp["source"]), "g": np.asarray(xp["g"]),
            "flux": np.asarray(table, float), "reason": np.asarray(why, str), "used": used,
            "resp_keep": np.ones((n, 3), bool), "cat_xy": cat[:, :2], "pred_xy": pred, "resid": resid,
            "cam_pred": out_rgb @ np.linalg.inv(np.asarray(M, float)).T, "M_xyz": np.asarray(M, float)}


def white_xyz(T: float = 5778.0) -> np.ndarray:
    import colour
    xyz = colour.sd_to_XYZ(colour.sd_blackbody(T))
    return xyz / xyz[1]


def bradford(src_white, dst_white):
    Mb = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
    s, d = Mb @ src_white, Mb @ dst_white
    return np.linalg.inv(Mb) @ np.diag(d / s) @ Mb


def star_photometry(img, xy, r_ap):
    """Aperture photometry with local annulus background: flux (N, 3), peak (N, 3)."""
    from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry
    ap = CircularAperture(xy, r=r_ap)
    an = CircularAnnulus(xy, r_in=2.5 * r_ap, r_out=3.6 * r_ap)
    flux, peak = [], []
    for c in range(3):
        bk = ApertureStats(img[c], an, sigma_clip=None).median
        st = ApertureStats(img[c], ap)
        f = aperture_photometry(img[c], ap)["aperture_sum"].data - bk * ap.area
        flux.append(f)
        peak.append(st.max)
    return np.stack(flux, 1), np.stack(peak, 1)
