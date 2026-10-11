"""Colour calibration: spectral response of the sensor from Gaia XP spectra, camera -> sRGB matrix."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, nnls

from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage, StageError

XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660, 1.8760108, 0.0415560],
                        [0.0556434, -0.2040259, 1.0572252]])
# typical HII region spectrum (Balmer case B reddened, [NII], [SII], weak [OIII])
HII_LINES = {656.3: 1.0, 658.4: 0.30, 654.8: 0.10, 486.1: 1 / 7.0, 434.0: 1 / 7.0 * 0.35, 671.6: 0.08, 673.1: 0.07,
             500.7: 0.05, 495.9: 0.017}
OIII_LINES = {500.7: 1.0, 495.9: 0.33, 486.1: 0.3}


class ColorParams(StageParams):
    aperture: float = P(5.0, label="Photometry aperture", level=ADVANCED, ge=1.5, le=30, step=0.5, unit="px",
                        help="Radius of the circular aperture used to measure the colour stars on the stack.",
                        tip="About twice the star FWHM; larger apertures are safer with poor seeing but noisier.")
    white_temp: float = P(5778.0, label="White reference", ge=3000, le=12000, step=100, unit="K",
                          help="Colour temperature rendered as neutral white. 5778 K is the Sun: a G2V star appears "
                               "white, hotter stars blue, cooler stars orange, as in professional colour images.",
                          tip="This is the physical calibration; the final rendering tab can still warm or cool "
                              "the image without touching this.")
    hii_weight: float = P(60.0, label="H-alpha anchor weight", level=ADVANCED, ge=0, le=500,
                          help="How strongly the matrix is constrained to render a typical HII emission spectrum "
                               "with its true colour (stars alone constrain the continuum only).")


class Color(Stage):
    name = "color"
    title = "Colour calibration"
    description = ("Measures a few thousand isolated stars whose full spectrum is known from Gaia XP, reconstructs "
                   "the spectral sensitivity curves of your camera (sensor + filter) and derives the colour matrix "
                   "camera -> sRGB. Star colours become physical (no 'neutral background' guess) and emission "
                   "nebulae are anchored on the true colour of an H-alpha + H-beta + [NII] + [SII] spectrum.")
    deps = ("catalogs", "stack", "register")
    Params = ColorParams
    version = "1"

    def auto(self, ctx: Context) -> dict[str, Auto]:
        try:
            S = float(ctx.load_json("stack", "drz_meta.json")["scale"])
            f = float(ctx.metric_of("register", "fwhm_px"))
            return {"aperture": Auto(round(max(4.0, 0.7 * f * S), 1), f"star FWHM {f:.1f} px x drizzle {S}")}
        except Exception:  # noqa: BLE001
            return {}

    def run(self, ctx: Context, p: ColorParams) -> None:
        import colour
        img = ctx.load("stack", "drz_full.npy").astype(np.float32)
        xp = np.load(ctx.input("catalogs", "gaia_xp.npz"))
        lam = xp["lam"]
        Hh, Ww = img.shape[1:]
        x, y = xp["x"], xp["y"]
        m = (x > 50) & (x < Ww - 50) & (y > 50) & (y < Hh - 50)
        flux, peak = star_photometry(img, np.c_[x[m], y[m]], p.aperture)
        chmax = np.array([np.nanpercentile(img[c], 99.999) for c in range(3)])
        ok = (flux > 0).all(1) & (peak < 0.45 * chmax).all(1) & np.isfinite(flux).all(1)
        idx = np.where(m)[0][ok]
        cam = flux[ok]
        if len(idx) < 50:
            raise StageError(f"only {len(idx)} colour stars could be measured (need 50)")
        ctx.progress(0.3, f"fitting the spectral response on {len(idx)} stars")
        S, rel = fit_response(xp["flux"][idx], lam, cam)
        cam_hii, xyz_hii = line_vec(S, lam, HII_LINES)
        cam_oiii, xyz_oiii = line_vec(S, lam, OIII_LINES)
        xyz = xp["xyz"][idx]
        cam_n = cam / cam.sum(1, keepdims=True)
        xyz_n = xyz / xyz.sum(1, keepdims=True)
        anchors = [(cam_hii, xyz_hii, p.hii_weight), (cam_oiii, xyz_oiii, p.hii_weight / 6)]
        M0 = np.linalg.lstsq(cam_n, xyz_n, rcond=None)[0].T

        def resid(q):
            Mq = q.reshape(3, 3)
            v = cam_n @ Mq.T
            v = v / v.sum(1, keepdims=True)
            r = [((v - xyz_n)[:, :2]).ravel()]
            for cv, xv, w in anchors:
                qa = Mq @ (cv / cv.sum())
                qa = qa / qa.sum()
                r.append(w * (qa - xv / xv.sum())[:2])
            return np.concatenate(r)

        sol = least_squares(resid, M0.ravel(), loss="soft_l1", f_scale=0.01)
        M = sol.x.reshape(3, 3)
        q = cam_n @ M.T
        q = q / q.sum(1, keepdims=True)
        rs = np.hypot(*(q - xyz_n)[:, :2].T)
        W_ref = white_xyz(p.white_temp)
        W_d65 = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
        Mtot = XYZ_TO_SRGB @ bradford(W_ref, W_d65) @ M
        white_cam = np.linalg.solve(Mtot, np.ones(3))
        Mtot = Mtot * white_cam.mean()
        ctx.save("M.npy", Mtot)
        np.savez(ctx.out("specresp.npz"), lam=lam, S=S, cam_hii=cam_hii, cam_oiii=cam_oiii)
        ctx.metric("stars", int(len(idx)))
        ctx.metric("fit_residual", [round(float(v), 4) for v in rel])
        ctx.metric("chromaticity_residual", round(float(np.median(rs)), 4))
        ctx.metric("matrix", np.round(Mtot, 4).tolist())
        resp = {f"{l0:.0f}nm": [round(float(np.interp(l0, lam, S[c]) / S.max()), 3) for c in range(3)]
                for l0 in (486.1, 500.7, 656.3)}
        ctx.metric("response_at_lines", resp)
        if np.median(rs) > 0.03 or (np.diag(Mtot) < 0).any():
            ctx.log("the colour fit is poor: check that the colour pattern (Bayer) of the camera is right",
                    "warning")
        ctx.log(f"spectral response from {len(idx)} Gaia XP stars, chromaticity residual {np.median(rs):.4f}")
        ctx.preview("spectral_response", response_plot(lam, S, len(idx)))


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


def fit_response(spec, lam, cam, mu=3.0):
    """cam (N,3) = integral spec(N,L) S_c(L) dl ; S_c = non-negative sum of smooth gaussians, regularised."""
    centers = np.arange(380, 1001, 15.0)
    B = np.exp(-0.5 * ((lam[:, None] - centers[None]) / 12.0) ** 2)
    dl = np.gradient(lam)
    Fm = (spec * dl) @ B
    K = len(centers)
    D = np.diff(np.eye(K), 2, axis=0)
    curves, rel = [], []
    for c in range(3):
        yv = cam[:, c]
        A = Fm * (1.0 / yv)[:, None]
        sc = A.std(0) + 1e-30
        An = A / sc
        Aa = np.vstack([An, np.sqrt(mu) * D / sc * sc.mean()])
        ya = np.concatenate([np.ones(len(yv)), np.zeros(len(D))])
        keep = np.ones(len(yv), bool)
        a = np.zeros(K)
        r = np.zeros(len(yv))
        for _ in range(4):
            rows = np.concatenate([np.where(keep)[0], len(yv) + np.arange(len(D))])
            a, _ = nnls(Aa[rows], ya[rows], maxiter=5000)
            r = An @ a - 1
            s = 1.4826 * np.median(np.abs(r[keep]))
            keep = np.abs(r) < 4 * s
        curves.append(B @ (a / sc))
        rel.append(float(np.median(np.abs(r[keep]))))
    return np.stack(curves), rel


def line_vec(S, lam, lines):
    import colour
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    cam = np.zeros(3)
    xyz = np.zeros(3)
    for l0, f in lines.items():
        cam += f * np.array([np.interp(l0, lam, S[c]) for c in range(3)])
        xyz += f * np.array([np.interp(l0, cmfs.wavelengths, cmfs.values[:, i]) for i in range(3)])
    return cam, xyz


def response_plot(lam, S, n):
    from ..core import plots
    fig, ax = plots.figure(size=(8, 3.6))
    for c, col in zip(range(3), ("#e0605a", "#6fbf73", "#5aa0d8")):
        ax.plot(lam, S[c] / S.max(), color=col, lw=2)
    for l0, nm in ((486.1, "Hb"), (500.7, "[OIII]"), (656.3, "Ha")):
        ax.axvline(l0, color=plots.MUTED, lw=0.8, ls=":")
        ax.text(l0 + 3, 1.02, nm, fontsize=9, color=plots.FG)
    ax.set_xlim(350, 1000)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("relative response")
    ax.set_title(f"Sensor spectral response reconstructed from {n} Gaia XP stars", fontsize=11)
    return plots.to_rgb(fig)
