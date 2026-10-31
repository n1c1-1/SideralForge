"""Registration: per-frame star catalogues, reference frame, polynomial (order 3) sub-pixel alignment, quality."""
from __future__ import annotations

import os
import pickle
from multiprocessing import get_context

import numpy as np
from scipy.spatial import cKDTree

from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage, StageError
from .calibrate import VALID_MAX, open_cache


class RegisterParams(StageParams):
    detect_sigma: float = P(8.0, label="Star detection threshold", level=ADVANCED, ge=3, le=50, unit="sigma",
                            help="Minimum significance of the stars used for alignment.",
                            tip="Lower it (5) for very short or very faint frames with few stars; raise it in very "
                                "dense Milky Way fields to keep only clean stars.")
    max_stars: int = P(2500, label="Stars per frame", level=ADVANCED, ge=100, le=20000,
                       help="Brightest stars kept per frame for matching and for the alignment polynomial.")
    workers: int = P(8, label="Parallel workers", level=ADVANCED, ge=1, le=64,
                     help="CPU processes used to detect stars in parallel.")


class Register(Stage):
    name = "register"
    title = "Registration"
    description = ("Detects the stars of every calibrated frame, picks the sharpest frame of the middle of the night "
                   "as reference, and fits for each frame a 3rd-order polynomial mapping it onto the reference. A "
                   "polynomial (rather than a rotation + shift) absorbs field rotation, refraction and optical "
                   "distortion to a fraction of a pixel. Also measures, per frame, the star size (FWHM) and the "
                   "transparency (star flux relative to the reference), used to weight or reject frames.")
    deps = ("calibrate",)
    Params = RegisterParams
    version = "1"

    def auto(self, ctx: Context) -> dict[str, Auto]:
        n = max(1, min(12, (os.cpu_count() or 4) - 2))
        return {"workers": Auto(n, "{cores} CPU cores detected", {"cores": os.cpu_count()})}

    def run(self, ctx: Context, p: RegisterParams) -> None:
        mm, info = open_cache(ctx)
        n, H, W = info["n"], info["height"], info["width"]
        meta = ctx.load_json("inventory", "frames.json")["lights"]
        cache_p = ctx.ckpt / "catalogs.pkl"
        if cache_p.exists():
            cats, fwhm, rmsb = pickle.loads(cache_p.read_bytes())
        else:
            cats, fwhm, rmsb = [None] * n, np.zeros(n), np.zeros(n)
        todo = [i for i in range(n) if cats[i] is None]
        if todo:
            init = (info["path"], n, H, W, p.detect_sigma, p.max_stars)
            with get_context("spawn").Pool(p.workers, initializer=_init, initargs=init) as pool:
                for c, (idx, cat, fw, rb) in enumerate(pool.imap_unordered(_catalog, todo, chunksize=4)):
                    cats[idx], fwhm[idx], rmsb[idx] = cat, fw, rb
                    if (c + 1) % 50 == 0 or c + 1 == len(todo):
                        cache_p.write_bytes(pickle.dumps((cats, fwhm, rmsb)))
                    ctx.progress(0.9 * (n - len(todo) + c + 1) / n, f"stars detected in {n - len(todo) + c + 1}/{n} frames")
        nst = np.array([len(c) for c in cats])
        if np.median(nst) < 30:
            raise StageError(f"too few stars detected per frame (median {np.median(nst):.0f}); lower the detection "
                             "threshold or check the focus / exposure")
        cand = np.where(nst >= np.percentile(nst, 50))[0]
        mid = cand[(cand > n * 0.3) & (cand < n * 0.7)]
        cand = mid if len(mid) else cand
        ref = int(cand[np.argmin(fwhm[cand])])
        refcat = cats[ref]
        tree = cKDTree(refcat[:, :2])
        coef = np.zeros((n, 2, 10))
        nmatch, rms, transp = np.zeros(n, int), np.full(n, np.inf), np.zeros(n)
        failed = []
        for i in range(n):
            r = fit_frame(cats[i], refcat, tree, W, H)
            if r is None:
                failed.append(meta[i]["path"])
            else:
                coef[i], nmatch[i], rms[i], transp[i] = r
            ctx.progress(0.9 + 0.1 * (i + 1) / n, f"aligning frame {i + 1}/{n}")
        fin = np.isfinite(rms)
        if fin.sum() >= 3:
            # a wrong match (6.5 px on the Stellina), not a softer frame (M33: 35 stacked frames at 1.0-1.9 px)
            lim = max(2.5, 8 * float(np.median(rms[fin])))
            off = np.where(fin & (rms > lim))[0]
            for i in off:                   # a residual far above the others: a wrong match, not a sharper frame
                failed.append(meta[i]["path"])
                rms[i] = np.inf
            if len(off):
                ctx.log(f"{len(off)} frame(s) aligned with a residual above {lim:.1f} px are left out", "warning")
        if failed:
            ctx.log(f"{len(failed)} frame(s) could not be aligned and will be ignored, e.g. {failed[0]}", "warning")
        np.savez(ctx.out("reg.npz"), coef=coef, fwhm=fwhm, transp=transp, nmatch=nmatch, rms=rms, ref=ref,
                 nstars=nst, bkgrms=rmsb)
        ok = np.isfinite(rms)
        ctx.metric("reference_frame", os.path.basename(meta[ref]["path"]))
        ctx.metric("aligned", int(ok.sum()))
        ctx.metric("rms_px", round(float(np.median(rms[ok])), 3))
        ctx.metric("fwhm_px", round(float(np.median(fwhm)), 2))
        ctx.metric("fwhm_p10_p90", [round(float(np.percentile(fwhm, 10)), 2), round(float(np.percentile(fwhm, 90)), 2)])
        ctx.metric("transparency_p10_p90", [round(float(np.percentile(transp[ok], 10)), 2),
                                            round(float(np.percentile(transp[ok], 90)), 2)])
        drift = np.hypot(coef[ok, 0, 0], coef[ok, 1, 0])
        ctx.metric("drift_px", round(float(np.ptp(drift)), 1))
        ctx.log(f"{ok.sum()}/{n} frames aligned, residual {np.median(rms[ok]):.2f} px, FWHM {np.median(fwhm):.2f} px, "
                f"transparency p10/p90 {np.percentile(transp[ok], 10):.2f}/{np.percentile(transp[ok], 90):.2f}")
        ctx.preview("frame_quality", quality_plot(meta, fwhm, transp, ok))


def poly_terms(x, y, W, H):
    """Monomials up to order 3 on coordinates normalised to [-1, 1] (sensor frame)."""
    u = x / (W / 2) - 1.0
    v = y / (H / 2) - 1.0
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3], -1)


def apply_poly(coef, x, y, W, H):
    T = poly_terms(x, y, W, H)
    return T @ coef[0], T @ coef[1]


def initial_shift(a, b, W, H):
    """Coarse translation by voting on pair differences (b - a), refined on nearest neighbours.

    Votes first within +-400 px (2 px bins, the usual drift); if no clear peak, within half the frame.
    """
    d = (b[:150, None, :2] - a[None, :150, :2]).reshape(-1, 2)
    s = np.zeros(2)
    for span, bins in ((400.0, 400), (max(W, H) / 2.0, 1000)):
        hh, xe, ye = np.histogram2d(d[:, 0], d[:, 1], bins=bins, range=[[-span, span], [-span, span]])
        i, j = np.unravel_index(hh.argmax(), hh.shape)
        s = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
        if hh.max() >= 8:
            break
    dd, jj = cKDTree(a[:, :2]).query(b[:, :2] - s, distance_upper_bound=6.0)
    m = np.isfinite(dd)
    if m.sum() > 10:
        s = s + np.median((b[m, :2] - s) - a[jj[m], :2], axis=0)
    return s


def _rotate(xy, ang_deg, W, H):
    """Positions turned by ang_deg about the centre of the frame."""
    t = np.deg2rad(ang_deg)
    c = np.array([W / 2.0, H / 2.0])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (xy - c) @ R.T + c


def initial_rotation(a, b, W, H, max_deg=45.0, step=0.5):
    """(angle, shift) such that b turned by the angle about the frame centre, minus the shift, falls on a: the field of
    an alt-azimuth mount turns during the night. Votes on pair differences for each trial angle (6 px bins, the angle
    is only known to the step), the angle of the highest vote kept, its shift refined on nearest neighbours."""
    span = max(W, H) / 2.0
    bins = int(2 * span / 6)
    best = (0, None, None)
    for ang in np.arange(-max_deg, max_deg + step / 2, step):
        bb = _rotate(b[:150, :2], ang, W, H)
        d = (bb[:, None, :] - a[None, :150, :2]).reshape(-1, 2)
        hh, xe, ye = np.histogram2d(d[:, 0], d[:, 1], bins=bins, range=[[-span, span], [-span, span]])
        k = hh.max()
        if k > best[0]:
            i, j = np.unravel_index(hh.argmax(), hh.shape)
            best = (k, float(ang), np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2]))
    k, ang, s = best
    if k < 8:
        return None, None
    rb = _rotate(b[:, :2], ang, W, H)
    dd, jj = cKDTree(a[:, :2]).query(rb - s, distance_upper_bound=12.0)
    m = np.isfinite(dd)
    if m.sum() > 10:
        s = s + np.median((rb[m] - s) - a[jj[m], :2], axis=0)
    return ang, s


def fit_frame(cat, ref, tree, W, H):
    """Registration polynomial of a frame on the reference: from a translation (the usual case), or, when that matches
    few stars, from a rotation and a translation (alt-azimuth mounts: the field turns). The turned fit is kept when it
    matches more stars with a small residual: in a dense field a loose start can lock on chance pairs (Stellina, 1 800
    stars a frame: 330 chance pairs at 6.5 px beat 138 true ones at 0.3 px)."""
    if cat is None or len(cat) < 30:
        return None
    r = _fit_from(cat, ref, tree, W, H, np.c_[cat[:, 0], cat[:, 1]] - initial_shift(ref, cat, W, H), 4.0)
    if r is not None and r[1] >= 0.3 * min(len(cat), len(ref)):
        return r
    ang, s = initial_rotation(ref, cat, W, H)
    if ang is None:
        return r
    r2 = _fit_from(cat, ref, tree, W, H, _rotate(cat[:, :2], ang, W, H) - s, 8.0)
    if r2 is not None and r2[2] < max(1.0, 2 * (r[2] if r is not None else 0.5)) and (r is None or r2[1] > r[1]):
        return r2
    return r


def _fit_from(cat, ref, tree, W, H, pred, tol):
    """Cubic polynomial from a first guess of the frame's stars on the reference (pred), the matching tolerance
    shrinking with the residuals."""
    T = poly_terms(cat[:, 0], cat[:, 1], W, H)
    cx = cy = None
    for _ in range(6):
        d, j = tree.query(pred, distance_upper_bound=tol)
        m = np.isfinite(d)
        if m.sum() < 30:
            return None
        A = T[m]
        tgt = ref[j[m], :2]
        cx = np.linalg.lstsq(A, tgt[:, 0], rcond=None)[0]
        cy = np.linalg.lstsq(A, tgt[:, 1], rcond=None)[0]
        pred = np.c_[T @ cx, T @ cy]
        res = np.hypot(*(pred[m] - tgt).T)
        tol = max(0.8, min(tol, 4 * np.median(res) * 1.4826))
    d, j = tree.query(pred, distance_upper_bound=tol)
    m = np.isfinite(d)
    rms = float(np.sqrt(np.mean(d[m] ** 2)))
    ratio = np.median(cat[m, 2] / ref[j[m], 2])
    return np.stack([cx, cy]), int(m.sum()), rms, float(ratio)


def quality_plot(meta, fwhm, transp, ok):
    from datetime import datetime

    from ..core import plots
    t = [datetime.fromisoformat(m["time"]) if m.get("time") else None for m in meta]
    x = t if all(t) else list(range(len(meta)))
    fig, ax = plots.figure(2, 1, size=(9, 4.6), sharex=True)
    ax[0].plot(x, np.where(ok, transp, np.nan), ".", ms=3, color=plots.AMBER)
    ax[0].set_ylabel("transparency")
    ax[1].plot(x, fwhm, ".", ms=3, color=plots.BLUE)
    ax[1].set_ylabel("FWHM (px)")
    fig.autofmt_xdate()
    return plots.to_rgb(fig)


# ---------------------------------------------------------------------- worker processes
_G: dict = {}


def _init(path, n, H, W, k, nmax):
    _G.update(mm=np.memmap(path, dtype=np.float16, mode="r", shape=(n, H, W)), H=H, W=W, k=k, nmax=nmax)


def _catalog(idx):
    import sep
    H, W = _G["H"], _G["W"]
    a = np.asarray(_G["mm"][idx], np.float32)[: H - H % 2, : W - W % 2].reshape(H // 2, 2, W // 2, 2)
    sat = (a > VALID_MAX).any(axis=(1, 3))
    s = np.nanmean(a, axis=(1, 3)) * 4.0
    s[sat] = np.nan
    mask = ~np.isfinite(s)
    s = np.where(mask, np.nanmedian(s), s).astype(np.float32)
    bkg = sep.Background(s, mask=mask, bw=64, bh=64)
    x = s - bkg.back()
    rms = bkg.globalrms
    sep.set_extract_pixstack(3000000)
    o = sep.extract(x, _G["k"], err=rms, minarea=4, mask=mask, deblend_cont=0.005)
    o = o[(o["flag"] == 0) & (o["npix"] < 200)]
    o = np.sort(o, order="flux")[::-1][: _G["nmax"]]
    if len(o) == 0:
        return idx, np.zeros((0, 4)), 0.0, float(rms)
    xw, yw, _ = sep.winpos(x, o["x"], o["y"], np.full(len(o), 0.9))
    hfr, _ = sep.flux_radius(x, xw, yw, np.full(len(o), 5.0), 0.5, subpix=5)
    cat = np.c_[2 * xw + 0.5, 2 * yw + 0.5, o["flux"], hfr]       # sensor coordinates
    cat = cat[np.isfinite(cat).all(1)]
    fwhm = 2.0 * 2.0 * np.median(cat[:300, 3])                     # super-pixel HFR -> sensor FWHM (gaussian)
    return idx, cat.astype(np.float64), float(fwhm), float(rms)
