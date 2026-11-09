"""Registration: per-frame star catalogues, reference frame, polynomial (order 3) sub-pixel alignment, quality.

Each frame's stars are detected on 2 x 2 super-pixels (``fwhm``: the reference frame and the photometry aperture are
chosen from it), and measured by an elliptical Moffat fit on its green samples at native resolution
(core/psf_moffat.py): reg.npz's ``fwhm_moffat``, ``roundness_moffat``, ``beta_moffat`` (NaN: fewer than 10 stars
fitted), which the stack's frame filters, drizzle scale rule and method advice read.
"""
from __future__ import annotations

import os
import pickle
from multiprocessing import get_context

import numpy as np
from scipy.spatial import cKDTree

from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage, StageError
from .calibrate import VALID_MAX, open_cache

#: stars fitted per frame for the Moffat FWHM (core/psf_moffat.py: about 0.1 to 0.2 s a frame)
MOFFAT_STARS = 150


class RegisterParams(StageParams):
    detect_sigma: float = P(8.0, label="Detection threshold", level=ADVANCED, ge=3, le=50, unit="sigma",
                            help="Minimum significance of the stars used for alignment.",
                            tip="Lower it (5) for very short or very faint frames with few stars; raise it in very "
                                "dense Milky Way fields to keep only clean stars.")
    max_stars: int = P(2500, label="Stars per frame", level=ADVANCED, ge=100, le=20000,
                       help="Brightest stars kept per frame for matching and for the alignment polynomial.")
    align: bool = P(True, label="Align frames", level=ADVANCED,
                    help="Off when the Registration block is cut (issue #43): the frames are taken as already aligned "
                         "(each stays where it is); the stars are still measured, to weigh and select the frames.")
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
    # reg.npz's "roundness" is an extra array (same results): an older result lacks it; 2: an affine map before the
    # cubic (a frame turned by 0.7 degree with a 0.2 % change of scale locked on a third of its stars, 6 px off at
    # the edges); the transparency from a fixed aperture (it followed the seeing); the pixel noise of each frame
    version = "2"

    def auto(self, ctx: Context) -> dict[str, Auto]:
        n = max(1, min(12, (os.cpu_count() or 4) - 2))
        return {"workers": Auto(n, "{cores} CPU cores detected", {"cores": os.cpu_count()})}

    def run(self, ctx: Context, p: RegisterParams) -> None:
        mm, info = open_cache(ctx)
        n, H, W = info["n"], info["height"], info["width"]
        meta = ctx.load_json("inventory", "frames.json")["lights"]
        cache_p = ctx.ckpt / "catalogs.pkl"
        got = pickle.loads(cache_p.read_bytes()) if cache_p.exists() else None
        psf: list = [None] * n               # the Moffat fit per frame
        if got is not None and len(got) in (5, 6, 7):        # 7: with the colour offsets of the removed per-block mode
            cats, fwhm, rmsb, rnd, noise = got[:5]
            if len(got) >= 6:
                psf = list(got[5])
        else:                                # no checkpoint, or one from an older version: those frames again
            cats, fwhm, rmsb, rnd = [None] * n, np.zeros(n), np.zeros(n), np.full(n, np.nan)
            noise = np.full(n, np.nan)
        todo = [i for i in range(n) if cats[i] is None or psf[i] is None]
        if todo:
            init = (info["path"], n, H, W, p.detect_sigma, p.max_stars, info["cfa"])
            # said at once, and one frame per task: a dense field takes 10 s and more per frame, so that a batch of 4
            # per worker kept the progress at 0 % for minutes (North America: first result after 141 s, issue #34)
            ctx.progress(0.0, f"detecting the stars of {len(todo)} frames ({p.workers} processes)")
            with get_context("spawn").Pool(p.workers, initializer=_init, initargs=init) as pool:
                for c, res in enumerate(pool.imap_unordered(_catalog_moffat, todo, chunksize=1)):
                    idx, cat, fw, rb, rd, nz, psf[idx] = res
                    cats[idx], fwhm[idx], rmsb[idx], rnd[idx], noise[idx] = cat, fw, rb, rd, nz
                    if (c + 1) % 50 == 0 or c + 1 == len(todo):
                        cache_p.write_bytes(pickle.dumps((cats, fwhm, rmsb, rnd, noise, psf)))
                    ctx.progress(0.9 * (n - len(todo) + c + 1) / n, f"stars detected in {n - len(todo) + c + 1}/{n} frames")
        nst = np.array([len(c) for c in cats])
        if not p.align:
            self._not_aligned(ctx, p, meta, cats, fwhm, rmsb, rnd, noise, psf, nst, W, H)
            return
        sparse = bool(np.median(nst) < SPARSE_STARS)         # the frames' polynomial may not find its 30 matched stars
        if np.median(nst) < SPARSE_MIN_MATCHES:
            raise StageError(f"too few stars detected per frame (median {np.median(nst):.0f}); lower the detection "
                             "threshold or check the focus / exposure")

        def align_on(ref):
            """Every frame on the reference ``ref``: (ref, its catalogue, its tree, coef, nmatch, rms, transp, failed,
            recovered, from_few)."""
            refcat = cats[ref]
            tree = cKDTree(refcat[:, :2])
            coef = np.zeros((n, 2, 10))
            nmatch, rms, transp = np.zeros(n, int), np.full(n, np.inf), np.zeros(n)
            failed = []
            recovered, from_few = [], 0
            turn = None                                  # the half turn of the last frame found so
            for i in range(n):
                r = None
                if turn is not None:
                    r, turn = fit_half_turn(cats[i], refcat, tree, W, H, turn)
                if r is None:
                    r = fit_frame(cats[i], refcat, tree, W, H)
                    if r is None or r[1] < 0.3 * min(len(cats[i]), len(refcat)):
                        h, at = fit_half_turn(cats[i], refcat, tree, W, H)
                        if h is not None and (r is None or h[1] > r[1]):
                            r, turn = h, at
                if r is None and cats[i] is not None and len(cats[i]) >= 30:
                    # no clear vote of the stars (a dense field: the brightest detections change from frame to frame):
                    # the shift from the images themselves (phase correlation), then the same fit (issue #37)
                    sh = phase_shift(mm, i, ref, H, W, info["cfa"])
                    if sh is not None:
                        r = _fit_from(cats[i], refcat, tree, W, H, cats[i][:, :2] - sh, 8.0)
                        if r is not None:
                            recovered.append(i)
                if r is None and sparse:
                    r = fit_sparse(cats[i], refcat, tree, W, H)
                    from_few += r is not None
                if r is None:
                    failed.append(meta[i]["path"])
                else:
                    coef[i], nmatch[i], rms[i], transp[i] = r
                ctx.progress(0.9 + 0.1 * (i + 1) / n, f"aligning frame {i + 1}/{n}")
            return ref, refcat, tree, coef, nmatch, rms, transp, failed, recovered, from_few

        # the reference is that of the session's majority: a frame of another field in the folder aligns almost none
        # of the others. Another one is tried only then, so that a session aligned on its first reference never pays
        # for this nor changes
        kept = None
        seen = np.zeros(n, bool)
        order = reference_order(nst, fwhm)
        tries = 0
        for r_try in order:
            if seen[r_try]:                                  # aligned on a reference already tried: the same field
                continue
            got = align_on(r_try)
            tries += 1
            ok_try = np.isfinite(got[5])
            seen |= ok_try
            seen[r_try] = True
            if kept is None or ok_try.sum() > np.isfinite(kept[5]).sum():
                kept = got
            if 2 * ok_try.sum() >= n or tries == REFERENCE_TRIES:
                break
        ref, refcat, tree, coef, nmatch, rms, transp, failed, recovered, from_few = kept
        if ref != order[0]:
            ctx.log(f"reference frame: {os.path.basename(meta[order[0]]['path'])} aligned too few of the frames (another "
                    f"field?), {os.path.basename(meta[ref]['path'])} is used")
        if from_few:
            ctx.log(f"sparse field: {from_few} frame(s) aligned from a few stars (rotation, shift and scale only)")
        fin = np.isfinite(rms)
        if fin.sum() >= 3:
            # a wrong match (6.5 px on the Stellina), not a softer frame (M33: 35 stacked frames at 1.0-1.9 px)
            lim = max(2.5, 8 * float(np.median(rms[fin])))
            off = np.where(fin & (rms > lim))[0]
            for i in off:                   # a residual far above the others: a wrong match, not a sharper frame
                rms[i] = np.inf
            if len(off):
                ctx.log(f"{len(off)} frame(s) aligned with a residual above {lim:.1f} px are left out, e.g. "
                        f"{os.path.basename(meta[off[0]]['path'])}", "warning")
        kept_rec = sum(1 for i in recovered if np.isfinite(rms[i]))      # each frame said once, by its final outcome
        if kept_rec:
            ctx.log(f"{kept_rec} frame(s) aligned from the shift the images' phase correlation gives (no clear vote "
                    "of their stars)")
        if failed:
            ctx.log(f"{len(failed)} frame(s) could not be aligned and will be ignored, e.g. {failed[0]}", "warning")
        # the transparency from the stars' flux in a fixed aperture (holding their light in every frame), not from
        # their flux above the detection threshold, which follows the seeing: on Iris a frame of 5.3 px stars read
        # 0.78 for 0.91 (its signal was scaled up 16 % too much, and its weight cut)
        ap = aperture_column(np.nanpercentile(fwhm, 90) / 4.0)
        for i in np.where(np.isfinite(rms))[0]:
            t_ap = aperture_ratio(cats[i], refcat, tree, coef[i], W, H, ap)
            if np.isfinite(t_ap):
                transp[i] = t_ap
        ctx.metric("photometry_aperture_px", 2 * APERTURES[ap - 4])
        extra = moffat_arrays(psf)
        ok_m = np.isfinite(extra["fwhm_moffat"])
        if ok_m.any():
            ctx.metric("fwhm_moffat_px", round(float(np.median(extra["fwhm_moffat"][ok_m])), 3))
            ctx.metric("roundness_moffat", round(float(np.median(extra["roundness_moffat"][ok_m])), 3))
            ctx.log(f"Moffat fit: FWHM {np.median(extra['fwhm_moffat'][ok_m]):.2f} px (green samples, "
                    f"{int(ok_m.sum())}/{n} frames measured), against {np.median(fwhm):.2f} px on 2 x 2 blocks")
        else:
            ctx.log("Moffat fit: too few isolated unsaturated stars in every frame: no Moffat FWHM", "warning")
        np.savez(ctx.out("reg.npz"), coef=coef, fwhm=fwhm, transp=transp, nmatch=nmatch, rms=rms, ref=ref,
                 nstars=nst, bkgrms=rmsb, roundness=rnd, noise=noise, **extra)
        ok = np.isfinite(rms)
        ctx.metric("reference_frame", os.path.basename(meta[ref]["path"]))
        ctx.metric("aligned", int(ok.sum()))
        if not ok.all():
            ctx.metric("not_aligned", int((~ok).sum()))
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

    def _not_aligned(self, ctx: Context, p: RegisterParams, meta, cats, fwhm, rmsb, rnd, noise, psf, nst, W, H):
        """The Registration block cut (issue #43): every frame stays where it is (the identity map), its stars still
        matched to the reference's within 3 px for the residual and the transparency."""
        n = len(cats)
        ctx.log("Registration cut: the frames are taken as already aligned")
        ref = int(np.argmin(np.where(nst >= np.percentile(nst, 50), fwhm, np.inf)))
        refcat = cats[ref] if cats[ref] is not None and len(cats[ref]) else np.zeros((1, 3))
        tree = cKDTree(refcat[:, :2])
        ident = np.zeros((2, 10))
        ident[0, 0], ident[0, 1] = W / 2, W / 2
        ident[1, 0], ident[1, 2] = H / 2, H / 2
        coef = np.repeat(ident[None], n, axis=0)
        nmatch, rms, transp = np.zeros(n, int), np.zeros(n), np.ones(n)
        for i in range(n):
            c = cats[i]
            if c is None or not len(c):
                continue
            d, j = tree.query(c[:, :2], distance_upper_bound=3.0)
            m = np.isfinite(d)
            nmatch[i] = int(m.sum())
            if m.any():
                rms[i] = float(np.sqrt(np.mean(d[m] ** 2)))
            if m.sum() >= 5:
                transp[i] = float(np.median(c[m, 2] / refcat[j[m], 2]))
        extra = moffat_arrays(psf)
        np.savez(ctx.out("reg.npz"), coef=coef, fwhm=fwhm, transp=transp, nmatch=nmatch, rms=rms, ref=ref,
                 nstars=nst, bkgrms=rmsb, roundness=rnd, noise=noise, **extra)
        ctx.metric("registration_cut", True)
        ctx.metric("reference_frame", os.path.basename(meta[ref]["path"]))
        ctx.metric("aligned", n)
        ctx.metric("rms_px", round(float(np.median(rms)), 3))
        ctx.metric("fwhm_px", round(float(np.median(fwhm)), 2))
        ok = np.ones(n, bool)
        ctx.metric("drift_px", 0.0)
        ctx.preview("frame_quality", quality_plot(meta, fwhm, transp, ok))


#: how many frames are tried as the reference when the first ones align less than half the session
REFERENCE_TRIES = 5


def reference_order(nst: np.ndarray, fwhm: np.ndarray) -> list[int]:
    """The frames in the order they are tried as the reference: first the sharpest of the middle of the night among
    those with at least the median number of stars, then the others of that kind by sharpness, then every other."""
    n = len(nst)
    rich = np.where(nst >= np.percentile(nst, 50))[0]
    mid = rich[(rich > n * 0.3) & (rich < n * 0.7)]
    cand = mid if len(mid) else rich
    first = int(cand[np.argmin(fwhm[cand])])
    sharp = np.argsort(fwhm, kind="stable").tolist()
    in_cand, in_rich = set(cand.tolist()), set(rich.tolist())
    return list(dict.fromkeys([first, *(i for i in sharp if i in in_cand), *(i for i in sharp if i in in_rich), *sharp]))


def moffat_arrays(psf: list) -> dict[str, np.ndarray]:
    """reg.npz's Moffat arrays from the workers' fits ((fwhm, roundness, beta, stars) per frame, NaN where fewer
    than psf_moffat.MIN_STARS stars were fitted): ``fwhm_moffat`` (sensor px), ``roundness_moffat`` (minor / major),
    ``beta_moffat``, ``stars_moffat``."""
    a = np.array([r if r is not None else (np.nan, np.nan, np.nan, 0) for r in psf], float).reshape(-1, 4)
    return {"fwhm_moffat": a[:, 0], "roundness_moffat": a[:, 1], "beta_moffat": a[:, 2],
            "stars_moffat": a[:, 3].astype(np.int32)}


def poly_terms(x, y, W, H):
    """Monomials up to order 3 on coordinates normalised to [-1, 1] (sensor frame)."""
    u = x / (W / 2) - 1.0
    v = y / (H / 2) - 1.0
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3], -1)


def apply_poly(coef, x, y, W, H):
    T = poly_terms(x, y, W, H)
    return T @ coef[0], T @ coef[1]


#: the Poisson sigmas a vote's peak stands above the mean count of a bin (the chance pairs)
VOTE_SIGMAS = 8.0


def initial_shift(a, b, W, H):
    """Coarse translation by voting on pair differences (b - a), refined on nearest neighbours.

    Votes first within +-400 px (2 px bins, the usual drift), on the 150 brightest stars, then on more (400, 1000):
    on a dense field the brightest detections change from frame to frame (the saturated stars masked, crowded stars
    merged or split), so that two lists of 150 share too few stars (North America, 5 frames of 109 lost, issue #37).
    A peak counts when it stands clear of the chance pairs (VOTE_SIGMAS Poisson sigmas over the mean count of a bin:
    more stars, more chance pairs). Only then within half the frame.
    """
    s = np.zeros(2)
    found = False
    for span, bins, nst in ((400.0, 400, 150), (400.0, 400, 400), (400.0, 400, 1000), (max(W, H) / 2.0, 1000, 150)):
        d = (b[:nst, None, :2] - a[None, :nst, :2]).reshape(-1, 2)
        hh, xe, ye = np.histogram2d(d[:, 0], d[:, 1], bins=bins, range=[[-span, span], [-span, span]])
        i, j = np.unravel_index(hh.argmax(), hh.shape)
        s = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
        mean = hh.sum() / hh.size
        if hh.max() >= max(8.0, mean + VOTE_SIGMAS * np.sqrt(mean) + 5.0):
            found = True
            break
    if not found:                                    # no clear peak anywhere: the brightest pairs' best guess
        d = (b[:150, None, :2] - a[None, :150, :2]).reshape(-1, 2)
        span = max(W, H) / 2.0
        hh, xe, ye = np.histogram2d(d[:, 0], d[:, 1], bins=1000, range=[[-span, span], [-span, span]])
        i, j = np.unravel_index(hh.argmax(), hh.shape)
        s = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
    dd, jj = cKDTree(a[:, :2]).query(b[:, :2] - s, distance_upper_bound=6.0)
    m = np.isfinite(dd)
    if m.sum() > 10:
        s = s + np.median((b[m, :2] - s) - a[jj[m], :2], axis=0)
    return s


#: the phase correlation's reduction (sensor pixels per pixel of the images it compares) and the least peak it trusts
PHASE_BIN, PHASE_MIN = 4, 0.1


def phase_shift(mm, i: int, ref: int, H: int, W: int, cfa: str = "") -> np.ndarray | None:
    """The shift (frame minus reference, sensor pixels) of two calibrated frames by phase correlation (C. D. Kuglin,
    D. C. Hines, "The phase correlation image alignment method", IEEE Conf. on Cybernetics and Society, 1975), on
    their images reduced PHASE_BIN times, the sky taken off and the brightest stars clipped, under a Hanning window;
    None when its peak is under PHASE_MIN. An X-Trans frame (``cfa``) has its colours levelled first: its blocks
    hold unequal mixes of colours, a pattern that stays still and would be found as the shift."""
    import cv2

    from ..core.cfa import equalised

    def small(k):
        # bad pixels and the float16 cache's overflows (a star brighter than 65504: inf) count as empty sky
        a = np.nan_to_num(equalised(np.asarray(mm[k], np.float32), cfa, VALID_MAX), nan=0.0, posinf=0.0, neginf=0.0)
        a = a[: H - H % PHASE_BIN, : W - W % PHASE_BIN]
        a = a.reshape(a.shape[0] // PHASE_BIN, PHASE_BIN, a.shape[1] // PHASE_BIN, PHASE_BIN).mean((1, 3))
        a = a - cv2.GaussianBlur(a, (0, 0), 10)
        return np.clip(a, 0, np.percentile(a, 99.9)).astype(np.float32)
    s_ref, s_i = small(ref), small(i)
    win = cv2.createHanningWindow(s_ref.shape[::-1], cv2.CV_32F)
    (dx, dy), peak = cv2.phaseCorrelate(s_ref, s_i, win)
    return None if not peak >= PHASE_MIN or not np.isfinite([dx, dy]).all() else np.array([dx, dy]) * PHASE_BIN


def _rotate(xy, ang_deg, W, H):
    """Positions turned by ang_deg about the centre of the frame."""
    t = np.deg2rad(ang_deg)
    c = np.array([W / 2.0, H / 2.0])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (xy - c) @ R.T + c


HALF_TURN = 180.0


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


def fit_half_turn(cat, ref, tree, W, H, near=None):
    """(fit, its angle) of a frame turned by about half a turn from the reference (the two sides of a meridian flip, a
    camera put back the other way round: issue #63), (None, None) when it is not one. ``near``: the angle of the last
    frame found so, tried first, without a search: the frames of a series share it."""
    if cat is None or len(cat) < 30:
        return None, None

    def good(r):
        return r is not None and r[1] >= 0.3 * min(len(cat), len(ref)) and r[2] < 1.0
    if near is not None:
        turned = np.c_[_rotate(cat[:, :2], near, W, H), cat[:, 2:]]
        r = _fit_from(cat, ref, tree, W, H, turned[:, :2] - initial_shift(ref, turned, W, H), 8.0)
        if good(r):
            return r, near
    turned = np.c_[_rotate(cat[:, :2], HALF_TURN, W, H), cat[:, 2:]]
    ang, s = initial_rotation(ref, turned, W, H)
    if ang is None:
        return None, None
    r = _fit_from(cat, ref, tree, W, H, _rotate(turned[:, :2], ang, W, H) - s, 8.0)
    return (r, HALF_TURN + ang) if good(r) else (None, None)


def _fit_from(cat, ref, tree, W, H, pred, tol):
    """Cubic polynomial from a first guess of the frame's stars on the reference (pred). An affine map first, at the
    starting tolerance: it extends the matches from where the guess is right to the whole field (a guess from a shift
    alone is right only near the centre of a turn: a cubic fitted there locked on a third of the stars, 6 px off at
    the edges, for a turn of 0.7 degree with a 0.2 % change of scale); then the cubic, the matching tolerance
    shrinking with the residuals."""
    T = poly_terms(cat[:, 0], cat[:, 1], W, H)
    cx = cy = None
    for it in range(7):
        k = 3 if it < 3 else T.shape[1]
        d, j = tree.query(pred, distance_upper_bound=tol)
        m = np.isfinite(d)
        if m.sum() < 30:
            return None
        A = T[m, :k]
        tgt = ref[j[m], :2]
        cx = np.zeros(T.shape[1])
        cy = np.zeros(T.shape[1])
        cx[:k] = np.linalg.lstsq(A, tgt[:, 0], rcond=None)[0]
        cy[:k] = np.linalg.lstsq(A, tgt[:, 1], rcond=None)[0]
        pred = np.c_[T @ cx, T @ cy]
        res = np.hypot(*(pred[m] - tgt).T)
        if k == T.shape[1]:
            tol = max(0.8, min(tol, 4 * np.median(res) * 1.4826))
    d, j = tree.query(pred, distance_upper_bound=tol)
    m = np.isfinite(d)
    rms = float(np.sqrt(np.mean(d[m] ** 2)))
    ratio = np.median(cat[m, 2] / ref[j[m], 2])
    return np.stack([cx, cy]), int(m.sum()), rms, float(ratio)


#: a field is sparse under this median of stars per frame (a frame of 100 stars or more reaches the 30 matches of the
#: polynomial; the vote and the polynomial are never replaced above it)
SPARSE_STARS = 100
#: the triangle matching's stars (the brightest), its neighbours per star, its match radius in the space of the
#: triangles' two shape ratios, and the distance (px) at which a star counts as matched: the values of Beroiz et al.
TRIANGLE_STARS, TRIANGLE_NEIGHBOURS, TRIANGLE_RADIUS, STAR_TOL = 50, 5, 0.1, 2.0
#: fewest matched stars for a transform: the triangle's 3, which any transform fits, and 3 more it did not use (a star
#: falls within STAR_TOL of one of 50 others by chance once in 10 000 tries on a 6 Mpx frame, so three never do)
SPARSE_MIN_MATCHES = 6
#: largest residual (px) of a transform from few stars, as ``fit_frame`` asks of a turned fit; the largest change of
#: scale between two frames of one instrument (a triangle fitted to a far smaller one maps every star onto one point)
SPARSE_RMS, SPARSE_SCALE = 1.0, 0.05


def _triangles(xy, nearest=True):
    """(star indices, shape) of the triangles each star makes with its nearest neighbours (``nearest`` False: of every
    three stars): its vertices ordered by the length of the opposite side, and the two ratios of its sorted sides
    (longest / middle, middle / shortest), which neither a rotation, a shift nor a change of scale alters."""
    from itertools import combinations
    if nearest:
        k = min(TRIANGLE_NEIGHBOURS, len(xy))
        near = cKDTree(xy).query(xy, k=k)[1].reshape(len(xy), k)
        tri = np.array(sorted({tuple(sorted(t)) for row in near.tolist() for t in combinations(row, 3)}), int)
    else:
        tri = np.array(list(combinations(range(len(xy)), 3)), int)
    tri = tri.reshape(-1, 3)
    p = xy[tri]
    side = np.stack([np.hypot(*(p[:, 1] - p[:, 2]).T), np.hypot(*(p[:, 0] - p[:, 2]).T),
                     np.hypot(*(p[:, 0] - p[:, 1]).T)], 1)
    order = np.argsort(side, axis=1)
    side, tri = np.take_along_axis(side, order, 1), np.take_along_axis(tri, order, 1)
    ok = side[:, 0] > 0
    return tri[ok], np.c_[side[ok, 2] / side[ok, 1], side[ok, 1] / side[ok, 0]]


def _matched(pred, tree):
    """(frame star, reference star, distance) of the stars within STAR_TOL of a reference star, each reference star
    taken once, by the nearest."""
    d, j = tree.query(pred, distance_upper_bound=STAR_TOL)
    i = np.where(np.isfinite(d))[0]
    i = i[np.argsort(d[i], kind="stable")]
    _, first = np.unique(j[i], return_index=True)
    i = np.sort(i[first])
    return i, j[i], d[i]


def _similarity(a, b):
    """The rotation and scale ``s`` and the shift ``t`` of the least-squares similarity b = s a + t, positions as
    complex numbers (Umeyama, IEEE TPAMI 13, 1991; no mirror image)."""
    ma, mb = a.mean(-1, keepdims=True), b.mean(-1, keepdims=True)
    s = ((b - mb) * np.conj(a - ma)).sum(-1) / np.maximum((np.abs(a - ma) ** 2).sum(-1), 1e-12)
    return s, mb[..., 0] - s * ma[..., 0]


def triangle_similarities(a, b, nearest=True, max_scale=SPARSE_SCALE):
    """The similarities z_b = s z_a + t that the triangles of the positions ``a`` similar to one of ``b`` propose: (s, t,
    the three stars of ``a`` each comes from), those within ``max_scale`` of the unit scale whose triangle lands within
    STAR_TOL of its match. In a fixed order: no random draw."""
    none = np.zeros(0, complex), np.zeros(0, complex), np.zeros((0, 3), int)
    ta, sa = _triangles(a, nearest)
    tb, sb = _triangles(b, nearest)
    if not len(ta) or not len(tb):
        return none
    near = cKDTree(sa).query_ball_tree(cKDTree(sb), r=TRIANGLE_RADIUS)
    ia = np.array([i for i, v in enumerate(near) for _ in v], int)
    ib = np.array([j for v in near for j in sorted(v)], int)
    if not len(ia):
        return none
    p, q = (a[:, 0] + 1j * a[:, 1])[ta[ia]], (b[:, 0] + 1j * b[:, 1])[tb[ib]]
    s, t = _similarity(p, q)
    keep = (np.abs(np.abs(s) - 1) < max_scale) & (np.abs(s[:, None] * p + t[:, None] - q).max(1) < STAR_TOL)
    return s[keep], t[keep], ta[ia][keep]


def fit_sparse(cat, ref, tree, W, H):
    """Registration of a frame of a sparse field, where the vote on the shifts has too few stars and the polynomial
    cannot find its 30 matches: the triangle matching of V. Valdes et al. (PASP 107, 1119, 1995) and E. Groth (AJ 91,
    1244, 1986), with the triangles from each star's nearest neighbours of M. Beroiz, J. Cabral, B. Sanchez, "Astroalign"
    (Astronomy and Computing 32, 100384, 2020). Each triangle of the frame similar to one of the reference proposes a
    similarity; all are tried and the one that brings the most stars is kept, so that the result does not depend on
    any random draw. It is then refined on every star it matches, and stands only with SPARSE_MIN_MATCHES stars and a
    residual under SPARSE_RMS. A similarity has four unknowns: with so few stars, a polynomial would follow their
    noise. Returns what ``fit_frame`` returns, the polynomial's terms above order 1 at zero; None when no transform
    stands."""
    if cat is None or min(len(cat), len(ref)) < SPARSE_MIN_MATCHES:
        return None
    a, b = cat[:, :2], ref[:, :2]
    s, t, _ = triangle_similarities(a[:TRIANGLE_STARS], b[:TRIANGLE_STARS])
    za, zb = a[:, 0] + 1j * a[:, 1], b[:, 0] + 1j * b[:, 1]
    best, top = None, (0, 0.0)
    for s1, t1 in zip(s, t):
        z = s1 * za + t1
        i, _, d = _matched(np.c_[z.real, z.imag], tree)
        score = (len(i), -float(np.sum(d ** 2)))
        if score > top:
            best, top = (s1, t1), score
    if best is None:
        return None
    s1, t1 = best
    for _ in range(3):
        z = s1 * za + t1
        i, j, d = _matched(np.c_[z.real, z.imag], tree)
        if len(i) < SPARSE_MIN_MATCHES:
            return None
        s1, t1 = _similarity(za[i], zb[j])
    z = s1 * za + t1
    i, j, d = _matched(np.c_[z.real, z.imag], tree)
    if len(i) < SPARSE_MIN_MATCHES or abs(abs(s1) - 1) >= SPARSE_SCALE:
        return None
    rms = float(np.sqrt(np.mean(d ** 2)))
    if rms > SPARSE_RMS:
        return None
    # the similarity on the polynomial's coordinates (poly_terms: u = x / (W / 2) - 1, v = y / (H / 2) - 1)
    cx, cy = np.zeros(10), np.zeros(10)
    cx[:3] = s1.real * W / 2 - s1.imag * H / 2 + t1.real, s1.real * W / 2, -s1.imag * H / 2
    cy[:3] = s1.imag * W / 2 + s1.real * H / 2 + t1.imag, s1.imag * W / 2, s1.real * H / 2
    return np.stack([cx, cy]), len(i), rms, float(np.median(cat[i, 2] / ref[j, 2]))


#: radii (super-pixels) of the apertures the stars' flux is measured in; the catalogue's columns 4 onwards
APERTURES = (3.0, 5.0, 8.0, 12.0)


def aperture_column(hfr_sp: float) -> int:
    """The catalogue column of the smallest aperture of at least 3 half-flux radii of the larger stars (``hfr_sp``,
    super-pixels): it holds nearly all their light whatever the seeing."""
    for k, r in enumerate(APERTURES):
        if r >= 3 * hfr_sp:
            return 4 + k
    return 4 + len(APERTURES) - 1


def aperture_ratio(cat, ref, tree, coef, W, H, col: int) -> float:
    """Median flux ratio (frame / reference) of the stars matched within 1.5 px through the frame's polynomial, in
    the aperture column ``col``; NaN when fewer than 10 stars have it in both."""
    u, v = apply_poly(coef, cat[:, 0], cat[:, 1], W, H)
    d, j = tree.query(np.c_[u, v], distance_upper_bound=1.5)
    m = np.isfinite(d)
    a, b = cat[m, col], ref[j[m], col]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return float(np.median(a[ok] / b[ok])) if ok.sum() >= 10 else float("nan")


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


def _init(path, n, H, W, k, nmax, cfa):
    _G.update(mm=np.memmap(path, dtype=np.float16, mode="r", shape=(n, H, W)), H=H, W=W, k=k, nmax=nmax, cfa=cfa)


def _catalog_moffat(idx):
    """``_catalog`` plus the Moffat fit on the same frame and catalogue: (fwhm, roundness, beta, stars; NaN under
    psf_moffat.MIN_STARS stars)."""
    out = _catalog(idx)
    cat = out[1]
    from ..core.psf_moffat import measure_psf
    r = measure_psf(np.asarray(_G["mm"][idx], np.float32), cat[:, :2], cfa=_G["cfa"], channel="G",
                    saturation=VALID_MAX, max_stars=MOFFAT_STARS)
    psf = (float(r.fwhm), float(r.roundness), float(r.beta), int(r.n)) if r.ok else \
        (float("nan"), float("nan"), float("nan"), int(r.n))
    return (*out, psf)


def _catalog(idx):
    import sep
    H, W = _G["H"], _G["W"]
    from ..core.cfa import equalised
    # an X-Trans frame's colours levelled first (its 2 x 2 cells hold 1 to 3 greens: a pattern of the sky otherwise)
    a = equalised(np.asarray(_G["mm"][idx], np.float32), _G["cfa"], VALID_MAX)
    a = a[: H - H % 2, : W - W % 2].reshape(H // 2, 2, W // 2, 2)
    sat = (a > VALID_MAX).any(axis=(1, 3))
    s = np.nanmean(a, axis=(1, 3)) * 4.0
    s[sat] = np.nan
    mask = ~np.isfinite(s)
    s = np.where(mask, np.nanmedian(s), s).astype(np.float32)
    bkg = sep.Background(s, mask=mask, bw=64, bh=64)
    x = s - bkg.back()
    rms = bkg.globalrms
    sep.set_extract_pixstack(3000000)
    from ..core.imgutil import sep_extract
    o = sep_extract(x, _G["k"], err=rms, minarea=4, mask=mask, deblend_cont=0.005)
    o = o[(o["flag"] == 0) & (o["npix"] < 200)]
    o = np.sort(o, order="flux")[::-1][: _G["nmax"]]
    # the noise of the frame: adjacent super-pixels differ by their noise only (the sky and the objects larger than
    # a few pixels cancel; the stars are a minority the median ignores)
    dd = (x[:, 1:] - x[:, :-1])[::3, ::3]
    dd = dd[np.isfinite(dd) & ~mask[::3, :-1][:, ::3][: dd.shape[0], : dd.shape[1]]]
    noise = float(1.4826 * np.median(np.abs(dd - np.median(dd))) / np.sqrt(2)) if dd.size else float("nan")
    if len(o) == 0:
        return idx, np.zeros((0, 4 + len(APERTURES))), 0.0, float(rms), float("nan"), noise
    xw, yw, _ = sep.winpos(x, o["x"], o["y"], np.full(len(o), 0.9))
    hfr, _ = sep.flux_radius(x, xw, yw, np.full(len(o), 5.0), 0.5, subpix=5)
    aps = []
    good = (~mask).astype(np.float32)
    for r in APERTURES:
        # a star whose background annulus is wholly masked (beside a saturated or bad area) has no measure: sep raises
        # for the whole call on it, so it is left out first
        seen, _, _ = sep.sum_circann(good, xw, yw, r + 2.0, r + 6.0)
        ok = seen > 1.0
        f = np.full(len(xw), np.nan)
        if ok.any():
            fo, _, fl = sep.sum_circle(x, xw[ok], yw[ok], r, mask=mask, bkgann=(r + 2.0, r + 6.0))
            f[ok] = np.where(fl == 0, fo, np.nan)
        aps.append(f)
    cat = np.c_[2 * xw + 0.5, 2 * yw + 0.5, o["flux"], hfr, *aps]   # sensor coordinates
    fin = np.isfinite(cat[:, :4]).all(1)
    cat = cat[fin]
    fwhm = 2.0 * 2.0 * np.median(cat[:300, 3])                     # super-pixel HFR -> sensor FWHM (gaussian)
    return idx, cat.astype(np.float64), float(fwhm), float(rms), _roundness(o[fin][:300]), noise


def _roundness(o) -> float:
    """Median minor / major axis (second moments of the detection) of the stars the FWHM is measured on: 1 for round
    stars, lower when they are stretched (tracking, wind, tilt)."""
    a, b = np.asarray(o["a"], float), np.asarray(o["b"], float)
    r = b / np.where(a > 0, a, np.nan)
    r = r[np.isfinite(r)]
    return float(np.median(r)) if len(r) else float("nan")
