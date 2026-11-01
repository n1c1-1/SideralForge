"""Measures of the bench: our linear stacks against Siril's, on the same sky, plus the side-by-side sheet of the finals.

    python tools/bench/measure.py CASE LEVEL [--no-sheet]

Both images keep their own pixel grid (ours is drizzled, Siril's is not): every measure is placed on the sky with each
image's own astrometric solution, and sizes are angular. The star list is Gaia DR3 (the catalogue our engine
downloaded for the field: positions at the date of the frames, G, BP, RP); only the stars inside both images are used.

On the raw linear stacks (ours: stack/drz_full; Siril: stack_linear.fit), luminance = mean of the three channels:
- photometric zero point: G + 2.5 log10(flux) on isolated unsaturated stars, so that both images speak in magnitudes;
- background noise: robust spread of boxes of the same angular size in the sky (stars masked, smooth background
  removed), as a surface brightness (mag/arcsec2: higher = less noise); and the 5-sigma depth of a point source
  (empty apertures of radius 1 FWHM: higher = deeper);
- faint stars: Gaia stars measured in an aperture of 1 FWHM above 5 sigma (local empty-aperture noise); their count
  and the magnitude where half the stars are found;
- target SNR: median signal of the target's boxes (above the darkest sky) over the box noise inside the target;
- stars: elliptical Gaussian fit on unsaturated isolated stars: FWHM (arcsec), roundness (minor / major), at the
  centre and in the outer part of the field;
- trails and rejection residuals: high-pass image over its noise, Gaia stars masked; long thin features (trails),
  compact spots above +6 / below -6 sigma; a spot found in one image only (not at a spot of the other) is an artefact.
On the gradient-removed, colour-calibrated linear images (ours: render_simple/lin; Siril: color_linear.fit):
- background flatness: floor of 8 x 8 tiles (median of the darkest fifth, stars masked); spread p90 - p10, as a surface
  brightness (mag/arcsec2 through the zero point: higher = flatter);
- star colours: -2.5 log10(B/R) and -2.5 log10(G/R) of isolated stars against Gaia BP-RP: robust scatter around a
  straight line (mag: lower = more faithful), the slope, and the BP-RP that comes out white.
Writes <case>/measures_<level>.json and <case>/sheet_<level>.jpg.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

os.environ["USERPROFILE"] = str(C.HOME)
os.environ["ASTRO_GPU_HOME"] = str(C.AG_HOME)
sys.path.insert(0, str(C.REPO))
C.utf8_console()

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

MAD = 1.4826


def rstd(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(MAD * np.median(np.abs(v - np.median(v)))) if v.size else float("nan")


# ---------------------------------------------------------------------------------------------- loading

class Img:
    """A linear image (3, H, W) on its own grid, with its WCS and its coverage mask."""

    def __init__(self, name: str, rgb: np.ndarray, wcs: WCS, cover: np.ndarray):
        self.name, self.rgb, self.wcs, self.cover = name, rgb, wcs, cover
        self.H, self.W = rgb.shape[1:]
        self.L = rgb.mean(0, dtype=np.float32)
        sc = np.abs(np.linalg.det(wcs.pixel_scale_matrix)) ** 0.5 * 3600
        self.scale = float(sc)                      # arcsec / px
        self.pixarea = self.scale ** 2

    def pix(self, ra, dec):
        x, y = self.wcs.all_world2pix(np.asarray(ra, float), np.asarray(dec, float), 0)
        return x + self.dx, y + self.dy

    dx = dy = 0.0

    def inside(self, x, y, margin: float = 0) -> np.ndarray:
        xi, yi = np.round(x).astype(int), np.round(y).astype(int)
        ok = (xi >= margin) & (yi >= margin) & (xi < self.W - margin) & (yi < self.H - margin)
        out = np.zeros(len(x), bool)
        out[ok] = self.cover[yi[ok], xi[ok]]
        return out


def _erode(m: np.ndarray, px: int) -> np.ndarray:
    small = m[::8, ::8]
    e = ndi.binary_erosion(small, iterations=max(1, px // 8), border_value=0)
    return np.kron(e, np.ones((8, 8), bool))[: m.shape[0], : m.shape[1]] & m


def common_cover(a: Img, b: Img) -> np.ndarray:
    """a's coverage limited to the sky b also covers (b's coverage looked up through both solutions, on an 8-px
    lattice of a)."""
    step = 8
    ys, xs = np.mgrid[0:a.H:step, 0:a.W:step]
    ra, dec = a.wcs.all_pix2world(xs.ravel().astype(float), ys.ravel().astype(float), 0)
    xb, yb = b.wcs.all_world2pix(ra, dec, 0)
    ok = b.inside(xb, yb, 0).reshape(xs.shape)
    full = np.kron(ok, np.ones((step, step), bool))[: a.H, : a.W]
    return a.cover & full


def load_siril(case: str, level: str, which: str) -> Img:
    f = C.BENCH / case / f"siril_{level}" / f"{which}.fit"
    with fits.open(f, memmap=False) as h:
        data = np.asarray(h[0].data, np.float32)
        hdr = h[0].header
    if data.ndim == 2:
        data = np.repeat(data[None], 3, 0)
    cover = _erode(np.all(data != 0, axis=0), 40)
    return Img(f"Siril {which}", data, WCS(hdr, naxis=2), cover)


def ours_dirs(case: str, level: str) -> dict[str, Path]:
    from astro_gpu import api
    root = C.BENCH / case / f"ours_{level}"
    p = api.open_project(root)
    w = next(w for w in p.workflows if w.name == "Classic stack")
    out = {}
    for s in api.chain(p, w.id):
        o = api.stage_outputs(p, w.id, s)
        if o is not None:
            out[s] = Path(o.dir)
    return out


def load_ours(d: dict[str, Path], which: str) -> Img:
    wcs = WCS(fits.getheader(d["astrometry"] / "wcs.fits"), naxis=2)
    wt = np.load(d["stack"] / "drz_weight.npy", mmap_mode="r")
    wl = np.asarray(wt, np.float32).min(0)
    good = wl > 0.5 * np.median(wl[wl > 0])
    if which == "stack":
        data = np.array(np.load(d["stack"] / "drz_full.npy", mmap_mode="r"), np.float32)
        cover = _erode(good, 40)
    else:
        data = np.array(np.load(d["render_simple"] / "lin.npy", mmap_mode="r"), np.float32)
        valid = np.load(d["render_simple"] / "valid.npy")
        cover = _erode(good & valid, 40)
    return Img(f"ours {which}", data, wcs, cover)


def gaia(d: dict[str, Path], ours: Img) -> dict[str, np.ndarray]:
    """Gaia stars of the field: the deep list (G < 18.5, with BP, RP) and the faint one (to G 20.5, positions through
    our solution), as sky coordinates."""
    deep = np.load(d["catalogs"] / "gaia_deep.npz")
    ra, dec, g = deep["ra"], deep["dec"], deep["g"]
    bp, rp = deep["bp"], deep["rp"]
    f = d["catalogs"] / "gaia_faint.npz"
    if f.exists():
        fa = np.load(f)
        if len(fa["g"]):
            sky = ours.wcs.all_pix2world(fa["x"].astype(float), fa["y"].astype(float), 0)
            ra = np.concatenate([ra, sky[0]])
            dec = np.concatenate([dec, sky[1]])
            g = np.concatenate([g, fa["g"]])
            bp = np.concatenate([bp, np.full(len(fa["g"]), np.nan)])
            rp = np.concatenate([rp, np.full(len(fa["g"]), np.nan)])
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(g)
    return {"ra": ra[ok], "dec": dec[ok], "g": g[ok].astype(float), "bp": bp[ok].astype(float),
            "rp": rp[ok].astype(float)}


# ---------------------------------------------------------------------------------------------- pixel tools

def disk_offsets(r: float) -> tuple[np.ndarray, np.ndarray]:
    R = int(np.ceil(r))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    m = xx ** 2 + yy ** 2 <= r * r
    return yy[m], xx[m]


def ring_offsets(r1: float, r2: float) -> tuple[np.ndarray, np.ndarray]:
    R = int(np.ceil(r2))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d2 = xx ** 2 + yy ** 2
    m = (d2 >= r1 * r1) & (d2 <= r2 * r2)
    return yy[m], xx[m]


def gather(img: np.ndarray, x: np.ndarray, y: np.ndarray, off: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    H, W = img.shape
    xi = np.clip(np.round(x).astype(int)[:, None] + off[1][None], 0, W - 1)
    yi = np.clip(np.round(y).astype(int)[:, None] + off[0][None], 0, H - 1)
    return img[yi, xi]


def star_mask(im: Img, st: dict, fwhm_arcsec: float, max_frac: float = 0.4) -> np.ndarray:
    """Gaia stars masked: radius 2 FWHM + 6" x 10^(-0.2 (G - 13)) (bright stars carry halos); in dense fields the
    faintest are left out until at most max_frac of the image is masked."""
    x, y = im.pix(st["ra"], st["dec"])
    on = (x > -50) & (y > -50) & (x < im.W + 50) & (y < im.H + 50)
    x, y, g = x[on], y[on], st["g"][on]
    r = (2.0 * fwhm_arcsec + np.minimum(6.0 * 10 ** (-0.2 * (g - 13.0)), 300.0)) / im.scale
    area = float(im.cover.sum())
    glim = 21.0
    for glim in (21.0, 20.0, 19.5, 19.0, 18.5, 18.0, 17.5, 17.0, 16.5, 16.0, 15.5, 15.0, 14.5):
        sel = g < glim                                   # coverage estimate (overlaps ignored: an upper bound)
        if np.pi * np.sum(r[sel] ** 2) / area <= 1.3 * max_frac:
            break
    sel = g < glim
    x, y, r = x[sel], y[sel], r[sel]
    m = np.zeros((im.H, im.W), bool)
    small = r <= 12
    # small disks: one distance transform per radius step (points -> distance to the nearest star centre)
    for lo, hi in ((0, 2.5), (2.5, 3.5), (3.5, 5), (5, 7), (7, 9.5), (9.5, 12.01)):
        k = small & (r > lo) & (r <= hi)
        if not k.any():
            continue
        pts = np.ones((im.H, im.W), bool)
        xi, yi = np.round(x[k]).astype(int), np.round(y[k]).astype(int)
        ok = (xi >= 0) & (yi >= 0) & (xi < im.W) & (yi < im.H)
        pts[yi[ok], xi[ok]] = False
        m |= ndi.distance_transform_edt(pts) <= hi
    for i in np.where(~small)[0]:                        # big disks (bright stars): one by one
        R = r[i]
        x0, x1 = int(max(0, x[i] - R)), int(min(im.W, x[i] + R + 1))
        y0, y1 = int(max(0, y[i] - R)), int(min(im.H, y[i] + R + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        m[y0:y1, x0:x1] |= (xx - x[i]) ** 2 + (yy - y[i]) ** 2 <= R * R
    return m


def background(L: np.ndarray, ok: np.ndarray, block: int) -> np.ndarray:
    """Smooth background: median of the unmasked pixels per block, holes filled, bilinear back to full size."""
    H, W = L.shape
    hb, wb = H // block, W // block
    Lc = L[: hb * block, : wb * block].reshape(hb, block, wb, block).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
    oc = ok[: hb * block, : wb * block].reshape(hb, block, wb, block).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
    v = np.where(oc, Lc, np.nan)
    frac = oc.mean(-1)
    med = np.nanmedian(np.where(frac[..., None] > 0.2, v, np.nan), axis=-1)
    bad = ~np.isfinite(med)
    if bad.all():
        return np.full_like(L, np.nanmedian(L))
    if bad.any():
        idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
        med = med[tuple(idx)]
    med = ndi.median_filter(med, 3)
    out = ndi.zoom(med, (H / hb, W / wb), order=1)
    full = np.empty_like(L)
    full[: out.shape[0], : out.shape[1]] = out[:H, :W]
    if out.shape[0] < H:
        full[out.shape[0]:] = full[out.shape[0] - 1]
    if out.shape[1] < W:
        full[:, out.shape[1]:] = full[:, out.shape[1] - 1:out.shape[1]]
    return full


def boxes(L: np.ndarray, ok: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    H, W = L.shape
    hb, wb = H // b, W // b
    Lc = L[: hb * b, : wb * b].reshape(hb, b, wb, b)
    oc = ok[: hb * b, : wb * b].reshape(hb, b, wb, b)
    n = oc.sum((1, 3))
    s = np.where(oc, Lc, 0).sum((1, 3), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / n).astype(np.float32), n / float(b * b)


# ---------------------------------------------------------------------------------------------- stars

def gauss_fit(img: np.ndarray, x: float, y: float, r: int):
    from scipy.optimize import least_squares
    xi, yi = int(round(x)), int(round(y))
    H, W = img.shape
    if xi - r < 0 or yi - r < 0 or xi + r + 1 > W or yi + r + 1 > H:
        return None
    cut = img[yi - r:yi + r + 1, xi - r:xi + r + 1].astype(np.float64)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    b0 = np.median(np.r_[cut[0], cut[-1], cut[:, 0], cut[:, -1]])
    a0 = cut.max() - b0
    if a0 <= 0:
        return None

    def model(p):
        a, x0, y0, sa, sb, th, b = p
        c, s = np.cos(th), np.sin(th)
        u = (xx - x0) * c + (yy - y0) * s
        v = -(xx - x0) * s + (yy - y0) * c
        return a * np.exp(-0.5 * ((u / sa) ** 2 + (v / sb) ** 2)) + b

    p0 = [a0, x - xi, y - yi, r / 4, r / 4, 0.0, b0]
    try:
        res = least_squares(lambda p: (model(p) - cut).ravel(), p0, method="lm", max_nfev=400)
    except Exception:  # noqa: BLE001
        return None
    a, x0, y0, sa, sb, th, b = res.x
    sa, sb = abs(sa), abs(sb)
    if not (a > 0 and 0.3 < sa < r and 0.3 < sb < r and abs(x0) < 3 and abs(y0) < 3):
        return None
    big, small = max(sa, sb), min(sa, sb)
    return {"fwhm_major": 2.3548 * big, "fwhm_minor": 2.3548 * small, "x": xi + x0, "y": yi + y0, "peak": a}


def isolated(st: dict, sel: np.ndarray, radius_deg: float, dmag: float) -> np.ndarray:
    """Stars of sel with no Gaia neighbour brighter than G + dmag within radius."""
    dec0 = np.radians(np.median(st["dec"]))
    pts = np.c_[st["ra"] * np.cos(dec0), st["dec"]]
    # only stars that can spoil one matter: brighter than the faintest candidate + dmag
    gmax = st["g"][sel].max() + dmag if sel.any() else 0
    cand = st["g"] <= gmax
    tree = cKDTree(pts[cand])
    gc = np.r_[st["g"][cand], np.inf]
    ids = np.where(sel)[0]
    out = np.zeros(len(st["g"]), bool)
    for s in range(0, len(ids), 200_000):
        part = ids[s:s + 200_000]
        d, j = tree.query(pts[part], k=24, distance_upper_bound=radius_deg)
        gn = gc[j]                                   # missing neighbours: index len -> inf
        self_ = d < 1e-9                             # the star itself
        bad = (~self_) & (gn <= st["g"][part][:, None] + dmag)
        out[part] = ~bad.any(1)
    return out


def refine_shift(im: Img, st: dict, sel: np.ndarray) -> None:
    """A global offset between the catalogue positions and the image (the solution's zero point), from the centroids
    of bright stars; kept on the image (im.dx, im.dy)."""
    x, y = im.pix(st["ra"][sel], st["dec"][sel])
    ok = im.inside(x, y, 30)
    x, y = x[ok][:400], y[ok][:400]
    if len(x) < 10:
        return
    r = 4
    off = np.mgrid[-r:r + 1, -r:r + 1]
    cut = gather(im.L, x, y, (off[0].ravel(), off[1].ravel())).reshape(len(x), 2 * r + 1, 2 * r + 1)
    cut = cut - np.median(cut.reshape(len(x), -1), axis=1)[:, None, None]
    cut = np.clip(cut, 0, None)
    s = cut.sum((1, 2))
    good = s > 0
    cx = (cut * off[1][None]).sum((1, 2))[good] / s[good]
    cy = (cut * off[0][None]).sum((1, 2))[good] / s[good]
    fx = x[good] - np.round(x[good])
    fy = y[good] - np.round(y[good])
    im.dx += float(np.median(cx - fx))
    im.dy += float(np.median(cy - fy))


def star_shapes(im: Img, st: dict, cand: np.ndarray, frame: Img | None = None) -> dict:
    """FWHM and roundness from elliptical Gaussian fits on unsaturated isolated stars."""
    x, y = im.pix(st["ra"][cand], st["dec"][cand])
    ok = im.inside(x, y, 30)
    idx = np.where(cand)[0][ok]
    x, y = x[ok], y[ok]
    g = st["g"][idx]
    order = np.argsort(g)
    # first pass: a guess of the size from the brightest unsaturated-looking stars
    fits_ = []
    r_guess = 6
    for k in order[: min(len(order), 1500)]:
        f = gauss_fit(im.L, x[k], y[k], r_guess)
        if f is not None:
            fits_.append((k, f))
        if len(fits_) >= 60:
            break
    if not fits_:
        return {}
    fw = np.median([f["fwhm_major"] for _, f in fits_])
    r = int(np.clip(round(2.2 * fw), 4, 25))
    res = []
    for k in order[: min(len(order), 4000)]:
        f = gauss_fit(im.L, x[k], y[k], r)
        if f is not None:
            f["g"] = g[k]
            res.append(f)
        if len(res) >= 1200:
            break
    if len(res) < 20:
        return {}
    peak = np.array([f["peak"] for f in res])
    gg = np.array([f["g"] for f in res])
    # saturated stars: flattened cores give a low peak for their magnitude; keep the ones on the faint stars' line
    zp_peak = np.median((gg + 2.5 * np.log10(np.maximum(peak, 1e-30)))[gg > np.percentile(gg, 50)])
    unsat = (gg + 2.5 * np.log10(np.maximum(peak, 1e-30))) > zp_peak - 0.35
    res = [f for f, u in zip(res, unsat) if u]
    if len(res) < 20:
        return {}
    fma = np.array([f["fwhm_major"] for f in res])
    fmi = np.array([f["fwhm_minor"] for f in res])
    fx = np.array([f["x"] for f in res])
    fy = np.array([f["y"] for f in res])
    # centre and edge of the field in one frame for both images (Siril's, through the sky): the same stars
    fr = frame or im
    if fr is not im:
        ra, dec = im.wcs.all_pix2world(fx - im.dx, fy - im.dy, 0)
        fx, fy = fr.pix(ra, dec)
    rr = np.hypot(fx - fr.W / 2, fy - fr.H / 2) / np.hypot(fr.W / 2, fr.H / 2)
    fwhm = np.sqrt(fma * fmi) * im.scale
    rnd = fmi / fma
    centre, outer = rr < 0.35, rr > 0.7
    out_n = {"centre": int(centre.sum()), "outer": int(outer.sum())}
    out = {"stars_fitted": len(res), "fwhm_arcsec": float(np.median(fwhm)), "roundness": float(np.median(rnd)),
           "fwhm_px": float(np.median(np.sqrt(fma * fmi))), "fit_radius_px": r, "zones": out_n}
    if centre.sum() >= 8:
        out["fwhm_centre_arcsec"] = float(np.median(fwhm[centre]))
        out["roundness_centre"] = float(np.median(rnd[centre]))
    if outer.sum() >= 8:
        out["fwhm_outer_arcsec"] = float(np.median(fwhm[outer]))
        out["roundness_outer"] = float(np.median(rnd[outer]))
    out["max_g_unsaturated_bright"] = float(np.min([f["g"] for f in res]))
    return out


# ---------------------------------------------------------------------------------------------- the measures

def normalise_channels(im: Img, st: dict, sel: np.ndarray, fwhm_px: float, zp_ref: float = 25.0) -> list[float]:
    """Each channel brought to the same photometric scale (a star of magnitude G gives 10^(-0.4 (G - zp_ref)) in its
    aperture), so that the luminance (mean of the channels) mixes them the same way in both programs: the channel
    scalings of a pipeline (Siril's background equalisation, our colour space) then do not weigh on the noise."""
    x, y = im.pix(st["ra"][sel], st["dec"][sel])
    ok = im.inside(x, y, 40)
    x, y, g = x[ok], y[ok], st["g"][sel][ok]
    big = disk_offsets(2.5 * fwhm_px)
    ring = ring_offsets(3.5 * fwhm_px, 5.0 * fwhm_px)
    zps = []
    for c in range(3):
        ch = im.rgb[c]
        loc = np.median(gather(ch, x, y, ring), 1)
        F = gather(ch, x, y, big).sum(1) - loc * len(big[0])
        good = F > 0
        zp = float(np.median(g[good] + 2.5 * np.log10(F[good]))) if good.sum() >= 10 else float("nan")
        zps.append(zp)
        if np.isfinite(zp):
            ch *= np.float32(10 ** (-0.4 * (zp - zp_ref)))
    im.L = im.rgb.mean(0, dtype=np.float32)
    return zps


def photometry(im: Img, st: dict, iso: np.ndarray, fwhm_px: float, mask: np.ndarray, gmin_unsat: float,
               tiles: int = 4, rng=None) -> dict:
    """Zero point, empty-aperture noise, faint-star detection."""
    rng = rng or np.random.default_rng(1)
    B = background(im.L, im.cover & ~mask, max(32, int(8 * fwhm_px)))
    R = im.L - B
    r_det = max(1.5, 1.0 * fwhm_px)
    ap = disk_offsets(r_det)
    big = disk_offsets(2.5 * fwhm_px)
    ring = ring_offsets(3.5 * fwhm_px, 5.0 * fwhm_px)
    x, y = im.pix(st["ra"], st["dec"])
    inside = im.inside(x, y, 40)
    # zero point on bright isolated unsaturated stars
    zsel = iso & inside & (st["g"] >= gmin_unsat + 0.3) & (st["g"] < gmin_unsat + 3.5)
    xs, ys = x[zsel], y[zsel]
    loc = np.median(gather(R, xs, ys, ring), 1)
    F = gather(R, xs, ys, big).sum(1) - loc * len(big[0])
    okf = F > 0
    zp = float(np.median(st["g"][zsel][okf] + 2.5 * np.log10(F[okf]))) if okf.sum() >= 10 else float("nan")
    # empty apertures in the sky (no star, inside the coverage), noise per tile
    free = im.cover & ~mask
    free_e = _erode(free, int(r_det) + 2) if free.any() else free
    yy, xx = np.nonzero(free_e[::4, ::4])
    n = min(len(yy), 6000)
    pick = rng.choice(len(yy), n, replace=False) if len(yy) > n else np.arange(len(yy))
    ex, ey = xx[pick] * 4.0, yy[pick] * 4.0
    ef = gather(R, ex, ey, ap).sum(1)
    sig_all = rstd(ef)
    tx = np.clip((ex / im.W * tiles).astype(int), 0, tiles - 1)
    ty = np.clip((ey / im.H * tiles).astype(int), 0, tiles - 1)
    sig_tile = np.full((tiles, tiles), sig_all)
    for i in range(tiles):
        for j in range(tiles):
            m = (ty == i) & (tx == j)
            if m.sum() >= 40:
                sig_tile[i, j] = rstd(ef[m])
    # faint stars: forced photometry at the Gaia positions, local sky from the smooth background
    fsel = iso & inside
    xf, yf = x[fsel], y[fsel]
    flux = gather(R, xf, yf, ap).sum(1)
    sx = np.clip((xf / im.W * tiles).astype(int), 0, tiles - 1)
    sy = np.clip((yf / im.H * tiles).astype(int), 0, tiles - 1)
    snr = flux / sig_tile[sy, sx]
    return {"zp": zp, "zp_stars": int(okf.sum()), "r_det_px": r_det, "sigma_ap": sig_all,
            "depth5_mag": zp - 2.5 * np.log10(5 * sig_all) if np.isfinite(zp) else float("nan"),
            "snr": snr, "fsel": fsel, "B": B, "R": R}


def box_noise(im: Img, R: np.ndarray, mask: np.ndarray, b: int, b_exact: float) -> dict:
    v, frac = boxes(R, im.cover & ~mask, b)
    good = frac > 0.7
    med = ndi.median_filter(np.where(good, v, np.nan), 9, mode="nearest") if good.any() else v
    d = np.where(good, v - np.nan_to_num(med), np.nan)
    s = rstd(d[good]) * (b / b_exact)          # same angular box: white-noise scaling of the rounding
    return {"sigma_box": s, "box_px": b, "boxes": int(good.sum())}


def target_snr(sc: Img, si: Img, ou: Img, si_mask, ou_mask, bs: int, bo: int, bo_exact: float,
               zp_s: float = float("nan"), zp_o: float = float("nan")) -> dict:
    """SNR on the target. The target and a sky reference are chosen once, on Siril's gradient-removed image (sc, the
    same grid as its stack si), in boxes of bs pixels: the target is the brightest extended region (the largest
    connected parts of the top 5 % of the smoothed box map, at most 5 % of the field), the sky reference the darker
    half of a band around it. On each raw stack: signal = median of the target boxes - median of the sky boxes;
    noise = spread of the target boxes around the median of their 3 x 3 neighbours. Our boxes take the choice of the
    Siril box their centre falls in."""
    out = {}
    vc, fc = boxes(sc.L, sc.cover & ~si_mask, bs)
    good = fc > 0.6
    if good.sum() < 200:
        return {"target_boxes": 0}
    noise_c = rstd((vc - ndi.median_filter(np.nan_to_num(vc), 3, mode="nearest"))[good])
    sky_v = np.nanpercentile(vc[good], 10)
    sky = good & (vc <= sky_v)
    # the target: the brightest extended region of the smoothed map (stars masked), its largest connected parts,
    # at most 5 % of the field
    sm = ndi.median_filter(np.where(good, vc, np.nanmedian(vc[good])), 9, mode="nearest")
    thr = np.percentile(sm[good], 95)
    lab, n = ndi.label((sm >= thr) & good)
    if n == 0:
        return {"target_boxes": 0}
    sizes = ndi.sum_labels(np.ones_like(sm), lab, np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1]
    tgt = np.zeros_like(good)
    for k in order:
        if tgt.sum() >= 0.6 * sizes.sum() or tgt.sum() + sizes[k] > 0.05 * good.sum() and tgt.any():
            break
        tgt |= lab == k + 1
    tgt &= good
    rel = (vc - np.nanmedian(vc[sky])) / noise_c
    if tgt.sum() < 20:
        return {"target_boxes": int(tgt.sum())}
    # the sky next to the target (so that a large-scale gradient left in a raw stack does not count as signal): a
    # band around it, its darker half; the darkest tenth of the field if the band is too small
    r2 = max(8, int(0.04 * max(vc.shape)))
    band = ndi.binary_dilation(tgt, iterations=r2) & ~ndi.binary_dilation(tgt, iterations=3) & good
    if band.sum() >= 50:
        sky = band & (vc <= np.nanmedian(vc[band]))
    out["target_boxes"] = int(tgt.sum())
    out["target_fraction"] = float(tgt.sum() / good.sum())
    out["target_level_sigma"] = float(np.median(rel[tgt]))

    def local_noise(v, t):
        med = ndi.median_filter(np.nan_to_num(v), 3, mode="nearest")
        return rstd((v - med)[t]) / np.sqrt(1 + 1.5 / 9)

    vs, fs = boxes(si.L, si.cover & ~si_mask, bs)
    ts, ks = tgt & (fs > 0.6), sky & (fs > 0.6)
    sig_s, n_s = float(np.median(vs[ts]) - np.median(vs[ks])), local_noise(vs, ts)
    out["snr_siril"] = sig_s / n_s
    # signal and noise as surface brightness (mag/arcsec2), to tell a lost signal from a higher noise
    out["signal_sb_siril"] = float(zp_s - 2.5 * np.log10(max(sig_s, 1e-30) / si.pixarea))
    out["noise_sb_siril"] = float(zp_s - 2.5 * np.log10(n_s / si.pixarea))
    vo, fo = boxes(ou.L, ou.cover & ~ou_mask, bo)
    hb, wb = vo.shape
    cy, cx = (np.mgrid[0:hb, 0:wb] + 0.5) * bo
    ra, dec = ou.wcs.all_pix2world((cx - ou.dx).ravel(), (cy - ou.dy).ravel(), 0)
    xs, ys = si.pix(ra, dec)
    jx, jy = np.floor(xs / bs).astype(int), np.floor(ys / bs).astype(int)
    inb = (jx >= 0) & (jy >= 0) & (jx < tgt.shape[1]) & (jy < tgt.shape[0])
    to, ko = np.zeros(hb * wb, bool), np.zeros(hb * wb, bool)
    to[inb], ko[inb] = tgt[jy[inb], jx[inb]], sky[jy[inb], jx[inb]]
    to, ko = to.reshape(hb, wb) & (fo > 0.6), ko.reshape(hb, wb) & (fo > 0.6)
    if to.sum() >= 20 and ko.sum() >= 20:
        sig_o, n_o = float(np.median(vo[to]) - np.median(vo[ko])), local_noise(vo, to) * (bo / bo_exact)
        out["snr_ours"] = sig_o / n_o
        out["signal_sb_ours"] = float(zp_o - 2.5 * np.log10(max(sig_o, 1e-30) / ou.pixarea))
        out["noise_sb_ours"] = float(zp_o - 2.5 * np.log10(n_o / ou.pixarea))
    out["target_boxes_ours"] = int(to.sum())
    return out


def residuals(im: Img, mask: np.ndarray, fwhm_px: float) -> dict:
    """High-pass spots and trails outside the stars."""
    hp = im.L - ndi.median_filter(im.L, size=max(5, int(2 * fwhm_px) | 1))
    ok = im.cover & ~mask
    sig = rstd(hp[ok][:: max(1, ok.sum() // 2_000_000)])
    z = np.where(ok, hp / sig, 0)
    out = {"sigma_hp": sig}
    spots = {}
    # spots on the 3 x 3 mean of the high-pass, over its own noise: a hot pixel or a cosmic ray left by the rejection
    # spreads over a few pixels after registration; single-pixel noise (heavier-tailed on a drizzled grid) does not
    hs = ndi.uniform_filter(np.where(ok, hp, 0), 3)
    zs3 = np.where(ok, hs / rstd(hs[ok][:: max(1, ok.sum() // 2_000_000)]), 0)
    for name, cond in (("hot", zs3 > 6), ("cold", zs3 < -6)):
        lab, n = ndi.label(cond)
        if n:
            sizes = ndi.sum_labels(np.ones_like(z), lab, np.arange(1, n + 1))
            com = np.array(ndi.center_of_mass(np.ones_like(z), lab, np.arange(1, n + 1)))
            small = sizes <= max(9, (2 * fwhm_px) ** 2)
            spots[name] = com[small][:, ::-1]                # x, y
        else:
            spots[name] = np.zeros((0, 2))
    # trails: the smoothed high-pass above 3 sigma, long thin components
    zs = ndi.uniform_filter(z, 3) * 3.0
    lab, n = ndi.label(zs > 3.5)
    trails = []
    if n:
        objs = ndi.find_objects(lab)
        for i, sl in enumerate(objs, 1):
            hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if max(hh, ww) < 12 * fwhm_px:
                continue
            yy, xx = np.nonzero(lab[sl] == i)
            if len(yy) < 30:
                continue
            c = np.cov(np.vstack([xx, yy]))
            ev = np.sort(np.linalg.eigvalsh(c))
            if ev[0] <= 0 or ev[1] / ev[0] > 25:
                length = 4 * np.sqrt(ev[1])
                trails.append({"x": float(sl[1].start + xx.mean()), "y": float(sl[0].start + yy.mean()),
                               "length_arcmin": float(length * im.scale / 60)})
    area_deg2 = ok.sum() * im.pixarea / 3600 ** 2
    out.update({"spots": spots, "trails": trails, "area_deg2": float(area_deg2)})
    return out


def flatness(im: Img, mask: np.ndarray, zp: float, sky_sb: float, tiles: int = 8) -> dict:
    """Floor of tiles (median of the darkest fifth of the unmasked pixels), spread p90 - p10, as a share of the sky
    brightness of the raw stack (both in this image's photometric units through the zero point)."""
    ok = im.cover & ~mask
    H, W = im.L.shape
    floors = []
    for i in range(tiles):
        for j in range(tiles):
            sl = (slice(i * H // tiles, (i + 1) * H // tiles), slice(j * W // tiles, (j + 1) * W // tiles))
            v = im.L[sl][ok[sl]]
            if v.size < 500:
                continue
            q = np.percentile(v, 20)
            floors.append(float(np.median(v[v <= q])))
    floors = np.array(floors)
    if len(floors) < 16:
        return {}
    spread = float(np.percentile(floors, 90) - np.percentile(floors, 10))
    out = {"tiles": int(len(floors)), "spread": spread}
    if np.isfinite(zp) and spread > 0:
        sb = zp - 2.5 * np.log10(spread / im.pixarea)
        out["spread_sb"] = float(sb)
        out["spread_pct_of_sky"] = float(100 * 10 ** (-0.4 * (sb - sky_sb)))
    return out


def colours(im: Img, st: dict, sel: np.ndarray, fwhm_px: float) -> dict:
    """Star colours against Gaia BP-RP."""
    x, y = im.pix(st["ra"][sel], st["dec"][sel])
    ok = im.inside(x, y, 40)
    idx = np.where(sel)[0][ok]
    x, y = x[ok], y[ok]
    ap = disk_offsets(1.5 * fwhm_px)
    ring = ring_offsets(3.5 * fwhm_px, 5.0 * fwhm_px)
    F = []
    for c in range(3):
        ch = im.rgb[c]
        loc = np.median(gather(ch, x, y, ring), 1)
        s = gather(ch, x, y, ap).sum(1) - loc * len(ap[0])
        noise = np.array([rstd(v) for v in gather(ch, x, y, ring)]) * np.sqrt(len(ap[0]))
        F.append((s, noise))
    good = np.all([(f > 30 * n) & (f > 0) for f, n in F], axis=0)
    if good.sum() < 20:
        return {"stars": int(good.sum())}
    R, G, B = (F[c][0][good] for c in range(3))
    col = st["bp"][idx][good] - st["rp"][idx][good]
    fin = np.isfinite(col) & (col > -0.3) & (col < 3.0)
    col, R, G, B = col[fin], R[fin], G[fin], B[fin]
    out = {"stars": int(len(col))}
    for name, c in (("b_r", -2.5 * np.log10(B / R)), ("g_r", -2.5 * np.log10(G / R))):
        a, b0 = np.polyfit(col, c, 1)
        for _ in range(3):
            res = c - (a * col + b0)
            keep = np.abs(res) < 4 * rstd(res)
            a, b0 = np.polyfit(col[keep], c[keep], 1)
        res = c - (a * col + b0)
        from scipy.stats import spearmanr
        out[name] = {"scatter_mag": rstd(res), "slope": float(a), "white_at_bp_rp": float(-b0 / a) if a else None,
                     "scatter_bp_rp": rstd(res) / abs(a) if a else None,
                     "spearman": float(spearmanr(col, c).statistic)}
    return out


def cross(a: np.ndarray, b: np.ndarray, r: float) -> int:
    """How many points of a have no point of b within r (both in the same coordinates)."""
    if len(a) == 0:
        return 0
    if len(b) == 0:
        return len(a)
    d, _ = cKDTree(b).query(a, distance_upper_bound=r)
    return int(np.sum(~np.isfinite(d)))


def to_sky(im: Img, xy: np.ndarray, dec0: float) -> np.ndarray:
    """Pixel positions to local sky coordinates in arcsec (RA scaled by cos dec0)."""
    if len(xy) == 0:
        return np.zeros((0, 2))
    ra, dec = im.wcs.all_pix2world(xy[:, 0] - im.dx, xy[:, 1] - im.dy, 0)
    return np.c_[ra * np.cos(np.radians(dec0)) * 3600, dec * 3600]


# ---------------------------------------------------------------------------------------------- main

def measure(case: str, level: str) -> dict:
    t0 = time.time()
    d = ours_dirs(case, level)
    res: dict = {"case": case, "level": level}
    so = load_ours(d, "stack")
    ss = load_siril(case, level, "stack_linear")
    st = gaia(d, so)
    so.cover, ss.cover = common_cover(so, ss), common_cover(ss, so)    # every measure on the sky both cover
    res["gaia_stars"] = int(len(st["g"]))
    print(f"loaded: ours {so.W}x{so.H} at {so.scale:.2f}\"/px, Siril {ss.W}x{ss.H} at {ss.scale:.2f}\"/px; "
          f"{len(st['g'])} Gaia stars ({time.time() - t0:.0f} s)", flush=True)
    both = {}
    for im in (so, ss):
        x, y = im.pix(st["ra"], st["dec"])
        both[im.name] = im.inside(x, y, 40)
    common = both[so.name] & both[ss.name]
    # bright stars for the zero point of the astrometry
    bright = common & (st["g"] < 14) & (st["g"] > 9)
    for im in (so, ss):
        refine_shift(im, st, bright)
    res["astrometric_offset_px"] = {"ours": [so.dx, so.dy], "siril": [ss.dx, ss.dy]}
    # stars: isolated, measured in both
    iso_shape = isolated(st, common & (st["g"] < 16.5) & (st["g"] > 8), 40 / 3600, 3.0)
    shapes = {}
    for key, im in (("ours", so), ("siril", ss)):
        shapes[key] = star_shapes(im, st, iso_shape, ss)
        print(f"{key}: stars {shapes[key]} ({time.time() - t0:.0f} s)", flush=True)
    res["stars"] = shapes
    fwhm_as = max(shapes["ours"].get("fwhm_arcsec", 5), shapes["siril"].get("fwhm_arcsec", 5))
    masks = {key: star_mask(im, st, fwhm_as) for key, im in (("ours", so), ("siril", ss))}
    print(f"star masks ({time.time() - t0:.0f} s)", flush=True)
    gmin = max(shapes["ours"].get("max_g_unsaturated_bright", 10), shapes["siril"].get("max_g_unsaturated_bright", 10))
    # faint stars: isolated within 3 FWHM (no Gaia star brighter than G + 1.5)
    iso_f = isolated(st, common & (st["g"] > gmin), 3 * fwhm_as / 3600, 1.5)
    iso_z = isolated(st, common & (st["g"] > gmin + 0.3) & (st["g"] < gmin + 3.5), 30 / 3600, 2.5)
    res["channel_zp"] = {key: normalise_channels(im, st, iso_z, shapes[key].get("fwhm_px", 3.0))
                         for key, im in (("ours", so), ("siril", ss))}
    phot = {}
    for key, im in (("ours", so), ("siril", ss)):
        phot[key] = photometry(im, st, iso_f, shapes[key].get("fwhm_px", 3.0), masks[key], gmin)
    # the same stars in both: the ones measured in both
    common_f = phot["ours"]["fsel"] & phot["siril"]["fsel"]
    det = {}
    g_all = st["g"][common_f]
    for key in ("ours", "siril"):
        snr_full = np.full(len(st["g"]), np.nan)
        snr_full[phot[key]["fsel"]] = phot[key]["snr"]
        s = snr_full[common_f]
        found = s >= 5
        bins = np.arange(np.floor(g_all.min() * 4) / 4, 21.01, 0.25)
        frac, centres = [], []
        for lo in bins[:-1]:
            m = (g_all >= lo) & (g_all < lo + 0.25)
            if m.sum() >= 20:
                frac.append(found[m].mean())
                centres.append(lo + 0.125)
        frac, centres = np.array(frac), np.array(centres)
        g50 = float("nan")
        below = np.where(frac < 0.5)[0]
        if len(below) and below[0] > 0:
            k = below[0]
            g50 = float(np.interp(0.5, [frac[k], frac[k - 1]], [centres[k], centres[k - 1]]))
        det[key] = {"found": int(found.sum()), "of": int(len(s)), "g50": g50}
    res["faint_stars"] = det
    # noise in boxes of the same angular size
    bs = 4
    bo_exact = bs * ss.scale / so.scale
    bo = max(2, int(round(bo_exact)))
    noise = {}
    for key, im, b, bex in (("ours", so, bo, bo_exact), ("siril", ss, bs, bs)):
        bn = box_noise(im, phot[key]["R"], masks[key], b, bex)
        zp = phot[key]["zp"]
        bn["sb_mag_arcsec2"] = float(zp - 2.5 * np.log10(bn["sigma_box"] / im.pixarea))
        # the same at a 4 times larger scale (fine-scale smoothing by interpolation weighs less there)
        b4, bex4 = max(2, int(round(4 * bex))), 4 * bex
        bn["sb_large_mag_arcsec2"] = float(zp - 2.5 * np.log10(
            box_noise(im, phot[key]["R"], masks[key], b4, bex4)["sigma_box"] / im.pixarea))
        sky = float(np.median(phot[key]["B"][im.cover & ~masks[key]]))
        bn["sky_sb"] = float(zp - 2.5 * np.log10(sky / im.pixarea)) if sky > 0 else float("nan")
        bn.update({k: phot[key][k] for k in ("zp", "zp_stars", "r_det_px", "depth5_mag")})
        noise[key] = bn
    res["noise"] = noise
    res["box_arcsec"] = bs * ss.scale
    sc = load_siril(case, level, "color_linear")
    res["target"] = target_snr(sc, ss, so, masks["siril"], masks["ours"], bs, bo, bo_exact, noise["siril"]["zp"],
                               noise["ours"]["zp"])
    del sc
    print(f"noise {noise} target {res['target']} ({time.time() - t0:.0f} s)", flush=True)
    # residual spots and trails
    rs = {key: residuals(im, masks[key], shapes[key].get("fwhm_px", 3.0)) for key, im in (("ours", so), ("siril", ss))}
    match_r = 3 * fwhm_as
    dec0 = float(np.median(st["dec"]))
    gsky = np.c_[st["ra"] * np.cos(np.radians(dec0)) * 3600, st["dec"] * 3600]
    gtree = cKDTree(gsky)
    art = {}
    for key, other in (("ours", "siril"), ("siril", "ours")):
        im, imo = (so, ss) if key == "ours" else (ss, so)
        a = {"area_deg2": rs[key]["area_deg2"], "trails": len(rs[key]["trails"]),
             "trail_length_arcmin": float(sum(t["length_arcmin"] for t in rs[key]["trails"]))}
        for kind in ("hot", "cold"):
            mine = to_sky(im, rs[key]["spots"][kind], dec0)
            theirs = to_sky(imo, np.vstack([rs[other]["spots"]["hot"], rs[other]["spots"]["cold"]]), dec0)
            # a spot on a Gaia star (to G 20.5) is a star, not a defect
            rmatch = max(1.5 * fwhm_as, 2 * im.scale)
            on_star = (gtree.query(mine, distance_upper_bound=rmatch)[0] < np.inf) if len(mine) else np.zeros(0, bool)
            # a spot of one image counts only where the other image covers the sky too
            xo, yo = imo.pix(*im.wcs.all_pix2world(rs[key]["spots"][kind][:, 0] - im.dx,
                                                   rs[key]["spots"][kind][:, 1] - im.dy, 0)) \
                if len(mine) else (np.zeros(0), np.zeros(0))
            inside = imo.inside(xo, yo, 40) if len(mine) else np.zeros(0, bool)
            a[f"{kind}_spots"] = int(len(mine))
            a[f"{kind}_off_stars"] = int((~on_star).sum())
            a[f"{kind}_only_here"] = cross(mine[inside & ~on_star], theirs, match_r)
        art[key] = a
    res["residuals"] = art
    print(f"residuals {art} ({time.time() - t0:.0f} s)", flush=True)
    del phot
    # gradient-removed colour images
    so2 = load_ours(d, "lin")
    ss2 = load_siril(case, level, "color_linear")
    so2.dx, so2.dy, ss2.dx, ss2.dy = so.dx, so.dy, ss.dx, ss.dy
    del so, ss
    flat, col = {}, {}
    iso_c = isolated(st, common & np.isfinite(st["bp"]) & (st["g"] > gmin + 0.3) & (st["g"] < gmin + 5), 30 / 3600, 2.5)
    for key, im in (("ours", so2), ("siril", ss2)):
        fw = shapes[key].get("fwhm_px", 3.0)
        col[key] = colours(im, st, iso_c, fw)            # the colours as each program delivers them
        normalise_channels(im, st, iso_z, fw)            # then the flatness on the common photometric scale
        ph = photometry(im, st, iso_f, fw, masks[key], gmin)
        flat[key] = flatness(im, masks[key], ph["zp"], noise[key]["sky_sb"])
        print(f"{key}: flat {flat[key]} colours {col[key]} ({time.time() - t0:.0f} s)", flush=True)
    res["flatness"] = flat
    res["colours"] = col
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=list(C.CASES))
    ap.add_argument("level", choices=list(C.LEVELS))
    ap.add_argument("--no-sheet", action="store_true")
    a = ap.parse_args()
    out = C.BENCH / a.case / f"measures_{a.level}.json"
    r = measure(a.case, a.level)
    out.write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
    print(f"written {out}")
    if not a.no_sheet:
        import sheet
        sheet.make(a.case, a.level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
