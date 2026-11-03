"""Optics: lateral chromatic aberration, saturation magnitude of the setup, empirical variable chromatic PSF."""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares

from ..core.determinism import deterministic
from ..core.params import ADVANCED, P, StageParams
from ..core.stage import Context, Stage
from .gradient import stack_image

G_SAT_REF = 11.25      # saturation magnitude of the reference setup the default thresholds were tuned on
OS, RAD, ST = 3, 11, 9  # PSF oversampling, core radius (px), half-size of fitting stamps
NG = 2 * RAD * OS + 1
COL0 = 1.3
NB = 14


class OpticsParams(StageParams):
    lateral: bool = P(True, label="Correct lateral colour",
                      help="Measures the shift between the red, green and blue images of the stars across the field "
                           "(lateral chromatic aberration of the optics) and realigns red and blue onto green.")
    lateral_grid: int = P(7, label="Lateral colour grid", level=ADVANCED, ge=2, le=15,
                          help="Number of zones across the long side of the image in which the colour shift is measured.")
    psf_iterations: int = P(300, label="PSF fit iterations", level=ADVANCED, ge=50, le=5000, step=50,
                            help="Conjugate-gradient steps of the empirical PSF model, per colour channel and per "
                                 "robust round (three rounds).")
    psf_stars_per_cell: int = P(60, label="PSF stars per zone", level=ADVANCED, ge=5, le=500,
                                help="At most this many stars per zone of a 10x7 grid, so dense areas do not "
                                     "dominate the PSF model.")


class Optics(Stage):
    name = "optics"
    title = "Optics: colour shift and PSF"
    description = ("Characterises the optics from the stars themselves. First the lateral colour: red and blue star "
                   "images are slightly shifted from green towards the edges, and are realigned. Then an empirical "
                   "point spread function (PSF) is fitted on thousands of isolated Gaia stars: its shape varies "
                   "across the field and with the colour of the star, and its faint wings are measured on bright "
                   "stars. This real PSF is what the star photometry, the star removal and the star rendering use.")
    deps = ("denoise", "stack", "catalogs")
    Params = OpticsParams
    version = "2"          # 2: the PSF fit posed with one answer (backgrounds from an annulus, linear least squares,
                           # deterministic sums): the same data gave 8 % or 30 % flux scatter from run to run

    def run(self, ctx: Context, p: OpticsParams) -> None:
        dev = ctx.device
        n2n = ctx.load("denoise", "n2n_avg.npy", mmap=True)
        H, W = n2n.shape[1:]
        gr = max(2, int(round(p.lateral_grid * min(H, W) / max(H, W))))
        grid = measure_lateral(n2n, gr if H < W else p.lateral_grid, p.lateral_grid if H < W else gr) \
            if p.lateral else np.zeros((2, 2, 4))
        ctx.save_json("lateral.json", grid.tolist())
        sh = np.abs(grid[..., [0, 2]]).max()
        ctx.metric("lateral_shift_max_px", round(float(sh), 2))
        ctx.log(f"lateral colour shift up to {sh:.2f} px (red/blue vs green)")
        for h in ("A", "B", "full"):
            ctx.progress(0.05 + 0.05 * ("A", "B", "full").index(h), f"realigning colours of {h}")
            img = stack_image(ctx, f"drz_{h}.npy")
            ctx.save(f"lat_{h}.npy", apply_lateral(img, grid))
        del n2n
        img = ctx.load(self.name, "lat_full.npy").astype(np.float32)
        d = np.load(ctx.input("catalogs", "gaia_deep.npz"))
        x, y, g = d["x"], d["y"], d["g"]
        col = np.nan_to_num(d["bp"] - d["rp"], nan=COL0)
        g_sat = saturation_magnitude(img[1], x, y, g)
        dm = float(np.round((g_sat - G_SAT_REF) * 2) / 2) if abs(g_sat - G_SAT_REF) > 1.0 else 0.0
        ctx.metric("g_sat", round(g_sat, 2))
        ctx.metric("mag_offset", dm)
        ctx.log(f"stars reach half saturation at G = {g_sat:.2f}; magnitude thresholds shifted by {dm:+.1f}")
        sat = np.zeros(img.shape[1:], bool)
        for c in range(3):
            sat |= img[c] > 0.7 * np.percentile(img[c], 99.999)
        sel = select_psf_stars(x, y, g, sat, W, H, 9.5 + dm, 14.0 + dm, p.psf_stars_per_cell)
        ctx.log(f"{len(sel)} isolated stars for the PSF (G {g[sel].min():.1f}-{g[sel].max():.1f})")
        out = {"E": [], "frac": [], "flux": []}
        for c in range(3):
            def prog(f, msg, c=c):
                ctx.progress(0.15 + 0.25 * (c + f), f"PSF channel {'RGB'[c]}: {msg}")
            E, fl, frac = fit_channel(img[c], x[sel], y[sel], col[sel], W, H, p.psf_iterations, dev, prog)
            out["E"].append(E.cpu())
            out["flux"].append(fl)
            out["frac"].append(frac)
            ctx.log(f"PSF {'RGB'[c]}: residual {100 * frac:.1f} % of the star flux")
        rr, prof, nw = wings(img, x, y, g, W, H, 6.0 + dm, 9.0 + dm)
        ctx.log(f"PSF wings measured on {nw} bright stars")
        torch.save({"E": torch.stack(out["E"]), "OS": OS, "RAD": RAD, "COL0": COL0, "sel": sel,
                    "flux": np.array(out["flux"]), "frac": out["frac"], "wing_r": rr, "wing_prof": prof,
                    "width": W, "height": H}, ctx.out("psf_model.pt"))
        ctx.metric("psf_residual_pct", [round(100 * f, 1) for f in out["frac"]])
        ctx.preview("psf", psf_preview(torch.stack(out["E"])))


# ---------------------------------------------------------------------- lateral colour
def _centroid_fit(pch):
    yy, xx = np.mgrid[-6:7, -6:7]

    def f(t):
        return (t[0] * np.exp(-((xx - t[1]) ** 2 + (yy - t[2]) ** 2) / (2 * t[3] ** 2)) + t[4] - pch).ravel()
    if not np.isfinite(pch).all() or pch.max() <= 0:     # no star in this colour (a frame edge, a gap): no fit
        return 0.0, 0.0, False
    r = least_squares(f, [pch.max(), 0, 0, 1.5, 0], bounds=([0, -3, -3, 0.4, -np.inf], [np.inf, 3, 3, 6, np.inf]))
    return r.x[1], r.x[2], r.success


def measure_lateral(d, GR, GC):
    import sep
    H, W = d.shape[1:]
    chans = [np.array(d[c], dtype=np.float32, copy=True) for c in range(3)]      # writable (sep needs it)
    subs = [g - sep.Background(g, bw=64, bh=64).back() for g in chans]
    bkg = sep.Background(chans[1], bw=64, bh=64)
    sep.set_extract_pixstack(3000000)
    o = sep.extract(subs[1], 40, err=bkg.globalrms, minarea=6)
    o = o[(o["flag"] == 0) & (o["peak"] < 0.25 * np.percentile(o["peak"], 99.9))]
    cells: dict = {}
    for s in o:
        cx, cy = int(round(s["x"])), int(round(s["y"]))
        if cx < 9 or cy < 9 or cx > W - 10 or cy > H - 10:
            continue
        i, j = min(int(cy / H * GR), GR - 1), min(int(cx / W * GC), GC - 1)
        if len(cells.get((i, j), [])) >= 120:
            continue
        pos = []
        for c in range(3):
            x0, y0, ok = _centroid_fit(subs[c][cy - 6:cy + 7, cx - 6:cx + 7])
            if not ok:
                break
            pos.append((x0, y0))
        if len(pos) == 3:
            cells.setdefault((i, j), []).append([pos[0][0] - pos[1][0], pos[0][1] - pos[1][1],
                                                 pos[2][0] - pos[1][0], pos[2][1] - pos[1][1]])
    grid = np.zeros((GR, GC, 4))
    for (i, j), v in cells.items():
        grid[i, j] = np.median(np.array(v), 0)
    return grid


def apply_lateral(img, grid):
    grid = np.asarray(grid, np.float32)
    if not np.any(grid):
        return img
    H, W = img.shape[1:]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = img.copy()
    for c, k in ((0, 0), (2, 2)):
        dx = cv2.resize(grid[..., k], (W, H), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(grid[..., k + 1], (W, H), interpolation=cv2.INTER_CUBIC)
        out[c] = cv2.remap(img[c], xx + dx, yy + dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return out


# ---------------------------------------------------------------------- saturation magnitude
def saturation_magnitude(g1, x, y, g):
    """Gaia G at which the median star peak (green) falls to half of the saturation level of the stack."""
    H, W = g1.shape
    lvl = 0.7 * np.percentile(g1[::4, ::4], 99.999)
    bg = cv2.GaussianBlur(cv2.medianBlur(cv2.resize(g1, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA), 5),
                          (0, 0), 2)
    bg = cv2.resize(bg, (W, H))
    ok = (x > 5) & (x < W - 5) & (y > 5) & (y < H - 5) & (g < 17)
    xi, yi = np.round(x[ok]).astype(int), np.round(y[ok]).astype(int)
    pk = np.max(np.stack([g1[yi + a, xi + b] for a in (-1, 0, 1) for b in (-1, 0, 1)]), 0) - bg[yi, xi]
    gg = g[ok]
    for b in np.arange(3, 17, 0.25):
        m = (gg >= b) & (gg < b + 0.25)
        if m.sum() > 5 and np.median(pk[m]) < 0.5 * lvl:
            return float(b + 0.125)
    return G_SAT_REF


# ---------------------------------------------------------------------- PSF model
def basis(u, v, col):
    return torch.stack([torch.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3,
                        col, col * u, col * v, col * col], -1)


def sample_E(E, dx, dy):
    """E (NB, NG, NG); dx, dy (N, S, S) offsets in output pixels -> (N, S, S, NB), bilinear."""
    grid = torch.stack([dx / RAD, dy / RAD], -1).view(1, -1, 1, 2)
    out = F.grid_sample(E[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[0, :, :, 0].T.reshape(*dx.shape, -1)


def select_psf_stars(x, y, g, sat, W, H, gmin, gmax, per_cell):
    from scipy.spatial import cKDTree
    t = cKDTree(np.c_[x, y])
    ok = (g > gmin) & (g < gmax) & (x > 40) & (x < W - 40) & (y > 40) & (y < H - 40)
    keep = []
    for i in np.where(ok)[0]:
        if any(j != i and g[j] < g[i] + 3.0 for j in t.query_ball_point([x[i], y[i]], 8.0)):
            continue
        xi, yi = int(round(x[i])), int(round(y[i]))
        if sat[yi - ST:yi + ST + 1, xi - ST:xi + ST + 1].any():
            continue
        keep.append(i)
    keep = np.array(keep, int)
    out, cnt = [], {}
    for i in keep[np.argsort(g[keep])]:
        c = int(np.clip(x[i] / W * 10, 0, 9)) * 7 + int(np.clip(y[i] / H * 7, 0, 6))
        if cnt.get(c, 0) < per_cell:
            out.append(i)
            cnt[c] = cnt.get(c, 0) + 1
    return np.array(out, int)


def _lower_median(a, dim):
    """Median along dim (the lower of the two middle values), by sorting: deterministic on the GPU."""
    return a.sort(dim).values.select(dim, (a.shape[dim] - 1) // 2)


def fit_channel(img_c, xs, ys, cols, W, H, iters, dev, progress=None, rounds=3):
    """Field- and colour-dependent PSF model of one colour, and the flux of each star.

    Posed so that it has one answer, reached the same way every time:
      - each star's background is a plane fitted (with clipping) to an annulus just outside its stamp. Fitted together
        with the PSF, it could trade places with a flat part of the PSF model, and the fluxes with it: the same data
        gave star fluxes scattered by 8 % or by 30 % against their colours depending on the run;
      - each stamp minus its background is divided by its own sum, and the PSF model E (NB field/colour terms on a
        grid oversampled OS times) is the regularised weighted least-squares fit of all the normalised stamps at once:
        linear in E, so a single solution whatever the start or the order of the colours; solved by preconditioned
        conjugate gradients with sums in a fixed order (bit-identical runs on the same GPU);
      - stars that disagree with the model (neighbours, defects) weigh less in two further rounds;
      - each star's flux is then fitted (weighted least squares) with the model's shape at its place and colour, over
        its stamp, above the annulus background.
    Returns E (NB, NG, NG), the fluxes and the median residual (fraction of the star flux)."""
    with deterministic(strict=True):
        return _fit_channel(img_c, xs, ys, cols, W, H, iters, dev, progress, rounds)


def _fit_channel(img_c, xs, ys, cols, W, H, iters, dev, progress, rounds):
    N = len(xs)
    xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
    oy, ox = np.mgrid[-ST:ST + 1, -ST:ST + 1]
    S = 2 * ST + 1
    py = (yi[:, None, None] + oy[None]).astype(np.int64)
    px = (xi[:, None, None] + ox[None]).astype(np.int64)
    data = torch.from_numpy(img_c[py, px].astype(np.float32)).to(dev)
    dx = torch.from_numpy((px - xs[:, None, None]).astype(np.float32)).to(dev)
    dy = torch.from_numpy((py - ys[:, None, None]).astype(np.float32)).to(dev)
    # background: a plane on the annulus ST+1 .. ST+5 px, three rounds of 2.5-sigma clipping
    ra0, ra1 = ST + 1, ST + 5
    ay, ax = np.mgrid[-ra1:ra1 + 1, -ra1:ra1 + 1]
    ann_m = (np.hypot(ax, ay) >= ra0) & (np.hypot(ax, ay) <= ra1)
    ay, ax = ay[ann_m], ax[ann_m]
    Hh, Ww = img_c.shape
    ayy = np.clip(yi[:, None] + ay[None], 0, Hh - 1)
    axx = np.clip(xi[:, None] + ax[None], 0, Ww - 1)
    ann = torch.from_numpy(img_c[ayy, axx].astype(np.float32)).to(dev)
    adx = torch.from_numpy((axx - xs[:, None]).astype(np.float32)).to(dev)
    ady = torch.from_numpy((ayy - ys[:, None]).astype(np.float32)).to(dev)
    A3 = torch.stack([torch.ones_like(adx), adx, ady], -1)
    w = torch.ones_like(ann)
    for _ in range(3):
        AtA = (A3[:, :, :, None] * A3[:, :, None, :] * w[:, :, None, None]).sum(1) + 1e-6 * torch.eye(3, device=dev)
        Atb = (A3 * (w * ann)[:, :, None]).sum(1)
        q = torch.linalg.solve(AtA, Atb[..., None])[..., 0]
        res = ann - (A3 * q[:, None, :]).sum(-1)
        sig_s = 1.4826 * _lower_median(res.abs(), 1) + 1e-6
        w = (res.abs() < 2.5 * sig_s[:, None]).float()
    sub = data - (q[:, 0, None, None] + q[:, 1, None, None] * dx + q[:, 2, None, None] * dy)
    F0 = sub.sum((1, 2))
    good = F0 > 10 * sig_s * S
    D = sub / F0.clamp(min=1e-6)[:, None, None]
    wpix = (1.0 / ((sig_s / F0.clamp(min=1e-6))[:, None, None] ** 2 + (0.03 * D.clamp(min=0)) ** 2)) * good[:, None, None]
    # bilinear sampling of the PSF grid (-RAD..RAD, NG nodes) at every stamp pixel: fixed indices and weights
    h = 2.0 * RAD / (NG - 1)
    gx, gy = (dx + RAD) / h, (dy + RAD) / h
    x0, y0 = gx.floor(), gy.floor()
    fx, fy = gx - x0, gy - y0
    idx, wts = [], []
    for ddy, ddx, wv in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        xx, yy = x0 + ddx, y0 + ddy
        ok = (xx >= 0) & (xx < NG) & (yy >= 0) & (yy < NG)
        idx.append((yy.clamp(0, NG - 1) * NG + xx.clamp(0, NG - 1)).long().reshape(-1))
        wts.append((wv * ok).reshape(-1))
    u = torch.from_numpy((xs / (W / 2) - 1).astype(np.float32)).to(dev)
    v = torch.from_numpy((ys / (H / 2) - 1).astype(np.float32)).to(dev)
    cc = torch.from_numpy((cols - COL0).astype(np.float32)).to(dev)
    Bp = basis(u, v, cc)[:, None, :].expand(N, S * S, NB).reshape(-1, NB)

    def fwd(E):                                                   # (NB, NG*NG) -> one value per stamp pixel
        Et = E.T
        return (sum(wts[c][:, None] * Et[idx[c]] for c in range(4)) * Bp).sum(-1)

    def adj(R, sq=False):                                         # transpose of fwd (sq: of its square, for the diagonal)
        out = torch.zeros(NG * NG, NB, device=dev)
        for c in range(4):
            out.index_add_(0, idx[c], ((wts[c] ** 2 if sq else wts[c]) * R)[:, None] * (Bp * Bp if sq else Bp))
        return out.T

    def lap(E):
        G = E.reshape(NB, NG, NG)
        L = torch.zeros_like(G)
        L[:, 1:-1, 1:-1] = G[:, 2:, 1:-1] + G[:, :-2, 1:-1] + G[:, 1:-1, 2:] + G[:, 1:-1, :-2] - 4 * G[:, 1:-1, 1:-1]
        return L

    def lap_t(L):
        G = torch.zeros_like(L)
        Li = L[:, 1:-1, 1:-1]
        G[:, 2:, 1:-1] += Li
        G[:, :-2, 1:-1] += Li
        G[:, 1:-1, 2:] += Li
        G[:, 1:-1, :-2] += Li
        G[:, 1:-1, 1:-1] -= 4 * Li
        return G

    Dv = D.reshape(-1)
    npts = float(good.sum()) * S * S
    peak = float(D[good].amax((1, 2)).median())
    mu = 50.0 / peak ** 2 / float((NG - 2) ** 2 * NB)             # the smoothness weight of the former fit
    star_w = torch.ones(N, device=dev)
    E = torch.zeros(NB, NG * NG, device=dev)
    t0 = time.time()
    for rnd in range(rounds):
        Wv = (wpix * star_w[:, None, None]).reshape(-1) / npts

        dg = adj(Wv, sq=True)
        # parts of the grid no stamp reaches stay at zero (a tiny ridge): otherwise they continued the PSF smoothly
        # out to the grid's edge, a pedestal the later stages would read as wings ending in a cliff
        ridge = 1e-3 * float(dg[dg > 0].median())

        def normal(E_, Wv=Wv, ridge=ridge):
            return adj(Wv * fwd(E_)) + mu * lap_t(lap(E_)).reshape(NB, -1) + ridge * E_

        b = adj(Wv * Dv)
        Minv = 1.0 / (dg + 20.0 * mu + ridge).clamp(min=1e-30)
        r = b - normal(E)
        z = Minv * r
        p_ = z.clone()
        rz = rz0 = float((r * z).sum())
        for it in range(iters):
            Ap = normal(p_)
            alpha = rz / max(float((p_ * Ap).sum()), 1e-30)
            E = E + alpha * p_
            r = r - alpha * Ap
            z = Minv * r
            rz_new = float((r * z).sum())
            if rz_new < 1e-12 * rz0:
                break
            p_ = z + (rz_new / rz) * p_
            rz = rz_new
            if progress and (it + 1) % 100 == 0:
                progress((rnd + (it + 1) / iters) / rounds, f"round {rnd + 1}/{rounds}, step {it + 1}/{iters} "
                                                           f"({time.time() - t0:.0f} s)")
        res = (Dv - fwd(E)).reshape(N, S, S)
        chi = (wpix * res ** 2).mean((1, 2))
        star_w = torch.clamp(3 * float(chi[good].median()) / chi.clamp(min=1e-12), max=1.0)
    P_ = fwd(E).reshape(N, S, S)
    Pk = P_ / P_.sum((1, 2), keepdim=True).clamp(min=1e-9)
    fl = F0.clone()
    for _ in range(2):
        wd = 1.0 / (sig_s[:, None, None] ** 2 + (0.03 * fl.clamp(min=0)[:, None, None] * Pk) ** 2)
        fl = (wd * Pk * sub).sum((1, 2)) / (wd * Pk * Pk).sum((1, 2)).clamp(min=1e-12)
    frac = (sub - fl[:, None, None] * Pk).abs().sum((1, 2)) / fl.clamp(min=1e-6)
    return E.reshape(NB, NG, NG), fl.cpu().numpy(), float(frac[good].median())


def wings(img, x, y, g, W, H, gmin, gmax):
    """Median normalised radial profile (per channel) of bright isolated stars, r = 0..60 px."""
    rr = np.arange(0, 61)
    prof = np.zeros((3, len(rr)))
    sel = np.where((g > gmin) & (g < gmax) & (x > 80) & (x < W - 80) & (y > 80) & (y < H - 80))[0][:400]
    yy, xx = np.mgrid[-70:71, -70:71]
    R = np.hypot(xx, yy)
    stacks = [[] for _ in range(3)]
    for i in sel:
        xi, yi = int(round(x[i])), int(round(y[i]))
        flux_ref = 10 ** (-0.4 * (g[i] - 10.0))
        for c in range(3):
            cut = img[c, yi - 70:yi + 71, xi - 70:xi + 71]
            bgv = np.median(cut[R > 62])
            stacks[c].append(np.array([np.median(cut[(R >= r - 0.5) & (R < r + 0.5)]) - bgv for r in rr]) / flux_ref)
    for c in range(3):
        if stacks[c]:
            prof[c] = np.median(np.array(stacks[c]), 0)
    return rr, prof, len(sel)


def psf_preview(E):
    """Central PSF of each channel (log stretch), side by side."""
    dev = E.device
    oy, ox = torch.meshgrid(torch.linspace(-8, 8, 161, device=dev), torch.linspace(-8, 8, 161, device=dev),
                            indexing="ij")
    Bs = basis(torch.zeros(1, device=dev), torch.zeros(1, device=dev), torch.zeros(1, device=dev))
    tiles = []
    for c in range(3):
        P_ = (sample_E(E[c], ox[None], oy[None]) * Bs[:, None, None, :]).sum(-1)[0].cpu().numpy()
        P_ = np.clip(P_ / P_.max(), 1e-4, 1)
        v = (np.log10(P_) + 4) / 4
        rgb = np.zeros((3, *v.shape))
        rgb[c] = v
        rgb += 0.25 * v[None]
        tiles.append(np.clip(rgb, 0, 1))
    return np.concatenate(tiles, 2)


def lateral_from_json(path):
    return np.array(json.loads(open(path).read()))
