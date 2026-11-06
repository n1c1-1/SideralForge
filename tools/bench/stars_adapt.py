"""Star removal adapted to one field (issue #40): does a few minutes of fine-tuning on the field itself help?

For a bench case: the shipped network's starless image S0 of the field; the field's star shape measured on its star
layer O - S0 (an elliptical Moffat fitted on its isolated unsaturated stars: FWHM, beta, ellipticity and angle), its
star colours (the flux of each colour in those stars) and its saturation level (the top of each colour); then pairs
built from the field itself: S0 + synthetic stars of that shape -> S0 (the truth under each star is exact), the
generator and the loss of stars_train.py (a Moffat core, sometimes an r^-3 aureole and spikes, a power law of
brightnesses, saturation, photon noise; L1 on the network's fixed asinh stretch). A copy of the shipped network is
fine-tuned on them (Adam, a small learning rate, constant after a short warm-up), and applied to the real field after
N steps: S1.

    python tools/bench/stars_adapt.py --cases rosette,iris,m51 --steps 300,1000 --device cuda:1

Files: F:/astro_essais/stars_adapt/<case>/{S0.fits, S1_<N>.fits} (the units of stars_compare/<case>/input.fits),
F:/astro_essais/stars_adapt/<case>_sheet.jpg, results.json. Measures: stars_compare.py's (star light left by Gaia
magnitude, the background off the stars, the texture where the bright stars were), plus S1 against S0 off the stars.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "bench"))

OUT = Path("F:/astro_essais/stars_adapt")
LOOK = Path("F:/astro_essais/A_REGARDER")
PATCH, MARGIN = 256, 128


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- the field's stars
def moffat2d(p, yy, xx):
    A, x0, y0, ax, ay, th, beta = p
    c, s = np.cos(th), np.sin(th)
    u, v = c * (xx - x0) + s * (yy - y0), -s * (xx - x0) + c * (yy - y0)
    return A * (1 + (u / ax) ** 2 + (v / ay) ** 2) ** (-beta)


def star_shape(L3: np.ndarray, valid: np.ndarray, top: float) -> dict:
    """The field's star profile on its star layer (3, H, W) in noise units: an elliptical Moffat fitted on its
    isolated, unsaturated, well-exposed stars; their colours (flux per colour over the mean)."""
    from scipy import ndimage
    from scipy.optimize import least_squares
    L = L3.mean(0)
    H, W = L.shape
    mx = ndimage.maximum_filter(L, 15)
    ok = valid & ndimage.binary_erosion(valid, iterations=20)
    pk = (L == mx) & (L > 20) & (L < 0.3 * top) & ok
    pk[:20], pk[-20:], pk[:, :20], pk[:, -20:] = False, False, False, False
    ys, xs = np.nonzero(pk)
    order = np.argsort(-L[ys, xs])
    R = 12
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    fits_, cols = [], []
    for y, x in zip(ys[order], xs[order]):
        c = L[y - R:y + R + 1, x - R:x + R + 1].astype(np.float64)
        if c.shape != yy.shape:
            continue
        pkv = L[y, x]
        if (c > 0.3 * pkv).sum() >= 4 * max(1, (c > 0.5 * pkv).sum()):       # something else bright around it
            continue
        f = L3[:, y - 5:y + 6, x - 5:x + 6].reshape(3, -1).sum(1)
        if f.min() > 0:
            cols.append(f / f.mean())
        if len(fits_) >= 250:
            if len(cols) >= 1500:
                break
            continue
        try:
            r = least_squares(lambda p: (moffat2d(p, yy, xx) - c).ravel(),
                              [pkv, 0, 0, 2.0, 2.0, 0.0, 2.5],
                              bounds=([0, -3, -3, 0.3, 0.3, -np.pi, 1.01], [np.inf, 3, 3, 30, 30, np.pi, 15]))
        except Exception:  # noqa: BLE001
            continue
        A, _, _, ax, ay, th, beta = r.x
        if not r.success or beta > 14:
            continue
        k = 2 * np.sqrt(2 ** (1 / beta) - 1)
        fa, fb = ax * k, ay * k
        if fa < fb:
            fa, fb, th = fb, fa, th + np.pi / 2
        fits_.append((np.sqrt(fa * fb), beta, 1 - fb / fa, (th + np.pi / 2) % np.pi - np.pi / 2))
    F = np.array(fits_)
    if len(F) < 10:
        raise RuntimeError(f"too few stars to measure the shape ({len(F)})")
    # the angle: a circular median (angles of period pi)
    ang = 0.5 * np.arctan2(np.median(np.sin(2 * F[:, 3])), np.median(np.cos(2 * F[:, 3])))
    return {"fwhm": float(np.median(F[:, 0])), "fwhm_iqr": [float(v) for v in np.percentile(F[:, 0], [25, 75])],
            "beta": float(np.median(F[:, 1])), "beta_iqr": [float(v) for v in np.percentile(F[:, 1], [25, 75])],
            "ellipticity": float(np.median(F[:, 2])), "angle": float(ang), "stars_fitted": len(F),
            "colours": np.array(cols, np.float32), "stars_coloured": len(cols)}


def field_stars(n, shape, gen, dev):
    """stars_train.synthetic_stars with the field's profile: FWHM, beta, ellipticity and angle around the measured
    ones, colours drawn from the field's stars; the aureole, spikes, brightness law and photon noise as in training."""
    import torch
    G = PATCH + 2 * MARGIN
    yy, xx = torch.meshgrid(torch.arange(G, device=dev, dtype=torch.float32) - G // 2,
                            torch.arange(G, device=dev, dtype=torch.float32) - G // 2, indexing="ij")
    out = torch.zeros(n, 3, G, G, device=dev)
    psfs = torch.zeros(n, 3, G, G, device=dev)
    cols = torch.from_numpy(shape["colours"]).to(dev)

    def U(a, b, size=()):
        return a + (b - a) * torch.rand(size, generator=gen, device=dev)

    for i in range(n):
        fw = shape["fwhm"] * float(U(0.9, 1.1))
        beta = float(np.clip(shape["beta"] * float(U(0.85, 1.15)), 1.3, 10))
        alpha = fw / (2 * np.sqrt(2 ** (1 / beta) - 1))
        e = float(np.clip(shape["ellipticity"] + float(U(-0.04, 0.04)), 0, 0.5))
        th = shape["angle"] + float(U(-0.15, 0.15))
        c, s = np.cos(th), np.sin(th)
        u, v = c * xx + s * yy, -s * xx + c * yy
        r2 = (u / (1 - e / 2)) ** 2 + (v / (1 + e / 2)) ** 2
        core = (1 + r2 / alpha ** 2) ** (-beta)
        core = core / core.sum()
        fh = float(U(0, 0.08)) if float(U(0, 1)) < 0.6 else 0.0
        rh = fw * float(torch.exp(U(np.log(3.0), np.log(25.0))))
        fs = float(U(0.002, 0.02)) if float(U(0, 1)) < 0.2 else 0.0
        if fs:
            ts = float(U(0, np.pi / 2))
            spikes = torch.zeros_like(core)
            for k in range(2):
                a = ts + k * np.pi / 2
                along = np.cos(a) * xx + np.sin(a) * yy
                perp = -np.sin(a) * xx + np.cos(a) * yy
                spikes += torch.exp(-perp ** 2 / (2 * (0.35 * fw) ** 2)) / (1 + (along / fw) ** 2)
            spikes = spikes / spikes.sum()
        for ch in range(3):
            p = (1 - fh - fs) * core
            if fh:
                rc = rh * (1 + float(U(-0.25, 0.25)))
                halo = (1 + (xx ** 2 + yy ** 2) / rc ** 2) ** (-1.5)
                p = p + fh * halo / halo.sum()
            if fs:
                p = p + fs * spikes
            psfs[i, ch] = torch.fft.ifftshift(p)
        nstar = int(float(torch.exp(U(0.0, np.log(400.0)))) * (G / PATCH) ** 2)
        if nstar == 0:
            continue
        a_pow = float(U(0.7, 1.2))
        peak1 = (beta - 1) / (np.pi * alpha ** 2)
        fmin = float(U(0.3, 1.0)) / peak1
        flux = fmin * U(0, 1, (nstar,)).clamp(min=1e-9) ** (-1 / a_pow)
        flux = flux.clamp(max=50 * shape["top"] / peak1)
        pick = torch.randint(len(cols), (nstar,), generator=gen, device=dev)
        col = cols[pick].T                                            # (3, nstar), mean 1 over the colours
        px, py = U(0, G - 1.001, (nstar,)), U(0, G - 1.001, (nstar,))
        x0, y0 = px.floor().long(), py.floor().long()
        fx, fy = px - x0, py - y0
        img = out[i].view(3, -1)
        for dy, dx, w in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
            idx = ((y0 + dy) * G + (x0 + dx)).clamp(max=G * G - 1)
            for ch in range(3):
                img[ch].index_add_(0, idx, flux * col[ch] * w)
    stars = torch.fft.irfft2(torch.fft.rfft2(out) * torch.fft.rfft2(psfs), s=(G, G))
    stars = stars[:, :, MARGIN:MARGIN + PATCH, MARGIN:MARGIN + PATCH].clamp(min=0)
    k = U(0, 0.15, (n, 1, 1, 1))
    return stars + k * stars.sqrt() * torch.randn(stars.shape, generator=gen, device=dev)


class FieldTiles:
    """Random tiles of S0 (rotated, flipped), mostly inside the covered area."""

    def __init__(self, S0, valid, seed, lossm=None):
        self.S, self.v = S0, valid
        self.lm = valid if lossm is None else lossm
        self.rng = np.random.default_rng(seed)

    def get(self, n):
        H, W = self.v.shape
        t, m = [], []
        while len(t) < n:
            y, x = int(self.rng.integers(0, H - PATCH)), int(self.rng.integers(0, W - PATCH))
            if self.v[y:y + PATCH, x:x + PATCH].mean() < 0.9:
                continue
            mm = self.lm[y:y + PATCH, x:x + PATCH]
            a = self.S[:, y:y + PATCH, x:x + PATCH]
            k = int(self.rng.integers(4))
            a, mm = np.rot90(a, k, (1, 2)), np.rot90(mm, k)
            if self.rng.random() < 0.5:
                a, mm = a[:, :, ::-1], mm[:, ::-1]
            t.append(np.ascontiguousarray(a))
            m.append(np.ascontiguousarray(mm))
        return np.stack(t), np.stack(m)[:, None]


# ---------------------------------------------------------------------------------------------- adaptation
def catalogue(case: str):
    """The case's Gaia stars (x, y, G): stars_compare.load_case's rule (the deepest catalogue of the source workflow),
    or, when the workflow's catalogue stage is not current, the deepest catalogue on disk."""
    import stars_compare as sc
    api = sc._api()
    p = api.open_project(sc.BENCH / case / "ours_express")
    src = next(w for w in p.workflows if w.name == sc.SOURCE)
    cat = api.stage_outputs(p, src.id, "catalogs")
    root = Path(cat.dir).parent if cat is not None else sc.BENCH / case / "ours_express" / "store" / "catalogs"
    deep = max(root.glob("*/gaia_deep.npz"), key=lambda f: f.stat().st_size)
    z = np.load(deep)
    return np.asarray(z["x"], float), np.asarray(z["y"], float), np.asarray(z["g"], float)


def adapt(case: str, steps: list[int], device: str, lr: float, batch: int, seed: int, tag: str = "") -> dict:
    """tag "m": the loss ignores the discs of the field's catalogued stars, where S0 is not a sure truth (what the
    shipped network left of the real stars would be taught as background)."""
    import torch
    from astropy.io import fits
    from stars_train import bench_case

    from astro_gpu.stages.star_removal import apply, fwd, network
    dev = torch.device(device)
    torch.backends.cudnn.benchmark = True
    d = OUT / case
    d.mkdir(parents=True, exist_ok=True)
    O, _, (bk, unit, scale, nm) = bench_case(case)
    O = np.ascontiguousarray(O, np.float32)
    valid = nm > 0
    info = {"shape": list(O.shape), "unit": unit}

    def save(S, name):
        S = S.copy()
        S[:, ~valid] = O[:, ~valid]
        fits.PrimaryHDU(((S * unit + bk) / scale).astype(np.float32)).writeto(d / f"{name}.fits", overwrite=True)
        return S

    net = network(dev)
    t = time.time()
    S0 = save(apply(net, O, dev), "S0")
    info["apply_s"] = round(time.time() - t, 1)
    log(f"{case}: S0 in {info['apply_s']} s")

    top = np.array([float(O[c][valid].max()) for c in range(3)], np.float32)
    near = [float((O[c][valid] > 0.98 * top[c]).sum()) for c in range(3)]
    shape = star_shape(O - S0, valid, float(top.mean()))
    shape["top"] = float(top.mean())
    info["shape"] = {k: v for k, v in shape.items() if k != "colours"}
    info["shape"]["colour_spread"] = [round(float(v), 3) for v in shape["colours"].std(0)]
    info["saturation"] = {"top": top.tolist(), "pixels_within_2pct": near}
    log(f"{case}: shape {info['shape']}, saturation {info['saturation']}")

    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    lossm = valid
    if tag == "m":
        x, y, g = catalogue(case)
        known = np.zeros(valid.shape, np.uint8)
        for cx, cy, gg in zip(x, y, g):
            rad = int(round(1.5 * shape["fwhm"] + max(0.0, 15 - gg) * 3))
            cv2.circle(known, (int(round(cx)), int(round(cy))), max(rad, 3), 1, -1)
        lossm = valid & (known == 0)
        info["loss_ignored_pct"] = round(100 * float((known[valid] > 0).mean()), 1)
        log(f"{case}: loss ignores {info['loss_ignored_pct']} % of the field (catalogued stars)")
    tiles = FieldTiles(S0, valid, seed, lossm)
    sat = torch.from_numpy(top).to(dev)[None, :, None, None]

    def make(n):
        t_, m_ = tiles.get(n)
        t_, m_ = torch.from_numpy(t_).to(dev), torch.from_numpy(m_).to(dev).float()
        x_ = torch.minimum(t_ + field_stars(n, shape, gen, dev), torch.maximum(t_, sat.expand_as(t_)))
        return x_, t_, m_

    vx, vt, _ = make(16)                                    # fixed validation tiles: synthetic, exact truth

    def val():
        net.eval()
        err_all, err_star = [], []
        with torch.no_grad():
            for j in range(0, 16, 4):
                x, tt = vx[j:j + 4], vt[j:j + 4]
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    pred = net(fwd(x)).float()
                err = (pred - fwd(tt)).abs()
                star = ((fwd(x) - fwd(tt)).abs().max(1, keepdim=True)[0] > 0.01).expand_as(err)
                err_all.append(float(err.mean()))
                err_star.append(float(err[star].mean()) if star.any() else 0.0)
        return round(float(np.mean(err_all)), 5), round(float(np.mean(err_star)), 5)

    hist = [{"step": 0, "val": val()}]
    log(f"{case}: validation before {hist[0]['val']}")
    net.recompute = True
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99))
    warm, t0, run, it = 30, time.time(), None, 0
    train_s, apply_s = 0.0, {}
    for target in sorted(steps):
        while it < target:
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (it + 1) / warm)
            x, tt, m = make(batch)
            net.train()
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                pred = net(fwd(x)).float()
            loss = ((pred - fwd(tt)).abs() * m).sum() / (m.sum() * 3 + 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            opt.step()
            run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
            it += 1
            if it % 100 == 0:
                log(f"{case}: step {it}, loss {run:.5f}, {(time.time() - t0) / it:.2f} s/step")
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        train_s = time.time() - t0
        v = val()
        hist.append({"step": it, "val": v, "train_seconds": round(train_s, 1), "loss": round(run, 5)})
        net.eval()
        t = time.time()
        save(apply(net, O, dev), f"S1{tag}_{it}")
        apply_s[it] = round(time.time() - t, 1)
        log(f"{case}: {it} steps in {train_s:.0f} s, validation {v}, applied in {apply_s[it]} s")
        t0 += time.time() - t                                   # the time of the application is not training time
    info.update({"history": hist, "apply_adapted_s": apply_s, "lr": lr, "batch": batch, "device": device})
    del net, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return info


# ---------------------------------------------------------------------------------------------- measures and sheet
def measure(case: str, steps: list[int]) -> dict:
    import stars_compare as sc
    from stars_compare import off_stars, read_fits, star_light_left, star_mask, texture
    api = sc._api()
    O, _S, stars, wcs, fwhm, meta = sc.load_case(api, case)
    del _S
    if stars is None:   # the workflow's catalogue stage is not current: its deepest catalogue on disk (same rule)
        stars = catalogue(case)
    d = OUT / case
    S = {"S0": read_fits(d / "S0.fits")}
    files = sorted(d.glob("S1*_*.fits"), key=lambda f: (f.stem.split("_")[0], int(f.stem.split("_")[1])))
    for f in files:
        S[f.stem] = read_fits(f)
    sigma = sc.noise_of(O)
    mask = star_mask(O, stars, fwhm)
    res = {"fwhm_px_register": fwhm, "by_method": {}}
    for k, Sm in S.items():
        r = {"off_stars": off_stars(O, Sm, mask, sigma), "texture": texture(Sm, stars, fwhm),
             "star_light_left": star_light_left(O, Sm, stars, fwhm, sigma) if stars else {}}
        if k != "S0":
            r["vs_S0_off_stars"] = off_stars(S["S0"], Sm, mask, sigma)
            dd = sc.lum(Sm) - sc.lum(S["S0"])
            keep = ~mask
            r["vs_S0_abs_diff_sigma"] = {"off_stars_p50": round(float(np.median(np.abs(dd[keep]))) / sigma, 3),
                                         "off_stars_p99": round(float(np.percentile(np.abs(dd[keep]), 99)) / sigma, 2),
                                         "on_stars_p99": round(float(np.percentile(np.abs(dd[mask]), 99)) / sigma, 2)}
        r["bright_cores"] = bright_cores(Sm, stars, fwhm, sigma)
        res["by_method"][k] = r
    res["paired"] = paired(O, S, stars, fwhm, sigma)
    sheet(case, O, S, stars, sigma)
    return res


def _star_selection(O, stars, fwhm, sigma):
    """stars_compare.star_light_left's selection: aperture, flux in O, the nebula fifth, the stars well above noise."""
    from stars_compare import aperture_flux, lum
    x, y, g = stars
    r = max(3.0, 1.5 * fwhm)
    fo = aperture_flux(O, x, y, r)
    big = cv2.blur(lum(O), (65, 65))
    H, W = big.shape
    bgl = big[np.clip(np.round(y).astype(int), 0, H - 1), np.clip(np.round(x).astype(int), 0, W - 1)]
    neb = bgl >= np.nanpercentile(bgl, 80)
    ok = np.isfinite(fo) & (fo > 10 * sigma * np.sqrt(np.pi) * r)
    return r, fo, neb, ok


def paired(O, S, stars, fwhm, sigma) -> dict:
    """Star by star, the light left in S1 minus the one in S0 (points of the star's flux), median and a 95 % bootstrap
    interval over the stars: what the change is, measured on the same stars."""
    from stars_compare import aperture_flux
    x, y, g = stars
    r, fo, neb, ok = _star_selection(O, stars, fwhm, sigma)
    f = {k: aperture_flux(v, x, y, r) / fo for k, v in S.items()}
    rng = np.random.default_rng(0)
    out = {}
    for k in S:
        if k == "S0":
            continue
        for (lo, hi), where in (((16, 18.5), "sky"), ((16, 18.5), "nebula"), ((14, 16), "sky")):
            m = ok & ((~neb) if where == "sky" else neb) & (g >= lo) & (g < hi) & np.isfinite(f[k]) & np.isfinite(f["S0"])
            if m.sum() < 20:
                continue
            dlt = 100 * (f[k][m] - f["S0"][m])
            boot = [np.median(rng.choice(dlt, len(dlt))) for _ in range(300)]
            out[f"{k} G{lo}-{hi} {where}"] = {"stars": int(m.sum()), "median_change_pts": round(float(np.median(dlt)), 2),
                                             "ci95": [round(float(v), 2) for v in np.percentile(boot, [2.5, 97.5])],
                                             "left_S0": round(100 * float(np.median(f["S0"][m])), 2),
                                             "left_S1": round(100 * float(np.median(f[k][m])), 2)}
    return out


def bright_cores(Sm, stars, fwhm, sigma) -> dict:
    """Holes where the bright stars were: for the stars brighter than G 11, the lowest point of the smoothed starless
    luminance in the star's disc, minus the median of a ring around it, in noise units (a hole: strongly negative)."""
    from stars_compare import lum
    x, y, g = stars
    L = cv2.GaussianBlur(lum(Sm), (0, 0), 1.0)
    H, W = L.shape
    vals = []
    for cx, cy, gg in zip(x[g < 11], y[g < 11], g[g < 11]):
        rad = int(max(4, 1.5 * fwhm + (11 - gg) * 2))
        cx, cy = int(round(cx)), int(round(cy))
        if cx - 2 * rad < 0 or cy - 2 * rad < 0 or cx + 2 * rad >= W or cy + 2 * rad >= H:
            continue
        p = L[cy - 2 * rad:cy + 2 * rad + 1, cx - 2 * rad:cx + 2 * rad + 1]
        yy, xx = np.mgrid[-2 * rad:2 * rad + 1, -2 * rad:2 * rad + 1]
        rr = np.hypot(xx, yy)
        vals.append((p[rr <= rad].min() - np.median(p[(rr > 1.3 * rad) & (rr <= 2 * rad)])) / sigma)
    if not vals:
        return {}
    q = np.percentile(vals, [5, 50])
    return {"stars": len(vals), "min_dip_p5_sigma": round(float(q[0]), 2), "min_dip_median_sigma": round(float(q[1]), 2)}


def sheet(case, O, S, stars, sigma):
    """Two rows (the brightest star, a plain field of faint stars, zoomed 2x), columns: the original, S0, each S1 and
    the difference S1 (last) - S0 (grey = 0)."""
    L = O.mean(0)
    H, W = L.shape
    bk = np.median(O.reshape(3, -1)[:, ::7], axis=1)[:, None, None]
    s = sigma * 3
    top = np.arcsinh(float(np.percentile(O - bk, 99.95)) / s)

    def show(img):
        v = np.arcsinh(np.clip((img - bk) / s, -1, None)) / top
        return (np.clip(v, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)

    def show_diff(dd):
        v = 0.5 + 0.5 * np.arcsinh(dd / sigma) / np.arcsinh(20.0)
        g = (np.clip(v, 0, 1) * 255).astype(np.uint8)
        return np.repeat(g[:, :, None], 3, 2)

    big = cv2.blur(L, (65, 65))
    rows = []
    C = 360
    x, y, g = stars
    inside = (x > C) & (x < W - C) & (y > C) & (y < H - C)
    i = int(np.argmin(np.where(inside, g, np.inf)))
    boxes = [("brightest star", int(y[i]) - C // 2, int(x[i]) - C // 2, C, 1)]
    # the faint field: the 180 px square with the most G 16-18.5 stars on a dark sky and no star brighter than G 12
    c = 180
    best, bb = -1, None
    sky = big <= np.percentile(big, 50)
    for yy in range(c, H - 2 * c, c // 2):
        for xx in range(c, W - 2 * c, c // 2):
            sel = (x >= xx) & (x < xx + c) & (y >= yy) & (y < yy + c)
            if not sky[yy + c // 2, xx + c // 2] or (g[sel] < 12).any():
                continue
            nf = int(((g[sel] >= 16) & (g[sel] < 18.5)).sum())
            if nf > best:
                best, bb = nf, (yy, xx)
    if bb is None:
        bb = (H // 2, W // 2)
    boxes.append(("faint stars x2", bb[0], bb[1], c, 2))
    names = list(S)
    for label, y0, x0, size, zoom in boxes:
        y0, x0 = int(np.clip(y0, 0, H - size)), int(np.clip(x0, 0, W - size))
        sl = (slice(None), slice(y0, y0 + size), slice(x0, x0 + size))
        bk = np.median(O[sl].reshape(3, -1), axis=1)[:, None, None]          # this crop's own sky: neutral
        top = np.arcsinh(max(float(np.percentile(O[sl] - bk, 99.9)), 5 * s) / s)
        tiles = [("original", show(O[sl]))] + [(k, show(S[k][sl])) for k in names]
        tiles.append((f"{names[-1]} - S0", show_diff((S[names[-1]][sl] - S["S0"][sl]).mean(0))))
        row = []
        for k, t in tiles:
            t = np.ascontiguousarray(cv2.resize(t, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
                                     if zoom > 1 else t)
            cv2.putText(t, k, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            row.append(t)
        rows.append(np.hstack(row))
    f = OUT / f"{case}_sheet.jpg"
    cv2.imwrite(str(f), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    LOOK.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, LOOK / f"40_{case}.jpg")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default="rosette,iris,m51")
    ap.add_argument("--steps", default="300,1000")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--tag", default="", help='"m": the loss ignores the catalogued stars (files S1m_<N>)')
    a = ap.parse_args(argv)
    steps = [int(v) for v in a.steps.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    res_f = OUT / "results.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    for case in a.cases.split(","):
        t = time.time()
        r = res.get(case, {})
        if not a.measure_only:
            r["adapt" + a.tag] = adapt(case, steps, a.device, a.lr, a.batch, a.seed, a.tag)
        r["measures"] = measure(case, steps)
        res[case] = r
        res_f.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        log(f"{case}: done in {(time.time() - t) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
