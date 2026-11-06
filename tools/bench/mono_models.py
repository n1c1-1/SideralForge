"""Bench: how the denoise and star networks (trained on colour stacks) behave on monochrome images.

    python tools/bench/mono_models.py OUT_FOLDER BENCH_PROJECT [BENCH_PROJECT ...] [--colour BENCH_PROJECT]

For each workflow of the bench projects that ran Denoise and Split stars & background (tools/bench/mono_path.py), read
only from their stores:

* denoise: the background noise before and after; the flux, peak and width of the stars (bright ones and faint ones,
  aperture photometry with a local background) after against before: a denoiser that eats signal lowers them; the
  network's three outputs before their mean (a grey image given as three equal channels): how far apart it moves them,
  in noise units;
* stars (when the workflow has Split stars & background): the stars left in the starless image (their peak left, by brightness), the dark holes where stars were (the
  starless image under the local background), how much the image away from the stars changed (noise units), and the
  network's three outputs before their mean;
* a sheet per workflow: crops (the brightest star, the densest star field, the brightest nebula) of the stack, the
  denoised image, the starless image and the stars, on one stretch.

``--colour``: colour bench projects measured the same way, for reference. Results in OUT_FOLDER/mono_models.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.journal import Journal  # noqa: E402
from astro_gpu.core.stage import Context  # noqa: E402
from astro_gpu.core.store import Keys, Store, StoreProject  # noqa: E402
from astro_gpu.stages import denoise as dn  # noqa: E402
from astro_gpu.stages import star_removal as sr  # noqa: E402
from astro_gpu.stages.gradient import stack_image  # noqa: E402

import torch  # noqa: E402

DEV = torch.device("cuda:0")


def context(p, k, wid, stage, st):
    sp = StoreProject(k, p.workflow(wid), stage, devices={"main": DEV})
    return Context(sp, st, Journal(sp.logs), DEV)


def own(p, k, wid, stage, name):
    """A stage's own result (its context reads only its inputs)."""
    return np.load(Store(p.root).dir(stage, k.key(wid, stage)) / name).astype(np.float32)


def mad(a):
    v = a[np.isfinite(a)][::7]
    m = np.median(v)
    return float(m), float(1.4826 * np.median(np.abs(v - m)))


def stars_of(img, sig, bkg, lo, hi, n=300, iso=10):
    """Isolated, compact local maxima with a peak (above the background) between lo and hi noise sigmas: (y, x)
    arrays. Compact: the difference of two blurs (1 and 4 px) holds at least half the peak, which leaves out the
    nebula's knots."""
    sm = cv2.GaussianBlur(img, (0, 0), 1.0)
    dog = sm - cv2.GaussianBlur(img, (0, 0), 4.0)
    mx = cv2.dilate(sm, np.ones((2 * iso + 1, 2 * iso + 1), np.uint8))
    pk = (sm == mx) & (sm - bkg > lo * sig) & (sm - bkg < hi * sig) & (dog > 0.5 * (sm - bkg))
    pk[:24], pk[-24:], pk[:, :24], pk[:, -24:] = False, False, False, False
    y, x = np.nonzero(pk)
    order = np.argsort(-(sm[y, x]))
    return y[order][:n], x[order][:n]


def phot(img, y, x, r=4, r1=8, r2=12):
    """Per star: flux in a disc of radius r over the annulus' median (not clipped: no noise bias), its peak over it
    (on a 1 px blur: little noise bias), its width (second moments), its lowest point near the centre."""
    sm = cv2.GaussianBlur(img, (0, 0), 1.0)
    yy, xx = np.mgrid[-r2:r2 + 1, -r2:r2 + 1]
    d = np.hypot(yy, xx)
    disc, ann = d <= r, (d >= r1) & (d <= r2)
    out = []
    for cy, cx in zip(y, x):
        c = img[cy - r2:cy + r2 + 1, cx - r2:cx + r2 + 1]
        b = float(np.median(c[ann]))
        s = np.clip(c - b, 0, None) * disc
        f = float(((c - b) * disc).sum())
        w = float(np.sqrt((s * (yy ** 2 + xx ** 2)).sum() / max(float(s.sum()), 1e-12) / 2))
        out.append((f, float(sm[cy, cx] - b), w, float((c - b)[d <= 2].min()), b))
    return np.array(out).reshape(-1, 5)


def ratio(a, b):
    ok = a > 0
    return round(float(np.median(b[ok] / a[ok])), 3) if ok.any() else None


def spread(raw, sig):
    """How far the network moved three equal channels apart: the per-pixel spread of its outputs, in noise units."""
    s = raw.std(0)[::3, ::3]
    return {"median": round(float(np.median(s)) / sig, 3), "p99": round(float(np.percentile(s, 99)) / sig, 3)}


def measure_denoise(p, k, wid):
    ctx = context(p, k, wid, "denoise", dn.Denoise())
    full = stack_image(ctx)
    dF = own(p, k, wid, "denoise", "n2n_full.npy")
    A = stack_image(ctx, "drz_A.npy")
    bk, sg, nm = dn.norm_params(A, ctx.load("stack", "drz_weight.npy").astype(np.float32))
    net = dn.pretrained(DEV)
    saved = Store(p.root).dir("denoise", k.key(wid, "denoise")) / "n2n_model.pt"
    net.load_state_dict(torch.load(saved, map_location=DEV, weights_only=False)["net"])   # the network the run used
    net.eval()                                                                          # (adapted: its own)
    raw = dn.apply_net(net, full, (nm / np.sqrt(2)).astype(np.float32), bk, sg, DEV)
    g = full[1]
    bkg, sig = mad(g)
    res = {"noise_before": round(dn._noise(g), 4), "noise_after": round(dn._noise(dF[1]), 4)}
    res["noise_reduction"] = round(res["noise_before"] / max(res["noise_after"], 1e-12), 2)
    for name, lo, hi in (("bright", 50, 1e9), ("faint", 5, 15)):
        y, x = stars_of(g, sig, bkg, lo, hi)
        a, b = phot(g, y, x), phot(dF[1], y, x)
        res[f"{name}_stars"] = {"n": int(len(y)), "flux": ratio(a[:, 0], b[:, 0]), "peak": ratio(a[:, 1], b[:, 1]),
                                "width": ratio(a[:, 2], b[:, 2])}
    res["channel_spread"] = spread(raw, max(res["noise_after"], 1e-12) * 1.0)
    res["half_consistency"] = halves(ctx, net, A, nm, bk, sg, full)
    return res, full, dF


def halves(ctx, net, A, nm, bk, sg, full):
    """Invented dots. The two half-stacks' noise is independent: a real star is in both, a dot made of noise in one
    only. The compact dots of the denoised half-stack A above 1.5 sigma of a raw half-stack's noise, and the share of
    them the denoised half-stack B also holds (above 0.75 of that sigma); the same on the two raw half-stacks blurred
    by 1.5 px (no network), for reference. One scale for every image: the raw half-stack's noise. On the faint half of
    the image (the sky, blurred) and on the brighter half."""
    B = stack_image(ctx, "drz_B.npy")
    level = cv2.GaussianBlur(full[1], (0, 0), 8)
    dA = dn.apply_net(net, A, nm, bk, sg, DEV).mean(0)
    dB = dn.apply_net(net, B, nm, bk, sg, DEV).mean(0)
    rA = cv2.GaussianBlur(A.mean(0), (0, 0), 1.5)
    rB = cv2.GaussianBlur(B.mean(0), (0, 0), 1.5)
    sh = dn._noise(A.mean(0))
    out = {"half_noise": round(sh, 4)}
    for part, m in (("faint_sky", level <= np.median(level)), ("brighter_half", level > np.median(level))):
        r = {}
        for name, a, b in (("denoised", dA, dB), ("blurred", rA, rB)):
            ba = float(np.median(cv2.GaussianBlur(a, (0, 0), 6)[m][::7]))
            loc = cv2.GaussianBlur(a, (0, 0), 6)
            y, x = stars_of(a - loc + ba, sh, ba, 1.5, 30, n=3000, iso=4)
            keep = m[y, x]
            y, x = y[keep], x[keep]
            locb = cv2.GaussianBlur(b, (0, 0), 6)
            smB = cv2.GaussianBlur(b, (0, 0), 1.0) - locb
            r[name] = {"dots": int(len(y)),
                       "found_in_other_half": round(float((smB[y, x] > 0.75 * sh).mean()), 3) if len(y) else None}
        out[part] = r
    return out


def measure_stars(p, k, wid):
    ctx = context(p, k, wid, "star_removal", sr.StarRemoval())
    lin = ctx.load("render_simple", "lin.npy").astype(np.float32)
    valid = ctx.load("render_simple", "valid.npy")
    starless = own(p, k, wid, "star_removal", "starless.npy")
    stars = own(p, k, wid, "star_removal", "stars.npy")
    g = lin[1]
    bkg, sig = mad(np.where(valid, g, np.nan))
    res = {}
    for name, lo, hi in (("faint", 5, 15), ("medium", 15, 100), ("bright", 100, 1e9)):
        y, x = stars_of(g, sig, bkg, lo, hi)
        a, b = phot(g, y, x), phot(starless[1], y, x)
        left = b[:, 1] / np.maximum(a[:, 1], 1e-12)
        res[f"{name}_stars"] = {"n": int(len(y)), "peak_left": round(float(np.median(left)), 3) if len(y) else None,
                                "over_20pc_left": round(float(np.mean(left > 0.2)), 3) if len(y) else None,
                                "dark_holes": round(float(np.mean(b[:, 3] < -3 * sig)), 3) if len(y) else None}
    # away from the stars: how much the starless image differs from the image (noise units)
    m = (np.abs(stars[1]) > 2 * sig).astype(np.uint8)
    far = (cv2.dilate(m, np.ones((15, 15), np.uint8)) == 0) & valid
    res["away_from_stars"] = round(float(np.median(np.abs(starless[1] - g)[far][::5])) / sig, 3)
    # the network's raw output (before the mean of its three channels)
    prep = ctx.load_json("render_simple", "prep_simple.json")
    M = np.asarray(prep["M"], np.float64)
    _bk, sg, _nm = dn.norm_params(stack_image(ctx, "drz_A.npy"), ctx.load("stack", "drz_weight.npy", mmap=True))
    unit = float(np.mean(np.sqrt((M ** 2) @ (np.asarray(sg, np.float64) ** 2))))
    x0, x1, y0, y1 = prep["crop"]
    shown = np.zeros_like(valid)
    shown[y0:y1, x0:x1] = True
    shown &= valid
    bk = np.array([np.median(lin[c][shown][::7]) for c in range(3)], np.float32)[:, None, None]
    raw = sr.apply(sr.network(DEV), (lin - bk) / unit, DEV) * unit + bk
    res["channel_spread"] = spread(raw, sig)
    res["noise_sigma"] = round(sig, 5)
    return res, lin, starless, stars


def crops(g, sig, bkg, size=256):
    """Three crop corners: the brightest star, the densest star field, the brightest smooth area (nebula)."""
    H, W = g.shape
    h = size // 2
    sm = cv2.GaussianBlur(g, (0, 0), 1.0)
    inner = np.zeros_like(g, bool)
    inner[h:H - h, h:W - h] = True
    yb, xb = np.unravel_index(np.argmax(np.where(inner, sm, -np.inf)), g.shape)
    pk = (sm == cv2.dilate(sm, np.ones((5, 5), np.uint8))) & (sm - bkg > 10 * sig)
    dens = cv2.boxFilter(pk.astype(np.float32), -1, (size, size), normalize=False)
    yd, xd = np.unravel_index(np.argmax(np.where(inner, dens, -1)), g.shape)
    neb = cv2.GaussianBlur(g, (0, 0), 25)
    yn, xn = np.unravel_index(np.argmax(np.where(inner, neb, -np.inf)), g.shape)
    return [(int(yb) - h, int(xb) - h), (int(yd) - h, int(xd) - h), (int(yn) - h, int(xn) - h)]


def stretch(a, bkg, top):
    x = np.clip((a - bkg) / max(top - bkg, 1e-12), 0, None)
    return np.clip(np.arcsinh(40 * x) / np.arcsinh(40), 0, 1)


def sheet(out, title, full, dF, lin, starless, stars, size=256):
    g = full[1]
    bkg, sig = mad(g)
    top = float(np.percentile(g[::5, ::5], 99.8))
    if lin is not None:
        lb, _ls = mad(lin[1])
        ltop = float(np.percentile(lin[1][::5, ::5], 99.8))
    rows = []
    for y, x in crops(g, sig, bkg, size):
        tiles = [stretch(g[y:y + size, x:x + size], bkg, top), stretch(dF[1][y:y + size, x:x + size], bkg, top)]
        if lin is not None and lin.shape[1:] == g.shape:
            tiles += [stretch(starless[1][y:y + size, x:x + size], lb, ltop),
                      stretch(stars[1][y:y + size, x:x + size] + lb, lb, ltop)]
        rows.append(np.hstack([np.pad(t, 3, constant_values=0.25) for t in tiles]))
    a = (np.vstack(rows) * 255).astype(np.uint8)
    a = cv2.resize(a, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
    a = cv2.copyMakeBorder(a, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=40)
    labels = ["stack", "denoised", "starless", "stars"]
    for i, lab in enumerate(labels[:a.shape[1] // int(1.5 * (size + 6))]):
        cv2.putText(a, lab, (int(i * 1.5 * (size + 6)) + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 230, 2)
    cv2.putText(a, title, (a.shape[1] - 12 * len(title) - 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 200, 1)
    cv2.imwrite(str(out), a, [cv2.IMWRITE_JPEG_QUALITY, 90])


def evaluate(root: Path, out: Path, tag: str) -> dict:
    p = project2.load(root)
    k = Keys(p)
    res = {}
    for w in p.workflows:
        ch = k.chain(w.id)
        if "denoise" not in ch or (w.source or {}).get("fusion"):
            continue
        d, full, dF = measure_denoise(p, k, w.id)
        res[w.name] = {"denoise": d}
        lin = starless = stars = None
        if "star_removal" in ch:                         # Denoise alone: no starless columns
            s, lin, starless, stars = measure_stars(p, k, w.id)
            res[w.name]["stars"] = s
        print(root.name, w.name, json.dumps(res[w.name]), flush=True)
        sheet(out / f"mono_{tag}_{w.name}.jpg", f"{root.name} {w.name}", full, dF, lin, starless, stars)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("projects", nargs="+")
    ap.add_argument("--colour", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for pr in a.projects:
        allres[Path(pr).name] = evaluate(Path(pr), out, Path(pr).name)
    for c in a.colour:
        allres["colour:" + Path(c).name] = evaluate(Path(c), out, "colour_" + Path(c).name)
    (out / "mono_models.json").write_text(json.dumps(allres, indent=1), encoding="utf-8")
