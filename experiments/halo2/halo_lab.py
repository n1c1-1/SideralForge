"""The far halo of the dominant stars, offline: the stage's computation on a read-only copy of a project, the data
against the model around the brightest stars (radial profiles per channel) and starless-style views.

python halo_lab.py halo  --snap <copy holding work/> --out <dir> --tag old|new   # far halo + rays as the stage does
python halo_lab.py prof  --snap ... --out ... --tags old,new                      # profiles, views, numbers
Every run reads the copy only and writes to --out.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from astro_gpu.stages import far_halo as fh  # noqa: E402
from astro_gpu.stages.starless import _filled_radius  # noqa: E402


def load_common(snap: Path):
    w = snap / "work"
    img = np.load(w / "optics/lat_full.npy").astype(np.float32)
    model = np.load(w / "stars/stars_model.npy").astype(np.float32)
    model += np.load(w / "stars/stars_model_faint.npy")
    fs = np.load(w / "stars/star_fluxes.npz")
    dm = float(json.load(open(w / "optics/manifest.json"))["metrics"].get("mag_offset", 0.0))
    return img, model, fs, dm


def stage_halo(img, model, fs, dm, log=print, **kw):
    """Stars._far_halo then Stars._own_rays, as the stars stage (v12) runs them last."""
    x, y, g, Fp = fs["x"], fs["y"], fs["g"] - dm, fs["pred"]
    H, W = img.shape[1:]
    hs = np.where((g < 10.5) & (x > -40) & (x < W + 40) & (y > -40) & (y < H + 40))[0]
    sat = np.zeros((H, W), bool)
    for c in range(3):
        sat |= img[c] > 0.7 * np.percentile(img[c][::4, ::4], 99.999)
    r_in = np.array([_filled_radius(sat, x[i], y[i], rmax=800) + 4 for i in hs], float)
    halo, info = fh.far_halo(img - model, x[hs], y[hs], Fp[:, hs], r_in, log=log, **kw)
    dom = [(int(hs[int(k)]), float(r_in[int(k)])) for k in info.get("far_halo_own", {})]
    glow = info.get("_dominant")
    rays = None
    if dom:
        idx = np.array([i for i, _ in dom])
        rr = np.array([max(fh.WING_OUT, r) for _, r in dom])
        rays = fh.own_rays(img - model - halo, x[idx], y[idx], rr, halo)
    return halo, glow, rays, info, hs


def cmd_halo(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img, model, fs, dm = load_common(Path(a.snap))
    t0 = time.time()
    kw = json.loads(a.kw) if a.kw else {}
    halo, glow, rays, info, hs = stage_halo(img, model, fs, dm, **kw)
    tot = halo + (rays if rays is not None else 0)
    np.save(out / f"halo_{a.tag}.npy", tot.astype(np.float32))          # = stars_model_halo.npy
    gl = (glow if glow is not None else np.zeros_like(halo)) + (rays if rays is not None else 0)
    np.save(out / f"glow_{a.tag}.npy", gl.astype(np.float32))           # = stars_model_glow.npy
    fit = info.pop("_fit")
    fit["hs"] = hs
    info.pop("_dominant", None)
    pickle.dump(fit, open(out / f"fit_{a.tag}.pkl", "wb"))
    json.dump({k: v for k, v in info.items() if not k.startswith("_")}, open(out / f"info_{a.tag}.json", "w"),
              indent=1, default=str)
    print(f"halo {a.tag}: {time.time() - t0:.0f} s")


# ------------------------------------------------------------------------------------------------- profiles

def noise_of(img):
    hp = img[1, ::3, ::3] - cv2.GaussianBlur(img[1, ::3, ::3], (0, 0), 2)
    return 1.4826 * float(np.median(np.abs(hp - np.median(hp))))


def targets(fs, dm, fit, W, H, gmax=6.5, extra=()):
    """The stars to look at: every star brighter than gmax in the frame, the stars with a halo of their own."""
    g = fs["g"] - dm
    x, y = fs["x"], fs["y"]
    hs = fit["hs"]
    sel = set(int(i) for i in np.where((g < gmax) & (x > 0) & (x < W) & (y > 0) & (y < H))[0])
    sel |= set(int(hs[k]) for k in fit["strong"]) | set(int(hs[k]) for k in fit["measured"])
    for (ex, ey) in extra:
        sel.add(int(np.argmin(np.hypot(x - ex, y - ey) + 3 * g)))
    return sorted(sel, key=lambda i: g[i])


def ring_stats(E, h, re, edges, valid):
    """Per ring of the elliptical radius re: percentiles 10/25/50 of E per channel and the median of h."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (re >= a) & (re < b) & valid
        if m.sum() < 30:
            out.append(None)
            continue
        out.append(dict(r=float(np.sqrt(a * b)), n=int(m.sum()),
                        p10=[float(np.percentile(E[c][m], 10)) for c in range(3)],
                        p25=[float(np.percentile(E[c][m], 25)) for c in range(3)],
                        p50=[float(np.percentile(E[c][m], 50)) for c in range(3)],
                        h=[float(np.median(h[c][m])) for c in range(3)]))
    return out


def cmd_prof(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(a.out)
    snap = Path(a.snap)
    img, model, fs, dm = load_common(snap)
    H, W = img.shape[1:]
    tags = a.tags.split(",")
    fits = {t: pickle.load(open(out / f"fit_{t}.pkl", "rb")) for t in tags}
    halos = {t: np.load(out / f"halo_{t}.npy", mmap_mode="r") for t in tags}
    D = 4
    h4, w4 = H // D, W // D
    res = np.stack([cv2.resize(img[c] - model[c], (w4, h4), interpolation=cv2.INTER_AREA) for c in range(3)])
    Y4, X4 = (np.mgrid[0:h4, 0:w4] + 0.5) * D - 0.5
    sig = noise_of(img) / D            # noise of a 4 x 4 block mean (green)
    sl = np.load(snap / "work/starless/sl_mask.npy", mmap_mode="r")
    core4 = cv2.resize(np.asarray(sl, np.float32), (w4, h4), interpolation=cv2.INTER_AREA) > 0.05
    valid = ~core4 & np.isfinite(res).all(0)
    extra = [tuple(map(float, s.split(","))) for s in a.extra.split(";")] if a.extra else []
    f0 = fits[tags[0]]
    idx = targets(fs, dm, f0, W, H, a.gmax, extra)
    g = fs["g"] - dm
    report = {}
    pdir = out / "profiles"
    pdir.mkdir(exist_ok=True)
    edges = np.geomspace(20, 1600, 34)
    for i in idx:
        xi, yi = float(fs["x"][i]), float(fs["y"][i])
        name = f"G{g[i]:.1f}_{int(xi)}_{int(yi)}"
        rep = {"x": xi, "y": yi, "G": round(float(g[i]), 2)}
        fig, axs = plt.subplots(2, 3, figsize=(17, 8))
        for t in tags:
            fit = fits[t]
            hs = list(fit["hs"])
            k = hs.index(i) if i in hs else None
            sky = fh.sky_of(fit, X4, Y4)
            tot = np.stack([cv2.resize(np.asarray(halos[t][c], np.float32), (w4, h4), interpolation=cv2.INTER_AREA)
                            for c in range(3)])
            own = fh.star_halo(fit, k, X4, Y4) if k is not None else np.zeros_like(tot)
            E = res - sky - (tot - own)            # what this star's halo and the nebula must explain
            q, phi = (fit["Q"][k], fit["PHI"][k]) if k is not None else (1.0, 0.0)
            cp, sp = np.cos(phi), np.sin(phi)
            dx, dy = X4 - xi, Y4 - yi
            re = np.sqrt((dx * cp + dy * sp) ** 2 / q + (-dx * sp + dy * cp) ** 2 * q)
            st = ring_stats(E, own, re, edges, valid)
            rep[t] = dict(kap=None if k is None else round(float(fit["kap"][k]), 3),
                          q=round(float(q), 2), beta=None if k is None else round(float(fit["BETA"][k]), 2),
                          r_in=None if k is None else round(float(fit["r_in"][k]), 1),
                          own="strong" if k in fit["strong"] else "measured" if k in fit["measured"] else "shared",
                          rings=st)
            # over-subtraction: the model above the lower quarter of the data, in noise sigmas of a 4x4 block
            ex = [(s["r"], [(s["h"][c] - s["p25"][c]) / sig for c in range(3)]) for s in st if s is not None]
            rep[t]["excess_p25_sigma"] = [(round(r, 0), [round(v, 1) for v in e]) for r, e in ex]
            col = {0: "tab:red", 1: "tab:blue", 2: "tab:green"}[tags.index(t)]
            for c in range(3):
                rr = [s["r"] for s in st if s is not None]
                # top: the levels (data and this star's model); bottom: the data minus the model (pixel sigmas)
                if t == tags[0]:
                    for key, cl, lab in (("p10", "0.6", "data 10 %"), ("p25", "0.3", "data 25 %"),
                                         ("p50", "k", "data median")):
                        axs[0, c].plot(rr, [s[key][c] / sig for s in st if s is not None], color=cl, lw=1, label=lab)
                axs[0, c].plot(rr, [s["h"][c] / sig for s in st if s is not None], color=col, lw=2,
                               label=f"model {t}")
                for key, ls in (("p25", "-"), ("p50", ":")):
                    axs[1, c].plot(rr, [(s[key][c] - s["h"][c]) / (D * sig) for s in st if s is not None], ls,
                                   color=col, lw=1.5, label=f"data {key[1:]} % - model {t}")
        for c in range(3):
            for k_ in range(2):
                ax = axs[k_, c]
                ax.set_xscale("log")
                ax.axhline(0, color="0.5", lw=0.8)
                ax.grid(alpha=0.3)
            axs[0, c].set_yscale("symlog", linthresh=10)
            axs[0, c].set_title(f"{name} channel {'RGB'[c]}: levels (4x4 block sigmas)")
            axs[1, c].set_title("data minus model (pixel noise sigmas)")
            axs[1, c].set_ylim(-12, 12)
            axs[1, c].set_xlabel("elliptical radius (px)")
        axs[0, 0].legend(fontsize=7)
        axs[1, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(pdir / f"{name}.png", dpi=80)
        plt.close(fig)
        report[name] = rep
    json.dump(report, open(out / f"profiles_{'_'.join(tags)}.json", "w"), indent=1)
    print(f"{len(idx)} stars profiled")


def _label(img, text):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def cmd_view(a):
    """Starless-style views around the stars: the model starless image with each halo (the old one as computed by
    the project, the others substituted), a gentle and a hard stretch, and the light given back (old - new)."""
    out, snap = Path(a.out), Path(a.snap)
    w_ = snap / "work"
    tags = a.tags.split(",")
    sl = np.load(w_ / "starless/sl_final.npy", mmap_mode="r")
    h0 = np.load(w_ / "stars/stars_model_halo.npy", mmap_mode="r")
    msk = np.load(w_ / "starless/sl_mask.npy", mmap_mode="r")
    halos = {t: np.load(out / f"halo_{t}.npy", mmap_mode="r") for t in tags}
    fs = np.load(w_ / "stars/star_fluxes.npz")
    dm = float(json.load(open(w_ / "optics/manifest.json"))["metrics"].get("mag_offset", 0.0))
    g = fs["g"] - dm
    full = np.load(w_ / "optics/lat_full.npy", mmap_mode="r")
    H, W = sl.shape[1:]
    sig = noise_of(np.asarray(full[:, ::2, ::2]))
    bk = np.array([float(np.median(np.asarray(sl[c, ::8, ::8]))) for c in range(3)])
    fit = pickle.load(open(out / f"fit_{tags[0]}.pkl", "rb"))
    idx = targets(fs, dm, fit, W, H, a.gmax)
    # a few ordinary bright stars, isolated, inside the frame
    cand = np.where((g > 6.8) & (g < 8.5) & (fs["x"] > 700) & (fs["x"] < W - 700) & (fs["y"] > 700)
                    & (fs["y"] < H - 700))[0]
    br = np.where(g < 9.5)[0]
    iso = [i for i in cand if np.sort(np.hypot(fs["x"][br] - fs["x"][i], fs["y"][br] - fs["y"][i]))[1] > 250]
    idx = list(idx) + iso[:a.ordinary]
    R, S = a.half, 360
    rows = []
    for i in idx:
        xc, yc = int(round(fs["x"][i])), int(round(fs["y"][i]))
        x0, x1, y0, y1 = max(0, xc - R), min(W, xc + R), max(0, yc - R), min(H, yc + R)
        base = np.asarray(sl[:, y0:y1, x0:x1], np.float32) + np.asarray(h0[:, y0:y1, x0:x1], np.float32)
        # the filled cores of the model starless image hold no data (their fill was made with the old halo): grey
        fill = cv2.dilate(np.asarray(msk[y0:y1, x0:x1]).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        ims = {t: base - np.asarray(halos[t][:, y0:y1, x0:x1], np.float32) for t in tags}
        ref = cv2.GaussianBlur(ims[tags[0]].mean(0), (0, 0), 2)
        lo, hi = np.percentile(ref, 1), np.percentile(ref, 99.5)
        tiles = []
        for mode in ("gentle", "hard"):
            for t in tags:
                v = ims[t]
                if mode == "gentle":
                    d = np.arcsinh((v - bk[:, None, None]) / (2 * sig)) / np.arcsinh(3000 / 2)
                    d = np.clip(0.15 + 0.85 * d, 0, 1)
                else:
                    vb = cv2.GaussianBlur(v.mean(0), (0, 0), 2)            # brightness only: rings and lobes
                    d = np.repeat((np.clip((vb - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.6)[None], 3, 0)
                d = np.where(fill[None], 0.5, d)
                im = np.ascontiguousarray((np.transpose(d[::-1], (1, 2, 0)) * 255).astype(np.uint8))
                im = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)
                tiles.append(_label(im, f"{mode} {t}"))
        dif = np.where(fill, 0.0, (ims[tags[-1]] - ims[tags[0]]).mean(0) / sig)
        dv = np.clip(0.5 + dif / 20, 0, 1)
        dv = cv2.applyColorMap((cv2.resize(dv, (S, S), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8),
                               cv2.COLORMAP_JET)
        tiles.append(_label(dv, f"{tags[-1]} - {tags[0]} -10..10 sig"))
        row = np.concatenate([np.concatenate([t_, np.full((S, 4, 3), 255, np.uint8)], 1) for t_ in tiles], 1)
        hdr = np.full((22, row.shape[1], 3), 40, np.uint8)
        _label(hdr, f"G {g[i]:.1f} at ({xc}, {yc}), {2 * R} px square")
        rows.append(np.concatenate([hdr, row], 0))
    cv2.imwrite(str(out / f"views_{'_'.join(tags)}.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(idx)} views")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["halo", "prof", "view"])
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="old")
    ap.add_argument("--tags", default="old")
    ap.add_argument("--kw", default="", help="halo: keyword arguments of far_halo (JSON)")
    ap.add_argument("--gmax", type=float, default=6.5)
    ap.add_argument("--extra", default="", help="prof: more stars 'x,y;x,y'")
    ap.add_argument("--half", type=int, default=600, help="view: half size of the crops")
    ap.add_argument("--ordinary", type=int, default=3, help="view: number of ordinary bright stars")
    a = ap.parse_args()
    {"halo": cmd_halo, "prof": cmd_prof, "view": cmd_view}[a.cmd](a)


if __name__ == "__main__":
    main()
