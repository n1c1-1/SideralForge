"""Run H against the layer a project uses now (run G, or run E): the nebula each leaves (the stack minus its far halo
minus its kept layer), what the rendering loses of each around the saturated stars (the kept layer minus the star
light shown inside the footprint, faded below 1.5 sigma: see lost_light.py), ring levels around the brightest stars.

python compare_GH.py --snap <halo2 project copy> --base name=layer,halo --new name=layer,halo
                     --meta <prep/meta.json> --out <dir> [--spots "x,y,half;..."]
Writes <out>/compare_<base>_<new>.jpg and rings_<base>_<new>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lost_light import kept, synthetic_noise  # noqa: E402

from astro_gpu.stages.starless import _filled_radius  # noqa: E402


def footprints(Ys, sig, fs, full, smask, rmax=400):
    """Weight of the saturated stars' footprints (render_prep.saturated_star_light's rule)."""
    H, W = Ys.shape
    xs, ys = fs["x"], fs["y"]
    ok = (xs > 2) & (xs < W - 3) & (ys > 2) & (ys < H - 3)
    xi, yi = np.round(xs[ok]).astype(int), np.round(ys[ok]).astype(int)
    sat = np.zeros(ok.sum(), bool)
    for c in range(3):
        fc = np.asarray(full[c])
        lvl = np.percentile(fc[::4, ::4], 99.999)
        v = np.max([fc[yi + dy, xi + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 0)
        sat |= v > 0.7 * lvl
    from lost_light import ring_median
    fp = np.zeros((H, W), np.float32)
    for i in np.where(ok)[0][sat]:
        R = 4
        for r_ in range(4, rmax, 2):
            if not ring_median(Ys, xs[i], ys[i], r_) >= 2 * sig:
                break
            R = r_
        R = max(R, int(1.3 * max(60.0, _filled_radius(smask, xs[i], ys[i], rmax=rmax) + 4)))
        R = int(R * 1.15) + 4
        x0, x1 = max(0, int(xs[i]) - R - 2), min(W, int(xs[i]) + R + 3)
        y0, y1 = max(0, int(ys[i]) - R - 2), min(H, int(ys[i]) + R + 3)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        w_ = np.clip((R - np.hypot(xx - xs[i], yy - ys[i])) / (0.15 * R), 0, 1).astype(np.float32)
        fp[y0:y1, x0:x1] = np.maximum(fp[y0:y1, x0:x1], w_)
    return fp


def label(img, text):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spots", default="")
    ap.add_argument("--gmax", type=float, default=6.0)
    a = ap.parse_args()
    w_ = Path(a.snap) / "work"
    full = np.load(w_ / "optics/lat_full.npy", mmap_mode="r")
    L = np.asarray(full, np.float32)
    flat = np.load(w_ / "background/flat.npy").astype(np.float32)
    M = np.array(json.load(open(w_ / "render_prep_learned/prep.json"))["M"])
    lum = (np.array([0.2126, 0.7152, 0.0722]) @ M).astype(np.float32)
    fs = np.load(w_ / "stars/star_fluxes.npz")
    smask = np.load(w_ / "starless/sl_mask.npy")
    meta = json.load(open(a.meta))
    bk, sg = np.array(meta["bk"], np.float32), np.array(meta["sg"], np.float32)
    H, W = L.shape[1:]
    nz = synthetic_noise(L, meta)
    noise_y = np.tensordot(lum, np.stack([cv2.GaussianBlur(nz[c] / flat[c], (0, 0), 1.0) for c in range(3)]), 1)
    del nz
    runs = {}
    for spec in (a.base, a.new):
        name, rest = spec.split("=", 1)
        lay, hal = rest.split(",")
        SL = np.load(lay).astype(np.float32)
        SL *= kept(SL)[None]
        neb = L - np.load(hal).astype(np.float32) - SL                 # what the starless image holds
        Yl = np.tensordot(lum, np.stack([cv2.GaussianBlur(SL[c] / flat[c], (0, 0), 1.0) for c in range(3)]), 1)
        Ys = Yl + noise_y
        hp = Ys[::3, ::3] - cv2.GaussianBlur(Ys[::3, ::3], (0, 0), 2)
        sig = 1.4826 * float(np.median(np.abs(hp - np.median(hp))))
        fp = footprints(Ys, sig, fs, full, smask)
        lost = Yl - Ys * np.clip((Ys / sig - 1.5) / 2.0, 0, 1) * fp       # removed and not shown
        runs[name] = dict(neb=neb, lost=lost / sig, sig=sig)
        del SL, Yl, Ys
    names = list(runs)
    g = fs["g"]
    spots = []
    if a.spots:
        for s_ in a.spots.split(";"):
            x_, y_, h_ = s_.split(",")
            spots.append((float(x_), float(y_), int(h_), f"({x_},{y_})"))
    for i in np.argsort(g):
        if g[i] >= a.gmax:
            break
        if 0 < fs["x"][i] < W and 0 < fs["y"][i] < H:
            spots.append((float(fs["x"][i]), float(fs["y"][i]), 350, f"G{g[i]:.1f}"))
    for (x0, y0, x1, y1), nm in zip((meta["struct_box"], meta["faint_box"]), ("structured hold-out", "faint hold-out")):
        spots.append(((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) // 2 + 64, nm))
    S = 300
    rows, rings = [], {}
    for (xc, yc, h, nm) in spots:
        xa, xb, ya, yb = max(0, int(xc) - h), min(W, int(xc) + h), max(0, int(yc) - h), min(H, int(yc) + h)
        tiles = []
        ref = cv2.GaussianBlur(runs[names[0]]["neb"][:, ya:yb, xa:xb].mean(0), (0, 0), 2)
        lo, hi = np.percentile(ref, 1), np.percentile(ref, 99.5)
        for mode in ("gentle", "hard", "lost"):
            for n in names:
                if mode == "gentle":
                    v = runs[n]["neb"][:, ya:yb, xa:xb]
                    d = np.clip(0.15 + 0.85 * np.arcsinh((v - bk[:, None, None]) / (2 * sg[:, None, None]))
                                / np.arcsinh(3000 / 2), 0, 1)
                    im = (np.transpose(d[::-1], (1, 2, 0)) * 255).astype(np.uint8)
                elif mode == "hard":
                    vb = cv2.GaussianBlur(runs[n]["neb"][:, ya:yb, xa:xb].mean(0), (0, 0), 2)
                    im = cv2.cvtColor((np.clip((vb - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.6 * 255).astype(np.uint8),
                                      cv2.COLOR_GRAY2BGR)
                else:
                    # 2x2 means then a 7 x 7 median: the stars around (drawn, not lost) leave the map, a disc stays
                    lv = runs[n]["lost"][ya:yb - (yb - ya) % 2, xa:xb - (xb - xa) % 2]
                    lv = median_filter(lv.reshape(lv.shape[0] // 2, 2, lv.shape[1] // 2, 2).mean((1, 3)), size=7)
                    im = cv2.applyColorMap((np.clip(0.5 + lv / 2, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                im = cv2.resize(np.ascontiguousarray(im), (S, S), interpolation=cv2.INTER_AREA)
                tiles.append(label(im, f"{mode} {n}" + (" -1..1 sig (median 14 px)" if mode == "lost" else "")))
        row = np.concatenate([np.concatenate([t, np.full((S, 4, 3), 255, np.uint8)], 1) for t in tiles], 1)
        hdr = np.full((22, row.shape[1], 3), 40, np.uint8)
        label(hdr, f"{nm} at ({xc:.0f},{yc:.0f}), {2 * h} px: nebula left (gentle, hard stretch), light lost at rendering")
        rows.append(np.concatenate([hdr, row], 0))
        if nm.startswith("G"):
            yy, xx = np.mgrid[ya:yb, xa:xb]
            r = np.hypot(xx - xc, yy - yc)
            rings[f"{nm}@({xc:.0f},{yc:.0f})"] = {
                n: {f"{lo_}-{hi_}": [round(float((np.median(runs[n]["neb"][1, ya:yb, xa:xb][(r >= lo_) & (r < hi_)])
                                                 - bk[1]) / sg[1]), 1),
                                     round(float(np.median(runs[n]["lost"][ya:yb, xa:xb][(r >= lo_) & (r < hi_)])), 2)]
                    for lo_, hi_ in ((20, 40), (40, 80), (80, 150), (150, 250), (250, 350)) if hi_ <= h}
                for n in names}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"compare_{names[0]}_{names[1]}.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(rings, open(out / f"rings_{names[0]}_{names[1]}.json", "w"), indent=1)
    for k, v in rings.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
