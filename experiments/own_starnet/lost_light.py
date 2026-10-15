"""The light a learned star layer loses at rendering around the saturated stars.

The rendering (render_prep_learned) shows a saturated star by its own light inside a footprint only, and the
learned starless image lacks the layer everywhere: what the layer holds beyond the footprint of a saturated star
(diffuse light: not a neighbouring star, which is drawn) is lost - the disc with an edge around the Rosette's
G 5.6. This simulates it with the footprint rule of render_prep.saturated_star_light:
  layer kept as the starless stage keeps it (|layer| above 3 x its noise, both signs),
  Ys = luminance of (layer + the image's noise) / flat, 1 px blur; sig = its high-pass noise;
  saturated star: 3 x 3 max above 0.7 x the 99.999th percentile in a channel;
  footprint R: last radius (step 2 px, up to 400) where the ring median of Ys stands at 2 sig, at least
  1.3 x max(60, filled core of sl_mask + 4), then 1.15 R + 4.
What the rendering shows of the star: Ys inside the footprint (weight 1 out to 0.85 R), faded below 1.5 sig as in
saturated_star_light. Lost = the kept layer minus what is shown, per 4 px ring out to 4 R (1200 px at most), its
median (the stars around do not count: drawn or shown themselves) times the ring's area, inside the footprint
(the fade) and beyond it. Reported per star: the light lost against the light shown, the peak ring median (sig),
the radii between which it stands above 0.5 sig (a disc with an edge).

The noise of the image is simulated (sky and photon noise, see synthetic_noise), the same for every layer.
python lost_light.py --snap <project copy holding work/> --meta <prep/meta.json> --layers name=path,... --out f.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astro_gpu.stages.starless import _filled_radius  # noqa: E402


def ring_median(img, xc, yc, r_):
    H, W = img.shape
    a = np.linspace(0, 2 * np.pi, max(32, int(2 * np.pi * r_)), endpoint=False)
    px = np.round(xc + r_ * np.cos(a)).astype(int)
    py = np.round(yc + r_ * np.sin(a)).astype(int)
    m = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return float(np.median(img[py[m], px[m]])) if m.sum() >= 16 else np.nan


def synthetic_noise(L, meta, seed=0):
    """A noise image like the stack's, zero-mean everywhere: sky noise plus photon noise of the signal (the
    project's measured gain). The stack minus its denoised version, and the half-stack difference, are both biased
    at the cores of the bright stars."""
    rng = np.random.default_rng(seed)
    sg, bk, gain = (np.asarray(meta[k], np.float32) for k in ("sg", "bk", "gain"))
    return np.stack([rng.standard_normal(L.shape[1:], dtype=np.float32)
                     * np.sqrt(sg[c] ** 2 + gain[c] * np.clip(np.asarray(L[c], np.float32) - bk[c], 0, None))
                     for c in range(L.shape[0])])


def kept(SL):
    """The starless stage's significance weight of the layer (both signs)."""
    sb = cv2.GaussianBlur(SL.mean(0), (0, 0), 1.0)
    s0 = 1.4826 * float(np.median(np.abs(sb[::4, ::4] - np.median(sb[::4, ::4]))))
    w = np.clip((np.abs(sb) / max(s0, 1e-9) - 3.0) / 3.0, 0, 1)
    return cv2.GaussianBlur(cv2.dilate(w, np.ones((3, 3), np.uint8)), (0, 0), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--meta", default="", help="prep's meta.json (sky, noise, photon gain); default: the prep next to "
                                               "the project copy (<copy>/../prep/meta.json)")
    ap.add_argument("--nl", default="", help="unused (older drivers)")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rmax", type=int, default=400)
    a = ap.parse_args()
    w_ = Path(a.snap) / "work"
    full = np.load(w_ / "optics/lat_full.npy", mmap_mode="r")
    flat = np.load(w_ / "background/flat.npy").astype(np.float32)
    M = np.array(json.load(open(w_ / "render_prep_learned/prep.json"))["M"])
    lum = np.array([0.2126, 0.7152, 0.0722]) @ M
    fs = np.load(w_ / "stars/star_fluxes.npz")
    smask = np.load(w_ / "starless/sl_mask.npy")
    H, W = flat.shape[1:]
    noise = synthetic_noise(full, json.load(open(a.meta or Path(a.snap).parent / "prep/meta.json")))
    xs, ys, g = fs["x"], fs["y"], fs["g"]
    ok = (xs > 2) & (xs < W - 3) & (ys > 2) & (ys < H - 3)
    xi, yi = np.round(xs[ok]).astype(int), np.round(ys[ok]).astype(int)
    sat = np.zeros(ok.sum(), bool)
    for c in range(3):
        fc = np.asarray(full[c])
        lvl = np.percentile(fc[::4, ::4], 99.999)
        v = np.max([fc[yi + dy, xi + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 0)
        sat |= v > 0.7 * lvl
    idx = np.where(ok)[0][sat]
    idx = idx[np.argsort(g[idx])]
    res = {"stars": {}, "totals": {}}
    for spec in a.layers.split(","):
        name, path = spec.split("=", 1)
        SL = np.load(path).astype(np.float32)
        SL *= kept(SL)[None]
        Yl = np.tensordot(lum, np.stack([cv2.GaussianBlur(SL[c] / flat[c], (0, 0), 1.0) for c in range(3)]), 1)
        Ys = Yl + np.tensordot(lum, np.stack([cv2.GaussianBlur(noise[c] / flat[c], (0, 0), 1.0) for c in range(3)]), 1)
        hp = Ys[::3, ::3] - cv2.GaussianBlur(Ys[::3, ::3], (0, 0), 2)
        sig = 1.4826 * float(np.median(np.abs(hp - np.median(hp))))
        tl, ts = 0.0, 0.0
        for i in idx:
            R = 4
            for r_ in range(4, a.rmax, 2):
                if not ring_median(Ys, xs[i], ys[i], r_) >= 2 * sig:
                    break
                R = r_
            R = max(R, int(1.3 * max(60.0, _filled_radius(smask, xs[i], ys[i], rmax=a.rmax) + 4)))
            R = int(R * 1.15) + 4
            Rw = min(4 * R, 1200)
            x0, x1 = max(0, int(xs[i]) - Rw - 2), min(W, int(xs[i]) + Rw + 3)
            y0, y1 = max(0, int(ys[i]) - Rw - 2), min(H, int(ys[i]) + Rw + 3)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            rr = np.hypot(xx - xs[i], yy - ys[i])
            wf = np.clip((R - rr) / (0.15 * R), 0, 1)
            ysn = Ys[y0:y1, x0:x1]
            # what the rendering shows: the star's light inside its footprint, faded below 1.5 sig (per pixel)
            shown_map = ysn * np.clip((ysn / sig - 1.5) / 2.0, 0, 1) * wf
            lost_map = Yl[y0:y1, x0:x1] - shown_map            # removed from the nebula, not shown
            shown = float(shown_map.sum())
            lost_in = lost_out = neg = 0.0
            peak, trough, r_lo, r_hi, prof = 0.0, 0.0, None, None, []
            rk = (rr / 4).astype(int)
            order = np.argsort(rk.ravel())
            rs = rk.ravel()[order]
            lv = lost_map.ravel()[order]
            bounds = np.searchsorted(rs, np.arange(0, Rw // 4 + 2))
            for k in range(3, Rw // 4):                  # from 12 px: the saturated core itself is not judged
                v_ = lv[bounds[k]:bounds[k + 1]]
                if v_.size < 16:
                    continue
                m_ = float(np.median(v_))
                area = float(v_.size)
                if k * 4 < R:
                    lost_in += max(m_, 0.0) * area
                else:
                    lost_out += max(m_, 0.0) * area
                neg += min(m_, 0.0) * area
                peak, trough = max(peak, m_ / sig), min(trough, m_ / sig)
                if m_ >= 0.5 * sig:
                    r_lo = k * 4 if r_lo is None else r_lo
                    r_hi = k * 4 + 4
                prof.append(round(m_ / sig, 2))
            lost = lost_in + lost_out
            key = f"G{g[i]:.1f}@({xs[i]:.0f},{ys[i]:.0f})"
            res["stars"].setdefault(key, {})[name] = dict(
                footprint_px=R, shown=round(shown), lost=round(lost), lost_inside_footprint=round(lost_in),
                lost_beyond_footprint=round(lost_out), lost_sigma_px2=round(lost / sig),
                lost_pct=round(100 * lost / max(shown + lost, 1e-9), 2), negative_sigma_px2=round(neg / sig),
                peak_sigma=round(peak, 2), trough_sigma=round(trough, 2),
                lost_above_half_sigma_px=[r_lo, r_hi], rings_4px=prof[:120])
            tl += lost
            ts += shown
        res["totals"][name] = dict(n_saturated=int(len(idx)), lost=round(tl), shown=round(ts),
                                   lost_pct=round(100 * tl / max(ts, 1e-9), 2), noise_sigma=round(sig, 4))
        print(name, res["totals"][name])
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
