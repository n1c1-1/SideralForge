"""Do faint stars inside the glare of the brightest stars stay point-like in the network's star layer?

For every catalogue star G 11-16 at 40-300 px from one of the given bright stars (no other star G < 16 within
15 px), in a 41 px cutout of the star layer (and of the reference: denoised stack minus far halo), a plane
fitted on the 8-12 px annulus is removed (the glare gradient), then on the positive light within 6 px:
  elongation = sqrt(second moment along the direction of the bright star / across it)  (radial comet: >> 1)
  flux ratio = light within 5 px in the star layer / in the reference                    (removed fully: ~1)
python faint_in_glare.py --project <snapshot> --out <dir> --runs runE,runF --stars x,y[,label] ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import own_starnet as o  # noqa: E402


def measure(cut, u):
    """cut (3, 41, 41) -> (elongation within 6 px, tail ratio, flux within 5 px) of the RGB sum after removing a plane
    fitted on the 16-20 px annulus. Tail ratio: light 6-16 px from the star in a 4 px wide strip along the direction
    of the bright star, over the same across it (a radial comet: >> 1)."""
    a = cut.sum(0)
    yy, xx = np.mgrid[-20:21, -20:21]
    r = np.hypot(yy, xx)
    ann = (r >= 16) & (r <= 20)
    A = np.c_[np.ones(ann.sum()), xx[ann], yy[ann]]
    coef = np.linalg.lstsq(A, a[ann], rcond=None)[0]
    a = a - (coef[0] + coef[1] * xx + coef[2] * yy)
    pr = xx * u[0] + yy * u[1]                  # along the direction of the bright star
    pt = -xx * u[1] + yy * u[0]                 # across it
    band = (r >= 6) & (r <= 16)
    rad = np.clip(a, 0, None)[band & (np.abs(pt) <= 2)].sum()
    tan = np.clip(a, 0, None)[band & (np.abs(pr) <= 2)].sum()
    tail = float(rad / tan) if tan > 0 else np.nan
    inner = r < 6
    w = np.clip(a, 0, None) * inner
    tot = w.sum()
    if tot <= 0:
        return np.nan, tail, float(a[r < 5].sum())
    cx, cy = (w * xx).sum() / tot, (w * yy).sum() / tot
    dx, dy = xx - cx, yy - cy
    mrr = (w * (dx * u[0] + dy * u[1]) ** 2).sum() / tot
    mtt = (w * (-dx * u[1] + dy * u[0]) ** 2).sum() / tot
    return float(np.sqrt(mrr / max(mtt, 1e-6))), tail, float(a[r < 5].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project")
    ap.add_argument("--out")
    ap.add_argument("--runs", default="runE,runF")
    ap.add_argument("--stars", action="append", default=[])
    ap.add_argument("--rmin", type=float, default=40)
    ap.add_argument("--rmax", type=float, default=300)
    args = ap.parse_args()
    o.set_paths(args.project, args.out)
    d = np.load(o.PROJ / "stars/star_fluxes.npz")
    x, y, g = d["x"], d["y"], d["g"]
    NL = np.load(o.PREP / "n2n_lat.npy", mmap_mode="r")
    hf = o.PROJ / "stars/stars_model_halo.npy"
    halo = np.load(hf, mmap_mode="r") if hf.exists() else None
    layers = {r: np.load(o.OUT / r / "star_layer_net.npy", mmap_mode="r") for r in args.runs.split(",")}
    H, W = NL.shape[1:]
    res = {}
    for spec in args.stars:
        v = spec.split(",")
        bx, by = float(v[0]), float(v[1])
        lab = v[2] if len(v) > 2 else f"({v[0]},{v[1]})"
        rr = np.hypot(x - bx, y - by)
        sel = np.where((g >= 11) & (g < 16) & (rr >= args.rmin) & (rr <= args.rmax) & (x > 21) & (y > 21)
                       & (x < W - 21) & (y < H - 21))[0]
        br14 = np.where(g < 16)[0]
        out = {k: [] for k in ["reference"] + list(layers)}
        tails = {k: [] for k in ["reference"] + list(layers)}
        flux = {k: [] for k in layers}
        n = 0
        for i in sel:
            others = br14[br14 != i]
            if others.size and np.hypot(x[others] - x[i], y[others] - y[i]).min() < 15:
                continue
            xi, yi = int(round(x[i])), int(round(y[i]))
            sl = (slice(None), slice(yi - 20, yi + 21), slice(xi - 20, xi + 21))
            ref = np.asarray(NL[sl], np.float32) - (np.asarray(halo[sl], np.float32) if halo is not None else 0)
            u = np.array([x[i] - bx, y[i] - by]) / max(rr[i], 1e-6)
            e_ref, t_ref, f_ref = measure(ref, u)
            if not np.isfinite(e_ref) or f_ref <= 0:
                continue
            out["reference"].append(e_ref)
            tails["reference"].append(t_ref)
            for k, Lr in layers.items():
                e, t, f = measure(np.asarray(Lr[sl], np.float32), u)
                out[k].append(e)
                tails[k].append(t)
                flux[k].append(f / f_ref)
            n += 1
        res[lab] = {"stars": n,
                    "elongation_median": {k: round(float(np.nanmedian(v)), 3) if v else None for k, v in out.items()},
                    "elongated_over_1.5_pct": {k: round(100 * float(np.nanmean(np.array(v) > 1.5)), 1) if v else None
                                               for k, v in out.items()},
                    "tail_ratio_median": {k: round(float(np.nanmedian(v)), 3) if v else None for k, v in tails.items()},
                    "tail_over_2_pct": {k: round(100 * float(np.nanmean(np.array(v) > 2)), 1) if v else None
                                        for k, v in tails.items()},
                    "flux_ratio_median": {k: round(float(np.nanmedian(v)), 3) if v else None for k, v in flux.items()}}
        print(lab, json.dumps(res[lab]))
    (o.OUT / "faint_in_glare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
