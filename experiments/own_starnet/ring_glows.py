"""Ring glows left in the starless image around the saturated stars (saturated as the rendering sees them): for
each, the radial profile (median per 4 px ring, mean of the channels in noise sigmas) of the nebula a layer leaves
(stack minus far halo minus the kept layer); ring height = the highest ring between 40 and 160 px minus the ring at
16-28 px (a glare falling outward gives <= 0).
python ring_glows.py <project> "name=layer,halo;name=layer,halo"   (copies in D:/astro_gpu_diag/halo2/<project>,
                                                                    meta of D:/astro_gpu_diag/own_starnet/<project>/H)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lost_light import kept  # noqa: E402

P = sys.argv[1]
runs = [s.split("=", 1) for s in sys.argv[2].split(";")]
w = f"D:/astro_gpu_diag/halo2/{P}/snapshot/work/"
meta = json.load(open(f"D:/astro_gpu_diag/own_starnet/{P}/H/prep/meta.json"))
sg = np.array(meta["sg"], np.float32)
L = np.load(w + "optics/lat_full.npy", mmap_mode="r")
fs = np.load(w + "stars/star_fluxes.npz")
H, W = L.shape[1:]
xs, ys, g = fs["x"], fs["y"], fs["g"]
ok = (xs > 200) & (xs < W - 200) & (ys > 200) & (ys < H - 200)
lvl = np.array([np.percentile(np.asarray(L[c, ::4, ::4]), 99.999) for c in range(3)])
idx = []
for i in np.where(ok & (g < 11))[0]:
    xi, yi = int(round(xs[i])), int(round(ys[i]))
    if (np.asarray(L[:, yi - 1:yi + 2, xi - 1:xi + 2]).reshape(3, -1).max(1) > 0.7 * lvl).any():
        idx.append(i)
res = {}
for name, spec in runs:
    lay, hal = spec.split(",")
    SL = np.load(lay, mmap_mode="r")
    Hh = np.load(hal, mmap_mode="r")
    hts = []
    for i in idx:
        xi, yi = int(round(xs[i])), int(round(ys[i]))
        sl = (slice(None), slice(yi - 200, yi + 200), slice(xi - 200, xi + 200))
        lay_ = np.asarray(SL[sl], np.float32)
        neb = np.asarray(L[sl], np.float32) - np.asarray(Hh[sl], np.float32) - lay_ * kept(lay_)[None]
        v = (neb / sg[:, None, None]).mean(0)
        yy, xx = np.mgrid[-200:200, -200:200]
        rk = (np.hypot(yy, xx) / 4).astype(int)
        prof = np.array([np.median(v[rk == k]) for k in range(50)])
        hts.append(float(prof[10:40].max() - prof[4:7].mean()))
    res[name] = hts
    hts = np.array(hts)
    print(P, name, len(hts), "saturated stars; ring height > 0.5 sigma:", int((hts > 0.5).sum()),
          "> 1 sigma:", int((hts > 1).sum()), f"median {np.median(hts):.2f}", f"max {hts.max():.2f}")
