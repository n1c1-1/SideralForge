"""Own-data star removal network (experiment, see README.md).

A UNet learns to remove stars from ONE project's stack, trained only on that stack:
  target = the model-based starless image (sl_final) where it is reliable (away from the bright stars),
  input  = the real stack there (its real noise and real faint stars) + real bright-star stamps cut from the same
           stack (denoised stack - sl_final, around isolated stars) pasted at random places, scaled, re-saturated,
           with their photon noise.

Sub-commands (run them in this order; every Python run must see only the second GPU: CUDA_VISIBLE_DEVICES=1):
  prep  [--minus-halo]      stamps library, normalisation, star extent law, reliability map, hold-out regions,
                            evaluation stars, nebula colour (all measured on the project's own data)
  train --run R [flags]     see README.md (the recommended recipe is run E's flags)
  infer --run R             full-frame starless_net.npy and star_layer_net.npy (tiled, feathered)
  layer --run R             star_layer_net.npy from an existing starless_net.npy
  eval  --run R             star metrics, region checks, known-truth injection test, comparison sheets
  table --run R1,R2,...     Markdown tables of the metrics
All take --project <folder holding work/> (default: the Iris test project) and --out <dir> (default:
D:/astro_gpu_diag/own_starnet for the Iris, D:/astro_gpu_diag/own_starnet/<project folder name> otherwise).
The project folder is only read.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from astro_gpu.stages.denoise import UNet  # noqa: E402
from astro_gpu.stages.optics import apply_lateral  # noqa: E402

DEFAULT_PROJECT = Path("C:/Users/Nicolas/astro_gpu_scratch/projects/Iris")
DIAG = Path("D:/astro_gpu_diag/own_starnet")
PROJ = DEFAULT_PROJECT / "work"     # set by set_paths()
OUT = DIAG
PREP = OUT / "prep"
RADIUS_FIT = None      # (a, b): star extent R = exp(a + b G) measured by prep; None = the first prep's fixed law
K_ASINH = 4.0                 # asinh softening, in noise sigmas
WIDTHS = (32, 64, 128, 192, 256, 320)   # 5 down-samplings: receptive field ~ 400 px


def set_paths(project=None, out=None):
    global PROJ, OUT, PREP
    proj = Path(project) if project else DEFAULT_PROJECT
    PROJ = proj / "work"
    if out:
        OUT = Path(out)
    else:
        OUT = DIAG if proj.resolve() == DEFAULT_PROJECT.resolve() else DIAG / proj.name
    PREP = OUT / "prep"


def set_radius(meta):
    global RADIUS_FIT
    RADIUS_FIT = meta.get("radius_fit")


def radius_of(g):
    """Stamp radius (px) from the Gaia G magnitude: the law measured by prep on this frame's bright stars
    (R = exp(a + b G), 12..600 px). The very first Iris prep, which predates the measurement, used a fixed law
    (40 px at G 11 to 250 px at G 6, 12..400 px); it is kept for that prep only."""
    g = np.asarray(g, float)
    if RADIUS_FIT is None:
        return np.clip(40.0 * 6.25 ** ((11.0 - g) / 5.0), 12, 400)
    a, b = RADIUS_FIT
    return np.clip(np.exp(a + b * g), 12, 600)


def log(path, msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load(name, mmap=True):
    return np.load(PROJ / name, mmap_mode="r" if mmap else None)


def boxes(meta):
    """The two hold-out boxes (x0, y0, x1, y1): the structured one and the faint one (older metas: 'iris_box')."""
    return meta.get("struct_box", meta.get("iris_box")), meta["faint_box"]


# ----------------------------------------------------------------------------------------------------------- prep

def lib_name(minus_halo, stamps=""):
    return ("library_nohalo" if minus_halo else "library") + (f"_{stamps}" if stamps else "") + ".pt"


def load_input(minus_halo):
    """The network input domain: lat_full, or lat_full minus the measured far halo (glare) of the stars."""
    L = np.load(PROJ / "optics/lat_full.npy")
    if minus_halo:
        L -= np.load(PROJ / "stars/stars_model_halo.npy")
    return L


def taper(R):
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    r = np.hypot(yy, xx) / R
    w = np.where(r <= 0.6, 1.0, 0.5 * (1 + np.cos(np.pi * np.clip((r - 0.6) / 0.4, 0, 1))))
    return w.astype(np.float32), r


def measure_extent(SL, sg, x, y, g, dirs, H, W, rmax=1000, nmax=60):
    """Extent of the light of the brightest stars (G < 11, no brighter star within 100 px), measured on the star
    layer SL = denoised stack (- far halo) - starless: the radius where the ring median, and the mean over the spikes
    (measured directions) of the median along each spike minus the median 6-14 px beside it, fall below 0.25 noise
    sigma (RGB sum, 2 px Gaussian, 5-ring running median, 4 px rings); the ring median is taken relative to its
    level in the outer third of the window. Returns {index: (r_halo, r_spikes)}."""
    out = {}
    for i in np.argsort(g):
        if g[i] >= 11 or len(out) >= nmax:
            break
        xi, yi = int(round(x[i])), int(round(y[i]))
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        br = g < g[i]
        if br.any() and np.hypot(x[br] - x[i], y[br] - y[i]).min() < 100:
            continue
        y0, y1, x0, x1 = max(0, yi - rmax), min(H, yi + rmax + 1), max(0, xi - rmax), min(W, xi + rmax + 1)
        src = np.asarray(SL[:, y0:y1, x0:x1], np.float32)
        s = cv2.GaussianBlur((src / sg[:, None, None]).sum(0) / math.sqrt(3), (0, 0), 2)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - x[i], yy - y[i]
        r = np.hypot(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx)) % 360
        nb = int(min(r.max(), rmax) // 4)
        lab = np.zeros(r.shape, np.int8)      # 1+k on spike k, -(1+k) beside it, 0 elsewhere
        for k, t in enumerate(dirs):
            da = np.abs((ang - t + 180) % 360 - 180)
            perp = r * np.sin(np.radians(np.minimum(da, 90)))
            near = da < 30
            lab[near & (perp <= 2.5)] = k + 1
            lab[near & (perp >= 6) & (perp <= 14)] = -(k + 1)
        rb = (r // 4).astype(int).ravel()
        order = np.argsort(rb, kind="stable")
        fs, ss, ls = rb[order], s.ravel()[order], lab.ravel()[order]
        starts = np.searchsorted(fs, np.arange(nb + 1))
        med = np.zeros(nb)
        spk = np.zeros(nb)
        for kb in range(3, nb):
            seg, lb = ss[starts[kb]:starts[kb + 1]], ls[starts[kb]:starts[kb + 1]]
            if seg.size < 20:
                continue
            med[kb] = np.median(seg[lb == 0]) if (lb == 0).sum() > 10 else np.median(seg)
            ex = [np.median(seg[lb == k + 1]) - np.median(seg[lb == -(k + 1)]) for k in range(len(dirs))
                  if (lb == k + 1).sum() >= 4 and (lb == -(k + 1)).sum() >= 8]
            if ex:
                spk[kb] = np.mean(ex)

        def first_below(v, thr=0.25):
            v = median_filter(v, size=5, mode="nearest")
            b = np.where(v[3:] < thr)[0]
            return 4 * (int(b[0]) + 3) if b.size else 4 * len(v)
        # the halo is measured against the layer's own level far out (outer third of the window): the star layer
        # holds unresolved light and the starless image's offsets, which do not belong to this star
        ref = np.median(med[max(4, 2 * nb // 3):]) if nb > 12 else 0.0
        out[int(i)] = (first_below(med - ref), first_below(spk) if dirs else 0)
    return out


def fit_extent(ext, g):
    """R = 1.15 x max(halo, spikes) radius; robust log-linear fit of R against G. Returns (a, b) or None."""
    idx = np.array(list(ext.keys()))
    if idx.size < 5:
        return None
    G = g[idx]
    R = 1.15 * np.array([max(v) for v in ext.values()], float)
    ok = R > 12
    G, R = G[ok], np.log(R[ok])
    keep = np.ones(G.size, bool)
    for _ in range(3):
        b, a = np.polyfit(G[keep], R[keep], 1)
        res = R - (a + b * G)
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
        keep = np.abs(res) < 2.5 * s
    if b >= 0 or keep.sum() < 5:
        return None
    return float(a), float(b)


def down4(a):
    """(3, H, W) -> (3, H/4, W/4) block means."""
    c, h, w = a.shape
    h4, w4 = h // 4, w // 4
    return np.asarray(a[:, :h4 * 4, :w4 * 4], np.float32).reshape(c, h4, 4, w4, 4).mean((2, 4))


def box_scores(lum, zone, size, step, margin):
    """Mean, std of lum and zone fraction over every size x size box (top-left on a step grid), via integral
    images. Arrays at the same (reduced) scale."""
    def integ(a):
        return np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    I1, I2, IZ = integ(lum.astype(np.float64)), integ(lum.astype(np.float64) ** 2), integ(zone.astype(np.float64))
    h, w = lum.shape
    ys = np.arange(margin, h - size - margin + 1, step)
    xs = np.arange(margin, w - size - margin + 1, step)
    Y, X = np.meshgrid(ys, xs, indexing="ij")

    def bsum(I):
        return I[Y + size, X + size] - I[Y, X + size] - I[Y + size, X] + I[Y, X]
    n = size * size
    m = bsum(I1) / n
    sd = np.sqrt(np.maximum(bsum(I2) / n - m ** 2, 0))
    return Y, X, m, sd, bsum(IZ) / n


def nebula_colour(S4, lum4, bk, sg, x, y, g, R_all, H, W):
    """Colour of the frame's brightest extended nebula (for the brightness augmentation): median colour (noise
    units) of the brightest 0.5 % of the smoothed luminance at 1/4 scale, away from the filled star cores and from
    the inner half of the extent of the stars brighter than G 10.5. Returns (colour / its mean, description)."""
    inner4 = np.zeros(lum4.shape, np.uint8)
    for i in np.where(g < 10.5)[0]:
        cv2.circle(inner4, (int(round(x[i])) // 4, int(round(y[i])) // 4), int(0.5 * R_all[i] / 4) + 1, 1, -1)
    core4 = down4(np.load(PROJ / "starless/sl_mask.npy", mmap_mode="r")[None].astype(np.float32))[0] > 0
    core4 = cv2.dilate(core4.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    lumc = cv2.GaussianBlur(lum4, (0, 0), 4)
    okc = ~core4 & (inner4 == 0)
    selc = okc & (lumc >= np.percentile(lumc[okc], 99.5))
    col = np.array([np.median((S4[c][selc] - bk[c]) / sg[c]) for c in range(3)])
    yy_, xx_ = np.nonzero(selc)
    where = f"level {float(np.median(lumc[selc])):.1f} sigma around ({4 * xx_.mean():.0f},{4 * yy_.mean():.0f})"
    return np.clip(col / max(col.mean(), 1e-6), 0.2, 5.0).astype(np.float32), where


def membrane_stamps(args):
    """Bright stamps against a background that does not inherit the starless image's errors under the star: inside
    the star's measured extent (1.15 x the larger of its halo and spike radii; the fitted law for the stars that
    were not measured) the nebula is interpolated from the ring just outside it (membrane / Laplace fill of
    sl_final), and the stamp is the denoised stack minus the far halo minus that background. The window ends 25 %
    (at least 16 px) beyond the extent: the starless image's errors further out (rays around the brightest stars)
    stay out of the stamp. Faint stars inside the stamp stay in it; ring level removal, taper and clipping as before.
    The faint stamps are those of library_nohalo.pt. Writes library_nohalo_membrane.pt.

    --stamp-variant agreed (library_nohalo_agreed.pt): the stamp is the star as the input shows it, without the
    nebula. Bright part: light above the higher of the two background estimates (the interpolated one and sl_final,
    a model-based nebula estimate) outside sl_final's filled cores (dilated 4 px), where it is itself an
    interpolation, and above the interpolated one inside them. Dark part (inside the extent): where the input is
    below the interpolated background at a pixel AND at its mirror through the star (point symmetry: the star's
    optics and the far-halo model are point-symmetric, nebula lanes are not), the smaller darkness of the two. Run F
    showed why: the network dug a crater (-20 sigma) at the brightest stars, its reaction to the dark rings and
    lobes that the far-halo subtraction leaves around them and that no clipped stamp ever contained.

    --stamp-variant compact (library_nohalo_compact.pt, run H): the agreed stamp split into the star's compact light
    and its diffuse light. Compact: everything within a radius and, beyond it, what is narrow across (spikes, rays,
    the faint stars inside the stamp: the stamp minus its median over +-90 deg of arc and +-2 px of radius). The
    radius: for a saturated star (shown at rendering by its own light, faded out below a few sigmas of the smoothed
    noise), where its broad light falls below the pixel noise, at least its saturated core + 4 px; for the others
    the radius the star model holds (60 px, as the far halo sees it). Diffuse: the rest beyond that radius (fully
    from 1.3 x it), the glare the far halo model did not take. The network removes
    the compact light only: the diffuse light is pasted into the input and the target alike (the rendering shows a
    saturated star by its own light inside a footprint; diffuse light the network removed beyond it was lost)."""
    from astro_gpu.core.imgutil import membrane_fill
    lg = PREP / "prep.log"
    meta = json.load(open(PREP / "meta.json"))
    set_radius(meta)
    L, S = load("optics/lat_full.npy"), load("starless/sl_final.npy")
    NL = np.load(PREP / "n2n_lat.npy", mmap_mode="r")
    hf = PROJ / "stars/stars_model_halo.npy"
    halo = np.load(hf, mmap_mode="r") if hf.exists() else None
    H, W = L.shape[1:]
    d = np.load(PROJ / "stars/star_fluxes.npz")
    x, y, g = d["x"], d["y"], d["g"]
    ext = {int(k): v for k, v in meta.get("star_extent", {}).items()}
    sat99 = np.array(meta["sat99"], np.float32)
    old = torch.load(PREP / lib_name(True), weights_only=False)
    agreed = args.stamp_variant in ("agreed", "compact")
    compact = args.stamp_variant == "compact"
    FM = np.load(PROJ / "starless/sl_mask.npy", mmap_mode="r") if agreed else None
    if compact:                 # the radius the star model holds: the saturated core as the far halo sees it
        from astro_gpu.stages.starless import _filled_radius
        satm = np.zeros((H, W), bool)
        lvl = np.array([np.percentile(np.asarray(L[c, ::4, ::4]), 99.999) for c in range(3)])
        for c in range(3):
            satm |= np.asarray(L[c], np.float32) > 0.7 * lvl[c]
    bright = []
    t0 = time.time()
    for i in np.argsort(g):
        if g[i] >= 11:
            break
        xi, yi = int(round(x[i])), int(round(y[i]))
        R_fill = max(12.0, 1.15 * max(ext[i]) if i in ext else float(radius_of(g[i])))
        R = int(round(R_fill + max(16.0, 0.25 * R_fill)))
        Re = min(R, xi, yi, W - 1 - xi, H - 1 - yi)
        if Re < 0.75 * R or Re < 10:
            continue
        brighter = g < g[i]
        dd = np.hypot(x[brighter] - x[i], y[brighter] - y[i])
        if dd.size and dd.min() < 60:
            continue
        sl = (slice(None), slice(yi - Re, yi + Re + 1), slice(xi - Re, xi + Re + 1))
        nl = np.asarray(NL[sl], np.float32) - (np.asarray(halo[sl], np.float32) if halo is not None else 0)
        s_ = np.asarray(S[sl], np.float32)
        w, r = taper(Re)
        hole = r * Re < min(R_fill, 0.85 * Re)
        bg = np.stack([membrane_fill(s_[c], hole) for c in range(3)])
        win = nl - bg
        ring = (r >= 0.85) & (r <= 1.0)
        ped = np.median(win[:, ring], axis=1)
        if agreed:
            filled = cv2.dilate(np.asarray(FM[sl[1:]]).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            bright_bg = np.where(filled[None], bg, np.maximum(bg, s_))
            pos = np.clip(nl - bright_bg - ped[:, None, None], 0, None)
            neg = np.clip(win - ped[:, None, None], None, 0)
            neg = np.where(hole[None], np.maximum(neg, neg[:, ::-1, ::-1]), 0)
            st = ((pos + neg) * w).astype(np.float32)
        else:
            st = np.clip((win - ped[:, None, None]) * w, 0, None).astype(np.float32)
        core = np.asarray(L[:, yi - 4:yi + 5, xi - 4:xi + 5]).reshape(3, -1).max(1)
        # saturated as the rendering sees it (3 x 3 max above 0.7 x the 99.999th percentile): Alnitak's plateau and
        # most of the Rosette's saturated stars stood below 0.75 x that percentile and passed for unsaturated
        sat_ = (np.asarray(L[:, yi - 1:yi + 2, xi - 1:xi + 2]).reshape(3, -1).max(1) > 0.7 * lvl if compact
                else core >= 0.75 * sat99)
        rec = dict(x=float(x[i]), y=float(y[i]), g=float(g[i]), R=Re, fill=float(R_fill),
                   sat=sat_, plateau=core.astype(np.float32), ped=ped.astype(np.float32), t=st)
        if compact:
            # saturated: shown at rendering by its own light, faded out below the noise - compact out to where its
            # broad light falls below the pixel noise (at least its saturated core); else the star model's radius
            r_sat = _filled_radius(satm, x[i], y[i], rmax=800) + 4.0
            r_c = max(r_sat, bright_radius(st, meta["sg"])) if sat_.any() else max(60.0, r_sat)
            dif = diffuse_part(st, r_c)
            rec.update(t=(st - dif).astype(np.float32), d=dif.astype(np.float32), r_c=r_c)
        bright.append(rec)
    variant = args.stamp_variant
    torch.save({"bright": bright, "faint": old["faint"]}, PREP / lib_name(True, variant))
    if compact:
        log(lg, "compact radius of the saturated stamps: " + ", ".join(
            f"G{b['g']:.1f} {b['r_c']:.0f} px" for b in bright if b["sat"].any()))
        fr = [float(np.abs(b["d"]).sum() / max(np.abs(b["t"]).sum() + np.abs(b["d"]).sum(), 1e-9)) for b in bright]
        log(lg, f"compact stamps: diffuse light {100 * np.median(fr):.1f} % of a stamp's light (median; max "
                f"{100 * max(fr):.1f} %), compact radius {min(b['r_c'] for b in bright):.0f}-"
                f"{max(b['r_c'] for b in bright):.0f} px")
    log(lg, f"{variant} stamps: {len(bright)} bright stamps (radius {min(b['R'] for b in bright)}-"
            f"{max(b['R'] for b in bright)} px) against an interpolated background in {time.time() - t0:.0f} s; "
            f"{len(old['faint'])} faint stamps kept")


def bright_radius(st, sg, r0=4):
    """The radius where a stamp's broad light (median of each 1 px ring) first falls below the pixel noise (sg per
    channel, the mean over the channels in noise units): inside, the star's light stands above the noise; beyond,
    its faint glare (the rendering fades a star's light out below a few sigmas of its smoothed noise, about one
    sigma of a pixel's: what the network removed there was lost)."""
    R = st.shape[-1] // 2
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rk = np.hypot(yy, xx).astype(int).ravel()
    sig = np.mean([st[c].ravel() / max(float(sg[c]), 1e-9) for c in range(st.shape[0])], 0)
    order = np.argsort(rk, kind="stable")
    rs, vs = rk[order], sig[order]
    bounds = np.searchsorted(rs, np.arange(R + 2))
    for k in range(r0, R):
        if np.median(vs[bounds[k]:bounds[k + 1]]) < 1.0:
            return float(k)
    return float(R)


def diffuse_part(st, r_c, nt=120, arc_deg=180, rad_px=5):
    """The diffuse part of a stamp (3, 2R+1, 2R+1) beyond the compact radius r_c: its median over +-arc_deg/2 of arc
    and +-rad_px/2 of radius (broad across: not a spike, a ray nor a star - over half a ring: over 45 degrees the glare of
    sigma Orionis' companions passed for diffuse light), weighted 0 inside r_c, 1 beyond 1.3 r_c
    (cosine in between)."""
    R = st.shape[-1] // 2
    rr = np.arange(R + 1, dtype=np.float32)[:, None]
    th = (np.arange(nt) * 2 * np.pi / nt).astype(np.float32)[None]
    px, py = (R + rr * np.cos(th)).astype(np.float32), (R + rr * np.sin(th)).astype(np.float32)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    r = np.hypot(xx, yy)
    t = ((np.arctan2(yy, xx) % (2 * np.pi)) / (2 * np.pi) * nt).astype(np.float32)
    rb = np.minimum(r, R).astype(np.float32)
    k = int(round(arc_deg / 360 * nt)) | 1
    out = np.zeros_like(st)
    for c in range(st.shape[0]):
        pol = cv2.remap(np.ascontiguousarray(st[c]), px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        hk = k // 2                                        # periodic in angle
        med = median_filter(np.concatenate([pol[:, -hk:], pol, pol[:, :hk]], 1), size=(rad_px, k),
                            mode="nearest")[:, hk:hk + nt]
        med = np.concatenate([med, med[:, :1]], 1)
        out[c] = cv2.remap(np.ascontiguousarray(med), t, rb, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wt = 0.5 * (1 - np.cos(np.pi * np.clip((r - r_c) / (0.3 * r_c), 0, 1)))
    return (out * wt[None]).astype(np.float32)


def prep(args):
    PREP.mkdir(parents=True, exist_ok=True)
    lg = PREP / "prep.log"
    log(lg, f"project {PROJ.parent} -> {OUT}")
    L = load("optics/lat_full.npy")
    S = load("starless/sl_final.npy")
    H, W = L.shape[1:]
    nl_path = PREP / "n2n_lat.npy"
    if not nl_path.exists():      # the N2N-denoised stack, realigned in colour exactly like lat_full
        grid = np.array(json.load(open(PROJ / "optics/lateral.json")), np.float32)
        np.save(nl_path, apply_lateral(np.load(PROJ / "denoise/n2n_full.npy").astype(np.float32), grid))
    NL = np.load(nl_path, mmap_mode="r")
    A, B = load("optics/lat_A.npy"), load("optics/lat_B.npy")

    # normalisation: sky level of the starless image, pixel noise of the stack from the two half-stacks
    bk = np.array([np.median(np.asarray(S[c, ::4, ::4])) for c in range(3)], np.float32)
    sg = []
    for c in range(3):
        n = (np.asarray(A[c, ::4, ::4]) - np.asarray(B[c, ::4, ::4])) / 2
        sg.append(1.4826 * np.median(np.abs(n - np.median(n))))
    sg = np.array(sg, np.float32)
    sat99 = np.array([np.percentile(np.asarray(L[c]), 99.999) for c in range(3)], np.float32)
    log(lg, f"sky {bk.round(2)}  noise {sg.round(3)}  saturation (99.999th pct) {sat99.round(0)}")

    d = np.load(PROJ / "stars/star_fluxes.npz")
    x, y, g = d["x"], d["y"], d["g"]

    # photon noise gain: var(A-B)/2 = a + b * signal + c * signal^2 (the c term is the PSF difference between the
    # half-stacks, not noise of the full stack); b is used to give the pasted stars their photon noise
    sel = np.where((g > 10) & (g < 13) & (x > 50) & (y > 50) & (x < W - 50) & (y < H - 50))[0][:1500]
    gain = []
    for c in range(3):
        ss, vv = [], []
        for i in sel:
            xi, yi = int(round(x[i])), int(round(y[i]))
            s_ = (c, slice(yi - 6, yi + 7), slice(xi - 6, xi + 7))
            ss.append((np.asarray(NL[s_]) - np.asarray(S[s_])).ravel())
            vv.append((((np.asarray(A[s_]) - np.asarray(B[s_])) / 2) ** 2).ravel())
        s, v = np.concatenate(ss), np.concatenate(vv)
        m = s < 2500
        from scipy.optimize import nnls
        coef = nnls(np.c_[np.ones(m.sum()), s[m], s[m] ** 2], v[m])[0]
        gain.append(max(float(coef[1]), 1e-4))
        log(lg, f"  channel {c}: var = {coef[0]:.3f} + {coef[1]:.4f} s + {coef[2]:.2e} s^2")
    gain = np.array(gain, np.float32)
    log(lg, f"photon noise gain (var per ADU of star light) {gain.round(4)}")
    if args.minus_halo:      # hybrid variant: the stamps are cut after the measured far halo is removed
        NL = np.asarray(NL, np.float32) - np.load(PROJ / "stars/stars_model_halo.npy")

    # ---- how far the light of the stars reaches (halo and spikes), measured on the frame's own bright stars
    global RADIUS_FIT
    sman = json.load(open(PROJ / "starless/manifest.json"))
    dirs = [float(t) for t in sman.get("metrics", {}).get("spike_directions", [])]
    S_mm = np.load(PROJ / "starless/sl_final.npy", mmap_mode="r")

    class _Layer:      # star layer a - b, read window by window
        def __init__(self, a, b):
            self.a, self.b = a, b

        def __getitem__(self, sl):
            return np.asarray(self.a[sl], np.float32) - np.asarray(self.b[sl], np.float32)
    ext = measure_extent(_Layer(NL, S_mm), sg, x, y, g, dirs, H, W)
    RADIUS_FIT = fit_extent(ext, g)
    log(lg, f"spike directions {dirs}; star extent measured on {len(ext)} bright stars; law R = exp(a + b G): "
            f"{RADIUS_FIT}" + ("" if RADIUS_FIT is None else
                               " -> " + ", ".join(f"G{gg} {float(radius_of(gg)):.0f} px" for gg in (3, 5, 7, 9, 11, 13))))
    for i, (rh, rs) in list(ext.items())[:12]:
        log(lg, f"  G {g[i]:5.2f} ({x[i]:.0f},{y[i]:.0f}): halo {rh} px, spikes {rs} px, law {float(radius_of(g[i])):.0f}")
    star_R = {i: float(np.clip(max(float(radius_of(g[i])), 1.15 * max(v)), 12, 600)) for i, v in ext.items()}
    R_all = radius_of(g)
    for i, r_ in star_R.items():
        R_all[i] = r_
    R_all = R_all.astype(int)

    # ---- stamp libraries
    bright, faint = [], []
    order = np.argsort(g)
    for i in order:
        if g[i] >= 15:
            break
        xi, yi = int(round(x[i])), int(round(y[i]))
        R = int(R_all[i])
        Re = min(R, xi, yi, W - 1 - xi, H - 1 - yi)
        if Re < 0.75 * R or Re < 10:
            continue
        brighter = g < g[i]
        dd = np.hypot(x[brighter] - x[i], y[brighter] - y[i])
        iso = 60 if g[i] < 11 else 30
        if dd.size and dd.min() < iso:
            continue
        if g[i] >= 11:      # faint stamps must not hold part of a bright star
            bb = g < 11
            if np.any(np.hypot(x[bb] - x[i], y[bb] - y[i]) < R_all[bb] + Re):
                continue
        sl = (slice(None), slice(yi - Re, yi + Re + 1), slice(xi - Re, xi + Re + 1))
        win = np.asarray(NL[sl], np.float32) - np.asarray(S[sl], np.float32)
        w, r = taper(Re)
        ring = (r >= 0.85) & (r <= 1.0)
        ped = np.median(win[:, ring], axis=1)
        st = np.clip((win - ped[:, None, None]) * w, 0, None).astype(np.float32)
        core = np.asarray(L[:, yi - 4:yi + 5, xi - 4:xi + 5]).reshape(3, -1).max(1)
        sat = core >= 0.75 * sat99
        rec = dict(x=float(x[i]), y=float(y[i]), g=float(g[i]), R=Re, sat=sat, plateau=core.astype(np.float32),
                   ped=ped.astype(np.float32), t=st.astype(np.float16) if g[i] >= 11 else st)
        (bright if g[i] < 11 else faint).append(rec)
    rng = np.random.default_rng(0)
    if len(faint) > 2000:
        faint = [faint[k] for k in sorted(rng.choice(len(faint), 2000, replace=False))]
    log(lg, f"stamps: {len(bright)} bright (G<11, {sum(bool(b['sat'].any()) for b in bright)} saturated, radius "
            f"{min([b['R'] for b in bright] or [0])}-{max([b['R'] for b in bright] or [0])} px), "
            f"{len(faint)} faint (G 11-15)")
    torch.save({"bright": bright, "faint": faint}, PREP / lib_name(args.minus_halo))
    del NL

    # ---- reliability of the target (sl_final): not around the bright stars
    wm = np.ones((H, W), np.uint8)
    zone4 = np.zeros((H // 4, W // 4), np.uint8)
    for i in np.where(g < 10.5)[0]:
        rz = max(150.0, 1.2 * R_all[i]) if g[i] < 10 else 1.2 * R_all[i]
        c_ = (int(round(x[i])), int(round(y[i])))
        cv2.circle(wm, c_, int(rz), 0, -1)
        cv2.circle(zone4, (c_[0] // 4, c_[1] // 4), int((rz + 30) / 4), 1, -1)
    wm[:16], wm[-16:], wm[:, :16], wm[:, -16:] = 0, 0, 0, 0

    # luminance of the starless image in noise units, at 1/4 scale
    S4 = down4(S)
    lum4 = ((S4 - bk[:, None, None]) / sg[:, None, None]).mean(0)
    lumb = cv2.GaussianBlur(lum4, (0, 0), 2)

    # hold-out box 1, "structured": 384 px with the most nebula structure (std of the luminance), outside the
    # bright-star zones; box 2, "faint": 512 px among the faintest quarter of the admissible boxes, most structure,
    # apart from box 1. Neither is ever a training target.
    def pick(size, admissible_extra=None, faint=False):
        Y, X, mean_, sd_, zf = box_scores(lumb, zone4, size // 4, 16, 16)
        ok_ = zf == 0
        if not ok_.any():
            ok_ = zf <= np.percentile(zf, 5)
        if admissible_extra is not None:
            ok_ &= admissible_extra(Y, X)
        if faint and ok_.any():
            ok_ &= mean_ <= np.percentile(mean_[ok_], 25)
        if not ok_.any():
            raise RuntimeError(f"no admissible {size} px hold-out box")
        k_ = np.argmax(np.where(ok_, sd_, -1))
        y0_, x0_ = int(Y.flat[k_]) * 4, int(X.flat[k_]) * 4
        return (x0_, y0_, x0_ + size, y0_ + size), float(mean_.flat[k_]), float(sd_.flat[k_])
    sbox, s_lev, s_sd = pick(384)

    def apart(Y, X):
        x0_, y0_, x1_, y1_ = sbox
        bx0, by0 = X * 4, Y * 4
        return (bx0 + 512 < x0_ - 256) | (bx0 > x1_ + 256) | (by0 + 512 < y0_ - 256) | (by0 > y1_ + 256)
    fbox, f_lev, f_sd = pick(512, apart, faint=True)
    log(lg, f"structured hold-out box {sbox} (level {s_lev:.1f}, structure {s_sd:.2f} noise sigma); faint hold-out "
            f"box {fbox} (level {f_lev:.1f}, structure {f_sd:.2f})")
    for (x0, y0, x1, y1) in (sbox, fbox):
        wm[y0:y1, x0:x1] = 0
    np.save(PREP / "weight_map.npy", wm)

    neb_col, where = nebula_colour(S4, lum4, bk, sg, x, y, g, R_all, H, W)
    log(lg, f"brightest extended nebula: {where}; colour (R, G, B relative) {neb_col.round(3)}")

    # ---- evaluation stars: the 4 brightest with room around them (300 px from the edges, no brighter star within
    # 150 px), and 4 medium stars G 9-11 with no brighter star within 250 px; none inside a hold-out box
    def inside_boxes(i, pad=50):
        return any(b[0] - pad <= x[i] <= b[2] + pad and b[1] - pad <= y[i] <= b[3] + pad for b in (sbox, fbox))

    def roomy(i, dmin):
        if not (300 <= x[i] <= W - 300 and 300 <= y[i] <= H - 300) or inside_boxes(i):
            return False
        br = g < g[i]
        return not (br.any() and np.hypot(x[br] - x[i], y[br] - y[i]).min() < dmin)
    brightest = [int(i) for i in np.argsort(g) if roomy(i, 150)][:4]
    ok = [int(i) for i in np.where((g > 9) & (g < 11))[0] if roomy(i, 250) and int(i) not in brightest]
    med = []
    for gt in (9.3, 9.9, 10.4, 10.9):
        cands = [k for k in ok if k not in med]
        if cands:
            med.append(min(cands, key=lambda k: abs(g[k] - gt)))
    eval_stars = [dict(x=float(x[i]), y=float(y[i]), g=float(g[i])) for i in brightest + med]
    log(lg, "evaluation stars: " + ", ".join(f"({s['x']:.0f},{s['y']:.0f}) G{s['g']:.1f}" for s in eval_stars))
    meta = dict(project=str(PROJ.parent), bk=bk.tolist(), sg=sg.tolist(), sat99=sat99.tolist(), gain=gain.tolist(),
                struct_box=list(sbox), faint_box=list(fbox), eval_stars=eval_stars, n_bright_eval=len(brightest),
                neb_col=neb_col.tolist(), radius_fit=RADIUS_FIT, spike_directions=dirs,
                star_extent={str(i): [int(v[0]), int(v[1])] for i, v in ext.items()}, H=H, W=W)
    json.dump(meta, open(PREP / "meta.json", "w"), indent=1)


# ---------------------------------------------------------------------------------------------------- synthesis

class Synth:
    """Pastes real stamps into (3, P, P) GPU patches."""

    def __init__(self, meta, lib, dev, hold_eval=False):
        self.dev = dev
        self.sat99 = torch.tensor(meta["sat99"], device=dev)
        self.gain = torch.tensor(meta["gain"], device=dev)
        self.sg = torch.tensor(meta["sg"], device=dev)
        br = lib["bright"]
        if hold_eval:     # leave out every stamp that holds an evaluation star
            ev = meta["eval_stars"]
            br = [b for b in br if all(math.hypot(b["x"] - e["x"], b["y"] - e["y"]) > b["R"] + 60 for e in ev)]
        self.bright = [dict(b, t=torch.from_numpy(np.asarray(b["t"], np.float32)).to(dev),
                            d=None if b.get("d") is None else torch.from_numpy(np.asarray(b["d"], np.float32)).to(dev))
                       for b in br]
        self.faint = [dict(f, t=torch.from_numpy(np.asarray(f["t"], np.float32)).to(dev)) for f in lib["faint"]]
        gb = np.array([b["g"] for b in self.bright])
        w = 10 ** (-0.15 * (gb - 11))
        self.pb = w / w.sum()
        self.n_bright = len(self.bright)

    def plan(self, rng, P, n_bright=None, idx=None):
        plan = []
        nb = rng.integers(1, 7) if n_bright is None else n_bright
        for k in range(nb):
            i = int(rng.choice(self.n_bright, p=self.pb)) if idx is None else idx[k]
            b = self.bright[i]
            R = b["R"]
            sat = bool(b["sat"].any())
            s = rng.uniform(1.0, 2.0) if sat else rng.uniform(0.3, 2.0)
            m = 0.3 * R
            cy, cx = (rng.uniform(-m, P + m, 2)).astype(int)
            col = np.exp(rng.normal(0, 0.08, 3))
            clip = np.where(b["sat"], b["plateau"], np.array(self.sat99.tolist()) * rng.uniform(1.0, 1.35))
            plan.append(("b", i, cy, cx, s, col, clip))
        for _ in range(rng.integers(0, 25) if n_bright is None else 0):
            i = int(rng.integers(len(self.faint)))
            cy, cx = rng.integers(0, P, 2)
            plan.append(("f", i, cy, cx, rng.uniform(0.3, 2.0), np.exp(rng.normal(0, 0.08, 3)), None))
        return plan

    def apply(self, X, plan, T=None):
        """X (3, P, P) in ADU, modified in place. Returns the pasted light (3, P, P). A stamp's diffuse part (compact
        stamps) is pasted into the target T too (modified in place), when given."""
        P = X.shape[-1]
        light = torch.zeros_like(X)
        clipped = torch.zeros_like(X, dtype=torch.bool)
        for kind, i, cy, cx, s, col, clip in plan:
            st = (self.bright if kind == "b" else self.faint)[i]
            t = st["t"]
            R = t.shape[-1] // 2
            y0, x0 = cy - R, cx - R
            ys0, xs0 = max(0, -y0), max(0, -x0)
            ys1, xs1 = min(t.shape[-2], P - y0), min(t.shape[-1], P - x0)
            if ys1 <= ys0 or xs1 <= xs0:
                continue
            k_ = torch.tensor(s * col, device=self.dev, dtype=torch.float32).view(3, 1, 1)
            sub = t[:, ys0:ys1, xs0:xs1] * k_
            sy, sx = slice(y0 + ys0, y0 + ys1), slice(x0 + xs0, x0 + xs1)
            dd = st.get("d")
            if dd is not None:
                dsub = dd[:, ys0:ys1, xs0:xs1] * k_
                sub = sub + dsub
                if T is not None:
                    T[:, sy, sx] += dsub
            reg = X[:, sy, sx]
            new = reg + sub
            if clip is not None:
                lim = torch.maximum(torch.tensor(clip, device=self.dev, dtype=torch.float32).view(3, 1, 1), reg)
                cl = (new > lim) & (sub > 0)
                new = torch.where(cl, lim, new)
                clipped[:, sy, sx] |= cl
            light[:, sy, sx] += new - reg
            X[:, sy, sx] = new
        noise = torch.randn_like(X) * torch.sqrt(self.gain.view(3, 1, 1) * light.clamp(min=0))
        X += torch.where(clipped, torch.zeros_like(noise), noise)
        return light


def normalise(x, bk, sg):
    return torch.asinh((x - bk.view(1, 3, 1, 1)) / (K_ASINH * sg.view(1, 3, 1, 1)))


def denormalise(n, bk, sg):
    return torch.sinh(n) * (K_ASINH * sg.view(1, 3, 1, 1)) + bk.view(1, 3, 1, 1)


def make_net(dev):
    return UNet(cin=3, cout=3, w=WIDTHS).to(dev)


def forward(net, x, bk, sg):
    """Linear ADU in -> starless linear ADU out (residual in the asinh space)."""
    n = normalise(x, bk, sg)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        o = net(n).float()
    return n + o


# --------------------------------------------------------------------------------------------------------- train

def val_set(meta, synth, L, S, rng):
    """Fixed validation patches (256 px) inside the hold-out boxes, with and without pasted bright stars."""
    items = []
    P = 256
    for (x0, y0, x1, y1) in boxes(meta):
        for cy in np.linspace(y0 + 128, y1 - 128, 2).astype(int):
            for cx in np.linspace(x0 + 128, x1 - 128, 2 if x1 - x0 > 300 else 1).astype(int):
                ya, xa = cy - P // 2, cx - P // 2
                w = np.zeros((P, P), np.float32)
                w[max(0, y0 - ya):min(P, y1 - ya), max(0, x0 - xa):min(P, x1 - xa)] = 1
                items.append((ya, xa, w, synth.plan(rng, P, n_bright=2)))
    return items


def run_val(net, items, synth, L, S, bk, sg, dev):
    net.eval()
    tot, tot0, n = 0.0, 0.0, 0
    with torch.no_grad():
        for ya, xa, w, plan in items:
            X = torch.from_numpy(np.ascontiguousarray(L[:, ya:ya + 256, xa:xa + 256])).to(dev)
            T = torch.from_numpy(np.ascontiguousarray(S[:, ya:ya + 256, xa:xa + 256])).to(dev)[None]
            wt = torch.from_numpy(w).to(dev)[None, None]
            nt0 = normalise(T, bk, sg)
            out0 = forward(net, X[None], bk, sg)
            torch.manual_seed(0)
            synth.apply(X, plan, T[0])
            nt = normalise(T, bk, sg)
            out = forward(net, X[None], bk, sg)
            tot += float(((out - nt).abs() * wt).sum() / (3 * wt.sum()))
            tot0 += float(((out0 - nt0).abs() * wt).sum() / (3 * wt.sum()))
            n += 1
    net.train()
    return tot / n, tot0 / n


def nebula_boost(Dn, bk, sg, neb_col, valid=None, p=0.5, kmax=40.0):
    """Extra nebula light for a batch of starless patches Dn (B, 3, P, P): smoothed luminance above the sky (noise
    units, 6 px Gaussian, so that star-sized defects of the starless image are not amplified) x (k - 1), k
    log-uniform in 1..kmax for a fraction p of the patches (0 for the others),
    coloured like neb_col with 15 % jitter; `valid` (B, 1, P, P) = 0 on the filled star cores of the starless
    image, which are interpolated over (normalised convolution). Returns the added light in ADU."""
    dev, nb = Dn.device, Dn.shape[0]
    lum = ((Dn - bk.view(1, 3, 1, 1)) / sg.view(1, 3, 1, 1)).mean(1, keepdim=True)
    gk = torch.exp(-0.5 * (torch.arange(-18, 19, device=dev, dtype=torch.float32) / 6.0) ** 2)
    gk = gk / gk.sum()
    if valid is None:
        valid = torch.ones_like(lum)

    def blur(a):
        a = F.conv2d(F.pad(a, (18, 18, 0, 0), mode="reflect"), gk.view(1, 1, 1, 37))
        return F.conv2d(F.pad(a, (0, 0, 18, 18), mode="reflect"), gk.view(1, 1, 37, 1))
    den = blur(valid)
    lum = torch.where(den > 0.05, blur(lum * valid) / den.clamp(min=0.05), torch.zeros_like(den)).clamp(min=0)
    col = torch.as_tensor(neb_col, device=dev).view(1, 3, 1, 1) * torch.exp(0.15 * torch.randn(nb, 3, 1, 1, device=dev))
    kk = torch.exp(torch.rand(nb, 1, 1, 1, device=dev) * math.log(kmax))
    amp = torch.where(torch.rand(nb, 1, 1, 1, device=dev) < p, kk - 1, torch.zeros_like(kk))
    return amp * lum * col * sg.view(1, 3, 1, 1)


def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rd = OUT / args.run
    rd.mkdir(parents=True, exist_ok=True)
    lg = rd / "train.log"
    meta = json.load(open(PREP / "meta.json"))
    set_radius(meta)
    lib = torch.load(PREP / lib_name(args.minus_halo, args.stamps), weights_only=False)
    synth = Synth(meta, lib, dev, hold_eval=args.hold_eval)
    log(lg, f"run {args.run}: {synth.n_bright} bright stamps (hold-out of evaluation stars: {args.hold_eval}), "
            f"{len(synth.faint)} faint stamps, far halo removed first: {args.minus_halo}, noise-preserving target: "
            f"{args.noisy_target}, nebula brightness augmentation: {args.bright_aug}, star cores out of the loss: "
            f"{args.core_mask}, stamps: {args.stamps or 'starless image'}, "
            f"budget {args.minutes} min, "
            f"patch {args.patch}, batch {args.batch}")
    t_load = time.time()
    L = load_input(args.minus_halo)
    S = np.load(PROJ / "starless/sl_final.npy")
    S_den = np.load(PROJ / "starless/sl_final.npy") if args.bright_aug else None
    VAL = None
    if S_den is not None:   # 1 outside the filled star cores (dilated 3 px)
        VAL = (cv2.dilate(np.load(PROJ / "starless/sl_mask.npy").astype(np.uint8), np.ones((7, 7), np.uint8)) == 0)
    neb_col = None
    if S_den is not None:   # colour of the frame's brightest extended nebula (measured by prep; older metas: now)
        neb_col = np.asarray(meta["neb_col"], np.float32) if meta.get("neb_col") else None
        if neb_col is None:
            cat = np.load(PROJ / "stars/star_fluxes.npz")
            neb_col, where = nebula_colour(down4(S_den), ((down4(S_den) - np.array(meta["bk"])[:, None, None]) /
                                                          np.array(meta["sg"])[:, None, None]).mean(0),
                                           np.array(meta["bk"]), np.array(meta["sg"]), cat["x"], cat["y"], cat["g"],
                                           radius_of(cat["g"]), *S_den.shape[1:])
            log(lg, f"brightest extended nebula: {where}")
        log(lg, f"brightness augmentation colour (R, G, B relative): {neb_col.round(3)}")
    WM = np.load(PREP / "weight_map.npy")
    d = np.load(PROJ / "stars/star_fluxes.npz")
    cm = None
    if args.core_mask:       # cores of the real stars: no loss there (the starless image fills them with defects and
        # the denoised stack is no faithful noise reference there: lat_full - n2n has a colour bias of ~3 sigma at the
        # cores of G 9-15 stars); core removal is then learnt from the pasted stamps only, whose target is true sky
        cm = cv2.dilate(np.load(PROJ / "starless/sl_mask.npy").astype(np.uint8), np.ones((7, 7), np.uint8))
        for xs, ys_, gs in zip(d["x"], d["y"], d["g"]):
            if gs < 16:
                cv2.circle(cm, (int(round(xs)), int(round(ys_))), int(max(4, 0.12 * radius_of(gs))), 1, -1)
        WM = WM * (1 - cm)
        log(lg, f"star cores masked out of the loss: {100 * cm.mean():.1f} % of the frame")
    if args.noisy_target:    # star removal only: the target keeps the input's own noise (lat_full - denoised stack)
        nzr = np.load(PROJ / "optics/lat_full.npy") - np.load(PREP / "n2n_lat.npy")
        if cm is not None:
            nzr *= (1 - cm)[None]
        S += nzr
        del nzr
    H, W = L.shape[1:]
    x, y, g = d["x"], d["y"], d["g"]
    P = args.patch
    # patch centres: >= 150 px from any star brighter than G 10, patch inside the image
    ok = np.ones((H, W), np.uint8)
    for i in np.where(g < 10)[0]:
        cv2.circle(ok, (int(round(x[i])), int(round(y[i]))), 150, 0, -1)
    ok[:P // 2 + 1], ok[H - P // 2 - 1:], ok[:, :P // 2 + 1], ok[:, W - P // 2 - 1:] = 0, 0, 0, 0
    cys, cxs = np.nonzero(ok[::4, ::4])
    cys, cxs = cys * 4, cxs * 4
    keep = WM[cys, cxs] > 0
    cys, cxs = cys[keep], cxs[keep]
    log(lg, f"data loaded in {time.time() - t_load:.0f} s; {len(cys)} admissible patch centres")
    bk = torch.tensor(meta["bk"], device=dev)
    sg = torch.tensor(meta["sg"], device=dev)

    net = make_net(dev)
    log(lg, f"UNet widths {WIDTHS}, {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    items = val_set(meta, synth, L, S, np.random.default_rng(123))
    budget = args.minutes * 60.0
    t0 = time.time()
    it, run_l = 0, None
    hist = []
    while True:
        el = time.time() - t0
        frac = el / budget
        if frac >= 1:
            break
        lr = args.lr * min(1.0, (it + 1) / 300) * 0.5 * (1 + math.cos(math.pi * frac))
        for pg in opt.param_groups:
            pg["lr"] = lr
        k = rng.integers(len(cys), size=args.batch)
        Xs, Ts, Ws, Ds = [], [], [], []
        for j in k:
            ya, xa = cys[j] - P // 2, cxs[j] - P // 2
            Xs.append(np.ascontiguousarray(L[:, ya:ya + P, xa:xa + P]))
            Ts.append(np.ascontiguousarray(S[:, ya:ya + P, xa:xa + P]))
            Ws.append(np.ascontiguousarray(WM[ya:ya + P, xa:xa + P]))
            if S_den is not None:
                Ds.append(np.ascontiguousarray(S_den[:, ya:ya + P, xa:xa + P]))
                Ds.append(np.ascontiguousarray(VAL[None, ya:ya + P, xa:xa + P]).astype(np.float32))
        X = torch.from_numpy(np.stack(Xs)).to(dev, non_blocking=True)
        T = torch.from_numpy(np.stack(Ts)).to(dev, non_blocking=True)
        Wt = torch.from_numpy(np.stack(Ws)).to(dev, non_blocking=True).float()[:, None]
        if S_den is not None:
            # this frame has almost no bright nebula outside the bright stars' zones: half of the patches get their
            # own nebula amplified k = 1..40 times (smoothed luminance above the sky, in noise units, 6 px Gaussian,
            # coloured like the frame's brightest nebula +-15 %), with its photon noise, in the input and in the
            # target alike (the target keeps that noise only when it keeps the input's)
            Dn = torch.from_numpy(np.stack(Ds[0::2])).to(dev)
            Vn = torch.from_numpy(np.stack(Ds[1::2])).to(dev)
            delta = nebula_boost(Dn, bk, sg, neb_col, Vn)
            nz = torch.randn_like(delta) * torch.sqrt(synth.gain.view(1, 3, 1, 1) * delta)
            X += delta + nz
            T += delta + (nz if args.noisy_target else 0)
            del Dn, Vn, delta, nz
        lights = []
        for b in range(args.batch):
            lights.append(synth.apply(X[b], synth.plan(rng, P), T[b]))
        light = torch.stack(lights)
        fp = ((light.abs() / sg.view(1, 3, 1, 1)).amax(1, keepdim=True) > 1.0).float()   # dark stamp parts too
        fp = F.max_pool2d(fp, 7, 1, 3)
        wt = Wt * (1 + 4 * fp)
        out = forward(net, X, bk, sg)
        nt = normalise(T, bk, sg)
        loss = ((out - nt).abs() * wt).sum() / (3 * wt.sum().clamp(min=1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        it += 1
        run_l = loss.item() if run_l is None else 0.98 * run_l + 0.02 * loss.item()
        if it % 100 == 0:
            msg = (f"step {it}  {el / 60:.1f}/{args.minutes} min  lr {lr:.2e}  loss {run_l:.4f}  "
                   f"gpu {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
            if it % 500 == 0:
                v, v0 = run_val(net, items, synth, L, S, bk, sg, dev)
                msg += f"  val(stars) {v:.4f}  val(no stars) {v0:.4f}"
                hist.append(dict(step=it, minutes=el / 60, loss=run_l, val=v, val0=v0))
            log(lg, msg)
        if it % 2000 == 0:
            torch.save({"net": net.state_dict(), "it": it, "minus_halo": args.minus_halo,
                        "noisy_target": args.noisy_target, "bright_aug": args.bright_aug, "core_mask": args.core_mask,
                        "stamps": args.stamps},
                       rd / "model.pt")
    v, v0 = run_val(net, items, synth, L, S, bk, sg, dev)
    hist.append(dict(step=it, minutes=(time.time() - t0) / 60, loss=run_l, val=v, val0=v0))
    torch.save({"net": net.state_dict(), "it": it, "hist": hist, "hold_eval": args.hold_eval,
                "minus_halo": args.minus_halo, "noisy_target": args.noisy_target, "bright_aug": args.bright_aug,
                "core_mask": args.core_mask, "stamps": args.stamps},
               rd / "model.pt")
    log(lg, f"done: {it} steps in {(time.time() - t0) / 60:.1f} min, loss {run_l:.4f}, val(stars) {v:.4f}, "
            f"val(no stars) {v0:.4f}")


# --------------------------------------------------------------------------------------------------------- infer

@torch.no_grad()
def apply_tiled(net, img, bk, sg, dev, tile=512, ov=128, batch=4):
    """img (3, H, W) linear -> starless (3, H, W) linear; reflect-padded, overlapping tiles, linear feathering."""
    H, W = img.shape[1:]
    pad = ov // 2
    step = tile - ov
    Hp = int(math.ceil((H + 2 * pad - tile) / step)) * step + tile
    Wp = int(math.ceil((W + 2 * pad - tile) / step)) * step + tile
    xp = np.pad(img, ((0, 0), (pad, Hp - H - pad), (pad, Wp - W - pad)), mode="reflect")
    num = np.zeros((3, Hp, Wp), np.float32)
    den = np.zeros((Hp, Wp), np.float32)
    r = np.minimum(np.arange(tile) + 0.5, tile - np.arange(tile) - 0.5)
    w1 = np.clip(r / ov, 1e-3, 1.0).astype(np.float32)
    wt = np.outer(w1, w1)
    pos = [(yy, xx) for yy in range(0, Hp - tile + 1, step) for xx in range(0, Wp - tile + 1, step)]
    for k in range(0, len(pos), batch):
        pb = pos[k:k + batch]
        t = torch.from_numpy(np.stack([xp[:, yy:yy + tile, xx:xx + tile] for yy, xx in pb])).to(dev)
        o = denormalise(forward(net, t, bk, sg), bk, sg).cpu().numpy()
        for (yy, xx), oo in zip(pb, o):
            num[:, yy:yy + tile, xx:xx + tile] += oo * wt
            den[yy:yy + tile, xx:xx + tile] += wt
    return (num / den)[:, pad:pad + H, pad:pad + W]


def load_net(rd, dev):
    net = make_net(dev)
    ck = torch.load(rd / "model.pt", map_location=dev, weights_only=False)
    net.load_state_dict(ck["net"])
    net.eval()
    return net, ck


def infer(args):
    dev = torch.device(args.device)
    rd = OUT / args.run
    meta = json.load(open(PREP / "meta.json"))
    net, ck = load_net(rd, dev)
    bk, sg = torch.tensor(meta["bk"], device=dev), torch.tensor(meta["sg"], device=dev)
    L = load_input(ck.get("minus_halo", False))
    t0 = time.time()
    out = apply_tiled(net, L, bk, sg, dev)
    np.save(rd / "starless_net.npy", out.astype(np.float32))
    np.save(rd / "star_layer_net.npy", (L - out).astype(np.float32))
    log(rd / "train.log", f"inference of the full frame in {time.time() - t0:.0f} s -> starless_net.npy, "
                          f"star_layer_net.npy (input{' minus far halo' if ck.get('minus_halo') else ''} - starless)")


def layer(args):
    """star_layer_net.npy = network input (lat_full, minus the far halo for --minus-halo runs) - starless_net.npy,
    in stack units, from an existing inference."""
    rd = OUT / args.run
    ck = torch.load(rd / "model.pt", map_location="cpu", weights_only=False)
    L = load_input(ck.get("minus_halo", False))
    L -= np.load(rd / "starless_net.npy")
    np.save(rd / "star_layer_net.npy", L.astype(np.float32))
    log(rd / "train.log", f"star_layer_net.npy written (input{' minus far halo' if ck.get('minus_halo') else ''}"
                          f" - starless_net)")


# ---------------------------------------------------------------------------------------------------------- eval

def robust_fit_bg(img, mask, yy, xx, cy, cx):
    """Quadratic surface fitted robustly on the pixels of `mask` (per channel), evaluated everywhere."""
    u, v = (xx - cx) / 100.0, (yy - cy) / 100.0
    Aall = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], -1)
    out = np.zeros_like(img)
    for c in range(3):
        a, b = Aall[mask], img[c][mask]
        keep = np.ones(len(b), bool)
        for _ in range(4):
            coef = np.linalg.lstsq(a[keep], b[keep], rcond=None)[0]
            res = b - a @ coef
            s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
            keep = np.abs(res) < 3 * max(s, 1e-6)
        out[c] = Aall @ coef
    return out


def stretch(a, m, s, top=3000.0, base=0.25):
    v = np.arcsinh((a - m[:, None, None]) / s[:, None, None]) / np.arcsinh(top)
    return np.clip(base + (1 - base) * v, 0, 1)


def to_bgr(v):
    return np.ascontiguousarray((v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8))


def label(img, text):
    bar = np.full((26, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


METRIC_SMOOTH = 1.5   # px: every image is smoothed by this Gaussian before the metrics (sheets are not)


def smooth3(a, s=METRIC_SMOOTH):
    return np.stack([cv2.GaussianBlur(a[c], (0, 0), s) for c in range(3)])


def star_eval(st, arrs, A, B, half, rin=20, rout=150):
    """Metrics in the ring rin..rout around one star, for each starless image; sheet row."""
    cx, cy = int(round(st["x"])), int(round(st["y"]))
    H, W = arrs["input"].shape[1:]
    R2 = 260
    y0, y1, x0, x1 = max(0, cy - R2), min(H, cy + R2 + 1), max(0, cx - R2), min(W, cx + R2 + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    r = np.hypot(yy - st["y"], xx - st["x"])
    ann = (r >= rin) & (r <= rout)
    core = r < rin
    ring = (r > rout) & (r <= 250)
    nz = (np.asarray(A[:, y0:y1, x0:x1]) - np.asarray(B[:, y0:y1, x0:x1])) / 2
    sgl = np.array([1.4826 * np.median(np.abs(nz[c][ring] - np.median(nz[c][ring]))) for c in range(3)])
    ssum = float(np.sqrt((sgl ** 2).sum()))
    inp = np.asarray(arrs["input"][:, y0:y1, x0:x1], np.float32)
    res = {}
    for k in ("sl_final", "net"):
        im = smooth3(np.asarray(arrs[k][:, y0:y1, x0:x1], np.float32))
        bg = robust_fit_bg(im, ring, yy, xx, st["y"], st["x"])
        dsum = (im - bg).sum(0)
        isum = im.sum(0)
        dloc = isum - median_filter(isum, size=31, mode="reflect")
        layer = cv2.GaussianBlur((inp - im).sum(0), (0, 0), 2)
        res[k] = dict(
            residue_pct=round(100 * float((dsum[ann] > 3 * ssum).mean()), 2),
            hole_pct=round(100 * float((dsum[ann] < -3 * ssum).mean()), 2),
            small_scale_residue_pct=round(100 * float((dloc[ann] > 3 * ssum).mean()), 2),
            small_scale_hole_pct=round(100 * float((dloc[ann] < -3 * ssum).mean()), 2),
            small_scale_rms_sigma=round(float(np.sqrt((dloc[ann] ** 2).mean()) / ssum), 3),
            mean_dev_sigma=round(float(dsum[ann].mean() / ssum), 3),
            rms_dev_sigma=round(float(np.sqrt((dsum[ann] ** 2).mean()) / ssum), 3),
            star_layer_neg_sigma=round(float(-np.minimum(layer[ann], 0).mean() / ssum), 3),
            star_layer_mean_sigma=round(float(layer[ann].mean() / ssum), 2),
            core_rms_dev_sigma=round(float(np.sqrt((dsum[core] ** 2).mean()) / ssum), 2),
            core_mean_dev_sigma=round(float(dsum[core].mean() / ssum), 2),
        )
    res["noise_sigma_rgb"] = [round(float(v), 3) for v in sgl]
    # sheet row, 100 % scale
    m = np.array([np.median(np.asarray(arrs["sl_final"][c, y0:y1, x0:x1])[ring]) for c in range(3)])
    ys, xs = slice(cy - half - y0, cy + half - y0), slice(cx - half - x0, cx + half - x0)

    def crop(a):
        c = np.zeros((3, 2 * half, 2 * half), np.float32) + m[:, None, None]
        aa = np.asarray(a[:, y0:y1, x0:x1], np.float32)[:, max(0, ys.start):ys.stop, max(0, xs.start):xs.stop]
        c[:, max(0, -ys.start):max(0, -ys.start) + aa.shape[1], max(0, -xs.start):max(0, -xs.start) + aa.shape[2]] = aa
        return c
    ci, cs, cn = crop(arrs["input"]), crop(arrs["sl_final"]), crop(arrs["net"])
    zero = np.zeros(3)
    panels = [stretch(ci, m, sgl), stretch(cs, m, sgl), stretch(cn, m, sgl),
              stretch(ci - cs, zero, sgl, base=0.5), stretch(ci - cn, zero, sgl, base=0.5)]
    row = [to_bgr(p) for p in panels]
    # analysis ring outlines on the starless panels
    for j in (1, 2, 3, 4):
        for rr in (rin, rout):
            if rr < half:
                cv2.circle(row[j], (half, half), rr, (90, 90, 160), 1, cv2.LINE_AA)
    sep = np.full((2 * half, 4, 3), 255, np.uint8)
    out = []
    for p in row:
        out += [p, sep]
    rowimg = np.concatenate(out[:-1], 1)
    return res, rowimg


def eval_regions(meta, arrs, A, B):
    res = {}
    tiles = []
    for name, (x0, y0, x1, y1) in zip(("structured_holdout", "faint_nebula_holdout"), boxes(meta)):
        sl = (slice(None), slice(y0, y1), slice(x0, x1))
        nz = (np.asarray(A[sl]) - np.asarray(B[sl])) / 2
        sgl = np.array([1.4826 * np.median(np.abs(nz[c] - np.median(nz[c]))) for c in range(3)])
        s_, n_, i_ = (np.asarray(arrs[k][sl], np.float32) for k in ("sl_final", "net", "input"))
        d = n_ - s_
        hp = lambda a: a - np.stack([cv2.GaussianBlur(a[c], (0, 0), 3) for c in range(3)])  # noqa: E731
        # star-free pixels: away from the filled cores (dilated 5 px) and from the discs R(G) of stars G < 12
        sm = np.load(PROJ / "starless/sl_mask.npy", mmap_mode="r")[y0:y1, x0:x1].astype(np.uint8)
        sm = cv2.dilate(sm, np.ones((11, 11), np.uint8))
        cat = np.load(PROJ / "stars/star_fluxes.npz")
        for xs, ys_, gs in zip(cat["x"], cat["y"], cat["g"]):
            if gs < 12 and x0 - 200 < xs < x1 + 200 and y0 - 200 < ys_ < y1 + 200:
                cv2.circle(sm, (int(round(xs - x0)), int(round(ys_ - y0))), int(radius_of(gs)), 1, -1)
        free = sm == 0
        hn, hs = hp(n_), hp(s_)
        extra = {}
        if "sl_noisy" in arrs:    # noise-preserving network: how much does it change the non-star pixels?
            dn = n_ - np.asarray(arrs["sl_noisy"][sl], np.float32)
            extra = dict(
                identity_star_free_rms_sigma=[round(float(np.sqrt((dn[c][free] ** 2).mean()) / sgl[c]), 3)
                                              for c in range(3)],
                identity_star_free_mean_sigma=[round(float(dn[c][free].mean() / sgl[c]), 3) for c in range(3)],
                identity_star_free_rms_sigma_smoothed=[round(float(np.sqrt((smooth3(dn)[c][free] ** 2).mean())
                                                                   / sgl[c]), 3) for c in range(3)])
            d = dn
        res[name] = dict(**extra,
            box=[x0, y0, x1, y1],
            noise_sigma_rgb=[round(float(v), 3) for v in sgl],
            level_above_sky_sigma=[round(float((np.median(s_[c]) - meta["bk"][c]) / sgl[c]), 1) for c in range(3)],
            rms_net_minus_sl_sigma=[round(float(np.sqrt((d[c] ** 2).mean()) / sgl[c]), 3) for c in range(3)],
            star_free_pct=round(100 * float(free.mean()), 1),
            star_free_rms_net_minus_sl_sigma=[round(float(np.sqrt((d[c][free] ** 2).mean()) / sgl[c]), 3)
                                              for c in range(3)],
            star_free_mean_net_minus_sl_sigma=[round(float(d[c][free].mean() / sgl[c]), 3) for c in range(3)],
            star_free_fine_detail_ratio_net_over_sl=[round(float(hn[c][free].std() / hs[c][free].std()), 3)
                                                     for c in range(3)],
        )
        m = np.array([np.median(s_[c]) for c in range(3)])
        top = 300.0
        panels = [stretch(i_, m, sgl, top=top, base=0.4), stretch(s_, m, sgl, top=top, base=0.4),
                  stretch(n_, m, sgl, top=top, base=0.4), np.clip(0.5 + d / (6 * sgl[:, None, None]), 0, 1)]
        row = [to_bgr(p) for p in panels]
        h = row[0].shape[0]
        sep = np.full((h, 4, 3), 255, np.uint8)
        img = np.concatenate([row[0], sep, row[1], sep, row[2], sep, row[3]], 1)
        ref = "sl_final + input noise" if "sl_noisy" in arrs else "sl_final"
        tiles.append(label(img, f"{name}: input | sl_final | starless_net | (net - {ref}), grey = 0, "
                                f"black/white = -/+3 noise sigma"))
    wmax = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 6), (0, wmax - t.shape[1]), (0, 0))) for t in tiles]
    return res, np.concatenate(tiles, 0)


@torch.no_grad()
def injection_test(meta, net, synth_all, L, S, A, B, dev):
    """Known truth: paste real bright stamps at known places of the hold-out boxes; what is left by the network
    = output(with star) - output(without star), measured in the 20..150 px ring in noise sigmas."""
    bk, sg = torch.tensor(meta["bk"], device=dev), torch.tensor(meta["sg"], device=dev)
    lib = synth_all.bright
    picks = {}
    nbe = meta.get("n_bright_eval", 4)
    for st in meta["eval_stars"][:nbe] + meta["eval_stars"][nbe:nbe + 2]:
        j = int(np.argmin([math.hypot(b["x"] - st["x"], b["y"] - st["y"]) for b in lib]))
        if math.hypot(lib[j]["x"] - st["x"], lib[j]["y"] - st["y"]) < 3:
            picks[f"G{lib[j]['g']:.1f}@({st['x']:.0f},{st['y']:.0f})"] = j
    res = {}
    for bname, (x0, y0, x1, y1) in zip(("structured", "faint_nebula"), boxes(meta)):
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        P = 768
        ya, xa = cy - P // 2, cx - P // 2
        base = np.ascontiguousarray(L[:, ya:ya + P, xa:xa + P])
        o0 = apply_tiled(net, base, bk, sg, dev)
        nz = (np.asarray(A[:, ya:ya + P, xa:xa + P]) - np.asarray(B[:, ya:ya + P, xa:xa + P])) / 2
        sgl = np.array([1.4826 * np.median(np.abs(nz[c] - np.median(nz[c]))) for c in range(3)])
        ssum = float(np.sqrt((sgl ** 2).sum()))
        yy, xx = np.mgrid[0:P, 0:P]
        r = np.hypot(yy - P // 2, xx - P // 2)
        ann = (r >= 20) & (r <= 150)
        for name, j in picks.items():
            b = lib[j]
            X = torch.from_numpy(base.copy()).to(dev)
            torch.manual_seed(1)
            sat = bool(b["sat"].any())
            clip = np.where(b["sat"], b["plateau"], np.array(meta["sat99"]) * 1.15)
            Td = torch.zeros_like(X)
            synth_all.apply(X, [("b", j, P // 2, P // 2, 1.0 if sat else 1.0, np.ones(3), clip)], Td)
            o1 = apply_tiled(net, X.cpu().numpy(), bk, sg, dev)
            d = smooth3(o1 - o0 - Td.cpu().numpy()).sum(0)       # the diffuse part is meant to stay
            res[f"{bname}:{name}"] = dict(
                left_rms_sigma=round(float(np.sqrt((d[ann] ** 2).mean()) / ssum), 3),
                left_mean_sigma=round(float(d[ann].mean() / ssum), 3),
                residue_pct=round(100 * float((d[ann] > 3 * ssum).mean()), 2),
                hole_pct=round(100 * float((d[ann] < -3 * ssum).mean()), 2),
                core_r20_mean_sigma=round(float(d[r < 20].mean() / ssum), 2))
    return res


def evaluate(args):
    dev = torch.device(args.device)
    rd = OUT / args.run
    meta = json.load(open(PREP / "meta.json"))
    set_radius(meta)
    nbe = meta.get("n_bright_eval", 4)
    L = np.load(PROJ / "optics/lat_full.npy", mmap_mode="r")
    S = np.load(PROJ / "starless/sl_final.npy", mmap_mode="r")
    N = np.load(rd / "starless_net.npy", mmap_mode="r")
    A, B = np.load(PROJ / "optics/lat_A.npy", mmap_mode="r"), np.load(PROJ / "optics/lat_B.npy", mmap_mode="r")
    arrs = {"input": L, "sl_final": S, "net": N}
    ck0 = torch.load(rd / "model.pt", map_location="cpu", weights_only=False)
    if ck0.get("noisy_target", False):
        arrs["sl_noisy"] = np.asarray(S) + (np.asarray(L) - np.load(PREP / "n2n_lat.npy"))
    out = {"stars": [], "regions": None, "injection": None, "noisy_target": ck0.get("noisy_target", False),
           "metric_smoothing_px": METRIC_SMOOTH}
    del ck0
    rows_b, rows_m = [], []
    for k, st in enumerate(meta["eval_stars"]):
        half = 200 if k < nbe else 128
        res, row = star_eval(st, arrs, A, B, half)
        res.update(x=round(st["x"]), y=round(st["y"]), g=round(st["g"], 2))
        out["stars"].append(res)
        txt = (f"({st['x']:.0f},{st['y']:.0f}) G{st['g']:.1f}   input | sl_final | starless_net | star layer "
               f"sl_final | star layer net    ring 20-150 px: residue/hole % sl {res['sl_final']['residue_pct']}/"
               f"{res['sl_final']['hole_pct']}  net {res['net']['residue_pct']}/{res['net']['hole_pct']}")
        (rows_b if k < nbe else rows_m).append(label(row, txt))
    sfx = args.eval_suffix
    cv2.imwrite(str(rd / f"sheet_bright_stars{sfx}.jpg"), np.concatenate(rows_b, 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(rd / f"sheet_medium_stars{sfx}.jpg"), np.concatenate(rows_m, 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    # wide context at 50 % for the four bright stars (input | sl_final | net)
    wide = []
    for st in meta["eval_stars"][:nbe]:
        cx, cy = int(round(st["x"])), int(round(st["y"]))
        hw = 450
        y0, y1, x0, x1 = max(0, cy - hw), min(L.shape[1], cy + hw), max(0, cx - hw), min(L.shape[2], cx + hw)
        m = np.array([np.median(np.asarray(S[c, y0:y1:4, x0:x1:4])) for c in range(3)])
        sgl = np.array(meta["sg"])
        ps = []
        for a in (L, S, N):
            v = to_bgr(stretch(np.asarray(a[:, y0:y1, x0:x1], np.float32), m, sgl))
            ps.append(cv2.resize(v, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        sep = np.full((ps[0].shape[0], 4, 3), 255, np.uint8)
        wide.append(label(np.concatenate([ps[0], sep, ps[1], sep, ps[2]], 1),
                          f"({cx},{cy}) G{st['g']:.1f} at 50 %: input | sl_final | starless_net"))
    wmax = max(t.shape[1] for t in wide)
    wide = [np.pad(t, ((0, 6), (0, wmax - t.shape[1]), (0, 0))) for t in wide]
    cv2.imwrite(str(rd / f"sheet_bright_wide{sfx}.jpg"), np.concatenate(wide, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    out["regions"], img = eval_regions(meta, arrs, A, B)
    cv2.imwrite(str(rd / f"sheet_nebula_regions{sfx}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # whole reliable area (where sl_final is trusted; includes training positions)
    WM = np.load(PREP / "weight_map.npy", mmap_mode="r")
    acc = np.zeros(3)
    n = 0
    for y0 in range(0, L.shape[1], 512):
        sl = (slice(None), slice(y0, y0 + 512), slice(None))
        w = np.asarray(WM[y0:y0 + 512]) > 0
        d = np.asarray(N[sl], np.float32) - np.asarray(S[sl], np.float32)
        acc += (d[:, w] ** 2).sum(1)
        n += w.sum()
    out["reliable_area_rms_net_minus_sl_sigma"] = [round(float(np.sqrt(acc[c] / n) / meta["sg"][c]), 3)
                                                   for c in range(3)]
    net, ck = load_net(rd, dev)
    mh = ck.get("minus_halo", False)
    inj = args.inject_stamps if args.inject_stamps is not None else ck.get("stamps", "")
    lib = torch.load(PREP / lib_name(mh, inj), weights_only=False)
    out["injection_stamps"] = inj or "starless image"
    synth_all = Synth(meta, lib, dev, hold_eval=False)
    out["injection"] = injection_test(meta, net, synth_all, load_input(mh) if mh else L, S, A, B, dev)
    out["minus_halo"] = mh
    out["training"] = ck.get("hist", [])
    json.dump(out, open(rd / f"metrics{sfx}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "training"}, indent=1))


def table(args):
    """Markdown tables of the metrics of the runs named in --run (comma separated)."""
    runs = args.run.split(",")
    M = {r: json.load(open(OUT / r / f"metrics{args.eval_suffix}.json")) for r in runs}
    first = M[runs[0]]
    hdr = ("| star | G | method | residue % (large) | hole % (large) | mean dev (sigma) | residue % (small) "
           "| hole % (small) | rms small-scale (sigma) | core r<20 rms / mean (sigma) |")
    lines = [hdr, "|" + "---|" * 10]
    for k, st in enumerate(first["stars"]):
        rows = [("sl_final", st["sl_final"])] + [(r, M[r]["stars"][k]["net"]) for r in runs]
        for name, v in rows:
            lines.append(f"| ({st['x']},{st['y']}) | {st['g']:.1f} | {name} | {v['residue_pct']} | {v['hole_pct']} | "
                         f"{v['mean_dev_sigma']} | {v['small_scale_residue_pct']} | {v['small_scale_hole_pct']} | "
                         f"{v['small_scale_rms_sigma']} | {v.get('core_rms_dev_sigma')} / "
                         f"{v.get('core_mean_dev_sigma')} |")
    lines += ["", "| region | run | rms net-sl all px (sigma RGB) | star-free rms (sigma RGB) | star-free mean "
                  "(sigma RGB) | fine detail net/sl (RGB) |", "|---|---|---|---|---|---|"]
    for reg in first["regions"]:
        for r in runs:
            v = M[r]["regions"][reg]
            lines.append(f"| {reg} | {r} | {v['rms_net_minus_sl_sigma']} | {v['star_free_rms_net_minus_sl_sigma']} | "
                         f"{v['star_free_mean_net_minus_sl_sigma']} | "
                         f"{v['star_free_fine_detail_ratio_net_over_sl']} |")
    lines += ["", "| pasted star (known truth) | " + " | ".join(f"{r}: left rms / residue % / hole % / core mean"
                                                            for r in runs) + " |",
              "|---|" + "---|" * len(runs)]
    for key in first["injection"]:
        cells = []
        for r in runs:
            v = M[r]["injection"][key]
            cells.append(f"{v['left_rms_sigma']} / {v['residue_pct']} / {v['hole_pct']} / {v['core_r20_mean_sigma']}")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", "reliable area, rms(net - sl_final) in sigma RGB: " +
              ", ".join(f"{r} {M[r]['reliable_area_rms_net_minus_sl_sigma']}" for r in runs)]
    txt = chr(10).join(lines)
    print(txt)
    (OUT / ("tables_" + "_".join(runs) + args.eval_suffix + ".md")).write_text(txt, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "stamps", "train", "infer", "layer", "eval", "table"])
    ap.add_argument("--stamps", default="",
                    help="train: bright stamp library variant ('', 'membrane', 'agreed', 'compact')")
    ap.add_argument("--stamp-variant", default="membrane", choices=["membrane", "agreed", "compact"],
                    help="stamps: which")
    ap.add_argument("--eval-suffix", default="", help="eval/table: suffix of metrics*.json and sheet_*.jpg")
    ap.add_argument("--inject-stamps", default=None,
                    help="eval: stamp library of the known-truth test (default: the run's own)")
    ap.add_argument("--project", default=None, help="project folder holding work/ (read only); default: the Iris")
    ap.add_argument("--out", default=None, help="output folder; default D:/astro_gpu_diag/own_starnet[/<project>]")
    ap.add_argument("--run", default="runA")
    ap.add_argument("--hold-eval", action="store_true", help="leave the evaluation stars out of the stamp library")
    ap.add_argument("--core-mask", action="store_true",
                    help="no loss on the cores of the real catalogue stars (G < 16) nor on the filled cores")
    ap.add_argument("--bright-aug", action="store_true",
                    help="amplify the patch's own nebula 1-40x in half of the patches (bright nebula is rare here)")
    ap.add_argument("--noisy-target", action="store_true",
                    help="star removal only: target = sl_final + the input's own noise (no denoising learnt)")
    ap.add_argument("--minus-halo", action="store_true",
                    help="hybrid: subtract the measured far halo of the stars (stars_model_halo) before the network")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--patch", type=int, default=384)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="infer/eval only (cpu for tests while the GPU trains)")
    args = ap.parse_args()
    set_paths(args.project, args.out)
    {"prep": prep, "stamps": membrane_stamps, "train": train, "infer": infer, "layer": layer, "eval": evaluate,
     "table": table}[args.cmd](args)


if __name__ == "__main__":
    main()
