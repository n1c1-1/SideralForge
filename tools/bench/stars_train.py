"""Star removal, path B: a network trained once, on our stacks, to take the stars off (docs/chantiers/etoiles.md).

The pairs: each stack of the training set (stars_dataset.py: gradient removed, denoised by the shipped network, colours
balanced, in noise sigmas, background 0) and its starless image, StarNet's corrected by hand in Photoshop (the review
TIFFs). Two kinds of examples, mixed in each batch:

- **real**: the stack with its own stars -> the corrected starless image;
- **synthetic stars**: stars drawn on top (on the starless image, or on the stack itself) -> the starless image. Their
  profile is a Moffat core (Moffat 1969, A&A 3, 455), the FWHM measured on the field's own stars or drawn wider, an
  aureole falling as r^-3 (Racine 1996, PASP 108, 699), sometimes four diffraction spikes; brightnesses from a power law
  of the counts, colours spread around the white of the balance, saturation at the field's own top, some photon noise.
  Here the truth under each star is exact.

The network: NAFNet (Chen et al. 2022, arXiv:2204.04676), the image restoration network that Dark Star and SyQon use
for star removal (etoiles_etat_de_lart.md), trained with an L1 loss on a fixed asinh stretch of the data (the star
removal tools all work on a stretched image: the stretch here is known and undone exactly).

Left out: the bench's fields (denoise_dataset.HELD_OUT), set 53 (not reviewed), the levels of a reviewed field on
another pixel grid than the reviewed one. Ignored zones (no loss): what the review could not settle (Alnitak's halo, the
star of NGC 2023, a knot of M33), and in a field's other levels the zones retouched in the reviewed one.

Commands:
    python tools/bench/stars_train.py fixes            # the corrected starless images, ignore masks, star FWHM
    python tools/bench/stars_train.py train --tag mixed --hours 8.5 --device cuda:0
    python tools/bench/stars_train.py train --tag real --synthetic 0 --hours 8.5 --device cuda:1
    python tools/bench/stars_train.py bench --tag mixed     # the bench's starless images, then stars_compare.py

Files: F:/astro_essais/stars_dataset_v2/<pair>/{starless_fixed.npy, ignore.npy}, fields.json;
F:/astro_essais/stars_train/<tag>/{net.pt, best.pt, history.json, log}; F:/astro_essais/stars_compare/<case>/
net_<tag>_starless.fits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "bench"))

DS = Path("F:/astro_essais/stars_dataset_v2")
REVIEW = Path("F:/astro_essais/stars_dataset/_revue")
TRAIN = Path("F:/astro_essais/stars_train")
COMPARE = Path("F:/astro_essais/stars_compare")
BENCH = Path("F:/astro_essais/bench")
LEVELS = ("", "_half", "_quarter", "_eighth")
NOT_REVIEWED = ("set_53_",)
#: (field, x, y, radius px): what the review could not settle, left out of the loss
IGNORED = [("user_Horsehead", 2568, 1136, 520),            # Alnitak: its halo and spikes cannot be taken off cleanly
           ("user_Horsehead", 3132, 2056, 60),             # the star of NGC 2023, left in the starless image
           ("set_36_osc_refractor_asi294mc_star71_m33", 3346, 2424, 45)]   # a knot of M33 taken off: H II region?
STRETCH = 3.0          # sigmas: the network sees asinh(v / STRETCH) / SCALE
SCALE = 10.0
PATCH, MARGIN = 256, 128


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def field_of(name: str) -> str:
    for lv in LEVELS[1:]:
        if name.endswith(lv):
            return name[:-len(lv)]
    return name


def fwd(v):
    """The network's stretch (numpy or torch)."""
    mod = np if isinstance(v, np.ndarray) else __import__("torch")
    return mod.asinh(v / STRETCH) / SCALE


def inv(y):
    mod = np if isinstance(y, np.ndarray) else __import__("torch")
    return mod.sinh(y * SCALE) * STRETCH


# ---------------------------------------------------------------------------------------------- fixes
def fixes() -> int:
    """The review TIFFs back to the data: the retouched pixels (more than 4 codes from StarNet's) take the TIFF's value
    with the stretch undone, the others keep StarNet's at full precision."""
    import tifffile
    from scipy import ndimage
    st = json.loads((REVIEW / "stretch.json").read_text())
    metas = {json.loads(f.read_text())["name"]: json.loads(f.read_text()) for f in DS.glob("*/meta.json")}
    fields = {}
    for fld, s in sorted(st.items()):
        if fld.startswith(NOT_REVIEWED):
            continue
        d = DS / s["pair"]
        sl = np.load(d / "starless.npy").astype(np.float32)
        top = np.arcsinh((s["top"] + s["offset"]) / s["soft"])
        ref = np.round(np.clip(np.arcsinh(np.clip(sl + s["offset"], 0, None) / s["soft"]) / top, 0, 1) * 65535)
        ed = tifffile.imread(REVIEW / f"{fld}_starless.tif").transpose(2, 0, 1).astype(np.float32)
        changed = np.abs(ed - ref).max(0) > 4
        del ref
        v = s["soft"] * np.sinh(ed / 65535 * top) - s["offset"]
        fixed = np.where(changed[None], v, sl)
        np.save(d / "starless_fixed.npy", fixed.astype(np.float16))
        H, W = changed.shape
        ign = np.zeros((H, W), bool)
        yy, xx = np.ogrid[:H, :W]
        for f, x, y, r in IGNORED:
            if f == fld:
                ign |= (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        np.save(d / "ignore.npy", ign)
        # the star FWHM of the field: its isolated, unsaturated stars in the star layer (stack - starless)
        inp = np.load(d / "input.npy").astype(np.float32)
        L = (inp - fixed).mean(0)
        del inp, fixed
        mx = ndimage.maximum_filter(L, 15)
        pk = (L == mx) & (L > 30) & (L < 0.25 * s["top"])
        ys, xs = np.nonzero(pk[20:-20, 20:-20])
        ys, xs = ys + 20, xs + 20
        order = np.argsort(-L[ys, xs])[:400]
        fw = []
        for y, x in zip(ys[order], xs[order]):
            c = L[y - 15:y + 16, x - 15:x + 16]
            area = float((c > 0.5 * L[y, x]).sum())
            if (c > 0.3 * L[y, x]).sum() < 4 * area:                 # nothing else bright around it
                fw.append(2 * np.sqrt(area / np.pi))
        fwhm = float(np.median(fw)) if len(fw) >= 10 else 3.0
        fields[fld] = {"pair": s["pair"], "changed_px": int(changed.sum()), "ignored_px": int(ign.sum()),
                       "fwhm_px": round(fwhm, 2), "fwhm_stars": len(fw), "top": s["top"]}
        # the field's other levels: the zones retouched in this one are ignored there (same pixel grid only)
        zone = ndimage.binary_dilation(changed, iterations=8) | ign
        for lv in LEVELS[1:]:
            m = metas.get(fld + lv)
            if m is None:
                continue
            if tuple(m["shape"][1:]) == (H, W):
                np.save(DS / m["name"] / "ignore.npy", zone)
                fields[fld].setdefault("levels", []).append(lv)
            else:
                fields[fld].setdefault("levels_left_out", []).append(lv)
        log(f"{fld}: {fields[fld]}")
    (DS / "fields.json").write_text(json.dumps(fields, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------- the network
def nafnet(width=32, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2), recompute=False):
    """NAFNet (Chen, Chu, Zhang, Sun 2022, arXiv:2204.04676), as published (github.com/megvii-research/NAFNet).
    ``recompute``: in training, the blocks' activations are recomputed on the way back instead of kept (less memory,
    about a third slower)."""
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint

    class LayerNorm2d(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.w, self.b = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

        def forward(self, x):
            mu = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1, keepdim=True)
            return (x - mu) / (var + 1e-6).sqrt() * self.w[:, None, None] + self.b[:, None, None]

    class NAFBlock(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.n1, self.n2 = LayerNorm2d(c), LayerNorm2d(c)
            self.c1 = nn.Conv2d(c, 2 * c, 1)
            self.c2 = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
            self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
            self.c3 = nn.Conv2d(c, c, 1)
            self.c4 = nn.Conv2d(c, 2 * c, 1)
            self.c5 = nn.Conv2d(c, c, 1)
            self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
            self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

        def forward(self, inp):
            x = self.c2(self.c1(self.n1(inp)))
            a, b = x.chunk(2, 1)
            x = a * b
            x = self.c3(x * self.sca(x))
            y = inp + x * self.beta
            a, b = self.c4(self.n2(y)).chunk(2, 1)
            return y + self.c5(a * b) * self.gamma

    class NAFNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.intro = nn.Conv2d(3, width, 3, padding=1)
            self.ending = nn.Conv2d(width, 3, 3, padding=1)
            self.encs, self.downs, self.ups, self.decs = (nn.ModuleList() for _ in range(4))
            c = width
            for n in enc:
                self.encs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
                self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
                c *= 2
            self.mid = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
            for n in dec:
                self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
                c //= 2
                self.decs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.pad = 2 ** len(enc)

        def forward(self, inp):
            H, W = inp.shape[2:]
            ph, pw = (-H) % self.pad, (-W) % self.pad
            x = nn.functional.pad(inp, (0, pw, 0, ph))
            x0 = x
            x = self.intro(x)

            def run(m, v):
                return checkpoint(m, v, use_reentrant=False) if recompute and self.training else m(v)
            skips = []
            for e, dn in zip(self.encs, self.downs):
                x = run(e, x)
                skips.append(x)
                x = dn(x)
            x = run(self.mid, x)
            for d, up, sk in zip(self.decs, self.ups, reversed(skips)):
                x = run(d, up(x) + sk)
            return (self.ending(x) + x0)[:, :, :H, :W]

    return NAFNet()


# ---------------------------------------------------------------------------------------------- synthetic stars
def synthetic_stars(n: int, fwhm_field, top, gen, dev):
    """n tiles (n, 3, PATCH, PATCH) of stars only, in sigmas: drawn on a wider area (the margin holds the stars whose
    halo reaches the tile), one profile per tile, made by a product in Fourier space."""
    import torch
    G = PATCH + 2 * MARGIN
    yy, xx = torch.meshgrid(torch.arange(G, device=dev, dtype=torch.float32) - G // 2,
                            torch.arange(G, device=dev, dtype=torch.float32) - G // 2, indexing="ij")
    out = torch.zeros(n, 3, G, G, device=dev)
    psfs = torch.zeros(n, 3, G, G, device=dev)

    def U(a, b, size=()):
        return a + (b - a) * torch.rand(size, generator=gen, device=dev)

    for i in range(n):
        fw = float(fwhm_field[i]) * float(U(0.8, 1.25)) if float(U(0, 1)) < 0.7 else float(torch.exp(U(np.log(1.3),
                                                                                                        np.log(8.0))))
        beta = float(U(1.8, 4.5))
        alpha = fw / (2 * np.sqrt(2 ** (1 / beta) - 1))
        e, th = float(U(0, 0.25)), float(U(0, np.pi))
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
                rc = rh * (1 + float(U(-0.25, 0.25)))           # a halo a little larger in one colour (refractors)
                halo = (1 + (xx ** 2 + yy ** 2) / rc ** 2) ** (-1.5)
                p = p + fh * halo / halo.sum()
            if fs:
                p = p + fs * spikes
            psfs[i, ch] = torch.fft.ifftshift(p)
        # the stars: a power law of brightness, from about a third of a sigma at the peak up to far past saturation
        nstar = int(float(torch.exp(U(0.0, np.log(400.0)))) * (G / PATCH) ** 2)
        if nstar == 0:
            continue
        a_pow = float(U(0.7, 1.2))
        peak1 = (beta - 1) / (np.pi * alpha ** 2)                   # the peak of a star of flux 1
        fmin = float(U(0.3, 1.0)) / peak1
        flux = fmin * U(0, 1, (nstar,)).clamp(min=1e-9) ** (-1 / a_pow)
        flux = flux.clamp(max=50 * float(top[i]) / peak1)
        # colours: B-V spread as field stars (about the Sun's, redder in the dust), the balance's white at the Sun's
        # (0.65), turned into multipliers
        bv = (0.7 + 0.35 * torch.randn(nstar, generator=gen, device=dev)).clamp(-0.3, 2.0)
        col = torch.stack([10 ** (0.2 * (bv - 0.65)), torch.ones_like(bv), 10 ** (-0.36 * (bv - 0.65))], 0)
        col = col / col.mean(0, keepdim=True)
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


# ---------------------------------------------------------------------------------------------- data
def training_pairs():
    from denoise_dataset import HELD_OUT
    fields = json.loads((DS / "fields.json").read_text())
    pairs = []
    for f in sorted(DS.glob("*/meta.json")):
        m = json.loads(f.read_text())
        fld = field_of(m["name"])
        if fld not in fields or m["name"].startswith(HELD_OUT):
            continue
        lv = m["name"][len(fld):]
        if lv and lv not in fields[fld].get("levels", []):
            continue
        d = DS / m["name"]
        tgt = d / "starless_fixed.npy" if not lv else d / "starless.npy"
        ign = d / "ignore.npy"
        pairs.append({"name": m["name"], "field": fld, "input": d / "input.npy", "target": tgt,
                      "ignore": ign if ign.exists() else None, "fwhm": fields[fld]["fwhm_px"], "top": m["top"]})
    return pairs


class Tiles:
    """Random tiles from the pairs (memory-mapped: the files stay in the system's cache, shared by the two runs)."""

    def __init__(self, pairs, seed):
        self.pairs = pairs
        self.arr = [(np.load(p["input"], mmap_mode="r"), np.load(p["target"], mmap_mode="r"),
                     np.load(p["ignore"], mmap_mode="r") if p["ignore"] else None) for p in pairs]
        self.by_field = {}
        for i, p in enumerate(pairs):
            self.by_field.setdefault(p["field"], []).append(i)
        self.fields = sorted(self.by_field)
        self.rng = np.random.default_rng(seed)

    def get(self, n):
        P = PATCH
        xin, tgt, msk, fw, top = [], [], [], [], []
        while len(xin) < n:
            i = self.by_field[self.fields[self.rng.integers(len(self.fields))]]
            i = i[self.rng.integers(len(i))]
            A, T, I = self.arr[i]
            H, W = A.shape[1:]
            y, x = int(self.rng.integers(0, H - P)), int(self.rng.integers(0, W - P))
            a = np.asarray(A[:, y:y + P, x:x + P], np.float32)
            inside = np.abs(a).max(0) > 0
            m = inside & (~np.asarray(I[y:y + P, x:x + P]) if I is not None else inside)
            if m.mean() < 0.5:
                continue
            t = np.asarray(T[:, y:y + P, x:x + P], np.float32)
            k = int(self.rng.integers(4))
            a, t, m = np.rot90(a, k, (1, 2)), np.rot90(t, k, (1, 2)), np.rot90(m, k)
            if self.rng.random() < 0.5:
                a, t, m = a[:, :, ::-1], t[:, :, ::-1], m[:, ::-1]
            xin.append(np.ascontiguousarray(a))
            tgt.append(np.ascontiguousarray(t))
            msk.append(np.ascontiguousarray(m))
            fw.append(self.pairs[i]["fwhm"])
            top.append(self.pairs[i]["top"])
        return np.stack(xin), np.stack(tgt), np.stack(msk)[:, None], np.array(fw), np.array(top)


# ---------------------------------------------------------------------------------------------- the bench
def bench_case(case: str):
    """A bench case's common original (stars_samples.export) and StarNet's starless image, in the training set's units:
    each colour minus its sky, over the mean of the colours' noise (measured on the half-stacks before denoising, carried
    through the colour matrix)."""
    import os
    os.environ.setdefault("ASTRO_GPU_HOME", str(BENCH / "_state_probe"))
    from stars_compare import OUT, SOURCE, read_fits

    from astro_gpu import api
    from astro_gpu.stages.denoise import norm_params
    meta = json.loads((OUT / case / "meta.json").read_text())
    p = api.open_project(BENCH / case / "ours_express")
    w = next(x for x in p.workflows if x.name == SOURCE)
    st, col = api.stage_outputs(p, w.id, "stack"), api.stage_outputs(p, w.id, "color")
    M = np.load(Path(col.dir) / "M.npy") if col is not None else np.eye(3)
    half = Path(meta["source"]) / "lat_A.npy"           # M8's original comes from a stage without half-stacks
    A = np.load(half if half.exists() else Path(st.dir) / "drz_A.npy", mmap_mode="r")
    _, sg, nm = norm_params(np.asarray(A, np.float32), np.load(Path(st.dir) / "drz_weight.npy").astype(np.float32))
    unit = float(np.mean(np.sqrt((M ** 2) @ (np.asarray(sg, float) ** 2))))
    O = read_fits(OUT / case / "input.fits") * meta["scale"]
    bk = np.array([np.median(O[c][nm > 0][::7]) for c in range(3)], np.float32)[:, None, None]
    S = read_fits(OUT / case / "starnet_starless.fits") * meta["scale"] if (OUT / case / "starnet_starless.fits").exists() \
        else None
    to = lambda img: (img - bk) / unit   # noqa: E731
    return to(O), (to(S) if S is not None else None), (bk, unit, meta["scale"], nm)


def apply(net, img: np.ndarray, dev, tile: int = PATCH, overlap: int = 64) -> np.ndarray:
    """The network over a whole image, in tiles of the training size (its channel attention averages over the tile),
    blended with a cosine window."""
    import torch
    C, H, W = img.shape
    step = tile - overlap
    out = np.zeros_like(img, np.float32)
    wsum = np.zeros((H, W), np.float32)
    win1 = np.sin(np.linspace(0, np.pi, tile + 2)[1:-1]) ** 2 + 1e-3
    win = np.outer(win1, win1).astype(np.float32)
    ys = list(range(0, max(H - tile, 0) + 1, step)) + ([H - tile] if (H - tile) % step else [])
    xs = list(range(0, max(W - tile, 0) + 1, step)) + ([W - tile] if (W - tile) % step else [])
    net.eval()
    with torch.no_grad():
        for y in ys:
            batch = [img[:, y:y + tile, x:x + tile] for x in xs]
            for j in range(0, len(batch), 16):
                b = torch.from_numpy(np.stack(batch[j:j + 16])).to(dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    r = net(fwd(b)).float()
                r = inv(r).cpu().numpy()
                for k, x in enumerate(xs[j:j + 16]):
                    out[:, y:y + tile, x:x + tile] += r[k] * win
                    wsum[y:y + tile, x:x + tile] += win
    return out / wsum


# ---------------------------------------------------------------------------------------------- train
def train(tag: str, hours: float, device: str, synthetic: float, batch: int, lr: float, recompute: bool) -> int:
    import torch
    dev = torch.device(device)
    torch.backends.cudnn.benchmark = True
    out = TRAIN / tag
    out.mkdir(parents=True, exist_ok=True)
    pairs = training_pairs()
    log(f"{tag}: {len(pairs)} pairs over {len({p['field'] for p in pairs})} fields, synthetic share {synthetic}")
    data = Tiles(pairs, seed=0)
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    # validation: synthetic stars on StarNet's starless images of the bench (a truth known exactly), fixed tiles
    vgen = torch.Generator(device=dev)
    vgen.manual_seed(123)
    vrng = np.random.default_rng(7)
    vset = []
    for case in ("heart", "iris", "m51", "m8", "rosette", "seestar"):
        try:
            _, S, (_, _, _, nm) = bench_case(case)
        except Exception as e:  # noqa: BLE001
            log(f"validation: {case} left out ({type(e).__name__}: {e})")
            continue
        H, W = S.shape[1:]
        tiles = []
        while len(tiles) < 6:
            y, x = int(vrng.integers(0, H - PATCH)), int(vrng.integers(0, W - PATCH))
            if nm[y:y + PATCH, x:x + PATCH].min() > 0:
                tiles.append(S[:, y:y + PATCH, x:x + PATCH])
        t = torch.from_numpy(np.stack(tiles)).to(dev)
        stars = synthetic_stars(len(tiles), [3.0] * len(tiles), [float(S.max())] * len(tiles), vgen, dev)
        sat = float(S.max())
        vset.append((case, torch.minimum(t + stars, torch.maximum(t, torch.full_like(t, sat))), t))
        del S
    log(f"validation tiles: {sum(len(v[1]) for v in vset)}")

    torch.manual_seed(0)
    net = nafnet(recompute=recompute).to(dev)
    log(f"NAFNet: {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters, batch {batch}")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.9))

    def loss_of(xin, tgt, msk):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = net(fwd(xin)).float()
        return ((pred - fwd(tgt)).abs() * msk).sum() / (msk.sum() * 3 + 1)

    hist, best = [], None
    t0, it, run, warm = time.time(), 0, None, 1000
    while True:
        frac = (time.time() - t0) / (hours * 3600)
        if frac >= 1.0:
            break
        for g in opt.param_groups:                       # warm-up, then a cosine to zero at the end of the time given
            g["lr"] = lr * min(1.0, (it + 1) / warm) * 0.5 * (1 + np.cos(np.pi * frac))
        a, t, m, fw, top = data.get(batch)
        a, t, m = (torch.from_numpy(v).to(dev) for v in (a, t, m))
        m = m.float()
        if synthetic > 0:
            pick = torch.rand(batch, generator=gen, device=dev) < synthetic
            if pick.any():
                k = int(pick.sum())
                stars = synthetic_stars(k, fw[pick.cpu().numpy()], top[pick.cpu().numpy()], gen, dev)
                # on the starless image (stars of synthesis only) or on the stack (its own stars and more), half each
                base = torch.where((torch.rand(k, 1, 1, 1, generator=gen, device=dev) < 0.5), t[pick], a[pick])
                sat = torch.from_numpy(top[pick.cpu().numpy()]).float().to(dev)[:, None, None, None] \
                    * (0.5 + 0.5 * torch.rand(k, 1, 1, 1, generator=gen, device=dev))
                a[pick] = torch.minimum(base + stars, torch.maximum(base, sat))
        net.train()
        loss = loss_of(a, t, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        opt.step()
        run = loss.item() if run is None else 0.99 * run + 0.01 * loss.item()
        it += 1
        if it % 2000 == 0:
            net.eval()
            vl, vs = [], []
            with torch.no_grad():
                for _case, x, tt in vset:
                    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        pred = net(fwd(x)).float()
                    err = (pred - fwd(tt)).abs()
                    star = (fwd(x) - fwd(tt)).abs().max(1, keepdim=True)[0] > 0.01
                    vl.append(float(err.mean()))
                    vs.append(float(err[star.expand_as(err)].mean()) if star.any() else 0.0)
            score = float(np.mean(vs))
            hist.append({"step": it, "minutes": round((time.time() - t0) / 60, 1), "train": round(run, 5),
                         "val": round(float(np.mean(vl)), 5), "val_stars": round(score, 5),
                         "by_case": {c[0]: round(v, 5) for c, v in zip(vset, vs)}, "lr": opt.param_groups[0]["lr"]})
            (out / "history.json").write_text(json.dumps(hist, indent=1))
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it}, out / "net.pt")
            if best is None or score < best:
                best = score
                torch.save({"net": net.state_dict(), "it": it}, out / "best.pt")
            log(f"step {it}: train {run:.5f}, validation {np.mean(vl):.5f}, on the stars {score:.5f} "
                f"({', '.join(f'{c[0]} {v:.4f}' for c, v in zip(vset, vs))}), lr {opt.param_groups[0]['lr']:.1e}, "
                f"{(time.time() - t0) / it:.2f} s/step")
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it}, out / "net.pt")
    log(f"done: {it} steps in {(time.time() - t0) / 60:.0f} min")
    return 0


# ---------------------------------------------------------------------------------------------- bench
def bench(tag: str, device: str, which: str = "best") -> int:
    """The trained network on the bench's common originals: net_<tag>_starless.fits in the same units as input.fits,
    for stars_compare.py."""
    import torch
    from astropy.io import fits
    dev = torch.device(device)
    net = nafnet().to(dev)
    net.load_state_dict(torch.load(TRAIN / tag / f"{which}.pt", map_location=dev, weights_only=False)["net"])
    for case in ("heart", "iris", "m51", "m8", "rosette", "seestar"):
        try:
            O, _, (bk, unit, scale, nm) = bench_case(case)
        except Exception as e:  # noqa: BLE001
            log(f"{case}: left out ({type(e).__name__}: {e})")
            continue
        t = time.time()
        S = apply(net, O, dev)
        S[:, nm <= 0] = O[:, nm <= 0]
        S = (S * unit + bk) / scale
        fits.PrimaryHDU(S.astype(np.float32)).writeto(COMPARE / case / f"net_{tag}_starless.fits", overwrite=True)
        log(f"{case}: {O.shape[2]}x{O.shape[1]} in {time.time() - t:.1f} s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("fixes", "train", "bench"))
    ap.add_argument("--tag", default="mixed")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--synthetic", type=float, default=0.6, help="share of the tiles given synthetic stars")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--recompute", action="store_true", help="train: less memory, slower (an 8 GB card)")
    ap.add_argument("--then-bench", action="store_true", help="train: then the bench's starless images and measures")
    a = ap.parse_args(argv)
    if a.cmd == "fixes":
        return fixes()
    if a.cmd == "bench":
        return bench(a.tag, a.device)
    r = train(a.tag, a.hours, a.device, a.synthetic, a.batch, a.lr, a.recompute)
    if a.then_bench:
        bench(a.tag, a.device)
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "tools/bench/stars_compare.py")], cwd=str(ROOT))
    return r


if __name__ == "__main__":
    sys.exit(main())
