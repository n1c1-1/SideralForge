"""The Stars block: a network takes the stars off the image, which gives two images, the starless image and the stars
alone (docs/chantiers/etoiles.md).

The network is NAFNet (Chen, Chu, Zhang, Sun 2022, arXiv:2204.04676), an image restoration network, trained once on
our stacks: starless images corrected by hand, plus stars drawn on top of them (a Moffat core, an r^-3 aureole,
diffraction spikes, saturation), where the truth under each star is exact (tools/bench/stars_train.py, run "mixed").
On the bench it leaves less of the faint stars than the reference tool and takes the spikes and halos of the bright
stars off, the background unchanged (docs/POINT_MATINEE_2026-11-05.md).

It works on what it was trained on: the Output block's linear image (render_simple: the stack with its sky gradient
removed, denoised, white balanced, its sky taken off), each channel minus its background, over one noise unit (the
noise of the half-stacks before denoising, through the colour matrix, averaged over the three colours). The network sees
a fixed asinh stretch of it, undone exactly on the way out.

Outputs, in the units of the Output block's lin.npy (so that starless + stars is that image, to the pixel):

* starless.npy, the image without its stars; stars.npy, the stars alone (the image minus the starless one);
* valid.npy, the covered pixels (the Output block's), and prep_stars.json: the useful frame and the statistics of
  the whole image's stretch (render/stars.py stretches both images with it in Post).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..core.imgutil import asinh_preview
from ..core.params import StageParams
from ..core.stage import Context, Stage

MODEL = Path(__file__).resolve().parents[1] / "resources" / "stars" / "starless_nafnet.pt"
MODEL_ID = hashlib.sha256(MODEL.read_bytes()).hexdigest()[:10] if MODEL.exists() else "missing"
#: the network's stretch: it sees asinh(v / STRETCH) / SCALE, v in noise units
STRETCH, SCALE = 3.0, 10.0
#: the tiles the network works on (its training size: its channel attention averages over a tile), their overlap
TILE, OVERLAP = 256, 64


class StarRemovalParams(StageParams):
    """No setting: the network is applied as it was trained (its outputs are adjusted in Post)."""


class StarRemoval(Stage):
    name = "star_removal"
    title = "Star removal"
    description = ("A network trained on our stacks takes the stars off the Output block's image: the starless image "
                   "and the stars alone, two images that Post combines (the Combine stars & background layers block) or takes as "
                   "they are.")
    deps = ("render_simple", "stack")
    Params = StarRemovalParams
    version = f"2-{MODEL_ID}"            # 2: the network's grid notched (core/periodic.py, issue #28)

    def run(self, ctx: Context, p: StarRemovalParams) -> None:
        from .denoise import norm_params
        from .gradient import stack_image
        ctx.progress(0.02, "loading the image")
        prep = ctx.load_json("render_simple", "prep_simple.json")
        lin = ctx.load("render_simple", "lin.npy").astype(np.float32)
        valid = ctx.load("render_simple", "valid.npy")
        M = np.asarray(prep["M"], np.float64)
        x0, x1, y0, y1 = prep["crop"]
        shown = np.zeros_like(valid)
        shown[y0:y1, x0:x1] = True
        shown &= valid
        _bk, sg, _nm = norm_params(stack_image(ctx, "drz_A.npy"), ctx.load("stack", "drz_weight.npy", mmap=True))
        unit = float(np.mean(np.sqrt((M ** 2) @ (np.asarray(sg, np.float64) ** 2))))
        bk = np.array([np.median(lin[c][shown][::7]) for c in range(3)], np.float32)[:, None, None]
        ctx.log(f"noise unit {unit:.4g}, background {', '.join(f'{v:.3g}' for v in bk.ravel())}")
        dev = ctx.device
        net = network(dev)
        t0 = time.time()
        starless = apply(net, (lin - bk) / unit, dev, ctx) * unit + bk
        ctx.log(f"stars taken off in {time.time() - t0:.1f} s on {dev}")
        starless[:, ~valid] = lin[:, ~valid]
        # the network's own grid (its checkerboard, at the multiples of 1/16 px^-1) notched where it stands out
        from ..core.periodic import lattice_frequencies, remove_periodic
        starless, taken = remove_periodic(starless, lattice_frequencies(16), valid)
        ctx.metric("grid_peaks", len(taken))
        if taken:
            ctx.log(f"the network's grid: {len(taken)} peaks notched (strongest {max(t[3] for t in taken):.0f} x the "
                    "spectrum around)")
        stars = lin - starless
        ctx.progress(0.9, "statistics")
        ctx.save("starless.npy", starless)
        ctx.save("stars.npy", stars)
        np.save(ctx.out("valid.npy"), valid)
        out = {"crop": prep["crop"], "M": prep["M"], "drizzle_scale": prep.get("drizzle_scale", 1.0),
               "width": prep.get("width"), "height": prep.get("height"), "unit": unit, "background": bk.ravel().tolist(),
               "stats_full": prep["stats"]}
        ctx.save_json("prep_stars.json", out)
        ctx.progress(0.95, "preview")
        h, w = lin.shape[1:]
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        s = min(800, h, w) // 2
        box = (slice(None), slice(max(0, cy - s), cy + s), slice(max(0, cx - s), cx + s))
        a, b = lin[box], starless[box]                   # one stretch for both: before and after side by side
        ctx.preview("starless", asinh_preview(np.concatenate([a, np.zeros((3, a.shape[1], 6), np.float32), b], 2)))


class _LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / (var + 1e-6).sqrt() * self.w[:, None, None] + self.b[:, None, None]


class _NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = _LayerNorm2d(c), _LayerNorm2d(c)
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
    """NAFNet (Chen, Chu, Zhang, Sun 2022, arXiv:2204.04676), as published (github.com/megvii-research/NAFNet).
    ``recompute``: in training, the blocks' activations are recomputed on the way back instead of kept (less memory,
    about a third slower)."""

    def __init__(self, width=32, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2), recompute=False):
        super().__init__()
        self.recompute = recompute
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encs, self.downs, self.ups, self.decs = (nn.ModuleList() for _ in range(4))
        c = width
        for n in enc:
            self.encs.append(nn.Sequential(*[_NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            c *= 2
        self.mid = nn.Sequential(*[_NAFBlock(c) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decs.append(nn.Sequential(*[_NAFBlock(c) for _ in range(n)]))
        self.pad = 2 ** len(enc)

    def _run(self, m, v):
        if self.recompute and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(m, v, use_reentrant=False)
        return m(v)

    def forward(self, inp):
        H, W = inp.shape[2:]
        ph, pw = (-H) % self.pad, (-W) % self.pad
        x = nn.functional.pad(inp, (0, pw, 0, ph))
        x0 = x
        x = self.intro(x)
        skips = []
        for e, dn in zip(self.encs, self.downs):
            x = self._run(e, x)
            skips.append(x)
            x = dn(x)
        x = self._run(self.mid, x)
        for d, up, sk in zip(self.decs, self.ups, reversed(skips)):
            x = self._run(d, up(x) + sk)
        return (self.ending(x) + x0)[:, :, :H, :W]


def fwd(v):
    """The network's stretch (numpy or torch), v in noise units."""
    mod = np if isinstance(v, np.ndarray) else torch
    return mod.asinh(v / STRETCH) / SCALE


def inv(y):
    mod = np if isinstance(y, np.ndarray) else torch
    return mod.sinh(y * SCALE) * STRETCH


def network(dev) -> NAFNet:
    """The bundled network (resources/stars), in float32."""
    if not MODEL.exists():
        raise FileNotFoundError(f"the star removal network is missing: {MODEL}")
    net = NAFNet().to(dev)
    state = torch.load(MODEL, map_location=dev, weights_only=False)["net"]
    net.load_state_dict({k: v.float() for k, v in state.items()})
    net.eval()
    return net


@torch.no_grad()
def apply(net, img: np.ndarray, dev, ctx: Context | None = None, tile: int = TILE, overlap: int = OVERLAP) -> np.ndarray:
    """The network over a whole (3, H, W) image in noise units, in tiles of its training size, blended with a cosine
    window; returns the starless image in the same units."""
    img = np.asarray(img, np.float32)
    C, H, W = img.shape
    if H < tile or W < tile:                                    # a tiny image: padded to one tile
        ph, pw = max(0, tile - H), max(0, tile - W)
        return apply(net, np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="reflect"), dev, ctx, tile, overlap)[:, :H, :W]
    step = tile - overlap
    out = np.zeros_like(img, np.float32)
    wsum = np.zeros((H, W), np.float32)
    win1 = np.sin(np.linspace(0, np.pi, tile + 2)[1:-1]) ** 2 + 1e-3
    win = np.outer(win1, win1).astype(np.float32)
    ys = list(range(0, H - tile + 1, step)) + ([H - tile] if (H - tile) % step else [])
    xs = list(range(0, W - tile + 1, step)) + ([W - tile] if (W - tile) % step else [])
    for i, y in enumerate(ys):
        batch = [img[:, y:y + tile, x:x + tile] for x in xs]
        for j in range(0, len(batch), 16):
            b = torch.from_numpy(np.ascontiguousarray(np.stack(batch[j:j + 16]))).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                r = net(fwd(b)).float()
            r = inv(r).cpu().numpy()
            for k, x in enumerate(xs[j:j + 16]):
                out[:, y:y + tile, x:x + tile] += r[k] * win
                wsum[y:y + tile, x:x + tile] += win
        if ctx is not None:
            ctx.check_stop()
            ctx.progress(0.05 + 0.8 * (i + 1) / len(ys), f"taking the stars off: row {i + 1}/{len(ys)}")
    return out / wsum
