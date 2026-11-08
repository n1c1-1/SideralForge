"""The Stars block: a network takes the stars off the image, which gives two images, the starless image and the stars
alone (docs/chantiers/etoiles.md).

The network is NAFNet (Chen, Chu, Zhang, Sun 2022, arXiv:2204.04676), an image restoration network, trained once on
our stacks: starless images corrected by hand, plus stars drawn on top of them (a Moffat core, an r^-3 aureole,
diffraction spikes, saturation), where the truth under each star is exact (tools/bench/stars_train.py, run "mixed").
On the bench it leaves less of the faint stars than the reference tool and takes the spikes and halos of the bright
stars off, the background unchanged (docs/POINT_MATINEE_2026-11-05.md).

It works on what it was trained on: the Output block's linear image (render_simple: the stack with its sky gradient
removed, denoised, white balanced, its sky taken off), each channel minus its background, over one noise unit (the
noise of the half-stacks before denoising, through the colour matrix, averaged over the three colours; a linear Image
has no half-stacks: the spread of its own sky). The network sees a fixed asinh stretch of it, undone exactly on the
way out.

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

from ..core import devices
from ..core.imgutil import asinh_preview
from ..core.params import StageParams
from ..core.stage import Context, Stage

MODEL = Path(__file__).resolve().parents[1] / "resources" / "stars" / "starless_nafnet.pt"
MODEL_ID = hashlib.sha256(MODEL.read_bytes()).hexdigest()[:10] if MODEL.exists() else "missing"
#: the network's stretch: it sees asinh(v / STRETCH) / SCALE, v in noise units
STRETCH, SCALE = 3.0, 10.0
#: the size the network was trained on, and its tiles on an image that is smaller than a big tile
NET_TILE, NET_OVERLAP = 256, 64
#: the tiles it works on: larger than the largest star halos (M45: 280 px), so that the tile holding a star's core also
#: holds its halo. Tiles of the training size cut such a halo along their boundary: the tile with the core takes it off,
#: its neighbour, which only sees halo, leaves it (issue #45; 1024/256 on M45: 13 s, 256/64: 9 s, on an RTX 4060 Ti)
TILE, OVERLAP = 1024, 256
#: its channel attention averages over this window instead of over the whole tile, the training size. It is how the
#: network's authors run it on more than its training size: Test-time Local Converter (Chu, Chen, Chen, Lu 2022, "Improving
#: Image Restoration by Revisiting Global Information Aggregation", arXiv:2112.04491; NAFNetLocal in
#: github.com/megvii-research/NAFNet). Their default is 1.5 x the training size; their reported range starts at the
#: training size, which on M45 leaves no ridge on the tile bands where 384 leaves a few (max 3 against 17 noise units)
WINDOW = NET_TILE
#: tiles of NET_TILE through the network at once (a big tile counts for its area), and the card memory one of them takes:
#: 1.4 GB measured for 16 on 8 November 2026, in bfloat16; a 4 GB card holds the whole batch
BATCH, TILE_GB = 16, 0.09


class StarRemovalParams(StageParams):
    """No setting: the network is applied as it was trained (its outputs are adjusted in Post)."""


class StarRemoval(Stage):
    name = "star_removal"
    title = "Star removal"
    description = ("A network trained on our stacks takes the stars off the Output block's image: the starless image "
                   "and the stars alone, two images that Post combines (the Combine layers block) or takes as "
                   "they are.")
    deps = ("render_simple", "stack")
    Params = StarRemovalParams
    version = f"3-{MODEL_ID}"            # 2: the network's grid notched (core/periodic.py, issue #28); 3: big tiles (#45)

    def run(self, ctx: Context, p: StarRemovalParams) -> None:
        from .denoise import norm_params
        from .gradient import stack_image
        ctx.progress(0.02, "loading the image")
        prep = ctx.load_json("render_simple", "prep_simple.json")
        lin = ctx.load("render_simple", "lin.npy").astype(np.float32)
        valid = ctx.load("render_simple", "valid.npy")
        if ctx.load_json("stack", "drz_meta.json").get("starless"):
            self.fused(ctx, prep, lin, valid)          # a starless fusion (option A): its star layer, no network
            return
        M = np.asarray(prep["M"], np.float64)
        x0, x1, y0, y1 = prep["crop"]
        shown = np.zeros_like(valid)
        shown[y0:y1, x0:x1] = True
        shown &= valid
        if (ctx.project.stage_dir("stack") / "drz_A.npy").is_file():
            _bk, sg, _nm = norm_params(stack_image(ctx, "drz_A.npy"), ctx.load("stack", "drz_weight.npy", mmap=True))
            unit = float(np.mean(np.sqrt((M ** 2) @ (np.asarray(sg, np.float64) ** 2))))
        else:                                            # a linear Image (issue #55): no half-stack to read it on
            unit = sky_unit(lin, shown)
        bk = np.array([np.median(lin[c][shown][::7]) for c in range(3)], np.float32)[:, None, None]
        ctx.log(f"noise unit {unit:.4g}, background {', '.join(f'{v:.3g}' for v in bk.ravel())}")
        dev = ctx.device
        net = network(dev)
        t0 = time.time()
        starless = apply(net, (lin - bk) / unit, dev, ctx) * unit + bk
        ctx.log(f"stars taken off in {time.time() - t0:.1f} s on {dev}")
        if np.array_equal(lin[0, ::97, ::97], lin[1, ::97, ::97]) and np.array_equal(lin[2, ::97, ::97],
                                                                                    lin[1, ::97, ::97]):
            starless = np.repeat(starless.mean(0, keepdims=True), 3, axis=0)   # a monochrome image stays grey (#43)
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


    def fused(self, ctx: Context, prep: dict, lin: np.ndarray, valid: np.ndarray) -> None:
        """A starless fusion of filters (stages/fusion.py, option A): the starless image is the fusion's output, the
        stars the fusion's star layer; written as this block writes its result, for Post and the Combine layers block."""
        from .fusion import STARS_FILE
        stars = ctx.load("stack", STARS_FILE).astype(np.float32)
        stars[:, ~valid] = 0
        ctx.save("starless.npy", lin)
        ctx.save("stars.npy", stars)
        np.save(ctx.out("valid.npy"), valid)
        ctx.save_json("prep_stars.json", {"crop": prep["crop"], "M": prep["M"],
                                          "drizzle_scale": prep.get("drizzle_scale", 1.0), "width": prep.get("width"),
                                          "height": prep.get("height"), "unit": 1.0, "background": [0.0, 0.0, 0.0],
                                          "stats_full": prep["stats"], "fused": True})
        ctx.metric("grid_peaks", 0)
        ctx.log("the fusion's starless image and its star layer (the stars were taken off in each channel)")
        x0, x1, y0, y1 = prep["crop"]
        h, w = lin.shape[1:]
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        s = min(800, h, w) // 2
        box = (slice(None), slice(max(0, cy - s), cy + s), slice(max(0, cx - s), cx + s))
        a, b = (lin + stars)[box], lin[box]
        ctx.preview("starless", asinh_preview(np.concatenate([a, np.zeros((3, a.shape[1], 6), np.float32), b], 2)))


class _LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / (var + 1e-6).sqrt() * self.w[:, None, None] + self.b[:, None, None]


class _LocalPool(nn.Module):
    """The channel attention's average. In training, over the whole tile, as published. Applied, over a ``window`` around
    each position when the features are larger than that (Test-time Local Converter, see WINDOW): a tile larger than the
    training size then sees what the network was trained on."""

    def __init__(self, window):
        super().__init__()
        self.window = window

    def forward(self, x):
        h, w = x.shape[2:]
        if self.training or (self.window >= h and self.window >= w):
            return x.mean((2, 3), keepdim=True)
        k1, k2 = min(self.window, h), min(self.window, w)
        s = nn.functional.pad(x.float().cumsum(2).cumsum(3), (1, 0, 1, 0))            # float32: bfloat16 cannot sum
        o = (s[:, :, k1:, k2:] + s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2]) / (k1 * k2)
        dh, dw = h - o.shape[2], w - o.shape[3]
        return nn.functional.pad(o, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), mode="replicate").to(x.dtype)


class _NAFBlock(nn.Module):
    def __init__(self, c, window=WINDOW):
        super().__init__()
        self.n1, self.n2 = _LayerNorm2d(c), _LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, 2 * c, 1)
        self.c2 = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca = nn.Sequential(_LocalPool(window), nn.Conv2d(c, c, 1))
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
        c, win = width, WINDOW                                        # win: WINDOW in this level's own pixels
        for n in enc:
            self.encs.append(nn.Sequential(*[_NAFBlock(c, win) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            c, win = 2 * c, win // 2
        self.mid = nn.Sequential(*[_NAFBlock(c, win) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c, win = c // 2, win * 2
            self.decs.append(nn.Sequential(*[_NAFBlock(c, win) for _ in range(n)]))
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


def sky_unit(lin: np.ndarray, shown: np.ndarray) -> float:
    """The noise unit of an image without half-stacks: the spread of its sky over the shown pixels, in the colours of
    ``lin`` (its colour matrix is applied already), averaged over the three colours; 1 for an image without noise."""
    from .render_simple import sky_spread
    return float(np.mean([sky_spread(lin[c][shown][::7]) for c in range(3)])) or 1.0


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
def apply(net, img: np.ndarray, dev, ctx: Context | None = None, tile: int = TILE, overlap: int = OVERLAP,
          batch: int = BATCH) -> np.ndarray:
    """The network over a whole (3, H, W) image in noise units, in tiles larger than the largest stars (an image smaller
    than such a tile: tiles of the training size), blended with a cosine window; returns the starless image in the same
    units. On a card with little free memory the tiles go through fewer at a time, then are cut smaller; a memory error
    does the same (the batch size changes nothing in the result, the tile a little)."""
    if min(img.shape[1:]) < tile:
        tile, overlap = min(tile, NET_TILE), min(overlap, NET_OVERLAP)
    tile_gb = TILE_GB * (tile / NET_TILE) ** 2
    fits = devices.tile_that_fits(tile, dev, tile_gb)
    if fits < tile:
        overlap, tile_gb = overlap * fits // tile, tile_gb * (fits / tile) ** 2
        tile = fits
    batch = max(1, batch * NET_TILE ** 2 // tile ** 2)                # the same memory whatever the tile
    free = devices.free_memory(dev)
    if free is not None:
        batch = max(1, min(batch, int(free / 2 ** 30 / (tile_gb * 1.5))))
    while True:
        try:
            return _apply(net, img, dev, ctx, tile, overlap, batch)
        except Exception as e:  # noqa: BLE001 - only a memory error is retried
            if not devices.is_out_of_memory(e):
                raise
            devices.empty_cache(dev)
            if batch > 1:
                batch //= 2
            elif tile // 2 >= devices.SMALLEST_TILE:
                tile, overlap = tile // 2, overlap // 2
            else:
                raise


def _apply(net, img: np.ndarray, dev, ctx: Context | None, tile: int, overlap: int, batch_size: int) -> np.ndarray:
    img = np.asarray(img, np.float32)
    C, H, W = img.shape
    if H < tile or W < tile:                                    # a tiny image: padded to one tile
        ph, pw = max(0, tile - H), max(0, tile - W)
        return _apply(net, np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="reflect"), dev, ctx, tile, overlap,
                      batch_size)[:, :H, :W]
    step = tile - overlap
    out = np.zeros_like(img, np.float32)
    wsum = np.zeros((H, W), np.float32)
    win1 = np.sin(np.linspace(0, np.pi, tile + 2)[1:-1]) ** 2 + 1e-3
    win = np.outer(win1, win1).astype(np.float32)
    ys = list(range(0, H - tile + 1, step)) + ([H - tile] if (H - tile) % step else [])
    xs = list(range(0, W - tile + 1, step)) + ([W - tile] if (W - tile) % step else [])
    for i, y in enumerate(ys):
        batch = [img[:, y:y + tile, x:x + tile] for x in xs]
        for j in range(0, len(batch), batch_size):
            b = torch.from_numpy(np.ascontiguousarray(np.stack(batch[j:j + batch_size]))).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                r = net(fwd(b)).float()
            r = inv(r).cpu().numpy()
            for k, x in enumerate(xs[j:j + batch_size]):
                out[:, y:y + tile, x:x + tile] += r[k] * win
                wsum[y:y + tile, x:x + tile] += win
        if ctx is not None:
            ctx.check_stop()
            ctx.progress(0.05 + 0.8 * (i + 1) / len(ys), f"taking the stars off: row {i + 1}/{len(ys)}")
    return out / wsum
