"""The Denoise block: a Noise2Noise network pretrained on many half-stack pairs, applied to this stack, and, as the
second method, adapted to it first (docs/chantiers/debruitage.md).

Noise2Noise: the network learns to predict one half-stack from the other; their noise is independent, so the only
thing it can learn is the sky common to both, never detail that is not in the data. Pretrained on 76 pairs (a varied
dataset stacked by this chain at four signal-to-noise levels, and projects), it beat on the bench a network trained
from scratch on each image (21 min of GPU) and GraXpert; adapting it to the image (all its weights, a few hundred
Noise2Noise steps on the half-stacks) helps most when the image differs from what it has seen: a session without
dithering, whose noise is independent from one pixel to the next, needs the long adaptation.

Methods (the workflow's ``denoise`` option, ``method`` here): ``pretrained`` the network as it is (seconds, also on a
processor); ``adapted`` the network adapted first (``adaptation``: short, about 40 s on a GPU; long, about 5 min). With
``pretrained``, a monochrome stack of few frames has its own network (issue #45): the colour one turns its noise into
star-like dots. Fine-tuned on monochrome half-stacks (tools/bench/denoise_mono.py), it removes them and denoises as much,
but brightens the faint stars by 8 to 11 % on the bench: for pictures, not for photometry.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import field_validator

from ..core import devices
from ..core.imgutil import asinh_preview
from ..core.params import ADVANCED, Auto, P, StageParams, check_choice
from ..core.stage import Context, Stage
from .gradient import image_in

METHODS = ("pretrained", "adapted")
#: the adaptation lengths: (steps, peak learning rate), as measured on the bench (docs/chantiers/debruitage.md §5)
ADAPTATIONS = {"short": (500, 2e-4), "long": (4000, 3e-4)}
#: below this correlation of the noise between neighbouring pixels the session was not dithered: the pretrained
#: network has not seen such noise (its pairs go from 0.09 to 0.42; the bench's undithered M8: 0.03)
UNDITHERED_CORRELATION = 0.08
MODEL = Path(__file__).resolve().parents[1] / "resources" / "denoise" / "pretrained_unet.pt"
MODEL_ID = hashlib.sha256(MODEL.read_bytes()).hexdigest()[:10] if MODEL.exists() else "missing"
#: the network of the monochrome stacks of fewer light frames than MONO_FEW_FRAMES (issue #45): the colour one turns the
#: noise of those stacks into star-like dots (found again in the other half-stack: 12 and 13 frames 0-7 %, 20 frames
#: 33-44 %, against 100 % at 34 frames and 84-98 % at 40)
MODEL_MONO = MODEL.with_name("pretrained_unet_mono.pt")
MODEL_MONO_ID = hashlib.sha256(MODEL_MONO.read_bytes()).hexdigest()[:10] if MODEL_MONO.exists() else "missing"
MONO_FEW_FRAMES = 30
#: the Denoise panel's word on that network (the Pretrained method's text, api._METHODS, says the same)
MONO_TIP = ("A monochrome stack of fewer than 30 frames gets a network made for monochrome noise. It slightly brightens "
            "faint stars (+8 to +11 % on the bench): it is for pictures, not for photometry.")
PATCH, BATCH = 192, 12
#: the network's passes over the image, each with the image moved by (y, x) pixels (Lim et al. 2017, self-ensemble): the
#: network leaves a pattern on its own grid of 2, 4 and 8 px, which falls elsewhere on the image at each pass. These
#: four cancel the period of 2 px exactly and take the whole to less than half (issue #84: Rosette and Pleiades, the
#: stars unchanged, the error against the other half-stack 1.2 to 1.4 % lower)
PASSES = ((0, 0), (0, 1), (1, 0), (1, 1))
#: the card memory the network takes for a tile of 512 (with its padding): 0.21 GB measured on 8 November 2026 for 512
#: pixels in bfloat16, plus the padding's area; a 4 GB card holds it many times over
TILE_GB = 0.35


class DenoiseParams(StageParams):
    method: str = P("pretrained", label="Method", choices=list(METHODS),
                    help="Pretrained: the network as it is, in seconds (also on a processor). Adapted: the network "
                         "first learns this image's noise for a while on the half-stacks.",
                    tip=MONO_TIP)
    adaptation: str = P("short", label="Adaptation", choices=list(ADAPTATIONS),
                        help="How long the network learns this image before denoising it: short, 500 steps (about "
                             "40 s on a GPU); long, 4000 steps (about 5 min).",
                        tip="Long for a session without dithering (the noise independent from one pixel to the next), "
                            "which the pretrained network has not seen; short otherwise. On a processor, a step takes "
                            "about 4 s.")
    fast: bool = P(False, label="Fast pass",
                   help="One pass of the network instead of four: four times faster, but its grid pattern can show in "
                        "the sky under a strong stretch.")
    seed: int = P(0, label="Random seed", level=ADVANCED, ge=0,
                  help="Seed of the random tile selection of the adaptation (for reproducible runs).")
    #: set by the graph (issue #47): a later Denoise reads this one's denoised half-stacks
    keep_halves: bool = P(False, label="Keep half-stacks", level="internal",
                          help="Saves the denoised half-stacks too, for a Denoise block placed after this one.")

    @field_validator("method")
    @classmethod
    def _method(cls, v: str) -> str:
        return check_choice("method", v, METHODS)

    @field_validator("adaptation")
    @classmethod
    def _adaptation(cls, v: str) -> str:
        return check_choice("adaptation", v, tuple(ADAPTATIONS))


class Denoise(Stage):
    name = "denoise"
    title = "Denoise"
    description = ("A Noise2Noise network, pretrained on many pairs of half-stacks, removes the noise of the stack: "
                   "trained to predict one half-stack from the other, it cannot invent detail that is not in the "
                   "data. Optionally adapted to this image first.")
    deps = ("stack",)
    Params = DenoiseParams
    version = f"4-{MODEL_ID}"     # 4: the network's four passes (PASSES); 3: the sky level of each colour kept (keep_level)

    def mono_key_version(self, values: dict, lights: int) -> str:
        """The version in the store key of a monochrome workflow of ``lights`` frames: with its own network when it has
        few, for the Pretrained method (issue #45). No other key moves."""
        own = uses_mono_network(values.get("method", "pretrained"), lights)
        return f"{self.version}+mono-{MODEL_MONO_ID}" if own else str(self.version)

    def auto(self, ctx: Context) -> dict[str, Auto]:
        try:
            c = noise_correlation(np.load(ctx.input("stack", "drz_A.npy"), mmap_mode="r"),
                                  np.load(ctx.input("stack", "drz_B.npy"), mmap_mode="r"))
        except Exception:  # noqa: BLE001
            return {}
        if c < UNDITHERED_CORRELATION:
            return {"adaptation": Auto("long", "noise nearly independent between neighbouring pixels (correlation "
                                               "{c:.2f}): a session without dithering", {"c": c})}
        return {"adaptation": Auto("short", "noise correlation between neighbouring pixels {c:.2f}, as in the "
                                            "pretraining", {"c": c})}

    def run(self, ctx: Context, p: DenoiseParams) -> None:
        dev = ctx.device
        A = image_in(ctx, "A").img                   # the half-stacks as the blocks before it made them (#47)
        B = image_in(ctx, "B").img
        Wt = ctx.load("stack", "drz_weight.npy").astype(np.float32)
        bk, sg, nm = norm_params(A, Wt)
        corr = noise_correlation(ctx.load("stack", "drz_A.npy", mmap=True), ctx.load("stack", "drz_B.npy", mmap=True))
        ctx.metric("noise_correlation", round(corr, 3))
        full = image_in(ctx).img
        mono = is_grey(full)                         # a monochrome stack (its three channels one, core/cfa.MONO)
        keys, w = getattr(ctx.project, "keys", None), getattr(ctx.project, "workflow", None)
        lights = keys.mono_lights(w) if keys is not None and w is not None else None      # as the key counts them
        own = mono and lights is not None and uses_mono_network(p.method, lights)
        net = pretrained(dev, mono=own)
        if own:
            ctx.log(f"monochrome stack of {lights} frames: the network made for monochrome noise")
        if p.method == "adapted":
            steps, lr = ADAPTATIONS[p.adaptation]
            adapt(ctx, net, A, B, nm, bk, sg, dev, steps, lr, p.seed)
            ctx.metric("adaptation_steps", steps)
        else:
            ctx.metric("adaptation_steps", 0)
            if corr < UNDITHERED_CORRELATION:
                ctx.log(f"the noise is nearly independent between neighbouring pixels (correlation {corr:.2f}, a "
                        "session without dithering): the pretrained network has not seen such noise, the Adapted "
                        "method (long) denoises it much better", "warning")
        torch.save({"net": net.state_dict(), "bk": bk, "sg": sg}, ctx.out("n2n_model.pt"))
        net.eval()
        ctx.progress(0.92, "denoising the half-stacks")
        shifts = PASSES[:1] if p.fast else PASSES
        dA = apply_net(net, A, nm, bk, sg, dev, shifts=shifts, check=ctx.check_stop)
        dB = apply_net(net, B, nm, bk, sg, dev, shifts=shifts, check=ctx.check_stop)
        ctx.progress(0.96, "denoising the full stack")
        # a monochrome stack: the network moves the three channels a little apart; their mean, in the three, keeps it
        # grey (issue #43)
        grey = (lambda a: np.repeat(a.mean(0, keepdims=True), 3, axis=0)) if mono else (lambda a: a)
        dA, dB = keep_level(grey(dA), A, Wt), keep_level(grey(dB), B, Wt)
        ctx.save("n2n_avg.npy", 0.5 * (dA + dB))
        dF = keep_level(grey(apply_net(net, full, (nm / np.sqrt(2)).astype(np.float32), bk, sg, dev, shifts=shifts,
                                         check=ctx.check_stop)), full, Wt)
        ctx.save("n2n_full.npy", dF)
        if p.keep_halves:                            # for a later Denoise block (issue #47)
            ctx.save("n2n_A.npy", dA)
            ctx.save("n2n_B.npy", dB)
        g0 = _noise(full[1])
        g1 = _noise(dF[1])
        ctx.metric("noise_before", round(g0, 3))
        ctx.metric("noise_after", round(g1, 3))
        ctx.metric("noise_reduction", round(g0 / max(g1, 1e-9), 2))
        ctx.log(f"background noise divided by {g0 / max(g1, 1e-9):.1f}")
        ctx.preview("before_after", before_after(full, dF))


def keep_level(done: np.ndarray, src: np.ndarray, weight: np.ndarray, step: int = 7, rounds: int = 5,
               k: float = 3.0) -> np.ndarray:
    """``done`` (3, H, W) with the level of each colour of ``src``. The network moves the sky by a small amount that
    differs from colour to colour; the strong stretch of a clean image shows it (a green sky on Iris). Measured on the
    difference, the robust median of ``done - src`` over the covered pixels (every ``step``-th, clipped ``rounds``
    times at ``k`` sigmas): the objects of the field, which the network keeps, do not count, even a nebula that fills
    the field."""
    w = weight if weight.ndim == 2 else weight.min(0)
    covered = w > 0
    out = done.copy()
    for c in range(3):
        v = (done[c] - src[c])[covered][::step]
        v = v[np.isfinite(v)]
        for _ in range(rounds):
            d = np.abs(v - np.median(v))
            v = v[d <= k * 1.4826 * np.median(d)]                # (<=: an exact difference stays)
        if v.size:
            out[c] = np.where(covered, done[c] - float(np.median(v)), done[c])
    return out


def before_after(full: np.ndarray, done: np.ndarray, size: int = 800) -> np.ndarray:
    """The preview: a central square of the image before and after, side by side; a stack smaller than ``size`` gives
    a smaller square (GUI exploration 7.8: it failed under 800 px)."""
    h, w = full.shape[1:]
    s = min(size, h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    a = asinh_preview(full[:, y0:y0 + s, x0:x0 + s])
    b = asinh_preview(done[:, y0:y0 + s, x0:x0 + s])
    return np.concatenate([a, np.ones((3, s, 6)) * 0.2, b], 2)


def noise_correlation(A, B, size: int = 800) -> float:
    """The correlation of the noise between horizontally neighbouring pixels, on the difference of the two half-stacks
    (green, a central square; the sky only: differences under 6 noise sigmas). About 0.15 to 0.4 for a dithered
    session (registration and drizzle spread each pixel over its neighbours), near 0 without dithering."""
    H, W = A.shape[1:]
    y0, x0 = max(0, H // 2 - size // 2), max(0, W // 2 - size // 2)
    d = (np.asarray(A[1, y0:y0 + size, x0:x0 + size], np.float32)
         - np.asarray(B[1, y0:y0 + size, x0:x0 + size], np.float32))
    ok = d != 0
    if ok.sum() < 1000:
        return 0.25
    s = 1.4826 * np.median(np.abs(d[ok] - np.median(d[ok])))
    ok &= np.abs(d - np.median(d[ok])) < 6 * max(s, 1e-12)
    d = np.where(ok, d - d[ok].mean(), 0.0)
    den = float((d ** 2).sum())
    return float((d[:, 1:] * d[:, :-1]).sum() / den) if den > 0 else 0.25


def is_grey(full: np.ndarray) -> bool:
    """Whether a (3, H, W) stack is monochrome: its three channels one (core/cfa.MONO), on a sparse grid of pixels."""
    return bool(np.array_equal(full[0, ::97, ::97], full[1, ::97, ::97])
                and np.array_equal(full[2, ::97, ::97], full[1, ::97, ::97]))


def uses_mono_network(method: str, lights: int) -> bool:
    """Whether a monochrome stack of ``lights`` frames, with this method, is denoised by the monochrome network."""
    return method == "pretrained" and 0 < lights < MONO_FEW_FRAMES and MODEL_MONO.exists()


def pretrained(dev, mono: bool = False) -> nn.Module:
    """The bundled pretrained network (resources/denoise), in float32: with ``mono``, the monochrome stacks' one."""
    path = MODEL_MONO if mono else MODEL
    if not path.exists():
        raise FileNotFoundError(f"the pretrained denoising network is missing: {path}")
    net = UNet().to(dev)
    state = torch.load(path, map_location=dev, weights_only=False)["net"]
    net.load_state_dict({k: v.float() for k, v in state.items()})
    return net


def _noise(a):
    from ..core.imgutil import hp_noise
    return hp_noise(a, sub=5)


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GELU(), nn.Conv2d(co, co, 3, padding=1), nn.GELU())
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x):
        return self.c(x) + self.skip(x)


class UNet(nn.Module):
    def __init__(self, cin=4, cout=3, w=(48, 96, 192, 320)):
        super().__init__()
        self.inc = Block(cin, w[0])
        self.downs = nn.ModuleList([Block(w[i], w[i + 1]) for i in range(len(w) - 1)])
        self.mid = Block(w[-1], w[-1])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(w[i + 1], w[i], 2, 2) for i in reversed(range(len(w) - 1))])
        self.decs = nn.ModuleList([Block(2 * w[i], w[i]) for i in reversed(range(len(w) - 1))])
        self.out = nn.Conv2d(w[0], cout, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        h = skips[0]
        for d in self.downs:
            h = d(F.avg_pool2d(h, 2))
            skips.append(h)
        h = self.mid(skips.pop())
        for up, dec in zip(self.ups, self.decs):
            h = up(h)
            h = dec(torch.cat([h, skips.pop()], 1))
        return self.out(h)


def norm_params(A, Wt):
    """Background and noise per channel of a half-stack, and the relative noise map from the drizzle weights."""
    bk, sg = [], []
    for c in range(3):
        x = A[c][::4, ::4]
        m = np.median(x[x != 0])
        s = 1.4826 * np.median(np.abs(x - m)[x != 0])
        bk.append(m)
        sg.append(s)
    w = Wt.mean(0)
    med = np.median(w[w > 0])
    nm = np.sqrt(med / np.maximum(w, med * 0.05)).astype(np.float32)
    nm[w <= 0] = 0
    return np.array(bk, np.float32), np.array(sg, np.float32), nm


def to_input(x, nm, bk, sg):
    a = (x - bk.view(1, 3, 1, 1)) / sg.view(1, 3, 1, 1)
    return torch.cat([torch.asinh(a / 8.0), nm], 1)


def adapt(ctx, net, A, B, nm, bk, sg, dev, steps: int, lr: float, seed: int) -> None:
    """Noise2Noise steps on random tiles of the half-stacks (one predicts the other, both ways), all the weights,
    a one-cycle learning rate: the network learns this image's noise. The loss is weighted by the expected noise."""
    # the half-stacks stay on the host when the card has little free memory (the tiles go to the card one batch at a time)
    free = devices.free_memory(dev)
    home = dev if free is None or (A.nbytes + B.nbytes + nm.nbytes) * 2 < free else torch.device("cpu")
    tA, tB = torch.from_numpy(A).to(home), torch.from_numpy(B).to(home)
    tN = torch.from_numpy(nm).to(home)[None]
    bk_t, sg_t = torch.from_numpy(bk).to(dev), torch.from_numpy(sg).to(dev)
    Hh, Ww = tA.shape[1:]
    valid = (tN[0] > 0) & (tN[0] < 1.6)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    net.train()
    run = 0.0
    t0 = time.time()
    for it in range(steps):
        ys = torch.randint(0, Hh - PATCH, (BATCH * 2,), device="cpu")
        xs = torch.randint(0, Ww - PATCH, (BATCH * 2,), device="cpu")
        xin, tgt, nmp = [], [], []
        for y, x in zip(ys.tolist(), xs.tolist()):
            if not bool(valid[y + PATCH // 2, x + PATCH // 2]):
                continue
            src, dst = (tA, tB) if torch.rand(1).item() < 0.5 else (tB, tA)
            p_in, p_t, p_n = (src[:, y:y + PATCH, x:x + PATCH], dst[:, y:y + PATCH, x:x + PATCH],
                              tN[:, y:y + PATCH, x:x + PATCH])
            k = int(torch.randint(0, 4, (1,)))
            p_in, p_t, p_n = (torch.rot90(t, k, (1, 2)) for t in (p_in, p_t, p_n))
            xin.append(p_in)
            tgt.append(p_t)
            nmp.append(p_n)
            if len(xin) == BATCH:
                break
        if not xin:
            continue
        xin, tgt, nmp = torch.stack(xin).to(dev), torch.stack(tgt).to(dev), torch.stack(nmp).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = net(to_input(xin, nmp, bk_t, sg_t)).float()
        a_in = (xin - bk_t.view(1, 3, 1, 1)) / sg_t.view(1, 3, 1, 1)
        b_t = (tgt - bk_t.view(1, 3, 1, 1)) / sg_t.view(1, 3, 1, 1)
        with torch.no_grad():
            sm = F.avg_pool2d(a_in, 5, 1, 2).clamp(min=0)
            wgt = 1.0 / ((nmp ** 2).clamp(min=0.25) + sm / 40.0)
        loss = (wgt * (a_in + out - b_t) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        if (it + 1) % 50 == 0:
            ctx.check_stop()
            ctx.progress(0.9 * (it + 1) / steps, f"adapting to the image: step {it + 1}/{steps}, loss {run:.4f}")
    ctx.log(f"adapted to the image in {steps} steps ({(time.time() - t0) / 60:.1f} min), loss {run:.4f}")


@torch.no_grad()
def apply_net(net, x, nm, bk, sg, dev, tile=512, pad=48, shifts=PASSES, check=None):
    """Denoise a (3, H, W) linear image tile by tile with overlap, as the mean of ``shifts``. On a card with little free
    memory the tile is smaller (``devices.tile_that_fits``; a memory error halves it, ``shrink_on_memory``): only then
    does the result differ, by a fraction of the noise, at the seams of the tiles. ``check``: called at each row of
    tiles (the run's Stop: a pass takes minutes on the processor)."""
    tile = devices.tile_that_fits(tile, dev, TILE_GB * (tile / 512) ** 2)

    def passes(t):
        total = 0
        for dy, dx in shifts:
            sx = np.pad(x, ((0, 0), (dy, 0), (dx, 0)), mode="reflect")
            total = total + _apply_net(net, sx, np.pad(nm, ((dy, 0), (dx, 0)), mode="reflect"), bk, sg, dev, t, pad, check)[
                :, dy:, dx:]
        return (total / len(shifts)).astype(x.dtype)

    return devices.shrink_on_memory(passes, tile, dev)


@torch.no_grad()
def _apply_net(net, x, nm, bk, sg, dev, tile, pad, check=None):
    Hh, Ww = x.shape[1:]
    out = np.zeros_like(x)
    # the image goes to the card when it has the room; otherwise each tile does (the same numbers, only slower)
    free = devices.free_memory(dev)
    home = dev if free is None or (x.nbytes + nm.nbytes) * 2 < free else torch.device("cpu")
    tx = torch.from_numpy(np.ascontiguousarray(x)).to(home)
    tn = torch.from_numpy(np.ascontiguousarray(nm)).to(home)[None]
    bk_t = torch.as_tensor(np.asarray(bk), device=dev)
    sg_t = torch.as_tensor(np.asarray(sg), device=dev)
    for y in range(0, Hh, tile):
        if check is not None:
            check()
        for x0 in range(0, Ww, tile):
            y1, x1 = max(0, y - pad), max(0, x0 - pad)
            y2, x2 = min(Hh, y + tile + pad), min(Ww, x0 + tile + pad)
            ph, pw = (y2 - y1), (x2 - x1)
            ph8, pw8 = (ph + 7) // 8 * 8, (pw + 7) // 8 * 8
            pt = F.pad(tx[None, :, y1:y2, x1:x2].to(dev), (0, pw8 - pw, 0, ph8 - ph), mode="reflect")
            pn = F.pad(tn[None, :, y1:y2, x1:x2].to(dev), (0, pw8 - pw, 0, ph8 - ph), mode="reflect")
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = net(to_input(pt, pn, bk_t, sg_t)).float()
            a = (pt - bk_t.view(1, 3, 1, 1)) / sg_t.view(1, 3, 1, 1) + o
            res = (a * sg_t.view(1, 3, 1, 1) + bk_t.view(1, 3, 1, 1))[0, :, :ph, :pw]
            yy, xx = y - y1, x0 - x1
            out[:, y:min(y + tile, Hh), x0:min(x0 + tile, Ww)] = \
                res[:, yy:yy + min(tile, Hh - y), xx:xx + min(tile, Ww - x0)].cpu().numpy()
    return out


def load_model(ctx: Context):
    dev = ctx.device
    ck = torch.load(ctx.input("denoise", "n2n_model.pt"), map_location=dev, weights_only=False)
    net = UNet().to(dev)
    net.load_state_dict(ck["net"])
    net.eval()
    return net, ck["bk"], ck["sg"]
