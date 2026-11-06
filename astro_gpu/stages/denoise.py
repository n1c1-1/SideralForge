"""The Denoise block: a Noise2Noise network pretrained on many half-stack pairs, applied to this stack, and, as the
second method, adapted to it first (docs/chantiers/debruitage.md).

Noise2Noise: the network learns to predict one half-stack from the other; their noise is independent, so the only
thing it can learn is the sky common to both, never detail that is not in the data. Pretrained on 76 pairs (a varied
dataset stacked by this chain at four signal-to-noise levels, and projects), it beat on the bench a network trained
from scratch on each image (21 min of GPU) and GraXpert; adapting it to the image (all its weights, a few hundred
Noise2Noise steps on the half-stacks) helps most when the image differs from what it has seen: a session without
dithering, whose noise is independent from one pixel to the next, needs the long adaptation.

Methods (the workflow's ``denoise`` option, ``method`` here): ``pretrained`` the network as it is (seconds, also on a
processor); ``adapted`` the network adapted first (``adaptation``: short, about 40 s on a GPU; long, about 5 min).
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
PATCH, BATCH = 192, 12


class DenoiseParams(StageParams):
    method: str = P("pretrained", label="Method", choices=list(METHODS),
                    help="Pretrained: the network as it is, in seconds (also on a processor). Adapted: the network "
                         "first learns this image's noise for a while on the half-stacks.")
    adaptation: str = P("short", label="Adaptation", choices=list(ADAPTATIONS),
                        help="How long the network learns this image before denoising it: short, 500 steps (about "
                             "40 s on a GPU); long, 4000 steps (about 5 min).",
                        tip="Long for a session without dithering (the noise independent from one pixel to the next), "
                            "which the pretrained network has not seen; short otherwise. On a processor, a step takes "
                            "about 4 s.")
    seed: int = P(0, label="Random seed", level=ADVANCED, ge=0,
                  help="Seed of the random tile selection of the adaptation (for reproducible runs).")
    #: set by the graph (issue #47): a later Denoise reads this one's denoised half-stacks
    keep_halves: bool = P(False, label="Keep the half-stacks", level="internal",
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
    version = f"2-{MODEL_ID}"

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
        net = pretrained(dev)
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
        dA = apply_net(net, A, nm, bk, sg, dev)
        dB = apply_net(net, B, nm, bk, sg, dev)
        ctx.progress(0.96, "denoising the full stack")
        full = image_in(ctx).img
        # a monochrome stack (its three channels one, core/cfa.MONO): the network, trained on colour, moves them a
        # little apart; their mean, in the three, keeps it grey (issue #43)
        mono = bool(np.array_equal(full[0, ::97, ::97], full[1, ::97, ::97])
                    and np.array_equal(full[2, ::97, ::97], full[1, ::97, ::97]))
        grey = (lambda a: np.repeat(a.mean(0, keepdims=True), 3, axis=0)) if mono else (lambda a: a)
        dA, dB = grey(dA), grey(dB)
        ctx.save("n2n_avg.npy", 0.5 * (dA + dB))
        dF = grey(apply_net(net, full, (nm / np.sqrt(2)).astype(np.float32), bk, sg, dev))
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
        h, w = full.shape[1:]
        y0, x0 = h // 2 - 400, w // 2 - 400
        a = asinh_preview(full[:, y0:y0 + 800, x0:x0 + 800])
        b = asinh_preview(dF[:, y0:y0 + 800, x0:x0 + 800])
        ctx.preview("before_after", np.concatenate([a, np.ones((3, 800, 6)) * 0.2, b], 2))


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


def pretrained(dev) -> nn.Module:
    """The bundled pretrained network (resources/denoise), in float32."""
    if not MODEL.exists():
        raise FileNotFoundError(f"the pretrained denoising network is missing: {MODEL}")
    net = UNet().to(dev)
    state = torch.load(MODEL, map_location=dev, weights_only=False)["net"]
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
    tA, tB = torch.from_numpy(A).to(dev), torch.from_numpy(B).to(dev)
    tN = torch.from_numpy(nm).to(dev)[None]
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
        xin, tgt, nmp = torch.stack(xin), torch.stack(tgt), torch.stack(nmp)
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
def apply_net(net, x, nm, bk, sg, dev, tile=512, pad=48):
    """Denoise a (3, H, W) linear image tile by tile with overlap."""
    Hh, Ww = x.shape[1:]
    out = np.zeros_like(x)
    tx = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    tn = torch.from_numpy(np.ascontiguousarray(nm)).to(dev)[None]
    bk_t = torch.as_tensor(np.asarray(bk), device=dev)
    sg_t = torch.as_tensor(np.asarray(sg), device=dev)
    for y in range(0, Hh, tile):
        for x0 in range(0, Ww, tile):
            y1, x1 = max(0, y - pad), max(0, x0 - pad)
            y2, x2 = min(Hh, y + tile + pad), min(Ww, x0 + tile + pad)
            ph, pw = (y2 - y1), (x2 - x1)
            ph8, pw8 = (ph + 7) // 8 * 8, (pw + 7) // 8 * 8
            pt = F.pad(tx[None, :, y1:y2, x1:x2], (0, pw8 - pw, 0, ph8 - ph), mode="reflect")
            pn = F.pad(tn[None, :, y1:y2, x1:x2], (0, pw8 - pw, 0, ph8 - ph), mode="reflect")
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
