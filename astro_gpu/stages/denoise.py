"""Noise2Noise denoiser trained on the two independent half-stacks of the very data being processed."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.imgutil import asinh_preview
from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage
from .gradient import stack_image


class DenoiseParams(StageParams):
    iterations: int = P(24000, label="Training iterations", ge=500, le=200000, step=1000,
                        help="Number of training steps of the denoising network on your own data.",
                        tip="More steps give a slightly better denoiser; the gain flattens beyond ~20 000. Training "
                            "is checkpointed every 1000 steps and resumes after an interruption.")
    patch: int = P(192, label="Patch size", level=ADVANCED, ge=64, le=512, step=32, unit="px",
                   help="Size of the image tiles shown to the network at each step.")
    batch: int = P(12, label="Batch size", level=ADVANCED, ge=1, le=64,
                   help="Tiles per training step; lower it if the GPU runs out of memory.")
    learning_rate: float = P(3e-4, label="Learning rate", level=ADVANCED, gt=0, le=0.01,
                             help="Peak learning rate of the one-cycle schedule.")
    seed: int = P(0, label="Random seed", level=ADVANCED, ge=0,
                  help="Seed of the random tile selection (for reproducible runs).")
    reuse_model: str = P("", label="Reuse a trained network", level=ADVANCED,
                         help="Path of an n2n_model.pt trained earlier on the same kind of data: no training, the "
                              "network is applied as is (re-runs, A/B tests where only the processing should change).")


class Denoise(Stage):
    name = "denoise"
    title = "Noise2Noise denoiser"
    description = ("Trains a neural network (U-Net) to predict half-stack B from half-stack A and the reverse. "
                   "Their noise is independent, so the only thing the network can learn is the sky common to both: "
                   "it cannot invent detail that is not in the data. Trained on your own images, it knows exactly "
                   "your noise, sampling and stars. Used for measurements and as a source of fine nebula detail.")
    deps = ("stack",)
    Params = DenoiseParams
    version = "1"

    def auto(self, ctx: Context) -> dict[str, Auto]:
        try:
            m = ctx.load_json("stack", "drz_meta.json")
            mp = m["width"] * m["height"] / 1e6
            it = int(np.clip(round(24000 * np.sqrt(mp / 27.3) / 1000) * 1000, 6000, 30000))
            return {"iterations": Auto(it, "{mpx:.0f} Mpx stack", {"mpx": mp})}
        except Exception:  # noqa: BLE001
            return {}

    def run(self, ctx: Context, p: DenoiseParams) -> None:
        dev = ctx.device
        A = stack_image(ctx, "drz_A.npy")
        B = stack_image(ctx, "drz_B.npy")
        Wt = ctx.load("stack", "drz_weight.npy").astype(np.float32)
        bk, sg, nm = norm_params(A, Wt)
        if p.reuse_model:
            ck_ = torch.load(p.reuse_model, map_location=dev, weights_only=False)
            net = UNet().to(dev)
            net.load_state_dict(ck_["net"])
            bk, sg = ck_["bk"], ck_["sg"]
            ctx.log(f"network reused from {p.reuse_model} (no training)")
        else:
            ck = ctx.ckpt / "train.pt"
            net = train(ctx, A, B, nm, bk, sg, p, dev, ck)
        torch.save({"net": net.state_dict(), "bk": bk, "sg": sg}, ctx.out("n2n_model.pt"))
        net.eval()
        ctx.progress(0.92, "denoising the half-stacks")
        dA = apply_net(net, A, nm, bk, sg, dev)
        dB = apply_net(net, B, nm, bk, sg, dev)
        ctx.save("n2n_avg.npy", 0.5 * (dA + dB))
        ctx.progress(0.96, "denoising the full stack")
        full = stack_image(ctx)
        dF = apply_net(net, full, (nm / np.sqrt(2)).astype(np.float32), bk, sg, dev)
        ctx.save("n2n_full.npy", dF)
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


def train(ctx, A, B, nm, bk, sg, p, dev, ck_path):
    steps = p.iterations
    tA, tB = torch.from_numpy(A).to(dev), torch.from_numpy(B).to(dev)
    tN = torch.from_numpy(nm).to(dev)[None]
    bk_t, sg_t = torch.from_numpy(bk).to(dev), torch.from_numpy(sg).to(dev)
    Hh, Ww = tA.shape[1:]
    valid = (tN[0] > 0) & (tN[0] < 1.6)
    torch.manual_seed(p.seed)                       # before the network is made: its starting weights are drawn too
    net = UNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=p.learning_rate, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p.learning_rate, total_steps=steps, pct_start=0.05)
    start, run = 0, 0.0
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, run = ck["it"], ck["run"]
        torch.set_rng_state(ck["rng"].cpu())        # the checkpoint was loaded onto the GPU
        ctx.log(f"resuming the training at step {start}/{steps}")
    t0 = time.time()
    PATCH, BATCH = p.patch, p.batch
    for it in range(start, steps):
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
            fl = bool(torch.rand(1) < 0.5)
            p_in, p_t, p_n = (torch.rot90(t, k, (1, 2)) for t in (p_in, p_t, p_n))
            if fl:
                p_in, p_t, p_n = (torch.flip(t, (2,)) for t in (p_in, p_t, p_n))
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
        pred = a_in + out
        b_t = (tgt - bk_t.view(1, 3, 1, 1)) / sg_t.view(1, 3, 1, 1)
        with torch.no_grad():
            sm = F.avg_pool2d(a_in, 5, 1, 2).clamp(min=0)
            wgt = 1.0 / ((nmp ** 2).clamp(min=0.25) + sm / 40.0)
        loss = (wgt * (pred - b_t) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        if (it + 1) % 100 == 0:
            ctx.progress(0.9 * (it + 1) / steps, f"training step {it + 1}/{steps}, loss {run:.4f}")
        if (it + 1) % 1000 == 0 and it + 1 < steps:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "it": it + 1, "run": run, "rng": torch.get_rng_state()}, ck_path)
    ctx.log(f"training done in {(time.time() - t0) / 60:.1f} min, final loss {run:.4f}")
    return net


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
