"""Evaluation of the pretrained denoiser against our per-image Noise2Noise (docs/chantiers/debruitage.md).

    python tools/bench/denoise_eval.py [--cases heart,m8,...] [--model F:/astro_essais/denoise_dataset/pretrained_unet.pt]
                                       [--methods pretrained,ft500,ft2000,gain500,scratch] [--scratch-steps 24000]

On each bench case (held out of the pretraining), the half-stack pair is cut in two zones: the methods that learn on
the image (fine-tuning, GainTuning, our per-image Noise2Noise from scratch) learn on the first 80 % of its width only,
and every method is measured on the last 20 %, never seen. The measure is the Noise2Noise loss of the training (the
prediction from one half-stack against the other, weighted by the expected noise), both ways, over the whole held-out
zone: its expectation is the error to the true sky plus the noise of the target, the same for every method, so its
differences are the methods' differences. Also: the time on the GPU and on the CPU, and crops (input, each output) in
F:/astro_essais/denoise_eval/<case>.jpg.

Methods: ``none`` no denoising (the prediction is the input: its loss is about twice the floor the target's own noise
sets, so ``none / 2`` estimates that floor and each method's loss minus it estimates its error to the true sky);
``pretrained`` the model as it is; ``ftN`` all its weights trained N more steps on the image (Noise2Noise, a
small learning rate); ``gainN`` only a gain per channel of each convolution trained N steps (GainTuning, Mohan et al.,
NeurIPS 2021); ``scratch`` a new network trained on the image alone (our Denoise stage today).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from astro_gpu.stages.denoise import UNet  # noqa: E402

DATA = Path("F:/astro_essais/denoise_dataset")
OUT = Path("F:/astro_essais/denoise_eval")
CASES = ("heart", "iris", "m51", "m8", "rosette", "seestar")
SPLIT = 0.8                   # the methods learn left of this fraction of the width, all are measured right of it
MARGIN = 192                  # the image the networks see left of the measured zone (more than their receptive field)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load(case: str):
    d = DATA / f"bench_{case}"
    A = np.load(d / "A.npy").astype(np.float32)
    B = np.load(d / "B.npy").astype(np.float32)
    N = np.load(d / "nm.npy").astype(np.float32)[None]
    return A, B, N


def net_in(x, n):
    return torch.cat([torch.asinh(x / 8.0), n], 1)


def weighted(pred, tgt, xin, n):
    """The training's loss: squared error weighted by the expected noise (stages/denoise.train)."""
    sm = F.avg_pool2d(xin, 5, 1, 2).clamp(min=0)
    w = 1.0 / ((n ** 2).clamp(min=0.25) + sm / 40.0)
    return w * (pred - tgt) ** 2


@torch.no_grad()
def apply(net, x, n, dev, tile=512, pad=48):
    """The network on a normalised (3, H, W) image, tile by tile with overlap (as stages/denoise.apply_net)."""
    H, W = x.shape[1:]
    out = np.zeros_like(x)
    tx = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    tn = torch.from_numpy(np.ascontiguousarray(n)).to(dev)
    for y in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = max(0, y - pad), max(0, x0 - pad)
            y2, x2 = min(H, y + tile + pad), min(W, x0 + tile + pad)
            ph, pw = y2 - y1, x2 - x1
            ph8, pw8 = (ph + 7) // 8 * 8, (pw + 7) // 8 * 8
            pt = F.pad(tx[None, :, y1:y2, x1:x2], (0, pw8 - pw, 0, ph8 - ph), mode="reflect")
            pn = F.pad(tn[None, :, y1:y2, x1:x2], (0, pw8 - pw, 0, ph8 - ph), mode="reflect")
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = net(net_in(pt, pn)).float()
            res = (pt + o)[0, :, :ph, :pw]
            yy, xx = y - y1, x0 - x1
            out[:, y:min(y + tile, H), x0:min(x0 + tile, W)] = \
                res[:, yy:yy + min(tile, H - y), xx:xx + min(tile, W - x0)].cpu().numpy()
    return out


def held_out_loss(net, A, B, N, dev, x0: int) -> float:
    """The Noise2Noise loss, both ways, over the held-out zone (x >= x0; a margin of the receptive field kept)."""
    zA, zB, zN = A[:, :, x0 - MARGIN:], B[:, :, x0 - MARGIN:], N[:, :, x0 - MARGIN:]
    tot, cnt = 0.0, 0
    for src, tgt in ((zA, zB), (zB, zA)):
        pred = apply(net, src, zN, dev)
        p, t, s, n = (torch.from_numpy(a[:, :, MARGIN:])[None] for a in (pred, tgt, src, zN))
        ok = (n[0, 0] > 0) & (n[0, 0] < 1.6)
        L = weighted(p, t, s, n)[0].mean(0)
        tot += float(L[ok].sum())
        cnt += int(ok.sum())
    return tot / max(cnt, 1)


class Nothing(nn.Module):
    """No denoising: the correction is zero, the prediction is the input."""

    def forward(self, x):
        return torch.zeros_like(x[:, :3])


class Gain(nn.Module):
    """A convolution followed by a learnable gain per output channel (GainTuning)."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        self.g = nn.Parameter(torch.ones(conv.out_channels, 1, 1))

    def forward(self, x):
        return self.conv(x) * self.g


def with_gains(net: nn.Module) -> list[nn.Parameter]:
    """Every convolution of ``net`` wrapped with a gain; returns the gains (the only parameters then trained)."""
    gains = []
    for _name, mod in list(net.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)) and not isinstance(mod, Gain):
                g = Gain(child)
                setattr(mod, cname, g)
                gains.append(g.g)
    for p in net.parameters():
        p.requires_grad_(False)
    for g in gains:
        g.requires_grad_(True)
    return gains


def learn(net, params, A, B, N, dev, steps: int, lr: float, x1: int, patch=192, batch=12, seed=0) -> float:
    """Noise2Noise on random tiles of the learning zone (x < x1); returns the seconds."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0 if len(params) and params[0].ndim == 3 else 1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    H = A.shape[1]
    tA, tB, tN = (torch.from_numpy(np.ascontiguousarray(a)).to(dev) for a in (A, B, N))
    t0 = time.time()
    net.train()
    for _ in range(steps):
        xs, ts, ns = [], [], []
        while len(xs) < batch:
            y, x = int(rng.integers(0, H - patch)), int(rng.integers(0, x1 - patch))
            c = float(N[0, y + patch // 2, x + patch // 2])
            if not 0 < c < 1.6:
                continue
            s, t = (tA, tB) if rng.random() < 0.5 else (tB, tA)
            k = int(rng.integers(4))
            a, b, n = (torch.rot90(z[:, y:y + patch, x:x + patch], k, (1, 2)) for z in (s, t, tN))
            xs.append(a)
            ts.append(b)
            ns.append(n)
        xin, tgt, nmp = torch.stack(xs), torch.stack(ts), torch.stack(ns)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = net(net_in(xin, nmp)).float()
        loss = weighted(xin + out, tgt, xin, nmp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
    net.eval()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


def crops(case, A, outs: dict, x0: int):
    """Two crops of the held-out zone (the brightest and a faint area), input then each output, one stretch."""
    import cv2
    H, W = A.shape[1:]
    L = A[1, :, x0:]
    blk = cv2.resize(L, (max(1, (W - x0) // 64), max(1, H // 64)), interpolation=cv2.INTER_AREA)
    yb, xb = np.unravel_index(np.argmax(blk), blk.shape)
    yf, xf = np.unravel_index(np.argmin(np.abs(blk - np.percentile(blk, 60))), blk.shape)
    S = 320
    rows = []
    for (yc, xc) in ((yb * 64 + 32, x0 + xb * 64 + 32), (yf * 64 + 32, x0 + xf * 64 + 32)):
        y1 = int(np.clip(yc - S // 2, 0, H - S))
        x1 = int(np.clip(xc - S // 2, x0, W - S))
        tiles = []
        for name, im in [("input A", A)] + list(outs.items()):
            c = im[:, y1:y1 + S, x1:x1 + S]
            v = np.arcsinh(np.clip(c + 2, 0, None) / 2) / np.arcsinh(60)
            t = (np.clip(v, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
            t = np.ascontiguousarray(t)
            cv2.putText(t, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(t)
        rows.append(np.hstack(tiles))
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{case}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--model", default=str(DATA / "pretrained_unet.pt"))
    ap.add_argument("--methods", default="none,pretrained,ft500,ft2000,gain500,scratch")
    ap.add_argument("--scratch-steps", type=int, default=24000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(OUT / "results.json"))
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    res_f = Path(a.out)
    res_f.parent.mkdir(parents=True, exist_ok=True)
    allres = json.loads(res_f.read_text()) if res_f.exists() else {}
    state = torch.load(a.model, map_location=dev, weights_only=False)["net"]
    for case in a.cases.split(","):
        A, B, N = load(case)
        W = A.shape[2]
        x0 = int(W * SPLIT)
        r = allres.setdefault(case, {})
        outs = {}
        for m in a.methods.split(","):
            torch.manual_seed(0)
            net = (Nothing() if m == "none" else UNet()).to(dev)
            secs = 0.0
            if m not in ("scratch", "none"):
                net.load_state_dict(state)
            if m.startswith("ft"):                       # ftN or ftN@lr: N steps of all weights (lr 5e-5 by default)
                steps, _, lr = m[2:].partition("@")
                secs = learn(net, list(net.parameters()), A, B, N, dev, int(steps), float(lr or 5e-5), x0)
            elif m.startswith("gain"):
                gains = with_gains(net)
                net.to(dev)
                secs = learn(net, gains, A, B, N, dev, int(m[4:]), 1e-3, x0)
            elif m == "scratch":
                secs = learn(net, list(net.parameters()), A, B, N, dev, a.scratch_steps, 3e-4, x0)
            net.eval()
            t = time.time()
            loss = held_out_loss(net, A, B, N, dev, x0)
            r[m] = {"held_out_loss": round(loss, 4), "learn_s": round(secs, 1),
                    "apply_s": round(time.time() - t, 1)}
            log(f"{case} {m}: {r[m]}")
            outs[m] = apply(net, A[:, :, x0 - MARGIN:], N[:, :, x0 - MARGIN:], dev)
            res_f.write_text(json.dumps(allres, indent=1))
        full = {k: np.concatenate([np.zeros((3, A.shape[1], x0 - MARGIN), np.float32), v], 2)
                for k, v in outs.items()}
        crops(case, A, full, x0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
