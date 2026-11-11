"""A deconvolution network trained on sharp space images degraded by our stacks' star shapes and noise.

    python tools/bench/deconv_train.py library       the star shapes of the half-stack pairs (per zone and colour)
    python tools/bench/deconv_train.py train --hours 4 [--tag run1] [--steps N]

The star network's NAFNet, on one colour plane, restarted from its weights. The truth: Euclid mosaics at 1, 2 or 4
arcsec per pixel (tools/bench/deconv_truth.py) and our stacks denoised and reduced (deconv_truth_stacks.py), with stars
added. The input: the truth blurred by a star shape, measured on a stack or drawn (Moffat, coma, astigmatism, defocus,
tracking), plus the noise of a real stack, (A - B) / 2 of a half-stack pair, then through our Denoise: the network comes
after that block and sees what it gives. The target: the truth blurred to a star of 1.3 px, what the sampling can hold.
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
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from astro_gpu.stages import deconv_lib as lib  # noqa: E402
from astro_gpu.stages import denoise as dn  # noqa: E402
from astro_gpu.stages.star_removal import NAFNet, fwd, inv, network  # noqa: E402

TRUTH = Path("E:/float_train/truth")
PAIRS = Path("F:/astro_essais/denoise_dataset")
HOME = Path("E:/float_train")
TRUTH_STACKS = Path("E:/float_train/truth_stacks")
HELD_OUT = ("bench_", "set_02_", "set_03_", "user_Heart", "user_Iris", "user_Rosette", "set_18_", "set_33_", "set_37_", "set_38_")
LEVELS = ("_half", "_quarter", "_eighth")
TILE, BATCH = 256, 8
TARGET_FWHM = 1.3
R = 12                                                 # the radius of the star shapes measured on the stacks
RK = 22                                                # of the kernels of the blur: the widest stars and rings hold in it
#: the share of the star shapes drawn rather than measured, and of the stacks among the truths
DRAWN, STACK_TRUTH = 0.45, 0.5
FLUX_WEIGHT = 0.5


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def full_pairs() -> list[Path]:
    return [d for d in sorted(PAIRS.iterdir()) if (d / "pair.json").exists() and not d.name.startswith(HELD_OUT)
            and not d.name.endswith(LEVELS)]


def library() -> None:
    """Every zone's and colour's star shape of the full pairs, in one array (n, 25, 25); and how wide they are."""
    out, widths = [], []
    dev = "cuda"
    for d in full_pairs():
        a = np.load(d / "A.npy").astype(np.float32)
        b = np.load(d / "B.npy").astype(np.float32)
        nm = np.load(d / "nm.npy").astype(np.float32)
        full = 0.5 * (a + b)
        valid = nm > 0
        sat = np.zeros(valid.shape, bool)
        for c in range(3):
            sat |= full[c] > 0.7 * np.percentile(full[c][valid][::11], 99.999)
        st = lib.StackDeconvSettings(tiles_long=5, radius=R, log=lambda *x, **k: None)
        eng = lib.StackDeconvolver(st, dev)
        t = [torch.from_numpy(x).to(dev) for x in (full, a, b)]
        psf = eng.measure(t[0], t[1], t[2], torch.from_numpy(sat).to(dev), torch.from_numpy(valid).to(dev))
        if psf is None:
            log(d.name, "too few stars")
            continue
        k = psf.kernels.cpu().numpy().reshape(-1, 2 * R + 1, 2 * R + 1)
        out.append(k)
        widths.append(psf.fwhm.ravel())
        log(d.name, len(k), "shapes, width", round(float(np.median(psf.fwhm)), 2))
    k, w = np.concatenate(out).astype(np.float32), np.concatenate(widths)
    HOME.mkdir(parents=True, exist_ok=True)
    np.save(HOME / "star_shapes.npy", k)
    np.save(HOME / "star_widths.npy", w)
    log(len(k), "shapes; width", np.percentile(w, [1, 25, 50, 75, 99]).round(2))


def _grid():
    y, x = np.mgrid[-RK:RK + 1, -RK:RK + 1].astype(np.float32)
    return y, x


def moffat(fwhm: float, beta: float, ratio: float, angle: float) -> np.ndarray:
    yy, xx = _grid()
    c, s = math.cos(angle), math.sin(angle)
    u, v = xx * c + yy * s, (-xx * s + yy * c) / ratio
    k = (1 + (u * u + v * v) / (fwhm / (2 * math.sqrt(2 ** (1 / beta) - 1))) ** 2) ** -beta
    return (k / k.sum()).astype(np.float32)


def _along(k: np.ndarray, length: float, angle: float, decay: float) -> np.ndarray:
    """``k`` smeared along a direction: uniform (decay 0, a tracking error) or fading (a comatic tail)."""
    n = 2 * RK + 1
    line = np.zeros((n, n), np.float32)
    c, s = math.cos(angle), math.sin(angle)
    steps = max(2, int(length * 4))
    for i in range(steps):
        d = length * i / (steps - 1)
        x, y = RK + d * c, RK + d * s
        j, l = int(y), int(x)
        if 0 <= j < n - 1 and 0 <= l < n - 1:
            line[j, l] += math.exp(-decay * d / max(length, 1e-6))
    out = cv2.filter2D(k, -1, line / line.sum(), borderType=cv2.BORDER_CONSTANT)
    return (out / out.sum()).astype(np.float32)


def drawn_shape(r) -> np.ndarray:
    """A star shape no stack gave: sharp to wide, comatic, astigmatic, defocused, trailed."""
    fwhm = float(10 ** r.uniform(math.log10(1.2), math.log10(9.0)))
    k = moffat(fwhm, float(r.uniform(2.5, 5.0)), float(r.uniform(0.75, 1.0)), float(r.uniform(0, math.pi)))
    kind = r.random()
    if kind < 0.2:                                     # coma
        return _along(k, float(r.uniform(1.5, 7.0)), float(r.uniform(0, 2 * math.pi)), 2.5)
    if kind < 0.35:                                    # astigmatism
        return moffat(fwhm, float(r.uniform(2.5, 5.0)), float(r.uniform(0.4, 0.75)), float(r.uniform(0, math.pi)))
    if kind < 0.5:                                     # tracking
        return _along(k, float(r.uniform(1.5, 8.0)), float(r.uniform(0, 2 * math.pi)), 0.0)
    if kind < 0.65:                                    # defocus: a ring
        yy, xx = _grid()
        rad = float(r.uniform(2.5, 10.0))
        hole = float(r.uniform(0.2, 0.45))
        d = np.hypot(yy, xx)
        ring = ((d < rad) & (d > hole * rad)).astype(np.float32)
        out = cv2.GaussianBlur(ring, (0, 0), float(r.uniform(0.6, 1.5)))
        return (out / out.sum()).astype(np.float32)
    return k


class Data:
    """Batches of (input, target), in noise units of the stack."""

    def __init__(self, dev, seed=0):
        self.dev = dev
        self.rng = np.random.default_rng(seed)
        self.truth = []
        for f in sorted(TRUTH.glob("*.npy")):
            a = np.load(f)
            a = a - np.median(a[a != 0]) if (a != 0).any() else a
            self.truth.append([a, cv2.resize(a, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                               cv2.resize(a, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)])
        self.stacks = []
        for f in sorted(TRUTH_STACKS.glob("*.npy")):
            a = np.load(f)
            self.stacks.append(a - np.median(a))
        m = np.load(HOME / "star_shapes.npy")
        pad = RK - R
        self.kernels = np.pad(m, ((0, 0), (pad, pad), (pad, pad)))
        self.den = dn.pretrained(dev)
        self.den.eval()
        self.noise = [(np.load(d / "A.npy", mmap_mode="r"), np.load(d / "B.npy", mmap_mode="r")) for d in full_pairs()]
        log(len(self.truth), "Euclid truths,", len(self.stacks), "stack planes,", len(self.kernels), "star shapes,", len(self.noise), "noise sources")
        yy, xx = _grid()
        s = TARGET_FWHM / 2.3548
        t = np.exp(-(xx * xx + yy * yy) / (2 * s * s))
        self.target_k = torch.from_numpy((t / t.sum()).astype(np.float32))[None, None].to(dev)

    def crop_truth(self) -> np.ndarray:
        r = self.rng
        for _ in range(50):
            if self.stacks and r.random() < STACK_TRUTH:
                a = self.stacks[r.integers(len(self.stacks))]
                if min(a.shape) <= TILE:
                    continue
            else:
                a = self.truth[r.integers(len(self.truth))][r.choice(3, p=(0.5, 0.3, 0.2))]
            y, x = r.integers(0, a.shape[0] - TILE), r.integers(0, a.shape[1] - TILE)
            c = a[y:y + TILE, x:x + TILE]
            if (c != 0).mean() > 0.98:
                return np.rot90(c, r.integers(4)).copy() if r.random() < 0.5 else np.fliplr(c).copy()
        return c

    def crop_noise(self) -> np.ndarray:
        r = self.rng
        a, b = self.noise[r.integers(len(self.noise))]
        c = int(r.integers(3))
        y, x = r.integers(0, a.shape[1] - TILE), r.integers(0, a.shape[2] - TILE)
        d = (np.asarray(a[c, y:y + TILE, x:x + TILE], np.float32) - np.asarray(b[c, y:y + TILE, x:x + TILE], np.float32)) / 2
        return np.nan_to_num(np.clip(d, -20, 20))

    def kernel(self) -> np.ndarray:
        r = self.rng
        if r.random() > DRAWN:
            return self.kernels[r.integers(len(self.kernels))]
        return drawn_shape(r)

    def batch(self):
        r, dev = self.rng, self.dev
        xs, ks, ns = [], [], []
        for _ in range(BATCH):
            x = self.crop_truth().astype(np.float32)
            x = x / max(float(np.quantile(x, 0.999)), 1e-6)                   # the truth: its highlights near 1
            x = np.clip(x, -0.05, None) * float(10 ** r.uniform(0.3, 2.7))    # how far it stands above the noise
            n = int(10 ** r.uniform(0.3, 2.3))                                # stars, at a density of the sky's
            for _s in range(n):
                y0, x0, f = r.uniform(2, TILE - 4), r.uniform(2, TILE - 4), float(10 ** r.uniform(0.7, 3.6))
                j, i = int(y0), int(x0)
                dy, dx = y0 - j, x0 - i
                x[j, i] += f * (1 - dy) * (1 - dx)
                x[j + 1, i] += f * dy * (1 - dx)
                x[j, i + 1] += f * (1 - dy) * dx
                x[j + 1, i + 1] += f * dy * dx
            xs.append(x)
            ks.append(self.kernel())
            ns.append(self.crop_noise())
        x = torch.from_numpy(np.stack(xs))[:, None].to(dev)
        k = torch.from_numpy(np.stack(ks))[:, None].to(dev)
        n = torch.from_numpy(np.stack(ns))[:, None].to(dev)
        pad = (RK, RK, RK, RK)
        y = F.conv2d(F.pad(x, pad, mode="reflect").reshape(1, BATCH, TILE + 2 * RK, TILE + 2 * RK), k, groups=BATCH).reshape(BATCH, 1, TILE, TILE)
        t = F.conv2d(F.pad(x, pad, mode="reflect"), self.target_k)
        y = y + n
        if r.random() < 0.3:                                                  # saturated stars
            top = float(10 ** r.uniform(2.0, 3.6))
            y, t = y.clamp(max=top), t.clamp(max=top)
        return self.denoised(y), t

    @torch.no_grad()
    def denoised(self, y: torch.Tensor) -> torch.Tensor:
        """The noisy image through our Denoise (one colour plane given as the three it expects), in noise units."""
        three = y.repeat(1, 3, 1, 1)
        nm = torch.ones_like(y) * 2 ** -0.5
        bk, sg = torch.zeros(3, device=self.dev), torch.ones(3, device=self.dev)
        total = 0.0
        for dy, dx in dn.PASSES:                                        # the block's four passes (its grid averaged out)
            pad = (dx, (-(TILE + dx)) % 8, dy, (-(TILE + dy)) % 8)
            x, m = F.pad(three, pad, mode="reflect"), F.pad(nm, pad, mode="reflect")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.den(dn.to_input(x, m, bk, sg)).float()
            total = total + (x + out)[:, :, dy:dy + TILE, dx:dx + TILE]
        den = (total / len(dn.PASSES)).mean(1, keepdim=True).float()
        # the network moves the sky a little: the block takes that back (denoise.keep_level), so does this
        shift = (den - y).reshape(y.shape[0], -1).median(1).values[:, None, None, None]
        return den - shift


def one_plane(net: NAFNet) -> NAFNet:
    """The network on one colour plane: its first and last layers made from the three colours' weights."""
    w = net.intro.out_channels
    intro, ending = nn.Conv2d(1, w, 3, padding=1), nn.Conv2d(w, 1, 3, padding=1)
    with torch.no_grad():
        intro.weight.copy_(net.intro.weight.mean(1, keepdim=True) * 3)
        intro.bias.copy_(net.intro.bias)
        ending.weight.copy_(net.ending.weight.mean(0, keepdim=True))
        ending.bias.copy_(net.ending.bias.mean(0, keepdim=True))
    net.intro, net.ending = intro.to(net.intro.weight.device), ending.to(net.ending.weight.device)
    return net


def build(dev) -> NAFNet:
    star = network(dev).float()
    net = NAFNet(recompute=False).to(dev)
    net.load_state_dict(star.state_dict())
    return one_plane(net).to(dev)


def train(tag: str, hours: float, steps: int | None, resume: str | None = None) -> None:
    """``resume``: a snapshot of the run to go on from (its step; the optimiser's moments start again, the learning
    rate follows its schedule from there)."""
    dev = torch.device("cuda")
    run = HOME / tag
    run.mkdir(parents=True, exist_ok=True)
    data = Data(dev)
    net = build(dev)
    first = 1
    if resume:
        ck = torch.load(resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"])
        first = int(ck["step"]) + 1
    net.train()
    t0 = time.time()
    y, t = data.batch()
    log("batch ready", y.shape, round(float(y.max()), 1), round(float(t.max()), 1), round(time.time() - t0, 2), "s")
    per = 0.67
    total = steps or int(hours * 3600 / per)
    opt = torch.optim.AdamW(net.parameters(), lr=4e-4, weight_decay=1e-4, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total, 1e-6)
    for _ in range(first - 1):                         # a run resumed: the schedule where it was
        sched.step()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss_avg, hist = None, []
    if first > 1 and (run / "history.json").exists():  # the curve goes on
        hist = [h for h in json.loads((run / "history.json").read_text(encoding="utf-8")) if h[0] < first]
    t0 = time.time()
    if first > 1:
        log(f"resumed from {resume} at step {first}, lr {sched.get_last_lr()[0]:.2e}")
    for it in range(first, total + 1):
        y, t = data.batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(fwd(y))
            loss = F.l1_loss(out.float(), fwd(t))
            # the light of a neighbourhood stays what it is, however it is shared among its pixels
            loss = loss + FLUX_WEIGHT * F.l1_loss(fwd(F.avg_pool2d(inv(out.float()), 16)), fwd(F.avg_pool2d(t, 16)))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        scaler.step(opt)
        scaler.update()
        sched.step()
        v = float(loss.detach())
        loss_avg = v if loss_avg is None else 0.99 * loss_avg + 0.01 * v
        if it % 200 == 0:
            el = time.time() - t0
            log(f"step {it}/{total} loss {loss_avg:.5f} lr {sched.get_last_lr()[0]:.2e} {el / (it - first + 1):.3f} s/step")
            hist.append((it, loss_avg))
        if it % 2000 == 0 or it == total:
            torch.save({"net": net.state_dict(), "step": it}, run / f"net_{it:06d}.pt")
            (run / "history.json").write_text(json.dumps(hist), encoding="utf-8")
    log("finished", round((time.time() - t0) / 3600, 2), "h")


def sheet() -> None:
    """A picture of what the network will see: rows of input (blurred, noisy, denoised) and target, stretched alike."""
    dev = torch.device("cuda")
    data = Data(dev, seed=5)
    rows = []
    for _ in range(3):
        y, t = data.batch()
        y, t = y.cpu().numpy()[:, 0], t.cpu().numpy()[:, 0]
        cells = []
        for a, b in zip(y[:8], t[:8]):
            hi = max(float(np.quantile(b, 0.999)), 1.0)
            view = lambda c: np.clip(np.arcsinh(c / hi * 40) / np.arcsinh(40), 0, 1)
            cells.append(np.vstack([view(a), np.ones((4, TILE)), view(b)]))
            cells.append(np.ones((2 * TILE + 4, 6)))
        rows.append(np.hstack(cells[:-1]))
        rows.append(np.ones((10, rows[0].shape[1])))
    img = np.vstack(rows[:-1])
    out = "F:/astro_essais/A_REGARDER/deconv_exemples_entrainement.png"
    cv2.imwrite(out, (cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA) * 255).astype(np.uint8))
    log(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("library", "train", "sheet"))
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--resume", help="a snapshot to go on from")
    a = ap.parse_args()
    if a.what == "library":
        library()
    elif a.what == "sheet":
        sheet()
    else:
        train(a.tag, a.hours, a.steps, a.resume)


if __name__ == "__main__":
    main()
