"""Non-blind sharpening with the star shape measured on the stack, learned on the data themselves.

The star shape (PSF) is measured on isolated Gaia stars of the stack, everywhere in the field and per colour. A
sharpening target is chosen from it (a Gaussian the measured shape can reach), and the Noise2Noise network is
fine-tuned so that its output, blurred back from the target to the measured shape, predicts the other half-stack.
The network therefore learns to undo the blur with the only supervision available: the data, with their real noise.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

R, OS, DEG = 12, 4, 2             # PSF stamp radius (px), oversampling, degree of the field polynomial
NG = 2 * R * OS + 1
NB = (DEG + 1) * (DEG + 2) // 2


class PSFField:
    """Empirical star shape of a stack, oversampled, every cell a degree-2 polynomial of the field position."""

    def __init__(self, full, gx, gy, gg, gsat, log=print):
        C, H, W = full.shape
        self.C, self.H, self.W = C, H, W
        tree = cKDTree(np.c_[gx, gy])
        sel = np.where((gg > gsat + 0.3) & (gg < gsat + 3.5) & (gx > R + 2) & (gx < W - R - 3) & (gy > R + 2)
                       & (gy < H - R - 3))[0]
        iso = np.array([i for i in sel
                        if all(j == i or gg[j] > gg[i] + 3 for j in tree.query_ball_point([gx[i], gy[i]], 2.2 * R))])
        oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
        ring = np.hypot(ox, oy) >= R - 2
        xi, yi = np.round(gx[iso]).astype(int), np.round(gy[iso]).astype(int)
        cy_ = np.clip(np.round((oy[None] + (yi - gy[iso])[:, None, None] + R) * OS).astype(int), 0, NG - 1)
        cx_ = np.clip(np.round((ox[None] + (xi - gx[iso])[:, None, None] + R) * OS).astype(int), 0, NG - 1)
        Bs = self.basis(gx[iso], gy[iso])
        ci = cy_ * NG + cx_
        self.field = np.zeros((C, NG, NG, NB), np.float32)
        for c in range(C):
            st = np.stack([np.asarray(full[c, b - R:b + R + 1, a - R:a + R + 1], np.float32) for a, b in zip(xi, yi)])
            st -= np.median(st[:, ring], axis=1)[:, None, None]
            flux = st[:, np.hypot(ox, oy) < 6].sum(1)
            for _ in range(3):
                ok = flux > 0
                w = flux[ok]
                Aacc = np.zeros((NB, NB, NG * NG))
                bacc = np.zeros((NB, NG * NG))
                for k in range(NB):
                    bk = np.broadcast_to((w * Bs[ok, k])[:, None, None], ci[ok].shape)
                    bacc[k] = np.bincount(ci[ok].ravel(), (bk * st[ok]).ravel(), minlength=NG * NG)
                    for l_ in range(k, NB):
                        wkl = np.broadcast_to((w * w * Bs[ok, k] * Bs[ok, l_])[:, None, None], ci[ok].shape)
                        Aacc[k, l_] = Aacc[l_, k] = np.bincount(ci[ok].ravel(), wkl.ravel(), minlength=NG * NG)
                Am = np.moveaxis(Aacc, -1, 0) + 1e-6 * np.eye(NB)[None] * np.maximum(Aacc[0, 0][:, None, None], 1e-12)
                co = np.linalg.solve(Am, np.moveaxis(bacc, -1, 0)[..., None])[..., 0].reshape(NG, NG, NB)
                co = np.ascontiguousarray(co, np.float32)
                co = np.stack([gaussian_filter(co[:, :, k], 0.7) for k in range(NB)], -1)
                P_at = np.einsum("syxk,sk->syx", co[cy_, cx_], Bs)
                flux = (st * P_at).sum((1, 2)) / np.maximum((P_at * P_at).sum((1, 2)), 1e-12)
            self.field[c] = co
        self.n_stars = len(iso)
        log(f"star shape measured on {len(iso)} isolated stars")

    def basis(self, x, y):
        a, b = np.asarray(x) / (self.W / 2) - 1.0, np.asarray(y) / (self.H / 2) - 1.0
        return np.stack([a ** i * b ** j for i in range(DEG + 1) for j in range(DEG + 1 - i)], -1)

    def at(self, c, x, y):
        """Star shape of colour c at (x, y), on the stack pixel grid ((2R+1)^2, centred, normalised)."""
        p = np.clip(np.einsum("yxk,k->yx", self.field[c], self.basis(x, y)), 0, None)
        k = cv2.resize(p, (2 * R + 1, 2 * R + 1), interpolation=cv2.INTER_AREA)
        return (k / k.sum()).astype(np.float32)

    def fwhm(self, c, x, y):
        """Width at half maximum (px) from the half-maximum area of the oversampled shape."""
        p = np.einsum("yxk,k->yx", self.field[c], self.basis(x, y))
        return float(2 * np.sqrt((p > p.max() / 2).sum() / np.pi) / OS)


def gaussian_kernel(fwhm, size=2 * R + 1):
    yy, xx = np.mgrid[:size, :size] - size // 2
    g = np.exp(-0.5 * (xx * xx + yy * yy) / (fwhm / 2.355) ** 2)
    return (g / g.sum()).astype(np.float32)


def target_fwhm(psf: PSFField):
    """Sharpening target, from the measured shape: two thirds of the core width at the field centre (median over
    colours), never below 2 px (the sampling limit): a target the measured shape can reach."""
    w = np.median([psf.fwhm(c, psf.W / 2, psf.H / 2) for c in range(psf.C)])
    return float(max(2.0, round(2.0 / 3.0 * w, 1)))


def target_to_stack_kernels(psf: PSFField, tf, nx=8, ny=6, n=64):
    """Per field cell and colour, the kernel K with T * K = P (T: target Gaussian, P: measured shape): an image at
    the target shape, convolved by K, is at the measured shape. Returns the kernels and the worst reconstruction
    error of P (fraction), a check that the target is reachable."""
    Tk = gaussian_kernel(tf)
    Tf = np.fft.fft2(np.pad(Tk, (0, n - 2 * R - 1)))
    Ks = np.zeros((ny, nx, psf.C, 2 * R + 1, 2 * R + 1), np.float32)
    worst = 0.0
    for j in range(ny):
        for i in range(nx):
            for c in range(psf.C):
                Pk = psf.at(c, (i + 0.5) * psf.W / nx, (j + 0.5) * psf.H / ny)
                Pf = np.fft.fft2(np.pad(Pk, (0, n - 2 * R - 1)))
                Kf = np.conj(Tf) * Pf / (np.abs(Tf) ** 2 + 1e-4 * np.abs(Tf).max() ** 2)
                k = np.roll(np.real(np.fft.ifft2(Kf)), (R, R), (0, 1))[:2 * R + 1, :2 * R + 1]
                k /= k.sum()
                chk = cv2.filter2D(np.pad(Tk, R).astype(np.float32), -1, k[::-1, ::-1].astype(np.float32),
                                   borderType=cv2.BORDER_CONSTANT)[R:-R, R:-R]
                worst = max(worst, float(np.abs(chk - Pk).sum() / Pk.sum()))
                Ks[j, i, c] = k
    return Ks, worst


def finetune_sharp(net, A, B, nm, bk, sg, Ks, dev, steps=4000, patch=192, batch=12, lr=3e-4, seed=0, progress=None):
    """Fine-tune a trained Noise2Noise network (astro_gpu.stages.denoise.UNet) into a sharpening network: its output
    is an image at the target shape; the loss compares it, convolved by the local target->measured kernel, with the
    other half-stack. Same sampling, augmentation and weighting as the plain training (the kernel turns and flips
    with the patch)."""
    from ..stages.denoise import to_input
    H, W = A.shape[1:]
    tA, tB = torch.from_numpy(A).to(dev), torch.from_numpy(B).to(dev)
    tN = torch.from_numpy(nm).to(dev)[None]
    bk_t, sg_t = torch.as_tensor(np.asarray(bk), device=dev), torch.as_tensor(np.asarray(sg), device=dev)
    valid = (tN[0] > 0) & (tN[0] < 1.6)
    tK = torch.from_numpy(Ks).to(dev)
    ny, nx = Ks.shape[:2]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    torch.manual_seed(seed)
    net.train()
    run = 0.0
    for it in range(steps):
        ys = torch.randint(0, H - patch, (batch * 2,)).tolist()
        xs = torch.randint(0, W - patch, (batch * 2,)).tolist()
        xin, tgt, nmp, ker = [], [], [], []
        for y, x in zip(ys, xs):
            if not bool(valid[y + patch // 2, x + patch // 2]):
                continue
            src, dst = (tA, tB) if torch.rand(1).item() < 0.5 else (tB, tA)
            ts = [src[:, y:y + patch, x:x + patch], dst[:, y:y + patch, x:x + patch], tN[:, y:y + patch, x:x + patch],
                  tK[min((y + patch // 2) * ny // H, ny - 1), min((x + patch // 2) * nx // W, nx - 1)]]
            k = int(torch.randint(0, 4, (1,)))
            ts = [torch.rot90(t, k, (1, 2)) for t in ts]
            if bool(torch.rand(1) < 0.5):
                ts = [torch.flip(t, (2,)) for t in ts]
            xin.append(ts[0])
            tgt.append(ts[1])
            nmp.append(ts[2])
            ker.append(ts[3])
            if len(xin) == batch:
                break
        if not xin:
            continue
        xin, tgt, nmp = torch.stack(xin), torch.stack(tgt), torch.stack(nmp)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = net(to_input(xin, nmp, bk_t, sg_t)).float()
        a_in = (xin - bk_t.view(1, 3, 1, 1)) / sg_t.view(1, 3, 1, 1)
        pred = a_in + out
        nb = pred.shape[0]
        pk = Fn.pad(pred.reshape(1, nb * 3, patch, patch), (R, R, R, R), mode="reflect")
        wk = torch.flip(torch.stack(ker).reshape(nb * 3, 1, 2 * R + 1, 2 * R + 1), (2, 3))
        pred = Fn.conv2d(pk, wk, groups=nb * 3).reshape(nb, 3, patch, patch)
        b_t = (tgt - bk_t.view(1, 3, 1, 1)) / sg_t.view(1, 3, 1, 1)
        with torch.no_grad():
            sm = Fn.avg_pool2d(a_in, 5, 1, 2).clamp(min=0)
            wgt = 1.0 / ((nmp ** 2).clamp(min=0.25) + sm / 40.0)
        sl = np.s_[:, :, R:patch - R, R:patch - R]
        loss = (wgt[sl] * (pred[sl] - b_t[sl]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        if progress and (it + 1) % 200 == 0:
            progress((it + 1) / steps, f"sharpening step {it + 1}/{steps}, loss {run:.4f}")
    net.eval()
    return net, run
