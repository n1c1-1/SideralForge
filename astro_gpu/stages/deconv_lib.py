"""Deconvolution of the finished stack, zone by zone, with the star shape measured on the stack itself.

The star shape of a stack changes across the field (wider in the corners) far more than from one frame to the next,
so it is measured once, on the stack: per tile of the field and per colour, on the isolated unsaturated stars, each
star centred to a fraction of a pixel in its own colour and the tiles' medians shrunk towards a smooth model of the
field. Each zone is restored with its own shape and the zones are blended between their centres (Nagy and O'Leary 1998).

The method is Richardson-Lucy (Richardson 1972, Lucy 1974) with the handling of saturated pixels of Whyte, Sivic and
Zisserman (IJCV 2014): the latent image is split at each iteration into the bright pixels (S) and the rest (U); the
measured pixels the bright ones reach are left out of U's update, which keeps their errors from ringing into the
image, and S is updated with a smooth saturation in the forward model. A total variation term holds the noise back
(Dey et al. 2006). The same operation on the stack and on the two half-stacks: the later stages read the difference
of the half-stacks as the noise.

Against the rings around the other stars and the noise, four mechanisms (issue #92): the damped iteration of White (1994,
Eqs. 7, 9, 10): pixels already within a few noise units of the data are no longer pushed; a stop when the estimate
hardly changes any more, the mean of |change| / |estimate|, as the free software of the field does it; local deringing,
at each iteration, and global deringing, after them. The last two follow the principle published by the commercial tools
(a star support limiting the growth of rings at each iteration; the pixels of rings partly brought back afterwards):
no equation of theirs is published, this is our reading of it. The result is kept where the signal is strong (a
protection mask from the image itself).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class StackDeconvSettings:
    iterations: int = 30          # rl: its iterations
    regularization: float = 0.002  # rl: strength of the total variation term
    # the settings frozen on the tuning targets of issue #92 and judged on 33 others
    damping: float = 3.0          # White 1994: noise units below which a pixel is no longer pushed (0: none)
    stop: float = 0.003           # stops when the mean relative change of an iteration falls under this (0: never)
    local_deringing: float = 0.9  # 0..1: how much of a darkening is refused in the star support, at each iteration
    global_dark: float = 0.0      # 0..1: how much of what went below the measured image is given back, afterwards
    protect_low: float = 3.0      # noise units over the sky where the result starts to count (0: everywhere)
    protect_high: float = 15.0    # noise units over the sky where it counts in full
    tiles_long: int = 6           # tiles along the long side (the short side gets as many as keep them square)
    radius: int = 12              # half-size of the star stamps and kernels (stack pixels)
    min_stars: int = 20           # a tile with fewer stars leans on the smooth model of the field
    snr_min: float = 30.0         # peak signal-to-noise of the stars the shape is measured on
    isolation: float = 0.03       # no neighbour brighter than this fraction of the star within 1.5 radius
    log: object = field(default=print, repr=False)
    tick: object = field(default=None, repr=False)   # called with the share done, after each colour


def tile_grid(H: int, W: int, tiles_long: int) -> tuple[int, int]:
    """(rows, columns) of roughly square tiles with ``tiles_long`` tiles along the long side."""
    n = max(1, int(tiles_long))
    if W >= H:
        return max(1, int(round(n * H / W))), n
    return n, max(1, int(round(n * W / H)))


def _tent(n: int, size: int, dev) -> torch.Tensor:
    """(n, size) bilinear weights of n tile centres along one axis (each pixel's weights sum to 1)."""
    g = torch.arange(size, device=dev, dtype=torch.float32) + 0.5
    c = (torch.arange(n, device=dev, dtype=torch.float32) + 0.5) * size / n
    w = torch.zeros(n, size, device=dev)
    for i in range(n):
        wi = torch.ones(size, device=dev)
        if i > 0:
            wi = torch.where(g < c[i], ((g - c[i - 1]) / (c[i] - c[i - 1])).clamp(0, 1), wi)
        if i < n - 1:
            wi = torch.where(g >= c[i], ((c[i + 1] - g) / (c[i + 1] - c[i])).clamp(0, 1), wi)
        w[i] = wi
    return w


def _blur(a: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (..., H, W) tensor, reflected borders."""
    r = max(1, int(math.ceil(3 * sigma)))
    o = torch.arange(-r, r + 1, device=a.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (o / sigma) ** 2)
    k = k / k.sum()
    shp = a.shape
    x = a.reshape(-1, 1, shp[-2], shp[-1])
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x.reshape(shp)


def _robust_sigma(x: torch.Tensor) -> float:
    v = x[torch.isfinite(x)]
    if v.numel() > 2_000_000:
        v = v[torch.randperm(v.numel(), device=v.device)[:2_000_000]]
    m = v.median()
    return float(1.4826 * (v - m).abs().median())


def _sky(img: torch.Tensor, valid: torch.Tensor, b: int = 64) -> torch.Tensor:
    """Smooth sky of one plane: medians of b x b blocks (valid pixels), bilinearly interpolated."""
    H, W = img.shape
    nby, nbx = max(1, H // b), max(1, W // b)
    a = torch.where(valid, img, torch.full_like(img, float("nan")))[: nby * b, : nbx * b]
    blk = a.reshape(nby, b, nbx, b).permute(0, 2, 1, 3).reshape(nby, nbx, b * b)
    m = torch.nanmedian(blk, dim=-1).values
    m = torch.where(torch.isfinite(m), m, torch.nanmedian(m))
    return F.interpolate(m[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]


def _dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    """Dilation of a 2-D boolean mask by a disc of radius r."""
    if r <= 0:
        return m
    o = torch.arange(-r, r + 1, device=m.device, dtype=torch.float32)
    disk = ((o[:, None] ** 2 + o[None, :] ** 2) <= r * r + 0.5).float()[None, None]
    return F.conv2d(m.float()[None, None], disk, padding=r)[0, 0] > 0.5


def _fill_invalid(img: torch.Tensor, valid: torch.Tensor, sigma: float = 8.0) -> torch.Tensor:
    """Pixels without data take the blurred value of their valid neighbours (the filter must not see a cliff)."""
    if bool(valid.all()):
        return img
    v = valid.float()
    num = _blur(img * v, sigma)
    den = _blur(v, sigma)
    filled = num / den.clamp(min=1e-6)
    bad = den < 1e-3
    if bool(bad.any()):                                          # farther from the data: a wider pass, then the median
        num2, den2 = _blur(img * v, 4 * sigma), _blur(v, 4 * sigma)
        filled = torch.where(bad, num2 / den2.clamp(min=1e-6), filled)
        bad = bad & (den2 < 1e-3)
        filled = torch.where(bad, img[valid].median(), filled)   # beyond the reach of any filter: anything smooth
    return torch.where(valid, img, filled)


def kernel_fwhm(K: torch.Tensor, up: int = 8) -> torch.Tensor:
    """Width at half maximum (pixels) of each kernel (..., k, k) on the pixel grid: the area above half the peak of
    the kernel resampled 'up' times finer (bicubic), as the diameter of the disc of that area."""
    k = K.shape[-1]
    Kf = K.reshape(-1, 1, k, k).float()
    Ku = F.interpolate(Kf, scale_factor=up, mode="bicubic", align_corners=False)
    mx = Ku.amax((-1, -2), keepdim=True)
    area = (Ku > 0.5 * mx).float().sum((-1, -2)) / up ** 2
    return (2.0 * torch.sqrt(area / math.pi)).reshape(K.shape[:-2])


@dataclass
class StackPSF:
    kernels: torch.Tensor         # (C, ty, tx, k, k), normalised, on the stack grid
    fwhm: np.ndarray              # (C, ty, tx) pixels
    nstars: np.ndarray            # (ty, tx)
    stars: int


class StackDeconvolver:
    """Measures the star shape of a stack per tile and colour and deconvolves the stack (and its half-stacks) towards
    a narrower Gaussian with one linear filter per tile, blended between tile centres."""

    def __init__(self, settings: StackDeconvSettings | None = None, device="cpu"):
        self.s = settings or StackDeconvSettings()
        self.dev = torch.device(device)

    # ------------------------------------------------------------------ star shape
    def measure(self, full: torch.Tensor, A: torch.Tensor, B: torch.Tensor, sat: torch.Tensor,
                valid: torch.Tensor) -> StackPSF | None:
        """Per tile and colour, the median of the isolated unsaturated stars, each centred in its own colour (bicubic
        shift) and normalised to its flux; tiles with few stars shrunk towards a quadratic model of the field."""
        s = self.s
        dev = full.device
        C, H, W = full.shape
        R = s.radius
        k = 2 * R + 1
        nty, ntx = tile_grid(H, W, s.tiles_long)
        g = full[1] if C > 1 else full[0]
        sky = _sky(g, valid)
        noise = _blur(0.5 * (A[1 if C > 1 else 0] - B[1 if C > 1 else 0]), 1.0)
        sig = _robust_sigma(noise[valid])
        gs = _blur(g - sky, 1.0)
        mx = F.max_pool2d(gs[None, None], 7, 1, 3)[0, 0]
        ok_area = valid & ~_dilate(sat, R)
        ok_area = ok_area & (F.max_pool2d((~valid).float()[None, None], 2 * R + 1, 1, R)[0, 0] < 0.5)
        peaks_lo = (gs == mx) & (gs > 5 * sig)
        cand = peaks_lo & (gs > s.snr_min * sig) & ok_area
        cand[:R + 2] = False
        cand[-R - 3:] = False
        cand[:, :R + 2] = False
        cand[:, -R - 3:] = False
        cy, cx = torch.nonzero(cand, as_tuple=True)
        if cy.numel() < 10:
            s.log(f"stack deconvolution: only {cy.numel()} stars to measure the star shape on - left as it is")
            return None
        # isolation: no neighbour brighter than a few % of the star within 1.5 radius
        from scipy.spatial import cKDTree
        ly, lx = torch.nonzero(peaks_lo, as_tuple=True)
        lv = gs[ly, lx].cpu().numpy()
        lxy = np.c_[lx.cpu().numpy(), ly.cpu().numpy()]
        tree = cKDTree(lxy)
        pv = gs[cy, cx].cpu().numpy()
        pts = np.c_[cx.cpu().numpy(), cy.cpu().numpy()]
        keep = np.ones(len(pts), bool)
        for i, nb in enumerate(tree.query_ball_point(pts, 1.5 * R)):
            for j in nb:
                if lv[j] > s.isolation * pv[i] and (lxy[j] != pts[i]).any():
                    keep[i] = False
                    break
        idx = torch.from_numpy(np.where(keep)[0]).to(dev)
        cy, cx = cy[idx], cx[idx]
        n = int(cy.numel())
        if n < 10:
            s.log(f"stack deconvolution: only {n} isolated stars - left as it is")
            return None
        o = torch.arange(-R, R + 1, device=dev)
        oy, ox = torch.meshgrid(o, o, indexing="ij")
        r = torch.sqrt((oy ** 2 + ox ** 2).float())
        ring = r >= R - 2
        st = full[:, cy[:, None, None] + oy[None], cx[:, None, None] + ox[None]].permute(1, 0, 2, 3).float()  # (n,C,k,k)
        st = st - st[:, :, ring].median(-1).values[:, :, None, None]
        # centre of each star in each colour: Gaussian-weighted centroid, iterated
        dx = torch.zeros(n, C, device=dev)
        dy = torch.zeros(n, C, device=dev)
        pos = st.clamp(min=0)
        for _ in range(6):
            w = torch.exp(-((ox[None, None] - dx[..., None, None]) ** 2 + (oy[None, None] - dy[..., None, None]) ** 2)
                          / (2 * 1.5 ** 2)) * pos
            tot = w.sum((-1, -2)).clamp(min=1e-12)
            dx = ((w * ox).sum((-1, -2)) / tot).clamp(-1.5, 1.5)
            dy = ((w * oy).sum((-1, -2)) / tot).clamp(-1.5, 1.5)
        good = (dx.abs() < 1.0).all(1) & (dy.abs() < 1.0).all(1)
        # centred stamps: bicubic resampling at (offset + centre)
        gx = (ox[None, None].float() + dx[..., None, None] + R + 0.5) / k * 2 - 1
        gy = (oy[None, None].float() + dy[..., None, None] + R + 0.5) / k * 2 - 1
        grid = torch.stack([gx, gy], -1).reshape(n * C, k, k, 2)
        cs = F.grid_sample(st.reshape(n * C, 1, k, k), grid, mode="bicubic", padding_mode="border",
                           align_corners=False).reshape(n, C, k, k)
        flux = (cs * (r <= R - 3)).sum((-1, -2))
        good = good & (flux > 0).all(1)
        cs = cs / flux.clamp(min=1e-12)[..., None, None]
        cs, cy, cx = cs[good], cy[good], cx[good]
        n = int(cs.shape[0])
        ty = (cy * nty // H).clamp(max=nty - 1)
        tx = (cx * ntx // W).clamp(max=ntx - 1)
        E = torch.full((C, nty, ntx, k, k), float("nan"), device=dev)
        cnt = np.zeros((nty, ntx), int)
        for a in range(nty):
            for b in range(ntx):
                m = (ty == a) & (tx == b)
                cnt[a, b] = int(m.sum())
                if cnt[a, b] >= 3:
                    E[:, a, b] = cs[m].median(0).values
        E = self._field_model(E, cnt)
        E = self._radial_wings(E)
        E = E.clamp(min=0) * (r <= R - 0.5).float()
        E = E / E.sum((-1, -2), keepdim=True).clamp(min=1e-12)
        fw = kernel_fwhm(E).cpu().numpy()
        s.log(f"stack deconvolution: star shape measured on {n} isolated stars, {nty} x {ntx} tiles "
              f"({cnt.min()}-{cnt.max()} per tile); width " +
              ", ".join(f"{'RGB'[c] if C == 3 else c} {np.median(fw[c]):.2f} px ({fw[c].min():.2f}-{fw[c].max():.2f})"
                        for c in range(C)))
        return StackPSF(E, fw, cnt, n)

    def _field_model(self, E, cnt):
        """Each tile's shape shrunk towards a smooth (quadratic) model of the field, weights = its star count and
        ``min_stars``; tiles without stars take the model."""
        s = self.s
        dev = E.device
        C, nty, ntx, k, _ = E.shape
        nn = torch.tensor(cnt, device=dev, dtype=torch.float32).reshape(-1)
        u = (torch.arange(ntx, device=dev, dtype=torch.float32) + 0.5) / ntx * 2 - 1
        v = (torch.arange(nty, device=dev, dtype=torch.float32) + 0.5) / nty * 2 - 1
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        terms = [torch.ones_like(uu)]
        if nty * ntx >= 3:
            terms += [uu, vv]
        if nty * ntx >= 9 and nty >= 2 and ntx >= 2:
            terms += [uu * uu, uu * vv, vv * vv]
        T = torch.stack(terms, -1).reshape(-1, len(terms))
        Ef = E.reshape(C, nty * ntx, k * k)
        fin = torch.isfinite(Ef[:, :, :1])
        wgt = nn[None, :, None] * fin
        Ez = torch.nan_to_num(Ef, nan=0.0)
        if int(fin[0].sum()) >= T.shape[1] + 2:
            Am = torch.einsum("tb,tc,kt->kbc", T, T, wgt[..., 0]) + 1e-6 * torch.eye(T.shape[1], device=dev)
            bv = torch.einsum("tb,ktg->kgb", T, wgt * Ez)
            coef = torch.linalg.solve(Am[:, None], bv[..., None])[..., 0]          # (C, k*k, nb)
            model = torch.einsum("tb,kgb->ktg", T, coef)
        else:
            model = ((wgt * Ez).sum(1, keepdim=True) / wgt.sum(1, keepdim=True).clamp(min=1e-9)).expand_as(Ez)
        w = nn[None, :, None]
        out = torch.where(fin, (w * Ez + s.min_stars * model) / (w + s.min_stars), model)
        return out.reshape(C, nty, ntx, k, k)

    def _radial_wings(self, E, r_core: float = 4.0, r_wing: float = 6.0):
        """Beyond the core, the azimuthal median of each kernel: the wings of one tile are measured with little
        signal, and their noise would weigh like a real halo."""
        C, nty, ntx, k, _ = E.shape
        R = k // 2
        dev = E.device
        o = torch.arange(-R, R + 1, device=dev, dtype=torch.float32)
        oy, ox = torch.meshgrid(o, o, indexing="ij")
        r = torch.sqrt(oy ** 2 + ox ** 2)
        ring = torch.round(r).long().reshape(-1)
        Ef = E.reshape(-1, k * k)
        prof = torch.zeros(Ef.shape[0], int(ring.max()) + 1, device=dev)
        for j in range(prof.shape[1]):
            m = ring == j
            if bool(m.any()):
                prof[:, j] = Ef[:, m].median(1).values
        wing = prof[:, ring].reshape(E.shape)
        t = ((r - r_core) / (r_wing - r_core)).clamp(0, 1)
        return (1 - t) * E + t * wing

    def restore(self, images: list[torch.Tensor], psf: StackPSF, sat: torch.Tensor, valid: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor) -> list[torch.Tensor]:
        """Each image of ``images`` (C, H, W) restored per zone and colour with the zone's star shape, the zones blended
        between their centres; the clipped cores and the pixels without data left as they were. The sky is raised for
        the iterations so that it stands above zero (the method needs positive values), then taken back."""
        s = self.s
        dev = images[0].device
        C, H, W = images[0].shape
        nty, ntx = psf.fwhm.shape[1:]
        P = 4 * s.radius
        wy, wx = _tent(nty, H, dev), _tent(ntx, W, dev)
        core = _dilate(sat, 1).float()
        keep = torch.maximum((_blur(core, 1.0) * 2).clamp(0, 1), (~valid).float())
        outs = [torch.zeros_like(im, dtype=torch.float32) for im in images]
        for c in range(C):
            batch = torch.stack([_fill_invalid(im[c].float(), valid) for im in images])
            sig = _robust_sigma((0.5 * (A[c] - B[c]))[valid].float())
            shift = max(0.0, SKY_SIGMAS * sig - float(batch[0][valid].median()))
            level = saturation_level(batch[0], sat) + shift
            sky = float(batch[0][valid].median())
            counts = (sky + shift) / max(sig * sig, 1e-30)              # an equivalent of counts: its noise is the stack's
            light = batch[0] - _blur(batch[0], 12.0)
            support = (_dilate(light > 8 * sig, s.radius // 2).float() if s.local_deringing > 0 else None)
            support = None if support is None else _blur(support, 2.0)[None].repeat(len(images), 1, 1)
            for a in range(nty):
                for b in range(ntx):
                    rows, cols = torch.nonzero(wy[a] > 0)[:, 0], torch.nonzero(wx[b] > 0)[:, 0]
                    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
                    d = F.pad(batch[None, :, r0:r1, c0:c1] + shift, (P, P, P, P), mode="reflect")[0]
                    sup = None if support is None else F.pad(support[None, :, r0:r1, c0:c1], (P, P, P, P), mode="reflect")[0]
                    y = restore_saturated(d, psf.kernels[c, a, b], s.iterations, s.regularization, sig, level, damping=s.damping,
                                          counts=counts, stop=s.stop, support=sup, local=s.local_deringing)[:, P:-P, P:-P]
                    wt = wy[a][r0:r1, None] * wx[b][None, c0:c1]
                    for i, out in enumerate(outs):
                        v = y[i] - shift
                        if s.global_dark > 0:                       # what went dark is partly given back
                            m = batch[i, r0:r1, c0:c1]
                            v = torch.where(v < m, m - (m - v) * (1 - s.global_dark), v)
                        if s.protect_high > 0:                      # kept where the signal is strong
                            sm = _blur(batch[i, r0:r1, c0:c1] - sky, 3.0)
                            k = ((sm - s.protect_low * sig) / max((s.protect_high - s.protect_low) * sig, 1e-9)).clamp(0, 1)
                            v = batch[i, r0:r1, c0:c1] + k * (v - batch[i, r0:r1, c0:c1])
                        out[c, r0:r1, c0:c1] += wt * v
            del batch
            if s.tick is not None:
                s.tick((c + 1) / C)
        return _kept(images, outs, keep, valid)


#: Richardson-Lucy: the sky is raised at least this many noise units above zero for its iterations
SKY_SIGMAS = 8.0
#: the term of the total variation is kept from dividing by less than this (Dey et al. 2006: it must stay positive)
TV_FLOOR = 0.2
#: White 1994: how suddenly the damped likelihood becomes flat under the threshold (his N)
DAMPING_N = 10
#: Whyte et al. 2014: the smoothness of the saturation function (their a), the level above which a pixel is "bright"
SATURATION_SMOOTHNESS, BRIGHT_LEVEL = 50.0, 0.9


def saturation_level(x: torch.Tensor, sat: torch.Tensor) -> float:
    """The level at which this plane clips: the high end of its pixels marked saturated (a pixel marked in another
    colour only is low here); without any, a level nothing reaches."""
    if not bool(sat.any()):
        return 2.0 * float(x.max())
    v = x[sat].float()
    if v.numel() > 1_000_000:
        v = v[:: v.numel() // 1_000_000]
    return float(torch.quantile(v, 0.9))


def _kept(images, outs, keep, valid):
    """The results, with the saturated cores (``keep``) and the pixels without data as they were."""
    res = []
    for im, out in zip(images, outs):
        o = im.float() + (1 - keep)[None] * (out - im.float())
        res.append(torch.where(valid[None], o, im.float()))
    return res


def restore_saturated(d: torch.Tensor, K: torch.Tensor, iterations: int, reg: float, sig: float, level: float, *,
                      damping: float = 0.0, counts: float = 1.0, stop: float = 0.0, support: torch.Tensor | None = None,
                      local: float = 0.0) -> torch.Tensor:
    """A batch (n, H, W) of positive images restored by Richardson-Lucy with saturated pixels: Eqs. 18, 20, 21 and 23-25
    of Whyte, Sivic and Zisserman (IJCV 2014), in units of the clipping level, with the total variation of Dey et al.
    (2006) on the part below it. ``sig``: the noise, the scale below which a gradient is flat. ``damping``: White's T, in
    noise units, with ``counts``, what makes a value here a number of counts of the same noise."""
    k = K.shape[-1]
    R = k // 2
    Hp, Wp = d.shape[-2:]
    pad = torch.zeros(Hp, Wp, device=d.device)
    pad[:k, :k] = K
    Kh = torch.fft.rfft2(torch.roll(pad, (-R, -R), (0, 1)))

    def conv(a, adj=False):
        return torch.fft.irfft2(torch.fft.rfft2(a) * (Kh.conj() if adj else Kh), s=(Hp, Wp))

    a_ = SATURATION_SMOOTHNESS
    level = max(level, 1e-12)
    low = 1e-3 * max(sig / level, 1e-9)
    g = (d / level).clamp(min=0)
    f = g.clamp(min=low)
    for _ in range(iterations):
        u = (f <= BRIGHT_LEVEL).float()
        v = 1 - F.max_pool2d(F.pad((1 - u)[None], (R, R, R, R), mode="replicate"), k, 1)[0]    # not reached by S
        f_u, f_s = u * f, f - u * f
        model = conv(f_u).clamp(min=low)
        gain = g / model
        if damping > 0:                                                    # White 1994, Eqs. 7, 9 and 10
            D, I = (g * level * counts).clamp(min=1e-6), (model * level * counts).clamp(min=1e-6)
            U = (-2.0 / damping ** 2 * (D * torch.log(I / D) - I + D)).clamp(0, 1)
            gain = 1 + U ** (DAMPING_N - 1) * (DAMPING_N - (DAMPING_N - 1) * U) * (gain - 1)
        ratio_u = conv(gain * v + (1 - v), True).clamp(min=0)
        x = conv(f).clamp(min=low)
        slope = torch.sigmoid(-a_ * (x - 1))
        clipped = (x - F.softplus(a_ * (x - 1)) / a_).clamp(min=low)
        ratio_s = conv(g * slope / clipped + (1 - slope), True).clamp(min=0)
        div = 1.0
        if reg > 0:
            gx = torch.diff(f_u, dim=-1, append=f_u[..., -1:])
            gy = torch.diff(f_u, dim=-2, append=f_u[..., -1:, :])
            norm = torch.sqrt(gx ** 2 + gy ** 2 + (0.1 * sig / level) ** 2)
            gx, gy = gx / norm, gy / norm
            curv = torch.diff(gx, dim=-1, prepend=gx[..., :1] * 0) + torch.diff(gy, dim=-2, prepend=gy[..., :1, :] * 0)
            div = (1 - reg * curv).clamp(min=TV_FLOOR)
        before = f
        f = (f_u * ratio_u / div + f_s * ratio_s).clamp(min=low)
        if support is not None and local > 0:                              # the rings' growth refused around the stars
            f = torch.where(f < before, before + (f - before) * (1 - support * local), f)
        if stop > 0 and float(((f - before).abs() / before.abs().clamp(min=low)).mean()) < stop:
            break
    return f * level


def run(full: np.ndarray, A: np.ndarray, B: np.ndarray, sat: np.ndarray | None, valid: np.ndarray,
        settings: StackDeconvSettings, device="cpu"):
    """Measure on the full stack, deconvolve the full stack and both half-stacks. Arrays (C, H, W) float32; sat and
    valid (H, W) booleans (sat None: the brightest pixels of each colour, as the optics stage defines them).
    Returns (full, A, B) as numpy arrays (unchanged when the star shape cannot be measured) and an info dict."""
    dev = torch.device(device)
    tf = torch.from_numpy(np.ascontiguousarray(full, np.float32)).to(dev)
    tA = torch.from_numpy(np.ascontiguousarray(A, np.float32)).to(dev)
    tB = torch.from_numpy(np.ascontiguousarray(B, np.float32)).to(dev)
    tv = torch.from_numpy(np.ascontiguousarray(valid)).to(dev)
    if sat is None:
        sat = np.zeros(full.shape[1:], bool)
        for c in range(full.shape[0]):
            sat |= full[c] > 0.7 * np.percentile(full[c][valid], 99.999)
    ts = torch.from_numpy(np.ascontiguousarray(sat)).to(dev)
    eng = StackDeconvolver(settings, dev)
    psf = eng.measure(tf, tA, tB, ts, tv)
    if psf is None:
        return full, A, B, {"applied": False}
    of, oa, ob = eng.restore([tf, tA, tB], psf, ts, tv, tA, tB)
    after = eng.measure(of, oa, ob, ts, tv)
    info = {"applied": True, "stars": psf.stars, "nstars": psf.nstars, "fwhm": psf.fwhm, "kernels": psf.kernels.cpu().numpy(), "fwhm_after": None if after is None else after.fwhm}
    return of.cpu().numpy(), oa.cpu().numpy(), ob.cpu().numpy(), info
