"""Deconvolution of the finished stack, zone by zone, with the star shape measured on the stack itself.

The star shape of a stack changes across the field (wider in the corners) far more than from one frame to the next,
so it is measured once, on the stack: per tile of the field and per colour, on the isolated unsaturated stars, each
star centred to a fraction of a pixel in its own colour and the tiles' medians shrunk towards a smooth model of the
field. One linear filter per tile and colour turns the measured shape into a narrower Gaussian ("target"); the
filtered images are blended between the tile centres.

Being linear, the operation is the same for faint and bright stars, keeps every flux, and is exactly the same on the
full stack and on the two half-stacks - the later stages read the difference of the half-stacks as the noise and
their agreement as the signal, which stays true. Its gain is bounded at every spatial frequency, and frequencies where
the stack holds no signal above its noise (measured on the half-stacks, per tile) are not amplified. Saturated cores
are left as they were measured: the filter fades out around them.

The other method (``method`` "rl") is Richardson-Lucy (Richardson 1972, Lucy 1974) with the same star shapes and the
same blending between zones, regularised by total variation (Dey et al. 2006). It is not linear: it keeps the values
positive and follows the noise of each pixel, which a linear filter cannot do. It works on the luminance, and its gain is
applied to the three colours: each colour on its own converges at its own speed, which colours the cores of the stars.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class StackDeconvSettings:
    method: str = "rl"            # "rl": Richardson-Lucy; "zones": one linear filter per zone
    iterations: int = 30          # rl: its iterations
    regularization: float = 0.002  # rl: strength of the total variation term
    tiles_long: int = 6           # tiles along the long side (the short side gets as many as keep them square)
    radius: int = 12              # half-size of the star stamps and kernels (stack pixels)
    shrink: float = 0.06          # every star made narrower by this fraction of its measured width
    even: bool = False            # also bring every tile down to the width of the sharpest part of the field
    even_pct: float = 10.0        # percentile of the tile widths (per colour) that 'even' aims at
    target_shape: str = "measured"  # 'measured': the zone's own star shape made narrower; 'gaussian': a round Gaussian
    max_gain: float = 4.0         # largest amplification at any spatial frequency
    damping: float = 1.0          # frequencies whose signal is below damping^2 x the noise power are not amplified
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


def scaled_kernel(K: torch.Tensor, f: float) -> torch.Tensor:
    """The shape of K made f times as wide (f < 1: narrower), same flux, on the same pixel grid."""
    k = K.shape[-1]
    R = k // 2
    dev = K.device
    o = torch.arange(-R, R + 1, device=dev, dtype=torch.float32)
    oy, ox = torch.meshgrid(o, o, indexing="ij")
    gx = (ox / f + R + 0.5) / k * 2 - 1
    gy = (oy / f + R + 0.5) / k * 2 - 1
    grid = torch.stack([gx, gy], -1)[None]
    T = F.grid_sample(K.reshape(1, 1, k, k).float(), grid, mode="bicubic", padding_mode="zeros",
                      align_corners=False)[0, 0].clamp(min=0)
    return T / T.sum().clamp(min=1e-12)


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

    # ------------------------------------------------------------------ targets and filters
    def targets(self, psf: StackPSF) -> np.ndarray:
        """Target width per tile and colour: (1 - shrink) x the measured width, and with 'even' never wider than
        (1 - shrink) x the width of the sharpest part of the field (percentile ``even_pct`` of the tiles)."""
        s = self.s
        fw = psf.fwhm
        cap = np.full(fw.shape[0], np.inf)
        if s.even:
            cap = np.array([np.percentile(fw[c], s.even_pct) for c in range(fw.shape[0])])
        return (1.0 - s.shrink) * np.minimum(fw, cap[:, None, None])

    @staticmethod
    def _freq(Hp, Wp, dev):
        fy = torch.fft.fftfreq(Hp, device=dev)[:, None]
        fx = torch.fft.rfftfreq(Wp, device=dev)[None, :]
        return fy, fx

    def _filter(self, K, t, fw, Hp, Wp, damp_w=None):
        """Frequency response (Hp, Wp//2+1) of the filter of one tile and colour: T K* / (|K|^2 + |T|^2 / g^2),
        g = 2 x max_gain (so that its gain never exceeds max_gain), normalised to keep the flux; with damping, the
        amplification is only applied in proportion to the signal share w of each frequency: 1 + w (F - 1)."""
        dev = K.device
        fy, fx = self._freq(Hp, Wp, dev)
        if not np.isfinite(t) or t >= 0.995 * fw:
            return None                                       # nothing to do here: the identity
        k = K.shape[-1]
        R = k // 2

        def spectrum(ker):
            pad = torch.zeros(Hp, Wp, device=dev)
            pad[:k, :k] = ker
            return torch.fft.rfft2(torch.roll(pad, (-R, -R), (0, 1)))

        Kh = spectrum(K)
        if self.s.target_shape == "gaussian":
            sig_t = t / 2.3548
            Th = torch.exp(-2 * math.pi ** 2 * sig_t ** 2 * (fy ** 2 + fx ** 2))
        else:                                               # the zone's own shape, narrower
            Th = spectrum(scaled_kernel(K, t / fw))
        g = 2.0 * self.s.max_gain
        Fh = Th * Kh.conj() / (Kh.abs() ** 2 + (Th.abs() / g) ** 2)
        Fh = Fh / Fh[0, 0].real
        if damp_w is not None:
            Fh = 1 + damp_w * (Fh - 1)
        # a compact filter: its response beyond the star's own reach is cut (cosine taper from 'radius' to 1.5 x
        # 'radius'). Left long, the filter carried the noise of the measured shape and the sharp edges of the gain
        # bound as faint ripples around every bright star, out to 100 px and more
        h = torch.fft.irfft2(Fh, s=(Hp, Wp))
        yy = torch.minimum(torch.arange(Hp, device=dev), Hp - torch.arange(Hp, device=dev)).float()[:, None]
        xx = torch.minimum(torch.arange(Wp, device=dev), Wp - torch.arange(Wp, device=dev)).float()[None, :]
        rr = torch.sqrt(yy ** 2 + xx ** 2)
        r1, r2 = float(self.s.radius), 1.5 * float(self.s.radius)
        win = torch.where(rr <= r1, torch.ones_like(rr),
                          0.5 * (1 + torch.cos(math.pi * ((rr - r1) / (r2 - r1)).clamp(0, 1))))
        h = h * win
        return torch.fft.rfft2(h / h.sum())

    def _damping_weights(self, A, B, y0, y1, x0, x1, Hp, Wp):
        """Share of the signal at each spatial frequency in a tile, from the two half-stacks: S / (S + d^2 N) with
        S = their cross power (what they share) and N the power of their half difference (the noise of the full
        stack), azimuthally averaged, on the frequency grid of the padded image."""
        s = self.s
        if s.damping <= 0:
            return None
        dev = A.device
        a = A[y0:y1, x0:x1].float()
        b = B[y0:y1, x0:x1].float()
        h, w = a.shape
        win = torch.hann_window(h, periodic=False, device=dev)[:, None] * torch.hann_window(w, periodic=False,
                                                                                         device=dev)[None]
        a = (a - a.mean()) * win
        b = (b - b.mean()) * win
        Ah, Bh = torch.fft.rfft2(a), torch.fft.rfft2(b)
        cross = (Ah * Bh.conj()).real
        nse = ((Ah - Bh).abs() ** 2) / 4
        fy = torch.fft.fftfreq(h, device=dev)[:, None]
        fx = torch.fft.rfftfreq(w, device=dev)[None, :]
        fr = torch.sqrt(fy ** 2 + fx ** 2)
        nb = 48
        bi = (fr / 0.7072 * nb).long().clamp(max=nb - 1).reshape(-1)
        Sb = torch.zeros(nb, device=dev).index_add_(0, bi, cross.reshape(-1))
        Nb = torch.zeros(nb, device=dev).index_add_(0, bi, nse.reshape(-1))
        wb = (Sb.clamp(min=0) / (Sb.clamp(min=0) + s.damping ** 2 * Nb).clamp(min=1e-30)).clamp(0, 1)
        fyp, fxp = self._freq(Hp, Wp, dev)
        frp = torch.sqrt(fyp ** 2 + fxp ** 2) / 0.7072 * nb - 0.5
        i0 = frp.floor().long().clamp(0, nb - 1)
        i1 = (i0 + 1).clamp(max=nb - 1)
        t = (frp - i0.float()).clamp(0, 1)
        return (1 - t) * wb[i0] + t * wb[i1]

    # ------------------------------------------------------------------ deconvolution
    def deconvolve(self, images: list[torch.Tensor], psf: StackPSF, sat: torch.Tensor, valid: torch.Tensor,
                   A: torch.Tensor, B: torch.Tensor):
        """The same linear operation on each image of ``images`` (C, H, W): per tile and colour the filter of its
        star shape, the results blended between the tile centres; saturated cores (and the area around them) and
        pixels without data are left as they were. A, B: the half-stacks, for the damping weights. Returns the images
        and per tile and colour the target width and the noise gain (white noise)."""
        s = self.s
        dev = images[0].device
        C, H, W = images[0].shape
        nty, ntx = psf.fwhm.shape[1:]
        R = s.radius
        P = 4 * R
        Hp, Wp = H + 2 * P, W + 2 * P
        tgt = self.targets(psf)
        wy, wx = _tent(nty, H, dev), _tent(ntx, W, dev)
        keep = torch.maximum(_kept_zone(sat, R), (~valid).float())    # 1 at saturated stars, fading out
        outs = [torch.zeros_like(im, dtype=torch.float32) for im in images]
        gain = np.ones((C, nty, ntx))
        ys = [int(round(a * H / nty)) for a in range(nty + 1)]
        xs = [int(round(b * W / ntx)) for b in range(ntx + 1)]
        for c in range(C):
            spec = []
            for im in images:
                x = _fill_invalid(im[c].float(), valid)
                x = F.pad(x[None, None], (P, P, P, P), mode="reflect")[0, 0]
                spec.append(torch.fft.rfft2(x))
            for a in range(nty):
                for b in range(ntx):
                    dw = self._damping_weights(A[c], B[c], ys[a], ys[a + 1], xs[b], xs[b + 1], Hp, Wp)
                    Fh = self._filter(psf.kernels[c, a, b], float(tgt[c, a, b]), float(psf.fwhm[c, a, b]), Hp, Wp, dw)
                    wt = wy[a][:, None] * wx[b][None, :]
                    rows = torch.nonzero(wy[a] > 0)[:, 0]
                    cols = torch.nonzero(wx[b] > 0)[:, 0]
                    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
                    if Fh is None:
                        for im, out in zip(images, outs):
                            out[c, r0:r1, c0:c1] += wt[r0:r1, c0:c1] * im[c, r0:r1, c0:c1].float()
                        continue
                    full_w = torch.cat([Fh.abs() ** 2, Fh[:, 1:-1].abs() ** 2], 1)   # both halves of the spectrum
                    gain[c, a, b] = float(full_w.mean().sqrt())
                    for sp, out in zip(spec, outs):
                        y = torch.fft.irfft2(sp * Fh, s=(Hp, Wp))[P:P + H, P:P + W]
                        out[c, r0:r1, c0:c1] += wt[r0:r1, c0:c1] * y[r0:r1, c0:c1]
                    del Fh
            del spec
            if s.tick is not None:
                s.tick((c + 1) / C)
        return _kept(images, outs, keep, valid), tgt, gain

    def restore(self, images: list[torch.Tensor], psf: StackPSF, sat: torch.Tensor, valid: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor) -> list[torch.Tensor]:
        """Richardson-Lucy on each image of ``images`` (C, H, W), per zone and colour with the zone's star shape, the
        zones blended between their centres; saturated cores and pixels without data left as they were. The sky is
        raised for the iterations so that it stands above zero (the method needs positive values), then taken back."""
        s = self.s
        dev = images[0].device
        C, H, W = images[0].shape
        nty, ntx = psf.fwhm.shape[1:]
        R = s.radius
        P = 4 * R
        wy, wx = _tent(nty, H, dev), _tent(ntx, W, dev)
        keep = torch.maximum(_kept_zone(sat, R), (~valid).float())
        outs = [torch.zeros_like(im, dtype=torch.float32) for im in images]
        filled = [torch.stack([_fill_invalid(im[c].float(), valid) for c in range(C)]) for im in images]  # (C, H, W)
        sigs = [_robust_sigma((0.5 * (A[c] - B[c]))[valid].float()) for c in range(C)]
        shift = torch.tensor([max(0.0, SKY_SIGMAS * sigs[c] - float(filled[0][c][valid].median())) for c in range(C)],
                             device=dev, dtype=torch.float32)[:, None, None]
        sig = float(np.sqrt(sum(v * v for v in sigs))) / C               # the noise of the luminance
        done, total = 0, nty * ntx
        for a in range(nty):
            for b in range(ntx):
                rows, cols = torch.nonzero(wy[a] > 0)[:, 0], torch.nonzero(wx[b] > 0)[:, 0]
                r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
                raised = [f[:, r0:r1, c0:c1] + shift for f in filled]
                lum = torch.stack([x.mean(0) for x in raised])            # (n, h, w)
                d = F.pad(lum[None], (P, P, P, P), mode="reflect")[0]
                K = psf.kernels[:, a, b].mean(0)
                y = richardson_lucy(d, K / K.sum(), s.iterations, s.regularization, sig)[:, P:-P, P:-P]
                gain = (y / lum.clamp(min=1e-3 * max(sig, 1e-12))).clamp(0.05, 20.0)
                wt = wy[a][r0:r1, None] * wx[b][None, c0:c1]
                for i, out in enumerate(outs):
                    out[:, r0:r1, c0:c1] += wt[None] * (raised[i] * gain[i] - shift)
                done += 1
                if s.tick is not None and done % max(1, total // 8) == 0:
                    s.tick(done / total)
        return _kept(images, outs, keep, valid)


#: Richardson-Lucy: the sky is raised at least this many noise units above zero for its iterations
SKY_SIGMAS = 8.0
#: the term of the total variation is kept from dividing by less than this (Dey et al. 2006: it must stay positive)
TV_FLOOR = 0.2


def _kept_zone(sat: torch.Tensor, R: int) -> torch.Tensor:
    """Where the image stays as measured (0..1): the saturated stars and, around them, the reach of the star shape
    (1.5 R, the filters' own): a clipped core rings as far as the filter reaches. Its edge fades over R / 2 pixels."""
    zone = _dilate(sat, int(1.5 * R)).float()
    return (_blur(zone, R / 3.0) * 2).clamp(0, 1)


def _kept(images, outs, keep, valid):
    """The results, with the saturated cores (``keep``) and the pixels without data as they were."""
    res = []
    for im, out in zip(images, outs):
        o = im.float() + (1 - keep)[None] * (out - im.float())
        res.append(torch.where(valid[None], o, im.float()))
    return res


def richardson_lucy(d: torch.Tensor, K: torch.Tensor, iterations: int, reg: float, sig: float) -> torch.Tensor:
    """Richardson-Lucy of a batch (n, H, W) of positive images by the kernel K, regularised by total variation: each
    step divides by 1 - reg div(grad u / |grad u|), and stays above the local sky. ``sig``: the noise, the scale below
    which a gradient is flat."""
    k = K.shape[-1]
    R = k // 2
    Hp, Wp = d.shape[-2:]
    pad = torch.zeros(Hp, Wp, device=d.device)
    pad[:k, :k] = K
    Kh = torch.fft.rfft2(torch.roll(pad, (-R, -R), (0, 1)))
    low = 1e-3 * max(sig, 1e-12)
    d = d.clamp(min=low)
    u = d.clone()
    # no dark ring: nothing goes below the sky around it, a morphological opening of the smoothed image (the stars
    # taken off, a nebula's level kept: its darkest value alone would leave a star a dark moat on a bright nebula)
    low_ = -F.max_pool2d(F.pad(-_blur(d, 1.5)[None], (R, R, R, R), mode="reflect"), k, 1)
    floor = F.max_pool2d(F.pad(low_, (R, R, R, R), mode="reflect"), k, 1)[0] - sig
    for _ in range(iterations):
        est = torch.fft.irfft2(torch.fft.rfft2(u) * Kh, s=(Hp, Wp)).clamp(min=low)
        ratio = torch.fft.irfft2(torch.fft.rfft2(d / est) * Kh.conj(), s=(Hp, Wp)).clamp(min=0)
        div = 1.0
        if reg > 0:
            gx = torch.diff(u, dim=-1, append=u[..., -1:])
            gy = torch.diff(u, dim=-2, append=u[..., -1:, :])
            norm = torch.sqrt(gx ** 2 + gy ** 2 + (0.1 * sig) ** 2)
            gx, gy = gx / norm, gy / norm
            curv = torch.diff(gx, dim=-1, prepend=gx[..., :1] * 0) + torch.diff(gy, dim=-2, prepend=gy[..., :1, :] * 0)
            div = (1 - reg * curv).clamp(min=TV_FLOOR)
        new = u * ratio / div
        kept = torch.maximum(new, floor).clamp(min=low)
        # the light the floor added comes back off what stands above it: the flux stays what the iteration gave
        above = (kept - floor).sum((-2, -1), keepdim=True).clamp(min=low)
        u = floor + (kept - floor) * ((new - floor).sum((-2, -1), keepdim=True) / above).clamp(0.5, 1.0)
    return u


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
    if settings.method == "rl":
        (of, oa, ob), tgt, gain = eng.restore([tf, tA, tB], psf, ts, tv, tA, tB), None, None
    else:
        (of, oa, ob), tgt, gain = eng.deconvolve([tf, tA, tB], psf, ts, tv, tA, tB)
    after = eng.measure(of, oa, ob, ts, tv)
    info = {"applied": True, "stars": psf.stars, "nstars": psf.nstars, "fwhm": psf.fwhm, "target": tgt,
            "gain": gain, "kernels": psf.kernels.cpu().numpy(), "fwhm_after": None if after is None else after.fwhm}
    return of.cpu().numpy(), oa.cpu().numpy(), ob.cpu().numpy(), info
