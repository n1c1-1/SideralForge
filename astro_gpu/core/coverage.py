"""The drizzle coverage map, and the rule that chooses the drizzle scale, the drop size and the method. Entry
points: ``suggest_drizzle``, ``advise_method``, ``map_variation``, ``weight_map``.

The coverage map is the weight map of a Bayer drizzle without the pixel values: for every frame, every raw sample of
a colour is dropped where the frame's registration polynomial puts it on the output grid, as a square drop of
``pixfrac`` x ``scale`` output pixels, and each output pixel adds up the area of the drops that touch it (the frame
weights, if given). It is what stack.py's ``deposit`` accumulates in ``den``, computed before stacking.

The rule (DrizzlePac Handbook v3, §7.3 "Inspecting Drizzled Products after User Reprocessing", p. 145-146 [L2]):
"statistics performed on the drizzled weight image in the region of interest should yield an RMS value (standard
deviation) that is less than 20% of the median value", and "final_pixfrac should be slightly larger than the final
output scale" (DrizzlePac's output scale is our 1 / scale). Hence, per colour, the variation = std / median of the
weight map, and for red and blue (a quarter of the samples each) it must stay under 20 %:

* scale 1.5 or 2 is suggested only if the stars are undersampled (Moffat FWHM under 2 px, the Nyquist sampling:
  drizzle restores detail only then, Fruchter & Hook 2002 [L1], Siril's drizzle page [S5]) AND the rule holds at that
  scale; of the scales that hold, the smallest that samples the stars at 2 px or more, else the largest;
* the drop size is then the smallest value of 0.05 steps from 1 / scale up that keeps the rule;
* at scale 1, a drop of 1 (Siril, WBPP and APP for colour sensors [S5][P15][A8]);
* the method advice (``advise_method``), on the frames the selection keeps:

  ===========================================================  ===========================  ========================
  series                                                       advice                       source
  ===========================================================  ===========================  ========================
  colour sensor, fewer than 15 frames                          Classic                      PixInsight [P12]: drizzle
                                                                                            from 15-20 dithered frames
  colour, red / blue coverage varies by 20 % or more at 1×     Classic                      DrizzlePac 20 % rule [L2]
  colour, enough well-dithered frames                          drizzle (scale: the rule)    PixInsight [P12], [L2]
  colour, stars under 2 px and the coverage holds at 1.5×/2×   drizzle 1.5× or 2×           Siril [S5], [L1][L2]
  monochrome                                                   Classic; drizzle only with   the same rules (no CFA:
                                                               stars under 2 px, 15 frames  drizzle only adds detail
                                                               and the coverage holding     when undersampled)
                                                               at 1.5× or 2×
  ===========================================================  ===========================  ========================

  The advice is only shown: it never changes the method by itself (stages/stack.py exposes it as an advice ``Auto``).

A monochrome sensor (``cfa`` empty, None or "MONO") goes through the same code: every pixel is a sample of its one
plane, deposited and read as the middle channel ("G").

Fast estimate: the map is computed only on a grid of small patches of the output (``GRID`` x ``PATCH`` px) inside the
field, but there from every sample of every frame, each at its position through its frame's polynomial (linearised
over the patch: 0.002 px at most for real lens distortions); a frame that does not cover a whole patch is left out
of that patch, so the edges of the dithering do not count as variation, only the sampling pattern does. The variation
is the RMS over the patches of their std / median. ``suggest_drizzle`` takes about 0.5 s for 100 frames, 1.5 s for 300
(one core); the estimate is within a few points of the variation of the weight map of the stack actually made.
``weight_map`` is the exact full map (small images, tests);
after a stack, ``map_variation`` measures the same thing on its real ``drz_weight.npy``.

Callers
-------
* stages/stack.py (``drizzle_suggestion``, from ``Stack.auto``) calls ``suggest_drizzle`` on register's polynomials
  of the frames the selection keeps, with calibrate's frame size and CFA pattern and the median Moffat FWHM, and
  takes ``scale``, ``pixfrac`` and the method advice from it.

  ``coef`` is register's (n, 2, 10) polynomial of each frame (frame sensor px -> reference sensor px, the 10 cubic
  terms of register.poly_terms); the output grid is stack.py's: Xo = (X + 0.5) x scale - 0.5. Pass ``weights`` (the
  frame weights) once they are known, for the weighted map. ``s.coverage[(scale, pixfrac)]`` keeps every
  ``Coverage`` computed (per colour: ``cv``, ``cv_max``, the per-patch ``cv_map`` for Inspect).
* The reasons are templates written here, listed in ``REASON_TEMPLATES`` (texts.reason_sites() reads them there);
  their French is in client/locale/fr/engine.json.
* Inspect > Stacking: ``map_variation(np.load(drz_weight.npy))`` gives the measured variation per colour of the
  stack actually made, to show next to the estimate.
Pure numpy; no torch.

Sources:
[L1] Fruchter & Hook 2002, PASP 114, 144 (drizzle): https://arxiv.org/abs/astro-ph/9808087
[L2] DrizzlePac Handbook v3: https://hst-docs.stsci.edu/drizzpac
[S5] Siril, drizzle: https://siril.readthedocs.io/en/latest/preprocessing/drizzle.html
[P12] PixInsight Debayer documentation: https://gitlab.com/pixinsight/Reference-Documentation/-/raw/master/tools/Debayer/Debayer.pidoc
[P15] PixInsight WBPP source (BPP-Global.js, BPP-FrameGroup.js): https://gitlab.com/pixinsight/PJSR/-/tree/master/src/scripts/BatchPreprocessing
[A8] Astro Pixel Processor forum, Bayer drizzle: https://www.astropixelprocessor.com/community/tutorials-workflows/bayer-drizzle/
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import Auto

CHANNELS = "RGB"
#: DrizzlePac's rule: the weight map's standard deviation under 20 % of its median
LIMIT = 0.20
#: undersampled below this FWHM (pixels): the Nyquist sampling of the star profile
UNDERSAMPLED = 2.0
#: the scales the rule may suggest, besides 1
SCALES = (1.5, 2.0)
#: patches of the fast estimate: a GRID (rows, columns) of PATCH x PATCH output pixels, over the central 80 % of the field
GRID = (4, 4)
PATCH = 32
#: a patch covered by fewer than this share of the frames is not used
MIN_PATCH_FRAMES = 0.5
#: fewer kept frames than this: Classic advised (PixInsight: drizzle pays from 15 to 20 well-dithered frames [P12])
MIN_DRIZZLE_FRAMES = 15
#: the values of ``cfa`` that mean a monochrome sensor
MONO = ("", "MONO", "L", "NONE")

#: the method advice's reasons (short: a tooltip)
M_FEW = "{n} frames (drizzle needs 15): Classic"
M_HOLES = "red and blue leave holes: Classic"
M_UNEVEN = "coverage varies {cov:.1f} % (over 20 %): Classic"
M_DRIZZLE = "{n} dithered frames, coverage {cov:.1f} %: drizzle"
M_DRIZZLE_FINE = "stars of {fwhm:.1f} px, coverage {cov:.1f} %: drizzle {scale:g}×"
M_MONO = "monochrome, stars of {fwhm:.1f} px: Classic"
M_MONO_NO_FWHM = "monochrome, star size not measured: Classic"
M_MONO_FEW = "monochrome, {n} frames (drizzle needs 15): Classic"
M_MONO_UNEVEN = "monochrome, coverage varies {cov:.1f} % at {scale:g}×: Classic"
M_MONO_HOLES = "monochrome, holes at {scale:g}×: Classic"
M_MONO_DRIZZLE = "monochrome, stars of {fwhm:.1f} px: drizzle {scale:g}×"

#: every template this module writes (texts.reason_sites reads them here: the method advice's are not ``Auto(...)``
#: literals); their French is in client/locale/fr/engine.json
REASON_TEMPLATES = (
    M_FEW,
    M_HOLES,
    M_UNEVEN,
    M_DRIZZLE,
    M_DRIZZLE_FINE,
    M_MONO,
    M_MONO_NO_FWHM,
    M_MONO_FEW,
    M_MONO_UNEVEN,
    M_MONO_HOLES,
    M_MONO_DRIZZLE,
    "star size not measured: 1×",
    "stars of {fwhm:.1f} px: sampled well enough, drizzle would add no detail: 1×",
    "stars of {fwhm:.1f} px: undersampled, but at {scale:g}× the coverage would vary by {cov:.0f} %: 1×",
    "stars of {fwhm:.1f} px: undersampled, but at {scale:g}× the red and blue samples would leave holes: 1×",
    "stars of {fwhm:.1f} px: undersampled; coverage stays at {cov:.0f} %: {scale:g}×",
    "scale 1×: drops of a whole pixel",
    "the finest drop that keeps the red and blue coverage under {limit:.0f} % ({cov:.0f} %)",
)


# ------------------------------------------------------------------------------------------------------ geometry
def poly_eval(coef, x, y, W: int, H: int):
    """register.py's polynomial (``apply_poly``): frame sensor (x, y) -> reference sensor (X, Y). ``coef`` (2, 10), or
    (n, 2, 10) with x, y of shape (n, ...)."""
    coef = np.asarray(coef, float)
    u = np.asarray(x, float) / (W / 2) - 1.0
    v = np.asarray(y, float) / (H / 2) - 1.0
    T = (np.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3)
    if coef.ndim == 2:
        return sum(coef[0, k] * t for k, t in enumerate(T)), sum(coef[1, k] * t for k, t in enumerate(T))
    sh = (len(coef),) + (1,) * (u.ndim - 1)
    return (sum(coef[:, 0, k].reshape(sh) * t for k, t in enumerate(T)),
            sum(coef[:, 1, k].reshape(sh) * t for k, t in enumerate(T)))


def _poly_jacobian(coef, x, y, W, H):
    """d(X, Y) / d(x, y) of each frame's polynomial at (x, y): four arrays. ``coef`` (n, 2, 10), x, y (n, k)."""
    u = x / (W / 2) - 1.0
    v = y / (H / 2) - 1.0
    z, o = np.zeros_like(u), np.ones_like(u)
    du = (z, o, z, 2 * u, v, z, 3 * u * u, 2 * u * v, v * v, z)
    dv = (z, z, o, z, u, 2 * v, z, u * u, 2 * u * v, 3 * v * v)
    c = coef[:, :, :, None]
    Xu = sum(c[:, 0, k] * t for k, t in enumerate(du)) * (2 / W)
    Xv = sum(c[:, 0, k] * t for k, t in enumerate(dv)) * (2 / H)
    Yu = sum(c[:, 1, k] * t for k, t in enumerate(du)) * (2 / W)
    Yv = sum(c[:, 1, k] * t for k, t in enumerate(dv)) * (2 / H)
    return Xu, Xv, Yu, Yv


def invert(coef, X, Y, W: int, H: int, iters: int = 8):
    """The frame sensor positions (x, y), shape (n, k), that each frame's polynomial (``coef`` (n, 2, 10)) puts at the
    reference positions (X, Y) (k,): the affine part solved first, then Newton's method on the cubic."""
    coef = np.asarray(coef, float)
    X = np.broadcast_to(np.asarray(X, float), (len(coef), np.size(X)))
    Y = np.broadcast_to(np.asarray(Y, float), (len(coef), np.size(Y)))
    a = coef[:, :, :3]                                   # X = a0 + a1 u + a2 v (normalised coordinates)
    det = a[:, 0, 1] * a[:, 1, 2] - a[:, 0, 2] * a[:, 1, 1]
    rx, ry = X - a[:, 0, 0, None], Y - a[:, 1, 0, None]
    u = (rx * a[:, 1, 2, None] - ry * a[:, 0, 2, None]) / det[:, None]
    v = (ry * a[:, 0, 1, None] - rx * a[:, 1, 1, None]) / det[:, None]
    x, y = (u + 1.0) * (W / 2), (v + 1.0) * (H / 2)
    for _ in range(iters):
        fx, fy = poly_eval(coef, x, y, W, H)
        ex, ey = fx - X, fy - Y
        Xu, Xv, Yu, Yv = _poly_jacobian(coef, x, y, W, H)
        d = Xu * Yv - Xv * Yu
        x = x - (Yv * ex - Xv * ey) / d
        y = y - (-Yu * ex + Xu * ey) / d
    return x, y


def is_mono(cfa) -> bool:
    """Whether ``cfa`` names a monochrome sensor (no Bayer pattern)."""
    return cfa is None or str(cfa).strip().upper() in MONO


def _channels_of(cfa) -> np.ndarray:
    """The channel of each 2 x 2 position k = 2 row + column: the Bayer pattern's, or the middle one for a
    monochrome sensor (its one plane)."""
    if is_mono(cfa):
        return np.ones(4, int)
    return np.array([CHANNELS.index(c) for c in str(cfa).upper()])


def rule_channels(cfa) -> str:
    """The channels the coverage rule reads: red and blue (a quarter of the samples each) on a colour sensor, the
    one plane on a monochrome one."""
    return "G" if is_mono(cfa) else "RB"


# ------------------------------------------------------------------------------------------------------ the maps
def _deposit(xo, yo, plane, w, h, n_planes: int, P_h: int, P_w: int) -> np.ndarray:
    """stack.py's ``deposit`` with a value of 1: square drops of half-size ``h`` (output px) at (xo, yo), the overlap
    area x ``w`` added to the planes ``plane`` of an (n_planes, P_h, P_w) map. Samples landing more than 2 pixels
    outside are ignored."""
    keep = (xo > -2.5) & (xo < P_w + 1.5) & (yo > -2.5) & (yo < P_h + 1.5)
    xo, yo, plane = xo[keep], yo[keep], plane[keep]
    w = w[keep] if np.ndim(w) else np.full(len(xo), float(w))
    pad = int(np.ceil(h)) + 3                            # the accumulator's border: no test per cell
    Hp, Wp = P_h + 2 * pad, P_w + 2 * pad
    jx0 = np.floor(xo - h + 0.5)
    iy0 = np.floor(yo - h + 0.5)
    ncell = int(np.ceil(2 * h)) + 1
    wxs = [np.clip(np.minimum(xo + h, jx0 + d + 0.5) - np.maximum(xo - h, jx0 + d - 0.5), 0, None) * w
           for d in range(ncell)]
    wys = [np.clip(np.minimum(yo + h, iy0 + d + 0.5) - np.maximum(yo - h, iy0 + d - 0.5), 0, None)
           for d in range(ncell)]
    base = plane.astype(np.int64) * (Hp * Wp) + (iy0.astype(np.int64) + pad) * Wp + (jx0.astype(np.int64) + pad)
    acc = np.zeros(n_planes * Hp * Wp)
    for di in range(ncell):
        for dj in range(ncell):
            acc += np.bincount(base + (di * Wp + dj), weights=wxs[dj] * wys[di], minlength=acc.size)
    return acc.reshape(n_planes, Hp, Wp)[:, pad:pad + P_h, pad:pad + P_w]


def weight_map(coef, W: int, H: int, cfa: str, scale: float, pixfrac: float, weights=None) -> np.ndarray:
    """The exact drizzle weight map (3, Ho, Wo) of all the frames (every sample of every frame, through its full
    polynomial): for small images and tests. The stack itself saves the real one (drz_weight.npy)."""
    coef = np.asarray(coef, float)
    n = len(coef)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    Ho, Wo = int(round(H * scale)), int(round(W * scale))
    out = np.zeros((3, Ho, Wo))
    ch_of_k = _channels_of(cfa)
    yy, xx = np.mgrid[0:H, 0:W]
    ch = ch_of_k[2 * (yy % 2) + (xx % 2)].ravel()
    xx, yy = xx.ravel().astype(float), yy.ravel().astype(float)
    h = pixfrac * scale / 2.0
    for i in range(n):
        if w[i] <= 0:
            continue
        X, Y = poly_eval(coef[i], xx, yy, W, H)
        out += _deposit((X + 0.5) * scale - 0.5, (Y + 0.5) * scale - 0.5, ch, w[i], h, 3, Ho, Wo)
    return out


def variation(m: np.ndarray) -> float:
    """std / median of the weights ``m`` (DrizzlePac's statistic); inf when the median is 0 (holes in half the map)."""
    m = np.asarray(m, float).ravel()
    m = m[np.isfinite(m)]
    if m.size == 0:
        return float("nan")
    med = float(np.median(m))
    return float(np.std(m) / med) if med > 0 else float("inf")


def map_variation(wmap: np.ndarray, margin: float = 0.1) -> dict[str, float]:
    """The variation (std / median) of each colour of a full weight map (3, H, W), e.g. a stack's drz_weight.npy, on
    the field without a ``margin`` share on each side (the edges of the dithering)."""
    C, H, W = wmap.shape
    y0, x0 = int(H * margin), int(W * margin)
    return {CHANNELS[c]: variation(wmap[c, y0:H - y0, x0:W - x0]) for c in range(C)}


@dataclass
class Coverage:
    """The estimated coverage of a drizzle (one scale, one drop size), per colour."""

    scale: float
    pixfrac: float
    cv: dict                    # colour -> RMS over the patches of std / median (inf: holes)
    cv_max: dict                # colour -> worst patch
    cv_map: dict = field(repr=False, default_factory=dict)   # colour -> (rows, columns) per patch, NaN: not used
    frames: np.ndarray = field(repr=False, default=None)     # frames covering each patch
    origins: np.ndarray = field(repr=False, default=None)    # (rows, columns, 2): output (x, y) of each patch's corner
    n_frames: int = 0

    @property
    def red_blue(self) -> float:
        """The variation the rule reads: the larger of red and blue; on a monochrome sensor (only "G" measured) its
        one plane."""
        vals = [self.cv[c] for c in ("R", "B") if c in self.cv] or [self.cv.get("G", np.nan)]
        return max(vals)

    def holds(self, limit: float = LIMIT) -> bool:
        return bool(self.red_blue < limit)


class _Patches:
    """The samples of every frame that land in the patches of the fast estimate, at one scale: positions in patch
    pixels, colour, weight. Computed once per scale; ``coverage(pixfrac)`` deposits them."""

    def __init__(self, coef, W, H, cfa, scale, weights, grid, patch, channels):
        coef = np.asarray(coef, float)
        n = len(coef)
        w = np.ones(n) if weights is None else np.asarray(weights, float)
        self.scale, self.grid, self.P, self.n = scale, grid, patch, int((w > 0).sum())
        self.channels = [CHANNELS.index(c) for c in channels]
        gy, gx = grid
        Ho, Wo = H * scale, W * scale
        cxs = Wo * (0.1 + 0.8 * (np.arange(gx) + 0.5) / gx)
        cys = Ho * (0.1 + 0.8 * (np.arange(gy) + 0.5) / gy)
        ch_of_k = _channels_of(cfa)
        offsets = [(c, k // 2, k % 2) for c in self.channels for k in range(4) if ch_of_k[k] == c]
        use = w > 0
        coef, w = coef[use], w[use]
        self.frames = np.zeros(grid, int)
        self.origins = np.zeros(tuple(grid) + (2,), int)
        half = (patch / 2 + 2) / scale                   # half-size of a patch and its drops, reference px
        xs_l, ys_l, ch_l, w_l, pid_l = [], [], [], [], []
        for py in range(gy):
            for px in range(gx):
                ox, oy = int(round(cxs[px] - patch / 2)), int(round(cys[py] - patch / 2))
                self.origins[py, px] = ox, oy
                Xc = (ox + (patch - 1) / 2 + 0.5) / scale - 0.5               # the patch centre, reference px
                Yc = (oy + (patch - 1) / 2 + 0.5) / scale - 0.5
                xc, yc = invert(coef, [Xc], [Yc], W, H)
                # each frame's polynomial linearised about the 2x2 cell of the CFA nearest to the patch centre: over
                # a block of a few tens of pixels a registration cubic departs from its tangent by 0.002 px at most
                x0 = 2 * np.floor(xc[:, 0] / 2)
                y0 = 2 * np.floor(yc[:, 0] / 2)
                X0, Y0 = poly_eval(coef, x0[:, None], y0[:, None], W, H)
                Xu, Xv, Yu, Yv = _poly_jacobian(coef, x0[:, None], y0[:, None], W, H)
                det = Xu * Yv - Xv * Yu
                with np.errstate(divide="ignore", invalid="ignore"):
                    sx = (np.abs(Yv) + np.abs(Xv)) / np.abs(det)
                    sy = (np.abs(Yu) + np.abs(Xu)) / np.abs(det)
                # a degenerate polynomial (no inverse there: a failed registration) covers nothing
                good = (np.isfinite(x0) & np.isfinite(y0))[:, None] & np.isfinite(sx) & np.isfinite(sy) & (det != 0)
                if not good.any():
                    continue
                rbx = int(np.ceil(np.max(sx[good]) * half)) + 2
                rby = int(np.ceil(np.max(sy[good]) * half)) + 2
                ax = 2 * np.arange(-(rbx // 2) - 1, rbx // 2 + 2)                 # the lattice of a colour, x0 + ax + kx
                ay = 2 * np.arange(-(rby // 2) - 1, rby // 2 + 2)
                with np.errstate(invalid="ignore"):
                    cov = (x0 + ax[0] >= 0) & (x0 + ax[-1] + 1 <= W - 1) & (y0 + ay[0] >= 0) & \
                        (y0 + ay[-1] + 1 <= H - 1) & good[:, 0]
                self.frames[py, px] = int(cov.sum())
                if cov.sum() < max(1, MIN_PATCH_FRAMES * len(coef)):
                    continue
                X0, Y0, Xu, Xv, Yu, Yv = (a[cov] for a in (X0, Y0, Xu, Xv, Yu, Yv))
                for c, ky, kx in offsets:                # the samples of colour c: one lattice of step 2 per CFA cell
                    jy, jx = np.meshgrid(ay + ky, ax + kx, indexing="ij")
                    dx, dy = jx.ravel()[None].astype(float), jy.ravel()[None].astype(float)
                    xo = (X0 + Xu * dx + Xv * dy + 0.5) * scale - 0.5 - ox
                    yo = (Y0 + Yu * dx + Yv * dy + 0.5) * scale - 0.5 - oy
                    keep = (xo > -2) & (xo < patch + 1) & (yo > -2) & (yo < patch + 1)
                    xs_l.append(xo[keep])
                    ys_l.append(yo[keep])
                    ch_l.append(np.full(int(keep.sum()), c))
                    w_l.append(np.broadcast_to(w[cov][:, None], keep.shape)[keep])
                    pid_l.append(np.full(int(keep.sum()), py * gx + px))

        def cat(parts, dt):
            return np.concatenate(parts).astype(dt) if parts else np.zeros(0, dt)

        self.xo, self.yo = cat(xs_l, np.float64), cat(ys_l, np.float64)
        self.ch, self.w, self.pid = cat(ch_l, np.int64), cat(w_l, np.float64), cat(pid_l, np.int64)
        self.used = np.zeros(gy * gx, bool)
        self.used[np.unique(self.pid)] = True
        self.start = np.searchsorted(self.pid, np.arange(gy * gx + 1))     # the samples of patch k: start[k]:start[k+1]

    def coverage(self, pixfrac: float) -> Coverage:
        gy, gx = self.grid
        P = self.P
        acc = np.zeros((gy * gx, 3, P * P))
        for k in np.where(self.used)[0]:                 # patch by patch: a drop never reaches the next one
            sl = slice(self.start[k], self.start[k + 1])
            acc[k] = _deposit(self.xo[sl], self.yo[sl], self.ch[sl], self.w[sl], pixfrac * self.scale / 2.0,
                              3, P, P).reshape(3, P * P)
        cv, cv_max, cv_map = {}, {}, {}
        for c in self.channels:
            per = np.full(gy * gx, np.nan)
            for k in np.where(self.used)[0]:
                per[k] = variation(acc[k, c])
            name = CHANNELS[c]
            cv_map[name] = per.reshape(gy, gx)
            fin = per[self.used]
            if fin.size == 0:
                cv[name] = cv_max[name] = float("nan")
            else:
                cv[name] = float(np.sqrt(np.mean(fin ** 2)))
                cv_max[name] = float(np.max(fin))
        return Coverage(self.scale, pixfrac, cv, cv_max, cv_map, self.frames, self.origins, self.n)


def coverage(coef, W: int, H: int, cfa: str, scale: float, pixfrac: float, *, weights=None, grid=GRID, patch=PATCH,
             channels: str = "RGB") -> Coverage:
    """The estimated variation of the drizzle weight map of each colour (std / median, the RMS over GRID patches of
    PATCH x PATCH output pixels inside the field), for the frames' polynomials ``coef`` (n, 2, 10) on a sensor W x H of
    Bayer pattern ``cfa``, at ``scale`` with drops of ``pixfrac``."""
    return _Patches(coef, W, H, cfa, scale, weights, grid, patch, channels).coverage(pixfrac)


# ------------------------------------------------------------------------------------------------------ the rule
@dataclass
class DrizzleSuggestion:
    scale: Auto
    pixfrac: Auto
    method: Auto                # "drizzle" or "classic"
    fwhm: float
    coverage: dict = field(repr=False, default_factory=dict)    # (scale, pixfrac) -> Coverage, every one computed


def pixfrac_steps(scale: float) -> np.ndarray:
    """The drop sizes tried at a scale: from 1 / scale (rounded up to 0.05) to 1, by 0.05."""
    lo = np.ceil(round(20.0 / scale, 6)) / 20.0
    return np.round(np.arange(lo, 1.0 + 1e-9, 0.05), 2)


def suggest_drizzle(coef, W: int, H: int, cfa, fwhm: float | None, *, weights=None, limit: float = LIMIT,
                    undersampled: float = UNDERSAMPLED, scales=SCALES, grid=GRID, patch=PATCH) -> DrizzleSuggestion:
    """The drizzle scale, drop size and method advice for these frames (see the module docstring), each an ``Auto``
    with its reason; the method's is an advice (``Auto.advice``: shown, never applied). ``coef``: the polynomials of
    the frames the selection keeps; ``weights``: their weights (None: equal); ``fwhm``: the stars' Moffat FWHM in
    sensor pixels (core/psf_moffat), None or NaN when not measured; ``cfa``: the Bayer pattern, or empty / None /
    "MONO" for a monochrome sensor."""
    coef = np.asarray(coef, float)
    n = int(len(coef) if weights is None else (np.asarray(weights) > 0).sum())
    rc = rule_channels(cfa)
    found: dict = {}

    def cov_at(p: _Patches, pf: float) -> Coverage:
        key = (p.scale, float(pf))
        if key not in found:
            found[key] = p.coverage(float(pf))
        return found[key]

    c1 = cov_at(_Patches(coef, W, H, cfa, 1.0, weights, grid, patch, rc), 1.0).red_blue
    one = Auto(1.0, "scale 1×: drops of a whole pixel")
    f = float(fwhm) if fwhm is not None else float("nan")

    def done(scale: Auto, pixfrac: Auto, finer=None) -> DrizzleSuggestion:
        return DrizzleSuggestion(scale, pixfrac, advise_method(cfa, n, c1, f, scale.value, finer, limit=limit,
                                                               undersampled=undersampled), f, found)

    if not np.isfinite(f):
        return done(Auto(1.0, "star size not measured: 1×"), one)
    if f >= undersampled:
        return done(Auto(1.0, "stars of {fwhm:.1f} px: sampled well enough, drizzle would add no detail: 1×",
                         {"fwhm": f}), one)
    holding = []                                         # (scale, pixfrac, variation) of the scales that keep the rule
    first = None
    for S in sorted(scales):
        p = _Patches(coef, W, H, cfa, float(S), weights, grid, patch, rc)
        steps = pixfrac_steps(S)
        top = cov_at(p, steps[-1])
        if first is None:
            first = (float(S), top.red_blue)
        if not top.holds(limit):
            continue
        lo, hi = 0, len(steps) - 1                       # the smallest drop that holds (the variation falls as it grows)
        while lo < hi:
            mid = (lo + hi) // 2
            if cov_at(p, steps[mid]).holds(limit):
                hi = mid
            else:
                lo = mid + 1
        holding.append((float(S), float(steps[hi]), cov_at(p, steps[hi]).red_blue))
    if not holding:
        S, cv = first
        if np.isfinite(cv):
            scale = Auto(1.0, "stars of {fwhm:.1f} px: undersampled, but at {scale:g}× the coverage would vary by "
                              "{cov:.0f} %: 1×", {"fwhm": f, "scale": S, "cov": 100 * cv})
        else:
            scale = Auto(1.0, "stars of {fwhm:.1f} px: undersampled, but at {scale:g}× the red and blue samples would "
                              "leave holes: 1×", {"fwhm": f, "scale": S})
        return done(scale, one, first)
    nyquist = [h for h in holding if f * h[0] >= undersampled]
    S, pf, cv = nyquist[0] if nyquist else holding[-1]
    scale = Auto(S, "stars of {fwhm:.1f} px: undersampled; coverage stays at {cov:.0f} %: {scale:g}×",
                 {"fwhm": f, "cov": 100 * cv, "scale": S})
    pixfrac = Auto(pf, "the finest drop that keeps the red and blue coverage under {limit:.0f} % ({cov:.0f} %)",
                   {"limit": 100 * limit, "cov": 100 * cv})
    return done(scale, pixfrac, (S, cv))


def advise_method(cfa, n: int, cov1: float, fwhm: float, scale: float, finer=None, *, limit: float = LIMIT,
                  undersampled: float = UNDERSAMPLED, min_frames: int = MIN_DRIZZLE_FRAMES) -> Auto:
    """The method advice of the table in the module docstring, an advice ``Auto``
    ("drizzle" or "classic", a short reason): ``n`` the frames kept, ``cov1`` the coverage variation the rule reads
    at scale 1 and drop 1 (red and blue; the one plane of a monochrome sensor), ``fwhm`` the Moffat FWHM (NaN: not
    measured), ``scale`` the scale the rule suggests, ``finer`` (scale, variation) at the finer scale it tried (None
    when it tried none)."""
    def advice(value, template, args=None):
        return Auto(value, template, args, advice=True)

    if is_mono(cfa):
        # no CFA: at scale 1 every pixel is sampled, a drizzle only adds detail when the stars are undersampled
        if n < min_frames:
            return advice("classic", M_MONO_FEW, {"n": n})
        if not np.isfinite(fwhm):
            return advice("classic", M_MONO_NO_FWHM)
        if fwhm >= undersampled:
            return advice("classic", M_MONO, {"fwhm": fwhm})
        if scale > 1.0:
            return advice("drizzle", M_MONO_DRIZZLE, {"fwhm": fwhm, "scale": scale})
        S, cv = finer if finer is not None else (min(SCALES), float("inf"))
        if np.isfinite(cv):
            return advice("classic", M_MONO_UNEVEN, {"cov": 100 * cv, "scale": S})
        return advice("classic", M_MONO_HOLES, {"scale": S})
    if n < min_frames:
        return advice("classic", M_FEW, {"n": n})
    if not np.isfinite(cov1):
        return advice("classic", M_HOLES)
    if not cov1 < limit:
        return advice("classic", M_UNEVEN, {"cov": 100 * cov1})
    if scale > 1.0 and finer is not None:
        return advice("drizzle", M_DRIZZLE_FINE, {"fwhm": fwhm, "cov": 100 * finer[1], "scale": scale})
    return advice("drizzle", M_DRIZZLE, {"n": n, "cov": 100 * cov1})
