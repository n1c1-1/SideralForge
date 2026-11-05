"""Sky gradient: the smooth sky the light pollution, the Moon and the vignetting leave on the stack, modelled right after
stacking and removed from what every later stage reads (the state of the art: Siril, PixInsight and GraXpert remove it
on the linear stack, before the colour calibration, the denoising and the star removal).

Entry points: the stage ``Gradient``, and ``stack_image(ctx, name)``: a stack image (drz_full, drz_A, drz_B) with the
sky gradient removed when the workflow has the Gradient block. Only the gradient goes: its variation across the field,
not the sky's level, which the noise models of the later stages rely on (as Siril does).

Methods (``method``, set by the workflow's Gradient option):

* ``polynomial``: per camera channel, a polynomial of the pixel position fitted on sky samples, as Siril's and
  PixInsight ABE's background extraction. The field is cut into blocks (``samples`` x 4 per side), each one's median a
  sample. The sky samples are found on the luminance, from the whole field (``sky_blocks``): the surface is fitted, the
  samples more than ``object_clip`` sigmas above it are left out, and again until it settles; the sigma is the scatter
  between neighbouring samples. An object that fills most of the field (a large galaxy) is left out whole and the
  surface passes under it. Each channel is then fitted on those samples, leaving out, iteratively, the ones farther
  than ``object_clip`` sigmas from its surface.
* ``automatic``: Siril's automatic, sample-free model, written again (stages/gradient_auto.py): a smooth multiscale
  surface fitted on every pixel that survives an iterative rejection of the structures, each channel apart. Siril's
  thresholds are in units of an image scaled to 0..1: the stack is scaled as an image exported for it would be, from
  its darkest to its brightest pixels (0.01th and 99.99th percentiles of the covered field, 20 % of headroom above).

* ``rbf``: radial basis functions (thin-plate splines, with a smoothing), Siril's second sample-based model: the
  surface passes near every sample, so it follows a gradient of any shape, and depends on where the samples are.

The samples of the polynomial and the RBF: a grid found automatically (the sky between the objects), or the points the
user placed and moved on the image (``sample_points``, stack pixels). The areas the user excluded (``exclude_ellipses``,
``exclude_brush``: painted and erased dabs, in order) never enter any of the three models.

Outputs: sky.npy, the sky's variation (3, h, w) on a grid (its median over the covered field is 0 in each channel):
cells of ``SKY_FACTOR`` px read back bicubically for the polynomial, the automatic model's reduced grid read back
bilinearly (corners aligned) as Siril does; gradient.json (method, grid, the sky level kept, the model's settings);
samples.npz for the polynomial (the samples, those used); a preview of the stack with the gradient removed.
"""
from __future__ import annotations

import cv2
import numpy as np
from pydantic import field_validator

from ..core.imgutil import asinh_preview, poly_terms
from ..core.params import ADVANCED, P, StageParams, check_choice
from ..core.stage import Context, Stage, StageError
from .gradient_auto import AutoSettings, enlarge, sky_model

LUM = np.array([0.2126, 0.7152, 0.0722])
#: the methods of the stage (the workflow's Gradient option chooses one; the first is the default)
METHODS = ("polynomial", "rbf", "automatic")
#: how the image is scaled to 0..1 for the automatic model: these percentiles of the covered field, and headroom
UNIT_PERCENTILES = (0.01, 99.99)
UNIT_HEADROOM = 1.2
#: side of the cells of the grid sky.npy is kept on (the sky is smooth: bicubic interpolation gives it back)
SKY_FACTOR = 8
SKY_FILE = "sky.npy"
INFO_FILE = "gradient.json"


class GradientParams(StageParams):
    method: str = P("polynomial", label="Method", choices=list(METHODS),
                    help="How the sky gradient is modelled. Polynomial: a smooth surface fitted on the sky between the "
                         "objects. RBF: a surface through the samples, which follows a gradient of any shape. "
                         "Automatic: a flexible surface fitted on every pixel that does not look like an object.",
                    tip="The polynomial keeps faint, very wide structures best; the automatic model removes complex "
                        "gradients best, and with them part of the faintest wide structures.")
    degree: int = P(2, label="Surface degree", ge=1, le=5, step=1,
                    help="Degree of the smooth polynomial surface describing the sky gradient (light pollution, the "
                         "Moon, what is left of the vignetting).",
                    tip="2 for most fields; 3-4 for wide fields with complex gradients. A high degree on a field "
                        "filled with nebula can take a part of it for sky.")
    samples: int = P(16, label="Samples per side", ge=4, le=64, step=2,
                     help="With no sample placed. Polynomial: the field is cut into 4 times this many blocks per "
                          "side, the blocks of sky (not the objects) are the samples of the fit. RBF: this many "
                          "samples along the shorter side.",
                     tip="More, smaller blocks find sky between the objects of a crowded field; fewer, larger ones "
                         "average more noise.")
    object_clip: float = P(2.5, label="Object rejection", level=ADVANCED, ge=1.0, le=10, step=0.1, unit="sigma",
                           help="Samples farther from the surface than this are taken as objects (nebula, galaxy, a "
                                "bright star) and left out of the fit, iteratively.")
    scale: float = P(5.0, label="Model scale", ge=1.0, le=10.0, step=0.5,
                     help="Automatic model: the scale below which variations are taken as objects rather than "
                          "gradient, relative to the field.",
                     tip="Higher keeps only the large gradients (a smoother model); lower follows more local "
                         "gradients, and may take large faint objects for sky.")
    smoothness: float = P(1.0, label="Model smoothness", level=ADVANCED, ge=0.0, le=5.0, step=0.1,
                          help="Automatic model: extra smoothing of the final model (0: none).")
    protect: bool = P(True, label="Structure protection", level=ADVANCED,
                      help="Automatic model: masks the extended bright structures (nebulae) so that they are not "
                           "taken into the model.")
    protect_threshold: float = P(0.05, label="Protection threshold", level=ADVANCED, ge=0.0, le=1.0, step=0.01,
                                 help="Automatic model: brightness above the model, on the image scaled to 0..1, at "
                                      "which a pixel is taken as a structure. Lower protects more.")
    protect_amount: float = P(0.5, label="Protection growth", level=ADVANCED, ge=0.0, le=1.0, step=0.05,
                              help="Automatic model: how far the protection mask grows around the structures.")
    simplified: bool = P(False, label="Plane first", level=ADVANCED,
                         help="Automatic model: a stiff polynomial is fitted and removed first, the flexible model "
                              "then works on what it leaves. Helps on gradients as strong as the objects themselves "
                              "(a bright urban sky, a strong Moon gradient).",
                         tip="Can take part of a large bright object with it: keep its degree low.")
    simple_degree: int = P(1, label="Plane-first degree", level=ADVANCED, ge=1, le=6, step=1,
                           help="Automatic model: degree of the stiff polynomial removed first (1: a plane).")
    downsample: int = P(4, label="Working reduction", level=ADVANCED, ge=1, le=8, unit="x",
                        help="Automatic model: the model is computed on a copy reduced this many times: faster, "
                             "coarser; the scale of the background itself does not change.")
    protect_sigmas: float = P(0.0, label="Protection in sky noise", level=ADVANCED, ge=0.0, le=50.0, step=0.5,
                              help="Automatic model: when not 0, the protection threshold is this many times the "
                                   "noise of the sky instead of a fraction of the image's range (a faint nebula is "
                                   "far below that fraction when bright stars set the range).")
    rbf_smooth: float = P(0.5, label="RBF smoothing", level=ADVANCED, ge=0.0, le=1.0, step=0.05,
                          help="RBF: 0, the surface passes through every sample; higher, it passes near them, "
                               "smoother.")
    sample_points: list[list[float]] = P([], label="Sample points", level=ADVANCED,
                                         help="The samples placed on the image (stack pixels); empty: the automatic "
                                              "grid.")
    exclude_ellipses: list[list[float]] = P([], label="Excluded ellipses", level=ADVANCED,
                                            help="Areas left out of the sky model: centre x, y, half axes and "
                                                 "angle (degrees).")
    exclude_brush: list[list[float]] = P([], label="Excluded brush strokes", level=ADVANCED,
                                         help="Brush dabs x, y, radius, 1 to exclude / 0 to erase, in order.")
    avoid_emission: bool = P(True, label="Away from emission",
                             help="Samples chosen by the block (none placed) never sit where the professional H-alpha "
                                  "survey of the field sees nebulosity: the faint extensions of an emission nebula are "
                                  "not taken for sky. The survey only says where not to sample; the sky model never "
                                  "reads it. No survey for the field: no effect.",
                             tip="On by default; the samples placed by hand are always kept as placed.")

    @field_validator("method")
    @classmethod
    def _method(cls, v: str) -> str:
        return check_choice("method", v, METHODS)

    @field_validator("downsample")
    @classmethod
    def _downsample(cls, v: int) -> int:
        return check_choice("downsample", v, (1, 2, 4, 8))

    def auto_settings(self) -> AutoSettings:
        return AutoSettings(scale=self.scale, smoothness=self.smoothness, protect=self.protect,
                            protect_threshold=self.protect_threshold, protect_amount=self.protect_amount,
                            simplified=self.simplified, degree=self.simple_degree, downsample=self.downsample,
                            protect_sigmas=self.protect_sigmas)


# ---------------------------------------------------------------------------------------------- the polynomial sky
def coverage(ctx: Context, cover_min: float = 0.6) -> np.ndarray:
    """The pixels of the image: covered by more than ``cover_min`` of the typical number of frames, or less covered
    inside the image (a bright core saturated in part of the frames, a satellite trail rejected in some): only the
    low-coverage regions reaching the border of the stack are left out."""
    from scipy import ndimage
    wt = np.asarray(ctx.load("stack", "drz_weight.npy", mmap=True)[1], np.float32)
    pos = wt[wt > 0]
    if not pos.size:
        raise StageError("the stack covers no pixel")
    good = wt > cover_min * float(np.median(pos))
    lab, n_lab = ndimage.label(~good)
    if n_lab:
        border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        good |= (lab > 0) & ~np.isin(lab, border[border > 0]) & (wt > 0)
    return good


def block_size(H: int, W: int, samples: int) -> int:
    """Side of the square blocks: each cell of the samples x samples grid holds about 4 x 4 of them."""
    return max(4, int(min(H / samples, W / samples) // 4))


def fit_poly_sky(rgb: np.ndarray, valid: np.ndarray, degree: int, samples: int, clip: float, iters: int = 10) -> dict:
    """Polynomial sky of a linear (3, H, W) image, fitted on sky samples (see the module).

    Returns {coefs (3, n_terms) on ``poly_terms`` of the pixel position, x, y (block centres), values (3, n), candidate
    (n,), used (3, n), valid (n,), block}."""
    c, H, W = rgb.shape
    b = block_size(H, W, samples)
    hb, wb = H // b, W // b
    if hb < 2 or wb < 2:
        raise StageError(f"the image ({W} x {H}) is too small for {samples} samples per side")
    blk = np.stack([np.median(rgb[k, :hb * b, :wb * b].reshape(hb, b, wb, b).transpose(0, 2, 1, 3).reshape(hb, wb, -1),
                              axis=-1) for k in range(c)])
    vb = valid[:hb * b, :wb * b].reshape(hb, b, wb, b).mean(axis=(1, 3)) >= 0.99
    yc, xc = (np.mgrid[0:hb, 0:wb] + 0.5) * b - 0.5
    L = np.tensordot(LUM, blk, 1)
    if vb.sum() < 10:
        raise StageError("too few covered areas to fit a background")
    A = poly_terms(xc.ravel(), yc.ravel(), W, H, degree)
    ok = sky_blocks(L.ravel().astype(np.float64), vb.ravel(), A, clip, vb.shape)
    if ok.sum() < 3 * A.shape[1]:
        raise StageError(f"only {int(ok.sum())} sky samples for a degree-{degree} surface: lower the degree or the "
                         f"number of samples")
    coefs, used = [], []
    for k in range(c):
        y = blk[k].ravel().astype(np.float64)
        m = ok.copy()
        q = np.zeros(A.shape[1])
        for _ in range(iters):
            q = np.linalg.lstsq(A[m], y[m], rcond=None)[0]
            r = y - A @ q
            s = 1.4826 * float(np.median(np.abs(r[m] - np.median(r[m])))) + 1e-12
            new = ok & (np.abs(r) < clip * s)
            if new.sum() < A.shape[1] or (new == m).all():
                break
            m = new
        coefs.append(q)
        used.append(m)
    return {"coefs": np.array(coefs), "x": xc.ravel(), "y": yc.ravel(), "values": blk.reshape(c, -1),
            "candidate": ok, "used": np.array(used), "valid": vb.ravel(), "block": b}


def sky_blocks(y: np.ndarray, valid: np.ndarray, A: np.ndarray, clip: float, shape: tuple[int, int],
               iters: int = 50) -> np.ndarray:
    """The sky among the block samples ``y`` (the valid ones, on a grid of ``shape``): the surface ``A`` fitted on
    them, the samples more than ``clip`` sigmas above it left out, and again until it settles: the surface sinks to
    the sky under the objects, whatever share of the field they cover. The sigma is the scatter of neighbouring
    samples (1.4826 x the median of their differences / sqrt 2): the noise and the fine structure of the sky, not the
    objects, which are smooth from one block to the next. Samples below the surface are all kept (the sky is there)."""
    g = y.reshape(shape)
    v = valid.reshape(shape)
    dif = np.concatenate([(g[:, 1:] - g[:, :-1])[v[:, 1:] & v[:, :-1]], (g[1:] - g[:-1])[v[1:] & v[:-1]]])
    s = 1.4826 * float(np.median(np.abs(dif - np.median(dif)))) / np.sqrt(2) + 1e-12 if dif.size else 0.0
    m = valid.copy()
    for _ in range(iters):
        q = np.linalg.lstsq(A[m], y[m], rcond=None)[0]
        r = y - A @ q
        new = valid & (r < clip * s)
        if new.sum() < 3 * A.shape[1] or (new == m).all():
            break
        m = new
    return m


def poly_sky(coefs, degree: int, H: int, W: int) -> np.ndarray:
    """The polynomial surfaces (3, H, W) float32 of ``coefs`` (on ``poly_terms``), evaluated exactly, by rows."""
    u = np.arange(W, dtype=np.float64) / (W / 2) - 1
    v = np.arange(H, dtype=np.float64) / (H / 2) - 1
    coefs = np.asarray(coefs, np.float64)
    out = np.zeros((coefs.shape[0], H, W), np.float32)
    for c in range(coefs.shape[0]):
        k = 0
        for i in range(degree + 1):
            col = np.zeros(H)
            for j in range(degree + 1 - i):
                col += coefs[c, k] * v ** j
                k += 1
            out[c] += np.multiply.outer(col.astype(np.float32), (u ** i).astype(np.float32))
    return out


# ---------------------------------------------------------------------------------------------- the grid and its reading
def grid_shape(H: int, W: int, f: int = SKY_FACTOR) -> tuple[int, int]:
    return -(-H // f), -(-W // f)


def grid_centres(H: int, W: int, f: int = SKY_FACTOR) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of the centres of the grid's cells, in pixels of the stack (the convention of ``cv2.resize``)."""
    h, w = grid_shape(H, W, f)
    yy, xx = (np.mgrid[0:h, 0:w] + 0.5) * f - 0.5
    return xx, yy


def upsample(grid: np.ndarray, H: int, W: int, f: int = SKY_FACTOR) -> np.ndarray:
    """A sky grid (3, h, w) of ``f`` px cells brought back to (3, H, W), bicubic."""
    h, w = grid.shape[1:]
    out = np.empty((grid.shape[0], H, W), np.float32)
    for c in range(grid.shape[0]):
        out[c] = cv2.resize(grid[c].astype(np.float32), (w * f, h * f), interpolation=cv2.INTER_CUBIC)[:H, :W]
    return out


def read_grid(grid: np.ndarray, info: dict, H: int, W: int) -> np.ndarray:
    """A sky grid (3, h, w) brought back to (3, H, W) the way its method stores it (``info["grid"]``)."""
    if info.get("grid") == "corners":
        return np.stack([enlarge(g, H, W) for g in grid])
    return upsample(grid, H, W, int(info["factor"]))


def sky_variation(ctx: Context, H: int, W: int) -> np.ndarray:
    """The sky gradient the Gradient block found, (3, H, W): its variation, of median 0 over the covered field."""
    info = ctx.load_json("gradient", INFO_FILE)
    if (info["width"], info["height"]) != (W, H):
        raise StageError(f"the sky gradient was modelled on a {info['width']} x {info['height']} stack, this one is "
                         f"{W} x {H}: run the Gradient block again")
    return read_grid(ctx.load("gradient", SKY_FILE), info, H, W)


def enlarge_mask(valid: np.ndarray, shape: tuple[int, int], f: int) -> np.ndarray:
    """The cells of a reduced grid of ``shape`` (blocks of ``f`` px) wholly covered."""
    h, w = shape
    return valid[:h * f, :w * f].reshape(h, f, w, f).all(axis=(1, 3))


def unit_scale(stack: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """(lo, span): ``(stack - lo) / span`` is the stack in 0..1 as an image exported for Siril (see the module)."""
    lo, hi = np.percentile(stack[:, valid][:, ::7], UNIT_PERCENTILES)
    return float(lo), float(max(hi - lo, 1e-12) * UNIT_HEADROOM)


def automatic_grid(stack: np.ndarray, valid: np.ndarray, p: GradientParams, progress=None) -> np.ndarray:
    """The automatic model's background (3, H // downsample, W // downsample), in the stack's units."""
    lo, span = unit_scale(stack, valid)
    s = p.auto_settings()
    out = []
    for c in range(stack.shape[0]):
        if progress:
            progress(c)
        unit = np.clip((stack[c] - lo) / span, 0, 1).astype(np.float32)
        out.append(sky_model(unit, s, valid) * np.float32(span) + np.float32(lo))
    return np.stack(out)


def stack_image(ctx: Context, name: str = "drz_full.npy") -> np.ndarray:
    """A stack image (drz_full.npy, drz_A.npy or drz_B.npy) as float32, its sky gradient removed when the workflow has
    the Gradient block (the half-stacks share the full stack's sky)."""
    a = ctx.load("stack", name).astype(np.float32)
    if ctx.has("gradient"):
        a -= sky_variation(ctx, *a.shape[1:])
    return a


# ---------------------------------------------------------------------------------------------- samples and areas
def exclusion_mask(p: GradientParams, h: int, w: int, f: float = 1.0) -> np.ndarray | None:
    """The excluded areas on an array of (h, w) whose pixels are ``f`` stack pixels^-1 (a reduced copy: f < 1): True
    where excluded; None when nothing is excluded."""
    if not p.exclude_ellipses and not p.exclude_brush:
        return None
    m = np.zeros((h, w), np.uint8)
    for e in p.exclude_ellipses:
        cx, cy, rx, ry = e[:4]
        angle = float(e[4]) if len(e) > 4 else 0.0           # degrees, clockwise on the image (y down)
        cv2.ellipse(m, (round(cx * f), round(cy * f)), (max(1, round(rx * f)), max(1, round(ry * f))), angle, 0, 360,
                    1, -1)
    for x, y, r, paint in p.exclude_brush:
        cv2.circle(m, (round(x * f), round(y * f)), max(1, round(r * f)), 1 if paint else 0, -1)
    return m > 0


def point_values(rgb: np.ndarray, valid: np.ndarray, pts: np.ndarray, box: int) -> tuple[np.ndarray, np.ndarray]:
    """The median of each channel in a ``box`` px square around each point (array pixels); a point is usable when
    most of its square is valid. Returns (values (3, n), usable (n,))."""
    H, W = valid.shape
    vals = np.full((rgb.shape[0], len(pts)), np.nan)
    ok = np.zeros(len(pts), bool)
    r = max(1, box // 2)
    for i, (x, y) in enumerate(np.asarray(pts, float)):
        x0, x1, y0, y1 = int(x) - r, int(x) + r + 1, int(y) - r, int(y) + r + 1
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        v = valid[y0:y1, x0:x1]
        if v.mean() < 0.5:
            continue
        vals[:, i] = [np.median(rgb[c, y0:y1, x0:x1][v]) for c in range(rgb.shape[0])]
        ok[i] = True
    return vals, ok


def grid_points(rgb: np.ndarray, valid: np.ndarray, samples: int, clip: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Siril's generated grid: ``samples`` points along the shorter side, a square box around each; the samples of
    sky are kept (``sky_blocks``: the ones not above a smooth surface through them by ``clip`` sigmas). Returns
    (points (n, 2) array pixels, kept (n,), box px)."""
    H, W = valid.shape
    step = min(H, W) / samples
    box = max(4, int(step / 2))
    ys = np.arange(step / 2, H, step)
    xs = np.arange(step / 2, W, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pts = np.c_[xx.ravel(), yy.ravel()]
    vals, ok = point_values(rgb, valid, pts, box)
    L = LUM @ np.nan_to_num(vals)
    A = poly_terms(pts[:, 0], pts[:, 1], W, H, 2)
    kept = sky_blocks(L, ok, A, clip, xx.shape) if ok.sum() >= 3 * A.shape[1] else ok
    return pts, kept, box


#: a sample of the generated grid sits on emission when the H-alpha survey under it is above this level (0 the
#: survey's sky, 1 its bright emission: surveys.emission_level); on the Rosette, 2 % of that range is 3 % of the core
EMISSION_FREE = 0.02
#: the survey leaves no sample out when it would keep fewer than this share of them (a field without emission: the
#: survey's noise then spans its whole range)
EMISSION_MIN_KEPT = 0.25


def away_from_emission(pts: np.ndarray, kept: np.ndarray, level: np.ndarray, box: int) -> tuple[np.ndarray, bool]:
    """The grid's samples of sky (``kept``) less the ones on emission: the median of the survey's ``level`` (on the
    grid of ``pts``) in each sample's box above ``EMISSION_FREE``. A careful user's rule (no sample on nebulosity),
    the survey showing where the nebulosity is. Returns (kept, whether the survey was used)."""
    H, W = level.shape
    h = max(1, box // 2)
    on = np.zeros(len(pts), bool)
    for i, (x, y) in enumerate(pts):
        x0, y0 = int(max(0, x - h)), int(max(0, y - h))
        cell = level[y0:int(min(H, y + h + 1)), x0:int(min(W, x + h + 1))]
        v = cell[np.isfinite(cell)]
        on[i] = v.size > 0 and float(np.median(v)) > EMISSION_FREE
    out = kept & ~on
    if out.sum() < EMISSION_MIN_KEPT * max(int(kept.sum()), 1):
        return kept, False
    return out, True


def rbf_grid(pts: np.ndarray, vals: np.ndarray, xx: np.ndarray, yy: np.ndarray, H: int, W: int,
             smooth: float) -> np.ndarray:
    """Thin-plate splines through the samples, evaluated at (xx, yy). The coordinates are in units of the image's
    longer side and each channel's values in units of their own spread, so that the smoothing means the same on every
    image (0: through every sample)."""
    from scipy.interpolate import RBFInterpolator
    s = float(max(H, W))
    out = np.empty((vals.shape[0], *xx.shape), np.float32)
    q = np.c_[xx.ravel() / s, yy.ravel() / s]
    for c in range(vals.shape[0]):
        v = vals[c]
        mid, spread = float(np.median(v)), float(np.std(v)) + 1e-12
        f = RBFInterpolator(pts / s, (v - mid) / spread, kernel="thin_plate_spline", smoothing=float(smooth))
        out[c] = (f(q) * spread + mid).reshape(xx.shape)
    return out


def emission_mask(level: np.ndarray | None, valid: np.ndarray) -> np.ndarray | None:
    """Where the H-alpha survey (its ``level``, surveys.emission_level, on the grid of ``valid``) sees nebulosity:
    above ``EMISSION_FREE``. None when there is no survey, or when it would leave less than ``EMISSION_MIN_KEPT`` of
    the covered field (a field without emission: the survey's noise spans its whole range)."""
    if level is None:
        return None
    # the level smoothed over a cell of a 32-per-side grid before the threshold, as the editor takes each sample's
    # median: areas, not the survey's noise around the threshold (scattered free pixels leave no whole sky block)
    k = max(3, int(min(level.shape) / 32) | 1)
    on = cv2.blur(np.nan_to_num(level, nan=0.0).astype(np.float32), (k, k)) > EMISSION_FREE
    n = int(valid.sum())
    if n == 0 or int((valid & ~on).sum()) < EMISSION_MIN_KEPT * n:
        return None
    return on


def sky_model_of(stack: np.ndarray, valid: np.ndarray, p: GradientParams, f: float = 1.0, progress=None,
                 emission: np.ndarray | None = None, _ok: np.ndarray | None = None) -> dict:
    """The sky model of the stage's method on ``stack`` (3, H, W), which may be a copy reduced ``1 / f`` times (the
    preview of the editor: the points and areas are in stack pixels). ``emission``: the survey's emission level on the
    same grid (surveys.emission_level), or None; with ``avoid_emission``, the samples the block chooses (none placed)
    leave it out. Returns {grid (3, h, w), info (how to read it back), covered (the grid's covered cells), samples: {x,
    y, used, manual} in stack pixels or None, detail}."""
    H, W = stack.shape[1:]
    excl = exclusion_mask(p, H, W, f)
    ok = valid & ~excl if excl is not None else valid
    if _ok is not None:                        # the pixels already chosen (away from the emission)
        ok = _ok
    info = {"method": p.method}
    if p.method != "automatic" and not p.sample_points and p.avoid_emission:
        em = emission_mask(emission, ok)
        if em is not None:
            try:
                m = sky_model_of(stack, valid, p.model_copy(update={"avoid_emission": False}), f, progress,
                                 _ok=ok & ~em)
                m["info"]["away_from_emission"] = True
                return m
            except StageError as e:            # too little sky left beside the nebulosity: the image's own samples
                info["away_from_emission"] = False
                info["away_note"] = f"too little sky away from the emission ({e}): samples chosen on the image alone"
    samples = None
    if p.method == "automatic":
        # a reduced copy (the editor's preview): at least halved again, the model's cost falls with the pixels
        ds = int(min((d for d in (2, 4, 8) if d >= p.downsample * f), default=2)) if f < 1 else p.downsample
        pp = p.model_copy(update={"downsample": max(1, ds)})
        grid = automatic_grid(stack, ok, pp, progress)
        covered = enlarge_mask(valid, grid.shape[1:], pp.downsample)
        info.update(grid="corners", factor=pp.downsample, settings=pp.auto_settings().__dict__)
        detail = f"automatic model, scale {p.scale:g}"
        return {"grid": grid, "info": info, "covered": covered, "samples": None, "detail": detail}
    xx, yy = grid_centres(H, W)
    covered = cv2.resize(valid.astype(np.uint8), (xx.shape[1], xx.shape[0]), interpolation=cv2.INTER_AREA) > 0
    manual = bool(p.sample_points)
    if p.method == "polynomial" and not manual:
        fit = fit_poly_sky(stack, ok, p.degree, p.samples, p.object_clip)
        grid = (fit["coefs"] @ poly_terms(xx.ravel(), yy.ravel(), W, H, p.degree).T).reshape(3, *xx.shape)
        both = fit["used"].all(0)
        samples = {"x": fit["x"][fit["candidate"]] / f, "y": fit["y"][fit["candidate"]] / f,
                   "used": both[fit["candidate"]], "manual": False, "box": fit["block"] / f}
        info.update(degree=p.degree, coefs=fit["coefs"].tolist(), block_px=fit["block"] / f)
        detail = f"fitted on {int(both.sum())} of {int(fit['candidate'].sum())} sky samples"
        return {"grid": grid.astype(np.float32), "info": {**info, "grid": "centres", "factor": SKY_FACTOR},
                "covered": covered, "samples": samples, "detail": detail, "fit": fit}
    if manual:
        pts = np.asarray(p.sample_points, float) * f
        box = max(4, int(min(H, W) / p.samples / 2))
        vals, use = point_values(stack, ok, pts, box)
    else:
        pts, use, box = grid_points(stack, ok, p.samples, p.object_clip)
        vals, _ok = point_values(stack, ok, pts, box)
        use &= _ok
    n_min = 3 if p.method == "rbf" else (p.degree + 1) * (p.degree + 2) // 2 + 1
    if use.sum() < n_min:
        raise StageError(f"only {int(use.sum())} usable sky samples: place more samples, or exclude less")
    if p.method == "rbf":
        grid = rbf_grid(pts[use], vals[:, use], xx, yy, H, W, p.rbf_smooth)
        info.update(smoothing=p.rbf_smooth)
    else:
        A = poly_terms(pts[use, 0], pts[use, 1], W, H, p.degree)
        coefs = np.stack([np.linalg.lstsq(A, vals[c, use], rcond=None)[0] for c in range(3)])
        grid = (coefs @ poly_terms(xx.ravel(), yy.ravel(), W, H, p.degree).T).reshape(3, *xx.shape)
        info.update(degree=p.degree, coefs=coefs.tolist())
    samples = {"x": pts[:, 0] / f, "y": pts[:, 1] / f, "used": use, "manual": manual, "box": box / f}
    detail = f"{p.method} on {int(use.sum())} of {len(use)} {'placed' if manual else 'grid'} samples"
    return {"grid": grid.astype(np.float32), "info": {**info, "grid": "centres", "factor": SKY_FACTOR},
            "covered": covered, "samples": samples, "detail": detail}


# ---------------------------------------------------------------------------------------------- the stage
def survey_emission(ctx: Context, p: GradientParams) -> np.ndarray | None:
    """The H-alpha survey's emission level on the stack (surveys.emission_level) when the samples are the block's own
    and must avoid it, the chain has the survey and the survey covers the field; else None."""
    if not p.avoid_emission or p.method == "automatic" or p.sample_points or not ctx.has("surveys"):
        return None
    if ctx.load_json("surveys", "survey.json").get("survey", "none") == "none":
        return None
    from .surveys import emission_level
    return emission_level(ctx.load("surveys", "ref_ha.npy"))


class Gradient(Stage):
    name = "gradient"
    title = "Sky gradient"
    description = ("Models the sky gradient (light pollution, the Moon, what is left of the vignetting) on the stack, "
                   "right after stacking, and removes it from what the later blocks read: the colour calibration, the "
                   "denoising, the star removal and the output see a flat sky. The sky's level stays.")
    deps = ("stack",)
    Params = GradientParams
    version = "4"          # 2: the automatic method; the level kept is the model's median over the covered field;
                           # 3: the RBF, the placed samples and the excluded areas; 4: away from the survey's emission

    def run(self, ctx: Context, p: GradientParams) -> None:
        ctx.progress(0.05, "loading the stack")
        stack = ctx.load("stack", "drz_full.npy").astype(np.float32)
        H, W = stack.shape[1:]
        valid = coverage(ctx)
        ctx.progress(0.2, f"{p.method} sky model")
        m = sky_model_of(stack, valid, p, progress=lambda c: ctx.progress(0.2 + 0.25 * c, f"automatic model, "
                                                                                        f"{'RGB'[c]}"),
                         emission=survey_emission(ctx, p))
        if m["info"].get("away_from_emission"):
            ctx.log("the samples leave out the nebulosity the H-alpha survey sees")
        elif m["info"].get("away_note"):
            ctx.log(m["info"]["away_note"], "warning")
        grid, vg, detail = m["grid"], m["covered"], m["detail"]
        info = {**m["info"], "width": W, "height": H, "space": "camera RGB, linear (before the colour calibration)"}
        if m["samples"] is not None:
            sm = m["samples"]
            ctx.metric("samples_used", int(np.sum(sm["used"])))
            np.savez(ctx.out("samples.npz"), x=np.asarray(sm["x"], np.float32), y=np.asarray(sm["y"], np.float32),
                     used=np.asarray(sm["used"]), manual=bool(sm["manual"]), box=float(sm["box"]))
        level = np.array([float(np.median(grid[c][vg])) for c in range(3)])
        grid -= level.astype(np.float32)[:, None, None]
        np.save(ctx.out(SKY_FILE), grid)
        Ls = np.tensordot(LUM, grid, 1)[vg]
        lum_level = float(LUM @ level)
        spread = float(np.percentile(Ls, 99) - np.percentile(Ls, 1))
        ctx.metric("gradient_pct", round(100 * spread / max(abs(lum_level), 1e-9), 2))
        ctx.save_json(INFO_FILE, {**info, "level": level.tolist()})
        ctx.log(f"sky gradient: {detail}; {spread:.2f} across the field on a sky of {lum_level:.2f}")
        ctx.progress(0.9, "preview")
        k = max(1, int(np.ceil(max(H, W) / 1600)))
        h, w = H // k * k, W // k * k
        flat = stack[:, :h, :w] - read_grid(grid, info, H, W)[:, :h, :w]
        ctx.preview("gradient", asinh_preview(flat.reshape(3, h // k, k, w // k, k).mean(axis=(2, 4))))
