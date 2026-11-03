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
METHODS = ("polynomial", "automatic")
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
                         "objects. Automatic: a more flexible surface that follows complex gradients (a corner lit by "
                         "the Moon), fitted on every pixel that does not look like an object.",
                    tip="The polynomial keeps faint, very wide structures best; the automatic model removes complex "
                        "gradients best, and with them part of the faintest wide structures.")
    degree: int = P(2, label="Surface degree", ge=1, le=5, step=1,
                    help="Degree of the smooth polynomial surface describing the sky gradient (light pollution, the "
                         "Moon, what is left of the vignetting).",
                    tip="2 for most fields; 3-4 for wide fields with complex gradients. A high degree on a field "
                        "filled with nebula can take a part of it for sky.")
    samples: int = P(16, label="Samples per side", ge=4, le=64, step=2,
                     help="The field is cut into 4 times this many blocks per side; the blocks of sky (not the "
                          "objects) are the samples of the fit.",
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
                            simplified=self.simplified, degree=self.simple_degree, downsample=self.downsample)


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


# ---------------------------------------------------------------------------------------------- the stage
class Gradient(Stage):
    name = "gradient"
    title = "Sky gradient"
    description = ("Models the sky gradient (light pollution, the Moon, what is left of the vignetting) on the stack, "
                   "right after stacking, and removes it from what the later blocks read: the colour calibration, the "
                   "denoising, the star removal and the output see a flat sky. The sky's level stays.")
    deps = ("stack",)
    Params = GradientParams
    version = "2"          # 2: the automatic method; the level kept is the model's median over the covered field

    def run(self, ctx: Context, p: GradientParams) -> None:
        ctx.progress(0.05, "loading the stack")
        stack = ctx.load("stack", "drz_full.npy").astype(np.float32)
        H, W = stack.shape[1:]
        valid = coverage(ctx)
        info = {"method": p.method, "width": W, "height": H, "space": "camera RGB, linear (before the colour "
                                                                       "calibration)"}
        if p.method == "automatic":
            ctx.progress(0.1, "automatic model")
            grid = automatic_grid(stack, valid, p, lambda c: ctx.progress(0.1 + 0.25 * c, f"automatic model, "
                                                                                        f"{'RGB'[c]}"))
            vg = enlarge_mask(valid, grid.shape[1:], p.downsample)
            info.update(grid="corners", factor=p.downsample, settings=p.auto_settings().__dict__)
            detail = f"automatic model, scale {p.scale:g}, on a copy reduced {p.downsample} times"
        else:
            ctx.progress(0.3, "sky samples")
            fit = fit_poly_sky(stack, valid, p.degree, p.samples, p.object_clip)
            ctx.progress(0.7, "sky model")
            xx, yy = grid_centres(H, W)
            A = poly_terms(xx.ravel(), yy.ravel(), W, H, p.degree)
            grid = (fit["coefs"] @ A.T).reshape(3, *xx.shape).astype(np.float32)
            vg = cv2.resize(valid.astype(np.uint8), (xx.shape[1], xx.shape[0]), interpolation=cv2.INTER_AREA) > 0
            both = fit["used"].all(0)
            ctx.metric("samples_used", int(both.sum()))
            np.savez(ctx.out("samples.npz"), x=fit["x"].astype(np.float32), y=fit["y"].astype(np.float32),
                     values=fit["values"].astype(np.float32), candidate=fit["candidate"], used=fit["used"],
                     valid=fit["valid"])
            info.update(grid="centres", factor=SKY_FACTOR, degree=p.degree, coefs=fit["coefs"].tolist(),
                        block_px=fit["block"],
                        basis="u^i v^j for i in 0..degree, j in 0..degree-i; u = x / (W/2) - 1, v = y / (H/2) - 1 "
                              "(pixel indices)")
            detail = f"fitted on {int(both.sum())} of {int(fit['candidate'].sum())} sky samples ({fit['block']} px blocks)"
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
