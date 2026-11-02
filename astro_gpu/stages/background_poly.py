"""Simple sky background: a smooth polynomial surface fitted on the dark areas of the stack.

The 'Polynomial' gradient of the workflows without Stars, as a usual background extraction does it (Siril,
PixInsight's ABE): per channel, in display RGB (the colour matrix applied), a polynomial of the pixel position fitted on
sky samples. The field is cut into blocks (``samples`` x 4 per side), each one's median a sample. The sky samples are
found on the luminance, from the whole field (``sky_blocks``): the surface is fitted, the samples more than
``object_clip`` sigmas above it are left out, and again until it settles; the sigma is the scatter between neighbouring
samples. An object that fills most of the field (a large galaxy) is left out whole and the surface passes under it
(taking each cell's darkest blocks instead would sample the disc of M31, which the surface would then erase).
Each channel is then fitted on those samples, leaving out, iteratively, the ones farther than ``object_clip`` sigmas
from its surface.

Only the model is written (coefs.json, samples.npz); ``render_simple`` subtracts it (``poly_sky``).
"""
from __future__ import annotations

import numpy as np

from ..core.imgutil import asinh_preview, poly_terms
from ..core.params import ADVANCED, P, StageParams
from ..core.stage import Context, Stage, StageError

LUM = np.array([0.2126, 0.7152, 0.0722])


class BackgroundPolyParams(StageParams):
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


def has_input(ctx: Context, stage: str) -> bool:
    """Whether an optional dependency is in this workflow (``Context.has``)."""
    return bool(ctx.has(stage))


def stack_source(ctx: Context) -> tuple[np.ndarray, str]:
    """The stack a simple background or rendering works on: the denoised one when the workflow has Denoise."""
    if has_input(ctx, "denoise"):
        return ctx.load("denoise", "n2n_full.npy").astype(np.float32), "denoise/n2n_full"
    return ctx.load("stack", "drz_full.npy").astype(np.float32), "stack/drz_full"


def coverage(ctx: Context, cover_min: float = 0.6) -> np.ndarray:
    """The pixels of the image (render_prep's rule): covered by more than ``cover_min`` of the typical number of
    frames, or less covered inside the image (a bright core saturated in part of the frames, a satellite trail
    rejected in some): only the low-coverage regions reaching the border of the stack are left out."""
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
    (n,), used (3, n), block}."""
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


def sky_of(ctx: Context, H: int, W: int) -> np.ndarray:
    """The sky model written by background_poly, on a grid of H x W."""
    cj = ctx.load_json("background_poly", "coefs.json")
    if (cj["width"], cj["height"]) != (W, H):
        raise StageError(f"the simple background was fitted on a {cj['width']} x {cj['height']} grid, the stack is "
                         f"{W} x {H}: run the Gradient block again")
    return poly_sky(cj["coefs"], cj["degree"], H, W)


class BackgroundPoly(Stage):
    name = "background_poly"
    title = "Simple sky background"
    description = ("Removes the sky gradient (light pollution, the Moon, residual vignetting) with a smooth polynomial "
                   "surface fitted on the dark areas of the stack, as a usual background extraction does. Used by the "
                   "workflows without the Stars block; the physical background (H-alpha survey) needs the starless "
                   "image.")
    deps = ("stack", "color")
    Params = BackgroundPolyParams
    version = "2"          # 2: the sky samples from the whole field (sky_blocks), not the darkest blocks of every cell

    def run(self, ctx: Context, p: BackgroundPolyParams) -> None:
        ctx.progress(0.05, "loading the stack")
        src, name = stack_source(ctx)
        M = ctx.load("color", "M.npy").astype(np.float32)
        H, W = src.shape[1:]
        valid = coverage(ctx)
        rgb = np.tensordot(M, src, 1)
        del src
        ctx.progress(0.3, "sky samples")
        fit = fit_poly_sky(rgb, valid, p.degree, p.samples, p.object_clip)
        ctx.progress(0.7, "sky model")
        sky = poly_sky(fit["coefs"], p.degree, H, W)
        both = fit["used"].all(0)
        n_valid = max(int(fit["valid"].sum()), 1)
        ctx.metric("sky_fit_blocks_pct", round(100 * float(both.sum()) / n_valid, 1))
        Ls = np.tensordot(LUM, sky[:, ::8, ::8], 1)[valid[::8, ::8]]
        level = float(np.median(Ls))
        grad = float(np.percentile(Ls, 99) - np.percentile(Ls, 1))
        ctx.metric("gradient_pct", round(100 * grad / max(abs(level), 1e-9), 2))
        ctx.metric("samples_used", int(both.sum()))
        ctx.log(f"sky fitted on {int(both.sum())} of {int(fit['candidate'].sum())} sky samples "
                f"({fit['block']} px blocks); gradient {grad:.2f} on a sky of {level:.2f} ({name})")
        ctx.save_json("coefs.json", {"degree": p.degree, "width": W, "height": H, "coefs": fit["coefs"].tolist(),
                                     "basis": "u^i v^j for i in 0..degree, j in 0..degree-i; u = x / (W/2) - 1, "
                                              "v = y / (H/2) - 1 (pixel indices)",
                                     "space": "display RGB (colour matrix applied)", "source": name,
                                     "block_px": fit["block"]})
        np.savez(ctx.out("samples.npz"), x=fit["x"].astype(np.float32), y=fit["y"].astype(np.float32),
                 values=fit["values"].astype(np.float32), candidate=fit["candidate"], used=fit["used"],
                 valid=fit["valid"])
        ctx.progress(0.9, "preview")
        k = max(1, int(np.ceil(max(H, W) / 1600)))
        h, w = H // k * k, W // k * k
        small = (rgb[:, :h, :w] - sky[:, :h, :w]).reshape(3, h // k, k, w // k, k).mean(axis=(2, 4))
        ctx.preview("background", asinh_preview(small))
