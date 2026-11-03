"""Simple output preparation: the linear image a simple rendering stretches (render/simple.py).

The output of the workflows without Stars (Classic stack, Classic + denoise) and of the Raw stack exit:

* lin.npy = M x stack - sky, in display RGB: the denoised stack when the workflow has Denoise, else the stack with the
  sky gradient removed (stages/gradient.py); the sky is then a level: the one the Gradient block measured between
  the objects (a median over the whole field would take a nebula that fills it for sky, differently in each colour);
* with Color, M is the colour stage's white balance (one factor per colour);
* without Color (the Raw stack exit), M is the identity; the sky of each channel (a smooth polynomial) is subtracted,
  the channels are scaled to the same sky spread, and the sky's brightness is added back grey: the camera's colour
  balance and the colour of the light pollution do not tint the preview, the gradient stays visible (it is the
  Gradient block's job), the colours of the objects stay camera colours;
* lin_raw.npy (with Denoise): the same from the raw stack, for the noise slider of Post;
* valid.npy, the covered pixels, and prep_simple.json: M, the crop, the statistics of the stretch (median, MAD and
  99.995th percentile of the luminance and of each channel), the settings.
"""
from __future__ import annotations

import numpy as np
from pydantic import field_validator

from ..core.params import P, StageParams
from ..core.stage import Context, Stage, StageError
from .gradient import LUM, GradientParams, coverage, fit_poly_sky, poly_sky, stack_image
from .render_prep import crop_box

STRETCHES = ["auto"]                       # the other stretches of the spec (GHS, arcsinh, STF, none) are not in the engine


def sky_spread(v: np.ndarray) -> float:
    """Robust spread of the sky in a channel: 1.4826 x the median distance below the median (the stars and nebulae
    of the field, above the median, do not widen it)."""
    med = float(np.median(v))
    below = med - v[v < med]
    return 1.4826 * float(np.median(below)) if below.size else 0.0


def stack_source(ctx: Context) -> tuple[np.ndarray, str]:
    """The stack the rendering works on: the denoised one when the workflow has Denoise, else the stack (its sky
    gradient removed when the workflow has the Gradient block)."""
    if ctx.has("denoise"):
        return ctx.load("denoise", "n2n_full.npy").astype(np.float32), "denoise/n2n_full"
    return stack_image(ctx), "stack/drz_full"


def raw_exit_sky(ctx: Context, cam: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, str]:
    """The sky of the Raw stack exit, per camera channel: the Gradient block's polynomial with its default settings
    (in camera colours; the exit has no Gradient block). The exit does not remove it: the stage subtracts it per
    channel and adds back its brightness, grey. Where it cannot be fitted (a tiny stack), the median of each channel."""
    bp = GradientParams()
    try:
        fit = fit_poly_sky(cam, valid, bp.degree, bp.samples, bp.object_clip)
    except StageError as exc:
        ctx.log(f"the sky could not be modelled ({exc}): the median of each channel is taken", "warning")
        return np.array([np.median(cam[c][valid]) for c in range(3)], np.float32)[:, None, None], "channel medians"
    return poly_sky(fit["coefs"], bp.degree, *cam.shape[1:]), "camera-colour polynomial, made grey"


def repair_saturated(lin: np.ndarray, sat: np.ndarray, valid: np.ndarray, ring_px: float) -> tuple[np.ndarray, int]:
    """The colour of the places with clipped samples (``sat`` > 0, the stack's map: star cores), which is not
    measured there: a channel clipped in some frames and not in others, or the colours of a star landing a
    fraction of a pixel apart, gave the cores of the bright stars a coloured disc or a blue-cyan crescent. Each place
    (a connected region, grown by a pixel) takes the colour of the light just around it (the unsaturated pixels within
    ``ring_px``, sky removed, weighted by their brightness), at the lowest brightness that every channel allows (a
    clipped channel is a lower bound). In place on ``lin`` (3, H, W); returns (lin, number of places)."""
    from scipy import ndimage as ndi
    reg = ndi.binary_dilation(np.asarray(sat) > 0, iterations=1) & valid
    lab, n = ndi.label(reg)
    if n == 0:
        return lin, 0
    dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    ring = (dist <= ring_px) & (lab == 0) & valid
    near = lab[iy, ix]
    idx = np.arange(1, n + 1)
    sums = np.stack([ndi.sum_labels(lin[c], np.where(ring, near, 0), idx) for c in range(3)], 1)   # (n, 3)
    tot = sums.sum(1)
    col = np.where(tot[:, None] > 0, np.clip(sums, 0, None) / np.maximum(tot, 1e-30)[:, None], np.nan)
    ok = np.isfinite(col).all(1) & (tot > 0)
    col = np.vstack([np.full((1, 3), np.nan), col])          # label 0: none
    inside = reg & ok[np.maximum(lab - 1, 0)] & (lab > 0)
    c = col[lab[inside]]                                     # (m, 3)
    v = lin[:, inside].T                                     # (m, 3)
    I = np.max(np.where(c > 0.02, v / np.maximum(c, 1e-6), -np.inf), axis=1)
    I = np.maximum(I, 0)
    lin[:, inside] = (I[:, None] * c).T.astype(lin.dtype)
    return lin, int(ok.sum())


class RenderSimpleParams(StageParams):
    stretch: str = P("auto", label="Stretch", choices=STRETCHES,
                     help="How the linear image is brought to the screen. 'auto': an automatic stretch from the "
                          "statistics of the image (black just below the sky, the sky at a set brightness).")
    linked: bool = P(True, label="Linked channels",
                     help="One stretch for the three channels (keeps the colour balance of the image); off, each "
                          "channel is stretched on its own (neutralises a colour cast, also the real colours of the "
                          "sky).")

    @field_validator("stretch")
    @classmethod
    def _known_stretch(cls, v: str) -> str:
        if v not in STRETCHES:
            raise ValueError(f"stretch {v!r} is not in the engine yet (allowed: {STRETCHES})")
        return v


class RenderSimple(Stage):
    name = "render_simple"
    title = "Simple output"
    description = ("Prepares the output of a workflow without the Stars block, or pulled from Stacking: the stack (or "
                   "the denoised stack) in screen colours, with its sky removed, and the statistics of an automatic "
                   "stretch. Post renders it with its sliders (sky, contrast, saturation, noise).")
    deps = ("stack",)
    Params = RenderSimpleParams
    version = "6"          # 2: the emission anchor of render_prep (the H-alpha of Classic was violet on heart_a7s);
                           # 3: the Raw stack exit's sky made grey (it had the green of the light pollution);
                           # 4: the colour of the places near saturation from their surroundings (repair_saturated);
                           # 5: the sky gradient removed before (the Gradient block): a level here;
                           # 6: that level is the Gradient block's; no emission anchor (the colour is a white balance)

    def run(self, ctx: Context, p: RenderSimpleParams) -> None:
        from ..render.simple import SimpleRenderEngine, SimpleSettings, image_stats
        ctx.progress(0.05, "loading the stack")
        color, denoised = ctx.has("color"), ctx.has("denoise")
        src, name = stack_source(ctx)
        H, W = src.shape[1:]
        valid = coverage(ctx)
        y0, y1, x0, x1 = crop_box(valid, 0.995)
        shown = np.zeros_like(valid)
        shown[y0:y1, x0:x1] = True
        shown &= valid
        M = ctx.load("color", "M.npy").astype(np.float64) if color else np.eye(3)
        lin = np.tensordot(M.astype(np.float32), src, 1)
        ctx.progress(0.2, "sky")
        sky_name = "channel medians"
        if not color:
            # the spread of the sky of each channel, for the gains below, measured on the stack (the objects keep the
            # scaling they had before the sky was made grey)
            spread = np.array([sky_spread(lin[c][shown]) for c in range(3)])
        if not color:                               # the Raw stack exit: a grey sky (below)
            sky, sky_name = raw_exit_sky(ctx, lin, valid)
        elif ctx.has("gradient"):                   # the level the Gradient block measured between the objects
            level = np.asarray(ctx.load_json("gradient", "gradient.json")["level"], np.float64)
            sky = (M @ level).astype(np.float32)[:, None, None]
            sky_name = "sky level of the Gradient block"
        else:
            sky = np.array([np.median(lin[c][shown]) for c in range(3)], np.float32)[:, None, None]
        lin -= sky
        del src
        from .color import saturation_map
        sat = saturation_map(ctx, clipped=True)
        if sat is not None and sat.shape == valid.shape:
            S = float(ctx.load_json("stack", "drz_meta.json").get("scale", 1.0))
            lin, n_sat = repair_saturated(lin, sat, valid, 3.0 * S)
            ctx.metric("saturated_places", n_sat)
            if n_sat:
                ctx.log(f"{n_sat} clipped places (star cores) take the colour of their surroundings")
        gains = np.ones(3)
        if not color:
            # the camera's channels have different sensitivities: scaled to the same spread of the sky, the stretch
            # shows a neutral background (the spread below the median: the objects of the field are above it)
            gains = spread[1] / np.maximum(spread, 1e-12)
            lin *= gains.astype(np.float32)[:, None, None]
            if sky.ndim == 3 and sky.shape[1] > 1:
                # the brightness of the sky kept, grey: its variations (light pollution, the Moon, vignetting) stay
                # visible, without the colour of that light in camera RGB (the green corner of heart_a7s)
                grey = np.tensordot((LUM * gains).astype(np.float32), sky, 1)
                lin += (grey - np.median(grey[shown]))[None]
                del grey
            ctx.log("no colour calibration: camera colours, each channel's sky made grey and scaled to the same "
                    f"spread (gains {', '.join(f'{g:.3f}' for g in gains)})")
        lin[:, ~valid] = 0
        ctx.progress(0.5, "statistics")
        stats = image_stats(lin, shown)
        ctx.save("lin.npy", lin.astype(np.float32))
        prep = {"M": (np.diag(gains) @ M).tolist(), "crop": [int(x0), int(x1), int(y0), int(y1)],
                "median": stats["lum"]["median"], "mad": stats["lum"]["mad"], "p99995": stats["lum"]["p99995"],
                "stats": stats, "stretch": p.stretch, "linked": p.linked, "denoised": denoised, "color": color,
                "sky": sky_name, "source": name, "gains": gains.tolist(),
                "exit": None if color else "stacking", "width": W, "height": H,
                "drizzle_scale": float(ctx.load_json("stack", "drz_meta.json").get("scale", 1.0))}
        del lin
        if denoised:
            ctx.progress(0.65, "raw stack")
            raw = np.tensordot(M.astype(np.float32), stack_image(ctx), 1)
            raw -= sky
            raw *= gains.astype(np.float32)[:, None, None]
            raw[:, ~valid] = 0
            prep["stats_raw"] = image_stats(raw, shown)
            ctx.save("lin_raw.npy", raw.astype(np.float32))
            del raw
        np.save(ctx.out("valid.npy"), valid)
        ctx.save_json("prep_simple.json", prep)
        ctx.metric("crop", [int(x0), int(x1), int(y0), int(y1)])
        ctx.metric("sky_source", prep["sky"])
        ctx.metric("colour_calibrated", color)
        ctx.metric("denoised", denoised)
        ctx.log(f"simple output {W} x {H} from {name}, useful frame x {x0}-{x1}, y {y0}-{y1}; sky: {prep['sky']}")
        ctx.progress(0.85, "preview")
        eng = SimpleRenderEngine(ctx.dir, device="cpu")
        f = max(1, int(np.ceil(max(x1 - x0, y1 - y0) / 1600)))
        img = eng.render(SimpleSettings(), 1.0 / f).final.numpy()
        ctx.preview("output", img[:, y0 // f:y1 // f, x0 // f:x1 // f])
