"""Simple output preparation: the linear image a simple rendering stretches (render/simple.py).

The output of the workflows without Stars (Classic stack, Classic + denoise) and of the Raw stack exit:

* lin.npy = M x stack - sky, in display RGB: the denoised stack when the workflow has Denoise, else the stack; the sky
  of the Gradient block (background_poly) when there is one;
* without Color (the Raw stack exit), M is the identity and the sky the median of each channel; the channels are then
  scaled to the same sky spread, so that the camera's colour balance does not tint the preview;
* lin_raw.npy (with Denoise): the same from the raw stack, for the noise slider of Post;
* valid.npy, the covered pixels, and prep_simple.json: M, the crop, the statistics of the stretch (median, MAD and
  99.995th percentile of the luminance and of each channel), the settings.
"""
from __future__ import annotations

import numpy as np
from pydantic import field_validator

from ..core.params import P, StageParams
from ..core.stage import Context, Stage
from .background_poly import coverage, has_input, sky_of, stack_source
from .render_prep import crop_box

STRETCHES = ["auto"]                       # the other stretches of the spec (GHS, arcsinh, STF, none) are not in v1


def sky_spread(v: np.ndarray) -> float:
    """Robust spread of the sky in a channel: 1.4826 x the median distance below the median (the stars and nebulae
    of the field, above the median, do not widen it)."""
    med = float(np.median(v))
    below = med - v[v < med]
    return 1.4826 * float(np.median(below)) if below.size else 0.0


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
    version = "1"

    def run(self, ctx: Context, p: RenderSimpleParams) -> None:
        from ..render.simple import SimpleRenderEngine, SimpleSettings, image_stats
        ctx.progress(0.05, "loading the stack")
        color, denoised = has_input(ctx, "color"), has_input(ctx, "denoise")
        poly = has_input(ctx, "background_poly")
        src, name = stack_source(ctx)
        H, W = src.shape[1:]
        valid = coverage(ctx)
        y0, y1, x0, x1 = crop_box(valid, 0.995)
        shown = np.zeros_like(valid)
        shown[y0:y1, x0:x1] = True
        shown &= valid
        M = ctx.load("color", "M.npy").astype(np.float64) if color else np.eye(3)
        lin = np.tensordot(M.astype(np.float32), src, 1)
        del src
        ctx.progress(0.2, "sky")
        if poly:
            sky = sky_of(ctx, H, W)
        else:                                       # no Gradient block: the median of each channel
            sky = np.array([np.median(lin[c][shown]) for c in range(3)], np.float32)[:, None, None]
        lin -= sky
        gains = np.ones(3)
        if not color:
            # the camera's channels have different sensitivities: scaled to the same spread of the sky, the stretch
            # shows a neutral background (the spread below the median: the objects of the field are above it)
            spread = np.array([sky_spread(lin[c][shown]) for c in range(3)])
            gains = spread[1] / np.maximum(spread, 1e-12)
            lin *= gains.astype(np.float32)[:, None, None]
            ctx.log("no colour calibration: camera colours, each channel's sky set to zero and scaled to the same "
                    f"spread (gains {', '.join(f'{g:.3f}' for g in gains)})")
        lin[:, ~valid] = 0
        ctx.progress(0.5, "statistics")
        stats = image_stats(lin, shown)
        ctx.save("lin.npy", lin.astype(np.float32))
        prep = {"M": (np.diag(gains) @ M).tolist(), "crop": [int(x0), int(x1), int(y0), int(y1)],
                "median": stats["lum"]["median"], "mad": stats["lum"]["mad"], "p99995": stats["lum"]["p99995"],
                "stats": stats, "stretch": p.stretch, "linked": p.linked, "denoised": denoised, "color": color,
                "sky": "background_poly" if poly else "channel medians", "source": name, "gains": gains.tolist(),
                "exit": None if color else "stacking", "width": W, "height": H,
                "drizzle_scale": float(ctx.load_json("stack", "drz_meta.json").get("scale", 1.0))}
        del lin
        if denoised:
            ctx.progress(0.65, "raw stack")
            raw = np.tensordot(M.astype(np.float32), ctx.load("stack", "drz_full.npy").astype(np.float32), 1)
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
