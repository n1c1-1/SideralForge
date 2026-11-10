"""The Deconvolution block: the star shape measured on the stack, per zone and colour, and the stack and its two
half-stacks sharpened with it (stages/deconv_lib.py). Optional, before Denoise, on linear data.

Methods (the workflow's ``deconvolution`` option, ``method`` here): ``rl`` Richardson-Lucy, regularised; ``zones`` one
linear filter per zone. The half-stacks are treated like the stack, so that a Denoise placed after it still reads two
images whose difference is the noise.
"""
from __future__ import annotations

import numpy as np
from pydantic import field_validator

from ..core import devices
from ..core.params import ADVANCED, P, StageParams, check_choice
from ..core.stage import Context, Stage
from . import deconv_lib as lib
from .denoise import before_after
from .gradient import DECONV_FILE, image_in
from .stack import SATURATION

METHODS = ("rl", "zones")


class DeconvParams(StageParams):
    method: str = P("rl", label="Method", choices=list(METHODS),
                    help="Richardson-Lucy: iterations with the star shape measured across the field. By zones: one "
                         "sharpening filter per zone, faster.")
    iterations: int = P(30, label="Iterations", ge=1, le=100,
                        help="More iterations sharpen more, and raise the noise.")
    regularization: float = P(0.002, label="Regularization", ge=0.0, le=0.05, step=0.001,
                              help="Holds back noise and ringing.")
    shrink: float = P(0.15, label="Star reduction", ge=0.01, le=0.5, step=0.01,
                      help="How much narrower the stars become: a share of their measured width.")
    tiles: int = P(6, label="Zones", level=ADVANCED, ge=1, le=16,
                   help="Zones along the long side of the image, each with its own measured star shape.")

    @field_validator("method")
    @classmethod
    def _method(cls, v: str) -> str:
        return check_choice("method", v, METHODS)


class Deconv(Stage):
    name = "deconv"
    title = "Deconvolution"
    description = ("Sharpens the stack and its half-stacks with the star shape measured on the stack itself, per zone "
                   "of the field and per colour.")
    deps = ("stack",)
    Params = DeconvParams
    version = "1"

    def run(self, ctx: Context, p: DeconvParams) -> None:
        full, A, B = (image_in(ctx, h).img for h in ("full", "A", "B"))
        weight = ctx.load("stack", "drz_weight.npy").astype(np.float32)
        valid = (weight if weight.ndim == 2 else weight.min(0)) > 0
        try:
            s = ctx.load("stack", SATURATION)
            sat = (s > 0).any(0) if s.ndim == 3 else s > 0
        except (FileNotFoundError, OSError):
            sat = None
        dev = ctx.device
        free = devices.free_memory(dev)
        device = dev if free is None or (full.nbytes + A.nbytes + B.nbytes) * 3 < free else "cpu"

        def tick(share: float) -> None:
            ctx.check_stop()
            ctx.progress(0.1 + 0.8 * share, "deconvolving")

        st = lib.StackDeconvSettings(method=p.method, iterations=p.iterations, regularization=p.regularization,
                                     shrink=p.shrink, tiles_long=p.tiles, log=ctx.log, tick=tick)
        ctx.progress(0.02, "measuring the star shape")
        of, oa, ob, info = lib.run(full, A, B, sat, valid, st, device)
        if info["applied"]:
            before, after = float(np.median(info["fwhm"])), info["fwhm_after"]
            ctx.metric("star_width_before", round(before, 2))
            if after is not None:
                ctx.metric("star_width_after", round(float(np.median(after)), 2))
            ctx.metric("zones", int(info["fwhm"].shape[1] * info["fwhm"].shape[2]))
        else:
            ctx.log("too few isolated stars to measure the star shape: the image is left as it is", "warning")
        for half, a in (("full", of), ("A", oa), ("B", ob)):
            ctx.save(DECONV_FILE[half], np.asarray(a, np.float32))
        ctx.preview("before_after", before_after(full, of))
