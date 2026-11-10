"""The Deconvolution block: the star shape measured on the stack, per zone and colour, and the stack and its two
half-stacks sharpened with it (stages/deconv_lib.py). Optional, before Denoise, on linear data.

Its method (the workflow's ``deconvolution`` option, ``method`` here): ``rl`` Richardson-Lucy with the handling of
saturated pixels of Whyte et al. 2014, regularised. The half-stacks are treated like the stack, so that a Denoise placed after it still reads two
images whose difference is the noise.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from pydantic import field_validator

from ..core import devices
from ..core.params import ADVANCED, P, StageParams, check_choice
from ..core.stage import Context, Stage, StageError
from . import deconv_lib as lib
from .denoise import before_after
from .gradient import DECONV_FILE, image_in
from .stack import SATURATION

METHODS = ("rl", "neural")
#: the network of the Neural method, in beta: not bundled yet, read from where it was trained (tools/bench/deconv_train.py)
BETA_WEIGHTS = Path("E:/float_train/run1/net_021818.pt")
BETA_TILE, BETA_PAD = 512, 48


def _ramp(n: int, before: bool, after: bool) -> np.ndarray:
    """The weights of a tile along one axis: 1 inside, a cosine down to 0 over the overlap on the sides with a neighbour."""
    w = np.ones(n, np.float32)
    k = 2 * BETA_PAD
    fade = 0.5 - 0.5 * np.cos(np.pi * (np.arange(k, dtype=np.float32) + 0.5) / k)
    if before:
        w[:k] = fade
    if after:
        w[-k:] = fade[::-1]
    return w


class DeconvParams(StageParams):
    method: str = P("rl", label="Method", choices=list(METHODS),
                    help="Richardson-Lucy: iterations with the star shape measured across the field. Neural "
                         "network: beta, it also removes the noise.")
    iterations: int = P(30, label="Iterations", ge=1, le=100,
                        help="More iterations sharpen more, and raise the noise.")
    regularization: float = P(0.002, label="Regularization", ge=0.0, le=0.05, step=0.001,
                              help="Holds back noise and ringing.")
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
    version = "4"     # 4: damping, stop, local deringing, protection (#92); the network's tiles blended

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

        if p.method == "neural":
            self.neural(ctx, full, A, B, valid, sat, device)
            return
        st = lib.StackDeconvSettings(iterations=p.iterations, regularization=p.regularization, tiles_long=p.tiles,
                                     log=ctx.log, tick=tick)
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

    def neural(self, ctx: Context, full, A, B, valid, sat, device) -> None:
        """The network of the Neural method (beta), colour by colour, the stack and its half-stacks; clipped cores kept."""
        from .star_removal import NAFNet, fwd, inv
        if not BETA_WEIGHTS.is_file():
            raise StageError(f"the neural deconvolution's network is not here: {BETA_WEIGHTS}")
        net = NAFNet(recompute=False)
        w = net.intro.out_channels
        net.intro, net.ending = nn.Conv2d(1, w, 3, padding=1), nn.Conv2d(w, 1, 3, padding=1)
        net.load_state_dict(torch.load(BETA_WEIGHTS, map_location="cpu", weights_only=False)["net"])
        dev = torch.device(device)
        net = net.to(dev).eval()

        @torch.no_grad()
        def plane(x: np.ndarray) -> np.ndarray:
            H, W = x.shape
            out, total = np.zeros_like(x), np.zeros_like(x)
            for y in range(0, H, BETA_TILE):
                for x0 in range(0, W, BETA_TILE):
                    y1, x1 = max(0, y - BETA_PAD), max(0, x0 - BETA_PAD)
                    y2, x2 = min(H, y + BETA_TILE + BETA_PAD), min(W, x0 + BETA_TILE + BETA_PAD)
                    t = torch.from_numpy(np.ascontiguousarray(x[y1:y2, x1:x2]))[None, None].to(dev)
                    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        o = inv(net(fwd(t)).float())[0, 0].cpu().numpy()
                    # a tile fades out over the overlap with the next one: its edges leave no seam
                    wy = _ramp(y2 - y1, y1 > 0, y2 < H)
                    wx = _ramp(x2 - x1, x1 > 0, x2 < W)
                    w = (wy[:, None] * wx[None, :]).astype(np.float32)
                    out[y1:y2, x1:x2] += o * w
                    total[y1:y2, x1:x2] += w
            return out / np.maximum(total, 1e-6)

        keep = None
        if sat is not None and sat.any():
            core = cv2.dilate(sat.astype(np.uint8), np.ones((5, 5))).astype(np.float32)
            keep = np.clip(cv2.GaussianBlur(core, (0, 0), 1.5) * 2, 0, 1)
        res = [np.zeros_like(full), np.zeros_like(A), np.zeros_like(B)]
        for c in range(3):
            ctx.check_stop()
            ctx.progress(0.05 + 0.3 * c, "the network")
            d = 0.5 * (A[c] - B[c])[valid]
            s = max(1.4826 * float(np.median(np.abs(d - np.median(d)))), 1e-9)
            bk = float(np.median(full[c][valid]))
            for k, im in enumerate((full, A, B)):
                y = plane((im[c] - bk) / s) * s + bk
                y = np.where(valid, y, im[c])
                res[k][c] = y if keep is None else im[c] + (1 - keep) * (y - im[c])
        for half, a in (("full", res[0]), ("A", res[1]), ("B", res[2])):
            ctx.save(DECONV_FILE[half], a.astype(np.float32))
        ctx.preview("before_after", before_after(full, res[0]))
