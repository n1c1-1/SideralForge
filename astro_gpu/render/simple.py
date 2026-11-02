"""Simple rendering: the output of the workflows without Stars, and of the Raw stack exit.

Works from the outputs of the ``render_simple`` stage: the linear image in display RGB with its sky removed (lin.npy,
the denoised stack when there is one), the same from the raw stack (lin_raw.npy, with denoise), the covered pixels
(valid.npy) and the statistics of the whole image (prep_simple.json). ``render(settings, scale, region)`` returns the
``Layers`` of the engine (no star layer): a usual automatic stretch:

* the raw stack mixed back into the denoised one, or the faint areas smoothed (noise smoothing);
* a midtones transfer function from black = median - 2.8 MAD to white = the 99.995th percentile, the median sky at
  ``sky`` (linked channels: on the luminance; unlinked: each channel);
* an S-curve for the contrast, then the saturation around the luminance.

Every statistic comes from prep_simple.json, measured once on the whole image at full resolution: a reduced preview,
the full image and a region of it share one curve, so a region is the crop of the full rendering.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..core.params import ADVANCED, P, StageParams
from . import ops
from .engine import Layers

#: margin rendered around a region, in units of the smoothing radius (plus a few pixels)
REGION_PAD_SIGMAS = 4


class SimpleSettings(StageParams):
    """The settings of a simple rendering. The defaults are the suggested rendering (Post's sliders at 2)."""

    sky: float = P(0.20, label="Sky brightness", ge=0.02, le=0.6, step=0.01,
                   help="Display level of the median sky after the stretch (0 = black, 1 = white).",
                   tip="0.15-0.25 looks natural on screen.")
    contrast: float = P(0.0, label="Contrast", ge=-1.0, le=1.0, step=0.01,
                        help="S-curve after the stretch: above 0 the mid-tones are spread, below they are flattened.")
    saturation: float = P(1.2, label="Saturation", ge=0.0, le=3.0, step=0.05,
                          help="Colour saturation around the luminance, after the stretch. 1 = the colours of the "
                               "stretch.")
    noise_mix: float = P(1.0, label="Denoised share", level=ADVANCED, ge=0.0, le=1.0, step=0.05,
                         help="With Denoise: share of the denoised stack, the rest from the raw stack (finer grain, "
                              "more noise). 1 = the denoised stack alone.")
    smooth_px: float = P(0.0, label="Faint area smoothing", level=ADVANCED, ge=0.0, le=8.0, step=0.1, unit="px",
                         help="Smoothing of the faint areas (Gaussian radius); the bright areas keep full "
                              "resolution.")


def mtf(x, m: float):
    """Midtones transfer function (0 -> 0, m -> 0.5, 1 -> 1)."""
    return (m - 1) * x / ((2 * m - 1) * x - m)


def mtf_midtone(xm: float, target: float) -> float:
    """The midtone parameter that brings xm to target."""
    return xm * (1 - target) / (xm - 2 * target * xm + target)


def _smoothstep(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    t = ((x - a) / max(b - a, 1e-12)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _mix_stats(a: dict, b: dict | None, w: float) -> dict:
    """Statistics of w * a + (1 - w) * b, interpolated (the medians and spreads of the two stacks differ little)."""
    if b is None or w >= 1:
        return a
    return {k: w * a[k] + (1 - w) * b[k] for k in ("median", "mad", "p99995")}


class SimpleRenderEngine:
    """Renders a render_simple folder. Same interface as ``RenderEngine``: ``render(s, scale, region, fit)``,
    ``crop``, ``SC``, ``H``, ``W``, ``dev``; ``fit`` is accepted and ignored (the statistics are in the folder)."""

    def __init__(self, prep_dir: str | Path, device: str = "cuda:0"):
        self.dir = Path(prep_dir)
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.prep = json.loads((self.dir / "prep_simple.json").read_text(encoding="utf-8"))
        self.M = np.array(self.prep["M"], np.float64)
        self.SC = float(self.prep.get("drizzle_scale", 1.0))
        self.crop = self.prep["crop"]                                    # x0, x1, y0, y1
        self._host = {"lin": torch.from_numpy(np.load(self.dir / "lin.npy"))}
        if (self.dir / "lin_raw.npy").exists():
            self._host["raw"] = torch.from_numpy(np.load(self.dir / "lin_raw.npy"))
        self.H, self.W = self._host["lin"].shape[-2:]
        self.valid_host = torch.from_numpy(np.load(self.dir / "valid.npy"))
        self.has_raw = "raw" in self._host
        self._proxy: dict = {}
        self.last_fit = None

    @classmethod
    def from_array(cls, rgb, device: str = "cpu", linked: bool = True, sample: int = 2) -> SimpleRenderEngine:
        """An engine on a linear image in memory (a stage result Post shows as a layer): (H, W), (H, W, 3) or (3, H, W),
        any dtype. The covered pixels are the finite ones that are not 0 in every channel; the statistics of the
        stretch are measured on every ``sample``-th pixel of them (the median and the spreads move by a fraction of a
        level)."""
        a = np.asarray(rgb)
        if a.ndim == 2:
            a = a[None]
        elif a.ndim == 3 and a.shape[0] not in (1, 3) and a.shape[2] in (1, 3):
            a = np.moveaxis(a, -1, 0)
        if a.ndim != 3 or a.shape[0] not in (1, 3):
            raise ValueError(f"expected an image (H, W), (H, W, 3) or (3, H, W), got {a.shape}")
        a = np.ascontiguousarray(a, np.float32)
        if a.shape[0] == 1:
            a = np.repeat(a, 3, 0)
        finite = np.isfinite(a).all(0)
        if not finite.all():
            a = np.where(finite[None], a, 0).astype(np.float32)
        valid = finite & (a != 0).any(0)
        if not valid.any():
            valid = finite
        eng = cls.__new__(cls)
        eng.dir = None
        eng.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        H, W = a.shape[1:]
        sl = (slice(None), slice(None, None, sample), slice(None, None, sample))
        v = valid[sl[1:]]
        eng.prep = {"M": np.eye(3).tolist(), "crop": [0, W, 0, H], "drizzle_scale": 1.0, "linked": bool(linked),
                    "stats": image_stats(a[sl], v if v.any() else np.ones_like(v)), "exit": "image"}
        eng.M = np.eye(3)
        eng.SC = 1.0
        eng.crop = eng.prep["crop"]
        eng._host = {"lin": torch.from_numpy(a)}
        eng.H, eng.W = H, W
        eng.valid_host = torch.from_numpy(valid)
        eng.has_raw = False
        eng._proxy = {}
        eng.last_fit = None
        return eng

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._host.values()) + self.valid_host.numel()

    def fit_of(self, s=None, scale: float = 0.25) -> dict:
        return {"key": "", "f": max(1, int(round(1 / scale))), "stats": {}}

    # ------------------------------------------------------------------ helpers
    def _layer(self, name: str, f: int, box) -> torch.Tensor:
        """A layer on the device in float32, cut to the box (x0, x1, y0, y1) and reduced by f (box aligned on f)."""
        key = (name, f, box)
        if key in self._proxy:
            return self._proxy[key]
        x0, x1, y0, y1 = box
        t = (self.valid_host if name == "valid" else self._host[name])[..., y0:y1, x0:x1].to(self.dev)
        if name == "valid":
            t = t if f == 1 else F.avg_pool2d(t[None, None].float(), f)[0, 0] > 0.99
        else:
            t = t.float() if f == 1 else F.avg_pool2d(t.float()[None], f)[0]
        if f > 1 or box != (0, self.W, 0, self.H):      # the full resolution of the whole grid is not kept on the device
            if len(self._proxy) > 12:
                self._proxy.clear()
            self._proxy[key] = t
        return t

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def render(self, s: SimpleSettings, scale: float = 1.0, region=None, fit: dict | None = None, *,
               draft: bool = False, stop=None) -> Layers:
        """``draft`` and ``stop`` are accepted for the engine's interface: a simple rendering is quick and exact."""
        from ..core.determinism import deterministic
        with deterministic():
            return self._render(s, scale, region)

    def _render(self, s: SimpleSettings, scale: float, region) -> Layers:
        f = max(1, int(round(1 / scale)))
        Hf, Wf = self.H // f * f, self.W // f * f
        if region is None:
            x0, x1, y0, y1 = 0, Wf, 0, Hf
            box = (0, Wf, 0, Hf)
        else:
            x0, x1, y0, y1 = (int(round(float(v))) for v in region)
            x0, x1 = sorted((min(max(x0, 0), Wf), min(max(x1, 0), Wf)))
            y0, y1 = sorted((min(max(y0, 0), Hf), min(max(y1, 0), Hf)))
            x0, y0 = x0 // f * f, y0 // f * f
            x1, y1 = min(-(-x1 // f) * f, Wf), min(-(-y1 // f) * f, Hf)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"empty region {list(region)} on a {self.W} x {self.H} grid")
            pad = -(-int(REGION_PAD_SIGMAS * 8.0 + 8) // f) * f          # the widest smoothing, 8 px
            box = (max(0, x0 - pad), min(Wf, x1 + pad), max(0, y0 - pad), min(Hf, y1 + pad))
        if box == (0, Wf, 0, Hf) and f == 1:
            box = (0, self.W, 0, self.H)
        st = self.prep["stats"]
        x = self._layer("lin", f, box)
        linked = bool(self.prep.get("linked", True))
        mix = float(s.noise_mix) if self.has_raw else 1.0
        if mix < 1:
            x = mix * x + (1 - mix) * self._layer("raw", f, box)
        if s.smooth_px > 0:
            # faint areas smoothed, bright ones kept (decided on the denoised luminance, from its 40th to 85th
            # percentile)
            w = _smoothstep(ops.luminance(self._layer("lin", f, box)), st["lum_p40"], st["lum_p85"])[None]
            x = w * x + (1 - w) * ops.gaussian_blur(x, s.smooth_px / f)
        raw = self.prep.get("stats_raw")
        if linked:
            lum = _mix_stats(st["lum"], raw and raw["lum"], mix)
            y = self._stretch(x, lum, s.sky)
        else:
            y = torch.stack([self._stretch(x[c], _mix_stats(st["channels"][c], raw and raw["channels"][c], mix), s.sky)
                             for c in range(3)])
        if s.contrast != 0:
            y = (y + s.contrast * (3 * y * y - 2 * y ** 3 - y)).clamp(0, 1)
        Ly = ops.luminance(y)[None]
        y = (Ly + s.saturation * (y - Ly)).clamp(0, 1)
        y[:, ~self._layer("valid", f, box)] = 0
        bx0, _, by0, _ = box
        y = y[:, (y0 - by0) // f:(y1 - by0) // f, (x0 - bx0) // f:(x1 - bx0) // f]
        info = {"stretch": "auto", "linked": linked, "noise_mix": mix}
        if region is not None:
            info["region"] = [x0, x1, y0, y1]
        return Layers(y, y, None, 1.0 / f, info)

    @staticmethod
    def _stretch(x: torch.Tensor, st: dict, target: float) -> torch.Tensor:
        """Linked or one channel: black at median - 2.8 MAD, white at the 99.995th percentile, the median at target."""
        c0 = st["median"] - 2.8 * st["mad"]
        hi = max(st["p99995"], c0 + 1e-6)
        xm = min(max((st["median"] - c0) / (hi - c0), 1e-6), 1 - 1e-6)
        m = mtf_midtone(xm, target)
        return mtf(((x - c0) / (hi - c0)).clamp(0, 1), m).clamp(0, 1)


def image_stats(lin: np.ndarray, shown: np.ndarray) -> dict:
    """The statistics a simple rendering needs, on the shown pixels of a linear (3, H, W) image: of the luminance
    and of each channel (median, MAD, 99.995th percentile), and the luminance's 40th and 85th percentiles."""
    def one(v: np.ndarray) -> dict:
        med = float(np.median(v))
        return {"median": med, "mad": 1.4826 * float(np.median(np.abs(v - med))),
                "p99995": float(np.percentile(v, 99.995))}
    L = (0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])[shown]
    out = {"lum": one(L), "channels": [one(lin[c][shown]) for c in range(3)]}
    out["lum_p40"], out["lum_p85"] = (float(v) for v in np.percentile(L, [40, 85]))
    return out
