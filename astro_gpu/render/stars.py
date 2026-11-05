"""Rendering of the Stars block's two images (stages/star_removal.py): the starless image, the stars alone, and the two
recombined (the Stars and background block of Post).

The usual way (docs/chantiers/etoiles.md): the stars are taken off the linear image, each image is stretched on its
own, then the stars go back on top in Screen mode. Siril's star recomposition tool stretches the starless image and
the star mask apart before combining them; RC Astro advises stretching both and recombining in Screen (Add only while
linear); in Photoshop the stars layer goes over the starless one in Screen or Linear Dodge (Add). With the stars apart,
stretching them less is the star reduction: no reduction is needed on the recombined image.

* the starless image: the Output block's automatic stretch (render/simple.py) with the whole image's statistics (its
  own, without the stars' grain, would stretch it much harder: recombined, the image would not look like the Output
  block's), moved by the Sky, Contrast and Saturation sliders;
* the stars: black at 0 (their background is the starless image's), white and midtone those of the whole image, the
  midtone moved by ``star_strength``; their size by a grey erosion (smaller) or dilation (larger) of the stretched
  stars, as Photoshop's Minimum and Maximum filters; their colour by a saturation around the luminance;
* recombined: Screen 1 - (1 - a)(1 - b), Linear dodge (add) a + b (clipped), or Lighten max(a, b).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import field_validator

from ..core.params import P, check_choice
from . import ops
from .simple import Layers, SimpleRenderEngine, SimpleSettings, mtf, mtf_midtone

#: how the stars go back on the starless image (the names of Photoshop's modes)
BLENDS = ("Screen", "Linear dodge (add)", "Lighten")
#: what a rendering shows: the starless image, the stars alone, the two recombined
PARTS = ("starless", "stars", "combined")


class StarSettings(SimpleSettings):
    """The simple rendering's settings (for the starless image) and the stars' own. Their defaults are the suggested
    rendering (Post's sliders at 2)."""

    star_strength: float = P(1.0, label="Stars", ge=0.2, le=5.0, step=0.05,
                             help="How strongly the stars are stretched: 1 = the stretch of the whole image; less "
                                  "makes them fainter and smaller (the star reduction the separation allows), more "
                                  "brighter.")
    star_size: float = P(0.0, label="Star size", ge=-3.0, le=3.0, step=0.1, unit="px",
                         help="Below 0 the stars are shrunk by this radius (a grey erosion: each pixel takes the "
                              "darkest of its neighbours), above 0 grown (a grey dilation: the brightest).")
    star_saturation: float = P(1.2, label="Star color", ge=0.0, le=3.0, step=0.05,
                               help="Colour saturation of the stars around their luminance, after their stretch.")
    blend: str = P("Screen", label="Blend", choices=list(BLENDS),
                   help="How the stars go back on the starless image: Screen (the usual way), Linear dodge (add) or "
                        "Lighten.")

    @field_validator("blend")
    @classmethod
    def _blend(cls, v: str) -> str:
        return check_choice("blend", v, BLENDS)


def is_stars_folder(folder: str | Path) -> bool:
    return (Path(folder) / "prep_stars.json").exists()


class StarsRenderEngine(SimpleRenderEngine):
    """Renders a star_removal folder: ``render(settings, scale, region, part=...)`` with part starless, stars or
    combined. Same interface as ``SimpleRenderEngine`` otherwise."""

    def __init__(self, folder: str | Path, device: str = "cuda:0"):
        self.dir = Path(folder)
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.prep = json.loads((self.dir / "prep_stars.json").read_text(encoding="utf-8"))
        self.M = np.array(self.prep["M"], np.float64)
        self.SC = float(self.prep.get("drizzle_scale", 1.0))
        self.crop = self.prep["crop"]
        self._host = {k: torch.from_numpy(np.load(self.dir / f"{k}.npy")) for k in ("starless", "stars")}
        self.H, self.W = self._host["starless"].shape[-2:]
        self.valid_host = torch.from_numpy(np.load(self.dir / "valid.npy"))
        self.has_raw = False
        self._proxy = {}
        self.last_fit = None

    @torch.no_grad()
    def render(self, s: StarSettings, scale: float = 1.0, region=None, fit: dict | None = None, *,
               draft: bool = False, stop=None, part: str = "combined") -> Layers:
        from ..core.determinism import deterministic
        if part not in PARTS:
            raise ValueError(f"unknown part {part!r} (allowed: {PARTS})")
        if not isinstance(s, StarSettings):
            s = StarSettings(**s.model_dump())
        with deterministic():
            f, reg, box = self._geometry(scale, region)
            linked = bool(self.prep.get("linked", True))
            a = b = None
            if part in ("starless", "combined"):
                a = self.tone(self._layer("starless", f, box), self.prep["stats_full"], s, linked)
            if part in ("stars", "combined"):
                b = self.star_tone(self._layer("stars", f, box), s, f)
            y = a if b is None else b if a is None else blend(a, b, s.blend)
            return self._finish(y, f, reg, box, region, {"stretch": "auto", "linked": linked, "part": part})

    def star_tone(self, x: torch.Tensor, s: StarSettings, f: int) -> torch.Tensor:
        """The stars alone stretched: black at 0, the white and the midtone of the whole image (its sky at the
        Sky setting), the midtone divided by ``star_strength``; then their size and their colour."""
        st = self.prep["stats_full"]["lum"]
        c0 = st["median"] - 2.8 * st["mad"]
        hi = max(st["p99995"], c0 + 1e-6)
        xm = min(max((st["median"] - c0) / (hi - c0), 1e-6), 1 - 1e-6)
        m = min(max(mtf_midtone(xm, s.sky) / s.star_strength, 1e-6), 0.5)
        y = mtf((x / (hi - c0)).clamp(0, 1), m).clamp(0, 1)
        y = resize_stars(y, s.star_size / f)
        L = ops.luminance(y)[None]
        return (L + s.star_saturation * (y - L)).clamp(0, 1)


def _morph(y: torch.Tensor, k: int, grow: bool) -> torch.Tensor:
    """A grey dilation (grow) or erosion of radius k pixels, a (2k + 1) square."""
    if k <= 0:
        return y
    if grow:
        return F.max_pool2d(y[None], 2 * k + 1, 1, k)[0]
    return -F.max_pool2d(-y[None], 2 * k + 1, 1, k)[0]


def resize_stars(y: torch.Tensor, r: float) -> torch.Tensor:
    """The stars shrunk (r < 0) or grown (r > 0) by |r| pixels: the two whole radii around |r|, mixed by its
    fraction, so that the size moves smoothly with r."""
    if abs(r) < 1e-3:
        return y
    n, t = int(abs(r)), abs(r) - int(abs(r))
    lo = _morph(y, n, r > 0)
    return lo if t < 1e-3 else (1 - t) * lo + t * _morph(y, n + 1, r > 0)


def blend(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    """The stars ``b`` over the starless image ``a``."""
    if mode == "Linear dodge (add)":
        return (a + b).clamp(0, 1)
    if mode == "Lighten":
        return torch.maximum(a, b)
    return 1 - (1 - a) * (1 - b)
