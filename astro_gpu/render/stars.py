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
  midtone moved by ``star_strength``; the small stars faded out (``star_small``: the star reduction of photo editors'
  tutorials, the stars selected and the small ones removed, then Minimum; here the stars' support opened by
  reconstruction, L. Vincent 1993, so that the wider stars stay exactly as they were; on the stars alone, so the
  nebula's texture is never touched); their size by a round grey erosion (smaller) or dilation (larger), as Photoshop's
  Minimum and Maximum filters with Preserve roundness (an octagon: a cross and a square in turn, the usual
  approximation of a disc), never a square; smaller through a star mask (``star_mask``: the stars standing above
  their local background's noise), as those tutorials select the stars first, so that the grain between the stars of
  a dense field is not eroded into filaments; their colour by a saturation around the luminance;
* recombined: Screen 1 - (1 - a)(1 - b), Linear dodge (add) a + b (clipped), or Lighten max(a, b).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import field_validator

from ..core.params import P, check_choice
from . import ops
from .simple import Layers, SimpleRenderEngine, SimpleSettings, continuum_ratio, emission, mtf, mtf_midtone

#: how the stars go back on the starless image (the names of Photoshop's modes)
BLENDS = ("Screen", "Linear dodge (add)", "Lighten")
#: what a rendering shows: the starless image, the stars alone, the two recombined
PARTS = ("starless", "stars", "combined")


class StarSettings(SimpleSettings):
    """The simple rendering's settings (for the starless image) and the stars' own. Their defaults are the suggested
    rendering (Post's sliders at 2)."""

    star_strength: float = P(0.6, label="Stars", ge=0.2, le=5.0, step=0.05,
                             help="How strongly the stars are stretched: 1 = the stretch of the whole image; less "
                                  "makes them fainter and smaller (the star reduction the separation allows), more "
                                  "brighter. 0.6 by default, with a size of -1 (the user's choice on the Rosette, "
                                  "checked on Heart, Iris, M8 and M31, 5 November: the nebula reads first).")
    star_size: float = P(-1.0, label="Star size", ge=-3.0, le=3.0, step=0.1, unit="px",
                         help="Below 0 the stars are shrunk by this radius (a grey erosion: each pixel takes the "
                              "darkest of its neighbours), above 0 grown (a grey dilation: the brightest).")
    star_small: float = P(0.0, label="Small stars", ge=0.0, le=2.0, step=0.05,
                          help="The small stars taken away: those narrower than this many times the image's typical "
                               "star (its median width at half height); 0 = all kept (the star reduction of photo "
                               "editors' tutorials: the stars selected, the small ones removed). A crowded field "
                               "clears; the larger stars stay exactly as they are.")
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
                x = self._layer("starless", f, box)
                if s.halpha:
                    x = emission(x, self.continuum(), float(s.halpha), f)
                a = self.tone(x, self.prep["stats_full"], s, linked)
            if part in ("stars", "combined"):
                b = self.star_tone(self._layer("stars", f, box), s, f)
            y = a if b is None else b if a is None else blend(a, b, s.blend)
            return self._finish(y, f, reg, box, region, {"stretch": "auto", "linked": linked, "part": part})

    def star_fwhm(self) -> float:
        """The image's typical star width at half height (full-resolution pixels), measured once on the stars alone
        (``measure_fwhm``)."""
        if getattr(self, "_fwhm", None) is None:
            self._fwhm = measure_fwhm(self._host["stars"])
        return self._fwhm

    def continuum(self) -> float:
        """The continuum's red-to-green ratio (``continuum_ratio``), measured once on the stars alone."""
        if getattr(self, "_alpha", None) is None:
            self._alpha = continuum_ratio(np.maximum(self._host["stars"].numpy(), 0))
        return self._alpha

    def star_tone(self, x: torch.Tensor, s: StarSettings, f: int) -> torch.Tensor:
        """The stars alone stretched: black at 0, the white and the midtone of the whole image (its sky at the
        Sky setting), the midtone divided by ``star_strength``; then their size and their colour."""
        st = self.prep["stats_full"]["lum"]
        if self.prep["stats_full"].get("stretched"):    # an image already stretched (issue #43): its stars as they are
            y = (x * float(s.star_strength)).clamp(0, 1)
        else:
            c0 = st["median"] - 2.8 * st["mad"]
            hi = max(st["p99995"], c0 + 1e-6)
            xm = min(max((st["median"] - c0) / (hi - c0), 1e-6), 1 - 1e-6)
            m = min(max(mtf_midtone(xm, s.sky) / s.star_strength, 1e-6), 0.5)
            y = mtf((x / (hi - c0)).clamp(0, 1), m).clamp(0, 1)
        v = float(s.star_small)
        if v > 0:                                  # a gentle start: below SMALL_START they fade, above they go
            y = fade_small_stars(y, min(1.0, v / SMALL_START), max(v, SMALL_START) * self.star_fwhm() / f)
        y = resize_stars(y, s.star_size / f)
        L = ops.luminance(y)[None]
        return (L + s.star_saturation * (y - L)).clamp(0, 1)


#: Small stars: below this setting the small stars fade (by its share), above it they go and the size grows
SMALL_START = 0.4
#: a typical star width when the stars cannot be measured (full-resolution pixels)
DEFAULT_FWHM = 3.0
#: the second moment per axis of a Gaussian kept above half its peak, in units of its sigma squared
CLIPPED_MOMENT = 2 * (1 - (1 + math.log(2)) / 2)


def measure_fwhm(stars: torch.Tensor, n: int = 300) -> float:
    """The median width at half height of the stars of a stars-alone image (3, H, W, linear): the local maxima of its
    luminance, the middle of them by brightness (neither the noise nor the saturated cores), each measured by the
    second moments of its 9 x 9 neighbourhood above half its peak (FWHM = 2.355 sigma)."""
    L = ops.luminance(stars.float().clamp(min=0))
    peaks = (L == F.max_pool2d(L[None, None], 7, 1, 3)[0, 0]) & (L > 0)
    ys, xs = torch.nonzero(peaks, as_tuple=True)
    keep = (ys >= 4) & (xs >= 4) & (ys < L.shape[0] - 4) & (xs < L.shape[1] - 4)
    ys, xs = ys[keep], xs[keep]
    if ys.numel() < 20:
        return DEFAULT_FWHM
    v = L[ys, xs]
    order = torch.argsort(v)
    mid = order[int(0.6 * len(order)):int(0.95 * len(order))]
    if mid.numel() > n:
        mid = mid[torch.linspace(0, mid.numel() - 1, n).long()]
    yy, xx = torch.meshgrid(torch.arange(-4, 5.0), torch.arange(-4, 5.0), indexing="ij")
    widths = []
    for i in mid.tolist():
        y, x = int(ys[i]), int(xs[i])
        w = L[y - 4:y + 5, x - 4:x + 5]
        w = (w - w.min()).clamp(min=0)
        w = torch.where(w >= 0.5 * w.max(), w, torch.zeros_like(w))
        tot = float(w.sum())
        if tot <= 0:
            continue
        cy, cx = float((w * yy).sum()) / tot, float((w * xx).sum()) / tot
        var = float((w * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum()) / tot / 2
        # above half the peak only: a Gaussian's weighted second moment per axis there is 2 (1 - (1 + ln 2) / 2)
        # sigma^2 = 0.307 sigma^2 (its integral from 0 to the half-height radius)
        widths.append(2.355 * (var / CLIPPED_MOMENT) ** 0.5)
    return float(np.median(widths)) if len(widths) >= 10 else DEFAULT_FWHM


def _step(y: torch.Tensor, grow: bool, cross: bool) -> torch.Tensor:
    """One pixel of grey dilation (grow) or erosion: by a cross (the 4 neighbours) or a 3 x 3 square."""
    z = y if grow else -y
    if cross:
        p = F.pad(z[None], (1, 1, 1, 1), mode="replicate")[0]
        z = torch.stack([z, p[..., :-2, 1:-1], p[..., 2:, 1:-1], p[..., 1:-1, :-2], p[..., 1:-1, 2:]]).amax(0)
    else:
        z = F.max_pool2d(F.pad(z[None], (1, 1, 1, 1), mode="replicate"), 3, 1, 0)[0]
    return z if grow else -z


def _morph(y: torch.Tensor, k: int, grow: bool) -> torch.Tensor:
    """A grey dilation (grow) or erosion of radius k pixels, round: k steps alternating a cross and a square (an
    octagon, the usual approximation of a disc; a square would make the stars square)."""
    for i in range(max(0, k)):
        y = _step(y, grow, cross=(i % 2 == 0))
    return y


#: the geodesic dilations of the reconstruction, per pixel of radius of the largest star kept whole (full resolution)
RECONSTRUCT_STEPS = 48


def reconstruct(marker: torch.Tensor, mask: torch.Tensor, steps: int) -> torch.Tensor:
    """Grey reconstruction by dilation (L. Vincent, "Morphological grayscale reconstruction in image analysis",
    IEEE Trans. Image Processing 2, 1993): the marker dilated one pixel at a time, never above the mask, until it
    settles (or ``steps``)."""
    m = torch.minimum(marker, mask)
    for _ in range(max(1, steps)):
        n = torch.minimum(_step(m, True, cross=False), mask)
        if torch.equal(n, m):
            break
        m = n
    return m


#: the level of the stretched stars above which a pixel belongs to a star (their support)
STAR_LEVEL = 0.04


def large_stars(y: torch.Tensor, r: float) -> torch.Tensor:
    """Where the stars wider than r pixels are, whole (h, w in 0..1): their support (the luminance above STAR_LEVEL)
    opened by reconstruction (L. Vincent 1993): the support eroded by r (round) keeps a seed of each wide star only,
    then the seeds grow back within the support to the whole star; the narrower stars leave no seed. Grown by the
    radius again (the wings below the support's level) and its edge softened by a pixel."""
    sup = (ops.luminance(y) > STAR_LEVEL).float()[None]
    n = max(1, int(round(r)))
    seeds = _morph(sup, n, False)
    whole = reconstruct(seeds, sup, int(RECONSTRUCT_STEPS * max(r / 2.0, 1.0)))
    whole = _morph(whole, n + 1, True)                 # their faint wings, below the support's level, kept too
    return ops.gaussian_blur(whole, 0.7)[0].clamp(0, 1)


def fade_small_stars(y: torch.Tensor, k: float, r: float) -> torch.Tensor:
    """The stars narrower than r pixels taken away by k (0 kept .. 1 gone), the wider ones left exactly as they were
    (``large_stars``): the star reduction of photo editors' tutorials (the stars selected, the small ones removed),
    on the stars alone."""
    if k <= 0 or r < 0.5:
        return y
    keep = large_stars(y, r)[None]
    return y * (keep + (1 - keep) * (1 - k))


def resize_stars(y: torch.Tensor, r: float) -> torch.Tensor:
    """The stars shrunk (r < 0) or grown (r > 0) by |r| pixels: the two whole radii around |r|, mixed by its
    fraction, so that the size moves smoothly with r."""
    if abs(r) < 1e-3:
        return y
    n, t = int(abs(r)), abs(r) - int(abs(r))
    lo = _morph(y, n, r > 0)
    out = lo if t < 1e-3 else (1 - t) * lo + t * _morph(y, n + 1, r > 0)
    if r > 0:
        return out
    # shrunk through a star mask, as the published star reductions do (Photoshop: the Minimum filter through a
    # selection of the stars; PixInsight: MorphologicalTransformation through a star mask): an erosion of the faint
    # background between the stars draws a web of filaments on a dense field (issue #18)
    return y + (out - y) * star_mask(y, n + 1)


#: a star of the mask stands this many local noise units above its local background (the usual detection threshold)
MASK_SIGMAS = 3.0
#: the scale of the local background and noise (pixels of the grid rendered): wider than the stars, so that a star's
#: own light does not raise the background it is measured against (issue #30: its wings left the mask, a ring)
MASK_SCALE = 24.0
#: the clipping passes that take the stars out of that background (2 noise units, as the background estimators do)
MASK_PASSES, MASK_CLIP = 3, 2.0
#: the softened edge of the mask (pixels of the grid rendered)
MASK_FEATHER = 1.5


def star_mask(y: torch.Tensor, r: int) -> torch.Tensor:
    """Where the stars are (1, h, w in 0..1): the luminance above STAR_LEVEL and MASK_SIGMAS noise units above the
    local sky, the sky and its noise measured at MASK_SCALE with the stars clipped out (the iterative clipping of the
    astronomical background estimators, Bertin and Arnouts 1996), so that a star's wings stand above it as its core
    does (issue #30); on a dense field the faint grain between the stars stays under it (issue #18). Grown by r pixels
    (round) to hold the wings the reduction takes, its edge softened over MASK_FEATHER pixels."""
    L = ops.luminance(y)[None]
    bg = ops.gaussian_blur(L, MASK_SCALE)
    noise = 1.2533 * ops.gaussian_blur((L - bg).abs(), MASK_SCALE)          # the mean deviation, as a sigma
    for _ in range(MASK_PASSES):                                            # the stars out of the sky and its noise
        lo = torch.minimum(L, bg + MASK_CLIP * noise)
        bg = ops.gaussian_blur(lo, MASK_SCALE)
        noise = 1.2533 * ops.gaussian_blur((lo - bg).abs(), MASK_SCALE)
    sup = ((L > STAR_LEVEL) & (L > bg + MASK_SIGMAS * noise)).float()
    grown = _morph(sup, r, True)
    return torch.maximum(ops.gaussian_blur(grown, MASK_FEATHER), grown)[0].clamp(0, 1)     # whole on the stars


def blend(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    """The stars ``b`` over the starless image ``a``."""
    if mode == "Linear dodge (add)":
        return (a + b).clamp(0, 1)
    if mode == "Lighten":
        return torch.maximum(a, b)
    return 1 - (1 - a) * (1 - b)
