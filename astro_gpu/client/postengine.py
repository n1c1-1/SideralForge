"""The Post of the client, without its widgets: layers, masks, filters and the composition.

Ported from the mockup's ``postengine.py``. The images come from a ``Provider`` (the engine's layers, see
``services/render.py``), the composition is numpy and runs in the render worker, never in the GUI thread.

Layers (a list, top first, as in Photoshop; saved as it is in ``project.post["layers"]``):

* a workflow layer ``{id, kind: "workflow", pid, part, name, visible, opacity, mode, base, sliders, mask}``: the image of
  a workflow's output, rendered by the engine with its ``base`` (a named rendering of the project, a preset, or None
  for the suggestion) moved by its simplified ``sliders`` (positions 0-4, 2 = the base); ``part`` is None, "stars" or
  "starless";
* an image layer ``{id, kind: "image", key, name, visible, opacity, mode, sliders, linked, mask}``: the linear image of
  a stage result (a stack, a gradient-removed or a denoised image) sent from Inspect, stretched automatically and moved
  by the Sky, Contrast and Saturation sliders; ``key`` names the stage result (the pixels are never saved);
* a filter layer ``{id, kind: "filter", filter, name, visible, opacity, mode, params, mask, target}``: one of the
  filters (``FILTERS``: each a published or documented method, named in its ``source``; postfilters/ holds the new
  ones), applied to the one pixel layer (workflow or image) ``target`` names, or to everything below (``target`` is
  None).

The model is Affinity Photo's: a filter that applies to one layer is that layer's child; a filter that applies to
everything below is a layer of its own. In the list a layer's children sit just above it, top child first; the panel
shows each pixel layer with its children indented under it (``rows``), and reading the panel from the bottom up is the
order of the composition: a pixel layer, then its children from the lowest to the highest, then the next row up.

Every layer may have a ``mask = {parts, feather, density, invert}`` of parts (stars, H-alpha, luminance, color, gradient,
brush), each added, subtracted or intersected. Pixel sizes are full-resolution pixels: at a reduced resolution they are
divided by the reduction factor, so the draft looks like the full image.

Resolutions: level 0 is the full canvas (only a region of it, with a 64 px margin for the filters), level 1 the half
(the screen), level 2 the quarter (the draft while something moves), level 3 the eighth (a draft on a slow machine).
The canvas is the area every workflow layer covers, taken on multiples of 8 px so that the levels line up.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np

from .postfilters import colour as pf_colour
from .postfilters import curves as pf_curves
from .postfilters import detail as pf_detail
from .postfilters import tone as pf_tone

MODES = ["Normal", "Lighten", "Screen", "Multiply", "Soft light"]       # plain English: _() where shown
OPS = {"add": "Add", "sub": "Subtract", "inter": "Intersect"}
ALIGN = 8                                  # the canvas starts and ends on a multiple of this (the deepest level)
MAX_LEVEL = 3
REGION_MARGIN = 64                         # full-resolution pixels around a region, so that the filters see neighbours
THUMB = (54, 36)                           # the layer and mask thumbnails (width, height)
_ids = itertools.count(1)


def new_id() -> int:
    return next(_ids)


def reseed_ids(layers: list[dict]) -> None:
    """Make the next new id greater than every id of these layers (after loading a saved composition)."""
    global _ids
    top = 0
    for lay in layers:
        top = max(top, int(lay.get("id", 0)))
        for p in (lay.get("mask") or {}).get("parts", []):
            top = max(top, int(p.get("id", 0)))
    _ids = itertools.count(max(top, next(_ids) - 1) + 1)


class Superseded(Exception):
    """A newer request replaced the one being computed: leave at once."""


# ======================================================================== small image helpers
_LUM = np.array([[0.2126, 0.7152, 0.0722]], np.float32)


def lum(x: np.ndarray) -> np.ndarray:
    """The luminance of an RGB image (h, w, 3) -> (h, w), float32."""
    if x.dtype == np.float32 and x.ndim == 3 and x.flags.c_contiguous:
        return cv2.transform(x, _LUM)
    return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]).astype(np.float32)


def _gray3(a: np.ndarray) -> np.ndarray:
    """A (h, w) image as (h, w, 3), the same value in each channel."""
    return cv2.merge([a, a, a])


def _mix(a: np.ndarray, wa: float, b: np.ndarray, wb: float) -> np.ndarray:
    """wa * a + wb * b in one pass (two float32 images of the same shape)."""
    return cv2.addWeighted(_f32(a), float(wa), _f32(b), float(wb), 0.0)


def _torch(x: np.ndarray):
    import torch
    return torch.from_numpy(_f32(x))


def hue(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hue in degrees and saturation (0..1) of an RGB image."""
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn + 1e-6
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, d / (mx + 1e-6)


def hue_weight(h, center, width):
    dist = np.abs((h - center + 180) % 360 - 180)
    return np.clip(1 - dist / max(width, 1), 0, 1)


def _f32(x: np.ndarray) -> np.ndarray:
    return x if x.dtype == np.float32 and x.flags.c_contiguous else np.ascontiguousarray(x, np.float32)


def gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur of an (h, w) or (h, w, c) float image, reflecting at the borders. A wide blur is made on a
    reduced copy (a Gaussian is smooth: the result is the same within a fraction of a level)."""
    if sigma < 0.3:
        return x
    x = _f32(x)
    one = x.ndim == 3 and x.shape[2] == 1
    a = x[..., 0] if one else x
    if sigma > 12:
        m = int(sigma // 6)
        h, w = a.shape[:2]
        small = cv2.resize(a, (max(1, w // m), max(1, h // m)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / m, borderType=cv2.BORDER_REFLECT)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        out = cv2.GaussianBlur(a, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return out[..., None] if one else out


def _box(size: int) -> np.ndarray:
    return np.ones((int(size), int(size)), np.uint8)


def erode(x: np.ndarray, size: int) -> np.ndarray:
    """A grey erosion with a flat square (each channel apart)."""
    return cv2.erode(_f32(x), _box(size), borderType=cv2.BORDER_REPLICATE)


def dilate(x: np.ndarray, size: int) -> np.ndarray:
    return cv2.dilate(_f32(x), _box(size), borderType=cv2.BORDER_REPLICATE)


# ======================================================================== filters
HUES = [("All", None), ("Reds", 0), ("Yellows", 60), ("Greens", 120), ("Cyans", 180), ("Blues", 230),
        ("Magentas", 300)]


def P(key, label, lo, hi, default, unit="", step=None):
    return dict(key=key, label=label, type="float", lo=lo, hi=hi, default=default, unit=unit, step=step)


def C(key, label, choices, default):
    return dict(key=key, label=label, type="choice", choices=choices, default=default)


def B(key, label, default):
    return dict(key=key, label=label, type="bool", default=bool(default))


def CURVE(key):
    """The curves of the Curves tool: {channel: {"pts", "interp"}}; {} is the identity (postfilters.curves)."""
    return dict(key=key, label="Curves", type="curve", default={})


def f_stretch(x, p, ctx):
    """An arcsinh stretch around the symmetry point (the transcendental on all cores, through torch)."""
    d, sp = p["strength"], p["symmetry"]
    if d < 1e-3:                                       # the default: no stretch at all
        return x
    a0 = float(np.arcsinh(-d * sp))                    # python floats: a numpy scalar would promote the image to float64
    a1 = float(np.arcsinh(d * (1 - sp)))
    t = (_torch(x) - sp).mul_(d).asinh_().sub_(a0).mul_(1.0 / (a1 - a0))
    return t.numpy()


def f_curves(x, p, ctx):
    """The Curves tool (postfilters.curves): a table while a point moves, exact on release."""
    return pf_curves.apply(x, p["curves"], draft=ctx.draft)


def curves_of_old(sh: float, mid: float, hi: float) -> dict:
    """The curves of a composition saved before the Curves tool (three sliders moving the curve at .25, .5 and .75):
    the same piecewise-linear curve, so that it renders as it did."""
    if not (sh or mid or hi):
        return {}
    ys = np.clip([0, 0.25 + sh, 0.5 + mid, 0.75 + hi, 1], 0, 1)
    return {"RGB/K": {"pts": [[x, float(y)] for x, y in zip((0.0, 0.25, 0.5, 0.75, 1.0), ys)], "interp": "Linear"}}


def f_levels(x, p, ctx):
    if p["black"] <= 0 and p["white"] >= 1 and abs(p["gamma"] - 1) <= 1e-9:      # the default: the identity
        return x
    t = (_torch(x) - p["black"]).mul_(1.0 / max(p["white"] - p["black"], 1e-3)).clamp_(0, 1)
    if abs(p["gamma"] - 1) > 1e-9:
        t = t.pow_(1.0 / p["gamma"])
    return t.numpy()


def f_saturation(x, p, ctx):
    k = p["strength"] / 100
    if k == 0:                                         # the default: no change
        return x
    L = lum(x)
    center = dict(HUES)[p["hues"]]
    if center is None:                                 # every hue alike: L + (x - L) * (1 + k) in one pass
        return _mix(x, 1 + k, _gray3(L), -k)
    h, s = hue(x)
    w = (hue_weight(h, center, 45) * np.clip(s * 3, 0, 1))[..., None]
    L = L[..., None]
    return L + (x - L) * (1 + k * w)


def f_scnr(x, p, ctx):
    """SCNR, Siril's and PixInsight's four protections (postfilters.colour.scnr)."""
    return pf_colour.scnr(x, p["method"], p["strength"], bool(p["lightness"]))


def f_ghs(x, p, ctx):
    return pf_tone.ghs(x, p)


def f_asinh(x, p, ctx):
    return pf_tone.asinh(x, p)


def f_histogram(x, p, ctx):
    return pf_tone.histogram_transform(x, p)


def f_clahe(x, p, ctx):
    """CLAHE; on a region (100 %) the tile grid is the whole canvas's, so that it looks as in the whole image."""
    ax0, ay0 = getattr(ctx, "abox", None)[:2] if getattr(ctx, "abox", None) is not None else (0, 0)
    return pf_detail.clahe(x, p["amount"], p["clip"], p["radius"], ctx.level, (ax0, ay0),
                           getattr(ctx, "canvas_wh", None))


ML_KEYS = ("l1", "l2", "l3", "l4", "l5")
ML_LABELS = ("1 px", "2 px", "4 px", "8 px", "16 px")


def f_ml_sharpen(x, p, ctx):
    return pf_detail.sharpen(x, [p[k] for k in ML_KEYS], p["target"], ctx.level)


def f_ml_denoise(x, p, ctx):
    return pf_detail.denoise(x, [p[k] for k in ML_KEYS[:4]], p["amount"], p["target"], ctx.level)


def f_balance(x, p, ctx):
    t, g = p["temperature"], p["tint"]
    if not (t or g):                                   # the default: neutral
        return x
    return x * np.array([1 + 0.15 * t, 1 + 0.1 * g, 1 - 0.15 * t], np.float32)


def f_local_contrast(x, p, ctx):
    k = p["strength"]
    if k == 0:                                         # the default: no change
        return x
    return _mix(x, 1 + k, gauss(x, p["scale"] / ctx.f), -k)            # x + k * (x - blur)


def f_sharpen(x, p, ctx):
    k = p["strength"]
    if k == 0:                                         # the default: no change
        return x
    return _mix(x, 1 + k, gauss(x, max(p["radius"] / ctx.f, 0.35)), -k)


def f_denoise(x, p, ctx):
    if not (p["luminance"] or p["color"]):             # the default: nothing smoothed
        return x
    Y = lum(x)
    C_ = cv2.subtract(_f32(x), _gray3(Y))                              # the colour, what is left without the luminance
    a, b = p["luminance"], p["color"]
    Y2 = _mix(Y, 1 - a, gauss(Y, 1.6 / ctx.f), a)
    C2 = _mix(C_, 1 - b, gauss(C_, 4 / ctx.f), b)
    return cv2.add(_gray3(Y2), C2)


def f_star_reduce(x, p, ctx):
    if p["strength"] == 0:                             # the default: the stars as they are
        return x
    m = ctx.stars()[..., None]
    size = max(2, int(round(5 / ctx.f)))
    er = erode(x, size)
    return x + p["strength"] * m * (er - x)


def f_halos(x, p, ctx):
    if p["strength"] == 0:                             # the default: the halos as they are
        return x
    m = ctx.stars()
    r = max(1, int(round(p["radius"] / ctx.f)))
    ring = np.clip(dilate(m, 2 * r + 1) - m, 0, 1)
    ring = gauss(ring, r / 3)[..., None]
    bg = gauss(erode(x, 2 * r + 1), r / 2)
    return x + p["strength"] * ring * (np.minimum(x, bg) - x)


def f_star_color(x, p, ctx):
    if p["strength"] == 0:                             # the default: the star colours as they are
        return x
    m = ctx.stars()
    w = _gray3(_f32(m) * float(p["strength"]))                         # k * m, in each channel
    L3 = _gray3(lum(x))
    return cv2.add(L3, cv2.multiply(cv2.subtract(_f32(x), L3), cv2.add(w, 1.0)))


def f_halpha(x, p, ctx):
    if p["strength"] == 0:                             # the default: no red added
        return x
    y = x.copy()
    y[..., 0] = x[..., 0] + p["strength"] * 0.35 * ctx.halpha() * (1 - x[..., 0])
    return y


GHS = ("The generalised hyperbolic stretch (D. Payne and M. Cranfield): a strong stretch around a chosen level, the "
       "shadows and highlights protected")
FILTERS = {
    # ---- stretch
    "ghs": dict(name="Generalised hyperbolic stretch", family="Stretch", fn=f_ghs, graph=pf_tone.ghs_curve, hist=True,
                source=GHS,
                params=[P("stretch", "Stretch factor ln(D+1)", 0, 10, 0, step=0.01),
                        P("local", "Local intensity (b)", -5, 15, 0, step=0.1),
                        P("symmetry", "Symmetry point (SP)", 0, 1, 0, step=0.001),
                        P("shadows", "Shadow protection (LP)", 0, 1, 0, step=0.001),
                        P("highlights", "Highlight protection (HP)", 0, 1, 1, step=0.001),
                        P("black", "Black point (BP)", 0, 0.5, 0, step=0.001),
                        C("model", "Color model", pf_tone.COLOUR_MODES, pf_tone.COLOUR_MODES[0])]),
    "asinh": dict(name="Arcsinh stretch", family="Stretch", fn=f_asinh, graph=pf_tone.asinh_curve, hist=True,
                  source="An arcsinh stretch (after Lupton et al. 2004): the colours kept",
                  params=[P("stretch", "Stretch factor", 0, 250, 0, step=0.5),
                          P("black", "Black point", 0, 0.2, 0, step=0.001),
                          C("model", "Luminance", pf_tone.COLOUR_MODES[2:0:-1], pf_tone.COLOUR_MODES[2])]),
    "histogram": dict(name="Histogram transformation", family="Stretch", fn=f_histogram, graph=pf_tone.ht_curve,
                      hist=True, source="Shadows, midtones and highlights, through the midtones transfer function",
                      params=[P("shadows", "Shadows", 0, 1, 0, step=0.001),
                              P("midtones", "Midtones", 0.001, 0.999, 0.5, step=0.001),
                              P("highlights", "Highlights", 0, 1, 1, step=0.001),
                              C("channel", "Channel", pf_tone.HT_CHANNELS, "RGB/K")]),
    # the arcsinh stretch of the first Post (saved compositions still open with it; no longer offered)
    "stretch": dict(name="Arcsinh stretch (symmetry point)", family="Stretch", fn=f_stretch, hidden=True,
                    source="An arcsinh stretch around a symmetry point (the first Post's)",
                    params=[P("strength", "Strength", 0, 12, 0, step=0.1),
                            P("symmetry", "Symmetry point", 0, 0.6, 0.08, step=0.01)]),
    # ---- contrast and curves
    "curves": dict(name="Curves", family="Stretch", fn=f_curves, editor="curves", hist=True,
                   source="A curve per channel through control points, as Photoshop's Curves",
                   params=[CURVE("curves")]),
    "levels": dict(name="Levels", family="Stretch", fn=f_levels,
                   source="Photoshop's Levels: black point, white point, gamma",
                   params=[P("black", "Black point", 0, 0.3, 0, step=0.005),
                           P("white", "White point", 0.7, 1, 1, step=0.005),
                           P("gamma", "Gamma", 0.5, 2, 1, step=0.01)]),
    "local_contrast": dict(name="Local contrast", family="Contrast", fn=f_local_contrast,
                           source="A large-radius unsharp mask",
                           params=[P("strength", "Strength", 0, 1.5, 0, step=0.01),
                                   P("scale", "Scale", 5, 120, 40, "px", 1)]),
    "clahe": dict(name="Local histogram equalization (CLAHE)", family="Contrast", fn=f_clahe,
                  source="CLAHE (Zuiderveld 1994): the contrast equalised tile by tile, with a limit, on the "
                         "lightness",
                  params=[P("amount", "Amount", 0, 1, 0, step=0.01),
                          P("clip", "Contrast limit", 1, 8, 2, step=0.1),
                          P("radius", "Kernel radius", 16, 512, 64, "px", 1)]),
    # ---- colour
    "saturation": dict(name="Saturation", family="Color", fn=f_saturation,
                       source="Saturation around the luminance, by hue range",
                       params=[P("strength", "Strength", -100, 150, 0, "%", 1),
                               C("hues", "Hues", [h for h, _ in HUES], "All")]),
    "scnr": dict(name="SCNR (green removal)", family="Color", fn=f_scnr,
                 source="Removes a green cast: the green held under a neutral made of red and blue",
                 params=[P("strength", "Amount", 0, 1, 0, step=0.01),
                         C("method", "Protection", pf_colour.SCNR_METHODS, pf_colour.SCNR_METHODS[0]),
                         B("lightness", "Preserve lightness", True)]),
    "balance": dict(name="Color balance", family="Color", fn=f_balance,
                    source="Gains on red and blue (temperature) and on green (tint)",
                    params=[P("temperature", "Temperature", -1, 1, 0, step=0.01),
                            P("tint", "Green / magenta", -1, 1, 0, step=0.01)]),
    # ---- detail
    "sharpen": dict(name="Sharpen", family="Detail", fn=f_sharpen,
                    source="Unsharp mask",
                    params=[P("strength", "Strength", 0, 2, 0, step=0.01),
                            P("radius", "Radius", 0.5, 4, 1.2, "px", 0.1)]),
    "ml_sharpen": dict(name="Multiscale sharpening", family="Detail", fn=f_ml_sharpen,
                       source="Starlet wavelet layers (B3 spline a trous): each scale of detail raised or lowered "
                              "on its own",
                       params=[P(k, lb, -1, 3, 0, step=0.01) for k, lb in zip(ML_KEYS, ML_LABELS)] +
                       [C("target", "Applies to", pf_detail.TARGETS, pf_detail.TARGETS[0])]),
    # ---- noise
    "denoise": dict(name="Noise reduction", family="Noise", fn=f_denoise,
                    source="A Gaussian smoothing of the luminance and of the colour",
                    params=[P("luminance", "Luminance", 0, 1, 0, step=0.01),
                            P("color", "Color", 0, 1, 0, step=0.01)]),
    "ml_denoise": dict(name="Multiscale noise reduction", family="Noise", fn=f_ml_denoise,
                       source="Soft thresholding of the starlet wavelet layers (Starck and Murtagh): the noise "
                              "lowered scale by scale",
                       params=[P(k, lb, 0, 10, 0, "σ", 0.1) for k, lb in zip(ML_KEYS[:4], ML_LABELS[:4])] +
                       [P("amount", "Amount", 0, 1, 1, step=0.01),
                        C("target", "Applies to", pf_detail.NR_TARGETS, pf_detail.NR_TARGETS[0])]),
    # ---- stars
    "star_reduce": dict(name="Star reduction", family="Stars", fn=f_star_reduce, needs="stars",
                        source="A morphological erosion through the star mask",
                        params=[P("strength", "Strength", 0, 1, 0, step=0.01)]),
    "halos": dict(name="Halo reduction", family="Stars", fn=f_halos, needs="stars",
                  source="The ring around the star mask pulled to the eroded background",
                  params=[P("strength", "Strength", 0, 1, 0, step=0.01), P("radius", "Radius", 4, 40, 14, "px", 1)]),
    "star_color": dict(name="Star color", family="Stars", fn=f_star_color, needs="stars",
                       source="Saturation through the star mask",
                       params=[P("strength", "Strength", 0, 2, 0, step=0.01)]),
    # ---- combine
    "halpha_red": dict(name="H-alpha to red", family="Combine", fn=f_halpha, needs="halpha",
                       source="The engine's H-alpha map added to the red channel",
                       params=[P("strength", "Strength", 0, 1, 0, step=0.01)]),
}
FAMILIES = ["Stretch", "Contrast", "Color", "Detail", "Noise", "Stars", "Combine"]
#: the names of filters as the first Post saved them (a layer keeps its name): shown as today's names
OLD_NAMES = {"Stretch (GHS)": "Arcsinh stretch (symmetry point)"}

# ======================================================================== masks
MASK_SOURCES = {
    "stars": "Catalog stars", "halpha": "H-alpha", "luminance": "Luminance range", "color": "Color range",
    "gradient": "Linear gradient", "brush": "Brush",
}
DIRECTIONS = ["Top → bottom", "Bottom → top", "Left → right", "Right → left"]
#: the settings of a mask part: key → (label, lo, hi, unit); the defaults are those of ``new_part``
PART_SETTINGS = {
    "stars": [("grow", "Size around stars", 0, 24, "px")],
    "luminance": [("lo", "From", 0, 100, "%"), ("hi", "To", 0, 100, "%")],
    "color": [("hue", "Hue", 0, 359, "°"), ("width", "Width", 10, 120, "°")],
    "gradient": [("position", "Position", 0, 100, "%"), ("softness", "Softness", 1, 100, "%")],
}


def new_part(src: str) -> dict:
    p = {"id": new_id(), "src": src, "op": "add"}
    if src == "stars":
        p.update(grow=4.0)
    elif src == "luminance":
        p.update(lo=40.0, hi=100.0)
    elif src == "color":
        p.update(hue=0.0, width=40.0)
    elif src == "gradient":
        p.update(direction=DIRECTIONS[0], position=50.0, softness=40.0)
    elif src == "brush":
        p.update(strokes=[], size=60.0, hardness=50.0, paint=True)
    elif src != "halpha":
        raise ValueError(f"unknown mask source {src!r}")
    return p


def new_mask() -> dict:
    return {"parts": [], "feather": 0.0, "density": 100.0, "invert": False}


def new_workflow_layer(pid: int, name: str, visible=False, part=None, base=None) -> dict:
    return {"id": new_id(), "kind": "workflow", "pid": pid, "part": part, "name": name, "visible": visible,
            "opacity": 100.0, "mode": "Normal", "base": base, "sliders": {}, "mask": None}


def new_image_layer(key: str, name: str, visible: bool = True) -> dict:
    """A layer of a stage result's image (``key`` names it; the pixels come from the render worker's sources)."""
    return {"id": new_id(), "kind": "image", "key": str(key), "name": name, "visible": visible, "opacity": 100.0,
            "mode": "Normal", "sliders": {}, "linked": True, "mask": None}


PIXEL = ("workflow", "image")
#: the sliders of an image layer (its automatic stretch; the simple rendering's Sky, Contrast and Saturation, 0-4)
IMAGE_SLIDERS = (("sky", "Sky", ("darker", "brighter")), ("contrast", "Contrast", ("softer", "stronger")),
                 ("saturation", "Saturation", ("paler", "more vivid")))


def is_pixel(lay: dict) -> bool:
    """A layer with an image of its own (a workflow's or a stage result's), which filters can be nested in."""
    return lay.get("kind") in PIXEL


def source_id(lay: dict):
    """What the composition asks the provider for: a workflow's id, or ("image", key)."""
    return lay["pid"] if lay["kind"] == "workflow" else ("image", lay["key"])


def new_filter_layer(key: str, target=None) -> dict:
    """A filter layer. target: the id of the one layer it applies to (it sits just above it, as a Photoshop clipping
    mask); None: it applies to everything below it."""
    f = FILTERS[key]
    return {"id": new_id(), "kind": "filter", "filter": key, "name": f["name"], "visible": True, "opacity": 100.0,
            "mode": "Normal", "params": {q["key"]: copy.deepcopy(q["default"]) for q in f["params"]}, "mask": None,
            "target": target}


def normalize(layers: list[dict]) -> list[dict]:
    """Each layer's own filters go just above it, in their order; a filter whose layer is gone applies to everything
    below. The list is top first."""
    ids = {lay["id"] for lay in layers if is_pixel(lay)}
    for lay in layers:
        if lay["kind"] == "filter" and lay.get("target") not in ids:
            lay["target"] = None
    out = []
    for lay in layers:
        if lay["kind"] == "filter" and lay.get("target"):
            continue
        if is_pixel(lay):
            out += [f for f in layers if f["kind"] == "filter" and f.get("target") == lay["id"]]
        out.append(lay)
    return out


def children(layers: list[dict], lid: int) -> list[dict]:
    """The filters nested in a pixel layer, top first (the top one is applied last)."""
    return [f for f in layers if f["kind"] == "filter" and f.get("target") == lid]


def tree(layers: list[dict]) -> list[tuple[dict, list[dict]]]:
    """The panel's structure: the top-level rows, top first, each with its children (pixel layers only)."""
    return [(lay, children(layers, lay["id"]) if is_pixel(lay) else []) for lay in normalize(list(layers))
            if not (lay["kind"] == "filter" and lay.get("target"))]


def rows(layers: list[dict], collapsed=()) -> list[tuple[dict, int]]:
    """The rows of the panel, top first, with their depth: a pixel layer, then its children indented (unless it is
    collapsed), then the next top-level row."""
    out = []
    for lay, kids in tree(layers):
        out.append((lay, 0))
        if lay["id"] not in collapsed:
            out += [(k, 1) for k in kids]
    return out


def move(layers: list[dict], lid: int, parent: int | None, index: int) -> list[dict]:
    """The layers with one moved, as a drop in the panel does. ``parent`` None: it becomes the top-level row at
    ``index`` (0 = the top, counted without it; a filter then applies to everything below it); else it becomes the child
    of that pixel layer at ``index`` among its children (0 = the top child). A pixel layer moves with its children and
    stays top level."""
    by_id = {x["id"]: x for x in layers}
    lay = by_id[lid]
    if parent is not None and (is_pixel(lay) or parent == lid or not is_pixel(by_id[parent])):
        raise ValueError("only a filter is nested, and only in a pixel layer")
    group = [lay] + (children(layers, lid) if is_pixel(lay) else [])
    gid = {x["id"] for x in group}
    top = [(x, [k for k in kids if k["id"] not in gid]) for x, kids in tree(layers) if x["id"] not in gid]
    if parent is None:
        if lay["kind"] == "filter":
            lay["target"] = None
        top.insert(max(0, min(index, len(top))), (lay, group[1:]))
    else:
        lay["target"] = parent
        kids = next(k for x, k in top if x["id"] == parent)
        kids.insert(max(0, min(index, len(kids))), lay)
    out = []
    for x, kids in top:
        out += kids + [x]
    return out


def snapshot(layers: list[dict]) -> list[dict]:
    """A copy of the layers, for Undo and for a render request: nothing in it is shared with the layers being edited."""
    out = []
    for lay in layers:
        d = dict(lay)
        if "sliders" in d:
            d["sliders"] = dict(d["sliders"])
        if "params" in d:
            d["params"] = {k: copy.deepcopy(v) if isinstance(v, (dict, list)) else v for k, v in d["params"].items()}
        m = lay.get("mask")
        if m is not None:
            d["mask"] = dict(m, parts=[dict(p, strokes=list(p["strokes"])) if "strokes" in p else dict(p)
                                       for p in m["parts"]])
        out.append(d)
    return out


# ---------------------------------------------------------------------- saved compositions
def _num(v, default, lo=None, hi=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return min(max(x, lo), hi) if lo is not None else x


def _clean_mask(m) -> dict | None:
    if not isinstance(m, dict):
        return None
    out = new_mask()
    out["feather"] = _num(m.get("feather"), 0.0, 0, 60)
    out["density"] = _num(m.get("density"), 100.0, 0, 100)
    out["invert"] = bool(m.get("invert"))
    for p in m.get("parts") or []:
        src = p.get("src") if isinstance(p, dict) else None
        if src not in MASK_SOURCES:
            continue
        q = new_part(src)
        q["id"] = int(p.get("id") or q["id"])
        q["op"] = p.get("op") if p.get("op") in OPS else "add"
        for k, v in q.items():
            if k in ("id", "src", "op") or k not in p:
                continue
            if k == "strokes":
                q[k] = [(float(s[0]), float(s[1]), float(s[2]), float(s[3]), bool(s[4]))
                        for s in p[k] if isinstance(s, (list, tuple)) and len(s) == 5]
            elif k == "paint":
                q[k] = bool(p[k])
            elif k == "direction":
                q[k] = p[k] if p[k] in DIRECTIONS else v
            else:
                q[k] = _num(p[k], v)
        out["parts"].append(q)
    return out


def load_layers(raw, alive: set[int]) -> list[dict]:
    """The layers of a saved composition (``project.post["layers"]``), made safe: unknown filters and the layers of
    workflows that no longer exist are dropped, every field is brought to its type and range."""
    out: list[dict] = []
    for d in raw or []:
        if not isinstance(d, dict):
            continue
        if d.get("kind") == "workflow":
            if int(d.get("pid", -1)) not in alive or d.get("part") not in (None, "stars", "starless"):
                continue
            lay = new_workflow_layer(int(d["pid"]), str(d.get("name") or "?"), bool(d.get("visible")), d.get("part"),
                                     d.get("base"))
            lay["sliders"] = {k: _num(v, 2.0, 0, 4) for k, v in (d.get("sliders") or {}).items()}
        elif d.get("kind") == "image" and isinstance(d.get("key"), str) and d["key"]:
            lay = new_image_layer(d["key"], str(d.get("name") or "?"), bool(d.get("visible", True)))
            lay["sliders"] = {k: _num(v, 2.0, 0, 4) for k, v in (d.get("sliders") or {}).items()}
            lay["linked"] = bool(d.get("linked", True))
        elif d.get("kind") == "filter" and d.get("filter") in FILTERS:
            lay = new_filter_layer(d["filter"], d.get("target"))
            name = str(d.get("name") or lay["name"])
            lay["name"] = OLD_NAMES.get(name, name)
            lay["visible"] = bool(d.get("visible", True))
            raw = dict(d.get("params") or {})
            if d["filter"] == "curves" and "curves" not in raw:          # saved before the Curves tool
                raw["curves"] = curves_of_old(_num(raw.get("shadows"), 0.0, -0.2, 0.2),
                                              _num(raw.get("midtones"), 0.0, -0.2, 0.2),
                                              _num(raw.get("highlights"), 0.0, -0.2, 0.2))
            if d["filter"] == "scnr":                                    # saved before its protections: as it was
                raw.setdefault("method", pf_colour.SCNR_METHODS[0])
                raw.setdefault("lightness", False)
            for q in FILTERS[d["filter"]]["params"]:
                v = raw.get(q["key"], q["default"])
                if q["type"] == "choice":
                    v = v if v in q["choices"] else q["default"]
                elif q["type"] == "bool":
                    v = bool(v)
                elif q["type"] == "curve":
                    v = pf_curves.clean_curves(v)
                else:
                    v = _num(v, q["default"], q["lo"], q["hi"])
                lay["params"][q["key"]] = v
        else:
            continue
        lay["id"] = int(d.get("id") or lay["id"])
        lay["opacity"] = _num(d.get("opacity"), 100.0, 0, 100)
        lay["mode"] = d.get("mode") if d.get("mode") in MODES else "Normal"
        lay["mask"] = _clean_mask(d.get("mask"))
        out.append(lay)
    ids = {lay["id"] for lay in out}
    if len(ids) != len(out):                                  # duplicated ids: give them new ones
        seen: set[int] = set()
        for lay in out:
            if lay["id"] in seen:
                lay["id"] = new_id()
            seen.add(lay["id"])
    reseed_ids(out)
    return normalize(out)


def save_layers(layers: list[dict]) -> list[dict]:
    """The composition as JSON-able dicts."""
    return json.loads(json.dumps(snapshot(layers), default=list))


# ======================================================================== what the composition reads
@dataclass
class Canvas:
    """The image of Post: the area every workflow layer covers, in the pixels of the first (bottom) layer's grid."""
    W: int
    H: int
    origin: tuple[int, int] = (0, 0)

    def size(self, level: int) -> tuple[int, int]:
        f = 2 ** level
        return -(-self.W // f), -(-self.H // f)


class Provider:
    """What the composition needs from the engine. ``services/render.py`` implements it on the layers service."""

    def version(self, pid) -> int:
        """Changes when the result of this workflow (or image, ``("image", key)``) changes."""
        return 0

    def available(self, pid) -> bool:
        raise NotImplementedError

    def canvas(self, pids: list) -> Canvas:
        """The canvas of these workflows and images (bottom first, see ``source_id``)."""
        raise NotImplementedError

    def layer(self, lay: dict, level: int, box: tuple[int, int, int, int], draft: bool = False,
              stop=None) -> np.ndarray:
        """The pixel layer ``lay`` at the level, over the box (x0, y0, x1, y1) of the canvas at that level:
        (h, w, 3) float32 in [0, 1]. Above level 0 the box is the whole canvas. ``draft``: a quick look is enough (a
        slider moves); ``stop()``: True abandons the rendering (raise ``Superseded``)."""
        raise NotImplementedError

    def data_mask(self, name: str, level: int, box) -> np.ndarray | None:
        """The 'stars' or 'halpha' map over the box, (h, w) float32 in [0, 1], or None when the layers have none."""
        return None

    def has(self, name: str) -> bool:
        """Whether a data mask exists ('stars': a layer with the Stars block; 'halpha')."""
        return False

    def notes(self) -> set[str]:
        """What the engine said while rendering (a fallback to the CPU...), read and cleared by the composer."""
        return set()


class ByteLRU:
    """Images kept by content, the least recently used dropped first; an image no layer holds any more goes at once."""

    def __init__(self, cap: int = 1_500_000_000):
        self.cap = cap
        self.items: OrderedDict = OrderedDict()
        self.bytes = 0
        self._held: dict = {}                     # (layer id, level) -> key
        self._holders: dict = {}                  # key -> {(layer id, level)}

    def get(self, key):
        a = self.items.get(key)
        if a is not None:
            self.items.move_to_end(key)
        return a

    def put(self, key, a: np.ndarray, holder=None):
        if key not in self.items:
            self.items[key] = a
            self.bytes += a.nbytes
        self.items.move_to_end(key)
        if holder is not None:
            self.hold(key, holder)
        while self.bytes > self.cap and len(self.items) > 1:
            k = next(iter(self.items))
            if k == key:
                self.items.move_to_end(k)
                continue
            self._drop(k)

    def hold(self, key, holder):
        old = self._held.get(holder)
        if old is not None and old != key:
            hs = self._holders.get(old, set())
            hs.discard(holder)
            if not hs:
                self._drop(old)
        self._held[holder] = key
        self._holders.setdefault(key, set()).add(holder)

    def _drop(self, key):
        a = self.items.pop(key, None)
        if a is not None:
            self.bytes -= a.nbytes
        for h in self._holders.pop(key, set()):
            if self._held.get(h) == key:
                del self._held[h]

    def clear(self):
        self.items.clear()
        self._held.clear()
        self._holders.clear()
        self.bytes = 0


@dataclass
class Composite:
    rgb: np.ndarray                                           # (h, w, 3) float32 in [0, 1]: the box at the level
    level: int
    box: tuple[int, int, int, int]
    masks: dict[int, np.ndarray] = field(default_factory=dict)          # layer id -> (h, w) float32, those wanted
    mask_thumbs: dict[int, np.ndarray] = field(default_factory=dict)    # layer id -> (36, 54) uint8
    notes: set[str] = field(default_factory=set)
    layer_ms: float = 0.0                                     # time spent in the engine's renderings
    drawn: int = 0                                            # how many workflow layers were rendered (not cached)
    hists: dict[int, dict] = field(default_factory=dict)      # filter id -> the histograms of its input (``hists``)


class Ctx:
    """What a filter or a mask part needs, at one resolution and for one region."""

    def __init__(self, comp: Composer, level: int, abox, draft: bool = False):
        self.comp, self.level, self.abox = comp, level, abox
        self.f = 2 ** level
        self.draft = draft                                    # a quick look while something moves (tables, not exact)
        self.hist_ids: set = set()                            # the filters whose input's histograms are wanted
        self.hists: dict = {}
        self._memo: dict[str, np.ndarray] = {}

    def _data(self, name: str) -> np.ndarray:
        if name not in self._memo:
            m = self.comp.provider.data_mask(name, self.level, self.abox)
            x0, y0, x1, y1 = self.abox
            self._memo[name] = (np.zeros((y1 - y0, x1 - x0), np.float32) if m is None
                                else np.ascontiguousarray(m, np.float32))
        return self._memo[name]

    def stars(self) -> np.ndarray:
        return self._data("stars")

    def halpha(self) -> np.ndarray:
        return self._data("halpha")


def thumb_of(a: np.ndarray) -> np.ndarray:
    """A 54 x 36 uint8 thumbnail of a float image (h, w[, 3])."""
    u = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    return cv2.resize(u, THUMB, interpolation=cv2.INTER_AREA)


class Composer:
    """The composition of the layers. One per render worker: it owns the caches (the workflow layers' images by
    content, the brushes' rasters), so it is used from one thread only."""

    def __init__(self, provider: Provider, cache_bytes: int = 1_500_000_000):
        self.provider = provider
        self.images = ByteLRU(cache_bytes)
        self._brush: dict = {}                              # (part id, level) -> (n strokes, last stroke, raster)
        self._canvas: tuple | None = None                   # (pids, canvas)
        self._drawn = 0

    # ------------------------------------------------------------ geometry
    def canvas_of(self, layers: list[dict]) -> Canvas | None:
        pids = [source_id(lay) for lay in reversed(layers) if is_pixel(lay) and self.provider.available(source_id(lay))]
        if not pids:
            return None
        key = (tuple(pids), tuple(self.provider.version(p) for p in pids))
        if self._canvas is None or self._canvas[0] != key:
            self._canvas = (key, self.provider.canvas(pids))
        return self._canvas[1]

    def forget(self):
        """Drop every cached image (the results changed on disk, or an Undo)."""
        self.images.clear()
        self._brush.clear()
        self._canvas = None

    def clear_brushes(self):
        self._brush.clear()

    # ------------------------------------------------------------ one workflow layer
    def _key(self, lay: dict, level: int, abox, canvas: Canvas, draft: bool = False) -> tuple:
        """What an image of a pixel layer depends on: the same for a copy of the layer. The canvas is part of it: it
        changes when another workflow's result comes in. A draft (a quick look) is kept apart from the exact image."""
        sid = source_id(lay)
        return (sid, lay.get("part"), json.dumps(lay.get("sliders") or {}, sort_keys=True),
                json.dumps(lay.get("base"), sort_keys=True, default=str), lay.get("linked"), self.provider.version(sid),
                level, None if level > 0 else abox, (canvas.W, canvas.H, canvas.origin)) + (("draft",) if draft else ())

    def cached(self, lay: dict, level: int, canvas: Canvas) -> np.ndarray | None:
        """The image of a pixel layer over the whole canvas at a level if it is in the cache (level > 0)."""
        return self.images.get(self._key(lay, level, None, canvas))

    def layer_rgb(self, lay: dict, level: int, abox, canvas: Canvas, cache: bool = True, draft: bool = False,
                  stop=None) -> np.ndarray:
        """The image of a pixel layer over the box at the level; cached by content (a copy of a layer shares it). A
        draft takes the exact image when it is cached, else a quick look."""
        W, H = canvas.size(level)
        whole = (0, 0, W, H)
        want = whole if level > 0 else abox
        key = self._key(lay, level, abox, canvas)
        a = self.images.get(key) if cache else None
        if a is None and draft and level > 0:
            key = self._key(lay, level, abox, canvas, True)
            a = self.images.get(key) if cache else None
        if a is None:
            a = self.provider.layer(lay, level, want, draft=draft and level > 0, stop=stop)
            self._drawn += 1
            if cache:
                self.images.put(key, a, (lay["id"], level))
        elif cache:
            self.images.hold(key, (lay["id"], level))
        if level > 0 and abox != whole:
            x0, y0, x1, y1 = abox
            a = a[y0:y1, x0:x1]
        return a

    # ------------------------------------------------------------ masks
    def part_raster(self, part, level, abox, below, canvas: Canvas, ctx: Ctx) -> np.ndarray:
        f = ctx.f
        src = part["src"]
        if src == "stars":
            m = ctx.stars()
            r = int(round(part["grow"] / f))
            if r > 0:
                m = dilate(m, 2 * r + 1)
                m = gauss(m, r / 2)
            return np.clip(m * 1.5, 0, 1)
        if src == "halpha":
            return ctx.halpha()
        if src == "luminance":
            L = lum(np.clip(below, 0, 1))
            lo, hi, soft = part["lo"] / 100, part["hi"] / 100, 0.06
            return np.clip((L - (lo - soft)) / (2 * soft), 0, 1) * np.clip(((hi + soft) - L) / (2 * soft), 0, 1)
        if src == "color":
            h, sat = hue(np.clip(below, 0, 1))
            return hue_weight(h, part["hue"], part["width"]) * np.clip(sat * 4, 0, 1)
        if src == "gradient":
            W, H = canvas.size(level)
            x0, y0, x1, y1 = abox
            ys = np.arange(y0, y1, dtype=np.float32)[:, None] / H
            xs = np.arange(x0, x1, dtype=np.float32)[None, :] / W
            ys, xs = np.broadcast_arrays(ys, xs)
            t = dict(zip(DIRECTIONS, (ys, 1 - ys, xs, 1 - xs)))[part["direction"]]
            pos, soft = part["position"] / 100, max(part["softness"] / 100, 0.01)
            return np.clip((t - (pos - soft / 2)) / soft, 0, 1).astype(np.float32)
        if src == "brush":
            return self.brush_raster(part, level, abox, canvas)
        raise ValueError(src)

    def brush_raster(self, part, level, abox, canvas: Canvas) -> np.ndarray:
        """The strokes of a brush part at a level. Above level 0 the whole canvas, kept: only the new strokes are
        stamped. At level 0 only the box asked for."""
        strokes = part["strokes"]
        n, last = len(strokes), (strokes[-1] if strokes else None)
        f = 2 ** level
        W, H = canvas.size(level)
        if level == 0:
            x0, y0, x1, y1 = abox
            r = np.zeros((y1 - y0, x1 - x0), np.float32)
            for x, y, rad, hard, paint in strokes:
                self.stamp(r, x - x0, y - y0, rad, hard, paint)
            return r
        key = (part["id"], level, canvas.W, canvas.H)
        hit = self._brush.get(key)
        if hit is not None and hit[0] == n and hit[1] == last:
            r = hit[2]
        else:
            if hit is not None and hit[0] < n and (hit[0] == 0 or strokes[hit[0] - 1] == hit[1]):
                r, start = hit[2], hit[0]
            else:
                r, start = np.zeros((H, W), np.float32), 0
            for x, y, rad, hard, paint in strokes[start:]:
                self.stamp(r, x / f, y / f, rad / f, hard, paint)
            self._brush[key] = (n, last, r)
        x0, y0, x1, y1 = abox
        return r[y0:y1, x0:x1]

    @staticmethod
    def stamp(r, x, y, rad, hard, paint):
        H, W = r.shape
        rad = max(rad, 0.8)
        xa, xb = int(max(0, x - rad - 1)), int(min(W, x + rad + 2))
        ya, yb = int(max(0, y - rad - 1)), int(min(H, y + rad + 2))
        if xa >= xb or ya >= yb:
            return
        yy, xx = np.mgrid[ya:yb, xa:xb]
        d = np.hypot(xx - x, yy - y) / rad
        core = hard / 100
        v = np.clip((1 - d) / max(1 - core, 0.05), 0, 1).astype(np.float32)
        if paint:
            r[ya:yb, xa:xb] = np.maximum(r[ya:yb, xa:xb], v)
        else:
            r[ya:yb, xa:xb] *= 1 - v

    def mask(self, lay, level, abox, below, canvas: Canvas, ctx: Ctx) -> np.ndarray | None:
        m = lay["mask"]
        if m is None:
            return None
        if not m["parts"]:                                 # a new mask shows the whole layer, as in Photoshop
            return self._finish(np.ones(below.shape[:2], np.float32), m, level)
        out = None
        for part in m["parts"]:
            r = self.part_raster(part, level, abox, below, canvas, ctx)
            if out is None:
                out = np.zeros_like(r) if part["op"] == "add" else np.ones_like(r)
            if part["op"] == "add":
                out = np.maximum(out, r)
            elif part["op"] == "sub":
                out = out * (1 - r)
            else:
                out = out * r
        return self._finish(out, m, level)

    @staticmethod
    def _finish(out, m, level):
        if m["feather"]:
            out = gauss(out, m["feather"] / 2 ** level)
        if m["density"] < 100:
            out = 1 - (m["density"] / 100) * (1 - out)
        if m["invert"]:
            out = 1 - out
        return out

    # ------------------------------------------------------------ the whole image
    def compose(self, layers: list[dict], level: int, box=None, *, want=(), thumbs: bool = False, stop=None,
                cache: bool = True, draft: bool = False, hists=()) -> Composite | None:
        """The composite (float 0..1) at a resolution, for a box of the canvas at that level (default: all of it).
        ``want``: ids of layers whose mask is returned (also those hidden). ``thumbs``: also a thumbnail of every mask
        computed. ``stop``: called between the steps (and by the engine between its stages), True abandons the work
        (``Superseded``). ``draft``: a quick look of the pixel layers whose exact image is not cached, and of the filters
        that have one (the curves' tables). ``hists``: ids of filters whose input's histograms are returned (the
        Curves editor's and the stretches' graphs). None when no pixel layer has a result."""
        canvas = self.canvas_of(layers)
        if canvas is None:
            return None
        W, H = canvas.size(level)
        if box is None:
            box = (0, 0, W, H)
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
            raise ValueError(f"box {box} outside the {W} x {H} canvas at level {level}")
        pad = REGION_MARGIN if level == 0 and box != (0, 0, W, H) else 0
        abox = (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))
        ax0, ay0, ax1, ay1 = abox
        comp = np.zeros((ay1 - ay0, ax1 - ax0, 3), np.float32)
        out = Composite(comp, level, box)
        ctx = Ctx(self, level, abox, draft)
        ctx.hist_ids, ctx.hists = set(hists), out.hists
        ctx.canvas_wh = (W, H)                                            # the whole canvas at this level
        want = set(want)
        self._drawn = 0
        check = stop or (lambda: False)

        def checkpoint():
            if check():
                raise Superseded()

        def keep_mask(lay, m):
            if m is None:
                return
            if lay["id"] in want:
                out.masks[lay["id"]] = m[y0 - ay0:y0 - ay0 + (y1 - y0), x0 - ax0:x0 - ax0 + (x1 - x0)]
            if thumbs:
                out.mask_thumbs[lay["id"]] = thumb_of(m)

        stack = list(reversed(layers))                                    # bottom first
        for lay in stack:
            checkpoint()
            if lay["kind"] == "filter" and lay.get("target"):
                continue                                                  # applied with its own layer, below
            if not lay["visible"]:
                if lay["id"] in want or (thumbs and lay["mask"] is not None):
                    keep_mask(lay, self.mask(lay, level, abox, comp, canvas, ctx))
                continue
            if is_pixel(lay):
                if not self.provider.available(source_id(lay)):
                    continue
                src = self.layer_rgb(lay, level, abox, canvas, cache, draft, check)
                for fl in stack:                                          # this layer's own filters, bottom first
                    if fl["kind"] != "filter" or fl.get("target") != lay["id"]:
                        continue
                    checkpoint()
                    if not fl["visible"]:
                        if fl["id"] in want or (thumbs and fl["mask"] is not None):
                            keep_mask(fl, self.mask(fl, level, abox, src, canvas, ctx))
                        continue
                    src, fm = self._apply(fl, src, level, abox, canvas, ctx)
                    keep_mask(fl, fm)
                m = self.mask(lay, level, abox, src, canvas, ctx)
                keep_mask(lay, m)
                src = self.blend(lay["mode"], comp, src)
                a = lay["opacity"] / 100
                if m is None and a >= 0.999:
                    comp = src
                else:
                    a = a if m is None else (a * m)[..., None]
                    comp = comp + (src - comp) * a
            else:
                comp, m = self._apply(lay, comp, level, abox, canvas, ctx)
                keep_mask(lay, m)
        cut = (slice(y0 - ay0, y0 - ay0 + (y1 - y0)), slice(x0 - ax0, x0 - ax0 + (x1 - x0)))
        out.rgb = np.clip(comp[cut], 0, 1).astype(np.float32, copy=False)
        out.notes = set(self.provider.notes())
        out.drawn = self._drawn
        return out

    def _apply(self, fl, under, level, abox, canvas, ctx):
        """A filter layer on an image (its own layer's, or everything below): the filtered image, through the layer's
        mask, opacity and mode. Returns (result, mask)."""
        if fl["id"] in ctx.hist_ids:
            ctx.hists[fl["id"]] = pf_curves.histograms(under)
        out = FILTERS[fl["filter"]]["fn"](under, fl["params"], ctx)
        m = self.mask(fl, level, abox, under, canvas, ctx)
        out = self.blend(fl["mode"], under, out)
        a = fl["opacity"] / 100
        if m is None and a >= 0.999:
            return out, m
        a = a if m is None else (a * m)[..., None]
        return under + (out - under) * a, m

    @staticmethod
    def blend(mode, a, b):
        if mode == "Lighten":
            return np.maximum(a, b)
        if mode == "Screen":
            t = np.clip(b, 0, 1)
            np.subtract(1.0, t, out=t)
            t *= 1.0 - a                                     # (1 - a)(1 - b)
            np.subtract(1.0, t, out=t)
            return t
        if mode == "Multiply":
            return a * b
        if mode == "Soft light":
            b = np.clip(b, 0, 1)
            return (1 - 2 * b) * a * a + 2 * b * a
        return b


def only_layer(layers: list[dict], lid: int) -> list[dict]:
    """A pixel layer alone with its own filters (visible, opaque, Normal, no mask): what 'each layer as a separate
    file' writes."""
    base = next(lay for lay in layers if lay["id"] == lid)
    own = [dict(f, visible=f["visible"]) for f in layers if f["kind"] == "filter" and f.get("target") == lid]
    return snapshot(own + [dict(base, visible=True, opacity=100.0, mode="Normal", mask=None)])


def copy_layers(layers: list[dict]) -> list[dict]:
    return copy.deepcopy(layers)
