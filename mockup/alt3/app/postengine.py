"""The Post of mockup alt3, without its widgets: layers, masks, filters and the composition.

Speed: the image is composed at the screen's resolution (half of the full image), and at a quarter while a slider or
the brush moves; full resolution only for the region shown when zoomed in. Each workflow layer is cached; only what
changed is computed again.

Layers (top first in the list, as in Photoshop):
* a workflow layer: the image of a computed workflow, with its simplified sliders (prepared renderings at 5
  positions; a position between two is interpolated);
* a filter layer: a filter applied to everything below it, through the layer's mask (an adjustment layer).
Every layer may have a mask made of parts (as in Lightroom): automatic (catalogue stars, H-alpha emission, luminosity,
colour, gradient) or painted, each added, subtracted or intersected.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
from collections import OrderedDict

import numpy as np
from PIL import Image
from scipy import ndimage

from model import DATA

MODES = ["Normal", "Lighten", "Screen", "Multiply", "Soft light"]       # plain English: _() where shown
OPS = {"add": "Add", "sub": "Subtract", "inter": "Intersect"}
_ids = itertools.count(1)


def new_id() -> int:
    return next(_ids)


# ======================================================================== images at three resolutions
class Store:
    """Prepared images as uint8, at full resolution (0), half (1) and quarter (2); least recently used dropped."""

    def __init__(self, cap=90):
        self.cache: OrderedDict = OrderedDict()
        self.cap = cap

    def _put(self, key, a):
        self.cache[key] = a
        while len(self.cache) > self.cap:
            self.cache.popitem(last=False)

    def get(self, rel: str, level: int, gray=False) -> np.ndarray:
        key = (rel, level, gray)
        a = self.cache.get(key)
        if a is not None:
            self.cache.move_to_end(key)
            return a
        im = Image.open(DATA / rel).convert("L" if gray else "RGB")
        if level:
            w, h = im.size
            f = 2 ** level
            im = im.resize((-(-w // f), -(-h // f)), Image.BOX)
        a = np.asarray(im)
        self._put(key, a)
        return a


def to_f(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) * (1 / 255)


def lum(x: np.ndarray) -> np.ndarray:
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


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


def gauss(x, sigma):
    if sigma < 0.3:
        return x
    return ndimage.gaussian_filter(x, (sigma, sigma, 0) if x.ndim == 3 else sigma)


# ======================================================================== filters
HUES = [("All", None), ("Reds", 0), ("Yellows", 60), ("Greens", 120), ("Cyans", 180), ("Blues", 230),
        ("Magentas", 300)]


def P(key, label, lo, hi, default, unit="", step=None):
    return dict(key=key, label=label, type="float", lo=lo, hi=hi, default=default, unit=unit, step=step)


def C(key, label, choices, default):
    return dict(key=key, label=label, type="choice", choices=choices, default=default)


def f_stretch(x, p, ctx):
    d, sp = p["force"], p["symetrie"]
    if d < 1e-3:
        return x
    a0 = np.arcsinh(-d * sp)
    a1 = np.arcsinh(d * (1 - sp))
    return (np.arcsinh(d * (x - sp)) - a0) / (a1 - a0)


def f_curves(x, p, ctx):
    xs = [0, 0.25, 0.5, 0.75, 1]
    ys = [0, 0.25 + p["ombres"], 0.5 + p["moyens"], 0.75 + p["lumieres"], 1]
    return np.interp(x, xs, np.clip(ys, 0, 1)).astype(np.float32)


def f_levels(x, p, ctx):
    y = np.clip((x - p["noir"]) / max(p["blanc"] - p["noir"], 1e-3), 0, 1)
    return y ** (1 / p["gamma"])


def f_saturation(x, p, ctx):
    L = lum(x)[..., None]
    k = p["force"] / 100
    center = dict(HUES)[p["teinte"]]
    if center is None:
        w = 1.0
    else:
        h, s = hue(x)
        w = (hue_weight(h, center, 45) * np.clip(s * 3, 0, 1))[..., None]
    return L + (x - L) * (1 + k * w)


def f_scnr(x, p, ctx):
    y = x.copy()
    g = np.minimum(x[..., 1], (x[..., 0] + x[..., 2]) / 2)
    y[..., 1] = x[..., 1] + p["force"] * (g - x[..., 1])
    return y


def f_balance(x, p, ctx):
    t, g = p["temperature"], p["teinte"]
    return x * np.array([1 + 0.15 * t, 1 + 0.1 * g, 1 - 0.15 * t], np.float32)


def f_local_contrast(x, p, ctx):
    return x + p["force"] * (x - gauss(x, p["echelle"] / ctx.f))


def f_sharpen(x, p, ctx):
    return x + p["force"] * (x - gauss(x, max(p["rayon"] / ctx.f, 0.35)))


def f_denoise(x, p, ctx):
    Y = lum(x)[..., None]
    C_ = x - Y
    Y2 = Y + p["luminance"] * (gauss(Y, 1.6 / ctx.f) - Y)
    C2 = C_ + p["chrominance"] * (gauss(C_, 4 / ctx.f) - C_)
    return Y2 + C2


def f_star_reduce(x, p, ctx):
    m = ctx.stars()[..., None]
    size = max(2, int(round(5 / ctx.f)))
    er = ndimage.grey_erosion(x, size=(size, size, 1))
    return x + p["force"] * m * (er - x)


def f_halos(x, p, ctx):
    m = ctx.stars()
    r = max(1, int(round(p["rayon"] / ctx.f)))
    ring = np.clip(ndimage.grey_dilation(m, size=(2 * r + 1, 2 * r + 1)) - m, 0, 1)
    ring = gauss(ring, r / 3)[..., None]
    bg = gauss(ndimage.grey_erosion(x, size=(2 * r + 1, 2 * r + 1, 1)), r / 2)
    return x + p["force"] * ring * (np.minimum(x, bg) - x)


def f_star_color(x, p, ctx):
    m = ctx.stars()[..., None]
    L = lum(x)[..., None]
    return L + (x - L) * (1 + p["force"] * m)


def f_halpha(x, p, ctx):
    y = x.copy()
    y[..., 0] = x[..., 0] + p["force"] * 0.35 * ctx.halpha() * (1 - x[..., 0])
    return y


FILTERS = {
    "etirement": dict(name="Stretch (GHS)", family="Tone", fn=f_stretch,
                      params=[P("force", "Strength", 0, 12, 3, step=0.1), P("symetrie", "Symmetry point", 0, 0.6, 0.08,
                                                                            step=0.01)]),
    "courbes": dict(name="Curves", family="Tone", fn=f_curves,
                    params=[P("ombres", "Shadows", -0.2, 0.2, 0, step=0.01), P("moyens", "Midtones", -0.2, 0.2, 0,
                                                                               step=0.01),
                            P("lumieres", "Highlights", -0.2, 0.2, 0, step=0.01)]),
    "niveaux": dict(name="Levels", family="Tone", fn=f_levels,
                    params=[P("noir", "Black point", 0, 0.3, 0.02, step=0.005), P("blanc", "White point", 0.7, 1, 1,
                                                                                  step=0.005),
                            P("gamma", "Gamma", 0.5, 2, 1, step=0.01)]),
    "saturation": dict(name="Saturation", family="Color", fn=f_saturation,
                       params=[P("force", "Strength", -100, 150, 30, "%", 1), C("teinte", "Hues", [h for h, _ in HUES],
                                                                                "All")]),
    "vert": dict(name="SCNR (green removal)", family="Color", fn=f_scnr,
                 params=[P("force", "Strength", 0, 1, 0.8, step=0.01)]),
    "balance": dict(name="Color balance", family="Color", fn=f_balance,
                    params=[P("temperature", "Temperature", -1, 1, 0, step=0.01), P("teinte", "Green / magenta", -1, 1,
                                                                                    0, step=0.01)]),
    "contraste_local": dict(name="Local contrast", family="Detail", fn=f_local_contrast,
                            params=[P("force", "Strength", 0, 1.5, 0.4, step=0.01), P("echelle", "Scale", 5, 120, 40,
                                                                                      "px", 1)]),
    "nettete": dict(name="Sharpen", family="Detail", fn=f_sharpen,
                    params=[P("force", "Strength", 0, 2, 0.6, step=0.01), P("rayon", "Radius", 0.5, 4, 1.2, "px",
                                                                            0.1)]),
    "bruit": dict(name="Noise reduction", family="Detail", fn=f_denoise,
                  params=[P("luminance", "Luminance", 0, 1, 0.4, step=0.01), P("chrominance", "Color", 0, 1, 0.7,
                                                                               step=0.01)]),
    "etoiles_reduire": dict(name="Star reduction", family="Stars", fn=f_star_reduce,
                            params=[P("force", "Strength", 0, 1, 0.6, step=0.01)]),
    "halos": dict(name="Halo reduction", family="Stars", fn=f_halos,
                  params=[P("force", "Strength", 0, 1, 0.6, step=0.01), P("rayon", "Radius", 4, 40, 14, "px", 1)]),
    "etoiles_couleur": dict(name="Star color", family="Stars", fn=f_star_color,
                            params=[P("force", "Strength", 0, 2, 0.6, step=0.01)]),
    "halpha_rouge": dict(name="H-alpha to red", family="Combine", fn=f_halpha,
                         params=[P("force", "Strength", 0, 1, 0.5, step=0.01)]),
}
FAMILIES = ["Tone", "Color", "Detail", "Stars", "Combine"]

# ======================================================================== masks
MASK_SOURCES = {
    "etoiles": "Catalog stars", "halpha": "H-alpha", "lum": "Luminance range", "couleur": "Color range",
    "degrade": "Linear gradient", "pinceau": "Brush",
}
DIRECTIONS = ["Top → bottom", "Bottom → top", "Left → right", "Right → left"]


def new_part(src: str, first: bool) -> dict:
    p = {"id": new_id(), "src": src, "op": "add"}
    if src == "etoiles":
        p.update(taille=4.0)
    elif src == "lum":
        p.update(lo=40.0, hi=100.0)
    elif src == "couleur":
        p.update(teinte=0.0, largeur=40.0)
    elif src == "degrade":
        p.update(direction=DIRECTIONS[0], position=50.0, douceur=40.0)
    elif src == "pinceau":
        p.update(strokes=[], size=60.0, hardness=50.0, paint=True)
    return p


def new_mask() -> dict:
    return {"parts": [], "feather": 0.0, "density": 100.0, "invert": False}


def new_workflow_layer(pid: int, name: str, visible=False, part=None) -> dict:
    return {"id": new_id(), "kind": "workflow", "pid": pid, "part": part, "name": name, "visible": visible,
            "opacity": 100.0, "mode": "Normal", "sliders": {}, "mask": None}


def new_filter_layer(key: str, target=None) -> dict:
    """A filter layer. target: the id of the one layer it applies to (it sits just above it, as a Photoshop
    clipping mask); None: it applies to everything below it."""
    f = FILTERS[key]
    return {"id": new_id(), "kind": "filter", "filter": key, "name": f["name"], "visible": True, "opacity": 100.0,
            "mode": "Normal", "params": {q["key"]: q["default"] for q in f["params"]}, "mask": None,
            "target": target}


def normalize(layers: list[dict]) -> list[dict]:
    """Each layer's own filters go just above it, in their order; a filter whose layer is gone applies to everything
    below. The list is top first."""
    ids = {l["id"] for l in layers if l["kind"] == "workflow"}
    for l in layers:
        if l["kind"] == "filter" and l.get("target") not in ids:
            l["target"] = None
    out = []
    for l in layers:
        if l["kind"] == "filter" and l.get("target"):
            continue
        if l["kind"] == "workflow":
            out += [f for f in layers if f["kind"] == "filter" and f.get("target") == l["id"]]
        out.append(l)
    return out


# ======================================================================== the composition
class Ctx:
    """What a filter or a mask needs, at one resolution and for one region."""

    def __init__(self, eng, level, sl):
        self.eng, self.level, self.sl = eng, level, sl
        self.f = 2 ** level

    def gray(self, rel):
        return to_f(self.eng.store.get(rel, self.level, gray=True)[self.sl])

    def stars(self):
        return self.gray(self.eng.s.data["masks"]["etoiles"]["file"])

    def halpha(self):
        return self.gray(self.eng.s.data["masks"]["halpha"]["file"])


class Engine:
    def __init__(self, session):
        self.s = session
        self.store = Store()
        self.layers: list[dict] = []
        self._layer_cache = {}
        self._brush = {}                                   # (part id, level) -> (n strokes, raster)

    def path(self, pid):
        return next((q for q in self.s.paths if q.pid == pid), None)

    def size(self, level):
        W, H = self.s.data["size"]
        f = 2 ** level
        return -(-W // f), -(-H // f)

    # ------------------------------------------------------------ one workflow layer
    def layer_rgb(self, lay, level, sl) -> np.ndarray:
        sig = (lay["id"], level, sl if level == 0 else None, lay["part"],
               json.dumps(lay["sliders"], sort_keys=True), self.path(lay["pid"]).layer)
        hit = self._layer_cache.get(lay["id"])
        if hit is not None and hit[0] == sig:
            return hit[1]
        p = self.path(lay["pid"])
        info = self.s.layers[p.layer]
        get = lambda rel: to_f(self.store.get(rel, level)[sl])           # noqa: E731
        base = get(info["base"])
        out = base.copy()
        for k, pos in lay["sliders"].items():
            files = info["sliders"].get(k)
            if abs(pos - 2) < 1e-3 or not files or not self.s.slider_state(p, k)[0]:
                continue
            lo, hi = int(math.floor(pos)), int(math.ceil(pos))
            t = pos - lo
            a = get(files[lo]) if lo != 2 else base
            img = a if hi == lo else a + t * ((get(files[hi]) if hi != 2 else base) - a)
            out += img - base
        if lay["part"] in ("etoiles", "fond") and "nombre" in info["sliders"]:
            starless = get(info["sliders"]["nombre"][0])
            out = np.clip(out - starless, 0, 1) if lay["part"] == "etoiles" else starless
        self._layer_cache[lay["id"]] = (sig, out)
        return out

    # ------------------------------------------------------------ masks
    def part_raster(self, part, lay, level, sl, below) -> np.ndarray:
        ctx = Ctx(self, level, sl)
        f = ctx.f
        src = part["src"]
        if src == "etoiles":
            m = ctx.stars()
            r = int(round(part["taille"] / f))
            if r > 0:
                m = ndimage.grey_dilation(m, size=(2 * r + 1, 2 * r + 1))
                m = gauss(m, r / 2)
            return np.clip(m * 1.5, 0, 1)
        if src == "halpha":
            return ctx.halpha()
        if src == "lum":
            L = lum(np.clip(below, 0, 1))
            lo, hi, soft = part["lo"] / 100, part["hi"] / 100, 0.06
            return np.clip((L - (lo - soft)) / (2 * soft), 0, 1) * np.clip(((hi + soft) - L) / (2 * soft), 0, 1)
        if src == "couleur":
            h, sat = hue(np.clip(below, 0, 1))
            return hue_weight(h, part["teinte"], part["largeur"]) * np.clip(sat * 4, 0, 1)
        if src == "degrade":
            W, H = self.size(level)
            ys = np.arange(sl[0].start, sl[0].stop, dtype=np.float32)[:, None] / H
            xs = np.arange(sl[1].start, sl[1].stop, dtype=np.float32)[None, :] / W
            ys, xs = np.broadcast_arrays(ys, xs)
            t = dict(zip(DIRECTIONS, (ys, 1 - ys, xs, 1 - xs)))[part["direction"]]
            pos, soft = part["position"] / 100, max(part["douceur"] / 100, 0.01)
            return np.clip((t - (pos - soft / 2)) / soft, 0, 1).astype(np.float32)
        if src == "pinceau":
            return self.brush_raster(part, level)[sl]
        raise ValueError(src)

    def brush_raster(self, part, level) -> np.ndarray:
        key = (part["id"], level)
        n = len(part["strokes"])
        hit = self._brush.get(key)
        if hit is not None and hit[0] == n:
            return hit[1]
        if hit is not None and hit[0] < n:
            r, start = hit[1], hit[0]
        else:
            W, H = self.size(level)
            r, start = np.zeros((H, W), np.float32), 0
        f = 2 ** level
        for x, y, rad, hard, paint in part["strokes"][start:]:
            self.stamp(r, x / f, y / f, rad / f, hard, paint)
        self._brush[key] = (n, r)
        return r

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

    def mask(self, lay, level, sl, below) -> np.ndarray | None:
        m = lay["mask"]
        if m is None:
            return None
        if not m["parts"]:                                 # a new mask shows the whole layer, as in Photoshop
            return self._finish(np.ones(below.shape[:2], np.float32), m, level)
        out = None
        for part in m["parts"]:
            r = self.part_raster(part, lay, level, sl, below)
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
    def compose(self, level, x0=0, y0=0, x1=None, y1=None, upto=None, want_mask=None):
        """The composite (float 0..1) at a resolution, for a region; `upto` stops below that layer index (for the
        before/after); `want_mask` also returns the mask of that layer id."""
        W, H = self.size(level)
        x1 = W if x1 is None else x1
        y1 = H if y1 is None else y1
        pad = 0 if level else 64
        ax0, ay0, ax1, ay1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
        sl = (slice(ay0, ay1), slice(ax0, ax1))
        comp = np.zeros((ay1 - ay0, ax1 - ax0, 3), np.float32)
        got_mask = None
        stack = list(reversed(self.layers))                                # bottom first
        if upto is not None:
            stack = stack[:upto]
        for lay in stack:
            if lay["kind"] == "filter" and lay.get("target"):
                continue                                   # applied with its own layer, below
            if not lay["visible"]:
                if want_mask == lay["id"]:
                    got_mask = self.mask(lay, level, sl, comp)
                continue
            if lay["kind"] == "workflow":
                if self.path(lay["pid"]) is None:
                    continue
                src = self.layer_rgb(lay, level, sl)
                for fl in stack:                           # this layer's own filters, bottom first
                    if fl["kind"] != "filter" or fl.get("target") != lay["id"]:
                        continue
                    if not fl["visible"]:
                        if want_mask == fl["id"]:
                            got_mask = self.mask(fl, level, sl, src)
                        continue
                    src, fm = self._apply(fl, src, level, sl)
                    if want_mask == fl["id"]:
                        got_mask = fm
                m = self.mask(lay, level, sl, src)
                if want_mask == lay["id"]:
                    got_mask = m
                src = self.blend(lay["mode"], comp, src)
                a = lay["opacity"] / 100
                if m is None and a >= 0.999:
                    comp = src
                else:
                    a = a if m is None else (a * m)[..., None]
                    comp = comp + (src - comp) * a
            else:
                comp, m = self._apply(lay, comp, level, sl)
                if want_mask == lay["id"]:
                    got_mask = m
        cut = (slice(y0 - ay0, y0 - ay0 + (y1 - y0)), slice(x0 - ax0, x0 - ax0 + (x1 - x0)))
        out = np.clip(comp[cut], 0, 1)
        if want_mask is not None:
            return out, (None if got_mask is None else got_mask[cut])
        return out

    def _apply(self, fl, under, level, sl):
        """A filter layer on an image (its own layer's, or everything below): the filtered image, through the
        layer's mask, opacity and mode. Returns (result, mask)."""
        out = FILTERS[fl["filter"]]["fn"](under, fl["params"], Ctx(self, level, sl))
        m = self.mask(fl, level, sl, under)
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
            return 1 - (1 - a) * (1 - np.clip(b, 0, 1))
        if mode == "Multiply":
            return a * b
        if mode == "Soft light":
            b = np.clip(b, 0, 1)
            return (1 - 2 * b) * a * a + 2 * b * a
        return b

    def snapshot(self):
        return copy.deepcopy(self.layers)
