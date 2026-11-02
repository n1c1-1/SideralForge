"""The Curves tool: PixInsight's CurvesTransformation, Siril 1.4's Curves Transformation, Photoshop's Curves.

* **Curves**, one per channel, each a list of control points (x, y) in [0, 1], sorted by x, 2 to 100 of them (Siril's
  limits). Outside its end points a curve is flat (Photoshop: the end points can move inwards; PixInsight keeps them at
  0 and 1). Outputs are clipped to [0, 1].
* **Interpolation**, as in Siril and PixInsight: Akima (the default; H. Akima, "A new method of interpolation and smooth
  curve fitting based on local procedures", J. ACM 17(4), 1970: the slope at a point is
  (|m(i+1) - m(i)| m(i-1) + |m(i-1) - m(i-2)| m(i)) / (|m(i+1) - m(i)| + |m(i-1) - m(i-2)|), the end slopes extended
  as Akima's paper does; two points give a line), the natural cubic spline (zero second derivative at both ends), or
  linear.
* **Channels**, applied in this order (Siril's, then the hue): RGB/K (the same curve on R, G and B), R, G, B, L* (CIE
  L*/100, a* and b* kept: PixInsight's and Siril's L), S (the HSV saturation, the original CIE L* put back: PixInsight's
  "HSVL*" S curve), c* (CIE chroma c*/128, hue and lightness kept: Siril's C, PixInsight's c), H (CIE hue h/360, the
  lightness and the chroma kept: PixInsight's H).
* **Speed**: while a point is dragged (a draft) each curve is a 4096-entry table; on release it is evaluated exactly
  (the piecewise polynomial of each pixel's value).
* **Presets**: Linear, the nine of Photoshop (their points read from the .acv files Photoshop 2026 ships in
  Presets/Curves), and four astrophotography classics built from documented functions; the user's own presets, kept
  in a file beside the settings; Photoshop's .acv files read and written (Adobe Photoshop File Formats Specification,
  "Curves file format": a version, a count, then for each curve its points as (output, input) in 0-255).
"""
from __future__ import annotations

import copy
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch

from . import colour

CHANNELS = ["RGB/K", "R", "G", "B", "L*", "S", "c*", "H"]
#: what each channel is, said in the editor (English; translated where shown)
CHANNEL_TIPS = {
    "RGB/K": "The same curve on the red, green and blue channels",
    "R": "The red channel", "G": "The green channel", "B": "The blue channel",
    "L*": "Lightness (CIE L*): the colours kept",
    "S": "Saturation (HSV, the lightness kept as CIE L*): PixInsight's S curve",
    "c*": "Chroma (CIE c*): the hue and the lightness kept",
    "H": "Hue (CIE h): input hue to output hue, the lightness and the chroma kept",
}
INTERPS = ["Akima", "Cubic spline", "Linear"]
MAX_POINTS = 100
EPS = 1e-3                                             # the least gap between two points' x
LUT_SIZE = 4096


# ---------------------------------------------------------------------------------------------- the curves
def identity() -> dict:
    return {"pts": [[0.0, 0.0], [1.0, 1.0]], "interp": INTERPS[0]}


def clean_points(raw) -> list[list[float]] | None:
    """Control points made safe: numbers in [0, 1], sorted by x, x at least EPS apart, 2 to MAX_POINTS; None when not
    a curve."""
    pts = []
    for p in raw or []:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            pts.append([min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)])
    pts.sort(key=lambda p: p[0])
    out: list[list[float]] = []
    for p in pts:
        if out and p[0] - out[-1][0] < EPS:
            continue
        out.append(p)
    if len(out) < 2:
        return None
    if len(out) > MAX_POINTS:
        keep = np.linspace(0, len(out) - 1, MAX_POINTS).round().astype(int)
        out = [out[i] for i in keep]
    return out


def clean_curves(raw) -> dict:
    """{channel: {"pts", "interp"}} made safe (unknown channels and broken curves dropped)."""
    out = {}
    if not isinstance(raw, dict):
        return out
    for ch, c in raw.items():
        if ch not in CHANNELS or not isinstance(c, dict):
            continue
        pts = clean_points(c.get("pts"))
        if pts is None:
            continue
        out[ch] = {"pts": pts, "interp": c.get("interp") if c.get("interp") in INTERPS else INTERPS[0]}
    return out


def is_identity(c: dict | None) -> bool:
    """A curve whose points all lie on the diagonal from (0, 0) to (1, 1) is the identity (each interpolation of
    collinear points is that line)."""
    if not c:
        return True
    pts = c["pts"]
    return pts[0][0] <= 1e-9 and pts[-1][0] >= 1 - 1e-9 and all(abs(x - y) < 1e-9 for x, y in pts)


def active(curves: dict) -> list[tuple[str, dict]]:
    """The channels whose curve does something, in the order they are applied."""
    return [(ch, curves[ch]) for ch in CHANNELS if ch in (curves or {}) and not is_identity(curves[ch])]


class Spline:
    """A curve as a piecewise cubic: y = a + b t + c t^2 + d t^3 with t = x - x_i on [x_i, x_i+1]."""

    def __init__(self, pts, interp: str = "Akima"):
        p = np.asarray(pts, np.float64)
        self.xs, self.ys = p[:, 0].copy(), p[:, 1].copy()
        n = len(self.xs)
        h = np.diff(self.xs)
        m = np.diff(self.ys) / h
        a = self.ys[:-1].copy()
        if interp == "Linear" or n == 2:
            b, c, d = m.copy(), np.zeros(n - 1), np.zeros(n - 1)
        elif interp == "Cubic spline":                 # natural: M_0 = M_n-1 = 0
            A = np.zeros((n, n))
            r = np.zeros(n)
            A[0, 0] = A[-1, -1] = 1
            for i in range(1, n - 1):
                A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
                r[i] = 6 * (m[i] - m[i - 1])
            M = np.linalg.solve(A, r)
            b = m - h * (2 * M[:-1] + M[1:]) / 6
            c = M[:-1] / 2
            d = (M[1:] - M[:-1]) / (6 * h)
        else:                                          # Akima 1970
            ext = np.empty(n + 3)
            ext[2:n + 1] = m
            ext[1] = 2 * m[0] - m[1]
            ext[0] = 2 * ext[1] - m[0]
            ext[n + 1] = 2 * m[-1] - m[-2]
            ext[n + 2] = 2 * ext[n + 1] - m[-1]
            t = np.empty(n)
            for i in range(n):                         # m(i) is ext[i + 2]
                w1, w2 = abs(ext[i + 3] - ext[i + 2]), abs(ext[i + 1] - ext[i])
                t[i] = (ext[i + 1] + ext[i + 2]) / 2 if w1 + w2 < 1e-12 else \
                    (w1 * ext[i + 1] + w2 * ext[i + 2]) / (w1 + w2)
            b = t[:-1]
            c = (3 * m - 2 * t[:-1] - t[1:]) / h
            d = (t[:-1] + t[1:] - 2 * m) / h ** 2
        self.coef = np.stack([a, b, c, d])            # (4, n - 1)

    def __call__(self, x) -> np.ndarray:
        """The curve at x (numbers or an array; float64)."""
        x = np.clip(np.asarray(x, np.float64), self.xs[0], self.xs[-1])
        i = np.clip(np.searchsorted(self.xs, x, side="right") - 1, 0, len(self.xs) - 2)
        t = x - self.xs[i]
        a, b, c, d = (k[i] for k in self.coef)
        return np.clip(a + t * (b + t * (c + t * d)), 0, 1)

    def table(self, n: int = LUT_SIZE) -> np.ndarray:
        return self(np.linspace(0, 1, n)).astype(np.float32)

    def map(self, v: np.ndarray, draft: bool = False) -> np.ndarray:
        """The curve of an image's values (float32, any shape): a table lookup for a draft, else exact."""
        t = torch.from_numpy(colour.f32(v))
        if draft:
            lut = torch.from_numpy(self.table())
            idx = t.clamp(0, 1).mul(LUT_SIZE - 1).add_(0.5).long()
            return lut[idx].numpy()
        xs = torch.from_numpy(self.xs.astype(np.float32))
        xc = t.clamp(float(self.xs[0]), float(self.xs[-1]))
        i = torch.bucketize(xc, xs[1:-1].contiguous())
        co = torch.from_numpy(self.coef.astype(np.float32))
        dx = xc - xs[i]
        y = co[3][i].mul_(dx).add_(co[2][i]).mul_(dx).add_(co[1][i]).mul_(dx).add_(co[0][i])
        return y.clamp_(0, 1).numpy()


def spline(c: dict) -> Spline:
    return Spline(c["pts"], c.get("interp", INTERPS[0]))


# ---------------------------------------------------------------------------------------------- the image
def apply(x: np.ndarray, curves: dict, draft: bool = False) -> np.ndarray:
    """Every channel's curve, in CHANNELS' order. The identity (no curve, or curves on the diagonal) returns x."""
    act = active(curves)
    if not act:
        return x
    y = colour.f32(x)
    for ch, c in act:
        s = spline(c)
        if ch == "RGB/K":
            y = s.map(y, draft)
        elif ch in ("R", "G", "B"):
            k = "RGB".index(ch)
            y = y.copy()
            y[..., k] = s.map(np.ascontiguousarray(y[..., k]), draft)
        elif ch == "L*":
            lab = colour.to_lab(y)
            lab[..., 0] = s.map(lab[..., 0] / 100, draft) * 100
            y = colour.from_lab(lab)
        elif ch == "S":
            y = np.clip(y, 0, 1)
            L0 = colour.lightness(y)
            V = y.max(-1)
            S = np.where(V > 1e-7, (V - y.min(-1)) / np.maximum(V, 1e-7), 0).astype(np.float32)
            S2 = s.map(S, draft)
            k = np.where(S > 1e-7, S2 / np.maximum(S, 1e-7), 1.0).astype(np.float32)[..., None]
            y = colour.with_lightness(V[..., None] - (V[..., None] - y) * k, L0)
        else:
            lab = colour.to_lab(y)
            a, b = lab[..., 1], lab[..., 2]
            C = np.hypot(a, b)
            if ch == "c*":
                C2 = s.map(np.clip(C / 128, 0, 1), draft) * 128
                k = np.where(C > 1e-6, C2 / np.maximum(C, 1e-6), 0).astype(np.float32)
                lab[..., 1], lab[..., 2] = a * k, b * k
            else:
                h = (np.degrees(np.arctan2(b, a)) % 360) / 360
                h2 = np.radians(s.map(h.astype(np.float32), draft) * 360)
                lab[..., 1], lab[..., 2] = C * np.cos(h2), C * np.sin(h2)
            y = colour.from_lab(lab)
    return y


def histograms(x: np.ndarray, bins: int = 256, max_px: int = 250_000) -> dict[str, np.ndarray]:
    """The histograms the editor shows under each channel's curve, on a regular subsample of the image: R, G, B, L*,
    S (HSV), c* and H, each ``bins`` counts (float32)."""
    h, w = x.shape[:2]
    k = max(1, int(math.ceil(math.sqrt(h * w / max_px))))
    s = np.ascontiguousarray(np.clip(x[::k, ::k], 0, 1), np.float32)

    def hist(v):
        i = np.clip((v.ravel() * (bins - 1) + 0.5).astype(np.int32), 0, bins - 1)
        return np.bincount(i, minlength=bins).astype(np.float32)

    lab = colour.to_lab(s)
    V = s.max(-1)
    C = np.hypot(lab[..., 1], lab[..., 2])
    out = {"R": hist(s[..., 0]), "G": hist(s[..., 1]), "B": hist(s[..., 2]), "L*": hist(lab[..., 0] / 100),
           "S": hist(np.where(V > 1e-7, (V - s.min(-1)) / np.maximum(V, 1e-7), 0)),
           "c*": hist(np.clip(C / 128, 0, 1))}
    hue = (np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360) / 360
    out["H"] = hist(hue[C > 2.0]) if (C > 2.0).any() else np.zeros(bins, np.float32)   # the coloured pixels only
    return out


# ---------------------------------------------------------------------------------------------- presets
def _ps(*pts) -> list[list[float]]:
    return [[i / 255, o / 255] for i, o in pts]


def _curve(pts, interp="Cubic spline") -> dict:
    return {"pts": pts, "interp": interp}


def _sampled(fn, xs, interp="Akima") -> dict:
    return {"pts": [[float(x), round(float(fn(x)), 4)] for x in xs], "interp": interp}


def _mtf(m, x):
    return (m - 1) * x / ((2 * m - 1) * x - m) if x > 0 else 0.0


def _ghs(p):
    from .tone import ghs_curve
    return lambda x: float(ghs_curve(p, np.array([x]))[0])


PS = "Photoshop"
_GRID = [0, 0.04, 0.12, 0.25, 0.45, 0.7, 1]


def builtin_presets() -> list[dict]:
    """The presets, in the menu's order: {"name", "source", "curves"}. Photoshop's are the points of its .acv files
    (input, output in 0-255), drawn with the natural cubic spline (Photoshop does not document its interpolation)."""
    out = [{"name": "Linear", "source": "The identity: the default of PixInsight, Siril and Photoshop", "curves": {}}]
    ps = [
        ("Medium Contrast", {"RGB/K": _ps((0, 0), (73, 56), (163, 164), (255, 255))}),
        ("Strong Contrast", {"RGB/K": _ps((0, 0), (77, 50), (151, 153), (175, 188), (255, 255))}),
        ("Increase Contrast", {"RGB/K": _ps((0, 0), (38, 17), (212, 231), (231, 250), (255, 255))}),
        ("Linear Contrast", {"RGB/K": _ps((0, 0), (78, 73), (177, 182), (255, 255))}),
        ("Lighter", {"RGB/K": _ps((0, 0), (103, 125), (255, 255))}),
        ("Darker", {"RGB/K": _ps((0, 0), (130, 101), (255, 255))}),
        ("Negative", {"RGB/K": _ps((0, 255), (255, 0))}),
        ("Color Negative", {"R": _ps((33, 255), (119, 127), (185, 0)), "G": _ps((28, 255), (77, 127), (132, 0)),
                            "B": _ps((25, 255), (60, 127), (108, 0))}),
        ("Cross Process", {"R": _ps((0, 0), (64, 40), (128, 125), (175, 190), (255, 255)),
                           "G": _ps((0, 0), (64, 48), (97, 128), (190, 208), (255, 208)),
                           "B": _ps((0, 0), (59, 24), (181, 223), (255, 255))}),
    ]
    for name, cs in ps:
        out.append({"name": name, "source": PS, "curves": {ch: _curve(p) for ch, p in cs.items()}})
    out += [
        {"name": "Lift the midtones", "source": "Histogram transformation, midtones balance 0.35 (PixInsight, Siril)",
         "curves": {"RGB/K": _sampled(lambda x: _mtf(0.35, x), _GRID)}},
        {"name": "Midtone contrast, highlights protected",
         "source": "GHS tutorial (Siril): SP on the midtones, b = 2, HP = 0.85, ln(D+1) = 1",
         "curves": {"RGB/K": _sampled(_ghs(dict(stretch=1.0, local=2.0, symmetry=0.35, highlights=0.85)), _GRID)}},
        {"name": "Brighten, contrast kept",
         "source": "GHS tutorial (Siril): a negative b (-1.4, close to asinh), ln(D+1) = 1",
         "curves": {"RGB/K": _sampled(_ghs(dict(stretch=1.0, local=-1.4, symmetry=0.0)), _GRID)}},
        {"name": "Saturation boost",
         "source": "PixInsight's S curve raised in the middle: the weak colours raised, the strong ones kept",
         "curves": {"S": _curve([[0.0, 0.0], [0.35, 0.47], [1.0, 1.0]], "Akima")}},
    ]
    return out


def preset_curves(name: str, user: list[dict] | None = None) -> dict | None:
    for p in builtin_presets() + (user or []):
        if p["name"] == name:
            return copy.deepcopy(p["curves"])
    return None


# ---------------------------------------------------------------------------------------------- the user's presets
def presets_file() -> Path | None:
    """Where the user's presets are kept: beside the settings file (None under pytest without ASTRO_GPU_SETTINGS:
    then they live in memory only)."""
    from .. import prefs
    f = prefs.settings_file()
    return None if f is None else f.with_name("post_curves_presets.json")


_MEMORY: list[dict] = []


def user_presets(path: Path | None = None) -> list[dict]:
    path = path or presets_file()
    if path is None:
        return copy.deepcopy(_MEMORY)
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for p in raw.get("presets", []) if isinstance(raw, dict) else []:
        if isinstance(p, dict) and isinstance(p.get("name"), str) and p["name"].strip():
            out.append({"name": p["name"].strip(), "source": "", "curves": clean_curves(p.get("curves"))})
    return out


def _write_user(presets: list[dict], path: Path | None):
    path = path or presets_file()
    if path is None:
        _MEMORY[:] = copy.deepcopy(presets)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"presets": [{"name": p["name"], "curves": p["curves"]} for p in presets]}, indent=1),
                   encoding="utf-8")
    tmp.replace(path)


def save_user_preset(name: str, curves: dict, path: Path | None = None) -> None:
    """Keep the curves under a name (a preset of the same name is replaced)."""
    name = name.strip()
    if not name:
        raise ValueError("a preset needs a name")
    ps = [p for p in user_presets(path) if p["name"] != name]
    ps.append({"name": name, "source": "", "curves": clean_curves(curves)})
    _write_user(ps, path)


def delete_user_preset(name: str, path: Path | None = None) -> None:
    _write_user([p for p in user_presets(path) if p["name"] != name], path)


# ---------------------------------------------------------------------------------------------- files
ACV_CHANNELS = ["RGB/K", "R", "G", "B"]


def read_acv(data: bytes) -> dict:
    """A Photoshop .acv file: its RGB, R, G and B curves (the points 0-255; drawn with the natural cubic spline)."""
    if len(data) < 4:
        raise ValueError("not a Photoshop curves file")
    _version, n = struct.unpack(">hh", data[:4])
    o, out = 4, {}
    for k in range(n):
        if o + 2 > len(data):
            raise ValueError("a truncated Photoshop curves file")
        cnt, = struct.unpack(">h", data[o:o + 2])
        o += 2
        if cnt < 2 or o + 4 * cnt > len(data):
            raise ValueError("a damaged Photoshop curves file")
        pts = [struct.unpack(">hh", data[o + 4 * j:o + 4 * j + 4]) for j in range(cnt)]
        o += 4 * cnt
        if k < len(ACV_CHANNELS):
            c = _curve(clean_points([[i / 255, v / 255] for v, i in pts]))
            if not is_identity(c):
                out[ACV_CHANNELS[k]] = c
    return out


def write_acv(curves: dict) -> bytes:
    """The RGB/K, R, G and B curves as a Photoshop .acv file (version 4, five curves as Photoshop writes them; the
    points rounded to 0-255, 19 at most per curve)."""
    blobs = [struct.pack(">hh", 4, 5)]
    for ch in ACV_CHANNELS + [None]:
        c = curves.get(ch) if ch else None
        pts = c["pts"] if c else [[0.0, 0.0], [1.0, 1.0]]
        if len(pts) > 19:
            raise ValueError("a Photoshop curve holds 19 points at most")
        q = sorted({(int(x * 255 + 0.5), int(y * 255 + 0.5)) for x, y in pts})
        blobs.append(struct.pack(">h", len(q)) + b"".join(struct.pack(">hh", o, i) for i, o in q))
    return b"".join(blobs)


def read_preset_file(path) -> dict:
    """Curves from a file: a Photoshop .acv, or a .json of this program (one preset)."""
    path = Path(path)
    if path.suffix.lower() == ".acv":
        return read_acv(path.read_bytes())
    raw = json.loads(path.read_text(encoding="utf-8"))
    return clean_curves(raw.get("curves") if isinstance(raw, dict) and "curves" in raw else raw)


def write_preset_file(path, curves: dict, name: str = "") -> None:
    path = Path(path)
    if path.suffix.lower() == ".acv":
        path.write_bytes(write_acv(curves))
    else:
        path.write_text(json.dumps({"name": name, "curves": clean_curves(curves)}, indent=1), encoding="utf-8")
