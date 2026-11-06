"""The Colour Mixer: hue, saturation and luminance of each colour range (issue #23).

Eight bands around the CIE L*c*h* hue circle (red, orange, yellow, green, aqua, blue, purple, magenta), each with a
hue shift, a saturation and a luminance setting, as the colour mixers of photo editors (their "HSL" panel) and the
hue-dependent saturation of the astronomy software; computed in CIE L*c*h* (CIE 1976, D65) rather than HSV, which
mixes lightness and colour (the astronomy software's documentation warns that HSV hue changes are colorimetrically
unsound). Each pixel takes from each band a share that falls smoothly with its distance to the band's hue (a raised
cosine reaching zero at the next band), and nothing where it has no colour (its chroma under NEUTRAL_C): the grey
background and the white star cores are left alone. The uses in astrophotography: the weak teal of OIII against the
dominant red in a dual-band image, the magenta of stars and halos in a narrowband palette, an H-alpha too pink.
"""
from __future__ import annotations

import numpy as np

from .colour import from_lab, to_lab

#: the bands: their names and their hue on the CIE L*c*h* circle (degrees; the hues of the sRGB primaries and of
#: their mixtures: red 40, yellow 102, green 136, cyan 196, blue 306, magenta 328)
BANDS = (("red", 30.0), ("orange", 65.0), ("yellow", 100.0), ("green", 136.0), ("aqua", 196.0), ("blue", 262.0),
         ("purple", 306.0), ("magenta", 335.0))
#: under this chroma a pixel has no colour of its own: the mixer leaves it (a smooth start up to twice it)
NEUTRAL_C = 4.0
#: the settings' ends: a hue shift in degrees, saturation and luminance in percent
HUE_MAX, SAT_MAX, LUM_MAX = 30.0, 100.0, 100.0


def keys() -> list[str]:
    """The settings, band by band: hue_<band>, sat_<band>, lum_<band>."""
    return [f"{what}_{name}" for name, _h in BANDS for what in ("hue", "sat", "lum")]


def is_identity(p: dict) -> bool:
    return all(abs(float(p.get(k, 0.0))) < 1e-9 for k in keys())


def _weights(h: np.ndarray) -> np.ndarray:
    """(n bands, ...) the share of each band in each pixel of hue ``h`` (degrees): a raised cosine from the band's
    hue to its neighbours' (the shares add up to 1 around the circle)."""
    centers = np.array([c for _n, c in BANDS])
    n = len(centers)
    out = np.zeros((n,) + h.shape, np.float32)
    for i in range(n):
        c, lo, hi = centers[i], centers[i - 1], centers[(i + 1) % n]
        d = (h - c + 180.0) % 360.0 - 180.0                       # signed distance to the band's hue
        left = (c - lo) % 360.0
        right = (hi - c) % 360.0
        span = np.where(d < 0, left, right)
        t = np.clip(np.abs(d) / span, 0, 1)
        out[i] = 0.5 * (1 + np.cos(np.pi * t))
    return out


def mix(x: np.ndarray, p: dict) -> np.ndarray:
    """``x`` (h, w, 3, sRGB 0..1) through the mixer's settings ``p`` (``keys``); the identity at zero."""
    if is_identity(p):
        return x
    lab = to_lab(np.clip(x, 0, 1))
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    w = _weights(h) * np.clip((C - NEUTRAL_C) / NEUTRAL_C, 0, 1)[None]   # nothing on what has no colour
    dh = np.zeros_like(h, dtype=np.float32)
    ds = np.zeros_like(h, dtype=np.float32)
    dl = np.zeros_like(h, dtype=np.float32)
    for i, (name, _c) in enumerate(BANDS):
        dh += w[i] * np.clip(float(p.get(f"hue_{name}", 0.0)), -HUE_MAX, HUE_MAX)
        ds += w[i] * np.clip(float(p.get(f"sat_{name}", 0.0)), -SAT_MAX, SAT_MAX) / 100.0
        dl += w[i] * np.clip(float(p.get(f"lum_{name}", 0.0)), -LUM_MAX, LUM_MAX) / 100.0
    h2 = np.radians(h + dh)
    C2 = C * np.clip(1.0 + ds, 0.0, 3.0)
    # the luminance of a colour: up or down by its share of the way to white or black (a colourful pixel only)
    L2 = np.where(dl >= 0, L + (100.0 - L) * dl * 0.5, L * (1.0 + dl * 0.5))
    out = np.stack([L2, C2 * np.cos(h2), C2 * np.sin(h2)], -1).astype(np.float32)
    return np.clip(from_lab(out), 0, 1).astype(np.float32)
