"""The simplified sliders of Post: continuous sliders from 0 to 4, position 2 = the layer's base.

A workflow layer is rendered with settings derived from its base (a named rendering, the suggestion) and its slider
positions: tables on the rendering's settings (``SimpleSettings``, and ``StarSettings`` for the Stars block's images),
whose position 2 is the default. For another base the table's *step* from position 2 is added (sky, contrast, noise
smoothing, star size) or its *ratio* to position 2 multiplies the base (the others). Between integer positions the
value is linear. Values stay inside the settings' bounds.

The kinds of workflow layers: the Output block's image (simple), one pulled from Stacking (stacking), and the Stars
block's images: the starless image (starless), the stars alone (stars), the two recombined (combined, the Stars and
background block). A slider the layer cannot offer is greyed with its reason (``availability``); its position is then
ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .simple import SimpleSettings
from .stars import StarSettings

#: the kinds of workflow layers
KINDS = ("simple", "stacking", "starless", "stars", "combined")
STAR_KINDS = ("starless", "stars", "combined")
#: settings moved by a step (the others by a ratio)
ADDITIVE = frozenset({"sky", "contrast", "noise_mix", "smooth_px", "star_size"})
SUGGESTED = 2.0
LO, HI = 0.0, 4.0

# the reasons of the greyed sliders (translated by the client's catalogues)
WHY_STARS = "the star sliders work on the stars of the Stars block"
WHY_STARS_ONLY = "this layer holds the stars only"
WHY_DENOISE = "noise smoothing needs the Denoise block"
WHY_WHOLE = "noise smoothing works on the whole image (the Output layer)"
WHY_EXIT_STARS = "this layer leaves the graph before the Stars block"
WHY_EXIT_DENOISE = "this layer leaves the graph before the Denoise block"


@dataclass(frozen=True)
class Slider:
    key: str
    name: str
    group: str
    ends: tuple[str, str]
    #: setting -> its values at positions 0, 1, 2 (the default rendering), 3, 4
    table: dict[str, tuple[float, ...]]
    #: what the layer needs for this slider: "" | "stars" | "denoise"
    needs: str = ""


NEBULA, STARS = "Nebula and background", "Stars"
SLIDERS: tuple[Slider, ...] = (
    Slider("sky", "Sky", NEBULA, ("darker", "brighter"), {"sky": (0.10, 0.15, 0.20, 0.26, 0.32)}),
    Slider("contrast", "Contrast", NEBULA, ("softer", "stronger"), {"contrast": (-0.35, -0.18, 0.0, 0.25, 0.5)}),
    Slider("saturation", "Saturation", NEBULA, ("paler", "more vivid"), {"saturation": (0.6, 0.9, 1.2, 1.5, 1.85)}),
    # the denoised layer: the raw stack mixed back below 2, faint areas smoothed above
    Slider("noise", "Noise smoothing", NEBULA, ("less", "more"),
           {"noise_mix": (0.4, 0.7, 1.0, 1.0, 1.0), "smooth_px": (0.0, 0.0, 0.0, 0.8, 1.6)}, needs="denoise"),
    Slider("star_strength", "Strength", STARS, ("subtler", "bolder"),
           {"star_strength": (0.35, 0.6, 1.0, 1.6, 2.6)}, needs="stars"),
    Slider("star_size", "Star size", STARS, ("smaller", "larger"), {"star_size": (-2.0, -1.0, 0.0, 1.0, 2.0)},
           needs="stars"),
    Slider("star_color", "Star color", STARS, ("whiter", "more colorful"),
           {"star_saturation": (0.4, 0.8, 1.2, 1.7, 2.3)}, needs="stars"),
)
BY_KEY = {s.key: s for s in SLIDERS}
KEYS = tuple(BY_KEY)
#: a group's title on a layer of the Stars block: named after the image it acts on, as the block's outputs are
PART_GROUPS = {NEBULA: "Starless image", STARS: "Stars alone"}


def group_title(group: str, part: str | None) -> str:
    """The title of a group of sliders on a layer (``part``: one of the Stars block's images, or None)."""
    return PART_GROUPS.get(group, group) if part else group


def _check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown layer kind {kind!r} (allowed: {KINDS})")
    return kind


def availability(kind: str, has_denoise: bool) -> dict[str, str]:
    """Per slider, '' when the layer offers it, else the reason it is greyed. ``kind``: one of ``KINDS`` (a layer
    pulled from Stacking leaves the graph before Denoise and Stars)."""
    _check_kind(kind)
    out = {}
    for s in SLIDERS:
        why = ""
        if s.needs == "stars":
            if kind not in ("stars", "combined"):
                why = WHY_EXIT_STARS if kind == "stacking" else WHY_STARS
        elif kind == "stars":
            why = WHY_STARS_ONLY
        elif s.needs == "denoise":
            if kind == "stacking":
                why = WHY_EXIT_DENOISE
            elif kind in STAR_KINDS:
                why = WHY_WHOLE
            elif not has_denoise:
                why = WHY_DENOISE
        out[s.key] = why
    return out


def _bounds(model, name: str) -> tuple[float, float]:
    lo, hi = float("-inf"), float("inf")
    for m in model.model_fields[name].metadata:
        for attr in ("ge", "gt"):
            if getattr(m, attr, None) is not None:
                lo = max(lo, float(getattr(m, attr)))
        for attr in ("le", "lt"):
            if getattr(m, attr, None) is not None:
                hi = min(hi, float(getattr(m, attr)))
    return lo, hi


def value_at(name: str, table: tuple[float, ...], base: float, pos: float) -> float:
    """A setting at a slider position: the base at 2, the table's step added or its ratio applied elsewhere, linear
    between integer positions. A ratio on a base of 0 would freeze the slider: the step is added instead."""
    pos = min(max(float(pos), LO), HI)

    def at(i: int) -> float:
        if i == 2:
            return base
        if name in ADDITIVE or base == 0 or table[2] == 0:
            return base + table[i] - table[2]
        return base * table[i] / table[2]
    i0 = min(int(pos), 3)
    t = pos - i0
    return at(i0) + t * (at(i0 + 1) - at(i0)) if t else at(i0)


def settings_at(kind: str, base: Any = None, positions: dict[str, float] | None = None,
                has_denoise: bool = True) -> SimpleSettings:
    """The settings of a workflow layer: its base (``base_settings``) moved by the slider positions. Positions of
    unknown or greyed sliders are ignored."""
    b = base_settings(kind, base)
    model = type(b)
    ok = availability(kind, has_denoise)
    vals = b.model_dump()
    for key, pos in (positions or {}).items():
        sl = BY_KEY.get(key)
        if sl is None or ok[key] or abs(float(pos) - SUGGESTED) < 1e-9:
            continue
        for name, table in sl.table.items():
            lo, hi = _bounds(model, name)
            v = value_at(name, table, float(getattr(b, name)), pos)
            vals[name] = min(max(v, lo), hi)
    return model(**vals)


def base_settings(kind: str, base: Any = None) -> SimpleSettings:
    """A layer's base: None (the suggestion), a dict of settings (a named rendering of the project; unknown settings
    ignored) or settings. The Stars block's images have the star settings too (``StarSettings``)."""
    model = StarSettings if _check_kind(kind) in STAR_KINDS else SimpleSettings
    if base is None or base == "suggested":
        return model()
    if isinstance(base, SimpleSettings):
        return model(**{k: v for k, v in base.model_dump().items() if k in model.model_fields})
    if isinstance(base, str):
        raise ValueError(f"a layer has no preset {base!r}")
    return model(**{k: v for k, v in dict(base).items() if k in model.model_fields})
