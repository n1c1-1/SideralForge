"""The simplified sliders of Post (POST.md 2.1): 9 continuous sliders from 0 to 4, position 2 = the layer's base.

A workflow layer is rendered with settings derived from its base (a named rendering, a preset, the suggestion) and
its slider positions:

* engine layers (render_prep, render_prep_learned): each slider moves engine settings. The table gives their values
  at positions 0-4 for the default rendering; for another base the table's *step* from position 2 is added (sky_level,
  faint_smooth, g_lim, d10, d7) or its *ratio* to position 2 multiplies the base (lc_*, chroma, neb_sharp, fine_k,
  star_sat, star_glow). Between integer positions the value is linear. Values stay inside the settings' bounds;
* simple layers (render_simple): the tables of the classic layers of mockup/alt2/build/prepare.py, on the simple
  rendering's settings, with the same rule (the defaults of ``SimpleSettings`` are those tables' position 2).

A slider the layer cannot offer is greyed with its reason (``availability``); its position is then ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .settings import RenderSettings
from .simple import SimpleSettings

#: the kinds of workflow layers: engine (with Stars), simple (without Stars), stacking (pulled from Stacking)
KINDS = ("engine", "simple", "stacking")
#: settings moved by a step (the others by a ratio)
ADDITIVE = frozenset({"sky_level", "faint_smooth", "g_lim", "d10", "d7", "sky", "contrast", "noise_mix", "smooth_px"})
SUGGESTED = 2.0
LO, HI = 0.0, 4.0

# the reasons of the greyed sliders (the mockup's texts, so that its French translations apply)
WHY_STARS = "the star sliders need the Stars block"
WHY_DETAIL = "sharpening needs the starless image (Stars block)"
WHY_DENOISE = "noise smoothing needs the Denoise block"
WHY_EXIT_STARS = "this layer leaves the graph before the Stars block"
WHY_EXIT_DENOISE = "this layer leaves the graph before the Denoise block"


@dataclass(frozen=True)
class Slider:
    key: str
    name: str
    group: str
    ends: tuple[str, str]
    #: engine setting -> its values at positions 0, 1, 2 (the default rendering), 3, 4
    engine: dict[str, tuple[float, ...]]
    #: simple rendering setting -> its values at positions 0-4 (empty: the slider needs Stars)
    simple: dict[str, tuple[float, ...]] = field(default_factory=dict)
    #: what the workflow needs for this slider: "" | "stars" | "denoise"
    needs: str = ""


NEBULA, STARS = "Nebula and background", "Stars"
SLIDERS: tuple[Slider, ...] = (
    Slider("sky", "Sky", NEBULA, ("darker", "brighter"),
           {"sky_level": (12, 18, 24, 30, 36)},
           {"sky": (0.10, 0.15, 0.20, 0.26, 0.32)}),
    Slider("contrast", "Contrast", NEBULA, ("softer", "stronger"),
           {"lc_fine": (0.95, 1.0, 1.05, 1.15, 1.3), "lc_medium": (1.0, 1.15, 1.3, 1.55, 1.85),
            "lc_large": (1.0, 1.12, 1.25, 1.5, 1.8)},
           {"contrast": (-0.35, -0.18, 0.0, 0.25, 0.5)}),
    Slider("saturation", "Saturation", NEBULA, ("paler", "more vivid"),
           {"chroma": (0.45, 0.68, 0.88, 1.08, 1.3)},
           {"saturation": (0.6, 0.9, 1.2, 1.5, 1.85)}),
    Slider("detail", "Detail", NEBULA, ("smoother", "sharper"),
           {"neb_sharp": (0.2, 0.55, 0.9, 1.4, 2.0), "fine_k": (0.6, 0.8, 1.0, 1.3, 1.6)}, needs="stars"),
    Slider("noise", "Noise smoothing", NEBULA, ("less", "more"),
           {"faint_smooth": (0, 16, 32, 52, 80)},
           # the denoised classic layer: the raw stack mixed back below 2, faint areas smoothed above
           {"noise_mix": (0.4, 0.7, 1.0, 1.0, 1.0), "smooth_px": (0.0, 0.0, 0.0, 0.8, 1.6)}, needs="denoise"),
    Slider("stars_number", "Number of stars", STARS, ("fewer", "more"),
           {"g_lim": (11, 13, 15, 16.5, 18)}, needs="stars"),
    Slider("star_size", "Star size", STARS, ("smaller", "larger"),
           {"d10": (2.6, 3.5, 4.4, 5.5, 6.8), "d7": (5.4, 7.2, 9.0, 11.2, 14.0)}, needs="stars"),
    Slider("star_color", "Star color", STARS, ("whiter", "more colorful"),
           {"star_sat": (0.15, 0.4, 0.7, 1.0, 1.4)}, needs="stars"),
    Slider("star_halo", "Bright star halo", STARS, ("none", "wider"),
           {"star_glow": (0.0, 0.15, 0.3, 0.5, 0.8)}, needs="stars"),
)
BY_KEY = {s.key: s for s in SLIDERS}
KEYS = tuple(BY_KEY)
#: the mockup's internal keys (the client keeps them, DESIGN.md §7.7)
MOCKUP_KEYS = {"sky": "ciel", "contrast": "contraste", "saturation": "saturation", "detail": "detail",
               "noise": "bruit", "stars_number": "nombre", "star_size": "taille", "star_color": "couleur",
               "star_halo": "halo"}


def _check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown layer kind {kind!r} (allowed: {KINDS})")
    return kind


def availability(kind: str, has_denoise: bool) -> dict[str, str]:
    """Per slider, '' when the layer offers it, else the reason it is greyed. ``kind``: engine, simple or stacking
    (a layer pulled from Stacking leaves the graph before Denoise and Stars)."""
    _check_kind(kind)
    out = {}
    for s in SLIDERS:
        why = ""
        if s.needs == "stars" and kind != "engine":
            why = WHY_EXIT_STARS if kind == "stacking" else (WHY_DETAIL if s.key == "detail" else WHY_STARS)
        elif s.needs == "denoise" and (kind == "stacking" or not has_denoise):
            why = WHY_EXIT_DENOISE if kind == "stacking" else WHY_DENOISE
        out[s.key] = why
    return out


def position_text(pos: float) -> str:
    """'suggested' at 2, else the offset: '+0.7', '−1.2'."""
    d = round(float(pos) - SUGGESTED, 1)
    if abs(d) < 0.05:
        return "suggested"
    return f"+{d:.1f}" if d > 0 else f"−{-d:.1f}"


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
                has_denoise: bool = True) -> RenderSettings | SimpleSettings:
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
        for name, table in (sl.engine if kind == "engine" else sl.simple).items():
            lo, hi = _bounds(model, name)
            v = value_at(name, table, float(getattr(b, name)), pos)
            vals[name] = min(max(v, lo), hi)
    return model(**vals)


def base_settings(kind: str, base: Any = None) -> RenderSettings | SimpleSettings:
    """A layer's base: None (the suggestion), a preset name (engine layers), a dict of settings (a named rendering of
    the project; unknown settings ignored) or settings."""
    if _check_kind(kind) == "engine":
        if base is None:
            return RenderSettings()
        if isinstance(base, RenderSettings):
            return base
        if isinstance(base, str):
            from .settings import PRESETS
            if base not in PRESETS:
                raise ValueError(f"unknown rendering preset {base!r} (known: {sorted(PRESETS)})")
            return RenderSettings(**PRESETS[base])
        return RenderSettings.from_saved(dict(base))
    if base is None or base == "suggested":
        return SimpleSettings()
    if isinstance(base, SimpleSettings):
        return base
    if isinstance(base, str):
        raise ValueError(f"a simple layer has no preset {base!r}")
    return SimpleSettings(**{k: v for k, v in dict(base).items() if k in SimpleSettings.model_fields})
