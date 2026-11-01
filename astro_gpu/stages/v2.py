"""The stages of v2 projects: v1's stages plus the new ones, and what the stage keys need to know about them.

v1's ``Pipeline`` keeps ``all_stages()``: it never runs the new stages. ``core/workflow.py`` builds the chains on
``V2_STAGES`` (loaded lazily) and ``core/store.py`` keys the results with the tables below (DESIGN.md §1.3, §2.2).
"""
from __future__ import annotations

from ..core.stage import Stage

#: fields an ``auto()`` can set, per stage: never stripped from a key at their static default (a stored value equal to
#: the default overrides the automatic one)
AUTO_FIELDS: dict[str, frozenset[str]] = {
    "calibrate": frozenset({"workers"}),
    "register": frozenset({"workers"}),
    "stack": frozenset({"scale", "min_transparency", "half_block", "read_noise", "gain"}),
    "astrometry": frozenset({"ra", "dec", "target", "pixel_um", "focal_mm"}),
    "color": frozenset({"aperture"}),
    "denoise": frozenset({"iterations"}),
    "starless": frozenset({"learned_stars"}),
    "background": frozenset({"large_scale_px"}),
}
#: fields an ``auto()`` sets only under another setting: {stage: {field: (setting, values that select its rule)}}.
#: Automatic (kept in a key at their default) only where that setting has one of those values; elsewhere a field like
#: any other, dropped at its default (core/store.py). ``stack.pixfrac``: the coverage rule of stacking v2 sets it
AUTO_WHEN: dict[str, dict[str, tuple[str, tuple]]] = {
    "stack": {"pixfrac": ("scale_rule", ("coverage",))},
}
#: fields naming a file: keyed by the file (name, size, modification time), not by the path string
PATH_FIELDS: dict[str, frozenset[str]] = {
    "starless": frozenset({"learned_stars"}),
    "denoise": frozenset({"reuse_model"}),
}
#: fields without effect on the results (left out of the keys)
NON_KEY: dict[str, frozenset[str]] = {
    "calibrate": frozenset({"workers"}),
    "register": frozenset({"workers"}),
    "star_net": frozenset({"device"}),
}
#: the stages v2 adds to v1's
NEW_STAGES = ("background_poly", "render_simple")


def new_stages() -> list[Stage]:
    from .background_poly import BackgroundPoly
    from .render_simple import RenderSimple
    return [BackgroundPoly(), RenderSimple()]


def V2_STAGES() -> list[Stage]:            # noqa: N802 (a table, built on first use: it imports every stage)
    """v1's stages in their order, then the new ones (``core/workflow.ORDER`` places them)."""
    from . import all_stages
    return all_stages() + new_stages()


def auto_fields(stage: str) -> frozenset[str]:
    return AUTO_FIELDS.get(stage, frozenset())


def path_fields(stage: str) -> frozenset[str]:
    return PATH_FIELDS.get(stage, frozenset())


def non_key(stage: str) -> frozenset[str]:
    return NON_KEY.get(stage, frozenset())
