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
    "stack": frozenset({"scale", "pixfrac", "min_transparency", "half_block", "read_noise", "gain"}),
    "astrometry": frozenset({"ra", "dec", "target", "pixel_um", "focal_mm"}),
    "color": frozenset({"aperture"}),
    "denoise": frozenset({"iterations"}),
    "starless": frozenset({"learned_stars"}),
    "background": frozenset({"large_scale_px"}),
}
#: settings whose default changed after results had been keyed under the previous one: {stage: {field: the previous
#: default}}. The key holds such a field's effective value unless it is the previous default (core/store.py), so a
#: definition that stored the previous default keeps its key and one that leaves the field unset gets a new one
KEY_BASELINE: dict[str, dict[str, object]] = {
    "stack": {"selection": "thresholds"},
}
#: settings removed from a stage: a definition that still stores one is read without it
RETIRED: dict[str, frozenset[str]] = {
    "stack": frozenset({"rejection", "weighting", "normalisation", "scale_rule"}),
    "register": frozenset({"channel_align", "fwhm_method"}),
}
#: removed settings whose last value stays in every key, so that the results computed with it keep their keys:
#: {stage: {field: value}} (the Moffat star measure and the drizzle coverage rule are now the only behaviour)
KEY_CONSTANTS: dict[str, dict[str, object]] = {
    "stack": {"scale_rule": "coverage"},
    "register": {"fwhm_method": "moffat"},
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
