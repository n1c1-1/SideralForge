"""The stages of v2 projects (``V2_STAGES``) and the tables the stage keys need.

``V2_STAGES()`` is ``stages.all_stages()`` (the stages v1's ``Pipeline`` runs) followed by the stages only v2
workflows have (``NEW_STAGES``). ``core/workflow.py`` builds the chains on it and ``core/store.py`` keys the results
with the tables below.
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
    "denoise": frozenset({"adaptation"}),                # long when the half-stacks' noise is not dithered
    "starless": frozenset({"learned_stars"}),
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
    "color": frozenset({"correction", "hii_weight", "camera_profile", "synthetic_weight", "profile_tolerance"}),
    "background": frozenset({"sky_order", "large_scale_px"}),
    # the network trained from scratch on each image, replaced by the pretrained one (stages/denoise.py)
    "denoise": frozenset({"iterations", "patch", "batch", "learning_rate", "reuse_model"}),
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
}
#: fields without effect on the results (left out of the keys)
NON_KEY: dict[str, frozenset[str]] = {
    "calibrate": frozenset({"workers"}),
    "register": frozenset({"workers"}),
    "star_net": frozenset({"device"}),
}
#: the stages only v2 workflows have
NEW_STAGES = ("gradient", "render_simple")


def new_stages() -> list[Stage]:
    from .gradient import Gradient
    from .render_simple import RenderSimple
    return [Gradient(), RenderSimple()]


def V2_STAGES() -> list[Stage]:            # noqa: N802 (a table, built on first use: it imports every stage)
    """``all_stages()`` in their order, then ``NEW_STAGES`` (``core/workflow.ORDER`` places them)."""
    from . import all_stages
    return all_stages() + new_stages()
