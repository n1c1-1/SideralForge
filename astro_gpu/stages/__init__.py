"""Processing stages, in execution order (each stage only depends on stages listed before it)."""
from __future__ import annotations

from ..core.stage import Stage


def all_stages() -> list[Stage]:
    from .astrometry import Astrometry
    from .background import Background
    from .calibrate import Calibrate
    from .catalogs import Catalogs
    from .color import Color
    from .denoise import Denoise
    from .inventory import Inventory
    from .optics import Optics
    from .reference import Reference
    from .register import Register
    from .render_prep import RenderPrep, RenderPrepLearned
    from .stack import Stack
    from .starless import Starless
    from .stars import Stars
    from .surveys import Surveys
    return [Inventory(), Calibrate(), Register(), Stack(), Astrometry(), Reference(), Catalogs(), Surveys(), Color(),
            Denoise(), Optics(), Stars(), Starless(), Background(), RenderPrep(), RenderPrepLearned()]
