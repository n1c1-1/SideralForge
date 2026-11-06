"""Issue #38: the pixel size added to a project's target counts in the astrometry's key only when it is set, so that
the projects made before it keep their keys (their computed results)."""
from __future__ import annotations

from dataclasses import asdict

from astro_gpu.core import store
from astro_gpu.core.project import Target


def test_an_unset_pixel_size_leaves_the_astrometry_key_as_before():
    class P:
        target = Target(name="IC 1805", ra=40.76, dec=61.11, focal_mm=272.0)

    sp = store.Keys.__new__(store.Keys)
    sp.project = P()
    part = sp.project_part("astrometry")
    old = {k: float(v) if isinstance(v, (int, float)) else v for k, v in asdict(P.target).items() if k != "pixel_um"}
    assert part == old
    P.target = Target(name="IC 1805", ra=40.76, dec=61.11, focal_mm=272.0, pixel_um=5.36)
    assert sp.project_part("astrometry")["pixel_um"] == 5.36
