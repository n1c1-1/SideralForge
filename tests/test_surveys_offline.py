"""Issue #39: the H-alpha survey is read through its tiles first (each tile cached: any field of a sky already seen
works offline); the cut-out service only when the tiles cannot be had, or for a survey without its own server."""
from __future__ import annotations

import numpy as np

from astro_gpu.stages import surveys


def test_the_tiles_come_first_and_the_cut_out_is_the_fallback(monkeypatch):
    asked = []
    monkeypatch.setattr(surveys, "fetch_hips_tiles", lambda url, *a, **k: (asked.append("tiles"), np.zeros((2, 2)))[1])
    monkeypatch.setattr(surveys, "fetch_reprojected", lambda hips, *a, **k: (asked.append("cut-out"),
                                                                              np.ones((2, 2)))[1])
    assert surveys.fetch_survey("nsns", None, 2, 2, 1.0).sum() == 0 and asked == ["tiles"]

    def down(*a, **k):
        raise OSError("no connection")
    monkeypatch.setattr(surveys, "fetch_hips_tiles", down)
    said = []
    asked.clear()
    assert surveys.fetch_survey("nsns", None, 2, 2, 1.0, lambda m, *k: said.append(m)).sum() == 4
    assert asked == ["cut-out"] and "server" in said[0]
    asked.clear()
    surveys.fetch_survey("finkbeiner", None, 2, 2, 1.0)                     # no server of its own: the cut-out
    assert asked == ["cut-out"]
