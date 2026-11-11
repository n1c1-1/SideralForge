"""The Calibration matching lines of a session (Inspect, Sources): each session's own calibration frames, masters made by
another program said as used as they are, and what the frames contradict by the Calibration's own rules said as a
warning (a session of the night test pass: a dark "at 0 ADU for a black of 512" had a green tick, three sessions of one
flat each said "3 frames", a master flat on the 0-1 scale "5 % of saturation")."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from astropy.io import fits  # noqa: E402

from astro_gpu.client.services.inspect import project_frames  # noqa: E402
from astro_gpu.client.sources import SourcesView  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

H, W = 24, 32


def _fits(path: Path, a, **hdr) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.Header()
    for k, v in {"BAYERPAT": "RGGB", **hdr}.items():
        h[k] = v
    fits.PrimaryHDU(np.asarray(a), h).writeto(path, overwrite=True)
    return path


def _project(root: Path, nights: int, dark_level: float, flat, flat_hdr: dict) -> project2.Project2:
    """``nights`` sessions of 16-bit lights of a 14-bit camera (black 512, white 16383), each with one flat of its own;
    one master dark and one master bias on the 0-1 scale for all of them."""
    p = project2.create(root / "P", "P", Target(name="NGC 7000"))
    cal = root / "cal"
    dark = _fits(cal / "dark.fit", np.full((H, W), dark_level / 65535, np.float32), EXPTIME=300.0, STACKCNT=16)
    bias = _fits(cal / "bias.fit", np.full((H, W), 509 / 65535, np.float32), STACKCNT=50)
    for i in range(nights):
        lights = root / f"L{i}"
        for j in range(2):
            _fits(lights / f"l{j}.fits", np.full((H, W), 900, np.uint16), EXPTIME=300.0, PEDESTAL=512, DATAMAX=16383,
                  **{"DATE-OBS": f"2026-06-1{i}T23:0{j}:00"})
        f = _fits(root / f"F{i}" / "flat.fit", flat, EXPTIME=0.05, **flat_hdr)
        p.sessions.append(project2.Session(i + 1, f"night {i + 1}", lights=[str(lights)], darks=[str(dark)],
                                           flats=[str(f)], bias=[str(bias)]))
    project2.save(p)
    return p


def _lines(p, sid: int) -> list[tuple]:
    d = project_frames(p, None)
    view = SimpleNamespace(data=d, s=SimpleNamespace(frames_excluded=[]),
                           _row=lambda: next(r for r in d.sessions if r.id == sid),
                           black=lambda: float(d.camera.get("black") or 0))
    return SourcesView.checks(view)


def test_masters_of_another_program_are_said_used_as_they_are_with_each_session_s_own_flat(tmp_path):
    p = _project(tmp_path, 3, 508.0, np.full((H, W), 8400 / 65535, np.float32), {"STACKCNT": 30})
    lines = _lines(p, 2)
    assert (True, "Darks: median level 508 ADU, nominal black level 512 ADU.") in lines        # in ADU, not 0
    assert (None, "Darks: a master of 16 frames, used as it is.") in lines
    assert (None, "Bias: a master of 50 frames, used as it is.") in lines
    # 8400 ADU over the 14-bit camera's range: 53 %, never 13 % (0.13 of a white level of 1, or of 65535)
    assert (None, "Flats: a master of 30 frames at 53 % of saturation, exposure 1/20 s: already calibrated, used "
                  "as it is.") in lines, lines
    assert not any("3 frames" in t or "corrected with" in t for _ok, t in lines)          # a master flat: nothing off


def test_darks_without_their_bias_and_flats_under_the_bias_are_warnings(tmp_path):
    flat = np.full((H, W), 3000, np.uint16)
    flat[:9, :9] = 100                                      # one flat frame whose bias was taken off already
    p = _project(tmp_path, 2, 4.0, flat, {})
    lines = _lines(p, 1)
    assert (False, "Darks: median level 4 ADU, far under the bias (509 ADU): they will not take it off the "
                   "lights.") in lines, lines
    assert not any(ok is True and t.startswith("Darks: median level") for ok, t in lines)
    assert (None, "Flats: 1 frame at 16 % of saturation, exposure 1/20 s.") in lines                 # its own: one
    assert (False, "Flats: their lowest levels (100 ADU) are under the bias (509 ADU): their bias was taken off "
                   "already, and would be taken off again.") in lines
