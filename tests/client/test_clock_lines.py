"""A session whose frames' own date contradicts the date their file names say (a camera whose clock was never set):
one warning among its matching lines in Inspect and in the checks before a run, from the headers already read, in
English and in French; nothing when the names hold no date."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from astropy.io import fits  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.frames import check_lines  # noqa: E402
from astro_gpu.client.services.inspect import project_frames  # noqa: E402
from astro_gpu.client.sources import SourcesView  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

EN = ("Dates: the camera says 1 Jan 2014, the file names say 17 Jun 2025: the camera's clock was probably never set. "
      "The star positions are computed for the camera's date.")
FR = ("Dates : l'appareil indique le 1 janv. 2014, les noms des fichiers le 17 juin 2025 : son horloge n'a sans doute "
      "jamais été réglée. Les positions des étoiles sont calculées pour la date de l'appareil.")


def _project(root: Path, names: list[str], date_obs: str | None = "2014-01-01T00:16:00") -> project2.Project2:
    p = project2.create(root / "P", "P", Target(name="M 8"))
    lights = root / "L"
    lights.mkdir(parents=True)
    later = datetime(2026, 6, 17, 12).timestamp()
    for n in names:                                       # the camera's own date: its clock was never set
        h = fits.Header()
        h["BAYERPAT"], h["EXPTIME"] = "RGGB", 120.0
        if date_obs:
            h["DATE-OBS"] = date_obs
        fits.PrimaryHDU(np.full((8, 8), 900, np.uint16), h).writeto(lights / n)
        os.utime(lights / n, (later, later))              # copied a year after the night
    p.sessions.append(project2.Session(1, "night 1", lights=[str(lights)]))
    project2.save(p)
    return p


def _lines(d) -> list[tuple]:
    view = SimpleNamespace(data=d, s=SimpleNamespace(frames_excluded=[]), _row=lambda: d.sessions[0],
                           black=lambda: float(d.camera.get("black") or 0))
    return SourcesView.checks(view)


@pytest.fixture
def english():
    i18n.set_language("en")
    yield
    i18n.set_language("en")


def test_the_session_says_the_camera_clock_and_the_checks_once(tmp_path, english):
    p = _project(tmp_path, [f"Light_M8_120.0s_ISO800_2025-06-17_23-4{j}-58.fits" for j in range(3)])
    d = project_frames(p, None)
    assert d.sessions[0].clock == {"camera": "2014-01-01", "names": "2025-06-17"}
    assert (False, EN) in _lines(d)
    assert [t for _lv, t in check_lines(d) if "clock" in t] == ["night 1: " + EN]       # once, never per frame
    i18n.set_language("fr")
    assert (False, FR) in _lines(d)
    assert [t for _lv, t in check_lines(d) if "horloge" in t] == ["night 1 : " + FR]


def test_names_without_a_date_say_nothing(tmp_path, english):
    d = project_frames(_project(tmp_path, ["DSC01234.fits", "DSC01235.fits"]), None)
    assert d.sessions[0].clock is None
    assert not any("Dates" in t for _ok, t in _lines(d))
    assert not any(c.key == "clock" for c in d.checks)


def test_a_date_on_disk_is_never_said_to_be_a_wrong_camera_clock(tmp_path, english):
    """Frames without a date of their own (no DATE-OBS), named by date, copied a year later: their time is the file's,
    no camera's clock to doubt."""
    d = project_frames(_project(tmp_path, [f"Light_M8_2025-06-17_23-4{j}-58.fits" for j in range(2)], None), None)
    assert d.sessions[0].first.startswith("2026-06-17")      # what the engine uses stays as it was
    assert d.sessions[0].clock is None
    assert not any("Dates" in t for _ok, t in _lines(d))
    assert not any(c.key == "clock" for c in d.checks)
