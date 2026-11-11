"""Frames whose own date contradicts the date their file names say: a camera whose clock was never set writes 2014-01-01
00:16 in frames a capture program named after the night of 17 June 2025. Said, never changed: the engine keeps the
camera's date (the night, the stars' epoch); nothing is guessed when the names hold no date."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

from astro_gpu import api
from astro_gpu.core import project2
from astro_gpu.core.project import Target
from astro_gpu.io import frames
from astro_gpu.io.frames import clock_disagreement, name_date

NAME = "Light_M8_120.0s_ISO800_2025-06-17_23-46-58.arw"          # a capture program's file name


def test_the_dates_a_name_says():
    assert name_date(NAME) == datetime(2025, 6, 17, 23, 46, 58)
    assert name_date("C:/nuit/" + NAME) == datetime(2025, 6, 17, 23, 46, 58)
    assert name_date("M42_Light_20250617_234658_0012.fits") == datetime(2025, 6, 17, 23, 46, 58)
    assert name_date("M42_Light_20250617-2346_0012.fit") == datetime(2025, 6, 17, 23, 46)
    assert name_date("2025-06-17T23:46:58_Ha.fits") == datetime(2025, 6, 17, 23, 46, 58)
    assert name_date("Light_2025-06-17.fits") == datetime(2025, 6, 17)
    # the same date twice (a folder-like prefix and the time stamp): one date, its time kept
    assert name_date("20250617_M8_2025-06-17_23-46-58.fits") == datetime(2025, 6, 17, 23, 46, 58)


def test_numbers_that_are_not_dates_and_two_dates_give_none():
    for name in ("DSC01234.ARW", "IMG_0001.CR2", "Light_00012345.fits",       # counters
                 "M8_20251399_0001.fits",                                    # no 13th month
                 "M8_20250231.fits",                                         # no 31 February
                 "M8_19890617.fits",                                         # before 1990
                 f"M8_{datetime.now().year + 2}0101.fits",                   # after a year from now
                 "M8_120250617.fits",                                        # inside a longer number
                 "M8_2025061.fits", "M8_2025-6-17.fits",                     # not a whole date
                 "M8_2025-06-17_then_2025-06-18.fits"):                      # two different dates: which one?
        assert name_date(name) is None, name


def test_a_camera_clock_never_set_disagrees_and_a_night_across_midnight_does_not():
    assert clock_disagreement([(NAME, "2014-01-01T00:16:00")]) == {"camera": "2014-01-01", "names": "2025-06-17"}
    # a frame of 23:59 named 00:01 the next day (another clock, a time zone): two minutes, not a disagreement
    assert clock_disagreement([("L_2025-06-18_00-01-00.arw", "2025-06-17T23:59:00")]) is None
    # one day apart across midnight (a name with the date only, a frame after midnight): not a disagreement
    assert clock_disagreement([("L_20250617.fits", "2025-06-18T02:30:00")]) is None
    assert clock_disagreement([("L_2025-06-17_22-00-00.fits", "2025-06-18T21:00:00")]) is None
    # FITS-style names and headers: DATE-OBS in UTC, with a Z or fractions of a second
    assert clock_disagreement([("M42_Light_20250617_234658.fits", "2025-06-17T21:46:58.123Z")]) is None
    assert clock_disagreement([("M42_Light_20250617_234658.fits", "2025-06-10T21:46:58")]) == \
        {"camera": "2025-06-10", "names": "2025-06-17"}
    # no date in the names: nothing said, whatever the frames' date
    assert clock_disagreement([("DSC01234.ARW", "2014-01-01T00:16:00")]) is None
    assert clock_disagreement([(NAME, None), (NAME, "")]) is None


def _project(root: Path, names: list[str], date_obs: str) -> project2.Project2:
    p = project2.create(root / "P", "P", Target(name="M 8"))
    for i, night in enumerate(("night 1", "night 2")):
        lights = root / f"L{i}"
        lights.mkdir(parents=True)
        for n in names:
            h = fits.Header()
            h["BAYERPAT"], h["EXPTIME"], h["DATE-OBS"] = "RGGB", 120.0, date_obs
            fits.PrimaryHDU(np.zeros((8, 8), np.uint16), h).writeto(lights / n)
        p.sessions.append(project2.Session(i + 1, night, lights=[str(lights)]))
    project2.save(p)
    return p


def test_the_checks_before_a_run_say_it_once_per_session(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_disk_check", lambda p: api.Check("ok", "disk", ""))
    names = [f"Light_M8_120.0s_2025-06-17_23-4{j}-58.fits" for j in range(3)]
    p = _project(tmp_path / "a", names, "2014-01-01T00:16:00")
    dates = {s.id: [(str(Path(s.lights[0]) / n), "2014-01-01T00:16:00") for n in names] for s in p.sessions}
    checks = [c for c in api.health_checks(p, dates) if c.key == "clock"]
    assert [c.text for c in checks] == [
        f"night {i}: dates: the camera says 2014-01-01, the file names say 2025-06-17: the camera's clock was probably "
        "never set. The star positions are computed for the camera's date" for i in (1, 2)]   # once each, not per frame
    assert checks[0].level == "warn" and checks[0].args == {"camera": "2014-01-01", "names": "2025-06-17",
                                                            "session": "night 1"}
    # before the Inventory has run and without the caller's dates nothing is read: nothing said
    assert not [c for c in api.health_checks(p) if c.key == "clock"]
    # the frames' date agrees with their names: nothing
    ok = {sid: [(n, "2025-06-17T23:40:00") for n, _t in v] for sid, v in dates.items()}
    assert not [c for c in api.health_checks(p, ok) if c.key == "clock"]


def test_names_without_a_date_are_never_looked_at(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_disk_check", lambda p: api.Check("ok", "disk", ""))
    p = _project(tmp_path / "b", ["DSC01234.fits", "DSC01235.fits"], "2014-01-01T00:16:00")
    dates = {s.id: [("DSC01234.fits", "2014-01-01T00:16:00")] for s in p.sessions}
    monkeypatch.setattr(frames, "clock_disagreement", lambda f: (_ for _ in ()).throw(AssertionError("looked at")))
    assert not [c for c in api.health_checks(p, dates) if c.key == "clock"]


def test_a_frame_without_a_date_of_its_own_is_never_said_to_have_a_wrong_clock(tmp_path):
    """No EXIF date, no DATE-OBS: its time is the file's date on disk (a copy's), marked so and never doubted."""
    f = tmp_path / "Light_M8_2025-06-17_23-46-58.fits"
    fits.PrimaryHDU(np.zeros((8, 8), np.uint16)).writeto(f)
    later = datetime(2026, 6, 17, 12).timestamp()                       # copied a year later
    os.utime(f, (later, later))
    info = frames.read_info(f)
    assert info.time_from_file and info.time.startswith("2026-06-17") and "time_from_file" not in info.to_dict()
    raw = tmp_path / "Light_M8_2025-06-17_23-46-58.arw"                 # a raw whose EXIF says no date
    raw.write_bytes(b"\0" * 64)
    os.utime(raw, (later, later))
    r = frames.FrameInfo(str(raw), "raw", 8, 8, "RGGB", 0.0, 1.0)
    frames._exif_into(r, raw)
    assert r.time_from_file
    h = fits.Header()
    h["DATE-OBS"] = "2014-01-01T00:16:00"
    fits.PrimaryHDU(np.zeros((8, 8), np.uint16), h).writeto(tmp_path / "own.fits")
    assert not frames.read_info(tmp_path / "own.fits").time_from_file
