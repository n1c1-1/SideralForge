"""The night of a session (noon to noon), across a month and a year: a camera whose clock was never set dated its frames
1 January 2014 at 00:16, which the Sources said "night of 31/01" (the night's day with the frame's month)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from astro_gpu.client.frames import night_text  # noqa: E402
from astro_gpu.client.sources import SourcesView  # noqa: E402


@pytest.mark.parametrize("first, night, text", [
    ("2014-01-01T00:16:00", "31/12", "31 Dec → 1 Jan 2014"),          # across a year
    ("2015-03-01T00:30:00", "28/02", "28 Feb → 1 Mar 2015"),          # across a month, not a leap year
    ("2016-03-01T00:30:00", "29/02", "29 Feb → 1 Mar 2016"),          # a leap year
    ("2026-08-16T22:00:00", "16/08", "16 → 17 Aug 2026"),             # before midnight: its own day
])
def test_the_night_of_a_frame_is_the_day_it_began_on(first, night, text):
    row = SimpleNamespace(first=first, exposure=300.0, iso=2000.0)
    sub = SourcesView._sub(SimpleNamespace(), "lights", 24, row)
    assert sub == f"24 lights · 300 s · ISO 2000 · night of {night}", sub
    assert night_text(first) == text


def test_one_frame_is_said_in_the_singular():
    """A master given as a session's darks is one file: the line under "Darks" said "1 frames · master_DOF"."""
    from astro_gpu.client import i18n
    view = SimpleNamespace(sid=1, s=SimpleNamespace(session_files=lambda sid, key: ["D:/night/master_DOF/dark.fit"]))
    row = SimpleNamespace(first="2025-06-17T23:46:58", exposure=300.0, iso=2000)
    try:
        assert SourcesView._sub(view, "darks", 1, None) == "1 frame · master_DOF"
        assert SourcesView._sub(view, "darks", 16, None) == "16 frames · master_DOF"
        assert SourcesView._sub(view, "lights", 1, row).startswith("1 light · ")
        i18n.set_language("fr")
        assert SourcesView._sub(view, "darks", 1, None) == "1 image · master_DOF"
        assert SourcesView._sub(view, "lights", 1, row).startswith("1 brute · ")
        assert not i18n.missing()
    finally:
        i18n.set_language("en")
