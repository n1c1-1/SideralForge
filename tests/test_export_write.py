"""An export replaces its file in one step (issue #85)."""
from __future__ import annotations

import os

import pytest

from astro_gpu.render import layers


def write(text):
    return lambda t: t.write_text(text)


def test_an_existing_file_is_replaced_and_no_temporary_stays(tmp_path):
    p = tmp_path / "a.tif"
    p.write_text("old")
    layers._safely(p, write("new"))
    assert p.read_text() == "new" and [f.name for f in tmp_path.iterdir()] == ["a.tif"]


def test_the_temporary_keeps_the_suffix_that_says_the_format(tmp_path):
    seen = []
    layers._safely(tmp_path / "a.png", lambda t: (seen.append(t.suffix), t.write_text("x")))
    assert seen == [".png"]


def test_a_refusal_that_clears_is_retried(tmp_path, monkeypatch):
    p = tmp_path / "a.tif"
    p.write_text("old")
    real, calls = os.replace, []

    def flaky(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise OSError(22, "Invalid argument")
        real(a, b)
    monkeypatch.setattr(layers.os, "replace", flaky)
    monkeypatch.setattr(layers, "REPLACE_WAIT", 0.0)
    layers._safely(p, write("new"))
    assert p.read_text() == "new" and len(calls) == 3


def test_a_failure_leaves_the_old_file_and_names_the_file(tmp_path, monkeypatch):
    p = tmp_path / "a.tif"
    p.write_text("old")

    def refused(a, b):
        raise OSError(22, "Invalid argument")
    monkeypatch.setattr(layers.os, "replace", refused)
    monkeypatch.setattr(layers, "REPLACE_WAIT", 0.0)
    with pytest.raises(layers.WriteFailed) as e:
        layers._safely(p, write("new"))
    assert e.value.name == "a.tif" and p.read_text() == "old" and [f.name for f in tmp_path.iterdir()] == ["a.tif"]
