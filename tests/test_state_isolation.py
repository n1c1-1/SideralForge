"""The tests never write the user's files (tests/conftest.py): the engine state is private to the session, in this
process and in the workers a test detaches, and the guard of the real folder works."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_engine_state_is_the_sessions_private_folder(private_home, guard):
    import astro_gpu.core.project as project
    from astro_gpu.core import estimate
    assert project.STATE == private_home == Path(os.environ["ASTRO_GPU_HOME"])
    assert estimate.history_path().parent == private_home
    assert private_home not in guard.homes
    assert not any(h == project.REGISTRY.parent or h in project.REGISTRY.parents for h in guard.homes)    # per test, private


def test_a_detached_worker_inherits_the_private_folder(private_home):
    """A subprocess does not inherit a monkeypatch: only the environment, which the session fixed before any import."""
    out = subprocess.run([sys.executable, "-c", "from astro_gpu.core import project, estimate; "
                          "print(project.STATE); print(project.REGISTRY); print(estimate.history_path())"],
                         cwd=ROOT, capture_output=True, text=True, timeout=120, check=True).stdout.split("\n")
    assert [Path(x) for x in out[:3]] == [private_home, private_home / "projects.json", private_home / "timings2.json"]


def test_a_write_in_the_users_folder_is_refused(guard, tmp_path):
    """The audit hook raises before anything is written (it is called directly: the real file is never opened)."""
    real = guard.homes[0]
    before = len(guard.violations)
    try:
        for event, args in (("open", (str(real / "projects.json"), "w", 0)),
                            ("open", (str(real / "timings.json"), None, os.O_WRONLY | os.O_CREAT)),
                            ("os.rename", (str(real / "x.tmp"), str(real / "timings2.json"), None, None)),
                            ("os.remove", (str(real / "projects.json"), None))):
            with pytest.raises(guard.error):
                guard.audit(event, args)
        for event, args in (("open", (str(real / "projects.json"), "r", 0)),                     # reading is fine
                            ("open", (str(real / "cache" / "gaia.npz"), "wb", 0)),               # the shared download cache
                            ("open", (str(tmp_path / "projects.json"), "w", 0))):
            guard.audit(event, args)
        assert len(guard.violations) == before + 4
    finally:
        del guard.violations[before:]               # the refusals above are the test, not a leak: keep the session green


def test_the_users_folder_is_untouched_so_far(guard):
    """Nothing before this test wrote the user's real state folder (the conftest checks again at the end)."""
    hard, _live = guard.changes()
    assert not hard and not guard.violations, (hard, guard.violations)
