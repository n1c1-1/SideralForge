"""Shared test configuration.

THE RULE: a test never writes the user's files. The engine keeps its per-user state (the list of recent projects, the
stage timings behind the estimates, the camera colour profiles) in ``astro_gpu.core.project.STATE``, which is read ONCE,
from ``ASTRO_GPU_HOME`` (else ``~/.astro_gpu``), when astro_gpu is first imported. A fixture that patches ``STATE`` or
``REGISTRY`` for one test is too late for the module- and session-scoped fixtures (they run before the function-scoped
ones and create projects, which register themselves) and it is invisible to the detached workers a test starts (they
are other processes: they only inherit the environment). So, here, before any test module imports astro_gpu:

1. ``ASTRO_GPU_HOME`` points to a private temporary folder for the whole session (this process and every subprocess);
2. a guard watches the user's real state folder, ``~/.astro_gpu`` (and the folder ``ASTRO_GPU_HOME`` named when the
   session started, if any), apart from its ``cache/`` (the shared download cache that real runs fill):
   - an audit hook refuses, in this process, any write there and names the test that tried;
   - the files' mtimes and sizes are taken now and compared when the session ends; any file that changed, appeared or
     went away fails the session. ``timings.json`` / ``timings2.json`` are the exception: a real run of the user also
     writes them while a session runs, so a change there only warns (the audit hook and the private folder are what
     protect them).
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def gpu_hidden() -> bool:
    """True when this run must not use a GPU (the projects' runs have them): CUDA_VISIBLE_DEVICES empty or -1 (on this
    Windows machine only -1 hides the GPUs, an empty value does not, but both say "no GPU"), or no GPU is visible."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is not None and v.strip() in ("", "-1"):
        return True
    try:
        import torch
        return not torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return True


# the ComfyUI nodes pick a CUDA device when imported: without a visible GPU their test files cannot even be collected
collect_ignore = ["test_comfy_masks.py", "test_comfy_project_node.py"] if gpu_hidden() else []

# ------------------------------------------------------------------------------------------ the private state folder

#: what the engine would use if the tests did not move it: the user's real folders, to be left alone
_REAL_HOMES: list[Path] = []
for _h in (Path.home() / ".astro_gpu", Path(os.environ["ASTRO_GPU_HOME"]) if os.environ.get("ASTRO_GPU_HOME") else None):
    if _h is not None and _h not in _REAL_HOMES:
        _REAL_HOMES.append(_h)

PRIVATE_HOME = Path(tempfile.mkdtemp(prefix="astro_gpu_test_home_"))
os.environ["ASTRO_GPU_HOME"] = str(PRIVATE_HOME)          # BEFORE astro_gpu is imported; workers inherit it
# the camera colour profiles the tests may read are the user's (a copy: read there, written here)
for _h in reversed(_REAL_HOMES):                           # the folder ASTRO_GPU_HOME named, if any, else the default
    if (_h / "camera_profiles").is_dir():
        shutil.copytree(_h / "camera_profiles", PRIVATE_HOME / "camera_profiles", dirs_exist_ok=True)
        break

# astro_gpu already imported (a plugin, or pytest.main() called from a script): move what it computed at import time
_project = sys.modules.get("astro_gpu.core.project")
if _project is not None:                                  # pragma: no cover
    _project.STATE = PRIVATE_HOME
    _project.REGISTRY = PRIVATE_HOME / "projects.json"
_color = sys.modules.get("astro_gpu.stages.color")
if _color is not None:                                    # pragma: no cover
    _color.PROFILES = PRIVATE_HOME / "camera_profiles"

# ----------------------------------------------------------------------------------------------------- the guard

#: files a real run of the user writes at any time (the stage that just finished adds its duration)
_LIVE_FILES = frozenset({"timings.json", "timings2.json"})


def _norm(p) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(p)))


_WATCHED = [_norm(h) for h in _REAL_HOMES]
_CACHE = [w + os.sep + "cache" for w in _WATCHED]


def _is_watched(path) -> bool:
    """True for a path inside a real state folder, outside its cache/."""
    try:
        s = _norm(path)
    except (TypeError, ValueError):                        # a file descriptor, None
        return False
    for w, c in zip(_WATCHED, _CACHE):
        if (s == w or s.startswith(w + os.sep)) and not (s == c or s.startswith(c + os.sep)):
            return True
    return False


def _snapshot() -> dict[str, tuple[int, int]]:
    """{path: (mtime_ns, size)} of every file under the real state folders (not their cache/)."""
    out: dict[str, tuple[int, int]] = {}
    for home in _REAL_HOMES:
        stack = [home]
        while stack:
            d = stack.pop()
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                if not _is_watched(e.path):
                    continue
                try:
                    st = e.stat()
                    if e.is_dir():
                        stack.append(Path(e.path))
                    else:
                        out[_norm(e.path)] = (st.st_mtime_ns, st.st_size)
                except OSError:
                    pass
    return out


_BEFORE = _snapshot()
_REPORT: list = []                    # [problems, live changes], for the terminal summary
_VIOLATIONS: list[str] = []
_CURRENT = ["(collection)"]

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


class UserFolderWrite(RuntimeError):
    """A test tried to write inside the user's real state folder."""


def _audit(event: str, args: tuple) -> None:
    try:
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            by_mode = isinstance(mode, str) and any(c in mode for c in "wax+")
            if not (by_mode or (isinstance(flags, int) and flags & _WRITE_FLAGS)):
                return
            targets = [path]
        elif event == "os.rename":
            targets = [args[0], args[1]]
        elif event in ("os.remove", "os.rmdir", "os.mkdir", "os.truncate"):
            targets = [args[0]]
        else:
            return
        for t in targets:
            if isinstance(t, (str, bytes, os.PathLike)) and _is_watched(os.fsdecode(t) if isinstance(t, bytes) else t):
                what = f"{event} {os.fsdecode(t) if isinstance(t, bytes) else os.fspath(t)}"
                _VIOLATIONS.append(f"{_CURRENT[0]}: {what}")
                raise UserFolderWrite(f"a test must never write the user's files ({what}; test {_CURRENT[0]}): "
                                      f"ASTRO_GPU_HOME is {os.environ.get('ASTRO_GPU_HOME')}")
    except UserFolderWrite:
        raise
    except Exception:  # noqa: BLE001 - the guard must never break anything else
        return


sys.addaudithook(_audit)


def pytest_collectstart(collector) -> None:
    _CURRENT[0] = f"(collecting {getattr(collector, 'nodeid', '') or collector})"


def pytest_runtest_logstart(nodeid, location) -> None:
    _CURRENT[0] = nodeid


def changed_real_files() -> tuple[list[str], list[str]]:
    """(files that changed, appeared or went away since the session began; the same for timings*.json alone)."""
    now = _snapshot()
    hard, live = [], []
    for p in sorted(set(_BEFORE) | set(now)):
        if _BEFORE.get(p) != now.get(p):
            kind = "new" if p not in _BEFORE else "removed" if p not in now else "changed"
            (live if os.path.basename(p) in _LIVE_FILES else hard).append(f"{kind}: {p}")
    return hard, live


def pytest_sessionfinish(session, exitstatus) -> None:
    hard, live = changed_real_files()
    problems = [*(f"write refused: {v}" for v in _VIOLATIONS), *hard]
    _REPORT[:] = [problems, live]
    if problems and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    shutil.rmtree(PRIVATE_HOME, ignore_errors=True)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    problems, live = _REPORT or ([], [])
    if problems:
        terminalreporter.section("THE USER'S STATE FOLDER WAS TOUCHED BY THE TESTS", red=True, bold=True)
        for line in problems:
            terminalreporter.write_line(line, red=True)
        terminalreporter.write_line("Tests never write the user's files: find the path that ignores ASTRO_GPU_HOME "
                                    "(tests/conftest.py) and undo what it wrote.", red=True)
    if live:
        terminalreporter.write_line("note (not a failure): " + "; ".join(live) + " - a real run of yours writes these "
                                    "while a session runs; a write by a test would have been refused above.")


@pytest.fixture(scope="session")
def private_home() -> Path:
    """The session's private ASTRO_GPU_HOME (engine state of the tests)."""
    return PRIVATE_HOME


@pytest.fixture(scope="session")
def guard():
    """The guard of the user's real state folders (tests/test_state_isolation.py checks that it works)."""
    return SimpleNamespace(homes=list(_REAL_HOMES), audit=_audit, error=UserFolderWrite, violations=_VIOLATIONS,
                           changes=changed_real_files)


@pytest.fixture(autouse=True)
def _private_project_registry(tmp_path, monkeypatch):
    """Each test has its own list of recent projects (inside the session's private folder, never the user's)."""
    import astro_gpu.core.project as project
    monkeypatch.setattr(project, "REGISTRY", tmp_path / "projects.json")
