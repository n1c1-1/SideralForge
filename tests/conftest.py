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

Parallel runs (pytest-xdist, ``tools/t.py``): every worker is a process of its own that imports this file, so each one
gets its own private ``ASTRO_GPU_HOME`` and its own temporary folder (screenshots, settings files), and the user's real
folder is the one the CONTROLLER saw before it moved ASTRO_GPU_HOME (``ASTRO_GPU_TEST_REAL_HOME``). The audit hook
runs in every worker; the end-of-session comparison of the real folder is done once, by the controller, which also
gets the refusals the workers recorded.

Test levels (tests/README.md): the default run is level 2 (``-m "not level3 and not slow and not network"`` in
pyproject.toml). ``level3`` is added here, from tests/levels.toml, to the tests that are too heavy for the fast suite,
so that no test file has to be edited for it. At the end of a level 1 / 2 run, the tests over 2 s and the total are
listed, with a warning when the budget is over.

The test project (``TEST_PROJECT``, built by tools/make_test_project.py) is named here once for every test that reads a
computed project; the tests skip with "run tools/make_test_project.py" when it is absent, and never write into it.
"""
from __future__ import annotations

import faulthandler
import fnmatch
import os
import shutil
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

#: "gw0", "gw1"... in a pytest-xdist worker (set by xdist before this file is imported), else None
WORKER = os.environ.get("PYTEST_XDIST_WORKER")
# no test opens a window: Qt draws offscreen in this process and in each worker (set QT_QPA_PLATFORM to see them)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_REAL_ENV = "ASTRO_GPU_TEST_REAL_HOME"
_T0 = time.time()


# ------------------------------------------------------------------------------------------ the private state folder

# ASTRO_GPU_HOME as the user had it. In an xdist worker the environment already holds the controller's private folder,
# so the controller passes the original along in ASTRO_GPU_TEST_REAL_HOME ("-" for "none": Windows deletes an
# environment variable that is set to an empty string).
if WORKER:
    _NAMED_HOME = os.environ.get(_REAL_ENV, "-")
    _NAMED_HOME = "" if _NAMED_HOME == "-" else _NAMED_HOME
else:
    _NAMED_HOME = os.environ.get("ASTRO_GPU_HOME", "")
    os.environ[_REAL_ENV] = _NAMED_HOME or "-"

#: what the engine would use if the tests did not move it: the user's real folders, to be left alone
_REAL_HOMES: list[Path] = []
for _h in (Path.home() / ".astro_gpu", Path(_NAMED_HOME) if _NAMED_HOME else None):
    if _h is not None and _h not in _REAL_HOMES:
        _REAL_HOMES.append(_h)

if not WORKER:      # folders a killed session left (a worker killed by the timeout cannot clean up): older than a day
    for _old in Path(tempfile.gettempdir()).glob("astro_gpu_test_*"):
        try:
            if time.time() - _old.stat().st_mtime > 86400:
                shutil.rmtree(_old, ignore_errors=True)
        except OSError:
            pass

PRIVATE_HOME = Path(tempfile.mkdtemp(prefix=f"astro_gpu_test_home_{WORKER or ''}_"))
os.environ["ASTRO_GPU_HOME"] = str(PRIVATE_HOME)          # BEFORE astro_gpu is imported; workers inherit it
PRIVATE_TMP: Path | None = None
if WORKER:
    # no temporary path shared between the workers: the folders the client tests build from %TEMP% (p10_settings, the
    # screenshots) and the ones a subprocess makes are the worker's own
    PRIVATE_TMP = Path(tempfile.mkdtemp(prefix=f"astro_gpu_test_tmp_{WORKER}_"))
    for _var in ("TEMP", "TMP", "TMPDIR"):
        os.environ[_var] = str(PRIVATE_TMP)
    tempfile.tempdir = str(PRIVATE_TMP)
    _shots = os.environ.get("ASTRO_GPU_SHOTS")             # a folder named by the user is kept, one sub-folder per worker
    os.environ["ASTRO_GPU_SHOTS"] = str(Path(_shots) / WORKER) if _shots else str(PRIVATE_TMP / "shots")
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
    if WORKER:      # the controller compares the real folders once and fails the session; it gets the refusals from here
        session.config.workeroutput["astro_violations"] = list(_VIOLATIONS)
        session.config.workeroutput["astro_counts"] = dict(_COUNTS)
    else:
        hard, live = changed_real_files()
        problems = [*(f"write refused: {v}" for v in _VIOLATIONS), *hard]
        _REPORT[:] = [problems, live]
        if problems and session.exitstatus == 0:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
    shutil.rmtree(PRIVATE_HOME, ignore_errors=True)
    if PRIVATE_TMP is not None:
        shutil.rmtree(PRIVATE_TMP, ignore_errors=True)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:            # xdist, in the controller: what a worker found
    out = getattr(node, "workeroutput", None) or {}
    for v in out.get("astro_violations", ()):
        _VIOLATIONS.append(f"[{getattr(node, 'gateway', None) and node.gateway.id}] {v}")
    if out.get("astro_counts") and not _COUNTS:           # every worker collects the same tests
        _COUNTS.update(out["astro_counts"])


# ----------------------------------------------------------------------------------------------------- the levels

#: the -m of the default run (level 2), as pyproject.toml's addopts has it
DEFAULT_MARK = "not level3 and not slow and not network"
LEVELS_FILE = Path(__file__).with_name("levels.toml")
ROOT = Path(__file__).resolve().parents[1]
#: seconds a level 1 / 2 / 3 run should not exceed, and the duration over which a level 1 / 2 test is listed
BUDGET = {"1": 60.0, "2": 180.0, "3": 1200.0}
SLOW_TEST = 2.0
MAX_LISTED = 40
_COUNTS: dict[str, int] = {}


def level3_patterns() -> list[str]:
    """The node ids (or file names, or fnmatch patterns on node ids) tests/levels.toml puts in level 3."""
    try:
        with open(LEVELS_FILE, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return []
    return [str(p).replace("\\", "/") for p in data.get("level3", [])]


def in_level3(nodeid: str, patterns: list[str]) -> bool:
    """A pattern names a file or a test (and all its parametrizations), or is an fnmatch pattern on the node id."""
    for p in patterns:
        if nodeid == p or nodeid.startswith((p + "::", p + "[")) or fnmatch.fnmatchcase(nodeid, p):
            return True
    return False


def _names_tests(config) -> bool:
    """True when the command line names tests: a node id (file::test) or a -k expression."""
    return any("::" in str(a) for a in config.args) or bool(config.option.keyword)


def pytest_configure(config) -> None:
    # naming a test runs it: a node id lifts every level, a -k expression lifts level 3 (not slow, not network)
    if config.option.markexpr == DEFAULT_MARK and _names_tests(config):
        config.option.markexpr = "" if any("::" in str(a) for a in config.args) else "not slow and not network"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items) -> None:
    """Mark the level 3 tests (BEFORE the -m of the run is applied) and count what each level holds."""
    patterns = level3_patterns()
    for item in items:
        if patterns and in_level3(item.nodeid, patterns):
            item.add_marker(pytest.mark.level3)
    _COUNTS.clear()
    for name in ("level3", "slow", "network"):
        _COUNTS[name] = sum(1 for i in items if i.get_closest_marker(name))
    _COUNTS["all"] = len(items)


#: seconds one test may take before the process is killed with the traceback of every thread (0: no limit). A test that
#: waits for ever (a modal dialog offscreen, a worker that never answers) would otherwise freeze a parallel run: the
#: controller waits for its worker without end. tools/t.py sets it from the level; xdist then names the test that was
#: running ("node down ... crashed") and goes on with a new worker.
TEST_TIMEOUT = float(os.environ.get("ASTRO_TEST_TIMEOUT") or 0)


def _real_stderr(config):
    """The terminal's stderr as pytest's own faulthandler plugin keeps it (a test's stderr is captured, and lost on exit)."""
    try:
        from _pytest.faulthandler import fault_handler_stderr_fd_key
        return config.stash[fault_handler_stderr_fd_key]
    except Exception:  # noqa: BLE001 - private API of pytest: fall back to the current stderr
        return sys.__stderr__


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    if TEST_TIMEOUT > 0:
        faulthandler.dump_traceback_later(TEST_TIMEOUT, exit=True, file=_real_stderr(item.config))
    try:
        yield
    finally:
        if TEST_TIMEOUT > 0:
            faulthandler.cancel_dump_traceback_later()


_TIMES: dict[str, dict[str, float]] = {}


def pytest_runtest_logreport(report) -> None:
    t = _TIMES.setdefault(report.nodeid, {})
    t[report.when] = t.get(report.when, 0.0) + report.duration


def _level(config) -> str | None:
    """"1" (named files), "2" (the whole fast suite), "3": from tools/t.py (ASTRO_TEST_LEVEL), else from the -m."""
    lv = os.environ.get("ASTRO_TEST_LEVEL")
    if lv:
        return lv
    if config.option.markexpr != DEFAULT_MARK:
        return None
    roots = {ROOT.resolve(), (ROOT / "tests").resolve()}
    whole = all(Path(str(a)).resolve() in roots for a in config.args)
    return "2" if whole else "1"


def _dur(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m} min {s:02d} s" if m else f"{sec:.1f} s"


def _time_budget(terminalreporter, config) -> None:
    level = _level(config)
    if level not in BUDGET or config.option.collectonly:
        return
    total = time.time() - _T0
    budget = BUDGET[level]
    tw = terminalreporter
    tw.section(f"time budget, level {level}")
    parallel = bool(getattr(config.option, "numprocesses", None))
    if level in ("1", "2"):
        slow = sorted(((sum(t.values()), n, t) for n, t in _TIMES.items() if sum(t.values()) > SLOW_TEST), reverse=True)
        if slow:
            tw.write_line(f"{len(slow)} test(s) over {SLOW_TEST:.0f} s, setup included (lighten them, or add them to "
                          f"tests/levels.toml)" + ("; a parallel run times them 1.5 to 3 times longer than a test alone "
                                                    "(python tools/t.py 1 file)" if parallel else "") + ":")
            for d, n, t in slow[:MAX_LISTED]:
                parts = ", ".join(f"{k} {v:.1f}" for k, v in t.items() if v >= 0.1)
                tw.write_line(f"  {d:6.1f} s  {n}  ({parts})")
            if len(slow) > MAX_LISTED:
                tw.write_line(f"  ... and {len(slow) - MAX_LISTED} more")
        else:
            tw.write_line(f"no test over {SLOW_TEST:.0f} s")
    run = sum(sum(t.values()) for t in _TIMES.values())
    left = {k: v for k, v in _COUNTS.items() if k != "all" and v}
    if _COUNTS and level in ("1", "2") and left:
        tw.write_line("not in this run: " + ", ".join(f"{v} {k}" for k, v in left.items())
                      + " (python tools/t.py 3 runs level 3; 4 the slow ones)")
    ok = total <= budget
    line = (f"total {_dur(total)} for {len(_TIMES)} tests ({_dur(run)} of test time)"
            + (f", within the {_dur(budget)} of level {level}" if ok else f": OVER the {_dur(budget)} budget of level {level}"))
    tw.write_line(line, green=ok, yellow=not ok, bold=not ok)
    if not ok and level in ("1", "2"):
        tw.write_line("WARNING: the fast suite got too slow for daily work: lighten or move the longest tests above"
                      + ("" if parallel or level == "1" else " (and run it in parallel: python tools/t.py 2)") + ".",
                      yellow=True, bold=True)


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
    _time_budget(terminalreporter, config)


# ------------------------------------------------------------------------------------------------ the test project

#: the computed project the client tests read: a copy of a case of the bench (Heart, express session), its workflows
#: computed with the code of the day (tools/bench/gradient_projects.py made the case; it has several workflows, so a
#: test that needs a known set takes it down with ``full_only``, and reads its numbers from it); ASTRO_TEST_PROJECT
#: names another one. Tests only read it: a test that changes a project works on a copy (``scratch_project`` below, or
#: its own copy in tmp_path).
TEST_PROJECT = Path(os.environ.get("ASTRO_TEST_PROJECT") or "F:/astro_essais/bench/heart/test_project")
#: a workflow whose results are in the test project's store (name, template, settings), for the tests that add it in
#: memory and need it computed
TEST_PROJECT_CLASSIC = ("Classic", "Classic stack", {})
HAVE_TEST_PROJECT = (TEST_PROJECT / "project.astroproj").is_file()
NO_TEST_PROJECT = f"the test project {TEST_PROJECT.as_posix()} is not here: run tools/bench/gradient_projects.py"
#: the raw frames the test project is made of (40 lights, 15 darks, 15 flats of a Sony A7S)
TEST_DATA = Path("F:/astro_essais/donnees/heart_a7s")
needs_test_project = pytest.mark.skipif(not HAVE_TEST_PROJECT, reason=NO_TEST_PROJECT)
needs_test_data = pytest.mark.skipif(not (TEST_DATA / "lights").is_dir(),
                                     reason=f"the raw test frames {TEST_DATA.as_posix()} are not here")


def scratch_project(folder: Path, source: Path = TEST_PROJECT) -> Path:
    """A copy of the test project's definition in ``folder`` whose ``store`` and ``inputs`` are directory junctions onto
    the real ones: the results are read (done states, previews), and what a test writes (state.json, the definition)
    lands in ``folder``. Skips where junctions cannot be made."""
    import subprocess
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "project.astroproj", folder / "project.astroproj")
    for name in ("store", "inputs"):
        if (source / name).is_dir() and not (folder / name).exists():
            try:
                r = subprocess.run(["cmd", "/c", "mklink", "/J", str(folder / name), str(source / name)],
                                   capture_output=True, text=True)
            except OSError as e:
                pytest.skip(f"no directory junction here: {e}")
            if r.returncode:
                pytest.skip(f"no directory junction here: {r.stderr.strip()}")
    return folder


def full_only(p, fresh_ids: bool = False):
    """The test project's definition as the graph tests start from it, whatever workflows the bench added since: one
    workflow, "Full" (id 1: the first with the Denoise and Stars blocks, computed in the store), and no layer in Post.
    In memory only.
    ``fresh_ids``: the workflows added next get ids the project never had (with ids 2, 3... they would find the
    states the bench's workflows of those ids left)."""
    import dataclasses

    full = next((w for w in p.workflows if w.options.get("denoise") and w.options.get("stars")), None)
    if full is None:
        pytest.skip("the test project has no workflow with the Denoise and Stars blocks")
    p.workflows[:] = [dataclasses.replace(full, id=1, name="Full", end=None, variant_of=None, breakpoints=[])]
    p.next_workflow_id, p.current = (max(p.next_workflow_id, 2) if fresh_ids else 2), 1
    from astro_gpu.core.project2 import new_post
    p.post = new_post()
    return p


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
def _collect_on_the_main_thread(request):
    """Client tests: Python's cyclic collector runs on the main (GUI) thread only, after each test, as the client's
    gcguard.GuiGarbageCollector does; a collection started by a worker thread can destroy a Qt object there and crash
    the process. (Enabled again after the test, for the engine tests that may follow in the same process.)"""
    if "client" not in Path(str(request.fspath)).parts:
        yield
        return
    import gc
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        gc.enable()


@pytest.fixture(autouse=True)
def _private_project_registry(tmp_path, monkeypatch):
    """Each test has its own list of recent projects (inside the session's private folder, never the user's)."""
    import astro_gpu.core.project as project
    monkeypatch.setattr(project, "REGISTRY", tmp_path / "projects.json")
