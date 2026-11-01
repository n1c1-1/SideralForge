# Tests: four levels

Run them with `python tools/t.py <level>` (the venv's python). It sets `CUDA_VISIBLE_DEVICES=-1` (on this machine an
empty value does NOT hide the GPUs), `QT_QPA_PLATFORM=offscreen`, one BLAS thread per worker and the number of workers.

| Level | Command | What | Budget |
|---|---|---|---|
| 1 | `t.py 1 tests/test_x.py` (same as `pytest tests/test_x.py`) | one file, one process, level-2 tests only | under 1 min |
| 2 | `t.py 2` (the selection of a plain `pytest`) | the fast suite, engine + client, in parallel (pytest-xdist). No screenshot, no real data set, no run longer than a few seconds | under 3 min |
| 3 | `t.py 3` | level 2 + the heavy tests: client walkthrough, screenshots, small real data (heart_a7s), detached workers | under 20 min |
| 4 | `t.py 4` | only the `slow` tests (real data, hours), one at a time. The bench is apart: `tools/bench/` | never by default |

- A plain `pytest` has the level-2 selection (`addopts` in `pyproject.toml`: `-m "not level3 and not slow and not
  network"`) but one process; `t.py 2` is the same selection in parallel. Any `-m` you pass wins.
- Naming a test runs it, whatever its level: `pytest file.py::test_x` runs it; `-k name` also lifts level 3 (not `slow`).
  The end of a run says how many tests were left out.
- Parallel needs `pytest-xdist` (`uv pip install pytest-xdist`). `t.py -j N` sets the workers (default: three quarters of
  the cores, fewer when the machine lacks memory to commit: each worker imports torch and Qt, about 1.5 to 3 GB, and
  Windows fails the imports with "paging file is too small" otherwise); `-j 0` runs in this process.
- A test running more than 120 s (level 3: 600 s) is killed with the traceback of its threads (`ASTRO_TEST_TIMEOUT`), so
  that a test waiting for ever (a modal dialog offscreen...) fails by name instead of freezing the run.
- Every xdist worker has its own `ASTRO_GPU_HOME`, temporary folder and screenshot folder (`tests/conftest.py`); the
  guard on the user's `~/.astro_gpu` is kept (each worker refuses writes, the controller compares the folder at the
  end). The screenshots of a parallel run are deleted with the worker's folder: `ASTRO_GPU_SHOTS=<folder>` keeps them,
  one sub-folder per worker.

## Which test is in which level

- `level3` is a marker added from `tests/levels.toml` (a list of files, node ids or fnmatch patterns, with the reason),
  so no test file is edited to move a test. The `slow` and `network` markers stay in the test files.
- Level 3 by name, with no edit: `tests/client/*::test_real_*`, `*screenshots*`, `*::test_shots*`.
- A level-2 test stays under about 2 s. At the end of a level 1 / 2 run the conftest lists every test over 2 s (setup
  included: the first test of a module pays for the module's fixtures, and a parallel run times everything 1.5 to 3
  times longer than a test alone) and the total, and warns when the budget is over. Make such a test lighter (smaller
  synthetic data) or add it to `levels.toml`.
- A new test that opens a window to take a screenshot, reads a real data set (`F:/astro_essais`), or starts a detached
  engine worker or another long subprocess goes to level 3.
