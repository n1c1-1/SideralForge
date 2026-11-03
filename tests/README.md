# Tests

`tests/` holds the engine's tests, `tests/client/` the client's (PySide6, offscreen). Run them with the venv's python
through `tools/t.py`, which sets `CUDA_VISIBLE_DEVICES=-1` (on this machine only -1 hides the GPUs), offscreen Qt, one
BLAS thread per worker and the number of parallel workers.

## Levels

| Level | Command | What | Budget |
|---|---|---|---|
| 1 | `python tools/t.py 1 tests/test_x.py` | one file (or a few), one process, level-2 tests only | under 1 min |
| 2 | `python tools/t.py 2` | the fast suite, engine + client, in parallel: no screenshot, no real data, no long run | under 3 min |
| 3 | `python tools/t.py 3` | level 2 + the heavy tests (`levels.toml`): screenshots, real data, detached workers | under 20 min |
| 4 | `python tools/t.py 4` | only the `slow` tests (real data, long runs), one at a time | never by default |

- A plain `pytest` has the level-2 selection (`addopts` in `pyproject.toml`) in one process. Naming a test runs it
  whatever its level (`pytest file.py::test_x`); `-k name` also lifts level 3.
- `t.py -j N` sets the workers (default three quarters of the cores, fewer when memory is short); `-j 0` runs in one
  process. Parallel runs need `pytest-xdist`.
- A test running more than 120 s (level 3: 600 s) is killed with the traceback of its threads (`ASTRO_TEST_TIMEOUT`).
- At the end of a level 1 / 2 run the tests over 2 s and the total time are listed; lighten a slow test or add it to
  `levels.toml` (a list of files, node ids or fnmatch patterns, each with its reason). By name, with no edit:
  `tests/client/*::test_real_*`, `*screenshots*`, `*::test_shots*` are level 3; the `slow` and `network` markers stay
  in the test files.

## The user's files are never touched

`tests/conftest.py` gives every run (and every xdist worker) a private `ASTRO_GPU_HOME` and temporary folder, refuses
any write into the user's `~/.astro_gpu` (its `cache/` apart) and compares that folder at the end of the session. The
client tests point `ASTRO_GPU_SETTINGS` to a temporary file. `ASTRO_GPU_SHOTS=<folder>` keeps the screenshots.

## Test data

- **The test project** `F:/astro_essais/tests/heart_small` (`TEST_PROJECT` in `tests/conftest.py`; the variable
  `ASTRO_TEST_PROJECT` names another): the 40 lights, 15 darks and 15 flats of `F:/astro_essais/donnees/heart_a7s`
  (Sony A7S, IC 1805), workflow Full computed, and in its store the results of a Classic workflow
  (`TEST_PROJECT_CLASSIC`) that the Post tests add in memory. The client tests only read it (a test that changes it
  works on a copy); without it they skip with "run tools/make_test_project.py".
- **Rebuild it** after an engine change that makes its results outdated:
  `python tools/make_test_project.py --force` (cuda:0, about 50 minutes once the Gaia tiles are in
  `F:/astro_essais/tests/_state/cache`; `--device` picks another GPU).
- Other real data the tests read when present (they skip otherwise): `F:/astro_essais/donnees/heart_a7s` (raw frames),
  `F:/astro_essais/projets/heart_a7s` (a project of the older version, for the import), `F:/astro_essais/bench/...`
  (bench projects, read only).
