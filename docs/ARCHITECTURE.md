# How the code is organised

astro_gpu turns the raw frames of a colour camera into a calibrated, stacked, colour-corrected image, then lets the
user finish it in a layer-based Post. The client drives everything; the engine does the work.

```
client (astro_gpu/client, PySide6)
   │  only through
   ▼
api.py ──► core/ (projects, workflows, store, runner) ──► stages/ (one module per processing step)
                                                     └──► render/ (Post layers, sliders, export)
```

## The engine

| Where | What |
|---|---|
| `api.py` | The only door between the client and the engine: projects, workflows, settings with their suggestions, runs, results to inspect, renders, export. |
| `core/project2.py` | A project (`project.astroproj`): sessions of frames, workflows with their settings, frames excluded by hand. |
| `core/workflow.py` | A workflow's options → its chain of stages; the blocks of the graph; which stages two workflows share. |
| `core/store.py` | Stage results in `store/<stage>/<key>/`. The key hashes everything that decides a result, so a shared block is computed once and a change recomputes only what follows it. Rules in `stages/v2.py`. |
| `core/runner2.py` | Plans the stage jobs of a run and executes them in a detached worker process (progress, stop, resume). |
| `core/params.py` | How a setting is declared (`P`: default, bounds, label, help, level) and how automatic values (`Auto`, with their reason) are resolved. |
| `stages/` | The processing steps, in order: inventory, calibrate, register, stack, astrometry, catalogs, surveys, gradient, color, denoise, render_simple (the Output), star_removal (the Stars block: a trained network, `resources/stars/`). Each is a `Stage` with its `Params`, `auto()` and `run()`. |
| `core/` (methods) | The algorithms the stages call: `debayer`, `resample`, `rejection_classic`, `stack_classic` (Classic stacking), `coverage` (drizzle scale rule), `frame_weights` (frame selection), `psf_moffat` (star size), `blindsolve`. Each names its published source. |
| `core/` (data) | Sky catalogues and objects (`gaia_archive`, `gaia_extracts`, `sirilcat`, `skycache`, `skyobjects`, `online`), the time estimates (`estimate`), the diagnostics Inspect shows (`diagnostics`). |
| `render/` | Post's engine: the layers of a workflow and the export (`layers`), the simplified sliders (`sliders`), the stretch of the Output image (`simple`) and of the Stars block's two images, recombined or not (`stars`), GPU primitives (`ops`). |
| `texts.py` | The engine's display texts (labels, reasons, metrics); French in `client/locale/fr/engine.json`. |
| `core/project.py`, `core/runner.py`, `cli.py`, `core/import_v1.py` | The older project format (`project.json`, one chain per project), still used by the command line; the client opens such a project by importing it (`import_v1`). |

## The client

| Where | What |
|---|---|
| `app.py`, `home.py` | The main window and the Home page. |
| `model.py` | `ProjectModel`: the one adapter between the views and `api.py`. |
| `graph.py`, `blocks.py`, `panel.py`, `settings.py` | Graph mode: the workflows as blocks, the settings of the selected block; the Composition section (the Stars and background blocks: Post's layers of part "combined", their settings shared with Post through `ProjectModel.set_post_layer`). |
| `tune.py`, `steps.py`, `charts.py`, `chartview.py` | Inspect: one block, its images and measurements, interactive charts. |
| `frames.py`, `sources.py` | The sessions, the health check of the data, the frame browser. |
| `imageview.py`, `stretch.py` | The image viewer used everywhere (compare, blink, display stretch). |
| `post3.py`, `postengine.py`, `postfilters/`, `curveseditor.py`, `viewer.py` | Post: layers, masks, filters, curves. |
| `runs.py`, `services/` | Runs and every slow task, off the GUI thread (`services/worker.py`, `run.py`, `inspect.py`, `render.py`). |
| `i18n.py`, `locale/fr/` | Every visible text is English in the code and goes through `_()`; French in `locale/fr/*.json`. |
| `prefs.py`, `settingsdlg.py` | The user's settings (`ASTRO_GPU_SETTINGS` points tests to a scratch file). |

## Rules the code follows

- Every method of the engine is a published one, named with its source in its module.
- A setting's stored value never changes meaning. When a setting is removed, `stages/v2.py` says how projects that
  still store it are read (`RETIRED`, `KEY_CONSTANTS`, `KEY_BASELINE`), so results already computed keep their keys.
- The client never computes on the GUI thread and never imports a stage: it goes through `api.py`.
- Tests: [tests/README.md](../tests/README.md).
