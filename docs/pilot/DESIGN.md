# Design: the client on the real engine

Written 2026-10-31 by the architect agent, from PILOT.md, ARCHITECTURE.md, ENGINE_KNOWLEDGE.md, spec/*, mockup/alt3 and
the engine code. Paths are relative to the repository root. Python: `C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe`.

## 0. The decisions in brief
1. **v2 lives beside v1**: a folder with `project.astroproj`, `state.json`, `store/`. v1 (`project.json`, the
   Pipeline, `gui/`, `tools/qc`) is untouched and keeps working; nothing writes into a v1 folder.
2. **Results are addressed by a key** (stage, version, stored parameters, upstream keys, project inputs). Auto values
   are not in it: they follow from the upstream results (stages are deterministic since 2026-10-16).
3. **Stage code runs unchanged** through a shim (`StoreProject`) that maps every ancestor to its key folder, as
   `comfyui_astro/blocks.py` does; jobs are executed by v1's `Pipeline._run_stage`.
4. **Two new stages only**, `background_poly` and `render_simple`, make the four templates real. Deconvolution, the
   new denoisers and the reorder are deferred, behind methods.
5. **Import adopts results** under their v2 keys: 7 of the 9 real projects keep inventory → optics (≈ 1–1.5 h of GPU
   each).
6. **The client is the mockup** (alt3 + alt2's viewer and widgets) in `astro_gpu/client/`, over a `ProjectModel`
   adapter on `astro_gpu/api.py`; runs in a detached process, renders in a worker thread.

## 1. Project format v2
### 1.1 Files
```
<workspace>/<project>/                      workspace default F:/astro_projets
  project.astroproj        the definition (JSON, atomic writes, written only through api)
  state.json               {"last_keys": {"<wf id>": {"<stage>": "<key>"}}}: the previous result of an Outdated block
  store/<stage>/<key>/     one result: outputs, manifest.json, previews/, _ckpt/ while resumable
  store/<stage>/<key>.link {"path": "<old work/stage folder>"}: an adopted result left in place, read-only
  logs/events.jsonl  logs/run.lock  logs/STOP  logs/runs/<run_id>.plan.json  logs/runs/<run_id>.log
  cache/thumbs/  cache/thumb.jpg (Home card)   inputs/ (reference image, external star layers)   export/<name>/
<scratch_dir or project/scratch>/<key>/<stage>/   large temporaries (the calibrated frame cache)
```
### 1.2 project.astroproj
```json
{"format": 2, "name": "HeartSoul", "description": "", "created": "2026-11-01T02:10:00",
 "target": {"name": "IC 1805", "ra": null, "dec": null, "focal_mm": 272.0},
 "scratch_dir": "", "reference_image": "inputs/full.png", "devices": {"main": "cuda:1", "aux": ""},
 "sessions": [{"id": 1, "name": "2026-08-16", "lights": ["F:/photos/heart_nebula/raw/signal"],
               "darks": ["F:/photos/heart_nebula/raw/dark"], "flats": ["F:/photos/heart_nebula/raw/flat"],
               "bias": [], "flat_darks": []}],
 "excluded": ["DSC01234.ARW"], "stack_mode": "together",
 "workflows": [{"id": 1, "name": "Full", "options": {"denoise": "n2n", "stars": "psf", "gradient": "halpha",
                "deconvolution": null}, "params": {"stack": {"scale": 1.5}}, "end": null, "variant_of": null}],
 "current": 1, "render_bases": {"faithful": {"sky_level": 24.0}}, "post": {"layers": []}}
```
- **Paths and ids.** Paths may be relative to the project. Workflow ids are integers, never reused; the name is
  display only, so renaming is free.
- **`params`** keeps stored values only. A value equal to the static default is dropped on save, and the API drops one
  equal to the current suggestion (as the mockup does). NEW (◇) settings are never stored in v1.
- **Sessions** are real for display and checks. The engine gets the union of each kind (one master dark and flat, as
  today); flats per session are ◇.
- **`excluded`** (light file names) becomes the new parameter `stack.exclude` in every workflow. Only Stacking and later
  stages change (this avoids the mockup's open point 3).
- **`post.layers`**: the POST.md dicts. A workflow layer has `base` (a `render_bases` name, or a preset) and `sliders`.

### 1.3 Key and store
```
key(wf, s) = sha1(canonical_json({"s": s, "v": STAGES[s].version,
    "p": key_params(wf, s),                          # stored ∪ implied ∪ injected, minus NON_KEY and static defaults
    "u": {d: key(wf, d) for d in deps_in_chain(wf, s)},
    "x": project_part(s)}))[:16]                     # canonical: sort_keys, separators (",", ":")
```
- **Implied** parameters: `star_net.enabled=True` with psf_neural; `surveys.survey="none"` with polynomial + Stars.
  **Injected**: `stack.exclude`. **NON_KEY** (no effect on results): `calibrate.workers`, `register.workers`,
  `star_net.device`.
- **project_part**: for inventory, the ordered (kind, file name, size) list as `list_frames` returns it over the union
  of sessions (content-based, so copied raws keep the key; ordered, because the half-stack blocks follow the order);
  astrometry: the target; reference: the image's (name, size); others: null.
- **Left out on purpose**: auto values, the device, which catalogue service answered, camera profiles in
  ASTRO_GPU_HOME. `describe()` compares a done key's manifest params with a fresh resolve and reports any drift
  ("suggested values changed since this result"); nothing re-runs automatically.
- **Done**: `store/<s>/<key>/manifest.json` (or the `.link` target's) has status done. The manifest is v1's plus
  `key`, `key_parts`, `adopted_from`, with `run_id = key`, so starless' auto `learned_stars` stays stable.
- **usable_as_input(s, dir)** is false when a done result lost external data (inventory: a frames.json path; calibrate:
  the cache.json path). A plan that needs it as an input re-runs it under the same key.
- **Old keys are kept.** `cache_usage()` lists keys no workflow references (current or last); "Clear unused results"
  deletes them after a confirmation only.

### 1.4 StoreProject, the shim (core/store.py)
- **Attributes**: `root`; `logs` (the project's, so STOP is shared); `results` = export/; `scratch` =
  `<scratch>/<key>`; `config` (name, `inputs` = union of sessions, target, devices, reference_image, empty
  params/variants/active_variant); `overrides(s)` = stored + implied + injected values.
- **`stage_dir(name)`** returns the key folder of the stage and of every ancestor (transitive `deps_in_chain`); any
  other stage gets `store/_absent/<name>`, never created (`ctx.input` raises "missing input", `star_net.stage_layer`
  returns None, export's reference test is false). New helper `Context.has(stage)`. A grep shows that stages reach
  outside `stage_dir` only through config.target, inputs, name, reference_image, devices, and STATE.

### 1.5 Backward compatibility
- **Loading.** `project2.load(root)` reads `project.astroproj`. A folder with only `project.json` raises
  `LegacyProject`; its `from_v1(root)` builds the v2 definition in memory (one session from `inputs`, one workflow
  from main, each variant a workflow with merged params). The client offers Import (1.6).
- **The workflow from main** has denoise n2n, and gradient polynomial if `surveys.survey == "none"`, else halpha. Its
  stars and name: psf_neural "Full + neural stars" when `star_net.enabled` (default true); psf_neural without
  star_net, "Full + earlier neural layer", when `starless.learned_stars` names a file; else psf "Full".

### 1.6 Importing a v1 project with its results: possible
`core/import_v1.py`. Nothing is ever written in the old folder.
1. **Write the v2 definition** with the new raw locations; `path_map` (old name → new path) is checked by name and
   size.
2. **Walk each chain in order.** Adopt a stage only if its old manifest is done with the current version; all its
   `deps_in_chain` are adopted and `manifest.inputs[d]` equals the old `d` run_id; its old params equal a fresh
   resolve (a read-only shim over the old `work/`, NON_KEY ignored, `_same_params`; auto values read existing folders,
   so no mkdir happens); and, for inventory, the old frames.json names and sizes, in order, equal the new
   `project_part`.
3. **Store** it under the new key, "copy" (the default, without `_ckpt`) or "link" (a `.link`; the runner makes a
   fresh folder before re-running). Rewrite the manifest and relocate the frames.json paths. Calibrate's cache.json
   points at the new scratch; the cache is not copied, so calibrate re-runs if register or stack must.
4. **Report** per stage: adopted (key) or recomputed (reason).

The parameter check matters: a version bump alone misses changed overrides and auto rules. Expected, from the
manifests read on 2026-10-31:

| Project | Adopted | Why not the rest |
|---|---|---|
| Heart, FlamingStar, Horsehead, Rosette, Wizard, M31, Iris | inventory → optics (reference where present) | stars v13 < 15 |
| NGC7822, M33 | inventory only | old calibrate and stack |
| F:/astro_essais/projets/heart_a7s | all 15 stages of Full | all current: the test case |

## 2. Optional blocks and methods (core/workflow.py)
### 2.1 Options → chain
- **Options** `{denoise: None|"n2n", stars: None|"psf"|"psf_neural", gradient: "halpha"|"polynomial",
  deconvolution: None}` (other values ◇). `end` is None, or "stacking" for the Raw stack exit.
- **Global order**: inventory, calibrate, register, stack, astrometry, reference, catalogs, surveys, color, denoise,
  optics, stars, star_net, starless, background | background_poly, render_prep | render_prep_learned | render_simple.

| Template | denoise | stars | gradient | Stages after stack | Output |
|---|---|---|---|---|---|
| Classic stack | – | – | polynomial | astrometry, catalogs, color, background_poly | render_simple |
| Classic + denoise | n2n | – | polynomial | same + denoise | render_simple |
| Full | n2n | psf | halpha | astrometry, catalogs, surveys, color, denoise, optics, stars, starless, background | render_prep |
| Full + neural stars | n2n | psf_neural | halpha | Full + star_net | render_prep_learned |
| Raw stack (exit) | – | – | – | none | render_simple |

- **Reference**: in every chain when the project has a reference image (hidden, in the Astrometry block).
- **Sharing**: Classic shares inventory → color with Full; Classic + denoise also shares Full's denoise. The four
  together cost about 1 h more than Full alone, 40 min of it star_net.
- **Rules** (`api.set_option` returns toast notes, as the mockup's `with_tool`/`without_tool`): Stars needs Denoise;
  removing Stars removes neural and turns H-alpha into polynomial; in v1 H-alpha needs Stars (the existing background
  works on the starless image). Polynomial with Stars = `background` with `surveys.survey="none"` (the existing
  physical background without the survey); without Stars = `background_poly`. psf_neural with a stored
  `starless.learned_stars` file omits star_net.
- **BLOCKS**: Calibration = inventory, calibrate; Registration = register; Selection follows register (Run to here
  runs register); Stacking = stack; Astrometry = astrometry, reference, catalogs, surveys; Color = color; Denoise =
  denoise; Stars = optics, stars, star_net, starless; Gradient = background | background_poly; Output = the output
  stage. A block's key is the tuple of its stage keys; Astrometry's leaves out surveys (it follows the Gradient
  method), as in the mockup's `blocks.block_key`.

### 2.2 How a stage finds its inputs
- **Dependencies.** `DEPS2[s]` = declared deps ∪ `EXTRA[s]`, with `EXTRA = {starless: (star_net,)}` (today an implicit
  read through starless' auto value). `OPTIONAL[s]` may be absent: starless: star_net; background_poly: denoise;
  render_simple: color, denoise, background_poly. In a chain, `deps_in_chain = DEPS2 ∩ chain`; a missing required
  dependency is a ValueError naming the block.
- **At run time** the shim maps every ancestor; an absent stage resolves to a non-existent folder (1.4). Only the new
  stages test `ctx.has()`; existing stages never sit in a chain that lacks one of their required dependencies.
- **Stage lists.** `stages.all_stages()` is unchanged; `workflow.V2_STAGES` adds the new stages, so v1's Pipeline
  never runs them. `Chains` (stages, order, DEPS2, OPTIONAL, EXTRA, BLOCKS, `chain(options, end)`) defaults to V2;
  tests build toy Chains.

### 2.3 New stages for v1 (P4)
- **background_poly** (CPU, ≈ 1 min; deps stack, color; optional denoise): per channel, in display RGB (M applied), a
  polynomial on the darkest blocks of a grid with 2.5 σ iterations (`simple_sky` in `mockup/alt2/build/prepare.py`).
  Params degree 2 (1–5), samples 16 per side (4–64), object_clip 2.5. Writes coefs.json, samples.npz, preview
  "background"; metrics sky_fit_blocks_pct, gradient_pct.
- **render_simple** (CPU, ≈ 30 s; deps stack; optional color, denoise, background_poly): lin.npy = M·src − sky (src =
  n2n_full when denoised, else drz_full; without color, identity M and a per-channel median sky), lin_raw.npy (with
  denoise), valid.npy, prep_simple.json {M, crop, luminance median, MAD, p99.995}. Params `stretch` "auto" (others ◇),
  `linked`.
- **render/simple.py**: `SimpleRenderEngine(dir).render(SimpleSettings, scale)` returns the `Layers` type (stars
  None): linked MTF stretch, S-curve contrast, saturation, noise mix (`stretch()` in prepare.py).
- **render/sliders.py**: the 9 sliders of POST.md 2.1. For engine layers, position 2 is the layer's base
  RenderSettings; other positions add the table's step (sky_level, faint_smooth, g_lim, d10, d7) or multiply by its
  ratio (lc_*, chroma, neb_sharp, fine_k, star_sat, star_glow), linearly between integers. Simple layers use
  prepare.py's classic tables. `availability(kind, has_denoise)` gives the reasons for greyed sliders.

### 2.4 Deferred, and how v1 shows the real order
- **Deferred**: Deconvolution; fine-tuned / pretrained / NL-Bayes denoise; Gradient before Color and Stars; a PSF
  before Denoise; the stretch choice; flats per session; the master library.
- **Real order in the graph**: Frames, Calibration, Registration, Selection, Stacking, Astrometry, Color, Denoise,
  Stars, Gradient, Output, Post. "Linear processing" runs Astrometry → Gradient; with Stars, Gradient's wire comes
  from Stars ("starless image"). Spec mode notes "◇ before Color in a later version" and shows the Deconvolution card,
  disabled.
- **The reorder comes later as new methods, never as a switch on old stages.** Example: Gradient "H-alpha survey, on
  the stack" = new stage `background_early` (deps stack, astrometry, catalogs, surveys), with color and stars depending
  on it through EXTRA/OPTIONAL. Its block sits in a column "Gradient (early)" before Color, shown only when used; old
  keys stay valid; it becomes the default only after non-regression and blind QC. Deconvolution comes the same way (a
  column after Color, its own PSF stage).

## 3. Runner v2 (core/runner2.py, core/estimate.py)
- **Plan:**
  ```
  need(wf, s, as_input=False):
      k = key(wf, s)
      if done(k) and (not as_input or usable_as_input(s, k)) or k in jobs: return
      for d in deps_in_chain(wf, s): need(wf, d, as_input=True)
      jobs[k] = Job(k, s, block, wf=sharing(k) over all workflows, deps={d: key(wf, d)}, est_s, disk_gb)
  ```
  Run all = `need` on every chain stage (reference included); Run to block = `BLOCKS[block] ∩ chain(wf)`. Jobs are
  sorted by global stage order then workflow; each key appears once.
- **Launch** (in the API, without Qt, so the CLI and the client share it): write `logs/runs/<run_id>.plan.json`
  {run_id, device, targets, snapshot of the workflow definitions, jobs}, then start `python -m astro_gpu.core.runner2
  <project> <plan>` detached (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`, output to `logs/runs/<run_id>.log`). The
  plan is frozen: edits during a run do not change it.
- **Worker**: refuse if run.lock is alive; write run.lock {pid, run_id}; remove STOP. Per job: skip it if done and
  usable; give a linked key a fresh folder; build the StoreProject and assert the key against the snapshot; run
  `Pipeline(shim, V2 stages of the chain, RunJournal)._run_stage(stage)` (v1's checkpoints, `deterministic()`,
  manifest, Interrupted / failed); rewrite the manifest (key fields, run_id = key); `estimate.record(…)`. Stop at the
  first failed or interrupted job, then `run_end`.
- **Events** in `logs/events.jsonl` (append-only, byte offsets): every event carries `run`; stage events add `key`,
  `wf` (ids) and `block`; a new `plan` event lists the jobs. At run_start a file over 64 MB is renamed
  `events_<date>.jsonl`, never deleted.
- **Cancel and locking**: the STOP file, as today (interrupted, resumed from `_ckpt` when the same key runs again).
  One runner per project (run.lock); two projects may run at once on two GPUs (the plan's device, else `devices.main`).
- **The client follows** with `RunFollower`: it tails `events.jsonl` every 500 ms from the launch offset (that run
  only) and updates the run rows (by key), the block stripes and the Log console. At `stage_end done` it refreshes the
  states (`api.states()` writes last_keys; a done output stage makes its Post layer available). On project open with a
  live run.lock it reattaches through the plan file.
- **Estimates**: `units(s)` = inventory frames; calibrate / register lights × sensor Mpx; stack lights × Mpx × scale²;
  denoise iterations; star_net steps; other stages stack grid Mpx. With history in `STATE/timings2.json` (last 20
  [s, units, gpu|cpu] per stage): median(s/units) × units, "measured". Otherwise "≈": the Heart reference
  (`mockup/alt2/data/data.json` stage seconds and gb, Heart units) × the units ratio, × the mockup's `CPU_FACTOR`
  without a GPU; new stages 60 s and 30 s at Heart size. Disk likewise.

## 4. The engine API (astro_gpu/api.py; rendering in render/layers.py)
| Function | Contract § | Built on |
|---|---|---|
| `list_projects(ws)`, `create_project(root, name, target=None)`, `open_project(root)` (LegacyProject), `save(p)`, `save_as(p, root)` (definition only) | 1 | project2 |
| `import_v1(old_root, new_root, path_map, mode="copy", dry_run=False) -> ImportReport` | 1 | import_v1 |
| `scan_folder(folder, progress)`, `add_frames(p, scan, session=None)`, `remove_session`, `group_sessions(infos, criteria)` (night noon-noon, exposure, ISO) | 2 | list_frames, read_info, newproject |
| `set_excluded(p, names, excluded)`; `health_checks(p)` (no bias, FWHM per session, disk; the rest ◇) | 2 | stack.exclude, space_text |
| `frame_table(p, wf)`, `selection_preview(p, wf, min_transp, max_fwhm_ratio)`, `frame_thumb(p, path, size)` (one stretch per session) | 2 | reg.npz, frames.json, `select_frames`, read_cfa |
| `TEMPLATES`, `workflows(p)`, `add_workflow(p, name, options, params=None, copy_of=None)`, `rename_workflow`, `duplicate_workflow`, `delete_workflow` | 3 | project2, workflow |
| `set_option(p, wf_ids, key, value) -> notes`, `set_param(p, wf_ids, stage, name, value or None)`; scope = list of ids ("sharing this block" = `sharing()`) | 3 | workflow rules |
| `make_variant(p, wf, change, name)`, `keep_variant`, `discard_variant`, `add_exit(p, wf, "stacking", name)`, `set_workflows(p, list)` (undo) | 3 | project2 |
| `methods(block) -> [MethodInfo(key, name, short, runs_on, stage, new)]`, `chain(p, wf)`, `deps(p, wf, s)`, `blocks(p, wf) -> [BlockRef(block, stages, key)]` | 3–4 | workflow, store |
| `stage_key(p, wf, s)`, `sharing(p, wf, s)`, `cache_usage(p)`, `clear_unused(p)` | 4 | store |
| `states(p) -> {(wf, s): StageState(state, reason, key, previous_key)}`: done, pending, stale, running, failed, interrupted | 5 | store, state.json |
| `plan_all(p)`, `plan_to_block(p, wf, block)`, `cost(p, workflows) -> Cost` (what-if: Replace, variant, toolbar, New workflow) | 5 | runner2 |
| `estimate(p, wf, s) -> Estimate(seconds, measured)`, `machine()` (GPUs, VRAM, cores, RAM) | 5 | estimate, torch, psutil |
| `start_run(p, plan, device=None) -> RunInfo`, `running(p)`, `cancel(p)`, `read_events(p, offset)` | 6 | runner2, journal |
| `stage_outputs(p, wf, s, which="current"/"previous")`, `stage_array(p, wf, s, name, scale)`, `pixel_to_sky`, `spectral_response` | 7 | manifests, npy, wcs.fits, specresp.npz |
| `describe(p, wf, s) -> [ParamInfo(…, value, suggested, origin, reason, drift)]` | 8 | `Pipeline(shim).effective_params` |
| `sliders_of(p, wf)`, `render_layer(p, wf, base, sliders, scale, part=None) -> LayerImage(rgb, crop, info)`, `post_masks(p, wf, scale)` | 9 | layers.py, RenderEngine, SimpleRenderEngine |
| `export_image(p, rgb, meta, options, progress)`, `export_workflow(p, wf, base, sliders, options)` (headless) | 11 | layers.py, render/export |

- **Layers**: `render_layer` returns the full stack grid, uncropped, plus the crop; Post crops to the intersection of
  the visible layers; another drizzle scale is resampled ("resampled"); a full-resolution region is a crop of a cached
  full render. `RenderEngine` is built on the key folder (`StarRemovalEngines` assumes sibling folders: not used).
- **Export in v1**: TIFF 16 / 32f, PNG 16, JPEG, FITS 32f + WCS; sRGB; full, 2048 or 4096 px; v1 orientation; layers
  and masks as files; a history sidecar. **◇**: master library, match_calibration, benchmark, preview_region,
  identified objects, script export, XISF, other profiles, linear outputs of Stars workflows.
- **In the client**: the composition and the 13 filters (`postengine`). **cli2** (`python -m astro_gpu.cli2`): new,
  import, add-workflow, set, status, plan, run [--detach], stop, export.

## 5. The client (astro_gpu/client/)
### 5.1 What is copied
`python -m astro_gpu.client`. Copied from mockup/alt3/app and alt2/app (viewer.py, widgets.py) with relative imports;
`alt3/app/locale/fr/*.json` goes to `client/locale/fr/`. Removed: the Demo menu, `datatext.py`, every alt2/data
reference; "About this mockup" becomes About.

| Mockup | Client | Change |
|---|---|---|
| app3.py | app.py | real New / Open / Import / Recent; Save flushes; Run, Run to here, Stop via services/run |
| project.py + model.py | model.py | `ProjectModel` adapter (5.2) |
| blocks.py, graph.py | same | v1 column order (2.4); `block_key` from `api.blocks` |
| panel.py, settings.py, dialogs.py | same | `resolve()` reads `api.describe` (cached, async); NEW rows in Spec mode only, disabled |
| steps.py, tune.py, charts.py | same | previews and `stage_array`; metrics; charts from frame_table, spectral_response |
| frames.py, sources.py | same | real scan, sessions, thumbnails (worker); Keep / Exclude → `set_excluded` |
| runs.py | same | plan dialog from `api.plan_*`; non-modal progress; Stop |
| post3.py, postengine.py, exportdlg.py | same | `layer_rgb` → RenderWorker; `Ctx.stars/halpha` → `post_masks`; real export |
| home.py, settingsdlg.py, logview.py, i18n.py, prefs.py | same | workspace scan; prefs path; Log = history + run console |

### 5.2 The model adapter
- **`ProjectModel(QObject)`** keeps the surface the copied views use, each member a thin api call; edits autosave
  (debounced 300 ms). Signals pathsChanged, currentChanged, costChanged, sourcesChanged; attributes paths, path,
  current, sel_step, scope, pending_change, sessions, stack_mode, gpu, history; methods state, pending_upto,
  all_pending, seconds, stage_key, owner, cost_with, apply_change, add/keep/drop_variant, add_path, remove_path,
  set_opts, has/with/without_tool, slider_state, thresholds_now, set_selection, kept_frames, add_exit…
- **`WorkflowView` replaces PathDef** (pid = workflow id, name, params, end, variant_of, computed, rendered). Its
  `opts` speaks the mockup's vocabulary: `debruitage`/`dn` ↔ denoise, `etoiles`/`appris` ↔ stars, `fond` ↔
  gradient, `deconv` ↔ deconvolution. The views keep their internal block keys (calibration, alignement…).
- **`s.data[...]` disappears**: each package replaces it in its own files with `model.info()`, `model.image_size()` or
  a service call.

### 5.3 Threads and processes (the GUI never blocks)
- **GUI thread**: widgets, workflow edits, keys, states and plans (JSON and small manifests, cached, invalidated by
  edits and run events). Never there: array loads, `auto()`, renders.
- **ApiWorker** (QThread, `services/worker.py`): describe, outputs, arrays, scans, thumbnails, import. Requests carry
  a channel; the latest per channel wins; results come back as signals.
- **RenderWorker** (QThread, `services/render.py`): one LayerService per workflow (at most 2 engines loaded) plus the
  postengine. Requests (layers, level, region, upto, want_mask) are coalesced by sequence number, latest wins (as
  `gui/render_tab.RenderWorker`): 1/4 while dragging, 1/2 on release, the full-resolution region at zoom ≥ 50 %.
- **Runs**: the detached worker (§3), followed by RunFollower (a QTimer tailing a file).

### 5.4 Languages and preferences
- **i18n.py** is unchanged: English source, French in `locale/fr/*.json`; engine labels, help and reasons also pass
  through `_()` (missing French shows English); a test lists untranslated strings (report only).
- **`prefs.py`**: `%APPDATA%/astro_gpu/settings.json`, overridden by `ASTRO_GPU_SETTINGS`; defaults workspace
  F:/astro_projets, en, spec_mode off, device auto, recent []. Tests and walkthroughs set `ASTRO_GPU_SETTINGS`,
  `ASTRO_GPU_HOME` (scratch) and `QT_QPA_PLATFORM=offscreen`; a test checks the real settings file is untouched.

### 5.5 What stays ◇ in v1
Shown in Spec mode only, disabled, with "◇ Not in the engine yet"; hidden otherwise; the status note reads "Spec mode ·
◇ not in the engine yet". The list: the 41 NEW settings; fine-tuned / pretrained / NL-Bayes denoise; Deconvolution;
the stretch choice; Gradient sample areas; flats per session and one stack per session; the master library;
temperature and library checks; Benchmark; Gaia local copy, ASTAP, astrometry.net; File formats; Cache limit; Preview
on a region; Identified objects; export to XISF, Adobe RGB / P3, linear outputs, project as a script.

## 6. Work packages
Every package keeps `pytest -m "not slow and not network" -q` green, adds its own tests, commits on `pro-client` and
writes a JOURNAL entry; no two packages edit the same file.

**Waves**: (1) P1 · (2) P2, P4 and P9a in parallel · (3) P3 · (4) P5 and P6 in parallel, P5 committing its
signatures first · (5) P7, then P8 in the background, in parallel with P9b · (6) P10, P11 and P12 in parallel ·
(7) P13 · (8) P14.

- **P1 · Project v2 and workflow chains** (2 h). Files `core/project2.py`, `core/workflow.py`, `tests/test_project2.py`,
  `tests/test_workflow.py`. Scope 1.2, 1.5, 2.1, 2.2 (Chains, toy Chains, rules with notes, BLOCKS, implied params,
  templates, columns). Accept: the 5 chains equal 2.1's table; invalid options raise; the mockup's 8 with/without_tool
  cases hold; save/load round-trips; `from_v1` on the 9 real project.json gives 1.5's names and options, and a
  file-list + mtime snapshot shows nothing was written. Risk low.
- **P2 · The stage store** (2–3 h, after P1). Files `core/store.py`, `core/stage.py` (`Context.has` only),
  `tests/test_store.py`. Scope 1.3, 1.4, `states()` with last_keys, `sharing`, `cache_usage`, `.link`. Accept, on toy
  stages A → B (optional) → C: the same upstream shares a key, and a parameter changes only the downstream keys;
  reverting restores the old key; workers is ignored; a subprocess computes the same key; copied inputs (same names and
  sizes) keep the inventory key, reordered frames change it; absent stages are never created. Risk medium (the
  foundation).
- **P3 · Runner v2, estimates, hand exclusions** (2–3 h, after P2). Files `core/runner2.py`, `core/estimate.py`,
  `stages/stack.py` (the `exclude` param only), `tests/test_runner2.py`. Accept, on toy stages: shared stages run once;
  events carry run, key and wf; a second plan is empty; a change plans only its downstream, and reverting plans nothing;
  run to a block works; STOP → interrupted → resume from the checkpoint (the `test_core.StageB` pattern); the lock
  refuses a second worker; a linked key gets a fresh folder; a detached subprocess is followed through `read_events`. On
  the engine: `select_frames` honours `exclude`; `Pipeline.status()` of F:/astro_essais/projets/heart_a7s stays all
  done; estimates are within 10 % of Heart's stack and denoise times.
- **P4 · New stages and simple rendering** (2–3 h, after P1, in parallel with P2 and P3). Files
  `stages/background_poly.py`, `stages/render_simple.py`, `render/simple.py`, `render/sliders.py`, `render/layers.py`
  (`LayerService(dir, kind, device)`: render, post_masks for stars and H-alpha as in prepare.py, export_image),
  `tests/test_simple_stages.py`, `tests/test_sliders.py` (a local dict shim, not P2). Accept: a synthetic degree-2
  gradient is recovered within 1 %; render_simple writes its files; the simple render of 2000 × 1500 at scale 1 takes
  < 2 s; each slider's position 2 equals the base and each slider is monotonic; greyed sliders give their reasons;
  `render_layer` works on `test_render_engine`'s synthetic prep; exports re-open with the right dtype and size.
- **P5 · Engine API and cli2** (3 h, after P2, P3 and P4). Files `api.py`, `cli2.py`, `tests/test_api.py`. Accept: a
  toy scenario (create; 4 workflows; a parameter on 2; make, keep and discard a variant; rename; delete; undo through
  `set_workflows`) in which `cost()` equals the plan after each change. After P6: `describe()` on a real adopted stack
  gives value, suggested, origin and reason.
- **P6 · Import of v1 projects** (2 h, after P3, in parallel with P5). Files `core/import_v1.py`, `tools/import_v1.py`,
  `tests/test_import_v1.py`. Accept: a toy v1 project adopts the right prefix, and a changed old parameter stops
  adoption there. Slow test, copying F:/astro_essais/projets/heart_a7s → F:/astro_essais/v2/heart_a7s: 15 stages
  adopted; `plan(Full)` empty; `plan(Classic)` = background_poly + render_simple; link mode copies 0 bytes; the old
  snapshot is unchanged. `--dry-run` on the 9 real projects matches 1.6 and writes nothing. Risk medium: if a snapshot
  check ever fails, never touch a real project.
- **P7 · End to end on a reduced real set** (1–2 h, ≈ 40 min of GPU, after P5 and P6). Files
  `tests/test_integration_v2.py` (slow), `tools/v2_smoke.py`. Set-up: a v2 project from
  F:/astro_essais/donnees/heart_a7s (`inventory.max_lights` 12, denoise 500, star_net steps 300, `starless.sharpen`
  off), the 4 templates plus a Raw stack exit, run all through cli2, detached. Accept: jobs = distinct keys (inventory →
  color once); all done, and a second run plans nothing; every layer renders at 1/4 without NaN; Full exports; the times
  and disk per stage are logged.
- **P8 · The user's projects on F:** (2 h of agent work + ≈ 10 h of runs, after P7). Files `tools/projects/projects.py`
  and the data under F:/astro_projets/. Order: Heart, Iris, Rosette, Horsehead, FlamingStar, M31, Wizard, NGC7822, M33.
  (1) Copy the raws not on F: (sources: ENGINE_KNOWLEDGE §2) to `F:/astro_projets/_raw/<name>/<kind>/` (robocopy
  /COPY:DAT), with folder names that keep the old frames.json order (FlamingStar has two folders); Heart and NGC7822
  raws are referenced in place. (2) Import in copy mode with the path map; log the report; star layers and references go
  to `inputs/`. (3) Workflows Classic stack, Classic + denoise, Full, Full + neural stars, plus the imported "Full +
  earlier neural layer" where it exists; run all via PowerShell Start-Process in two queues (cuda:0, cuda:1).
  (4) Bases faithful / artistic from the old `results/*/*_settings.json`; export each Full* with both and Classic with
  its defaults; default Post = Full visible at the bottom. Accept: `cli2 status` all up to date; exports exist; bulletin
  numbers logged; the old folders' snapshots unchanged. Risks: catalogue outages, 8 GB on cuda:1, run time.
- **P9a · Client shell** (2–3 h, after P1). Files `client/__init__.py`, `__main__.py`, `app.py`, `i18n.py`, `locale/`,
  `prefs.py`, `widgets.py`, `viewer.py`, `home.py`, `settingsdlg.py`, `logview.py`, `dialogs.py`, `services/worker.py`,
  `pyproject.toml` (entry `astro-gpu-client`), `tests/client/test_shell.py`. Accept (offscreen): Home lists a scratch
  workspace; New project writes a v2 folder; French and back rebuild the menus; prefs go only to the
  `ASTRO_GPU_SETTINGS` file; a grep of `client/` finds no "alt2", "data.json", "Demo" or mockup import; shots home_en,
  home_fr, settings.
- **P9b · Graph and panel on real workflows** (3 h, after P5). Files `client/model.py`, `blocks.py`, `graph.py`,
  `panel.py`, `settings.py`, `tests/client/test_graph.py`. Accept, on heart_a7s v2 with 4 workflows: the tree has
  Calibration → Color shared by 4, Denoise by 3, a split at Stars (the methods differ), and 4 outputs; the states come from the API; adding and
  removing Denoise and Stars follow the rules, with toasts; a change on a shared done block offers Replace or Try as
  variant, with `api.cost` times; Undo and Redo work; shots graph, panel_shared, variant.
- **P10 · Inspect and Frames** (3 h, after P9b). Files `steps.py`, `tune.py`, `charts.py`, `frames.py`, `sources.py`,
  `services/inspect.py`, `tests/client/test_inspect.py`. Accept: the real stack shows with auto-stretch, histogram and
  metrics; Selection thresholds update the kept count through `selection_preview`; the Color chart draws; an Outdated
  block shows "previous result"; Compare works; importing a heart_a7s folder gives the right counts, sessions and
  thumbnails; Exclude changes the Stacking key; shots.
- **P11 · Run** (2 h, after P9b, in parallel with P10 and P12). Files `runs.py`, `services/run.py`,
  `tests/client/test_run.py`. Accept: the plan dialog shows the blocks, the sharing workflows and the durations (≈
  marks); a real render_simple job runs with the heartbeat under 200 ms, the stripes turn green and the layer appears;
  Stop during denoise (3000 iterations) gives interrupted, and the next run resumes; closing and reopening during a run
  reattaches.
- **P12 · Post and Export** (3 h, after P9b). Files `post3.py`, `postengine.py`, `exportdlg.py`, `services/render.py`,
  `tests/client/test_post.py`. Accept, with the Full and Classic layers of heart_a7s v2: a drag renders the draft at 1/4,
  then 1/2 on release (times logged); the stars, H-alpha, luminance and brush masks work; filters, split stars /
  starless and Undo work; TIFF16, JPEG and FITS32 + WCS export and re-read; shots.
- **P13 · Integration and walkthrough** (2 h, after P10, P11 and P12). Files `client/app.py` (final wiring),
  `tools/client_walkthrough.py`, `tests/client/test_walkthrough.py`. Accept: every screen of SPEC §3 shot in EN and FR in
  `F:/astro_projets/_shots/<date>/`; the heartbeat stays under 200 ms outside project open; all suites green.
- **P14 · QC** (2–3 h, after P8 and P13). Files `tools/qc/*` (a `--v2 WORKFLOW` option where needed), output in
  `F:/astro_projets/_qc/`. (1) The bulletin on the new exports. (2) Blind sheets: new against old faithful / artistic
  (the old project read-only, for the framing), and Classic against Full. (3) Independent judges (`qc_prompt.md`), plus
  your own look at 100 %. (4) An independent GUI review of the walkthrough shots against the mockup. Accept: a verdict
  per project (better / same / worse, and where) and the defects per screen, in the JOURNAL. Nothing reaches the user
  without passing.
- **P15 · Phase D (optional).** New methods, each a stage plus a method entry with its ◇ removed: Gradient early
  (`background_early`), Richardson-Lucy deconvolution, NL-Bayes denoise, stretch choices. Each needs non-regression on
  the 4 datasets and blind QC before it becomes the default.

## 7. Decisions and risks
**Decisions and why:**
1. **v2 beside v1, import rather than in-place migration**: real projects stay read-only; the QC tools keep working.
2. **Keys without auto values**: they derive from the upstream keys. If `describe()` ever shows drift, bump the stage
   version; never add auto values to the key.
3. **Jobs run by v1's `_run_stage` through a shim**: one runner logic, already tested.
4. **Two new stages** make 4 templates real; Classic + denoise reuses Full's denoise.
5. **The real order in the graph**; the reorder only as new methods: quality never changes silently.
6. **Import by copy, with checks**: clean, self-contained folders on F:; link mode for a quick look.
7. **The mockup's internal keys stay in the client** (less porting risk); the engine and the API use English keys.
8. **The Post compositor stays in the client** (numpy, as in the mockup); the engine renders layers on the stack grid.
9. **A frozen plan file per run**: editing stays possible during runs that last hours.

**What the orchestrator must watch:**
1. **Writes to real projects**: check the snapshot tests of P1, P6 and P8; `project.json` is never written.
2. **Detached runs**: jobs started inside an agent session die with it. Launch long runs with PowerShell
   Start-Process and track them by PID.
3. **The GPU**: P8 holds both GPUs while P11–P13 need renders. Shoot the walkthroughs on a free GPU or on the CPU;
   cuda:1 drives the display.
4. **Disk**: 30–60 GB of store per project, plus 17–30 GB of calibrated cache when calibrate re-runs (F: has 2.5 TB
   free). Old keys are cleared only after a confirmation.
5. **Raw order** (FlamingStar): a different order makes the import refuse and recompute. That costs 3 h, not
   correctness.
6. **Demo leftovers in a 9 000-line port**: every client package extends P9a's grep test.
7. **Catalogue outages**: the adopted catalogs and the cache cover the 9 fields; do not stall on them.
8. **Two failed attempts on a package**: stop it, log it, move on. Priority: P1–P7 and P9–P13, then P8, then P15. P8
   still starts as soon as P7 passes, since it runs in the background.
