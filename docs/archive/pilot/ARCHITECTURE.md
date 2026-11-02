# astro_gpu: architecture map (branch pro-client, 2026-10-31)

This file maps the existing code for the pilot. It is factual and was produced read-only: no code was changed. Paths
are relative to the repository root, and `file:N` is a line number. Python:
`C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe` (astro_gpu installed editable). About 15 100 lines in
`astro_gpu/`.

## 1. Package layout

| Module | Role |
|---|---|
| `astro_gpu/__init__.py` | Sets `CUBLAS_WORKSPACE_CONFIG` before torch loads (reproducible GPU sums). `__version__` 0.1.0. |
| `cli.py` | argparse CLI: `new`, `status`, `run`, `stop`, `params`, `set`, `render`, `variant`, `gui` (`main` :164). Entry point `astro-gpu`. |
| `core/project.py` | `Project`, `ProjectConfig`, `Inputs`, `Target`, variants, `stage_graph()`, per-user `STATE` / `REGISTRY`, `list_frames`. |
| `core/stage.py` | `Stage` base class (name, title, deps, version, Params, `auto`, `run`) and `Context` (files, journal, progress, previews, stop, device). |
| `core/params.py` | `P()` field factory with UI metadata, `StageParams.describe()`, `Auto(value, reason)`, `resolve()`, `params_hash()`. |
| `core/runner.py` | `Pipeline`: status, plan, run, manifests, checkpoints, run lock, stop file, timing history. |
| `core/journal.py` | `Journal` writes `logs/events.jsonl` and `logs/journal.md`; `read_events(path, offset)` tails the events. |
| `core/methods.py` | `DECISIONS`: 8 plain-language choices, each option a `{stage: {param: value}}` patch (Methods tab, ComfyUI). |
| `core/newproject.py` | Guesses from the lights folder: calibration folders, project name/folder, scratch, target, disk space. |
| `core/determinism.py` | `deterministic()`: deterministic torch / cuDNN / cuBLAS. Every stage and every render runs inside it. |
| `core/imgutil.py` | numpy/OpenCV helpers: robust sigma, polynomial fit, hole filling, starlet, guided filter. |
| `core/sharpen.py` | PSF field + fine-tuning of the N2N network for non-blind sharpening (used by starless). |
| `core/gaia_archive.py` | Gaia DR3, XP spectra, Hipparcos from VizieR, with GAVO and ESA fallbacks. |
| `core/blindsolve.py` | All-sky blind solver, quad hashing on Hipparcos / Tycho-2; index cached in `~/.astro_gpu/cache`. |
| `core/plots.py` | Matplotlib figures in the dark theme, returned as RGB for previews. |
| `io/frames.py` | RAW (rawpy, EXIF) and FITS (.fz, .gz) readers: `read_info`, `read_cfa`, `FrameInfo`, `cfa_channel_map`. |
| `stages/__init__.py` | `all_stages()` (:7): the 17 stages in execution order. |
| `stages/<stage>.py` | One module per stage (section 4). Helpers that are not stages: `far_halo.py`, `star_shape.py`, `star_net_lib.py`. |
| `render/engine.py` | `RenderEngine` (:56) renders on the GPU from render_prep layers; `StarRemovalEngines` (:510) switches model / learned star removal. |
| `render/ops.py` | torch primitives: blur, box filter (float64 cumsum, :68), guided, median, percentile, sRGB, OKLab, gamut, screen. |
| `render/settings.py` | `RenderSettings` (:15): 73 sliders in 5 sections, 40 basic. `PRESETS` (:261). |
| `render/export.py` | `ExportSettings` (:16), `export()` (:63), `quality_checks()` (:136). |
| `render/views.py` | Views = named (variant, render settings) pairs: `default_views`, `ensure_default_views` (:69). |
| `render/clipping.py` | Clipping overlay (black, white, burnt), shared with ComfyUI. |
| `gui/*` | PySide6 heavy client (section 6). |
| `resources/` | `GUIDE.md` (F1 help), `generic_bayer_response.npz`. |
| `comfyui_astro/` | ComfyUI nodes. `bridge.py` runs the pipeline in-process (`run_up_to` :167, `engine_for` :198); `blocks.py` = elementary blocks cached by a fingerprint of inputs + settings (`_key` :56, `_run` :88). |

## 2. The project model

**On disk** (docstring `core/project.py:1-19`):
```
<project>/project.json               ProjectConfig (below)
<project>/work/<stage>/              outputs + manifest.json + previews/*.jpg + _ckpt/ (while resumable)
<project>/variants/<slug>/<stage>/   the stages a variant computes itself
<project>/results/<name>/            exports + <name>_settings.json
<project>/logs/                      events.jsonl, journal.md, run.lock (pid), STOP, worker.log
<scratch_dir>/<stage>/               large temporary data, e.g. calibrate/lights.f16 (float16 memmap of calibrated frames)
```
`Project.create` (:89) defaults scratch to `work/_scratch`; the New project dialog proposes `~/astro_gpu_scratch/<name>`
(`newproject.guess_names` :56).

**project.json** = `ProjectConfig` (:66), saved atomically by `save()` (:113); `FORMAT = 1`, no migrations. Fields:
- `name`, `description`, `created`, `format`; `scratch_dir`; `reference_image` (a previous processing, for framing / QC).
- `inputs` = `Inputs` (:49): `lights`, `darks`, `flats`, `bias`, `flat_darks`, each a list of folders **or files**
  (FlamingStar lists 100 dark files). `list_frames` (:331) / `frame_kind` (:37) accept RAW, FITS, .fz, .gz.
- `target` = `Target` (:58): `name`, `ra`, `dec`, `focal_mm` (hints for the astrometry).
- `params`: `{stage: {param: value}}`, the user overrides of main.
- `render`: `{"settings": {last sliders}, "views": {name: {variant, settings, created, note}}}`.
- `devices`: `{"main": "cuda:N", "aux": "cuda:M"}` (Heart: `cuda:1`; only star_net uses `aux`).
- `variants`: `{name: {params, note, created, slug}}`; `active_variant` (`''` = main).

**Per-user state** `STATE = $ASTRO_GPU_HOME or ~/.astro_gpu` (`project.py:292`): `projects.json` (recent projects, 30
kept, `remember` :296), `timings.json` (last 20 durations per stage, for ETAs, `runner._record_timing` :273),
`camera_profiles/` (`color.py:22`). The download cache is always `~/.astro_gpu/cache` whatever ASTRO_GPU_HOME says
(gaia/, xp/, hipparcos/, quad indexes; `catalogs.py:16`, `blindsolve.py:18`); surveys use the astropy cache. Tests patch
`REGISTRY` (`tests/conftest.py`); `tools/qc/jeux.py` sets ASTRO_GPU_HOME to `F:/astro_essais/etat`.

**manifest.json** (written by `runner._run_stage` :177-204): `stage`, `title`, `version`, `status` (running | done |
failed | interrupted), `params` (full effective values), `origin` (per parameter: `default` | `auto` | `user`),
`auto_reasons` (one line per automatic value), `params_hash`, `inputs` (`{dep: dep.run_id}`, plus `_project_inputs` for
inventory), `started`, `finished`, `duration`, `run_id` (random uuid4), `metrics` (flat dict from `ctx.metric`), and on
failure `error`, `traceback`.

**Parameters.** Each stage declares `class XParams(StageParams)` with fields built by
`P(default, label, help, tip, level, unit, ge/le, step, choices)` (`params.py:29`). `describe()` (:47) returns
`[{name, type, default, ge, le, ..., level, label, help, tip, unit, step, choices}]`. 115 fields over 17 stages, 23 basic;
render_prep_learned repeats render_prep's 14, so 101 distinct fields, 22 basic; 32 have a `tip`.

`Stage.auto(ctx)` returns `{name: Auto(value, reason)}` computed from the dependencies' outputs. Examples: stack `scale`
("FWHM 4.0 px with 721 frames: under-sampled, enough frames to drizzle"); calibrate/register `workers` = CPU count;
denoise `iterations` ∝ sqrt(Mpx), 6000-30000. `resolve()` (:74) applies default < auto < user, returns the model and
`origin`, and refuses unknown keys. `Pipeline.effective_params` (:69) returns `(params, origin, auto)` and swallows auto
errors when the dependencies have not run. `cli params` prints the same (`cli.py:73`).

## 3. The pipeline and the runner

**Declaration.** A stage class sets `name`, `title`, `description`, `deps` (tuple of names), `version` (string; a bump
makes old results stale), `Params`, `cost_hint` (never used) and `network`. The order is the fixed list of `all_stages()`
(`stages/__init__.py:7`): inventory, calibrate, register, stack, astrometry, reference, catalogs, surveys, color,
denoise, optics, stars, star_net, starless, background, render_prep, render_prep_learned. `stage_graph()`
(`project.py:281`) = `{name: deps}`. No optional or alternative stages: star_net is turned off by `enabled=False`,
reference does nothing without a reference image, render_prep_learned writes `none.json` when there is no learned layer.

**Starting a run.**
- CLI: `python -m astro_gpu.cli run <project> [--variant NAME|main] [--until STAGE] [--force S ...]` (`cli.py:188`), exit
  0 when done. Also `status [--variant]`, `stop` (writes `logs/STOP`), `params <project> [stage]`,
  `set <project> stage.key=value|auto [--variant]`,
  `new <project> --lights ... [--darks --flats --bias --flat-darks --target --ra --dec --focal --scratch --reference]`.
- Python: `Pipeline(Project(root), all_stages()).run(until, force)` (`runner.py:132`) returns done | failed | interrupted.
- GUI: detached process `QProcess.startDetached(python, ["-m","astro_gpu.cli","run",root,"--variant",v,...])`
  (`gui/runner.py:48-71`).
- ComfyUI: in-process `bridge.run_up_to` (:167) inside `_spawn_without_main()` (:151), because Windows spawn workers
  re-import `__main__`.

**Up to date or stale.** `Pipeline.status()` (:83) says `done` only when: the manifest is `done`; every dependency is
done; `version` is unchanged; every `inputs[dep]` equals that dependency's current `run_id`; for inventory,
`_project_inputs` equals `sha1(input lists)` (`_inputs_fingerprint` :223: the **path lists** only, not file contents or
dates); and the current effective params (auto recomputed) hash to `params_hash`, or `_same_params` (:231) accepts them
(new parameters at their default and removed ones are tolerated). Otherwise: `ready`, `pending`, `stale` (with a reason),
`failed`, `interrupted`, or `running` when `logs/run.lock` holds a live pid (`_run_alive` :251).

`plan(until, force)` (:118) keeps what is not done, plus forced stages and everything below them, limited to `until` and
its ancestors (`_ancestors` :263). `run()` (:132) plans again after each stage (auto values may read optional outputs)
and never runs a stage twice in one run. Because `run_id` is random, any rerun of a stage invalidates everything below it,
even with identical output. One result per stage per variant folder: caching is not content-addressed.

**Inside a stage** (`_run_stage` :162): the checkpoint tag `{params_hash, inputs, version}` is compared with
`_ckpt/meta.json`; if it differs, `_ckpt` and every output in the stage folder are wiped (`_clear_outputs` :211). The
manifest is written `running`, `st.run(ctx, p)` runs inside `deterministic()`, then the manifest becomes `done` with a
new `run_id` and the metrics, `_ckpt` is cleared and the duration recorded. `ctx.check_stop()` (called by
`ctx.progress`) raises `Interrupted` when `logs/STOP` exists. Any other exception marks the stage `failed`.

**Events** in `logs/events.jsonl`, one JSON object per line `{"t": epoch, "type": ...}` (`journal.py:1-11`):
`run_start {stages, variant}`, `run_end {status}`, `stage_start {stage, title}`, `stage_end {stage, status, duration}`,
`progress {stage, frac, msg, eta, elapsed}` (at most one per second, `stage.py:126`), `log {stage, level, msg}`,
`metric {stage, key, value}`, `preview {stage, name, path}`. `ctx.preview(name, rgb)` writes
`work/<stage>/previews/<name>.jpg`, 1600 px max (`stage.py:137`). The GUI tails the file every 500 ms with a byte offset
(`gui/runner.py:85`). One file per project, append-only, never rotated; only `run_start` names the variant, and no event
carries a run id.

**Parallelism.** The runner is strictly sequential: one stage at a time, one run per project (run.lock). Inside stages:
calibrate and register use `multiprocessing` spawn pools of `workers` processes (`calibrate.py:184`, `register.py:55`);
inventory uses `ThreadPoolExecutor(8)`; catalogs uses thread pools of 6 and 10 for network queries. One GPU per project,
`devices["main"]`, chosen in the GUI toolbar (`gui/app.py:251-258`); star_net can train on `devices["aux"]`
(`star_net.py:82` `choose_device`). Nothing runs on two GPUs at once.

**Variants and methods (the only form of several workflows today).** A variant is a named parameter patch over main
(`project.py:15-18`). `variant_stages(name)` (:182) = the changed stages plus everything downstream; only those get
folders in `variants/<slug>/`, the others are read from main's `work/`. Exactly **one** variant is active:
`Project.stage_dir` (:135) and `overrides` (:142) follow `config.active_variant`, which `set_active_variant` (:218) saves
to disk (`cli --variant` sets it in memory only). `create_variant` (:199) drops no-op entries; `promote_variant` (:236)
moves the folders into `work/`, merges the params and can keep the old main as a variant; `delete_variant` (:224).
`core/methods.py` `DECISIONS` (:38) has 8 questions (star_layer, glare, star_shape, faint_stars, star_flat, ha_reference,
star_guard, pattern); each option is a parameter patch, applied to the shown processing or made into a variant
(`gui/methods_tab.py:491` `_try_variant`, :516 `_apply_here`). A variant always shares main's upstream folders, even when
its settings equal another variant's. None of the 9 real projects has a variant.

## 4. The stages

Durations in minutes, from the manifests of Heart (721 lights, Sony A7S, 12 Mpx, drizzle ×2) and Iris (1268 lights); each
is the stage's last run, on a 2-GPU machine.

| Stage (file:class) | Deps | Main outputs in its stage folder | Main params (basic in bold) | Device | Heart / Iris |
|---|---|---|---|---|---|
| inventory (`inventory.py:22`) | none | frames.json; preview middle_frame | max_lights | CPU, 8 threads | 0.1 / 0.8 |
| calibrate (`calibrate.py:51`) | inventory | master_dark/flat.npy, badmask.npy, frame_stats.npy, cache.json → `scratch/calibrate/lights.f16` | hot_sigma, flat_defect, saturation, workers | CPU spawn pool | 5.2 / 13.3 |
| register (`register.py:27`) | calibrate | reg.npz (polynomial transforms, FWHM, transparency); preview frame_quality | detect_sigma, max_stars, workers | CPU spawn pool | 2.0 / 1.6 |
| stack (`stack.py:67`) | calibrate, register | drz_full / drz_A / drz_B (half-stacks) / drz_weight.npy, drz_meta.json, frames.json (weight, transparency, FWHM, excluded) | **scale, min_transparency**, pixfrac, kappa, max_fwhm_ratio, half_block, pattern_removal | GPU (Bayer drizzle, torch) | 34.4 / 12.5 |
| astrometry (`astrometry.py:44`) | inventory, stack | wcs.fits; metrics up_position_angle, parity | **target, ra, dec, focal_mm**, blind, sip_order | CPU + network | 2.1 / 2.5 |
| reference (`reference.py:17`) | stack | align_ref.json | **max_stars** | CPU, optional | 1.5 / 1.4 |
| catalogs (`catalogs.py:39`) | inventory, astrometry, stack | gaia_deep / gaia_faint / gaia_xp.npz | g_deep, g_faint, xp_* | network | 3.0 / 0.1 |
| surveys (`surveys.py:34`) | astrometry, stack | ref_ha.npy, survey.json | **survey** (auto, nsns, finkbeiner, none) | network | 0.7 / 0.6 |
| color (`color.py:56`) | catalogs, stack, register | M.npy (colour matrix), specresp.npz; preview spectral_response | **white_temp**, camera_profile | CPU | 0.0 / 0.0 |
| denoise (`denoise.py:34`) | stack | n2n_model.pt, n2n_full.npy, n2n_avg.npy; preview before_after | **iterations** (auto 6k-30k), patch, batch, seed | GPU training (UNet, bf16 autocast) | 31.1 / 33.5 |
| optics (`optics.py:38`) | denoise, stack, catalogs | psf_model.pt, lat_*.npy, lateral.json; preview psf | **lateral**, psf_iterations | GPU (PSF fit) | 3.0 / 1.6 |
| stars (`stars.py:56`) | optics, catalogs | stars_model{,_ext,_faint,_glow,_halo}.npy, star_fluxes.npz, faint_fluxes.npz, star_shape.npz | **faint_model, local_psf, angular_shape, far_halo**, halo_bound | GPU (photometry, model) | 32.2 / 3.3 |
| *star_shape* (`star_shape.py`) | not a stage | angular PSF (spikes, diamond), called from stars and render_prep | none | CPU numpy | |
| *far_halo* (`far_halo.py`) | not a stage | glare model of the optics, called from stars | none | CPU numpy | |
| star_net (`star_net.py:109`) | stars, optics, denoise, color | star_net_model.pt, star_layer.npy; preview star_removal | **enabled, steps** (11 000), device (auto, aux, main) | GPU training | 40.9 / off |
| starless (`starless.py:60`) | stars, optics, denoise, stack, color, catalogs | sl_final(_learned).npy, n2n_final(_learned).npy, n2n_sharp_final.npy, sl_mask.npy | **wiener_k, sharpen**, learned_stars, sharpen_steps | GPU (N2N apply, sharpen fine-tune; `:433` hardcodes cuda:0) | 10.4 / 8.7 |
| background (`background.py:38`) | starless, stars, optics, surveys, astrometry, stack, color, catalogs | astro.npy, sky.npy, flat.npy (star flat), coefs.json, ha_extraction.json | **sky_order, star_flat**, bin_px, object_clip | CPU | 0.7 / 0.3 |
| render_prep (`render_prep.py:77`) | background, starless, stars, color, optics, stack | prep.json, N / Eg0 / dY / R / chroma_fine / dS / Dglow / O.npy layers, Sreal / Sshape / Shalo(L) / Wsp / Ssp.npy, stars.npz, profile.npz, valid.npy | **star_guard**, 13 advanced | CPU, small torch | 1.9 / 1.6 |
| render_prep_learned (`render_prep.py:942`) | same as render_prep | the same layers from the learned starless (or none.json) | same as render_prep | CPU | 1.7 / 0.7 |

A full Heart run takes about 2.8 h; the test sets (30 to 60 frames) 0.3 to 1.3 h. Current versions: stars v15, starless
v24, render_prep v44, background v5. A bump makes the stage and everything below it stale.

## 5. Rendering

**`RenderSettings`** (`render/settings.py:15`, a `StageParams`): 73 fields, each with a `section`. Colour: white_temp,
tint, chroma, ha_boost, oiii_*, resid_boost, colour_detail, hue_*, snr_*. Sky background: sky_level, dark_level, toe,
diffuse_cut, sky_neutral, glow_*, speck_k. Nebula: white_mult, hdr_*, lc_fine / medium / large, neb_sharp, sharpen,
fine_*, faint_*. Highlights: hl_*. Stars: stars, real_stars, learned_stars, spike_*, g_lim, t_lim, d10, d7, star_glow,
star_sat, core_white, glare_nebula. `from_saved` (:17) drops unknown keys. `PRESETS` (:261): Natural, Faithful, Artistic,
H-alpha emphasised, Prominent stars, Dense Milky Way, Starless, Rich detail & colour, Emission nebula OIII.

**How an image is made.** `RenderEngine(prep_dir, device, calibrated_white)` (`engine.py:56`) loads the render_prep
layers into host memory. `render(s, scale)` (:251) runs `_render` (:256) inside `deterministic()` and `no_grad`:
1. `f = round(1/scale)`. Below full scale the layers are avg-pooled and every spatial parameter divided by f (`_layer`
   :125); the stars are always drawn at full resolution, then reduced.
2. `_nebula_linear` (:273) mixes N with the optional layers (fine detail, N2N, fine colour, OIII, glow, glare), then
   applies the colour matrix `M_eff` and the sky neutralisation.
3. `_nebula` (:341) fits the tone curve on percentiles (`fit_tone_curve` :563, `tone_curve` :551), then guided-filter HDR,
   local contrast and sharpening, the highlights shoulder (`_highlights` :407), OKLab chroma and gamut mapping.
4. `_stars` (:446) fits the star display curve (`fit_star_curve` :610) and draws the catalogue stars with the measured
   profile (`draw_stars` :665), adding Sreal (stars shown by their real light) and the photographed spikes.
5. `final = screen(neb, stars)`, invalid border set to zero; returns `Layers(final, starless, stars, scale, info)`, sRGB
   tensors in [0,1].

Partial caching: `_cached(name, keys, s, f)` (:241) is keyed by the settings in `NEB_LIN_KEYS`, `NEB_KEYS`, `STAR_KEYS`
(:32-44), so moving a star slider redraws only the stars. `StarRemovalEngines` (:510) holds one engine per preparation
(render_prep, or render_prep_learned when `learned_stars`), loaded lazily. The engine falls back to CPU without CUDA (:59).

**Interactive use.** `gui/render_tab.py`: `RenderWorker` (:36) owns the engine in a `QThread` (:308); requests
`(seq, values, scale)` are coalesced (only the latest is rendered); it emits uint8 arrays plus `info.render_ms` and
`gpu_mb`. `RenderTab` (:106) handles presets, undo/redo, snapshots, views (`render/views.py`), compare, clipping and
export. ComfyUI keeps up to 2 engines loaded (`bridge.engine_for` :198) and serves HTTP previews
(`comfyui_astro/server_routes.py`).

**Export.** `export(project, engine, s, ExportSettings, progress)` (`export.py:63`) renders at scale 1, orients the image
(`orientation_ops` :32: as captured, north up, match reference) and writes into `results/<name>/`: `<name>_full_resolution_16bit.tif`
(zlib, sRGB ICC), `<name>.png` (16-bit, sensor scale), `<name>.jpg` (quality 93), the same three for `_starless` and
`_stars`, `_reference_framing.png`, and `<name>_settings.json` = `{date, settings, export, orientation, qa, files, info,
variant, preparation}`. `quality_checks` (:136): clipping, burnt highlights, dark dips at stars. CLI:
`astro-gpu render <p> --preset Faithful|Artistic [--set k=v] [--use-saved] [--name] [--orientation] [--variant]`
(`cli.py:98`).

## 6. The existing GUI (`astro_gpu/gui`, about 3 060 lines, PySide6)

`gui/app.py` `MainWindow` (:191): a toolbar (New, Open, Recent, Run/resume F5, Stop, Advanced toggle, variant label,
Guide F1, GPU combo); a left dock listing the stages with their states; a bottom dock with the journal, the current
stage bar and an overall bar whose ETA comes from `timings.json` medians (`_update_overall` :604). Tabs (:288-301):
- **Processing**: stage previews (`work/<stage>/previews/*.jpg`) and a `StagePage` (:56) with state and reason, an
  auto-generated `ParamForm` (origin and auto reason per field), help box, metrics table, frames table for stack, and
  "Run up to here" / "Recompute" (:95-96, `start_run(until, force)`).
- **Methods** (`methods_tab.py:355`): one `DecisionCard` per decision (apply, try as variant, switch, promote, delete);
  `CompareDialog` renders the same zone of every processing at 100 % in a `QThread`.
- **Final adjustment** (`render_tab.py:106`, section 5). **Results** (`results_tab.py:31`): description, exported images.
- `NewProjectDialog` (`project_dialog.py:60`).

**How it calls the engine.** Processing runs in a **detached child process** (`python -m astro_gpu.cli run`), followed
only through `events.jsonl`; `RunController` (`gui/runner.py:18`) re-attaches after a restart and stops a run through
the STOP file. Status and params are computed in the GUI process (`Pipeline.status()` and `effective_params()`, which
also run the stages' `auto()`). Rendering runs in-process on a QThread, so the GUI and the worker may share one GPU.

**Reusable by a new client:** `RunController` + `read_events` (subprocess and tail model); `ParamForm`, `ImageView`
(zoom/pan), `HelpBox`, `Histogram`, `JournalView`, `Debouncer` (`widgets.py`); the coalescing `RenderWorker`;
`clipping.py`, `views.py`, `style.py`, `newproject.py`; the CLI as the process boundary. The mockup (`mockup/alt3`,
about 9 100 lines, simulated) reuses none of these: it has its own model, settings catalogue and post engine.

## 7. Tests and QC

**Tests:** 82 test functions in 17 files (86 collected with parametrize). Markers in `pyproject.toml`: `gpu` (declared,
never used), `network`, `slow`. Fast subset:
```
C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe -m pytest -m "not slow and not network" -q
```
Run on 2026-10-31: **84 passed, 2 deselected (test_blindsolve, slow test of test_far_halo), 5 warnings, 69 s**. The
warnings come from the non-deterministic CUDA median at `render/ops.py:99`; the tests use the GPU when present. Main
files: `test_core.py` (resolve, describe, up to date, stale downstream, resume, stop, journal, new param at default),
`test_variants.py` (6), `test_render_*`, `test_star_net.py` (12), `test_generalisation.py` (other cameras' readers),
`test_comfy_*`.

**tools/qc** (`tools/qc/README.md`), all run with the venv python:
- `nonreg.py SOURCE COPY [--name faithful --name artistic] [--set stage.k=v] [--skip-before]`: copies the project, renders
  the deliverables before, recomputes the stale stages, renders after, writes `nonreg.json` (mean, p99.9, % > 2 levels)
  and diff maps. Never writes to the source.
- `blind.py PROJECT OUT SHORT label=render.png ... [--zones x:y]`: blind sheets under letters (key saved apart), in the
  reference framing, bright and mid stars at 100 %, zones. Judges get `qc_prompt.md`.
- `judge.py PROJECT OUT_PREFIX label=png ...`: overview and zoom sheets in the reference framing (used by blind).
- `variants_sheet.py`: stars and zones side by side. `changed_zones.py BEFORE.tif AFTER.tif [N]`: the most changed zones.
- `jeux.py build|run|list [NAMES] [--until --force --label]`: frozen test sets of other cameras on `F:/astro_essais`, own
  ASTRO_GPU_HOME, `TEST_PARAMS` (star_net off, denoise 6000), rendered with Natural.
- `bulletin.py PROJECT... --render NAME [--sheet] [--out]`: report card (stage metrics, export QA, sky flatness and cast,
  star size and elongation measured on the render).
- `render_saved.py PROJECT NAME... [--suffix]`: render again with the saved settings.
- `run_stage.py PROJECT [--param stage.k=v] STAGE...`: run stages without the dependency check, params in memory only.

Protocol: nonreg on copies of Heart, Iris, Rosette and Horsehead, then blind sheets, then independent judges. A change is
kept only if it is not worse anywhere.

## 8. The user's real projects

Read only. In all 9, every computed stage is `done` and there is no variant. Registry `~/.astro_gpu/projects.json`:
FlamingStar, Wizard, M31, Rosette, Horsehead, Iris, Heart (ASTRO_GPU_HOME not set). Per PLAN 11.16 the stars and
render_prep versions were bumped after some projects ran, so the GUI may show them stale from stars onward.

| Project folder | Name | Raws (inputs) | State |
|---|---|---|---|
| `F:/photos/heart_nebula/astro_gpu` | HeartSoul | `F:/photos/heart_nebula/raw/{signal,dark,flat}`, present | All 17 stages (~2.8 h); artistic, faithful. Scratch `C:/Users/Nicolas/astro_gpu_scratch/HeartSoul`; devices main=cuda:1. Reference project. |
| `F:/photos/NGC7822/astro_gpu` | NGC7822 | `F:/photos/NGC7822/{signal,dark,flat}`, present | 15 stages (older chain, no star_net), several old renders. Not in the registry. |
| `E:/astro_gpu/projects/FlamingStar` | Flaming Star | lights + darks `F:/backup_pc/photos/28.12.24/...` **missing**; copies in `I:/backup_pc_LENOVO/photos/28.12.24/` and `G:/Astro/2024-12-28_divers_astro/`; flats `D:/photos/2.3.25_Nebuleuse_iris/raw/flat` | 17 stages; artistic, faithful. Recomputes from stars onward without the raws. |
| `E:/astro_gpu/projects/Horsehead` | Horsehead | as Flaming Star (missing, same copies) | 17 stages; artistic, faithful. Reference project. |
| `E:/astro_gpu/projects/Rosette` | Rosette | as Flaming Star (missing, same copies) | 17 stages; artistic, faithful, render, render2. Reference project. |
| `E:/astro_gpu/projects/Wizard` | NGC 7380 Wizard | `E:/Astro-photo/Sorciere/raw/{signal,dark,flat}`, present | 16 stages, nothing exported. |
| `E:/astro_gpu/projects/M31` | M31 (2023) | `F:/backup_pc/photos/24.08.23/{signal,flat}` **missing**; copies in `I:/backup_pc_LENOVO/photos/24.08.23/` and `G:/Astro/2023-08-24_M31_top/` | 16 stages, nothing exported. |
| `C:/Users/Nicolas/astro_gpu_scratch/projects/Iris` | Iris | `D:/photos/2.3.25_Nebuleuse_iris/raw/{signal,dark,flat}`, present | 17 stages; artistic, faithful, compare_reference. Reference project. |
| `C:/Users/Nicolas/astro_gpu_scratch/projects/M33` | M33 | `D:/astro_photo/22_Aout_25_Triangle/{Raw2,Flat}`, no darks | 15 stages (older chain). Not in the registry. |

**Test base `F:/astro_essais`** (donnees, projets, scratch, etat = its ASTRO_GPU_HOME, journaux, runs.json), 9 sets.
Complete (17 stages, Natural renders): canon40d_rosette, canon6d_m51, heart_a7s, iris_a7s, seestar_m31,
stellina_ic1318, stellina_ngc891. Partial: asi2600mc_m8 (catalogs left "running"), canonra_northamerica (astrometry
interrupted). `D:/astro_donnees_test`: 15 downloaded raw sets, no projects. Bench copies, not real projects:
`D:/astro_gpu_diag`, `C:/Users/Nicolas/astro_gpu_scratch/nonreg_2026-10-28` (~115 GB). ComfyUI block cache:
`D:/astro_gpu/blocks`.

## 9. Gaps against the mockup (`mockup/alt3`, `docs/SPEC_IHM.md`)

1. **Several workflows per project, settings per workflow, caching keyed by settings.**
   - The mockup (`alt3/app/project.py:429` `stage_key`): a workflow is a full set of settings and methods; two workflows
     share a block when `(stage, options, settings of the upstream chain, project-level choices such as excluded
     frames)` are equal. It also expects optional blocks and per-block methods (deconv, fond_simple, denoise_ft /
     denoise_ai / denoise_cl, `DN_STAGE` :49), `throws_away` and `add_variant`.
   - The engine: main plus named variants, one active; folders keyed by variant slug, not by settings; a random `run_id`
     that prevents content addressing; `run.lock`, `STOP` and `events.jsonl` per project (one run at a time); a fixed
     list of 17 stages with no optional or alternative stage; no manual per-frame exclusion.
   - Precedent: `comfyui_astro/blocks.py:56-120` already runs unchanged stage code in folders
     `work/<stage>/<sha1(stage, version, upstream keys, settings)>` through a `_BlockProject` shim, i.e. content-addressed
     caching.
2. **Run up to a block.** Exists per stage (`Pipeline.plan/run(until)`, `--until`, GUI "Run up to here"). Missing: the
   mapping of a block to its last stage (e.g. Stars = optics + stars + star_net), plans per workflow, and a run recap
   across workflows that counts shared stages once.
3. **Progress events a GUI can follow.** Exist: events.jsonl, `read_events` offsets, progress each second with ETA,
   stage start and end. Missing: a workflow or run id on each event; a plan event with estimated durations (`cost_hint`
   unused; `timings.json` keeps global medians unrelated to image size); rotation of the file; an in-process callback
   API (`bridge.HookJournal` shows how).
4. **Per-stage previews and measurements.** Exist: `previews/*.jpg` (1600 px max), `preview` events, manifest `metrics`,
   `stack/frames.json`, `color/specresp.npz`. Missing: previews for astrometry, catalogs and reference; labels and units
   on the flat metrics; structured data for the mockup's charts (frames chart with thresholds, before/after, sensor
   response); previews at reduced scale or on a region.
5. **CPU fallback.** Exists: `Context.device` returns CPU without CUDA (`stage.py:161-174`), `RenderEngine` falls back
   (`engine.py:59`), the GUI offers "CPU (no CUDA GPU found)". Missing: nothing was ever measured or tested on CPU;
   denoise (6k-30k UNet steps), star_net (11k steps) and the sharpen fine-tune would take hours; no CPU method
   (NL-Bayes, pretrained network); `starless.py:433` and `comfyui_astro/blocks.py:108` hardcode `cuda:0`;
   `ops.box_filter` uses float64 (blocks MPS); the mockup's `CPU_FACTOR` (`alt3/app/project.py:89`) is a guess.
