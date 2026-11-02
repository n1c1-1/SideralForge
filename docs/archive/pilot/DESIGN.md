# Design: the client on the real engine

Written 2026-10-31 by the architect agent, from PILOT.md, ARCHITECTURE.md, ENGINE_KNOWLEDGE.md, spec/*, mockup/alt3 and
the engine code. Corrected the same day after the review (DESIGN_REVIEW.md, "Orchestrator's decision"). Paths are
relative to the repository root. Python: `C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe`.

## Changes after review
Point numbers are those of DESIGN_REVIEW.md.
1. **Key rule**: values are validated through Params before hashing; a field with an auto rule is never stripped; a
   value is dropped only when it equals a known suggestion; paths are normalised. §1.2, §1.3, §2.2, P1, P2, P5.
2. **Paths and state**: paths made absolute; the worker runs in the project root; files keyed by name, size and mtime;
   pilot ASTRO_GPU_HOME. §1.3, §1.4, §3, P7, P8.
3. **Re-running a done key**: jobs write into `<key>.tmp`, renamed on success; calibrate rebuilds its cache only; the
   client never memory-maps. §1.1, §3, §5.3, P3.
4. **Import check**: names, order, count, EXIF time and exposure; temporary folder for missing stages; write guard.
   §1.6, P6.
5. **No link mode**: adoption copies only what later stages read, after a free-space check. §0, §1.1, §1.3, §1.6, §4,
   P2, P3, P6.
6. **Earlier star layers** are tied to the key of their stack. §1.5, §1.6, P6.
7. **Exclusions** are keyed by (session, name). §1.2, §1.3, P1.
8. **Cheap safety items**: run_id passed in, O_EXCL lock, atomic timings, rotation and STOP removal at launch, full
   definition in the plan. The CUDA random state is left as is. §3, §7, P3.
9. **Runs**: skip downstream on failure, CUDA_VISIBLE_DEVICES, CREATE_BREAKAWAY_FROM_JOB, missing reference dropped,
   cheap workflows first. §1.6, §3, P3, P8.
10. **Estimates** are accepted on Iris within ±50 %. P3.
11. **QC**: pairs by preparation, each render remapped with its own project, no `--sheet` on old projects, a QC tools
    port. P14, new P17.
12. **Raw copies**: sessions keep file lists; FlamingStar's folders are merged back into one. §1.2, P8.
13. **v1/v2 checks**. P7, P12.
14. **Clutter**: test projects under F:/astro_essais; `list_projects` skips `_*`; walkthroughs never open P8's
    projects; export/ holds only the QC winners; `exclude` hidden in the v1 form. §4, P3, P5, P8, P13, P14.
15. **Agents never commit**; package tests during a wave; tests on the CPU during P8. §6.
16. **Ownership**: P9b delivers the whole model surface; per-package locale files; `stages/v2.py` owned by P4. §2.2,
    §6.
17. **Engine freeze** during P8, tag `p8-engine`. §6, P8.
18. **Missing work** spread over the packages; new P16 and P17; durations re-estimated. §6.
19. **Engine texts**: reasons as a template plus arguments, a metric catalogue, French. §5.4, new P16.
20. **Post**: budgets, real region rendering, LRU engines, a free GPU or the CPU with a fallback. §2.3, §4, §5.3, P4,
    P12.
21. **No disk scans in the GUI thread.** §1.3, §5.3.
22. **Spec fidelity**: raw-stack channel normalisation, distinct internal polynomial names, the ◇ list, the
    flats-merge check. §1.2, §2.1, §2.3, §5.5.
23. **Risks**: heavy P8 jobs on cuda:0; walkthroughs render on the CPU during P8; the Classic rings on M31 and M33 are
    stated in the report; P13 gets more time. §7, P8, P13, P14.

## 0. The decisions in brief
1. **v2 lives beside v1**: a folder with `project.astroproj`, `state.json`, `store/`. v1 (`project.json`, the
   Pipeline, `gui/`, `tools/qc`) keeps working; nothing writes into a v1 folder.
2. **Results are addressed by a key** (stage, version, validated stored parameters, upstream keys, project inputs).
   Auto values are not in it: they follow from the upstream results (stages are deterministic since 2026-10-16).
3. **Stage code runs unchanged** through a shim (`StoreProject`) that maps every ancestor to its key folder, as
   `comfyui_astro/blocks.py` does; jobs are executed by v1's `Pipeline._run_stage`.
4. **Two new stages only**, `background_poly` and `render_simple`, make the four templates real. Deconvolution, the
   new denoisers and the reorder are deferred, behind methods.
5. **Import adopts results** by copying them under their v2 keys (never linking): 7 of the 9 real projects keep
   inventory → optics (≈ 1–1.5 h of GPU each).
6. **The client is the mockup** (alt3 + alt2's viewer and widgets) in `astro_gpu/client/`, over a `ProjectModel`
   adapter on `astro_gpu/api.py`; runs in a detached process, renders in a worker thread.

## 1. Project format v2
### 1.1 Files
```
<workspace>/<project>/                      workspace default F:/astro_projets
  project.astroproj        the definition (JSON, atomic writes, written only through api)
  state.json               {"last_keys": {"<wf id>": {"<stage>": "<key>"}}}: the previous result of an Outdated block
  store/<stage>/<key>/     one done result: outputs, manifest.json, previews/
  store/<stage>/<key>.tmp/ a job in progress (with _ckpt/); renamed <key> on success
  store/_trash/            done folders replaced by a re-run; emptied by "Clear unused results"
  logs/events.jsonl  logs/run.lock  logs/STOP  logs/runs/<run_id>.plan.json  logs/runs/<run_id>.log
  cache/thumbs/  cache/thumb.jpg (Home card)   inputs/ (reference image, earlier star layers + .json)   export/<name>/
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
 "excluded": [[1, "DSC01234.ARW"]], "stack_mode": "together",
 "workflows": [{"id": 1, "name": "Full", "options": {"denoise": "n2n", "stars": "psf", "gradient": "halpha",
                "deconvolution": null}, "params": {"stack": {"scale": 1.5}}, "end": null, "variant_of": null}],
 "current": 1, "render_bases": {"faithful": {"sky_level": 24.0}}, "post": {"layers": []}}
```
- **Paths and ids.** Paths may be relative to the project in the file; the engine always gets them absolute. Workflow
  ids are integers, never reused; the name is display only, so renaming is free.
- **`params`** keeps the values the user set.
  - Saving never drops a value, even one equal to the static default: under default < auto < user, typing the default
    is how a user overrides an auto value (stack.scale 1.5 against auto 2.0).
  - Only the API drops a value, and only when it equals a **known** suggestion (1.3). NEW (◇) settings are never
    stored in v1.
- **Sessions** keep lists of folders **or files**, as v1 `Inputs` does (the 100 listed darks of the 28.12.24 nights).
  The engine gets the union of each kind: one master dark and one master flat, as today. A health check warns when
  flats of several sessions merge into one master. Flats per session are ◇.
- **`excluded`** holds (session id, file name) pairs, since camera counters restart between cards and nights. The API
  passes them as the new parameter `stack.exclude` (absolute frame paths) to every workflow. Only Stacking and later
  stages change (this avoids the mockup's open point 3).
- **`post.layers`**: the POST.md dicts. A workflow layer has `base` (a `render_bases` name, or a preset) and `sliders`.

### 1.3 Key and store
```
key(wf, s) = sha1(canonical_json({"s": s, "v": STAGES[s].version, "p": key_params(wf, s),
                                  "u": {d: key(wf, d) for d in deps_in_chain(wf, s)},
                                  "x": project_part(s)}))[:16]          # canonical: sort_keys, separators (",", ":")

key_params(wf, s):
    v = stored(wf, s) | implied(wf, s) | injected(wf, s)                   # later wins
    v = {k: x for k, x in v.items() if k not in NON_KEY[s]}
    m = STAGES[s].Params(**v).model_dump(mode="json")                      # validated: 2 and 2.0 hash the same
    p = {k: m[k] for k in v}
    p = {k: file_id(x) if k in PATH_FIELDS[s] else x for k, x in p.items()}     # (name, size, mtime)
    p = {k: x for k, x in p.items() if k in AUTO_FIELDS[s] or x != default(s, k)}
    if s == "stack" and "exclude" in p: p["exclude"] = sorted((session id, name) pairs)
    return p
```
- **The rule**:
  - A field with an auto rule (`AUTO_FIELDS[s]`) is never stripped: a stored value equal to the static default still
    differs from "not stored".
  - A field without an auto rule is stripped at its default: its effective value is the same.
  - An invalid value is a ValueError naming the setting.
- **When the API drops a stored value** (`set_param` with the suggested value, or ↺): only when the suggestion is
  known. That means `auto()` ran without error on done upstream results, or the field has no auto rule (the suggestion
  is the static default). If upstream is not done or `auto()` fails, the value stays stored.
- **`AUTO_FIELDS`** (fields an `auto()` can set) and **`PATH_FIELDS`** (`starless.learned_stars`,
  `denoise.reuse_model`) are declared per stage in `stages/v2.py`.
- **Implied** parameters: `star_net.enabled=True` with psf_neural; `surveys.survey="none"` with polynomial + Stars.
  **Injected**: `stack.exclude`. **NON_KEY** (no effect on results): `calibrate.workers`, `register.workers`,
  `star_net.device`.
- **project_part**:
  - inventory: the ordered (kind, file name, size) list, as `list_frames` returns it over the union of sessions. It is
    content-based, so copied raws keep the key; it is ordered, because the half-stack blocks follow the order;
  - astrometry: the target;
  - reference: `file_id` of the image;
  - others: null.

  It is computed in the client's ApiWorker, cached per session, never in the GUI thread.
- **Left out on purpose**: auto values, the device, which catalogue service answered. Camera profiles in
  ASTRO_GPU_HOME are also left out; P7 and P8 run with a pilot ASTRO_GPU_HOME holding a copy of the profiles, so
  color does not depend on other runs. `describe()` compares a done key's manifest params with a fresh resolve and
  reports any drift ("suggested values changed since this result"); nothing re-runs automatically.
- **Done**: `store/<s>/<key>/manifest.json` has status done. The manifest is v1's plus `key`, `key_parts`,
  `adopted_from`, with `run_id = key`, so starless' auto `learned_stars` stays stable. A `.tmp` folder is never done.
- **usable_as_input(s, dir)** is false when a done result lost external data (inventory: a frames.json path;
  calibrate: the cache.json path). A plan that needs it as an input re-runs it (3). It is checked in the ApiWorker and
  cached per key.
- **Old keys are kept.** `cache_usage()` lists the keys no workflow references (current or last), `_trash`, the
  `.tmp` folders of interrupted jobs and the scratch caches. "Clear unused results" deletes them after a confirmation,
  and refuses during a run.

### 1.4 StoreProject, the shim (core/store.py)
- **Attributes**: `root`; `logs` (the project's, so STOP is shared); `results` = export/; `scratch` =
  `<scratch>/<key>`; `config` (name, `inputs` = union of sessions, target, devices, reference_image, empty
  params/variants/active_variant); `overrides(s)` = the validated stored + implied + injected values.
- **Paths**: every path it hands out (inputs, reference image, path fields) is absolute and normalised. The worker's
  working directory is the project root.
- **`stage_dir(name)`** returns:
  - for the job's own stage, `<key>.tmp`;
  - for every ancestor (transitive `deps_in_chain`), its done key folder;
  - for any other stage, `store/_absent/<name>`, never created (`ctx.input` raises "missing input",
    `star_net.stage_layer` returns None, export's reference test is false).
- **New helper** `Context.has(stage)`. A grep shows that stages reach outside `stage_dir` only through config.target,
  inputs, name, reference_image, devices, and STATE.

### 1.5 Backward compatibility
- **Loading.** `project2.load(root)` reads `project.astroproj`. A folder with only `project.json` raises
  `LegacyProject`; its `from_v1(root)` builds the v2 definition in memory (one session from `inputs`, one workflow
  from main, each variant a workflow with merged params). The client offers Import (1.6).
- **The workflow from main** has denoise n2n, and gradient polynomial if `surveys.survey == "none"`, else halpha. Its
  stars and name:
  - psf_neural, "Full + neural stars", when `star_net.enabled` (default true);
  - psf_neural without star_net, "Full + earlier neural layer", when `starless.learned_stars` names a file;
  - else psf, "Full".
- **An earlier star layer belongs to one stack** (it is in that stack's grid). The import writes
  `inputs/<layer>.json` with the key of the stack it came from. "Full + earlier neural layer" runs only while its
  stack key equals it; otherwise its Stars block says "the earlier layer belongs to another stack" and the plan skips
  that workflow.

### 1.6 Importing a v1 project with its results: possible
`core/import_v1.py`. Nothing is ever written in the old folder.
1. **Write the v2 definition** with the new raw locations. The old frames.json has no file size, so `path_map` (old
   name → new path) is checked by name, order and count. A reference image that no longer exists (M31) is dropped,
   with a note.
2. **Walk each chain in order.** Adopt a stage only if:
   - its old manifest, read first, is done with the current version;
   - all its `deps_in_chain` are adopted, and `manifest.inputs[d]` equals the old `d` run_id;
   - its old params equal a fresh resolve. The resolve goes through a read-only shim over the old `work/`. Stages
     absent from the old `work/` map to a temporary folder in the new project, because `Context` creates its folder.
     NON_KEY is ignored and `_same_params` is used;
   - for inventory: the old frames.json and the new files agree on name, order, count, and per-frame EXIF time and
     exposure.
3. **Copy** each adopted stage under its new key. There is no link mode.
   - Copy only what later stages read: outputs, previews, manifest. Never copy `_ckpt` or a rebuildable cache (the
     calibrated frames).
   - Before copying, check free space on F: (the bytes to copy + 10 %); refuse with a message otherwise.
   - Rewrite the manifest and relocate the frames.json paths. Calibrate's cache.json points at the new scratch, so
     calibrate rebuilds its cache if register or stack must re-run.
4. **Report** per stage: adopted (key) or recomputed (reason).

**The write guard**: dry-run and import run under an audit hook (`sys.addaudithook`). It raises on any open-for-write,
mkdir, rename or remove under the old root. It runs on copies first (heart_a7s), then on the real projects.

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
- **Rules** (`api.set_option` returns toast notes, as the mockup's `with_tool`/`without_tool`):
  - Stars needs Denoise. Removing Stars removes neural and turns H-alpha into polynomial.
  - In v1, H-alpha needs Stars: the existing background works on the starless image.
  - psf_neural with a stored `starless.learned_stars` file omits star_net.
- **The two polynomials** are distinct methods internally:
  - `poly_simple` = `background_poly`, without Stars;
  - `poly_physical` = `background` with `surveys.survey="none"`, with Stars (the existing physical background,
    without the survey).

  The definition keeps gradient "polynomial", and the GUI shows "Polynomial" for both.
- **BLOCKS**:
  - Calibration = inventory, calibrate; Registration = register; Selection follows register (Run to here runs
    register); Stacking = stack;
  - Astrometry = astrometry, reference, catalogs, surveys; Color = color; Denoise = denoise;
  - Stars = optics, stars, star_net, starless; Gradient = background | background_poly; Output = the output stage.

  A block's key is the tuple of its stage keys. Astrometry's leaves out surveys (it follows the Gradient method), and
  Output's includes the workflow id (never shared), as in the mockup's `blocks.block_key`.

### 2.2 How a stage finds its inputs
- **Dependencies.**
  - `DEPS2[s]` = declared deps ∪ `EXTRA[s]`, with `EXTRA = {starless: (star_net,)}` (today an implicit read through
    starless' auto value).
  - `OPTIONAL[s]` may be absent: starless: star_net; background_poly: denoise; render_simple: color, denoise,
    background_poly.
  - In a chain, `deps_in_chain = DEPS2 ∩ chain`. A missing required dependency is a ValueError naming the block.
- **At run time** the shim maps every ancestor; an absent stage resolves to a non-existent folder (1.4). Only the new
  stages test `ctx.has()`; existing stages never sit in a chain that lacks one of their required dependencies.
- **Stage lists.** `stages.all_stages()` is unchanged.
  - `stages/v2.py` (owned by P4) holds `V2_STAGES` (all_stages + the new stages), `AUTO_FIELDS` and `PATH_FIELDS`, so
    v1's Pipeline never runs the new stages.
  - `workflow.Chains` (stages, order, DEPS2, OPTIONAL, EXTRA, BLOCKS, `chain(options, end)`) imports it lazily and
    defaults to V2; tests build toy Chains.

### 2.3 New stages for v1 (P4)
- **background_poly** (CPU, ≈ 1 min; deps stack, color; optional denoise):
  - per channel, in display RGB (M applied), a polynomial on the darkest blocks of a grid with 2.5 σ iterations
    (`simple_sky` in `mockup/alt2/build/prepare.py`);
  - params degree 2 (1–5), samples 16 per side (4–64), object_clip 2.5;
  - writes coefs.json, samples.npz, preview "background"; metrics sky_fit_blocks_pct, gradient_pct.
- **render_simple** (CPU, ≈ 30 s; deps stack; optional color, denoise, background_poly):
  - lin.npy = M·src − sky; src = n2n_full when denoised, else drz_full;
  - without color (the Raw stack exit), identity M, a per-channel median sky, and channels normalised to the same sky
    spread, so the preview has no colour cast;
  - also writes lin_raw.npy (with denoise), valid.npy, prep_simple.json {M, crop, luminance median, MAD, p99.995};
  - params `stretch` "auto" (others ◇), `linked`.
- **render/simple.py**: `SimpleRenderEngine(dir).render(SimpleSettings, scale, region=None)` returns the `Layers` type
  (stars None): linked MTF stretch, S-curve contrast, saturation, noise mix (`stretch()` in prepare.py).
- **render/engine.py** (region only): `render(s, scale, region=None, fit=None)`. A full-resolution region uses the
  tone fit of the 1/4 render, so 100 % zoom never renders the whole grid.
- **render/sliders.py**: the 9 sliders of POST.md 2.1.
  - Engine layers: position 2 is the layer's base RenderSettings. Other positions add the table's step (sky_level,
    faint_smooth, g_lim, d10, d7) or multiply by its ratio (lc_*, chroma, neb_sharp, fine_k, star_sat, star_glow),
    linearly between integers.
  - Simple layers use prepare.py's classic tables.
  - `availability(kind, has_denoise)` gives the reasons for greyed sliders.

### 2.4 Deferred, and how v1 shows the real order
- **Deferred**: Deconvolution; fine-tuned / pretrained / NL-Bayes denoise; Gradient before Color and Stars; a PSF
  before Denoise; the stretch choice; flats per session; the master library.
- **Real order in the graph**: Frames, Calibration, Registration, Selection, Stacking, Astrometry, Color, Denoise,
  Stars, Gradient, Output, Post.
  - "Linear processing" runs Astrometry → Gradient. With Stars, Gradient's wire comes from Stars ("starless image").
  - Spec mode notes "◇ before Color in a later version" and shows the Deconvolution card, disabled.
- **The reorder comes later as new methods, never as a switch on old stages.**
  - Example: Gradient "H-alpha survey, on the stack" = new stage `background_early` (deps stack, astrometry, catalogs,
    surveys), with color and stars depending on it through EXTRA/OPTIONAL.
  - Its block sits in a column "Gradient (early)" before Color, shown only when used; old keys stay valid.
  - It becomes the default only after non-regression and blind QC. Deconvolution comes the same way (a column after
    Color, its own PSF stage).

## 3. Runner v2 (core/runner2.py, core/estimate.py)
- **Plan:**
  ```
  need(wf, s, as_input=False):
      k = key(wf, s)
      if done(k) and (not as_input or usable_as_input(s, k)) or k in jobs: return
      for d in deps_in_chain(wf, s): need(wf, d, as_input=True)
      jobs[k] = Job(k, s, block, wf=sharing(k) over all workflows, deps={d: key(wf, d)}, est_s, disk_gb)
  ```
  - Run all = `need` on every chain stage (reference included). Run to block = `BLOCKS[block] ∩ chain(wf)`.
  - Jobs are in dependency order. Ties go to the workflow with the least remaining time, so cheap workflows finish
    first. Each key appears once.
- **Launch**: `runner2.launch`, called by `api.start_run`, without Qt.
  1. Remove STOP. Rotate `events.jsonl` (over 64 MB, renamed `events_<date>.jsonl`, never deleted) before the
     follower takes its offset.
  2. Write `logs/runs/<run_id>.plan.json`: {run_id, device, targets, a snapshot of the whole definition, jobs}.
  3. Start `python -m astro_gpu.core.runner2 <project> <plan>`:
     - detached, with `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB`, so it survives an
       agent session;
     - working directory = the project root; output to `logs/runs/<run_id>.log`;
     - `CUDA_VISIBLE_DEVICES=<n>` for the plan's GPU, with device cuda:0 inside. `starless.py`'s hard-coded cuda:0
       and star_net's "aux" then stay on that GPU.

  The plan is frozen: edits during a run do not change it.
- **Worker**:
  1. Create run.lock with O_EXCL {pid, run_id}. A lock whose pid is dead is replaced; a live one makes the worker
     refuse.
  2. For each job:
     - skip it if it is done and usable, or if an upstream job failed in this run (recorded as "skipped: upstream
       failed");
     - build the StoreProject (own stage → `<key>.tmp`) and assert the key against the snapshot;
     - run `Pipeline(shim, V2 stages of the chain, RunJournal)._run_stage(stage, run_id=key)`. This is v1's code:
       checkpoints, `deterministic()`, the manifest, Interrupted / failed. The only v1 change is the optional
       `run_id` argument;
     - on success, rename `<key>.tmp` → `<key>`. A done folder being replaced (an unusable calibrate) moves to
       `store/_trash/` only at that moment, so a failed re-run never destroys the old result;
     - write the manifest's key fields and call `estimate.record(…)`; timings files are written atomically.
  3. A failed job stops only the jobs downstream of it; the others go on. STOP ends the run, and the `.tmp` folder
     keeps its `_ckpt` for the resume.
  4. `run_end` lists the jobs done, failed, skipped and interrupted.
- **Calibrate with a lost cache** rebuilds `lights.f16` only. Its masters are copied into the `.tmp` folder, and
  calibrate takes a cache-only path when they are present for the same params (P3, no version bump).
- **Events** go to `logs/events.jsonl` (append-only, byte offsets). Every event carries `run`; stage events add `key`,
  `wf` (ids) and `block`; a new `plan` event lists the jobs.
- **Cancel** is the STOP file, as today. **One runner per project.** Two projects may run at once on the two GPUs.
- **The client follows** with `RunFollower`:
  - it tails `events.jsonl` every 500 ms from the launch offset, keeping that run only;
  - it updates the run rows (by key), the block stripes and the Log console;
  - at `stage_end done` it refreshes the states: `api.states()` writes last_keys, and a done output stage makes its
    Post layer available;
  - on project open with a live run.lock, it reattaches through the plan file.
- **Estimates**:
  - `units(s)`: inventory frames; calibrate / register lights × sensor Mpx; stack lights × Mpx × scale²; denoise
    iterations; star_net steps; other stages stack grid Mpx.
  - With history in `STATE/timings2.json` (last 20 [s, units, gpu|cpu] per stage): median(s/units) × units,
    "measured".
  - Otherwise "≈": the Heart reference (`mockup/alt2/data/data.json` stage seconds and gb, Heart units) × the units
    ratio, × the mockup's `CPU_FACTOR` without a GPU. New stages: 60 s and 30 s at Heart size. Disk is estimated the
    same way.
- **Not fixed (accepted):** star_net saves only the CPU random state, so a resumed training differs slightly from an
  uninterrupted one.

## 4. The engine API (astro_gpu/api.py; rendering in render/layers.py)
| Function | Contract § | Built on |
|---|---|---|
| `list_projects(ws)` (skips `_*` folders and folders without `project.astroproj`), `create_project(root, name, target=None)`, `open_project(root)` (LegacyProject), `save(p)`, `save_as(p, root)` (definition only) | 1 | project2 |
| `import_v1(old_root, new_root, path_map, dry_run=False) -> ImportReport` (copy only) | 1 | import_v1 |
| `scan_folder(folder, progress)`, `add_frames(p, scan, session=None)`, `remove_session`, `group_sessions(infos, criteria)` (night noon-noon, exposure, ISO) | 2 | list_frames, read_info, newproject |
| `set_excluded(p, [(session, name)], excluded)`; `health_checks(p)` (no bias, FWHM per session, flats of several sessions merged, disk; the rest ◇) | 2 | stack.exclude, space_text |
| `frame_table(p, wf)`, `selection_preview(p, wf, min_transp, max_fwhm_ratio)`, `frame_thumb(p, path, size)` (one stretch per session) | 2 | reg.npz, frames.json, `select_frames`, read_cfa |
| `TEMPLATES`, `workflows(p)`, `add_workflow(p, name, options, params=None, copy_of=None)`, `rename_workflow`, `duplicate_workflow`, `delete_workflow` | 3 | project2, workflow |
| `set_option(p, wf_ids, key, value) -> notes`, `set_param(p, wf_ids, stage, name, value or None)` (drops a value only at a known suggestion, 1.3); scope = list of ids ("sharing this block" = `sharing()`) | 3 | workflow rules |
| `make_variant(p, wf, change, name)`, `keep_variant`, `discard_variant`, `add_exit(p, wf, "stacking", name)`, `set_workflows(p, list)` (undo) | 3 | project2 |
| `methods(block) -> [MethodInfo(key, name, short, runs_on, stage, new)]`, `chain(p, wf)`, `deps(p, wf, s)`, `blocks(p, wf) -> [BlockRef(block, stages, key)]` | 3–4 | workflow, store |
| `stage_key(p, wf, s)`, `sharing(p, wf, s)`, `cache_usage(p)`, `clear_unused(p)` | 4 | store |
| `states(p) -> {(wf, s): StageState(state, reason, key, previous_key)}`: done, pending, stale, running, failed, interrupted, skipped | 5 | store, state.json |
| `plan_all(p)`, `plan_to_block(p, wf, block)`, `cost(p, workflows) -> Cost` (what-if: Replace, variant, toolbar, New workflow) | 5 | runner2 |
| `estimate(p, wf, s) -> Estimate(seconds, measured)`, `machine()` (GPUs, VRAM, cores, RAM) | 5 | estimate, torch, psutil |
| `start_run(p, plan, device=None) -> RunInfo`, `running(p)`, `cancel(p)`, `read_events(p, offset)` | 6 | runner2, journal |
| `stage_outputs(p, wf, s, which="current"/"previous")`, `stage_array(p, wf, s, name, scale)`, `pixel_to_sky`, `spectral_response` | 7 | manifests, npy, wcs.fits, specresp.npz |
| `describe(p, wf, s) -> [ParamInfo(…, value, suggested, suggestion_known, origin, reason, reason_args, drift)]`; `metric_info(s, key)` | 8 | `Pipeline(shim).effective_params`, texts.py |
| `sliders_of(p, wf)`, `render_layer(p, wf, base, sliders, scale, region=None, part=None) -> LayerImage(rgb, crop, info)`, `post_masks(p, wf, scale)` | 9 | layers.py, RenderEngine, SimpleRenderEngine |
| `export_image(p, rgb, meta, options, progress)`, `export_workflow(p, wf, base, sliders, options)` (headless) | 11 | layers.py, render/export |

- **Layers**:
  - `render_layer` returns the full stack grid (or the asked region), uncropped, plus the crop.
  - Post crops to the intersection of the visible layers; another drizzle scale is resampled ("resampled").
  - `RenderEngine` is built on the key folder (`StarRemovalEngines` assumes sibling folders, so it is not used).
  - `LayerService` evicts the least-recently-used engine when host memory runs short (Heart's render_prep is 1.4 GB).
- **Export in v1**: TIFF 16 / 32f, PNG 16, JPEG, FITS 32f + WCS; sRGB; full, 2048 or 4096 px; v1 orientation; layers
  and masks as files; a history sidecar.
- **◇ in the API**: master library, match_calibration, benchmark, preview_region, identified objects, script export,
  XISF, other profiles, history in the header, linear outputs of Stars workflows.
- **In the client**: the composition and the 13 filters (`postengine`). **cli2** (`python -m astro_gpu.cli2`): new,
  import, add-workflow, set, status, plan, run [--detach], stop, export.

## 5. The client (astro_gpu/client/)
### 5.1 What is copied
`python -m astro_gpu.client`. Copied from mockup/alt3/app and alt2/app (viewer.py, widgets.py) with relative imports;
the mockup's `alt3/app/locale/fr/*.json` go to `client/locale/fr/`. Removed: the Demo menu, `datatext.py`, every
alt2/data reference; "About this mockup" becomes About.

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
- **`ProjectModel(QObject)`** keeps the whole surface the copied views use, each member a thin api call. Edits
  autosave, debounced 300 ms. P9b delivers all of it; later packages never edit model.py.
  - Signals: pathsChanged, currentChanged, costChanged, sourcesChanged.
  - Attributes: paths, path, current, sel_step, scope, pending_change, sessions, stack_mode, gpu, history.
  - Methods: state, pending_upto, all_pending, seconds, stage_key, owner, cost_with, apply_change,
    add/keep/drop_variant, add_path, remove_path, set_opts, has/with/without_tool, slider_state, thresholds_now,
    set_selection, kept_frames, add_exit, plus `info()` and `image_size()`, which replace `s.data[...]`.
- **`WorkflowView` replaces PathDef** (pid = workflow id, name, params, end, variant_of, computed, rendered).
  - Its `opts` speaks the mockup's vocabulary: `debruitage`/`dn` ↔ denoise, `etoiles`/`appris` ↔ stars, `fond` ↔
    gradient, `deconv` ↔ deconvolution.
  - The views keep their internal block keys (calibration, alignement…).
- **What the model does not serve** (images, arrays, renders, runs) goes through the services of P10–P12.

### 5.3 Threads and processes (the GUI never blocks)
- **GUI thread**: widgets, workflow edits, and keys, states and plans from caches.
  - Never in the GUI thread: disk scans (`project_part`, `usable_as_input`), array loads, `auto()`, renders.
  - The caches are filled by the ApiWorker, per session and per key, and refreshed on edits and run events.
- **ApiWorker** (QThread, `services/worker.py`): scans, keys' file parts, describe, outputs, arrays, thumbnails,
  import. Requests carry a channel; the latest per channel wins; results come back as signals.
- **RenderWorker** (QThread, `services/render.py`):
  - LayerServices with least-recently-used engine eviction, plus the postengine.
  - Requests (layers, level, region, upto, want_mask) are coalesced by sequence number, latest wins (as
    `gui/render_tab.RenderWorker`): 1/4 while dragging, 1/2 on release, a real full-resolution region at zoom ≥ 50 %.
  - It renders on a GPU with no running job, else on the CPU. A GPU out-of-memory error falls back to the CPU, with a
    note.
- **Files**: the client never memory-maps store files, so a re-run can always replace them.
- **Runs**: the detached worker (§3), followed by RunFollower (a QTimer tailing a file).

### 5.4 Languages and preferences
- **i18n.py** is unchanged: English source, French in `client/locale/fr/*.json`, one file per package (`shell.json`,
  `graph.json`, `inspect.json`, `run.json`, `post.json`, `engine.json`). A test lists the untranslated strings.
- **Engine texts** (P16):
  - auto reasons become a template plus arguments (`Auto(value, reason, args)`, with `Auto.reason` still returning
    the English text for v1). The client formats the arguments with a decimal comma in French;
  - `astro_gpu/texts.py` is a metric catalogue (label, unit, format) for every metric key;
  - engine labels, help, reasons and metric labels are translated in `engine.json`.
- **`prefs.py`**: `%APPDATA%/astro_gpu/settings.json`, overridden by `ASTRO_GPU_SETTINGS`. Defaults: workspace
  F:/astro_projets, en, spec_mode off, device auto, recent [].
- **Tests and walkthroughs** set `ASTRO_GPU_SETTINGS`, `ASTRO_GPU_HOME` (scratch) and `QT_QPA_PLATFORM=offscreen`. A
  test checks that the real settings file is untouched.

### 5.5 What stays ◇ in v1
Shown in Spec mode only, disabled, with "◇ Not in the engine yet"; hidden otherwise. The status note reads "Spec mode ·
◇ not in the engine yet".
- **Blocks and settings**:
  - the 41 NEW settings. Unlike SPEC 5.7, their values are not stored and do not change the stage keys;
  - fine-tuned / pretrained / NL-Bayes denoise; Deconvolution; the stretch choice; Gradient sample areas.
- **Frames and Selection**:
  - flats per session, and one stack per session;
  - the master library; temperature and library checks;
  - Keep a frame against the thresholds (v1 can only exclude).
- **Settings pages**: Memory limit; Benchmark; Gaia local copy, ASTAP, astrometry.net; File formats; Cache limit.
- **Inspect and Help**: Preview on a region; Identified objects; Help > Glossary.
- **Export**: XISF, Adobe RGB / P3, the processing history in the header (v1 writes a sidecar file), linear outputs,
  the project as a script.

## 6. Work packages
### 6.1 Rules for every package
- **Commits**: agents never commit. The orchestrator commits each package by explicit paths, one at a time, and owns
  JOURNAL.md and RAPPORT.md.
- **Tests**: during a wave, each package runs its own tests (`pytest tests/test_<its files> -q`). The full fast suite
  (`-m "not slow and not network"`) runs at the end of the wave.
- **During P8**:
  - tests run on the CPU (`CUDA_VISIBLE_DEVICES=-1`: on this Windows machine an empty value does NOT hide the GPUs);
  - walkthroughs and client tests never open P8's projects: they use the F:/astro_essais/v2 projects;
  - client renders run on the CPU.
- **Engine freeze**: from P8's start (commit tagged `p8-engine`) to its end, nothing under `astro_gpu/` outside
  `astro_gpu/client/` changes. Engine fixes found meanwhile are queued.
- **Shared files**: a package edits only the files the table below gives it. `widgets.py` is frozen after P9a; new
  widgets go in the package's own modules. Each package writes its own `client/locale/fr/<package>.json`.

### 6.2 Ownership and order
| Package | Wave | Duration | Files owned |
|---|---|---|---|
| P1 Project v2 and chains | 1 | 2–3 h | `core/project2.py`, `core/workflow.py`, `tests/test_project2.py`, `tests/test_workflow.py` |
| P2 Stage store and keys | 2 | 3 h | `core/store.py`, `core/stage.py` (`Context.has` only), `tests/test_store.py` |
| P4 New stages, simple rendering | 2 | 3–4 h | `stages/v2.py`, `stages/background_poly.py`, `stages/render_simple.py`, `render/engine.py` (region only), `render/simple.py`, `render/sliders.py`, `render/layers.py`, `tests/test_simple_stages.py`, `tests/test_sliders.py`, `tests/test_layers.py` |
| P9a Client shell | 2 | 3 h | `client/__init__.py`, `__main__.py`, `app.py`, `i18n.py`, `prefs.py`, `widgets.py`, `viewer.py`, `home.py`, `settingsdlg.py`, `logview.py`, `dialogs.py`, `services/worker.py`, `locale/fr/` (copied files + `shell.json`), `pyproject.toml`, `tests/client/test_shell.py` |
| P3 Runner v2 | 3 | 3–4 h | `core/runner2.py`, `core/estimate.py`, `core/runner.py` (`run_id` argument, atomic timings only), `stages/stack.py` (`exclude` only), `stages/calibrate.py` (cache-only path only), `gui/widgets.py` (hide `internal` params only), `tests/test_runner2.py` |
| P5 API and cli2 | 4 | 3–4 h | `api.py`, `cli2.py`, `tests/test_api.py` |
| P6 Import of v1 projects | 4 | 3 h | `core/import_v1.py`, `tools/import_v1.py`, `tests/test_import_v1.py` |
| P16 Engine texts | 4 | 3 h | `core/params.py` (`Auto` only), the reason strings of `stages/*.py`, `texts.py`, `client/locale/fr/engine.json`, `tests/test_texts.py` |
| P7 End to end and v1/v2 check | 5 | 2–3 h + ≈ 2 h GPU | `tests/test_integration_v2.py`, `tools/v2_smoke.py` |
| P9b Graph, panel, model | 5 | 4–5 h | `client/model.py`, `blocks.py`, `graph.py`, `panel.py`, `settings.py`, `locale/fr/graph.json`, `tests/client/test_graph.py`, `tests/client/test_model_surface.py` |
| P8 The user's projects | 6 | 3 h + ≈ 12 h runs | `tools/projects/*`, data under F:/astro_projets/ |
| P17 QC tools port | 6 | 3 h | `tools/qc/*`, `tests/test_qc_v2.py` |
| P10 Inspect and Frames | 6 | 4 h | `steps.py`, `tune.py`, `charts.py`, `frames.py`, `sources.py`, `services/inspect.py`, `locale/fr/inspect.json`, `tests/client/test_inspect.py` |
| P11 Run | 6 | 3 h | `runs.py`, `services/run.py`, `locale/fr/run.json`, `tests/client/test_run.py` |
| P12 Post and Export | 6 | 5 h | `post3.py`, `postengine.py`, `exportdlg.py`, `services/render.py`, `locale/fr/post.json`, `tests/client/test_post.py` |
| P13 Integration and walkthrough | 7 | 4 h | `client/app.py` (final wiring), `tools/client_walkthrough.py`, `tests/client/test_walkthrough.py` |
| P14 QC | 8 | 3 h | outputs in `F:/astro_projets/_qc/` and `<project>/export/` (no code) |
| P15 Phase D (optional) | 9 | – | new stage files and their entries in `stages/v2.py` and `core/workflow.py`, after the freeze |

The orchestrator runs P8 in the background from wave 6 to its end; P14 waits for P8 and P17.

### 6.3 Acceptance
- **P1**:
  - the 5 chains equal 2.1's table; invalid options raise; the mockup's 8 with/without_tool cases hold;
  - save/load keeps a stored value equal to its static default; `excluded` holds (session, name) pairs; sessions take
    file lists;
  - `from_v1` on the 9 real project.json gives 1.5's names and options, and a file-list + mtime snapshot shows nothing
    was written.
- **P2**, on toy stages A → B (optional) → C, with an EXTRA edge and a stage with an auto rule:
  - the key rule:
    - a stored value equal to the static default on an `AUTO_FIELDS` field gives a key different from "not stored";
    - on a field without an auto rule it gives the same key;
    - 2 and 2.0 on a float field, and "2" coerced by Params, hash the same; an invalid value raises;
    - a path field written relative, absolute, with `\` or `/` gives one key; the same name, size and mtime elsewhere
      keeps it; a newer mtime changes it;
    - excluded (session, name) pairs hash the same after the raws move;
  - sharing: the same upstream shares a key; a parameter changes only the downstream keys; reverting restores the old
    key; workers is ignored; a subprocess computes the same key;
  - optional and EXTRA dependencies enter `deps_in_chain` only when present;
  - inputs: copied inputs (same names and sizes) keep the inventory key, and reordered frames change it;
  - store: absent stages are never created, and a `.tmp` folder is never done.
- **P3**:
  - on toy stages:
    - shared stages run once; events carry run, key and wf; a second plan is empty;
    - a change plans only its downstream, and reverting plans nothing; run to a block works;
    - every job writes into `<key>.tmp` and renames on success; a failing re-run of a done key leaves the old folder
      intact;
    - a failed job skips only its downstream jobs, and an independent workflow still completes;
    - STOP → interrupted → resume from the `.tmp` checkpoint (`test_core.StageB` pattern);
    - run.lock is O_EXCL, and a stale lock is replaced;
  - the worker:
    - its environment has `CUDA_VISIBLE_DEVICES`, and it is created with `CREATE_BREAKAWAY_FROM_JOB`;
    - a detached worker is followed through `read_events` and survives the end of the launching process;
  - the engine:
    - `select_frames` keeps its positional signature and honours `exclude`; `exclude` (level `internal`) is hidden
      in the v1 form;
    - the cache-only calibrate gives the same `lights.f16` as a full run on 8 heart_a7s frames;
    - `Pipeline.status()` of F:/astro_essais/projets/heart_a7s stays all done;
    - estimates are within ±50 % of Iris's measured times.
- **P4**:
  - stages and rendering:
    - a synthetic degree-2 gradient is recovered within 1 %; render_simple writes its files; the Raw stack exit has no
      colour cast (channel medians equal within 2 %);
    - the simple render of 2000 × 1500 at scale 1 takes < 2 s;
    - a region render with the 1/4 fit equals the crop of the full render within 2 levels (8-bit) at p99.9, away from
      a 64 px margin;
  - sliders: each slider's position 2 equals the base and each slider is monotonic; greyed sliders give their reasons;
  - `stages/v2.py`: every key `auto()` returns on heart_a7s is in `AUTO_FIELDS`;
  - exports re-open with the right dtype and size; LayerService evicts the least-recently-used engine.
- **P5**:
  - a toy scenario (create; 4 workflows; a parameter on 2; make, keep and discard a variant; rename; delete; undo via
    `set_workflows`) in which `cost()` equals the plan after each change;
  - `set_param` drops a value only at a known suggestion, and keeps it while upstream is not done;
  - `list_projects` skips `_*` folders; the flats-merge health check fires on two sessions;
  - after P6, `describe()` on a real adopted stack gives value, suggested, origin and reason.
- **P6**:
  - a toy v1 project adopts the right prefix, and a changed old parameter stops adoption there;
  - the write guard raises on a deliberate write under the old root;
  - **slow test**, copying F:/astro_essais/projets/heart_a7s → F:/astro_essais/v2/heart_a7s:
    - 15 stages adopted, no `_ckpt` or calibrated cache copied;
    - `plan(Full)` empty; `plan(Classic)` = background_poly + render_simple;
    - the old snapshot is unchanged;
  - the free-space check refuses when F: is too small (simulated);
  - an earlier layer gets its stack-key file; a missing reference is dropped with a note;
  - `--dry-run`, under the guard, first on the copy and then on the 9 real projects, matches 1.6 and writes nothing.
- **P16**:
  - no auto reason is an f-string any more (grep); `Auto.reason` still gives the English text;
  - `describe()` in French shows decimal commas;
  - every metric key in the heart_a7s v2 manifests has a catalogue entry;
  - the coverage report shows zero missing French for engine labels, help, reasons and metrics;
  - `Pipeline.status()` of heart_a7s stays all done.
- **P7**:
  - **end to end**: a v2 project at F:/astro_essais/v2/heart_a7s_e2e from F:/astro_essais/donnees/heart_a7s
    (`inventory.max_lights` 12, denoise 500, star_net steps 300, `starless.sharpen` off), with the 4 templates and a
    Raw stack exit, run all through cli2, detached:
    - jobs = distinct keys (inventory → color once);
    - all done, and a second run plans nothing;
    - every layer renders at 1/4 without NaN; Full exports; times and disk per stage logged;
  - **v1/v2 check**: Full run on all of `donnees/heart_a7s` with `projets/heart_a7s`'s parameters, on the same physical
    GPU. stack, denoise and render_prep equal `projets/heart_a7s`'s outputs bit for bit;
  - both runs use a pilot ASTRO_GPU_HOME with a copy of the profiles.
- **P8**:
  1. **Freeze.** Tag `p8-engine` at start. Pilot ASTRO_GPU_HOME at `F:/astro_projets/_etat` (a copy of the profiles).
  2. **Raws.** Copy the raws not on F: (sources: ENGINE_KNOWLEDGE §2) to `F:/astro_projets/_raw/<name>/<kind>/`
     (robocopy /COPY:DAT).
     - Sessions keep file lists: only the 100 listed darks.
     - FlamingStar's two G: folders are merged back into one, as in its project.json, so its old order is kept.
     - Heart and NGC7822 raws are referenced in place.
  3. **Import** in copy mode after the free-space check; log the report. Star layers and references go to `inputs/`.
  4. **Workflows**: Classic stack, Classic + denoise, Full, Full + neural stars, plus "Full + earlier neural layer"
     where it exists.
  5. **Run all** through `cli2 run --detach`, in two queues. Heavy projects (Heart, Horsehead, FlamingStar, Wizard,
     NGC7822) go on cuda:0 (16 GB); the others on cuda:1 (8 GB, the display).
  6. **Candidates.** Bases faithful / artistic come from the old `results/*/*_settings.json`. Export each Full* with
     both, and Classic with its defaults, into `F:/astro_projets/_qc/<name>/candidates/`. Default Post: Full visible
     at the bottom.

  Accept:
  - `cli2 status` all up to date, or failures and skipped jobs reported;
  - candidates exist;
  - the old folders' snapshots are unchanged;
  - no engine file changed since the tag.
- **P17**:
  - `tools/qc` addresses `PROJECT::WORKFLOW`;
  - each old image is paired with the workflow of the same preparation: an old `render_prep_learned` image goes with
    "Full + neural stars" or "Full + earlier neural layer";
  - each render is remapped with its own project's files;
  - the tools refuse to write under a v1 project (no `--sheet` there);
  - blind sheets of heart_a7s v1 against v2 come out with the right pairs.
- **P9a** (offscreen):
  - Home lists a scratch workspace and skips `_*`; New project writes a v2 folder;
  - French and back rebuild the menus; prefs go only to the `ASTRO_GPU_SETTINGS` file;
  - a grep of `client/` finds no "alt2", "data.json", "Demo" or mockup import;
  - shots home_en, home_fr, settings.
- **P9b**, on F:/astro_essais/v2/heart_a7s with 4 workflows:
  - `test_model_surface` statically lists every `s.<member>` the copied views use, and checks each exists on
    ProjectModel;
  - the tree: Calibration → Color shared by 4, Denoise by 3, a split at Stars (the methods differ), 4 outputs;
  - the states come from the API; adding and removing Denoise and Stars follow the rules, with toasts;
  - a change on a shared done block offers Replace or Try as variant, with `api.cost` times; Undo and Redo work;
  - shots graph, panel_shared, variant.
- **P10**:
  - the real stack shows with auto-stretch, histogram and catalogue metrics;
  - Selection thresholds update the kept count through `selection_preview`;
  - the Color chart draws; an Outdated block shows "previous result"; Compare works;
  - importing a heart_a7s folder gives the right counts, sessions and thumbnails; Exclude changes the Stacking key;
  - no disk access happens in the GUI thread (a debug hook counts them);
  - shots.
- **P11**:
  - the plan dialog shows the blocks, the sharing workflows and the durations (≈ marks);
  - a real render_simple job runs with the heartbeat under 200 ms; the stripes turn green and the layer appears;
  - a failed job shows its skipped downstream blocks;
  - Stop during denoise after step 1000 (it checkpoints every 1000) → the next run resumes at 1000;
  - killing the worker by PID → interrupted, the stale lock is replaced, and the next run resumes;
  - closing and reopening during a run reattaches;
  - the denoise and kill tests run on a free GPU, or after P8; a slow toy stage covers them on the CPU.
- **P12**:
  - with the Full and Classic layers of heart_a7s v2:
    - masks (stars, H-alpha, luminance, brush), filters, split stars / starless and Undo work;
    - TIFF16, JPEG and FITS32 + WCS export and re-read;
    - the export reduced to screen size matches the 1/2 preview within 2 levels at p99.9;
  - **budgets**, measured with `render/layers.py` on Heart's existing render_prep (read-only, never opened in the
    client), on a free GPU, or after P8:
    - draft ≤ 150 ms, 1/2 ≤ 1 s, 100 % region ≤ 2 s;
    - a simulated GPU out-of-memory falls back to the CPU;
  - **v1/v2 check**: `render_layer(faithful, sliders at 2)` equals `RenderEngine.render(faithful)` bit for bit;
  - render fixes needed in `render/` wait for the end of P8;
  - shots.
- **P13**:
  - every screen of SPEC §3 is shot in EN and FR into `F:/astro_projets/_shots/<date>/`, from F:/astro_essais/v2
    projects only, rendering on the CPU during P8;
  - zero untranslated strings on the FR shots;
  - the heartbeat stays under 200 ms outside project open;
  - all suites are green.
- **P14** (after P8 and P17):
  1. The bulletin on the candidates.
  2. Blind sheets, new against old, paired by preparation (P17), and Classic against Full.
  3. Independent judges (`qc_prompt.md`), plus your own look at 100 %.
  4. An independent GUI review of P13's shots against the mockup.
  5. Copy only the winning faithful and artistic images into `<project>/export/`. The report states the rings the
     Classic polynomial is expected to leave on M31 and M33 (open problem 7).

  Accept: a verdict per project (better / same / worse, and where) and the defects per screen, written for the JOURNAL.
- **P15** (optional, after the freeze):
  - new methods, each a stage plus a method entry with its ◇ removed: Gradient early (`background_early`),
    Richardson-Lucy deconvolution, NL-Bayes denoise, stretch choices;
  - each needs non-regression on the 4 datasets and blind QC before it becomes the default.

## 7. Decisions and risks
**Decisions and why:**
1. **v2 beside v1, import rather than in-place migration**: real projects stay read-only; the QC tools keep working.
2. **Keys without auto values, from validated stored values**: auto values derive from the upstream keys. If
   `describe()` ever shows drift, bump the stage version; never add auto values to the key.
3. **Jobs run by v1's `_run_stage` through a shim**, into a `.tmp` folder renamed on success: one runner logic, and a
   failed re-run never destroys a result.
4. **Two new stages** make 4 templates real; Classic + denoise reuses Full's denoise.
5. **The real order in the graph**; the reorder comes only as new methods, so quality never changes silently.
6. **Import by copy only, with checks and a write guard**: clean, self-contained folders on F:.
7. **The mockup's internal keys stay in the client** (less porting risk); the engine and the API use English keys.
8. **The Post compositor stays in the client** (numpy, as in the mockup); the engine renders layers or regions on
   the stack grid.
9. **A frozen plan file per run, and an engine frozen during P8**: the user's projects come from one code version.

**What the orchestrator must watch:**
1. **Writes to real projects**: the audit-hook guard and the snapshot tests of P1, P6 and P8. `project.json` is never
   written.
2. **Detached runs**: workers use `CREATE_BREAKAWAY_FROM_JOB`. Check them by PID; kill by PID only.
3. **The GPU**: heavy P8 projects on cuda:0 (16 GB); cuda:1 also drives the display. Tests, client renders and
   walkthroughs use the CPU during P8.
4. **Disk**: 30–60 GB of store per project, plus 17–30 GB of calibrated cache when calibrate rebuilds it. The import
   checks free space; old keys and `_trash` are cleared only after a confirmation.
5. **Raw order**: FlamingStar's merged folder must reproduce the old order, or the import recomputes (3 h lost, not
   correctness).
6. **Demo leftovers in a 9 000-line port**: every client package extends P9a's grep test.
7. **Catalogue outages**: the adopted catalogs and the cache cover the 9 fields; do not stall on them.
8. **Known and accepted**: a resumed star_net training differs slightly (CUDA random state not saved). The Classic
   polynomial leaves rings on M31 and M33; the report says so.
9. **Two failed attempts on a package**: stop it, log it, move on.
   - Priority: P1–P7, P16 and P9–P13, then P8 and P17, then P15.
   - P8 still starts as soon as P7 passes, because it runs in the background; P13 has 4 h for the integration.
