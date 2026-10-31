# Engine contract

This is the API the heavy client needs from the engine, derived from what the mockup `mockup/alt3` calls. Each item
was checked by grep against `astro_gpu` (`core/`, `stages/`, `render/`, `io/`, `cli.py`) on 2026-10-31.

## 0. Conventions

**Status marks.**
- **EXISTS**: usable as it is. The engine function is named.
- **PARTIAL**: something close exists. The item says what is missing.
- **MISSING**: nothing yet.

**Process.**
- A run happens in a worker process. The CLI already does this: `astro-gpu run`. A lock allows one run per project.
- The client follows the run through `logs/events.jsonl`. The other calls are in-process and fast.

**Names.**
- Stage and parameter names are the engine's.
- The workflow options get English keys, proposed below. The mockup's keys are noted beside them.

```python
WorkflowId = str                      # engine today: "" = main, else a variant name
StageRef = tuple[WorkflowId, str]     # (workflow, stage)

class Options(TypedDict):
    """Which optional blocks a workflow has, and their methods (SPEC.md 1)."""
    denoise: Literal[None, "n2n", "finetuned", "pretrained", "nlbayes"]   # mockup: debruitage + dn
    deconvolution: Literal[None, "rl", "neural"]                           # mockup: deconv + deconv_m
    stars: Literal[None, "psf", "psf_neural"]                              # mockup: etoiles + appris
    gradient: Literal["halpha", "polynomial"]                              # mockup: fond
```

## 1. Project and workspace

```python
def list_projects(workspace: Path) -> list[ProjectSummary]:
    """The projects of the workspace, most recent first, for the Home cards.

    Each summary has: name, target, nights, integration (h), last date, state text and a thumbnail.

    PARTIAL: core.project.recent_projects_described() returns (name, folder, description) from a per-user registry.
    Missing: a scan of the workspace folder, nights, integration, the state ("2 workflows · 1 outdated") and the
    thumbnail.
    """

def create_project(root: Path, name: str, target: Target | None = None) -> Project:
    """A new, empty project folder. Frames come later (section 2).

    EXISTS: core.project.Project.create(root, name, inputs, target, scratch_dir).
    The project is a folder with project.json. The mockup's file filter is "*.astroproj": the client opens the folder
    or its project.json.
    """

def open_project(root: Path) -> Project:
    """EXISTS: Project(root); core.project.remember(project) adds it to the recent projects."""

def Project.save() -> None:
    """EXISTS: an atomic write of project.json. Settings and variants are saved as soon as they are set."""

def Project.save_as(root: Path) -> Project:
    """MISSING: copies the project (its definition, and optionally its cache) to a new folder."""
```

## 2. Frames, sessions and masters

```python
def scan_folder(folder: Path) -> ScanResult:
    """Reads a folder: every frame with its type and header, and the calibration folders found next to it.

    PARTIAL:
    - core.project.list_frames and frame_kind list the files.
    - io.frames.read_info returns a FrameInfo: size, CFA, black and white levels, exposure, ISO / gain, time, camera,
      focal length, pixel size, temperature, RA / Dec, object.
    - core.newproject.guess_calibration and guess_names look for the calibration folders.
    Missing: filter, binning, rotation, and sorting the files by type from their content (level, exposure).
    """

def group_sessions(frames: list[FrameInfo], criteria: GroupingCriteria) -> list[Session]:
    """The sessions: one per night (noon to noon), split further by exposure, ISO, filter, temperature (± 2 °C),
    binning, rotation and, optionally, object. Flats belong to their session.

    MISSING: the engine has one flat list per project (Inputs).
    """

def add_frames(project: Project, files: list[Path], session: str | None = None) -> None:
    """Adds a folder or a session (Import folder…, Add session…).

    PARTIAL: edit project.config.inputs, then save(). The inputs fingerprint (runner._inputs_fingerprint) then makes
    the whole chain run again. There are no sessions.
    """

def set_excluded(project: Project, frames: list[str], excluded: bool) -> None:
    """Frames excluded by hand, for the whole project. They are an input of the stack, or of the calibration for
    calibration frames.

    PARTIAL: a frame can only be removed from Inputs, and then everything runs again from the inventory. There is no
    exclusion list read by the stack.
    """

def match_calibration(project: Project) -> list[CalibrationMatch]:
    """For each session: the darks, flats, bias and dark flats used, from the session or from the master library,
    with the tolerance checks.

    MISSING: the calibrate stage builds its masters from the project's own lists at every run.
    """

def masters(library: Path) -> list[MasterInfo]: ...
def build_masters(folder: Path, library: Path, progress: Callable) -> list[MasterInfo]: ...
    """The master library: kind, exposure, ISO, temperature, frame count, date, and where each is used.
    MISSING."""

def health_checks(project: Project) -> list[Check]:
    """A list of (level ok | warn | info, text): darks matched, temperature apart, no bias, rotated camera, FWHM per
    session, disk needed against free space.

    MISSING. The disk part is PARTIAL: core.newproject.space_text.
    """

def frame_table(project: Project, wf: WorkflowId) -> FrameTable:
    """One row per light: name, time, FWHM, transparency, residual and status. The Selection chart reads it.

    EXISTS:
    - work/register/reg.npz holds fwhm, transp, rms, coef and ref for each frame;
    - work/stack/frames.json holds path, time, weight and the reason a frame was excluded.
    The NEW criteria (background, roundness, SNR) are MISSING.
    """

def frame_image(path: Path, size: int, stretch: SharedStretch) -> np.ndarray:
    """A thumbnail or a viewer image of a raw frame, with one stretch shared by a group of frames (for blinking).

    PARTIAL: io.frames.read_cfa and the inventory's superpixel debayer exist. There is no thumbnail service and no
    shared stretch.
    """
```

## 3. Workflows, options and settings

```python
def list_workflows(project: Project) -> list[Workflow]:
    """Each workflow: id, name, options, params ({stage: {name: value}}, stored values only), end (exit block or
    None) and variant_of.

    PARTIAL: main plus project.config.variants (name -> params, note, slug). Variants are parameter overlays on main.
    Missing: options, exit blocks, variants of variants, and workflows that are not built on main.
    """

def add_workflow(project: Project, name: str, options: Options, params: dict | None = None,
                 copy_of: WorkflowId | None = None) -> Workflow:
    """New workflow (from a template or a copy). PARTIAL: Project.create_variant(name, params); no options."""

def rename_workflow(project: Project, wf: WorkflowId, name: str) -> None:
    """MISSING: the variant name is also its key."""

def remove_workflow(project: Project, wf: WorkflowId) -> None:
    """PARTIAL: Project.delete_variant(name). It deletes the variant's own stage folders; main cannot be removed."""

def set_options(project: Project, workflows: list[WorkflowId], **changes) -> None:
    """Adds or removes an optional block, or changes a method, in these workflows (SPEC.md 5.3 rules).

    MISSING: every stage always runs; only star_net.enabled turns one off. There is no optional Denoise, Stars or
    Deconvolution, and no method stage (denoise_ft, denoise_ai, denoise_cl, fond_simple, deconv, rendu_simple).
    """

def set_param(project: Project, workflows: list[WorkflowId], stage: str, name: str, value: Any | None) -> None:
    """Stores a value; None goes back to the suggestion.

    PARTIAL: Project.set_override and clear_override act on the active variant (or main) only, one workflow at a
    time.
    """

def workflow_stages(project: Project, wf: WorkflowId) -> list[str]: ...
def stage_deps(project: Project, wf: WorkflowId, stage: str) -> list[str]: ...
    """The stages of a workflow in order, and what each depends on, given its options.

    PARTIAL: core.project.stage_graph() gives the fixed chain of 17 stages. The new order is MISSING: Gradient
    before Color and Stars; Deconvolution after Color and before Denoise, with a PSF measured before Denoise.
    """

def make_variant(project: Project, wf: WorkflowId, change: Change, name: str) -> Workflow: ...
def keep_variant(project: Project, variant: WorkflowId) -> None: ...
def discard_variant(project: Project, variant: WorkflowId) -> None: ...
    """Try as variant, Keep and Discard (SPEC.md 5.2).

    PARTIAL: create_variant (always from main), promote_variant(name, keep_main_as) for Keep, and delete_variant for
    Discard.
    """

def add_exit(project: Project, wf: WorkflowId, block: str, name: str) -> Workflow:
    """A workflow that copies wf up to `block` (Stacking), then goes to its own simple output (SPEC.md 5.5).

    MISSING (needs rendu_simple).
    """

def methods(block: str) -> list[MethodInfo]:
    """Each method of a block: key, name, short name, description, runs_on ("gpu" | "cpu" | "both"), stage, new.

    PARTIAL: core.methods.DECISIONS has the concept (a Decision with Options, as parameter sets, and evidence).
    Missing: runs_on, and the NEW methods (BLOCKS.md).
    """
```

## 4. Stage key and cache

```python
def stage_key(project: Project, wf: WorkflowId, stage: str) -> str:
    """A hash of: the stage name and version; the options that change this stage or its upstream stages; the stored
    settings of the stage and its upstream stages; the project-level decisions. Equal keys share one result
    (SPEC.md 5.1).

    PARTIAL: a stage is up to date when its manifest has the same version, params_hash and upstream run_ids
    (core.runner). But results are found by folder (work/<stage>, or variants/<slug>/<stage>), not by key.
    """

def sharing(project: Project, wf: WorkflowId, stage: str) -> list[WorkflowId]:
    """The workflows that compute this stage exactly as wf does.

    PARTIAL: Project.variant_stages(name) gives a variant's own stages. A variant shares with main only the stages
    before its first change, and two variants never share a stage.
    """

def cache_lookup(project: Project, key: str) -> Path | None:
    """The stored result of a key. Old results are kept, so that going back to old settings costs nothing.

    MISSING: a stage folder holds one result, and runner._clear_outputs erases it when the parameters change.
    """

def cache_usage(project: Project) -> CacheUsage: ...
def clear_cache(project: Project, keep: Callable[[str], bool]) -> int: ...
    """The disk used, and the cleaning (Settings > Folders & disk).
    PARTIAL: Project.variant_size(name), and the scratch_dir per project."""
```

## 5. State, pending stages and durations

```python
def states(project: Project) -> dict[StageRef, StageState]:
    """For each stage of each workflow: done | pending | stale, with a reason.

    PARTIAL: Pipeline.status(), for the active variant only. Its states are done, ready, stale, pending, failed,
    interrupted and running. The client maps them:
    - done → Up to date;
    - ready or pending → Pending;
    - stale, failed or interrupted → Outdated.
    The outputs of an Outdated stage stay in place until it runs again, so Inspect can show the previous result.
    """

def pending(project: Project, workflows: list[WorkflowId] | None = None,
            upto: StageRef | None = None) -> list[StageRef]:
    """Run: everything pending, each key once. Run to here: the stages needed for `upto`.

    PARTIAL: Pipeline.plan(until, force), one variant at a time.
    """

def estimate(project: Project, stage: str, device: Device) -> Estimate:
    """Seconds for this stage on this machine, and whether that figure was measured or estimated.

    PARTIAL:
    - runner._record_timing keeps per-machine durations (~/.astro_gpu/timings.json);
    - gui.runner.stage_history_seconds returns their median;
    - Stage.cost_hint is declared but is 0 in every stage.
    Missing: scaling with frame count and megapixels, the CPU estimate, and NEW stages.
    """

def disk_estimate(project: Project, stages: list[StageRef]) -> float:
    """GB a run will write (Run dialog, health check). MISSING."""

def machine() -> MachineInfo:
    """GPU name, memory, CUDA availability, CPU cores, RAM.

    PARTIAL: torch is used directly; Context.device falls back to the CPU; project.config.devices names the GPUs.
    No report.
    """

def benchmark(progress: Callable[[float, str], None]) -> BenchmarkResult:
    """Measures this machine on a small data set: GPU and CPU ratios to the reference machine. MISSING."""
```

## 6. Run, progress and cancel

```python
def start_run(project: Project, items: list[StageRef]) -> RunHandle:
    """Runs these stages, each stage key once, in dependency order, in a worker process.

    PARTIAL: Pipeline.run(until, force) runs the active variant up to one stage (CLI: astro-gpu run --variant
    --until), and replans after each stage. Missing: a list that spans several workflows.
    """

# Events, appended to logs/events.jsonl. EXISTS: core.journal.Journal.emit, read_events(path, start), tail(path).
#   run_start(stages, variant)        stage_start(stage, title)
#   progress(stage, frac, msg, eta, elapsed)                    (sent at most once per second)
#   log(stage, level, msg)            metric(stage, key, value)    preview(stage, name, path)
#   stage_end(stage, status, duration)                          run_end(status)

def cancel(project: Project) -> None:
    """Stops at the next progress check. The stage is marked interrupted and resumes from its checkpoint when it is
    run again with the same parameters.

    EXISTS: core.runner.request_stop (a STOP file), Context.check_stop, and checkpoints in work/<stage>/_ckpt.
    """

def preview_region(project: Project, wf: WorkflowId, block: str, frame: str, rect: Rect) -> RunHandle:
    """Runs the blocks on a region drawn on a frame, at full resolution (Inspect: "Preview on a region"). MISSING."""
```

## 7. Outputs of a stage

```python
def stage_outputs(project: Project, wf: WorkflowId, stage: str) -> StageOutputs:
    """Previews {name: jpg path}, metrics {key: value}, params, origins, auto reasons, duration, finished.

    EXISTS: the stage's manifest.json (metrics, params, origin, auto_reasons, duration) and
    previews/<name>.jpg (Context.preview).
    The previews written today are:
    master_flat, calibrated_frame, middle_frame, fixed_pattern, frame_quality, stack, halpha_reference,
    spectral_response, before_after, psf, star_model, starless, star_removal, background, nebula_layer.
    """

def stage_array(project: Project, wf: WorkflowId, stage: str, name: str) -> np.ndarray:
    """Linear data, for auto-stretch, the histogram and the ADU readout in Inspect.

    PARTIAL: Context.load reads the .npy files of a stage, but there is no public, read-only access.
    """

def identified_objects(project: Project, wf: WorkflowId) -> list[SkyObject]:
    """Name, x and y of the catalogue objects in the field (Astrometry: "Identified objects").
    MISSING: the mockup's labels came from its build script."""

def pixel_to_sky(project: Project, wf: WorkflowId, x: float, y: float) -> tuple[float, float]:
    """RA and Dec of a pixel, for the readout.
    PARTIAL: the astrometry stage fits a WCS (astropy) but does not expose it."""

def spectral_response(project: Project, wf: WorkflowId) -> SpectralCurves:
    """Wavelengths and the R, G, B response, so that the client draws the Color chart.
    PARTIAL: the color stage draws a preview image of it; the data are not returned."""
```

## 8. Parameter descriptions

```python
def describe(project: Project, wf: WorkflowId, stage: str) -> list[ParamInfo]:
    """For each parameter: name, label, help, tip, type, default, value, origin (default | auto | user), the reason
    of an automatic value, ge / le, step, unit, choices, and level (basic | advanced; the GUI adds "system").

    EXISTS for the engine's parameters:
    - StageParams.describe() gives name, type, default, bounds, level, label, help, tip, unit, step and choices;
    - Pipeline.effective_params(stage) gives (params, origin, {name: Auto(value, reason)}); reasons need the
      upstream stages done.
    MISSING for the 41 NEW settings (BLOCKS.md).
    """
```

## 9. Rendering a workflow layer

```python
def render_layer(project: Project, wf: WorkflowId, sliders: dict[str, float], scale: float,
                 region: Rect | None = None, part: Literal[None, "stars", "starless"] = None) -> Layer:
    """The workflow's output, with its simplified sliders (positions 0–4, 2 = suggested).

    - scale 1/4 is the draft while a slider moves, 1/2 is the screen render, 1 is full resolution.
    - region: full resolution for the visible part only.
    - part: the halves of "Split stars and starless".

    PARTIAL:
    - render.engine.RenderEngine.render(settings, scale) returns Layers(final, starless, stars, scale, info). It
      reduces by an integer factor (1/4 and 1/2 work) and caches the nebula and star parts.
    - StarRemovalEngines chooses between the measured and the neural star removal.
    Missing:
    - render of a region;
    - render of workflows without Stars (rendu_simple), and the stretch choice;
    - the slider table below.
    """

def slider_settings(key: str, position: float) -> dict[str, float]:
    """The engine RenderSettings a slider moves, interpolated between its five positions (POST.md 2.1). For example,
    "Sky" → sky_level in 12, 18, 24, 30, 36.

    MISSING in the engine: the table is in the mockup's prepared data. RenderSettings has every field it moves.
    """

def sliders_of(project: Project, wf: WorkflowId) -> list[SliderInfo]:
    """Each slider: name, group, ends, available, and the reason when it is not (SPEC.md 5.5, BLOCKS.md 12).
    MISSING: the client can derive it from the options."""

def post_masks(project: Project, wf: WorkflowId, scale: float) -> dict[str, np.ndarray]:
    """{"stars": every drawn star and the halo of bright stars, "halpha": the H-alpha emission map, 0 at the median,
    1 at the brightest 3 %}.

    PARTIAL: render_prep holds the star positions and magnitudes (stars.npz) and the emission layers, but no ready
    mask.
    """
```

## 10. Post composition

```python
def compose(layers: list[LayerSpec], scale: float, region: Rect | None = None,
            upto: int | None = None, want_mask: int | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """The composite, bottom first: workflow layers with their own filters, filter layers on everything below,
    masks made of parts, blend modes and opacity (POST.md 4). `upto` gives the "before" image; `want_mask` also
    returns one layer's mask.

    MISSING in the engine. Its only composition is final = screen(nebula, stars). The mockup's version is
    postengine.Engine.compose: it can live in the client, on the layers given by render_layer.
    """

FILTERS: dict[str, FilterInfo]
    """The 13 filters, with their parameters and families (POST.md 2.3).
    MISSING: render.ops has building blocks (gaussian_blur, screen, Oklab chroma)."""
```

## 11. Export

```python
def export(project: Project, composition: list[LayerSpec], options: ExportOptions,
           progress: Callable[[float, str], None]) -> ExportResult:
    """Writes the final image at full resolution (SPEC.md 3.7).

    The options are: format (TIFF 16, TIFF 32f, FITS 32f, XISF 32f, PNG 16, JPEG), colour profile, size (full,
    2048, 4096 px), path, WCS in the header, history in the header, each layer, the masks, and each workflow's
    linear output as FITS.

    PARTIAL: render.export.export(project, engine, settings, ExportSettings, progress) exports one rendering:
    - a 16-bit TIFF at full resolution with an sRGB ICC profile;
    - a 16-bit PNG and a JPEG at sensor scale;
    - the starless and stars layers;
    - the orientation (as captured, north up, match reference) and the reference framing;
    - quality checks and a settings JSON.
    Missing: the composition, float TIFF, FITS and XISF, the profile choice, web sizes, WCS and history headers,
    masks, and linear outputs.
    """

def export_script(project: Project, path: Path) -> None:
    """The project as a reproducible recipe (File > Export project as script…).

    MISSING. The CLI has the pieces: new, set, variant, run and render.
    """
```

## 12. Application settings the engine must honour

| Setting (Settings dialog) | Status | Engine today |
|---|---|---|
| Device: Automatic / GPU / CPU only | PARTIAL | `config.devices` per project; Context.device falls back to the CPU |
| CPU threads | PARTIAL | the `workers` parameter of calibrate and register |
| Memory limit | MISSING | — |
| Cache and temporary folders, cache limit | PARTIAL | `scratch_dir` per project; no limit |
| Gaia DR3 local or online, plate solver (Internal / ASTAP / astrometry.net), survey cache | PARTIAL | online catalogues (`core/gaia_archive.py`), an internal blind solver (`core/blindsolve.py`) |
| Intermediate format (FITS / XISF, bit depth, compression), keep calibrated frames | MISSING | `.npy` files and a scratch cache |
| Master library folder and tolerances | MISSING | — |

## 13. Stages and changes to build (Spec mode ◇)

| Item | Status |
|---|---|
| `fond_simple`: polynomial gradient | MISSING |
| `deconv`: Richardson-Lucy or neural, on both half-stacks | MISSING |
| `denoise_ft`, `denoise_ai`, `denoise_cl`: fine-tuned, pretrained, NL-Bayes | MISSING |
| `rendu_simple`: output without Stars, with a stretch choice | MISSING |
| Gradient computed before Color and Stars, from the stack and the catalogue (today `background` depends on `starless`) | MISSING |
| A PSF measured before Denoise (today `optics` depends on `denoise`) | MISSING |
| The 41 NEW settings (BLOCKS.md) | MISSING |
| A CPU path: measure each stage without a GPU, then decide where a classic method is needed | MISSING |

## 14. Summary

| Status | Items |
|---|---|
| EXISTS | create and open a project, save, the frame table, run events, cancel and resume, stage outputs (previews and metrics), parameter descriptions (engine parameters) |
| PARTIAL | list projects, scan a folder, add frames, exclude frames, list / add / remove workflows, set_param, stage list and dependencies, variants (make / keep / discard), methods, stage key, sharing, cache usage, states, pending, estimate, machine, start a run, stage arrays, pixel to sky, spectral response, render a layer, Post masks, export, application settings |
| MISSING | save as, group sessions, match calibration, master library, health checks, rename a workflow, set options (optional blocks and methods), exit blocks, a cache by key, disk estimate, benchmark, preview on a region, identified objects, the slider table, slider availability, the Post composition and filters, export as a script, the new stages and the new order |
