# Review of DESIGN.md

An independent agent reviewed this design on 2026-10-31 against PILOT.md, SPEC.md, ENGINE_CONTRACT.md,
ARCHITECTURE.md, ENGINE_KNOWLEDGE.md and the engine code. The real projects were only read (manifests, project.json,
results/*_settings.json). Severity: **blocking** = fix before P2 starts; **important** = fix in the design before the
package concerned; **minor** = when convenient.

## Verdict
The architecture reaches the goal: v2 beside v1, a content-addressed store over unchanged stage code, a checked
import, the mockup on an API; most of the spec is covered or openly ◇. One key rule is wrong (point 1): fix it before
P2, since changing the key later invalidates every stored result, P8's 10 h included. The weak points are at the
edges: paths and external files, in-place re-runs, old-vs-new QC pairing, engine texts in French, and Post at 100 %.
No package adds an instrument-specific rule.

## 1. Key, store and import
**1. Blocking · §1.2, §1.3 · A stored value equal to the static default is dropped, even if auto differs.**
Precedence is default < auto < user (`core/params.py:74-90`). `stack.scale`: default 1.5 (`stages/stack.py:18`), auto
2.0 or 1.0 (`:88-93`). `min_transparency`: default 0.6 (`:27`), auto 0.3 on hazy nights (`:97-99`), the M33 and
Wizard case. `denoise.iterations`: default 24000, auto 6k–30k (`denoise.py:17,45-50`).
A user who types 1.5 or 0.6 to override the auto value loses it on save. If such a value reaches the overrides by
another path (from_v1, implied values), the key strips it, and two parameter sets share one key. "Drop a value equal
to the suggestion" fails in the same way before upstream has run: `effective_params` swallows the auto error
(`core/runner.py:72-75`), so the suggestion falls back to the default.
→ Drop a value only when it equals a *known* suggestion. Never strip a field that has an auto rule (declare
`auto_fields` per stage). Hash values after validation through Params (`model_dump(mode="json")`), so 2 and 2.0 hash
the same, and normalise paths. P2 tests both.

**2. Important · §1.2, §1.4 · Relative paths, path-valued parameters, per-user state.** Relative paths resolve
against the runner's working directory: `reference.py:28-34` tests `Path(ref_path).exists()` and raises;
`star_net.layer_file` returns `Path(v)` (`star_net.py:72-79`). `starless.learned_stars` and `denoise.reuse_model` are
keyed by their path string, not by the file. `color` reads and writes `STATE/camera_profiles` (`color.py:22,94,126`),
so its results depend on the run order and on ASTRO_GPU_HOME. → StoreProject makes every path absolute; the worker
runs in the project root. Key files by name, size and mtime. P7 and P8 use a pilot ASTRO_GPU_HOME with a copy of
the profiles.

**3. Important · §1.3, §3 · Re-running a done key in place destroys it first.** This happens for an unusable
inventory or calibrate, a linked key, and a failed or interrupted key. `_run_stage` wipes the folder before computing
(`runner.py:172-174,211-220`). If the run then fails (raws on an unplugged drive), the adopted masters are gone while
register and stack stay "done" above them. On Windows, replacing a file the client has memory-mapped raises an error.
→ Write each job into `<key>.tmp-<run>` and rename it on success. A lost calibrate cache rebuilds `lights.f16` only.
The client never memory-maps store files.

**4. Important · §1.6 step 2 · The import check uses missing data and assumes no writes.**
- The old frames.json has no file size (`io/frames.py:32-49`; checked on Rosette), and 4 projects have dead raw paths.
  → Compare name, order, count and per-frame EXIF time and exposure.
- "No mkdir happens" is false: `Context.__init__` creates its folder (`core/stage.py:60-61`), and `effective_params`
  builds a Context (`runner.py:71`). Resolving a stage absent from the old work/ writes into a real project. → Map
  missing stages to a temporary folder, and read the manifest before resolving. Run the dry-run under a guard that
  raises on any write under the old root, on copies first.

**5. Important · §1.6 step 3 · Link mode reuses changed results.** A linked folder lives in a real project; if the
user re-runs it in v1, the v2 key silently points at another result. → No link mode on real projects. Otherwise store
the old run_id and `finished` in the `.link` file, and treat any mismatch as not done.

**6. Important · §1.5, §1.6 · An earlier learned star layer belongs to one stack.** Rosette, Horsehead, Flaming Star
and Iris set `learned_stars` to `D:/astro_gpu_diag/...npy` (checked). The layer is in the old stack's grid. An
exclusion, or a FlamingStar re-import in another order, changes the stack, and starless then subtracts a misaligned
layer. → Record the stack key beside the layer; "Full + earlier neural layer" runs only while that key matches.

**7. Important · §1.2 · Exclusions are keyed by file name.** Camera counters restart between cards and nights, so two
sessions can hold one name, and excluding one frame excludes both. → Key exclusions by (session, name).

**8. Minor · §1.3, §3 · Reproducibility, cleanup, races.** star_net saves only the CPU random state
(`star_net_lib.py:612,621`) but draws noise on the GPU (:642): a resumed training differs. Save the CUDA state too (no
version bump); the two GPUs also differ, as the device is not keyed. `clear_unused` refuses during a run, deletes
`<scratch>/<key>`, counts the `_ckpt` of interrupted keys; nothing clears calibrated caches (17–30 GB per key). Pass
run_id into `_run_stage` (`runner.py:202-204`); snapshot the whole definition in the plan; remove STOP in `start_run`;
create run.lock with O_EXCL; write `timings*.json` atomically (`:273-284`); rotate events.jsonl in `start_run`, since
rotating at run_start strands the follower's offset (`core/journal.py:79-96`).

## 2. Runner and runs
**9. Important · §3, P8 · One failure stops a 10 h queue, and GPUs are not pinned.** M31's reference path is dead
(`F:/backup_pc/.../m31_flat.jpg`, checked) and `reference.py:33-34` raises, so stop-at-first-failure halts the whole
project. `starless.py:433` hard-codes cuda:0, and star_net's "auto" takes `devices.aux` (`star_net.py:82-98`): the
cuda:1 queue also allocates on cuda:0. → After a failure, skip only the jobs downstream of it and report them. Start
each worker with `CUDA_VISIBLE_DEVICES=<n>` and device cuda:0. The import drops a missing reference. Run cheap
workflows first. Add CREATE_BREAKAWAY_FROM_JOB so `cli2 run --detach` survives an agent session.

**10. Minor · §3, P3 · Estimates.** "Within 10 % of Heart" is circular: Heart is the reference. Heart and Iris
differ by 2–10× from the units model (stack 34 vs 12.5 min, stars 32 vs 3.3 min). Accept on Iris within ±50 %.

## 3. The user's rules
**11. Important · P8, P14 · The old-versus-new QC compares the wrong things, and a QC tool writes into projects.**
Heart's, Horsehead's and Iris's old images, and Rosette's faithful one, come from `render_prep_learned` (checked in
`results/*/*_settings.json`). Judged against the new "Full" (model removal), they mix two changes. `judge.py:37-39`
maps every render through the old project's alignment, but the framing moves a few px. `bulletin.py --sheet` writes
into `<project>/results/` (`bulletin.py:7,162`). → Pair each old image with the workflow of the same preparation,
and remap each render with its own project's files. Never use `--sheet` on old projects. Port the QC tools to v2 in
their own package during P8; P14 at 2–3 h cannot.

**12. Important · P8 · Raw copies.** The 28.12.24 darks are 100 listed files out of the 2467 in
`G:/Astro/2024-12-28_divers_astro/957MSDCF` (checked). Copying the whole folder changes the key and brings in the darks
with the growing central light. Flaming Star's lights were one folder in project.json: merge the two G: folders back
into one, so the old order is kept, inventory → optics is adopted, and 3 h are saved. → Sessions keep file lists.

**13. Important · §6 · No non-regression of the v2 path.** Nothing shows that the shim and runner2 match v1, nor that
`render_layer` at position 2 matches v1's render. → In P7, run Full on `donnees/heart_a7s` with `projets/heart_a7s`'s
parameters, on the same GPU. Compare stack, denoise and render_prep bit for bit. In P12, check that
`render_layer(faithful, 2)` equals `RenderEngine.render(faithful)`.

**14. Minor · rules 10 and 14, P3 · Clutter, reporting, v1 forms.** Put P7's project under F:/astro_essais.
`list_projects` skips `_*` folders and folders without `project.astroproj`. Walkthroughs never open P8's projects
(autosave would edit them). `export/` keeps only the QC winner's faithful and artistic images. No package owns the
P8 live logbook or the French RAPPORT.md. `stack.exclude` shows as a text field in the v1 form
(`gui/widgets.py:373-378`): hide it there, and keep `select_frames` positional for `comfyui_astro/blocks.py:404`.

## 4. Work packages
**15. Important · §6 waves · Parallel agents share one working tree and both GPUs.** Rule 19 forbids worktrees, so
`commit -a` sweeps up other agents' files, and "keep the suite green" fails on half-written code. Tests use the GPU
while P8 holds both. → Commit by explicit path, serialised by the orchestrator. Run each package's own tests during a
wave and the full suite at its end. Run tests on the CPU during P8.

**16. Important · §6 · File ownership has gaps.** `client/model.py` (P9b) is needed by P10, P11 and P12, which run in
parallel. `locale/fr/*.json` and `widgets.py` (P9a) receive additions from every package. P4's stages must be
registered in `core/workflow.py` (P1). → P9b delivers the whole §5.2 surface, with a test that every member the views
use exists; P10–P12 never edit it. Each package writes its own `locale/fr/<package>.json`. P4 owns a `stages/v2.py`
registry.

**17. Important · P8 · Engine freeze.** A fix in `stages/`, `core/store`, `core/runner2` or `render/` during P8's 10 h
mixes code versions. → Tag P8's starting commit; only client changes until P8 ends.

**18. Important · Missing packages and weak acceptance.** Missing: performance budgets (point 20), French engine
texts (19), the QC tools port (11), the v1/v2 check (13). P2 also tests optional and EXTRA dependencies and link
staleness. P11 stops denoise after step 1000 (checkpoints every 1000, `denoise.py:213`) and kills a worker. P13
requires zero untranslated strings on its shots. P9b, P12 and P13 port about 5,000 lines: 2–3 h each is optimistic.

## 5. The client
**19. Important · §5.4 · Engine texts cannot be translated as designed.** Auto reasons are f-strings with numbers
(`stack.py:85-116`, `denoise.py:50`), so a dictionary never matches them. About 100 metric keys have no label or unit,
and many engine help texts have no French: the French screen will be mixed, against SPEC 5.8. → Write reasons as a
template plus arguments, formatted with a decimal comma. Add a metric catalogue and a translation package.

**20. Important · §5.3 · Post at 100 % and render latency are not designed.** A "crop of a cached full render"
re-renders Heart's whole 6384 × 4272 grid (checked) after every release at zoom ≥ 50 %, at 330 MB per layer. P8
gives up to 3 engine layers, but only 2 engines stay loaded, each holding all its layers in RAM
(`render/engine.py:76-116`; Heart's render_prep is 1.4 GB). Rendering on a GPU where a run is training can make that
run fail for lack of GPU memory. → Set budgets in P12 on Heart: draft ≤ 150 ms, 1/2 ≤ 1 s, 100 % ≤ 2 s. Render a
real region at 100 %, using the tone fit of the 1/4 render. Evict the least-recently-used engine. Render on a free GPU
or the CPU, with a fallback when GPU memory runs out. Check that the export, reduced, matches the preview.

**21. Important · §5.3 · The GUI thread scans disks.** Keys need `project_part`, which lists and stats every session
file (`core/project.py:331-340`). Plans stat 721–1268 frame paths on USB drives (`usable_as_input`). Both run after
every edit. → Compute them in the ApiWorker, cache them per session and per key, and refresh them on run events.

**22. Minor · Spec fidelity.** Raw stack without Color is camera RGB under a linked stretch, so it shows a colour
cast: normalise channels first. "Polynomial" names two different algorithms. An Output block is never shared (SPEC
5.1). Astrometry and Catalogs have no preview. Dropped silently, to list in §5.5: Memory limit, Glossary, history in
the export header, Keep against a threshold, SPEC 5.7. Warn when several sessions' flats merge into one master.

## 6. Risks the design underestimates
- cuda:1 (8 GB, drives the display) carries a P8 queue, star_net on Heart included, while walkthroughs render.
- Classic's polynomial will leave rings on M31 and M33 (open problem 7). Say so in the report instead of letting the
  judges find it.
- P13 inherits the integration debt of three parallel packages in a 750-line `app.py`, in 2 h. Everything the user
  judges depends on P14, which comes last, is small, and has tools not yet ported to v2.

## Orchestrator's decision (2026-10-31)

Triage rules (PILOT.md §4): the spec is frozen; only verified blocking or important points are applied; correct, do
not redesign. Spot-checked in the code: points 1 (`params.py:74-90`), 3 (`runner.py:172-174`), 4 (`stage.py:61`),
9 (`starless.py:433`). None of the 22 points reopens a product decision.

- **Applied as proposed**: 1, 2, 3, 4, 6, 7, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 10, 14.
- **5**: the strict option. No link mode on real projects: adoption copies only the outputs that later stages read
  (no rebuildable caches), after checking free space on F:.
- **8** (minor): only the cheap safety items (run_id passed in, run.lock with O_EXCL, atomic timings, events rotation
  and STOP removal in `start_run`, definition snapshot in the plan). The CUDA random state is left as it is: a resumed
  training that differs slightly is acceptable; noted.
- **18**: the missing work goes into existing packages where it fits; new packages only for the QC tools port
  (during P8) and the engine texts. Durations re-estimated.
- **22** (minor): normalise channels for the raw-stack preview only; distinct internal names for the two
  polynomials, GUI names unchanged; list the dropped items in §5.5 as ◇ instead of dropping them silently; the
  flats-merge warning is a health check.
- **Risks**: heavy P8 jobs go on the large GPU; walkthroughs render on the CPU during P8; the report states the
  expected rings of the Classic polynomial on M31 and M33; P13 gets more time.
- **Ownership**: the orchestrator owns JOURNAL.md, RAPPORT.md and every commit (explicit paths, one at a time).
