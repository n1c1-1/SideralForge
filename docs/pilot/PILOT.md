# Pilot brief: from the mockup to the software

Started 2026-10-31 evening. The user is away. Their words: "go tu es autonome… tu utilises des agents, tu es
l'orchestrateur, tu dois préserver ta fenêtre de contexte pour garder le cap. À la fin je veux les projets prêts que je
puisse réévaluer et un GUI de qualité pro."

**Read this file first after any summary of the conversation.** Then read JOURNAL.md (what was done) and the documents
listed below.

## 1. The goal

1. **A pro-quality heavy client** that implements the mockup `mockup/alt3`.
   - The mockup is the specification. Its English GUI, French translation, Graph / Inspect / Post, sessions,
     settings and export are reproduced.
   - The client drives the real engine `astro_gpu`.
2. **The user's projects ready to re-evaluate in that client.**
   - Each project is re-run in a clean folder on F: (T9).
   - Each gets its workflows: at least Classic stack, Full, and Full + neural stars where the data allow.
   - Results are computed with the current engine, plus the changes below that passed QC.
3. **Nothing is lost.**
   - Real projects stay read-only: work on copies.
   - No deletion without the user's OK (recursive deletion is blocked anyway).
   - Every engine change is checked for non-regression before it is kept.

## 2. Reference documents (docs/)

| File | What |
|---|---|
| pilot/ARCHITECTURE.md | map of the engine and the existing GUI (written by an agent) |
| pilot/ENGINE_KNOWLEDGE.md | the user's rules, projects, data sets, QC procedure, pitfalls, open problems |
| spec/SPEC.md, BLOCKS.md, POST.md, ENGINE_CONTRACT.md | the specification generated from the mockup |
| ../mockup/alt3/TODO.md | what is deliberately left for a later version (mono / filters, crop, …) |
| pilot/JOURNAL.md | the live logbook of this run |

## 3. The plan, in phases

Each phase ends with a commit on branch `pro-client` and an entry in JOURNAL.md. Commits are by Nicolas with
Co-Authored-By; never push.

- **A. Groundwork**: architecture map, engine knowledge, spec (three agents). Then the design (DESIGN.md).
  - **The user asked (2026-10-31)**: when the design is written, an independent agent reviews it (documents only).
    The orchestrator then judges the review with distance, applies the changes that hold, and only then dispatches
    the work packages.
- **B. Engine contract** (what the client needs, before any algorithm change):
  - workflows per project with settings per workflow, and stage caching keyed by the settings up to the stage, so
    that a shared block is computed once;
  - run a list of stages / run up to a block, with progress events and cancel, in a separate process;
  - per-stage preview images and measurements in a form the client reads;
  - parameter descriptions with suggestion, reason and range;
  - render of a layer at given slider positions;
  - export.
- **C. The client**: a new package, built from the mockup's code, with the mockup's demo data replaced by the
  engine. Keep:
  - the i18n system;
  - Spec mode (◇ = not in the engine);
  - every screen.
- **D. Engine changes from the spec**, each behind its own method or option, each checked with non-regression and
  blind QC before it becomes the default:
  1. Gradient (background) before Color and Stars;
  2. Deconvolution (Richardson-Lucy with the measured PSF) before Denoise, on the half-stacks;
  3. Denoise: NL-Bayes (CPU) and pretrained + fine-tuned;
  4. Output stretch choice;
  5. Post filters and masks at full resolution, and export.

  Anything that fails its QC stays a non-default method and is reported.
- **E. Projects**: copy each real project's inputs into clean folders on F:, build its workflows, run, and check at
  100 %.
- **Checkpoint with the user (asked 2026-11-01 morning).** A review pack in French (docs/pilot/POINT_ETAPE.md):
  the finished projects' images beside the old ones, screenshots of every screen, and the decisions that are the
  user's. Their answers come before the image fixes are applied and before the blind QC.
- **F. Final QC.** The user's explicit request: at the END OF IMPLEMENTATION, an independent agent evaluates the
  result. The orchestrator judges that evaluation with distance and applies the changes that hold, before handing
  back. This is in addition to the design review done in A2. The checks are:
  - an independent GUI review on screenshots;
  - blind judges on the renders against the previous results;
  - a final report in French for the user, plain language, with images.

Order of priority if time runs short: B and C (the client on the real engine) > E (projects ready) > D (new methods).
A working client on the current algorithms is worth more than new methods without a client.

## 3b. Sunday plan (2026-11-01, from about 09:30; the user is away, "tu as la main, pas de questions bloquantes")

The user's review changed the course:
- **Target:** Siril users must find at least what they get from Siril + Photoshop: a calibrated, stacked,
  colour-corrected image, plus Photoshop-like post. Star processing and neural methods are extras.
- **No more long runs of the full projects** before the design is validated. Iterate on a bench.
- **P8 queues paused** at 09:30. Resume with `tools/projects/queue.py resume gpu0|gpu1`. M33 still to redo on
  cuda:0. The user decides in the evening whether to re-run all 9 projects with the final engine.
- **Engine unfrozen:** P8 no longer runs.

Three lanes run in parallel:
1. **Bench and Siril reference** (opus).
   - 5 cases, no crop:
     - Heart (A7S + TS61);
     - Iris (A7S + Newton 130/900);
     - M51 (Canon 6D + Newton 200, D:\astro_donnees_test);
     - M8 (ASI2600MC, the Siril tutorial set);
     - Seestar M31.
   - Express = 30 frames per case (about 10–15 min, with Siril). Base = 100 frames (about 40 min). Complete = all
     workflows, at night only.
   - Siril 1.4 basic: calibration, registration, stacking with rejection, background extraction, SPCC, autostretch.
     The recipe is documented, for the user to validate.
   - Measures on the linear stacks and the finals: background noise, SNR, FWHM, roundness, faint stars, trails,
     background flatness, star colours against Gaia. Plus side-by-side sheets.
   - Code in tools/bench/, data under F:/astro_essais/bench/, report in docs/pilot/BENCH.md (French).
2. **GUI evolutions** (the user's remarks of 09:00), in waves so that each file has one owner:
   - A: quick fixes (neutral filter defaults, final image in Output, duration for the right device, Astrometry
     view order); interactive charts (QtCharts: registration quality, x,y drift against time, x against y; colour;
     selection; zoom, CSV export); the common image viewer built standalone.
   - B: the viewer wired into Frames, Calibration, Stacking, Denoise, Gradient and Output; Affinity-like Post layers
     (a filter nested under its layer, "everything below" as a top-level row); breakpoints in the graph.
   - C: calibration frames review, exclusion and quality detection; Astrometry annotated image; Post CPU speed.
3. **Tests diet**: 4 levels (file under 1 min; fast suite under 3 min with xdist; walkthrough under 20 min; bench
   only when results change), and time budgets.

Plus the **engine fixes**: patches 1–3, store mtime ceil-to-even, manifest replace retry, catalogs reload, timings2
lock.

**No heavy run (the user's projects, a full bench) starts without a review pass done together with the user**
(their rule, 2026-11-01 18:20).

**PRIORITY 1 since 11:00 (the user, angry, rightly so): the assembled GUI.** They asked for it three times ("n'oublie
pas le GUI") and found standalone pieces on return.
- Rule: every GUI evolution ships wired into the client. Never "built standalone, wired later" when the user will
  look at it.
- Deliver one client.bat with everything, checked by the orchestrator screen by screen, plus a short French page
  "ce qui a changé, où le voir".
- The bench and engine work go on in the background and never delay the GUI.

Checkpoints with the user: Express verdict and quick fixes; Base verdict, viewer and charts; evening (Post, manual
control, the night plan).

## 4. Rules for the orchestrator

- **Reviews must not overcorrect** (the user's worry, 2026-10-31). When triaging any independent review:
  1. The spec (mockup alt3) and the product decisions taken with the user are frozen. A remark that reopens them
     is set aside.
  2. Only blocking or important points, verified in the code or the data, are applied.
  3. Correct, do not redesign: change lines, keep the plan.

- **Agent model: frugal but quality first** (the user, 2026-10-31: "qualité avant tout… mais sans gaspiller avec des
  agents lourds pour des tâches qui n'en valent pas la peine"):
  - **opus** where a mistake costs results or data: the runner (P3), the import of real projects (P6), the
    projects' runs (P8), integration (P13), the final independent evaluation;
  - **sonnet** for well-specified work: the API (P5), engine texts (P16), the end-to-end test (P7), the ports of
    mockup views (P9b, P10–P12), the QC tools port (P17), QC judging (P14);
  - **haiku** for simple look-ups and checks only, never for code;
  - give sonnet agents sharper briefs (exact files, exact acceptance);
  - this split is a starting point, not a rule (the user: "pas à graver dans le marbre… ajuste en fonction du retour
    d'expérience"). If a lighter agent's work needs heavy rework, move that kind of task up. If an opus agent's task
    turns out routine, move it down. Note each change in JOURNAL.md.
- **No Monitor tool in agents' work** (lesson of 2026-11-01): three agents waited 3 h on Monitor conditions that
  never came true. Every brief says: do not use Monitor, never wait more than 10 minutes, run long commands in the
  background and check them with quick commands. To detect a stall, look at the mtimes of the agent's own files: the
  task output files do not grow while an agent runs.
- **CPU only means `CUDA_VISIBLE_DEVICES=-1`.** On this Windows machine an empty value does NOT hide the GPUs:
  torch still sees 2 (checked on 2026-11-01).
- **Delegate** reading and coding to agents with precise briefs. Keep only conclusions here. Never read big files or
  transcripts into this context.
- **One owner per file at a time**: never two agents on the same files.
- **Check agents' work.** Run the tests and the walkthroughs, look at screenshots. An agent's report is not proof.
- **Two failed attempts on the same problem: stop that line**, write it down in JOURNAL.md, move on to the next item.
- **Test settings**: test scripts set ALT3_SETTINGS (or the client's equivalent) to a scratch file, and never write
  the user's settings.
- **Disk**: F: (T9) is healthy and fast (708 MB/s write, 368 MB/s read over USB 3 UASP). Projects and caches can go
  there. Real projects and raws are read-only.
- **Online catalogues** (VizieR, GAVO, SIMBAD, NED) may be down: use the fallbacks; do not stall on them.
- **Python**: C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe.
- **Plain language** in everything written for the user (French). No internal codes in their reports.

## 5. Where the work goes

- Code: branch `pro-client` of C:/Users/Nicolas/sandbox/astro.
- Client package: `astro_gpu/client/`, decided in phase A (see JOURNAL.md).
- Clean project folders: `F:/astro_projets/<project>/` (inputs copied or referenced read-only, results inside).
- Logbook: docs/pilot/JOURNAL.md. Final report: docs/pilot/RAPPORT.md, in French.
