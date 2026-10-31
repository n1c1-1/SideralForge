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

- **A. Groundwork**: architecture map, engine knowledge, spec (three agents). Then decide the client architecture.
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
- **F. Final QC**:
  - an independent GUI review on screenshots;
  - blind judges on the renders against the previous results;
  - a final report in French for the user, plain language, with images.

Order of priority if time runs short: B and C (the client on the real engine) > E (projects ready) > D (new methods).
A working client on the current algorithms is worth more than new methods without a client.

## 4. Rules for the orchestrator

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
