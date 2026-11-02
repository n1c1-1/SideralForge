# Heavy client: functional specification

Written from the mockup `mockup/alt3` on 2026-10-31. The mockup's behaviour is the reference. Where the mockup is
silent or contradicts itself, this document says so under "Open points".

The specification has four files:
- SPEC.md (this file): the product, its screens and its rules.
- [BLOCKS.md](BLOCKS.md): each block, with its outputs, methods and settings.
- [POST.md](POST.md): layers, masks and filters.
- [ENGINE_CONTRACT.md](ENGINE_CONTRACT.md): what the client needs from the engine, checked against `astro_gpu`.

Out of scope for this version (TODO.md): mono cameras with filters (LRGB, SHO), dual-band extraction, the filter per
session, crop, hand-placed gradient samples, headless runs and queues, dedicated CMOS cameras, mosaics, comets, live
stacking, photometry.

## 1. Object model

| Object | What it is |
|---|---|
| Workspace | The folder that holds the projects. Set in Home and in Settings > Folders & disk. |
| Project | One target: its frames, sessions, workflows, the cache of computed stages, the Post composition and the history (Log). The mockup's file filter is `*.astroproj`. |
| Session | One night of acquisition (noon to noon): lights, exposure, ISO or gain, sensor temperature, filter, flats, darks, bias, rotation. Darks and bias come from the session or from the master library; flats always belong to their session. |
| Frames | All the input files: lights, darks, flats, bias, dark flats. Grouped into sessions automatically (3.4). A light can be excluded by hand, for the whole project. |
| Masters | Combined calibration frames. The master library keeps them between projects; a master is matched to a session by exposure, ISO, temperature (± 2 °C) and age (180 days). |
| Workflow | A complete chain, from Calibration to its Output: a name, options (which blocks and which methods) and its own settings at every step. There is no common trunk. |
| Block | One step as the user sees it: Frames, Calibration, Registration, Selection, Stacking, Astrometry, Gradient, Color, Deconvolution, Denoise, Stars, Output, Post. Denoise, Deconvolution and Stars are optional. |
| Stage | The engine's unit of computation and caching. A block runs one or more stages (BLOCKS.md). |
| Method | One way of computing a block: a name, where it runs (GPU, CPU, or either) and its own settings. Gradient, Deconvolution, Denoise and Stars have methods. |
| Setting | One parameter of a stage: a suggested value with its reason; a level (normal, expert, system); existing in the engine or NEW. Only the values that differ from the suggestion are stored. |
| Variant | A workflow copied from another with one change. It shares every block before the change, and is kept (it replaces the original) or discarded. |
| Layer | An item of the Post composition: a workflow layer (the output of a workflow) or a filter layer (on one layer or on everything below). Any layer may have a mask. See POST.md. |

Workflow options in the mockup (`PathDef.opts`; the engine keys proposed are in ENGINE_CONTRACT.md):

| Option | Values | Meaning |
|---|---|---|
| `debruitage`, `dn` | off / `appris` (Noise2Noise), `affine` (fine-tuned), `generique` (pretrained), `nlbayes` | Denoise block and its method |
| `deconv`, `deconv_m` | off / `psf` (Richardson-Lucy), `neural` | Deconvolution block and its method |
| `etoiles`, `appris` | `garder` (no Stars block) / `mesurer` (PSF model), with or without neural removal | Stars block and its method |
| `fond` | `halpha` / `simple` (polynomial) | Gradient method (the block is always there) |
| `end` | a block key, or none | the workflow leaves for Post at this block (5.5) |
| `variant_of` | a workflow id | the workflow this variant was made from |

Project-level decisions count for every workflow: the lights and calibration frames excluded by hand, the sessions,
and the stacking mode (all sessions together, or one stack per session).

## 2. Main window
- Top bar: the project name and target; the mode switch Graph | Inspect | Post; the machine (orange without a GPU);
  the button **Run · total time**. The window title is "astro_gpu · <project>".
- Status bar: "N workflow(s) · to run: t" or "all up to date"; the Spec mode note; the **Log** toggle.
- Post is the Inspect view of the Post block: Ctrl+3, or a double-click on the Post block, opens it.

## 3. Screens

### 3.1 Home
- Left: the Workspace path with Change…; New project… (primary), Import folder as new project…, Open project…;
  "N projects in the workspace".
- Right: Recent projects, a grid of cards: thumbnail, name, target, "N night(s) · integration · date", and the state
  ("All workflows up to date", "2 workflows · 1 outdated", "Never run").
- New project dialog: Name, Target ("optional: filled in by astrometry"), Folder (`<workspace>/…`).

### 3.2 Graph
- The workflows as a tree of blocks. Columns, left to right: Frames, Calibration, Registration, Selection, Stacking,
  Astrometry, Gradient, Color, Deconvolution, Denoise, Stars, Output, Post. The optional columns appear only when a
  workflow uses them. Two bands: "Pre-processing" (Calibration to Stacking), "Linear processing" (Astrometry to Stars).
- One lane per workflow. A block shared by several workflows is drawn once, between their lanes; the branch splits at
  the first block where they differ. Every Output is wired to the one Post block.
- A block shows its title, a one-line summary (BLOCKS.md), a state stripe (green, grey, orange) and a tooltip with
  the state. Blocks outside the current workflow are dimmed. Optional blocks show a × on hover. Stacking has an
  output port ● (5.5).
- The selected block shows its side inputs as dashed, labelled wires: Denoise ← Stacking "half-stacks"; Stars ←
  Astrometry "Gaia stars"; Color ← Astrometry "Gaia spectra"; Gradient (H-alpha) ← Astrometry "H-alpha survey".
- Toolbar "Add step for "<workflow>"": one card per optional block (Denoise, Deconvolution, Stars), saying "+time",
  "same duration" (under 30 s), "already there" or "after the output". Drag a card onto a lane (dashed ghosts show
  where it can go), or double-click to add it to the current workflow.
- A dashed "+ New workflow" lane opens the New workflow dialog: start from Classic stack, Classic + denoise or Full
  ("≈ duration" each), or from "Copy of "<workflow>"" with its settings. The name is filled in, unique.
- Context menu: Output: Rename, Duplicate, Delete workflow (confirmed). Optional block: Remove from the N workflows /
  from "X" only. Any block: Inspect. Stacking: Send the output to Post.
- "Show all" fits the whole graph; off, the zoom stays at 80 % or more and the view scrolls to the selected block.
- Click selects: the current workflow becomes one of the block's workflows, and the scope all of them. Double-click
  or Enter opens Inspect (Post for the Post block). Toasts report what happened.
- The block panel, on the right (390 px):
  - title (Output: the editable workflow name), the state, Remove for optional blocks;
  - "Changes apply to:" on a shared block (5.2); the variant row (Keep / Discard); the pending-change box;
  - Measurements (Inspect only); Method cards (what it does; CPU / GPU / "CPU or GPU"; duration on this machine, ≈
    when estimated; why it is unavailable); the block's own controls (Selection thresholds, Stacking sessions);
  - the settings by group, one line each: name (orange when changed), ◇ in Spec mode, editor, unit, ↺ (back to the
    suggestion). A choice gets a line of its own. "Advanced settings (N)" adds the expert settings and the group
    "System: speed and memory";
  - a click on a setting opens its details: a range bar (suggestion tick, value dot; log scale when max/min > 100),
    "Suggested: v · reason", the help, the tip, and "◇ Not in the engine yet" in Spec mode.

### 3.3 Inspect
- Strip: "◂ Graph" (Esc), a workflow selector, then Frames › the workflow's blocks › Post, each with a border in the
  colour of its state.
- Header: "<Block> · <workflow>", image tabs when there are several images, and **Compare with** another workflow
  that has this block: side by side, zoom and pan in sync, "same block" when shared. A new variant opens compared
  with its original.
- An Outdated block shows its previous result, tagged "previous result", with "Run to here · t". A Pending block
  shows "Not computed yet", "Run to here · t" (the tooltip lists the blocks) and "Preview on a region · ≈ t".
- Footer: Auto-stretch (linear blocks only; display only, the data stay linear; remembered), Histogram (RGB, log
  counts), Fit, 1:1, and the pixel readout "x y R G B ADU RA Dec".
- On the right, the same panel as in Graph, with Measurements.
- Selection shows the frames chart (transparency against FWHM) instead of an image: drag the thresholds; click a
  point to pick a frame, shown with its time, FWHM, transparency and state, and Keep / Exclude.

### 3.4 Frames (Inspect of the Frames block)
- Buttons: Import folder…, Add session…, Grouping…. Tabs: Sessions and Browse.
- Sessions: a table (Session, Night, Lights, Exposure, ISO / gain, Temp., Filter, Flats, Darks, Bias, FWHM, Kept)
  and how the frames were grouped.
- Health check (ok / warn / info): the library dark matched; a temperature more than 2 °C from the master dark (the
  dark is scaled); no bias (the camera black level is used); a rotated camera (its session uses its own flats); a
  session with FWHM more than 10 % worse than the best (its frames weigh less); disk needed against free space.
- With more than one session, **Stacking**: All sessions together, or One stack per session (5.2).
- Grouping dialog: night (noon to noon), exposure, ISO / gain, filter, sensor temperature (± 2 °C), binning, camera
  rotation; FITS OBJECT keyword (off by default).
- Browse: groups Lights, and Calibration (Darks, Flats, Bias, Dark flats); Grid, Viewer with filmstrip, or List; Sort
  by; Blink at 5 frames/s (Space); ← →; Keep / Exclude (X); Draw test area.
- Import folder: the files are read with a progress bar, then "Found: N lights, N darks and N flats…", then Add to
  project.

### 3.5 Post
Layers as in Photoshop, masks made of parts as in Lightroom, filter layers, continuous sliders: see POST.md. The Graph
panel of the Post block lists the layers with their state and source, and has "Open Post".

### 3.6 Settings (Edit > Settings…, Ctrl+,): application preferences, kept between sessions

| Page | Content |
|---|---|
| General | Language (English, Français), applied at once; Spec mode. |
| Folders & disk | Workspace, Cache (calibrated frames), Temporary files, Cache limit (GB), disk use bars, "Clear cache of closed projects". |
| Compute | Device (Automatic / the GPU by name / CPU only), CPU threads (0 = automatic), Memory limit (GB), Benchmark (7). |
| Catalogs & astrometry | Gaia DR3 local copy or online (VizieR); its folder; "Download Gaia DR3 extract (12 GB)…"; plate solver Internal (Gaia), ASTAP or astrometry.net (local); H-alpha survey cache. |
| File formats | Intermediate files FITS or XISF; 32-bit float or 16-bit integer; compression None, Rice (FITS) or Gzip; keep the calibrated and registered frames. |
| Master library | Folder; temperature tolerance (°C); maximum age of a dark (days); the masters (type, exposure, ISO, sensor, frames, date, used here); "Build masters from a folder…". |
| Display | Auto-stretch linear images by default; angular units (arcsec or px). |

### 3.7 Export (File > Export image…, Ctrl+E, or Export… in Post)
- Format: TIFF 16-bit, TIFF 32-bit float, FITS 32-bit float, XISF 32-bit float, PNG 16-bit, JPEG. Color profile:
  sRGB, Adobe RGB (1998), Display P3, Linear sRGB. Size: Full resolution, Web (2048 px), Web (4096 px). File:
  `<workspace>/<project>/export/<project>.tif`.
- Include: WCS in the header (on), processing history in the header (on), each layer as a separate file (off), the
  masks (off), each workflow's output, linear, as FITS (off). A progress bar, then Close.
- File > Export project as script… writes the project as a reproducible recipe.

### 3.8 Log (View > Log, Ctrl+L, or the Log toggle)
A bottom panel, 200 px high. Left: "Time | Event", newest first (project opened, folder imported, session added,
each change with the workflows it went to, Replace, variant created, block added or removed, workflow created or
deleted, layer sent to Post, "Run: N stages · t"). Right: the console of the last run. "Export log…".

## 4. Menus and shortcuts

| Menu | Items (shortcut) |
|---|---|
| File | Home (Ctrl+H), New project… (Ctrl+N), Open project… (Ctrl+O), Recent projects ▸, Save (Ctrl+S), Save as…, Import folder… (Ctrl+I), Export image… (Ctrl+E), Export project as script…, Close project (Ctrl+W), Quit (Ctrl+Q) |
| Edit | Undo (Ctrl+Z), Redo (Ctrl+Y), Settings… (Ctrl+,) |
| View | Graph (Ctrl+1), Inspect (Ctrl+2), Post (Ctrl+3), Log (Ctrl+L), Spec mode (check) |
| Run | Run all (F5), Run to selected block (Shift+F5) |
| Demo (mockup only) | New project / Existing project (Heart); Add a second night; Machine: GPU / CPU only |
| Help | About (F1), Glossary |

Other keys: Graph: Enter opens the selected block. Inspect: Esc goes back to the Graph. Browse: ← →, X (keep /
exclude), Space (blink). Post: Ctrl+Z / Ctrl+Y, C (before / after), M (show mask), F (fit), 1 (100 %), X (paint /
erase), [ ] (brush size), Ctrl+J (duplicate).

## 5. Rules

### 5.1 A shared block is computed once
- Each stage of a workflow has a **stage key**: the stage name (a method may have its own stage, such as
  `denoise_cl`); the workflow options that change this stage or any stage it depends on; the stored settings of the
  stage and of every stage it depends on; the project-level decisions.
- Two workflows with the same key for a stage share its result: computed and stored once.
- In the graph, a block is one node for two workflows as long as their stage keys are equal for it and for every
  block before it. Selection is keyed by its thresholds, Gradient by its method; Output is never shared. After a
  split, a block may have the same key in two branches (Denoise depends only on the stack): it is drawn in each branch
  but computed once.
- A result stays in the cache when settings change: going back to earlier settings (Undo, or by hand) makes the block
  Up to date again without running anything.

### 5.2 Changing a computed block: Replace or Try as variant
- Scope: a change goes to every workflow through the selected block. A shared block offers "Changes apply to: all N
  workflows / "X" only" (a list beyond 3 workflows). A change for one workflow only splits the branch at that block.
- When nothing computed would be lost, the change is applied at once and logged.
- When a stage at or after the changed one is computed in a scope workflow, a box offers:
  - **Replace · t**: the change goes to the scope; their results downstream become Outdated; t is what runs again;
    the tooltip names the workflows.
  - **Try as variant · t** (primary): a copy of the current workflow with the change, named "<workflow> · <change>",
    sharing everything up to this block. It is selected and shown side by side with its original in Inspect.
  - **Cancel**.
- A variant shows "Variant of "X"" with **Keep** (it replaces X and takes its name) and **Discard**.
- Settings, method cards and Selection thresholds all follow this rule. ↺, or the suggested value typed in, removes
  the stored value.
- Frames excluded by hand (Browse or chart) apply to the whole project without asking.
- Stacking "One stack per session" makes one variant of the current workflow per session ("<workflow> · <session>"),
  to compare the nights.

### 5.3 Optional blocks and their dependencies
- Adding Stars also adds Denoise ("Denoise added: the star measurement needs it"). "PSF + neural removal" needs Stars.
- Denoise cannot be removed while Stars is there. Removing Stars also removes the neural removal and turns an H-alpha
  Gradient into the polynomial one (each said in a toast).
- Gradient cannot be removed; only its method changes. Deconvolution is independent: after Color, before Denoise.
- No block can be added after an early output (the toolbar says "after the output").

### 5.4 Run and Run to here
- **Run** (F5, or the Run button): every pending stage of every workflow, each distinct stage key once. Disabled
  until frames are imported ("No frames yet") and when nothing is pending.
- **Run to here** (Shift+F5, or the Inspect button): the selected block's stages in the current workflow, and every
  stage they need that is not computed (for Stars, the starless image too).
- The Run dialog lists one row per block, in graph order: block, the workflows sharing it, duration, progress bar;
  "Already up to date: N blocks"; "Duration: t · disk: N GB"; Run, then Close. The mockup allows Cancel only before
  the start; the client needs a stop that can resume later.
- At the end: each computed stage key is marked done for every workflow that has it; a workflow whose every stage is
  done becomes *rendered*, and its layer appears in Post; a Log line is written.
- **Preview on a region** (Inspect, Pending block): runs the blocks on a region drawn on a frame, at full resolution.
  A button only in the mockup: to build.

### 5.5 Outputs pulled to Post (Stacking only)
- Drag the ● port of Stacking onto the Post block (highlighted), or use "Send the output to Post".
- This makes a workflow "Raw stack" (unique name): the source workflow's blocks and settings up to Stacking, then its
  own Output (simple output preparation); its Output says "from Stacking".
- Its layer joins Post once computed. The sliders that need later blocks are greyed: "this layer leaves the graph
  before the Stars block" (or the Denoise block).
- Only Stacking has the port: before Gradient the image still carries the sky gradient, and Astrometry does not
  change the image.

### 5.6 The machine, with or without a GPU
- The device comes from Settings > Compute (Automatic / GPU / CPU only); the mockup also switches it from the Demo
  menu. The top bar names the machine, in orange without a GPU.
- Without a GPU: durations become CPU estimates (7); a GPU-only method is greyed, "Needs an NVIDIA GPU (≈ t on the
  CPU)", or, when computed earlier, "Computed earlier on a GPU; running it again here takes ≈ t"; a Denoise block
  that is added, or a new workflow with Denoise, uses "Pretrained + fine-tuned".

### 5.7 Spec mode
- View > Spec mode, or Settings > General; the choice is kept.
- It marks ◇ after the name of each NEW setting and on each NEW method card, adds "◇ Not in the engine yet." to the
  details, and shows "Spec mode · ◇ not in the engine yet · demo data" in the status bar.
- The ◇ items are the list of engine work. Their values are kept and change the stage keys like any other setting,
  so changing one marks its block to run again.

### 5.8 Languages
- English is the source language. Every displayed text is written in English and goes through `_()` when shown,
  never at import time. French: `locale/fr/*.json`, dictionaries {English: French}. A change of language rebuilds
  the menus, Home and the project screen at once.
- The engine's labels, help and reasons are English keys, translated the same way. Names typed by the user are never
  translated; names the client builds ("<layer> · stars", "(copy)") are translated piece by piece.
- Style (GLOSSARY.md): names a Siril or PixInsight user recognises; one name per object; a neutral tone (no "you",
  no "tu"); numbers with their unit; a decimal comma in French.

### 5.9 Undo
- Graph: Edit > Undo / Redo (Ctrl+Z / Ctrl+Y) restores the workflows (names, options, settings, exits, variants)
  and the stacking mode. Computed results are not removed (5.1).
- Post has its own undo stack (POST.md).

## 6. State of a block

| State | Stage rule | Shown as |
|---|---|---|
| Up to date | its key is in the cache | green stripe, ✓ |
| Pending | not in the cache, never computed for this workflow | grey stripe; the duration |
| Outdated | not in the cache, but computed earlier for this workflow with other settings | orange stripe, "outdated · t"; Inspect shows the previous result |

- A block is Outdated if any of its stages is Outdated in any of its workflows; otherwise Pending if any is Pending;
  otherwise Up to date. Selection follows Registration. Frames is Up to date once frames are imported. Post is Up to
  date when every workflow is rendered.
- The Post block lists each layer as Up to date, Outdated (rendered once, now out of date) or Pending.

## 7. Time estimates
- Each stage has a duration on the reference machine: the Heart project (721 frames of 12 Mpx, RTX 4060 Ti). NEW
  stages are estimated: `deconv` 150 s, `denoise_ft` 180 s, `denoise_ai` 45 s, `denoise_cl` 300 s, `fond_simple`
  60 s, `rendu_simple` 30 s.
- A block's duration is the sum of its stages in that workflow (Stars includes the starless image; Astrometry the
  catalogues, and the H-alpha survey when used).
- Without a GPU, a stage takes this many times longer (estimates, to be measured on the engine): calibrate 1.3,
  register 1.5, stack 5, color 1.5, denoise 45, denoise_ft and denoise_ai 8, deconv 6, optics 3, stars 8, star_net
  60, starless 4, background 2, render_prep 10, rendu_simple 2; inventory, astrometry, catalogs, surveys,
  fond_simple and denoise_cl 1; any other stage 2.
- A value shown with "≈" is an estimate (no GPU, or a NEW method). Format: "≤ 1 min", "N min", "H h MM".

| Where | Duration shown |
|---|---|
| Run button, Run dialog | the distinct pending stage keys (and disk: the sum of the stage sizes) |
| Run to here | the pending stages that block needs |
| Replace / Try as variant | what the scope workflows / the variant alone will compute after the change |
| Toolbar card | the difference with and without the block ("same duration" under 30 s) |
| New workflow | the full cost from scratch |
| Toast | "+ t to run" when a change adds 30 s or more |
| Preview on a region | ≈ max(t / 12, 30 s) |

- Benchmark (Settings > Compute) measures this machine on a small data set and stores its ratios against the
  reference machine ("GPU 1.00× and CPU 0.92×"). Until it is run, the reference durations are used.

## 8. Open points found in the mockup
1. TODO.md puts Gradient before Color and Stars, from the stack and the catalogue, and the mockup's stage
   dependencies do so too. Yet choosing the H-alpha method still adds Stars ("the anchored background is computed on
   the starless image").
2. A Gradient method card changes the method directly; it does not offer Replace or Try as variant like the others.
3. Hand exclusions are part of every stage key, Calibration included: excluding one light invalidates Calibration as
   well as Stacking.
4. Deconvolution has no image and no measurements in Inspect yet (BLOCKS.md).
5. The mockup's Run dialog cannot be stopped once started; the engine can stop and resume.
