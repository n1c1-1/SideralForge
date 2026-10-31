# Logbook of the pilot run

Newest entries at the bottom. Times are local.

## 2026-10-31 evening: start

- Branch `pro-client` created; mockups alt1-alt3 and the GUI spec committed (c13804d).
- docs/pilot/PILOT.md written: goal, plan A-F, rules.
- Phase A launched: three agents, one each for the architecture map, the engine knowledge and the spec from the
  mockup.

## Phase A done
- ARCHITECTURE.md:
  - 17 fixed stages; the cache is keyed by variant name and random run_id, so no sharing between workflows;
  - the ComfyUI blocks already cache by settings hash (pattern to reuse);
  - one run per project; events.jsonl; --until exists;
  - fast tests: 84 passed.
- ENGINE_KNOWLEDGE.md:
  - rules: read-only real projects, nonreg 4 datasets + blind judges, 100 % checks, no patch stacking;
  - the raws of Rosette / Horsehead / Flaming Star / M31-2023 are now only under G:\Astro.
- spec/:
  - SPEC, BLOCKS, POST, ENGINE_CONTRACT;
  - 41 NEW settings, 6 NEW methods / stages;
  - missing in the engine: workflows as options, the settings-keyed store, sessions, Post composition, …
- Next: phase A2, the design of the engine contract and the client (design agent), then dispatch.

## Phase A2: design
- DESIGN.md:
  - v2 projects beside v1;
  - a content-addressed store with the existing stages run through an adapter;
  - 2 new stages (polynomial gradient, simple output);
  - the graph shows the real engine order;
  - import reuses old results (7 of 9 projects);
  - work packages P1-P15.
- Independent design review launched (DESIGN_REVIEW.md), as the user asked. Then: judge it, apply, dispatch.
- The user worried that reviews could overcorrect work already done. Answer: the earlier reviews judged the GUI; this
  one checks the technical plan (store keys, invalidation, reuse of old results, run order), which nobody had
  reviewed. Triage rules added to PILOT.md §4: spec frozen, only verified blocking/important points, correct without
  redesigning.
- Design review received: 22 points (1 blocking: the key dropped a value equal to the static default even when the
  auto value differs). Four points spot-checked in the code. Decision written at the end of DESIGN_REVIEW.md: 18
  applied, link mode refused in favour of copying (F: has 2.5 TB free), minor points reduced to the cheap safety
  items. The design agent is applying it to DESIGN.md.
- DESIGN.md revised (709 lines, "Changes after review" at the top). Two new packages: P16 engine texts, P17 QC tools
  port. Order: P1 → P2, P4, P9a → P3 → P5, P6, P16 → P7, P9b → P8 (background) with P17, P10, P11, P12 → P13 → P14
  → P15 optional. Engine freeze during P8 (tag `p8-engine`). Agents never commit; the orchestrator does.

## Phase B/C: implementation
- Wave 1: P1 (project v2 and chains) dispatched.
- P1 done and committed (f6de621):
  - Project2 / from_v1 / the chain table;
  - fast suite 167 passed (checked by the orchestrator);
  - from_v1 on the 9 real projects: nothing written;
  - noted deviations (none touches the spec): `next_workflow_id` so ids are never reused; the Raw stack exit has no
    reference stage; temporary stand-ins until P4's stages/v2.py; saving refuses a folder with a v1 project.json.
- An hourly heartbeat is scheduled, so the run restarts if it stalls.
- Wave 2 dispatched in parallel: P2 (store and keys), P4 (new stages, rendering), P9a (client shell).
- The user asked for frugality in agent models; the split is in PILOT.md §4, adjusted from experience.
- P2 done and committed (aa92fd1):
  - Keys / InputScan / Store / StoreProject; 18 tests;
  - 110 passed on store + project + core (checked by the orchestrator);
  - until stages/v2.py exists, every field counts as automatic (correct, less sharing); files keyed by name, size and
    mtime;
  - known minor gap: astrometry may look the target up by the project name while its key holds only the target name.
- P3 (runner v2, opus) dispatched early: it needs P1 and P2 only, and none of P4's files.
- P9a done and committed (23cd44a):
  - client shell; 17 tests; checked by the orchestrator, plus screenshots of Home (en) and Settings › Compute (fr):
    clean and faithful to the mockup;
  - mount points in app.py, held in a `READY` dict, for P9b–P13;
  - the mockup's French files merged into locale/fr/base.json;
  - polish for P13: "1 night(s)" / "3 workflow(s)" need real plural forms.
- Waiting: P4 (stages, rendering) and P3. P16 waits for P3, because both edit stages/stack.py.
- P4 done and committed (d892984):
  - stages/v2.py registry, BackgroundPoly, RenderSimple, region rendering, sliders, LayerService;
  - 101 passed on its tests plus store and workflow (checked by the orchestrator);
  - v1 render unchanged bit for bit (CPU and GPU);
  - budgets on Heart: 1/2 0.4 s, 100 % region 0.7–0.8 s, nebula draft 0.15–0.18 s;
  - star-slider draft 1.1 s (budget 150 ms, missed: v1 draws stars at full resolution). For P12: a preview-only
    lower-resolution star draft is allowed, as long as the release render and the export stay v1-exact.
- P6 (import of v1 projects, opus) dispatched early: it needs P1, P2 and P4 only. It includes a read-only dry-run on
  the 9 real projects; no real import before P8.
- P3 done and committed (45095a8):
  - runner2 (plans, shared jobs, tmp-then-rename, skip downstream, O_EXCL lock, detached worker with
    CREATE_BREAKAWAY_FROM_JOB), estimate;
  - 86 runner/core tests and the fast suite (261) pass (checked by the orchestrator);
  - Iris estimate: +98 % from Heart alone (Heart's stack and stars ran 5–10× slower), +8 % from the other 8 projects'
    timings. So P8 seeds a pilot ASTRO_GPU_HOME with `learn_from_v1` on the 9 projects;
  - the earlier star layer is tied by `{"stack_key": …}` in `inputs/<layer>.json` (passed on to P6).
- Wave 4 dispatched: P5 (API and cli2, sonnet) and P16 (engine texts, sonnet), beside P6 (opus).
- P6 done and committed (b7fbf49):
  - import_v1 with WriteGuard, copy only, EXIF frame check, star layer tied to its stack key;
  - 28 tests including the 9 real dry-runs, rerun by the orchestrator: all pass; F:/astro_projets still absent;
  - dry-run:
    - Heart, Iris, Rosette, Horsehead, Flaming Star adopt inventory → optics and recompute from stars (stars
      version 13 < 15), 3.3–3.5 GB to copy each;
    - M31 adopts to optics and recomputes reference (its reference now found on G:);
    - Wizard adopts to optics and recomputes from stars;
    - NGC 7822 and M33 recompute from calibrate (version 1 < 3);
  - proof of no writes: file lists before and after, all 9 unchanged;
  - F:/astro_essais/v2/heart_a7s created for the client packages.
- P17 (QC tools port, sonnet) dispatched early: it needs only committed code.
- P16 done and committed (77f4872), on sonnet:
  - 22 reason templates, 157 metrics, every help text in French;
  - stage diffs checked by the orchestrator: only how reasons are stored changes, the values are identical;
  - fast suite 305 passed;
  - sonnet was enough for this well-specified work: keep the split.
