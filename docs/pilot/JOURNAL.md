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
