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
- P5 done and committed (9aa92dc), on sonnet:
  - api.py and cli2; 31 tests; loading it loads no torch (checked);
  - import_v1 and metric_info wired, no stubs left;
  - additions: warm/refresh caches, with_change / apply_change / throws_away for the variant choice.
- Wave 5 dispatched (sonnet): P9b (graph, panel, model; screenshots on F:/astro_essais/v2/heart_a7s) and P7 (Full and
  Classic stack via the API on heart_a7s under F:/astro_essais/p7/, bit-for-bit against v1). P17 still running.
- 03:34 heartbeat:
  - no file written by P9b since 01:39 or by P17 since 01:33;
  - P7's run ended at 02:29 ("12 checks recorded, 0 required failed"; Full chain v1 76 min, v2 83 min);
  - the agents seem to be in sleep loops. Status messages sent to all three. If they are still silent at the next
    heartbeat, stop them and relaunch with sharper briefs.
- 04:34 heartbeat: still silent, and the messages were not picked up.
  - The cause: they were waiting on Monitor conditions that never came true; the `sleep` processes were Monitor's
    loops. Not a model problem.
  - All three stopped (about 3 h lost) and relaunched once to finish from their files on disk, with a ban on Monitor
    and on waits over 10 minutes: P9b on opus (P10–P12 depend on its model), P7 and P17 on sonnet (little left).
  - The rule is added to PILOT.md §4.
- P17 done and committed (df7d359), on sonnet:
  - tools/qc/qc2.py: pairs, pair-sheets, sheets, bulletin, nonreg, all with `--out`;
  - 44 tests on the CPU (rerun by the orchestrator);
  - pairing: Heart → "Full + neural stars"; Iris, Horsehead, Flaming Star and Rosette faithful → "Full + earlier
    neural layer"; Rosette artistic, NGC 7822 and M33 → "Full";
  - real check on heart_a7s: aligned sheets, nonreg without differences.
  - **To do before P8** (tests on the CPU): 3 runner2 tests expect a visible GPU, and 2 comfy test files fail at
    collection with an empty CUDA_VISIBLE_DEVICES. Make them skip without a GPU.
- P9b done and committed (202872f), resumed on opus:
  - ProjectModel with the whole §5.2 surface, the graph and the panel; 54 client tests pass (checked);
  - the orchestrator looked at graph_en and panel_denoise_fr. Four defects sent back to the same agent for a
    follow-up:
    - the default view clips Frames on the left and "Post-traitement" on the right;
    - French block titles overflow;
    - the Classic branches look as if they come out of Denoise;
    - panel editors are misaligned when only some rows have a unit.
- P9b polish committed (9fe07d0):
  - all four defects fixed, checked on graph_fr: the whole chain is visible, Classic stack leaves from Color, and
    the editors are aligned;
  - left for P13: long workflow names are elided in their blocks ("Classic + deno…"); check the tooltips.
- P11 done and committed, on sonnet:
  - RunController: plan dialog, progress bars over the graph, Stop, reattach, recap, status widget;
  - 21 fast tests pass (rerun) and 2 slow ones;
  - polish for P13:
    - check the plan's durations with the GPU visible (the shot, taken without a GPU, shows Débruitage 5 h 50 and
      neural stars 41 h on heart_a7s);
    - when all workflows share a block, say "les 4 workflows" instead of an elided list;
    - about 30 common engine log lines are translated, the others stay English.
- P7 done and committed (ca34ffa), resumed on sonnet. **Verdict: v2 matches v1.**
  - Stack and denoise are identical bit for bit in the full chain. Every stage is identical when v2 recomputes it from
    v1's own inputs. render_layer equals RenderEngine.
  - Shared stages run once with the same key; adopted v1 keys equal v2's.
  - The full chain differs after astrometry (WCS within 1e-13 px) because sep.extract is not deterministic (v1
    too): final pictures at PSNR 34–36 dB, median difference 2–3 levels of 255.
  - Defects found (engine, for after P8):
    - **violet H-alpha in the Classic exits** (seen by the orchestrator on sheet.png: Full is red, Classic violet);
    - the sky slider is dead at positions 0–3 (fit_tone_curve);
    - a green sky cast in the Raw stack exit;
    - optics.measure_lateral is not reproducible (sep).
  - Estimates: 65.6 min announced against 83.6 measured; per stage, off by up to 5×.
- Tests made CPU-safe: the GPU-pinning tests are skipped and the ComfyUI files ignored when CUDA_VISIBLE_DEVICES is
  empty; the fast suite passes on the CPU (448). Committed (06757c6).
- **Engine freeze: tag `p8-engine` on 06757c6.**
- P8 (opus) dispatched: import the 9 projects to F:/astro_projets/, add workflows, set up two detached GPU queues,
  write tools/projects/status.py. P10 and P12 told to work on the CPU.
- A diagnosis agent (opus) prepares fixes for the three image defects as patches (docs/pilot/patches/,
  FIXES_AFTER_P8.md), without editing the engine. They are applied after P8; only the cheap last stages recompute.
- P10 done and committed, on sonnet:
  - Inspect (Steps, TuneView, histogram, pixel readout with RA/Dec, selection chart) and Frames (sessions, checks,
    grouping, Browse);
  - 25 tests on the CPU (rerun);
  - the orchestrator looked at inspect_stack_hist_readout_fr and frames_sessions_fr: faithful to the mockup;
  - polish for P13:
    - "Pose empilée 0,17 h" should read "10 min", and "Pixels rejetés 1,821 %" should read "1,8 %" (metric
      formats);
    - the stack preview has a green cast under the linked auto-stretch;
    - "1 workflow(s)" in the status bar;
  - model fixes for P13:
    - `model.p` is missing from SURFACE;
    - set_excluded and session edits do not mark the model dirty (P10 calls `_touch()`);
    - method cards call api.estimate on the GUI thread (64 manifest reads per walk).
  - Under machine load, two P11 tests failed once in P10's full-suite run: check them in P13.
- P8 set up and committed (bee6996), on opus:
  - the 9 projects imported into F:/astro_projets/ with no-write proofs OK; each has 4 workflows (Classic stack,
    Classic + denoise, Full, and Full + neural stars or Full + earlier neural layer);
  - raws: 84 GB copied to `_raw/` (frame files only; the 28.12.24 darks limited to the 100 listed);
  - two detached queues (gpu0 4060 Ti 16 GB, gpu1 3070 8 GB), about 4.5–5 h each: the Classic workflows of every
    project first, then the rest. Status: `tools/projects/status.py`;
  - at 06:23, the Classic workflows of 7 projects were done, and NGC 7822 and M33 were calibrating.
  - Engine bugs noted, for after P8:
    - store.py:80 keys a file by its mtime in whole seconds, but exFAT rounds to 2 s, so a copy NTFS → F: changes
      the key. Worked around by copying first; fix: compare on a 2-second grid;
    - timings2.json is updated by both workers without a lock;
    - `.gitignore` `projects/` hides tools/projects/ (`git add -f`).
  - **Tests wrote into the user's ~/.astro_gpu** (projects.json lists pytest projects; timings changed at 06:12).
    A sonnet agent is fixing the isolation (tests only, after a backup) and removing only the entries the tests
    added.
- P12 done and committed, on sonnet:
  - Post (layers, clipped filters, masks from parts, brush) and Export, renders on the worker;
  - 48 fast tests on the CPU (rerun), and 2 slow ones;
  - `render_layer(faithful, 2)` equals the engine bit for bit;
  - the orchestrator looked at post_mask_fr: Photoshop-like and faithful to the mockup;
  - timings on Heart:

    | Render | GPU | CPU |
    |---|---|---|
    | nebula draft | 0.32 s | 2.4 s |
    | star draft | 1.07 s | 4.9 s |
    | 1/2 on release | 1.1 s | 11 s |
    | 100 % region | 0.5–1.8 s | 2–11 s |

  - **For after P8** (render/):
    - a star-slider draft at reduced resolution;
    - CPU speed of Post (the user wants it usable without a GPU);
    - the export at reduced size differs from the 1/2 preview by 16 levels at p99.9, and the engine alone by 19.5.
- **Lesson: `CUDA_VISIBLE_DEVICES=""` does not hide the GPUs on Windows; `-1` does (checked).** Earlier "CPU" test
  runs may have used the GPUs during P8. The rule is fixed in PILOT.md and DESIGN.md, and passed to the isolation
  agent for conftest and needs_gpu.
- Test isolation fixed and committed (1349676):
  - conftest sets ASTRO_GPU_HOME to a private folder before any import, refuses writes under ~/.astro_gpu, and
    fails the session on a change there; 4 tests in test_state_isolation.py;
  - the leaks came from older tests (test_core, test_star_net, test_runner2 and a module fixture in
    test_simple_stages), not from the new client tests;
  - **the user's files cleaned, with a loss to report:**
    - projects.json: the test entries had pushed all of the user's own entries past the 30-entry cap. The 8 that
      still exist were restored from bak_20261028; entries added after 10-28 cannot be recovered;
    - timings.json: the toy stages were removed. The real history of `stack` and `inventory` had already been
      pushed out of its 20-entry window, so it is lost;
    - backups in the scratchpad (astro_gpu_home_backup_20261101_*).
- Image defect fixes prepared (docs/pilot/FIXES_AFTER_P8.md, patches 1–3; nothing applied; only render_simple's
  version is bumped):
  1. violet H-alpha in Classic: the simple output lacked Full's `emission_anchor`. After the patch the nebula is
     **magenta**, no longer violet but still not Full's red. A second, targeted attempt has been asked for (where
     does the hue depart: matrix, sky or stretch);
  2. dead sky slider: a floor at strength 1 in fit_tone_curve. The slider now moves the dark level too; position 2
     equals v1 bit for bit. Accepted;
  3. green raw-stack sky: a constant per channel cannot remove a coloured gradient; a smooth sky per channel is now
     subtracted. Accepted (seen on sheet_after: grey sky, red nebula).
  - Incident: before the warning, this agent ran tests from a scratch checkout with the old conftest and an empty
    CUDA_VISIBLE_DEVICES. It wrote toy timings into ~/.astro_gpu again (removed by the orchestrator with
    clean_state.py --apply, after a backup), and ran v2_smoke on cuda:0 at 06:50–06:53 during P8, with no P8
    failure. P7's sheet.png was overwritten (test area).
- Defect 1, second attempt: **the measured emission colour is pink-magenta** (OKLab hue −27° to −33°, in Full's
  linear data too). Full's red (+15°) comes from two render conventions, `ha_boost` (+16°) and a pull towards 33°
  (+27°).
  - With patch 1, Classic shows the measured colour (−31° against Full's −28° without the conventions).
  - Porting the conventions to Classic: the hue pull alone is clean (−0.2°); ha_boost browns Classic stack's sky.
  - **Decision:** apply patch 1 as it is (measured colour). Whether Classic should carry the "red H-alpha" convention
    is the user's call, in the report, with the safe option named (the hue pull alone).

- 07:34 heartbeat:
  - **M33 failed at stack on gpu1** (cuda:1, the 3070 that drives the display): "CUDA error: unspecified launch
    failure". 3 jobs done, 7 skipped downstream; the queue moved on (Rosette pass 2 runs fine on gpu1);
  - likely a display-card reset (TDR), or a test that touched cuda:1 while GPUs were not hidden;
  - **Action:** when queue gpu0 ends, run `tools/projects/queue.py start gpu0b --device cuda:0 M33` (done jobs are
    kept). Do not retry it on cuda:1.
  - NGC 7822 denoise at 80 % on gpu0, about 3.7 h left; P13 active.

- 08:20 the user asked for a checkpoint with their manual review. A review pack (POINT_ETAPE.md, sonnet, CPU,
  read-only, images under F:/astro_projets/_review/) is being prepared. The image fixes and the QC wait for their
  answers.

- POINT_ETAPE.md is ready: sheets for the 4 finished projects, the Classic stack of 8 projects, 11 French screens,
  and the decisions (Classic colour, the set of 4 workflows, the order of what remains). Updated at 9:00 to say the
  application is assembled.
- P13 done, on opus:
  - everything wired (all READY flags on); `client/previous.py` is the v1 import dialog; all 8 polish items done;
  - the walkthrough: 87 screens EN+FR, no warning, no English left in French;
  - fast suite 518 on the CPU, client 154.
  - Engine bugs for the queue:
    - runner2 replaces a manifest with no retry, so a job fails if the client reads that file at that instant
      (Windows). The client works around it by waiting 400 ms of quiet;
    - the catalogs stage reloads a compressed array inside its loop: 4–5 min of CPU per run, and Astrometry is
      estimated at 2 min against 7 measured.
- M33: rather than waiting for gpu0 (about 11:30), retry it on gpu1 when M31 ends (about 9:15) with
  `queue.py resume gpu1`. The 06:35 failure may have come from tests that used cuda:1 while GPUs were not hidden.
  If it fails again at stack, use gpu0b after gpu0.

### Queue for the end of P8 (engine unfrozen)
1. Apply patches 1–3, run the tests and the 4-dataset non-regression, recompute only render_simple (Classic and Raw
   stack), and re-check the sky slider in Post.
2. store.py file id: exFAT rounds mtimes UP to an even second, so an NTFS → F: copy changes the key. Fix by rounding
   UP to the even second (ceil), which leaves every key already computed on F: unchanged (their mtimes are already
   even). Test that P8's store keys do not move.
3. timings2.json: a lock or an atomic merge between workers.
4. Post speed on the CPU (1/2 render 11 s) and a reduced-resolution star draft: render/ work, if time allows (P15).
5. Then P14: QC of the projects with the P17 tools; then phase F (independent evaluation) and RAPPORT.md.
- P13 (integration, polish, walkthrough, opus) dispatched, CPU only.
  - It carries the 8 polish items gathered from P9a–P12. One explicit exception to the freeze: astro_gpu/texts.py,
    for display formatting only.
  - The walkthrough covers a project created from scratch on the heart_a7s raws (F:/astro_essais/p13/) and the
    import of a v1 copy.
- Wave 6 (client views) dispatched on sonnet, each on its own copy of the test project (heart_a7s_p10/p11/p12): P10
  Inspect and Frames, P11 Run, P12 Post and Export. P8 waits for P7's verdict.
