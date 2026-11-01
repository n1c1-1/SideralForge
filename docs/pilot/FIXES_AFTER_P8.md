# Fixes after P8: three image defects found by P7

Prepared 2026-11-01 on the frozen engine (tag `p8-engine`, 06757c6). **Nothing is applied.** The engine was not
edited: the patches were made and tested in a scratch copy of HEAD. The files they touch are unchanged up to
1349676.

- Patches, to apply from the repository root with `git apply`, **in this order**:
  1. `docs/pilot/patches/1-classic-emission-anchor.patch`: violet H-alpha in the Classic workflows.
  2. `docs/pilot/patches/2-sky-slider-dark-level.patch`: the Post sky slider dead over part of its range.
  3. `docs/pilot/patches/3-raw-stack-grey-sky.patch`: green sky in the Raw stack exit.
- Patches 1 and 2 each apply to HEAD on their own. Patch 3 is based on patch 1 (both touch `render_simple.py` and its
  version line). If patch 1 is dropped, patch 3 needs only `version = "2"` set by hand.
- All the evidence images are in `F:/astro_essais/fixes/`. The overview is `sheet_before.png` against
  `sheet_after.png`. `sheet_before.png` is rebuilt from P7's untouched v1 results, because P7's own `sheet.png` was
  overwritten (see "Incident").
- The stage outputs computed from the P7 store are in `F:/astro_essais/fixes/work/` (about 2 GB). `fix3_raw` there is
  a superseded first version; `fix3_raw_b` is the final one.
- The data used, read only: the P7 store of heart_a7s, and the v1 test projects in `F:/astro_essais/projets/`.

## 1. Violet H-alpha in the Classic workflows

**Cause.**
- `render_simple.py:73` uses the colour stage's matrix `M.npy` as it is.
- That matrix is fitted on the stars. The stars only fix it on the plane of star colours. The direction off that
  plane, where line emission lies, is fixed by a single anchor: the HII spectrum as the reconstructed camera response
  predicts it (`color.py:129-156`).
  - On heart_a7s that prediction is camera R 1 : G 0.40 : B 0.12, and M renders it red: (0.57, 0.00, 0.17).
  - The emission the camera really records is 1 : 0.35 : 0.19.
  - The matrix has large coefficients (−1.96 on camera blue in display red, +3.02 in display blue), so the real
    emission comes out as (0.35, −0.10, 0.42). That is more blue than red: violet.
- Full does not have this problem because `render_prep.py:150` calls `emission_anchor` (`render_prep.py:421-466`).
  - It measures the emission's colour on the starless nebula.
  - It then changes M only across the plane of star colours (a rank-one update), so that the measured emission gets
    the colour M gives the HII spectrum. The star colours do not move.
- The simple output has no such step.
- Gamut mapping is not the cause. M's rendering of the emission is only slightly out of gamut, and its own hue is
  violet. Bringing it into gamut by adding white keeps that hue: (0.45, 0, 0.52), still violet.

**Fix: the same emission anchor as Full, in `render_simple`.**
- The matrix update is moved out of `emission_anchor` into `anchor_matrix` (`render_prep.py`) and shared.
  `emission_anchor` returns the same matrices, bit for bit. Checked on P7 heart_a7s, iris, rosette, seestar_m31 and
  ngc891, with weights 0, 0.07 and 1.
- The star colours come from the stars the colour stage itself fitted on. Its selection is moved into
  `color.colour_stars`; the code is identical, only moved.
- **The emission is measured differently from Full, because the stars are still in the image.**
  - On the stack, Full's band-pass sees the stars (G/R 1.06, a star colour). The weight is then 0 and nothing would be
    corrected.
  - I also tried median filters of 2-5 FWHM before the band-pass, and a regression on the off-plane signal. Both
    failed on the Heart: G/R between 0.46 and 1.4, or negative.
  - What works: in blocks of 128 px (the same area as render_prep's blocks), the median of each channel of the stack
    minus its sky. Then, as render_prep does, the reddest fifth of the brightest fifth of the blocks. The medians are
    not moved by the stars.
- **The sky for that measurement is a polynomial fitted on the raw stack in camera colours** (background_poly's fit,
  default settings). A camera colour has to be measured in camera colours.
  - A first version measured on each workflow's own image and display-space sky. It gave G/R 0.31 for Classic and 0.42
    for Classic + denoise, because the two skies differ by about 3 ADU and the emission's green is only a few ADU.
  - On the stack with the camera-colour sky the result is 0.39. It stays at 0.39-0.40 with 8, 16 or 32 samples, and
    0.41 with degree 3. Full measures 0.35-0.37.
  - Both Classic workflows therefore get the same matrix.
- The image is converted, not recomputed: `lin = T x (M x stack − sky)` with `T = Ma x M⁻¹`. The Gradient block's sky
  goes through the same `T`, so background_poly does not change.

**Evidence.**
- heart_a7s: measured 1 : 0.39 : 0.24, weight 0.89 (Full: 1 : 0.35 : 0.19, weight 1).
- The nebula turns from violet to red in both workflows: `fix1_classic_overview.png` (with Full for reference),
  `fix1_classic_crop1_x1400_y800.png`, `fix1_classic_crop2_x2400_y700.png`, `fix1_classic_denoise_overview.png` and
  `fix1_classic_denoise_crop1_x1400_y800.png`.
- Star colours look the same. The cyan cores of saturated stars become white: they lie off the star plane.
- **Still different from Full** (the orchestrator saw magenta against Full's salmon). Measured on the same pixels
  (`h1_hue.py`, the 50 brightest emission blocks, OKLab hue):

| Image or step | Hue |
|---|---|
| Full, linear nebula (M_full × starless N) | −33° |
| Full, final | +14.8° |
| Full, final without its hue pull | −12.2° |
| Full, final without hue pull and without `ha_boost` | −28.4° |
| Classic + denoise (patch 1), linear after the matrix and sky subtraction | −27.2° |
| Classic + denoise (patch 1), after the stretch | −28.1° |
| Classic + denoise (patch 1), final | −30.9° |
| Classic + denoise before patch 1, final | −62.7° |

  - **The hue does not depart in the simple path.** The matrix, the sky subtraction and the stretch together move it
    by less than 4°.
  - **Full's red comes from its two render-time H-alpha conventions.** `ha_boost` 0.8 adds the extracted emission in
    the HII colour (+16°). The hue pull towards 33° (`hue_shift` 0.6) adds +27°.
  - Before those two steps, Full and Classic agree within 3°. Measured colour on the Heart is pink-magenta in both,
    which the Faithful preset already notes: "measured colours without the H-alpha hue convention came out magenta".
- **Test: porting both conventions to the simple renderer** (`h2_convention.py` and `h3_sheet2.py`, at 1/4; not a
  patch). Overview: `F:/astro_essais/fixes/sheet_after2.png`.
  - Classic + denoise reaches +14.5° (Full +14.8°). It also reddens the gradient left in the bottom-right corner:
    R 52 → 60 levels.
  - Classic stack overshoots to +26.9° and the whole sky turns brown: R−B from about 0 to +6/+8 levels
    (`test_conventions_classic_stack_crop.png`).
  - The cause: `ha_boost` clips `R − αG` at zero on an image that still has its noise and stars. Full applies it to
    the denoised starless layer.
  - The hue pull alone is clean (a per-pixel display step) but gives only −0.2°.
- **Decision (orchestrator, one attempt): stop.** Patch 1 is applied as an improvement: violet −63° becomes
  pink-magenta −31°, the same as Full's measured colour.
- Whether Classic should carry the Faithful H-alpha convention goes to the user. The hue pull alone is safe. An
  `ha_boost` would need its excess measured on a smoothed or starless map.
- Weight 0.89 also keeps 11 % of the old rendering.

The other test sets: weight 0, so their Classic image is unchanged (`lin.npy` identical).

| Set | Classic (patched): emission, weight | Full: emission, weight |
|---|---|---|
| heart_a7s | 1 : 0.39 : 0.24, **0.89** | 1 : 0.37 : 0.21, 1.0 |
| iris_a7s | 1 : 1.06 : 0.57, 0 | 1 : 0.96 : 0.50, 0 |
| canon40d_rosette | 1 : 0.41 : 0.33, 0 | 1 : 0.41 : 0.39, 0.07 |
| canon6d_m51 | 1 : 1.81 : 1.29, 0 | 1 : 1.50 : 1.00, 0.03 |
| seestar_m31 | 1 : 1.04 : 0.55, 0 | 1 : 0.59 : 0.93, **1.0** |
| stellina_ic1318 | 1 : 0.58 : 0.79, 0 | 1 : 0.40 : 0.50, 0 |
| stellina_ngc891 | 1 : 0.75 : 1.12, 0 | 1 : 0.18 : 0.67, **1.0** |

A side finding, not changed here: on two galaxy fields, Full's anchor fires at full weight (Seestar M31 and NGC 891).
That may be the cause of the known red blotches (M31) and magenta sky (NGC 891), open problems 10 and 11.

**Non-regression.**
- **Stage versions.**
  - `render_simple` goes from 1 to 2, or to 3 with patch 3.
  - `color` and `render_prep` are **not** bumped: their code is only refactored, and their output is unchanged.
  - So only `render_simple` recomputes: 10-30 s per Classic workflow, on the CPU. Full's keys do not change.
- **After applying**, the plan of each project should list only `render_simple` jobs, one per Classic workflow and
  Raw stack exit.
- **Tests.**
  - `tests/test_simple_stages.py`: the new `test_render_simple_gives_the_emission_its_hii_colour` (with and without
    denoise) checks a synthetic violet case. The anchored emission gets the HII colour, stars move less than 3 %, and
    `lin` equals `Ma x stack − T x sky`.
  - Projects without a catalogue keep their matrix.
  - Then `test_color.py`, `test_texts.py` and the fast suite (results at the end).
- **Data.**
  - The 7 test sets in the table: only the Heart's Classic changes.
  - On Heart, Iris, Rosette and Horsehead, the Classic workflows added by P8 recompute; Full does not.
  - Expect only the Heart's Classic to change. Judge it blind against the P8 one.

## 2. The Post sky slider is dead over part of its range

**Cause: a floor on the curve's strength.**
- `fit_tone_curve` (`engine.py:712-752`) must put the 1st percentile (the dark clouds) at `dark_level` and the 30th
  percentile (the sky) at `sky_level`.
- `strength()` searches the curve strength b between 1 and 10⁷ (`engine.py:720`). The black point is then chosen for
  the dark level (`engine.py:730-741`).
- When the sky lies just above the darkest areas, as in a field filled with nebula, even b = 1 puts the sky above the
  level asked. heart_a7s: dark 1.63, sky 2.91, white 14.8.
  - b then stays at its floor, and the black point is set by the dark level alone.
  - The sky level no longer has any effect.
- On heart_a7s, faithful: every sky level from 0 to 34 gives the same image, with the sky at 34.6/255. **The default
  24 is inside that zone.** Artistic: 0 to 30 gives 30.9.
- The slider moved `sky_level` alone (`sliders.py:54-56`), so its lower positions fell into the dead zone.

**Two engine changes were ruled out.**
- Lowering the floor does not help. Even a linear curve cannot put this sky below 32/255 while the dark stays at 10.
- Giving the sky precedence over the dark level would change the faithful rendering at its default, which is in the
  dead zone. The hard constraint forbids that.

**Fix (`sliders.py` only).**
- The sky slider also moves the dark level, in the same proportion: 5, 7.5, 10, 12.5, 15 against 12 to 36. The steps
  are added to the base, like `sky_level`.
- The slider then always moves a level the curve holds.
- Position 2 is the base itself, so the curve and the image are identical to v1. The engine is not touched.

**Evidence.**
- heart_a7s (P7, Full), mean of a crop at 1/4, positions 0-4:
  - faithful, before: 58.8, 58.8, 58.8, 58.8, 61.2; after: 52.8, 56.0, **58.8**, 61.5, 64.1;
  - Artistic, before: 52.6, 52.6, 58.1, 68.2, 77.6; after: 46.6, 49.7, **58.1**, 66.6, 74.8.
  - Images: `fix2_sky_slider_faithful.png` and `fix2_sky_slider_artistic.png` (top row v1, bottom row patched).
- Sky level reached (/255) at positions 0-4, on 7 test sets:

| Set, base | v1 table | Patched |
|---|---|---|
| heart_a7s, faithful (defaults) | 37.4 ×5 (4 dead steps) | 31.6, 34.7, 37.4, 40.1, 42.6 |
| heart_a7s, Artistic | 33.3, 33.3, 34, 40, 46 | 27.6, 30.6, 34, 40, 46 |
| iris_a7s, faithful | 24.2, 24.2, 24.2, 30, 36 | 18.3, 21.4, 24.2, 30, 36 |
| canon40d_rosette, faithful | 29.3, 29.3, 29.3, 30, 36 | 23.4, 26.5, 29.3, 31.9, 36 |
| stellina_ngc891, faithful | 71.0 ×5 | 65.3, 68.3, 71.0, 73.6, 76.1 |
| stellina_ngc891, Artistic | 56.1 ×5 | 50.5, 53.4, 56.1, 58.7, 61.2 |
| canon6d_m51, seestar_m31, stellina_ic1318 (both bases); iris and rosette, Artistic | already alive | same sky levels; the dark clouds now follow |

**Non-regression.**
- No stage version: these are render settings only, so nothing recomputes. Renders at the default positions are bit
  for bit identical.
- Tests: `test_sliders.py`.
  - The new `test_the_sky_slider_never_falls_in_the_dead_zone_of_the_tone_curve` uses the heart_a7s statistics. It
    fails on the v1 table (34.6 everywhere) and passes with the patch.
  - `test_position_2_is_the_base`, `test_each_slider_is_monotonic` and `test_rendered_sliders_move_the_image` still
    pass.
- `tests/client/test_post.py` (P12, committed in 25af444) compares `{"sky": 3}` with a rendering at sky level 30.
  The patch adds `dark_level` 12.5 to that comparison (line 727).
- P7's smoke check "the sky slider moves the image between 0 and 2" will now pass.
- **Limit left as it is.** The engine's own "Sky brightness" setting, moved alone in the Render tab, still has the dead
  zone. The faithful Heart sky sits at 34.6, not the 24 it asks for. Changing that would change every existing render.
  The reached level could be shown next to it.

## 3. Green sky in the Raw stack exit

**Cause.**
- Without Color and Gradient, `render_simple.py:79-80` removes one number per channel (its median). Lines 83-90 then
  scale the channels to the same noise spread.
- Constants cannot neutralise a gradient whose colour differs from the mean sky.
  - On heart_a7s the camera sky is 169 : 237 : 105 at the centre, and 27 : 30 : 7 ADU brighter in the bottom-right
    corner.
  - After the exit's gains that corner is +20.6 : +25.7 : +8.3: green-yellow.
- A white balance on the sky level would not fix it either. The gradient's colour (G/R 1.13) differs from the mean
  sky's (1.40), so the corner would turn orange instead.

**Fix: fit the sky per camera channel, subtract it, add back its brightness in grey.**
- The Raw exit fits the sky of each camera channel with background_poly's polynomial (default settings), subtracts
  it, and adds back the sky's brightness as grey (the luminance of the scaled sky, minus its median).
- What the preview keeps and changes:
  - Light pollution and vignetting stay visible as brightness: no gradient removal, which is the Gradient block's
    job.
  - There is no colour calibration.
  - The channel gains are measured on the stack exactly as before, so the objects keep their v1 scaling. Only the
    sky changes.
  - A first version measured the gains after removing the gradient. That made the stars greener, so it was dropped.
- The sky label is now "camera-colour polynomial, made grey", with English and French texts (`texts.py`,
  `locale/fr/engine.json`).

**Evidence.**
- Colour of the sky per zone on the rendered image, in levels of 255 (R G B, and their spread):

| Zone | Before | After |
|---|---|---|
| top-left | 45.1 42.4 47.8 (5.3) | 44.2 43.1 42.9 (1.3) |
| top-right | 43.3 43.3 46.8 (3.5) | 43.2 43.6 43.1 (0.5) |
| bottom-left | 49.4 51.5 49.8 (2.1) | 50.9 50.9 50.7 (0.2) |
| bottom-right | 67.3 69.9 56.8 (**13.2**) | 69.1 68.2 69.6 (1.4) |
| centre (nebula and cluster) | 53.7 51.2 53.6 | 54.4 51.3 50.7 |

- Images: `fix3_raw_stack_overview.png`, `fix3_raw_stack_crop1_x5400_y3400.png` (the corner, olive then grey) and
  `fix3_raw_stack_crop2_x2400_y1300.png` (nebula and stars unchanged).
- P7's own check (whole-image channel medians within 5 %) still passes: 1.029.

**Non-regression.**
- `render_simple` version 3, shared with fix 1. Only the Raw stack exits recompute: 12 s on heart_a7s, on the CPU.
- Tests:
  - `test_raw_stack_exit_has_no_colour_cast` is updated. Every corner of a synthetic sky with a different gradient
    per channel must be neutral within 2.5 levels, with its brightness kept. On the code before the patch the corner
    spread is 0.115 (29 levels), so the test fails.
  - The old check "channel medians equal" was removed: in that scene the median pixel is red nebula, not sky.
  - `test_integration_v2::test_toy_v2_run_equals_v1_bit_for_bit` had the same median-pixel check. It now checks the
    four corners: unpatched they are 39-100 % off, patched within 15 %.
  - The 15 % is a property of this 192 x 256 toy. Its sky samples are 4 px blocks, and picking the darkest of them
    (by luminance, mostly green) leaves the green sky 5-8 ADU too low, against a noise of 15 ADU. Real frames use
    blocks of 30-70 px. The same bias exists in background_poly; it is not touched here.
  - `test_texts.py` covers the new label and its French.
- Data: the Raw stack exits of the test sets and of the P8 projects recompute. Classic and Full are unaffected.

## Test results (HEAD + the 3 patches, CPU)

All runs were on a `git archive` copy of HEAD with the patches applied, on the CPU (`CUDA_VISIBLE_DEVICES` empty).

**First fast-suite run** (`-m "not slow and not gpu and not network"`, earlier draft of patch 3): 459 passed,
5 skipped, 3 failed.
- `test_runner2::test_the_reference_is_heart_from_the_mockup_data`: the copy lacks `mockup/alt2/data/data.json`. It
  fails the same way on unpatched HEAD; not related to the patches.
- `client/test_run::…followed_instead_of_started_twice` and `test_integration_v2::test_toy_v2_run_equals_v1_bit_for_bit`:
  these tests start a detached worker.
  - From outside the copy, the worker imports the installed engine (`sandbox/astro`, `render_simple` version 1), so
    the planned key and the run key differ.
  - With `PYTHONPATH` set to the patched copy, `test_run` passes.
  - The toy test then ran through, v2 equal to v1 file by file. It stopped only at its Raw-exit median-pixel check,
    now corrected in patch 3 (section 3).
- A slow GPU test, `test_heart_a7s_v2_equals_v1`, also ran in that rerun because no marker filter was set. It is P7's
  whole end-to-end run and needs the GPU, so it is for after P8.

**Final patches, fast suite, done the safe way.**
- Setup: a copy of HEAD 1349676 (with its new `conftest.py`) plus the 3 patches; `ASTRO_GPU_HOME` set to a
  temporary folder; `CUDA_VISIBLE_DEVICES=-1`; `PYTHONPATH` set to the copy.
- Engine tests: 365 passed, 5 skipped, 1 failed (the mockup-data test above).
- Client tests: 148 passed.
- The conftest guard reported no write by the tests. The change it noted in `timings.json` is the user's live runs.

**End-to-end, by accident** (see "Incident"): P7's `v2_smoke.py all` ran on the patched engine.
- It planned exactly 3 jobs, the three `render_simple` keys, and nothing else.
- Every required check passed, including "Raw stack exit: no colour cast" and "Classic stack: the sky slider is
  monotonic".
- "Full: the sky slider moves the image between 0 and 2" now passes: 0.185 against 0.208, where it was 0.208
  against 0.208 before.

**To run after applying**, in the repository:
1. The fast suite.
2. P7's `tools/v2_smoke.py` render phase on heart_a7s (Classic and Raw stack previews, slider checks).
3. Plan each P8 project and check that only `render_simple` jobs appear.

## Incident: test runs that were not isolated (06:00-07:00)

Before the orchestrator's warning, I ran pytest in my scratch copies (`scratchpad/r`, `v`, `w`, `w2`, `w3`, `w4` and
`b`). Two things were wrong:
- Those copies had the old `tests/conftest.py`, and `ASTRO_GPU_HOME` was not set. The tests wrote into the user's
  `~/.astro_gpu` (projects.json, timings). The orchestrator has cleaned that up.
- I set `CUDA_VISIBLE_DEVICES=""` in Git Bash. On Windows the empty variable is dropped, so the GPUs stayed visible.
  Tests that pick `cuda:0` when available ran on the GPU, the 4060 Ti used by the P8 queue. This includes one full
  fast suite (about 9.5 min) and a run of about 4 min that was stopped part-way.

One run did more. `w3` with `PYTHONPATH` and no marker filter ran the slow GPU test `test_heart_a7s_v2_equals_v1`.
That test launched `tools/v2_smoke.py --root F:/astro_essais/p7 all` with the patched engine, from 06:49:57 to
06:53:04, on `cuda:0`:
- **In the P7 store**: 3 new `render_simple` results (keys `0655c09dc2517aec`, `7ae40e3baa1fa1af` and
  `ef16c50d954d22b1`). The original v1 results and every other stage are untouched.
- **Rewritten**: `heart_a7s/project.astroproj`, `state.json` and `report.json`; `smoke.log` was appended.
- **Overwritten previews** (`previews/*.png`, `sheet.png` included): they now show the patched Classic and Raw
  outputs. `sheet_before.png` in the fixes folder replaces the lost "before" sheet.
- It used its own `ASTRO_GPU_HOME` (`p7/etat`), so it did not touch the user's state.

Nothing was deleted. The image scripts I wrote ran on the CPU (explicit `device="cpu"`) and only read the P7 store
and the v1 test projects; they wrote only under `F:/astro_essais/fixes/` and my scratch folder.
