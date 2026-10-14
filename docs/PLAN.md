# astro_GPU – processing plan

*Revised 2026-10-11. A plain-language report with images, in French, explains where things stand and the decision
in progress: `F:\photos\heart_nebula\rapport\rapport.md`.*

## 1. Goal

Starting from raw frames, produce automatically a better image than the user's own manual processing, with a method
that works for any instrument (no rule written for one telescope or one camera).

Two references to beat:

- **Refractor** (TS61EDPH + astro-modified Sony A7S), Heart & Soul nebulae: `F:\photos\heart_nebula\full.png`
  (processed by hand with Siril, StarNet2, Topaz and Photoshop).
- **Newton 130/900** (same camera), Iris nebula: the user's manual image,
  `C:\Users\Nicolas\astro_gpu_scratch\projects\Iris\reference\iris_user_manual.webp`.

If it works, the project will be published on GitHub (not sold); its licence is whatever the libraries used impose.

## 2. Words used in this document

- **Stack**: the image obtained by adding up all the frames.
- **Half-stacks**: two stacks made from every other frame. What is the same in both is real signal; what differs is
  noise. This is our main way to tell real detail from noise.
- **Star shape (PSF)**: the small blur spot a point-like star makes on the image (optics + atmosphere).
- **Sharpening (deconvolution)**: undoing that blur, knowing its shape. "Non-blind" means the shape is measured on the
  stars, not guessed.
- **Noise-reduction network (Noise2Noise)**: our denoiser, a neural network trained only on the user's own images: it
  learns to predict one half-stack from the other, so it can only learn the real signal.
- **Test bench**: the user's images plus artificial objects of known shape (fake stars, faint filaments, faint
  patches), so that every method is judged against a known truth.
- **Pan-STARRS**: a professional sky survey (1.8 m telescope), sharper than our images, used as an outside reference.

## 3. Principles

1. **Free tools only**: open-source code, or free tools whose trained network is closed (e.g. StarNet2). Nothing paid.
   Paid tools (RC Astro, Topaz) are mentioned only for comparison.
2. **Reuse, and modify what we reuse.** Existing libraries can be changed (camera colour pattern, our image alignment,
   our noise model, our measured star shapes...).
3. **Treat the processing as one chain.** No single all-in-one model is imposed. When one step destroys information the
   next step needs, the two are not separated: the information is passed on, or the two are solved together.
   Elsewhere the steps stay separate. Each case is decided by measurement.
4. **Existing tool or our own, decided by measurement.** Every part is measured against the free existing tools on the
   same test bench. If an existing tool is as good or better, we use it. We only develop where the bench shows a real
   gain.
5. **Short test loops.** Each test starts with what we expect and a numeric success criterion written *before* the
   test. Quick tests take 15-20 minutes. After two failed attempts: stop and report to the user, with numbers.
6. **No setting tuned by eye on one image.** Every setting is checked on the test bench, on both datasets.
7. **Same presentation at every step**: table of measurements, 100 % crops at the same fixed places, comparison with the
   previous version. Reports in plain language, with images.
8. **Nothing is launched without the user's go.**

## 4. Information that must not be lost between steps

| Information | Exists only in | Needed by | How to keep it |
|---|---|---|---|
| Star shape per frame, per place in the field, per colour | single frames, before stacking | sharpening, star measurement | measure it per frame; sharpen before or while stacking |
| Exact sub-pixel positions and the camera's colour pattern | raw frames | resolution of each colour | stack the raw colour pixels directly, or rebuild the image from all frames |
| Saturated or unreliable pixels | raw frames | star removal, sharpening | carry them as "missing" instead of filling them with invented values |
| Two independent noise samples | the two half-stacks | noise reduction, checking | keep the halves separate through every step |
| Sky glow, vignetting and sky, which move differently from frame to frame | single frames | background model | separate them using their motions, or on the stack if the bench shows no loss |
| Star brightness and colour | measurement on the data | star layer, colour calibration | measure every Gaia star with the measured star shape |

## 5. Test bench

Datasets: **Heart & Soul** (refractor, 721 frames) and **Iris** (Newton, 1242 frames). NGC 7822 and M33 later, to
check that the method generalises.

| Test | Frames | Use | Time |
|---|---|---|---|
| Quick test | 100 frames spread over the whole night | development, go / stop | 15-20 min |
| Full test | all frames | confirmation, comparison with the user's references | ~1 hour per dataset |

The quick test is reliable for systematic defects (rings, holes, halos, colour shifts). It is not reliable for faint
signal and fine detail, which must be confirmed on the full test.

How a method is judged:

1. **Fake objects**: fake bright stars (copied frame by frame with their spikes and saturation), faint filaments
   (0.8 to 3 pixels wide) and faint patches, at 0.7 to 3 times the noise. For methods that work on the stack, fake
   objects can be added directly into the full stack, blurred with the measured star shape: the truth is then exactly
   known. We measure what each method keeps, removes, distorts or invents.
2. **Half-stack agreement, scale by scale**: detail that both half-stacks share is real.
3. **Pan-STARRS**: detail that Pan-STARRS also shows is real. Limits: in dense Milky Way fields it only judges large
   structures; the red (H-alpha) Heart nebula is nearly invisible in it.
4. For methods that model the frames themselves: prediction of frames kept aside, and the leftover after the model
   must look like pure noise.

## 6. Steps

| # | Step | What is compared | Success criterion (written before testing) | Status |
|---|---|---|---|---|
| 1 | Test bench | our chain, StarNet2, nox | the bench reproduces known numbers; table "who wins what" | done |
| 2 | Star-shape measurement | one shape for the whole image, per place, per frame, per colour | predicts stars kept aside | done |
| 3 | Sharpening (deconvolution) | on the stack (classic filter, existing tools, our network with the measured blur); from all frames together | sharper on fake objects, closer to the truth, nothing invented, faint signal kept | 5 attempts, none passes; decision pending |
| 4 | Noise reduction | our network, GraXpert, others | less noise, no loss of detail or faint signal, no over-brightening | to do |
| 5 | Stars | our model of each star, StarNet2, nox, mixed | no ring and no hole around bright stars; faint nebula kept | to do |
| 6 | Sky background | our physical model, separation by motion, GraXpert | faint nebula kept, even background | to do |
| 7 | Colour | our camera response from Gaia spectra, Siril colour calibration | Gaia star colours recovered | to do |
| 8 | Final image | best chain from steps 2-7 | 100 % crops of all bright stars and fixed zones; comparison with full.png and the Iris reference | to do |

## 7. What we have measured so far

**Test bench and star removal (step 1).** On the bright fake stars, our star model leaves far smaller rings than
StarNet2 and nox (typical / worst ring, in times the noise: Heart 0.2 / 0.9 for us, 0.6 / 3.4 StarNet2, 3 to 8 nox;
Iris 0.2 / 0.6 for us, 1.1 / 14 StarNet2, 2 to 21 nox). On faint filaments and patches, the existing tools keep more
(Heart filaments: 66 % for us, 53 % StarNet2, 83-94 % nox). Far from any object our chain fluctuates 15 times more than
StarNet2 (a "pepper" grain), probably because star brightnesses are refitted over the whole image.

**Star shape (step 2).** Measured from Gaia stars with an error of 4-5 %. It varies mostly across the field, little
from frame to frame (Heart 3.8-4.3 px, Iris 4.3-5.0 px over the night), and with colour only on the refractor.

**Sharpening (step 3).**
- The stack noise has a 2-3 pixel grain (each stacked pixel mixes neighbouring pixels). Existing tools trained on
  ordinary photos take this grain for detail and turn it into invented patterns, unless the grain is measured and
  accounted for.
- On faint nebula, the resolution is limited by the noise, not by the blur: with all 1242 Iris frames, the two
  half-stacks share 0-3 % of their detail below 2.5 pixels, 7-18 % between 2.5 and 5 pixels, 30-60 % between 5 and
  10 pixels. Below 2-3 pixels the faint detail is simply not in the data.
- On bright zones the gain is real: fine streaks near the Iris core, confirmed by Pan-STARRS, appear only after
  sharpening.
- The sharpening target must be one the measured star shape can reach: 3 px for the Newton, only 2 px for the
  refractor (sharper, undersampled star cores).
- Five attempts (classic Wiener filter; the DPIR tool from the deepinv library, as-is and corrected for the grain; our
  network trained with the measured blur, twice). None is both sharper and closer to the truth. Two problems upstream
  limit every attempt: the rings around the Iris central star already exist after star subtraction (step 5), and our
  noise-reduction network does not return faint structures at their true brightness (often too bright, sometimes too
  faint; step 4).

## 8. Decisions

- Decided: free tools only (open source or closed networks); libraries may be modified; licence follows the libraries;
  the chain is designed as a whole where steps depend on each other; reference datasets Heart and Iris; quick test on
  100 frames, then full test; user references full.png and the manual Iris image.
- Pending (see the French report, section 5): what to do after the sharpening attempts. Recommended: fix the star
  subtraction around very bright stars and the faint-signal bias of the noise reduction first, then the final rendering
  (dust, colour, stars), then sharpening restricted to zones where the half-stacks prove the detail exists.
- Open question for the user: how much "detail not proven by the data" is acceptable in the final image (strict
  fidelity, or some controlled aesthetic sharpening).

## 9. History and lessons

- 2026-10-08: the chosen idea included a physical model of the raw colour frames with a star shape per frame and per
  colour (sharpening from all frames together).
- The first attempt was not that (one Gaussian blur per frame, no sharpening, two short runs) and its "no gain" was
  wrongly generalised. Then classic sharpening on the stack (stars 12.9" -> 8.7"); the measured star shape was later
  used only for star measurement, and nebula sharpening was dropped after a test on faint nebula only.
- 2026-10-10: astro_GPU ported that chain without sharpening; star removal grew into ~8 stacked repairs whose edges
  produce rings around bright stars (Iris central star).
- Lessons: several independent checks, never one; never drop a part without telling the user; no stacked repairs; no
  launch without the user's go; plain language in every report.

## 10. State of the code (2026-10-11, after the rebuild)

The user's preferred image is the first Heart render of 2026-10-10 10:34 (`results_archive/natural`). The chain was
rebuilt from the code of that morning (commit 4632904) and the later changes were re-injected one at a time, each
judged at 100 % against that image on Heart ("equivalent or better") and on Iris ("works": core intact, no holes,
no patchwork):

| Change | Verdict |
|---|---|
| Fixed-pattern (walking noise) removal at stacking (stack v2) | kept (neutral on Heart) |
| Background: bounded fit, object clipping, zero-point floor (v3) | kept (Iris: no more clipped black) |
| Missing data decided on the measurements only (no mask by model brightness) | rejected: grey discs and glare residuals on the refractor (the wide mask hides the wing errors of the model) |
| Dip repair only where a dip is measured | kept (neutral) |
| Glare vs nebula decided per star on the fall-off | rejected: with the wide masks nothing is left to measure; the Iris G6.9 glare stayed |
| Full-resolution render base (no emission/continuum model) | needed for reflection nebulae (Iris core), but mottled the dense Heart field |
| ... with the continuum smoothed only where star light dominates (render_prep v16) | kept: Heart equivalent, Iris core structured |
| Extended star template (halo, spikes) alone | rejected: large ring around the Iris central star, haze |
| Nebula-aware core mask | no effect, dropped |
| Colour calibration v2 (camera profile, synthetic spectra) | kept (neutral on Heart, Iris closer to the reference) |

Not re-injected: the sharpening network (`core/sharpen.py`, kept in the repository, not called by the rebuilt
starless stage) and the night's glare zones. Known remaining defects: the Iris G6.9 glare (blue blob) and the
plateau of the filled core around the Iris central star (fill disc of the cleaning step). The founding principle of
the project, per-frame per-tile deconvolution before stacking, has still not been tried: it is the next session.

- Render presets `Faithful` (reference rendering, measured colours) and `Artistic` (`render/settings.py`);
  `astro-gpu render PROJECT --preset faithful` writes `results/faithful/`. GUI: "Results" tab, project descriptions.
- Earlier work in progress stopped by the user (stacked repairs) is kept in `git stash` and not used.
- Test bench and night scripts (outside the repository): `C:\Users\Nicolas\astro_gpu_scratch\bench\` and
  `C:\Users\Nicolas\astro_gpu_scratch\night\` (rebuild patches, judge script, logs).

## 11. Iris against the user's own image (2026-10-11 afternoon)

The user's full-size Iris processing (3749 x 2514) is registered as the project's reference image and aligned to the
stack (1284 stars, 0.14 px). `judge_iris.py NAME` compares a render with it at 100 % on six zones, in the reference
framing (`results/compare_reference/`). What the comparison showed, in order of impact:

1. **Stale files.** The stage engine never deleted the outputs of an earlier version, and the render preparation
   picks up optional files when they exist. The extended star template of a rejected night version was still in
   `work/stars/` and classified the whole Iris nebula as "star light": the continuum was smoothed at 24 px over 53 %
   of the field, the nebula included. Fixed in the runner (`_clear_outputs` at every fresh start of a stage); both
   projects cleaned by hand. Every verdict of section 10 was made with those files present.
2. **Reflection nebula = star light.** Even without stale files, "smooth the continuum where star light dominates"
   is a rule made for emission nebulae: a reflection nebula is scattered star light. Deciding "dominates" against the
   nebula brightness instead of the noise was tried and rejected (nothing gained on Iris, star-residual mottling
   back on Heart). The rule stays as in version 16; step 1 alone brought the Iris core structure back
   (smoothed area 53 % -> 25 %, core contrast 2.0 -> 2.6, user's image 4.7).
3. **Halo of the bright blue star (G 6.9).** Its halo extends to 300 px; the removal (nominal radius from the
   magnitude, a refractor scale) covered 130 px, partially. Measuring the extent on each star hollowed the Iris core
   (seen from one star, a nebula centred on it is a halo). Kept rule: the instrument's typical halo is the median
   profile of the eight brightest stars normalised by their catalogue flux (fitted and predicted fluxes both fail on
   saturated stars); each of those eight loses min(its own profile, three times the typical halo), out to where the
   typical halo falls into the scatter of the far background; every other star keeps the version-4 treatment. Applied
   to all 1113 bright stars of the Heart field it brightened and yellowed the sky: rejected in that form.
4. **Spikes.** The diffraction spikes of the two brightest Iris stars are not in the star model and stay in the
   starless image. A detector (narrow peaks of the azimuthal profile at the same angles over several annuli) finds
   them; masking and filling the wedges leaves a bright bar (the fill takes the halo's value). Parked: subtract a
   measured spike profile instead of filling.
5. **Not defects of the chain but of its design, to decide with the user**: the darkest 1 % of the field is rendered
   at 10/255 (dark_level); on Iris the darkest region is plain sky in a field that has no empty sky, hence "black
   patches" that the same policy renders as dark nebulae on Heart. Stars are drawn small and dim compared with the
   user's image (settings `d10`, `d7`, `star_sat`). The faint dust is rendered yellow-olive where the user's is
   neutral brown.
6. **Star flat.** The correction derived from star photometry (-23 to +11 % on Iris, -6 to +24 % on Heart) makes the
   sky *less* uniform on both data sets (lower envelope range x2.5): it measures aperture losses of the PSF across
   the field, not a flat error. Present since the first version; to be gated on "makes the sky flatter" later.

### 11.1 Halos around the stars on Iris (2026-10-11, late afternoon)

The user found halos everywhere on Iris at 100 % (around stars in the nebula core, around faint field stars, on
the spikes of the bright stars) that the zone judge had not caught. Rule since then: every result is inspected at
100 % on faint stars, stars in the nebula, bright stars with spikes, on both data sets (`night/qc_sheet.py`),
before being shown. Causes and fixes, all general (no instrument rule), each checked on Iris and Heart:

| Cause | Fix | Commit |
|---|---|---|
| Bubbles around every star in the reflection nebula: the render preparation smoothed the continuum at 24 px in a disc around each star | smoothing only where the field is too dense to see the continuum between the stars, or where it is flat between them | 55737bf |
| Rings / commas around every star: a Newtonian's star shape changes across the field faster than the fitted PSF | local star shape measured on the image (grid of cells) + per-star flux, position, width refinement | a0f9bfa |
| Grey disc around the Iris central star: its real halo and the nebula counted as model error | a residual is a model error only below 60 % of the model | a0f9bfa |
| Spikes of the Newton left in the nebula layer, or removed and lost | spikes measured strip by strip and moved to the star layer | a0f9bfa |
| Bright saturated stars: nothing can model them (saturation, coma, spikes) | with spikes in the optics, saturated stars are shown by their real light (image minus nebula), nebula interpolated under them unless a structured nebula surrounds them; without spikes (refractor) every star is drawn as before - Heart identical | 55737bf |

Rejected on the way (each made one star better and another worse): a 2-D halo-and-spike template for bright stars
(black disc), disc rules for bright cores (three variants), the halo extent followed in the starless image (the
footprint of the Iris central star grew and its nebula was erased), OpenCV inpainting (garbage on large holes).

Remaining on Iris: about 14 stars of G 9.5-11 have a slight dark disc - their fitted flux is about twice the
catalogue prediction (core near saturation) and their modelled wings are removed twice too strongly; thin straight
spike traces cross the Iris core. The user wants to keep the spikes ("the charm of a Newtonian image").

### 11.2 Dark grit on Heart, odd spikes on Iris (2026-10-11, evening)

The user saw dark grit in the Heart nebula and odd spikes on some medium stars of Iris (the Iris nebula itself:
"excellent").

**Dark grit (Heart) - found and fixed.** The starless image was clean at those places (a slight bright bump on
filled star cores, never a hole). The render preparation made the holes: on a star footprint the star guard gives
the brightness it keeps a neutral colour, which is star-coloured light plus a negative emission (neutral light is
bluer than the average star); the continuum smoothing that followed removed the star-coloured part and left the
negative part, a dark, slightly coloured hole. Fix: the continuum is smoothed before the guard, on the true colours;
the guard then fills whatever hole is left on the footprints. Dark specks of the faithful render 235 -> 133 (the
morning image has 129), deep ones 22 -> 11; a compact bright knot that came out as a black hole is back as in the
morning image. Iris: no visible change, same speck count within the run-to-run spread.

Rejected on the way, each measured on both data sets: every star bounded by what the image holds (Heart specks
249 -> 505); lifting the emission under positive star residuals over the whole field (more specks: the lifted
spots make their surroundings look darker) or on footprints only (235 -> 371, same reason).

**Odd spikes on Iris medium stars - diagnosed, not fixed.** In the user's image these stars are small diamonds (the
inner part of the spikes merged into the star). Ours are a round drawn star plus a spike that starts about 10 px
away: the inner part of the spike lies in the filled core of the starless image, and a drawn star is round, so
neither layer holds it. Tried: a spike profile per colour channel (the widths are equal, 8.5/8/8 px), the direction
of each spike measured on each star (stars fan by up to 4 degrees; remnants 40 -> 34 stars), the intensity along the
spike following the data instead of a power law (remnants in the nebula layer 2-3 times weaker, no visible gain in
the render, a bright stub at the start of some spikes), the stars with measured spikes shown by their real light
(spike attached, but medium stars swell into soft discs, as saturated stars did on Heart). All removed. The real fix
is to draw the stars with their measured local shape (the star stage already measures it, cell by cell): it would
also give the corner stars their coma instead of a white blob with a dark crescent. Waiting for the user's decision.

Noted: rerunning the star and star-removal stages with unchanged code changes the result slightly (Iris specks
354 -> 379, render preparation almost unchanged): comparisons are made within one run of the upstream stages, never across
reruns.

### 11.3 Spikes: one measured star shape instead of a spike model (2026-10-12)

The user rejected the spike traces left on Iris after 11.2 ("step back, no tinkering"). Stepping back: a diffraction
spike is not an object of its own; it is part of the shape the optics give every star (diamond core, four spikes,
secondary rays of the mirror clips), the same for every star up to its flux. Every earlier attempt cut the star into
pieces - a round drawn star, a spike fitted star by star on noisy data, a gap between the two - and the pieces did not
meet. Measured facts that led to the design: the field rotates only 0.27 degree during the night (no fanning); the
stacked shape of 63 isolated stars shows the diamond and the four spikes cleanly; the spike amplitude fitted star by
star scatters by 30-70 % around the flux (nebula and neighbours), so no per-star spike fit can be clean; the
photometry of a saturated star underestimates it (G 6.9 by 4x) and the catalogue is wrong for a very red star
(G 4.6, BP-RP 5.7).

Design (`stages/star_shape.py`, star stage v8, star removal v19, render preparation v24, engine):

* the angular part of the star shape is measured once on the residual image around the stars brighter than G 11
  (polar grid, each ring minus its median over the angle), per unit flux, weights flux^2 capped at the 5th brightest
  star reaching each radius, smoothed along the radius in proportion to it and along the arc at the pixel scale;
* the stars are split into two halves measured independently; the shape is where both halves show it above twice
  their noise, the noise being the larger of the pixel noise and the difference between the halves; its value is the
  mean of the halves; structure broader across than three spike widths is removed; one shape (green) for all colours;
* cross-check: the shape of one half must be found in the stars of the other half at more than 5 standard errors.
  Iris: 144 and 238. Heart (refractor): -65 - no shared shape, round stars only, nothing changes;
* flux scale calibrated on the unsaturated stars; saturated stars get the flux measured on their own spikes;
* every star's model is its flux times the shape (subtracted from the nebula; the unreliable-core mask ignores it, so
  spikes are subtracted, never cut out); the strip spike model of the star removal is gone;
* drawing: saturated stars keep their real light, their spikes continue it outside the footprint; a drawn star gets
  its measured shape (core diamond from the local templates, spikes outside) attenuated like its wings, i.e. by the
  same star-size reduction the rendering applies to the round part (at full strength it showed as a ring).

Rejected on the way: a free shape without the halves test (nebula and dense-field neighbours passed as shape: Heart
1.7 % of the light), one shape per colour channel (coloured streaks), the smaller of the two halves as value (thinned
the faint spike ends, which stayed in the nebula), the energy explained as cross-check (noise-dominated, 1-6 % even on
Iris), the shape at full strength on drawn stars (rings).

Result on Iris: the horizontal spike traces of the medium stars are gone from the nebula layer and from the render; the
messy double star in the corner is clean; dark specks 360 -> 332 (artistic), 232 -> 215 (faithful); zone figures
against the user's image unchanged; bright stars keep their real spikes.

Heart: the cross-check rejects the shape (-67 and -64 standard errors in two runs), the code path is the one of
before. Two runs with identical code differ by as much as the new run differs from the previous final (12.5 % of the
pixels by more than 3/255; dark specks 133 / 153 / 145): the star stage is not deterministic (its star model differs
on 89 % of the pixels between two runs). Making it deterministic is needed before fine comparisons across reruns.

### 11.4 Three objects of 28.12.2024 without flats (2026-10-13 night)

Flaming Star (IC 405), Horsehead, Rosette: A7S + Newton 130/900, 15 s ISO 4000, darks found unsorted in the camera
folder, no flats that night. Projects on E: (C: was nearly full). Darks: a first selection on level and absence of
stars (557 of 722) was wrong - most of those frames hold a growing central light (dawn through the capped tube?) up to
24 ADU; the master dark then had a +8 ADU disc, and with the Rosette's low sky noise 7 % of the sensor (a central
disc) was flagged as hot: the stack covered the centre at 13 %, rendered black. Darks must also be spatially uniform
(centre - corners <= 4 ADU, one quantisation step): 175 qualify, 100 used.

* Flats: the star-derived flat (cubic) cannot carry the Newton's vignetting (down to 21 % in the corners): rings and
  green corners. The flats of Iris (March 2025) and M33 (August 2025) have the same vignetting as this night's star
  flat (correlation 0.96 / 0.95): calibrate option `flat_scale_px` (40 px) keeps only the large scales of borrowed
  flats (vignetting, not their dust). Rejected: the star-flat degree chosen by cross-validation (degree 5: the
  H-alpha fit flipped and the image went green).
* Hot pixels: calibrate v2 also flags pixels that stand out in the lower quartile of 40 lights in sensor coordinates
  (skipped if the sky moved less than 12 px): 1247 isolated pixels on the Flaming Star, 1054 on Iris. The green dotted
  trail below IC 405 is NOT explained by it: it is in both half-stacks as dots along the drift path, but no sensor
  pixel of the calibrated cache traces it (outliers in 10 % of frames or more, pixels marked saturated in 5 % or more,
  checked against the frame-to-reference mapping on a star), and the stack's fixed-pattern estimate has nothing there
  (recomputed: features only at the edges and at the brightest stars). It starts at a star. Cause found on
  2026-10-13: one green sensor pixel, in the bad-pixel map, was marked saturated in most frames - the saturation mark
  was applied after the bad-pixel exclusion and overwrote it, and the stack takes a saturated pixel for the core of
  a bright star (drawn along the drift). Calibrate v3: a bad pixel stays excluded when it saturates; the three
  dotted trails are gone.
* Runner: adding flats to an existing project was ignored (the inventory did not rerun): the first stage now records
  a fingerprint of the project's input lists.
* Bright saturated stars in nebula (AE Aurigae): the whole-footprint interpolation smooths the nebula over ~200 px
  around the star. Three variants were tried and rejected (interpolating only the unreliable core: Iris G 6.9 became
  a blue disc; structure measured inside the footprint: star residues read as nebula; upper-quartile structure test +
  core interpolation: holes, blooming streak and a starburst of rays around AE Aur and the G 4.1 star). The validated
  rule is kept; the real fix is a better star removal around very bright saturated stars (blooming columns, rays).

### 11.5 The far halo of the stars (2026-10-13)

Horsehead was the worst of the three: a blue glare over a quarter of the frame, colour patches, an "eye" (dark disc in
a bright ring) around sigma Orionis and other bright stars, sky fitted on 46 % of the field. Diagnosis:

* Alnitak (V 1.74) is missing from Gaia DR3: not modelled, its glare stayed in the nebula. Catalogs v2 adds the
  Hipparcos stars of the field with no Gaia counterpart, G and BP-RP from V and B-V through relations fitted on the
  Hipparcos stars of the same field that Gaia has (Horsehead: Alnitak alone, 20 stars for the relations; Iris and
  Heart: none).
* The star flat (-33 %..+17 %) is real: aperture photometry of isolated stars gives the same map (-15 to -24 % on
  the right edge). Not the cause.
* The glare is part of each star, proportional to its flux. It was modelled out to 60 px only (stars stage) and then
  cut star by star in the starless image with a local reference level and a taper (the eye). The far halo is now one
  shape for the instrument, measured on all stars brighter than G 10.5 jointly with a smooth sky, robustly, on a
  coarse grid: non-increasing in log r, falling at least as 1/r (a free shape levelled off and took the Iris dust
  for glare), achromatic (the colour of each star; per-channel shapes took the red emission for glare), on an
  elliptical radius whose axis ratio and angle come from the second angular harmonic of the fit (Horsehead 1.27 at
  -3 deg). Inside the radius the star model holds (60 px or the saturated core), it is flat. A star whose halo is
  measurable against its surroundings gets its own amplitude (0 to 1.5: T Cephei by the Iris, a Mira fainter than its
  Gaia magnitude, would lose a dark disc); a star whose glare dominates them (Alnitak, sigma Ori) also its own
  elongation, angle and radial slope (Alnitak: elongated at -30 deg, not like the instrument), the candidates being
  compared on the same trimmed residual (a selection on the pixels kept by one candidate chose that candidate).
* The halo is measured in the stars stage and saved apart (stars_model_halo.npy): the starless stage subtracts it
  from the half-stacks with the star model, but not from the core model that decides what is filled and where
  residuals are guarded (the glare would have been filled away with the Flame). The angular shape (spikes) and the
  halo are measured in turn, each without the other: a long spike passed for an elongated glare, the glare for
  star shape.
* The angular shape was then rejected on Horsehead (cross-check 0.1 sigma for one half): the two companions of
  sigma Ori, inside its core, carried its residual per unit of their own flux into their half. Stars within 60 px of
  a star 1.5 mag brighter no longer enter the shape measurement (cross-check back to several hundred sigma).
* Rendering: the real light of saturated stars no longer includes the far halo (it was drawn out to 400 px and cut);
  the halo is a layer of its own drawn with the stars as a glow, attenuated like the wings of the drawn stars
  (setting "Bright star glow", a fraction of the measured halo, 0.3 by default: the wing attenuation of the drawn
  stars, 0.06 on Horsehead, made it invisible).
* Open: the fine rays of Alnitak and sigma Ori (and their colour fringes) are star light that the shared angular
  shape does not hold: they stay in the starless layer, which the render keeps under these stars (structured
  surroundings), and are stretched as nebula. Tried and removed: attenuating the real light of saturated stars
  beyond their core like the drawn wings, and taking its colour at 4 px - both act on the star layer, where the rays
  are not. Next: the own radial structure of the stars whose glare dominates (sharp across the arc, smooth along
  the radius), measured like the halo and moved to the star light - with care for the Flame next to Alnitak.
* The background floor (no significant area below zero) was set by blocks the halo removal leaves uncertain: along
  Alnitak's horizontal spike 90-140 px out (~30 ADU too much), 400 px south of it (~5 ADU, 10 % of the halo), and the
  filled core of the corner star: +5 ADU, unequal in the three channels, a blue cast over the whole sky. Background
  v4: the floor is set only by measured blocks (filled cores excluded) where 20 % of the removed halo (the precision
  measured around Alnitak: -5 % at 90 px to +25 % at 300 px) cannot explain the deficit. Horsehead: floor 1.0-1.6
  ADU (as before the halo model), IC 434 red again, the dust lane south of Alnitak black as in the user's own stack;
  Iris and Heart unchanged (0.6-1.2 ADU).
* Tried and not kept: the own rays of the dominant stars as angular structure that is the same over a factor 1.5-2
  in radius (radial median in log r, per channel): fine rays reduced, but the broader dark and blue structure within
  300 px of Alnitak stays.
* Finals of the three objects (2026-10-13 13:20, archives of the previous ones in E:stro_gpurchive): Horsehead
  without the glare and the colour patches, Flame and IC 434 clean; Flaming Star with the filaments around AE Aur (a
  smoothed disc before) and no glow at the bottom; Rosette with its central cavity (filled by the glare of the cluster
  before). At 100 %, around the two or three brightest stars of each field: fine rays and colour fringes (Alnitak,
  sigma Ori, the G 4.1 star of the Flaming Star) and shallow dips of ~1 ADU that the stretch turns black (above the
  G 4.9 star of the Flaming Star); the central star of the Rosette is a white disc in a dark ring. The Rosette's
  angular shape is still rejected by the cross-check (3.0 and 37 sigma; -10.6 this morning).
* Regression found on these finals and fixed (starless v21): the per-star cleaning of what is left around the bright
  stars (bright_halos) is bounded by K x the typical halo of their brightness, measured on the brightest stars; after
  the far halo that template was nearly zero, so nothing was cleaned any more near the cores of saturated stars (white
  disc of the Rosette's central star, clustered cores of the Horsehead G 6.2 star). The template is measured with the
  far halo (on the starless image plus the halo), the cleaning applies to what is left. Iris: the magenta arcs around
  the blue G 6.9 star are gone too.
* That fix brought back the old fault around AE Aurigae (a smooth pink disc where the filaments were): the per-star
  cleaning acted out to its nominal radius (up to 320 px), where the far halo now holds the glare. Starless v22: it
  acts only where the star model holds the star (1.3 x the radius inside which the far halo is flat, ~80 px). AE Aur
  keeps its filaments; Horsehead and Iris keep the v21 gains. Still open: the white disc of the Rosette's central
  star (its filled core), the clustered cores of the Flaming Star G 4.1 and G 5.3 stars, the dip above G 4.9.
* The real-light footprints of the saturated stars shrank with the far halo out of their light (Horsehead: 5.1 % of
  the field -> 1.2 %): the core residue spilled out, made the surroundings look structured, and the nebula was no
  longer interpolated under 14 of 50 stars (clustered white cores, the Horsehead G 6.2 star). Render prep v26: a
  footprint covers at least the unreliable core (1.3 x the radius inside which the far halo is flat). Horsehead: 3.5 %
  of the field, interpolated under 40 of 50; G 6.2 and the corner star's companion clean again; the NGC 2023 star
  keeps its clustered core (structured reflection nebula around).
* Left for next time: three bright stars at the bottom of the Flaming Star (G 4.1, G 5.3) and the NGC 2023 star keep
  their core residue in the starless layer: their surroundings are judged structured (nebula, the rays of G 4.1), so
  the nebula is kept under their footprint. Their star layer is clean. Interpolating always within the unreliable
  core and keeping the nebula only beyond it would remove the residue (tried before the far halo: Iris G 6.9 turned
  into a blue disc - to be re-tested now that the glare is modelled).

### 11.6 Against the user's own images (2026-10-13 night)

The user's finished Horsehead, Flaming Star and Rosette (project reference folders, aligned by the reference stage -
image features when star matching fails) are the guide; an independent agent judges every render on aligned 100 %
crops (stars, star field, colour, structure, faint detail, artefacts) and must pass it before it is shown. The
Horsehead finals of 14:21 failed on everything (1-3 / 5).

* Diagonal streaks (walking noise): they were drawn by the fixed-pattern removal itself. On the star-free background
  of the Rosette stack, without it the fine structure is isotropic (ratio along / across the drift 0.97); with it,
  streaks along the drift (1.67). The pattern estimate (frames minus their smooth sky, averaged in sensor
  coordinates) also holds the tracks of the faint stars that cross the sensor with the drift; subtracting it from
  every frame redraws them as streaks. It had been neutral on Heart: stack v3 turns it off by default.
* Saturated star cores rendered "as photographed" (nebula and star light together under the nebula curve, no
  separation): the shards around Alnitak went, but the nebula curve made a large flat white disc of every bright
  star and the undenoised stack under the footprints a grainy disc. Removed.
* The glare of the stars (far halo, star_glow) is drawn with the nebula instead of the stars (engine): the residual
  rays of the brightest stars stay in the nebula layer, and on a dark base (glare taken out, drawn back with the
  gentle star curve) they stood out as multicoloured shards with black holes between them. On the glare, as in a
  photograph, they are soft (Alnitak, sigma Ori, the G 4.9 star with no dark ellipse; Iris G 6.9 a larger glow).
* Colours: the colour matrix is fitted on the stars and anchored on the camera's response to an HII spectrum,
  reconstructed from the stars. That response is poorly determined at the lines: green / red at H-alpha 0.04 to 0.43
  on five fields of the same camera. The emission actually recorded (colour of the band-pass structure of the
  nebula, zero-point free) is R 1 : G 0.3-0.5 : B 0.17-0.34 on all of them; the Horsehead matrix sent it to blue-violet
  with a negative luminance (IC 434 and the Horsehead nearly vanished, magenta wash). Render prep v27: the matrix is
  changed only across the plane of the star colours (rank one), so that the measured emission gets the colour the
  calibration gives the HII spectrum; the stars keep their colours exactly; not applied when the emission is within
  the colours of the stars (Iris: unchanged). Horsehead: IC 434 red, Flame salmon, sky neutral; Heart: a slightly more
  neutral sky. The independent judge ranks it clearly above the previous version (still a fail overall).
* Reflection nebulae centred on their star (NGC 2023, IC 435) came out hollow with a bright ring: the per-star
  cleaning of the starless stage removed, for every star but the eight brightest, everything above the level just
  beyond its nominal radius, unbounded. Starless v23: bounded by K x the typical residual of these stars measured on
  themselves (median per unit flux). NGC 2023 keeps its core (115 ADU at 15 px instead of 25 flat); IC 435 follows
  the data (a plateau); no change elsewhere.
* Render setting "Diffuse glow removed" (render prep layer Dglow: 3rd percentile of the nebula over a fifth of the
  frame, smoothed): the level a gradient removal would take as background. The user's Horsehead treats the diffuse
  H-alpha west of IC 434 as background (neutral grey where our data have R 4.9, G 2.1); subtracting it is also what
  turns the dust east of IC 434 teal. 0 in Faithful.
* Own rays of the dominant stars (stars v10-v11): radially coherent (running median along the radius over a factor 2),
  narrow (+-12 px arc high-pass), per channel, significant against the noise of the median, bounded by the star's
  glare (their contrast to it measured just outside the star model, at most four times it, tapered before 1200 px);
  saved with the glare. The star below sigma Ori is clean again; a first version without the glare bound drew a
  ring of dark radial streaks 600-1200 px out. Tried and not kept: +-30 px (no visible gain).
* Render prep v29: the star guard also covers where star light dominates the noise: dark moats around the G 9.5-11.5
  stars (their wings slightly over-subtracted 6-20 px out, outside the filled core) and black specks in the Flame.
* Engine: the tone curve is set on the nebula without the glare (more glare made the whole image hazier and dimmer).
* Render prep v30: the diffuse part of the glare goes with 'Diffuse glow removed' (the far glare of Alnitak veiled the
  faint areas when the diffuse glow was removed).
* Render prep / stars v31 / v12: only the glare of the stars whose glare dominates their surroundings is drawn back
  (Horsehead: Alnitak, sigma Ori, G 4.9; Flaming Star: the four bright stars at the bottom; Rosette: none). Drawn back
  for every star, the glare of the cluster filled the Rosette's central cavity. The measured emission colour also
  separates emission, continuum and OIII (the green of the emission passed for OIII wherever there is H-alpha).
* Artistic preset = the user's own slider settings on the Rosette (2026-10-13 22:33) + diffuse glow 0.6. New setting
  'Neutral sky' (background neutralisation, 0 by default).
* Tried and not kept: render settings fitted automatically on the user's references (OKLab band-pass and colour
  distances; pixelwise comparisons favour flat grey images, contrast amplitudes still give grey desaturated ones);
  the Noise2Noise network in the nebula base up to 32 px (the faint dust it shows is real, but star residues came
  back around the bright stars); stronger OIII boost on the Rosette (our OIII map is not where the user's teal core is:
  whitish patches in the ring, teal at the frame edges - the OSC separation is not reliable at this signal).
* User feedback 2026-10-14 midday: the Heart covered with black spots; dark spots around the spikes of the bright
  stars of the Rosette and the Flaming Star. Causes and fixes (render prep v32-v34):
  - Heart: the star guard extended to wherever star light dominates the noise (v29, for the Horsehead's dark moats)
    covers 98 % of a Milky Way field; the guard's interpolation then drew dark discs (60 spots, 36 before). Reverted.
  - The real light of a saturated star was image minus the starless image plus the change made to the nebula; it
    missed what the preparation changed in the starless image itself. Where the core fill of the starless image is
    brighter than the image between the spikes (Rosette, no spike model), the star light was negative: dark notches
    in a diamond. Now image minus the nebula rendered under the star: nebula + star light give the image back.
  - Under a saturated star in structured nebula the nebula was kept as is, with the dark notches the symmetric star
    model leaves between its rays (Flaming Star G 5.3, 5.9, 6.1, AE Aur). It is now kept at large scales only
    (Gaussian of R/8 over the footprint) where the star model light exceeds the nebula (weight 0 at half of it, 1 at
    twice it); the finer scales go to the star light. Smoothing everywhere under the footprint washed out the Iris
    around its star (the nebula is 2-3 x the star model there; 5-40 x around the Flaming Star stars).

### 11.7 Star artefacts: the leads (2026-10-14 evening)

The user (Iris G 6.9 star, dark blobs along its spikes): step back and solve the star problem for good. Every star
artefact so far (notches, rings, discs, blobs) comes from subtracting a model of each star and drawing it back: the
model never matches each star exactly. Leads, to be sorted at the end (each in its own branch, each judged by an
independent agent on aligned 100 % crops):

| Lead | Branch / place | Idea | Status |
|---|---|---|---|
| 1b real stars | real-stars | stars brighter than G 12: spikes (and saturated cores, rays) shown as photographed = image minus the nebula under them, the nebula interpolated across the spikes; fainter star cores drawn as now | in test |
| 2 StarNet2 / nox | D:/astro_gpu_diag/starnet_test | off-the-shelf learned star removers on the problem stars | agent running |
| 3 own network | star-net, D:/astro_gpu_diag/own_starnet | a star remover trained on the project's own data (its real bright stars injected on clean nebula) | agent running |

Lead 1b, first findings on the Iris (G 6.9 star):
* all stars G < 12 shown by their own light (discs + spikes): the medium stars came out large and soft (their
  photographed wings escape the star size settings) - only their spikes are taken as real light now, their round
  core stays drawn;
* the spike strips were missed on saturated stars (the saturated core broke the spike length measurement at the
  start): the length is now where less than 60 % of the last 15 px stand above twice the noise;
* the fine detail, colour and sharpening layers come from the starless images where the shared spike model was
  subtracted: off under the strips too;
* the dark blobs left were the star's fine secondary rays beyond its footprint (measured with the ring median, which
  narrow rays do not move): footprints of saturated stars now out to where the ring's 90th percentile falls below
  twice the noise (Iris: 4 % -> 14 % of the field). G 6.9 clean, the Iris core unchanged; to check: faint dust under
  the larger interpolated footprints.
