# astro_GPU - processing plan (revision of 2026-10-10, evening)

## 1. Goal

Beat the state of the art on the user's own data, with a method that generalises to any instrument (no
instrument-specific rules). References to beat, one per optical case:

- refractor (TS61EDPH + Sony A7S astro-modified): `F:\photos\heart_nebula\full.png`
  (Siril + StarNet2 + Topaz + Photoshop);
- Newton 130/900: the user's manual Iris processing,
  `C:\Users\Nicolas\astro_gpu_scratch\projects\Iris\reference\iris_user_manual.webp`
  (copy of the user's file; 2000 x 1340, about 1.5x smaller than the sensor scale, same orientation as the stack;
  confirmed by the user to come from these raw frames).

If it works well, the project will be published on GitHub (not sold). Its licence is whatever the libraries used
impose.

## 2. Principles

1. **Free tools only.** Open-source code, or free tools with closed weights (e.g. StarNet2). Nothing paid. Paid tools
   (RC Astro) are only literature context.
2. **Reuse first, and modify what we reuse.** Libraries can be forked, patched or extended (Bayer sampling, our warps,
   our noise model, our measured PSFs...). Licences are not a constraint.
3. **Master the chain.** No single model is imposed. Where a step destroys information that the next one needs, the
   two are not decoupled: the information is passed on, or the two are estimated jointly. Elsewhere the steps stay
   separate and modular. Each coupling is decided on the bench, case by case.
4. **Make or buy, decided by measurement.** Every component is measured against the free shelf alternatives on the
   same bench. If a shelf tool is as good or better, it is used (as-is, inside the chain, or modified). We develop only
   where the bench shows a real gain.
5. **Fast validation loops.** Every step starts with a hypothesis and a numeric pass criterion written before the
   test; tests take 15-20 min on the reduced batch; then go / stop. Two failed attempts: stop and report with numbers.
6. **No threshold tuned by eye on a single case.** Any parameter is validated on the bench, on both datasets.
7. **Same presentation at every iteration:** table of judges + 100 % crops at the same fixed positions + comparison
   with the baseline. Logbook updated at every decision.
8. **Nothing is launched without the user's explicit go.**

## 3. Couplings to examine (information that must not be lost between steps)

| Information | Exists only in / at | Needed by | Coupling option |
|---|---|---|---|
| PSF per frame, per tile, per colour | single frames, sensor frame (before stacking) | deconvolution, star photometry | measure per frame; deconvolve per frame / per tile before stacking, or jointly over frames |
| Sub-pixel phases, Bayer sampling | raw frames | resolution of the colour planes | Bayer drizzle, or multi-frame reconstruction on raw pixels |
| Saturated / unreliable pixels | raw frames | star removal, deconvolution | carry a mask (missing data) instead of filling with invented values |
| Independent noise (two halves) | half-stacks | denoising, validation | keep the halves separate through every step |
| Background components moving differently (sky glow vs flat vs sky) | frame by frame | background model | estimate per frame from their motions, or on the stack if the bench shows no loss |
| Star fluxes / colours | photometry on the data | star layer, colour calibration | Gaia forced photometry with the measured PSF |

## 4. Free shelf candidates (first survey; to be completed at J0)

| Component | Candidates |
|---|---|
| Inverse-problem backbone | **deepinv** (PyTorch; matrix-free operators; known space-varying blur via eigen-PSFs; plug-and-play solvers; custom physics; self-supervised losses). First choice for the deconvolution. Its pretrained denoisers come from natural photos: use our Noise2Noise as the prior |
| Multi-frame deconvolution | torchmfbd (PyTorch, GPU; multi-frame **blind** deconvolution for solar data; space-variant PSF by apodised patches). Inspiration only for the multi-frame / patch machinery: we are **not blind** (see below); dropped if a measured PSF cannot easily be imposed |
| Forward modelling of astro images | **AstroPhot** (PyTorch), scarlet2 (JAX), the Tractor |
| Star removal | StarNet2 (installed in `F:\photos\heart_nebula\tools`, free, closed weights), nox (open, trained on synthetic stars) |
| Background | GraXpert (AI, ONNX), Siril background extraction |
| Denoising | GraXpert denoise, Noise2Noise / Noise2Void implementations, ASTERIS (Science 2026; code to be checked) |
| Deconvolution (single image) | GraXpert stellar / object models, Siril deconvolution |
| Stacking / Bayer SR | Siril 1.4 (HST drizzle, SPCC, Python API), STScI drizzle, Handheld multi-frame SR (raw Bayer) |
| PSF / bright-star wings | photutils EPSF, Piff, PSFEx, Gnuastro PSF scripts (Infante-Sainz), elderflower |
| Masked-data filling | maskfill |
| Colour calibration | Siril SPCC |

**Not blind.** Thousands of point sources with Gaia positions (~0.1 px) and measured fluxes / colours give the PSF per
frame, per tile and per colour directly: the deconvolution is non-blind (at most "myopic": measured PSF, slightly
refined with the star photometry). Subtlety: the PSF measured on stars (continuum SEDs) differs from the PSF of the
nebula light (mostly H-alpha at 656 nm, where the doublet focuses differently); the colour-dependent PSF model (Gaia XP
effective wavelength) is extrapolated to the emission-line wavelengths.

## 5. Validation bench

Reference datasets: **Heart & Soul** (refractor, 721 frames) and **Iris** (Newton, 1242 frames). NGC 7822 and M33
later, for generalisation.

Procedure, step by step: a method is first validated on the reduced batch; only then is it tested on all the frames
and the full stack.

| Level | Frames | Area | Use | Time per test |
|---|---|---|---|---|
| 1. Batch | 100 frames, stratified over the whole night (time) and the FWHM distribution | 4 sensor tiles of 512 px: centre, corner, bright star, nebula | development and go / stop decisions | 15-20 min |
| 2. Full | all frames | whole field (full stack) | validation of what passed level 1; comparison with the user's references | hours |

Why level 1 is enough for some judges and not for others (Heart: 100 of 721 frames, noise x2.7, about 1.1 mag):
reliable for systematic effects (rings, holes, halos, colour shifts, convergence, speed); biased for faint-signal
fidelity, measured resolution limits and regularisation strengths, which are therefore confirmed at level 2. The batch
must span the whole night (drift for sub-pixel phases and fixed-pattern separation, field rotation for sky glow).

**Judges** (identical at every milestone):
1. Prediction of held-out frames (raw pixels), for every method that models the frames.
2. Whiteness of the residuals (structure left = something missing, and where).
3. Injection-recovery in the raw frames (saturated and bright stars, faint stars, filaments, faint extended patches,
   point sources) with each frame's PSF: resolution, flux, rings / holes, faint signal kept.
4. Pan-STARRS (~1"/px, covers both fields) brought to the target PSF: recovered detail is real, not invented.
5. Agreement of the two independent halves (correlation per scale and per zone).

The bench must first reproduce known numbers of the current chain (FWHM, injected fluxes); otherwise it is not used.

Measured fact to keep in mind: per-frame FWHM spread is narrow (Heart p10-p90 3.81-4.33 px, Iris 4.30-5.03); the
gain of per-frame PSF modelling may come more from spatial / colour variation and Bayer sampling than from seeing
changes over the night. J1 / J2 must show it.

## 6. Milestones

Each milestone compares the options (ours, shelf as-is, shelf modified, coupled or not) with the judges, decides, and
records the decision. Level 1 first, level 2 for what passes.

| # | Content | Options compared | Pass criterion (fixed before the test) |
|---|---|---|---|
| J0 | Bench and judges; GitHub survey (activity, weights, fit); baselines | current chain, StarNet2, GraXpert, Siril | bench reproduces known numbers; table "who wins what". **Stop point with the user** |
| J1 | PSF per frame / tile / colour, from Gaia stars, colour-dependent (extrapolated to H-alpha / emission lines) | current stack PSF vs per-frame PSF (ours, photutils / Piff) | predicts held-out stars; variation over frames, field and colour quantified |
| J2 | Deconvolution (non-blind, measured PSF) | on the stack (ours, GraXpert, Siril) / per frame per tile in sensor frame then stack / multi-frame joint (deepinv-based) | gain against Pan-STARRS and in injection, no invented detail, faint signal kept |
| J3 | Denoising, and its coupling with J2 (prior inside the deconvolution or after) | Noise2Noise, GraXpert denoise, others found at J0 | noise down without resolution or faint-signal loss |
| J4 | Stars (separation, bright stars, saturated cores as missing data) | Gaia model subtraction with measured PSF, StarNet2, nox, hybrids | zero rings / holes on injected bright stars; faint diffuse kept; better than StarNet2 |
| J5 | Background | current physical model, per-frame separation by motion, GraXpert | NSNS correlation kept or better; consistent flat |
| J6 | Colour | current (Gaia XP sensor response), Siril SPCC | Gaia star colours recovered |
| J7 | Full data, rendering | best chain from J1-J6 | 100 % sheets of all bright stars and fixed zones; comparison with full.png and the Iris reference |

## 7. Decisions

- Decided: free tools only (open source or closed weights); libraries may be modified; licence follows the libraries;
  chain mastery instead of an imposed single model; reference datasets Heart and Iris; batch of 100 frames + tiles, then
  all frames and full stack; user references full.png and the manual Iris image.
- Open: Pan-STARRS as external truth (internet access).

## 8. History and lessons (2026-10-08 to 2026-10-10)

- 2026-10-08: the chosen "rupture" idea included a physical model on the raw Bayer frames with a PSF per frame and per
  channel (multi-frame deconvolution / super-resolution).
- First attempt (`F:\photos\heart_nebula\sota\code\mfsr.py`) was not that: one Gaussian blur per frame, no
  deconvolution, two short runs, one measure -> "no gain", wrongly generalised.
- Then Richardson-Lucy on the stack (stars 12.9" -> 8.7"); run 2 built the empirical PSF but used it only for star
  photometry, and dropped nebula deconvolution after a half-stack limit measured on faint nebula only.
- astro_GPU (2026-10-10) ported the run-3 chain: no deconvolution; star removal grew into ~8 stacked repairs whose edges
  produce rings around bright stars (Iris central star).
- Lessons: several independent judges, never one; never drop a component without telling the user and relating it to
  the original goal; no repair stacking; no launch without the user's go.

## 9. State of the code (2026-10-10 evening)

- Committed (f6a4543): N2N fine scales in the base, extended PSF template with field sky and physical constraints,
  bright-star zones, residual glare per star, 'Fine detail' factor. Renders v31 exist for Heart, NGC and Iris; the
  Iris central star shows rings (not acceptable as final).
- Uncommitted work in progress, stopped by the user (repair stacking, superseded by this plan): single reconstruction
  region per star, continuous wing lift, ring detector in the render QA.
