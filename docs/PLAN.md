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

## 10. State of the code

- Committed: the processing chain (renders v31 for Heart, NGC 7822 and Iris; the Iris central star shows rings, not
  acceptable as final); this plan.
- Not committed, stopped by the user (stacked repairs, replaced by this plan): changes in `starless.py`, `stars.py`,
  `imgutil.py`, `export.py`.
- Test bench scripts (outside the repository): `C:\Users\Nicolas\astro_gpu_scratch\bench\`.
