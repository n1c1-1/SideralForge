# Backlog

## Rendering: subtle colours and fine faint detail (feedback 2026-10-10)

Feedback on the Heart & Soul render compared with the original hand processing (full.png):

1. the fine blue-green tones of the original are lost;
2. the original shows sharper detail in the faint, fine nebulosity.

Probable causes (conservative defaults):

* colour discarded below 8 px in the starless Wiener step (`starless.chroma_min_scale = 3`);
* the component that is neither H-alpha nor starlight (OIII, reflection) smoothed at 6 px, then 32 px in faint
  areas (`render_prep.resid_smooth`, `render_prep.faint_smooth`);
* nebula saturation 0.88 by default;
* Noise2Noise fine detail injected only above the 80th emission percentile (`render_prep.fine_lo_pct`);
* sharpening only in high signal-to-noise emission.

Done (render_prep version 2, starless version 2): the choices are now live sliders of the final adjustment tab,
without heavy reprocessing. `render_prep` stores separate layers (emission, continuum, rest, unsmoothed rest,
Noise2Noise emission detail, fine colour) that the engine mixes on the fly:

* **Fine colour** (`colour_detail`): colour finer than 8 px, kept aside by the Wiener step where both half-stacks
  agree, switched off on star footprints;
* **Blue-green / reflection tones** (`resid_boost`): gain on the rest component (OIII, H-beta, reflection);
* **Fine detail** / **Fine detail start** (`fine_k`, `fine_lo_pct`): Noise2Noise detail injection, now down to
  any brightness;
* **Faint area smoothing** (`faint_smooth` and limits).

Default values reproduce the previous render (NGC 7822: median difference 0.002 ADU; remaining differences are
star residuals no longer re-injected by the detail layer). Preset "Rich detail & colour" to calibrate on Heart.

Still to do:

* measure on matched crops which blue-green structures and fine details of the original are confirmed by both
  half-stacks (real) and which are noise or denoiser texture; calibrate the preset accordingly;
* data-driven colour scales (finer where the signal allows);
* a "structure" control using the Noise2Noise layer more widely.

## Robustness on galaxy fields and difficult data (M33 test, 2026-10-10)

M33 (A7S, 900 mm, 20 s at ISO 4000, no darks, no dithering, strong vignetting) exposed:

* colour: at high galactic latitude the stars are too similar to determine the sensitivity curves; the free fit
  gave an unphysical red channel (centred at 576 nm). Fixed: physical check of the curves and camera profiles
  (the profile of the same camera predicted the M33 star colours within 1.8 %);
* colour matrix: done - fitted on the measured stars plus spectra predicted with the sensitivity curves (black bodies
  2500-30000 K, reddened up to E(B-V) = 2); the matrices of the four test fields are now consistent and no longer
  amplify the colour noise;
* walking noise: done - the sensor fixed pattern is measured in sensor coordinates (median, then clipped mean over
  all frames) and removed before drizzling, when the drift allows it. Possible next step: a pattern per time block
  (dark current changes with temperature during the night);
* sky background: done - bounded fit (H-alpha and unresolved light >= 0), objects rejected iteratively, H-alpha
  signature checked against the camera (imposed when the survey correlates with a galaxy or a reflection nebula),
  zero-point floor. Still to do: when a smooth object fills most of the frame its faint wings bias the sky (no
  estimator can tell a smooth halo from a smooth gradient): mask an ellipse of the catalogued size of the target
  (SIMBAD dimensions) in the sky fit;
* 9 % pure black after the tone curve (fixed by the sky fit), pink frame at the edges: the useful frame now
  excludes an effective exposure (coverage x flat^2) below 15 % of the centre.

## Frame selection

* The automatic transparency rule excluded 36 % of the M33 frames: refine the rule and flag large exclusions in
  the interface.

## Generalisation: other sensors and instruments (test data downloaded 2026-10-28 night)

Fifteen public raw sessions are in `D:\astro_donnees_test` (catalogue `CATALOGUE.md`, what the chain reads of
them `LECTURE_PAR_LA_CHAINE.md`): Canon 6D / 40D (CR2), Canon EOS Ra (CR3), Nikon D810A, Fujifilm GFX 50R (Bayer),
Pentax 645Z, ZWO ASI2600MC, two Stellina and two Seestar smart telescopes, ASI1600MM mono in LRGB / RGB+H-alpha /
SHO, the LCO 0.4 m network (mono, B V r' H-alpha) and an Apogee CCD. Still missing: Fuji X-Trans, RASA, SCT.

Found by reading one frame of each with `io/frames.py`:

* compressed FITS (`.fits.fz`, `.fit.gz`) are not listed as frames;
* CR3 and RAF: camera, exposure and ISO not read (exifread does not parse them);
* Stellina writes EXPTIME in milliseconds (read as 10 000 s);
* Pentax 645Z: black level 0, the background noise is clipped at zero in the raw file;
* monochrome cameras: no colour pattern, not processed (one stack per filter, then colour combination: the largest
  piece of work);
* smart telescopes on alt-azimuth mounts: the field rotates during the night (registration and stacking to check).

First pass on nine test sets, 2026-10-30 night (PLAN 11.18; sets in `F:\astro_essais`, `tools/qc/jeux.py`). Fixed:
compressed FITS, CR3 / RAF metadata, ms exposures, turned fields, colour without a camera profile, star flat under
coma, H-alpha amplitude collapse in emission fields, catalogue / name / survey services down. Still open:

* stars with coma (Newtonian without corrector, Canon 6D) leave white / black blobs in the starless layer;
* smart telescopes (Seestar, Stellina): star-model residual 23-55 % in red and blue (undersampled stars, FWHM
  1.7 px) - stars left orange in the starless layer of IC 1318, its nebulosity lost; not the drizzle (without it
  the residual is 31-36 %);
* the H-alpha boost of the rendering (R - alpha G) takes a large-scale red excess of the sky for emission (Seestar
  M31); telling a line from a red continuum needs the blue channel or the survey;
* astrometry with an unknown scale queries Gaia once per trial scale (22 scales, ~1 min each while VizieR is down);
  the local blind solver could give the scale first;
* dense Milky Way fields (M8) and wide fields (135 mm lens: 234 catalogue cells): the catalogue stage is
  impractical without VizieR; its depth (G 18.5 / 20.5) should follow the field and the exposure;
* Stellina NGC 891 (frames already processed by the telescope, light-pollution filter): magenta sky, colour fit 8 %.

## Reproducibility: the learned star removal is trained for a time, not a number of steps (found 2026-10-29)

`star_net` trains for `minutes` (32 by default), so the number of training steps depends on what else uses the GPU:
two runs of the Heart on identical inputs (same star model, bit for bit) made 11 181 and 11 408 steps and gave
different star layers; the renders then differ by up to ~60 levels on ~2 % of the pixels (inside the nebulae, at
star positions). Fix: train for a fixed number of steps (the time setting converted once), stage version raised -
every project using the learned layer would then retrain once.
