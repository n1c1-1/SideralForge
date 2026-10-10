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
