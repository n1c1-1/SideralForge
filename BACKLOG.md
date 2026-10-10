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

## Frame selection

* The automatic transparency rule excluded 36 % of the M33 frames: refine the rule and flag large exclusions in
  the interface.
