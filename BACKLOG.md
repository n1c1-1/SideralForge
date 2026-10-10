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

Plan (to discuss before changing defaults):

* measure on matched crops which blue-green structures and fine details of the original are confirmed by both
  half-stacks (real) and which are noise or denoiser texture;
* data-driven colour scales (finer where the signal allows);
* a dedicated OIII / blue-green component with its own boost slider;
* expose the faint-area smoothing and the detail injection in the final adjustment tab (with fast recompute);
* a "structure" control using the Noise2Noise layer more widely.
