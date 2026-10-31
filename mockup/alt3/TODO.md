# To do: what the spec (mockup alt3) does not cover yet

Decided on 2026-10-31: these items wait for a later version. The mockup covers the rest of what Siril does in deep
sky.

## Next version

1. **Mono cameras with filters**
   - One session per filter and one stack per filter.
   - A **Channels** block for LRGB, SHO and HOO composition.
   - The data model goes from one stack to several stacks combined.
2. **Dual-band filters on colour cameras**: H-alpha / OIII extraction.
3. **The filter in the optical path** (light-pollution, dual-band), declared per session and used by the Color block.
4. **Crop**: automatic for the stacking edges, or manual.
5. **Gradient**: place or exclude sample areas by hand, on the image, in Inspect.
6. **Running without the GUI**: run a project from the command line, and a queue of several projects.
7. **Dedicated CMOS cameras**: gain and offset instead of ISO, amp glow, and per-filter masters in the master library.

## Small additions, marked ◇ in the existing blocks

- Registration: interpolation method (Lanczos, bicubic…).
- Calibration:
  - bad pixel map given as a file;
  - CFA equalization of the flats;
  - Canon banding reduction.
- Stacking: rejection maps as an output.
- FITS header viewer.
- Object annotations on the image and at export.
- Rotate, mirror, binning.
- PixelMath, for expert users.
- Presets per block: save and load.
- Per-frame metrics in Selection: eccentricity, PSF SNR, star count, background. A choice of weighting formula.
- Preview on a region: real, not only a button.
- The starless image looks too yellow with the display stretch. Choose linked or unlinked display stretch.

## Later (only their place is noted in the spec)

- Mosaics.
- Comet registration.
- Live stacking.
- Photometry and light curves (not image processing).

## Consequences for the engine (the mockup is its spec)

- The background (Gradient) is extracted **before** Color and Stars, from the stack and the catalogue. Today it is
  extracted after, from the starless image.
- Deconvolution comes **before** Denoise (as BlurX before NoiseX). It is therefore applied to both half-stacks.
- Settings belong to each workflow, not to a common trunk. The cache is keyed by the settings up to each stage, so
  a block shared by several workflows is computed once.
- New methods:
  - Denoise: pretrained + fine-tuned, pretrained, NL-Bayes;
  - Deconvolution: Richardson-Lucy, neural;
  - Gradient: polynomial;
  - Output: stretch choice.
- Post: layers, masks made of parts, filter layers on one layer or on everything below, computed at screen size
  while a slider moves.
- Catalogues offline: local Gaia DR3 and a local solver. The catalogue depth must follow the field.
- A CPU path: measure each stage without a GPU first, then decide where a classic method is needed.
