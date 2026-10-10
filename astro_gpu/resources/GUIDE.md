# astro_GPU user guide

## 1. Create a project

**New project** (Ctrl+N), then choose the folder of your light frames. The dark, flat, bias and flat-dark folders
next to it are found automatically when they are named `dark(s)`, `flat(s)`, `bias` / `offset(s)`, `flat_dark(s)`.

* **Project folder**: results, intermediate files and the journal (a few GB).
* **Fast scratch folder**: the calibrated frames cache, read many times. Put it on an internal SSD; the dialog shows
  the space needed.
* **Target**: the object name is resolved by SIMBAD to find the field centre. With FITS files that contain the
  pointing, it is not needed.
* **Reference image** (optional): your own earlier processing of the same data. It is aligned automatically for
  side-by-side comparison and for exports with exactly the same framing.

Only colour cameras (DSLR, mirrorless, one-shot colour astro cameras) are supported in this version.

## 2. Run the processing

**Run / resume** (F5) computes every stage that is not up to date, in order. You can close the interface: the
processing continues in the background, and reopening the project re-attaches to it.

**Stop** (Shift+F5) stops at the next checkpoint. Run again later: the processing resumes where it stopped. A
calibration resumes frame by frame, and the denoiser training resumes at its last step.

Each stage shows a symbol:

| Symbol | Meaning |
|---|---|
| ✔ green | up to date |
| ↻ amber | must be recomputed: never run, parameters changed, or an earlier stage was recomputed |
| ⏸ | interrupted, will resume |
| ✖ red | failed; the error is shown in the stage page and in the journal |
| ○ | waiting for earlier stages |

## 3. Parameters

Select a stage to see what it does, its parameters, its measured results and its previews.

* **Basic / Advanced**: the basic mode shows the few parameters that matter most; *Advanced settings* shows all.
* **auto** (green): the value was computed from your data. Hover it to read why, for example "FWHM 4.8 px with 446
  frames: under-sampled, enough frames to drizzle".
* **yours** (amber): you changed it. The ↺ button returns to the automatic value.
* Click or hover a parameter to read, in the help panel, what it does, when to change it and its range.

Changing a parameter marks the stage and everything after it for recomputation. Nothing is lost: only what depends on
the change is recomputed.

## 4. Final adjustment

Once *Rendering preparation* is done, the **Final adjustment** tab re-renders the image on the GPU in a fraction of
a second for every slider. The preview is computed at reduced resolution while you move a slider, and at full
resolution when you zoom in (double click or **1** for 100 %, **F** to fit).

* **Colour**: calibrated white balance (5778 K = the Sun is neutral), nebula saturation, H-alpha boost and hue.
* **Sky background**: sky and dark-cloud levels, shadow softness (the cores of dark globules keep their gradation),
  Milky Way glow (light of the stars too faint to be drawn).
* **Nebula**: brightness, large-scale compression (keeps the core of bright nebulae detailed), local contrast at
  three scales, sharpening (only where the signal is strong).
* **Highlights**: logarithmic shoulder with re-injected local detail: bright cores never burn out.
* **Stars**: faintest drawn star, sizes of G=10 and G=7 stars, colour, white cores. Stars are drawn with the real
  profile of your optics, measured on the image.

Tools:

* **Presets**: *Natural*, *H-alpha emphasised*, *Prominent stars*, *Dense Milky Way*, *Starless*.
* **Snapshot** keeps the current image; **Compare with snapshot** shows before / after with a draggable divider.
* **Compare with reference image** shows your own version aligned under the same divider.
* **Show clipping**: blue = pure black, red = pure white, amber = nebula above 92 % (loss of detail).
* The **checks** under the histogram turn amber when something is clipped or burnt.
* **Undo / Redo**: Ctrl+Z / Ctrl+Y.

## 5. Export

**Export...** writes, in `results/<name>/`:

* a 16-bit TIFF at full (drizzled) resolution with an sRGB ICC profile,
* a 16-bit PNG and a JPEG at the sensor scale,
* the starless and stars-only layers (recombine with the stars on top in *Screen* mode),
* the image with the framing of your reference image, if any,
* `<name>_settings.json`: every setting used and the automatic quality checks.

Orientation: *as captured*, *north up* (from the astrometry) or *match reference*; always an exact rotation by
multiples of 90°, never resampled.

## 6. Command line

Everything is also available from the command line (`astro-gpu --help`): `new`, `run`, `status`, `stop`, `params`,
`set`, `render`.
