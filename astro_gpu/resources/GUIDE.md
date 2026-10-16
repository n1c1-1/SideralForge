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

## 4. Methods and variants

No single processing suits every instrument and object. The **Methods** tab lists the choices of method in plain
words - how the stars are removed (star model or a network trained on the image), the glare of the bright stars,
the shape of the stars, the faint stars, the flat correction, the H-alpha reference, the repairs around the stars,
the sensor pattern, the colour fringes - each option with what it does and what was seen on real images.

* **Processing shown** (top): *main* is the project's own processing; the others are **variants**, the same data
  processed another way, kept side by side. The processing shown is the one the Processing tab, Final adjustment
  and the exports use (its name is in the toolbar).
* Choose another option of a decision to see what it would cost (stages computed again, time, disk), then
  **Try as a new variant** (computed in its own folders, main untouched) or **Apply** to the processing shown.
* A variant shares with main every stage before the ones it changes: only what differs is computed and stored.
* **Compare...** renders the same zone of every computed processing at 100 %, with the current Final adjustment
  settings; click the overview to choose the zone.
* **Make it main** keeps a variant as the project's processing (the former main is kept as a variant until you
  delete it); **Delete variant** frees its disk space. A variant's exports get its name in their folder name.
* Choices that need no computing (star removal used, spikes as photographed, sharpening, glare) are made in Final
  adjustment.

## 5. Final adjustment

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

**Views** (the bar above the image): a view is a rendering of the image saved under a name - the processing (main
or a variant, see the Methods tab) and these settings. No single rendering suits every part of an image: keep at
least one good view per zone.

* The first views are your delivered renderings (one row each) and, beside each, the alternatives that cost no
  computing: the glare of the bright stars drawn with the stars, the spikes drawn, the other star removal.
* Click a view to show it at the same zoom and place; **Left / Right**: previous / next view; **B**: back to the view
  shown before (press it repeatedly to flip between two views).
* **+ New view** saves the current settings and processing; right click a view to update it with the current
  settings, rename, export or delete it. A star (*) marks a view whose settings were changed since it was saved.
* **Side by side...** renders the same zone of the views at 100 %.
* A variant computed in the Methods tab gets its own view.

Tools:

* **Presets**: *Natural*, *H-alpha emphasised*, *Prominent stars*, *Dense Milky Way*, *Starless*.
* **Snapshot** keeps the current image; **Compare with snapshot** shows before / after with a draggable divider.
* **Compare with reference image** shows your own version aligned under the same divider.
* **Show clipping**: blue = pure black, red = pure white, amber = nebula above 92 % (loss of detail).
* The **checks** under the histogram turn amber when something is clipped or burnt.
* **Undo / Redo**: Ctrl+Z / Ctrl+Y.

## 6. Export

**Export...** writes, in `results/<name>/`:

* a 16-bit TIFF at full (drizzled) resolution with an sRGB ICC profile,
* a 16-bit PNG and a JPEG at the sensor scale,
* the starless and stars-only layers (recombine with the stars on top in *Screen* mode),
* the image with the framing of your reference image, if any,
* `<name>_settings.json`: every setting used and the automatic quality checks.

Orientation: *as captured*, *north up* (from the astrometry) or *match reference*; always an exact rotation by
multiples of 90°, never resampled.

## 7. Command line

Everything is also available from the command line (`astro-gpu --help`): `new`, `run`, `status`, `stop`, `params`,
`set`, `render`, `variant` (list, create, activate, delete, promote); `run`, `status`, `params`, `set` and `render`
take `--variant NAME` (`main` for the project's own processing).
