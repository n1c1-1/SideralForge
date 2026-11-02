# Blocks

One section per block, in graph order. Taken from `mockup/alt3` on 2026-10-31: `project.py` (stages and methods),
`steps.py` (images and measurements), `settings.py` and `panel.py` (the settings catalogue), `blocks.py` (summaries
and side inputs).

## How to read the tables

- **stage.param** is the engine name.
  - A NEW setting is given the name proposed for the engine. The mockup stores it as `new_<name>`.
  - `render_prep(_learned)` means `render_prep`, or `render_prep_learned` when the neural star removal is used. The
    mockup writes this `@render`.
- **Type**:
  - `optfloat`: a number, or empty for "auto";
  - `zone`: areas drawn on the image;
  - `str`: free text, or a file.
- **Range**: the engine's bounds for an existing setting, the mockup's bounds for a NEW one.
- **Suggested**: the value the engine suggested on the Heart project. Many values are computed from the data, each
  with a one-line reason.
- **Level**:
  - normal: always shown;
  - expert: shown under "Advanced settings (N)";
  - system: shown under "System: speed and memory", in expert mode. The engine has only `basic` and `advanced`; the
    `system` level is the GUI's.
- **Durations**: measured on Heart (RTX 4060 Ti). NEW stages are estimated (SPEC.md 7).

Totals:
- 96 existing settings and 41 NEW settings. The catalogue has 43 NEW entries, because Stretch and Linked channels are
  declared once for each output stage.
- 10 methods, 6 of them NEW.

Five engine parameters are not in the catalogue:
- `stack.min_transparency` and `stack.max_fwhm_ratio`: they are the Selection thresholds.
- `reference.max_stars`: the stage `reference` is not shown in the GUI.
- `star_net.enabled`: it is replaced by the Stars method.
- `starless.learned_stars`: an internal file path.

### Methods

| Block | Method | Short name | Runs on | Stage | Status |
|---|---|---|---|---|---|
| Gradient | H-alpha survey | H-alpha survey | CPU | `background` (+ `surveys`) | existing |
| Gradient | Polynomial (as in Siril) | polynomial | CPU | `fond_simple` | **NEW** |
| Deconvolution | Measured PSF (Richardson-Lucy) | Richardson-Lucy | CPU or GPU | `deconv` | **NEW** |
| Deconvolution | Neural (BlurXTerminator-like) | neural | CPU or GPU | `deconv` | **NEW** |
| Denoise | Noise2Noise on this image | Noise2Noise | GPU | `denoise` | existing |
| Denoise | Pretrained + fine-tuned | fine-tuned | CPU or GPU | `denoise_ft` | **NEW** |
| Denoise | Pretrained (GraXpert-like) | pretrained | CPU or GPU | `denoise_ai` | **NEW** |
| Denoise | NL-Bayes (as in Siril) | NL-Bayes | CPU | `denoise_cl` | **NEW** |
| Stars | PSF model (Gaia) | PSF model | CPU or GPU | `optics`, `stars`, `starless` | existing |
| Stars | PSF model + neural removal | PSF + neural | GPU | the same + `star_net` | existing |

The Output's stretch choice is a NEW setting, not a method.

---

## 1. Frames

**Purpose.**
- The input files of the project: lights and calibration frames.
- Their grouping into sessions (nights).
- Their match with calibration frames and masters.
- A health check of the data set.
- A frame browser.

**Inputs.** Folders of lights, darks, flats, bias and dark flats, and the master library.

**Stages.** None of its own. The inventory runs in Calibration.

**Graph summary.** "to add" before the import, then "N session(s) · N lights". The block is drawn dashed until frames
exist.

**Panel.**
- Before the import: Import folder….
- After the import:
  - one line per session: night, "N lights · N flats · darks: library";
  - a warning when the camera was rotated;
  - Excluded by hand: N;
  - Stacking: all sessions together, or one stack per session;
  - Open frames and Add session….

**Outputs in Inspect.**
- The Sessions table: Session, Night, Lights, Exposure, ISO / gain, Temp., Filter, Flats, Darks, Bias, FWHM, Kept.
- The health check.
- The stacking mode.
- Browse: groups, grid, viewer, list, blink, keep or exclude, test area (SPEC.md 3.4).

**Methods.** None.

**Settings.** None in the catalogue. The project-level decisions live here:
- frames excluded by hand (lights and calibration frames);
- the grouping criteria;
- the stacking mode;
- the use of library masters.

**Needs from the engine.**
- Read the header of each file:
  - type, exposure, ISO or gain, sensor temperature, time;
  - camera, filter, binning, rotation, object, RA / Dec.
- Sort the files by type, then group them into sessions with the chosen criteria.
- Match the calibration frames, the master library included, with temperature and age tolerances.
- The health checks, and the disk space needed.
- Thumbnails and viewer images, with one stretch shared by all the lights, so that a hazy frame looks darker when
  blinking.
- Add a session to an existing project.
- Take a list of frames excluded by hand as an input.

---

## 2. Calibration

**Purpose.**
- Builds the dark and flat masters.
- Calibrates each light: the thermal signal, the vignetting and the defective pixels are removed.
- There is no debayering: Stacking works on the sensor's colour pixels.

**Inputs.** The frames of each session, the library masters, and the project-level exclusions.

**Stages.** `inventory` (5 s), `calibrate` (5 min).

**Graph summary.** "28 darks, 46 flats, no bias".

**Images in Inspect.** Master flat; A calibrated frame; The same frame, raw.

**Measurements** (the engine metric is in brackets).
- Vignetting: "the corner gets 36 % of the center light" (`vignetting_min`).
- Defective pixels: "p %, of which N hot found in the lights" (`bad_pixels_pct`, `hot_pixels_from_lights`).
- Dark level: "L ADU, noise N ADU" (`dark_level`, `dark_frame_noise`).
- Sky noise of one frame, in ADU (`sky_noise_adu`).
- Calibrated frames in cache, in GB (`cache_gb`).

**Methods.** None.

**Settings.**

| Group | Label | stage.param | Type | Range / choices | Suggested | Level | Status |
|---|---|---|---|---|---|---|---|
| Masters | Dark combination | `calibrate.darks_combine` | choice | Average with rejection / Median / Average | Average with rejection | normal | **NEW** |
| Masters | Dark scaling | `calibrate.dark_scaling` | bool | — | no | normal | **NEW** |
| Cosmetic correction | Hot pixel threshold | `calibrate.hot_sigma` | float | 0.1–10 step 0.1 (× sky noise) | 0.6 | normal | existing |
| Cosmetic correction | Auto-detect hot pixels in the lights | `calibrate.hot_from_lights` | bool | — | yes | normal | existing |
| Cosmetic correction | Correct cold pixels too | `calibrate.cold_pixels` | bool | — | no | normal | **NEW** |
| Flats | Flat defect threshold | `calibrate.flat_defect` | float | 0.05–0.9 step 0.05 | 0.3 | normal | existing |
| Flats | Flat: large scales only | `calibrate.flat_scale_px` | float | 0–500 step 5 (px) | 0.0 | expert | existing |
| Sensor | Saturation level | `calibrate.saturation` | float | 0.5–1 step 0.001 (× white level) | 0.976 | expert | existing |
| Sensor | Limit the number of frames | `inventory.max_lights` | int | ≥ 0 (0 = all) | 0 | expert | existing |
| System | Parallel workers | `calibrate.workers` | int | 1–64 | 12 | system | existing |
| Debayer | Debayer method | `calibrate.debayer` | choice | RCD / VNG / AHD / Bilinear | RCD | normal | **NEW** |
| Debayer | CFA pattern | `calibrate.cfa_pattern` | choice | From header / RGGB / BGGR / GRBG / GBRG | From header | expert | **NEW** |
| Flat calibration | Flat offset | `calibrate.flat_offset` | choice | Bias from library / Dark flats / Camera black level | Bias from library | normal | **NEW** |
| Flat calibration | Output pedestal | `calibrate.pedestal` | int | 0–1000 step 10 (ADU) | 0 | expert | **NEW** |

**Needs from the engine.**
- The master flat as a preview.
- One calibrated frame and the same frame raw, with the same stretch.
- The five metrics.
- The NEW settings: the dark combination and dark scaling, cold pixels, the debayer method (not used with Bayer
  drizzle), the CFA pattern override, the flat offset source, and the pedestal.
- Masters taken from the library.

---

## 3. Registration

**Purpose.** Measures the stars of each frame (size and transparency), then aligns every frame on the sharpest one.

**Stages.** `register` (2 min).

**Graph summary.** "on the sharpest frame".

**Images.** Frame quality.

**Measurements.**
- Reference frame (`reference_frame`).
- Registered frames (`aligned`).
- Registration residual, in px (`rms_px`).
- Star size: "a to b px on 80 % of the frames" (`fwhm_p10_p90`).
- Drift over the night, in px (`drift_px`).

**Methods.** None.

**Settings.**

| Group | Label | stage.param | Type | Range / choices | Suggested | Level | Status |
|---|---|---|---|---|---|---|---|
| Reference | Reference frame | `register.reference` | choice | Automatic: the sharpest / Picked in the sources | Automatic: the sharpest | normal | **NEW** |
| Detected stars | Star detection threshold | `register.detect_sigma` | float | 3–50 (σ) | 8.0 | normal | existing |
| Detected stars | Stars per frame | `register.max_stars` | int | 100–20000 | 2500 | expert | existing |
| Distortion | Transformation model | `register.transform` | choice | Automatic / Rotation and translation / Homography / Polynomial, degree 2 / Polynomial, degree 3 | Automatic | normal | **NEW** |
| System | Parallel workers | `register.workers` | int | 1–64 | 12 | system | existing |
| Distortion and framing | Distortion correction | `register.distortion` | choice | Automatic / None / SIP polynomial / Thin-plate splines | Automatic | normal | **NEW** |
| Distortion and framing | Framing | `register.framing` | choice | Reference frame / Union (maximum) / Intersection (minimum) / Centre of gravity | Reference frame | normal | **NEW** |

**Needs from the engine.**
- A per-frame table: name, time, FWHM, transparency, residual and status. Selection's chart reads it.
- The reference frame chosen in Browse.
- The transformation models.
- The interpolation method, still to come (TODO ◇).

---

## 4. Selection

**Purpose.** Chooses which frames go into the stack, and how much each one weighs. Frames are chosen by thresholds
on star size and transparency, or one by one.

**Stages.**
- None of its own. Selection reads the per-frame measurements of Registration.
- Its thresholds are Stacking settings (`stack.*`).
- Its state follows Registration.
- Two workflows with different thresholds split here.

**Graph summary.** "transparency ≥ 60 % · stars ≤ 8.0 px".

**Inspect.** The frames chart, instead of an image.
- Each point is a frame, placed by its FWHM and its transparency.
- The two thresholds can be dragged; the suggested thresholds are marked.
- Frames excluded by hand are shown.
- A click opens the raw frame, with its Time, FWHM, Transparency (in % of the 90th percentile of the frames) and
  State, and Keep / Exclude.
- Under the chart: "k of n frames kept · h".

**Measurements.**
- Frames kept: "k of n".
- Integration, in h.
- Minimum transparency: "% of the best frames".
- Maximum star size, in px.
- Excluded by hand: N.

**Panel thresholds.** These are controls, not catalogue rows. They are stored together, per workflow, as
`stack.selection = (t %, f px)`.

| Label | Engine parameter | Shown as | Engine range | Suggested | Status |
|---|---|---|---|---|---|
| Minimum transparency | `stack.min_transparency` | % of the best, 0–100, step 1 | 0–1 step 0.05 | 0.6 → 60 % | existing |
| Maximum FWHM | `stack.max_fwhm_ratio` | px, 1–20, step 0.1 | 1–5 step 0.1 (× median FWHM) | 2.0 → 2 × median | existing |

**Methods.** None.

**Settings.**

| Group | Label | stage.param | Type | Range / choices | Suggested | Level | Status |
|---|---|---|---|---|---|---|---|
| Other criteria | Maximum background | `stack.max_background` | float | 0–5 step 0.1 (× median) | 0.0 (off) | normal | **NEW** |
| Other criteria | Minimum star roundness | `stack.min_roundness` | float | 0–1 step 0.05 | 0.0 (off) | normal | **NEW** |
| Other criteria | Keep the best | `stack.keep_best` | int | 10–100 step 5 (% of frames) | 100 | normal | **NEW** |
| Weighting | Frame weights | `stack.weighting` | choice | Transparency and star size / Signal-to-noise ratio / Equal for all | Transparency and star size | normal | **NEW** |

**Needs from the engine.**
- Per-frame FWHM and transparency, and, for the NEW criteria, the background, roundness and SNR of each frame. More
  per-frame metrics will follow: eccentricity, PSF SNR, star count (TODO ◇).
- The thresholds given in GUI units: px, and % of the best. The engine stores a ratio to the median and a fraction.
- The list of frames excluded by hand, as an input of the stack.
- The kept count for given thresholds. The client can compute it from the per-frame table.

---

## 5. Stacking

**Purpose.**
- Stacks the frames straight from the sensor's colour pixels (drizzle without debayering).
- Rejects trails and outlier pixels.
- When the workflow has Denoise, two half-stacks are made with the stack, from alternate blocks of frames, at no
  extra time.

**Stages.** `stack` (34 min).

**Graph summary.** "drizzle ×1.5 · rejection at 5 σ". It adds " · + half-stacks" when a workflow through the block
uses Denoise.

**Images.** Stack (stretched for display).

**Measurements.**
- Stacked frames: "N (h h)" (`frames_used`, `integration_used_h`).
- Stack size: "W × H px" (`grid`).
- Rejected pixels: "p % (trails, cosmic rays)" (`rejected_pct`).
- Half-stacks: "two, of about 360 frames each", or "none".

**Other controls.**
- The output port ●, to Post (SPEC.md 5.5).
- With more than one session, Sessions: All together, or One per session. One per session makes one variant per
  session, with `stack.sessions = (session id,)`.

**Methods.** None.

**Settings.** The last row is shown only when the workflow has Denoise.

| Group | Label | stage.param | Type | Range / choices | Suggested | Level | Status |
|---|---|---|---|---|---|---|---|
| Resolution | Drizzle scale | `stack.scale` | float | 1–3 step 0.25 (×) | 1.5 | normal | existing |
| Resolution | Pixfrac | `stack.pixfrac` | float | 0.3–1 step 0.05 | 0.7 | normal | existing |
| Pixel rejection | Rejection method | `stack.rejection` | choice | Automatic: around the median / Sigma clipping / Winsorized sigma clipping / Linear fit clipping / Generalized ESD | Automatic: around the median | normal | **NEW** |
| Pixel rejection | Rejection threshold | `stack.kappa` | float | 2–20 step 0.5 (σ) | 5.0 | normal | existing |
| Pixel rejection | Separate low and high thresholds | `stack.split_thresholds` | bool | — | no | normal | **NEW** |
| Pixel rejection | Star core tolerance | `stack.core_tolerance` | float | 0–2 step 0.1 | 0.5 | expert | existing |
| Normalization | Frame normalization | `stack.normalization` | choice | Automatic: offset and scale / Offset only / None | Automatic: offset and scale | normal | **NEW** |
| Sensor | Sensor gain | `stack.gain` | float | ≥ 0 (e⁻/ADU) | 0.1531 | expert | existing |
| Sensor | Read noise | `stack.read_noise` | float | ≥ 0 (ADU) | 11.86 | expert | existing |
| Fixed sensor pattern | Fixed pattern removal | `stack.pattern_removal` | bool | — | no | expert | existing |
| Fixed sensor pattern | Frames for the reference pattern | `stack.pattern_frames` | int | 8–400 | 48 | expert | existing |
| Advanced stacking | Local normalization | `stack.local_norm` | bool | — | no | normal | **NEW** |
| Advanced stacking | Drizzle kernel | `stack.drizzle_kernel` | choice | Square / Point / Turbo / Gaussian / Lanczos-3 | Square | expert | **NEW** |
| Half-stacks | Half-stack block | `stack.half_block` | int | 1–200 (frames) | 20 | expert | existing |

**Needs from the engine.**
- The stack, as a preview and as linear data (for auto-stretch, the histogram and the ADU readout).
- The metrics.
- The half-stacks only when a workflow needs them.
- A stack of a subset of sessions (`stack.sessions`).
- The rejection maps as an output (TODO ◇).

---

## 6. Astrometry

**Purpose.**
- Finds where the image points on the sky, by matching its stars to Gaia.
- Then fetches the catalogue stars, and the H-alpha survey when the Gradient uses it.

**Stages.**
- `astrometry` (2 min) and `catalogs` (3 min).
- `surveys` (43 s), only with the H-alpha Gradient. Its time counts here.

**Graph summary.** "plate solved · Gaia DR3".

**Images.**
- Identified objects: the stack with circles and names, such as "IC 1805 · Heart Nebula".
- Reference H-alpha survey (with the H-alpha Gradient only).

**Measurements.**
- Center: α and δ (`ra`, `dec`).
- Field (`field_deg`).
- Scale: "s″ per pixel" (`scale_arcsec`).
- Gaia stars used: "N, residual r px" (`stars_used`, `residual_px`).
- Measured focal length, in mm (`focal_estimate_mm`).

**Side outputs.**
- Gaia stars, to Stars.
- Gaia spectra, to Color.
- The H-alpha survey, to Gradient.

**Methods.** None.

**Settings.**

| Group | Label | stage.param | Type | Range / choices | Suggested | Level | Status |
|---|---|---|---|---|---|---|---|
| Initial position | Object name | `astrometry.target` | str | — | IC 1805 | normal | existing |
| Initial position | Center: right ascension | `astrometry.ra` | optfloat | 0–360 (°) | auto | normal | existing |
| Initial position | Center: declination | `astrometry.dec` | optfloat | -90–90 (°) | auto | normal | existing |
| Initial position | Focal length | `astrometry.focal_mm` | float | ≥ 0 (mm) | 272.0 | normal | existing |
| Plate solving | Distortion order | `astrometry.sip_order` | int | 0–5 | 3 | expert | existing |
| Plate solving | Blind solve | `astrometry.blind` | bool | — | yes | expert | existing |
| Plate solving | Center search radius | `astrometry.search_offsets` | float | 0–1 step 0.05 (× field) | 0.2 | expert | existing |
| Plate solving | Pixel size | `astrometry.pixel_um` | float | ≥ 0 (µm) | 8.4 | expert | existing |
| Plate solving | Image scale | `astrometry.scale` | optfloat | ≥ 0 (″/px) | auto | expert | existing |
| System | Query cell size | `catalogs.cell_deg` | float | 0.1–5 (°) | 1.0 | system | existing |

**Needs from the engine.**
- The WCS, for the RA / Dec readout in Inspect and for north-up export.
- The list of identified objects (name, x, y) in stack coordinates.
- The survey, registered on the image.
- An offline Gaia DR3 and a local solver (Settings > Catalogs). The catalogue depth must follow the field (TODO).

---

## 7. Gradient

**Purpose.**
- Removes the sky gradient (light pollution, the Moon, what is left of the vignetting).
- The spec places it **before** Color and Stars, as in Siril and PixInsight, and computes it from the stack and the
  catalogue. The engine today computes it after, from the starless image.

**Stages.**
- `background` (45 s) for the H-alpha method.
- `fond_simple` (NEW, ≈ 1 min) for the polynomial method.

**Graph summary.** The method's short name.

**Images.**
- H-alpha: Background.
- Polynomial: "Stack with a simple background (preview)".

**Measurements.**
- H-alpha:
  - Image used as sky, in % (`sky_fit_blocks_pct`);
  - H-alpha survey: "registered on this image";
  - Dips repaired under the stars (`emission_dips.below_8sigma`).
- Polynomial:
  - Method: "degree-2 surface on the dark areas";
  - Duration: "≈ 1 min (estimate)".

**Side input.** The H-alpha survey, from Astrometry.

**Methods.** H-alpha survey (CPU, existing); Polynomial (CPU, NEW).

**Settings.**
- The rows depend on the method.
- For an "Areas to exclude" zone, "Draw on the image…" opens the stack at full size. The areas are drawn with a brush
  or a lasso, and the background is computed again without them.

| Group | Label | stage.param | Type | Range / choices | Suggested | Level | Status |
|---|---|---|---|---|---|---|---|
| H-alpha · Reference survey | Reference H-alpha survey | `surveys.survey` | choice | auto, NSNS, Finkbeiner, none | auto | normal | existing |
| H-alpha · Reference survey | Minimum survey coverage | `surveys.min_coverage` | float | 0–1 step 0.05 | 0.95 | expert | existing |
| H-alpha · Sky gradient | Polynomial degree | `background.sky_order` | int | 1–5 | 3 | normal | existing |
| H-alpha · Sky gradient | Residual flat correction (from the stars) | `background.star_flat` | bool | — | no | normal | existing |
| H-alpha · Sky gradient | Sky sampling | `background.bin_px` | int | 8–256 (px) | 32 | expert | existing |
| H-alpha · Sky gradient | Large-scale correction | `background.large_scale_px` | float | 0–2000 (px) | 200.0 | expert | existing |
| H-alpha · Sky gradient | Object rejection | `background.object_clip` | float | 1–10 step 0.1 (σ) | 2.5 | expert | existing |
| H-alpha · Samples | Areas to exclude | `background.exclusions` | zone | areas drawn on the image | none | normal | **NEW** |
| H-alpha · Samples | Show sample points | `background.show_samples` | bool | — | no | normal | **NEW** |
| Polynomial · Sky gradient | Surface degree | `fond_simple.degree` | int | 1–5 step 1 | 2 | normal | **NEW** |
| Polynomial · Sky gradient | Samples per side | `fond_simple.samples` | int | 4–64 step 2 | 16 | normal | **NEW** |
| Polynomial · Samples | Areas to exclude | `fond_simple.exclusions` | zone | areas drawn on the image | none | normal | **NEW** |
| Polynomial · Samples | Show sample points | `fond_simple.show_samples` | bool | — | no | normal | **NEW** |

**Needs from the engine.**
- The background model, as a preview.
- The polynomial stage.
- Exclusion areas as an input.
- The sample points as an output, so that they can be shown, moved or removed.
- The stage moved before Color and Stars.

---

## 8. Color

**Purpose.**
- Measures the sensor's response on the Gaia spectra of the field stars.
- Then computes the matrix to the true sky colours (SPCC).

**Stages.** `color` (3 s).

**Graph summary.** "SPCC · Gaia spectra".

**Images.**
- Sensor response: a chart of the R, G and B response against wavelength.
- Engine preview.

**Measurements.**
- Color calibration stars (`stars`).
- Deviation from Gaia colors, in % (`fit_agreement_pct`).
- H-alpha response (656 nm): "red r: astro-modified camera" (`response_at_lines["656nm"][0]`).

**Side input.** The Gaia spectra, from Astrometry.

**Methods.** None.

**Settings.**

| Group | Label | stage.param | Type | Range / choices | Suggested | Level | Status |
|---|---|---|---|---|---|---|---|
| White reference | White reference | `color.white_ref` | choice | The Sun / Average of the field stars / Average spiral galaxy / Custom temperature | The Sun | normal | **NEW** |
| White reference | White temperature | `color.white_temp` | float | 3000–12000 step 100 (K) | 5778 | normal | existing |
| White reference | Background neutralization area | `color.neutral_zone` | choice | Automatic: nebula-free / Drawn on the image | Automatic: nebula-free | normal | **NEW** |
| Color measurement | Photometric aperture | `color.aperture` | float | 1.5–30 step 0.5 (px) | 4.2 | expert | existing |
| Color measurement | Camera profile | `color.camera_profile` | bool | — | yes | expert | existing |
| Color measurement | Synthetic spectra weight | `color.synthetic_weight` | float | 0–5 step 0.1 | 0.5 | expert | existing |
| Color measurement | Profile agreement | `color.profile_tolerance` | float | 1–20 step 0.5 (%) | 5.0 | expert | existing |
| Color measurement | H-alpha anchor weight | `color.hii_weight` | float | 0–500 | 60.0 | expert | existing |
| Color stars (Gaia) | Color stars: brightest | `catalogs.xp_gmin` | float | 3–15 step 0.5 (G) | 7.5 | expert | existing |
| Color stars (Gaia) | Color stars: faintest | `catalogs.xp_gmax` | float | 8–17.5 step 0.5 (G) | 14.0 | expert | existing |
| Color stars (Gaia) | Color stars: number | `catalogs.xp_max` | int | 100–20000 | 4000 | expert | existing |

**Needs from the engine.**
- The response curves as data, so that the client draws the chart: wavelengths, and R, G, B.
- The matrix and the metrics.
- The white reference choices and a drawn neutral area (NEW).
- Later, the filter in the optical path, declared per session (TODO).

---

## 9. Deconvolution (NEW block)

**Purpose.**
- Sharpens the image with the point spread function measured on the stars.
- It goes after Color and before Denoise, as BlurXTerminator goes before NoiseXTerminator.
- It is applied to both half-stacks, so that Denoise still works.

**Stages.** `deconv` (NEW, ≈ 2 min 30 s; × 6 on the CPU).

**Graph summary.** "Richardson-Lucy" or "neural".

**Images and measurements.** None in the mockup: Inspect says "No image for this block". To define. A
before / after view at 100 % and a FWHM before and after would match the other blocks.

**Methods.** Measured PSF (Richardson-Lucy) and Neural (BlurXTerminator-like). Both are NEW and run on the CPU or the
GPU.

**Settings.**

| Group | Label | stage.param | Type | Range / choices | Suggested | Level | Status |
|---|---|---|---|---|---|---|---|
| Deconvolution | Iterations | `deconv.iterations` | int | 5–200 step 5 | 30 | normal | **NEW** |
| Deconvolution | Correct the stars too | `deconv.stars` | bool | — | yes | normal | **NEW** |
| Deconvolution | Regularization | `deconv.regularization` | float | 0–0.05 step 0.001 | 0.002 | expert | **NEW** |

**Needs from the engine.**
- The stage, with both methods.
- A PSF measured across the field **before** Denoise. Today the `optics` stage, which measures the PSF, depends on
  `denoise`.
- Its output must cover both half-stacks.

---

## 10. Denoise

**Purpose.**
- Removes the noise without inventing detail.
- By default (Noise2Noise), a network learns, on this image alone, to predict one half-stack from the other.

**Stages.** One per method:
- `denoise` (31 min);
- `denoise_ft` (NEW, ≈ 3 min);
- `denoise_ai` (NEW, ≈ 45 s);
- `denoise_cl` (NEW, ≈ 5 min, CPU).

**Graph summary.** "Noise2Noise · 24 000 it.", or the method's short name.

**Images.**
- Before | after, at 100 %: a crop with the labels "Before: the stack" and "After: denoised".
- Engine preview.

**Measurements.**
- Sky noise: "before → after, divided by k" (`noise_before`, `noise_after`, `noise_reduction`).
- Training: "N steps on the half-stacks".

**Side input.** The half-stacks, from Stacking.

**Methods.** The four in the table at the top. Without a GPU, Noise2Noise is greyed, and an added Denoise uses the
fine-tuned method.

**Settings.** The rows depend on the method:
- Denoise, Training and System: Noise2Noise.
- Fine-tuning: fine-tuned.
- Pretrained network: pretrained.
- NL-Bayes: NL-Bayes.

| Group | Label | stage.param | Type | Range / choices | Suggested | Level | Status |
|---|---|---|---|---|---|---|---|
| Denoise | Strength | `denoise.strength` | int | 0–100 step 5 (%) | 100 | normal | **NEW** |
| Denoise | Training steps | `denoise.iterations` | int | 500–200000 step 1000 | 24000 | normal | existing |
| Training | Patch size | `denoise.patch` | int | 64–512 step 32 (px) | 192 | expert | existing |
| Training | Learning rate | `denoise.learning_rate` | float | 0–0.01 | 0.0003 | expert | existing |
| Training | Reuse a trained network | `denoise.reuse_model` | str | a file | (empty) | expert | existing |
| System | Batch size | `denoise.batch` | int | 1–64 | 12 | system | existing |
| System | Random seed | `denoise.seed` | int | ≥ 0 | 0 | system | existing |
| Fine-tuning | Fine-tuning iterations | `denoise_ft.steps` | int | 100–3000 step 100 | 600 | normal | **NEW** |
| Fine-tuning | Starting network | `denoise_ft.start` | choice | Closest earlier project / Bundled generic network | Closest earlier project | normal | **NEW** |
| Pretrained network | Strength | `denoise_ai.force` | float | 0–1 step 0.05 | 0.8 | normal | **NEW** |
| Pretrained network | Model | `denoise_ai.model` | choice | Bundled / GraXpert (licence to check) | Bundled | normal | **NEW** |
| NL-Bayes | Strength | `denoise_cl.force` | float | 0–3 step 0.05 | 1.0 | normal | **NEW** |
| NL-Bayes | Stabilise photon noise | `denoise_cl.vst` | bool | — | yes | normal | **NEW** |
| NL-Bayes | DA3D refinement | `denoise_cl.da3d` | bool | — | no | expert | **NEW** |

**Needs from the engine.**
- The three NEW stages.
- A before / after crop at 100 %.
- The same metrics for every method.
- A store of the networks trained in earlier projects ("Closest earlier project").

---

## 11. Stars

**Purpose.**
- Each star is measured with the real PSF of the optics, positioned by Gaia, then subtracted.
- The stars are redrawn at output. This gives the starless image and the star sliders in Post.
- Stars needs Denoise (SPEC.md 5.3).

**Stages.**
- `optics` (3 min), `stars` (32 min), `starless` (10 min).
- `star_net` (41 min), with the neural method only.

**Graph summary.** "PSF model" or "PSF + neural".

**Images.** Starless image; Star model; Measured shape of a star.

**Measurements.**
- Stars measured one by one (`modelled_stars`).
- Faint stars modelled (`faint_stars`).
- Predicted / measured flux deviation, in % (the mean of `prediction_scatter_pct`).
- Star shape: "measured on a grid of a × b zones" (`local_psf_grid`).
- The mockup also has two measurements it does not show: "Learned layer kept %" and "Surface rebuilt under the stars
  %" (`starless`).

**Side input.** The Gaia stars, from Astrometry.

**Methods.** PSF model (Gaia) and PSF model + neural removal. The neural removal needs a GPU.

**Settings.**
- The "Neural removal" rows are shown only with the neural method.
- The "Starless image" rows are always shown.

| Group | Label | stage.param | Type | Range / choices | Suggested | Level | Status |
|---|---|---|---|---|---|---|---|
| Which stars | Limiting magnitude (G) | `catalogs.g_deep` | float | 12–21 step 0.5 (G) | 18.5 | normal | existing |
| Which stars | Model the faint stars | `stars.faint_model` | bool | — | yes | normal | existing |
| Which stars | Modelled star limit | `catalogs.g_faint` | float | 0–21 step 0.5 (G) | 20.5 | expert | existing |
| Which stars | Maximum number of modelled stars | `catalogs.max_faint` | int | ≥ 0 | 4000000 | expert | existing |
| Star shape | Locally corrected star shape | `stars.local_psf` | bool | — | yes | normal | existing |
| Star shape | Measured star shape (spikes) | `stars.angular_shape` | bool | — | yes | normal | existing |
| Star shape | Extended star halo | `stars.far_halo` | bool | — | yes | normal | existing |
| Star shape | Lateral chromatic aberration | `optics.lateral` | bool | — | yes | normal | existing |
| Star shape | Far halo bounded by the data | `stars.halo_bound` | bool | — | no | expert | existing |
| Star shape | Color fringe grid | `optics.lateral_grid` | int | 2–15 | 7 | expert | existing |
| Star shape | Star shape iterations | `optics.psf_iterations` | int | 50–5000 step 50 | 300 | expert | existing |
| Star shape | Stars per area for the shape | `optics.psf_stars_per_cell` | int | 5–500 | 60 | expert | existing |
| Fit | Color prior strength | `stars.color_prior` | float | 0.02–5 step 0.05 | 0.3 | expert | existing |
| Fit | Position refinement passes | `stars.refine_passes` | int | 0–5 | 2 | expert | existing |
| Fit | Background / flux iterations | `stars.solver_passes` | int | 1–10 | 4 | expert | existing |
| Fit | Solver iterations | `stars.cg_iterations` | int | 10–500 | 60 | expert | existing |
| Neural removal · Training | Training steps | `star_net.steps` | int | 10–100000 step 500 | 11000 | normal | existing |
| Neural removal · Training | Keep a nebula centred on its star | `star_net.glow` | bool | — | yes | normal | existing |
| Neural removal · System | GPU | `star_net.device` | choice | auto, main, secondary | auto | system | existing |
| Neural removal · System | Batch size | `star_net.batch` | int | 1–32 | 6 | system | existing |
| Neural removal · System | Random seed | `star_net.seed` | int | ≥ 0 | 0 | system | existing |
| Starless image · Denoise | Denoise strength | `starless.wiener_k` | float | 0.2–4 step 0.1 | 1.0 | normal | existing |
| Starless image · Denoise | Finest color scale | `starless.chroma_min_scale` | int | 0–6 | 3 | expert | existing |
| Starless image · Star residuals | Star residual removal | `starless.residual_k` | float | 0–5 step 0.1 | 1.5 | normal | existing |
| Starless image · Star residuals | Residual scales | `starless.residual_scales` | int | 1–7 | 5 | expert | existing |
| Starless image · Star residuals | Unreliable core threshold | `starless.mask_k` | float | 5–500 (σ) | 40.0 | expert | existing |
| Starless image · Star residuals | Optical halo limit | `starless.halo_gmax` | float | 4–15 step 0.5 (G) | 10.5 | expert | existing |
| Starless image · Star residuals | Dip repair limit | `starless.lift_gmax` | float | 10–21 step 0.5 (G) | 17.5 | expert | existing |
| Starless image · Nebula | Nebula protection | `starless.nebula_gate` | float | 0–1 step 0.05 | 0.85 | normal | existing |
| Starless image · Sharpening | Sharpening network | `starless.sharpen` | bool | — | yes | normal | existing |
| Starless image · Sharpening | Sharpening training steps | `starless.sharpen_steps` | int | 500–20000 step 500 | 4000 | expert | existing |

**Needs from the engine.**
- Previews of the starless image, the star model and the measured PSF.
- The metrics.
- For Post: the star mask, and the starless and stars-only layers. Post uses them to split stars and starless.

---

## 12. Output

**Purpose.**
- The end of a workflow. It prepares the render that the workflow gives to Post: its layer and its sliders.
- The block carries the workflow's name, which can be edited in the panel.

**Stages.**
- `render_prep` (2 min) with Stars.
- `render_prep_learned` (1 min 40 s) with the neural method.
- `rendu_simple` (NEW, ≈ 30 s) without Stars, and for outputs pulled from Stacking.

**Graph summary.** "to Post", or "from Stacking". The panel also says "a of n sliders".

**Images.** "Output, suggested settings": the layer, with every slider at its suggestion.

**Measurements.** One row per Post slider: "available", or "greyed out: reason".
- The star sliders (Number, Size, Color, Halo) need the Stars block.
- Detail needs the starless image, from the Stars block.
- Noise smoothing needs the Denoise block.
- A layer pulled from Stacking also greys what it leaves out.

**Context menu.** Rename, Duplicate, Delete workflow.

**Methods.** None. The stretch is a setting.

**Settings.**
- Stretch and Linked channels are shown for every workflow.
- The other rows only with Stars (`render_prep`).

| Group | Label | stage.param | Type | Range / choices | Suggested | Level | Status |
|---|---|---|---|---|---|---|---|
| Stretch | Stretch | `render_prep(_learned).stretch`, `rendu_simple.stretch` | choice | Engine (statistical) / GHS / Arcsinh / Histogram (STF) / None: linear output | Engine (statistical) | normal | **NEW** |
| Stretch | Linked channels | `render_prep(_learned).linked`, `rendu_simple.linked` | bool | — | yes | normal | **NEW** |
| Stars in the render | Protect against star artifacts | `render_prep(_learned).star_guard` | bool | — | yes | normal | existing |
| Stars in the render | Guard tolerance | `render_prep(_learned).guard_sigma` | float | 0–10 step 0.1 (σ) | 3.0 | expert | existing |
| Stars in the render | Spikes measured down to magnitude | `render_prep(_learned).real_g` | float | 0–16 step 0.5 (G) | 12.0 | expert | existing |
| Stars in the render | Spikes kept from a signal-to-noise of | `render_prep(_learned).spike_snr` | float | 1–50 step 0.5 (× noise) | 2.0 | expert | existing |
| Image edges | Minimum edge coverage | `render_prep(_learned).cover_min` | float | 0.1–1 step 0.05 | 0.6 | expert | existing |
| Image edges | Edge cleanliness | `render_prep(_learned).edge_frac` | float | 0.9–1 step 0.001 | 0.995 | expert | existing |
| Image edges | Minimum effective exposure | `render_prep(_learned).min_exposure` | float | 0–1 step 0.01 | 0.15 | expert | existing |
| Detail and color | Fine detail scale | `render_prep(_learned).fine_sigma` | float | 0.5–10 (px) | 3.0 | expert | existing |
| Detail and color | Background smoothing under the star light | `render_prep(_learned).cont_smooth` | float | 0–60 (px) | 24.0 | expert | existing |
| Detail and color | Residual color smoothing | `render_prep(_learned).resid_smooth` | float | 0–50 (px) | 6.0 | expert | existing |
| Detail and color | Learned denoising in the base | `render_prep(_learned).n2n_base` | float | 0–1 step 0.05 | 0.0 | expert | existing |
| Detail and color | Learned denoising up to the scale | `render_prep(_learned).n2n_scale` | float | 0–200 step 1 (px) | 16.0 | expert | existing |
| Detail and color | Diffuse glow scale | `render_prep(_learned).diffuse_scale` | float | 0.05–1 step 0.01 | 0.2 | expert | existing |
| Detail and color | OIII map base smoothing | `render_prep(_learned).oiii_scale` | float | 0–50 (px) | 4.0 | expert | existing |

**Needs from the engine.**
- `rendu_simple`: a simple output preparation, with the choice of stretch.
- The render of the layer at given slider positions (POST.md).
- Which sliders each workflow offers, and why the others are greyed.

---

## 13. Post

**Purpose.** Combines the workflows' layers, with filter layers and masks, into the final image.

**Graph summary.** "N layer(s)". The block is Up to date when every workflow is rendered.

**Panel in Graph.**
- The layers, each with its state (Up to date, Outdated or Pending) and its source ("from Stacking").
- "Open Post".

Everything else is in POST.md.
