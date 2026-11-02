# Post

Taken from `mockup/alt3/app/post3.py` (the screen) and `postengine.py` (the model and the composition) on
2026-10-31.

Post combines the outputs of the workflows:
- layers, as in Photoshop;
- masks made of parts, as in Lightroom;
- filter layers, applied to one layer or to everything below;
- continuous sliders.

It is the Inspect view of the Post block (Ctrl+3). When no workflow has been rendered yet, it shows "No layers yet ·
Each computed workflow becomes a layer here · Go to graph →".

## 1. Screen

- **Centre**: the image.
  - The bar holds Fit, 100 %, the zoom in %, Before / after, Show mask, ↶ ↷ and Export….
  - A hint line under the image explains the current tool. For example: "Brush: drag to paint · X paint / erase ·
    [ ] size · M show mask".
- **Right** (400 px):
  - the **Layers** list, top first;
  - the buttons **+ Filter** (a menu by family), **Duplicate** (Ctrl+J), **Mask** ("Mask ✓" once the layer has one:
    it then edits it) and **Delete**;
  - under them, the properties of what is selected: the layer, or its mask.
- **A layer row** shows:
  - the eye (visible or hidden);
  - "↳" when the filter applies only to the layer just below;
  - a thumbnail: the image, or an italic "ƒ" in the colour of the filter's family;
  - the mask thumbnail (white shows the layer, black hides it);
  - the name;
  - a sub-line: "on "X"" or "on everything below", then the mode and opacity, then "preview" when the workflow's image
    is still an approximation.
- **Mouse on a row**:
  - a click on the thumbnail selects the layer;
  - a click on the mask thumbnail selects the mask;
  - a drag reorders the rows (a drop line shows where);
  - a double-click renames;
  - a right-click opens the menu: Duplicate, Add mask / Delete mask, Split stars and starless, Show workflow →,
    Rename, Delete.
- The workflows not computed yet are listed under the layers as links: '"X": not computed yet →'. A link opens
  Inspect on that workflow's Output.

## 2. Layers

### 2.1 Workflow layer
`{id, kind: "workflow", pid, part: None | "stars" | "starless", name, visible, opacity 0–100, mode, sliders: {key:
position}, mask}`

- **When it appears.**
  - A workflow gets its layer once it has been rendered, that is, computed to the end once. New layers are inserted
    at the top, hidden.
  - When Post opens for the first time, the "Full" layer (or else the bottom one) goes to the bottom and is made
    visible.
  - If no workflow layer is visible, the bottom one is made visible.
  - Deleting a workflow removes its layer.
- **Properties.**
  - Blend mode: Normal, Lighten, Screen, Multiply or Soft light.
  - Opacity: 0–100 %.
  - The workflow's simplified **sliders**, in two groups:

| Group | Slider | Ends | Engine settings moved (values at positions 0, 1, 2 = suggested, 3, 4) |
|---|---|---|---|
| Nebula and background | Sky | darker – brighter | `sky_level` 12, 18, 24, 30, 36 |
| Nebula and background | Contrast | softer – stronger | `lc_fine` 0.95, 1, 1.05, 1.15, 1.3 · `lc_medium` 1, 1.15, 1.3, 1.55, 1.85 · `lc_large` 1, 1.12, 1.25, 1.5, 1.8 |
| Nebula and background | Saturation | paler – more vivid | `chroma` 0.45, 0.68, 0.88, 1.08, 1.3 |
| Nebula and background | Detail | smoother – sharper | `neb_sharp` 0.2, 0.55, 0.9, 1.4, 2 · `fine_k` 0.6, 0.8, 1, 1.3, 1.6 |
| Nebula and background | Noise smoothing | less – more | `faint_smooth` 0, 16, 32, 52, 80 |
| Stars | Number of stars | fewer – more | `g_lim` 11, 13, 15, 16.5, 18 |
| Stars | Star size | smaller – larger | `d10` 2.6, 3.5, 4.4, 5.5, 6.8 · `d7` 5.4, 7.2, 9, 11.2, 14 |
| Stars | Star color | whiter – more colorful | `star_sat` 0.15, 0.4, 0.7, 1, 1.4 |
| Stars | Bright star halo | none – wider | `star_glow` 0, 0.15, 0.3, 0.5, 0.8 |

- **How a slider works.**
  - A slider is continuous from 0 to 4. Position 2 is the suggestion, shown as "suggested"; other positions show as
    "+0.7" or "−1.2".
  - A double-click on its name brings it back to the suggestion.
  - A slider the workflow cannot offer is greyed, with the reason. Star sliders and Detail need Stars; Noise
    smoothing needs Denoise; a layer pulled from Stacking leaves the graph before them.
  - In the mockup, each position is a pre-rendered image and the effects of the sliders are added. The client asks
    the engine for the layer rendered with the interpolated engine settings (ENGINE_CONTRACT.md 9).

### 2.2 Filter layer
`{id, kind: "filter", filter, name, visible, opacity, mode, params, mask, target}`

- **Applies to.**
  - "X" only: `target` is that workflow layer. The filter sits just above it, as a Photoshop clipping mask.
  - Everything below: `target` is None. The filter acts on the composite of all the layers under it.
- **A new filter** applies to the selected workflow layer, or to whatever the selected filter applies to. It goes on
  top of that layer's own filters.
- **List order** (`normalize`):
  - a layer's own filters always sit just above it, in their order;
  - a filter whose layer is deleted applies to everything below.
- **Dragging a filter** retargets it:
  - dropped just above a workflow layer: it applies to that layer;
  - dropped above a filter of a group: it joins that group;
  - dropped at the bottom: it applies to everything below.
- **Duplicate** puts a copy above. A workflow layer is copied with its own filters, retargeted to the copy. The copy is
  named "<name> (copy)".
- **Delete**: a workflow layer goes with its own filters. The last workflow layer cannot be deleted.

### 2.3 The 13 filters

| Family | Filter | Parameters: range (default) |
|---|---|---|
| Tone | Stretch (GHS) | Strength 0–12 (3), Symmetry point 0–0.6 (0.08). An arcsinh stretch around the symmetry point. |
| Tone | Curves | Shadows, Midtones, Highlights −0.2 to 0.2 (0). They move the curve at 0.25, 0.5 and 0.75. |
| Tone | Levels | Black point 0–0.3 (0.02), White point 0.7–1 (1), Gamma 0.5–2 (1) |
| Color | Saturation | Strength −100 to 150 % (30); Hues: All, Reds, Yellows, Greens, Cyans, Blues or Magentas (All) |
| Color | SCNR (green removal) | Strength 0–1 (0.8) |
| Color | Color balance | Temperature −1 to 1 (0), Green / magenta −1 to 1 (0) |
| Detail | Local contrast | Strength 0–1.5 (0.4), Scale 5–120 px (40) |
| Detail | Sharpen | Strength 0–2 (0.6), Radius 0.5–4 px (1.2) |
| Detail | Noise reduction | Luminance 0–1 (0.4), Color 0–1 (0.7) |
| Stars | Star reduction | Strength 0–1 (0.6). It works through the catalogue star mask. |
| Stars | Halo reduction | Strength 0–1 (0.6), Radius 4–40 px (14). It works on a ring around the star mask. |
| Stars | Star color | Strength 0–2 (0.6). It works through the star mask. |
| Combine | H-alpha to red | Strength 0–1 (0.5). It works through the H-alpha map. |

The pixel sizes are full-resolution pixels. At a reduced resolution they are divided by the reduction factor, so the
draft looks the same as the full image.

## 3. Masks

`mask = {parts: [...], feather 0–60 px (0), density 0–100 % (100), invert (no)}`.

- **Adding a mask.** Any layer can have one: use the Mask button or "Add mask". A new mask has no parts, and the layer
  then shows everywhere ("Empty: the layer shows everywhere.").
- **Parts.** "+ Add to mask" offers six sources. Each part has an operation (**Add**, **Subtract** or **Intersect**)
  and a ✕ to remove it.

| Source | Settings: range (default) | Data needed |
|---|---|---|
| Catalog stars | Size around stars 0–24 px (4): a dilation, then a smoothing | the star mask from the engine: every drawn star, and the halo of stars brighter than G 7.5 |
| H-alpha | none | the H-alpha emission map from the engine: 0 at the median, 1 at the brightest 3 %, smoothed |
| Luminance range | From 0–100 % (40), To 0–100 % (100); 6 % soft edges | the composite below |
| Color range | Hue 0–359° (0), Width 10–120° (40); weighted by saturation | the composite below |
| Linear gradient | Direction: Top → bottom, Bottom → top, Left → right or Right → left; Position 0–100 % (50); Softness 1–100 % (40) | none |
| Brush | Paint or Erase (X), Size 4–400 px (60, with [ and ]), Hardness 0–100 % (50), Clear all, Paint on image | the strokes, in image coordinates |

- **How the parts combine**, in list order:
  - the result starts at 0 if the first part is Add, and at 1 otherwise;
  - Add takes max(result, part); Subtract takes result × (1 − part); Intersect takes result × part.
- **Then the whole mask is processed:**
  - feather: a Gaussian blur of the feather radius;
  - density: 1 − density × (1 − mask);
  - invert.
- **Editing a mask.**
  - When the mask is selected, the image shows it as a red overlay, where the layer shows.
  - **Show mask** (M) shows the mask in grey instead.
  - Selecting a mask that has a Brush part turns the cursor into the brush. A stroke fills the gaps of a fast movement.
- **Mask thumbnails** are refreshed after each full render.

## 4. The composition

- The layers are composed **bottom first**.
- **Hidden** layers are skipped.
- **A workflow layer** is handled in four steps:
  1. its image, at the requested resolution and region;
  2. its own filters, bottom first, each through its mask, opacity and mode;
  3. the blend mode, against the composite below;
  4. the mix: composite + (image − composite) × opacity × mask.
- **A filter layer that applies to everything below** filters the composite, then mixes the result in through its
  mask, opacity and mode.
- **Blend modes**, with a the image below and b the layer:
  - Lighten: max(a, b);
  - Screen: 1 − (1 − a)(1 − b);
  - Multiply: a × b;
  - Soft light: (1 − 2b)a² + 2ab.
- **Clipping.** The result is clipped to 0–1. Values are display-referred, 0–1, after the Output stretch.

## 5. Split stars and starless

- This is in the right-click menu of a workflow layer that has a star model.
- The layer becomes two layers:
  - "<name> · stars": the stars alone (the layer minus its starless image, clipped at 0), in **Screen** mode, with no
    mask, on top;
  - "<name> · starless": the starless image, below.
- The layer's own filters stay on the starless layer.
- Each half has its own sliders, filters and mask.

## 6. Before / after

- **Before / after** (C) shows a divider that can be dragged across the image.
- "Before" is the bottom workflow layer at its suggested settings, without filters or masks.

## 7. Undo

- Post has its own undo stack, separate from the graph's: ↶ ↷, Ctrl+Z and Ctrl+Y.
- Each step is a snapshot of all the layers.
- One step is recorded per action: visibility, order, add, duplicate, delete, rename, mode, a choice, mask added or
  removed, part added or removed, invert.
- A slider drag is **one** step: the state before is taken at the first movement and recorded on release.
- A brush stroke is **one** step.
- After an undo, the layer caches are cleared. The selection moves to a layer that still exists.

## 8. Speed

| When | Resolution | Detail |
|---|---|---|
| A slider or the brush moves | **1/4** (draft) | The draft is rendered at once, then enlarged ×2 to screen size. |
| On release, or 220 ms after the last move | **1/2** (screen) | The whole image. The mask thumbnail is refreshed. |
| Zoom at 50 % or more | **full**, visible region only | Only the visible region is composed at full resolution, with a 64 px margin so that the filters see their neighbours. |

The caches:
- The prepared images are kept at the three resolutions, the least recently used dropped first; the mockup keeps 90.
- Each workflow layer is cached for its id, resolution, region (full resolution only), part, sliders and workflow
  image. Only what changed is computed again.
- The brush is rasterised incrementally, for each part and resolution: only the new strokes are stamped.

These are requirements for the client. The engine must render a layer at 1/4 and 1/2 resolution quickly, and at full
resolution for a region (ENGINE_CONTRACT.md 9).

## 9. Export

- Export… in the Post bar, or File > Export image… (Ctrl+E), opens the Export dialog (SPEC.md 3.7).
- The dialog offers:
  - the format, the colour profile and the size;
  - the WCS and the processing history in the header;
  - each layer as a separate file, the masks, and each workflow's linear output as FITS.
- The final image is the full composition, rendered at full resolution.
