# astro_GPU in ComfyUI

This code base (C:/Users/Nicolas/sandbox/astro, cloned from astro_GPU on 2026-10-17, which stays untouched as
it was) is the reference from now on. The processing chain runs as ComfyUI nodes; the astro_gpu library underneath
is the same (stages, variants, rendering engine). The heavy client (astro-gpu-gui) still works on the same projects;
its Final adjustment and views will move to ComfyUI later.

## Install and start

- ComfyUI: C:/Users/Nicolas/sandbox/ComfyUI (version 0.41), with its own Python environment (.venv: Python 3.12,
  PyTorch 2.11 CUDA 12.8) in which this repository is installed in editable mode (`uv pip install -e`).
- The nodes: `comfyui_astro/`, linked into `ComfyUI/custom_nodes/comfyui_astro` (a folder junction).
- The heavy client of this code base: `astro_client.bat` (this repository; it runs in ComfyUI's environment, with the
  same code as the nodes). The old launcher (sandbox/astro_GPU) runs the old code: since 2026-10-17 the projects
  hold stage versions it does not know (it would see them out of date and compute them again the old way).
- Start: `ComfyUI_astro.bat` (this repository), then open http://127.0.0.1:8188. Restart after changing the nodes'
  code: `comfyui_astro/tools/restart_comfy.ps1`.

## The processing blocks (categories "astro/1 sources", "astro/2 calibration", "astro/3 alignment", "astro/4 stacking",
"astro/5 field and colour", "astro/6 denoising and stars", "astro/7 rendering", "astro/8 combine")

Elementary blocks: the data travels on the wires, each block does one thing and shows what it did (an image and a
one-line summary). Wire them in the order you want, and branch: two blocks fed by the same output give two
versions to compare.

| Block | In -> out | What it does |
|---|---|---|
| Light frames | folders -> frames | The light frames of a session (RAW or FITS), checked; shows the middle frame. |
| Master offset / Master dark / Master flat | folder -> master | One calibration master from its own folder (the flat can take flat darks or the offset). |
| Calibrate | frames + masters -> calibrated frames | Dark subtracted, flat divided, hot / cold / dusty pixels excluded; cached on the scratch disk. |
| Align | calibrated -> aligned | Polynomial alignment on the sharpest frame; transparency and star size curves. |
| Frame selection | aligned -> aligned | Which frames are stacked and their weight (optional: without it the stack applies the same rule). |
| Stack | aligned -> linear image | Drizzle without debayering, outliers rejected, two half-stacks for the denoising; stretched view. |
| Identify the field | linear -> linear | Where the image points on the sky (stars matched to Gaia); immediate with the object name and focal length, blind otherwise. |
| Gaia stars | linear -> stars | The Gaia stars of the field placed on the image (positions, brightness, colours, spectra); needs the identified field. |
| Hydrogen survey | linear -> survey | A professional H-alpha survey of the field reprojected onto the image (reference for the sky background). |
| Colour calibration | linear + stars -> linear | The camera's colour response measured on the Gaia spectra; the colour matrix to true sky colours. |
| Denoise (two half-stacks) | linear -> linear | A network trained on the image to predict one half-stack from the other: denoises without inventing. |
| Star shape and colour fringes | linear + stars -> linear | The star shape across the field (core, wings, spikes) and the colour fringes, corrected. |
| Measure the stars | linear + stars -> star model | Every Gaia star measured with that shape: the light of the stars alone. |
| Learned star removal | linear + star model -> star layer | Optional (about 35 min): a star-removal network trained on the image itself. |
| Remove stars | linear + star model (+ star layer) -> starless | Stars subtracted, starless image denoised; with a star layer also the learned starless image; trains the sharpening network used at rendering. |
| Sky background | starless + survey -> sky | The real sky separated from the instrument and the atmosphere (residual flat, sky glow). |

Each block keeps its result on disk (folders set in `comfyui_astro/blocks_config.json`: work E:/astro_gpu/blocks,
large caches C:/Users/Nicolas/astro_gpu_scratch/blocks), under a fingerprint of its inputs and settings: a block is
not computed again (even after a restart) while nothing upstream changed. A setting left at its default value means
automatic (chosen from the data; the block's summary says what it did).

### Rendering blocks ("astro/7 rendering")

What travels on the wires after **Prepare the rendering** is a *rendering*: the layers the rendering engine works on
plus the settings chosen so far. Each setting block sets its own settings on the rendering it receives and passes it
on; a block left out keeps the faithful rendering's values. The engine applies the settings in its own physical
order (colour in linear light, then stretch, contrast, highlights, stars): the order of the blocks on the wire does
not change the image. Two chains from the same preparation are two versions.

| Block | What it sets |
|---|---|
| Prepare the rendering | sky -> rendering: the layers (nebula at full resolution, detail and colour layers, stars as photographed, star profile); also the learned star removal's layers when it was made. About a minute. |
| Rendering preset | All the settings of a ready-made rendering (Faithful, Artistic, H-alpha emphasised...): put it first. |
| White balance | Colour temperature shown as neutral, tint. |
| Stretch | Sky brightness, darkest clouds level, shadow softness, white point. |
| Sky glow and specks | Milky Way glow of the undrawn stars, diffuse glow removed, neutral sky, dark specks removed. |
| Emission colours | H-alpha red (strength, hue), OIII teal, blue of the reflection nebulae. |
| Saturation | Nebula saturation, fine colour, colour of the faint areas smoothed by their signal-to-noise. |
| Contrast | Local contrast at three scales, compression of the large bright areas. |
| Sharpening | Classic sharpening and the sharpening network (measured star shape undone), from a brightness on. |
| Fine detail and smoothing | Fine detail brought back in the bright areas, faint areas smoothed. |
| Highlights | Highlight compression, its ceiling, detail and colour. |
| Stars: removal and spikes | Stars shown or not, which starless image (learned or star model), saturated stars and spikes as photographed, glare kept with the nebula. |
| Stars: size and colour | Faintest drawn star, sizes, glow, colour, white cores. |
| Show the rendering | rendering -> final, starless, stars images at a scale (0.33 by default; 1 = full resolution). |
| Save the image | A rendering (full resolution, with its starless and stars layers and its settings) or any image, as 16-bit TIFF with sRGB profile and JPEG; default folder <work>/images. |

### Combining versions ("astro/8 combine")

| Block | What it does |
|---|---|
| Bright star mask | A mask around the bright stars (Gaia), framed like 'Show the rendering' of the same preparation at the same scale. |
| Brightness mask | A mask from an image's brightness, smoothed into zones, thresholds in percentiles of the image or in levels. |
| Painted mask | A mask painted by hand with a brush over an image, in the node. |
| Blend versions | Up to five versions laid over a base, each where its mask is white; a coverage view shows which version is used where. |

### Starting graphs (ComfyUI library, folder "astro")

1. **1 - Traitement complet** (comfyui_astro/workflows/full_chain.json): from the raw frames to the two renderings
   (faithful and artistic), every block with its view and summary underneath, in four labelled groups (frames to
   stack, field and colour, denoising and stars, rendering). 24 frames for a quick test ('max_frames' 0 = all).
2. **2 - Rendu en detail et combinaison** (render_detail.json): the same processing (already computed, not
   recomputed), then every setting block chained after a preset, and two versions combined (faithful around the
   bright stars, artistic elsewhere).

The graphs used to test the stages one by one are in the library's sub-folder "etapes (essais)" (stage1.json to
stage3.json). Graphs are built from their API version with `tools/build_ui_graph.py <graph>_spec.json` (layout,
groups and note in the spec).

Note on small tests: with a few tens of frames, the drizzle leaves each colour plane of each half-stack unevenly
covered (blobs of low coverage a few hundred pixels across); the star removal, which keeps what the two halves agree
on, then shows them as coloured patches on a regular pattern. They are a property of the subset, not of the blocks:
judge the image on a full session.

The project nodes below (category "astro/old - projects") are the first design, kept for the existing projects.

## The nodes (category "astro/old - projects")

All the nodes pass on an **ASTRO** value: the project folder, the variant ("" = the main processing) and the last
stage computed.

| Node | What it does |
|---|---|
| Astro: New project | Makes a project from the folders of your frames, as the heavy client's New project: the light frames' folder (or files, one per line); the darks, flats, bias and flat darks next to it, the project name, its folder, the scratch folder and the target are guessed when left empty. A folder that already holds a project is opened as it is, never overwritten. |
| Astro: Project | Opens a project folder (the one holding project.json, made by the astro_GPU interface or command line). |
| Astro: Methods | The processing choices in plain words, as in the heavy client's Methods tab (learned star removal, glare of the bright stars, shape of the stars, faint stars, flat field, H-alpha reference, repairs around the stars, sensor pattern, colour fringes): one list per question, "as main" = unchanged; hover a list for what each option does and what was seen on real data. The choices are computed as a variant (main is never changed), named after them unless named in the node. |
| Astro: Variant | The nodes after it compute a variant: the same data processed another way, kept in its own folders (variants/<name>/) next to the main processing; the stages before the first change are shared. |
| Astro: <stage> (one per stage: Inventory ... Rendering preparation) | Computes everything needed up to its stage that is not up to date; a stage already up to date is not touched. It has one input per step it depends on directly (e.g. Optics takes Denoising and Catalogues), so a graph draws the real dependencies; or just the project ('astro'). |
| Astro: Render | Renders the image of a processing: outputs final, starless, stars and the settings used. |
| Astro: Render (live) | The rendering settings as sliders, by section, with a preview in the node that follows them at once (see below). |
| Astro: Star mask | A mask around the bright stars (Gaia catalogue), framed like a render at the same scale. |
| Astro: Export | The full export into the project's results/<name>/ (16-bit TIFF with colour profile, PNG, JPEG, layers, reference framing, settings). |
| Astro: Save 16-bit | Any image (a composite of views...) as a 16-bit TIFF. |
| Blend versions (now astro/8 combine) | Combines up to five views zone by zone: each view laid over the ones before it where its mask is white; outputs the image and a coverage map (which view is used where). |
| Brightness mask (now astro/8 combine) | A mask from an image's brightness, smoothed into zones, thresholds in percentiles of the image or in levels. |
| Painted mask (now astro/8 combine) | A mask painted by hand with a brush over an image, in the node. |

Stage parameters: every parameter is a widget with its label, range and help (hover it); the advanced ones are in
the node's advanced section. **settings: project** (the default) keeps the parameters already chosen for the project
and ignores the widgets, so an existing project is never recomputed by surprise. **settings: this node** takes the
widgets: a widget left at its default value means *automatic* (the value the stage derives from the data).

Recomputation: ComfyUI keeps each node's result and computes again only a node whose inputs changed and the nodes
after it; the chain itself also keeps its results on disk (work/<stage>/, variants/<name>/<stage>/) and recomputes
only what is out of date, so a restarted ComfyUI does not recompute anything that is done.

Progress: the running stage drives the node's progress bar; ComfyUI's stop button stops the stage at its next
checkpoint (it can be resumed).

Render settings (Astro: Render): **source** = the settings of a delivered rendering (results/faithful, artistic),
the interface's last settings or a preset; **view** = a view saved in the interface (its settings replace the
source); **changes** = settings changed on top, as JSON, e.g. `{"glare_nebula": 0, "chroma": 1.0}`; **scale** =
output size relative to the full drizzled resolution. Several render nodes on the same processing share its layers
on the GPU.

Live render (Astro: Render (live)): run the graph once; the node then shows the rendering settings as sliders
(section list, "advanced" for every setting of the section) starting from the chosen source, and a preview. Each
slider moved re-renders the preview at the node's width in about 0.15 s, without running the graph (the server
keeps the processing's layers on the GPU). A changed slider is highlighted; double-click a slider to put it back
to the source's value; Reset puts them all back. The changes are kept in the node (saved with the graph). Running
the graph gives the image at the chosen scale with the sliders' settings, for the nodes after it (save, masks...).
**start from** lists the project's delivered renderings, its views (of the node's processing) and the presets:
choosing one sets the node's source or view and starts the sliders from it. **Save as view...** saves the sliders
as a view of the project, which the heavy client's views bar shows too (and the other way round). **Compare
(hold)** shows the starting rendering while the button is held. **clipping** marks, as in the heavy client, the
pure black (blue), the pure white (red) and the nebula above 92 % where its detail is lost (amber).

## Zones: one good view per part of the image

No single rendering suits every part of an image: the glare drawn with the stars may suit the bright stars and
the glare drawn with the nebula the rest, the nebula may want more contrast than the sky. Render the views
(Astro: Render or Astro: Render (live), at the same scale), make one mask per view, and combine them with
**Blend versions**: the base is used wherever no mask takes another view; each view is laid over the ones before
it, so put the most local corrections last (a painted correction on top). Its **coverage** output shows which view
is used where (grey = the base, then orange, cyan, magenta, green): check that every zone has the view wanted.

Masks:
- **Astro: Star mask**: around the bright stars (Gaia catalogue), radius growing with brightness.
- **Brightness mask**: from an image's brightness, e.g. the nebula from the starless layer (percentiles 60
  to 90), the stars from the stars layer (levels 3 to 15 %, grown), bright cores from the final image (percentiles
  99.5 to 99.95). Percentiles are measured on each image, so the same graph adapts to another object. Note: the
  starless layer holds the bright stars' glare (when the glare is drawn with the nebula), so a nebula mask also
  takes their surroundings: lay the star view over it.
- **Painted mask**: run the graph once, the image appears in the node; paint with the left button, erase
  with the right one (or Erase), mouse wheel = brush size, Undo, Clear, Fill; "overlay" sets how strongly the mask
  shows. The mask is kept in the node (saved with the graph) and follows the image at any scale.

Sizes (smoothing, growth, soft edges) are in % of the image width, so a graph works at any scale.

## Example graphs of the project nodes (comfyui_astro/workflows/, library sub-folder "anciens (projets)")

Open the files without "_api" by dragging them onto the ComfyUI window (the "_api" files are the same graphs in
ComfyUI's API format, also accepted, and used by `tools/run_prompt.py`).

1. **01_render_project**: a project rendered with its delivered faithful settings, previews, 16-bit save.
2. **02_full_chain**: every stage as a node, wired by its real dependencies (after the stack the astrometry, the
   denoising and the reference branch out and meet again at the optics; at the end the two star removals give two
   renders): change a stage's parameters there. ComfyUI computes one node at a time, in an order that respects the
   wires; changing a stage recomputes only the branches after it.
3. **03_compare_variant**: the main processing and a variant (far halo bounded by the data) side by side.
4. **04_views_mask**: two views of the same image (glare drawn with the nebula / with the stars) combined with a
   mask around the bright stars, saved in 16 bits.
5. **05_render_live**: the live render node (sliders and preview), its image previewed and saved in 16 bits.
6. **06_zones**: four views of the Flaming Star combined by zones: faithful as the base, more colour and contrast
   in the nebula (brightness mask on the starless layer), glare drawn with the stars around the bright stars (star
   mask, laid over the nebula), and the artistic rendering where it is painted by hand (empty at first: paint it).
7. **07_methods**: one processing choice (repairs around the stars: as removed) computed as a variant and
   rendered next to the main processing. Note: some choices act only on one star removal (this one only on the
   star model: render both with `{"learned_stars": false}` to see it).
8. **08_glare_zones**: the Horsehead's milky veil comes from the glare of Alnitak drawn with the nebula (its wide
   wings lift the whole sky). The base view draws a quarter of the glare with the nebula (dark sky, contrast back);
   around the brightest stars (Gaia G < 2.5) the photographed glare is kept, fading out radially (soft star mask),
   where the glare drawn with the stars would show a disc and coarse rays.
9. **09_new_project**: a new project from your frames (fill in the light frames' folder): computed from the
   calibration to the rendering, previewed and exported as results/faithful. The whole chain takes one to two hours
   on a large session (the denoising and star removal networks are trained on the data).

## Tools

- `comfyui_astro/tools/run_prompt.py WORKFLOW_api.json [--set node.input=value]`: run a graph on the running
  ComfyUI from the command line (tests, batches).
- `comfyui_astro/tools/api_to_ui.py api.json ui.json`: make an interface graph from an API graph (automatic
  layout); `build_ui_graph.py spec.json`: the same with a hand layout, groups and a note.
- `comfyui_astro/tools/ui_check.py`, `live_check.py`: check the graphs and the live node in a headless browser
  (screenshots).

## Next

- The interface's views and its Final adjustment moved into ComfyUI.
