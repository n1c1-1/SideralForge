# astro_GPU in ComfyUI

This code base (C:/Users/Nicolas/sandbox/astro_comfy, cloned from astro_GPU on 2026-10-17, which stays untouched as
it was) is the reference from now on. The processing chain runs as ComfyUI nodes; the astro_gpu library underneath
is the same (stages, variants, rendering engine). The heavy client (astro-gpu-gui) still works on the same projects;
its Final adjustment and views will move to ComfyUI later.

## Install and start

- ComfyUI: C:/Users/Nicolas/sandbox/ComfyUI (version 0.41), with its own Python environment (.venv: Python 3.12,
  PyTorch 2.11 CUDA 12.8) in which this repository is installed in editable mode (`uv pip install -e`).
- The nodes: `comfyui_astro/`, linked into `ComfyUI/custom_nodes/comfyui_astro` (a folder junction).
- Start: `ComfyUI_astro.bat` (this repository), then open http://127.0.0.1:8188. Restart after changing the nodes'
  code: `comfyui_astro/tools/restart_comfy.ps1`.

## The nodes (category "astro")

All the nodes pass on an **ASTRO** value: the project folder, the variant ("" = the main processing) and the last
stage computed.

| Node | What it does |
|---|---|
| Astro: Project | Opens a project folder (the one holding project.json, made by the astro_GPU interface or command line). |
| Astro: Variant | The nodes after it compute a variant: the same data processed another way, kept in its own folders (variants/<name>/) next to the main processing; the stages before the first change are shared. |
| Astro: <stage> (one per stage: Inventory ... Rendering preparation) | Computes everything needed up to its stage that is not up to date; a stage already up to date is not touched. |
| Astro: Render | Renders the image of a processing: outputs final, starless, stars and the settings used. |
| Astro: Render (live) | The rendering settings as sliders, by section, with a preview in the node that follows them at once (see below). |
| Astro: Star mask | A mask around the bright stars (Gaia catalogue), framed like a render at the same scale. |
| Astro: Export | The full export into the project's results/<name>/ (16-bit TIFF with colour profile, PNG, JPEG, layers, reference framing, settings). |
| Astro: Save 16-bit | Any image (a composite of views...) as a 16-bit TIFF. |

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

## Example graphs (comfyui_astro/workflows/)

Open the files without "_api" by dragging them onto the ComfyUI window (the "_api" files are the same graphs in
ComfyUI's API format, also accepted, and used by `tools/run_prompt.py`).

1. **01_render_project**: a project rendered with its delivered faithful settings, previews, 16-bit save.
2. **02_full_chain**: every stage as a node, in order, then the render: change a stage's parameters there.
3. **03_compare_variant**: the main processing and a variant (far halo bounded by the data) side by side.
4. **04_views_mask**: two views of the same image (glare drawn with the nebula / with the stars) combined with a
   mask around the bright stars, saved in 16 bits.
5. **05_render_live**: the live render node (sliders and preview), its image previewed and saved in 16 bits.

## Tools

- `comfyui_astro/tools/run_prompt.py WORKFLOW_api.json [--set node.input=value]`: run a graph on the running
  ComfyUI from the command line (tests, batches).
- `comfyui_astro/tools/api_to_ui.py api.json ui.json`: make an interface graph from an API graph.
- `comfyui_astro/tools/ui_check.py`, `live_check.py`: check the graphs and the live node in a headless browser
  (screenshots).

## Next

- Masks drawn by hand per zone and more automatic masks (by brightness, around the nebula).
- The interface's views and its Final adjustment moved into ComfyUI.
