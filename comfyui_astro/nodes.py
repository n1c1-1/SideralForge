"""ComfyUI nodes of astro_GPU: the processing chain as nodes, variants as branches, the rendering and the export.

Every node passes on an ASTRO value (project folder, variant, last stage computed: see bridge.py).
- 'Astro: Project' opens a project folder (made by the astro_GPU interface or command line).
- 'Astro: Variant' makes the nodes after it compute a variant: the same data processed another way, in its own
  folders next to the main processing (the stages before the first change are shared).
- One node per processing stage ('Astro: Stacking', 'Astro: Star photometry'...). A stage node computes everything
  needed up to its stage that is not up to date; a stage already up to date is not touched. Its parameters come from
  the project ('settings: project', the default: what was chosen before) or from the node's widgets ('settings: this
  node': a widget left at its default value means 'automatic', the value the stage derives from the data).
- 'Astro: Render' renders the image (final, starless, stars) with rendering settings; 'Astro: Export' writes the
  16-bit TIFF, PNG, JPEG, layers and reference framing into the project's results folder.
"""
from __future__ import annotations

import json
import time

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from . import bridge

ASTRO = io.Custom("ASTRO")
CATEGORY = "astro"


def _hooks():
    import comfy.model_management as mm
    from comfy.utils import ProgressBar
    bar = ProgressBar(1000)

    def on_progress(stage, frac, msg):
        bar.update_absolute(int(1000 * min(max(frac, 0.0), 1.0)))
    return on_progress, mm.processing_interrupted


def _run(value: dict, stage: str) -> dict:
    import comfy.model_management as mm
    on_progress, interrupted = _hooks()
    try:
        return bridge.run_up_to(value["root"], value["variant"], stage, on_progress, interrupted)
    except RuntimeError:
        mm.throw_exception_if_processing_interrupted()
        raise


def _ready_to_render(value: dict) -> dict:
    """Everything a rendering needs, computed if needed: the learned star removal network first (it is optional for
    the stages after it, so they would not ask for it; the stage does nothing when switched off), then both
    rendering preparations (star model and learned removal)."""
    _run(value, "star_net")
    _run(value, "render_prep")
    return _run(value, "render_prep_learned")


def _tooltip(d: dict) -> str:
    parts = [d.get("help", "")]
    if d.get("tip"):
        parts.append(d["tip"])
    if d.get("unit"):
        parts.append(f"Unit: {d['unit']}.")
    parts.append(f"Default: {d.get('default')!r} (left at the default = automatic, chosen from the data).")
    return "\n".join(p for p in parts if p)


def _widget(d: dict):
    """A ComfyUI widget for a stage parameter description (Params.describe())."""
    name, typ, default = d["name"], d.get("type", ""), d.get("default")
    common = {"display_name": d.get("label") or name, "tooltip": _tooltip(d), "advanced": d.get("level") == "advanced",
              "optional": True}                   # a parameter left out of a graph is automatic
    if typ == "bool":
        return io.Boolean.Input(name, default=bool(default), **common)
    if typ == "int" and isinstance(default, int):
        return io.Int.Input(name, default=default, min=int(d["ge"]) if "ge" in d else None,
                            max=int(d["le"]) if "le" in d else None, step=int(d.get("step") or 1), **common)
    if typ == "float" and isinstance(default, (int, float)):
        return io.Float.Input(name, default=float(default), min=float(d["ge"]) if "ge" in d else None,
                              max=float(d["le"]) if "le" in d else None, step=float(d.get("step") or 0.01),
                              round=False, **common)
    if d.get("choices"):
        return io.Combo.Input(name, options=[str(c) for c in d["choices"]], default=str(default), **common)
    if typ == "str" and (default is None or isinstance(default, str)):
        return io.String.Input(name, default=default or "", **common)
    # anything else (lists, optional numbers): JSON text
    return io.String.Input(name, default=json.dumps(default), **{**common, "tooltip": common["tooltip"] + "\n(JSON)"})


def _parse(d: dict, value):
    """Back from the widget to the parameter's value."""
    typ, default = d.get("type", ""), d.get("default")
    if typ in ("bool", "int", "float") and not isinstance(value, str):
        return value
    if d.get("choices") or (typ == "str" and (default is None or isinstance(default, str))):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


# ---------------------------------------------------------------------- project and variant
class AstroProject(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroProject", display_name="Astro: Project", category=CATEGORY,
            description="Opens an astro_GPU project folder (the one holding project.json). Connect it to the first "
                        "processing node, or straight to a later one: a stage node computes what it needs first.",
            inputs=[io.String.Input("project_folder", default="",
                                    tooltip="Folder of the project (made by the astro_GPU interface or command line)")],
            outputs=[ASTRO.Output(display_name="astro"), io.String.Output(display_name="status")])

    @classmethod
    def fingerprint_inputs(cls, project_folder):
        # computed again when anything of the project changed on disk (another program, the interface)
        return bridge.project_signature(project_folder)

    @classmethod
    def execute(cls, project_folder) -> io.NodeOutput:
        from astro_gpu.core.project import Project
        p = Project(project_folder.strip().strip('"'))
        return io.NodeOutput(bridge.astro_value(p.root), bridge.status_text(p.root))


class AstroVariant(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroVariant", display_name="Astro: Variant", category=CATEGORY,
            description="The nodes after this one compute a variant: the same data processed another way, kept in "
                        "its own folders next to the main processing (empty name = the main processing). Set the "
                        "parameters that differ on the following nodes ('settings: this node').",
            inputs=[ASTRO.Input("astro"),
                    io.String.Input("name", default="", tooltip="Name of the variant (empty = main processing)"),
                    io.String.Input("note", default="", tooltip="What this variant tries (shown in the interface)")],
            outputs=[ASTRO.Output(display_name="astro")])

    @classmethod
    def execute(cls, astro, name, note) -> io.NodeOutput:
        name = name.strip()
        bridge.ensure_variant(astro["root"], name, note)
        return io.NodeOutput(bridge.astro_value(astro["root"], name, astro.get("stage"), astro.get("run_id")))


# ---------------------------------------------------------------------- one node per processing stage
def _stage_node(stage) -> type[io.ComfyNode]:
    descs = stage.Params.describe()
    by = {d["name"]: d for d in descs}

    def define_schema(cls) -> io.Schema:
        inputs = [ASTRO.Input("astro", tooltip="The project (and variant) to compute, from the previous node"),
                  io.Combo.Input("settings", options=[bridge.SETTINGS_FROM_PROJECT, bridge.SETTINGS_FROM_NODE],
                                 default=bridge.SETTINGS_FROM_PROJECT,
                                 tooltip="'project': the parameters already chosen for this project (the widgets "
                                         "below are ignored). 'this node': the widgets below; a widget left at its "
                                         "default value means automatic (chosen from the data).")]
        inputs += [_widget(d) for d in descs]
        return io.Schema(node_id=f"Astro_{stage.name}", display_name=f"Astro: {stage.title}",
                         category=f"{CATEGORY}/processing", description=stage.description,
                         inputs=inputs, outputs=[ASTRO.Output(display_name="astro")])

    def execute(cls, astro, settings, **params) -> io.NodeOutput:
        if settings == bridge.SETTINGS_FROM_NODE:
            bridge.apply_node_params(astro["root"], astro["variant"], stage.name,
                                     {k: _parse(by[k], v) for k, v in params.items() if k in by})
        return io.NodeOutput(_run(astro, stage.name))

    return type(f"Astro_{stage.name}", (io.ComfyNode,),
                {"define_schema": classmethod(define_schema), "execute": classmethod(execute),
                 "__doc__": stage.description})


STAGE_NODES = [_stage_node(s) for s in bridge.STAGES]


# ---------------------------------------------------------------------- rendering and export
def _source_options() -> list[str]:
    from astro_gpu.render.settings import PRESETS
    return ["delivered: faithful", "delivered: artistic", "interface settings"] + [f"preset: {k}" for k in PRESETS]


def _source(opt: str) -> str:
    if opt.startswith("delivered: "):
        return "results/" + opt.split(": ", 1)[1]
    if opt.startswith("preset: "):
        return opt.split(": ", 1)[1]
    return "project"


class AstroRender(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroRender", display_name="Astro: Render", category=CATEGORY,
            description="Renders the image of a processing (main or variant) with rendering settings: the delivered "
                        "renderings' settings, the interface's last settings or a preset, then a saved view, then "
                        "your own JSON changes. Several render nodes on the same processing share its layers.",
            inputs=[ASTRO.Input("astro"),
                    io.Combo.Input("source", options=_source_options(), default="delivered: faithful",
                                   tooltip="Where the rendering settings start from"),
                    io.String.Input("view", default="", tooltip="A view saved in the interface (its settings replace "
                                                                "the source); empty = none"),
                    io.String.Input("changes", default="", multiline=True,
                                    tooltip='Settings changed on top, as JSON, e.g. {"glare_nebula": 0, "chroma": 1.0}'),
                    io.Float.Input("scale", default=0.5, min=0.1, max=1.0, step=0.05,
                                   tooltip="Output size relative to the full (drizzled) resolution, "
                                           "rounded to 1, 1/2, 1/3, 1/4...")],
            outputs=[io.Image.Output(display_name="final"), io.Image.Output(display_name="starless"),
                     io.Image.Output(display_name="stars"), io.String.Output(display_name="settings")])

    @classmethod
    def execute(cls, astro, source, view, changes, scale) -> io.NodeOutput:
        import torch

        from astro_gpu.render.settings import RenderSettings
        ready = _ready_to_render(astro)
        eng, p = bridge.engine_for(ready["root"], ready["variant"])
        vals = bridge.render_settings(p, _source(source), view.strip(), changes)
        t0 = time.time()
        lay = eng.render(RenderSettings(**vals), float(scale))
        x0, x1, y0, y1 = eng.crop
        f = 1.0 / lay.scale
        sl = (slice(None), slice(int(y0 / f), int(y1 / f)), slice(int(x0 / f), int(x1 / f)))
        out = [getattr(lay, k)[sl].clamp(0, 1).permute(1, 2, 0).unsqueeze(0).float().cpu() for k in
               ("final", "starless", "stars")]
        del lay
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[astro] rendered {ready['root']} [{ready['variant'] or 'main'}] in {time.time() - t0:.1f} s")
        return io.NodeOutput(*out, json.dumps(vals))


class AstroExport(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroExport", display_name="Astro: Export", category=CATEGORY, is_output_node=True,
            description="Writes the rendering at full resolution into the project's results/<name>/: 16-bit TIFF "
                        "with colour profile, 16-bit PNG and JPEG at sensor scale, starless and stars layers, the "
                        "framing of your reference image, and the settings used.",
            inputs=[ASTRO.Input("astro"),
                    io.String.Input("settings", default="{}", force_input=True,
                                    tooltip="Rendering settings: connect the 'settings' output of an Astro: Render"),
                    io.String.Input("name", default="comfy", tooltip="Folder name in results/"),
                    io.Combo.Input("orientation", options=["as captured", "north up", "match reference"],
                                   default="as captured")],
            outputs=[io.String.Output(display_name="folder")])

    @classmethod
    def execute(cls, astro, settings, name, orientation) -> io.NodeOutput:
        from astro_gpu.render.export import ExportSettings, export
        from astro_gpu.render.settings import RenderSettings
        ready = _ready_to_render(astro)
        eng, p = bridge.engine_for(ready["root"], ready["variant"])
        on_progress, _ = _hooks()
        export(p, eng, RenderSettings(**json.loads(settings or "{}")),
               ExportSettings(name=name.strip() or "comfy", orientation=orientation),
               lambda frac, msg: on_progress("export", frac, msg))
        folder = str(p.results / (name.strip() or "comfy"))
        return io.NodeOutput(folder, ui={"text": [folder]})


class AstroRenderLive(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroRenderLive", display_name="Astro: Render (live)", category=CATEGORY, is_output_node=True,
            description="The rendering with its sliders and a live preview in the node: run the graph once (the "
                        "sliders then start from the chosen source), then every slider moved re-renders the preview "
                        "at once, without running the graph. Running the graph gives the image at the chosen scale "
                        "with the sliders' settings.",
            inputs=[ASTRO.Input("astro"),
                    io.Combo.Input("source", options=_source_options(), default="delivered: faithful",
                                   tooltip="Where the sliders start from (used while the sliders were not moved)"),
                    io.String.Input("view", default="", tooltip="A view saved in the interface (replaces the source)"),
                    io.String.Input("settings_json", default="", multiline=True,
                                    tooltip="The settings changed with the sliders, on top of the source (kept by the "
                                            "node; empty = the source as it is)"),
                    io.Float.Input("scale", default=0.5, min=0.1, max=1.0, step=0.05,
                                   tooltip="Output size relative to the full (drizzled) resolution, "
                                           "rounded to 1, 1/2, 1/3, 1/4...")],
            outputs=[io.Image.Output(display_name="final"), io.Image.Output(display_name="starless"),
                     io.Image.Output(display_name="stars"), io.String.Output(display_name="settings")])

    @classmethod
    def execute(cls, astro, source, view, settings_json, scale) -> io.NodeOutput:
        import torch

        from astro_gpu.render.settings import RenderSettings
        ready = _ready_to_render(astro)
        eng, p = bridge.engine_for(ready["root"], ready["variant"])
        base = bridge.render_settings(p, _source(source), view.strip(), "")
        vals = {**base, **json.loads(settings_json)} if settings_json.strip() else base   # the sliders' changes
        lay = eng.render(RenderSettings(**vals), float(scale))
        x0, x1, y0, y1 = eng.crop
        f = 1.0 / lay.scale
        sl = (slice(None), slice(int(y0 / f), int(y1 / f)), slice(int(x0 / f), int(x1 / f)))
        out = [getattr(lay, k)[sl].clamp(0, 1).permute(1, 2, 0).unsqueeze(0).float().cpu() for k in
               ("final", "starless", "stars")]
        del lay
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        info = {"root": ready["root"], "variant": ready["variant"], "settings": vals, "source_settings": base}
        return io.NodeOutput(*out, json.dumps(vals), ui={"astro_live": [info]})


class AstroStarMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroStarMask", display_name="Astro: Star mask", category=CATEGORY,
            description="A mask of the surroundings of the bright stars (from the Gaia catalogue matched to the image), "
                        "framed like an Astro: Render of the same processing at the same scale: to take one view "
                        "around the bright stars and another elsewhere (e.g. with ImageCompositeMasked).",
            inputs=[ASTRO.Input("astro"),
                    io.Float.Input("scale", default=0.5, min=0.1, max=1.0, step=0.05,
                                   tooltip="Same scale as the Astro: Render the mask goes with (rounded the "
                                           "same way: 1, 1/2, 1/3...)"),
                    io.Float.Input("brighter_than", default=9.0, min=2.0, max=14.0, step=0.5,
                                   tooltip="Stars brighter than this Gaia G magnitude"),
                    io.Float.Input("radius_g6", default=120.0, min=5.0, max=2000.0, step=5.0,
                                   tooltip="Radius around a G 6 star, in full-resolution pixels"),
                    io.Float.Input("growth", default=1.6, min=1.0, max=4.0, step=0.05,
                                   tooltip="Radius factor per magnitude brighter (1 = the same radius for all)"),
                    io.Float.Input("feather", default=0.3, min=0.0, max=1.0, step=0.05,
                                   tooltip="Soft edge, as a fraction of each radius")],
            outputs=[io.Mask.Output(display_name="mask")])

    @classmethod
    def execute(cls, astro, scale, brighter_than, radius_g6, growth, feather) -> io.NodeOutput:
        import numpy as np
        import torch
        ready = _ready_to_render(astro)
        eng, _ = bridge.engine_for(ready["root"], ready["variant"])
        x0, x1, y0, y1 = eng.crop
        f = float(max(1, int(round(1.0 / float(scale)))))     # as the rendering engine: 1, 1/2, 1/3...
        w, h = int(x1 / f) - int(x0 / f), int(y1 / f) - int(y0 / f)
        m = np.zeros((h, w), np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        sel = np.where((eng.G < brighter_than) & (eng.X > x0 - 2000) & (eng.X < x1 + 2000)
                       & (eng.Y > y0 - 2000) & (eng.Y < y1 + 2000))[0]
        for i in sel:
            r = radius_g6 * growth ** (6.0 - float(eng.G[i])) / f
            cx, cy = eng.X[i] / f - int(x0 / f), eng.Y[i] / f - int(y0 / f)
            r0 = max(r * (1 - feather), 0.0)
            xa, xb = int(max(cx - r - 2, 0)), int(min(cx + r + 3, w))
            ya, yb = int(max(cy - r - 2, 0)), int(min(cy + r + 3, h))
            if xa >= xb or ya >= yb:
                continue
            d = np.hypot(xx[ya:yb, xa:xb] - cx, yy[ya:yb, xa:xb] - cy)
            v = np.clip((r - d) / max(r - r0, 1e-3), 0, 1)
            v = v * v * (3 - 2 * v)                       # smooth edge
            m[ya:yb, xa:xb] = np.maximum(m[ya:yb, xa:xb], v)
        return io.NodeOutput(torch.from_numpy(m)[None])


class AstroSave16(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroSave16", display_name="Astro: Save 16-bit", category=CATEGORY, is_output_node=True,
            description="Saves any image (a render, a composite of views...) as a 16-bit TIFF with an sRGB colour "
                        "profile: into the project's results/<name>/ when a project is connected, else into "
                        "ComfyUI's output folder.",
            inputs=[io.Image.Input("image"),
                    io.String.Input("name", default="composite", tooltip="File and folder name"),
                    ASTRO.Input("astro", optional=True, tooltip="Optional: write into this project's results")],
            outputs=[io.String.Output(display_name="file")])

    @classmethod
    def execute(cls, image, name, astro=None) -> io.NodeOutput:
        from pathlib import Path

        import numpy as np
        import tifffile

        from astro_gpu.core.project import Project
        from astro_gpu.render.export import _icc
        name = (name or "composite").strip()
        if astro:
            out = Project(astro["root"]).results / name
        else:
            import folder_paths
            out = Path(folder_paths.get_output_directory()) / "astro"
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for k, img in enumerate(image):
            a = (np.clip(img.cpu().numpy(), 0, 1) * 65535 + 0.5).astype(np.uint16)
            f = out / (f"{name}.tif" if len(image) == 1 else f"{name}_{k + 1}.tif")
            tifffile.imwrite(f, a, photometric="rgb", iccprofile=_icc(), compression="zlib")
            files.append(str(f))
        return io.NodeOutput("; ".join(files), ui={"text": files})


class AstroExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        from .masks import MASK_NODES
        from .methods_node import AstroMethods
        from .project_node import AstroNewProject
        return [AstroNewProject, AstroProject, AstroVariant, AstroMethods, *STAGE_NODES, AstroRender, AstroRenderLive,
                AstroStarMask, AstroExport, AstroSave16, *MASK_NODES]


async def comfy_entrypoint() -> AstroExtension:
    return AstroExtension()
