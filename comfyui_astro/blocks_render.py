"""Rendering blocks for ComfyUI: the final image made of small blocks, each setting one aspect of the rendering.

'Prepare the rendering' makes, from the sky background (and everything upstream), the layers the rendering engine
works on. What then travels on the wires is a rendering: those layers plus the settings chosen so far. Each setting
block (white balance, stretch, saturation, contrast, sharpening, stars...) sets its own settings on the rendering it
receives and passes it on; a block left out keeps the engine's defaults (the faithful rendering). The engine applies
the settings in its own physical order (colour in linear light, then stretch, contrast, highlights, stars), so the
order of the blocks on the wire does not change the image. 'Show the rendering' renders the image at any point of
the chain, 'Save the image' writes it at full resolution. Two chains from the same preparation are two versions;
they combine zone by zone with the blocks of 'astro/8 combine'.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
from comfy_api.latest import io

from .blocks import SKY, _config, _key, _metrics, _param_widgets, _parsed, _run, _stage

RENDER = io.Custom("ASTRO_RENDER")
CAT_RENDER = "astro/7 rendering"
CAT_COMBINE = "astro/8 combine"

_ENGINES: OrderedDict = OrderedDict()
_LOCK = threading.Lock()
MAX_ENGINES = 2


# ---------------------------------------------------------------------- the rendering engine of a preparation
def _engine(render: dict, settings):
    """The rendering engine of a preparation (the learned one when the settings ask for it and it was prepared),
    kept loaded: the blocks rendering the same preparation share it."""
    import torch

    from astro_gpu.render.engine import RenderEngine
    d = render["learned_prep"] if settings.learned_stars and render.get("learned_prep") else render["prep"]
    with _LOCK:
        if d in _ENGINES:
            _ENGINES.move_to_end(d)
            return _ENGINES[d]
        while len(_ENGINES) >= MAX_ENGINES:
            _ENGINES.popitem(last=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        eng = RenderEngine(d, device="cuda:0", calibrated_white=render.get("white", 5778.0))
        _ENGINES[d] = eng
        return eng


def _settings(render: dict):
    from astro_gpu.render.settings import RenderSettings
    return RenderSettings(**render["settings"])


def _layers(render: dict, scale: float):
    """final, starless, stars of a rendering at a scale, cropped to the valid image, as ComfyUI images."""
    import torch
    s = _settings(render)
    eng = _engine(render, s)
    with torch.inference_mode():
        lay = eng.render(s, float(scale))
    x0, x1, y0, y1 = eng.crop
    f = 1.0 / lay.scale
    sl = (slice(None), slice(int(y0 / f), int(y1 / f)), slice(int(x0 / f), int(x1 / f)))
    out = [getattr(lay, k)[sl].clamp(0, 1).permute(1, 2, 0).unsqueeze(0).float().cpu()
           for k in ("final", "starless", "stars")]
    del lay
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out, eng


# ---------------------------------------------------------------------- preparation
class AstroPrepareRendering(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        st = _stage("render_prep")
        return io.Schema(
            node_id="AstroPrepareRendering", display_name="Prepare the rendering", category=CAT_RENDER,
            description="Makes the layers the rendering works on, from the sky background and everything upstream: "
                        "the calibrated nebula at full resolution, its fine detail and colour layers, the stars as "
                        "photographed (saturated cores, spikes) and their measured profile. When the star removal "
                        "made a learned starless image, its layers are prepared too (chosen in 'Stars: removal and "
                        "spikes'). About a minute.",
            inputs=[SKY.Input("sky")] + _param_widgets(st),
            outputs=[RENDER.Output(display_name="rendering"), io.String.Output(display_name="summary")])

    @classmethod
    def execute(cls, sky, **kw) -> io.NodeOutput:
        st = _stage("render_prep")
        widgets = _parsed(st, kw)
        key = _key("render_prep", st.version, sky["key"], widgets)
        out = _run(st, sky["dirs"], key, widgets)
        learned = None
        if sky.get("learned"):
            sl = _stage("render_prep_learned")
            lkey = _key("render_prep_learned", sl.version, sky["key"], widgets)
            lout = _run(sl, {**sky["dirs"], "render_prep": str(out)}, lkey, widgets)
            if (lout / "N.npy").exists():
                learned = str(lout)
        white = 5778.0
        col = Path(sky["dirs"].get("color", "")) / "manifest.json"
        if col.is_file():
            white = float(json.loads(col.read_text(encoding="utf-8"))["params"].get("white_temp", 5778.0))
        prep = json.loads((out / "prep.json").read_text(encoding="utf-8"))
        x0, x1, y0, y1 = prep["crop"]
        summary = (f"image {x1 - x0}x{y1 - y0} px, star removal: model"
                   + (" and learned" if learned else ""))
        m = _metrics(out)
        if m.get("guarded_pct") is not None:
            summary += f", star guard on {m['guarded_pct']} %"
        return io.NodeOutput({"kind": "rendering", "key": key, "prep": str(out), "learned_prep": learned,
                              "white": white, "settings": {}, "dirs": sky["dirs"]}, summary)


# ---------------------------------------------------------------------- setting blocks
def _render_widget(d: dict):
    """A widget for a rendering setting (its default is the faithful rendering)."""
    parts = [d.get("help", ""), d.get("tip", ""), f"Unit: {d['unit']}." if d.get("unit") else "",
             f"Default: {d.get('default')!r} (the faithful rendering)."]
    common = {"display_name": d.get("label") or d["name"], "tooltip": "\n".join(p for p in parts if p),
              "advanced": d.get("level") == "advanced"}
    name, typ, default = d["name"], d.get("type", ""), d.get("default")
    if typ == "bool":
        return io.Boolean.Input(name, default=bool(default), **common)
    if typ == "int":
        return io.Int.Input(name, default=int(default), min=int(d["ge"]) if "ge" in d else None,
                            max=int(d["le"]) if "le" in d else None, step=int(d.get("step") or 1), **common)
    return io.Float.Input(name, default=float(default), min=float(d["ge"]) if "ge" in d else None,
                          max=float(d["le"]) if "le" in d else None, step=float(d.get("step") or 0.01),
                          round=False, **common)


# block id, title, description, settings it owns
SETTING_BLOCKS = [
    ("AstroWhiteBalance", "White balance",
     "The colour temperature shown as neutral and the green / magenta tint. The calibrated values show stars and "
     "nebulae in their measured colours; moving them is an aesthetic choice.",
     ["white_temp", "tint"]),
    ("AstroStretch", "Stretch",
     "Brings the linear image to the screen: how bright the sky background is, how dark the darkest clouds go, the "
     "softness of the shadows and the white point of the nebula.",
     ["sky_level", "dark_level", "toe", "white_mult", "sky_pct", "dark_pct"]),
    ("AstroSkyGlow", "Sky glow and specks",
     "The diffuse light of the sky: the Milky Way glow of the stars too faint to draw, the diffuse glow removed, a "
     "neutral (grey) sky, the dark specks removed.",
     ["glow_k", "glow_sigma", "g_glow", "diffuse_cut", "sky_neutral", "speck_k"]),
    ("AstroEmissionColours", "Emission colours",
     "The colours of the gas: the H-alpha red (strength and hue), the OIII teal, the blue of the reflection "
     "nebulae.",
     ["ha_boost", "ha_sigma", "hue_target", "hue_shift", "oiii_boost", "oiii_smooth", "oiii_threshold",
      "resid_boost"]),
    ("AstroSaturation", "Saturation",
     "How saturated the nebula and background are (perceptual), the fine colour kept, the colour of the faint "
     "areas smoothed by their signal-to-noise.",
     ["chroma", "chroma_protect", "colour_detail", "chroma_sigma", "snr_lo", "snr_hi"]),
    ("AstroContrast", "Contrast",
     "Local contrast at three scales (fine, medium, large) and the compression of the large bright areas (so a "
     "bright core keeps its structure).",
     ["lc_fine", "lc_medium", "lc_large", "hdr_strength", "hdr_knee", "hdr_r", "hdr_eps"]),
    ("AstroSharpening", "Sharpening",
     "Sharpening of the nebula: the classic sharpening, and the sharpening network trained on your stack (the "
     "measured star shape undone), from a brightness on.",
     ["neb_sharp", "sharp_white", "sharpen", "sharpen_lo_pct"]),
    ("AstroDetailAndSmoothing", "Fine detail and smoothing",
     "The fine detail of the denoised image brought back in the bright areas, the faint areas smoothed.",
     ["fine_k", "fine_lo_pct", "fine_hi_pct", "faint_smooth", "faint_lo_pct", "faint_hi_pct"]),
    ("AstroHighlights", "Highlights",
     "The brightest parts: where the highlight compression starts, its ceiling, the detail kept in it, its colour.",
     ["hl_knee", "hl_max", "hl_detail", "hl_dmin", "hl_r1", "hl_r2", "hl_eps", "hl_shoulder", "hl_keep_chroma"]),
    ("AstroStarRemovalChoice", "Stars: removal and spikes",
     "Stars shown or not, which starless image (learned star removal or star model), the saturated stars and the "
     "spikes as photographed, the glare of the bright stars kept with the nebula, the colour taken out of their halo.",
     ["stars", "learned_stars", "real_stars", "stars_photographed", "spike_photo", "spike_g_max", "spike_smooth",
      "glare_nebula", "bright_halo_desat"]),
    ("AstroStarLook", "Stars: size and colour",
     "How the stars are drawn: the faintest one drawn and its brightness, the size of bright stars, their glow, "
     "colour and white cores.",
     ["g_lim", "t_lim", "d10", "d7", "t9", "star_glow", "star_sat", "sat_min", "core_white"]),
]


def _setting_block(node_id: str, title: str, desc: str, keys: list[str]) -> type[io.ComfyNode]:
    def define_schema(cls) -> io.Schema:
        from astro_gpu.render.settings import RenderSettings
        by = {d["name"]: d for d in RenderSettings.describe()}
        return io.Schema(node_id=node_id, display_name=title, category=CAT_RENDER,
                         description=desc + " A block left out keeps the faithful rendering's values.",
                         inputs=[RENDER.Input("rendering")] + [_render_widget(by[k]) for k in keys],
                         outputs=[RENDER.Output(display_name="rendering")])

    def execute(cls, rendering, **kw) -> io.NodeOutput:
        vals = {k: v for k, v in kw.items() if k in keys}
        return io.NodeOutput({**rendering, "settings": {**rendering["settings"], **vals}})

    return type(node_id, (io.ComfyNode,), {"define_schema": classmethod(define_schema),
                                           "execute": classmethod(execute), "__doc__": desc})


SETTING_NODES = [_setting_block(*b) for b in SETTING_BLOCKS]


class AstroRenderingPreset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        from astro_gpu.render.settings import PRESETS
        return io.Schema(
            node_id="AstroRenderingPreset", display_name="Rendering preset", category=CAT_RENDER,
            description="Sets all the settings of a ready-made rendering (Faithful, Artistic, H-alpha emphasised...). "
                        "The setting blocks after it change what they own; put it first in the chain.",
            inputs=[RENDER.Input("rendering"),
                    io.Combo.Input("preset", options=list(PRESETS), default="Faithful")],
            outputs=[RENDER.Output(display_name="rendering")])

    @classmethod
    def execute(cls, rendering, preset) -> io.NodeOutput:
        from astro_gpu.render.settings import PRESETS
        return io.NodeOutput({**rendering, "settings": dict(PRESETS[preset])})


# ---------------------------------------------------------------------- showing and saving
class AstroShowRendering(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroShowRendering", display_name="Show the rendering", category=CAT_RENDER,
            description="Renders the image with the settings set so far on the chain: the final image, the starless "
                        "nebula and the stars alone, at a reduced scale for speed (1 = full resolution). Connect "
                        "several along a chain to see what each block changes.",
            inputs=[RENDER.Input("rendering"),
                    io.Float.Input("scale", default=0.33, min=0.1, max=1.0, step=0.01,
                                   tooltip="Size relative to the full (drizzled) resolution, rounded to 1, 1/2, "
                                           "1/3, 1/4...")],
            outputs=[io.Image.Output(display_name="final"), io.Image.Output(display_name="starless"),
                     io.Image.Output(display_name="stars"), io.String.Output(display_name="settings")])

    @classmethod
    def execute(cls, rendering, scale) -> io.NodeOutput:
        (final, starless, stars), _ = _layers(rendering, scale)
        return io.NodeOutput(final, starless, stars, json.dumps(rendering["settings"]))


class AstroSaveImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroSaveImage", display_name="Save the image", category=CAT_RENDER, is_output_node=True,
            description="Writes the image at full resolution into a folder: 16-bit TIFF with an sRGB colour profile "
                        "and a JPEG, the starless and stars layers too when a rendering is connected, and the "
                        "settings used. Takes a rendering, or any image (a blend of versions).",
            inputs=[RENDER.Input("rendering", optional=True, tooltip="A rendering (saved at full resolution)"),
                    io.Image.Input("image", optional=True, tooltip="Or an image (a blend of versions), saved as it is"),
                    io.String.Input("name", default="faithful", tooltip="File name"),
                    io.String.Input("folder", default="", tooltip="Folder; empty = <work folder>/images")],
            outputs=[io.String.Output(display_name="files")])

    @classmethod
    def execute(cls, name, folder, rendering=None, image=None) -> io.NodeOutput:
        import cv2
        import tifffile

        from astro_gpu.render.export import _icc
        if (rendering is None) == (image is None):
            raise ValueError("connect a rendering or an image (one of the two)")
        name = (name or "image").strip()
        out = Path(folder.strip()) if folder.strip() else Path(_config()["work"]) / "images"
        out.mkdir(parents=True, exist_ok=True)
        layers = {}
        if rendering is not None:
            (final, starless, stars), _ = _layers(rendering, 1.0)
            layers = {name: final[0], f"{name}_starless": starless[0], f"{name}_stars": stars[0]}
            (out / f"{name}_settings.json").write_text(json.dumps(
                {"settings": rendering["settings"], "preparation": rendering["prep"],
                 "learned_preparation": rendering.get("learned_prep"), "date": datetime.now().isoformat()},
                indent=1), encoding="utf-8")
        else:
            layers = {name if len(image) == 1 else f"{name}_{k + 1}": im for k, im in enumerate(image)}
        files = []
        for n, im in layers.items():
            a = np.clip(im.cpu().numpy(), 0, 1)
            tifffile.imwrite(out / f"{n}.tif", (a * 65535 + 0.5).astype(np.uint16), photometric="rgb",
                             iccprofile=_icc(), compression="zlib")
            files.append(str(out / f"{n}.tif"))
            if n == name or rendering is None:
                cv2.imwrite(str(out / f"{n}.jpg"), (a[..., ::-1] * 255 + 0.5).astype(np.uint8),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                files.append(str(out / f"{n}.jpg"))
        return io.NodeOutput("; ".join(files), ui={"text": files})


class AstroBrightStarMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroBrightStarMask", display_name="Bright star mask", category=CAT_COMBINE,
            description="A mask of the surroundings of the bright stars (Gaia stars on the image), framed like 'Show "
                        "the rendering' of the same preparation at the same scale: to take one version around the "
                        "bright stars and another elsewhere ('Blend versions').",
            inputs=[RENDER.Input("rendering"),
                    io.Float.Input("scale", default=0.33, min=0.1, max=1.0, step=0.01,
                                   tooltip="Same scale as the rendering the mask goes with"),
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
    def execute(cls, rendering, scale, brighter_than, radius_g6, growth, feather) -> io.NodeOutput:
        import torch
        eng = _engine(rendering, _settings(rendering))
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
            m[ya:yb, xa:xb] = np.maximum(m[ya:yb, xa:xb], v * v * (3 - 2 * v))
        return io.NodeOutput(torch.from_numpy(m)[None])


RENDER_NODES = [AstroPrepareRendering, AstroRenderingPreset, *SETTING_NODES, AstroShowRendering, AstroSaveImage,
                AstroBrightStarMask]
