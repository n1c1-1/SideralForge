"""Elementary processing blocks for ComfyUI: data travels on the wires, each block does one thing.

Stage 1 of the block library: sources (light frames, master offset / dark / flat from their folders), calibration,
alignment, frame selection, stacking. Each block runs the processing code of astro_gpu (the same as the heavy
client) in a small context made of the folders of the blocks wired upstream: no project, nothing hidden. Its result
is kept on disk under a fingerprint of its inputs and settings (work folder, see blocks_config.json), so a block is
not computed again after a restart of ComfyUI as long as nothing upstream changed.

What travels on the wires (dicts): frames (a list of raw frames and their metadata), master (one calibration master),
calibrated frames, aligned frames, linear image (the stack and its two half-stacks).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from comfy_api.latest import io

from . import bridge

FRAMES = io.Custom("ASTRO_FRAMES")
MASTER = io.Custom("ASTRO_MASTER")
CALIBRATED = io.Custom("ASTRO_CALIBRATED")
ALIGNED = io.Custom("ASTRO_ALIGNED")
LINEAR = io.Custom("ASTRO_LINEAR")
STARS = io.Custom("ASTRO_STARS")
SURVEY = io.Custom("ASTRO_SURVEY")

CAT_SOURCES = "astro/1 sources"
CAT_CALIB = "astro/2 calibration"
CAT_ALIGN = "astro/3 alignment"
CAT_STACK = "astro/4 stacking"
CAT_FIELD = "astro/5 field and colour"
CONFIG = Path(__file__).with_name("blocks_config.json")


def _config() -> dict:
    d = {"work": "E:/astro_gpu/blocks", "scratch": str(Path.home() / "astro_gpu_scratch" / "blocks")}
    if CONFIG.exists():
        d.update(json.loads(CONFIG.read_text(encoding="utf-8")))
    return d


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _folders(text: str) -> list[str]:
    return [ln.strip().strip('"') for ln in (text or "").splitlines() if ln.strip()]


# ---------------------------------------------------------------------- running astro_gpu code inside a block
class _BlockProject:
    """What the processing code asks of a project, made of the folders of the blocks upstream."""

    def __init__(self, dirs: dict[str, Path], out: Path, scratch: Path, inputs=None):
        from astro_gpu.core.project import Inputs
        self.dirs = dict(dirs)
        self.root = out
        self.logs = out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.scratch = scratch
        self.results = out / "results"
        self.config = SimpleNamespace(name="blocks", inputs=inputs or Inputs(), params={}, render={}, devices={},
                                      variants={},
                                      active_variant="", target=SimpleNamespace(name="", ra=None, dec=None,
                                                                                focal_mm=None))

    def stage_dir(self, name: str) -> Path:
        if name not in self.dirs:
            from astro_gpu.core.stage import StageError
            raise StageError(f"this block needs the result of '{name}': connect the block that makes it")
        return Path(self.dirs[name])


def _run(stage, dirs: dict[str, str], key: str, widgets: dict, run=None, inputs=None) -> Path:
    """Run ``stage`` (its ``run``, or ``run(ctx, params)``) in the block folder for ``key`` unless it is already
    done there; ``widgets`` are the parameter values set on the block (a value left at its default = automatic)."""
    import torch

    from astro_gpu.core.params import resolve
    from astro_gpu.core.stage import Context

    from .nodes import _hooks
    cfg = _config()
    out = Path(cfg["work"]) / stage.name / key
    man = out / "manifest.json"
    if man.exists() and json.loads(man.read_text(encoding="utf-8")).get("status") == "done":
        return out
    out.mkdir(parents=True, exist_ok=True)
    proj = _BlockProject({**{k: Path(v) for k, v in dirs.items()}, stage.name: out}, out,
                         Path(cfg["scratch"]) / key, inputs)
    (proj.logs / "STOP").unlink(missing_ok=True)
    on_progress, interrupted = _hooks()
    journal = bridge.HookJournal(proj.logs, on_progress, interrupted)
    ctx = Context(proj, stage, journal, "cuda:0")
    defaults = {d["name"]: d["default"] for d in stage.Params.describe()}
    overrides = {k: v for k, v in widgets.items() if k in defaults and v != defaults[k]}
    with torch.inference_mode(False), torch.enable_grad(), bridge._spawn_without_main():
        auto = stage.auto(ctx)
        p, _ = resolve(stage.Params, auto, overrides)
        journal.emit("stage_start", stage=stage.name, title=stage.title)
        (run or stage.run)(ctx, p)
        journal.emit("stage_end", stage=stage.name, status="done", duration=0)
    man.write_text(json.dumps({"status": "done", "stage": stage.name, "version": stage.version,
                               "params": p.model_dump(mode="json"), "metrics": ctx.metrics}, indent=1, default=str),
                   encoding="utf-8")
    return out


def _preview(folder: Path, name: str):
    """A block's preview image (saved by the processing code) as a ComfyUI image, or a black square."""
    import cv2
    import torch
    f = folder / "previews" / f"{name}.jpg"
    a = cv2.imread(str(f)) if f.exists() else None
    if a is None:
        a = np.zeros((64, 64, 3), np.uint8)
    return torch.from_numpy(a[:, :, ::-1].astype(np.float32) / 255.0)[None]


def _metrics(folder: Path) -> dict:
    return json.loads((folder / "manifest.json").read_text(encoding="utf-8")).get("metrics", {})


def _stage(name: str):
    return bridge.BY_NAME[name]


def _param_widgets(stage, keep=None, drop=()):
    from .nodes import _widget
    return [_widget(d) for d in stage.Params.describe()
            if (keep is None or d["name"] in keep) and d["name"] not in drop]


def _parsed(stage, kw: dict) -> dict:
    from .nodes import _parse
    by = {d["name"]: d for d in stage.Params.describe()}
    return {k: _parse(by[k], v) for k, v in kw.items() if k in by}


# ---------------------------------------------------------------------- 1. sources
class AstroLights(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroLights", display_name="Light frames", category=CAT_SOURCES,
            description="The light frames of a session: one or several folders (or files), one per line, RAW or "
                        "FITS. Reads their metadata (size, colour pattern, exposure, camera) and checks they are "
                        "consistent. Shows the middle frame, uncalibrated.",
            inputs=[io.String.Input("folders", multiline=True, default="",
                                    tooltip="Folders (or files) of light frames, one per line"),
                    io.Int.Input("max_frames", default=0, min=0, max=100000, advanced=True,
                                 tooltip="Use only N frames spread over the session (quick test); 0 = all")],
            outputs=[FRAMES.Output(display_name="frames"), io.Image.Output(display_name="middle frame"),
                     io.String.Output(display_name="summary")])

    @classmethod
    def execute(cls, folders, max_frames=0) -> io.NodeOutput:
        from astro_gpu.core.project import Inputs, list_frames
        fl = _folders(folders)
        files = [str(f) for f in list_frames(fl)]
        if not files:
            raise ValueError("no RAW or FITS frame found in: " + "; ".join(fl or ["(nothing given)"]))
        st = _stage("inventory")
        key = _key("lights", st.version, files, max_frames)
        out = _run(st, {}, key, {"max_lights": max_frames}, inputs=Inputs(lights=fl))
        m = _metrics(out)
        summary = (f"{m.get('n_lights')} frames of {m.get('exposure_s')} s ({m.get('integration_h')} h), "
                   f"{m.get('width')}x{m.get('height')} {m.get('cfa')}, camera {m.get('camera')}")
        return io.NodeOutput({"kind": "lights", "key": key, "dirs": {"inventory": str(out)}},
                             _preview(out, "middle_frame"), summary)


class _MasterStage:
    """A minimal stage for the master blocks (they have no setting of their own beyond the block's widgets)."""

    def __init__(self, name, title):
        from astro_gpu.core.params import StageParams
        self.name, self.title, self.version, self.Params = name, title, "1", StageParams

    def auto(self, ctx):
        return {}


def _master(kind: str, title: str, folder: str, make) -> dict:
    from astro_gpu.core.project import list_frames
    files = [str(f) for f in list_frames(_folders(folder))]
    if not files:
        raise ValueError(f"no RAW or FITS frame found in: {folder or '(nothing given)'}")
    st = _MasterStage(f"master_{kind}", title)
    key = _key("master", kind, files, make.__name__, getattr(make, "extra", None))

    def run(ctx, p):
        arr, metrics = make(files, ctx)
        ctx.save("master.npy", arr.astype(np.float32))
        for k, v in metrics.items():
            ctx.metric(k, v)
        ctx.metric("frames", len(files))
        v = arr.astype(np.float32)
        lo, hi = np.percentile(v[::4, ::4], [1, 99.5])
        ctx.preview(kind, np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1))
    out = _run(st, {}, key, {}, run=run)
    m = _metrics(out)
    return {"kind": kind, "key": key, "file": str(out / "master.npy"), "frames": files, "metrics": m,
            "dir": str(out)}


class AstroMasterBias(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroMasterBias", display_name="Master offset", category=CAT_SOURCES,
            description="Master offset (bias): the median of the offset frames of a folder.",
            inputs=[io.String.Input("folder", multiline=True, default="", tooltip="Folder(s) of offset frames")],
            outputs=[MASTER.Output(display_name="offset"), io.Image.Output(display_name="view")])

    @classmethod
    def execute(cls, folder) -> io.NodeOutput:
        from astro_gpu.stages.calibrate import master_bias

        def make(files, ctx):
            return master_bias(files, ctx), {}
        v = _master("bias", "Master offset", folder, make)
        return io.NodeOutput(v, _preview(Path(v["dir"]), "bias"))


class AstroMasterDark(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroMasterDark", display_name="Master dark", category=CAT_SOURCES,
            description="Master dark: the median of the dark frames of a folder (same exposure and temperature as the "
                        "lights). Also measures the noise of one dark frame (read noise for the stacking).",
            inputs=[io.String.Input("folder", multiline=True, default="", tooltip="Folder(s) of dark frames")],
            outputs=[MASTER.Output(display_name="dark"), io.Image.Output(display_name="view")])

    @classmethod
    def execute(cls, folder) -> io.NodeOutput:
        from astro_gpu.stages.calibrate import master_dark

        def make(files, ctx):
            dark, noise = master_dark(files, None, 0, 0, 0.0, ctx)
            return dark, {"dark_frame_noise": round(float(noise), 2) if noise else None}
        v = _master("dark", "Master dark", folder, make)
        return io.NodeOutput(v, _preview(Path(v["dir"]), "dark"))


class AstroMasterFlat(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroMasterFlat", display_name="Master flat", category=CAT_SOURCES,
            description="Master flat: the flat frames of a folder, minus their own dark (flat darks if given, else "
                        "the offset if connected, else the black level of the camera), combined with rejection and "
                        "normalised to 1 at the centre. Shows the vignetting and the dust.",
            inputs=[io.String.Input("folder", multiline=True, default="", tooltip="Folder(s) of flat frames"),
                    io.String.Input("flat_darks", multiline=True, default="", optional=True,
                                    tooltip="Optional: folder(s) of dark frames taken with the flats"),
                    MASTER.Input("offset", optional=True, tooltip="Optional: master offset (Master offset block)"),
                    io.Float.Input("large_scales_only", default=0.0, min=0.0, max=500.0, step=5.0, optional=True,
                                   tooltip="Keep only the scales of the flat larger than this (px): for flats "
                                           "borrowed from another night - their vignetting, not their dust. 0 = all")],
            outputs=[MASTER.Output(display_name="flat"), io.Image.Output(display_name="view")])

    @classmethod
    def execute(cls, folder, flat_darks="", offset=None, large_scales_only=0.0) -> io.NodeOutput:
        from astro_gpu.core.project import list_frames
        from astro_gpu.io.frames import read_info
        from astro_gpu.stages.calibrate import master_flat
        fd = [str(f) for f in list_frames(_folders(flat_darks))]
        bias = np.load(offset["file"]) if offset else None

        def make(files, ctx):
            inf = read_info(files[0])
            flat = master_flat(files, fd, bias, float(inf.black), inf.height, inf.width, float(large_scales_only),
                               ctx)
            return flat, {"vignetting_min": round(float(flat.min()), 3)}
        make.extra = [fd, offset["key"] if offset else None, large_scales_only]
        v = _master("flat", "Master flat", folder, make)
        return io.NodeOutput(v, _preview(Path(v["dir"]), "flat"))


# ---------------------------------------------------------------------- 2. calibration
class AstroCalibrate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        st = _stage("calibrate")
        return io.Schema(
            node_id="AstroCalibrate", display_name="Calibrate", category=CAT_CALIB,
            description="Calibrates the light frames with the masters connected: minus the dark (or the offset, or "
                        "the black level), divided by the flat; hot, cold and dusty pixels are excluded. The frames "
                        "stay raw colour mosaics (no debayering) and are cached on the scratch disk.",
            inputs=[FRAMES.Input("lights"),
                    MASTER.Input("dark", optional=True, tooltip="Master dark (Master dark block)"),
                    MASTER.Input("flat", optional=True, tooltip="Master flat (Master flat block)"),
                    MASTER.Input("offset", optional=True, tooltip="Master offset, used when there is no dark")]
                   + _param_widgets(st, drop=("flat_scale_px",)),
            outputs=[CALIBRATED.Output(display_name="calibrated"), io.Image.Output(display_name="calibrated frame"),
                     io.String.Output(display_name="summary")])

    @classmethod
    def execute(cls, lights, dark=None, flat=None, offset=None, **kw) -> io.NodeOutput:
        from astro_gpu.stages.calibrate import calibrate_lights, master_dark
        st = _stage("calibrate")
        widgets = _parsed(st, kw)
        inv = Path(lights["dirs"]["inventory"])
        im = _metrics(inv)
        H, W = int(im["height"]), int(im["width"])
        for m in (dark, flat, offset):
            if m is not None:
                shp = np.load(m["file"], mmap_mode="r").shape
                if shp != (H, W):
                    raise ValueError(f"the master {m['kind']} is {shp[1]}x{shp[0]}, the lights {W}x{H}")
        key = _key("calibrate", st.version, lights["key"], [m["key"] if m else None for m in (dark, flat, offset)],
                   {k: v for k, v in widgets.items() if k != "workers"})

        def run(ctx, p):
            frames = json.loads((inv / "frames.json").read_text(encoding="utf-8"))
            bias = np.load(offset["file"]) if offset else None
            if dark is not None:
                d, noise = np.load(dark["file"]), dark["metrics"].get("dark_frame_noise")
            else:
                d, noise = master_dark([], bias, H, W, float(im["black_level"]), ctx)
            f = np.load(flat["file"]) if flat is not None else np.ones((H, W), np.float32)
            if flat is None:
                ctx.log("no master flat: vignetting and dust will not be corrected", "warning")
            calibrate_lights(ctx, p, frames["lights"], d, f, dark is not None, noise, im["cfa"],
                             float(im["white_level"]))
        out = _run(st, lights["dirs"], key, widgets, run=run)
        m = _metrics(out)
        summary = (f"{m.get('cache_gb')} GB of calibrated frames, {m.get('bad_pixels_pct')} % pixels excluded, "
                   f"sky noise {m.get('sky_noise_adu')} ADU")
        v = {"kind": "calibrated", "key": key, "dirs": {**lights["dirs"], "calibrate": str(out)}}
        return io.NodeOutput(v, _preview(out, "calibrated_frame"), summary)


# ---------------------------------------------------------------------- 3. alignment
class AstroAlign(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        st = _stage("register")
        return io.Schema(
            node_id="AstroAlign", display_name="Align", category=CAT_ALIGN,
            description="Detects the stars of every calibrated frame, takes the sharpest frame of the middle of the "
                        "night as reference and fits for each frame a polynomial mapping it onto the reference "
                        "(rotation, refraction and distortion included). Measures per frame the star size and the "
                        "transparency: the curves are shown.",
            inputs=[CALIBRATED.Input("calibrated")] + _param_widgets(st),
            outputs=[ALIGNED.Output(display_name="aligned"), io.Image.Output(display_name="frame quality"),
                     io.String.Output(display_name="summary")])

    @classmethod
    def execute(cls, calibrated, **kw) -> io.NodeOutput:
        st = _stage("register")
        widgets = _parsed(st, kw)
        key = _key("align", st.version, calibrated["key"], {k: v for k, v in widgets.items() if k != "workers"})
        out = _run(st, calibrated["dirs"], key, widgets)
        m = _metrics(out)
        summary = (f"{m.get('aligned')} frames aligned, residual {m.get('rms_px')} px, star size {m.get('fwhm_px')} "
                   f"px, transparency {m.get('transparency_p10_p90')}, drift {m.get('drift_px')} px")
        v = {"kind": "aligned", "key": key, "dirs": {**calibrated["dirs"], "register": str(out)}}
        return io.NodeOutput(v, _preview(out, "frame_quality"), summary)


class AstroFrameSelection(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroFrameSelection", display_name="Frame selection", category=CAT_ALIGN,
            description="Chooses the frames to stack and their weight: (transparency / noise)^2 x sharpness; frames "
                        "hazed or clouded (transparency below the threshold x the best frames) or with stars too "
                        "large (wind, focus) are left out. Without this block the stacking uses the same rule with "
                        "its own settings.",
            inputs=[ALIGNED.Input("aligned"),
                    io.Float.Input("min_transparency", default=0.6, min=0.0, max=1.0, step=0.05,
                                   tooltip="Frames below this fraction of the best frames' star flux are left out"),
                    io.Float.Input("max_star_size", default=2.0, min=1.0, max=5.0, step=0.1,
                                   tooltip="Frames whose stars are larger than this x the median are left out")],
            outputs=[ALIGNED.Output(display_name="selected"), io.String.Output(display_name="summary")])

    @classmethod
    def execute(cls, aligned, min_transparency, max_star_size) -> io.NodeOutput:
        from astro_gpu.stages.stack import select_frames
        reg_dir = Path(aligned["dirs"]["register"])
        key = _key("selection", aligned["key"], min_transparency, max_star_size)
        out = Path(_config()["work"]) / "selection" / key
        if not (out / "selection.npz").exists():
            out.mkdir(parents=True, exist_ok=True)
            reg = np.load(reg_dir / "reg.npz")
            stats = np.load(Path(aligned["dirs"]["calibrate"]) / "frame_stats.npy")
            weight, reasons = select_frames(reg["transp"].astype(float), reg["fwhm"], reg["rms"], stats,
                                            min_transparency, max_star_size)
            shutil.copy2(reg_dir / "reg.npz", out / "reg.npz")
            shutil.copy2(reg_dir / "manifest.json", out / "manifest.json")
            np.savez(out / "selection.npz", weight=weight, reasons=np.asarray(reasons, dtype=str))
        sel = np.load(out / "selection.npz")
        reasons = sel["reasons"]
        n, kept = len(reasons), int((reasons == "").sum())
        parts = [f"{kept}/{n} frames kept"] + [f"{int((reasons == r).sum())} {r}" for r in
                                               ("low transparency", "large stars", "not aligned")
                                               if (reasons == r).any()]
        v = {"kind": "aligned", "key": key, "dirs": {**aligned["dirs"], "register": str(out)}}
        return io.NodeOutput(v, ", ".join(parts))


# ---------------------------------------------------------------------- 4. stacking
class AstroStackBlock(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        st = _stage("stack")
        return io.Schema(
            node_id="AstroStackBlock", display_name="Stack", category=CAT_STACK,
            description="Stacks the aligned frames straight from their raw colour pixels onto a finer grid (drizzle, "
                        "no debayering), each frame weighted, outliers rejected (satellites, planes, cosmic rays). "
                        "Also makes two independent half-stacks, used by the denoising. The result is a linear "
                        "image (not stretched); the view shown is stretched for display only.",
            inputs=[ALIGNED.Input("aligned")] + _param_widgets(st),
            outputs=[LINEAR.Output(display_name="linear"), io.Image.Output(display_name="view"),
                     io.String.Output(display_name="summary")])

    @classmethod
    def execute(cls, aligned, **kw) -> io.NodeOutput:
        st = _stage("stack")
        widgets = _parsed(st, kw)
        key = _key("stack", st.version, aligned["key"], widgets)
        out = _run(st, aligned["dirs"], key, widgets)
        m = _metrics(out)
        summary = (f"{m.get('frames_used')} frames ({m.get('integration_used_h')} h), grid {m.get('grid')}, "
                   f"{m.get('rejected_pct')} % pixels rejected")
        v = {"kind": "linear", "key": key, "dirs": {**aligned["dirs"], "stack": str(out)}}
        return io.NodeOutput(v, _preview(out, "stack"), summary)


# ---------------------------------------------------------------------- 5. field and colour
def _same_stack(*values) -> dict:
    """The folders of several inputs, which must come from the same stack (one image, several results on it)."""
    stacks = {v["dirs"].get("stack") for v in values}
    if len(stacks) > 1:
        raise ValueError("these inputs come from different stacks: connect results of the same image")
    dirs: dict = {}
    for v in values:
        dirs.update(v["dirs"])
    return dirs


def _need_field(v: dict) -> None:
    if "astrometry" not in v["dirs"]:
        raise ValueError("identify the field first (block 'Identify the field')")


class AstroIdentifyField(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        st = _stage("astrometry")
        return io.Schema(
            node_id="AstroIdentifyField", display_name="Identify the field", category=CAT_FIELD,
            description="Finds exactly where the image points on the sky (its stars matched to the Gaia catalogue). "
                        "The object name, or its coordinates, and the focal length make it immediate; without them "
                        "the field is searched blind (a few minutes). Every later step that uses the sky (Gaia "
                        "stars, colour, surveys) relies on it.",
            inputs=[LINEAR.Input("linear")] + _param_widgets(st),
            outputs=[LINEAR.Output(display_name="linear"), io.String.Output(display_name="summary")])

    @classmethod
    def execute(cls, linear, **kw) -> io.NodeOutput:
        st = _stage("astrometry")
        widgets = _parsed(st, kw)
        key = _key("field", st.version, linear["key"], widgets)
        out = _run(st, linear["dirs"], key, widgets)
        m = _metrics(out)
        summary = (f"centre RA {m.get('ra')} Dec {m.get('dec')}, {m.get('scale_arcsec')} arcsec/px, field "
                   f"{m.get('field_deg')} deg, residual {m.get('residual_px')} px on {m.get('stars_used')} stars"
                   + (f", focal about {m.get('focal_estimate_mm')} mm" if m.get("focal_estimate_mm") else ""))
        return io.NodeOutput({"kind": "linear", "key": key, "dirs": {**linear["dirs"], "astrometry": str(out)}},
                             summary)


class AstroGaiaStars(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        st = _stage("catalogs")
        return io.Schema(
            node_id="AstroGaiaStars", display_name="Gaia stars", category=CAT_FIELD,
            description="The Gaia stars of the field placed on the image (positions, brightness, colours, spectra of "
                        "the brighter ones), down to the faintest the image can show. Needs the identified field. "
                        "Downloaded once (internet), then kept.",
            inputs=[LINEAR.Input("linear", tooltip="An image whose field is identified")] + _param_widgets(st),
            outputs=[STARS.Output(display_name="stars"), io.String.Output(display_name="summary")])

    @classmethod
    def execute(cls, linear, **kw) -> io.NodeOutput:
        _need_field(linear)
        st = _stage("catalogs")
        widgets = _parsed(st, kw)
        key = _key("gaia", st.version, linear["key"], widgets)
        out = _run(st, linear["dirs"], key, widgets)
        m = _metrics(out)
        summary = (f"{m.get('stars_deep')} stars measured, {m.get('stars_faint')} fainter, "
                   f"{m.get('xp_spectra')} spectra")
        return io.NodeOutput({"kind": "stars", "key": key, "dirs": {**linear["dirs"], "catalogs": str(out)}},
                             summary)


class AstroHalphaSurvey(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        st = _stage("surveys")
        return io.Schema(
            node_id="AstroHalphaSurvey", display_name="Hydrogen survey", category=CAT_FIELD,
            description="A professional H-alpha survey of the field (NSNS in the north, Finkbeiner elsewhere) "
                        "reprojected onto the image: the absolute reference of the faint emission, used by the sky "
                        "background. Needs the identified field; downloaded once (internet).",
            inputs=[LINEAR.Input("linear", tooltip="An image whose field is identified")] + _param_widgets(st),
            outputs=[SURVEY.Output(display_name="survey"), io.Image.Output(display_name="view"),
                     io.String.Output(display_name="summary")])

    @classmethod
    def execute(cls, linear, **kw) -> io.NodeOutput:
        _need_field(linear)
        st = _stage("surveys")
        widgets = _parsed(st, kw)
        key = _key("survey", st.version, linear["key"], widgets)
        out = _run(st, linear["dirs"], key, widgets)
        m = _metrics(out)
        summary = f"survey {m.get('survey')}, coverage {m.get('coverage')}, H-alpha p50/p99 {m.get('ha_p50_p99')}"
        return io.NodeOutput({"kind": "survey", "key": key, "dirs": {**linear["dirs"], "surveys": str(out)}},
                             _preview(out, "halpha_reference"), summary)


class AstroColourCalibration(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        st = _stage("color")
        return io.Schema(
            node_id="AstroColourCalibration", display_name="Colour calibration", category=CAT_FIELD,
            description="Measures the colour response of your camera on the Gaia stars of the field (their measured "
                        "spectra) and gives the image its true colours: the colour matrix from camera to sky. The "
                        "measured response is shown.",
            inputs=[LINEAR.Input("linear"), STARS.Input("stars", tooltip="Gaia stars of the same image")]
                   + _param_widgets(st),
            outputs=[LINEAR.Output(display_name="linear"), io.Image.Output(display_name="camera response"),
                     io.String.Output(display_name="summary")])

    @classmethod
    def execute(cls, linear, stars, **kw) -> io.NodeOutput:
        st = _stage("color")
        widgets = _parsed(st, kw)
        dirs = _same_stack(linear, stars)
        key = _key("colour", st.version, linear["key"], stars["key"], widgets)
        out = _run(st, dirs, key, widgets)
        m = _metrics(out)
        summary = (f"colour matrix from {m.get('stars')} stars, residual {m.get('fit_residual')}, response "
                   f"{m.get('response_source')}")
        return io.NodeOutput({"kind": "linear", "key": key, "dirs": {**dirs, "color": str(out)}},
                             _preview(out, "spectral_response"), summary)


BLOCK_NODES = [AstroLights, AstroMasterBias, AstroMasterDark, AstroMasterFlat, AstroCalibrate, AstroAlign,
               AstroFrameSelection, AstroStackBlock, AstroIdentifyField, AstroGaiaStars, AstroHalphaSurvey,
               AstroColourCalibration]
