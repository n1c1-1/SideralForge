"""'Astro: New project': a project made from the folders of the frames, as the heavy client's New project dialog does
(same guesses: the calibration folders next to the lights, the project name, folders and target). An existing project
is never overwritten: it is opened as it is.
"""
from __future__ import annotations

from pathlib import Path

from comfy_api.latest import io

from . import bridge

ASTRO = io.Custom("ASTRO")
KINDS = ("darks", "flats", "bias", "flat_darks")


def _lines(text: str) -> list[str]:
    return [ln.strip().strip('"') for ln in (text or "").splitlines() if ln.strip()]


def _num(text: str) -> float | None:
    try:
        return float(text) if str(text).strip() else None
    except ValueError:
        raise ValueError(f"not a number: {text!r}") from None


def _counts(inputs: dict[str, list[str]]) -> str:
    from astro_gpu.core.project import list_frames
    c = {k: len(list_frames(v)) for k, v in inputs.items()}
    return ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in c.items() if n) or "no frames found"


class AstroNewProject(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        multi = {"multiline": True, "optional": True}
        return io.Schema(
            node_id="AstroNewProject", display_name="Astro: New project", category="astro",
            description="Makes a project from the folders of your frames (as the heavy client's New project): give "
                        "the light frames' folder; the darks, flats, bias and flat darks next to it, the project "
                        "name, its folder, the scratch folder and the target are guessed when left empty. A folder "
                        "that already holds a project is opened as it is (never overwritten). Connect it to the "
                        "processing nodes, or to a render node: everything needed is then computed.",
            inputs=[io.String.Input("lights", multiline=True, default="",
                                    tooltip="Light frames: folders or files, one per line (RAW or FITS)"),
                    io.String.Input("darks", default="", tooltip="Darks: folders or files, one per line (empty = "
                                                                 "guessed next to the lights)", **multi),
                    io.String.Input("flats", default="", tooltip="Flats, as above", **multi),
                    io.String.Input("bias", default="", tooltip="Bias / offsets, as above", **multi),
                    io.String.Input("flat_darks", default="", tooltip="Flat darks, as above", **multi),
                    io.Boolean.Input("guess_calibration", default=True, optional=True,
                                     tooltip="Look for the calibration folders next to the lights by their usual "
                                             "names (dark, flat, offset...) when a list is empty"),
                    io.String.Input("project_folder", default="", optional=True,
                                    tooltip="Where the project goes (a few GB; empty = <lights' folder>/astro_gpu)"),
                    io.String.Input("name", default="", optional=True, tooltip="Name (empty = the folder's name)"),
                    io.String.Input("target", default="", optional=True,
                                    tooltip="Object name, helps the astrometry (empty = guessed from the folder)"),
                    io.String.Input("ra", default="", optional=True, advanced=True,
                                    tooltip="Centre right ascension in degrees (optional)"),
                    io.String.Input("dec", default="", optional=True, advanced=True,
                                    tooltip="Centre declination in degrees (optional)"),
                    io.String.Input("focal_mm", default="", optional=True, advanced=True,
                                    tooltip="Focal length in mm, with reducer (optional: makes the astrometry "
                                            "immediate)"),
                    io.String.Input("scratch_folder", default="", optional=True, advanced=True,
                                    tooltip="Calibrated frames cache (large; an internal SSD is best; empty = "
                                            "<home>/astro_gpu_scratch/<name>)"),
                    io.String.Input("reference_image", default="", optional=True, advanced=True,
                                    tooltip="Optional: your own earlier processing of these data, for comparisons "
                                            "and same-framing exports")],
            outputs=[ASTRO.Output(display_name="astro"), io.String.Output(display_name="status")])

    @classmethod
    def fingerprint_inputs(cls, lights, project_folder="", **kw):
        # computed again when the project changed on disk (another program, the interface), as Astro: Project
        from astro_gpu.core.newproject import guess_names
        lt = _lines(lights)
        root = project_folder.strip().strip('"') or (guess_names(lt[0])["folder"] if lt else "")
        return f"{root}|{bridge.project_signature(root)}" if root else ""

    @classmethod
    def execute(cls, lights, darks="", flats="", bias="", flat_darks="", guess_calibration=True, project_folder="",
                name="", target="", ra="", dec="", focal_mm="", scratch_folder="",
                reference_image="") -> io.NodeOutput:
        from astro_gpu.core.newproject import guess_calibration as guess_cal
        from astro_gpu.core.newproject import guess_names, space_text
        from astro_gpu.core.project import Inputs, Project, Target, list_frames
        lt = _lines(lights)
        if not lt:
            raise ValueError("give the light frames' folder (or files), one per line")
        missing = [f for f in lt if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(f"not found: {', '.join(missing)}")
        g = guess_names(lt[0])
        root = Path(project_folder.strip().strip('"') or g["folder"])
        if (root / "project.json").exists():
            p = Project(root)
            note = f"existing project opened (not changed): {root}"
            if [str(Path(f)) for f in p.config.inputs.lights] != [str(Path(f)) for f in lt]:
                note += " - its light frames differ from this node's: they are kept as they are"
            return io.NodeOutput(bridge.astro_value(p.root), note + "\n" + bridge.status_text(p.root))
        given = {"darks": _lines(darks), "flats": _lines(flats), "bias": _lines(bias),
                 "flat_darks": _lines(flat_darks)}
        found = guess_cal(lt[0]) if guess_calibration else {k: [] for k in KINDS}
        used = {k: given[k] or found[k] for k in KINDS}
        guessed = [k.replace("_", " ") for k in KINDS if not given[k] and found[k]]
        nm = name.strip() or g["name"]
        scratch = scratch_folder.strip().strip('"') or str(Path(g["scratch"]).parent / nm)
        tgt = Target(target.strip() or g["target"], _num(ra), _num(dec), _num(focal_mm))
        p = Project.create(root, nm, Inputs(lights=lt, **used), tgt, scratch, reference_image.strip().strip('"'))
        lines = [f"project created: {p.root}", f"frames: {_counts({'lights': lt, **used})}",
                 f"target: {tgt.name or '(none)'}; scratch folder: {scratch}"]
        if guessed:
            lines.append(f"guessed next to the lights: {', '.join(guessed)}")
        try:
            lines.append(space_text(list_frames(lt), scratch))
        except Exception as e:  # noqa: BLE001
            lines.append(f"(could not read the first frame: {e})")
        return io.NodeOutput(bridge.astro_value(p.root), "\n".join(x for x in lines if x))
