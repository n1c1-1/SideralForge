"""Bridge between ComfyUI and astro_gpu: projects, variants, stage runs with progress and interruption, and the
rendering engines kept loaded between renders.

Nothing here imports ComfyUI at module level: the ComfyUI hooks (progress bar, interruption) are passed in, so the
bridge can be used and tested without a running ComfyUI.

The value carried between the nodes (ComfyUI type "ASTRO") is a small dict:
    {"root": project folder, "variant": "" for main or a variant name, "stage": last stage computed or None,
     "run_id": that stage's run id (a new computation = a new value)}
"""
from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from astro_gpu.core.journal import Journal
from astro_gpu.core.project import Project
from astro_gpu.core.runner import Pipeline, read_manifest
from astro_gpu.stages import all_stages

LOG = logging.getLogger("astro")          # stage messages, in ComfyUI's console
STAGES = all_stages()
BY_NAME = {s.name: s for s in STAGES}
ORDER = [s.name for s in STAGES]

# a node's parameters come from the project (what the earlier interface or graph saved) or from the node's widgets
SETTINGS_FROM_PROJECT = "project"
SETTINGS_FROM_NODE = "this node"


def project_signature(project_folder) -> str:
    """Changes when anything of the project changed on disk (its settings, a stage computed by another program)."""
    root = Path(str(project_folder).strip().strip('"'))
    sig = []
    for f in [root / "project.json"] + sorted(root.glob("work/*/manifest.json")) + \
            sorted(root.glob("variants/*/*/manifest.json")):
        try:
            sig.append(f"{f}:{f.stat().st_mtime_ns}")
        except OSError:
            pass
    return "|".join(sig)


def astro_value(root, variant: str = "", stage: str | None = None, run_id: str | None = None) -> dict:
    return {"root": str(Path(root)), "variant": variant or "", "stage": stage, "run_id": run_id}


def open_project(root, variant: str = "") -> Project:
    """The project with ``variant`` active in memory only (the processing shown in the interface is not changed)."""
    p = Project(root)
    if variant and variant not in p.config.variants:
        raise KeyError(f"no variant named {variant!r} in {root}")
    p.config.active_variant = variant or ""
    return p


def ensure_variant(root, variant: str, note: str = "") -> None:
    """Create an empty variant (it gets stages of its own when nodes set parameters in it)."""
    if not variant:
        return
    p = Project(root)
    if variant not in p.config.variants:
        p.create_variant(variant, {}, note=note or "made by a ComfyUI graph")


def apply_node_params(root, variant: str, stage: str, values: dict[str, Any]) -> bool:
    """Store a node's parameter values in the project (main, or the variant): a value equal to the parameter's
    default means 'automatic' (no override). Returns True when something changed. The project is read and saved
    afresh, so the processing shown in the interface (active variant) is kept."""
    defaults = {d["name"]: d["default"] for d in BY_NAME[stage].Params.describe()}
    p = Project(root)
    before = json.dumps([p.config.params, p.config.variants], sort_keys=True, default=str)
    if variant:
        if variant not in p.config.variants:
            p.create_variant(variant, {}, note="made by a ComfyUI graph")
            p = Project(root)
        vp = p.config.variants[variant].setdefault("params", {})
        cur = vp.setdefault(stage, {})
        main = p.config.params.get(stage, {})
        for k, v in values.items():
            if k not in defaults:
                continue
            if v == defaults[k]:
                if k in main:
                    cur[k] = None                    # main sets it: back to the automatic value in the variant
                else:
                    cur.pop(k, None)
            elif main.get(k) == v:
                cur.pop(k, None)                     # same as main: no change of its own
            else:
                cur[k] = v
        if not cur:
            vp.pop(stage, None)
    else:
        mp = p.config.params.setdefault(stage, {})
        for k, v in values.items():
            if k not in defaults:
                continue
            if v == defaults[k]:
                mp.pop(k, None)
            else:
                mp[k] = v
        if not mp:
            p.config.params.pop(stage, None)
    after = json.dumps([p.config.params, p.config.variants], sort_keys=True, default=str)
    if after != before:
        p.save()
        return True
    return False


class HookJournal(Journal):
    """The project's journal, with the progress of the running stage forwarded to ``on_progress(stage, frac, msg)``,
    its messages and the start and end of every stage to ComfyUI's console (logger 'astro'), and a stop request when
    ``interrupted()`` says so (the stage stops at its next checkpoint)."""

    def __init__(self, log_dir: Path, on_progress: Callable | None = None, interrupted: Callable | None = None):
        super().__init__(log_dir, echo=False)
        self._on_progress, self._interrupted = on_progress, interrupted
        self._log_dir = Path(log_dir)

    def emit(self, type_: str, **kw: Any) -> None:
        super().emit(type_, **kw)
        try:
            st = kw.get("stage", "")
            if type_ == "log":
                level = {"warning": logging.WARNING, "error": logging.ERROR}.get(kw.get("level", ""), logging.INFO)
                LOG.log(level, f"{st}: {kw.get('msg', '')}")
            elif type_ == "stage_start":
                LOG.info(f"{st}: started")
            elif type_ == "stage_end":
                LOG.info(f"{st}: {kw.get('status', '')} ({float(kw.get('duration') or 0) / 60:.1f} min)")
        except Exception:  # noqa: BLE001 - a display problem never stops a computation
            pass
        if type_ == "progress" and self._on_progress is not None:
            try:
                self._on_progress(kw.get("stage", ""), float(kw.get("frac", 0.0)), kw.get("msg", ""))
            except Exception:  # noqa: BLE001 - a display problem never stops a computation
                pass
        if self._interrupted is not None and self._interrupted():
            (self._log_dir / "STOP").write_text("stop requested by ComfyUI")


@contextmanager
def _spawn_without_main():
    """The calibration and the registration compute in worker processes ('spawn'). On Windows a spawned worker runs
    the parent's main script again before its work: ComfyUI's main.py, which fails there (the workers die and are
    started again without end, the stage never finishes). While the pipeline runs, the main module is replaced by an
    empty one, so the workers only import what their work needs."""
    import sys
    import types
    main = sys.modules.get("__main__")
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        yield
    finally:
        if main is not None:
            sys.modules["__main__"] = main


def run_up_to(root, variant: str, stage: str, on_progress: Callable | None = None,
              interrupted: Callable | None = None) -> dict:
    """Compute every stage needed up to ``stage`` (in the variant, or main) that is not up to date, then return the
    ASTRO value of that stage. Stages already up to date are not touched."""
    p = open_project(root, variant)
    pipe = Pipeline(p, STAGES, journal=HookJournal(p.logs, on_progress, interrupted))
    todo = pipe.plan(until=stage)
    if todo:
        import torch
        # ComfyUI runs the nodes without gradients (inference mode): the stages that train a network (denoising,
        # learned star removal, sharpening) need them
        with torch.inference_mode(False), torch.enable_grad(), _spawn_without_main():
            status = pipe.run(until=stage)
        if status != "done":
            m = read_manifest(p, next((s for s in todo if read_manifest(p, s).get("status") != "done"), stage))
            raise RuntimeError(f"{stage}: the computation {status}" + (f": {m.get('error')}" if m.get("error") else ""))
    return astro_value(root, variant, stage, read_manifest(p, stage).get("run_id"))


def status_text(root, variant: str = "") -> str:
    p = open_project(root, variant)
    pipe = Pipeline(p, STAGES)
    return "\n".join(f"{s.state:12s} {s.name}" + (f"  ({s.reason})" if s.reason else "") for s in pipe.status())


# ---------------------------------------------------------------------- rendering engines
_ENGINES: OrderedDict = OrderedDict()
_ENGINES_LOCK = threading.Lock()
MAX_ENGINES = 2


def engine_for(root, variant: str, device: str | None = None):
    """The rendering engine of a processing (both star removals), kept loaded: several render nodes on the same
    processing share it. Reloaded when the rendering preparation was computed again."""
    from astro_gpu.render.engine import StarRemovalEngines
    p = open_project(root, variant)
    key = (str(p.root), variant or "", read_manifest(p, "render_prep").get("run_id"),
           read_manifest(p, "render_prep_learned").get("run_id"), device)
    with _ENGINES_LOCK:
        if key in _ENGINES:
            _ENGINES.move_to_end(key)
            return _ENGINES[key], p
        for k in [k for k in _ENGINES if k[:2] == key[:2]]:       # an older preparation of the same processing
            del _ENGINES[k]
        while len(_ENGINES) >= MAX_ENGINES:
            _ENGINES.popitem(last=False)
        col = p.stage_dir("color") / "manifest.json"
        white = json.loads(col.read_text())["params"].get("white_temp", 5778.0) if col.exists() else 5778.0
        eng = StarRemovalEngines(p.stage_dir("render_prep"), device=device or p.config.devices.get("main", "cuda:0"),
                                 calibrated_white=white)
        _ENGINES[key] = eng
        return eng, p


def release_engines() -> None:
    with _ENGINES_LOCK:
        _ENGINES.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def render_settings(p: Project, source: str, view: str = "", overrides: str = "") -> dict:
    """Rendering settings (non-default values) from a source: 'project' (the last settings of the interface), a
    delivered rendering ('results/<name>'), a preset name, then a saved view, then JSON overrides on top."""
    from astro_gpu.render.settings import PRESETS, RenderSettings
    vals: dict = {}
    if source == "project":
        vals = RenderSettings.from_saved(p.config.render.get("settings", {})).model_dump(exclude_defaults=True)
    elif source.startswith("results/"):
        name = source.split("/", 1)[1]
        f = p.results / name / f"{name}_settings.json"
        if not f.exists():
            raise FileNotFoundError(f"no delivered rendering '{name}' in {p.results}")
        meta = json.loads(f.read_text(encoding="utf-8"))
        vals = RenderSettings.from_saved(meta.get("settings", {})).model_dump(exclude_defaults=True)
        if meta.get("preparation") == "render_prep":
            vals["learned_stars"] = False
    elif source in PRESETS:
        vals = dict(PRESETS[source])
    if view:
        v = p.config.render.get("views", {}).get(view)
        if v is None:
            raise KeyError(f"no view named {view!r}; views: {', '.join(p.config.render.get('views', {})) or '(none)'}")
        vals = dict(v["settings"])
    if overrides.strip():
        vals.update(json.loads(overrides))
    RenderSettings(**vals)                                   # validated here, with a readable error
    return vals
