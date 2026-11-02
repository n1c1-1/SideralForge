"""Command line of v1 projects (project.json): create a project, run / resume the pipeline, inspect and set
parameters, manage variants, export renders (``astro-gpu``; v2 projects have ``cli2.py``)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# less GPU memory fragmentation for the large, variable-size tensors of the processing
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _project(project_dir, variant=None):
    """The project, with ``variant`` active for this command only ('main' = no variant; None = as saved)."""
    from .core.project import Project
    p = Project(project_dir)
    if variant is not None:
        v = "" if variant == "main" else variant
        if v and v not in p.config.variants:
            raise SystemExit(f"no variant named {v!r}; variants: {', '.join(p.config.variants) or '(none)'}")
        p.config.active_variant = v
    return p


def _pipeline(project_dir, variant=None):
    from .core.runner import Pipeline
    from .stages import all_stages
    p = _project(project_dir, variant)
    return p, Pipeline(p, all_stages())


def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def cmd_new(a):
    from .core.project import Inputs, Project, Target
    inputs = Inputs(lights=a.lights, darks=a.darks or [], flats=a.flats or [], bias=a.bias or [],
                    flat_darks=a.flat_darks or [])
    t = Target(name=a.target or "", ra=a.ra, dec=a.dec, focal_mm=a.focal)
    p = Project.create(a.project, a.name or Path(a.project).name, inputs, t, a.scratch, a.reference or "")
    print(f"project created: {p.root}")


def cmd_status(a):
    p, pl = _pipeline(a.project, a.variant)
    print(f"variant: {p.config.active_variant or 'main'}")
    for s in pl.status():
        dur = f"{s.duration / 60:6.1f} min" if s.duration else " " * 10
        print(f"  {s.state:12s} {s.name:12s} {dur}  {s.reason}")


def cmd_run(a):
    _, pl = _pipeline(a.project, a.variant)
    status = pl.run(until=a.until, force=tuple(a.force or ()))
    print(f"run {status}")
    sys.exit(0 if status == "done" else 1)


def cmd_stop(a):
    from .core.project import Project
    from .core.runner import request_stop
    request_stop(Project(a.project))
    print("stop requested: the current stage will stop at its next checkpoint")


def cmd_params(a):
    _, pl = _pipeline(a.project, a.variant)
    for st in pl.stages:
        if a.stage and st.name != a.stage:
            continue
        p, origin, auto = pl.effective_params(st)
        print(f"[{st.name}] {st.title}")
        for d in st.Params.describe():
            n = d["name"]
            extra = f"  <- {auto[n].reason}" if origin[n] == "auto" else ""
            print(f"    {n:22s} = {getattr(p, n)!r:14}  ({origin[n]}, {d['level']}){extra}")


def cmd_set(a):
    p = _project(a.project, a.variant)
    for item in a.assignments:
        key, val = item.split("=", 1)
        stage, name = key.split(".", 1)
        if val == "auto":
            p.clear_override(stage, name)
        else:
            p.set_override(stage, name, _parse_value(val))
    print("ok")


def cmd_render(a):
    from .render.engine import StarRemovalEngines
    from .render.export import ExportSettings, export
    from .render.settings import PRESETS, RenderSettings
    p = _project(a.project, a.variant)
    vals = {}
    if a.preset:
        key = next((k for k in PRESETS if k.lower() == a.preset.lower()), None)
        if key is None:
            raise SystemExit(f"unknown preset '{a.preset}'; available: {', '.join(PRESETS)}")
        vals = dict(PRESETS[key])
        if a.name is None:
            a.name = key.lower().replace(" ", "_").replace(",", "").replace("&", "and")
    a.name = a.name or "render"
    if a.use_saved:
        vals.update(RenderSettings.from_saved(p.config.render.get("settings", {})).model_dump(exclude_defaults=True))
    for item in a.set or []:
        k, v = item.split("=", 1)
        vals[k] = _parse_value(v)
    s = RenderSettings(**vals)
    col = json.loads((p.stage_dir("color") / "manifest.json").read_text())["params"]
    eng = StarRemovalEngines(p.stage_dir("render_prep"), device=p.config.devices.get("main", "cuda:0"),
                       calibrated_white=col.get("white_temp", 5778.0))
    e = ExportSettings(name=a.name, orientation=a.orientation)
    meta = export(p, eng, s, e, lambda f, m: print(f"  {100 * f:5.1f} %  {m}", flush=True))
    print(json.dumps(meta["qa"], indent=1))


def cmd_variant(a):
    """list | create NAME stage.key=value ... | activate NAME | delete NAME | promote NAME [--keep NAME]"""
    from .core.project import Project
    p = Project(a.project)
    act, rest = a.action, a.args
    if act == "list":
        cur = p.config.active_variant
        print(("* " if not cur else "  ") + "main")
        for n, v in p.config.variants.items():
            ch = ", ".join(f"{st}.{k}={val!r}" for st, kv in v.get("params", {}).items() for k, val in kv.items())
            print(("* " if n == cur else "  ") + f"{n}  [{ch}]  {p.variant_size(n) / 2 ** 30:.1f} GB  "
                  f"own stages: {', '.join(sorted(p.variant_stages(n)))}")
    elif act == "create":
        params: dict = {}
        for item in rest[1:]:
            key, val = item.split("=", 1)
            stage, name = key.split(".", 1)
            params.setdefault(stage, {})[name] = None if val == "auto" else _parse_value(val)
        p.create_variant(rest[0], params, note=a.note or "")
        print(f"variant created: {rest[0]}")
    elif act == "activate":
        p.set_active_variant("" if rest[0] == "main" else rest[0])
        print(f"active: {rest[0]}")
    elif act == "delete":
        p.delete_variant(rest[0])
        print(f"deleted: {rest[0]}")
    elif act == "promote":
        p.promote_variant(rest[0], keep_main_as=a.keep or "")
        print(f"'{rest[0]}' is now main" + (f"; the former main is kept as '{a.keep}'" if a.keep else ""))
    else:
        raise SystemExit(f"unknown action {act!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="astro-gpu", description="GPU astrophotography processing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="create a project")
    n.add_argument("project")
    n.add_argument("--name")
    n.add_argument("--lights", nargs="+", required=True)
    n.add_argument("--darks", nargs="+")
    n.add_argument("--flats", nargs="+")
    n.add_argument("--bias", nargs="+")
    n.add_argument("--flat-darks", nargs="+")
    n.add_argument("--target")
    n.add_argument("--ra", type=float)
    n.add_argument("--dec", type=float)
    n.add_argument("--focal", type=float, help="effective focal length in mm (optional)")
    n.add_argument("--scratch")
    n.add_argument("--reference")
    n.set_defaults(fn=cmd_new)
    for name, fn in (("status", cmd_status), ("stop", cmd_stop)):
        s = sub.add_parser(name)
        s.add_argument("project")
        if name == "status":
            s.add_argument("--variant", help="variant to show ('main' for the project's own processing)")
        s.set_defaults(fn=fn)
    r = sub.add_parser("run", help="run or resume the pipeline")
    r.add_argument("project")
    r.add_argument("--variant", help="variant to compute ('main' for the project's own processing)")
    r.add_argument("--until")
    r.add_argument("--force", nargs="+")
    r.set_defaults(fn=cmd_run)
    pa = sub.add_parser("params", help="show the effective parameters")
    pa.add_argument("project")
    pa.add_argument("stage", nargs="?")
    pa.add_argument("--variant")
    pa.set_defaults(fn=cmd_params)
    se = sub.add_parser("set", help="override parameters: stage.name=value (value 'auto' removes the override)")
    se.add_argument("project")
    se.add_argument("assignments", nargs="+")
    se.add_argument("--variant", help="set them in this variant ('main' for the project's own processing)")
    se.set_defaults(fn=cmd_set)
    re_ = sub.add_parser("render", help="render and export")
    re_.add_argument("project")
    re_.add_argument("--preset")
    re_.add_argument("--set", nargs="+")
    re_.add_argument("--use-saved", action="store_true")
    re_.add_argument("--name", help="result name (default: the preset name, or 'render')")
    re_.add_argument("--orientation", default="as captured")
    re_.add_argument("--variant", help="render this variant ('main' for the project's own processing)")
    re_.set_defaults(fn=cmd_render)
    va = sub.add_parser("variant", help="variants: the same data processed another way, side by side")
    va.add_argument("project")
    va.add_argument("action", choices=["list", "create", "activate", "delete", "promote"])
    va.add_argument("args", nargs="*", help="NAME [stage.key=value ...] (value 'auto': back to automatic)")
    va.add_argument("--note")
    va.add_argument("--keep", help="promote: keep the former main as a variant with this name")
    va.set_defaults(fn=cmd_variant)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
