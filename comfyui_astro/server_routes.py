"""HTTP routes of the live render node, added to ComfyUI's server:
- GET  /astro/render_schema        every rendering setting by section (label, range, step, default, level, help);
- GET  /astro/render_sources?root=&variant=   the delivered renderings, the views and the presets of a project;
- POST /astro/preview              {root, variant, settings, width} -> a JPEG preview rendered at a reduced scale
                                   (the processing's layers stay loaded on the GPU between previews);
- POST /astro/save_view            {root, variant, name, settings} -> the settings saved as a view of the project
                                   (the heavy client's views bar shows it too).
"""
from __future__ import annotations

import asyncio

import cv2
import numpy as np
from aiohttp import web
from server import PromptServer

from . import bridge

routes = PromptServer.instance.routes
_LOCK = asyncio.Lock()


def render_schema() -> list[dict]:
    from astro_gpu.render.settings import SECTIONS, RenderSettings
    sections: dict[str, list] = {s: [] for s in SECTIONS}
    for name, f in RenderSettings.model_fields.items():
        meta = dict(f.json_schema_extra or {})
        d = {"name": name, "label": meta.get("label", name), "level": meta.get("level", "basic"),
             "help": meta.get("help", ""), "tip": meta.get("tip", ""), "unit": meta.get("unit", ""),
             "default": f.default, "type": "bool" if f.annotation is bool else ("int" if f.annotation is int else "float")}
        for m in f.metadata:
            for key in ("ge", "le"):
                if hasattr(m, key):
                    d[key] = getattr(m, key)
        if d["type"] != "bool":
            d["step"] = meta.get("step", 1 if d["type"] == "int" else 0.01)
        sections.setdefault(meta.get("section", "Other"), []).append(d)
    return [{"section": s, "fields": v} for s, v in sections.items() if v]


@routes.get("/astro/render_schema")
async def get_render_schema(request):
    return web.json_response(render_schema())


@routes.get("/astro/render_sources")
async def get_render_sources(request):
    from astro_gpu.render.settings import PRESETS
    root, variant = request.query.get("root", ""), request.query.get("variant", "")
    try:
        p = bridge.open_project(root, variant)
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": str(e)}, status=400)
    delivered = sorted(f.parent.name for f in p.results.glob("*/*_settings.json"))
    views = [k for k, v in p.config.render.get("views", {}).items() if (v.get("variant") or "") == variant]
    return web.json_response({"delivered": delivered, "views": views, "presets": list(PRESETS)})


def _settings_of(root: str, variant: str, source: str) -> dict:
    p = bridge.open_project(root, variant)
    if source.startswith("view:"):
        return bridge.render_settings(p, "project", view=source[5:])
    return bridge.render_settings(p, source)


@routes.get("/astro/source_settings")
async def get_source_settings(request):
    """The settings a source stands for (to set the sliders): source = 'results/<name>', 'project', a preset name or
    'view:<name>'."""
    try:
        vals = _settings_of(request.query["root"], request.query.get("variant", ""), request.query["source"])
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response(vals)


def _preview(root: str, variant: str, settings: dict, width: int, clip: bool = False) -> bytes:
    from astro_gpu.render.settings import RenderSettings
    eng, _ = bridge.engine_for(root, variant)
    x0, x1, y0, y1 = eng.crop
    scale = float(np.clip(width / max(x1 - x0, 1), 0.05, 1.0))
    lay = eng.render(RenderSettings(**settings), scale)
    f = 1.0 / lay.scale
    sl = (slice(None), slice(int(y0 / f), int(y1 / f)), slice(int(x0 / f), int(x1 / f)))

    def u8(t):
        return (t[sl].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    a = u8(lay.final)
    if clip:                                    # the heavy client's clipping display (blue, red, amber)
        from astro_gpu.render.clipping import with_clipping
        a = with_clipping(a, u8(lay.starless))
    ok, buf = cv2.imencode(".jpg", a[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


@routes.post("/astro/save_view")
async def post_save_view(request):
    from datetime import datetime

    from astro_gpu.core.project import Project
    from astro_gpu.render.settings import RenderSettings
    body = await request.json()
    try:
        name = str(body["name"]).strip()
        if not name:
            raise ValueError("empty name")
        settings = RenderSettings(**dict(body.get("settings") or {})).model_dump(exclude_defaults=True)
        p = Project(body["root"])                      # fresh from disk: what the heavy client saved is kept
        p.config.render.setdefault("views", {})[name] = {
            "variant": body.get("variant", "") or "", "settings": settings,
            "created": datetime.now().isoformat(timespec="seconds"), "note": "saved from ComfyUI"}
        p.save()
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=400)
    return web.json_response({"saved": name})


@routes.post("/astro/preview")
async def post_preview(request):
    body = await request.json()
    try:
        async with _LOCK:                       # one preview at a time on the GPU
            data = await asyncio.to_thread(_preview, body["root"], body.get("variant", ""),
                                           dict(body.get("settings") or {}), int(body.get("width", 1000)),
                                           bool(body.get("clip")))
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=400)
    return web.Response(body=data, content_type="image/jpeg")
