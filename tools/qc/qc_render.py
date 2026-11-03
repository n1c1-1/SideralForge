"""Render a v2 workflow's output into the QC folder, when no exported image of it is at hand.

The image is made the way the v1 deliverables' sensor-scale PNG was: the layer at full resolution, cropped to the
useful frame of its preparation, reduced by the drizzle scale with INTER_AREA, 16-bit. It goes to ``<out>/renders/``
(never into the project's export/), next to a ``_history.json`` that says how it was made. Rendering reads the
project's store; it writes nothing there.

``base``: None (the suggested settings), a preset or ``render_bases`` name, or a dict of render settings. To compare a
new workflow with an old image like for like, pass the old image's saved settings (``old.settings["settings"]``).

A module of tools/qc/qc2.py (``python tools/qc/qc2.py --help``): not run by itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from qc_addr import QCError, Source
from qc_out import RENDERS, slug, sub

from astro_gpu.core import project2 as p2


def render_v2(src: Source, out: str | Path, base: Any = None, sliders: dict[str, float] | None = None,
              device: str = "cpu", service: Any = None, name: str | None = None) -> Path:
    """Render ``src`` (a v2 workflow) into ``<out>/renders/`` and set ``src.image`` to the file. Returns the file."""
    if src.version != 2:
        raise QCError(f"{src.address}: only a v2 workflow can be rendered here; a v1 deliverable has its image")
    from astro_gpu.render.layers import LayerService
    svc = service or LayerService(device)
    if isinstance(base, str):
        proj = p2.load(src.root)
        base = proj.render_bases.get(base, base)
    lay = svc.render(src.out_dir, base, sliders, 1.0)
    x0, x1, y0, y1 = lay.crop
    rgb = np.asarray(lay.rgb[:, y0:y1, x0:x1], np.float32)
    w, h = max(1, round((x1 - x0) / src.drizzle)), max(1, round((y1 - y0) / src.drizzle))
    small = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA) for c in rgb])
    u16 = (np.clip(small, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 65535 + 0.5).astype(np.uint16)
    stem = name or f"{slug(src.project_name)}__{slug(src.name)}"
    d = sub(Path(out), RENDERS)
    png = d / f"{stem}.png"
    ok, buf = cv2.imencode(".png", u16)
    if not ok:
        raise QCError(f"cannot encode {png}")
    png.write_bytes(buf.tobytes())
    (d / f"{stem}_history.json").write_text(json.dumps(
        {"source": src.address, "rot90": 0, "flip": False, "base": base if isinstance(base, (str, dict)) else None,
         "sliders": sliders or {}, "crop": [x0, x1, y0, y1], "drizzle_scale": src.drizzle, "render": lay.info},
        indent=1, default=str), encoding="utf-8")
    src.image, src.rot90, src.flip = png, 0, False
    return png
