"""The render worker of Post: the engine's layers, the composition and the export, off the GUI thread (DESIGN §5.3).

* ``LayerSource`` is the ``postengine.Provider`` on the engine: a workflow's output folder rendered by a ``LayerService``
  (render/layers.py) with a base and slider positions, on the canvas every layer covers. The release render, the 100 %
  region and the export are the engine's own renderings; nothing here changes a pixel of them.
* ``Renderer`` runs a request (a draft at 1/4, the screen at 1/2, a region at full resolution, the before image, the
  thumbnails) and an export. ``RenderWorker`` is the QThread that owns one: requests are coalesced by lane, the newest of
  a lane wins, and a request being computed is abandoned at its next checkpoint when a newer one arrives.
* The device: a GPU that no run is using, else the CPU; a GPU out-of-memory error renders again on the CPU, with a note.

    w = RenderWorker(project)
    w.rendered.connect(on_result)
    w.request(Request(kind="half", layers=snapshot(layers), level=1))
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..postengine import (
    ALIGN,
    Canvas,
    Composer,
    Provider,
    Superseded,
    only_layer,
    thumb_of,
)

#: the output stages (the last one of a chain is the workflow's layer)
OUTPUT_STAGES = ("render_prep", "render_prep_learned", "render_simple")
#: the budgets of DESIGN §5.3, in seconds
BUDGET = {"draft": 0.15, "half": 1.0, "region": 2.0}
DRAFT_LEVEL, HALF_LEVEL = 2, 1
NOTE_CPU_TEXT = "GPU memory full: rendered on the CPU"


# ---------------------------------------------------------------------------------------------- the device
def gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001 - no torch, no driver
        return 0


def choose_device(pref: str = "auto", busy: str | None = None, need_gb: float = 3.0) -> str:
    """Where Post renders: the GPU the settings name, else the free GPU with the most memory, else the CPU. ``busy``:
    the GPU a running job holds (never used for Post while it runs). A GPU with less than ``need_gb`` free is skipped."""
    if (pref or "auto").lower() == "cpu":
        return "cpu"
    n = gpu_count()
    if n == 0:
        return "cpu"
    import torch
    free = []
    order = [f"cuda:{i}" for i in range(n)]
    if pref and pref != "auto" and pref in order:
        order = [pref] + [d for d in order if d != pref]
    for d in order:
        if d == busy:
            continue
        try:
            gb = torch.cuda.mem_get_info(int(d.split(":")[1]))[0] / 2 ** 30
        except Exception:  # noqa: BLE001
            continue
        if gb >= need_gb:
            free.append((d, gb))
    if not free:
        return "cpu"
    if pref and pref != "auto" and free[0][0] == pref:
        return pref
    return max(free, key=lambda t: t[1])[0]


# ---------------------------------------------------------------------------------------------- the engine's layers
@dataclass
class Folder:
    path: Path
    previous: bool = False                   # the result an Outdated stage had, not the current one
    mtime: float = 0.0


@dataclass
class Grid:
    W: int
    H: int
    crop: tuple[int, int, int, int]
    kind: str


def read_grid(folder: Path) -> Grid:
    """The grid and the useful frame of an output folder, from its small files (the arrays are not loaded)."""
    from ...render.layers import layer_kind
    kind = layer_kind(folder)
    pj = json.loads((folder / ("prep.json" if kind == "engine" else "prep_simple.json")).read_text(encoding="utf-8"))
    with open(folder / "valid.npy", "rb") as fh:
        v = np.lib.format.read_magic(fh)
        shape = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(fh)[0]
    return Grid(int(shape[1]), int(shape[0]), tuple(int(c) for c in pj["crop"]), kind)


def api_resolver(project) -> Callable[[int], Folder | None]:
    """pid -> the folder of the workflow's output: the current result, else the one an Outdated output had."""
    from ... import api

    def resolve(pid: int) -> Folder | None:
        try:
            outs = [s for s in api.chain(project, pid) if s in OUTPUT_STAGES]
            if not outs:
                return None
            so = api.stage_outputs(project, pid, outs[-1]) or api.stage_outputs(project, pid, outs[-1], "previous")
        except (OSError, ValueError, KeyError):
            return None
        if so is None:
            return None
        d = Path(so.dir)
        try:
            mt = (d / "manifest.json").stat().st_mtime
        except OSError:
            mt = 0.0
        return Folder(d, so.which == "previous", mt)
    return resolve


def bases_of(project) -> Callable[[Any], Any]:
    """A named rendering of the project -> its settings; anything else (a preset, settings, None) as it is."""
    def base(b):
        rb = getattr(project, "render_bases", None) or {}
        return rb[b] if isinstance(b, str) and b in rb else b
    return base


class LayerSource(Provider):
    """The layers of the workflows as the composition reads them. Used in one thread (the worker's)."""

    def __init__(self, resolver: Callable[[int], Folder | None], service, bases: Callable[[Any], Any] | None = None):
        self.resolver = resolver
        self.service = service
        self.bases = bases or (lambda b: b)
        self._folders: dict[int, Folder | None] = {}
        self._grids: dict[str, Grid] = {}
        self._epoch: dict[int, int] = {}
        self._notes: set[str] = set()
        self._masks: dict[tuple, dict] = {}
        self._cv: Canvas | None = None
        self._ref: Grid | None = None
        self._pids: list[int] = []
        self.engine_ms = 0.0                                  # time the engine spent since the last reset

    # ------------------------------------------------------------ the results on disk
    def folder(self, pid: int) -> Folder | None:
        if pid not in self._folders:
            self._folders[pid] = self.resolver(pid)
        return self._folders[pid]

    def refresh(self) -> set[int]:
        """The results may have changed: look again. The pids whose result changed (a new folder, or the same folder
        written again) lose their engine and their cached images."""
        changed = set()
        for pid in list(self._folders):
            old, new = self._folders[pid], self.resolver(pid)
            self._folders[pid] = new
            same = old is not None and new is not None and old.path == new.path and old.mtime == new.mtime \
                and old.previous == new.previous
            if same or (old is None and new is None):
                continue
            changed.add(pid)
            self._epoch[pid] = self._epoch.get(pid, 0) + 1
            if old is not None:
                self.service.evict(old.path)
        if changed:
            self._masks.clear()
            self._cv = None
        return changed

    def version(self, pid: int) -> int:
        return self._epoch.get(pid, 0)

    def available(self, pid: int) -> bool:
        return self.folder(pid) is not None

    def previous(self, pid: int) -> bool:
        f = self.folder(pid)
        return bool(f and f.previous)

    def grid(self, pid: int) -> Grid:
        f = self.folder(pid)
        if f is None:
            raise FileNotFoundError(f"workflow {pid} has no result to show")
        k = str(f.path)
        if k not in self._grids:
            self._grids[k] = read_grid(f.path)
        return self._grids[k]

    def kind(self, pid: int) -> str:
        return self.grid(pid).kind

    # ------------------------------------------------------------ geometry
    def canvas(self, pids: list[int]) -> Canvas:
        """The area every layer covers, in the pixels of the first one's grid, on multiples of ``ALIGN``."""
        grids = [self.grid(p) for p in pids]
        ref = grids[0]
        x0, x1, y0, y1 = (float(c) for c in ref.crop)
        for g in grids[1:]:
            rx, ry = ref.W / g.W, ref.H / g.H                    # a pixel of g is this many pixels of the reference
            x0, x1 = max(x0, g.crop[0] * rx), min(x1, g.crop[1] * rx)
            y0, y1 = max(y0, g.crop[2] * ry), min(y1, g.crop[3] * ry)
        ox, oy = -(-int(np.ceil(x0)) // ALIGN) * ALIGN, -(-int(np.ceil(y0)) // ALIGN) * ALIGN
        ex, ey = int(np.floor(x1)) // ALIGN * ALIGN, int(np.floor(y1)) // ALIGN * ALIGN
        if ex - ox < ALIGN or ey - oy < ALIGN:
            raise ValueError("the layers of these workflows share no area")
        self._cv, self._ref, self._pids = Canvas(ex - ox, ey - oy, (ox, oy)), ref, list(pids)
        return self._cv

    def _window(self, pid: int, level: int, box) -> tuple[int, tuple, tuple[int, int]]:
        """How a layer of grid g covers a box of the canvas at a level: the reduction ``fi`` to render it with, the
        region of g's full-resolution grid the box covers, and the size the box has at the level."""
        cv, ref, g = self._cv, self._ref, self.grid(pid)
        rx, ry = g.W / ref.W, g.H / ref.H
        f = 2 ** level
        x0, y0, x1, y1 = box
        fi = max(1, int(round(f * rx))) if level > 0 else 1
        ox, oy = cv.origin
        region = ((ox + x0 * f) * rx, (ox + x1 * f) * rx, (oy + y0 * f) * ry, (oy + y1 * f) * ry)
        return fi, region, (x1 - x0, y1 - y0)

    def _cut(self, arr: np.ndarray, fi: int, region, size: tuple[int, int], whole_grid: bool) -> np.ndarray:
        """A rendering (C, h, w) or (h, w) of the whole grid at 1/fi (or of the region at full resolution), cut to the
        region and resized to the size the box has at the level."""
        x0, x1, y0, y1 = region
        if whole_grid:
            sx0, sx1, sy0, sy1 = (int(round(v / fi)) for v in (x0, x1, y0, y1))
            arr = arr[..., sy0:sy1, sx0:sx1]
        w, h = size
        if arr.shape[-2:] != (h, w):
            a = np.moveaxis(arr, 0, -1) if arr.ndim == 3 else arr
            a = cv2.resize(np.ascontiguousarray(a), (w, h),
                           interpolation=cv2.INTER_AREA if a.shape[0] > h else cv2.INTER_LINEAR)
            arr = np.moveaxis(a, -1, 0) if arr.ndim == 3 else a
        return arr

    # ------------------------------------------------------------ images
    def layer(self, lay: dict, level: int, box) -> np.ndarray:
        pid = lay["pid"]
        f = self.folder(pid)
        cv = self._cv
        whole = cv.size(level)
        fi, region, size = self._window(pid, level, box)
        base = self.bases(lay.get("base"))
        part = lay.get("part")
        full = tuple(box) == (0, 0, *whole)
        try:
            if level == 0 and not full:
                li = self.service.render(f.path, base, lay.get("sliders"), 1.0,
                                         region=tuple(int(round(v)) for v in region), part=part)
                arr = self._cut(li.rgb, 1, region, size, False)
            else:                                           # the whole grid at 1/fi: v1's rendering, cut to the canvas
                li = self.service.render(f.path, base, lay.get("sliders"), 1.0 / fi, part=part)
                arr = self._cut(li.rgb, fi, region, size, True)
        except ValueError:                                  # a part of a layer that has no star model any more
            if part is None:
                raise
            if part == "stars":
                return np.zeros((size[1], size[0], 3), np.float32)
            li = self.service.render(f.path, base, lay.get("sliders"), 1.0 / fi)
            arr = self._cut(li.rgb, fi, region, size, level > 0 or full)
        if li.info.get("note"):
            self._notes.add(li.info["note"])
        self.engine_ms += float(li.info.get("render_ms") or 0.0)
        return np.ascontiguousarray(np.moveaxis(arr, 0, -1), np.float32)

    def has(self, name: str) -> bool:
        if name == "stars":
            return any(self.kind(p) == "engine" for p in self._pids if self.available(p))
        return any(self.available(p) for p in self._pids)

    def _mask_pid(self, name: str) -> int | None:
        for p in self._pids:                                  # bottom first
            if self.available(p) and (name != "stars" or self.kind(p) == "engine"):
                return p
        return None

    def data_mask(self, name: str, level: int, box) -> np.ndarray | None:
        pid = self._mask_pid(name)
        if pid is None or self._cv is None:
            return None
        f = self.folder(pid)
        fi, region, size = self._window(pid, level, box)
        whole = self._cv.size(level)
        full = tuple(box) == (0, 0, *whole)
        if level == 0 and not full:
            m = self.service.masks(f.path, 1.0, region=tuple(int(round(v)) for v in region))[name]
            return None if m is None else self._cut(m, 1, region, size, False)
        key = (pid, fi, str(f.path), self.version(pid))
        if key not in self._masks:
            if len(self._masks) >= 4:
                self._masks.pop(next(iter(self._masks)))
            self._masks[key] = self.service.masks(f.path, 1.0 / fi)
        m = self._masks[key][name]
        return None if m is None else self._cut(m, fi, region, size, True)

    def notes(self) -> set[str]:
        n, self._notes = self._notes, set()
        return n


# ---------------------------------------------------------------------------------------------- requests
@dataclass
class Request:
    """What the view asks for. ``kind``: draft (quarter, while something moves), half (the screen, on release), region
    (full resolution, the visible part), before (the bottom layer alone, at its suggestion), thumbs (the layers'
    thumbnails)."""
    kind: str = "half"
    layers: list = field(default_factory=list)          # a snapshot (postengine.snapshot)
    level: int = HALF_LEVEL
    box: tuple | None = None                            # at the level, in canvas pixels (None: all of it)
    want: tuple = ()                                    # ids of the layers whose mask is returned
    thumbs: bool = False                                # also the mask thumbnails
    overlay: bool = False                               # a red overlay of the first wanted mask
    need_thumbs: tuple = ()                             # (thumbs kind) the ids of the workflow layers to thumbnail
    seq: int = 0
    tag: Any = None


@dataclass
class Result:
    seq: int
    kind: str
    level: int
    box: tuple | None
    rgb: np.ndarray | None                              # uint8 (h, w, 3)
    canvas: Canvas | None
    masks: dict = field(default_factory=dict)           # id -> (h, w) float32
    overlay: np.ndarray | None = None                   # uint8 (h, w, 4): red where the wanted mask shows the layer
    mask_thumbs: dict = field(default_factory=dict)     # id -> (36, 54) uint8
    layer_thumbs: dict = field(default_factory=dict)    # id -> (36, 54, 3) uint8
    previews: set = field(default_factory=set)          # pids whose layer comes from an outdated result
    ms: float = 0.0                                     # the whole request
    layer_ms: float = 0.0                               # spent in the engine
    drawn: int = 0                                      # workflow layers the engine rendered (the others were cached)
    notes: set = field(default_factory=set)
    device: str = "cpu"
    tag: Any = None


_RETIRED: list = []                    # workers asked to stop that were still busy: kept until their thread ends
LANES = {"draft": "main", "half": "main", "region": "region", "before": "before", "thumbs": "thumbs"}
PRIORITY = ("main", "region", "before", "thumbs")


def to_u8(a: np.ndarray) -> np.ndarray:
    """A float image in [0, 1] as 8-bit (rounded, clipped)."""
    a = np.ascontiguousarray(a, np.float32)
    return cv2.convertScaleAbs(np.clip(a, 0, 1) if a.min() < 0 else a, alpha=255.0)


def red_overlay(mask: np.ndarray) -> np.ndarray:
    """Where the layer shows, in red at about half strength (the mask being edited)."""
    m = np.clip(mask, 0, 1)
    a = np.empty(m.shape + (4,), np.uint8)
    a[..., 0], a[..., 1], a[..., 2] = 255, 70, 60
    a[..., 3] = (m * 120).astype(np.uint8)
    return a


class Renderer:
    """Runs the requests with a composer. No Qt: the worker's thread uses it, and so do the tests and the export."""

    def __init__(self, source: LayerSource, composer: Composer | None = None, device: str = "cpu"):
        self.source = source
        self.composer = composer or Composer(source)
        self.device = device
        self.wcs_of: Callable[[int], Any] | None = None      # pid -> the wcs.fits of the workflow (the FITS export)
        self.writer: Callable | None = None                  # (rgb, folder, ExportOptions, meta) -> files

    def run(self, req: Request, stop: Callable[[], bool] | None = None) -> Result:
        t0 = time.perf_counter()
        cp = self.composer
        self.source.engine_ms = 0.0
        if req.kind == "thumbs":
            return self._thumbs(req, stop, t0)
        layers = req.layers
        if req.kind == "before":
            layers = self._before_layers(layers)
        comp = cp.compose(layers, req.level, req.box, want=req.want, thumbs=req.thumbs, stop=stop)
        if stop is not None and stop():                       # a newer request came meanwhile: this answer is stale
            raise Superseded()
        if comp is None:
            return Result(req.seq, req.kind, req.level, req.box, None, None, device=self.device, tag=req.tag)
        res = Result(req.seq, req.kind, req.level, comp.box, to_u8(comp.rgb), cp.canvas_of(layers), comp.masks,
                     None, comp.mask_thumbs, {}, set(), 0.0, self.source.engine_ms, comp.drawn, comp.notes,
                     self.device, req.tag)
        if req.overlay and req.want and req.want[0] in comp.masks:
            res.overlay = red_overlay(comp.masks[req.want[0]])
        res.previews = {lay["pid"] for lay in layers if lay["kind"] == "workflow" and self.source.available(lay["pid"])
                        and self.source.previous(lay["pid"])}
        if req.kind == "half" and req.box is None:                    # thumbnails of what was just drawn, for free
            res.layer_thumbs = self._visible_thumbs(layers, cp.canvas_of(layers))
        res.ms = 1000 * (time.perf_counter() - t0)
        return res

    def _before_layers(self, layers: list[dict]) -> list[dict]:
        """The bottom workflow layer alone: its base, every slider at the suggestion, no filter, no mask. The others
        stay in the list, hidden, so that the canvas is the same as the composition's."""
        wl = [lay for lay in layers if lay["kind"] == "workflow" and self.source.available(lay["pid"])]
        if not wl:
            return []
        bottom = wl[-1]
        out = [dict(lay, visible=False) for lay in layers if lay["kind"] == "workflow" and lay is not bottom]
        return out + [dict(bottom, visible=True, opacity=100.0, mode="Normal", sliders={}, mask=None, part=None)]

    def _visible_thumbs(self, layers, canvas) -> dict:
        """The thumbnails of the workflow layers whose quarter-resolution image is in the cache (the draft's)."""
        out = {}
        if canvas is None:
            return out
        for lay in layers:
            if lay["kind"] != "workflow" or not self.source.available(lay["pid"]):
                continue
            a = self.composer.cached(lay, 2, canvas)
            if a is not None:
                out[lay["id"]] = thumb_of(a)
        return out

    def _thumbs(self, req: Request, stop, t0) -> Result:
        res = Result(req.seq, "thumbs", 2, None, None, None, device=self.device, tag=req.tag)
        cv = self.composer.canvas_of(req.layers)
        if cv is None:
            return res
        res.canvas = cv
        W, H = cv.size(2)
        for lay in req.layers:
            if stop and stop():
                raise Superseded()
            if lay["id"] in req.need_thumbs and lay["kind"] == "workflow" and self.source.available(lay["pid"]):
                res.layer_thumbs[lay["id"]] = thumb_of(self.composer.layer_rgb(lay, 2, (0, 0, W, H), cv))
        res.notes = self.source.notes()
        res.ms = 1000 * (time.perf_counter() - t0)
        return res

    # ------------------------------------------------------------ the export
    def export(self, job: ExportJob, progress: Callable[[float, str], None] = lambda f, m: None,
               stop: Callable[[], bool] | None = None) -> list[Path]:
        return run_export(self, job, progress, stop)


# ---------------------------------------------------------------------------------------------- the export
@dataclass
class ExportJob:
    layers: list
    folder: str                                          # the one folder the files go into
    name: str = "image"
    format: str | list = "tiff16"                        # tiff16 | tiff32f | png16 | jpeg | fits32f (or several, one render)
    size: int = 0                                        # long side in pixels, 0 = full resolution
    wcs: bool = True
    history: bool = True
    each_layer: bool = False
    masks: bool = False
    project_name: str = ""
    jpeg_quality: int = 93


def _slug(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "layer"


def history_of(job: ExportJob, renderer: Renderer, canvas: Canvas | None) -> dict:
    """The sidecar of an export: the composition and what rendered it."""
    from datetime import datetime
    src = renderer.source
    return {"project": job.project_name, "created": datetime.now().isoformat(timespec="seconds"),
            "layers": job.layers, "format": job.format, "size": job.size, "device": renderer.device,
            "canvas": None if canvas is None else {"width": canvas.W, "height": canvas.H, "origin": list(canvas.origin)},
            "workflows": {str(lay["pid"]): {"folder": str(src.folder(lay["pid"]).path),
                                            "outdated": src.previous(lay["pid"])}
                          for lay in job.layers if lay["kind"] == "workflow" and src.available(lay["pid"])}}


def run_export(renderer: Renderer, job: ExportJob, progress, stop=None,
               writer: Callable | None = None) -> list[Path]:
    """The final image at full resolution (the engine's whole-grid rendering of each layer, composed), written into
    ``job.folder`` and nowhere else; with ``each_layer`` and ``masks`` the layers and masks in its subfolders."""
    from ...render.layers import ExportOptions, export_rgb
    cp = renderer.composer
    layers = job.layers
    canvas = cp.canvas_of(layers)
    if canvas is None:
        raise ValueError("no layer has a result to export")
    folder = Path(job.folder)
    write = writer or getattr(renderer, "writer", None) or (lambda rgb, out, opts, meta: export_rgb(
        rgb, out, opts, meta.get("wcs"), tuple(meta.get("origin") or (0, 0)), meta.get("history")))
    formats = [job.format] if isinstance(job.format, str) else list(job.format)
    opts = {"name": job.name, "formats": formats, "size": job.size, "jpeg_quality": job.jpeg_quality}
    ids_with_mask = [lay["id"] for lay in layers if lay.get("mask") is not None]
    stop = stop or (lambda: False)
    n_steps = 1 + (sum(1 for lay in layers if lay["kind"] == "workflow") if job.each_layer else 0) \
        + (len(ids_with_mask) if job.masks else 0)
    done = 0

    def step(msg):
        progress(0.9 * done / n_steps, msg)

    step("composing the image")
    comp = cp.compose(layers, 0, want=(), stop=stop, cache=False)
    files: list[Path] = []
    pids = [lay["pid"] for lay in reversed(layers) if lay["kind"] == "workflow" and renderer.source.available(lay["pid"])]
    ref_pid = pids[0]
    wcs = renderer.wcs_of(ref_pid) if job.wcs and getattr(renderer, "wcs_of", None) else None
    meta = {"wcs": wcs, "origin": canvas.origin, "wf": ref_pid if job.wcs else None,
            "history": history_of(job, renderer, canvas) if job.history else None}
    files += write(np.ascontiguousarray(np.moveaxis(comp.rgb, -1, 0)), folder, ExportOptions(**opts), meta)
    done += 1
    del comp
    if job.each_layer:
        for lay in [x for x in layers if x["kind"] == "workflow"]:
            step(f"layer {lay['name']}")
            c = cp.compose(only_layer(layers, lay["id"]), 0, cache=False, stop=stop)
            o = dict(opts, name=f"{job.name}_layer_{_slug(lay['name'])}")
            files += write(np.ascontiguousarray(np.moveaxis(c.rgb, -1, 0)), folder / "layers", ExportOptions(**o),
                           {"wcs": wcs, "origin": canvas.origin})
            done += 1
    if job.masks:
        for lid in ids_with_mask:
            step("masks")
            c = cp.compose(layers, 0, want=(lid,), cache=False, stop=stop)
            m = c.masks.get(lid)
            if m is None:
                done += 1
                continue
            lay = next(x for x in layers if x["id"] == lid)
            o = dict(opts, name=f"{job.name}_mask_{_slug(lay['name'])}", formats=["png16"])
            files += write(np.repeat(m[None], 3, 0).astype(np.float32), folder / "masks", ExportOptions(**o),
                           {"origin": canvas.origin})
            done += 1
    progress(1.0, "done")
    return files


# ---------------------------------------------------------------------------------------------- the thread
class RenderWorker(QThread):
    """The thread that renders for Post. Requests are kept by lane (main: draft and half, region, before, thumbs): a
    newer request replaces the one waiting in its lane, and abandons the one being computed in it; a request of a lane
    that comes first in ``PRIORITY`` abandons the others. Results come back as signals, in the GUI thread."""

    rendered = Signal(object)                    # Result
    failed = Signal(int, str, str)               # seq, kind, text
    progress = Signal(float, str)                # export
    exported = Signal(object, str)               # the files written (list of paths), or [] and the error text
    deviceChanged = Signal(str)

    def __init__(self, project=None, *, resolver=None, device: str = "auto", bases=None, service=None,
                 parent=None):
        super().__init__(parent)
        self.project = project
        self.pref = device
        self._resolver = resolver
        self._bases = bases
        self._service = service
        self._cond = threading.Condition()
        self._slots: dict[str, Request] = {}
        self._export: ExportJob | None = None
        self._seq = 0
        self._running: tuple[str, int] | None = None        # (lane, seq) being computed
        self._stopping = False
        self._refresh = False
        self._cancel_export = False
        self.renderer: Renderer | None = None
        self.device = ""
        self._busy_at = 0.0
        self._last_busy: str | None = None
        self.stats: list[tuple[str, float, float]] = []     # (kind, total ms, engine ms), the last ones
        self.start()

    # ------------------------------------------------------------ GUI thread
    def request(self, req: Request) -> int:
        lane = LANES[req.kind]
        with self._cond:
            self._seq += 1
            req.seq = self._seq
            self._slots[lane] = req
            self._cond.notify()
        return req.seq

    def cancel(self, lane: str):
        with self._cond:
            self._slots.pop(lane, None)

    def refresh(self):
        """The results on disk may have changed (a run ended): look again before the next request."""
        with self._cond:
            self._refresh = True

    def export(self, job: ExportJob):
        with self._cond:
            self._export = job
            self._cancel_export = False
            self._cond.notify()

    def cancel_export(self):
        """Stop the export at its next step (the files already written stay)."""
        self._cancel_export = True

    def busy(self) -> bool:
        with self._cond:
            return bool(self._slots) or self._running is not None or self._export is not None

    def settle(self, timeout: float = 60.0) -> bool:
        """Process events until everything asked is delivered (tests and scripted walkthroughs)."""
        from PySide6.QtCore import QCoreApplication
        end = time.monotonic() + timeout
        app = QCoreApplication.instance()
        while self.busy() and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.005)
        for _ in range(3):
            app.processEvents()
            time.sleep(0.01)
        return not self.busy()

    def stop(self, timeout_ms: int = 8000):
        """End the thread after the request it is running (an engine opening cannot be interrupted: a worker that is
        still busy when the wait ends is kept alive, a stopped Qt thread must not be destroyed while it runs)."""
        with self._cond:
            self._stopping = True
            self._slots.clear()
            self._cond.notify()
        if self.isRunning() and QThread.currentThread() is not self:
            if not super().wait(timeout_ms):
                _RETIRED.append(self)
        _RETIRED[:] = [w for w in _RETIRED if w.isRunning()]

    # ------------------------------------------------------------ the thread
    def _make(self):
        from ...render.layers import LayerService
        dev = self._pick_device()
        self.device = dev
        service = self._service or LayerService(dev)
        resolver = self._resolver or api_resolver(self.project)
        bases = self._bases or (bases_of(self.project) if self.project is not None else None)
        self.renderer = Renderer(LayerSource(resolver, service, bases), device=dev)
        if self.project is not None:
            self.renderer.wcs_of = self._wcs_of
            self.renderer.writer = self._write
        self.deviceChanged.emit(dev)

    def _write(self, rgb, out_dir, opts, meta):
        """The export through the engine API (render/export), into the folder chosen and nowhere else."""
        from ... import api
        m = {"origin": meta.get("origin"), "history": meta.get("history"), "out_dir": str(out_dir)}
        if meta.get("wf") is not None:
            m["wf"] = meta["wf"]
        return api.export_image(self.project, rgb, m, opts)

    def _wcs_of(self, pid: int):
        from ... import api
        try:
            f = Path(api.stage_outputs(self.project, pid, "astrometry").dir) / "wcs.fits"
        except (AttributeError, OSError, ValueError, KeyError):
            return None
        return f if f.is_file() else None

    def _pick_device(self) -> str:
        self._last_busy = self._busy_device()
        return choose_device(self.pref, self._last_busy)

    def _busy_device(self) -> str | None:
        if self.project is None:
            return None
        try:
            from ... import api
            r = api.running(self.project)
            return r.device if r is not None else None
        except Exception:  # noqa: BLE001 - no run information: no busy GPU
            return None

    def _recheck_device(self):
        """A run may have started or ended since the engines were opened: when the GPU a run holds changes, pick the
        device again between two requests (never otherwise: reopening the engines takes seconds)."""
        if self.renderer is None or time.monotonic() - self._busy_at < 3.0:
            return
        self._busy_at = time.monotonic()
        busy = self._busy_device()
        if busy == self._last_busy:
            return
        self._last_busy = busy
        want = choose_device(self.pref, busy)
        if want != self.device:
            svc = self.renderer.source.service
            svc.evict("all")
            svc.device = want
            svc._on_cpu.clear()
            self.renderer.device = want
            self.renderer.composer.forget()
            self.device = want
            self.deviceChanged.emit(want)

    def _superseded(self, lane: str, seq: int, kind: str = "") -> bool:
        """Whether the request being computed should be abandoned. A newer request of its lane replaces it, a request of
        a more urgent lane too; but a draft is only abandoned by the release (the half image), never by another draft:
        while a slider moves, each draft that finishes is shown and the newest waiting one comes next."""
        with self._cond:
            if self._stopping:
                return True
            if lane == "export":
                return False
            mine = PRIORITY.index(lane)
            for other, r in self._slots.items():
                if other == lane and r.seq > seq and (kind != "draft" or r.kind != "draft"):
                    return True
                if other in PRIORITY and PRIORITY.index(other) < mine:
                    return True
        return False

    def _next(self):
        with self._cond:
            while not self._stopping and not self._slots and self._export is None:
                self._cond.wait()
            if self._stopping:
                return None
            if self._export is not None:
                job, self._export = self._export, None
                self._running = ("export", 0)
                return "export", job
            for lane in PRIORITY:
                if lane in self._slots:
                    req = self._slots.pop(lane)
                    self._running = (lane, req.seq)
                    return lane, req
        return None

    def run(self):
        try:
            self._make()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(0, "start", f"{type(e).__name__}: {e}")
            self.renderer = None
        try:
            self._loop()
        finally:
            if self.renderer is not None:                       # the engines (gigabytes) go with the page
                self.renderer.source.service.evict("all")
                self.renderer = None

    def _loop(self):
        while True:
            item = self._next()
            if item is None:
                return
            lane, work = item
            try:
                if self.renderer is None:
                    self._make()
                if self._refresh:
                    self._refresh = False
                    if self.renderer.source.refresh():
                        self.renderer.composer.forget()
                if lane == "export":
                    self._run_export(work)
                else:
                    if work.kind == "half":
                        self._recheck_device()
                    res = self.renderer.run(work, lambda l=lane, s=work.seq, k=work.kind: self._superseded(l, s, k))
                    self.stats = (self.stats + [(work.kind, res.ms, res.layer_ms)])[-50:]
                    self.rendered.emit(res)
            except Superseded:
                if lane in ("region", "before", "thumbs"):          # abandoned for a more urgent lane: keep it
                    with self._cond:
                        if lane not in self._slots and not self._stopping:
                            self._slots[lane] = work
            except Exception as e:  # noqa: BLE001 - every failure goes back to the GUI as text
                import traceback
                if lane == "export":
                    self.exported.emit([], f"{type(e).__name__}: {e}")
                else:
                    self.failed.emit(work.seq, work.kind, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}")
            finally:
                with self._cond:
                    self._running = None

    def _run_export(self, job: ExportJob):
        try:
            files = self.renderer.export(job, lambda f, m: self.progress.emit(f, m),
                                         lambda: self._stopping or self._cancel_export)
            self.exported.emit([str(p) for p in files], "")
        except Superseded:
            self.exported.emit([], "stopped")
        except Exception as e:  # noqa: BLE001
            self.exported.emit([], f"{type(e).__name__}: {e}")


def default_device_pref() -> str:
    """The device of Settings > Compute for Post's renders ('auto', 'cpu', 'cuda:N')."""
    try:
        from .. import prefs
        return str(prefs.get("device") or "auto")
    except Exception:  # noqa: BLE001
        return os.environ.get("ASTRO_GPU_DEVICE", "auto")
