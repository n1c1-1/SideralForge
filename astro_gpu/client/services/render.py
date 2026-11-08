"""The render worker of Post: the engine's layers, the composition and the export, off the GUI thread.

* ``LayerSource`` is the ``postengine.Provider`` on the engine: a workflow's Output block folder (or its Stars block
  folder, for the starless image, the stars alone and the two recombined) rendered by a ``LayerService``
  (render/layers.py) with a base and slider positions, on the canvas every layer covers. The release render, the 100 %
  region and the export are the engine's own renderings; nothing here changes a pixel of them. It also renders the
  image layers: a stage result's linear image (an array, a file, or a function giving one, registered under its key;
  else found from the key by ``image_resolver``), stretched by the simple rendering (render/simple.py).
* ``Renderer`` runs a request (a draft while a slider moves, the screen at 1/2 - first the 1/4 when ``refine`` -, a
  region at full resolution, the before image, the thumbnails) and an export. ``RenderWorker`` is the QThread that owns
  one: requests are coalesced by lane, the newest of a lane wins, and a request being computed is abandoned at its next
  checkpoint (between two layers, or two stages of the engine) when a newer one arrives.
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
    is_pixel,
    only_layer,
    source_id,
    thumb_of,
)

#: the stage of a workflow's layer: its Output block's image, and its Stars block's images (``postengine.source_id``:
#: the workflow's id, or ("stars", id))
OUTPUT_STAGE, STARS_STAGE = "render_simple", "star_removal"
#: the time budgets of a request, in seconds
BUDGET = {"draft": 0.15, "half": 1.0, "region": 2.0}
DRAFT_LEVEL, HALF_LEVEL = 2, 1
#: the draft's level on the CPU: an eighth of the image (the nebula of a 6000-pixel image in about 0.3 s)
CPU_DRAFT_LEVEL = 3
#: the main arrays of the stage results an image layer's key can name without its file (``store_image_resolver``)
STAGE_IMAGES = {"stack": "drz_full", "denoise": "n2n_full", "render_simple": "lin", "star_removal": "starless",
                "color": None}
NOTE_CPU_TEXT = "GPU memory full: rendered on the CPU"


# ---------------------------------------------------------------------------------------------- the device
def gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001 - no torch, no driver
        return 0


def _free(device: str) -> int | None:
    """Bytes free on the card now, None when it cannot tell."""
    try:
        from ...core import devices
        return devices.free_memory(device)
    except Exception:  # noqa: BLE001 - no torch, no driver
        return None


def _available(device: str) -> bool:
    try:
        from ...core import devices
        return devices.available(device)
    except Exception:  # noqa: BLE001
        return False


def _first_other() -> str:
    """The first card of another kind (an Intel, Apple or AMD card as the settings offer it), else the CPU."""
    try:
        from ...core import devices
        cards = [d.device for d in devices.list_devices() if not d.device.startswith("cuda")]
    except Exception:  # noqa: BLE001
        return "cpu"
    return cards[0] if cards else "cpu"


def choose_device(pref: str = "auto", busy: str | None = None, need_gb: float = 3.0) -> str:
    """Where Post renders: the GPU the settings name, else the free GPU with the most memory, else the CPU. ``busy``:
    the GPU a running job holds (never used for Post while it runs). A GPU with less than ``need_gb`` free is skipped."""
    if (pref or "auto").lower() == "cpu":
        return "cpu"
    if pref and pref != "auto" and not pref.startswith("cuda"):        # another card, named in the settings (beta)
        return pref if _available(pref) else "cpu"
    n = gpu_count()
    if n == 0:
        return _first_other()
    free = []
    order = [f"cuda:{i}" for i in range(n)]
    if pref and pref != "auto" and pref in order:
        order = [pref] + [d for d in order if d != pref]
    for d in order:
        if d == busy:
            continue
        room = _free(d)
        if room is None:
            continue
        gb = room / 2 ** 30
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
    """The grid and the useful frame of an Output or Stars block folder, from its small files (the arrays are not
    loaded)."""
    from ...render.layers import layer_kind
    from ...render.stars import is_stars_folder
    stars = is_stars_folder(folder)
    kind = "stars" if stars else layer_kind(folder)
    pj = json.loads((folder / ("prep_stars.json" if stars else "prep_simple.json")).read_text(encoding="utf-8"))
    with open(folder / "valid.npy", "rb") as fh:
        v = np.lib.format.read_magic(fh)
        shape = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(fh)[0]
    return Grid(int(shape[1]), int(shape[0]), tuple(int(c) for c in pj["crop"]), kind)


def api_resolver(project) -> Callable[[Any], Folder | None]:
    """A workflow's id -> the folder of its Output block; ("stars", id) -> the folder of its Stars block: the current
    result, else the one an Outdated stage had."""
    from ... import api

    def resolve(sid) -> Folder | None:
        pid, stage = (sid[1], STARS_STAGE) if isinstance(sid, tuple) else (sid, OUTPUT_STAGE)
        try:
            chain = api.chain(project, pid)
            if stage not in chain:                       # an already stretched Image stands for the Output (it
                from ...core import workflow as wfl  # writes what the Output would: workflow.PROVIDES)
                stage = next((s for s in chain if stage in wfl.PROVIDES.get(wfl.base(s), ())), None)
                if stage is None:
                    return None
            so = api.stage_outputs(project, pid, stage) or api.stage_outputs(project, pid, stage, "previous")
        except (OSError, ValueError, KeyError):
            return None
        if so is None:
            return None
        d = Path(so.dir)
        if stage != OUTPUT_STAGE and stage != STARS_STAGE and not (d / "prep_simple.json").is_file():
            return None                                  # a linear image: no Output image yet
        try:
            mt = (d / "manifest.json").stat().st_mtime
        except OSError:
            mt = 0.0
        return Folder(d, so.which == "previous", mt)
    return resolve


def load_image(source) -> np.ndarray:
    """The pixels of an image layer's source: an array (numpy or torch), a file (.npy, .fits, .tif, .png, .jpg) or a
    function returning either (called here, in the render worker)."""
    if callable(source):
        source = source()
    if hasattr(source, "detach") and hasattr(source, "cpu"):
        source = source.detach().cpu().numpy()
    if isinstance(source, np.ndarray):
        return source
    path = Path(source)
    suf = path.suffix.lower()
    if suf == ".npy":
        return np.load(path)
    if suf in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        with fits.open(path) as h:
            hdu = next(x for x in h if x.data is not None and x.data.ndim >= 2)
            return np.asarray(hdu.data, np.float32)
    if suf in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise OSError(f"cannot read the image {path}")
    return a[..., ::-1] if a.ndim == 3 else a


def store_image_resolver(root) -> Callable[[str], Any]:
    """key -> the file of a stage result in the project's store: "<stage>/<store key>[/<array>]" (the array's name with
    or without .npy; without it, the stage's main image, STAGE_IMAGES), or None when it is not there."""
    store = Path(root) / "store"

    def resolve(key: str):
        parts = [x for x in str(key).replace("\\", "/").split("/") if x]
        if len(parts) < 2 or any(x in ("..", ".") for x in parts):
            return None
        d = store / parts[0] / parts[1]
        if not (d / "manifest.json").is_file():
            return None
        name = parts[2] if len(parts) > 2 else STAGE_IMAGES.get(parts[0])
        if name is None:
            big = sorted(d.glob("*.npy"), key=lambda f: f.stat().st_size, reverse=True)
            return big[0] if big else None
        f = d / (name if name.endswith((".npy", ".fits", ".tif")) else name + ".npy")
        return f if f.is_file() else None
    return resolve


def bases_of(project) -> Callable[[Any], Any]:
    """A named rendering of the project -> its settings; anything else (a preset, settings, None) as it is."""
    def base(b):
        rb = getattr(project, "render_bases", None) or {}
        return rb[b] if isinstance(b, str) and b in rb else b
    return base


class LayerSource(Provider):
    """The layers of the workflows as the composition reads them. Used in one thread (the worker's)."""

    def __init__(self, resolver: Callable[[Any], Folder | None], service, bases: Callable[[Any], Any] | None = None):
        self.resolver = resolver
        self.service = service
        self.bases = bases or (lambda b: b)
        self._folders: dict[Any, Folder | None] = {}          # by source id: a workflow's, or ("stars", its id)
        self._grids: dict[str, Grid] = {}
        self._epoch: dict[Any, int] = {}
        self._notes: set[str] = set()
        self._masks: dict[tuple, dict] = {}
        self._cv: Canvas | None = None
        self._ref: Grid | None = None
        self._pids: list = []
        self.engine_ms = 0.0                                  # time the engine spent since the last reset
        # the image layers: their sources by key (set from the GUI through the worker), the images loaded
        self.image_sources: dict[str, Any] = {}
        self.image_resolver: Callable[[str], Any] | None = None
        self._img: dict[str, Any] = {}                       # key -> (SimpleRenderEngine, Grid), or None: not there
        self._img_epoch: dict[str, int] = {}

    # ------------------------------------------------------------ the results on disk
    def folder(self, sid) -> Folder | None:
        """The folder of a workflow's Output block (``sid`` its id) or Stars block (``sid`` ("stars", its id))."""
        if sid not in self._folders:
            self._folders[sid] = self.resolver(sid)
        return self._folders[sid]

    def refresh(self) -> set:
        """The results may have changed: look again. The source ids whose result changed (a new folder, or the same
        folder written again) lose their engine and their cached images."""
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
        for key in [k for k, ent in self._img.items() if ent is None]:   # a missing image may be there now
            del self._img[key]                        # (found, it changes the layers composed: a new canvas)
        if changed:
            self._masks.clear()
            self._cv = None
        return changed

    # ------------------------------------------------------------ the image layers
    @staticmethod
    def _is_image(sid) -> bool:
        return isinstance(sid, tuple) and len(sid) == 2 and sid[0] == "image"

    def _is_workflow(self, sid) -> bool:
        return not self._is_image(sid)

    def set_image(self, key: str, source) -> None:
        """The source of an image layer (an array, a file, a function); a new source for a key replaces its image."""
        self.image_sources[key] = source
        if key in self._img:
            self._img.pop(key)
            self._img_epoch[key] = self._img_epoch.get(key, 0) + 1     # (a new version: the canvas is measured again)

    def image(self, key: str):
        """(engine, grid) of an image layer's key, loaded once; None when its source cannot be found or read."""
        if key not in self._img:
            src = self.image_sources.get(key)
            if src is None and self.image_resolver is not None:
                try:
                    src = self.image_resolver(key)
                except Exception:  # noqa: BLE001 - a key that cannot be resolved: the layer says its image is missing
                    src = None
            ent = None
            if src is not None:
                from ...render.simple import SimpleRenderEngine
                try:
                    eng = SimpleRenderEngine.from_array(load_image(src), device="cpu")
                    ent = (eng, Grid(eng.W, eng.H, (0, eng.W, 0, eng.H), "image"))
                except Exception:  # noqa: BLE001 - an unreadable source: the same (the other layers are composed)
                    ent = None
            self._img[key] = ent
        return self._img[key]

    def version(self, pid) -> int:
        if self._is_image(pid):
            return self._img_epoch.get(pid[1], 0)
        return self._epoch.get(pid, 0)

    def available(self, pid) -> bool:
        if self._is_image(pid):
            return self.image(pid[1]) is not None
        return self.folder(pid) is not None

    def previous(self, pid) -> bool:
        if self._is_image(pid):
            return False
        f = self.folder(pid)
        return bool(f and f.previous)

    def grid(self, pid) -> Grid:
        if self._is_image(pid):
            ent = self.image(pid[1])
            if ent is None:
                raise FileNotFoundError(f"the image {pid[1]!r} is not there")
            return ent[1]
        f = self.folder(pid)
        if f is None:
            raise FileNotFoundError(f"workflow {pid} has no result to show")
        k = str(f.path)
        if k not in self._grids:
            self._grids[k] = read_grid(f.path)
        return self._grids[k]

    def kind(self, pid) -> str:
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
    def layer(self, lay: dict, level: int, box, draft: bool = False, stop=None) -> np.ndarray:
        if lay["kind"] == "image":
            return self._image_layer(lay, level, box)
        sid = source_id(lay)
        f = self.folder(sid)
        cv = self._cv
        whole = cv.size(level)
        fi, region, size = self._window(sid, level, box)
        base = self.bases(lay.get("base"))
        kw = {"part": lay.get("part"), "blend": lay.get("blend")}
        if level == 0 and tuple(box) != (0, 0, *whole):
            li = self.service.render(f.path, base, lay.get("sliders"), 1.0, region=tuple(int(round(v)) for v in region),
                                     **kw)
            arr = self._cut(li.rgb, 1, region, size, False)
        else:                                               # the whole grid at 1/fi, cut to the canvas
            li = self.service.render(f.path, base, lay.get("sliders"), 1.0 / fi, **kw)
            arr = self._cut(li.rgb, fi, region, size, True)
        if li.info.get("note"):
            self._notes.add(li.info["note"])
        self.engine_ms += float(li.info.get("render_ms") or 0.0)
        return np.ascontiguousarray(np.moveaxis(arr, 0, -1), np.float32)

    def _image_layer(self, lay: dict, level: int, box) -> np.ndarray:
        """An image layer: the stage result stretched like a simple layer (Sky, Contrast, Saturation), linked or each
        channel apart, cut to the canvas."""
        from ...render.sliders import settings_at
        sid = source_id(lay)
        eng = self.image(lay["key"])[0]
        eng.prep["linked"] = bool(lay.get("linked", True))
        s = settings_at("simple", None, lay.get("sliders"), has_denoise=False)
        fi, region, size = self._window(sid, level, box)
        t0 = time.perf_counter()
        if level == 0 and tuple(box) != (0, 0, *self._cv.size(0)):
            reg = tuple(int(round(v)) for v in region)
            arr = self._cut(eng.render(s, 1.0, region=reg).final.cpu().numpy(), 1, region, size, False)
        else:
            arr = self._cut(eng.render(s, 1.0 / fi).final.cpu().numpy(), fi, region, size, True)
        self.engine_ms += 1000 * (time.perf_counter() - t0)
        return np.ascontiguousarray(np.moveaxis(arr, 0, -1), np.float32)

    def has(self, name: str) -> bool:
        return any(self.available(p) for p in self._pids if self._is_workflow(p))

    def _mask_pid(self, name: str):
        for p in self._pids:                                  # bottom first (the workflows: the images have no map)
            if self._is_workflow(p) and self.available(p):
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
    thumbnails), inspect (a workflow's final image for Inspect: the same rendering as half, on a lane of its own so that
    it never replaces or stops what Post asked for), exact (after the release, the half image made from the full
    resolution: what the export writes, reduced)."""
    kind: str = "half"
    layers: list = field(default_factory=list)          # a snapshot (postengine.snapshot)
    level: int = HALF_LEVEL
    box: tuple | None = None                            # at the level, in canvas pixels (None: all of it)
    want: tuple = ()                                    # ids of the layers whose mask is returned
    thumbs: bool = False                                # also the mask thumbnails
    overlay: bool = False                               # a red overlay of the first wanted mask
    need_thumbs: tuple = ()                             # (thumbs kind) the ids of the workflow layers to thumbnail
    refine: bool = False                                # (half) first the exact 1/4, shown at once ("refine")
    hists: tuple = ()                                   # ids of filters whose input's histograms come back (not drafts)
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
    missing: tuple = ()                                 # image layers' keys whose image was not found
    hists: dict = field(default_factory=dict)           # filter id -> {channel: counts}: the Curves editor's histograms


_RETIRED: list = []                    # workers asked to stop that were still busy: kept until their thread ends
LANES = {"draft": "main", "half": "main", "refine": "main", "exact": "main", "region": "region", "before": "before",
         "inspect": "inspect", "thumbs": "thumbs"}
PRIORITY = ("main", "region", "before", "inspect", "thumbs")


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

    def run(self, req: Request, stop: Callable[[], bool] | None = None,
            emit: Callable[[Result], None] | None = None) -> Result:
        """The request's image. ``emit``: where a "refine" result goes before the half image (the exact 1/4, when
        ``req.refine``)."""
        t0 = time.perf_counter()
        cp = self.composer
        self.source.engine_ms = 0.0
        if req.kind == "thumbs":
            return self._thumbs(req, stop, t0)
        if req.kind == "exact":
            return self._exact(req, stop, t0)
        layers = req.layers
        if req.kind == "before":
            layers = self._before_layers(layers)
        if req.kind == "half" and req.refine and req.box is None and emit is not None:
            first = Request(kind="refine", layers=layers, level=req.level + 1, want=req.want, overlay=req.overlay,
                            hists=req.hists, seq=req.seq, tag=req.tag)
            emit(self.run(first, stop))
            t0 = time.perf_counter()
            self.source.engine_ms = 0.0
        comp = cp.compose(layers, req.level, req.box, want=req.want, thumbs=req.thumbs, stop=stop,
                          draft=req.kind == "draft", hists=() if req.kind == "draft" else req.hists)
        if stop is not None and stop():                       # a newer request came meanwhile: this answer is stale
            raise Superseded()
        if comp is None:
            return Result(req.seq, req.kind, req.level, req.box, None, None, device=self.device, tag=req.tag)
        res = Result(req.seq, req.kind, req.level, comp.box, to_u8(comp.rgb), cp.canvas_of(layers), comp.masks,
                     None, comp.mask_thumbs, {}, set(), 0.0, self.source.engine_ms, comp.drawn, comp.notes,
                     self.device, req.tag)
        res.hists = comp.hists
        if req.overlay and req.want and req.want[0] in comp.masks:
            res.overlay = red_overlay(comp.masks[req.want[0]])
        res.previews = {lay["pid"] for lay in layers if lay["kind"] == "workflow" and
                        self.source.available(source_id(lay)) and self.source.previous(source_id(lay))}
        res.missing = tuple(sorted({lay["key"] for lay in layers if lay["kind"] == "image"
                                    and not self.source.available(source_id(lay))}))
        if req.kind in ("half", "refine") and req.box is None:        # thumbnails of what was just drawn, for free
            res.layer_thumbs = self._visible_thumbs(layers, cp.canvas_of(layers))
        res.ms = 1000 * (time.perf_counter() - t0)
        return res

    def _exact(self, req: Request, stop, t0) -> Result:
        """The half image made from the full resolution (GUI exploration 4, Q1): the composition at level 0, clipped as
        the export writes it, then reduced. A filter run on the reduced image sees its colour noise already averaged
        (a strong Saturation turned the sky another colour on screen than in the file); this one shows the file."""
        cp = self.composer
        comp = cp.compose(req.layers, 0, stop=stop)
        if stop is not None and stop():
            raise Superseded()
        canvas = cp.canvas_of(req.layers)
        if comp is None or canvas is None:
            return Result(req.seq, "exact", req.level, None, None, None, device=self.device, tag=req.tag)
        w, h = canvas.size(req.level)
        small = cv2.resize(np.clip(comp.rgb, 0, 1), (w, h), interpolation=cv2.INTER_AREA)
        res = Result(req.seq, "exact", req.level, None, to_u8(small), canvas, device=self.device, tag=req.tag)
        res.layer_ms, res.drawn, res.notes = self.source.engine_ms, comp.drawn, comp.notes
        res.ms = 1000 * (time.perf_counter() - t0)
        return res

    def _before_layers(self, layers: list[dict]) -> list[dict]:
        """The bottom pixel layer alone: its base, every slider at the suggestion, no filter, no mask. The others stay
        in the list, hidden, so that the canvas is the same as the composition's. Its id is its own (negative), so that
        its image is cached beside the composition's, not instead of it."""
        wl = [lay for lay in layers if is_pixel(lay) and self.source.available(source_id(lay))]
        if not wl:
            return []
        bottom = wl[-1]
        out = [dict(lay, visible=False) for lay in layers if is_pixel(lay) and lay is not bottom]
        return out + [dict(bottom, id=-abs(bottom["id"]), visible=True, opacity=100.0, mode="Normal", sliders={},
                           mask=None, part=None)]

    def _visible_thumbs(self, layers, canvas) -> dict:
        """The thumbnails of the pixel layers whose quarter-resolution image is in the cache (the refine's)."""
        out = {}
        if canvas is None:
            return out
        for lay in layers:
            if not is_pixel(lay) or not self.source.available(source_id(lay)):
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
            if lay["id"] in req.need_thumbs and is_pixel(lay) and self.source.available(source_id(lay)):
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
    #: burned into the PNG / JPEG / TIFF (never the FITS): {"grid": bool, "objects": bool}, or None
    annotations: dict | None = None


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
            "workflows": {str(lay["pid"]) + (f" {lay['part']}" if lay.get("part") else ""):
                          {"folder": str(src.folder(source_id(lay)).path), "outdated": src.previous(source_id(lay))}
                          for lay in job.layers if lay["kind"] == "workflow" and src.available(source_id(lay))},
            "images": {lay["key"]: lay["name"] for lay in job.layers if lay["kind"] == "image"}}


def _solution(sol):
    """An astropy WCS from what the render worker gives (a wcs.fits path, a header or a WCS)."""
    if sol is None or hasattr(sol, "all_world2pix"):
        return sol
    import warnings

    from astropy.io import fits
    from astropy.wcs import WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(fits.getheader(sol) if isinstance(sol, str | Path) else sol, naxis=2)


def export_header(sol):
    """The solution as the export writes it: the header of the workflow's wcs.fits without its empty image axes (that
    file holds a header only, NAXIS = 0, which astropy notes with a warning on every FITS export when it builds the
    WCS of the exported image). A header or a WCS goes through as it is."""
    if not isinstance(sol, str | Path):
        return sol
    from astropy.io import fits
    try:
        h = fits.getheader(sol)
    except OSError:
        return sol
    for key in [k for k in h if k == "NAXIS" or (k.startswith("NAXIS") and k[5:].isdigit())]:
        del h[key]
    return h


def canvas_annotations(sol, canvas, show: dict):
    """The sky grid and the named objects of Post's canvas (``annotate.field_annotations`` on the solution of the
    bottom workflow's grid, moved to the canvas's origin)."""
    from ..annotate import crop_wcs, field_annotations
    w = crop_wcs(_solution(sol), canvas.origin)
    return field_annotations(w, canvas.W, canvas.H, grid=bool(show.get("grid")), names=bool(show.get("objects")))


def _burned(rgb: np.ndarray, ann, show: dict) -> np.ndarray:
    from ..annotate import burn
    return np.ascontiguousarray(burn(np.clip(rgb, 0.0, 1.0), ann, show), dtype=np.float32)


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
    n_steps = 1 + (sum(1 for lay in layers if is_pixel(lay)) if job.each_layer else 0) \
        + (len(ids_with_mask) if job.masks else 0)
    done = 0

    def step(msg):
        progress(0.9 * done / n_steps, msg)

    step("composing the image")
    comp = cp.compose(layers, 0, want=(), stop=stop, cache=False)
    files: list[Path] = []
    # the solution of the bottom layer's grid (the canvas is in its pixels): a workflow's astrometry; an image layer of
    # the stack grid has none of its own, the first workflow's then (the same grid when the drizzle scales agree)
    pids = [lay["pid"] for lay in reversed(layers) if lay["kind"] == "workflow"
            and renderer.source.available(source_id(lay))]
    bottom = next((lay for lay in reversed(layers) if is_pixel(lay) and renderer.source.available(source_id(lay))), None)
    ref_pid = pids[0] if pids else None
    if bottom is not None and bottom["kind"] == "image" and ref_pid is not None:
        ga = renderer.source.grid(next(source_id(x) for x in reversed(layers) if x["kind"] == "workflow"
                                       and x["pid"] == ref_pid))
        gb = renderer.source.grid(source_id(bottom))
        if (ga.W, ga.H) != (gb.W, gb.H):
            ref_pid = None
    sol = renderer.wcs_of(ref_pid) if ref_pid is not None and getattr(renderer, "wcs_of", None) else None
    wcs = export_header(sol) if job.wcs else None
    meta = {"wcs": wcs, "origin": canvas.origin, "wf": ref_pid if job.wcs else None,
            "history": history_of(job, renderer, canvas) if job.history else None}
    rgb = np.ascontiguousarray(np.moveaxis(comp.rgb, -1, 0))
    show = dict(job.annotations or {})
    pictures = [f for f in formats if f != "fits32f"]
    if (show.get("grid") or show.get("objects")) and pictures and sol is not None:
        # the annotations (sky grid, named objects) burned into the pictures; the FITS file stays clean, with its WCS
        step("annotations")
        ann = canvas_annotations(sol, canvas, show)
        files += write(_burned(rgb, ann, show), folder, ExportOptions(**dict(opts, formats=pictures)), meta)
        rest = [f for f in formats if f not in pictures]
        if rest:
            files += write(rgb, folder, ExportOptions(**dict(opts, formats=rest)), meta)
    else:
        files += write(rgb, folder, ExportOptions(**opts), meta)
    del rgb
    done += 1
    del comp
    if job.each_layer:
        for lay in [x for x in layers if is_pixel(x)]:
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
        self._images: dict[str, Any] = {}                     # image sources to hand to the source (GUI -> worker)
        self._resolver_img: Callable[[str], Any] | None = None
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

    def set_image(self, key: str, source) -> None:
        """The source of an image layer (an array, a file, a function returning one): given to the worker's source
        before its next request; the pixels are read there, never in the GUI thread."""
        with self._cond:
            self._images[key] = source

    def set_image_resolver(self, fn: Callable[[str], Any] | None) -> None:
        """key -> source for the image layers whose source was not given (a composition opened again)."""
        with self._cond:
            self._resolver_img = fn
            self._images.setdefault("", None)                 # (an empty key: the resolver is applied at the next turn)

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
        root = getattr(self.project, "root", None)
        self.renderer.source.image_resolver = store_image_resolver(root) if root else None
        if self.project is not None:
            self.renderer.wcs_of = self._wcs_of
            self.renderer.writer = self._write
        self.deviceChanged.emit(dev)

    def _write(self, rgb, out_dir, opts, meta):
        """The export through the engine API (api.export_image), into the folder chosen and nowhere else."""
        from ... import api
        m = {"origin": meta.get("origin"), "history": meta.get("history"), "out_dir": str(out_dir)}
        if meta.get("wcs") is not None:              # the solution's header as run_export read it (export_header)
            m["wcs"] = meta["wcs"]
        elif meta.get("wf") is not None:
            m["wf"] = meta["wf"]
        return api.export_image(self.project, rgb, m, opts)

    def _wcs_of(self, pid: int):
        from ... import api
        try:                                         # an Outdated workflow's layer shows its previous result: its solution
            out = api.stage_outputs(self.project, pid, "astrometry") or \
                api.stage_outputs(self.project, pid, "astrometry", "previous")
            f = Path(out.dir) / "wcs.fits"
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
                self._take_images()
                if lane == "export":
                    self._run_export(work)
                else:
                    if work.kind in ("half", "inspect"):
                        self._recheck_device()
                    res = self.renderer.run(work, lambda l=lane, s=work.seq, k=work.kind: self._superseded(l, s, k),
                                            self.rendered.emit)
                    self.stats = (self.stats + [(work.kind, res.ms, res.layer_ms)])[-50:]
                    self.rendered.emit(res)
            except Superseded:
                if lane in ("region", "before", "inspect", "thumbs"):   # abandoned for a more urgent lane: keep it
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

    def _take_images(self):
        """The image sources and resolver the GUI gave since the last request, into the source (worker thread)."""
        with self._cond:
            images, self._images = self._images, {}
            fn = self._resolver_img
        if not images:
            return
        src = self.renderer.source
        if fn is not None:
            store = src.image_resolver
            src.image_resolver = (lambda key, a=fn, b=store: a(key) or (b(key) if b else None))
            self._resolver_img = None
            for key in [k for k, ent in src._img.items() if ent is None]:   # looked for before: look again
                del src._img[key]
        for key, source in images.items():
            if key:
                src.set_image(key, source)

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
    """The device of Settings > Compute for Post's renders ('auto', 'cpu', 'cuda:N'): the CPU when the card test sent
    Post there."""
    try:
        from ...core import selftest
        from .. import prefs
        dev = str(prefs.get("device") or "auto")
        return "cpu" if selftest.stage_on_cpu(prefs.get("compute_profiles"), dev, "post") else dev
    except Exception:  # noqa: BLE001
        return os.environ.get("ASTRO_GPU_DEVICE", "auto")
