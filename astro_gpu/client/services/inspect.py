"""The worker side of Inspect and Frames (DESIGN §5.3): everything that reads the disk or takes time.

The views (tune.py, frames.py, sources.py) never touch a file. They ask an ``InspectService``; it runs the job in a
worker thread and answers with a signal in the GUI thread:

* images: the array of a block (stack, denoised, starless...) or the engine's preview, with the automatic display stretch,
  the histograms and the astrometric solution (``ShownImage``); the last images are kept (least recently used);
* the metrics of a done result (``api.stage_outputs``), the frame table of Selection, the sessions with their headers
  and health checks, the frames of a group, their thumbnails and the image of one frame;
* the scan of a folder to import, with its progress.

Three workers: ``images`` (arrays, metrics, tables, scans), ``thumbs`` (the thumbnails of the grid and the filmstrip) and
``view`` (the one frame shown large), so that a long load never keeps another one waiting. Each request carries a
channel and the latest one wins (``ApiWorker``).

``GuiDiskWatch`` is the debug hook of DESIGN §6.3: it counts the file accesses the GUI thread makes under some folders.
"""
from __future__ import annotations

import os
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from ... import api
from .worker import ApiWorker

# ---------------------------------------------------------------------------------------------- the GUI disk watch
_WATCHED = ("open", "os.listdir", "os.scandir", "glob.glob", "os.mkdir", "os.rename", "os.remove", "os.rmdir",
            "shutil.copyfile")
_watch: dict[str, Any] = {"on": False, "tid": None, "roots": (), "hits": [], "where": []}
_installed = False


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _hook(event: str, args: tuple):
    if not _watch["on"] or event not in _WATCHED or threading.get_ident() != _watch["tid"]:
        return
    if not args or not isinstance(args[0], str | bytes | os.PathLike):
        return                                              # a file descriptor, or nothing to compare
    try:
        path = _norm(os.fsdecode(args[0]))
    except (TypeError, ValueError):
        return
    if any(path.startswith(r) for r in _watch["roots"]):
        _watch["hits"].append((event, path))
        if len(_watch["where"]) < 400:                      # who asked: the calls of the client that led here
            import traceback
            fr = [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in traceback.extract_stack()
                  if "/astro_gpu/client/" in f.filename.replace("\\", "/") and f.name != "_hook"]
            _watch["where"].append((path, fr))


class GuiDiskWatch:
    """``with GuiDiskWatch(folders) as w: ...``: ``w.hits`` lists the (event, path) the calling thread opened, listed,
    created, renamed or removed under ``folders`` (DESIGN §6.3: no disk access in the GUI thread); ``w.where`` says, for
    the first of them, which calls of the client led there. The hook is installed once (``sys.addaudithook`` cannot be
    removed) and costs nothing while no watch is open."""

    def __init__(self, folders):
        self.roots = tuple(_norm(str(f)) for f in folders)
        self.hits: list[tuple[str, str]] = []
        self.where: list[tuple[str, list[str]]] = []

    def __enter__(self):
        global _installed
        if not _installed:
            sys.addaudithook(_hook)
            _installed = True
        _watch.update(on=True, tid=threading.get_ident(), roots=self.roots, hits=self.hits, where=self.where)
        return self

    def __exit__(self, *exc):
        _watch.update(on=False, tid=None, roots=(), hits=[], where=[])

    @property
    def count(self) -> int:
        return len(self.hits)


# ---------------------------------------------------------------------------------------------- what a view shows
@dataclass(frozen=True)
class VisualSpec:
    """One picture of a block: where it comes from and how to read it."""

    key: str                 # the tab: "stack", "starless"...
    caption: str             # English, translated where shown
    kind: str                # array | preview | chart
    stage: str
    name: str = ""           # the array (without .npy) or the preview (without .jpg); "" for a chart
    grid: bool = False       # on the stack grid: the RA / Dec readout
    linear: bool = False     # linear data: the automatic stretch applies
    spec: bool = False       # ◇: not in the engine yet (a disabled tab, in Spec mode only)


@dataclass
class ShownImage:
    """What the worker hands the view for one picture. ``stretched`` is what is shown (the engine's preview as it is, or
    the automatic stretch of an array); ``linear`` is the same data without the stretch (None for a preview)."""

    key: tuple
    spec: VisualSpec
    stretched: Any = None                      # a PixelSource
    linear: Any = None                         # a PixelSource, or None
    lin: np.ndarray | None = None              # (3, H, W) float32, the data (ADU); None for a preview
    size: tuple[int, int] = (0, 0)             # (w, h) of the pixels shown
    grid: tuple[int, int] | None = None        # (w, h) of the stack grid the pixels cover (sky readout), or None
    wcs: Any = None
    hist: dict[str, np.ndarray] = field(default_factory=dict)     # "stretched" / "linear" -> (3, 256) counts
    stretch: dict[str, list[float]] = field(default_factory=dict)  # black, white, balance per channel
    curves: Any = None                         # SpectralCurves, for a chart
    nbytes: int = 0


class PixelSource:
    """A fixed picture (H x W x 3 uint8) and its half-resolution copy, an average of 2 x 2 pixels: what the viewer asks
    for when it shows the picture reduced (the viewer's own ``StaticSource`` drops three pixels in four, and the stars
    of a reduced picture would twinkle)."""

    def __init__(self, arr: np.ndarray):
        self.arr = np.ascontiguousarray(arr)
        self.half = reduce2(self.arr)
        self.version = 0

    def size(self):
        return self.arr.shape[1], self.arr.shape[0]

    def region(self, level, x0, y0, x1, y1):
        a = self.arr if level == 0 else self.half
        return a[y0:y1, x0:x1]

    @property
    def nbytes(self) -> int:
        return self.arr.nbytes + self.half.nbytes


def reduce2(a: np.ndarray) -> np.ndarray:
    """H x W x 3 uint8 -> half size, each pixel the mean of 2 x 2 (an odd last row or column is repeated)."""
    h, w = a.shape[:2]
    if h % 2 or w % 2:
        a = np.pad(a, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
    s = a[::2, ::2].astype(np.uint16) + a[1::2, ::2] + a[::2, 1::2] + a[1::2, 1::2]
    return ((s + 2) >> 2).astype(np.uint8)


# ---------------------------------------------------------------------------------------------- the display stretch
def mtf(m: float, x):
    """The midtones transfer function: 0 -> 0, m -> 1/2, 1 -> 1."""
    return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)


def stretch_params(lin: np.ndarray) -> dict[str, np.ndarray]:
    """The automatic stretch of a (3, H, W) linear image, channel by channel: the black point sits 2.8 sigma under the
    median (sigma from the median absolute deviation), the white point at the 99.99th percentile, and the midtones put
    the median at 0.25. A sample of the image gives the numbers: the stretch is a display, never a measurement."""
    c = lin.shape[0]
    sub = lin[:, ::4, ::4].reshape(c, -1)
    sub = sub[:, np.isfinite(sub).all(0)] if not np.isfinite(sub).all() else sub
    med = np.median(sub, axis=1)
    mad = 1.4826 * np.median(np.abs(sub - med[:, None]), axis=1)
    bp = med - 2.8 * np.maximum(mad, 1e-7 * np.maximum(np.abs(med), 1.0))
    wp = np.maximum(np.percentile(sub, 99.99, axis=1), bp + 1e-6)
    mn = np.clip((med - bp) / (wp - bp), 1e-4, 0.99)
    return {"black": bp, "white": wp, "balance": mtf(0.25, mn)}


def _lut(balance: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
    y = x if abs(balance - 0.5) < 1e-6 else mtf(balance, x)
    return (np.clip(y, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


def render8(lin: np.ndarray, black, white, balance) -> np.ndarray:
    """(3, H, W) float -> (H, W, 3) uint8: (x - black) / (white - black), then the midtones function (balance 0.5: none).
    Each channel has its own numbers (or one number for all, a scalar)."""
    c, h, w = lin.shape
    bp = np.broadcast_to(np.asarray(black, np.float64), (c,))
    wp = np.broadcast_to(np.asarray(white, np.float64), (c,))
    mb = np.broadcast_to(np.asarray(balance, np.float64), (c,))
    out = np.empty((h, w, 3), np.uint8)
    luts = {}
    for k in range(c):
        luts[k] = _lut(float(mb[k]))
    for r0 in range(0, h, 512):
        blk = lin[:, r0:r0 + 512]
        for k in range(c):
            x = (blk[k] - np.float32(bp[k])) * np.float32(65535.0 / (wp[k] - bp[k]))
            np.nan_to_num(x, copy=False, nan=0.0, posinf=65535.0, neginf=0.0)
            np.clip(x, 0, 65535, out=x)
            out[r0:r0 + 512, :, k] = luts[k][x.astype(np.uint16)]
    if c == 1:
        out[:, :, 1] = out[:, :, 2] = out[:, :, 0]
    return out


def histogram8(a: np.ndarray) -> np.ndarray:
    """(3, 256) counts of the values of an (H, W, 3) uint8 picture (every second pixel is enough)."""
    s = a[::2, ::2]
    return np.stack([np.bincount(s[:, :, k].ravel(), minlength=256) for k in range(3)])


def stretch_image(lin: np.ndarray):
    """(stretched uint8, linear uint8, parameters): the automatic stretch, and the data as stored (one black and one white
    for the three channels, no midtones), each ready to show."""
    sp = stretch_params(lin)
    stretched = render8(lin, sp["black"], sp["white"], sp["balance"])
    linear = render8(lin, float(sp["black"].min()), float(sp["white"].max()), 0.5)
    return stretched, linear, sp


# ---------------------------------------------------------------------------------------------- loading a picture
def _rgb_of_file(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _wcs_of(p, wid: int, which: str):
    out = api.stage_outputs(p, wid, "astrometry", which)
    if out is None or not (Path(out.dir) / "wcs.fits").is_file():
        return None
    import warnings

    from astropy.io import fits
    from astropy.wcs import WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(fits.getheader(Path(out.dir) / "wcs.fits"))


def load_visual(p, wid: int, spec: VisualSpec, which: str = "current") -> ShownImage:
    """One picture of a done result, ready to show (worker thread). ``which``: current, or previous (the result an
    Outdated block had). Raises ``api.NotDone`` when the result or the file is not there."""
    out = api.stage_outputs(p, wid, spec.stage, which)
    if out is None:
        raise api.NotDone(f"the {spec.stage} result of workflow {wid} is not computed ({which})")
    key = (spec.stage, out.key, spec.name, which)
    if spec.kind == "chart":
        return ShownImage(key, spec, curves=api.spectral_response(p, wid, which))
    if spec.kind == "preview":
        f = out.previews.get(spec.name)
        if f is None:
            raise api.NotDone(f"{spec.stage} has no preview {spec.name!r}")
        a = _rgb_of_file(f)
        src = PixelSource(a)
        return ShownImage(key, spec, stretched=src, size=src.size(), hist={"stretched": histogram8(a)},
                          wcs=_wcs_of(p, wid, which) if spec.grid else None, nbytes=src.nbytes)
    lin = np.asarray(api.stage_array(p, wid, spec.stage, spec.name, which=which), np.float32)
    if lin.ndim == 2:
        lin = lin[None]
    if lin.shape[0] == 1:
        lin = np.repeat(lin, 3, axis=0)
    h, w = lin.shape[1:]
    stretched, linear, sp = stretch_image(lin)
    s8, l8 = PixelSource(stretched), PixelSource(linear)
    grid = None
    wcs = None
    if spec.grid:
        wcs = _wcs_of(p, wid, which)
        grid = (w, h)
    return ShownImage(key, spec, stretched=s8, linear=l8, lin=lin, size=(w, h), grid=grid, wcs=wcs,
                      hist={"stretched": histogram8(stretched), "linear": histogram8(linear)},
                      stretch={k: [float(x) for x in v] for k, v in sp.items()},
                      nbytes=lin.nbytes + s8.nbytes + l8.nbytes)



class ImageCache:
    """The last pictures, least recently used first out, within a budget of bytes (the arrays are big)."""

    def __init__(self, budget: int = 2_000_000_000):
        self.budget = budget
        self._d: OrderedDict[tuple, ShownImage] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> ShownImage | None:
        with self._lock:
            im = self._d.get(key)
            if im is not None:
                self._d.move_to_end(key)
            return im

    def put(self, im: ShownImage):
        with self._lock:
            self._d[im.key] = im
            self._d.move_to_end(im.key)
            while len(self._d) > 1 and sum(x.nbytes for x in self._d.values()) > self.budget:
                self._d.popitem(last=False)

    def clear(self):
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        return len(self._d)

    def keys(self) -> list[tuple]:
        return list(self._d)


# ---------------------------------------------------------------------------------------------- sessions and frames
@dataclass
class SessionRow:
    id: int
    name: str
    lights: int = 0
    darks: int = 0
    flats: int = 0
    bias: int = 0
    flat_darks: int = 0
    first: str = ""              # time of the first light, ISO
    last: str = ""
    exposure: float | None = None
    iso: float | None = None
    temperature: float | None = None
    filter: str = ""
    fwhm: float | None = None    # median star size of its frames, once Registration is done
    kept: int | None = None      # lights the stack takes (None: not known yet)


@dataclass
class ProjectFrames:
    """The sessions of a project with what their headers say, and the health checks (worker thread)."""

    sessions: list[SessionRow] = field(default_factory=list)
    checks: list = field(default_factory=list)                      # api.Check
    camera: dict[str, Any] = field(default_factory=dict)           # camera, width, height, cfa, black, white
    flat_pct: float | None = None                                   # median level of a flat, % of saturation
    dark_level: float | None = None                                 # median level of a dark (ADU)
    darks: dict[str, Any] = field(default_factory=dict)            # exposure, iso of the first dark
    flats: dict[str, Any] = field(default_factory=dict)
    folders: dict[str, list[str]] = field(default_factory=dict)    # kind -> folders as the sessions list them
    counts: dict[str, int] = field(default_factory=dict)           # kind -> files in the project


def _files(p, entries) -> list[Path]:
    from ...core.project import list_frames
    return list_frames([p.abspath(x) for x in entries])


def _median_level(path, info) -> float:
    from ...io.frames import read_cfa
    a = read_cfa(path)[::8, ::8]
    return float(np.median(a))


def project_frames(p, wid: int | None, headers: dict | None = None) -> ProjectFrames:
    """The sessions, headers, star sizes and health checks of a project. ``headers`` caches the headers read (a dict
    owned by the service)."""
    from ...io.frames import read_info
    hdr = headers if headers is not None else {}
    out = ProjectFrames()
    table = None
    if wid is not None:
        try:
            table = api.frame_table(p, wid)
        except (api.NotDone, OSError, ValueError, KeyError):
            table = None
    by_session: dict[int, list] = {}
    if table is not None:
        for r in table.rows:
            by_session.setdefault(r.session, []).append(r)
    first_info = None
    for s in p.sessions:
        row = SessionRow(s.id, s.name)
        lights = _files(p, s.lights)
        row.lights = len(lights)
        row.darks, row.flats = len(_files(p, s.darks)), len(_files(p, s.flats))
        row.bias, row.flat_darks = len(_files(p, s.bias)), len(_files(p, s.flat_darks))
        for kind in ("lights", "darks", "flats", "bias", "flat_darks"):
            out.counts[kind] = out.counts.get(kind, 0) + getattr(row, kind)
        if lights:
            k = (str(lights[0]), str(lights[-1]), len(lights))
            if k not in hdr:
                a = read_info(lights[0])
                hdr[k] = (a, read_info(lights[-1], like=a) if len(lights) > 1 else a)
            a, b = hdr[k]
            first_info = first_info or a
            row.exposure, row.iso, row.temperature = a.exposure, a.iso, a.temperature
            times = [x for x in (a.time, b.time) if x]
            row.first, row.last = (min(times), max(times)) if times else ("", "")
        rows = by_session.get(s.id)
        if rows:
            fw = [r.fwhm for r in rows if np.isfinite(r.fwhm)]
            row.fwhm = float(np.median(fw)) if fw else None
            row.kept = sum(1 for r in rows if not r.status)
        else:
            ex = sum(1 for sid, _n in p.excluded if sid == s.id)
            row.kept = max(row.lights - ex, 0)
        out.sessions.append(row)
    if first_info is not None:
        out.camera = dict(camera=first_info.camera, width=first_info.width, height=first_info.height,
                          cfa=first_info.cfa, black=first_info.black, white=first_info.white)
    for kind in ("darks", "flats", "bias", "flat_darks"):
        seen = []
        for s in p.sessions:
            seen += [p.abspath(x) for x in getattr(s, kind)]
        out.folders[kind] = list(dict.fromkeys(seen))
    # the first dark and the first flat tell what they are (header), their level what they hold (one decode each)
    for kind in ("darks", "flats"):
        fl = [f for s in p.sessions for f in _files(p, getattr(s, kind))]
        if not fl:
            continue
        mid = fl[len(fl) // 2]
        try:
            info = read_info(mid)
            d = {"exposure": info.exposure, "iso": info.iso, "time": info.time, "file": mid.name}
            lvl = hdr.get(("level", str(mid)))
            if lvl is None:
                lvl = hdr[("level", str(mid))] = _median_level(mid, info)
            if kind == "darks":
                out.dark_level, out.darks = lvl, d
            else:
                out.flat_pct = 100 * (lvl - info.black) / max(info.white - info.black, 1.0)
                out.flats = d
        except Exception:  # noqa: BLE001 - a hint: an unreadable frame is reported by the import, not here
            pass
    try:
        out.checks = api.health_checks(p)
    except Exception:  # noqa: BLE001
        out.checks = []
    return out


def add_scan(p, scan, criteria=None, session: int | None = None) -> list:
    """The frames of a scanned folder into the project (GUI thread: memory only, the model saves). One session per night by
    default; with ``criteria`` that split a night (exposure, ISO / gain), one per group. ``session``: everything goes into it."""
    lights = scan.frames.get("lights") or []
    if session is not None or not lights or criteria is None:
        return api.add_frames(p, scan, session)
    by_night = api.group_sessions(lights, api.GroupingCriteria(True, False, False))
    groups = api.group_sessions(lights, criteria)
    if len(groups) <= len(by_night):
        return api.add_frames(p, scan)
    made = [api.add_session(p, lights=g.paths, name=g.name) for g in groups]
    from datetime import datetime
    mids = [datetime.fromisoformat(g.start) if g.start else None for g in groups]
    for kind in ("darks", "flats", "bias", "flat_darks"):
        for folder in scan.folders.get(kind, []):
            fl = [i for i in scan.frames.get(kind, []) if os.path.normcase(os.path.normpath(str(Path(i.path).parent)))
                  == os.path.normcase(os.path.normpath(folder))]
            if not fl:
                continue
            ts = sorted(t for t in map(api._when, fl) if t)
            near = 0
            if ts and any(mids):
                mid = ts[len(ts) // 2]
                near = min((i for i, m in enumerate(mids) if m), key=lambda i: abs(mids[i] - mid))
            lst = getattr(made[near], kind)
            if not any(os.path.normcase(os.path.normpath(folder)) == os.path.normcase(os.path.normpath(x)) for x in lst):
                lst.append(folder)
    p.validate()
    return made


@dataclass
class FrameItem:
    """One frame of a group of the browser."""

    path: str
    name: str
    kind: str                         # lights | darks | flats | bias | flat_darks
    session: int | None = None
    time: str = ""                    # ISO, or the file's date for a calibration frame
    fwhm: float | None = None
    transparency: float | None = None
    weight: float | None = None
    status: str = ""                  # '' kept, else the reason the stack leaves it out
    excluded: bool = False            # by hand
    index: int | None = None          # in the frame table
    level: float | None = None        # median level above black, ADU (lights: from the calibration statistics)


def browse_group(p, wid: int | None, kind: str, headers: dict | None = None) -> list[FrameItem]:
    """The frames of a group, in name order, with what is known of each (worker thread). Lights come with their star
    size, transparency and weight once Registration is done."""
    import datetime as dt

    from ...io.frames import read_info
    hdr = headers if headers is not None else {}
    by_path: dict[str, Any] = {}
    if kind == "lights" and wid is not None:
        try:
            t = api.frame_table(p, wid)
            by_path = {os.path.normcase(os.path.normpath(r.path)): r for r in t.rows}
        except (api.NotDone, OSError, ValueError, KeyError):
            by_path = {}
    items: list[FrameItem] = []
    times: dict[str, str] = {}
    levels = None
    if by_path:                                    # the sky level of each frame, measured at calibration
        cal = api.stage_outputs(p, wid, "calibrate")
        f = Path(cal.dir) / "frame_stats.npy" if cal is not None else None
        if f is not None and f.is_file():
            levels = np.load(f)[:, :, 0].mean(1)
    if kind == "lights" and not by_path:
        inv = api.stage_outputs(p, wid, "inventory") if wid is not None else None
        if inv is not None:
            import json
            try:
                fj = json.loads((Path(inv.dir) / "frames.json").read_text(encoding="utf-8"))["lights"]
                times = {os.path.normcase(os.path.normpath(f["path"])): f.get("time") or "" for f in fj}
            except (OSError, ValueError, KeyError):
                times = {}
    like = None
    for s in p.sessions:
        for f in _files(p, getattr(s, kind)):
            norm = os.path.normcase(os.path.normpath(str(f)))
            it = FrameItem(str(f), f.name, kind, s.id)
            r = by_path.get(norm)
            if r is not None:
                it.time, it.fwhm, it.transparency, it.weight = r.time or "", r.fwhm, r.transparency, r.weight
                it.status, it.index = r.status, r.index
                if levels is not None and r.index < len(levels):
                    it.level = float(levels[r.index])
            elif kind == "lights":
                it.time = times.get(norm, "")
                if not it.time:
                    k = str(f)
                    if k not in hdr:
                        try:
                            hdr[k] = read_info(f, like=like).time or ""
                            like = like or read_info(f)
                        except Exception:  # noqa: BLE001
                            hdr[k] = ""
                    it.time = hdr[k]
            else:
                try:
                    it.time = dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")
                except OSError:
                    it.time = ""
            it.excluded = (s.id, f.name) in {(a, b) for a, b in p.excluded} if kind == "lights" else False
            items.append(it)
    return items


def frame_stats(path: str) -> dict[str, float]:
    """The median level above black and the noise of one frame (ADU): the decode is the cost (worker thread)."""
    from ...io.frames import read_cfa, read_info
    info = read_info(path)
    a = read_cfa(path)[::4, ::4]
    med = float(np.median(a))
    noise = float(1.4826 * np.median(np.abs(a - med)))
    return {"level": med - info.black, "noise": noise, "pct": 100 * (med - info.black) / max(info.white - info.black, 1.0),
            "black": info.black, "white": info.white}


# ---------------------------------------------------------------------------------------------- the service
class Cancelled(Exception):
    """A scan stopped by the user."""


class InspectService(QObject):
    """The requests of the views, and their answers as signals (GUI thread). One service per model (``service_of``)."""

    imageReady = Signal(int, int, object)           # slot, token, ShownImage
    imageFailed = Signal(int, int, str)             # slot, token, reason
    outputsReady = Signal(int, str, str)            # workflow id, stage, which: the metrics are in ``outputs``
    framesReady = Signal(object)                    # ProjectFrames
    tableReady = Signal(int, int, object)           # workflow id, token, FrameTable (or an error text)
    groupReady = Signal(str, int, object)           # kind, token, [FrameItem]
    thumbReady = Signal(str, object)                # path, (h, w, 3) uint8
    frameReady = Signal(int, str, object)           # token, path, (h, w, 3) uint8 (the shown frame)
    statsReady = Signal(str, object)                # path, {level, noise, pct}
    scanProgress = Signal(int, float, str)          # token, fraction, text
    scanDone = Signal(int, object)                  # token, api.ScanResult
    scanFailed = Signal(int, str)
    previewReady = Signal(int, int, int)            # token, kept, total: what the thresholds of Selection keep

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.images = ApiWorker(self)               # arrays, metrics, tables, scans
        self.thumbs = ApiWorker(self)
        self.view = ApiWorker(self)
        self.prefetch = ApiWorker(self)             # the frames the blink shows next
        for w in (self.images, self.thumbs, self.view, self.prefetch):
            w.done.connect(self._done)
            w.failed.connect(self._failed)
        self.cache = ImageCache()
        self.outputs: dict[tuple, Any] = {}          # (wf, stage, which, key) -> StageOutputs (or None)
        self.headers: dict = {}
        self._tokens: dict[str, int] = {}
        self._thumb_token = 0
        self._scan_cancel: set[int] = set()
        self.thumb_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.frame_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.stats: dict[str, dict] = {}
        self._wb = None                              # the gains that make the sky of a light neutral
        self.criteria = api.GroupingCriteria()       # how the next imported folder is grouped (the Grouping dialog)
        self.requests = 0                            # jobs sent to the workers (the tests count them)

    # ------------------------------------------------------------ the workers' answers
    def _done(self, channel: str, _n: int, result):
        head = channel.split(":", 1)[0]
        if head == "inspect.image":
            slot, token, im = result
            self.imageReady.emit(slot, token, im)
        elif head == "inspect.outputs":
            wid, stage, which, key, out = result
            self.outputs[(wid, stage, which, key)] = out
            self.outputsReady.emit(wid, stage, which)
        elif head == "inspect.frames":
            self.framesReady.emit(result)
        elif head == "inspect.table":
            wid, token, t = result
            self.tableReady.emit(wid, token, t)
        elif head == "inspect.group":
            kind, token, items = result
            self.groupReady.emit(kind, token, items)
        elif head == "inspect.thumbs":
            pass
        elif head == "inspect.frame":
            token, path, arr = result
            self.frame_cache[path + "|view"] = arr
            while len(self.frame_cache) > 60:
                self.frame_cache.popitem(last=False)
            self.frameReady.emit(token, path, arr)
        elif head == "inspect.stats":
            path, st = result
            self.stats[path] = st
            self.statsReady.emit(path, st)
        elif head == "inspect.scan":
            token, res = result
            self.scanDone.emit(token, res)
        elif head == "inspect.preview":
            token, kept, total = result
            self.previewReady.emit(token, kept, total)

    def _failed(self, channel: str, _n: int, text: str):
        head = channel.split(":", 1)[0]
        why = text.splitlines()[0] if text else "?"
        if head == "inspect.image":
            slot = int(channel.split(":")[1])
            self.imageFailed.emit(slot, self._tokens.get(channel, 0), why)
        elif head == "inspect.table":
            self.tableReady.emit(int(channel.split(":")[1]), self._tokens.get(channel, 0), why)
        elif head == "inspect.scan":
            self.scanFailed.emit(self._tokens.get(channel, 0), why)

    def _token(self, channel: str) -> int:
        self._tokens[channel] = self._tokens.get(channel, 0) + 1
        return self._tokens[channel]

    def stop(self):
        for w in (self.images, self.thumbs, self.view, self.prefetch):
            w.stop()

    def wait(self, timeout: float = 60.0) -> bool:
        """Process events until the workers are idle (tests and scripted walkthroughs)."""
        return all(w.wait(timeout) for w in (self.images, self.thumbs, self.view, self.prefetch))

    def busy(self) -> bool:
        return any(w.busy() for w in (self.images, self.thumbs, self.view, self.prefetch))

    # ------------------------------------------------------------ requests
    def request_image(self, slot: int, wid: int, spec: VisualSpec, which: str = "current") -> int:
        """Load a picture into ``slot`` (0: the shown one, 1: the one it is compared with). Answer: ``imageReady`` or
        ``imageFailed``, with the token returned here; a newer request of the slot makes the older answer stale. (A view
        looks in ``cache`` first: a picture seen lately is there.)"""
        channel = f"inspect.image:{slot}"
        token = self._token(channel)
        self.requests += 1
        self.images.submit(channel, self._image_job, slot, token, wid, spec, which)
        return token

    def _image_job(self, slot, token, wid, spec, which):
        p = self.model.p
        out = api.stage_outputs(p, wid, spec.stage, which)
        if out is not None:
            hit = self.cache.get((spec.stage, out.key, spec.name, which))
            if hit is not None:
                return slot, token, hit
        im = load_visual(p, wid, spec, which)
        self.cache.put(im)                          # at once: the next request of the worker finds it
        return slot, token, im

    def request_outputs(self, wid: int, stage: str, which: str, key: str):
        """The metrics of a done result (``api.stage_outputs``); answer: ``outputsReady``, then ``outputs_of``."""
        self.requests += 1
        self.images.submit(f"inspect.outputs:{wid}:{stage}:{which}", self._outputs_job, wid, stage, which, key)

    def _outputs_job(self, wid, stage, which, key):
        return wid, stage, which, key, api.stage_outputs(self.model.p, wid, stage, which)

    def outputs_of(self, wid: int, stage: str, which: str, key: str):
        return self.outputs.get((wid, stage, which, key))

    def request_frames(self, wid: int | None):
        """The sessions with their headers and the health checks; answer: ``framesReady``."""
        self.requests += 1
        self.images.submit("inspect.frames", self._frames_job, wid)

    def _frames_job(self, wid):
        return project_frames(self.model.p, wid, self.headers)

    def request_table(self, wid: int) -> int:
        channel = f"inspect.table:{wid}"
        token = self._token(channel)
        self.requests += 1
        self.images.submit(channel, self._table_job, wid, token)
        return token

    def _table_job(self, wid, token):
        return wid, token, api.frame_table(self.model.p, wid)

    def request_preview(self, wid: int, min_transparency: float, max_fwhm_ratio: float) -> int:
        """How many frames these thresholds keep (``api.selection_preview``); answer: ``previewReady``."""
        token = self._token("inspect.preview")
        self.requests += 1
        self.images.submit("inspect.preview", self._preview_job, wid, min_transparency, max_fwhm_ratio, token)
        return token

    def _preview_job(self, wid, tmin, ratio, token):
        t = api.selection_preview(self.model.p, wid, tmin, ratio)
        return token, t.kept, t.total

    def request_group(self, wid: int | None, kind: str) -> int:
        token = self._token("inspect.group")
        self.requests += 1
        self.images.submit("inspect.group", self._group_job, wid, kind, token)
        return token

    def _group_job(self, wid, kind, token):
        return kind, token, browse_group(self.model.p, wid, kind, self.headers)

    def _balanced(self, arr: np.ndarray, balance: bool) -> np.ndarray:
        """A raw frame is green (its pixels are not white balanced): the sky of the first light seen is made neutral and the
        same gains serve every light, so that blinking still shows what changes from frame to frame."""
        if not balance:
            return arr
        if self._wb is None:
            flat = arr.reshape(-1, 3)[::7].astype(np.float32)
            med = np.median(flat, axis=0)
            hi = np.percentile(flat, 99.5, axis=0)
            gain = np.clip((hi[1] - med[1]) / np.maximum(hi - med, 1.0), 0.4, 4.0)
            self._wb = (med, gain.astype(np.float32), med[1])
        med, gain, ref = self._wb
        return np.clip(ref + (arr.astype(np.float32) - med) * gain, 0, 255).astype(np.uint8)

    def request_thumbs(self, paths: list[str], size: int = 192, balance: bool = False) -> int:
        """Thumbnails, in this order, the ones already made left out; a new request drops what is not made yet. Each
        answer is a ``thumbReady`` signal."""
        self._thumb_token += 1
        token = self._thumb_token
        todo = [x for x in paths if x not in self.thumb_cache]
        if todo:
            self.requests += 1
            self.thumbs.submit("inspect.thumbs", self._thumbs_job, todo, size, token, balance)
        return token

    def _thumbs_job(self, paths, size, token, balance):
        p = self.model.p
        for path in paths:
            if token != self._thumb_token:
                break
            if path in self.thumb_cache:
                continue
            try:
                arr = self._balanced(api.frame_thumb(p, path, size), balance)
            except Exception:  # noqa: BLE001 - one unreadable frame does not stop the others
                continue
            self.thumb_cache[path] = arr
            while len(self.thumb_cache) > 4000:
                self.thumb_cache.popitem(last=False)
            self.thumbReady.emit(path, arr)
        return None

    def request_frame(self, path: str, size: int = 1280, balance: bool = False) -> int:
        """One frame, large; answer: ``frameReady``. A frame seen already comes back at once."""
        token = self._token("inspect.frame")
        hit = self.frame_cache.get(path + "|view")
        if hit is not None:
            QTimer.singleShot(0, lambda: self.frameReady.emit(token, path, hit))     # after the caller has the token
            return token
        self.requests += 1
        self.view.submit("inspect.frame", self._frame_job, path, size, token, balance)
        return token

    def _frame_job(self, path, size, token, balance):
        return token, path, self._balanced(api.frame_thumb(self.model.p, path, size), balance)

    def prefetch_frames(self, paths: list[str], size: int = 1280, balance: bool = False):
        """Make these frames in the background, so that the viewer finds them (blink)."""
        todo = [x for x in paths if x + "|view" not in self.frame_cache]
        if todo:
            self.requests += 1
            self.prefetch.submit("inspect.prefetch", self._prefetch_job, todo, size, balance)

    def _prefetch_job(self, paths, size, balance):
        for path in paths:
            if path + "|view" in self.frame_cache:
                continue
            try:
                self.frame_cache[path + "|view"] = self._balanced(api.frame_thumb(self.model.p, path, size), balance)
            except Exception:  # noqa: BLE001
                continue
            while len(self.frame_cache) > 60:
                self.frame_cache.popitem(last=False)
        return None

    def request_stats_batch(self, paths: list[str]):
        """The level and noise of these frames, one after the other in the background; each answer is ``statsReady``."""
        todo = [x for x in paths if x not in self.stats]
        if todo:
            self.requests += 1
            self.prefetch.submit("inspect.statsbatch", self._stats_batch_job, todo)

    def _stats_batch_job(self, paths):
        for path in paths:
            if path in self.stats:
                continue
            try:
                st = frame_stats(path)
            except Exception:  # noqa: BLE001
                continue
            self.stats[path] = st
            self.statsReady.emit(path, st)
        return None

    def request_stats(self, path: str):
        if path in self.stats:
            self.statsReady.emit(path, self.stats[path])
            return
        self.requests += 1
        self.view.submit("inspect.stats", self._stats_job, path)

    def _stats_job(self, path):
        return path, frame_stats(path)

    def scan(self, folder: str) -> int:
        """Read a folder to import; answers: ``scanProgress``, then ``scanDone`` or ``scanFailed``."""
        token = self._token("inspect.scan")
        self.requests += 1
        self.images.submit("inspect.scan", self._scan_job, folder, token)
        return token

    def cancel_scan(self, token: int):
        self._scan_cancel.add(token)

    def _scan_job(self, folder, token):
        def progress(f, text):
            if token in self._scan_cancel:
                raise Cancelled()
            self.scanProgress.emit(token, float(f), text)
        return token, api.scan_folder(folder, progress)


_services: WeakKeyDictionary = WeakKeyDictionary()


def service_of(model) -> InspectService:
    """The one service of a model (made at the first request)."""
    sv = _services.get(model)
    if sv is None:
        sv = _services[model] = InspectService(model)
    return sv
