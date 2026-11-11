"""How long a stage job takes and how much disk it leaves, before it runs.

A stage's cost grows with its *units*:

    inventory              frames (all kinds)
    calibrate, register    light frames x sensor Mpx
    stack                  light frames x sensor Mpx x drizzle scale²
    denoise                adaptation steps + DENOISE_MPX_UNITS x stack grid Mpx (applying the network)
    any other stage        stack grid Mpx

With a history of this machine's runs (``STATE/timings2.json``: the last 20 [seconds, units, "gpu" | "cpu"] per
stage) an estimate is median(seconds / units) x units, *measured*. Otherwise it is *≈*: the time of the reference
project (``HEART``: 721 lights of 12.1 Mpx, drizzle 1.5) x the units
ratio, x ``CPU_FACTOR`` without a GPU. Disk follows the reference the same way, plus the calibrated frame cache that
calibrate writes on the scratch disk (light frames x sensor pixels x 2 bytes).

``learn_from_v1`` reads the durations of a v1 project's manifests, so that a machine's past projects can seed the
history (``seed``) before its first v2 run.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .files import write_json

HISTORY_FILE = "timings2.json"
KEEP = 20                                      # entries kept per stage
LOCK_WAIT = 5.0                                # seconds a writer of the history waits for another one
LOCK_STALE = 30.0                              # a lock file older than this was left by a process that died

#: the reference project, Heart: (seconds, GB in the stage folder) per stage, from its manifests; gradient and
#: render_simple are guesses at Heart size (60 s and 30 s), star_removal from the bench (8 s for 12 Mpx on the RTX
#: 4060 Ti, 5 November 2026, plus the reading and writing of its images)
HEART: dict[str, tuple[float, float]] = {
    "inventory": (5.0, 0.0), "calibrate": (310.8, 0.11), "register": (121.4, 0.0), "stack": (2064.0, 1.31),
    "astrometry": (124.1, 0.0), "reference": (89.8, 0.0), "catalogs": (178.3, 0.07), "surveys": (42.5, 0.11),
    "synth_stars": (60.0, 0.3), "color": (2.9, 0.0), "deconv": (85.0, 1.0), "denoise": (20.4, 0.68), "gradient": (60.0, 0.01), "render_simple": (30.0, 0.7),
    "star_removal": (30.0, 0.66),
}
#: a stage the table does not know (tests' toy stages): one minute at Heart size
OTHER = (60.0, 0.0)
#: without a GPU, each stage takes this many times longer (guesses, not measured; star_removal measured on 5 November
#: 2026, issue #9: 7.7 s per Mpx on 8 processor threads, about 4 times the reference's GPU time with its reading and
#: writing; the 40 guessed before announced 50 minutes for North America, where the run takes about 4). Stacking
#: measured on 11 November 2026 (issue #130) on a laptop of 8 cores, drizzle scale 1, 12 Mpx: 50 frames in 156 and
#: 186 s, 754 frames in 2766 s, 2.5 to 2.9 times the reference; the 5 guessed before announced 80 minutes for 46
CPU_FACTOR: dict[str, float] = {
    "inventory": 1, "calibrate": 1.3, "register": 1.5, "stack": 2.7, "astrometry": 1, "catalogs": 1, "surveys": 1,
    "synth_stars": 1, "color": 1.5, "deconv": 20, "denoise": 45, "gradient": 1, "render_simple": 2, "star_removal": 4,
}
DEFAULT_CPU_FACTOR = 2.0
#: Denoise (stages/denoise.py): applying the pretrained network costs this many adaptation steps per Mpx of the stack
#: grid; measured on the bench on 4 November 2026 (RTX 4060 Ti; 8 processor threads): seconds per adaptation step and
#: per Mpx applied. Its two parts do not slow down alike without a GPU, so without a history of this kind of machine
#: the estimate is these times, not the reference's x CPU_FACTOR.
DENOISE_MPX_UNITS = 8.5
DENOISE_SECONDS = {"gpu": {"step": 0.088, "mpx": 0.75}, "cpu": {"step": 3.6, "mpx": 3.3}}
#: the adaptation lengths (stages/denoise.ADAPTATIONS: steps)
DENOISE_ADAPT_STEPS = {"short": 500, "long": 4000}
#: the network's passes over the image (stages/denoise.PASSES; the Fast pass setting: one). The times above are per pass
DENOISE_PASSES = 4


@dataclass
class Facts:
    """What the units of a workflow's stages are made of."""

    frames: int = 795                # every frame the inventory reads (Heart: 721 lights, 28 darks, 46 flats)
    lights: int = 721
    mpx: float = 4256 * 2848 / 1e6   # sensor
    scale: float = 1.5               # drizzle
    grid_mpx: float = 6384 * 4272 / 1e6
    iterations: int = 0              # denoise: adaptation steps (0: the pretrained network as it is)
    passes: int = DENOISE_PASSES     # denoise: passes of the network over the image


HEART_FACTS = Facts()


@dataclass
class Estimate:
    seconds: float
    measured: bool                   # from this machine's history; False: ≈, from the reference project
    disk_gb: float = 0.0             # what the job leaves on disk (its folder, and calibrate's frame cache)


def units(stage: str, f: Facts) -> float:
    if stage == "inventory":
        return float(f.frames)
    if stage in ("calibrate", "register"):
        return f.lights * f.mpx
    if stage == "stack":
        return f.lights * f.mpx * f.scale ** 2
    if stage == "denoise":
        return float(f.iterations) + DENOISE_MPX_UNITS * f.grid_mpx * f.passes
    return f.grid_mpx


def cache_gb(f: Facts) -> float:
    """The calibrated frame cache (float16 per sensor pixel and light frame)."""
    return f.lights * f.mpx * 1e6 * 2 / 1e9


# ---------------------------------------------------------------------------------------------- history

def history_path() -> Path:
    from . import project
    return project.STATE / HISTORY_FILE


def load_history(path: str | Path | None = None) -> dict[str, list[list]]:
    """{stage: [[seconds, units, "gpu" | "cpu"], ...]} ({} when absent or unreadable)."""
    try:
        d = json.loads(Path(path or history_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _kind(gpu: bool) -> str:
    return "gpu" if gpu else "cpu"


def record(stage: str, seconds: float, n_units: float, gpu: bool, path: str | Path | None = None) -> None:
    """Add a job's duration to the history (written atomically; never raises)."""
    seed([(stage, seconds, n_units, _kind(gpu))], path, dedupe=False)


@contextmanager
def _locked(p: Path) -> Iterator[None]:
    """Hold ``<history>.lock`` (created with O_EXCL) while the history is read, changed and written: the workers of
    the GPU queues finish stages at the same time, and without it the second writer rewrote the file from what it
    had read before the first one wrote, losing the first one's entry. A lock older than ``LOCK_STALE`` was left by a
    process that died: it is removed. After ``LOCK_WAIT`` the update goes ahead without it (a job never waits long
    for a hint)."""
    lock = p.with_name(p.name + ".lock")
    deadline = time.monotonic() + LOCK_WAIT
    fd = None
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except (FileExistsError, PermissionError):   # held (Windows: PermissionError while it is being deleted)
            pass
        except OSError:                                # no lock possible there
            break
        try:
            if time.time() - lock.stat().st_mtime > LOCK_STALE:
                lock.unlink()
                continue
        except OSError:
            pass
        if time.monotonic() > deadline:
            break
        time.sleep(0.01)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
            try:
                lock.unlink()
            except OSError:
                pass


def seed(entries: list[tuple[str, float, float, str]], path: str | Path | None = None, dedupe: bool = True) -> int:
    """Add (stage, seconds, units, "gpu" | "cpu") entries to the history; ``dedupe`` skips those already there.
    The read and the write are one step under a lock file (``_locked``), so workers that record at once all keep
    their entries. Returns the number added. Never raises: the history is a hint."""
    p = Path(path or history_path())
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return 0
    with _locked(p):
        d = load_history(p)
        added = 0
        for stage, s, u, kind in entries:
            if not (s and u and s > 0 and u > 0):
                continue
            e = [round(float(s), 3), round(float(u), 3), kind]
            rows = d.setdefault(stage, [])
            if dedupe and e in rows:
                continue
            rows.append(e)
            d[stage] = rows[-KEEP:]
            added += 1
        if added:
            try:
                write_json(p, d, indent=None, default=None)
            except OSError:
                return 0
    return added


# ---------------------------------------------------------------------------------------------- estimates

def _rates(history: dict, stage: str, kind: str) -> list[float]:
    out = []
    for row in history.get(stage) or []:
        try:
            s, u, k = float(row[0]), float(row[1]), row[2]
        except (TypeError, ValueError, IndexError):
            continue
        if k == kind and s > 0 and u > 0:
            out.append(s / u)
    return out


def estimate(stage: str, f: Facts, gpu: bool = True, history: dict | None = None) -> Estimate:
    """Seconds and disk of a stage job on this machine (``gpu``: the run has a GPU)."""
    u = units(stage, f)
    hist = load_history() if history is None else history
    ref_s, ref_gb = HEART.get(stage, OTHER)
    ref_u = units(stage, HEART_FACTS)
    gb = ref_gb * u / ref_u + (cache_gb(f) if stage == "calibrate" else 0.0)
    factor = CPU_FACTOR.get(stage, DEFAULT_CPU_FACTOR)
    own = _rates(hist, stage, _kind(gpu))
    if own:
        return Estimate(statistics.median(own) * u, True, gb)
    if stage == "denoise":
        t = DENOISE_SECONDS[_kind(gpu)]
        return Estimate(t["step"] * f.iterations + t["mpx"] * f.grid_mpx * f.passes, False, gb)
    other = _rates(hist, stage, _kind(not gpu))          # measured on the other kind of machine: scaled, ≈
    if other:
        return Estimate(statistics.median(other) * u * (1 / factor if gpu else factor), False, gb)
    return Estimate(ref_s * u / ref_u * (1 if gpu else factor), False, gb)


# ---------------------------------------------------------------------------------------------- facts

_SENSOR: dict[str, float | None] = {}


def _sensor_mpx(path: str) -> float | None:
    """Mpx of a raw frame from its header (cached; None when it cannot be read)."""
    if path not in _SENSOR:
        try:
            from ..io.frames import read_info
            i = read_info(path)
            _SENSOR[path] = i.width * i.height / 1e6 if i.width and i.height else None
        except Exception:  # noqa: BLE001 - an estimate never fails
            _SENSOR[path] = None
    return _SENSOR[path]


def denoise_steps(method: str | None, adaptation: str | None) -> int:
    """The adaptation steps of a Denoise job: 0 for the pretrained network, else those of its adaptation length."""
    return DENOISE_ADAPT_STEPS.get(adaptation or "short", 500) if method == "adapted" else 0


def facts(keys: Any, wf: Any, store: Any = None) -> Facts:
    """The facts of a v2 workflow: from its done results (inventory, stack, denoise manifests) when there are some,
    else from its settings and its frame files; the reference project's values for what is unknown.
    ``keys``: a core.store.Keys; ``store``: a core.store.Store (None: results are not read)."""
    f = Facts()
    try:
        chain = keys.chain(wf)
    except (ValueError, KeyError):
        return f

    def done(stage: str) -> dict:
        if store is None or stage not in chain:
            return {}
        try:
            k = keys.key(wf, stage)
            return store.manifest(stage, k) if store.done(stage, k) else {}
        except (ValueError, OSError):
            return {}

    def given(stage: str) -> dict:
        try:
            return keys.overrides(wf, stage) if stage in chain else {}
        except (ValueError, OSError):
            return {}

    inv = done("inventory").get("metrics") or {}
    if inv.get("n_lights"):
        f.lights = int(inv["n_lights"])
        f.frames = f.lights + sum(int(inv.get(f"n_{k}") or 0) for k in ("darks", "flats", "bias", "flat_darks"))
    else:
        try:
            frames = keys.scan.frames(keys.project)
        except (OSError, ValueError):
            frames = []
        if frames:
            lights = [f[1] for f in frames if f[0] == "lights"]          # [kind, name, size(, filter)]
            ml = int(given("inventory").get("max_lights") or 0)
            f.lights = min(len(lights), ml) if ml else len(lights)
            f.frames = len(frames) - len(lights) + f.lights
    if inv.get("width") and inv.get("height"):
        f.mpx = inv["width"] * inv["height"] / 1e6
    else:
        for s in keys.project.sessions:
            try:
                files = keys.scan.session_frames(keys.project, s)["lights"]
            except (OSError, ValueError):
                files = []
            if files:
                f.mpx = _sensor_mpx(str(files[0][0])) or f.mpx
                break
    stack = done("stack")
    scale = (stack.get("params") or {}).get("scale") or given("stack").get("scale")
    if not scale and "stack" in keys.rules.chains.stages:
        scale = keys.rules.defaults("stack").get("scale")
    f.scale = float(scale or f.scale)
    grid = (stack.get("metrics") or {}).get("grid")
    f.grid_mpx = grid[0] * grid[1] / 1e6 if grid else f.mpx * f.scale ** 2
    dn = {**given("denoise"), **(done("denoise").get("params") or {})}
    f.iterations = denoise_steps(dn.get("method") or (getattr(wf, "options", None) or {}).get("denoise"),
                                 dn.get("adaptation"))
    f.passes = 1 if dn.get("fast") else DENOISE_PASSES
    # an image opened (not frames): its own size is the grid (the night test pass: the reference's 27 Mpx gave "≈ 3 min"
    # for a picture of 2 Mpx computed in 10 s)
    w = wf if hasattr(wf, "source") else next((x for x in keys.project.workflows if x.id == wf), None)
    path = (getattr(w, "source", None) or {}).get("path")
    if path and not grid:
        mpx = _sensor_mpx(str(keys.project.abspath(path)) if hasattr(keys.project, "abspath") else str(path))
        if mpx:
            f.mpx, f.scale, f.grid_mpx = mpx, 1.0, mpx
    return f


# ---------------------------------------------------------------------------------------------- v1 projects

def v1_facts(root: str | Path) -> Facts:
    """The facts of a v1 project from the manifests of its ``work/`` folder."""
    work = Path(root) / "work"

    def man(stage: str) -> dict:
        try:
            m = json.loads((work / stage / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return m if m.get("status") == "done" else {}

    f = Facts()
    inv = man("inventory").get("metrics") or {}
    if inv.get("n_lights"):
        f.lights = int(inv["n_lights"])
        f.frames = f.lights + sum(int(inv.get(f"n_{k}") or 0) for k in ("darks", "flats", "bias", "flat_darks"))
    if inv.get("width") and inv.get("height"):
        f.mpx = inv["width"] * inv["height"] / 1e6
    stack = man("stack")
    f.scale = float((stack.get("params") or {}).get("scale") or f.scale)
    grid = (stack.get("metrics") or {}).get("grid")
    f.grid_mpx = grid[0] * grid[1] / 1e6 if grid else f.mpx * f.scale ** 2
    f.iterations = 0                 # v1 trained its network from scratch: no adaptation steps to count
    return f


def v1_durations(root: str | Path) -> dict[str, float]:
    """{stage: seconds} of the done stages of a v1 project (a stage that was off counts 0)."""
    out = {}
    work = Path(root) / "work"
    if not work.is_dir():
        return out
    for d in sorted(work.iterdir()):
        try:
            m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if m.get("status") == "done" and isinstance(m.get("duration"), (int, float)):
            out[d.name] = float(m["duration"])
    return out


def learn_from_v1(root: str | Path, gpu: bool = True) -> list[tuple[str, float, float, str]]:
    """History entries (stage, seconds, units, kind) from the done stages of a v1 project, read only. Stages that
    did nothing (a stage that was off: under a second) are left out. v1 runs used the GPU: ``gpu`` True."""
    f = v1_facts(root)
    return [(s, sec, units(s, f), _kind(gpu)) for s, sec in v1_durations(root).items() if sec >= 1.0]
